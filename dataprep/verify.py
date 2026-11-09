"""Check the Earshot data pack and write data/manifests/SUMMARY.md.

Run from the repo root:  .venv/bin/python -m dataprep.verify [--full]

Checks - any violation makes the exit code 1 (warnings do not):
 1. Every line of every data/manifests/*.jsonl parses and follows the schema in dataprep/common.py.
 2. Every file exists. Every eval/test file plus a fixed 10 % sample of the others is decoded and must be 16 kHz
    mono with the manifest's duration (--full decodes all of them; .npy files are loaded and checked for NaN/Inf).
 3. A speaker never appears in two splits.
 4. No eval or test speaker appears in a train or val manifest. Ids are compared source-aware: "vctk:p225" and
    "globe:p225" are different people, but LibriSpeech, LibriTTS and LibriTTS-R share LibriVox reader numbers, so
    "librispeech:1089" and "librittsr:1089" are the same person (see canonical_speaker).
 5. Every license id in a manifest is listed in dataprep/LICENSES.md, and non-commercial / evaluation-only licenses
    appear only on evaluation entries (split "test" and use exactly ["eval"]), so training can never load them.
 6. Eval definition files (data/eval/libri2mix_test.jsonl, data/eval/dipco_segments.jsonl) point only at files that
    exist.
 7. RIR banks (data/rir/<name>.npy with one line per row in <name>.jsonl, written by rir_build) load, hold only
    finite values, have as many rows as their .jsonl has lines, and carry licenses listed in LICENSES.md. Rows
    without a license field are our own pyroomacoustics rooms (id `generated`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import DATA, MANIFESTS, ROOT, SR, free_gb

LICENSES_MD = ROOT / "dataprep" / "LICENSES.md"
ALLOWED_USE = {"target", "interferer", "babble", "enrollment", "noise", "music", "eval"}
SPLITS = {"train", "val", "test"}
REQUIRED = {"path": str, "source": str, "duration": (int, float), "sr": int, "split": str, "license": str,
            "use": list}
# Corpora that share LibriVox reader ids as speaker ids.
LIBRI_FAMILY = {"librispeech", "libritts", "librittsr", "libritts_r", "libritts-r", "libri", "librimix",
                "libri2mix", "voices"}
# Licenses that must never reach training, besides any id containing "NC" (non-commercial).
EVAL_ONLY_LICENSES = {"BUT-NTT-EVAL"}
AUDIO_EXT = {".flac", ".wav", ".ogg"}


def canonical_speaker(speaker: str | None, source: str | None = None) -> str | None:
    """Source-aware speaker id used for every cross-manifest comparison.

    "librittsr:0103" -> "libri:103", "LibriSpeech:103" -> "libri:103", "vctk:p225" -> "vctk:p225".
    An id without a "source:" prefix gets the entry's source as prefix."""
    if speaker is None:
        return None
    s = str(speaker).strip().lower()
    if not s:
        return None
    src, sep, sid = s.partition(":")
    if not sep:
        src, sid = (source or "").strip().lower(), s
    if src in LIBRI_FAMILY:
        if sid.startswith("sp") and sid[2:].isdigit():   # VOiCES style "sp0083"
            sid = sid[2:]
        if sid.isdigit():
            sid = sid.lstrip("0") or "0"
        return f"libri:{sid}"
    return f"{src}:{sid}" if src else sid


def is_eval_entry(e: dict) -> bool:
    return e.get("split") == "test" or "eval" in (e.get("use") or [])


def is_train_entry(e: dict) -> bool:
    return e.get("split") in ("train", "val") and "eval" not in (e.get("use") or [])


def eval_only_license(lic: str) -> bool:
    return bool(re.search(r"(^|-)NC(-|$)", lic.upper())) or lic.upper() in EVAL_ONLY_LICENSES


def license_ids(md_text: str) -> set[str]:
    """License ids listed in LICENSES.md - every table row whose first cell is a `backticked` id."""
    return set(re.findall(r"^\|\s*`([^`]+)`\s*\|", md_text, flags=re.M))


def in_sample(path: str, frac: float) -> bool:
    if frac >= 1.0:
        return True
    return int(hashlib.sha1(path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


@dataclass
class Report:
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    entries: list[tuple[str, dict]] = field(default_factory=list)   # (manifest name, entry)
    sizes: dict[str, int] = field(default_factory=dict)              # path -> bytes, existing files only
    decoded: int = 0
    manifests: list[str] = field(default_factory=list)

    def bad(self, msg: str) -> None:
        self.violations.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def load_manifests(mdir: Path, rep: Report) -> None:
    for p in sorted(mdir.glob("*.jsonl")):
        rep.manifests.append(p.name)
        raw = p.read_text(encoding="utf-8")
        lines = raw.split("\n")
        for k, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                last = k == len(lines) - 1 and not raw.endswith("\n")
                (rep.warn if last else rep.bad)(f"{p.name}:{k + 1}: line does not parse"
                                                 + (" (incomplete last line - a fetcher may still be writing)"
                                                    if last else ""))
                continue
            rep.entries.append((p.name, e))


def check_schema(name: str, e: dict, rep: Report) -> bool:
    where = f"{name}: {e.get('path')}"
    for k, t in REQUIRED.items():
        if k not in e:
            rep.bad(f"{where}: missing field '{k}'")
            return False
        if not isinstance(e[k], t) or (k == "sr" and isinstance(e[k], bool)):
            rep.bad(f"{where}: field '{k}' has type {type(e[k]).__name__}")
            return False
    ok = True
    p = e["path"]
    if p.startswith("/") or ".." in Path(p).parts:
        rep.bad(f"{where}: path must be relative to data/ and stay inside it")
        ok = False
    if e["sr"] != SR:
        rep.bad(f"{where}: sr {e['sr']} (must be {SR})")
        ok = False
    if e["split"] not in SPLITS:
        rep.bad(f"{where}: split '{e['split']}' (must be train, val or test)")
        ok = False
    if not e["duration"] > 0:
        rep.bad(f"{where}: duration {e['duration']}")
        ok = False
    bad_use = set(e["use"]) - ALLOWED_USE
    if bad_use or not e["use"]:
        rep.bad(f"{where}: use {e['use']} (allowed: {', '.join(sorted(ALLOWED_USE))})")
        ok = False
    if not e["license"].strip():
        rep.bad(f"{where}: empty license")
        ok = False
    if e.get("gender") not in (None, "F", "M"):
        rep.warn(f"{where}: gender '{e.get('gender')}' is not F, M or null")
    spk = e.get("speaker")
    if spk is not None and (not isinstance(spk, str) or ":" not in spk):
        rep.warn(f"{where}: speaker id '{spk}' has no 'source:' prefix")
    return ok


def decode_check(path: Path, duration: float) -> str | None:
    """None if the file is fine, else what is wrong."""
    try:
        if path.suffix == ".npy":
            a = np.load(path, mmap_mode="r")
            if a.size == 0:
                return "empty array"
            flat = a.reshape(-1)
            for i in range(0, flat.size, 1 << 24):  # in pieces, so a big RIR bank never sits in memory twice
                if not np.isfinite(flat[i:i + (1 << 24)]).all():
                    return "NaN or Inf values"
            return None
        if path.suffix.lower() not in AUDIO_EXT:
            return None if path.stat().st_size > 0 else "empty file"
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != SR:
            return f"{sr} Hz"
        if x.shape[1] != 1:
            return f"{x.shape[1]} channels"
        if x.shape[0] == 0:
            return "no samples"
        dur = x.shape[0] / sr
        if abs(dur - duration) > max(0.02, 0.001 * duration):
            return f"duration {dur:.3f} s but manifest says {duration:.3f} s"
        if not np.isfinite(x).all():
            return "NaN or Inf samples"
    except Exception as ex:  # unreadable file
        return f"{type(ex).__name__}: {ex}"
    return None


def check_files(rep: Report, data: Path, full: bool, sample: float, workers: int) -> None:
    to_decode: dict[str, float] = {}
    seen_split: dict[str, tuple[str, str]] = {}
    repeats: Counter = Counter()
    for name, e in rep.entries:
        p = e["path"]
        if p in seen_split:
            first_name, first_split = seen_split[p]
            if first_split != e["split"]:
                rep.bad(f"{p}: listed as '{first_split}' in {first_name} and '{e['split']}' in {name}")
            else:
                repeats[name] += 1
            continue
        seen_split[p] = (name, e["split"])
        f = data / p
        try:
            rep.sizes[p] = f.stat().st_size
        except FileNotFoundError:
            rep.bad(f"{name}: missing file {p}")
            continue
        if full or is_eval_entry(e) or in_sample(p, sample):
            to_decode[p] = float(e["duration"])
    for name, n in sorted(repeats.items()):
        rep.warn(f"{name}: {n} lines repeat a path listed earlier (same split)")
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda kv: (kv[0], decode_check(data / kv[0], kv[1])), to_decode.items()))
    for p, err in results:
        if err:
            rep.bad(f"{p}: {err}")
    rep.decoded = len(results)


def check_speakers(rep: Report) -> None:
    splits: dict[str, set[str]] = defaultdict(set)
    where: dict[str, set[str]] = defaultdict(set)
    eval_spk: dict[str, set[str]] = defaultdict(set)
    train_spk: dict[str, set[str]] = defaultdict(set)
    for name, e in rep.entries:
        c = canonical_speaker(e.get("speaker"), e.get("source"))
        if c is None:
            continue
        splits[c].add(e["split"])
        where[c].add(name)
        if is_eval_entry(e):
            eval_spk[c].add(name)
        if is_train_entry(e):
            train_spk[c].add(name)
    crossed = sorted(c for c, s in splits.items() if len(s) > 1)
    for c in crossed[:50]:
        rep.bad(f"speaker {c} is in splits {sorted(splits[c])} ({', '.join(sorted(where[c]))})")
    if len(crossed) > 50:
        rep.bad(f"... and {len(crossed) - 50} more speakers in more than one split")
    leaked = sorted(set(eval_spk) & set(train_spk))
    for c in leaked[:50]:
        rep.bad(f"eval/test speaker {c} ({', '.join(sorted(eval_spk[c]))}) also in train/val "
                f"({', '.join(sorted(train_spk[c]))})")
    if len(leaked) > 50:
        rep.bad(f"... and {len(leaked) - 50} more eval speakers that are also in train/val")


def check_licenses(rep: Report, known: set[str]) -> None:
    unknown: Counter = Counter()
    for name, e in rep.entries:
        lic = e["license"]
        if lic not in known:
            unknown[(name, lic)] += 1
        if eval_only_license(lic) and not (e["split"] == "test" and e["use"] == ["eval"]):
            rep.bad(f"{name}: {e['path']} has evaluation-only license {lic} but split '{e['split']}' and use "
                    f"{e['use']} (must be 'test' and ['eval'])")
        if "eval" in e["use"] and e["split"] != "test":
            rep.bad(f"{name}: {e['path']} has use 'eval' but split '{e['split']}'")
    for (name, lic), n in sorted(unknown.items()):
        rep.bad(f"{name}: license '{lic}' ({n} entries) is not listed in dataprep/LICENSES.md")


def check_definitions(rep: Report, data: Path) -> dict[str, int]:
    """Paths inside eval definition files must exist. Returns lines checked per file."""
    counts = {}
    l2m = data / "eval" / "libri2mix_test.jsonl"
    if l2m.exists():
        missing: Counter = Counter()
        n = 0
        for line in l2m.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            n += 1
            paths = [s["path"] for s in d["sources"]] + [d["noise"]["path"]]
            paths += [x["path"] for x in d.get("enrollment", []) if x]
            for p in paths:
                if not (data / p).exists():
                    missing[p.split("/")[1] if p.startswith("eval/") else p.split("/")[0]] += 1
        for kind, k in missing.items():
            rep.bad(f"libri2mix_test.jsonl: {k} references to missing files under {kind}/")
        counts["libri2mix_test.jsonl"] = n
    seg = data / "eval" / "dipco_segments.jsonl"
    if seg.exists():
        n, bad = 0, set()
        for line in seg.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                n += 1
                for k in ("far_path", "close_path"):
                    if d.get(k) and not (data / d[k]).exists():
                        bad.add(d[k])
        for p in sorted(bad):
            rep.bad(f"dipco_segments.jsonl: points at missing file {p}")
        counts["dipco_segments.jsonl"] = n
    return counts


def check_rir_banks(rep: Report, data: Path, known: set[str]) -> list[list]:
    """Rows of (bank, split, rows, shape, GB) for every finished RIR bank in data/rir/."""
    out = []
    rdir = data / "rir"
    if not rdir.is_dir():
        return out
    for partial in sorted(rdir.glob("*.partial.npy")):
        rep.warn(f"rir/{partial.name}: bank still being built (rir_build running or interrupted)")
    for npy in sorted(rdir.glob("*.npy")):
        if npy.name.endswith(".partial.npy"):
            continue
        meta = npy.with_suffix(".jsonl")
        name = f"rir/{npy.name}"
        try:
            a = np.load(npy, mmap_mode="r")
        except Exception as ex:
            rep.bad(f"{name}: does not load ({type(ex).__name__}: {ex})")
            continue
        err = decode_check(npy, 0.0)
        if err:
            rep.bad(f"{name}: {err}")
        rows = []
        if meta.exists():
            rows = [json.loads(x) for x in meta.read_text(encoding="utf-8").splitlines() if x.strip()]
            if len(rows) != a.shape[0]:
                rep.bad(f"{name}: {a.shape[0]} rows but {meta.name} has {len(rows)} lines")
        else:
            rep.bad(f"{name}: no {meta.name} next to it")
        lic = Counter(r.get("license", "generated") for r in rows)
        for l, n in sorted(lic.items()):
            if l not in known:
                rep.bad(f"{name}: license '{l}' ({n} rows) is not listed in dataprep/LICENSES.md")
            if eval_only_license(l):
                rep.bad(f"{name}: evaluation-only license {l} in an RIR bank")
        splits = sorted({r.get("split") for r in rows if r.get("split")}) or \
            [x for x in ("train", "val", "test") if f"_{x}" in npy.stem]
        out.append([name, "/".join(splits) or "?", int(a.shape[0]), "x".join(map(str, a.shape)),
                    npy.stat().st_size / 1e9, ", ".join(f"{k} {v}" for k, v in sorted(lic.items()))])
    return out


def summarize(rep: Report) -> tuple[list[list], list[list]]:
    """Rows of (source, split, files, speakers, hours, GB) and (manifest, files, speakers, hours, GB)."""
    by_src: dict[tuple[str, str], dict] = defaultdict(lambda: {"n": 0, "spk": set(), "s": 0.0, "paths": set()})
    by_man: dict[str, dict] = defaultdict(lambda: {"n": 0, "spk": set(), "s": 0.0, "paths": set()})
    for name, e in rep.entries:
        for d in (by_src[(e.get("source", "?"), e.get("split", "?"))], by_man[name]):
            d["n"] += 1
            d["s"] += float(e.get("duration") or 0)
            d["paths"].add(e.get("path"))
            c = canonical_speaker(e.get("speaker"), e.get("source"))
            if c:
                d["spk"].add(c)

    def gb(paths: set) -> float:
        return sum(rep.sizes.get(p, 0) for p in paths) / 1e9

    src_rows = [[s, sp, d["n"], len(d["spk"]), d["s"] / 3600, gb(d["paths"])]
                for (s, sp), d in sorted(by_src.items())]
    man_rows = [[m, d["n"], len(d["spk"]), d["s"] / 3600, gb(d["paths"])] for m, d in sorted(by_man.items())]
    return src_rows, man_rows


def write_summary(rep: Report, src_rows, man_rows, def_counts, out: Path, mode: str, seconds: float,
                  rir_rows=()) -> str:
    def table(head, rows, fmt):
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(f.format(v) for f, v in zip(fmt, r)) + " |" for r in rows]
        return "\n".join(lines)

    train = [r for r in src_rows if r[1] in ("train", "val")]
    test = [r for r in src_rows if r[1] == "test"]
    tot = lambda rows: [sum(r[2] for r in rows), sum(r[4] for r in rows), sum(r[5] for r in rows)]
    t_tr, t_te = tot(train), tot(test)
    fmt_src = ["{}", "{}", "{:,}", "{:,}", "{:.2f}", "{:.3f}"]
    parts = [
        "# Earshot data pack - summary",
        "",
        f"Written by `python -m dataprep.verify` on {time.strftime('%Y-%m-%d %H:%M')} ({mode}, {rep.decoded:,} files "
        f"decoded, {seconds:.0f} s). Free disk {free_gb():.1f} GB.",
        "",
        f"**Result - {'PASS' if not rep.violations else 'FAIL'}** - {len(rep.violations)} violations, "
        f"{len(rep.warnings)} warnings.",
        "",
        f"Train + val - {t_tr[0]:,} files, {t_tr[1]:.1f} h, {t_tr[2]:.2f} GB. "
        f"Test / eval - {t_te[0]:,} files, {t_te[1]:.1f} h, {t_te[2]:.2f} GB.",
        "",
        "## Per source and split",
        "",
        table(["source", "split", "files", "speakers", "hours", "GB"], src_rows, fmt_src),
        "",
        "## Per manifest",
        "",
        table(["manifest", "files", "speakers", "hours", "GB"], man_rows, ["{}", "{:,}", "{:,}", "{:.2f}", "{:.3f}"]),
        "",
    ]
    if rir_rows:
        parts += ["## RIR banks (data/rir)", "",
                  table(["bank", "split", "rows", "shape", "GB", "licenses (rows)"], rir_rows,
                        ["{}", "{}", "{:,}", "{}", "{:.3f}", "{}"]), ""]
    if def_counts:
        parts += ["## Eval definition files", ""]
        parts += [f"- `data/eval/{k}` - {v:,} lines, every referenced file checked" for k, v in def_counts.items()]
        parts += [""]
    for title, items in (("Violations", rep.violations), ("Warnings", rep.warnings)):
        parts += [f"## {title} ({len(items)})", ""]
        parts += [f"- {x}" for x in items[:200]] or ["- none"]
        if len(items) > 200:
            parts.append(f"- ... and {len(items) - 200} more")
        parts.append("")
    text = "\n".join(parts)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out)
    return text


def run(data: Path = DATA, mdir: Path = MANIFESTS, licenses_md: Path = LICENSES_MD, full: bool = False,
        sample: float = 0.1, workers: int = 8, summary: Path | None = None) -> tuple[Report, str]:
    t0 = time.time()
    rep = Report()
    if not mdir.is_dir():
        rep.bad(f"{mdir} does not exist")
        return rep, ""
    load_manifests(mdir, rep)
    rep.entries = [(n, e) for n, e in rep.entries if check_schema(n, e, rep)]
    known = license_ids(licenses_md.read_text(encoding="utf-8")) if licenses_md.exists() else set()
    if not known:
        rep.bad(f"no license ids found in {licenses_md}")
    check_files(rep, data, full, sample, workers)
    check_speakers(rep)
    check_licenses(rep, known)
    def_counts = check_definitions(rep, data)
    rir_rows = check_rir_banks(rep, data, known)
    src_rows, man_rows = summarize(rep)
    mode = "full decode" if full else f"eval files + {sample:.0%} sample decoded"
    text = write_summary(rep, src_rows, man_rows, def_counts, summary or (mdir / "SUMMARY.md"), mode, time.time() - t0,
                         rir_rows)
    return rep, text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="decode every file, not a 10 %% sample")
    ap.add_argument("--sample", type=float, default=0.1, help="fraction of train/val files to decode (default 0.1)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    rep, text = run(full=args.full, sample=args.sample, workers=args.workers)
    print(text)
    print(f"\n{len(rep.violations)} violations, {len(rep.warnings)} warnings - summary in "
          f"{(MANIFESTS / 'SUMMARY.md').relative_to(ROOT)}")
    return 1 if rep.violations else 0


if __name__ == "__main__":
    sys.exit(main())
