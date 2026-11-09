"""Earshot evaluation pack (data/eval, about 1.2 GB) - every set is evaluation only (split "test", use ["eval"]).

Run all sets, or some of them:

    cd earshot   # the repo root
    .venv/bin/python -m dataprep.eval_fetch                      # all sets, in the order below
    .venv/bin/python -m dataprep.eval_fetch --only wham edacc    # just these

Sets -
  librispeech  manifest for LibriSpeech test-clean, already on disk (40 speakers, nothing converted)
  wham         WHAM! noise "tt" clips used by Libri2Mix test, read out of the remote 18 GB zip, channel 0
               (CC BY-NC 4.0 - NON-COMMERCIAL, EVALUATION ONLY)
  libri2mix    data/eval/libri2mix_test.jsonl - Libri2Mix test "min" definitions (sources, gains, offsets, noise,
               enrollment), not rendered audio; see eval_libri2mix.py for how to render them
  edacc        EdAcc test segments, all 60 speakers (eval_hf.py)
  voxpopuli    VoxPopuli en_accented test segments of speakers unseen in VoxPopuli train (eval_hf.py)
  dipco        DiPCo dinner-party recordings, a streamed partial subset (eval_dipco.py)
VOiCES is documented but not fetched (see voices_note below and dataprep/README.md).

Every set is idempotent and resumable - finished files are skipped (flac_ok + manifest), and network errors are
retried with backoff. When free disk drops below the 12 GB floor the run stops with a "STOP disk floor" log line
and exit code 3. Logs - data/logs/eval_fetch.log.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import DATA, MANIFESTS, DiskFull, ManifestWriter, check_disk, entry, flac_ok, free_gb, get_logger, \
    read_manifest, write_flac
from .eval_libri2mix import build_definition, parse_info_csv, parse_mixture_csv, utt_id
from .eval_remote import RangeFile, get_bytes, make_session

log = get_logger("eval_fetch")

EVAL = DATA / "eval"
LIBRISPEECH = DATA / "LibriSpeech"
EVAL_BUDGET_GB = 1.2

LIBRIMIX_REV = "b898ca375c03e4039cdb3dbe3dd485aad58ec7d1"   # JorisCos/LibriMix master on 2026-11-07
LIBRIMIX_RAW = f"https://raw.githubusercontent.com/JorisCos/LibriMix/{LIBRIMIX_REV}/"
LIBRIMIX_FILES = {
    "mix": "metadata/Libri2Mix/libri2mix_test-clean.csv",
    "info": "metadata/Libri2Mix/libri2mix_test-clean_info.csv",
    "wham": "metadata/Wham_noise/test.csv",
    "librispeech": "metadata/LibriSpeech/test-clean.csv",
    "license": "LICENSE",
}

WHAM_URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
WHAM_LICENSE = "CC-BY-NC-4.0"
WHAM_DIR = "eval/wham_tt"


# ---------------------------------------------------------------------------------------------------------------------
# LibriSpeech test-clean (already on disk)
# ---------------------------------------------------------------------------------------------------------------------

def librispeech_speakers(path: Path) -> dict[str, tuple[str, str]]:
    """SPEAKERS.TXT -> {speaker id: (gender, subset)}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        out[parts[0]] = (parts[1], parts[2])
    return out


def librispeech_transcripts(subset_dir: Path) -> dict[str, str]:
    out = {}
    for t in subset_dir.glob("*/*/*.trans.txt"):
        for line in t.read_text(encoding="utf-8").splitlines():
            if line.strip():
                u, text = line.split(" ", 1)
                out[u] = text.strip()
    return out


def do_librispeech() -> None:
    subset = LIBRISPEECH / "test-clean"
    if not subset.is_dir():
        raise FileNotFoundError(f"{subset} is missing; download test-clean.tar.gz from https://www.openslr.org/12/")
    spk = librispeech_speakers(LIBRISPEECH / "SPEAKERS.TXT")
    text = librispeech_transcripts(subset)
    n_new = 0
    with ManifestWriter("eval_librispeech_test.jsonl") as mw:
        for f in sorted(subset.glob("*/*/*.flac")):
            rel = f.relative_to(DATA).as_posix()
            if rel in mw:
                continue
            dur = flac_ok(rel)
            if dur is None:
                log.warning("librispeech: %s does not decode as 16 kHz mono, skipped", rel)
                continue
            s, chapter, u = f.parent.parent.name, f.parent.name, f.name.split(".")[0]
            mw.add(entry(rel, "librispeech", dur, "test", "CC-BY-4.0", ["eval"], speaker=f"librispeech:{s}",
                         gender=spk.get(s, (None,))[0], accent=None, group=chapter, utt=u, text=text.get(u),
                         subset="test-clean"))
            n_new += 1
    log.info("librispeech: %d new entries (test-clean)", n_new)


# ---------------------------------------------------------------------------------------------------------------------
# LibriMix metadata (MIT) - cached under data/eval/librimix/
# ---------------------------------------------------------------------------------------------------------------------

def librimix_csvs() -> dict[str, str]:
    out_dir = EVAL / "librimix"
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = make_session()
    texts = {}
    for key, rel in LIBRIMIX_FILES.items():
        local = out_dir / Path(rel).name
        if not local.exists():
            data = get_bytes(sess, LIBRIMIX_RAW + rel, log)
            tmp = local.with_suffix(local.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, local)
        texts[key] = local.read_text(encoding="utf-8")
    return texts


# ---------------------------------------------------------------------------------------------------------------------
# WHAM! noise tt clips (CC BY-NC 4.0, evaluation only)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Span:
    """One HTTP read covering neighbouring zip members."""
    start: int
    end: int
    members: list = field(default_factory=list)


def plan_spans(infos: list[zipfile.ZipInfo], max_gap: int = 64 << 10, max_span: int = 48 << 20,
               slack: int = 1024) -> list[Span]:
    """Group zip members (by local-header offset) into reads of at most max_span bytes. Members are read together
    when the gap between them is at most max_gap. slack covers the local header's name and extra field."""
    spans: list[Span] = []
    for i in sorted(infos, key=lambda i: i.header_offset):
        end = i.header_offset + 30 + len(i.filename.encode()) + len(i.extra) + slack + i.compress_size
        if spans and i.header_offset - spans[-1].end <= max_gap and end - spans[-1].start <= max_span:
            spans[-1].end = max(spans[-1].end, end)
            spans[-1].members.append(i)
        else:
            spans.append(Span(i.header_offset, end, [i]))
    return spans


def wham_rel(name: str) -> str:
    return f"{WHAM_DIR}/{Path(name).stem}.flac"


def parse_wham_noise_meta(text: str) -> dict[str, dict]:
    """noise_meta_tt.csv -> {file name: {"recording", "location", "day", "noise_band", "reverb"}}."""
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        out[r["utterance_id"]] = {"recording": r["File ID"], "location": r["Location ID"], "day": r["Location Day ID"],
                                  "noise_band": r["Noise Band"], "reverb": r["Reverberation Level"],
                                  "lr_width": r["L to R Width (cm)"]}
    return out


def do_wham() -> None:
    csvs = librimix_csvs()
    need = sorted({Path(r["noise"][0]).name for r in parse_mixture_csv(csvs["mix"])})
    with ManifestWriter("eval_wham_tt.jsonl") as mw:
        todo = [n for n in need if not (wham_rel(n) in mw and flac_ok(wham_rel(n)))]
        log.info("wham: Libri2Mix test uses %d distinct tt clips, %d still to fetch", len(need), len(todo))
        if not todo:
            return
        check_disk()
        rf = RangeFile(WHAM_URL, sess=make_session(), log=log)
        zf = zipfile.ZipFile(rf)
        infos = {Path(i.filename).name: i for i in zf.infolist() if "/tt/" in i.filename and i.filename.endswith(".wav")}
        meta_dir = EVAL / "wham_tt"
        meta_dir.mkdir(parents=True, exist_ok=True)
        for doc in ("README_metadata.md", "noise_meta_tt.csv"):
            if not (meta_dir / doc).exists():
                (meta_dir / doc).write_bytes(zf.read(f"wham_noise/metadata/{doc}"))
        meta = parse_wham_noise_meta((meta_dir / "noise_meta_tt.csv").read_text())
        missing = [n for n in todo if n not in infos]
        if missing:
            raise RuntimeError(f"{len(missing)} needed tt clips are not in the zip, e.g. {missing[:3]}")
        spans = plan_spans([infos[n] for n in todo])
        t0, done = time.time(), 0
        for k, span in enumerate(spans):
            check_disk()
            rf.prefetch(span.start, span.end)
            for info in span.members:
                name = Path(info.filename).name
                x, sr = sf.read(io.BytesIO(zf.read(info)), dtype="float32", always_2d=True)
                if sr != 16000:
                    raise RuntimeError(f"{name} is {sr} Hz, expected 16 kHz")
                y = x[:, 0]                                   # LibriMix uses channel 0 of the binaural clip
                peak = float(np.max(np.abs(y)))
                scale = 1.0 if peak <= 0.999 else 0.999 / peak
                rel = wham_rel(name)
                write_flac(rel, y * scale)
                m = meta.get(name, {})
                mw.add(entry(rel, "wham", len(y) / 16000, "test", WHAM_LICENSE, ["eval"], group=m.get("recording"),
                             wham_id=f"tt/{name}", channel=0, scale=round(scale, 6), eval_only=True,
                             location=m.get("location"), day=m.get("day"), noise_band=m.get("noise_band"),
                             reverb=m.get("reverb")))
                done += 1
            if k % 25 == 0 or k == len(spans) - 1:
                el = time.time() - t0
                log.info("wham: %d/%d clips, %.0f MB downloaded, %.1f MB/s", done, len(todo), rf.fetched / 1e6,
                         rf.fetched / 1e6 / max(el, 1e-6))


# ---------------------------------------------------------------------------------------------------------------------
# Libri2Mix test definitions
# ---------------------------------------------------------------------------------------------------------------------

def do_libri2mix() -> None:
    csvs = librimix_csvs()
    rows = parse_mixture_csv(csvs["mix"])
    info = parse_info_csv(csvs["info"])
    lengths, by_spk = {}, {}
    for f in sorted((LIBRISPEECH / "test-clean").glob("*/*/*.flac")):
        u = f.name.split(".")[0]
        n = sf.info(f).frames
        lengths[u] = n
        by_spk.setdefault(u.split("-")[0], {})[u] = n / 16000
    wham = {}
    wm = MANIFESTS / "eval_wham_tt.jsonl"
    if wm.exists():
        for e in read_manifest(wm):
            wham[e["extra"]["wham_id"]] = e
    defs, no_noise = [], 0
    for r in rows:
        missing = [p for p, _ in r["sources"] if utt_id(p) not in lengths]
        if missing:
            raise RuntimeError(f"LibriSpeech test-clean file(s) missing for {r['id']}: {missing}")
        name = Path(r["noise"][0]).name
        e = wham.get(f"tt/{name}")
        if e is None:
            no_noise += 1
        scale = e["extra"].get("scale", 1.0) if e else 1.0
        meta = {"recording": e["group"], "clip_samples": sf.info(DATA / e["path"]).frames} if e else None
        defs.append(build_definition(r, info.get(r["id"]), lengths, by_spk, wham_rel(name), scale, meta))
    out = EVAL / "libri2mix_test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(d) + "\n" for d in defs), encoding="utf-8")
    os.replace(tmp, out)
    hours = sum(d["length"] for d in defs) / 16000 / 3600
    log.info("libri2mix: wrote %d definitions (%.2f h in min mode) to %s; %d still lack their WHAM! clip",
             len(defs), hours, out.relative_to(DATA), no_noise)


# ---------------------------------------------------------------------------------------------------------------------
# VOiCES - documented, not fetched
# ---------------------------------------------------------------------------------------------------------------------

def voices_note() -> None:
    log.info("voices: not fetched - S3 bucket lab41openaudiocorpus holds only tar.gz archives (devkit 29.5 GB, "
             "competition 20.9 GB, release 449 GB). gzip cannot be seeked, and the devkit stream starts with "
             "room-response and distractor-only recordings (the first 1.58 GB read were tones, swoops, impulses and "
             "105 MB-per-mic TV-distractor tracks), so a speech subset would need most of the archive. "
             "See dataprep/README.md.")


# ---------------------------------------------------------------------------------------------------------------------

def dir_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9 if path.exists() else 0.0


def main(argv: list[str] | None = None) -> int:
    from .eval_dipco import fetch_dipco
    from .eval_hf import fetch_edacc, fetch_voxpopuli

    steps = {"librispeech": do_librispeech, "wham": do_wham, "libri2mix": do_libri2mix,
             "edacc": lambda: fetch_edacc(log), "voxpopuli": lambda: fetch_voxpopuli(log),
             "dipco": lambda: fetch_dipco(log), "voices": voices_note}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(steps), help="run only these sets (default: all)")
    ap.add_argument("--skip", nargs="+", choices=list(steps), default=[], help="skip these sets")
    args = ap.parse_args(argv)
    todo = [s for s in (args.only or list(steps)) if s not in args.skip]
    if "libri2mix" in todo and "wham" in todo and todo.index("libri2mix") < todo.index("wham"):
        todo.remove("libri2mix")
        todo.insert(todo.index("wham") + 1, "libri2mix")
    stop = {"asked": False}

    def on_signal(signum, frame):
        # nohup'd background jobs ignore SIGINT by default; this makes kill -INT / kill -TERM stop cleanly (temp
        # files removed, manifests flushed) instead of killing the process mid-write.
        stop["asked"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    log.info("eval_fetch start: %s (%.1f GB free, data/eval holds %.2f GB)", " ".join(todo), free_gb(), dir_gb(EVAL))
    failed = []
    for name in todo:
        t0 = time.time()
        try:
            steps[name]()
            log.info("%s: done in %.0f s", name, time.time() - t0)
        except DiskFull as e:
            log.error("STOP disk floor - %s (during %s). Free some space, then re-run the same command to resume.",
                      e, name)
            return 3
        except KeyboardInterrupt:
            log.error("STOP asked by a signal during %s - re-run the same command to resume", name)
            return 130
        except Exception:
            if stop["asked"]:   # a signal that arrived inside pyarrow comes back wrapped in another exception
                log.error("STOP asked by a signal during %s - re-run the same command to resume", name)
                return 130
            log.exception("%s: FAILED (the other sets still run; re-run to retry)", name)
            failed.append(name)
    gb = dir_gb(EVAL)
    log.info("eval_fetch end: data/eval holds %.2f GB (budget %.1f GB)%s; failed: %s", gb, EVAL_BUDGET_GB,
             " - OVER BUDGET" if gb > EVAL_BUDGET_GB else "", ", ".join(failed) or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
