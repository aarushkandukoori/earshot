"""EdAcc test and VoxPopuli en_accented test as speaker-labelled 16 kHz FLAC segments (evaluation only).

Neither set is downloaded whole. pyarrow reads each parquet footer and the small metadata columns through HTTP
range requests (eval_remote.RangeFile); we choose rows from that; then only the row groups that hold chosen rows
have their audio column fetched (100 rows per row group for EdAcc, 1,000 for VoxPopuli).

EdAcc (CC BY-SA 4.0) - 60 test speakers in 30 Zoom conversations, spread over 10 shards that are ordered by
conversation. Instead of 1-2 whole shards (16 speakers) we fetch, for every speaker, the row group(s) holding at least
EDACC_MIN_ROWS of their segments. That costs about the same download as 2 shards and covers all 60 speakers and all
24 accent labels. Kept per speaker - up to EDACC_CAP_S seconds of segments 1 to 30 s long, in corpus order.

VoxPopuli en_accented (CC0) - 364 test speakers, but 238 of them also speak in VoxPopuli's English train shards
and 27 in its validation split (speaker ids are the same across configs, measured 2026-11-07). Our training pack
uses VoxPopuli train shards, so we keep only speakers that appear in no English train or validation shard and that
have at least 2 segments (target + enrollment), up to VP_CAP_S seconds each. Rows are shuffled across row groups, so
this reads all 10 row groups of the 2 shards (about 4.9 GB of download for about 1.6 h kept).
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from .common import DATA, ManifestWriter, check_disk, entry, flac_ok, is_silent, read_manifest, resample, to_mono, \
    write_flac
from .eval_remote import RangeFile, hf_url, make_session

EDACC_REPO, EDACC_REV = "edinburghcstr/edacc", "d9ae7bd344f0562b766ec93ee5ce8f2f9568ce66"
EDACC_SHARDS = [
    "data/test-00000-of-00010-f0aceb1ca4406ff1.parquet", "data/test-00001-of-00010-856b016d9d438ff3.parquet",
    "data/test-00002-of-00010-2b021baedb4deb8a.parquet", "data/test-00003-of-00010-4de275e704375a02.parquet",
    "data/test-00004-of-00010-806407c9bc68112a.parquet", "data/test-00005-of-00010-9c97c4c4c8d01f82.parquet",
    "data/test-00006-of-00010-cc4648d0f66f65a4.parquet", "data/test-00007-of-00010-ea5ed4464ecff3c9.parquet",
    "data/test-00008-of-00010-d1aa19b51ad423da.parquet", "data/test-00009-of-00010-648ce5002d124496.parquet",
]
EDACC_CAP_S = 180.0
EDACC_MIN_S, EDACC_MAX_S = 1.0, 30.0
EDACC_MIN_ROWS = 20
EDACC_LICENSE = "CC-BY-SA-4.0"

VP_REPO, VP_REV = "facebook/voxpopuli", "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
VP_ACCENTED = ["en_accented/test-00000-of-00002.parquet", "en_accented/test-00001-of-00002.parquet"]
VP_SEEN = [f"en/train-{i:05d}-of-00030.parquet" for i in range(30)] + ["en/validation-00000-of-00001.parquet"]
VP_CAP_S = 300.0
VP_MIN_S, VP_MAX_S = 2.0, 60.0
VP_LICENSE = "CC0-1.0"
VP_ACCENT_NAMES = {"en_cs": "Czech", "en_de": "German", "en_es": "Spanish", "en_et": "Estonian", "en_fi": "Finnish",
                   "en_fr": "French", "en_hr": "Croatian", "en_hu": "Hungarian", "en_it": "Italian",
                   "en_lt": "Lithuanian", "en_nl": "Dutch", "en_pl": "Polish", "en_ro": "Romanian", "en_sk": "Slovak",
                   "en_sl": "Slovene"}

GENDER = {"male": "M", "female": "F", "m": "M", "f": "F"}


# ---------------------------------------------------------------------------------------------------------------------
# Small pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def edacc_conversation(speaker: str) -> str:
    """'EDACC-C08-A' -> 'C08'"""
    return speaker.split("-")[1]


def edacc_text_ok(text: str | None) -> bool:
    """EdAcc marks stretches that ASR scoring should skip with IGNORE_TIME_SEGMENT_IN_SCORING; we skip them too."""
    return bool(text) and "IGNORE_TIME_SEGMENT_IN_SCORING" not in text


def choose_row_groups(rows: list[dict], audio_bytes: dict[tuple[int, int], int], min_rows: int) -> list[tuple[int, int]]:
    """Fewest row groups so every speaker has min(min_rows, all their rows) rows in them.

    rows - dicts with "shard", "rg", "speaker". For each speaker (sorted) we add the row group holding most of their
    rows; ties go to the group with more rows overall, then the smaller download, then the earlier group."""
    per_rg: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for r in rows:
        per_rg[(r["shard"], r["rg"])][r["speaker"]] += 1
    total = Counter(r["speaker"] for r in rows)
    chosen: list[tuple[int, int]] = []
    have: Counter = Counter()
    for spk in sorted(total):
        need = min(min_rows, total[spk])
        while have[spk] < need:
            cands = [k for k in per_rg if k not in chosen and per_rg[k][spk] > 0]
            best = max(cands, key=lambda k: (per_rg[k][spk], sum(per_rg[k].values()), -audio_bytes.get(k, 0),
                                             -k[0], -k[1]))
            chosen.append(best)
            have.update(per_rg[best])
    return sorted(chosen)


def vp_safe_name(audio_id: str) -> str:
    """VoxPopuli ids contain ':' (clock times); keep file names portable."""
    return audio_id.replace(":", "-").replace("/", "_")


def vp_session(audio_id: str) -> str:
    """'20111116-0900-PLENARY-8-en_20111116-16:02:33_0' -> '20111116-0900-PLENARY-8' (one plenary session)."""
    return audio_id.split("-en_")[0] if "-en_" in audio_id else audio_id


def select_unseen(rows: list[dict], seen: set[str], min_rows: int = 2) -> list[dict]:
    """Rows whose speaker never appears in `seen` and has at least min_rows rows."""
    keep = [r for r in rows if r["speaker"] not in seen and r["speaker"] not in ("", "None", None)]
    n = Counter(r["speaker"] for r in keep)
    return [r for r in keep if n[r["speaker"]] >= min_rows]


def speaker_seconds(manifest: Path) -> Counter:
    """Seconds already kept per speaker (for the per-speaker caps when a run resumes)."""
    acc: Counter = Counter()
    if manifest.exists():
        for e in read_manifest(manifest):
            acc[e["speaker"]] += e["duration"]
    return acc


# ---------------------------------------------------------------------------------------------------------------------
# Parquet access
# ---------------------------------------------------------------------------------------------------------------------

class Shards:
    """Open remote parquet files lazily and keep them open (the footer is read once per shard)."""

    def __init__(self, urls: list[str], log: logging.Logger):
        self.urls, self.log = urls, log
        self.sess = make_session()
        self._pf: dict[int, tuple[RangeFile, pq.ParquetFile]] = {}

    def get(self, i: int) -> pq.ParquetFile:
        if i not in self._pf:
            rf = RangeFile(self.urls[i], sess=self.sess, log=self.log)
            # buffer_size makes pyarrow read a column chunk in 8 MB pieces instead of all at once, so a 580 MB
            # VoxPopuli audio chunk never sits in memory whole
            self._pf[i] = (rf, pq.ParquetFile(rf, buffer_size=8 << 20, pre_buffer=False))
        return self._pf[i][1]

    def fetched(self) -> int:
        return sum(rf.fetched for rf, _ in self._pf.values())


def read_meta(shards: Shards, i: int, columns: list[str]) -> list[dict]:
    """Metadata columns of shard i as row dicts with "shard", "rg" and "row" (index inside the row group)."""
    pf = shards.get(i)
    tb = pf.read(columns=columns).to_pylist()
    out, k = [], 0
    for rg in range(pf.metadata.num_row_groups):
        for j in range(pf.metadata.row_group(rg).num_rows):
            r = tb[k]
            if isinstance(r.get("audio"), dict):  # nested audio.path column
                r["audio_path"] = r.pop("audio")["path"]
            r.update(shard=i, rg=rg, row=j)
            out.append(r)
            k += 1
    return out


def audio_bytes_per_rg(shards: Shards, i: int) -> dict[tuple[int, int], int]:
    md = shards.get(i).metadata
    out = {}
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        out[(i, rg)] = sum(g.column(c).total_compressed_size for c in range(g.num_columns)
                           if g.column(c).path_in_schema == "audio.bytes")
    return out


def iter_rg_audio(shards: Shards, shard: int, rg: int, batch: int = 32):
    """Yield (row index, audio bytes) for every row of one row group, in order, 32 rows at a time (the only
    large download; memory stays at one batch)."""
    k = 0
    for b in shards.get(shard).iter_batches(batch_size=batch, row_groups=[rg], columns=["audio.bytes"]):
        col = b.column(0).field("bytes")
        for j in range(len(col)):
            yield k, col[j].as_py()
            k += 1


def decode_16k(data: bytes) -> tuple[np.ndarray, int]:
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = resample(to_mono(x), sr)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:  # float WAVs can exceed full scale; keep the waveform, lose only gain
        y = y * (0.999 / peak)
    return y.astype(np.float32), sr


# ---------------------------------------------------------------------------------------------------------------------
# Shared row-group loop
# ---------------------------------------------------------------------------------------------------------------------

def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"done_rgs": []}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, path)


def process(rows: list[dict], shards: Shards, mw: ManifestWriter, acc: Counter, cap_s: float, min_s: float,
            max_s: float, make_entry, state_path: Path, log: logging.Logger, tag: str) -> dict:
    """Write the chosen rows as FLAC, one row group at a time, in (shard, row group, row) order.

    rows        - chosen rows, each with "rel" (output path under data/), "speaker", "shard", "rg", "row".
    acc         - seconds already kept per speaker id (updated in place); a speaker stops at cap_s.
    make_entry  - f(row, duration, orig_sr) -> manifest entry.
    Row groups listed in state["done_rgs"] are skipped without downloading, so a resumed run is cheap."""
    state = _load_state(state_path)
    done_rgs = set(state.get("done_rgs", []))
    by_rg: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_rg[(r["shard"], r["rg"])].append(r)
    stats = Counter()
    for key in sorted(by_rg):
        rkey = f"{key[0]}/{key[1]}"
        if rkey in done_rgs:
            continue
        group = sorted(by_rg[key], key=lambda r: r["row"])
        pending = [r for r in group if not (r["rel"] in mw and flac_ok(r["rel"]))
                   and acc[r["speaker_id"]] < cap_s]
        if pending:
            check_disk()
            log.info("%s: shard %d row group %d - fetching audio for %d rows", tag, key[0], key[1], len(pending))
            wanted = {r["row"]: r for r in group}
            for k, data in iter_rg_audio(shards, key[0], key[1]):
                r = wanted.get(k)
                if r is None or (r["rel"] in mw and flac_ok(r["rel"])):
                    continue
                if acc[r["speaker_id"]] >= cap_s:
                    stats["capped"] += 1
                    continue
                if not data:
                    stats["no_audio"] += 1
                    continue
                try:
                    y, sr = decode_16k(data)
                except Exception as e:
                    log.warning("%s: cannot decode %s (%s)", tag, r["rel"], e)
                    stats["bad_audio"] += 1
                    continue
                dur = len(y) / 16000
                if dur < min_s or dur > max_s:
                    stats["length"] += 1
                    continue
                if is_silent(y):
                    stats["silent"] += 1
                    continue
                write_flac(r["rel"], y)
                mw.add(make_entry(r, dur, sr))
                acc[r["speaker_id"]] += dur
                stats["written"] += 1
        done_rgs.add(rkey)
        state["done_rgs"] = sorted(done_rgs)
        _save_state(state_path, state)
        pa.default_memory_pool().release_unused()  # hand the row group's buffers back to the OS
    log.info("%s: %s, %.1f MB downloaded", tag, dict(stats), shards.fetched() / 1e6)
    return dict(stats)


# ---------------------------------------------------------------------------------------------------------------------
# EdAcc
# ---------------------------------------------------------------------------------------------------------------------

def fetch_edacc(log: logging.Logger) -> None:
    out_dir = DATA / "eval" / "edacc"
    meta_cache = out_dir / "test_metadata.json"
    shards = Shards([hf_url(EDACC_REPO, s, EDACC_REV) for s in EDACC_SHARDS], log)
    if meta_cache.exists():
        cache = json.loads(meta_cache.read_text())
        rows, sizes = cache["rows"], {tuple(map(int, k.split("/"))): v for k, v in cache["audio_bytes"].items()}
    else:
        cols = ["speaker", "text", "accent", "raw_accent", "gender", "l1", "audio.path"]
        with ThreadPoolExecutor(5) as ex:
            parts = list(ex.map(lambda i: read_meta(shards, i, cols), range(len(EDACC_SHARDS))))
        rows = [r for p in parts for r in p]
        sizes = {}
        for i in range(len(EDACC_SHARDS)):
            sizes.update(audio_bytes_per_rg(shards, i))
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_cache.write_text(json.dumps({"repo": EDACC_REPO, "revision": EDACC_REV, "rows": rows,
                                          "audio_bytes": {f"{k[0]}/{k[1]}": v for k, v in sizes.items()}}))
    usable = [r for r in rows if edacc_text_ok(r["text"])]
    groups = choose_row_groups(usable, sizes, EDACC_MIN_ROWS)
    chosen = set(groups)
    stems = Counter(Path(r["audio_path"]).stem for r in usable)
    picked = []
    for r in usable:
        if (r["shard"], r["rg"]) not in chosen:
            continue
        stem = Path(r["audio_path"]).stem
        if stems[stem] > 1:
            stem = f"{stem}_s{r['shard']}r{r['rg']}-{r['row']}"
        r = dict(r, speaker_id=f"edacc:{r['speaker']}", rel=f"eval/edacc/{r['speaker']}/{stem}.flac")
        picked.append(r)
    mb = sum(sizes[k] for k in groups) / 1e6
    log.info("edacc: %d test rows, %d speakers; %d row groups chosen (%.0f MB of audio to download), %d rows offered",
             len(rows), len({r['speaker'] for r in rows}), len(groups), mb, len(picked))

    def make_entry(r: dict, dur: float, sr: int) -> dict:
        return entry(r["rel"], "edacc", dur, "test", EDACC_LICENSE, ["eval"], speaker=r["speaker_id"],
                     gender=GENDER.get((r["gender"] or "").lower()), accent=r["accent"],
                     group=edacc_conversation(r["speaker"]), text=r["text"], l1=r["l1"], raw_accent=r["raw_accent"],
                     orig_sr=sr, hf=f"{EDACC_REPO}@{EDACC_REV[:12]}:{EDACC_SHARDS[r['shard']]}#rg{r['rg']}:{r['row']}")

    manifest = "eval_edacc_test.jsonl"
    acc = speaker_seconds(DATA / "manifests" / manifest)
    with ManifestWriter(manifest) as mw:
        process(picked, shards, mw, acc, EDACC_CAP_S, EDACC_MIN_S, EDACC_MAX_S, make_entry, out_dir / "state.json",
                log, "edacc")


# ---------------------------------------------------------------------------------------------------------------------
# VoxPopuli en_accented
# ---------------------------------------------------------------------------------------------------------------------

def vp_seen_speakers(log: logging.Logger, cache: Path) -> set[str]:
    """Every speaker id in VoxPopuli's English train (30 shards) and validation split (only that column is read)."""
    if cache.exists():
        d = json.loads(cache.read_text())
        if d.get("revision") == VP_REV:
            return set(d["speakers"])
    sess = make_session()

    def one(path: str) -> list[str]:
        rf = RangeFile(hf_url(VP_REPO, path, VP_REV), sess=sess, log=log)
        return pq.ParquetFile(rf).read(columns=["speaker_id"]).column("speaker_id").to_pylist()

    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(one, VP_SEEN))
    seen = sorted({s for p in parts for s in p if s})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"repo": VP_REPO, "revision": VP_REV, "files": VP_SEEN, "speakers": seen}))
    log.info("voxpopuli: %d speakers in en train + validation", len(seen))
    return set(seen)


def fetch_voxpopuli(log: logging.Logger) -> None:
    out_dir = DATA / "eval" / "voxpopuli_accented"
    seen = vp_seen_speakers(log, out_dir / "seen_speakers.json")
    shards = Shards([hf_url(VP_REPO, s, VP_REV) for s in VP_ACCENTED], log)
    meta_cache = out_dir / "test_metadata.json"
    if meta_cache.exists():
        rows = json.loads(meta_cache.read_text())["rows"]
    else:
        cols = ["audio_id", "raw_text", "normalized_text", "gender", "speaker_id", "is_gold_transcript", "accent"]
        rows = [r for i in range(len(VP_ACCENTED)) for r in read_meta(shards, i, cols)]
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_cache.write_text(json.dumps({"repo": VP_REPO, "revision": VP_REV, "rows": rows}))
    for r in rows:
        r["speaker"] = r["speaker_id"]
    picked = []
    for r in select_unseen(rows, seen):
        acc_code = r["accent"]
        picked.append(dict(r, speaker_id=f"voxpopuli:{r['speaker']}",
                           rel=f"eval/voxpopuli_accented/{acc_code}/{r['speaker']}/{vp_safe_name(r['audio_id'])}.flac"))
    log.info("voxpopuli: %d en_accented test rows, %d speakers; %d unseen speakers with 2+ rows, %d rows offered",
             len(rows), len({r['speaker'] for r in rows}), len({r['speaker'] for r in picked}), len(picked))

    def make_entry(r: dict, dur: float, sr: int) -> dict:
        return entry(r["rel"], "voxpopuli", dur, "test", VP_LICENSE, ["eval"], speaker=r["speaker_id"],
                     gender=GENDER.get((r["gender"] or "").lower()), accent=r["accent"],
                     group=vp_session(r["audio_id"]), text=r["raw_text"], normalized_text=r["normalized_text"],
                     gold_transcript=r["is_gold_transcript"], l1=VP_ACCENT_NAMES.get(r["accent"]),
                     audio_id=r["audio_id"], orig_sr=sr,
                     hf=f"{VP_REPO}@{VP_REV[:12]}:{VP_ACCENTED[r['shard']]}#rg{r['rg']}:{r['row']}")

    manifest = "eval_voxpopuli_accented.jsonl"
    acc = speaker_seconds(DATA / "manifests" / manifest)
    with ManifestWriter(manifest) as mw:
        process(picked, shards, mw, acc, VP_CAP_S, VP_MIN_S, VP_MAX_S, make_entry, out_dir / "state.json", log,
                "voxpopuli")
