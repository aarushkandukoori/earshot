"""Helpers shared by the speech fetchers (LibriTTS-R, VoxPopuli, GLOBE V2, VCTK).

What happens to every clip we keep
  1. decode it at its native rate,
  2. reject it if it is clipped or silent (common.is_clipped / common.is_silent),
  3. resample to 16 kHz mono (soxr VHQ) and scale it down if resampling pushed the peak over 0.999,
  4. write 16-bit FLAC to data/speech/<source>/<speaker>/<file>.flac (atomic, checks the 12 GB disk floor),
  5. add one line to data/manifests/speech_<split>.jsonl (schema in common.py).

Re-running is safe. A file that already decodes (flac_ok) is never written again, a path already in a manifest is
never added again, and finished speakers / shards / tarballs are listed in small state files under
data/speech/<source>/_state/ so a restarted run skips them.

Disk rules. Each source has its own share of the 5.2 GB speech budget (BUDGET_GB) and stops when it is used up. When
free disk drops below the 12 GB floor we stop writing, wait up to 15 minutes for other jobs to free space (they often
delete temporary files), and stop cleanly with a DiskFull log line if it does not come back.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import (DATA, MANIFESTS, MIN_FREE_GB, SR, DiskFull, ManifestWriter, entry, flac_ok, free_gb, is_clipped,
                     is_silent, read_manifest, resample, to_mono, write_flac)

SPEECH = DATA / "speech"
RAW = DATA / "raw"
SPLITS = ("train", "val", "test")
USE = ["target", "interferer", "babble", "enrollment"]
LICENSES = {"librittsr": "CC-BY-4.0", "voxpopuli": "CC0-1.0", "globe": "CC0-1.0", "vctk": "CC-BY-4.0"}

# Share of the 5.2 GB speech budget per source, in GB of allocated disk. Plan section 1.3 expects about 2.6 / 0.9 /
# 0.9 / 0.25 GB; the shares leave room for FLAC compressing a little worse than the 68 MB/h measured on LibriSpeech.
BUDGET_GB = {"librittsr": 3.2, "voxpopuli": 0.6, "globe": 1.1, "vctk": 0.3}
TOTAL_BUDGET_GB = 5.2
DISK_WAIT_S = 15 * 60


class BudgetFull(RuntimeError):
    """This source used up its share of the speech disk budget."""


# ---------------------------------------------------------------------------------------------------------------
# small pure helpers


def stable_hash(key: str) -> int:
    """Deterministic pseudo-random number for a string (same on every machine and run)."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:12], 16)


def safe_name(s: str) -> str:
    """File-system safe id: 'a:b c' -> 'a-b-c'. VoxPopuli ids contain ':' which macOS tools treat badly."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(s).strip()).strip("-.")
    return s or "x"


def norm_gender(g) -> str | None:
    g = str(g or "").strip().lower()
    if g in ("f", "female", "woman"):
        return "F"
    if g in ("m", "male", "man"):
        return "M"
    return None


def norm_accent(a) -> str | None:
    a = str(a or "").strip()
    return None if a.lower() in ("", "none", "null", "nan", "unknown") else a


def prepare_clip(x: np.ndarray, sr: int) -> tuple[np.ndarray | None, str]:
    """Quality checks plus resampling. Returns (16 kHz mono float32, "ok") or (None, reason)."""
    x = to_mono(np.asarray(x, dtype=np.float32))
    if x.size == 0 or not np.isfinite(x).all():
        return None, "broken"
    if is_clipped(x):
        return None, "clipped"
    if is_silent(x):
        return None, "silent"
    y = resample(x, sr, SR)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:  # resampling can overshoot a little; scale down instead of letting the FLAC writer clip
        y = y * np.float32(0.999 / peak)
    return y.astype(np.float32), "ok"


def allocated_bytes(path: Path) -> int:
    return os.stat(path).st_blocks * 512


def tree_bytes(root: Path) -> int:
    """Allocated bytes of every file under root (what du counts)."""
    total = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            try:
                total += allocated_bytes(Path(dirpath) / f)
            except FileNotFoundError:
                pass
    return total


# ---------------------------------------------------------------------------------------------------------------
# disk floor and budget


def wait_for_disk(log: logging.Logger, need_gb: float = 0.0, wait_s: float = DISK_WAIT_S,
                  poll_s: float = 30.0) -> None:
    """Return once (free space - need_gb) is above the 12 GB floor; raise DiskFull after waiting wait_s seconds."""
    deadline = time.monotonic() + wait_s
    warned = False
    while True:
        gb = free_gb()
        if gb - need_gb >= MIN_FREE_GB:
            if warned:
                log.info("disk recovered (%.1f GB free), continuing", gb)
            return
        if time.monotonic() >= deadline:
            raise DiskFull(f"only {gb:.1f} GB free and {need_gb:.1f} GB needed, floor is {MIN_FREE_GB:.0f} GB")
        if not warned:
            log.warning("disk floor - %.1f GB free, %.1f GB needed above the %.0f GB floor; pausing up to %d min",
                        gb, need_gb, MIN_FREE_GB, wait_s // 60)
            warned = True
        time.sleep(poll_s)


def save_flac(rel: str, y: np.ndarray, log: logging.Logger) -> Path:
    """common.write_flac, but a short dip below the disk floor pauses instead of killing a multi-hour stream."""
    while True:
        try:
            return write_flac(rel, y)
        except DiskFull:
            wait_for_disk(log)


class Budget:
    """Disk budget of one source - bytes already under data/speech/<source> plus what this run writes."""

    def __init__(self, source: str, cap_gb: float | None = None):
        self.source = source
        self.cap = (BUDGET_GB[source] if cap_gb is None else cap_gb) * 1e9
        self.used = tree_bytes(SPEECH / source)
        self._lock = threading.Lock()

    def add(self, path: Path) -> None:
        with self._lock:
            self.used += allocated_bytes(path)

    def check(self) -> None:
        if self.used >= self.cap:
            raise BudgetFull(f"{self.source} used its {self.cap / 1e9:.2f} GB share of the speech budget")


# ---------------------------------------------------------------------------------------------------------------
# state and manifests


class DoneSet:
    """Append-only text file of finished keys (speakers, shards, tarballs), one per line. Thread-safe."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.keys = set(path.read_text().split()) if path.exists() else set()
        self._lock = threading.Lock()

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def add(self, key: str) -> None:
        assert key and not any(c.isspace() for c in key), key
        with self._lock:
            if key in self.keys:
                return
            with open(self.path, "a") as f:
                f.write(key + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.keys.add(key)


def state_dir(source: str) -> Path:
    return SPEECH / source / "_state"


class SpeechManifests:
    """speech_train.jsonl, speech_val.jsonl and speech_test.jsonl. add() routes an entry by its split; thread-safe."""

    def __init__(self):
        self.w = {s: ManifestWriter(f"speech_{s}.jsonl") for s in SPLITS}
        self._lock = threading.Lock()

    def __contains__(self, rel_path: str) -> bool:
        return any(rel_path in w for w in self.w.values())

    def add(self, e: dict) -> None:
        with self._lock:
            self.w[e["split"]].add(e)

    def close(self) -> None:
        for w in self.w.values():
            w.close()

    def __enter__(self) -> "SpeechManifests":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def source_entries(source: str) -> list[dict]:
    """Every manifest entry of one source, from all three speech manifests."""
    out = []
    for s in SPLITS:
        p = MANIFESTS / f"speech_{s}.jsonl"
        if p.exists():
            out += [e for e in read_manifest(p) if e.get("source") == source]
    return out


def seconds_per_speaker(entries: list[dict], only=None) -> defaultdict:
    """speaker id -> seconds already kept (only = optional filter on the entry)."""
    tot = defaultdict(float)
    for e in entries:
        if only is None or only(e):
            tot[e["speaker"]] += e["duration"]
    return tot


class SourceLock:
    """Stops two runs of the same source from writing at the same time (flock on data/speech/.<source>.lock)."""

    def __init__(self, source: str):
        self.path = SPEECH / f".{source}.lock"
        self.f = None

    def __enter__(self) -> "SourceLock":
        SPEECH.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.f.close()
            raise RuntimeError(f"another run is already writing this source (lock file {self.path})")
        return self

    def __exit__(self, *exc) -> None:
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


# ---------------------------------------------------------------------------------------------------------------
# storing clips


@dataclass
class Ctx:
    """Everything a fetcher needs while it runs."""
    source: str
    log: logging.Logger
    man: SpeechManifests
    budget: Budget
    counts: Counter = field(default_factory=Counter)
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.counts[key] += n


def make_entry(ctx: Ctx, rel: str, duration: float, info: dict) -> dict:
    """info = {speaker, split, gender, accent, group, extra}; every clip may be used in all four roles."""
    return entry(rel, ctx.source, duration, info["split"], LICENSES[ctx.source], USE, speaker=info["speaker"],
                 gender=info.get("gender"), accent=info.get("accent"), group=info.get("group"),
                 **info.get("extra", {}))


def reuse_existing(ctx: Ctx, rel: str, info: dict) -> float | None:
    """If data/<rel> is already a good 16 kHz FLAC (earlier run), make sure it is in the manifest and return its
    duration; otherwise None."""
    d = flac_ok(rel)
    if d is None:
        return None
    ctx.man.add(make_entry(ctx, rel, d, info))
    ctx.count("already_on_disk")
    return d


def store_clip(ctx: Ctx, rel: str, x: np.ndarray, sr: int, info: dict) -> float | None:
    """Check, resample, write and register one clip. Returns its 16 kHz duration, or None if it was rejected."""
    y, why = prepare_clip(x, sr)
    if y is None:
        ctx.count(why)
        return None
    ctx.budget.check()
    out = save_flac(rel, y, ctx.log)
    ctx.budget.add(out)
    d = len(y) / SR
    ctx.man.add(make_entry(ctx, rel, d, info))
    ctx.count("written")
    return d


# ---------------------------------------------------------------------------------------------------------------
# reporting


def source_stats(source: str) -> dict:
    es = source_entries(source)
    speakers = {e["speaker"] for e in es}
    gender = Counter()
    seen = set()
    for e in es:
        if e["speaker"] not in seen:
            seen.add(e["speaker"])
            gender[e.get("gender") or "?"] += 1
    nbytes = 0
    for e in es:
        p = DATA / e["path"]
        if p.exists():
            nbytes += allocated_bytes(p)
    return {
        "source": source,
        "files": len(es),
        "speakers": len(speakers),
        "speakers_by_split": {s: len({e["speaker"] for e in es if e["split"] == s}) for s in SPLITS},
        "hours": sum(e["duration"] for e in es) / 3600,
        "gb": nbytes / 1e9,
        "gender": dict(gender),
    }


def format_stats(st: dict) -> str:
    sp = st["speakers_by_split"]
    return (f"{st['source']}: {st['speakers']} speakers ({sp['train']} train / {sp['val']} val / {sp['test']} test), "
            f"{st['files']} files, {st['hours']:.2f} h, {st['gb']:.3f} GB, speakers by gender {st['gender']}")
