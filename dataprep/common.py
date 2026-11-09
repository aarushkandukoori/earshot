"""Shared helpers for every Earshot data fetcher (package: dataprep).

Every fetcher writes 16 kHz mono 16-bit FLAC files under data/ and describes them in JSON Lines manifests under
data/manifests/. One manifest line = one audio file:

    {
      "path": "speech/librittsr/1034/1034_121119_000001_000000.flac",  # relative to data/
      "source": "librittsr",           # corpus short name
      "speaker": "librittsr:1034",     # globally unique speaker id, or null for noise/music
      "gender": "F",                   # "F", "M" or null
      "accent": "us",                  # free text or null
      "group": "121119",               # chapter / session / video id, or null
      "duration": 4.31,                # seconds
      "sr": 16000,
      "split": "train",                # "train", "val" or "test"
      "license": "CC-BY-4.0",
      "use": ["target", "interferer", "babble", "enrollment"],   # allowed uses
      "extra": {}                      # optional source-specific fields
    }

Allowed "use" values - target, interferer, babble, enrollment, noise, music, eval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

import numpy as np
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFESTS = DATA / "manifests"
LOGS = DATA / "logs"
SR = 16000
MIN_FREE_GB = 12.0

T = TypeVar("T")


class DiskFull(RuntimeError):
    """Raised when free disk space drops below the floor. Fetchers stop cleanly when they see it."""


def free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def check_disk(min_gb: float = MIN_FREE_GB) -> None:
    gb = free_gb()
    if gb < min_gb:
        raise DiskFull(f"only {gb:.1f} GB free, floor is {min_gb:.1f} GB")


def get_logger(name: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        for h in (logging.FileHandler(LOGS / f"{name}.log"), logging.StreamHandler(sys.stderr)):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


def retry(fn: Callable[[], T], tries: int = 8, base_s: float = 2.0, log: logging.Logger | None = None) -> T:
    """Call fn, retrying with exponential backoff (2, 4, 8 ... s, capped at 120 s) on any exception except DiskFull."""
    for i in range(tries):
        try:
            return fn()
        except DiskFull:
            raise
        except Exception as e:  # network errors come in many types
            if i == tries - 1:
                raise
            wait = min(base_s * 2**i, 120.0)
            if log:
                log.warning("attempt %d/%d failed (%s: %s), retrying in %.0f s", i + 1, tries, type(e).__name__, e, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def to_mono(x: np.ndarray, channel: int | None = None) -> np.ndarray:
    """(samples,) or (samples, channels) -> (samples,). channel=None averages, an int picks one channel."""
    if x.ndim == 1:
        return x
    return x[:, channel] if channel is not None else x.mean(axis=1)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out:
        return x
    return soxr.resample(x, sr_in, sr_out, quality="VHQ").astype(np.float32)


def is_clipped(x: np.ndarray, level: float = 0.999, max_frac: float = 0.001) -> bool:
    return float(np.mean(np.abs(x) >= level)) > max_frac


def is_silent(x: np.ndarray, floor_db: float = -50.0) -> bool:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))
    return 20 * np.log10(rms) < floor_db


def write_flac(rel_path: str | Path, x: np.ndarray, sr: int = SR) -> Path:
    """Atomically write 16-bit FLAC to data/<rel_path>. Checks the disk floor first. Returns the absolute path."""
    check_disk()
    out = DATA / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    fd, tmp = tempfile.mkstemp(suffix=".flac", dir=out.parent)
    os.close(fd)
    try:
        sf.write(tmp, x, sr, format="FLAC", subtype="PCM_16")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return out


def flac_ok(rel_path: str | Path) -> float | None:
    """Duration in seconds if data/<rel_path> exists and decodes as 16 kHz mono, else None (used for resume)."""
    p = DATA / rel_path
    if not p.exists():
        return None
    try:
        info = sf.info(p)
    except Exception:
        return None
    if info.samplerate != SR or info.channels != 1 or info.frames == 0:
        return None
    return info.frames / SR


def split_for(key: str, val_frac: float = 0.03) -> str:
    """Deterministic speaker-level split. Same key -> same split on every machine and every run."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def entry(path: str | Path, source: str, duration: float, split: str, license: str, use: Iterable[str],
          speaker: str | None = None, gender: str | None = None, accent: str | None = None,
          group: str | None = None, **extra) -> dict:
    return {
        "path": str(path), "source": source, "speaker": speaker, "gender": gender, "accent": accent,
        "group": group, "duration": round(float(duration), 4), "sr": SR, "split": split, "license": license,
        "use": list(use), "extra": extra,
    }


class ManifestWriter:
    """Append-only JSON Lines writer. Re-running a fetcher skips paths that are already in the file."""

    def __init__(self, name: str):
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        self.path = MANIFESTS / name
        self.seen = {e["path"] for e in read_manifest(self.path)} if self.path.exists() else set()
        self._f = open(self.path, "a", encoding="utf-8")

    def __contains__(self, rel_path: str) -> bool:
        return str(rel_path) in self.seen

    def add(self, e: dict) -> None:
        if e["path"] in self.seen:
            return
        self._f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._f.flush()
        self.seen.add(e["path"])

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_manifest(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
