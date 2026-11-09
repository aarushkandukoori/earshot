"""Small helpers shared by the noise fetchers (FSD50K, DEMAND, MUSAN)."""

from __future__ import annotations

import fcntl
import hashlib
import io
import re

import numpy as np
import soundfile as sf

from . import common
from .common import ManifestWriter, SR, resample, to_mono

_CC_URL = re.compile(r"creativecommons\.org/licenses/([a-z+\-]+)/(\d(?:\.\d)?)")


def license_tag(text: str | None) -> str | None:
    """Turn a license string or URL into a short tag like "CC-BY-3.0", "CC0-1.0", "PD", "CC-BY-NC-3.0".

    Returns None when the text is empty or unrecognised. Use license_ok() to decide whether to keep a clip."""
    if not text:
        return None
    t = text.strip().lower()
    if "publicdomain/zero" in t or re.search(r"\bcc0\b", t) or "cc-zero" in t or "cc zero" in t:
        return "CC0-1.0"
    m = _CC_URL.search(t)
    if m:
        kind, ver = m.group(1), m.group(2)
        if kind.startswith("sampling"):
            return f"CC-Sampling+-{ver}"
        return "CC-" + kind.upper() + f"-{ver}"
    if "publicdomain/mark" in t or "public domain" in t or "public-domain" in t:
        return "PD"
    # plain-text forms, for example "Attribution 3.0", "Attribution-NonCommercial 3.0" or "CC BY-SA 4.0"
    ver = re.search(r"(\d\.\d)", t)
    v = f"-{ver.group(1)}" if ver else ""
    words = t.replace("_", " ").replace("-", " ")
    squashed = words.replace(" ", "")
    if "sampling" in words:
        return f"CC-Sampling+{v}"
    if "attribution" in words or re.search(r"\bcc by\b", words):
        parts = ["CC-BY"]
        if "noncommercial" in squashed or re.search(r"\bnc\b", words):
            parts.append("NC")
        if "noderiv" in squashed or re.search(r"\bnd\b", words):
            parts.append("ND")
        if "sharealike" in squashed or re.search(r"\bsa\b", words):
            parts.append("SA")
        return "-".join(parts) + v
    return None


def license_ok(tag: str | None) -> bool:
    """Commercial use and adaptations allowed - CC0, public domain, CC BY, CC BY-SA (any version)."""
    if not tag:
        return False
    if tag in ("CC0-1.0", "PD"):
        return True
    base = re.sub(r"-\d(\.\d)?$", "", tag)
    return base in ("CC-BY", "CC-BY-SA")


def stable_unit(key: str) -> float:
    """Deterministic number in [0, 1) from a string (same on every machine and run)."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:12], 16) / float(16**12)


def decode_audio(data: bytes) -> tuple[np.ndarray, int, int]:
    """Decode WAV/FLAC bytes. Returns (samples x channels float32 array, sample rate, channel count)."""
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return x, sr, x.shape[1]


def to_16k_mono(x: np.ndarray, sr: int, channel: int | None = 0) -> np.ndarray:
    """(samples, channels) -> 16 kHz mono float32, peak kept below full scale so FLAC writing never clips."""
    y = resample(to_mono(x, channel), sr, SR)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:
        y = y * (0.999 / peak)
    return y.astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


class NoiseManifests:
    """The three noise manifests (noise_train/val/test.jsonl). add() routes an entry by its split.

    The FSD50K, DEMAND and MUSAN fetchers may run as separate processes at the same time, so every append takes an
    exclusive file lock (fcntl.flock) - two processes never interleave half-written lines."""

    def __init__(self):
        common.MANIFESTS.mkdir(parents=True, exist_ok=True)
        self._lock = open(common.MANIFESTS / ".noise.lock", "a")
        self.w = {s: ManifestWriter(f"noise_{s}.jsonl") for s in ("train", "val", "test")}

    def __contains__(self, rel_path: str) -> bool:
        return any(rel_path in w for w in self.w.values())

    def add(self, e: dict) -> None:
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        try:
            self.w[e["split"]].add(e)
        finally:
            fcntl.flock(self._lock, fcntl.LOCK_UN)

    def close(self) -> None:
        for w in self.w.values():
            w.close()
        self._lock.close()

    def __enter__(self) -> "NoiseManifests":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def bytes_on_disk(rel_paths) -> int:
    total = 0
    for p in rel_paths:
        f = common.DATA / p
        if f.exists():
            total += f.stat().st_size
    return total
