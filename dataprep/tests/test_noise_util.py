"""Offline tests for dataprep.noise_util - license parsing, deterministic hashing, resampling, manifest routing."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep import common  # noqa: E402
from dataprep import noise_util as nu  # noqa: E402


@pytest.mark.parametrize("text,tag", [
    ("http://creativecommons.org/licenses/by/3.0/", "CC-BY-3.0"),
    ("https://creativecommons.org/licenses/by-nc/4.0/", "CC-BY-NC-4.0"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0"),
    ("http://creativecommons.org/licenses/sampling+/1.0/", "CC-Sampling+-1.0"),
    ("http://creativecommons.org/publicdomain/mark/1.0/", "PD"),
    ("Attribution 3.0 Unported (CC BY 3.0)", "CC-BY-3.0"),
    ("Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)", "CC-BY-NC-SA-3.0"),
    ("Attribution-NoDerivs 3.0 Unported (CC BY-ND 3.0)", "CC-BY-ND-3.0"),
    ("X by Y is licensed under a Attribution-Share Alike 3.0 United States License.", "CC-BY-SA-3.0"),
    ("Seeing Things by BrokeMC is licensed under a Attribution License.", "CC-BY"),
    ("License:Public Domain", "PD"),
    ("CC BY-SA 4.0", "CC-BY-SA-4.0"),
    ("License: CC BY 3.0", "CC-BY-3.0"),
    ("", None),
    ("http://freemusicarchive.org/music/Some_Artist/Album/", None),
])
def test_license_tag(text, tag):
    assert nu.license_tag(text) == tag


@pytest.mark.parametrize("tag,ok", [
    ("CC0-1.0", True), ("PD", True), ("CC-BY-3.0", True), ("CC-BY", True), ("CC-BY-SA-2.5", True),
    ("CC-BY-NC-3.0", False), ("CC-BY-ND-3.0", False), ("CC-BY-NC-SA-3.0", False), ("CC-Sampling+-1.0", False),
    (None, False),
])
def test_license_ok(tag, ok):
    assert nu.license_ok(tag) is ok


def test_stable_unit_is_deterministic():
    a = [nu.stable_unit(f"k{i}") for i in range(1000)]
    assert a == [nu.stable_unit(f"k{i}") for i in range(1000)]
    assert 0.0 <= min(a) and max(a) < 1.0
    assert 0.4 < np.mean(a) < 0.6


def test_to_16k_mono_picks_channel_and_limits_peak():
    sr = 44100
    t = np.arange(sr) / sr
    x = np.stack([1.2 * np.sin(2 * np.pi * 440 * t), np.zeros(sr)], axis=1).astype(np.float32)
    y = nu.to_16k_mono(x, sr)
    assert abs(len(y) - 16000) <= 1
    assert y.dtype == np.float32 and np.max(np.abs(y)) <= 0.999 + 1e-6
    assert np.max(np.abs(nu.to_16k_mono(x, sr, channel=1))) == 0.0


def test_noise_manifests_route_by_split(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "MANIFESTS", tmp_path / "manifests")
    with nu.NoiseManifests() as man:
        for split in ("train", "val", "test", "train"):
            man.add(common.entry(f"noise/x/{split}.flac", "x", 1.0, split, "CC0-1.0", ["noise"]))
        assert "noise/x/val.flac" in man
    lines = {s: (tmp_path / f"manifests/noise_{s}.jsonl").read_text().splitlines() for s in ("train", "val", "test")}
    assert {s: len(v) for s, v in lines.items()} == {"train": 1, "val": 1, "test": 1}   # duplicate skipped
    assert json.loads(lines["val"][0])["split"] == "val"
