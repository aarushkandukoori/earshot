"""Offline tests for dataprep.speech_common (tiny synthetic signals, temporary data dir)."""

import json

import numpy as np
import pytest

from dataprep import common, speech_common as sc


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Point every data/ path at a temporary folder and pretend the disk has room."""
    data = tmp_path / "data"
    monkeypatch.setattr(common, "DATA", data)
    monkeypatch.setattr(common, "MANIFESTS", data / "manifests")
    monkeypatch.setattr(common, "free_gb", lambda path=None: 100.0)
    monkeypatch.setattr(sc, "DATA", data)
    monkeypatch.setattr(sc, "MANIFESTS", data / "manifests")
    monkeypatch.setattr(sc, "SPEECH", data / "speech")
    monkeypatch.setattr(sc, "free_gb", lambda path=None: 100.0)
    return data


def tone(seconds, sr, amp=0.3, f=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_safe_name_and_normalizers():
    assert sc.safe_name("20180418-0900-PLENARY-3-en_20180418-08:50:36_17") == \
        "20180418-0900-PLENARY-3-en_20180418-08-50-36_17"
    assert sc.safe_name("  ") == "x"
    assert sc.norm_gender("female") == "F" and sc.norm_gender("M") == "M"
    assert sc.norm_gender("other") is None and sc.norm_gender(None) is None
    assert sc.norm_accent("None") is None and sc.norm_accent(" Unknown ") is None
    assert sc.norm_accent(" Irish English ") == "Irish English"


def test_stable_hash_is_deterministic():
    assert sc.stable_hash("abc") == sc.stable_hash("abc")
    assert sc.stable_hash("abc") != sc.stable_hash("abd")


def test_prepare_clip_resamples_and_rejects():
    y, why = sc.prepare_clip(tone(2.0, 24000), 24000)
    assert why == "ok" and y.dtype == np.float32 and abs(len(y) - 32000) <= 1
    clipped = np.clip(tone(1.0, 16000, amp=2.0), -1, 1)
    assert sc.prepare_clip(clipped, 16000) == (None, "clipped")
    assert sc.prepare_clip(tone(1.0, 16000, amp=1e-4), 16000) == (None, "silent")
    bad = tone(1.0, 16000)
    bad[10] = np.nan
    assert sc.prepare_clip(bad, 16000) == (None, "broken")
    stereo = np.stack([tone(1.0, 44100), tone(1.0, 44100)], axis=1)
    y, why = sc.prepare_clip(stereo, 44100)
    assert why == "ok" and y.ndim == 1 and abs(len(y) - 16000) <= 1


def test_prepare_clip_keeps_peak_below_full_scale():
    x = tone(1.0, 48000, amp=0.9985)  # resampling can overshoot past 0.999
    y, why = sc.prepare_clip(x, 48000)
    assert why == "ok" and float(np.max(np.abs(y))) <= 0.999 + 1e-6


def test_done_set_survives_reload(tmp_path):
    d = sc.DoneSet(tmp_path / "s" / "done.txt")
    d.add("train-clean-100/730")
    d.add("train-clean-100/730")
    assert "train-clean-100/730" in sc.DoneSet(tmp_path / "s" / "done.txt")
    assert (tmp_path / "s" / "done.txt").read_text() == "train-clean-100/730\n"
    with pytest.raises(AssertionError):
        d.add("has space")


def test_wait_for_disk(monkeypatch):
    import logging
    log = logging.getLogger("t")
    monkeypatch.setattr(sc, "free_gb", lambda path=None: 20.0)
    sc.wait_for_disk(log, need_gb=5.0, wait_s=0)  # 20 - 5 >= 12, returns
    with pytest.raises(common.DiskFull):
        sc.wait_for_disk(log, need_gb=9.0, wait_s=0, poll_s=0)
    seq = iter([5.0, 5.0, 30.0])
    monkeypatch.setattr(sc, "free_gb", lambda path=None: next(seq))
    sc.wait_for_disk(log, wait_s=10, poll_s=0)  # recovers on the third poll


def test_store_clip_writes_flac_and_manifest(tmp_data):
    import logging
    with sc.SpeechManifests() as man:
        ctx = sc.Ctx("vctk", logging.getLogger("t"), man, sc.Budget("vctk", cap_gb=1.0))
        info = {"speaker": "vctk:p1", "split": "test", "gender": "F", "accent": "Irish", "group": None,
                "extra": {"mic": "mic1"}}
        d = sc.store_clip(ctx, "speech/vctk/p1/p1_001_mic1.flac", tone(3.0, 48000), 48000, info)
        assert d == pytest.approx(3.0, abs=1e-3)
        assert ctx.budget.used > 0
        # second time the file is reused, not rewritten, and the manifest has one line
        assert sc.reuse_existing(ctx, "speech/vctk/p1/p1_001_mic1.flac", info) == pytest.approx(3.0, abs=1e-3)
        assert sc.store_clip(ctx, "speech/vctk/p1/silent.flac", tone(1.0, 48000, amp=1e-5), 48000, info) is None
    lines = (tmp_data / "manifests" / "speech_test.jsonl").read_text().splitlines()
    assert len(lines) == 1
    e = json.loads(lines[0])
    assert e["use"] == ["target", "interferer", "babble", "enrollment"]
    assert e["license"] == "CC-BY-4.0" and e["sr"] == 16000 and e["extra"] == {"mic": "mic1"}
    assert ctx.counts["silent"] == 1 and ctx.counts["written"] == 1
    st = sc.source_stats("vctk")
    assert st["speakers"] == 1 and st["speakers_by_split"]["test"] == 1 and st["files"] == 1
    assert "vctk: 1 speakers" in sc.format_stats(st)


def test_budget_full(tmp_data):
    b = sc.Budget("globe", cap_gb=0.0)
    with pytest.raises(sc.BudgetFull):
        b.check()


def test_manifests_route_by_split(tmp_data):
    with sc.SpeechManifests() as man:
        for split in ("train", "val", "test"):
            man.add(common.entry(f"speech/x/{split}.flac", "x", 1.0, split, "CC0-1.0", sc.USE, speaker="x:1"))
        assert "speech/x/val.flac" in man and "speech/x/nope.flac" not in man
    for split in ("train", "val", "test"):
        assert len((tmp_data / "manifests" / f"speech_{split}.jsonl").read_text().splitlines()) == 1
    assert sc.seconds_per_speaker(sc.source_entries("x")) == {"x:1": 3.0}


def test_source_lock_blocks_a_second_run(tmp_data):
    with sc.SourceLock("vctk"):
        with pytest.raises(RuntimeError):
            with sc.SourceLock("vctk"):
                pass
    with sc.SourceLock("vctk"):  # released again
        pass
