"""Tests for dataprep.eval_hf - EdAcc / VoxPopuli selection logic and the resumable row-group loop (offline)."""

import io
import json
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

import dataprep.common as common
import dataprep.eval_hf as eh


def test_small_helpers():
    assert eh.edacc_conversation("EDACC-C08-A") == "C08"
    assert eh.edacc_text_ok("HELLO THERE")
    assert not eh.edacc_text_ok("IGNORE_TIME_SEGMENT_IN_SCORING")
    assert not eh.edacc_text_ok("")
    assert eh.vp_safe_name("20111116-0900-PLENARY-8-en_20111116-16:02:33_0") == "20111116-0900-PLENARY-8-en_20111116-16-02-33_0"
    assert eh.vp_session("20111116-0900-PLENARY-8-en_20111116-16:02:33_0") == "20111116-0900-PLENARY-8"


def _rows(spec):
    """spec: list of (shard, rg, speaker, count)"""
    return [{"shard": s, "rg": g, "speaker": spk} for s, g, spk, n in spec for _ in range(n)]


def test_choose_row_groups_covers_every_speaker():
    rows = _rows([(0, 0, "C01-A", 30), (0, 0, "C01-B", 25), (0, 1, "C01-A", 40), (0, 1, "C02-A", 5),
                  (1, 0, "C02-A", 10), (1, 0, "C02-B", 3), (1, 1, "C02-B", 30)])
    sizes = {(0, 0): 10, (0, 1): 50, (1, 0): 5, (1, 1): 7}
    got = eh.choose_row_groups(rows, sizes, min_rows=20)
    have = Counter()
    for r in rows:
        if (r["shard"], r["rg"]) in got:
            have[r["speaker"]] += 1
    total = Counter(r["speaker"] for r in rows)
    for spk, n in total.items():
        assert have[spk] >= min(20, n), spk
    assert got == sorted(got) and len(got) == len(set(got))
    assert eh.choose_row_groups(rows, sizes, 20) == got   # deterministic


def test_select_unseen_needs_two_rows():
    rows = [{"speaker": s} for s in ["1", "1", "2", "3", "3", "3", "None"]]
    out = eh.select_unseen(rows, seen={"3"})
    assert [r["speaker"] for r in out] == ["1", "1"]   # "2" has one row, "3" is seen, "None" has no id


def _wav(seconds, amp=0.1, sr=32000):
    buf = io.BytesIO()
    x = (amp * np.sin(np.arange(int(seconds * sr)) * 0.05)).astype(np.float32)
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", tmp_path)
    monkeypatch.setattr(common, "MANIFESTS", tmp_path / "manifests")
    monkeypatch.setattr(eh, "check_disk", lambda: None)
    monkeypatch.setattr(common, "check_disk", lambda min_gb=12.0: None)
    return tmp_path


def test_process_caps_filters_and_resumes(tmp_data, monkeypatch):
    audio = {(0, 0): [_wav(3.0), _wav(0.5), _wav(4.0), _wav(3.0, amp=0.0)], (0, 1): [_wav(2.0), _wav(3.0)]}
    fetched = []

    def fake_iter(shards, shard, rg, batch=32):
        fetched.append((shard, rg))
        yield from enumerate(audio[(shard, rg)])

    monkeypatch.setattr(eh, "iter_rg_audio", fake_iter)

    class FakeShards:
        def fetched(self):
            return 0

    rows = []
    for (s, g), clips in audio.items():
        for j in range(len(clips)):
            spk = "A" if j % 2 == 0 else "B"
            rows.append({"shard": s, "rg": g, "row": j, "speaker": spk, "speaker_id": f"t:{spk}",
                         "rel": f"eval/t/{spk}/{s}_{g}_{j}.flac"})

    def make_entry(r, dur, sr):
        return common.entry(r["rel"], "t", dur, "test", "CC0-1.0", ["eval"], speaker=r["speaker_id"], orig_sr=sr)

    log = common.get_logger("test_eval_hf")
    state = tmp_data / "state.json"
    acc = Counter()
    with common.ManifestWriter("eval_t.jsonl") as mw:
        stats = eh.process(rows, FakeShards(), mw, acc, cap_s=5.0, min_s=1.0, max_s=30.0, make_entry=make_entry,
                           state_path=state, log=log, tag="t")
    # A: 3.0 s kept, 4.0 s kept (cap reached at 7 s), later A rows capped; B: 0.5 s too short, silent one dropped
    assert stats["written"] == 3 and stats["length"] == 1 and stats["silent"] == 1 and stats["capped"] == 1
    lines = [json.loads(x) for x in (tmp_data / "manifests" / "eval_t.jsonl").read_text().splitlines()]
    assert {e["speaker"] for e in lines} == {"t:A", "t:B"}
    assert all(e["sr"] == 16000 and e["extra"]["orig_sr"] == 32000 for e in lines)
    assert all(common.flac_ok(e["path"]) for e in lines)
    assert json.loads(state.read_text())["done_rgs"] == ["0/0", "0/1"]
    # second run: every row group is marked done, nothing is downloaded again
    fetched.clear()
    with common.ManifestWriter("eval_t.jsonl") as mw:
        eh.process(rows, FakeShards(), mw, eh.speaker_seconds(tmp_data / "manifests" / "eval_t.jsonl"), 5.0, 1.0,
                   30.0, make_entry, state, log, "t")
    assert fetched == []


def test_speaker_seconds(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps({"path": f"p{k}", "speaker": "s:1", "duration": 1.5}) for k in range(3)) + "\n")
    assert eh.speaker_seconds(m) == Counter({"s:1": 4.5})
    assert eh.speaker_seconds(tmp_path / "missing.jsonl") == Counter()
