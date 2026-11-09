"""Offline tests for dataprep.speech_parquet - selection rules and both converters on tiny synthetic parquet files."""

import io
import json
import logging
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from dataprep import common, speech_common as sc
from dataprep import speech_parquet as P


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(common, "DATA", data)
    monkeypatch.setattr(common, "MANIFESTS", data / "manifests")
    monkeypatch.setattr(common, "free_gb", lambda path=None: 100.0)
    monkeypatch.setattr(sc, "DATA", data)
    monkeypatch.setattr(sc, "MANIFESTS", data / "manifests")
    monkeypatch.setattr(sc, "SPEECH", data / "speech")
    monkeypatch.setattr(sc, "free_gb", lambda path=None: 100.0)
    return data


def audio(seconds, sr, fmt, subtype, f=300.0, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    b = io.BytesIO()
    sf.write(b, (amp * np.sin(2 * np.pi * f * t)).astype(np.float32), sr, format=fmt, subtype=subtype)
    return b.getvalue()


def entries(data):
    out = []
    for s in ("train", "val", "test"):
        p = data / "manifests" / f"speech_{s}.jsonl"
        if p.exists():
            out += [json.loads(x) for x in p.read_text().splitlines()]
    return out


def ctx_for(source):
    man = sc.SpeechManifests()
    return sc.Ctx(source, logging.getLogger("test_parquet"), man, sc.Budget(source, cap_gb=1.0))


# --------------------------------------------------------------------------------------------- rules


def test_vox_decision():
    test = {"9"}
    assert P.vox_decision("None", 5, 0, test) == "no_speaker_id"
    assert P.vox_decision(None, 5, 0, test) == "no_speaker_id"
    assert P.vox_decision("9", 5, 0, test) == "official_test_speaker"
    assert P.vox_decision("1", 1.9, 0, test) == "under_2s"
    assert P.vox_decision("1", 10, 50, test) == "keep"
    assert P.vox_decision("1", 10.1, 50, test) == "speaker_full"


def test_globe_decision():
    us, test = "United States English", {"S_9"}
    assert P.globe_decision("S_1", "Irish English", 0.5, 999, test) == "keep"  # non-US: every row
    assert P.globe_decision("S_1", "United States English,Canadian English", 0.5, 999, test) == "keep"
    assert P.globe_decision("S_1", us, 1.5, 0, test) == "us_under_2s"
    assert P.globe_decision("S_1", us, 3.0, 27.0, test) == "keep"
    assert P.globe_decision("S_1", us, 3.1, 27.0, test) == "us_speaker_full"
    assert P.globe_decision("S_9", "Irish English", 3, 0, test) == "official_test_speaker"
    assert P.globe_is_us(" United States English ") and not P.globe_is_us(None)


def test_name_helpers():
    assert P.vox_session("20180418-0900-PLENARY-3-en_20180418-08:50:36_17") == "20180418-0900-PLENARY-3-en"
    assert P.name_stem("000030.v2.vad.flac") == "000030.v2.vad"
    assert P.name_stem("a/b/c d.wav") == "c-d"


def test_shard_lists():
    assert P.VOX_SHARDS == ["en/train-00000-of-00030.parquet", "en/train-00015-of-00030.parquet"]
    assert len(P.GLOBE_SHARDS) == 8 and len(set(P.GLOBE_SHARDS)) == 8


# --------------------------------------------------------------------------------------------- converters


def vox_parquet(path):
    rows = []

    def add(audio_id, spk, secs, gender="female"):
        rows.append({"audio_id": audio_id, "language": 0,
                     "audio": {"bytes": audio(secs, 16000, "WAV", "FLOAT", f=200 + 9 * len(rows)),
                               "path": audio_id + ".wav"},
                     "raw_text": "raw", "normalized_text": f"text {audio_id}", "gender": gender, "speaker_id": spk,
                     "is_gold_transcript": True, "accent": "None"})

    for i in range(6):
        add(f"20180418-0900-PLENARY-3-en_20180418-08:50:{i:02d}_1", "1", 3.0)  # 4 fit in the 12 s test cap
    add("20180418-0900-PLENARY-4-en_x_1", "None", 3.0)
    add("20180418-0900-PLENARY-4-en_x_2", "9", 3.0, "male")  # official test speaker
    add("20180418-0900-PLENARY-4-en_x_3", "2", 1.5, "male")  # under 2 s
    add("20180418-0900-PLENARY-4-en_x_4", "2", 5.0, "male")
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)


def test_convert_voxpopuli(tmp_data, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "VOX_SECONDS", 12.0)
    path = tmp_path / "vox.parquet"
    vox_parquet(path)
    ctx = ctx_for("voxpopuli")
    P.convert_vox_shard(ctx, path, "en/train-x", {"9"}, defaultdict(float))
    ctx.man.close()
    es = entries(tmp_data)
    spk = defaultdict(list)
    for e in es:
        spk[e["speaker"]].append(e)
    assert set(spk) == {"voxpopuli:1", "voxpopuli:2"}
    assert len(spk["voxpopuli:1"]) == 4 and sum(e["duration"] for e in spk["voxpopuli:1"]) == pytest.approx(12.0)
    assert len(spk["voxpopuli:2"]) == 1 and spk["voxpopuli:2"][0]["gender"] == "M"
    e = spk["voxpopuli:1"][0]
    assert ":" not in e["path"] and e["path"].startswith("speech/voxpopuli/1/")
    assert e["group"] == "20180418-0900-PLENARY-3-en" and e["accent"] is None and e["license"] == "CC0-1.0"
    assert e["extra"]["text"].startswith("text ") and e["extra"]["orig_sr"] == 16000
    assert ctx.counts["no_speaker_id"] == 1 and ctx.counts["official_test_speaker"] == 1
    assert ctx.counts["under_2s"] == 1 and ctx.counts["speaker_full"] == 2

    # resume: same shard again, totals rebuilt from the manifest, nothing new is written
    ctx2 = ctx_for("voxpopuli")
    P.convert_vox_shard(ctx2, path, "en/train-x", {"9"}, sc.seconds_per_speaker(sc.source_entries("voxpopuli")))
    ctx2.man.close()
    assert ctx2.counts["written"] == 0 and len(entries(tmp_data)) == 5


def globe_parquet(path):
    rows = []

    def add(spk, accent, secs, name, gender="male"):
        rows.append({"audio": {"bytes": audio(secs, 44100, "FLAC", "PCM_16", f=250 + 11 * len(rows)),
                               "path": name},
                     "speaker_id": spk, "transcript": f"say {name}", "accent": accent, "duration": secs,
                     "age": "twenties", "gender": gender})

    for i in range(5):
        add("S_1", "United States English", 3.0, f"{i:06d}.v2.vad.flac")  # cap 9 s in the test -> 3 kept
    add("S_1", "United States English", 1.5, "000099.v2.vad.flac")  # US under 2 s
    for i in range(4):
        add("S_2", "Irish English", 1.0, f"{i:06d}.v2.vad.flac", gender="female")  # non-US: all kept
    add("S_3", "United States English,Canadian English", 1.2, "000000.v2.vad.flac", gender="other")
    add("S_9", "Irish English", 3.0, "000000.v2.vad.flac")  # test speaker
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=5)


def test_convert_globe(tmp_data, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "GLOBE_US_SECONDS", 9.0)
    path = tmp_path / "globe.parquet"
    globe_parquet(path)
    ctx = ctx_for("globe")
    us_kept = sc.seconds_per_speaker([], None)
    P.convert_globe_shard(ctx, path, "data/train-x", {"S_9"}, us_kept)
    ctx.man.close()
    es = entries(tmp_data)
    count = defaultdict(int)
    for e in es:
        count[e["speaker"]] += 1
        assert e["extra"]["denoised_by_adobe_enhance"] is True and e["extra"]["orig_sr"] == 44100
        assert e["use"] == ["target", "interferer", "babble", "enrollment"]
        assert sf.info(tmp_data / e["path"]).samplerate == 16000
    assert dict(count) == {"globe:S_1": 3, "globe:S_2": 4, "globe:S_3": 1}
    s3 = [e for e in es if e["speaker"] == "globe:S_3"][0]
    assert s3["gender"] is None and s3["accent"] == "United States English,Canadian English"
    assert ctx.counts["us_under_2s"] == 1 and ctx.counts["us_speaker_full"] == 2
    assert ctx.counts["official_test_speaker"] == 1
    # the US cap after a restart counts only US rows already kept
    us = sc.seconds_per_speaker(sc.source_entries("globe"), lambda e: P.globe_is_us(e.get("accent")))
    assert us == {"globe:S_1": pytest.approx(9.0)}


def test_globe_us_cap_uses_real_length(tmp_data, tmp_path, monkeypatch):
    """GLOBE's duration column can be shorter than the audio; the 30 s US cap must use the real length."""
    monkeypatch.setattr(P, "GLOBE_US_SECONDS", 8.7)
    rows = [{"audio": {"bytes": audio(3.0, 44100, "FLAC", "PCM_16", f=300 + 20 * i), "path": f"{i:06d}.v2.vad.flac"},
             "speaker_id": "S_1", "transcript": "t", "accent": "United States English", "duration": 2.5,
             "age": "", "gender": "female"} for i in range(4)]
    path = tmp_path / "g.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    ctx = ctx_for("globe")
    us_kept = sc.seconds_per_speaker([], None)
    P.convert_globe_shard(ctx, path, "data/train-y", set(), us_kept)
    ctx.man.close()
    assert us_kept["globe:S_1"] == pytest.approx(6.0)  # a third 3 s clip would pass 8.7 s
    assert entries(tmp_data)[0]["extra"]["age"] is None
