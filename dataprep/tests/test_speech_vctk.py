"""Offline tests for dataprep.speech_vctk - speaker-info parser, zip member byte ranges, and one speaker end to end
from an in-memory zip that stands in for the remote one."""

import io
import json
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf

from dataprep import common, speech_common as sc
from dataprep import speech_vctk as V

INFO = """ID  AGE  GENDER  ACCENTS  REGION COMMENTS
p225  23  F    English    Southern  England
p280  25  F    Unknown  France (mic2 files unavailable)
p315  18  M    American  New  England  (Text and mic2 files unavailable)
p362  29  F    American
s5    22  F    British (new speaker, more data would be released for training a speaker-dependent model)
"""


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


def flac48(seconds, f):
    t = np.arange(int(seconds * 48000)) / 48000
    b = io.BytesIO()
    sf.write(b, (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32), 48000, format="FLAC", subtype="PCM_24")
    return b.getvalue()


def test_parse_speaker_info():
    s = V.parse_speaker_info(INFO)
    assert set(s) == {"p225", "p280", "p315", "p362", "s5"}
    assert s["p225"] == {"age": 23, "gender": "F", "accent": "English", "region": "Southern England"}
    assert s["p280"]["accent"] is None and s["p280"]["region"] == "France"
    assert s["p315"]["region"] == "New England"
    assert s["p362"]["region"] is None
    assert s["s5"]["accent"] == "British" and s["s5"]["region"] is None


def test_mic1_members():
    names = ["wav48_silence_trimmed/p225/p225_001_mic1.flac", "wav48_silence_trimmed/p225/p225_001_mic2.flac",
             "wav48_silence_trimmed/p225/p225_002_mic1.flac", "wav48_silence_trimmed/s5/s5_003_mic1.flac",
             "txt/p225/p225_001.txt", "wav48_silence_trimmed/log.txt",
             "wav48_silence_trimmed/p226/p225_004_mic1.flac"]  # speaker mismatch is ignored
    m = V.mic1_members(names)
    assert set(m) == {"p225", "s5"}
    assert sorted(u for u, _ in m["p225"]) == ["p225_001", "p225_002"]
    assert m == V.mic1_members(list(reversed(names)))  # order does not depend on the zip order


def test_test_speakers():
    assert len(V.TEST_SPEAKERS) == 8 and len(set(V.TEST_SPEAKERS)) == 8
    assert not set(V.TEST_SPEAKERS) & {"p280", "p315", "s5"}


def zip_bytes(files, method=zipfile.ZIP_DEFLATED):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", compression=method) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return b.getvalue()


@pytest.mark.parametrize("method", [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
def test_member_range_and_unpack(method):
    files = {"speaker-info.txt": INFO.encode(), "wav48_silence_trimmed/p1/p1_001_mic1.flac": flac48(0.5, 300)}
    blob = zip_bytes(files, method)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    for zi in zf.infolist():
        start, end = V.member_byte_range(zi, len(blob))
        assert V.unpack_member(blob[start:end + 1], zi) == files[zi.filename]
        with pytest.raises(EOFError):
            V.unpack_member(blob[start:start + 40], zi)
        broken = bytearray(blob[start:end + 1])
        broken[0] = 0
        with pytest.raises(IOError):
            V.unpack_member(bytes(broken), zi)


class FakeZip:
    """Same interface as RemoteZip (info + read), backed by an in-memory zip."""

    def __init__(self, files):
        self.zf = zipfile.ZipFile(io.BytesIO(zip_bytes(files)))
        self.info = {zi.filename: zi for zi in self.zf.infolist()}

    def read(self, name):
        return self.zf.read(name)


def test_do_speaker_caps_and_splits(tmp_data, monkeypatch):
    monkeypatch.setattr(V, "SECONDS_PER_SPEAKER", 7.0)
    spk = V.TEST_SPEAKERS[0]
    files = {}
    for i in range(5):
        files[f"wav48_silence_trimmed/{spk}/{spk}_{i:03d}_mic1.flac"] = flac48(3.0, 200 + 30 * i)
        files[f"txt/{spk}/{spk}_{i:03d}.txt"] = f"sentence {i}".encode()
    files[f"wav48_silence_trimmed/{spk}/{spk}_009_mic1.flac"] = flac48(1.0, 999)  # under 2 s
    files["wav48_silence_trimmed/p999/p999_001_mic1.flac"] = flac48(3.0, 500)
    rz = FakeZip(files)
    members = V.mic1_members(rz.info)
    man = sc.SpeechManifests()
    ctx = sc.Ctx("vctk", logging.getLogger("test_vctk"), man, sc.Budget("vctk", cap_gb=1.0))
    with ThreadPoolExecutor(2) as pool:
        kept = V.do_speaker(ctx, rz, pool, spk, members[spk], {"gender": "F", "accent": "English", "age": 22,
                                                               "region": "Manchester"})
        assert kept == pytest.approx(6.0)  # two 3 s clips; a third would pass 7 s and 1 s is left
        kept2 = V.do_speaker(ctx, rz, pool, "p999", members["p999"], {})
        assert kept2 == pytest.approx(3.0)
        # running the same speaker again reuses what is there and writes nothing new
        written = ctx.counts["written"]
        assert V.do_speaker(ctx, rz, pool, spk, members[spk], {"gender": "F"}) == pytest.approx(6.0)
        assert ctx.counts["written"] == written
    man.close()
    test = [json.loads(x) for x in (tmp_data / "manifests" / "speech_test.jsonl").read_text().splitlines()]
    assert len(test) == 2 and all(e["speaker"] == f"vctk:{spk}" and e["split"] == "test" for e in test)
    assert all(e["extra"]["text"].startswith("sentence ") for e in test)
    assert test[0]["gender"] == "F" and test[0]["accent"] == "English" and test[0]["extra"]["orig_sr"] == 48000
    other = [json.loads(x) for s in ("train", "val") for x in
             ((tmp_data / "manifests" / f"speech_{s}.jsonl").read_text().splitlines()
              if (tmp_data / "manifests" / f"speech_{s}.jsonl").exists() else [])]
    assert len(other) == 1 and other[0]["split"] == common.split_for("vctk:p999") and "text" not in other[0]["extra"]
