"""Offline tests for dataprep.speech_librittsr - parsers, selection, and a full stream of a tiny synthetic tarball
served from a local HTTP server that drops the first connection half-way (the reader must resume, not restart)."""

import hashlib
import io
import json
import logging
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from dataprep import common, speech_common as sc, speech_http
from dataprep import speech_librittsr as L


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
    monkeypatch.setattr(speech_http, "RETRY_BASE_S", 0.01)
    return data


# --------------------------------------------------------------------------------------------- parsers


def test_parse_md5sums():
    text = "2c1f5312914890634cc2d15783032ff3  dev_clean.tar.gz\n6DF668D8F5F33E70876BFA33862AD02B *train_clean_100.tar.gz\n"
    assert L.parse_md5sums(text) == {"dev_clean.tar.gz": "2c1f5312914890634cc2d15783032ff3",
                                     "train_clean_100.tar.gz": "6df668d8f5f33e70876bfa33862ad02b"}


def test_parse_speakers():
    text = """; comment line
;ID  |SEX| SUBSET           |MINUTES| NAME
14   | F | train-clean-360  | 25.03 | Kristin LeMoine
17   | M | train-clean-360  | 25.04 | Gord | Mackenzie
bad  | X | nothing
"""
    assert L.parse_speakers(text) == {"14": "F", "17": "M"}


def test_parse_bad_list():
    text = "./train-clean-100/1040/133433/1040_133433_000042_000000.wav\n\n./x/y/2_3_4_5.wav\n"
    assert L.parse_bad_list(text) == {"1040_133433_000042_000000", "2_3_4_5"}


def test_parse_member():
    assert L.parse_member("LibriTTS_R/train-clean-100/730/358/730_358_000005_000000.wav") == \
        ("train-clean-100", "730", "358", "730_358_000005_000000", "wav")
    assert L.parse_member("./LibriTTS_R/train-clean-100/730/358/730_358_000005_000000.normalized.txt") == \
        ("train-clean-100", "730", "358", "730_358_000005_000000", "text")
    assert L.parse_member("LibriTTS_R/train-clean-100/730/358/730_358_000005_000000.original.txt") is None
    assert L.parse_member("LibriTTS_R/train-clean-100/730/358/730_358.trans.tsv") is None
    assert L.parse_member("LibriTTS_R/train-clean-100/730/358") is None
    assert L.parse_member("LibriTTS_R/SPEAKERS.txt") is None
    assert L.parse_member("LibriTTS_R/train-clean-100/730/358/999_358_000005_000000.wav") is None  # wrong speaker


def test_wav_seconds():
    assert L.wav_seconds(44 + 48000) == pytest.approx(1.0)
    assert L.wav_seconds(10) == 0.0


# --------------------------------------------------------------------------------------------- selection


def test_round_robin_spreads_over_chapters():
    chapter_of = {f"s_{ch}_{i:06d}": ch for ch in ("100", "200", "300") for i in range(5)}
    order = L.round_robin_order(chapter_of)
    assert sorted(order) == sorted(chapter_of)
    assert {chapter_of[u] for u in order[:3]} == {"100", "200", "300"}  # first pass takes one from each
    assert order == L.round_robin_order(dict(reversed(list(chapter_of.items()))))  # independent of input order


def test_pick_within_budget():
    order = ["a", "b", "c", "d", "e"]
    secs = {"a": 30.0, "b": 25.0, "c": 10.0, "d": 4.0, "e": 3.0}
    kept, total = L.pick_within_budget(order, secs, 60.0, lambda u: secs[u])
    assert kept == ["a", "b", "d"] and total == 59.0  # c would overflow, e no longer fits after d
    kept, total = L.pick_within_budget(order, secs, 60.0, lambda u: None if u == "a" else secs[u])
    assert kept == ["b", "c", "d", "e"] and total == 42.0  # a rejected (for example clipped)


def test_plan_lanes():
    keys = list(L.TARBALLS)
    assert L.plan_lanes(keys, 2) == [["train_other_500"], ["train_clean_100", "train_clean_360"]]
    assert L.plan_lanes(keys, 1) == [["train_clean_100", "train_clean_360", "train_other_500"]]
    assert L.plan_lanes(["train_clean_100"], 4) == [["train_clean_100"]]


# --------------------------------------------------------------------------------------------- streaming


def wav_bytes(seconds, f, amp=0.3, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    x = np.clip(amp * np.sin(2 * np.pi * f * t), -1, 1).astype(np.float32)
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def make_tarball():
    """Speaker 11 - 2 chapters x 15 clips of 3 s (90 s). Speaker 22 - 1 chapter: 20 clips of 4.5 s, plus a failed-
    restoration clip, a 1 s clip, a 16 s clip and a clipped clip. Members in tar order like the real thing."""
    files = []
    for ch in ("100", "200"):
        for i in range(15):
            files.append(("11", ch, f"11_{ch}_{i:06d}_000000", wav_bytes(3.0, 200 + 10 * i)))
    for i in range(20):
        files.append(("22", "300", f"22_300_{i:06d}_000000", wav_bytes(4.5, 300 + 7 * i)))
    files.append(("22", "300", "22_300_000100_000000", wav_bytes(5.0, 500)))  # in the failed list
    files.append(("22", "300", "22_300_000101_000000", wav_bytes(1.0, 500)))  # too short
    files.append(("22", "300", "22_300_000102_000000", wav_bytes(16.0, 500)))  # too long
    files.append(("22", "300", "22_300_000103_000000", wav_bytes(5.0, 500, amp=3.0)))  # clipped
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def add(name, data):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        add("LibriTTS_R/README_libritts_r.txt", b"readme")
        for spk, ch, utt, data in files:
            base = f"LibriTTS_R/train-clean-100/{spk}/{ch}/{utt}"
            add(base + ".normalized.txt", f"text of {utt}".encode())
            add(base + ".original.txt", b"original")
            add(base + ".wav", data)
    return buf.getvalue()


class RangeHandler(BaseHTTPRequestHandler):
    """Serves `blob` with HEAD and byte ranges. drop_first_after=N sends only N bytes of the first range, then hangs
    up, like a flaky connection."""
    blob = b""
    drop_first_after = None
    starts: list = []

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.blob)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", '"v1"')
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        start, end = 0, len(self.blob) - 1
        if rng:
            a, b = rng.split("=")[1].split("-")
            start, end = int(a), (int(b) if b else len(self.blob) - 1)
        RangeHandler.starts.append(start)
        body = self.blob[start:end + 1]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", '"v1"')
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.blob)}")
        self.end_headers()
        if self.drop_first_after and len(RangeHandler.starts) == 1:
            self.wfile.write(body[: self.drop_first_after])  # then hang up early
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    RangeHandler.starts = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def make_run(blob, connections=3):
    log = logging.getLogger("test_librittsr")
    man = sc.SpeechManifests()
    ctx = sc.Ctx("librittsr", log, man, sc.Budget("librittsr", cap_gb=1.0))
    meta = L.Meta(md5={"train_clean_100.tar.gz": hashlib.md5(blob).hexdigest()}, gender={"11": "F", "22": "M"},
                  bad={"22_300_000100_000000"})
    return L.Run(ctx, meta, max_speakers=None, connections=connections)


def test_stream_selects_resumes_and_skips_done(tmp_data, server, monkeypatch):
    blob = make_tarball()
    RangeHandler.blob = blob
    RangeHandler.drop_first_after = len(blob) // 3
    monkeypatch.setattr(L, "BASE", f"http://127.0.0.1:{server.server_address[1]}/")

    r = make_run(blob)
    assert r.stream("train_clean_100") is True  # read to the end and the md5 matched
    r.ctx.man.close()
    assert len(RangeHandler.starts) == len(set(RangeHandler.starts)) + 1  # the dropped chunk was fetched again

    entries = [json.loads(x) for s in ("train", "val") for x in
               (tmp_data / "manifests" / f"speech_{s}.jsonl").read_text().splitlines()]
    by_spk = {}
    for e in entries:
        by_spk.setdefault(e["speaker"], []).append(e)
    s11, s22 = by_spk["librittsr:11"], by_spk["librittsr:22"]
    assert len(s11) == 20 and sum(e["duration"] for e in s11) == pytest.approx(60.0, abs=0.01)
    assert sorted({e["group"] for e in s11}) == ["100", "200"]
    assert sum(e["group"] == "100" for e in s11) == 10  # round robin gives each chapter half
    assert len(s22) == 13 and sum(e["duration"] for e in s22) == pytest.approx(58.5, abs=0.01)
    names = {e["path"].rsplit("/", 1)[-1] for e in s22}
    assert not names & {"22_300_000100_000000.flac", "22_300_000101_000000.flac", "22_300_000102_000000.flac"}
    e = s11[0]
    assert e["gender"] == "F" and e["accent"] is None and e["license"] == "CC-BY-4.0"
    assert e["extra"]["subset"] == "train-clean-100" and e["extra"]["text"].startswith("text of 11_")
    assert e["split"] == common.split_for("librittsr:11") and s22[0]["gender"] == "M"
    info = sf.info(tmp_data / e["path"])
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
    assert r.ctx.counts["failed_restoration_skipped"] == 1

    # a second run streams again but skips both finished speakers without writing anything
    RangeHandler.drop_first_after = None
    r2 = make_run(blob)
    assert r2.stream("train_clean_100") is True
    r2.ctx.man.close()
    assert r2.ctx.counts["written"] == 0 and r2.ctx.counts["speakers"] == 0
    n_lines = sum(len((tmp_data / "manifests" / f"speech_{s}.jsonl").read_text().splitlines()) for s in ("train", "val"))
    assert n_lines == 33


def test_half_finished_speaker_is_completed_identically(tmp_data, server, monkeypatch):
    """Crash in the middle of a speaker: on restart the same clips are chosen and existing files are reused."""
    blob = make_tarball()
    RangeHandler.blob = blob
    RangeHandler.drop_first_after = None
    monkeypatch.setattr(L, "BASE", f"http://127.0.0.1:{server.server_address[1]}/")
    r = make_run(blob)
    assert r.stream("train_clean_100")
    r.ctx.man.close()
    first = sorted(p.name for p in (tmp_data / "speech" / "librittsr" / "11").iterdir())
    # forget speaker 11 was finished and delete half of its files and the manifests
    done = tmp_data / "speech" / "librittsr" / "_state" / "done_speakers.txt"
    done.write_text("train-clean-100/22\n")
    for p in sorted((tmp_data / "speech" / "librittsr" / "11").iterdir())[:10]:
        p.unlink()
    for s in ("train", "val"):
        (tmp_data / "manifests" / f"speech_{s}.jsonl").unlink(missing_ok=True)
    r2 = make_run(blob)
    assert r2.stream("train_clean_100")
    r2.ctx.man.close()
    assert sorted(p.name for p in (tmp_data / "speech" / "librittsr" / "11").iterdir()) == first
    assert r2.ctx.counts["written"] == 10 and r2.ctx.counts["already_on_disk"] == 10


def test_md5_mismatch_does_not_mark_done(tmp_data, server, monkeypatch):
    blob = make_tarball()
    RangeHandler.blob = blob
    RangeHandler.drop_first_after = None
    monkeypatch.setattr(L, "BASE", f"http://127.0.0.1:{server.server_address[1]}/")
    r = make_run(blob)
    r.meta.md5["train_clean_100.tar.gz"] = "0" * 32
    assert r.stream("train_clean_100") is False
    r.ctx.man.close()
