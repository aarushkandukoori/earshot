"""DiPCo (Dinner Party Corpus, CDLA-Permissive-1.0) - a partial, streamed subset of real dinner-party recordings.

DiPCo is one 13.4 GB DipCo.tgz on Zenodo (record 8122551). A gzip stream cannot be seeked, so we read it from the
start and stop early. Measured layout (2026-11-07) - README, LICENSE and all 10 transcription JSONs come first, then
the WAV files of audio/eval/ in no particular order (S08_U03.CH7, S01_U02.CH1, S08_U02.CH2, ...). So the first few
hundred MB give us real far-field recordings of several whole sessions: 4 people talking over dinner, music from a
fixed time mark on, one 16 kHz channel of a 7-mic device a few metres away. Close-talk (headset) files are kept too if
they turn up before we stop.

What we keep -
- the transcriptions, README.md and LICENSE.txt,
- the first far-field channel we meet of each session (one channel per session; channels of one device are
  centimetres apart, so one is enough for single-mic evaluation),
- every close-talk file we meet,
until DIPCO_STREAM_CAP compressed bytes have been read or DIPCO_DISK_CAP bytes written.

Segments - data/eval/dipco_segments.jsonl has one line per transcribed utterance of the kept sessions, with the
speaker, words and start/end seconds on the kept far-field channel and on the close-talk channel.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import DATA, ManifestWriter, check_disk, entry, flac_ok, resample, to_mono, write_flac
from .eval_remote import HttpStream, make_session

DIPCO_URL = "https://zenodo.org/api/records/8122551/files/DipCo.tgz/content"
DIPCO_STREAM_CAP = 2_500_000_000   # stop after this many compressed bytes (about 17 min from Zenodo)
DIPCO_DISK_CAP = 240_000_000       # or after writing this many bytes of FLAC (one file may overshoot)
LICENSE = "CDLA-Permissive-1.0"
OUT = "eval/dipco"

# Music start per session, from the DiPCo README table (hh:mm:ss)
MUSIC_START = {"S01": "00:38:52", "S02": "00:19:30", "S03": "00:33:45", "S04": "00:23:25", "S05": "00:31:15",
               "S06": "00:06:17", "S07": "00:10:05", "S08": "00:01:02", "S09": "00:12:18", "S10": "00:07:10"}

_FAR = re.compile(r"(S\d\d)_(U\d\d)\.(CH\d)\.wav$")
_CLOSE = re.compile(r"(S\d\d)_(P\d\d)\.wav$")
_JSON = re.compile(r"transcriptions/(dev|eval)/(S\d\d)\.json$")


def parse_time(t: str) -> float:
    """'00:02:12.79' -> 132.79 seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def member_kind(name: str) -> tuple | None:
    """Classify a tar member name. Returns ("far", session, device, channel, split), ("close", session, speaker,
    split), ("json", split, session), ("doc", file name) or None (skip, including macOS '._' resource forks)."""
    base = name.rsplit("/", 1)[-1]
    if base.startswith("._"):
        return None
    split = "eval" if "/eval/" in name else "dev" if "/dev/" in name else None
    if m := _FAR.search(name):
        return ("far", m.group(1), m.group(2), m.group(3), split)
    if m := _CLOSE.search(name):
        return ("close", m.group(1), m.group(2), split)
    if m := _JSON.search(name):
        return ("json", m.group(1), m.group(2))
    if base in ("README.md", "LICENSE.txt"):
        return ("doc", base)
    return None


class Keeper:
    """Decides which audio members to keep while streaming (pure logic, unit-tested)."""

    def __init__(self, disk_cap: int, have_far: dict[str, str] | None = None, written: int = 0):
        self.disk_cap = disk_cap
        self.far: dict[str, str] = dict(have_far or {})   # session -> kept far-field file stem
        self.written = written

    def want(self, kind: tuple) -> bool:
        if self.written >= self.disk_cap:
            return False
        if kind[0] == "far":
            stem = f"{kind[1]}_{kind[2]}.{kind[3]}"
            return self.far.get(kind[1], stem) == stem
        return kind[0] == "close"

    def kept(self, kind: tuple, nbytes: int) -> None:
        if kind[0] == "far":
            self.far.setdefault(kind[1], f"{kind[1]}_{kind[2]}.{kind[3]}")
        self.written += nbytes

    def full(self) -> bool:
        return self.written >= self.disk_cap


def segments(utts: list[dict], far_device: str | None, far_path: str | None, close_paths: dict[str, str]) -> list[dict]:
    """Transcribed utterances of one session as segment records (times in seconds)."""
    out = []
    for u in utts:
        spk = u["speaker_id"]
        rec = {"session": u["session_id"], "speaker": f"dipco:{spk}", "words": u["words"],
               "gender": {"male": "M", "female": "F"}.get(u.get("gender", "").lower()),
               "nativeness": u.get("nativeness"), "mother_tongue": u.get("mother_tongue"),
               "far_path": None, "far_start": None, "far_end": None,
               "close_path": close_paths.get(spk), "close_start": None, "close_end": None}
        if far_path and far_device in u["start_time"]:
            rec.update(far_path=far_path, far_start=parse_time(u["start_time"][far_device]),
                       far_end=parse_time(u["end_time"][far_device]))
        if "close-talk" in u["start_time"]:
            rec.update(close_start=parse_time(u["start_time"]["close-talk"]),
                       close_end=parse_time(u["end_time"]["close-talk"]))
        out.append(rec)
    return out


def _wav_to_16k(data: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return resample(to_mono(x, 0), sr)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def fetch_dipco(log: logging.Logger) -> None:
    root = DATA / OUT
    state_path = root / "state.json"
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        state = {"finished": False, "kept": {}}
    # a stream that stopped at an older, smaller cap is read again (from the start - gzip cannot resume mid-way)
    capped_early = state.get("stop_reason") == "stream cap" and state.get("stream_bytes", 0) < DIPCO_STREAM_CAP
    if not state.get("finished") or capped_early:
        _stream(log, root, state, state_path)
    _write_manifest_and_segments(log, root, state)


def _stream(log: logging.Logger, root: Path, state: dict, state_path: Path) -> None:
    kept: dict = state.setdefault("kept", {})       # rel path -> {"kind": [...], "bytes": n}
    keeper = Keeper(DIPCO_DISK_CAP,
                    {v["kind"][1]: Path(k).stem for k, v in kept.items() if v["kind"][0] == "far"},
                    sum(v["bytes"] for v in kept.values()))
    raw = HttpStream(DIPCO_URL, sess=make_session(), log=log)
    stream = io.BufferedReader(raw, 1 << 20)
    log.info("dipco: streaming %s (stop after %.1f GB read or %.0f MB written)", DIPCO_URL, DIPCO_STREAM_CAP / 1e9,
             DIPCO_DISK_CAP / 1e6)
    reason = "end of archive"
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tf:
            for m in tf:
                if raw.pos > DIPCO_STREAM_CAP:
                    reason = "stream cap"
                    break
                if keeper.full():
                    reason = "disk cap"
                    break
                if not m.isfile():
                    continue
                kind = member_kind(m.name)
                if kind is None:
                    continue
                if kind[0] in ("json", "doc"):
                    rel = (f"transcriptions/{kind[1]}/{kind[2]}.json" if kind[0] == "json" else kind[1])
                    _atomic_write(root / rel, tf.extractfile(m).read())
                    continue
                if not keeper.want(kind):
                    continue
                stem = Path(m.name).stem
                rel = f"{OUT}/{stem}.flac"
                if rel in kept and flac_ok(rel):
                    keeper.kept(kind, 0)
                    continue
                check_disk()
                y = _wav_to_16k(tf.extractfile(m).read())
                out = write_flac(rel, y)
                nbytes = out.stat().st_size
                kept[rel] = {"kind": list(kind), "bytes": nbytes, "duration": round(len(y) / 16000, 4)}
                keeper.kept(kind, nbytes)
                state["stream_bytes"] = raw.pos
                _atomic_write(state_path, json.dumps(state, indent=1).encode())
                log.info("dipco: kept %s (%.1f min, %.1f MB), %.0f MB read so far", rel, len(y) / 16000 / 60,
                         nbytes / 1e6, raw.pos / 1e6)
    finally:
        stream.close()
    state.update(finished=True, stop_reason=reason, stream_bytes=raw.pos)
    _atomic_write(state_path, json.dumps(state, indent=1).encode())
    log.info("dipco: stopped (%s) after %.0f MB of the %s archive", reason, raw.pos / 1e6, "13.4 GB")


def _write_manifest_and_segments(log: logging.Logger, root: Path, state: dict) -> None:
    trans: dict[str, list[dict]] = {}
    split_of: dict[str, str] = {}
    for p in sorted((root / "transcriptions").glob("*/S*.json")):
        trans[p.stem] = json.loads(p.read_text())
        split_of[p.stem] = p.parent.name
    kept = state.get("kept", {})
    far = {v["kind"][1]: (rel, v["kind"][2]) for rel, v in kept.items() if v["kind"][0] == "far"}
    close: dict[str, dict[str, str]] = {}
    for rel, v in kept.items():
        if v["kind"][0] == "close":
            close.setdefault(v["kind"][1], {})[v["kind"][2]] = rel
    with ManifestWriter("eval_dipco.jsonl") as mw:
        for rel, v in sorted(kept.items()):
            dur = flac_ok(rel)
            if dur is None:
                log.warning("dipco: %s is missing or unreadable, re-run to fetch it again", rel)
                continue
            kind = v["kind"]
            sess = kind[1]
            people = sorted({u["speaker_id"] for u in trans.get(sess, [])})
            info = next((u for u in trans.get(sess, []) if kind[0] == "close" and u["speaker_id"] == kind[2]), None)
            music = parse_time(MUSIC_START[sess]) if sess in MUSIC_START else None
            if kind[0] == "far":
                e = entry(rel, "dipco", dur, "test", LICENSE, ["eval"], group=sess, kind="far-field",
                          device=kind[2], channel=kind[3], dipco_split=split_of.get(sess),
                          participants=[f"dipco:{p}" for p in people], music_start_s=music)
            else:
                e = entry(rel, "dipco", dur, "test", LICENSE, ["eval"], speaker=f"dipco:{kind[2]}",
                          gender={"male": "M", "female": "F"}.get((info or {}).get("gender", "").lower()),
                          accent=(info or {}).get("mother_tongue"), group=sess, kind="close-talk",
                          nativeness=(info or {}).get("nativeness"), dipco_split=split_of.get(sess),
                          music_start_s=music)
            mw.add(e)
    lines = []
    for sess in sorted(set(far) | set(close)):
        far_rel, far_dev = far.get(sess, (None, None))
        lines += segments(trans.get(sess, []), far_dev, far_rel, close.get(sess, {}))
    out = DATA / "eval" / "dipco_segments.jsonl"
    _atomic_write(out, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines).encode())
    log.info("dipco: %d audio files (%d far-field sessions, %d close-talk), %d segments in %s", len(kept), len(far),
             sum(len(v) for v in close.values()), len(lines), out.relative_to(DATA))
