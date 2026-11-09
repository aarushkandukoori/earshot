"""VCTK 0.92 mic1 (CC BY 4.0) - pull single FLACs out of the 11.75 GB zip on Edinburgh DataShare by range request.

How a remote zip read works. A zip ends with a central directory: one record per member with its name, size, CRC and
the byte offset of its local header. Python's zipfile reads that directory through an fsspec HTTP file (a few range
requests, about 13 MB). For each member we keep, one more range request fetches its local header plus its compressed
bytes; we inflate them and check the CRC. No temporary copy of the zip is ever written.

Selection - 2 minutes per speaker of wav48_silence_trimmed/<spk>/<spk>_<nnn>_mic1.flac (clips of 2 to 15 s, in a fixed
pseudo-random order), 48 kHz to 16 kHz. Gender, age, accent and region come from speaker-info.txt. The 8 speakers in
TEST_SPEAKERS are held out with split "test" (one per accent that has at least 2 speakers, 4 F and 4 M, chosen by the
lowest sha1 of "vctk:<id>" among the speakers of that accent and gender; p280, p315 and s5 were not eligible because
of their known recording or transcript problems). Their transcripts go into extra["text"] for WER tests. All other
speakers get split_for(speaker). The README notes mic2 problems for p280 and p315; we only use mic1.
"""

from __future__ import annotations

import io
import re
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import fsspec
import requests
import soundfile as sf

from .common import get_logger, retry, split_for
from .speech_common import (Budget, Ctx, DoneSet, SourceLock, SpeechManifests, norm_accent, reuse_existing,
                            stable_hash, state_dir, store_clip)
from .speech_http import UA, get_range

SOURCE = "vctk"
URL = "https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content"
SECONDS_PER_SPEAKER = 120.0
MIN_S, MAX_S = 2.0, 15.0
TEST_SPEAKERS = ["p236", "p334", "p234", "p245", "p307", "p292", "p323", "p376"]
# English F, American M, Scottish F, Irish M, Canadian F, Northern Irish M, South African F, Indian M
WORKERS = 8
WINDOW = 16  # clips fetched in parallel per step; at most this many are fetched but not needed per speaker

MIC1 = re.compile(r"^wav48_silence_trimmed/([ps]\d+)/(\1_\d+)_mic1\.flac$")
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")  # signature, version, flags, method, time, date, crc, sizes, name/extra len


# ---------------------------------------------------------------------------------------------------------------
# pure helpers (tested in tests/test_speech_vctk.py)


def parse_speaker_info(text: str) -> dict[str, dict]:
    """speaker-info.txt ('p225  23  F    English    Southern  England') -> {'p225': {age, gender, accent, region}}."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or not re.fullmatch(r"[ps]\d+", parts[0]):
            continue
        region = re.sub(r"\(.*\)", "", " ".join(parts[4:])).strip() or None
        out[parts[0]] = {"age": int(parts[1]) if parts[1].isdigit() else None,
                         "gender": parts[2] if parts[2] in ("F", "M") else None,
                         "accent": norm_accent(parts[3]) if len(parts) > 3 else None,
                         "region": region}
    return out


def mic1_members(names) -> dict[str, list[tuple[str, str]]]:
    """Zip member names -> {speaker: [(utterance id, member name), ...]} for the mic1 FLACs, in a fixed pseudo-random
    order per speaker."""
    by_spk: dict[str, list[tuple[str, str]]] = {}
    for n in names:
        m = MIC1.match(n)
        if m:
            by_spk.setdefault(m.group(1), []).append((m.group(2), n))
    for spk in by_spk:
        by_spk[spk].sort(key=lambda t: (stable_hash(t[0]), t[0]))
    return by_spk


def member_byte_range(zi: zipfile.ZipInfo, file_size: int, slack: int = 1024) -> tuple[int, int]:
    """Inclusive byte range that covers a member's local header and data. The local header's extra field can be
    longer than the central directory's copy, hence the slack."""
    start = zi.header_offset
    end = start + LOCAL_HEADER.size + len(zi.filename.encode()) + len(zi.extra) + slack + zi.compress_size - 1
    return start, min(end, file_size - 1)


def unpack_member(blob: bytes, zi: zipfile.ZipInfo) -> bytes:
    """Local header + data bytes (from member_byte_range) -> the member's uncompressed bytes, CRC-checked.
    Raises EOFError if blob is too short (caller fetches more)."""
    sig, _, _, method, _, _, _, _, _, name_len, extra_len = LOCAL_HEADER.unpack_from(blob, 0)
    if sig != 0x04034B50:
        raise IOError(f"{zi.filename}: bad local header signature {sig:#x}")
    start = LOCAL_HEADER.size + name_len + extra_len
    comp = blob[start:start + zi.compress_size]
    if len(comp) < zi.compress_size:
        raise EOFError(start + zi.compress_size)
    if zi.compress_type == zipfile.ZIP_STORED:
        data = comp
    elif zi.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(comp, -15)
    else:
        raise IOError(f"{zi.filename}: unsupported compression {zi.compress_type}")
    if len(data) != zi.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != zi.CRC:
        raise IOError(f"{zi.filename}: size or CRC mismatch")
    return data


# ---------------------------------------------------------------------------------------------------------------
# remote zip


class RemoteZip:
    def __init__(self, url: str, log):
        self.url, self.log = url, log
        self.sess = requests.Session()
        self.sess.headers.update(UA)

        def open_dir():
            f = fsspec.filesystem("http").open(url, "rb", block_size=1 << 20, cache_type="readahead")
            return f, zipfile.ZipFile(f)

        self._f, zf = retry(open_dir, log=log)
        self.size = self._f.size
        self.info = {zi.filename: zi for zi in zf.infolist()}
        log.info("zip directory - %d members, %.2f GB", len(self.info), self.size / 1e9)

    def read(self, name: str) -> bytes:
        zi = self.info[name]

        def fetch() -> bytes:
            start, end = member_byte_range(zi, self.size)
            blob = get_range(self.sess, self.url, start, end, log=self.log)
            try:
                return unpack_member(blob, zi)
            except EOFError as e:  # unusually long local extra field: fetch exactly what is missing
                need = int(e.args[0])
                blob = get_range(self.sess, self.url, start, start + need - 1, log=self.log)
                return unpack_member(blob, zi)

        return retry(fetch, tries=6, log=self.log)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------------------------
# runner


def run(max_speakers: int | None = None) -> None:
    log = get_logger("speech_vctk")
    with SourceLock(SOURCE), SpeechManifests() as man:
        ctx = Ctx(SOURCE, log, man, Budget(SOURCE))
        done = DoneSet(state_dir(SOURCE) / "done_speakers.txt")
        rz = RemoteZip(URL, log)
        speakers = parse_speaker_info(rz.read("speaker-info.txt").decode("utf-8", "replace"))
        by_spk = mic1_members(rz.info)
        missing = sorted(set(by_spk) - set(speakers))
        if missing:
            log.warning("speakers with audio but no speaker-info line: %s", missing)
        for s in TEST_SPEAKERS:
            assert s in by_spk, f"test speaker {s} has no mic1 files"
        log.info("%d speakers with mic1 audio, %d already done, test speakers %s", len(by_spk), len(done),
                 TEST_SPEAKERS)
        pool = ThreadPoolExecutor(WORKERS)
        t0, n = time.monotonic(), 0
        try:
            for spk in sorted(by_spk):
                if spk in done:
                    continue
                kept_s = do_speaker(ctx, rz, pool, spk, by_spk[spk], speakers.get(spk, {}))
                done.add(spk)
                n += 1
                log.info("%s - %.1f s kept (%d/%d speakers this run, %.0f s elapsed, budget %.3f GB)", spk, kept_s,
                         n, len(by_spk), time.monotonic() - t0, ctx.budget.used / 1e9)
                if max_speakers and n >= max_speakers:
                    break
        finally:
            pool.shutdown(wait=True)
            rz.close()
        log.info("end - counts %s", dict(ctx.counts))


def do_speaker(ctx: Ctx, rz: RemoteZip, pool: ThreadPoolExecutor, spk: str, members: list[tuple[str, str]],
               info: dict) -> float:
    speaker = f"{SOURCE}:{spk}"
    test = spk in TEST_SPEAKERS
    split = "test" if test else split_for(speaker)
    total = 0.0
    for w in range(0, len(members), WINDOW):
        if SECONDS_PER_SPEAKER - total < MIN_S:
            break
        window = members[w:w + WINDOW]

        def meta_for(utt: str, text: str | None = None) -> dict:
            extra = {"mic": "mic1", "orig_sr": 48000, "age": info.get("age"), "region": info.get("region")}
            if text is not None:
                extra["text"] = text
            return {"speaker": speaker, "split": split, "gender": info.get("gender"), "accent": info.get("accent"),
                    "group": None, "extra": extra}

        rels = {utt: f"speech/{SOURCE}/{spk}/{utt}_mic1.flac" for utt, _ in window}
        todo = [(utt, name) for utt, name in window if rels[utt] not in ctx.man]  # not yet in a manifest
        audio = dict(zip([u for u, _ in todo], pool.map(lambda t: rz.read(t[1]), todo)))
        texts = {}
        if test:
            names = {utt: f"txt/{spk}/{utt}.txt" for utt, _ in todo if f"txt/{spk}/{utt}.txt" in rz.info}
            texts = dict(zip(names, pool.map(lambda n: rz.read(n).decode("utf-8", "replace").strip(),
                                             names.values())))
        for utt, _ in window:
            if SECONDS_PER_SPEAKER - total < MIN_S:
                break
            rel = rels[utt]
            if utt not in audio:  # already in the manifest from an earlier run
                d = reuse_existing(ctx, rel, meta_for(utt))
                if d is not None:
                    total += d
                continue
            try:
                x, sr = sf.read(io.BytesIO(audio[utt]), dtype="float32")
            except Exception as e:
                ctx.log.warning("%s: cannot decode (%s)", utt, e)
                ctx.count("decode_error")
                continue
            secs = len(x) / sr
            if not MIN_S <= secs <= MAX_S:
                ctx.count("outside_2_15s")
                continue
            if total + secs > SECONDS_PER_SPEAKER + 1e-6:
                ctx.count("speaker_full")
                continue
            meta = meta_for(utt, texts.get(utt))
            d = reuse_existing(ctx, rel, meta)
            if d is None:
                d = store_clip(ctx, rel, x, sr, meta)
            if d is not None:
                total += d
    return total
