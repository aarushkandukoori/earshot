"""Offline tests for dataprep.noise_remote - the split-zip reader and the resumable stream.

The fixtures fake the HTTP layer, so nothing touches the network."""

import io
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep import noise_remote as nr  # noqa: E402


class FakeResponse:
    def __init__(self, status, body=b"", headers=None, fail_after=None):
        self.status_code, self.content, self.headers = status, body, headers or {}
        self.raw = FakeRaw(body, fail_after)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise IOError(f"HTTP {self.status_code}")

    def close(self):
        pass


class FakeRaw:
    def __init__(self, body, fail_after):
        self.body, self.pos, self.fail_after, self.decode_content = body, 0, fail_after, True

    def readinto(self, b):
        if self.fail_after is not None and self.pos >= self.fail_after:
            raise ConnectionError("connection reset")
        n = min(len(b), len(self.body) - self.pos, 7)   # small reads, like a real socket
        b[:n] = self.body[self.pos:self.pos + n]
        self.pos += n
        return n


class FakeSession:
    """Serves byte ranges of in-memory files. files = {url: bytes}."""

    def __init__(self, files, fail_first_stream_after=None):
        self.files, self.calls, self.fail = files, [], fail_first_stream_after

    def get(self, url, headers=None, stream=False, timeout=None):
        data = self.files[url]
        rng = (headers or {}).get("Range")
        self.calls.append(rng)
        if rng is None:
            fail, self.fail = self.fail, None
            return FakeResponse(200, data, {"Content-Length": str(len(data))}, fail_after=fail)
        a, b = rng.removeprefix("bytes=").split("-")
        a = int(a)
        b = int(b) if b else len(data) - 1
        return FakeResponse(206, data[a:b + 1])


def make_split_zip(files: dict, cuts: list[int]) -> list[bytes]:
    """Build a real-format split zip - a 4-byte spanning signature, then members, with every central-directory entry
    pointing at (disk number, offset inside that disk), like FSD50K's .z01 ... .zip. cuts = part boundaries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            method = zipfile.ZIP_DEFLATED if name.endswith(".bin") else zipfile.ZIP_STORED
            z.writestr(zipfile.ZipInfo(name), data, compress_type=method)
    raw = b"PK\x07\x08" + buf.getvalue()
    i = raw.rfind(nr.EOCD_SIG)
    _, _, _, _, n_total, cd_size, cd_off, _ = struct.unpack("<4s4H2LH", raw[i:i + 22])
    cd_off += 4
    starts = [0] + cuts
    assert cuts[-1] < cd_off, "keep the central directory inside the last part"

    def disk_of(off):
        return max(k for k, s in enumerate(starts) if s <= off)

    cd = bytearray(raw[cd_off:cd_off + cd_size])
    p = 0
    while p < len(cd):
        nlen, elen, clen = struct.unpack("<3H", cd[p + 28:p + 34])
        off = struct.unpack("<L", cd[p + 42:p + 46])[0] + 4
        k = disk_of(off)
        struct.pack_into("<H", cd, p + 34, k)
        struct.pack_into("<L", cd, p + 42, off - starts[k])
        p += 46 + nlen + elen + clen
    last = len(starts) - 1
    eocd = struct.pack("<4s4H2LH", nr.EOCD_SIG, last, last, n_total, n_total, cd_size, cd_off - starts[last], 0)
    glued = raw[:cd_off] + bytes(cd) + eocd
    return [glued[s:e] for s, e in zip(starts, starts[1:] + [len(glued)])]


@pytest.fixture
def split_zip():
    files = {
        "A/first.bin": bytes(range(256)) * 40,          # deflated
        "A/second.wav": b"RIFF" + b"\x01\x02" * 3000,     # stored, will cross a part boundary
        "A/third.bin": b"hello world " * 500,
        "A/fourth.wav": b"x" * 100,
    }
    parts = make_split_zip(files, cuts=[2000, 5000])
    urls = [f"https://example.test/a.z{k:02d}" for k in range(1, len(parts))] + ["https://example.test/a.zip"]
    sess = FakeSession(dict(zip(urls, parts)))
    return files, sess, [(u, len(p)) for u, p in zip(urls, parts)]


def test_split_zip_directory_and_members(split_zip):
    files, sess, parts = split_zip
    rz = nr.RemoteZip(parts, sess)
    assert set(rz.members) == set(files)
    got = dict(rz.iter_members(["A/second.wav", "A/fourth.wav", "A/first.bin"]))
    assert got == {k: files[k] for k in ("A/second.wav", "A/fourth.wav", "A/first.bin")}
    # second.wav straddles the boundary at byte 5000, so at least one read was split over two parts
    m = rz.members["A/second.wav"]
    assert m.offset < 5000 < m.offset + 30 + len(m.name) + m.csize


def test_read_member_alone(split_zip):
    files, sess, parts = split_zip
    rz = nr.RemoteZip(parts, sess)
    assert rz.read_member(rz.members["A/third.bin"]) == files["A/third.bin"]


def test_corrupt_member_is_reported_as_none(split_zip):
    files, sess, parts = split_zip
    rz = nr.RemoteZip(parts, sess)
    m = rz.members["A/fourth.wav"]
    # flip one byte of the stored data in whichever part holds it
    k = max(i for i in range(len(parts)) if rz.starts[i] <= m.offset + 40)
    url = parts[k][0]
    blob = bytearray(sess.files[url])
    blob[m.offset + 30 + len(m.name) + 5 - rz.starts[k]] ^= 0xFF
    sess.files[url] = bytes(blob)
    assert dict(rz.iter_members(["A/fourth.wav"])) == {"A/fourth.wav": None}


def test_plan_spans_merges_neighbours():
    ms = [nr.Member("a", 8, 100, 200, 0, 0), nr.Member("b", 8, 100, 200, 300, 0), nr.Member("c", 8, 100, 200, 10_000, 0)]
    spans = nr.plan_spans(ms, max_gap=1000, max_span=1 << 20, slack=0)
    assert [[m.name for m in s.members] for s in spans] == [["a", "b"], ["c"]]
    spans = nr.plan_spans(ms, max_gap=0, max_span=1 << 20, slack=0)
    assert len(spans) == 3


def test_extract_checks_crc():
    data = b"abc" * 100
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = comp.compress(data) + comp.flush()
    header = nr.LOCAL_SIG + b"\x00" * 22 + struct.pack("<2H", 1, 0) + b"n"
    m = nr.Member("n", 8, len(raw), len(data), 0, zlib.crc32(data))
    assert nr.extract_from(header + raw, 0, m) == data
    with pytest.raises(ValueError):
        nr.extract_from(header + raw, 0, nr.Member("n", 8, len(raw), len(data), 0, 12345))


def test_eocd_missing():
    with pytest.raises(ValueError):
        nr.parse_eocd(b"no zip here")


def test_resumable_stream_reconnects(monkeypatch):
    monkeypatch.setattr(nr.time, "sleep", lambda s: None)
    body = bytes(range(256)) * 50
    sess = FakeSession({"https://example.test/big": body}, fail_first_stream_after=1000)
    s = nr.ResumableStream("https://example.test/big", sess=sess)
    out = io.BufferedReader(s).read()
    assert out == body
    # the fake socket hands out 7 bytes per read and dies once 1000 bytes are through (at byte 143 * 7 = 1001),
    # so the stream must reconnect exactly once, asking for the rest from byte 1001
    assert sess.calls == [None, "bytes=1001-"]


def test_prefetch_keeps_order():
    import random
    import time as _t

    def slow(x):
        _t.sleep(random.random() * 0.01)
        return x * x

    assert list(nr.prefetch(range(30), slow, workers=4)) == [x * x for x in range(30)]
    assert list(nr.prefetch(range(5), slow, workers=1)) == [0, 1, 4, 9, 16]


def test_parallel_member_reads(split_zip):
    files, sess, parts = split_zip
    rz = nr.RemoteZip(parts, sess, limiter=nr.RateLimiter(0.0))
    got = dict(rz.iter_members(list(files), max_gap=0, workers=3))
    assert got == files
