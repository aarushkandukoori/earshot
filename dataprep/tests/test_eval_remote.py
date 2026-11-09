"""Tests for dataprep.eval_remote - range reads, the seekable RangeFile (with zipfile and pyarrow on top) and the
reconnecting HttpStream. Offline: a fake requests session serves an in-memory blob."""

import io
import tarfile
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dataprep.eval_remote as er


class FakeRaw:
    def __init__(self, data: bytes, drop_after: int | None = None):
        self.data, self.pos, self.drop_after = data, 0, drop_after
        self.decode_content = True

    def readinto(self, b) -> int:
        if self.drop_after is not None and self.pos >= self.drop_after:
            raise ConnectionError("connection reset")
        n = min(len(b), len(self.data) - self.pos)
        if self.drop_after is not None:
            n = min(n, self.drop_after - self.pos)
        b[:n] = self.data[self.pos:self.pos + n]
        self.pos += n
        return n


class FakeResp:
    def __init__(self, status: int, body: bytes, headers: dict, drop_after: int | None = None):
        self.status_code, self.content, self.headers = status, body, headers
        self.raw = FakeRaw(body, drop_after)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        pass


class FakeSession:
    """Serves `blob` with HTTP Range semantics. `fail` = list of status codes to answer first (then normal);
    `ignore_range` answers 200 with the whole blob; `drop_first_stream_after` cuts the first streamed reply."""

    def __init__(self, blob: bytes, fail=(), ignore_range=False, drop_first_stream_after=None):
        self.blob, self.fail, self.ignore_range = blob, list(fail), ignore_range
        self.drop = drop_first_stream_after
        self.calls = []

    def get(self, url, headers=None, stream=False, timeout=None):
        rng = (headers or {}).get("Range")
        self.calls.append(rng)
        if self.fail:
            return FakeResp(self.fail.pop(0), b"", {})
        if rng is None or self.ignore_range:
            drop, self.drop = (self.drop, None) if stream else (None, self.drop)
            return FakeResp(200, self.blob, {"Content-Length": str(len(self.blob))}, drop)
        a, b = rng.split("=")[1].split("-")
        a = int(a)
        b = int(b) if b else len(self.blob) - 1
        body = self.blob[a:b + 1]
        return FakeResp(206, body, {"Content-Range": f"bytes {a}-{b}/{len(self.blob)}"})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(er.time, "sleep", lambda s: None)


BLOB = bytes(range(256)) * 400


def test_get_range_and_size():
    s = FakeSession(BLOB)
    assert er.get_range(s, "u", 10, 20) == BLOB[10:20]
    assert er.remote_size(s, "u") == len(BLOB)


def test_get_range_retries_then_succeeds():
    s = FakeSession(BLOB, fail=[503, 429])
    assert er.get_range(s, "u", 0, 5) == BLOB[:5]
    assert len(s.calls) == 3


def test_get_range_refuses_whole_file_answer():
    with pytest.raises(RuntimeError):
        er.get_range(FakeSession(BLOB, ignore_range=True), "u", 0, 5)


def test_rangefile_read_seek_and_big_reads(monkeypatch):
    monkeypatch.setattr(er, "CHUNK", 1000)          # force multi-piece reads
    f = er.RangeFile("u", sess=FakeSession(BLOB), readahead=64)
    assert f.size == len(BLOB)
    f.seek(100)
    assert f.read(10) == BLOB[100:110]
    assert f.read(10) == BLOB[110:120]              # served from the read-ahead buffer
    f.seek(-50, io.SEEK_END)
    assert f.read() == BLOB[-50:]
    f.seek(0)
    assert bytes(f.read(5000)) == BLOB[:5000]       # 5 pieces of 1000 bytes into one buffer


def test_zipfile_over_rangefile_with_prefetch():
    buf = io.BytesIO()
    members = {f"wham_noise/tt/c{k}.wav": bytes([k]) * (3000 + 17 * k) for k in range(6)}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    sess = FakeSession(buf.getvalue())
    rf = er.RangeFile("u", sess=sess)
    zf = zipfile.ZipFile(rf)
    infos = zf.infolist()
    first, last = min(i.header_offset for i in infos), max(i.header_offset + 2000 for i in infos)
    rf.prefetch(first, last + 4096)
    n_calls = len(sess.calls)
    for i in infos:
        assert zf.read(i) == members[i.filename]
    assert len(sess.calls) == n_calls               # every member came out of the prefetched block


def test_parquet_over_rangefile():
    buf = io.BytesIO()
    pq.write_table(pa.table({"speaker": ["a", "b", "c"], "audio": [{"bytes": b"x" * 10, "path": "p"}] * 3}), buf,
                   row_group_size=2)
    pf = pq.ParquetFile(er.RangeFile("u", sess=FakeSession(buf.getvalue())))
    assert pf.metadata.num_row_groups == 2
    assert pf.read(columns=["speaker"]).column("speaker").to_pylist() == ["a", "b", "c"]
    assert pf.read_row_group(1, columns=["audio.path"]).column("audio").to_pylist() == [{"path": "p"}]


def test_httpstream_reconnects_mid_tar():
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w:gz") as tf:
        for k in range(3):
            data = bytes([k]) * 20000
            ti = tarfile.TarInfo(f"Dipco/audio/eval/S0{k}_U01.CH1.wav")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    blob = tbuf.getvalue()
    sess = FakeSession(blob, drop_first_stream_after=len(blob) // 2)
    raw = er.HttpStream("u", sess=sess)
    with tarfile.open(fileobj=io.BufferedReader(raw, 4096), mode="r|gz") as tf:
        names = [m.name for m in tf]
    assert len(names) == 3
    assert any(c and c.startswith("bytes=") for c in sess.calls)   # it resumed with a Range request
    assert raw.pos == len(blob)
