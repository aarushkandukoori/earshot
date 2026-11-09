"""Offline tests for dataprep.speech_http - range reads and the parallel in-order stream, against a local HTTP
server that can hang up in the middle of a response."""

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from dataprep import speech_http


class Handler(BaseHTTPRequestHandler):
    blob = b""
    drop_first_after = None  # send only this many bytes of the first GET, then hang up
    ignore_range = False  # answer 200 with the whole file, like a server without range support
    etag = '"v1"'
    starts: list = []

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.blob)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", self.etag)
        self.end_headers()

    def do_GET(self):
        rng = None if self.ignore_range else self.headers.get("Range")
        start, end = 0, len(self.blob) - 1
        if rng:
            a, b = rng.split("=")[1].split("-")
            start, end = int(a), (int(b) if b else len(self.blob) - 1)
        Handler.starts.append(start)
        body = self.blob[start:end + 1]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", self.etag)
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.blob)}")
        self.end_headers()
        if self.drop_first_after and len(Handler.starts) == 1:
            self.wfile.write(body[: self.drop_first_after])
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def url(monkeypatch):
    monkeypatch.setattr(speech_http, "RETRY_BASE_S", 0.01)
    Handler.blob = bytes(range(256)) * 4000 + b"tail"
    Handler.drop_first_after, Handler.ignore_range, Handler.etag, Handler.starts = None, False, '"v1"', []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/f.bin"
    srv.shutdown()


def test_get_range(url):
    s = requests.Session()
    assert speech_http.get_range(s, url, 10, 19) == Handler.blob[10:20]
    Handler.drop_first_after = 5  # short read on the first try, then a retry succeeds
    Handler.starts = []
    assert speech_http.get_range(s, url, 100, 1099) == Handler.blob[100:1100]
    assert Handler.starts == [100, 100]


def test_get_range_refuses_whole_file_and_changed_file(url):
    s = requests.Session()
    Handler.ignore_range = True
    with pytest.raises(RuntimeError):
        speech_http.get_range(s, url, 0, 9)
    Handler.ignore_range = False
    with pytest.raises(RuntimeError):
        speech_http.get_range(s, url, 0, 9, validator='"other"')


def test_range_stream_reassembles_in_order(url):
    Handler.drop_first_after = 1000
    blob = Handler.blob
    with speech_http.RangeStream(url, connections=4, chunk_bytes=50_000) as f:
        got = b"".join(iter(lambda: f.read(7_777), b""))
        assert got == blob and f.pos == len(blob) and f.md5.hexdigest() == hashlib.md5(blob).hexdigest()
        assert f.read(10) == b"" and f.read_to_end() == 0
    assert len(Handler.starts) == len(set(Handler.starts)) + 1  # only the dropped chunk was fetched twice


def test_range_stream_read_to_end(url):
    with speech_http.RangeStream(url, connections=2, chunk_bytes=100_000) as f:
        head = f.read(123)
        assert head == Handler.blob[:123]
        assert f.read_to_end() == len(Handler.blob) - 123
        assert f.md5.hexdigest() == hashlib.md5(Handler.blob).hexdigest()
