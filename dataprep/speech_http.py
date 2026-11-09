"""HTTP helpers for the speech fetchers - a parallel forward-only stream and single range reads.

RangeStream is what lets LibriTTS-R stream an 8 to 46 GB tar.gz without saving it. tarfile only ever calls read(n)
on it. Behind that, the file is fetched as consecutive 8 MB range requests, a few of them in flight at once, and handed
out strictly in order. Two reasons -
  - Speed - one connection to openslr.org was often slow (1.0 to 1.2 MB/s per stream for the first minutes of the
    2026-11-07 run, while short curl tests got 4 to 7 MB/s); with 3 connections per tarball each tarball then ran at
    about 4.5 MB/s.
  - Robustness - a dropped connection only costs one chunk, which is retried with backoff; the gzip decoder inside
    tarfile never notices.
It also computes the MD5 of every byte handed out, so at the end we can compare with OpenSLR's md5sum.txt.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests

from .common import retry

# identity encoding: we want the raw bytes of the file, never a server-side gzip of it
UA = {"User-Agent": "earshot-dataprep/1.0 (research data fetcher)", "Accept-Encoding": "identity"}
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 120
RETRY_BASE_S = 2.0  # first wait after a failed request; doubles each time, capped at 2 minutes


def get_range(sess: requests.Session, url: str, start: int, end: int, tries: int = 8,
              log: logging.Logger | None = None, validator: str | None = None) -> bytes:
    """Bytes [start, end] (inclusive) of url. Retries network errors and HTTP 429/5xx with backoff.
    Refuses a 200 answer (server ignored Range), so a bug can never pull a whole multi-GB file by accident.
    validator = the ETag (or Last-Modified) seen first; a different one means the remote file changed."""
    want = end - start + 1
    for i in range(tries):
        try:
            r = sess.get(url, headers={**UA, "Range": f"bytes={start}-{end}"}, stream=True,
                         timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S))
            try:
                if r.status_code == 429 or r.status_code >= 500:
                    raise IOError(f"HTTP {r.status_code}")
                if r.status_code != 206:
                    raise RuntimeError(f"expected HTTP 206 for a range request, got {r.status_code}")
                v = r.headers.get("ETag") or r.headers.get("Last-Modified")
                if validator and v and v != validator:
                    raise RuntimeError(f"the remote file changed ({v!r} instead of {validator!r})")
                chunks, got = [], 0
                while got <= want:  # read one byte past the end to notice a server that sends too much
                    b = r.raw.read(min(1 << 20, want + 1 - got), decode_content=False)
                    if not b:
                        break
                    chunks.append(b)
                    got += len(b)
                data = b"".join(chunks)
            finally:
                r.close()
            if len(data) != want:
                raise IOError(f"short read, {len(data)} of {want} bytes")
            return data
        except RuntimeError:
            raise
        except Exception as e:  # requests, urllib3 and socket errors come in many types
            if i == tries - 1:
                raise
            wait = min(RETRY_BASE_S * 2**i, 120.0)
            if log:
                log.warning("range %d-%d failed (%s: %s), retrying in %.0f s", start, end, type(e).__name__, e, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


class RangeStream:
    """Read-only, forward-only file object over one URL, fetched as parallel range requests and served in order."""

    def __init__(self, url: str, log: logging.Logger | None = None, label: str = "", connections: int = 3,
                 chunk_bytes: int = 8 << 20, tries: int = 15, progress_every_s: float = 120.0):
        self.url, self.log = url, log
        self.label = label or url.rsplit("/", 1)[-1]
        self.chunk = chunk_bytes
        self.tries = tries
        self.connections = max(1, connections)
        self._local = threading.local()
        self.size, self.validator = retry(self._head, log=log)
        self._pool = ThreadPoolExecutor(self.connections, thread_name_prefix=f"{self.label}-get")
        self._futures: deque = deque()
        self._next = 0  # first byte not yet requested
        self._buf, self._off = b"", 0
        self.pos = 0  # bytes handed to the reader so far
        self.md5 = hashlib.md5()
        self._t0 = self._last_log = time.monotonic()
        self._progress_every_s = progress_every_s
        for _ in range(self.connections + 1):  # one extra so a connection never waits for the reader
            self._submit()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = requests.Session()
            s.headers.update(UA)
        return s

    def _head(self) -> tuple[int, str | None]:
        r = requests.head(self.url, headers=UA, allow_redirects=True, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S))
        if r.status_code == 429 or r.status_code >= 500:
            raise IOError(f"HTTP {r.status_code}")
        r.raise_for_status()
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError(f"{self.url} does not advertise range requests")
        return int(r.headers["Content-Length"]), r.headers.get("ETag") or r.headers.get("Last-Modified")

    def _fetch(self, start: int, end: int) -> bytes:
        return get_range(self._session(), self.url, start, end, tries=self.tries, log=self.log,
                         validator=self.validator)

    def _submit(self) -> None:
        if self._next < self.size:
            end = min(self._next + self.chunk, self.size) - 1
            self._futures.append(self._pool.submit(self._fetch, self._next, end))
            self._next = end + 1

    def read(self, n: int = -1) -> bytes:
        if self._off >= len(self._buf):
            if not self._futures:
                return b""
            self._buf, self._off = self._futures.popleft().result(), 0  # re-raises a chunk that failed for good
            self._submit()
        n = len(self._buf) - self._off if n is None or n < 0 else n
        out = self._buf[self._off:self._off + n]
        self._off += len(out)
        self.pos += len(out)
        self.md5.update(out)
        self._maybe_log()
        return out

    def read_to_end(self) -> int:
        """Consume what the reader left (tar padding) so the MD5 covers the whole file. Returns bytes read."""
        n = 0
        while True:
            b = self.read(1 << 20)
            if not b:
                return n
            n += len(b)

    def rate_mb_s(self) -> float:
        return self.pos / 1e6 / max(time.monotonic() - self._t0, 1e-6)

    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if self.log and now - self._last_log >= self._progress_every_s:
            self._last_log = now
            self.log.info("%s: %.2f / %.2f GB (%.1f %%), %.1f MB/s average over %d connections", self.label,
                          self.pos / 1e9, self.size / 1e9, 100.0 * self.pos / self.size, self.rate_mb_s(),
                          self.connections)

    def close(self) -> None:
        for f in self._futures:
            f.cancel()
        self._futures.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "RangeStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
