"""HTTP helpers for the evaluation fetchers.

Two ways to read a big remote file without saving it:

- RangeFile - a read-only, seekable file made of HTTP range requests. Python's zipfile and pyarrow's parquet
  reader accept it like a local file, so they fetch only the zip central directory / parquet footer and the
  members or row groups we ask for. Used for wham_noise.zip (18 GB) and the Hugging Face parquet shards.
- HttpStream - one long download read from the start. When the connection drops it reconnects with a Range header
  at the current byte, so a gzip/tar reader on top never notices. Used for DipCo.tgz, which we read only partly.

Every request is retried with exponential backoff (2, 4, 8 ... s, capped at 120 s) on network errors, short reads
and HTTP 429/5xx. A range request answered with HTTP 200 (server ignored the Range header) is an error, so we never
pull a whole multi-GB file by accident.
"""

from __future__ import annotations

import io
import logging
import time

import requests

from .common import DiskFull

UA = {"User-Agent": "earshot-dataprep/1.0 (research eval fetcher)"}
CHUNK = 16 << 20  # large reads are split into 16 MB range requests, each retried on its own


class RetryableHTTP(IOError):
    """HTTP 429 or 5xx, or a short read. wait = seconds the server asked us to wait (0 if it did not say)."""

    def __init__(self, msg: str, wait: float = 0.0):
        super().__init__(msg)
        self.wait = wait


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    return s


def backoff_s(attempt: int, server_wait: float = 0.0) -> float:
    """Seconds to sleep before retry number attempt+1: 2, 4, 8 ... capped at 120, or longer if the server asked."""
    return max(min(2.0 * 2**attempt, 120.0), server_wait)


def _short(url: str) -> str:
    return "/".join(url.split("?")[0].split("/")[-2:])


def get_range(sess: requests.Session, url: str, start: int, end: int, log: logging.Logger | None = None,
              tries: int = 10) -> bytes:
    """Bytes [start, end) of url (end exclusive)."""
    for i in range(tries):
        try:
            r = sess.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=(30, 180))
            if r.status_code == 429 or r.status_code >= 500:
                raise RetryableHTTP(f"HTTP {r.status_code}", float(r.headers.get("Retry-After", 0) or 0))
            if r.status_code != 206:
                raise RuntimeError(f"expected HTTP 206 for a range request, got {r.status_code} from {url}")
            if len(r.content) != end - start:
                raise RetryableHTTP(f"short read, {len(r.content)} of {end - start} bytes")
            return r.content
        except (RuntimeError, DiskFull):
            raise
        except Exception as e:  # network errors come in many types
            if i == tries - 1:
                raise
            wait = backoff_s(i, getattr(e, "wait", 0.0))
            if log:
                log.warning("range %d-%d of %s failed (%s: %s), retry %d/%d in %.0f s", start, end, _short(url),
                            type(e).__name__, e, i + 1, tries - 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def remote_size(sess: requests.Session, url: str, log: logging.Logger | None = None, tries: int = 8) -> int:
    """Size in bytes of a remote file, from a 1-byte range request (works on S3, Hugging Face and GitHub)."""
    for i in range(tries):
        try:
            r = sess.get(url, headers={"Range": "bytes=0-0"}, timeout=(30, 60))
            if r.status_code == 429 or r.status_code >= 500:
                raise RetryableHTTP(f"HTTP {r.status_code}", float(r.headers.get("Retry-After", 0) or 0))
            if r.status_code != 206 or "/" not in r.headers.get("Content-Range", ""):
                raise RuntimeError(f"no range support at {url} (HTTP {r.status_code})")
            return int(r.headers["Content-Range"].rsplit("/", 1)[1])
        except RuntimeError:
            raise
        except Exception as e:
            if i == tries - 1:
                raise
            wait = backoff_s(i, getattr(e, "wait", 0.0))
            if log:
                log.warning("size of %s failed (%s: %s), retrying in %.0f s", _short(url), type(e).__name__, e, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def get_bytes(sess: requests.Session, url: str, log: logging.Logger | None = None, tries: int = 8) -> bytes:
    """Whole (small) file, with retries. Use only for metadata files of a few MB."""
    for i in range(tries):
        try:
            r = sess.get(url, timeout=(30, 120))
            if r.status_code == 429 or r.status_code >= 500:
                raise RetryableHTTP(f"HTTP {r.status_code}", float(r.headers.get("Retry-After", 0) or 0))
            r.raise_for_status()
            return r.content
        except Exception as e:
            if i == tries - 1 or (isinstance(e, requests.HTTPError) and e.response is not None
                                  and e.response.status_code in (401, 403, 404)):
                raise
            wait = backoff_s(i, getattr(e, "wait", 0.0))
            if log:
                log.warning("GET %s failed (%s: %s), retrying in %.0f s", _short(url), type(e).__name__, e, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


class RangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests.

    Reads inside the current buffer cost nothing. A small read outside it loads `readahead` bytes; a big read goes
    straight to the network in 16 MB pieces. prefetch(start, end) loads one big block, which is how we read many
    neighbouring zip members with one request."""

    def __init__(self, url: str, size: int | None = None, sess: requests.Session | None = None,
                 log: logging.Logger | None = None, readahead: int = 256 << 10):
        super().__init__()
        self.url, self.log, self.readahead = url, log, readahead
        self.sess = sess or make_session()
        self.size = size if size is not None else remote_size(self.sess, url, log)
        self.pos = 0
        self.fetched = 0  # bytes downloaded so far
        self._buf, self._buf_start = b"", 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def fetch(self, start: int, end: int) -> bytes | bytearray:
        """Bytes [start, end). Reads over 16 MB are filled piece by piece into one preallocated buffer, so a big
        parquet column chunk is held in memory once, not twice (pyarrow and zipfile accept a bytearray)."""
        if end - start <= CHUNK:
            data = get_range(self.sess, self.url, start, end, self.log)
        else:
            data = bytearray(end - start)
            for a in range(start, end, CHUNK):
                b = min(end, a + CHUNK)
                data[a - start:b - start] = get_range(self.sess, self.url, a, b, self.log)
        self.fetched += len(data)
        return data

    def prefetch(self, start: int, end: int) -> None:
        end = min(end, self.size)
        self._buf, self._buf_start = self.fetch(start, end), start

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        a, b = self.pos, self.pos + n
        if self._buf_start <= a and b <= self._buf_start + len(self._buf):
            out = self._buf[a - self._buf_start:b - self._buf_start]
        elif n < self.readahead:
            self.prefetch(a, a + self.readahead)
            out = self._buf[:n]
        else:
            out = self.fetch(a, b)
        self.pos = b
        return out

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


class HttpStream(io.RawIOBase):
    """Sequential reader over one HTTP download that reconnects at the current byte when the connection drops."""

    def __init__(self, url: str, sess: requests.Session | None = None, log: logging.Logger | None = None,
                 tries: int = 30):
        super().__init__()
        self.url, self.log, self.tries = url, log, tries
        self.sess = sess or make_session()
        self.pos = 0
        self.size: int | None = None
        self._r = None

    def readable(self) -> bool:
        return True

    def _open(self) -> None:
        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        r = self.sess.get(self.url, headers=headers, stream=True, timeout=(30, 180))
        if r.status_code == 429 or r.status_code >= 500:
            r.close()
            raise RetryableHTTP(f"HTTP {r.status_code}", float(r.headers.get("Retry-After", 0) or 0))
        if self.pos and r.status_code != 206:
            r.close()
            raise RuntimeError(f"server ignored the Range header (HTTP {r.status_code}), cannot resume the stream")
        r.raise_for_status()
        if self.size is None and not self.pos:
            self.size = int(r.headers.get("Content-Length", 0)) or None
        r.raw.decode_content = False
        self._r = r

    def _drop(self) -> None:
        if self._r is not None:
            try:
                self._r.close()
            except Exception:
                pass
        self._r = None

    def readinto(self, b) -> int:
        if self.size is not None and self.pos >= self.size:
            return 0
        for i in range(self.tries):
            try:
                if self._r is None:
                    self._open()
                n = self._r.raw.readinto(b)
                if n == 0 and self.size is not None and self.pos < self.size:
                    raise RetryableHTTP(f"connection ended at byte {self.pos} of {self.size}")
                self.pos += n
                return n
            except RuntimeError:
                raise
            except Exception as e:
                self._drop()
                if i == self.tries - 1:
                    raise
                wait = backoff_s(i, getattr(e, "wait", 0.0))
                if self.log:
                    self.log.warning("stream error at byte %d (%s: %s), reconnecting in %.0f s", self.pos,
                                     type(e).__name__, e, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._drop()
        super().close()


def hf_url(repo: str, path: str, revision: str = "main") -> str:
    """Download URL of a file in a Hugging Face dataset repo."""
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
