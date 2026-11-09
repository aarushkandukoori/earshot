"""HTTP helpers for the noise and RIR fetchers - range reads, a resumable stream and a remote (split) zip reader.

Why this exists - FSD50K, DEMAND and OpenSLR 28 ship as big zips, and FSD50K is a *split* zip (FSD50K.dev_audio.z01 ..
.z05 + .zip). Python's zipfile refuses split archives, and we only want a few percent of the members anyway. So we read
the zip's central directory with HTTP range requests, then fetch only the byte ranges of the members we keep.

A split zip is just one long byte string cut into parts. Each central-directory entry says on which part ("disk") its
local header starts and at which offset inside that part. If we glue the parts end to end in our head, the member starts
at  sum(sizes of the earlier parts) + offset . Reads that cross a part boundary become two range requests.
"""

from __future__ import annotations

import collections
import io
import logging
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import requests

from .common import DiskFull

UA = {"User-Agent": "earshot-dataprep/1.0 (research data fetcher)"}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    return s


class RateLimiter:
    """Keeps at least min_interval seconds between requests (Zenodo allows 133 requests per minute per address)."""

    def __init__(self, min_interval: float = 0.0):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()   # several reader threads share one limiter

    def wait(self) -> None:
        with self._lock:
            dt = time.monotonic() - self._last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self._last = time.monotonic()


def prefetch(items: Iterable, fn: Callable, workers: int) -> Iterator:
    """fn(item) for every item on `workers` threads, results yielded in the original order. At most 2 x workers
    calls are in flight, so memory stays bounded. With workers <= 1 it is a plain loop."""
    if workers <= 1:
        for item in items:
            yield fn(item)
        return
    with ThreadPoolExecutor(workers) as ex:
        it = iter(items)
        futs = collections.deque(ex.submit(fn, x) for x in [y for _, y in zip(range(2 * workers), it)])
        while futs:
            res = futs.popleft().result()
            for nxt in it:
                futs.append(ex.submit(fn, nxt))
                break
            yield res


class RetryLater(IOError):
    """HTTP 429 or 5xx. wait = seconds the server asked us to wait (0 if it did not say)."""

    def __init__(self, msg: str, wait: float = 0.0):
        super().__init__(msg)
        self.wait = wait


def _respect_quota(r: requests.Response, log: logging.Logger | None) -> None:
    """Zenodo sends X-RateLimit-Remaining with every answer. When the shared quota (other downloads on this machine use
    the same address) is nearly gone, sleep until the window resets instead of running into HTTP 429."""
    rem = r.headers.get("X-RateLimit-Remaining")
    if rem is not None and rem.isdigit() and int(rem) <= 3:
        wait = min(float(r.headers.get("Retry-After", 30) or 30), 65.0)
        if log:
            log.info("rate-limit quota low (%s left), sleeping %.0f s", rem, wait)
        time.sleep(wait)


def get_range(sess: requests.Session, url: str, start: int, end: int, limiter: RateLimiter | None = None,
              log: logging.Logger | None = None, tries: int = 10) -> bytes:
    """Bytes [start, end] (inclusive) of url. Retries network errors and HTTP 429/5xx with backoff.
    Refuses a 200 answer (server ignored Range), so we never pull a whole multi-GB file by accident."""
    want = end - start + 1
    for i in range(tries):
        if limiter:
            limiter.wait()
        try:
            r = sess.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 120))
            try:
                if r.status_code == 429 or r.status_code >= 500:
                    raise RetryLater(f"HTTP {r.status_code}", float(r.headers.get("Retry-After", 0) or 0))
                if r.status_code != 206:
                    raise RuntimeError(f"expected 206 for a range request, got {r.status_code} from {url}")
                data = r.content
                _respect_quota(r, log)
            finally:
                r.close()
            if len(data) != want:
                raise IOError(f"short read {len(data)} of {want} bytes")
            return data
        except (DiskFull, RuntimeError):
            raise
        except Exception as e:
            if i == tries - 1:
                raise
            wait = max(min(2.0 * 2**i, 120.0), getattr(e, "wait", 0.0))
            if log:
                log.warning("range %d-%d of %s failed (%s: %s), retry %d/%d in %.0f s", start, end,
                            "/".join(url.split("/")[-3:]), type(e).__name__, e, i + 1, tries, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_bytes(sess: requests.Session, url: str, log: logging.Logger | None = None, tries: int = 8) -> bytes:
    """Whole (small) file into memory, with retries. Used for metadata zips and the 12 MB MIT IR zip."""
    for i in range(tries):
        try:
            r = sess.get(url, timeout=(30, 300))
            if r.status_code == 429 or r.status_code >= 500:
                raise RetryLater(f"HTTP {r.status_code}", float(r.headers.get("Retry-After", 0) or 0))
            r.raise_for_status()
            n = r.headers.get("Content-Length")
            if n is not None and int(n) != len(r.content):
                raise IOError(f"short read {len(r.content)} of {n} bytes")
            return r.content
        except Exception as e:
            if i == tries - 1:
                raise
            wait = max(min(2.0 * 2**i, 120.0), getattr(e, "wait", 0.0))
            if log:
                log.warning("GET %s failed (%s: %s), retry %d/%d in %.0f s", url, type(e).__name__, e, i + 1, tries, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def head_size(sess: requests.Session, url: str, tries: int = 6) -> int:
    """Content length of url (follows redirects)."""
    for i in range(tries):
        try:
            r = sess.head(url, allow_redirects=True, timeout=60)
            r.raise_for_status()
            return int(r.headers["Content-Length"])
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(2.0 * 2**i, 60.0))
    raise AssertionError("unreachable")


class ResumableStream(io.RawIOBase):
    """A read-only stream over one long HTTP GET. When the connection drops it reconnects with a Range header at the
    current byte, so a streaming decoder on top (gzip, tar) never notices. Used for musan.tar.gz (11 GB)."""

    def __init__(self, url: str, size: int | None = None, sess: requests.Session | None = None,
                 log: logging.Logger | None = None, tries: int = 30):
        super().__init__()
        self.url, self.size, self.log, self.tries = url, size, log, tries
        self.sess = sess or make_session()
        self.pos = 0
        self._resp = None

    def readable(self) -> bool:
        return True

    def _open(self) -> None:
        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        r = self.sess.get(self.url, headers=headers, stream=True, timeout=(30, 120))
        if self.pos and r.status_code != 206:
            r.close()
            raise RuntimeError(f"server ignored the Range header (HTTP {r.status_code}), cannot resume")
        r.raise_for_status()
        if self.size is None and not self.pos:
            self.size = int(r.headers.get("Content-Length", 0)) or None
        r.raw.decode_content = False
        self._resp = r

    def _drop(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
        self._resp = None

    def readinto(self, b) -> int:
        if self.size is not None and self.pos >= self.size:
            return 0
        for i in range(self.tries):
            try:
                if self._resp is None:
                    self._open()
                n = self._resp.raw.readinto(b)
                if n == 0 and self.size is not None and self.pos < self.size:
                    raise IOError(f"connection ended early at byte {self.pos} of {self.size}")
                self.pos += n
                return n
            except RuntimeError:
                raise
            except Exception as e:
                self._drop()
                if i == self.tries - 1:
                    raise
                wait = min(2.0 * 2**i, 120.0)
                if self.log:
                    self.log.warning("stream error at byte %d (%s: %s), reconnecting in %.0f s", self.pos,
                                     type(e).__name__, e, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._drop()
        super().close()


# ---------------------------------------------------------------------------------------------------------------------
# Remote zip reader (single or split archives)
# ---------------------------------------------------------------------------------------------------------------------

EOCD_SIG = b"PK\x05\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CDIR_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"


@dataclass
class Member:
    name: str
    method: int        # 0 = stored, 8 = deflate
    csize: int         # compressed size
    usize: int         # uncompressed size
    offset: int        # offset of the local header in the glued-together archive
    crc: int
    extra_len: int = 0  # length of the central-directory extra field (local one is usually the same)


@dataclass
class Span:
    """One HTTP read that covers one or more neighbouring members."""
    start: int
    end: int           # exclusive
    members: list = field(default_factory=list)


def parse_eocd(tail: bytes) -> dict:
    """Find and parse the end-of-central-directory record in the last bytes of the (last part of the) archive.
    Returns disk numbers, entry count, central directory size and offset, and where a ZIP64 locator would sit."""
    i = tail.rfind(EOCD_SIG)
    if i < 0:
        raise ValueError("no end-of-central-directory record found")
    (_, disk, cd_disk, n_here, n_total, cd_size, cd_off, _clen) = struct.unpack("<4s4H2LH", tail[i:i + 22])
    out = {"pos_in_tail": i, "disk": disk, "cd_disk": cd_disk, "n_total": n_total, "cd_size": cd_size,
           "cd_offset": cd_off, "zip64": False}
    j = i - 20
    if j >= 0 and tail[j:j + 4] == ZIP64_LOCATOR_SIG:
        _, z64_disk, z64_off, n_disks = struct.unpack("<4sLQL", tail[j:j + 20])
        out.update(zip64=True, z64_disk=z64_disk, z64_offset=z64_off, n_disks=n_disks)
    return out


def parse_zip64_eocd(rec: bytes) -> dict:
    if rec[:4] != ZIP64_EOCD_SIG:
        raise ValueError("bad ZIP64 end-of-central-directory signature")
    (_, _size, _vm, _vn, disk, cd_disk, _n_here, n_total, cd_size, cd_off) = struct.unpack("<4sQ2H2L4Q", rec[:56])
    return {"disk": disk, "cd_disk": cd_disk, "n_total": n_total, "cd_size": cd_size, "cd_offset": cd_off}


def parse_central_directory(cd: bytes, part_starts: list[int]) -> list[Member]:
    """Parse central-directory entries. part_starts[k] = global offset where part (disk) k begins."""
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == CDIR_SIG:
        (_, _vm, _vn, _flag, method, _t, _d, crc, csize, usize, nlen, elen, clen, disk, _ia, _ea,
         off) = struct.unpack("<4s4H2H3L5H2L", cd[p:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        # ZIP64 extra field (id 1) carries the 64-bit values, only for the fields that are maxed out above
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<2H", extra[q:q + 4])
            if hid == 1:
                vals = extra[q + 4:q + 4 + hlen]
                k = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                if disk == 0xFFFF:
                    disk = struct.unpack("<L", vals[k:k + 4])[0]; k += 4
            q += 4 + hlen
        out.append(Member(name, method, csize, usize, part_starts[disk] + off, crc, elen))
        p += 46 + nlen + elen + clen
    return out


def plan_spans(members: Iterable[Member], max_gap: int, max_span: int, slack: int = 1024) -> list[Span]:
    """Group members (sorted by offset) into reads. Two members share a read when the gap between them is at most
    max_gap bytes and the read stays under max_span bytes. slack covers the local header's name and extra field."""
    spans: list[Span] = []
    for m in sorted(members, key=lambda m: m.offset):
        end = m.offset + 30 + len(m.name.encode()) + m.extra_len + slack + m.csize
        if spans and m.offset - spans[-1].end <= max_gap and end - spans[-1].start <= max_span:
            spans[-1].end = max(spans[-1].end, end)
            spans[-1].members.append(m)
        else:
            spans.append(Span(m.offset, end, [m]))
    return spans


def extract_from(buf: bytes, buf_start: int, m: Member) -> bytes | None:
    """Pull member m out of buf (which starts at global offset buf_start). Returns None if buf is too short.
    Raises ValueError on a bad local header or a CRC / size mismatch (corrupt transfer)."""
    p = m.offset - buf_start
    if buf[p:p + 4] != LOCAL_SIG:
        raise ValueError(f"bad local header for {m.name}")
    nlen, elen = struct.unpack("<2H", buf[p + 26:p + 30])
    a = p + 30 + nlen + elen
    if a + m.csize > len(buf):
        return None
    raw = buf[a:a + m.csize]
    if m.method == 0:
        data = raw
    elif m.method == 8:
        data = zlib.decompress(raw, -15)
    else:
        raise ValueError(f"unsupported compression method {m.method} for {m.name}")
    if len(data) != m.usize or (zlib.crc32(data) & 0xFFFFFFFF) != m.crc:
        raise ValueError(f"size or CRC mismatch for {m.name}")
    return data


class RemoteZip:
    """Read members of a remote zip (one part, or a split zip given as all its parts in order) via range requests.
    parts = [(url, size), ...] in disk order (for FSD50K - .z01, .z02, ..., then .zip)."""

    def __init__(self, parts: list[tuple[str, int]], sess: requests.Session | None = None,
                 limiter: RateLimiter | None = None, log: logging.Logger | None = None):
        self.parts = parts
        self.sess = sess or make_session()
        self.limiter, self.log = limiter, log
        self.starts = [0]
        for _, size in parts:
            self.starts.append(self.starts[-1] + size)
        self.total = self.starts[-1]
        self.requests = 0       # range requests made (for progress logs)
        self.bytes = 0          # bytes downloaded
        self._count_lock = threading.Lock()
        self.members = {m.name: m for m in self._read_directory()}

    def read_at(self, start: int, n: int) -> bytes:
        """n bytes starting at global offset start, split over parts as needed."""
        out = []
        while n > 0:
            k = max(i for i in range(len(self.parts)) if self.starts[i] <= start)
            url, size = self.parts[k]
            local = start - self.starts[k]
            take = min(n, size - local)
            out.append(get_range(self.sess, url, local, local + take - 1, self.limiter, self.log))
            with self._count_lock:
                self.requests += 1
                self.bytes += take
            start += take
            n -= take
        return b"".join(out)

    def _read_directory(self) -> list[Member]:
        url, size = self.parts[-1]
        tail_len = min(size, 65536 + 22 + 20)
        tail = get_range(self.sess, url, size - tail_len, size - 1, self.limiter, self.log)
        e = parse_eocd(tail)
        if e["zip64"]:
            z = parse_zip64_eocd(self.read_at(self.starts[e["z64_disk"]] + e["z64_offset"], 56))
            e.update(z)
        cd = self.read_at(self.starts[e["cd_disk"]] + e["cd_offset"], e["cd_size"])
        members = parse_central_directory(cd, self.starts)
        if len(members) != e["n_total"]:
            raise ValueError(f"central directory has {len(members)} entries, EOCD says {e['n_total']}")
        return members

    def read_member(self, m: Member) -> bytes:
        """One member on its own (two reads - the local header to learn its exact length, then the data)."""
        nlen, elen = struct.unpack("<2H", self.read_at(m.offset, 30)[26:30])
        data = extract_from(self.read_at(m.offset, 30 + nlen + elen + m.csize), m.offset, m)
        assert data is not None
        return data

    def iter_members(self, names: Iterable[str], max_gap: int = 256 << 10, max_span: int = 48 << 20,
                     workers: int = 1) -> Iterator[tuple[str, bytes | None]]:
        """Yield (name, uncompressed bytes) for the given members in archive order, reading neighbours together
        (workers > 1 runs that many range requests at once). A member that fails its CRC check is read again on
        its own; if that fails too, it is yielded with None."""
        spans = plan_spans([self.members[n] for n in names], max_gap, max_span)
        reads = prefetch(spans, lambda sp: self.read_at(sp.start, min(sp.end, self.total) - sp.start), workers)
        for span, buf in zip(spans, reads):
            for m in span.members:
                try:
                    data = extract_from(buf, span.start, m)
                    if data is None:  # local extra field longer than our slack - read this one alone
                        data = self.read_member(m)
                except ValueError as e:
                    if self.log:
                        self.log.warning("%s - reading it again on its own", e)
                    try:
                        data = self.read_member(m)
                    except ValueError as e2:
                        if self.log:
                            self.log.error("giving up on %s (%s)", m.name, e2)
                        data = None
                yield m.name, data
