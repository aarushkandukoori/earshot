"""LibriTTS-R train (CC BY 4.0, OpenSLR 141) - stream the three train tarballs and keep up to 60 s per speaker.

Nothing big touches the disk. Each tarball is read once over HTTP through tarfile's stream mode ("r|gz") and only
the kept utterances are written (as 16 kHz FLAC). A single connection to openslr.org was often slow, so by default two
tarballs are streamed at once (other_500 in one lane, clean_100 then clean_360 in the other) and each is fetched over
3 connections (speech_http.RangeStream), 6 connections in all.

Restarting. A tar.gz cannot be resumed in the middle, so a restarted run streams the tarball again from the start
and skips every speaker listed in data/speech/librittsr/_state/done_speakers.txt without decoding anything (the
download is the only cost). A dropped connection does not restart anything: RangeStream retries only the 8 MB chunk
that failed. If a chunk still fails after about 20 minutes of retries, the lane waits 5 minutes and streams that
tarball again (up to 3 times). Finished tarballs are listed in _state/done_tarballs.txt and never fetched again.

Selection for one speaker (the tar stores a speaker's chapters next to each other, so we buffer one speaker at a time,
at most about 90 MB of 24 kHz audio)
  - utterances of 2 to 15 s that are not in libritts_r_failed_speech_restoration_examples,
  - chapters are visited round-robin, utterances inside a chapter in a fixed pseudo-random order, so the 60 s are
    spread over every chapter the speaker has (at least 2 when there are 2),
  - a clip is skipped when it would push the speaker past 60 s, or when it is clipped or silent.
Gender comes from SPEAKERS.txt in doc.tar.gz. LibriTTS-R does not label accents, so accent is null.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import requests
import soundfile as sf

from .common import get_logger, retry, split_for
from .speech_common import (RAW, BudgetFull, Budget, Ctx, DiskFull, DoneSet, SourceLock, SpeechManifests,
                            reuse_existing, stable_hash, state_dir, store_clip)
from .speech_http import UA, RangeStream

SOURCE = "librittsr"
BASE = "https://www.openslr.org/resources/141/"
TARBALLS = {"train_clean_100": "train-clean-100", "train_clean_360": "train-clean-360",
            "train_other_500": "train-other-500"}
SIZE_GB = {"train_clean_100": 8.1, "train_clean_360": 29.0, "train_other_500": 46.0}  # only used for scheduling
RAW_DIR = RAW / "librittsr"
SECONDS_PER_SPEAKER = 60.0
MIN_S, MAX_S = 2.0, 15.0
SR_IN = 24000


# ---------------------------------------------------------------------------------------------------------------
# parsers (pure functions, tested in tests/test_speech_librittsr.py)


def parse_md5sums(text: str) -> dict[str, str]:
    """md5sum.txt -> {file name: md5}."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0].lower()
    return out


def parse_speakers(text: str) -> dict[str, str]:
    """SPEAKERS.txt ('14   | F | train-clean-360  | 25.03 | Kristin LeMoine') -> {'14': 'F'}."""
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("F", "M"):
            out[parts[0]] = parts[1]
    return out


def parse_bad_list(text: str) -> set[str]:
    """'./train-clean-100/1040/133433/1040_133433_000042_000000.wav' lines -> {'1040_133433_000042_000000'}."""
    out = set()
    for line in text.splitlines():
        name = line.strip().rsplit("/", 1)[-1]
        if name.endswith(".wav"):
            out.add(name[:-4])
    return out


def parse_member(name: str) -> tuple[str, str, str, str, str] | None:
    """'LibriTTS_R/train-clean-100/730/358/730_358_000005_000000.wav'
    -> ('train-clean-100', '730', '358', '730_358_000005_000000', 'wav'). Text members give kind 'text'.
    Anything else (directories, *.original.txt, *.trans.tsv, docs) -> None."""
    parts = name.removeprefix("./").split("/")
    if len(parts) != 5 or parts[0] != "LibriTTS_R":
        return None
    _, subset, spk, chap, fname = parts
    for suffix, kind in ((".wav", "wav"), (".normalized.txt", "text")):
        if fname.endswith(suffix):
            utt = fname[: -len(suffix)]
            if utt.startswith(f"{spk}_{chap}_"):
                return subset, spk, chap, utt, kind
    return None


def wav_seconds(member_size: int, sr: int = SR_IN) -> float:
    """Duration of a 16-bit mono PCM WAV from its size (LibriTTS-R files have the plain 44-byte header)."""
    return max(member_size - 44, 0) / (2 * sr)


def round_robin_order(chapter_of: dict[str, str]) -> list[str]:
    """Utterances ordered chapter by chapter in turns: first utterance of every chapter, then the second, ...
    Inside a chapter the order is a fixed pseudo-random one (sha1 of the utterance id)."""
    by_ch: dict[str, list[str]] = defaultdict(list)
    for utt, ch in chapter_of.items():
        by_ch[ch].append(utt)
    chapters = sorted(by_ch, key=lambda c: (stable_hash(c), c))
    queues = [sorted(by_ch[c], key=lambda u: (stable_hash(u), u)) for c in chapters]
    order: list[str] = []
    for i in range(max((len(q) for q in queues), default=0)):
        order += [q[i] for q in queues if i < len(q)]
    return order


def pick_within_budget(order: list[str], seconds: dict[str, float], budget_s: float, accept,
                       min_s: float = MIN_S) -> tuple[list[str], float]:
    """Walk `order`; skip a clip that would overflow budget_s; accept(utt) stores it and returns its real duration
    (or None when it was rejected). Stops once less than min_s of budget is left."""
    kept, total = [], 0.0
    for utt in order:
        if budget_s - total < min_s:
            break
        if total + seconds[utt] > budget_s + 1e-6:
            continue
        d = accept(utt)
        if d is not None:
            kept.append(utt)
            total += d
    return kept, total


# ---------------------------------------------------------------------------------------------------------------
# metadata (small files, saved under data/raw/librittsr)


@dataclass
class Meta:
    md5: dict[str, str]
    gender: dict[str, str]
    bad: set[str]


def _fetch_small(name: str, md5: str | None, log) -> bytes:
    p = RAW_DIR / name
    if p.exists() and (md5 is None or hashlib.md5(p.read_bytes()).hexdigest() == md5):
        return p.read_bytes()

    def get() -> bytes:
        r = requests.get(BASE + name, headers=UA, timeout=(30, 120))
        r.raise_for_status()
        if md5 and hashlib.md5(r.content).hexdigest() != md5:
            raise IOError(f"md5 mismatch for {name}")
        return r.content

    data = retry(get, log=log)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return data


def _tar_texts(data: bytes, want) -> dict[str, str]:
    out = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf:
            if m.isfile() and want(m.name):
                out[m.name] = tf.extractfile(m).read().decode("utf-8", "replace")
    return out


def load_meta(log) -> Meta:
    md5 = parse_md5sums(_fetch_small("md5sum.txt", None, log).decode())
    doc = _fetch_small("doc.tar.gz", md5.get("doc.tar.gz"), log)
    speakers = _tar_texts(doc, lambda n: n.endswith("/SPEAKERS.txt"))
    if not speakers:
        raise RuntimeError("SPEAKERS.txt not found in doc.tar.gz")
    gender = parse_speakers(next(iter(speakers.values())))
    bad_tgz = _fetch_small("libritts_r_failed_speech_restoration_examples.tar.gz",
                           md5.get("libritts_r_failed_speech_restoration_examples.tar.gz"), log)
    bad = set()
    for text in _tar_texts(bad_tgz, lambda n: n.endswith("_bad_sample_list.txt")).values():
        bad |= parse_bad_list(text)
    log.info("metadata - %d speakers with gender, %d failed-restoration utterances to skip", len(gender), len(bad))
    return Meta(md5=md5, gender=gender, bad=bad)


# ---------------------------------------------------------------------------------------------------------------
# streaming


@dataclass
class SpeakerBuffer:
    chapter: dict[str, str] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    wav: dict[str, bytes] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)


class Run:
    """State shared by the stream threads of one run."""

    def __init__(self, ctx: Ctx, meta: Meta, max_speakers: int | None, connections: int = 3):
        self.ctx = ctx
        self.meta = meta
        self.max_speakers = max_speakers
        self.connections = connections
        self.done_speakers = DoneSet(state_dir(SOURCE) / "done_speakers.txt")
        self.done_tarballs = DoneSet(state_dir(SOURCE) / "done_tarballs.txt")
        self.kept_seconds = 0.0
        self.speakers = 0
        self.error: BaseException | None = None
        self.lock = threading.Lock()

    def finish_speaker(self, subset: str, spk: str, buf: SpeakerBuffer) -> None:
        ctx = self.ctx
        speaker = f"{SOURCE}:{spk}"
        split = split_for(speaker)
        gender = self.meta.gender.get(spk)

        def accept(utt: str) -> float | None:
            rel = f"speech/{SOURCE}/{spk}/{utt}.flac"
            info = {"speaker": speaker, "split": split, "gender": gender, "accent": None, "group": buf.chapter[utt],
                    "extra": {"subset": subset, "orig_sr": SR_IN, "text": buf.text.get(utt)}}
            d = reuse_existing(ctx, rel, info)
            if d is not None:
                return d
            try:
                x, sr = sf.read(io.BytesIO(buf.wav[utt]), dtype="float32")
            except Exception as e:  # one broken file must not stop a 46 GB stream
                ctx.log.warning("%s: cannot decode (%s)", utt, e)
                ctx.count("decode_error")
                return None
            if not MIN_S <= len(x) / sr <= MAX_S:
                ctx.count("bad_length")
                return None
            return store_clip(ctx, rel, x, sr, info)

        kept, total = pick_within_budget(round_robin_order(buf.chapter), buf.seconds, SECONDS_PER_SPEAKER, accept)
        self.done_speakers.add(f"{subset}/{spk}")
        chapters = len({buf.chapter[u] for u in kept})
        with self.lock:
            self.speakers += 1
            self.kept_seconds += total
            n = self.speakers
        ctx.count("speakers")
        if not kept:
            ctx.count("speakers_with_nothing_kept")
        elif chapters < 2 and len(set(buf.chapter.values())) >= 2:
            ctx.count("speakers_one_chapter_despite_two")
        if n % 50 == 0:
            ctx.log.info("%d speakers finished this run, %.2f h kept, budget %.2f / %.2f GB, counts %s", n,
                         self.kept_seconds / 3600, ctx.budget.used / 1e9, ctx.budget.cap / 1e9, dict(ctx.counts))

    def stream(self, key: str) -> bool:
        """Stream one tarball. Returns True when it was read to the end and its MD5 matched."""
        ctx, log = self.ctx, self.ctx.log
        subset = TARBALLS[key]
        url = f"{BASE}{key}.tar.gz"
        finished_here = 0
        log.info("%s: streaming %s", key, url)
        with RangeStream(url, log, label=key, connections=self.connections) as f:
            tf = tarfile.open(fileobj=f, mode="r|gz", bufsize=1 << 20)
            cur, buf = None, None
            for m in tf:
                if len(tf.members) > 2000:
                    tf.members.clear()  # stream mode keeps every header otherwise (600k of them in other_500)
                if ctx.stop.is_set():
                    return False
                p = parse_member(m.name) if m.isfile() else None
                if p is None:
                    continue
                sub, spk, chap, utt, kind = p
                if sub != subset:
                    continue
                if spk != cur:
                    if buf is not None:
                        self.finish_speaker(subset, cur, buf)
                        finished_here += 1
                        if self.max_speakers and finished_here >= self.max_speakers:
                            log.info("%s: stopping after %d speakers (--max-speakers)", key, finished_here)
                            return False
                    cur = spk
                    buf = None if f"{subset}/{spk}" in self.done_speakers else SpeakerBuffer()
                if buf is None:
                    continue  # finished in an earlier run - skip without decoding
                if kind == "text":
                    buf.text[utt] = tf.extractfile(m).read().decode("utf-8", "replace").strip()
                    continue
                if utt in self.meta.bad:
                    ctx.count("failed_restoration_skipped")
                    continue
                secs = wav_seconds(m.size)
                if not MIN_S - 0.01 <= secs <= MAX_S + 0.01:
                    continue
                buf.chapter[utt], buf.seconds[utt] = chap, secs
                buf.wav[utt] = tf.extractfile(m).read()
            if buf is not None:
                self.finish_speaker(subset, cur, buf)
            f.read_to_end()
            got, want = f.md5.hexdigest(), self.meta.md5.get(f"{key}.tar.gz")
            log.info("%s: finished in %.0f min (%.1f MB/s), md5 %s", key, f.elapsed_s() / 60, f.rate_mb_s(),
                     "OK" if got == want else f"MISMATCH (got {got}, expected {want})")
            if want and got != want:
                return False
        return True

    def lane(self, keys: list[str], attempts: int = 3, wait_s: float = 300.0) -> None:
        """One stream thread: its tarballs one after the other. A network failure that outlasts the chunk retries
        restarts that tarball (finished speakers are skipped) after a pause, up to `attempts` times."""
        try:
            for key in keys:
                for attempt in range(1, attempts + 1):
                    if self.ctx.stop.is_set():
                        return
                    try:
                        if self.stream(key):
                            self.done_tarballs.add(key)
                        break
                    except (OSError, tarfile.TarError, EOFError) as e:  # network or truncated stream
                        if attempt == attempts:
                            raise
                        self.ctx.log.warning("%s: stream failed (%s: %s), attempt %d/%d - restarting it in %.0f s",
                                             key, type(e).__name__, e, attempt, attempts, wait_s)
                        self.ctx.stop.wait(wait_s)
        except (DiskFull, BudgetFull) as e:
            self.ctx.log.error("STOP - %s: %s", type(e).__name__, e)
            self.error = e
            self.ctx.stop.set()
        except BaseException as e:  # keep the other stream from running on alone without anyone noticing
            self.ctx.log.exception("stream failed: %s", e)
            self.error = e
            self.ctx.stop.set()


def plan_lanes(keys: list[str], streams: int) -> list[list[str]]:
    """Give each tarball (largest first) to the stream with the least work; each stream then runs small to big."""
    lanes: list[list[str]] = [[] for _ in range(max(1, min(streams, len(keys))))]
    for k in sorted(keys, key=lambda k: -SIZE_GB[k]):
        min(lanes, key=lambda lane: sum(SIZE_GB[x] for x in lane)).append(k)
    return [sorted(lane, key=lambda k: SIZE_GB[k]) for lane in lanes if lane]


def run(streams: int = 2, connections: int = 3, max_speakers: int | None = None,
        only: list[str] | None = None) -> None:
    log = get_logger("speech_librittsr")
    with SourceLock(SOURCE), SpeechManifests() as man:
        ctx = Ctx(SOURCE, log, man, Budget(SOURCE))
        meta = load_meta(log)
        r = Run(ctx, meta, max_speakers, connections)
        todo = [k for k in TARBALLS if k not in r.done_tarballs and (not only or k in only)]
        if not todo:
            log.info("all tarballs already done")
            return
        lanes = plan_lanes(todo, streams)
        log.info("start - %d speakers already done, budget used %.2f / %.2f GB, lanes %s, %d connections each",
                 len(r.done_speakers), ctx.budget.used / 1e9, ctx.budget.cap / 1e9, lanes, connections)
        threads = [threading.Thread(target=r.lane, args=(lane,), name=f"stream{i}", daemon=True)
                   for i, lane in enumerate(lanes)]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=5)
        except KeyboardInterrupt:
            log.warning("interrupted - stopping streams")
            ctx.stop.set()
            for t in threads:
                t.join(timeout=60)
            raise
        log.info("end - %d speakers finished this run, %.2f h kept, counts %s", r.speakers, r.kept_seconds / 3600,
                 dict(ctx.counts))
        if r.error is not None:
            raise r.error
