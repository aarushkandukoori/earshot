"""VoxPopuli en and GLOBE V2 (both CC0) - Hugging Face parquet shards, one at a time: download, convert, delete.

VoxPopuli en (facebook/voxpopuli) - 2 train shards of about 3.2 GB (16 kHz float WAV inside). Keep up to 60 s per
speaker, skip clips under 2 s, skip rows without a speaker id ('None', about a third of the rows) and skip speakers
that appear in an official VoxPopuli test split (en test or en_accented test). Measured 2026-11-07 - those test
speakers are also in the train shards (176 of the 364 en_accented speakers are in shards 0 + 15), so without this
filter the accent test set would contain voices the model trained on.

GLOBE V2 (MushanW/GLOBE_V2) - 8 spread-out train shards of about 450 MB (44.1 kHz FLAC inside). Keep every row whose
accent is not "United States English" (rows that list several accents count as non-US), cap US speakers at 30 s
(clips of at least 2 s), resample 44.1 to 16 kHz. GLOBE was denoised with Adobe Podcast Enhance, so every entry gets
extra["denoised_by_adobe_enhance"] = true and the mixer can down-weight GLOBE as a target. The GLOBE test split
shares no speakers with the train shards (checked), but we filter its speakers out anyway.

Resume - a finished shard is listed in data/speech/<source>/_state/done_shards.txt and its parquet deleted; a
half-converted shard keeps its parquet, and on the next run rows already in the manifest are skipped.
"""

from __future__ import annotations

import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # no progress bars in nohup logs
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")  # no hidden chunk cache outside data/

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from .common import get_logger, retry, split_for
from .speech_common import (RAW, Budget, Ctx, DoneSet, SourceLock, SpeechManifests, norm_accent, norm_gender,
                            reuse_existing, safe_name, seconds_per_speaker, source_entries, state_dir, store_clip,
                            wait_for_disk)

VOX_REPO = "facebook/voxpopuli"
VOX_SHARDS = ["en/train-00000-of-00030.parquet", "en/train-00015-of-00030.parquet"]
VOX_TEST_FILES = ["en/test-00000-of-00001.parquet", "en_accented/test-00000-of-00002.parquet",
                  "en_accented/test-00001-of-00002.parquet"]
VOX_SECONDS, VOX_MIN_S = 60.0, 2.0

GLOBE_REPO = "MushanW/GLOBE_V2"
# 8 spread-out shards. Rows are grouped by speaker and some shards are one or two voices for hours (shard 71 is a
# single England English speaker, 70 has one with 2.7 h), and "keep every non-US row" would then add hours of one
# voice instead of accent variety. So for each evenly spaced position (0, 23, 47, 70, 93, 116, 140, 163) we take the
# nearest shard with at least 100 speakers and no non-US speaker over 30 minutes (ties go to the lower index).
# Chosen from the shards' metadata columns on 2026-11-07: 0 -> 1 (shard 0 has a 42 min speaker), 47 -> 46 (16
# speakers), 70 -> 68 (69 to 72 fail). Expected about 1,200 speakers and 11 h.
GLOBE_SHARDS = [f"data/train-{i:05d}-of-00164.parquet" for i in (1, 23, 46, 68, 93, 116, 140, 163)]
GLOBE_TEST_FILES = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]
GLOBE_US_ACCENT = "United States English"
GLOBE_US_SECONDS, GLOBE_US_MIN_S = 30.0, 2.0


# ---------------------------------------------------------------------------------------------------------------
# selection rules (pure functions, tested in tests/test_speech_parquet.py)


def vox_decision(speaker_id, seconds: float, kept_s: float, test_speakers: set[str]) -> str:
    """'keep' or the reason a VoxPopuli row is skipped."""
    if speaker_id in (None, "", "None"):
        return "no_speaker_id"
    if speaker_id in test_speakers:
        return "official_test_speaker"
    if seconds < VOX_MIN_S:
        return "under_2s"
    if kept_s + seconds > VOX_SECONDS + 1e-6:
        return "speaker_full"
    return "keep"


def globe_is_us(accent) -> bool:
    return str(accent or "").strip() == GLOBE_US_ACCENT


def globe_decision(speaker_id, accent, seconds: float, us_kept_s: float, test_speakers: set[str]) -> str:
    """'keep' or the reason a GLOBE row is skipped. Non-US rows are always kept (unless the speaker is a test one)."""
    if speaker_id in (None, ""):
        return "no_speaker_id"
    if speaker_id in test_speakers:
        return "official_test_speaker"
    if not globe_is_us(accent):
        return "keep"
    if seconds < GLOBE_US_MIN_S:
        return "us_under_2s"
    if us_kept_s + seconds > GLOBE_US_SECONDS + 1e-6:
        return "us_speaker_full"
    return "keep"


def vox_session(audio_id: str) -> str:
    """'20180418-0900-PLENARY-3-en_20180418-08:50:36_17' -> '20180418-0900-PLENARY-3-en' (one plenary item)."""
    return audio_id.split("_", 1)[0]


def name_stem(path: str) -> str:
    """GLOBE 'audio.path' '000030.v2.vad.flac' -> '000030.v2.vad' (unique per speaker)."""
    base = path.rsplit("/", 1)[-1]
    for ext in (".flac", ".wav", ".mp3"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return safe_name(base)


# ---------------------------------------------------------------------------------------------------------------
# Hugging Face access


def official_test_speakers(repo: str, files: list[str], cache: Path, log) -> set[str]:
    """Speaker ids of a repo's test split, read from the speaker_id column only (a few range requests)."""
    if cache.exists():
        return set(json.loads(cache.read_text()))
    fs = HfFileSystem()
    spk: set[str] = set()
    for fn in files:
        def read(fn=fn):
            with fs.open(f"datasets/{repo}/{fn}", "rb", block_size=1 << 18) as f:
                return pq.ParquetFile(f).read(columns=["speaker_id"]).column("speaker_id").to_pylist()
        spk |= set(retry(read, log=log))
    spk -= {None, "", "None"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    tmp.write_text(json.dumps(sorted(spk)))
    os.replace(tmp, cache)
    log.info("%d official test speakers in %s %s", len(spk), repo, files)
    return spk


def download_shard(repo: str, filename: str, raw_dir: Path, log) -> Path:
    """hf_hub_download into data/raw/<source>/ (resumes a partial download). Waits for disk room first."""
    size = retry(lambda: HfApi().get_paths_info(repo, [filename], repo_type="dataset")[0].size, log=log)
    target = raw_dir / filename
    if not (target.exists() and target.stat().st_size == size):
        wait_for_disk(log, need_gb=size / 1e9)
    t0 = time.monotonic()
    log.info("downloading %s/%s (%.2f GB)", repo, filename, size / 1e9)
    path = Path(retry(lambda: hf_hub_download(repo, filename, repo_type="dataset", local_dir=raw_dir), log=log))
    dt = time.monotonic() - t0
    log.info("downloaded %s in %.0f s (%.1f MB/s)", filename, dt, size / 1e6 / max(dt, 1e-6))
    if path.stat().st_size != size:
        raise IOError(f"{filename} has {path.stat().st_size} bytes, expected {size}")
    return path


def delete_shard(path: Path, log) -> None:
    if path.exists():
        path.unlink()
        log.info("deleted %s", path)


# ---------------------------------------------------------------------------------------------------------------
# converters


def convert_vox_shard(ctx: Ctx, path: Path, shard: str, test_speakers: set[str], kept: dict) -> None:
    pf = pq.ParquetFile(path)
    cols = ["audio_id", "audio", "gender", "speaker_id", "accent", "normalized_text"]
    for i, batch in enumerate(pf.iter_batches(batch_size=100, columns=cols)):
        for row in batch.to_pylist():
            spk = row["speaker_id"]
            speaker = f"voxpopuli:{spk}"
            rel = f"speech/voxpopuli/{safe_name(spk)}/{safe_name(row['audio_id'])}.flac"
            if rel in ctx.man:
                continue  # done in an earlier run, already counted in `kept`
            data = row["audio"]["bytes"]
            sr = None
            secs = 0.0
            if spk not in (None, "", "None") and spk not in test_speakers:
                try:
                    info = sf.info(io.BytesIO(data))
                except Exception as e:
                    ctx.log.warning("%s: cannot read header (%s)", row["audio_id"], e)
                    ctx.count("decode_error")
                    continue
                sr, secs = info.samplerate, info.frames / info.samplerate
            why = vox_decision(spk, secs, kept[speaker], test_speakers)
            if why != "keep":
                ctx.count(why)
                continue
            meta = {"speaker": speaker, "split": split_for(speaker), "gender": norm_gender(row["gender"]),
                    "accent": norm_accent(row["accent"]), "group": vox_session(row["audio_id"]),
                    "extra": {"shard": shard, "orig_sr": sr, "audio_id": row["audio_id"],
                              "text": row["normalized_text"]}}
            d = reuse_existing(ctx, rel, meta)
            if d is None:
                try:
                    x, sr = sf.read(io.BytesIO(data), dtype="float32")
                except Exception as e:
                    ctx.log.warning("%s: cannot decode (%s)", row["audio_id"], e)
                    ctx.count("decode_error")
                    continue
                d = store_clip(ctx, rel, x, sr, meta)
            if d is not None:
                kept[speaker] += d
        if (i + 1) % 10 == 0:
            ctx.log.info("%s: %d rows read, budget %.2f GB, counts %s", shard, (i + 1) * 100, ctx.budget.used / 1e9,
                         dict(ctx.counts))


def convert_globe_shard(ctx: Ctx, path: Path, shard: str, test_speakers: set[str], us_kept: dict) -> None:
    pf = pq.ParquetFile(path)
    cols = ["audio", "speaker_id", "transcript", "accent", "age", "gender"]
    for i, batch in enumerate(pf.iter_batches(batch_size=100, columns=cols)):
        for row in batch.to_pylist():
            spk = row["speaker_id"]
            speaker = f"globe:{spk}"
            rel = f"speech/globe/{safe_name(spk)}/{name_stem(row['audio']['path'])}.flac"
            if rel in ctx.man:
                continue
            us = globe_is_us(row["accent"])
            try:  # the real length from the FLAC header; the "duration" column is up to 0.6 s short on some rows
                info = sf.info(io.BytesIO(row["audio"]["bytes"]))
            except Exception as e:
                ctx.log.warning("%s %s: cannot read header (%s)", spk, row["audio"]["path"], e)
                ctx.count("decode_error")
                continue
            why = globe_decision(spk, row["accent"], info.frames / info.samplerate, us_kept[speaker], test_speakers)
            if why != "keep":
                ctx.count(why)
                continue
            meta = {"speaker": speaker, "split": split_for(speaker), "gender": norm_gender(row["gender"]),
                    "accent": norm_accent(row["accent"]), "group": None,
                    "extra": {"denoised_by_adobe_enhance": True, "shard": shard, "orig_sr": 44100,
                              "age": row["age"] or None, "text": row["transcript"]}}
            d = reuse_existing(ctx, rel, meta)
            if d is None:
                try:
                    x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                except Exception as e:
                    ctx.log.warning("%s %s: cannot decode (%s)", spk, row["audio"]["path"], e)
                    ctx.count("decode_error")
                    continue
                meta["extra"]["orig_sr"] = sr
                d = store_clip(ctx, rel, x, sr, meta)
            if d is not None and us:
                us_kept[speaker] += d
        if (i + 1) % 10 == 0:
            ctx.log.info("%s: %d rows read, budget %.2f GB, counts %s", shard, (i + 1) * 100, ctx.budget.used / 1e9,
                         dict(ctx.counts))


# ---------------------------------------------------------------------------------------------------------------
# runners


def _run(source: str, repo: str, shards: list[str], test_files: list[str], convert, kept_filter=None,
         max_shards: int | None = None) -> None:
    log = get_logger(f"speech_{source}")
    raw_dir = RAW / source
    with SourceLock(source), SpeechManifests() as man:
        ctx = Ctx(source, log, man, Budget(source))
        done = DoneSet(state_dir(source) / "done_shards.txt")
        for fn in shards:  # a crash right after marking a shard done can leave its parquet behind
            if fn in done:
                delete_shard(raw_dir / fn, log)
        test_speakers = official_test_speakers(repo, test_files, state_dir(source) / "test_speakers.json", log)
        kept = seconds_per_speaker(source_entries(source), kept_filter)
        log.info("start - %d shards done, %d speakers already kept, budget used %.2f / %.2f GB", len(done),
                 len(kept), ctx.budget.used / 1e9, ctx.budget.cap / 1e9)
        todo = [fn for fn in shards if fn not in done][: max_shards or None]
        for fn in todo:
            path = download_shard(repo, fn, raw_dir, log)
            convert(ctx, path, fn, test_speakers, kept)
            done.add(fn)
            delete_shard(path, log)
            log.info("%s done - counts %s", fn, dict(ctx.counts))
        log.info("end - counts %s", dict(ctx.counts))


def run_voxpopuli(max_shards: int | None = None) -> None:
    _run("voxpopuli", VOX_REPO, VOX_SHARDS, VOX_TEST_FILES, convert_vox_shard, max_shards=max_shards)


def run_globe(max_shards: int | None = None) -> None:
    _run("globe", GLOBE_REPO, GLOBE_SHARDS, GLOBE_TEST_FILES, convert_globe_shard,
         kept_filter=lambda e: globe_is_us(e.get("accent")), max_shards=max_shards)
