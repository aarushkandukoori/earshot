"""Audio decode throughput per CPU core: FLAC vs WAV vs Ogg Opus, read with soundfile (libsndfile).

1. Picks a fixed subset of LibriSpeech dev-clean (FLAC, 16 kHz).
2. Makes WAV (PCM 16-bit, soundfile) and Ogg Opus (32 and 64 kbit/s, /opt/homebrew/bin/ffmpeg) copies.
3. Times, in one process (= one core):
     full   - decode every file completely (best of 3 passes, files in page cache)
     crop   - open + seek to a random offset + read 4 s, once per file (what a training loader does)
   and the same with N worker processes to see how it scales across cores.

Run with python -I (the audio files are downloaded data):
    .venv/bin/python -I bench/bench_decode.py --root data/LibriSpeech/dev-clean --out data/bench_formats
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

FFMPEG = "/opt/homebrew/bin/ffmpeg"
SR = 16000
CROP = 4 * SR


def make_copies(files, root, out):
    """Create WAV and Opus versions; returns {format: [paths]} in the same order as files."""
    sets = {"flac": files, "wav": [], "opus32": [], "opus64": []}
    for f in files:
        rel = f.relative_to(root)
        wav = out / "wav" / rel.with_suffix(".wav")
        if not wav.exists():
            wav.parent.mkdir(parents=True, exist_ok=True)
            x, sr = sf.read(f, dtype="int16")
            sf.write(wav, x, sr, subtype="PCM_16")
        sets["wav"].append(wav)
        for kbps in (32, 64):
            op = out / f"opus{kbps}" / rel.with_suffix(".opus")
            if not op.exists():
                op.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", str(f), "-c:a", "libopus",
                     "-b:a", f"{kbps}k", "-vbr", "on", "-application", "audio", str(op)],
                    check=True,
                )
            sets[f"opus{kbps}"].append(op)
    return sets


def decode_full(paths):
    n = 0
    t0 = time.perf_counter()
    for p in paths:
        x, sr = sf.read(p, dtype="float32")
        n += x.shape[0]
    return n, time.perf_counter() - t0


def decode_crops(args):
    paths, seed = args
    rng = random.Random(seed)
    n = 0
    t0 = time.perf_counter()
    for p in paths:
        with sf.SoundFile(p) as f:
            total = f.frames
            start = rng.randrange(0, max(1, total - CROP))
            f.seek(start)
            x = f.read(CROP, dtype="float32")
        n += x.shape[0]
    return n, time.perf_counter() - t0


def worker_full(paths):
    return decode_full(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-files", type=int, default=200)
    ap.add_argument("--workers", nargs="*", type=int, default=[4, 8])
    ap.add_argument("--prepare-only", action="store_true", help="only create the WAV / Opus copies")
    args = ap.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    allf = sorted(root.rglob("*.flac"))
    random.Random(0).shuffle(allf)
    files = sorted(allf[: args.n_files])
    sets = make_copies(files, root, out)
    if args.prepare_only:
        return

    info = sf.info(sets["opus32"][0])
    print(json.dumps({"meta": {"libsndfile": sf.__libsndfile_version__, "soundfile": sf.__version__,
                               "opus_decoded_samplerate": info.samplerate, "opus_subtype": info.subtype,
                               "files": len(files), "loadavg": os.getloadavg()}}), flush=True)
    # durations from the FLAC originals
    secs = sum(sf.info(p).frames for p in files) / SR
    for fmt, paths in sets.items():
        size = sum(os.path.getsize(p) for p in paths)
        # sanity: decoded length equals the original within the codec padding
        x0, sr0 = sf.read(paths[0], dtype="float32")
        ref0 = sf.info(files[0]).frames
        decode_full(paths)  # warm the page cache
        full = [decode_full(paths) for _ in range(3)]
        n, t = min(full, key=lambda r: r[1])
        crops = [decode_crops((paths, s)) for s in range(3)]
        nc, tc = min(crops, key=lambda r: r[1])
        r = {
            "format": fmt,
            "files": len(paths),
            "audio_hours": round(secs / 3600, 3),
            "bytes": size,
            "MB_per_audio_hour": round(size / 1e6 / (secs / 3600), 1),
            "kbit_per_s": round(size * 8 / secs / 1e3, 1),
            "decoded_sr": sr0,
            "first_file_len_ratio": round(x0.shape[0] / (ref0 * sr0 / SR), 4),
            "full_decode_xrealtime_1core": round(n / sr0 / t, 1),
            "full_decode_s_per_audio_hour": round(t / (n / sr0) * 3600, 2),
            "crop4s_per_s_1core": round(len(paths) / tc, 1),
            "crop4s_ms_each": round(tc / len(paths) * 1e3, 3),
            "crop_xrealtime_1core": round(nc / sr0 / tc, 1),
        }
        for w in args.workers:  # multi-process scaling of the crop workload (each worker has all files)
            with mp.get_context("spawn").Pool(w) as pool:
                pool.map(decode_crops, [(paths, 100 + i) for i in range(w)])  # warm up workers
                t0 = time.perf_counter()
                res = pool.map(decode_crops, [(paths, 200 + i) for i in range(w)])
                wall = time.perf_counter() - t0
            r[f"crop_xrealtime_{w}proc"] = round(sum(a for a, _ in res) / sr0 / wall, 1)
        r["loadavg_1m"] = round(os.getloadavg()[0], 2)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
