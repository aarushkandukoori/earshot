"""Decode-speed benchmark for on-the-fly mixing: FLAC vs Ogg-Opus vs MP3 vs int16 memmap.

Usage: python -I bench_decode.py <librispeech_dir> <work_dir>
Measures (a) single-core full-file decode speed, (b) random 4 s crop latency,
(c) aggregate random-crop throughput with N worker processes (spawn, like macOS DataLoader).
"""
import glob
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np
import soundfile as sf

SR = 16000
CROP = 4 * SR
FFMPEG = "/opt/homebrew/bin/ffmpeg"


def encode(args):
    src, dst, codec_args = args
    if not os.path.exists(dst):
        subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", src, *codec_args, dst], check=True)


def prep(src_dir, work, n_files):
    files = sorted(glob.glob(f"{src_dir}/**/*.flac", recursive=True))
    random.Random(0).shuffle(files)
    files = files[:n_files]
    formats = {
        "flac": None,
        "opus24": ["-c:a", "libopus", "-b:a", "24k", "-application", "audio"],
        "opus32": ["-c:a", "libopus", "-b:a", "32k", "-application", "audio"],
        "mp3_48k": ["-c:a", "libmp3lame", "-b:a", "48k"],
    }
    out = {"flac": files}
    jobs = []
    for name, codec in formats.items():
        if codec is None:
            continue
        os.makedirs(f"{work}/{name}", exist_ok=True)
        ext = "opus" if name.startswith("opus") else "mp3"
        dsts = [f"{work}/{name}/{i:05d}.{ext}" for i in range(len(files))]
        out[name] = dsts
        jobs += [(s, d, codec) for s, d in zip(files, dsts)]
    with ThreadPoolExecutor(10) as ex:
        list(ex.map(encode, jobs))
    # int16 memmap: one flat file + offsets
    lens = []
    with open(f"{work}/all.int16", "wb") as fh:
        for f in files:
            a, sr = sf.read(f, dtype="int16")
            assert sr == SR
            fh.write(a.tobytes())
            lens.append(len(a))
    offsets = np.concatenate([[0], np.cumsum(lens)])
    np.save(f"{work}/offsets.npy", offsets)
    return out, offsets


def du(paths):
    return sum(os.path.getsize(p) for p in paths)


def full_decode(paths):
    t = time.perf_counter()
    total = 0
    for p in paths:
        a, sr = sf.read(p, dtype="float32")
        total += len(a) / sr
    return total / (time.perf_counter() - t)  # x realtime


def crop_reader(kind, paths, work):
    if kind == "memmap":
        mm = np.memmap(f"{work}/all.int16", dtype=np.int16, mode="r")
        offsets = np.load(f"{work}/offsets.npy")
        lens = np.diff(offsets)
        ok = np.where(lens >= CROP)[0]

        def read(rng):
            i = ok[rng.integers(len(ok))]
            s = offsets[i] + rng.integers(lens[i] - CROP + 1)
            return np.asarray(mm[s:s + CROP], dtype=np.float32) / 32768.0
        return read
    infos = [sf.info(p) for p in paths]
    ok = [i for i, inf in enumerate(infos) if inf.frames >= CROP + SR]
    srs = [infos[i].samplerate for i in range(len(infos))]

    def read(rng):
        i = ok[rng.integers(len(ok))]
        sr = srs[i]
        n = int(CROP * sr / SR)
        start = int(rng.integers(infos[i].frames - n))
        a, _ = sf.read(paths[i], start=start, frames=n, dtype="float32")
        return a
    return read


def crop_bench(kind, paths, work, n=400, seed=0):
    read = crop_reader(kind, paths, work)
    rng = np.random.default_rng(seed)
    read(rng)
    t = time.perf_counter()
    for _ in range(n):
        read(rng)
    dt = time.perf_counter() - t
    return n / dt  # crops per second, one core


def _worker(args):
    kind, paths, work, seconds, seed = args
    read = crop_reader(kind, paths, work)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    count = 0
    while time.perf_counter() - t0 < seconds:
        read(rng)
        count += 1
    return count / (time.perf_counter() - t0)


def parallel_bench(kind, paths, work, procs, seconds=6.0):
    ctx = get_context("spawn")
    with ctx.Pool(procs) as pool:
        rates = pool.map(_worker, [(kind, paths, work, seconds, s) for s in range(procs)])
    return sum(rates)


def main():
    src_dir, work = sys.argv[1], sys.argv[2]
    n_files = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    t = time.perf_counter()
    out, offsets = prep(src_dir, work, n_files)
    audio_s = offsets[-1] / SR
    res = {"n_files": n_files, "audio_hours": audio_s / 3600, "prep_s": time.perf_counter() - t}
    sizes = {k: du(v) for k, v in out.items()}
    sizes["memmap"] = os.path.getsize(f"{work}/all.int16")
    res["MB_per_hour"] = {k: v / 1e6 / (audio_s / 3600) for k, v in sizes.items()}
    res["full_decode_x_realtime_1core"] = {k: full_decode(v) for k, v in out.items()}
    kinds = list(out.keys()) + ["memmap"]
    res["crop4s_per_s_1core"] = {k: crop_bench(k, out.get(k), work) for k in kinds}
    res["crop4s_per_s_parallel"] = {}
    for procs in (4, 8, 10):
        res["crop4s_per_s_parallel"][procs] = {k: parallel_bench(k, out.get(k), work, procs) for k in kinds}
    print(json.dumps(res, indent=1))
    with open(f"{work}/results.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
