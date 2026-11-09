"""Re-run of the parallel random-crop benchmark with a start barrier so all workers overlap.
Usage: python -I bench_parallel.py <work_dir> <librispeech_dir>"""
import glob, json, random, sys, time
from multiprocessing import get_context
import numpy as np
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from bench_decode import crop_reader  # noqa: E402

def worker(args):
    kind, paths, work, seconds, seed, barrier = args
    read = crop_reader(kind, paths, work)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        read(rng)  # warm up (page faults, file handles)
    barrier.wait()
    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < seconds:
        read(rng); n += 1
    return n / (time.perf_counter() - t0)

def run(kind, paths, work, procs, seconds=8.0):
    ctx = get_context("spawn")
    mgr = ctx.Manager()
    barrier = mgr.Barrier(procs)
    with ctx.Pool(procs) as pool:
        rates = pool.map(worker, [(kind, paths, work, seconds, s, barrier) for s in range(procs)])
    mgr.shutdown()
    return round(sum(rates)), [round(r) for r in rates]

if __name__ == "__main__":
    work, src = sys.argv[1], sys.argv[2]
    flac = sorted(glob.glob(f"{src}/**/*.flac", recursive=True)); random.Random(0).shuffle(flac); flac = flac[:400]
    paths = {"flac": flac, "opus24": [f"{work}/opus24/{i:05d}.opus" for i in range(400)], "memmap": None}
    out = {}
    for procs in (1, 4, 6, 8, 10):
        out[procs] = {k: run(k, v, work, procs) for k, v in paths.items()}
        print(procs, out[procs], flush=True)
    json.dump(out, open(f"{work}/parallel_results.json", "w"), indent=1)
