"""Streaming inference cost per chunk on CPU with ONE thread (proxy for one iPhone P-core).

For every candidate and chunk size (64 samples = 4 ms, 128 samples = 8 ms at 16 kHz) it times
    eager  - plain PyTorch, torch.inference_mode()
    jit    - torch.jit.trace + torch.jit.freeze of the same step (less Python overhead)
and prints one JSON line per (model, chunk, mode).

    python bench_stream.py                 # all models
    python bench_stream.py --models tasnet64 stftlstm --modes eager
"""
import argparse
import json
import os
import platform
import statistics
import time

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import models as M  # noqa: E402
import streaming as S  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def op_overhead_us(n=20000):
    """Cost of one tiny eager op (add on a 1x256x4 tensor) = framework dispatch floor."""
    x = torch.randn(1, 256, 4)
    with torch.inference_mode():
        for _ in range(1000):
            x = x + 1.0
        t0 = time.perf_counter_ns()
        for _ in range(n):
            x = x + 1.0
        t1 = time.perf_counter_ns()
    return (t1 - t0) / n / 1e3


def count_ops(traced):
    nodes = [n.kind() for n in traced.graph.nodes()]
    return sum(1 for k in nodes if k.startswith("aten::"))


def run(name, chunk, mode, seconds, impl):
    torch.manual_seed(0)
    model = M.build(name).eval()
    emb = torch.randn(1, M.EMB_DIM)
    stream = S.make_stream(model, emb, chunk, tasnet_impl=impl).eval()
    st = stream.init_state()
    n = int(seconds * M.SR / chunk)
    warm = max(100, n // 10)
    xs = torch.randn(n + warm, chunk) * 0.1
    fn = stream
    n_ops = None
    with torch.no_grad():
        if mode == "jit":
            traced = torch.jit.trace(stream, (xs[0], *st), check_trace=False)
            fn = torch.jit.freeze(traced)
            n_ops = count_ops(fn)
    times = []
    with torch.inference_mode() if mode == "eager" else torch.no_grad():
        for i in range(n + warm):
            t0 = time.perf_counter_ns()
            y, *st = fn(xs[i], *st)
            t1 = time.perf_counter_ns()
            if i >= warm:
                times.append((t1 - t0) / 1e3)
    budget_us = chunk / M.SR * 1e6
    med = statistics.median(times)
    return {
        "model": name,
        "impl": impl if name.startswith("tasnet") else "-",
        "chunk_samples": chunk,
        "chunk_ms": chunk / M.SR * 1e3,
        "mode": mode,
        "chunks_timed": len(times),
        "median_us": round(med, 1),
        "mean_us": round(statistics.mean(times), 1),
        "p90_us": round(pct(times, 90), 1),
        "p99_us": round(pct(times, 99), 1),
        "p999_us": round(pct(times, 99.9), 1),
        "max_us": round(max(times), 1),
        "rtf_median": round(med / budget_us, 3),
        "rtf_p99": round(pct(times, 99) / budget_us, 3),
        "macs_per_chunk": model.macs_per_frame() * model.frames_per_second() * chunk / M.SR,
        "gflops_effective": round(2 * model.macs_per_frame() * model.frames_per_second() * chunk / M.SR / (med * 1e3), 2),
        "aten_ops_per_chunk": n_ops,
        "algorithmic_latency_ms": model.algorithmic_latency_samples(chunk) / M.SR * 1e3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=M.ALL)
    ap.add_argument("--chunks", nargs="*", type=int, default=[64, 128])
    ap.add_argument("--modes", nargs="*", default=["eager", "jit"])
    ap.add_argument("--seconds", type=float, default=10.0, help="audio seconds timed per config")
    ap.add_argument("--tag", default="", help="free text stored with results (e.g. p-core / e-core)")
    ap.add_argument("--tasnet-impls", nargs="*", default=["tm", "conv"],
                    help="tm = time-major F.linear + 3-tap depthwise; conv = nn.Conv1d calls")
    args = ap.parse_args()
    meta = {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "machine": platform.machine(),
        "loadavg": os.getloadavg(),
        "op_overhead_us": round(op_overhead_us(), 2),
        "tag": args.tag,
    }
    print(json.dumps({"meta": meta}), flush=True)
    for name in args.models:
        impls = args.tasnet_impls if name.startswith("tasnet") else ["tm"]
        for impl in impls:
            for chunk in args.chunks:
                for mode in args.modes:
                    try:
                        r = run(name, chunk, mode, args.seconds, impl)
                    except Exception as e:  # noqa: BLE001
                        r = {"model": name, "impl": impl, "chunk_samples": chunk, "mode": mode,
                             "error": f"{type(e).__name__}: {e}"[:300]}
                    r["tag"] = args.tag
                    r["loadavg_1m"] = round(os.getloadavg()[0], 2)
                    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
