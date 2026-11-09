"""Turn bench/results/*.jsonl into the markdown tables used in RESULTS.md.

    .venv/bin/python -I bench/summarize.py > /tmp/tables.md
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def load(name):
    p = RES / name
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def train_tables():
    rows = load("train.jsonl") + load("train_extra.jsonl")
    body = []
    for r in rows:
        prec = r["precision"] + (" +ckpt" if r.get("ckpt") else "") + (" (static scale)" if r.get("static_loss_scale") else "")
        if not r.get("ok"):
            body.append([r["torch"], r["model"], r["device"], prec, "FAILED: " + r.get("error", "")[:80]] + [""] * 9)
            continue
        s = r["step_median_s"]
        body.append([
            r["torch"], r["model"], r["device"], prec, fmt(s, 3), fmt(r["fwd_median_s"], 3),
            fmt(r["bwd_median_s"] + r["opt_median_s"], 3), fmt(r["audio_seconds_per_s"], 0),
            fmt(r.get("peak_allocated_GB")), fmt(r.get("peak_driver_GB")),
            f"{86400 / s / 1000:.0f}k", f"{129600 / s / 1000:.0f}k", f"{100000 * s / 3600:.1f}",
            fmt(r.get("loadavg_1m_start"), 1),
        ])
    return table(["torch", "model", "device", "precision", "s/step", "fwd s", "bwd+opt s", "audio s per s",
                  "peak alloc GB", "MPS pool GB", "steps in 24 h", "steps in 36 h", "h per 100k", "load"], body)


def stream_tables(name):
    rows = [r for r in load(name) if "model" in r]
    body = []
    for r in rows:
        if "error" in r:
            body.append([r["model"], r.get("impl", "-"), r["chunk_samples"], r["mode"], "ERROR " + r["error"][:60]] + [""] * 6)
            continue
        body.append([r["model"], r.get("impl", "-"), f"{r['chunk_ms']:.0f} ms", r["mode"], fmt(r["median_us"], 0),
                     fmt(r["p99_us"], 0), fmt(r["max_us"], 0), fmt(r["rtf_median"], 3), fmt(r["rtf_p99"], 3),
                     fmt(r.get("aten_ops_per_chunk")), fmt(r.get("loadavg_1m"), 1)])
    return table(["model", "impl", "chunk", "mode", "median us", "p99 us", "max us", "RTF median", "RTF p99",
                  "aten ops", "load"], body)


def sgemm_table(name):
    rows = [r for r in load(name) if "shape" in r]
    body = []
    for r in rows:
        body.append([r["shape"], f"{r['M']}x{r['K']} * {r['K']}x{r['N']}", fmt(r["sgemm_xW_us"], 2),
                     fmt(r["sgemm_xW_gflops"], 0), fmt(r["sgemm_xWT_us"], 2), fmt(r["sgemm_xWT_gflops"], 0),
                     fmt(r.get("sgemv_us"), 2), fmt(r.get("sgemv_gflops"), 0)])
    return table(["what", "shape (M x K) * (K x N)", "sgemm W[K,N] us", "GFLOPS", "sgemm W^T us", "GFLOPS",
                  "sgemv us", "GFLOPS"], body)


def skeleton_table(p_name, e_name):
    p = {(r["model"], r["chunk_samples"]): r for r in load(p_name) if "model" in r}
    e = {(r["model"], r["chunk_samples"]): r for r in load(e_name) if "model" in r}
    body = []
    for k, r in p.items():
        re_ = e.get(k, {})
        body.append([r["model"], f"{r['chunk_ms']:.0f} ms", fmt(r["weights_MB"]), f"{r['macs_per_chunk'] / 1e6:.2f}",
                     fmt(r["median_us"], 0), fmt(r["p99_us"], 0), fmt(r["max_us"], 0), fmt(r["rtf_median"], 3),
                     fmt(r["gflops"], 0), fmt(re_.get("median_us"), 0), fmt(re_.get("p99_us"), 0),
                     fmt(re_.get("rtf_median"), 3)])
    return table(["model", "chunk", "weights MB", "MMAC per chunk", "P median us", "P p99 us", "P max us",
                  "P RTF", "P GFLOPS", "E median us", "E p99 us", "E RTF"], body)


def decode_table():
    rows = [r for r in load("decode.jsonl") if "format" in r]
    body = []
    for r in rows:
        body.append([r["format"], fmt(r["kbit_per_s"], 0), fmt(r["MB_per_audio_hour"], 0), r["decoded_sr"],
                     fmt(r["full_decode_xrealtime_1core"], 0), fmt(r["crop4s_ms_each"], 2),
                     fmt(r["crop_xrealtime_1core"], 0), fmt(r.get("crop_xrealtime_4proc"), 0),
                     fmt(r.get("crop_xrealtime_8proc"), 0), fmt(r.get("loadavg_1m"), 1)])
    return table(["format", "kbit/s", "MB per audio hour", "decoded Hz", "full decode x realtime (1 core)",
                  "4 s crop ms", "crop x realtime (1 core)", "crop x realtime (4 proc)", "crop x realtime (8 proc)",
                  "load"], body)


def main():
    which = sys.argv[1:] or ["train", "stream", "sgemm", "skeleton", "decode"]
    if "train" in which:
        print("### Training step\n\n" + train_tables() + "\n")
    if "stream" in which:
        print("### Streaming PyTorch, 1 thread\n\n" + stream_tables("stream_cpu_1thread.jsonl") + "\n")
        if (RES / "stream_cpu_1thread_ecore.jsonl").exists():
            print("### Streaming PyTorch, 1 thread, E-core\n\n" + stream_tables("stream_cpu_1thread_ecore.jsonl") + "\n")
    if "sgemm" in which:
        print("### sgemm P-core\n\n" + sgemm_table("sgemm_1thread.jsonl") + "\n")
        print("### sgemm E-core\n\n" + sgemm_table("sgemm_1thread_ecore.jsonl") + "\n")
    if "skeleton" in which:
        print("### Accelerate skeleton\n\n" + skeleton_table("skeleton_1thread.jsonl", "skeleton_1thread_ecore.jsonl") + "\n")
    if "decode" in which:
        print("### Decode\n\n" + decode_table() + "\n")


if __name__ == "__main__":
    main()
