"""Time one training step (forward + SI-SDR loss + backward + grad clip + AdamW) for a candidate.

Run one config per process so memory numbers are clean, for example
    python bench_train.py --model tasnet32 --device mps --precision fp16
Prints one JSON line.
"""
import argparse
import json
import resource
import statistics
import time

import torch

import models as M


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=M.ALL)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--ckpt", action="store_true", help="activation checkpointing per TCN block / RNN layer")
    ap.add_argument("--no-fused", action="store_true", help="do not try fused AdamW")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--grad-scaler", action="store_true", help="use torch.amp.GradScaler for fp16 on MPS")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    torch.manual_seed(0)
    model = M.build(args.model).to(dev)
    n_params = M.n_params(model)

    if args.ckpt:
        from torch.utils.checkpoint import checkpoint

        if isinstance(model, M.ConvTasNetTSE):
            for blocks in model.repeats:
                for blk in blocks:
                    f = blk.forward
                    blk.forward = lambda x, f=f: checkpoint(f, x, use_reentrant=False)
        else:
            raise SystemExit("--ckpt only implemented for tasnet")

    fwd = torch.compile(model) if args.compile else model

    n = int(M.SR * args.seconds)
    mix = (torch.randn(args.batch, n) * 0.1).to(dev)
    tgt = (torch.randn(args.batch, n) * 0.1).to(dev)
    emb = torch.randn(args.batch, M.EMB_DIM).to(dev)

    fused = False
    opt = None
    if not args.no_fused and args.device == "mps":
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
            fused = True
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"note": f"fused AdamW unavailable: {e}"[:200]}))
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    scaler = None
    static_scale = None
    if amp_dtype is torch.float16:
        if args.device == "mps" and not args.grad_scaler:
            # torch 2.7.1: GradScaler.unscale_ casts the scale to float64, which MPS does not support
            # ("Cannot convert a MPS Tensor to float64"). Use a static loss scale with the same cost.
            static_scale = 1024.0
        else:
            scaler = torch.amp.GradScaler(args.device)

    peak_alloc = 0
    peak_driver = 0

    def mem_sample():
        nonlocal peak_alloc, peak_driver
        if args.device == "mps":
            peak_alloc = max(peak_alloc, torch.mps.current_allocated_memory())
            peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())

    def step():
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.autocast(device_type=args.device, dtype=amp_dtype):
                est = fwd(mix, emb)
        else:
            est = fwd(mix, emb)
        loss = M.neg_si_sdr(est, tgt)
        sync(args.device)
        mem_sample()
        t1 = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        elif static_scale is not None:
            (loss * static_scale).backward()
            torch._foreach_mul_([p.grad for p in model.parameters() if p.grad is not None], 1.0 / static_scale)
        else:
            loss.backward()
        sync(args.device)
        mem_sample()
        t2 = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        elif static_scale is None or torch.isfinite(norm).item():  # skip the step on fp16 overflow
            opt.step()
        sync(args.device)
        mem_sample()
        t3 = time.perf_counter()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        return t3 - t0, t1 - t0, t2 - t1, t3 - t2

    out = {
        "model": args.model,
        "device": args.device,
        "precision": args.precision,
        "batch": args.batch,
        "seconds": args.seconds,
        "params": n_params,
        "macs_per_audio_second": model.macs_per_frame() * model.frames_per_second(),
        "fused_adamw": fused,
        "grad_scaler": scaler is not None,
        "static_loss_scale": static_scale,
        "compile": args.compile,
        "ckpt": args.ckpt,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    try:
        t_start = time.perf_counter()
        first = step()  # includes kernel compilation / graph caching
        out["first_step_s"] = round(first[0], 4)
        for _ in range(args.warmup - 1):
            step()
        t_warm = time.perf_counter() - t_start
        times = [step() for _ in range(args.steps)]
        tot = [t[0] for t in times]
        out.update(
            {
                "ok": True,
                "warmup_total_s": round(t_warm, 2),
                "step_median_s": round(statistics.median(tot), 4),
                "step_mean_s": round(statistics.mean(tot), 4),
                "step_min_s": round(min(tot), 4),
                "step_max_s": round(max(tot), 4),
                "fwd_median_s": round(statistics.median(t[1] for t in times), 4),
                "bwd_median_s": round(statistics.median(t[2] for t in times), 4),
                "opt_median_s": round(statistics.median(t[3] for t in times), 4),
                "audio_seconds_per_s": round(args.batch * args.seconds / statistics.median(tot), 1),
            }
        )
        if len(tot) >= 50:  # sustained runs: median step time of each tenth of the run, in order
            k = len(tot) // 10
            out["step_median_by_tenth_s"] = [round(statistics.median(tot[i * k : (i + 1) * k]), 4) for i in range(10)]
        # achieved model FLOP rate: training ~ 3x forward MACs, 2 FLOPs per MAC
        flops = 6 * out["macs_per_audio_second"] * args.batch * args.seconds
        out["train_tflops_effective"] = round(flops / statistics.median(tot) / 1e12, 3)
    except Exception as e:  # noqa: BLE001
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:400]})
    if args.device == "mps":
        out["peak_allocated_GB"] = round(peak_alloc / 1e9, 3)
        out["peak_driver_GB"] = round(peak_driver / 1e9, 3)
        out["recommended_max_GB"] = round(torch.mps.recommended_max_memory() / 1e9, 2)
    out["max_rss_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 3)  # bytes on macOS
    print(json.dumps(out))


if __name__ == "__main__":
    main()
