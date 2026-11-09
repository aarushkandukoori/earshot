"""Export check: convert a streaming step (chunk in, explicit state in/out) to Core ML with coremltools,
compare it with PyTorch over many chunks, and time MLModel.predict on the Mac's CPU / Neural Engine.

Timings include Python + coremltools predict() overhead, so they are an upper bound for Swift.

    .venv/bin/python bench/check_coreml.py --models tasnet64 stftlstm
"""
import argparse
import json
import statistics
import time
import warnings

import numpy as np
import torch

import models as M
import streaming as S

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["tasnet64", "stftlstm"])
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--chunks", type=int, default=300)
    args = ap.parse_args()
    import coremltools as ct

    for name in args.models:
        r = {"model": name, "chunk_samples": args.chunk, "coremltools": ct.__version__, "torch": torch.__version__}
        try:
            torch.manual_seed(0)
            model = M.build(name).eval()
            emb = torch.randn(1, M.EMB_DIM)
            stream = S.make_stream(model, emb, args.chunk).eval()
            st0 = stream.init_state()
            x0 = torch.zeros(args.chunk)
            with torch.no_grad():
                traced = torch.jit.trace(stream, (x0, *st0), check_trace=False)
            in_names = ["x"] + [f"s{i}" for i in range(len(st0))]
            inputs = [ct.TensorType(name=n, shape=t.shape, dtype=np.float32) for n, t in zip(in_names, (x0, *st0))]
            t0 = time.time()
            ml = ct.convert(traced, inputs=inputs, convert_to="mlprogram",
                            minimum_deployment_target=ct.target.iOS18,
                            compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY)
            r["convert_s"] = round(time.time() - t0, 1)
            out_names = [o.name for o in ml.get_spec().description.output]
            # run both implementations over the same chunks, feeding each its own state back
            xs = torch.randn(args.chunks, args.chunk) * 0.1
            st_t = [t.clone() for t in st0]
            st_c = {n: t.numpy() for n, t in zip(in_names[1:], st0)}
            max_err, max_ref = 0.0, 0.0
            times = []
            for i in range(args.chunks):
                with torch.no_grad():
                    y_t, *st_t = stream(xs[i], *st_t)
                feed = {"x": xs[i].numpy(), **st_c}
                t1 = time.perf_counter()
                out = ml.predict(feed)
                times.append((time.perf_counter() - t1) * 1e6)
                y_c = out[out_names[0]]
                st_c = {n: out[o] for n, o in zip(in_names[1:], out_names[1:])}
                max_err = max(max_err, float(np.abs(y_c.reshape(-1) - y_t.numpy()).max()))
                max_ref = max(max_ref, float(y_t.abs().max()))
            r.update({"ok": True, "max_abs_err_vs_torch": max_err, "max_abs_out": max_ref,
                      "predict_us_median_cpu_fp32": round(statistics.median(times[20:]), 1)})
            # timing on other compute units (fp16 model for the Neural Engine)
            ml16 = ct.convert(traced, inputs=inputs, convert_to="mlprogram",
                              minimum_deployment_target=ct.target.iOS18,
                              compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.ALL)
            for label, mdl_units in [("cpu_fp16", ct.ComputeUnit.CPU_ONLY), ("cpu_and_ne_fp16", ct.ComputeUnit.CPU_AND_NE),
                                     ("all_fp16", ct.ComputeUnit.ALL)]:
                mdl = ct.models.MLModel(ml16.get_spec(), weights_dir=ml16.weights_dir, compute_units=mdl_units)
                feed = {"x": xs[0].numpy(), **{n: t.numpy() for n, t in zip(in_names[1:], st0)}}
                for _ in range(30):
                    mdl.predict(feed)
                ts = []
                for i in range(200):
                    t1 = time.perf_counter()
                    mdl.predict(feed)
                    ts.append((time.perf_counter() - t1) * 1e6)
                r[f"predict_us_median_{label}"] = round(statistics.median(ts), 1)
        except Exception as e:  # noqa: BLE001
            r.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:400]})
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
