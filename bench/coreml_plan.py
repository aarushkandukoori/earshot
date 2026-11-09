"""Where would Core ML run each streaming step? Uses MLComputePlan (macOS 14.4+) to list the preferred
compute device of every op in the fp16 ML Program, and times predict() of a trivial model to measure
the Python + coremltools per-call overhead that is included in check_coreml.py timings.

    .venv/bin/python bench/coreml_plan.py --models tasnet64 tasnet32 stftlstm stftgru
"""
import argparse
import collections
import json
import statistics
import time
import warnings

import numpy as np
import torch

import models as M
import streaming as S

warnings.filterwarnings("ignore")


def device_name(usage):
    if usage is None:
        return "none"
    return type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["tasnet64", "stftlstm"])
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    # per-call overhead of predict() from Python on a trivial model (one add)
    class Tiny(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    x0 = torch.zeros(64)
    tiny = ct.convert(torch.jit.trace(Tiny(), x0), inputs=[ct.TensorType(name="x", shape=(64,), dtype=np.float32)],
                      convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_ONLY)
    feed = {"x": np.zeros(64, np.float32)}
    for _ in range(200):
        tiny.predict(feed)
    ts = []
    for _ in range(2000):
        t0 = time.perf_counter()
        tiny.predict(feed)
        ts.append((time.perf_counter() - t0) * 1e6)
    print(json.dumps({"predict_overhead_us_trivial_model_cpu": round(statistics.median(ts), 1)}), flush=True)

    for name in args.models:
        r = {"model": name}
        try:
            torch.manual_seed(0)
            model = M.build(name).eval()
            stream = S.make_stream(model, torch.randn(1, M.EMB_DIM), args.chunk).eval()
            st0 = stream.init_state()
            with torch.no_grad():
                traced = torch.jit.trace(stream, (x0[: args.chunk], *st0), check_trace=False)
            names = ["x"] + [f"s{i}" for i in range(len(st0))]
            inputs = [ct.TensorType(name=n, shape=t.shape, dtype=np.float32) for n, t in zip(names, (x0, *st0))]
            ml = ct.convert(traced, inputs=inputs, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                            compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.ALL)
            path = ml.get_compiled_model_path()
            for label, units in [("all", ct.ComputeUnit.ALL), ("cpu_and_ne", ct.ComputeUnit.CPU_AND_NE)]:
                plan = MLComputePlan.load_from_path(path=path, compute_units=units)
                ops = plan.model_structure.program.functions["main"].block.operations
                counts = collections.Counter()
                ne_types = collections.Counter()
                for op in ops:
                    dev = device_name(plan.get_compute_device_usage_for_mlprogram_operation(op))
                    if op.operator_name in ("const",):
                        continue
                    counts[dev] += 1
                    if dev == "NeuralEngine":
                        ne_types[op.operator_name] += 1
                r[f"ops_by_device_{label}"] = dict(counts)
                if label == "all":
                    r["ops_on_neural_engine_by_type"] = dict(ne_types.most_common(8))
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {e}"[:400]
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
