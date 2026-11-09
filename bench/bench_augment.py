"""Data-pipeline cost per CPU core: room impulse response (RIR) simulation with pyroomacoustics
and convolving 4 s of speech with it (scipy). Tells whether RIRs can be simulated on the fly in the
training data loader or must be precomputed into a bank.

    .venv/bin/python -I bench/bench_augment.py
"""
import json
import os
import random
import time

import numpy as np
import pyroomacoustics as pra
import scipy
import scipy.signal as ss

SR = 16000


def one_rir(rng, rt60, max_order_cap=None):
    room_dim = [rng.uniform(3, 10), rng.uniform(3, 10), rng.uniform(2.5, 4.0)]
    e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
    if max_order_cap is not None:
        max_order = min(max_order, max_order_cap)
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_abs), max_order=max_order)
    src = [rng.uniform(0.5, d - 0.5) for d in room_dim]
    mic = [rng.uniform(0.5, d - 0.5) for d in room_dim]
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    return np.asarray(room.rir[0][0], dtype=np.float32), max_order


def main():
    rng = random.Random(0)
    out = {"meta": {"pyroomacoustics": pra.__version__, "scipy": scipy.__version__, "loadavg": os.getloadavg()}}
    print(json.dumps(out), flush=True)
    for rt60 in (0.3, 0.6, 0.9):
        for cap in (None, 12):
            one_rir(rng, rt60, cap)  # warm up
            n, t0 = 10, time.perf_counter()
            lens, orders = [], []
            for _ in range(n):
                h, mo = one_rir(rng, rt60, cap)
                lens.append(len(h))
                orders.append(mo)
            dt = (time.perf_counter() - t0) / n
            print(json.dumps({"rt60_s": rt60, "max_order_cap": cap, "mean_max_order": float(np.mean(orders)),
                              "rir_ms_each": round(dt * 1e3, 1), "rir_len_s": round(float(np.mean(lens)) / SR, 3),
                              "loadavg_1m": round(os.getloadavg()[0], 2)}), flush=True)
    x = np.random.default_rng(0).standard_normal(4 * SR).astype(np.float32)
    for rir_s in (0.25, 0.5, 1.0):
        h = np.random.default_rng(1).standard_normal(int(rir_s * SR)).astype(np.float32)
        for name, fn in (("fftconvolve", ss.fftconvolve), ("oaconvolve", ss.oaconvolve)):
            fn(x, h)
            n, t0 = 50, time.perf_counter()
            for _ in range(n):
                fn(x, h)
            dt = (time.perf_counter() - t0) / n
            print(json.dumps({"convolve": name, "signal_s": 4, "rir_s": rir_s, "ms_each": round(dt * 1e3, 3)}), flush=True)


if __name__ == "__main__":
    main()
