"""Design Earshot's resampler FIR tables (20-engine 9.1, contract 4.2 "Resamplers").

Run from the repo root:
    .venv/bin/python engine/tools/make_resampler_tables.py

Writes two files:
    engine/src/resampler_tables.c   polyphase tables for the C engine (both I/O rates, both directions)
    ml/assets/decim16k_fir.npy      64-tap float32 filter at 16 kHz that copies the decimator's
                                    0 to 8 kHz response, used by training augmentation (10-ml fir_decim)

and prints the filter figures. Exits non-zero if any figure misses the numbers of 20-engine 9.1.

Design (one prototype low-pass per I/O rate family):
    48 kHz family:   177 taps at 48 kHz
    44.1 kHz family: 26,080 taps at 7.056 MHz (= 44,100 x 160 = 16,000 x 441), the same 3.7 ms span
    both: firwin(numtaps, 7500 Hz, window=("kaiser", 5.65)), made minimum phase with
          minimum_phase(h, "homomorphic", half=False), then scaled so the taps sum to 1 (0 dB at DC).

Polyphase table for up L / down M: taps[p][j] = L * h[p + j L] (0 past the end), each phase stored
reversed (rev[p][Q-1-j] = taps[p][j]) so the C code computes one output with one dot product.
"""

import pathlib
import sys

import numpy as np
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_C = ROOT / "engine" / "src" / "resampler_tables.c"
OUT_NPY = ROOT / "ml" / "assets" / "decim16k_fir.npy"

CUTOFF_HZ = 7500.0
KAISER_BETA = 5.65

# The figures 20-engine 9.1 lists (analytic, scipy 1.18.1). The tool must reproduce them.
EXPECT = {
    "48k": {"ripple_lo": -0.0031, "ripple_hi": 0.0042, "f3db": 7385.0, "stop": -60.0, "gd_mean_ms": 0.1288,
            "delay_ms": 0.2575},
    "44k": {"ripple_lo": -0.0031, "ripple_hi": 0.0043, "f3db": None, "stop": -60.0, "gd_mean_ms": 0.1494,
            "delay_ms": 0.2987},
}
TOL_DB = 0.01      # passband ripple
TOL_STOP_DB = 1.0  # stopband level
TOL_MS = 0.01      # group delay
TOL_F3DB_HZ = 5.0  # -3 dB point (48 kHz family only)


def design(numtaps, fs):
    """The prototype low-pass at rate fs, minimum phase, taps summing to 1."""
    h = signal.firwin(numtaps, CUTOFF_HZ, window=("kaiser", KAISER_BETA), fs=fs)
    h = signal.minimum_phase(h, method="homomorphic", half=False)
    return h / np.sum(h)


def dtft(h, fs, freqs, weight_by_index=False):
    """Exact frequency response sum_n h[n] e^(-i 2 pi f n / fs) at the given frequencies (float64).
    With weight_by_index=True it returns the response of n * h[n] (used for the group delay).
    Done in chunks of 16 frequencies so the 26,080-tap filter does not need a huge matrix."""
    n = np.arange(len(h), dtype=np.float64)
    taps = h * n if weight_by_index else h
    out = np.empty(len(freqs), dtype=np.complex128)
    for i in range(0, len(freqs), 16):
        f = np.asarray(freqs[i:i + 16], dtype=np.float64)
        out[i:i + 16] = np.exp(-2j * np.pi * np.outer(f, n) / fs) @ taps
    return out


def figures(h, fs):
    """Passband ripple to 6.5 kHz, -3 dB point, stopband from 8 kHz, group delay over 1 to 4 kHz."""
    pass_f = np.arange(0.0, 6500.0 + 1e-9, 5.0)
    pass_db = 20 * np.log10(np.abs(dtft(h, fs, pass_f)))

    # -3 dB point: fine scan between 6.5 and 8 kHz, then linear interpolation between grid points.
    edge_f = np.arange(6500.0, 8000.0 + 1e-9, 0.5)
    edge_db = 20 * np.log10(np.abs(dtft(h, fs, edge_f)))
    k = int(np.argmax(edge_db < -3.0))
    f3db = edge_f[k - 1] + (edge_f[k] - edge_f[k - 1]) * (-3.0 - edge_db[k - 1]) / (edge_db[k] - edge_db[k - 1])

    # Stopband: a long zero-padded FFT covers every frequency from 8 kHz up to fs / 2.
    nfft = 1 << 18 if len(h) < 1000 else 1 << 23
    spec = np.abs(np.fft.rfft(h, nfft))
    fgrid = np.arange(len(spec)) * fs / nfft
    stop_db = 20 * np.log10(np.max(spec[fgrid >= 8000.0]))

    # Group delay = Re( DTFT(n h[n]) / DTFT(h[n]) ) samples, exact (no phase unwrapping).
    gd_f = np.arange(1000.0, 4000.0 + 1e-9, 10.0)
    gd = np.real(dtft(h, fs, gd_f, weight_by_index=True) / dtft(h, fs, gd_f)) / fs * 1000.0   # ms

    return {
        "ripple_lo": float(np.min(pass_db)), "ripple_hi": float(np.max(pass_db)), "f3db": float(f3db),
        "stop": float(stop_db), "gd_min_ms": float(np.min(gd)), "gd_max_ms": float(np.max(gd)),
        "gd_mean_ms": float(np.mean(gd)),
    }


def polyphase(h, L):
    """Reversed polyphase table: rev[p][Q-1-j] = L * h[p + j L]. Returns (Q, rev [L][Q] float32)."""
    Q = -(-len(h) // L)                         # ceil(len(h) / L)
    padded = np.zeros(Q * L)
    padded[:len(h)] = h
    taps = L * padded.reshape(Q, L).T           # taps[p][j] = L * h[p + j L]
    return Q, np.ascontiguousarray(taps[:, ::-1]).astype(np.float32)


def c_array(name, values):
    """A static const float array, 6 values per line, 9 significant digits (exact for float32)."""
    lines = [f"static const float {name}[{len(values)}] = {{"]
    for i in range(0, len(values), 6):
        chunk = values[i:i + 6]
        lines.append("  " + ", ".join(f"{float(v):.9e}f" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def write_c(tables, delays, figs):
    """tables: dict name -> (L, M, Q, rev). Writes engine/src/resampler_tables.c."""
    out = [
        "/* resampler_tables.c - GENERATED by engine/tools/make_resampler_tables.py, do not edit by hand.",
        " * The lead regenerates it (contract 11); any re-run gets a DECISIONS row.",
        " *",
        f" * Prototype low-pass: firwin(numtaps, {CUTOFF_HZ:.0f} Hz, kaiser beta {KAISER_BETA}), minimum phase",
        " * (homomorphic, half=False), taps summing to 1. 48 kHz family: 177 taps at 48 kHz.",
        " * 44.1 kHz family: 26,080 taps at 7.056 MHz.",
        " * Each table: rev[p * Q + i] = L * h[p + (Q - 1 - i) * L] (phases stored reversed).",
        " *",
    ]
    for fam in ("48k", "44k"):
        f = figs[fam]
        out.append(f" * {fam}: passband {f['ripple_lo']:+.4f} to {f['ripple_hi']:+.4f} dB to 6.5 kHz, -3 dB at "
                   f"{f['f3db']:.1f} Hz, stopband {f['stop']:.2f} dB from 8 kHz,")
        out.append(f" *      group delay 1 to 4 kHz {f['gd_min_ms']:.4f} to {f['gd_max_ms']:.4f} ms, mean "
                   f"{f['gd_mean_ms']:.5f} ms per direction")
    out += [" */", '#include "internal.h"', ""]
    for name, (L, M, Q, rev) in tables.items():
        out.append(f"/* {name}: up L = {L}, down M = {M}, Q = {Q} taps per phase */")
        out.append(c_array(name, rev.reshape(-1)))
        out.append("")
    out.append("/* Index 0 = 48 kHz, index 1 = 44.1 kHz. */")
    out.append("const es_rs_table es_rs_decim_tables[2] = {")
    for name in ("rs48_decim", "rs44_decim"):
        L, M, Q, _ = tables[name]
        out.append(f"  {{ {L}, {M}, {Q}, {name} }},")
    out.append("};")
    out.append("const es_rs_table es_rs_interp_tables[2] = {")
    for name in ("rs48_interp", "rs44_interp"):
        L, M, Q, _ = tables[name]
        out.append(f"  {{ {L}, {M}, {Q}, {name} }},")
    out.append("};")
    out.append("/* Decimator plus interpolator mean group delay over 1 to 4 kHz, in ms (20-engine 9.1). */")
    out.append(f"const float ES_RS_DELAY_MS[2] = {{ {delays[0]:.9e}f, {delays[1]:.9e}f }};")
    out.append("")
    OUT_C.write_text("\n".join(out))


def make_decim16k(h48):
    """64-tap filter at 16 kHz with the 48 kHz decimator's 0 to 8 kHz response (20-engine 9.1):
    sample the decimator's complex response at the 257 bins of a 512-point grid at 16 kHz, irfft to
    512 taps, keep the first 64 and taper taps 48 to 63 with a falling half Hann window."""
    bins_hz = np.arange(257) * 16000.0 / 512.0
    resp = dtft(h48, 48000.0, bins_hz)
    h512 = np.fft.irfft(resp, n=512)
    h64 = h512[:64].copy()
    h64[48:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(16) / 16.0)
    return h64


def check(name, value, expect, tol, failures, unit):
    ok = abs(value - expect) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {value:.4f} {unit} (expected {expect} +/- {tol})")
    if not ok:
        failures.append(name)


def main():
    failures = []
    h48 = design(177, 48000.0)
    h44 = design(26080, 7056000.0)

    figs = {"48k": figures(h48, 48000.0), "44k": figures(h44, 7056000.0)}
    delays = []
    for fam in ("48k", "44k"):
        f, e = figs[fam], EXPECT[fam]
        print(f"{fam} family")
        print(f"  passband ripple to 6.5 kHz: {f['ripple_lo']:+.4f} to {f['ripple_hi']:+.4f} dB")
        print(f"  -3 dB point: {f['f3db']:.1f} Hz")
        print(f"  stopband from 8 kHz: {f['stop']:.2f} dB")
        print(f"  group delay 1 to 4 kHz, one direction: {f['gd_min_ms']:.4f} to {f['gd_max_ms']:.4f} ms, "
              f"mean {f['gd_mean_ms']:.5f} ms")
        delay = 2.0 * f["gd_mean_ms"]      # the decimator and the interpolator share one prototype
        delays.append(delay)
        print(f"  ES_RS_DELAY_MS (both directions): {delay:.5f} ms")
        check(f"{fam} ripple low", f["ripple_lo"], e["ripple_lo"], TOL_DB, failures, "dB")
        check(f"{fam} ripple high", f["ripple_hi"], e["ripple_hi"], TOL_DB, failures, "dB")
        if e["f3db"] is not None:
            check(f"{fam} -3 dB point", f["f3db"], e["f3db"], TOL_F3DB_HZ, failures, "Hz")
        check(f"{fam} stopband", f["stop"], e["stop"], TOL_STOP_DB, failures, "dB")
        check(f"{fam} mean group delay", f["gd_mean_ms"], e["gd_mean_ms"], TOL_MS, failures, "ms")
        check(f"{fam} ES_RS_DELAY_MS", delay, e["delay_ms"], TOL_MS, failures, "ms")

    tables = {}
    for name, h, L, M in (("rs48_decim", h48, 1, 3), ("rs48_interp", h48, 3, 1),
                          ("rs44_decim", h44, 160, 441), ("rs44_interp", h44, 441, 160)):
        Q, rev = polyphase(h, L)
        tables[name] = (L, M, Q, rev)
        print(f"table {name}: L {L}, M {M}, Q {Q}")
    expected_q = {"rs48_decim": 177, "rs48_interp": 59, "rs44_decim": 163, "rs44_interp": 60}
    for name, q in expected_q.items():
        if tables[name][2] != q:
            print(f"  FAIL {name}: Q {tables[name][2]}, expected {q}")
            failures.append(name)

    # Training filter: magnitude within 0.05 dB of the decimator below 6.5 kHz; report the phase error.
    h64 = make_decim16k(h48)
    f_chk = np.arange(0.0, 6500.0 + 1e-9, 5.0)
    r48 = dtft(h48, 48000.0, f_chk)
    r16 = dtft(h64, 16000.0, f_chk)
    mag_err = np.max(np.abs(20 * np.log10(np.abs(r16) / np.abs(r48))))
    phase_err = np.max(np.abs(np.angle(r16 / r48))) * 180.0 / np.pi
    print(f"decim16k_fir: 64 taps, magnitude error below 6.5 kHz {mag_err:.4f} dB (limit 0.05), "
          f"max phase error {phase_err:.3f} degrees")
    if not mag_err <= 0.05:
        failures.append("decim16k_fir magnitude")

    if failures:
        print("FAILED: " + ", ".join(failures) + " (no files written)")
        return 1
    write_c(tables, delays, figs)
    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, h64.astype(np.float32))
    print(f"wrote {OUT_C.relative_to(ROOT)} and {OUT_NPY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
