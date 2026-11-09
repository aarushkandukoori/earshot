"""The fixed validation set (10-ml 11, contract 8.4) and listenable WAV previews of the loader's examples.

    .venv/bin/python -m ml.tools.render_val                          # 400 mixtures -> ml/cache/val_v1.npz
    .venv/bin/python -m ml.tools.render_val --preview 20             # 20 of them as WAVs -> ml/runs/preview/
    .venv/bin/python -m ml.tools.render_val --preview 20 --recipe B  # 20 training examples (train split) instead

Validation set: seed 2026, speech_val + noise_val (FSD50K and MUSAN; there are no DEMAND val rows, so DEMAND's share
goes to FSD50K) + sim_val rooms (90%) or dry (10%), no real RIRs. Recipe B values, 4 s segments, Le 6 s, use_wes
always on, rendered in test mode (10-ml 8.9: level -26 dBFS, high-pass 100 Hz, no EQ, no clipping, no Lombard), so
the reference is never distorted by clipping and every checkpoint is scored on the same clean scenes.
Example k uses default_rng([2026, k]). Subsets (10-ml 11 table):
    A 160  recipe B, Person present                         SI-SNRi, TSOS
    B  80  Person present, 1-3 talkers, SIR 0 dB, SNR 5 dB   picked rate at SIR 0 dB, SI-SNRi
    C  60  Person absent                                     absent attenuation
    D  60  Person present, own talker at +6 dB, active 40%   own-voice suppression, SI-SNRi
    E  40  Everyone, with talkers (not the nobody case)      SI-SNRi
e_t and e_o are computed once here with WeSpeaker (validation speakers are not in the own cache).

Subset D re-draws: own-voice suppression is measured only on hops where the own voice talks and the target does not
(metrics.own_suppression_db skips a clip with under 0.2 s of them). The target is fully active in 75% of present
examples, so with the table's overrides alone 37 of 60 D clips had too little own-only audio and the best-checkpoint
rule (own >= 20 dB, contract 8.4) was decided on 23 clips. A D example whose own-only audio is under 0.2 s is
therefore rendered again with default_rng([2026, k, a]) for a = 1, 2, ... (at most 20 tries); the overrides stay
exactly as in the table, and info["val_tries"] records the try that was kept.

Preview: writes <k>_<subset>_mix.wav, _ref.wav, _enr.wav (and _own.wav when an own enrollment is given) as float
WAVs, plus preview.json with the drawn values and simple level statistics of every file.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse      # noqa: E402
import dataclasses   # noqa: E402
import json          # noqa: E402
import math          # noqa: E402
import time          # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np   # noqa: E402
import soundfile as sf   # noqa: E402

from ml.earshot import data   # noqa: E402
from ml.earshot.metrics import OWN_MIN_SAMPLES   # noqa: E402
from ml.earshot.config import CACHE, RUNS, SR   # noqa: E402
from ml.earshot.sources import get_sources   # noqa: E402

VAL_SEED = 2026
VAL_L = 4 * SR             # 64,000 samples
VAL_LE = 6 * SR            # 96,000 samples
VAL_RECIPE = dataclasses.replace(data.RECIPES["B"], name="VAL", rooms=(0.9, 0.0, 0.1))   # sim_val 90 / dry 10
SUBSETS = [("A", 160), ("B", 80), ("C", 60), ("D", 60), ("E", 40)]
SUBSET_CODE = {name: i for i, (name, _) in enumerate(SUBSETS)}
VAL_PATH = CACHE / "val_v1.npz"
PREVIEW_DIR = RUNS / "preview"
D_MAX_TRIES = 20           # subset D: renders tried until the own-only audio reaches OWN_MIN_SAMPLES (0.2 s)


def plan():
    """[(k, subset, overrides)] for the 400 validation examples, subsets in table order."""
    out = []
    k = 0
    for name, count in SUBSETS:
        for i in range(count):
            ov = {"use_wes": True}
            if name == "A":
                ov.update(person=True, absent=False)
            elif name == "B":
                ov.update(person=True, absent=False, n_talkers=1 + i % 3, sir_db=0.0, quiet=False, snr_db=5.0)
            elif name == "C":
                ov.update(person=True, absent=True)
            elif name == "D":
                ov.update(person=True, absent=False, own_given=True, own_in_mix=True, own_db=6.0, own_frac=0.4)
            else:
                ov.update(person=False, absent=False)
            out.append((k, name, ov))
            k += 1
    return out


def own_only_samples(ex):
    """Samples where the own voice talks and the target does not (vad_o = 1, vad_t = 0), as metrics counts them."""
    return 64 * int(np.sum((np.asarray(ex["vad_o"]) > 0) & (np.asarray(ex["vad_t"]) == 0)))


def render_val_example(src, k, overrides, subset=None):
    """Validation example k (generator default_rng([2026, k])), in test mode, with the metric signals of 8.9.

    Subset D only: if the example has under 0.2 s of own-only audio, it is rendered again with
    default_rng([2026, k, a]) for a = 1, 2, ... until it has enough (module docstring). info["val_tries"] = a + 1.
    """
    for a in range(D_MAX_TRIES if subset == "D" else 1):
        rng = np.random.default_rng([VAL_SEED, k] if a == 0 else [VAL_SEED, k, a])
        ex = data.make_example(src, VAL_RECIPE, rng, VAL_L, VAL_LE, overrides, test_mode=True, metric_signals=True)
        ex["info"]["val_tries"] = a + 1
        if subset != "D" or own_only_samples(ex) >= OWN_MIN_SAMPLES:
            return ex
    print(f"warning: validation example {k} (subset D) has under 0.2 s of own-only audio after {D_MAX_TRIES} "
          "tries; own-voice suppression will skip it", flush=True)
    return ex


def dbfs(x):
    """RMS level in dBFS (-inf for silence)."""
    p = float(np.mean(np.asarray(x, dtype=np.float64) ** 2))
    return 10.0 * math.log10(p) if p > 0 else float("-inf")


def stats(x):
    """Level statistics of one signal: RMS dBFS, peak, and the fraction of samples at the peak magnitude
    (a hard-clipped signal has many samples exactly at its largest |x|)."""
    a = np.abs(np.asarray(x, dtype=np.float32))
    peak = float(a.max()) if len(a) else 0.0
    at_peak = float(np.mean(a >= peak * (1 - 1e-6))) if peak > 0 else 0.0
    return {"rms_dbfs": round(dbfs(x), 2), "peak": round(peak, 4), "frac_at_peak": round(at_peak, 6)}


# ---------------------------------------------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------------------------------------------
def preview_items(n, recipe, seed):
    """n (label, example) pairs: validation examples spread over the subsets, or training examples of a recipe."""
    if recipe is None:
        src = get_sources("val")
        by_subset = {name: [p for p in plan() if p[1] == name] for name, _ in SUBSETS}
        picks = []
        i = 0
        while len(picks) < n:                       # round robin: the first examples of A, B, C, D, E, then the next
            name = SUBSETS[i % len(SUBSETS)][0]
            items = by_subset[name]
            if i // len(SUBSETS) < len(items):
                picks.append(items[i // len(SUBSETS)])
            i += 1
        for k, name, ov in picks:
            yield f"{k:03d}_{name}", render_val_example(src, k, ov, name)
    else:
        src = get_sources("train")
        r = data.get_recipe(recipe)
        L = int(r.seconds * SR)
        for k in range(n):                          # example k = example k % 8 of batch k // 8, as the loader makes it
            j = k // data.EXAMPLE_STRIDE
            rng = np.random.default_rng([seed, k])
            yield f"{k:03d}_{r.name}", data.make_example(src, r, rng, L, data.draw_le(seed, j))


def write_preview(n, out_dir, recipe=None, seed=0):
    """Write n previews (float WAVs) and preview.json into out_dir -> the list of per-example records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for label, ex in preview_items(n, recipe, seed):
        files = {"mix": ex["mix"], "ref": ex["ref"]}
        if ex["has_t"]:
            files["enr"] = ex["enr_t"]
        if ex["has_o"]:
            files["own"] = ex["enr_o"]
        record = {"label": label, "info": ex["info"], "files": {}}
        for kind, x in files.items():
            name = f"{label}_{kind}.wav"
            sf.write(str(out_dir / name), np.asarray(x, dtype=np.float32), SR, subtype="FLOAT")
            record["files"][name] = stats(x)
        records.append(record)
        mix = record["files"][f"{label}_mix.wav"]
        print(f"{label}: mix {mix['rms_dbfs']:6.1f} dBFS peak {mix['peak']:.3f} | ref "
              f"{record['files'][f'{label}_ref.wav']['rms_dbfs']:6.1f} dBFS | {ex['info']['room']:4s} "
              f"talkers {ex['info']['n_talkers']} snr {ex['info']['snr_db']:5.1f} clip {ex['info']['clip']}",
              flush=True)
    with open(out_dir / "preview.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)
    return records


# ---------------------------------------------------------------------------------------------------------------
# The validation set
# ---------------------------------------------------------------------------------------------------------------
def write_val(out_path, count=None, device=None):
    """Render the validation set (or its first `count` examples) and save it as an .npz (10-ml 11 arrays)."""
    import torch                                       # imported here: previews need no torch device
    from ml.earshot.wespeaker import WeSpeaker

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    encoder = WeSpeaker.load().to(device)
    src = get_sources("val")
    items = plan()[:count] if count else plan()
    n = len(items)
    arrays = {"mix": np.zeros((n, VAL_L), np.float32), "ref": np.zeros((n, VAL_L), np.float32),
              "talkers": np.zeros((n, 3, VAL_L), np.float16), "own": np.zeros((n, VAL_L), np.float16),
              "vad_t": np.zeros((n, VAL_L // 64), np.uint8), "vad_o": np.zeros((n, VAL_L // 64), np.uint8),
              "vad_talkers": np.zeros((n, 3, VAL_L // 64), np.uint8),
              "enr_t": np.zeros((n, VAL_LE), np.float32), "tok_idx_t": np.zeros((n, 48, 3), np.int64),
              "tok_val_t": np.zeros((n, 48), bool), "enr_o": np.zeros((n, 128000), np.float32),
              "tok_idx_o": np.zeros((n, 16, 3), np.int64), "tok_val_o": np.zeros((n, 16), bool),
              "e_t": np.zeros((n, 256), np.float32), "e_o": np.zeros((n, 256), np.float32),
              "has_t": np.zeros(n, bool), "has_o": np.zeros(n, bool), "use_wes": np.zeros(n, bool),
              "mode": np.zeros((n, 2), np.float32), "present": np.zeros(n, bool), "person": np.zeros(n, bool),
              "subset": np.zeros(n, np.uint8)}
    infos = []
    t0 = time.time()
    for row, (k, name, ov) in enumerate(items):
        ex = render_val_example(src, k, ov, name)
        m = ex["scene"]["metric"]
        for key in ["mix", "ref", "vad_t", "vad_o", "enr_t", "tok_idx_t", "tok_val_t", "enr_o", "tok_idx_o",
                    "tok_val_o", "has_t", "has_o", "use_wes", "mode", "present", "person"]:
            arrays[key][row] = ex[key]
        for i, (r_i, v_i) in enumerate(zip(m["talkers"], m["vad_talkers"])):
            arrays["talkers"][row, i] = r_i
            arrays["vad_talkers"][row, i] = v_i
        arrays["own"][row] = m["own"]
        arrays["subset"][row] = SUBSET_CODE[name]
        with torch.no_grad():
            if ex["has_t"]:
                arrays["e_t"][row] = encoder(torch.from_numpy(ex["enr_t"])[None].to(device))[0].cpu().numpy()
            if ex["has_o"]:
                arrays["e_o"][row] = encoder(torch.from_numpy(ex["own_clean"])[None].to(device))[0].cpu().numpy()
        infos.append(dict(ex["info"], k=k, subset=name))
        if (row + 1) % 50 == 0:
            print(f"  {row + 1}/{n} examples, {time.time() - t0:.0f} s", flush=True)
    d_rows = [i for i, (_, name, _) in enumerate(items) if name == "D"]
    if d_rows:
        own_s = [64 * int(np.sum((arrays["vad_o"][i] > 0) & (arrays["vad_t"][i] == 0))) / SR for i in d_rows]
        redrawn = sum(infos[i]["val_tries"] > 1 for i in d_rows)
        print(f"subset D: own-only audio min {min(own_s):.2f} s, median {float(np.median(own_s)):.2f} s; "
              f"{redrawn} of {len(d_rows)} examples re-drawn", flush=True)
    arrays["info"] = np.array(json.dumps(infos))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB, {n} examples, {time.time() - t0:.0f} s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the validation set or WAV previews (10-ml 11).")
    parser.add_argument("--preview", type=int, default=0, help="write N previews as WAVs instead of the set")
    parser.add_argument("--recipe", default=None, help="with --preview: training examples of this recipe")
    parser.add_argument("--seed", type=int, default=0, help="with --preview --recipe: loader seed (default 0)")
    parser.add_argument("--out", default=None, help="output folder (preview) or .npz (validation set)")
    parser.add_argument("--count", type=int, default=0, help="validation set: only the first N (a quick check)")
    args = parser.parse_args(argv)
    if args.preview:
        write_preview(args.preview, Path(args.out) if args.out else PREVIEW_DIR, args.recipe, args.seed)
    else:
        out = Path(args.out) if args.out else VAL_PATH
        if args.count and out.resolve() == VAL_PATH.resolve():
            parser.error("--count writes an incomplete set; give --out somewhere else")
        write_val(out, args.count)


if __name__ == "__main__":
    main()
