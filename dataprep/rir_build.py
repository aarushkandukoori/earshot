"""RIR bank entry point - simulated rooms and measured RIRs into data/rir/ (plan sections 3.1, 3.2 and 7).

Usage (from the project root):
    .venv/bin/python -m dataprep.rir_build --sim --real          # both
    .venv/bin/python -m dataprep.rir_build --sim --procs 5        # only the pyroomacoustics bank

Outputs (layout in data/rir/README.md, which this script writes):
    sim_{train,val,test}.npy    float16 [rooms, 8, 17600]   3,800 / 100 / 100 rooms, separate seeds
    sim_{train,val,test}.jsonl  one line per room (geometry, absorption, measured RT60, distances, direct-path index)
    real_train.npy / .jsonl     OpenSLR 28 real RIRs, float16 [325, 24000]
    real_test.npy / .jsonl      MIT IR Survey, float16 [270, 24000], test only

Resumable - finished rooms are appended to sim_<split>.partial.jsonl and written into sim_<split>.partial.npy (a
memory-mapped file); a re-run skips them. The final files appear only when a split is complete. Stops cleanly
(exit code 3) if free disk drops below the 12 GB floor. Log - data/logs/rir_build.log.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from . import common
from .common import DiskFull, check_disk, get_logger

FLUSH_EVERY = 25


def rir_dir():
    return common.DATA / "rir"


def _init_worker() -> None:
    import pyroomacoustics as pra
    pra.constants.set("num_threads", 1)


def _task(args):
    from .rir_sim import build_room
    return build_room(*args)


def _write_jsonl(path, rows) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _count_lines(path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_sim_split(log, split: str, procs: int) -> None:
    from .rir_sim import L, ROLES, SPLITS
    n, _ = SPLITS[split]
    d = rir_dir()
    d.mkdir(parents=True, exist_ok=True)
    final_npy, final_jsonl = d / f"sim_{split}.npy", d / f"sim_{split}.jsonl"
    if final_npy.exists() and final_jsonl.exists() and _count_lines(final_jsonl) == n \
            and np.load(final_npy, mmap_mode="r").shape == (n, len(ROLES), L):
        log.info("sim_%s already complete (%d rooms)", split, n)
        return
    part_npy, part_jsonl = d / f"sim_{split}.partial.npy", d / f"sim_{split}.partial.jsonl"
    mm = None
    if part_npy.exists():
        try:
            mm = np.lib.format.open_memmap(part_npy, mode="r+")
            if mm.shape != (n, len(ROLES), L) or mm.dtype != np.float16:
                log.warning("partial bank %s has shape %s, starting this split again", part_npy, mm.shape)
                mm = None
        except Exception as e:
            log.warning("cannot open %s (%s), starting this split again", part_npy, e)
            mm = None
        if mm is None:
            part_npy.unlink()
            part_jsonl.unlink(missing_ok=True)
    if mm is None:
        check_disk()
        mm = np.lib.format.open_memmap(part_npy, mode="w+", dtype=np.float16, shape=(n, len(ROLES), L))
        part_jsonl.unlink(missing_ok=True)
    done = set()
    if part_jsonl.exists():
        with open(part_jsonl, encoding="utf-8") as f:
            done = {json.loads(line)["room"] for line in f if line.strip()}
    todo = [i for i in range(n) if i not in done]
    log.info("sim_%s - %d rooms done, %d to simulate with %d processes", split, len(done), len(todo), procs)
    t0 = time.time()
    if todo:
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(k, "1")
        ctx = mp.get_context("spawn")
        buf = []
        with ctx.Pool(procs, initializer=_init_worker) as pool, open(part_jsonl, "a", encoding="utf-8") as fj:
            for k, (idx, arr, meta) in enumerate(pool.imap_unordered(_task, [(split, i) for i in todo], chunksize=2)):
                mm[idx] = arr
                buf.append(meta)
                if len(buf) >= FLUSH_EVERY or k == len(todo) - 1:
                    mm.flush()
                    for m in buf:
                        fj.write(json.dumps(m) + "\n")
                    fj.flush()
                    buf = []
                    el = time.time() - t0
                    log.info("sim_%s %d/%d rooms (%.2f s per room wall clock, about %.0f s left)", split, k + 1,
                             len(todo), el / (k + 1), el / (k + 1) * (len(todo) - k - 1))
                    check_disk()
    mm.flush()
    del mm
    rows = {}
    with open(part_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                m = json.loads(line)
                rows[m["room"]] = m
    if sorted(rows) != list(range(n)):
        raise RuntimeError(f"sim_{split}: {n - len(rows)} rooms missing after the run")
    _write_jsonl(final_jsonl, [rows[i] for i in range(n)])
    os.replace(part_npy, final_npy)
    part_jsonl.unlink()
    rt = np.array([rows[i]["rt60"] for i in range(n)])
    log.info("sim_%s complete - %d rooms, RT60 median %.2f s (10th-90th percentile %.2f-%.2f s), %.2f GB",
             split, n, np.nanmedian(rt), np.nanpercentile(rt, 10), np.nanpercentile(rt, 90),
             final_npy.stat().st_size / 1e9)


def build_real(log) -> None:
    from .rir_real import fetch_mit, fetch_slr28
    d = rir_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name, fn in (("real_train", fetch_slr28), ("real_test", fetch_mit)):
        npy, jsonl = d / f"{name}.npy", d / f"{name}.jsonl"
        if npy.exists() and jsonl.exists() and np.load(npy, mmap_mode="r").shape[0] == _count_lines(jsonl):
            log.info("%s already complete", name)
            continue
        check_disk()
        arr, rows = fn(log)
        tmp = npy.with_name(npy.name + ".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, npy)
        _write_jsonl(jsonl, rows)
        log.info("%s - %s float16, %.1f MB", name, arr.shape, npy.stat().st_size / 1e6)


README = """# Room impulse responses (RIRs) for Earshot

Made by `dataprep/rir_build.py` (`python -m dataprep.rir_build --sim --real`), following
docs/research/data-training.md sections 3.1 and 3.2. Everything is 16 kHz, float16, one microphone (the phone).
An RIR is the phone microphone's view of one source - convolve a dry clip with it to place that clip in the room.

## Files

| File | Shape | What |
|---|---|---|
| sim_train.npy | [3800, 8, 17600] | simulated rooms for training (seed 1001) |
| sim_val.npy | [100, 8, 17600] | simulated rooms for validation (seed 2002) |
| sim_test.npy | [100, 8, 17600] | simulated rooms for testing (seed 3003) |
| real_train.npy | [325, 24000] | OpenSLR 28 real RIRs (RWCP, REVERB 2014, Aachen AIR), Apache 2.0 |
| real_test.npy | [270, 24000] | MIT IR Survey, CC BY 4.0, test only |

Load with `np.load(path, mmap_mode="r")` so a training worker reads only the rooms it needs. Each `.npy` has a
`.jsonl` next to it with one line per row, in the same order (line i describes row i).

## Simulated bank (sim_*.npy)

`rirs[room, k]` is the RIR from source position k to the microphone of that room, 17,600 samples (1.1 s). RIRs
longer than that end with a 100 ms half-cosine fade-out. The 8 positions are always in this order:

| k | Role | Distance to the mic |
|---|---|---|
| 0 | target - the person the user wants to hear | 0.5 to 2.5 m (log-uniform, mostly near) |
| 1, 2, 3 | competing talkers | 1 to 5 m |
| 4 | own voice - the phone owner's mouth | 0.25 to 0.8 m |
| 5, 6, 7 | extra positions for babble talkers and point noises | 2 to 8 m (shorter in small rooms) |

Two geometries (field `geometry`) -
- in_hand (60 %) - mic 1.0 to 1.4 m high, talkers seated (mouth 1.1 to 1.3 m) or standing (1.45 to 1.75 m),
  own voice 0.25 to 0.5 m away and 0.15 to 0.4 m above the mic, behind the phone.
- on_table (40 %) - phone on a 0.72 to 0.78 m table. Each source also has a mirror image about the table plane,
  scaled by `table.reflection` (0.6 to 0.9) and summed into the same RIR (a phone on a table is close to a boundary
  microphone). Talkers are seated, own voice is 0.4 to 0.8 m away, and every source is above the table and at most
  2 x table height - 5 cm high so its mirror image stays inside the room.

Rooms - length 4 to 20 m, width 3 to 15 m, height 2.5 to 5 m. pyroomacoustics image source model, max_order from
the Sabine formula capped at 40, `use_rand_ism` (5 cm), air absorption on, frequency-flat walls. The RT60 target is
drawn per room (90 % log-uniform 0.2 to 1.3 s, 10 % uniform 1.3 to 2.0 s); the absorption is corrected until the
target source's measured RT60 is within 10 % of it (up to 4 tries). Long RT60s get big, tall rooms - the wall
absorption must stay between 0.06 and 0.95, and with max_order 40 the image sources only reach an RT60 of about
0.42 x min(ab / sqrt(a^2 + b^2)) over the room's wall pairs (measured on test rooms), so the room is redrawn until
the target fits. Always use the measured `rt60`, not `rt60_target`.

Level - physical. The direct path has amplitude 1/r (r in metres), so a source 1 m away in free field gives a unit
impulse, and a talker 4 m away arrives 12 dB weaker (direct path) than one 1 m away. Scale dry clips by their level
at 1 m and let the RIR do the rest, or normalise per source if you prefer.

JSON line fields (one per room) -
- `room`, `split`, `seed` - `np.random.default_rng(seed)` rebuilds the room exactly.
- `geometry`, `dims` [x, y, z] m, `mic` [x, y, z], `table` (null or {height, reflection}).
- `rt60_target`, `rt60` (median of the 8 measured RIRs), `rt60_src` (per source), `rt60_probe` (target-only
  probe). Measured with pyroomacoustics `measure_rt60(decay_db=30)` (Schroeder T30) on the full-length RIR.
- `e_absorption`, `max_order`, `air_absorption`, `rand_ism_disp`.
- `roles`, `pos` (per source [x, y, z]), `dist` (per source, m), `raw_len` (samples before the cut).
- `dp` - per source, the sample index of the direct-path peak; `dp_amp` - the RIR value there.

pyroomacoustics delays every arrival by 40 samples (2.5 ms, half of its 81-tap fractional-delay filter), so
`dp` is about `dist / 343 * 16000 + 40`. This constant delay is the same for every source; leave it in.

## Building the target reference

Plan 3.2 - the training target is the dry target convolved with the direct path plus the first 50 ms of its RIR,
with a 5 ms half-cosine fade, without removing the propagation delay (the causal model never needs look-ahead):

```python
rirs = np.load("data/rir/sim_train.npy", mmap_mode="r")
meta = rows[room]                                          # that room's JSON line
h = rirs[room, 0].astype(np.float32)                       # source 0 = target
end = meta["dp"][0] + 800                                  # 50 ms after the direct path
early = h[:end + 80].copy()
early[end:end + 80] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, 80)))   # 5 ms fade
reference = scipy.signal.fftconvolve(dry_target, early)[:len(dry_target)]
mixture_part = scipy.signal.fftconvolve(dry_target, h)[:len(dry_target)]
```

## Real RIRs (real_*.npy)

Each row is one measured RIR, channel 0 of multichannel files, resampled to 16 kHz where needed, 24,000 samples
(1.5 s) with the same fade-out. Leading silence is removed so the direct path sits at `dp` (at most 32 samples in),
and every RIR is normalised so the direct-path sample is +1.0 (distances are unknown for most of them, so there is
no physical level).

JSON fields - `row`, `name`, `dp`, `trimmed` (samples removed at the start), `rt60` (rough T20; measured RIRs end in
a noise floor), `len_s`, `channels`, `sr_original`, `source`, `license`.
- real_train (OpenSLR 28) also has `rir_id`, `room_id` (93 rooms) and `db` (RWCP, AIR or REVERB2014).
- real_test (MIT IR Survey) also has `id`, `place` (from the file name), `category` (bar, restaurant, indoor,
  vehicle, outdoor) and the flags `bar`, `restaurant` (any food or drink venue - restaurants, cafes, a cafeteria,
  a food court, dining halls, pizzerias, sandwich and ice-cream shops, the wine bar), `indoor` and `vehicle`.
  All MIT IRs were measured 1.5 m from the source.

## Licenses

- Simulated bank - generated here with pyroomacoustics (MIT license); no third-party data inside.
- OpenSLR 28 - Apache 2.0 (openslr.org/28).
- MIT IR Survey - CC BY 4.0 as stated by dataset mirrors; the lab page itself names no license (plan section 8).
  Credit - Traer and McDermott, "Statistics of natural reverberation enable perceptual separation of sound and
  space", PNAS 2016. Used for testing only.
"""


def measured_section() -> str:
    """Numbers measured from the finished .jsonl files, appended to the README."""
    d = rir_dir()
    lines = []
    for split in ("train", "val", "test"):
        p = d / f"sim_{split}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        rt = np.array([r["rt60"] for r in rows], dtype=float)
        tg = np.array([r["rt60_target"] for r in rows], dtype=float)
        pr = np.array([r["rt60_probe"] for r in rows], dtype=float)
        table = np.mean([r["geometry"] == "on_table" for r in rows])
        lines.append(f"- sim_{split} - {len(rows)} rooms, {100 * table:.0f} % on_table; measured RT60 (median of the 8 "
                     f"RIRs) 10th / 50th / 90th percentile {np.nanpercentile(rt, 10):.2f} / {np.nanpercentile(rt, 50):.2f}"
                     f" / {np.nanpercentile(rt, 90):.2f} s, {100 * np.mean(rt <= 1.3):.0f} % at or below 1.3 s; target "
                     f"source within 10 % of its RT60 target in {100 * np.mean(np.abs(np.log(pr / tg)) < np.log(1.1)):.0f} "
                     f"% of rooms")
    for name in ("real_train", "real_test"):
        p = d / f"{name}.jsonl"
        if p.exists():
            rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            rt = np.array([r["rt60"] if r["rt60"] is not None else np.nan for r in rows], dtype=float)
            extra = ""
            if name == "real_test":
                extra = (f", {sum(r['bar'] for r in rows)} bar, {sum(r['restaurant'] for r in rows)} restaurant, "
                         f"{sum(r['indoor'] for r in rows)} indoor")
            lines.append(f"- {name} - {len(rows)} RIRs{extra}; RT60 label median {np.nanmedian(rt):.2f} s")
    return "\n## Measured on this build\n\n" + "\n".join(lines) + "\n" if lines else ""


def write_readme() -> None:
    """Write data/rir/README.md, with the shapes of the files that exist on disk and their measured numbers."""
    d = rir_dir()
    d.mkdir(parents=True, exist_ok=True)
    text = README
    for name, shape in (("real_train", "[325, 24000]"), ("real_test", "[270, 24000]")):
        p = d / f"{name}.npy"
        if p.exists():
            text = text.replace(f"| {name}.npy | {shape} |", f"| {name}.npy | {list(np.load(p, mmap_mode='r').shape)} |")
    (d / "README.md").write_text(text + measured_section(), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="build the pyroomacoustics bank")
    ap.add_argument("--real", action="store_true", help="fetch OpenSLR 28 and the MIT IR Survey")
    ap.add_argument("--procs", type=int, default=5, help="worker processes for --sim (4 to 6 recommended)")
    ap.add_argument("--splits", default="val,test,train", help="which simulated splits, comma separated")
    a = ap.parse_args(argv)
    if not (a.sim or a.real):
        ap.error("pass --sim and/or --real")
    log = get_logger("rir_build")
    log.info("start (sim=%s real=%s), %.1f GB free", a.sim, a.real, common.free_gb())
    try:
        write_readme()
        if a.real:
            build_real(log)
            write_readme()
        if a.sim:
            for split in a.splits.split(","):
                build_sim_split(log, split.strip(), a.procs)
        write_readme()
    except DiskFull as e:
        log.error("stopping cleanly - %s. Free some space and re-run the same command to resume.", e)
        return 3
    log.info("end, %.1f GB free", common.free_gb())
    return 0


if __name__ == "__main__":
    sys.exit(main())
