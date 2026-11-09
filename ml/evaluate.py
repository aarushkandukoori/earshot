"""Evaluate exported weights (or a checkpoint) on the contract 9 test suites (spec 10-ml section 12, contract 9).

    .venv/bin/python -m ml.evaluate --esw ml/exports/v1/earshot.esw --suites all --out docs/eval/report-v1.md
    .venv/bin/python -m ml.evaluate --ckpt ml/runs/main/best_1.pt --suites restaurant,absent --limit 100
    nice -n 15 .venv/bin/python -m ml.evaluate --esw ml/exports/random_v1/earshot.esw --device cpu --limit 8 \
        --out ml/runs/eval/sample-report.md          # quick CPU check (what the sample report was made with)

What it does:
  1. Load the weights: --esw reads the shipped file with export.load_esw, so the numbers belong to that file;
     --ckpt loads a training checkpoint's EMA weights (raw weights if it has no EMA). Either way the weights get
     an id = the .esw data_sha256 (for a checkpoint, the hash its export would have), written on every CSV row.
  2. Run the suites (testsets.py, stability.py, dipco.py build them; nothing is cached as audio, DECISIONS L
     2026-11-08): every mixture is rendered on the fly from its seed, embedded with WeSpeaker, run through the
     offline model in batches of up to 8, scored with metrics.py, and its result row is appended to
     ml/runs/eval/<suite>.csv. Those CSVs are also the resume cache: a rerun skips the rows already there only
     when the weights id AND the recipe id match (recipe_id: a hash of the code that makes a row, RECIPE_VERSION
     and the device), so rows of other weights, older code or another device are dropped when the file is
     reopened and never reach a report.
  3. Write the markdown report (evalreport.py): one table per contract 9 row with value, pass threshold, goal and
     PASS / FAIL, the restaurant factor marginals, timings, the data inventory and the credits and license notes.

Suites: restaurant (2,000), absent (500), libri2mix (3,000 x 2 targets), accents (500 EdAcc + 500 VoxPopuli),
stability (60 min stream through StreamRef on the CPU), dipco (4 targets, 10 listening clips), parity (the .est
files next to the .esw). --limit N keeps the first N items of each batch suite (accents: N per set), makes the
stability stream N x 8 s long and runs N DiPCo chunks in total; a limited report says it is not a release result.
Devices: --device mps (default) for real runs; the tests and quick checks use --device cpu. StreamRef always runs
on the CPU (contract 3.3 reference).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse          # noqa: E402
import copy              # noqa: E402
import csv               # noqa: E402
import hashlib           # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import multiprocessing   # noqa: E402
import time              # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np       # noqa: E402
import torch             # noqa: E402

from ml import export    # noqa: E402
from ml.earshot import dipco, evalreport, fileio, metrics, stability, testsets   # noqa: E402
from ml.earshot.config import MAX_O_TOK, ROOT, RUNS, SR, WESPK_DIM   # noqa: E402
from ml.earshot.data import RECIPE_VERSION   # noqa: E402
from ml.earshot.model import CondInputs   # noqa: E402
from ml.earshot.stream_ref import StreamRef, pack_cond   # noqa: E402
from ml.earshot.vad import es_vad   # noqa: E402
from ml.train import run_offline   # noqa: E402  (imported, never edited: the live training uses ml/train.py)

SUITES = ["restaurant", "absent", "libri2mix", "accents", "stability", "dipco", "parity"]
BATCH_SUITES = ("restaurant", "absent", "libri2mix", "accents")
EVAL_DIR = RUNS / "eval"
BATCH = 8                     # offline forward batch (as in validation)
UNIT_S = 8                    # --limit N makes the stability stream N x 8 s long
STREAM_PIECE = 60 * SR        # StreamRef runs the stream in 60 s pieces (state carried over)
PARITY_FILES = ("testvec_short.est", "testvec_long.est")   # the test vectors next to the .esw

# CSV columns per suite (fixed, so a cached file with another layout is recognized and rewritten).
GRID_COLS = ["index", "cell", "snr_db", "sir_db", "n_talkers", "own", "room", "enroll", "own_db"]
SCENE_COLS = ["target", "talkers", "own_speaker", "room_index", "geometry", "rt60", "babble", "noise",
              "noise_kinds", "target_frac", "enr_room", "enr_noise", "short_t", "snr_chain_db"]
SEP_COLS = ["si_snr_in", "si_snr_out", "sisnri"]
TIME_COLS = ["nan_out", "render_s", "embed_s", "forward_s", "metrics_s"]
COLUMNS = {
    "restaurant": ["weights", "recipe", "key"] + GRID_COLS + SCENE_COLS + SEP_COLS
    + ["picked", "eligible", "tsos_over", "tsos_active", "own_supp_db", "estoi_gain", "pesq_out", "pesq_in"]
    + TIME_COLS,
    "absent": ["weights", "recipe", "key"] + GRID_COLS + SCENE_COLS + ["atten_db"] + TIME_COLS,
    "accents": ["weights", "recipe", "key", "set"] + GRID_COLS + SCENE_COLS + ["attempt"] + SEP_COLS + TIME_COLS,
    "libri2mix": ["weights", "recipe", "key", "index", "id", "target", "target_speaker", "other_speaker", "L",
                  "Le", "enr_utt", "short_t"] + SEP_COLS + ["picked", "eligible"] + TIME_COLS,
    "stability": ["weights", "recipe", "key", "kind", "start_s", "end_s", "si_snr_in", "si_snr_out", "sisnri",
                  "nan_samples", "in_dbfs", "out_dbfs", "note"],
    "dipco": ["weights", "recipe", "key", "target", "session", "kind", "start_s", "end_s", "atten_db", "win",
              "corr_target", "corr_other", "others", "note"],
    "parity": ["weights", "recipe", "key", "file", "case", "check", "max_err", "tolerance", "snr_db", "snr_min",
               "result", "note"],
}


# ---------------------------------------------------------------------------------------------------------------
# Result rows: ml/runs/eval/<suite>.csv (also the resume cache)
# ---------------------------------------------------------------------------------------------------------------
def fmt(value):
    """A CSV cell: floats with 6 significant digits, None as an empty cell, dicts as JSON, else str()."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def read_rows(path):
    """All rows of a CSV file as dicts of strings ([] if the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# The source files whose code decides what a result row holds (rendering, the model, WeSpeaker, the metrics and
# this file). evalreport.py only formats rows, so it is left out: editing the report never throws rows away.
ROW_CODE = sorted(p for p in (ROOT / "ml" / "earshot").glob("*.py") if p.name != "evalreport.py") + [
    ROOT / "ml" / "evaluate.py", ROOT / "ml" / "train.py", ROOT / "ml" / "export.py",
    ROOT / "dataprep" / "eval_libri2mix.py", ROOT / "dataprep" / "common.py"]


def recipe_id(device):
    """A 16-hex id of HOW the rows are made: the bytes of every ROW_CODE file, RECIPE_VERSION and the device.

    It is written on every CSV row next to the weights id. A cached row is reused only when both ids match, so a
    development run (older code, a changed interpretation, CPU instead of MPS) can never leak into a release
    report that evaluates the same weights."""
    h = hashlib.sha256(f"recipe {RECIPE_VERSION} device {device}\n".encode())
    for path in ROW_CODE:
        if path.exists():
            h.update(path.name.encode() + b"\n" + path.read_bytes())
    return h.hexdigest()[:16]


class CsvLog:
    """The result rows of one suite for one set of weights and one recipe id (see recipe_id).

    Opening keeps the rows already in the file whose weights and recipe ids both match (their keys are in `done`)
    and drops every other row, including rows of an older column layout. The file is rewritten with the kept rows
    on the first add, then every new row is appended and flushed, so a crash loses at most the batch being scored.
    """

    def __init__(self, path, suite, weights, recipe):
        self.path = Path(path)
        self.columns = COLUMNS[suite]
        self.weights = weights
        self.recipe = recipe
        old = read_rows(self.path)
        same_layout = bool(old) and list(old[0].keys()) == self.columns
        self.rows = [r for r in old if same_layout and r.get("weights") == weights and r.get("recipe") == recipe]
        self.done = {r["key"] for r in self.rows}
        self._file = None
        self._writer = None

    def add(self, row):
        """Append one result row (a dict; missing columns are written empty, unknown keys are ignored)."""
        row = {c: fmt(row.get(c)) for c in self.columns}
        row["weights"] = self.weights
        row["recipe"] = self.recipe
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.columns)
            self._writer.writeheader()
            self._writer.writerows(self.rows)
        self._writer.writerow(row)
        self._file.flush()
        self.rows.append(row)
        self.done.add(row["key"])

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------------------------------------------
# Weights, WeSpeaker and the offline forward
# ---------------------------------------------------------------------------------------------------------------
def load_model(esw=None, ckpt=None):
    """-> (model in eval mode on the CPU, info). info["weights"] is the model id written on every CSV row."""
    if esw:
        model, header = export.load_esw(esw)
        return model, {"kind": "esw", "path": str(esw), "weights": header["data_sha256"],
                       "meta": header.get("meta", {})}
    model, step, used_ema = export.load_checkpoint_model(ckpt)
    blob, _ = fileio.build_data(export.to_export(model))        # the data region its earshot.esw would have
    return model, {"kind": "ckpt", "path": str(ckpt), "weights": fileio.sha256_hex(blob), "step": step,
                   "ema": used_ema, "meta": {}}


@torch.no_grad()
def embed(wes, clips, device):
    """WeSpeaker vectors of equal-length 1-D clips -> numpy float32 [n, 256] (contract 6, frozen module)."""
    x = torch.from_numpy(np.ascontiguousarray(np.stack(clips), dtype=np.float32)).to(device)
    return wes(x).float().cpu().numpy()


def model_arrays(items, wes, device):
    """Stack items (equal mix and enrollment lengths) into the arrays train.run_offline reads, with e_t from the
    target enrollment and e_o from the clean own clip (as render_val does); e_o is zeros without an own voice."""
    keys = [k for k in testsets.MODEL_KEYS if k != "ref"]
    arrays = {k: np.stack([np.asarray(it[k]) for it in items]) for k in keys}
    arrays["e_t"] = embed(wes, list(arrays["enr_t"]), device)
    arrays["e_o"] = np.zeros((len(items), WESPK_DIM), np.float32)
    own = [i for i, it in enumerate(items) if it["has_o"]]
    if own:
        arrays["e_o"][own] = embed(wes, [items[i]["own_clean"] for i in own], device)
    return arrays


def offline(model, arrays, device):
    """train.run_offline on all rows of `arrays` in one batch -> y float32 [n, L]."""
    return run_offline(model, arrays, device, batch=len(arrays["mix"]))


def packed_cond(model, arrays):
    """The PackedCond (engine layout) of row 0 of model_arrays output, for StreamRef (CPU)."""
    t = {k: torch.from_numpy(np.ascontiguousarray(v[:1])) for k, v in arrays.items()}
    with torch.no_grad():
        tok_t, g_t = model.encode_enrollment(t["enr_t"], t["tok_idx_t"], t["tok_val_t"])
        tok_o, g_o = model.encode_enrollment(t["enr_o"], t["tok_idx_o"], t["tok_val_o"])
        cond = model.build_cond(CondInputs(t["e_t"], tok_t, t["tok_val_t"], g_t, t["e_o"], tok_o, t["tok_val_o"], g_o,
                                           t["has_t"], t["has_o"], t["use_wes"], t["mode"]))
    return pack_cond(cond, 0)


def no_own(item):
    """Fill the own-voice inputs of an item that has none (DiPCo: nobody at the table owns the phone)."""
    item.update(enr_o=np.zeros(128000, np.float32), tok_idx_o=np.zeros((MAX_O_TOK, 3), np.int64),
                tok_val_o=np.zeros(MAX_O_TOK, bool), has_o=False, own_clean=None)
    return item


# ---------------------------------------------------------------------------------------------------------------
# Batch suites: restaurant, absent, libri2mix, accents
# ---------------------------------------------------------------------------------------------------------------
def job_key(job):
    """CSV key of a job: the item index, or "<set>:<index>" for accents."""
    suite, key = job
    return f"{key[0]}:{key[1]}" if suite == "accents" else str(key)


def item_metrics(suite, item, y):
    """The 12.2 metrics of one item's output y (numpy float64 math in metrics.py) -> dict of CSV fields."""
    x, s, row = item["mix"], item["ref"], item["row"]
    out = {"nan_out": int(not np.all(np.isfinite(y)))}
    if suite == "absent":
        out["atten_db"] = metrics.attenuation_db(x, y)
        return out
    out["si_snr_in"] = float(metrics.si_snr(x, s))
    out["si_snr_out"] = float(metrics.si_snr(y, s))
    out["sisnri"] = out["si_snr_out"] - out["si_snr_in"]
    if suite in ("restaurant", "libri2mix"):
        out["picked"], out["eligible"] = metrics.picked(y, s, item["talkers"], item["vad_t"], item["vad_talkers"])
    if suite == "restaurant":
        out["tsos_over"], out["tsos_active"] = metrics.tsos(s, y)
        if row["own"] == "on":
            out["own_supp_db"] = metrics.own_suppression_db(item["own"], y, item["vad_o"], item["vad_t"])
        # ESTOI and PESQ are slow (pystoi / pesq on the CPU); their seconds go to the timer under "_" keys.
        if row["snr_db"] in (0.0, 5.0):                  # ESTOI gain is gated on the SNR 0-5 dB cells
            t0 = time.perf_counter()
            out["estoi_gain"] = metrics.estoi_gain(s, y, x)
            out["_estoi_s"] = time.perf_counter() - t0
        if row["snr_db"] >= 5.0:                         # PESQ is reported on the SNR >= 5 dB cells
            t0 = time.perf_counter()
            out["pesq_out"] = metrics.pesq_wb(s, y)
            out["pesq_in"] = metrics.pesq_wb(s, x)
            out["_pesq_s"] = time.perf_counter() - t0
    return out


def metric_call_costs():
    """Seconds of one ESTOI gain (2 pystoi calls) and one PESQ pair (2 pesq calls) on 8 s of synthetic audio, for
    the full-run estimate when a quick run had no SNR >= 0 dB cells. None for a package that is not installed."""
    rng = np.random.default_rng(0)
    t = np.arange(8 * SR) / SR
    s = (0.05 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    x = s + 0.02 * rng.standard_normal(len(s)).astype(np.float32)
    costs = {}
    for name, call in (("estoi", lambda: metrics.estoi_gain(s, x, x)),
                       ("pesq", lambda: (metrics.pesq_wb(s, x), metrics.pesq_wb(s, s)))):
        if metrics.available(name):
            t0 = time.perf_counter()
            call()
            costs[name] = round(time.perf_counter() - t0, 4)
        else:
            costs[name] = None
    return costs


def render_iter(jobs, workers):
    """Items of the jobs in order: rendered here (workers 0) or by a spawn pool of `workers` processes. The workers
    inherit this process's environment, so their numeric libraries stay single-threaded (top of this file)."""
    if workers <= 0:
        for job in jobs:
            yield testsets.build_item(job)
        return
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool:
        yield from pool.imap(testsets.build_item, jobs, chunksize=1)


class Timer:
    """Seconds spent per stage and items done, for the report's timing table."""

    def __init__(self):
        self.s = {"embed": 0.0, "forward": 0.0, "metrics": 0.0, "render": 0.0, "wall": 0.0}
        self.items = 0
        self.cached = 0                                       # rows reused from the resume cache
        self.audio_s = 0.0
        self.calls = {"estoi": [0, 0.0], "pesq": [0, 0.0]}     # [calls, seconds], kept out of "metrics"

    def as_dict(self):
        n = max(self.items, 1)
        out = {"items": self.items, "cached_rows": self.cached, "audio_s": round(self.audio_s, 1),
               **{f"{k}_s_per_item": round(v / n, 4) for k, v in self.s.items()},
               "forward_s_per_8s": round(self.s["forward"] / max(self.audio_s / 8.0, 1e-9), 4)}
        for name, (count, secs) in self.calls.items():
            out[f"{name}_s_per_call"] = round(secs / count, 4) if count else None
        return out


def score_batch(suite, items, ctx, log, timer):
    """Embed, forward and score one batch (equal lengths); append the rows to the suite's CSV."""
    t0 = time.perf_counter()
    arrays = model_arrays(items, ctx.wes, ctx.device)
    t1 = time.perf_counter()
    y = offline(ctx.model_dev, arrays, ctx.device)
    t2 = time.perf_counter()
    rows = []
    slow = 0.0
    for item, y_i in zip(items, y):
        m = item_metrics(suite, item, y_i)
        for name in ("estoi", "pesq"):
            if f"_{name}_s" in m:
                secs = m.pop(f"_{name}_s")
                timer.calls[name][0] += 1
                timer.calls[name][1] += secs
                slow += secs
        rows.append({"key": item["key"], **item["row"], **m})
    t3 = time.perf_counter() - slow                   # metrics time without ESTOI / PESQ
    n = len(items)
    for row in rows:
        row.update(embed_s=(t1 - t0) / n, forward_s=(t2 - t1) / n, metrics_s=(t3 - t2) / n)
        log.add(row)
    timer.s["embed"] += t1 - t0
    timer.s["forward"] += t2 - t1
    timer.s["metrics"] += t3 - t2
    timer.s["render"] += sum(float(it["row"]["render_s"]) for it in items)
    timer.items += n
    timer.audio_s += sum(len(it["mix"]) for it in items) / SR


def run_batch_suite(suite, ctx):
    """Render, run and score every job of the suite not yet in its CSV (for these weights)."""
    jobs = testsets.suite_jobs(suite, ctx.limit)
    log = CsvLog(ctx.eval_dir / f"{suite}.csv", suite, ctx.weights, ctx.recipe)
    todo = [j for j in jobs if job_key(j) not in log.done]
    print(f"[{suite}] {len(jobs)} items, {len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
    timer = Timer()
    timer.cached = len(jobs) - len(todo)              # rows reused from the CSV (the report says how many)
    t_start = time.perf_counter()
    pending = []
    next_report = 50                                  # print progress every 50 items
    try:
        for job, item in zip(todo, render_iter(todo, ctx.workers)):
            item["key"] = job_key(job)
            shape = (len(item["mix"]), len(item["enr_t"]))
            if pending and (len(pending) == BATCH or shape != (len(pending[0]["mix"]), len(pending[0]["enr_t"]))):
                score_batch(suite, pending, ctx, log, timer)
                pending = []
                if timer.items >= next_report:
                    next_report += 50
                    elapsed = time.perf_counter() - t_start
                    eta = elapsed / timer.items * (len(todo) - timer.items)
                    print(f"  {timer.items}/{len(todo)} items, {elapsed:.0f} s, about {eta:.0f} s left", flush=True)
            pending.append(item)
        if pending:
            score_batch(suite, pending, ctx, log, timer)
    finally:
        log.close()
    timer.s["wall"] = time.perf_counter() - t_start
    keys = {job_key(j) for j in jobs}
    return [r for r in log.rows if r["key"] in keys], timer.as_dict()


# ---------------------------------------------------------------------------------------------------------------
# Stability: one long stream through StreamRef on the CPU
# ---------------------------------------------------------------------------------------------------------------
def span_row(kind, lo, hi, x, s, y, nan_samples):
    """One stability CSV row for samples [lo, hi) of the aligned signals."""
    s_in, s_out = float(metrics.si_snr(x[lo:hi], s[lo:hi])), float(metrics.si_snr(y[lo:hi], s[lo:hi]))
    return {"key": f"{kind}:{lo / SR:g}", "kind": kind, "start_s": lo / SR, "end_s": hi / SR, "si_snr_in": s_in,
            "si_snr_out": s_out, "sisnri": s_out - s_in, "nan_samples": nan_samples, "in_dbfs": dipco.dbfs(x[lo:hi]),
            "out_dbfs": dipco.dbfs(y[lo:hi])}


def run_stability(ctx):
    """Build the stream (stability.py), stream it hop by hop with StreamRef, check for NaN, compare the first and
    last windows. Reuses a complete result for the same weights and stream length."""
    seconds = stability.STREAM_S if ctx.limit is None else int(ctx.limit) * UNIT_S
    path = ctx.eval_dir / "stability.csv"
    old = [r for r in read_rows(path) if ctx.matches(r)]
    if any(r["kind"] == "last" for r in old) and any(r["kind"] == "scene" and float(r["end_s"]) == seconds
                                                    for r in old):
        print(f"[stability] reusing {path}", flush=True)
        return old, {"items": 0, "cached": True, "cached_rows": len(old)}
    log = CsvLog(path, "stability", ctx.weights, ctx.recipe)
    log.rows, log.done = [], set()                    # a new stream replaces any partial result
    t0 = time.perf_counter()
    scene = stability.build_stream(testsets.get_eval_sources(), seconds)
    t_build = time.perf_counter() - t0
    arrays = model_arrays([scene], ctx.wes, ctx.device)
    pc = packed_cond(ctx.model_cpu, arrays)
    ref = StreamRef(ctx.model_cpu)
    x_all, n = scene["mix"], len(scene["mix"])
    y = np.zeros(n, dtype=np.float32)
    print(f"[stability] {seconds} s stream built in {t_build:.0f} s; streaming {n // 64} hops", flush=True)
    t1 = time.perf_counter()
    for lo in range(0, n, STREAM_PIECE):
        hi = min(n, lo + STREAM_PIECE)
        y[lo:hi], _ = ref.run(pc, x_all[lo:hi], reset=(lo == 0))
        if (lo // STREAM_PIECE) % 5 == 4:
            done = hi / n
            el = time.perf_counter() - t1
            print(f"  {hi / SR / 60:.0f} min streamed, {el:.0f} s, about {el / done - el:.0f} s left", flush=True)
    t_stream = time.perf_counter() - t1
    # The stream is the offline output delayed by one hop: y_stream[n + 64] = y_offline[n]. Align before scoring.
    ya, xa, sa = y[64:], x_all[:-64], scene["ref"][:-64]
    m = len(ya)
    log.add({"key": "scene", "kind": "scene", "start_s": 0, "end_s": seconds, "note": scene["info"]})
    for lo in range(0, m, 60 * SR):
        hi = min(m, lo + 60 * SR)
        log.add(span_row("minute", lo, hi, xa, sa, ya, int(np.sum(~np.isfinite(y[lo + 64:hi + 64])))))
    (a0, a1), (b0, b1) = stability.window_bounds(m)
    log.add(span_row("first", a0, a1, xa, sa, ya, int(np.sum(~np.isfinite(ya[a0:a1])))))
    log.add(span_row("last", b0, b1, xa, sa, ya, int(np.sum(~np.isfinite(ya[b0:b1])))))
    log.add({"key": "total", "kind": "total", "start_s": 0, "end_s": m / SR,
             "nan_samples": int(np.sum(~np.isfinite(y))),
             "note": {"build_s": round(t_build, 1), "stream_s": round(t_stream, 1),
                      "ms_per_hop": round(1000 * t_stream / max(n // 64, 1), 3)}})
    log.close()
    return log.rows, {"items": 1, "build_s": round(t_build, 1), "stream_s": round(t_stream, 1),
                      "ms_per_hop": round(1000 * t_stream / max(n // 64, 1), 3), "seconds": seconds}


# ---------------------------------------------------------------------------------------------------------------
# DiPCo: real dinner-party recordings, 4 targets
# ---------------------------------------------------------------------------------------------------------------
def dipco_counts(limit):
    """Chunks per target: all (None) for a full run; --limit N spreads N chunks over the 4 targets in turn."""
    if limit is None:
        return [None] * len(dipco.TARGETS)
    n = int(limit)
    return [n // len(dipco.TARGETS) + (1 if k < n % len(dipco.TARGETS) else 0) for k in range(len(dipco.TARGETS))]


def run_dipco(ctx):
    """Enrollment, chunked offline forward, absent and envelope windows, 10 listening clips (dipco.py)."""
    sessions = dipco.load()
    log = CsvLog(ctx.eval_dir / "dipco.csv", "dipco", ctx.weights, ctx.recipe)
    log.rows, log.done = [], set()                    # always recomputed (minutes on MPS)
    far, close, outputs, candidates = {}, {}, {}, {}
    timer = Timer()
    t_start = time.perf_counter()
    for (ses_name, target), count in zip(dipco.TARGETS, dipco_counts(ctx.limit)):
        if count == 0:
            continue
        ses = sessions[ses_name]
        enr = dipco.enrollment(ses, target)
        if ses_name not in far:
            far[ses_name] = dipco.far_input(ses)
        x = far[ses_name]
        # A quick run starts at the chunk holding the end of the enrollment span, so it has windows to score.
        first = 0 if count is None else int(enr["span"][1] * SR) // dipco.CHUNK
        plan = dipco.chunk_plan(len(x), first, count)
        item = no_own({"enr_t": enr["enr"], "tok_idx_t": enr["tok_idx"], "tok_val_t": enr["tok_val"], "has_t": True,
                       "use_wes": True, "mode": np.array([1.0, 0.0], np.float32)})
        y = np.zeros(len(x), dtype=np.float32)
        t0 = time.perf_counter()
        arrays = model_arrays([dict(item, mix=x[:64])], ctx.wes, ctx.device)    # e_t once per target
        timer.s["embed"] += time.perf_counter() - t0
        for in_lo, out_lo, out_hi in plan:
            arrays["mix"] = x[in_lo:out_hi][None]
            t1 = time.perf_counter()
            y[out_lo:out_hi] = offline(ctx.model_dev, arrays, ctx.device)[0][out_lo - in_lo:]
            timer.s["forward"] += time.perf_counter() - t1
            timer.items += 1
            timer.audio_s += (out_hi - in_lo) / SR
        lo_s, hi_s = plan[0][1] / SR, plan[-1][2] / SR
        outputs[target] = (ses_name, y, lo_s, hi_s, enr["span"])
        # "full" = every chunk of the session ran (no --limit); "nan_samples" = non-finite output samples, which
        # the absent and envelope windows cannot show (the report fails the row when it is above 0).
        log.add({"key": f"{target}:enroll", "target": target, "session": ses_name, "kind": "enroll",
                 "start_s": enr["span"][0], "end_s": enr["span"][1],
                 "note": {"segments": enr["segments"], "speech_s": enr["speech_s"], "short": enr["short"],
                          "scored_from_s": lo_s, "scored_to_s": hi_s, "chunks": len(plan), "full": count is None,
                          "nan_samples": int(np.sum(~np.isfinite(y[plan[0][1]:plan[-1][2]])))}})
        t0 = time.perf_counter()
        for t_w in dipco.absent_windows(ses, target, lo_s, hi_s, enr["span"]):
            a, b = int(t_w * SR), int((t_w + dipco.WIN_S) * SR)
            log.add({"key": f"{target}:absent:{t_w:g}", "target": target, "session": ses_name, "kind": "absent",
                     "start_s": t_w, "end_s": t_w + dipco.WIN_S, "atten_db": metrics.attenuation_db(x[a:b], y[a:b])})
        windows = dipco.envelope_windows(ses, target, lo_s, hi_s, enr["span"])
        if windows and ses_name not in close:
            close[ses_name] = {spk: dipco.read_row(path) for spk, path in ses["close"].items()}
        for t_w, rivals in windows:
            a, b = int(t_w * SR), int((t_w + dipco.WIN_S) * SR)
            c = close[ses_name]
            win, c_t, c_o = dipco.envelope_win(y[a:b], c[target][a:b], [c[j][a:b] for j in rivals])
            log.add({"key": f"{target}:envelope:{t_w:g}", "target": target, "session": ses_name, "kind": "envelope",
                     "start_s": t_w, "end_s": t_w + dipco.WIN_S, "win": int(win), "corr_target": c_t,
                     "corr_other": c_o, "others": "|".join(rivals)})
        candidates[target] = dipco.listen_candidates(ses, target, lo_s, hi_s, enr["span"])
        timer.s["metrics"] += time.perf_counter() - t0
    clips = []
    for target, t_w in dipco.pick_listening(candidates):
        ses_name, y, _, _, _ = outputs[target]
        a, b = int(t_w * SR), int((t_w + dipco.LISTEN_S) * SR)
        x_c, y_c = far[ses_name][a:b], y[a:b]
        ses = sessions[ses_name]
        close_t = dipco.read_part(ses["close"][target], a, b - a)
        clips.append({"x": x_c, "y": y_c, "target": target, "session": ses_name, "start_s": t_w,
                      "target_talk_s": round(dipco.talk_seconds(ses["segments"][target], t_w, t_w + 10), 2),
                      "others_talk_s": round(dipco.talk_seconds(dipco.others_of(ses, target), t_w, t_w + 10), 2),
                      "in_dbfs": round(dipco.dbfs(x_c), 2), "out_dbfs": round(dipco.dbfs(y_c), 2),
                      "env_corr_in": round(metrics.env_corr(x_c, close_t), 4),
                      "env_corr_out": round(metrics.env_corr(y_c, close_t), 4)})
    listen_dir = ctx.eval_dir / "dipco_listen"
    csv_path = dipco.write_listening(listen_dir, clips)
    log.add({"key": "listening", "kind": "listening", "note": {"clips": len(clips), "csv": str(csv_path)}})
    log.close()
    timer.s["wall"] = time.perf_counter() - t_start
    return log.rows, timer.as_dict()


# ---------------------------------------------------------------------------------------------------------------
# Parity: the .est test vectors next to the .esw, rerun through StreamRef (12.3; the C side is WP-ENGINE's)
# ---------------------------------------------------------------------------------------------------------------
def compare(name, case, check, got, want, tol, log, snr_min=None):
    """One parity check: max abs error (and SNR) of `got` vs the stored `want` against the file's tolerance."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if got.shape != want.shape:
        log.add({"key": f"{name}:{case}:{check}", "file": name, "case": case, "check": check, "result": "FAIL",
                 "note": f"shape {list(got.shape)} vs stored {list(want.shape)}"})
        return False
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    row = {"key": f"{name}:{case}:{check}", "file": name, "case": case, "check": check, "max_err": err,
           "tolerance": tol}
    ok = err <= tol
    if snr_min is not None:
        noise = float(np.sum((got - want) ** 2))
        snr = 10.0 * math.log10(float(np.sum(want ** 2)) / noise) if noise > 0 else float("inf")
        row.update(snr_db=snr, snr_min=snr_min)
        ok = ok and snr >= snr_min
    row["result"] = "PASS" if ok else "FAIL"
    log.add(row)
    return ok


def run_parity(ctx):
    """Reload each .est next to the .esw, recompute every stored array from its inputs (vad, mask_o, tokens, g,
    the 4 conditioning sets, StreamRef outputs, heads, dbg taps) and compare with the file's own meta.tolerances.

    This is a PyTorch self-check: the same export.py and StreamRef code that wrote the .est recompute it, so it
    proves the files and weights belong together (file integrity), not that the reference model is right. The
    meaningful parity is WP-ENGINE's C run against these files."""
    log = CsvLog(ctx.eval_dir / "parity.csv", "parity", ctx.weights, ctx.recipe)
    log.rows, log.done = [], set()
    if ctx.info["kind"] != "esw":
        log.add({"key": "none", "result": "NOT RUN", "note": "parity needs --esw (test vectors live next to it)"})
        log.close()
        return log.rows, {"items": 0}
    t0 = time.perf_counter()
    folder = Path(ctx.info["path"]).parent
    model = ctx.model_cpu
    for name in PARITY_FILES:
        path = folder / name
        if not path.exists():
            log.add({"key": f"{name}:file", "file": name, "check": "file", "result": "FAIL", "note": "missing"})
            continue
        _, header, a = fileio.read_container(path, expect_magic="EST1")
        meta = header["meta"]
        tol = meta.get("tolerances", {})
        keys_ok = list(tol) == export.TOLERANCE_KEYS
        hash_ok = meta.get("weights_sha256") == ctx.weights
        log.add({"key": f"{name}:header", "file": name, "check": "header", "result": "PASS" if keys_ok and hash_ok
                 else "FAIL", "note": f"tolerance keys {'ok' if keys_ok else list(tol)}; weights_sha256 "
                                      f"{'matches' if hash_ok else 'differs from'} the .esw"})
        if not keys_ok:
            continue
        # es_vad of both enrollments: vad_t is stored as is; mask_o is es_vad(enr_o) by the file's own notes
        # (export.make_testvecs), so it checks the own-voice side of the VAD as well.
        for check, clip, stored in (("vad", "enr_t", "vad_t"), ("vad_o", "enr_o", "mask_o")):
            got = es_vad(a[clip])
            mism = int(np.sum(got != a[stored])) if len(got) == len(a[stored]) else -1
            log.add({"key": f"{name}::{check}", "file": name, "check": check, "max_err": mism,
                     "tolerance": tol["vad"], "result": "PASS" if 0 <= mism <= tol["vad"] else "FAIL",
                     "note": f"frames where es_vad({clip}) differs from the stored {stored}"})
        ci, tok_t, g_t, tok_o, g_o = export.case_cond_inputs(model, a["enr_t"], a["mask_t"], a["enr_o"], a["mask_o"],
                                                             a["e_t"], a["e_o"])
        for check, got in (("tok_t", tok_t), ("tok_o", tok_o)):
            compare(name, "", check, got, a[check], tol["tokens"], log)
        for check, got in (("g_t", g_t), ("g_o", g_o)):
            compare(name, "", check, got, a[check], tol["g"], log)
        with torch.no_grad():
            cond = model.build_cond(ci)
        ref = StreamRef(model)
        for i, case in enumerate(export.CASES):
            pc = pack_cond(cond, i)
            for part in ("film", "k", "v"):
                compare(name, case, part, getattr(pc, part).numpy(), a[f"cond.{case}.{part}"], tol[part], log)
            n_ok = pc.n == int(a[f"cond.{case}.n"][0])
            log.add({"key": f"{name}:{case}:n", "file": name, "case": case, "check": "n", "result":
                     "PASS" if n_ok else "FAIL", "note": f"{pc.n} vs stored {int(a[f'cond.{case}.n'][0])}"})
            debug = export.DEBUG_HOPS if (case == "person" and "dbg.feat" in a) else 0
            res = ref.run(pc, a["in"], debug=debug)
            compare(name, case, "out", res[0], a[f"out.{case}"], tol["out"], log, snr_min=tol["out_snr_db"])
            compare(name, case, "heads", res[1], a[f"heads.{case}"], tol["heads"], log)
            if debug:
                for tap, got in res[2].items():
                    compare(name, case, f"dbg.{tap}", got, a[f"dbg.{tap}"], tol["dbg"], log)
    log.close()
    return log.rows, {"items": 1, "wall_s": round(time.perf_counter() - t0, 1)}


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------
class Context:
    """Everything the suite runners share (weights, devices, options)."""

    def __init__(self, args):
        self.limit = args.limit
        self.workers = args.workers
        self.device = args.device
        self.eval_dir = Path(args.eval_dir)
        self.model_cpu, self.info = load_model(args.esw, args.ckpt)
        self.weights = self.info["weights"]
        self.recipe = recipe_id(self.device)
        self.info["recipe"] = self.recipe             # shown in the report header
        self.model_dev = self.model_cpu if self.device == "cpu" else copy.deepcopy(self.model_cpu).to(self.device)
        from ml.earshot.wespeaker import WeSpeaker   # imported here: loading it reads models/ (26 MB)
        self.wes = WeSpeaker.load().to(self.device)

    def matches(self, row):
        """Was this CSV row made by these weights with this code and device (so it may be reused)?"""
        return row.get("weights") == self.weights and row.get("recipe") == self.recipe


RUNNERS = {"stability": run_stability, "dipco": run_dipco, "parity": run_parity}


def parse_suites(text):
    """"all" or a comma list -> suite names in SUITES order."""
    names = SUITES if text.strip() == "all" else [s.strip() for s in text.split(",") if s.strip()]
    unknown = [s for s in names if s not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; choose from {SUITES} or all")
    return [s for s in SUITES if s in names]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate Earshot weights on the contract 9 suites (10-ml 12).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--esw", help="exported weights (earshot.esw); the numbers belong to this file")
    src.add_argument("--ckpt", help="training checkpoint (EMA weights if present)")
    p.add_argument("--suites", default="all", help="all or a comma list of " + ",".join(SUITES))
    p.add_argument("--out", default=str(EVAL_DIR / "report.md"), help="markdown report path")
    p.add_argument("--device", default="mps", help="mps (default) or cpu for the offline forward and WeSpeaker")
    p.add_argument("--limit", type=int, default=None, help="quick run: N items per batch suite (see docstring)")
    p.add_argument("--workers", type=int, default=0, help="spawn processes that render mixtures (0 = inline)")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    p.add_argument("--eval-dir", default=str(EVAL_DIR), help="folder of the per-suite CSVs and listening clips")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; pass --device cpu")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    torch.set_num_threads(args.threads)
    suites = parse_suites(args.suites)
    t0 = time.perf_counter()
    ctx = Context(args)
    print(f"weights {ctx.info['path']} ({ctx.info['kind']}, id {ctx.weights[:16]}), recipe {ctx.recipe}, "
          f"device {ctx.device}, suites {','.join(suites)}, limit {args.limit}", flush=True)
    results, timings = {}, {}
    for suite in suites:
        t_suite = time.perf_counter()
        runner = run_batch_suite if suite in BATCH_SUITES else RUNNERS[suite]
        if suite in BATCH_SUITES:
            results[suite], timings[suite] = runner(suite, ctx)
        else:
            results[suite], timings[suite] = runner(ctx)
        timings[suite]["suite_wall_s"] = round(time.perf_counter() - t_suite, 1)
        if suite == "restaurant" and timings[suite].get("items"):
            costs = metric_call_costs()               # used only where this run made no such call
            for name in ("estoi", "pesq"):
                if timings[suite].get(f"{name}_s_per_call") is None and costs[name] is not None:
                    timings[suite][f"{name}_s_per_call"] = costs[name]
                    timings[suite][f"{name}_calibrated"] = True
        print(f"[{suite}] done in {timings[suite]['suite_wall_s']:.0f} s", flush=True)
    # Rows of suites not run this time come from their CSVs when the weights AND recipe ids match (e.g. accents
    # needs the restaurant all-cells median); the report marks them as cached.
    cached = {}
    for suite in SUITES:
        if suite not in results:
            rows = [r for r in read_rows(Path(args.eval_dir) / f"{suite}.csv") if ctx.matches(r)]
            if rows:
                cached[suite] = rows
    text = evalreport.build_report(
        info=ctx.info, results=results, cached=cached, timings=timings, limit=args.limit, device=args.device,
        workers=args.workers, threads=args.threads, metric_status=metrics.status(),
        inventory=evalreport.inventory(), command=" ".join(["python -m ml.evaluate"] + list(argv or [])),
        wall_s=time.perf_counter() - t0, accent_excluded=testsets.get_accent_sources().excluded)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"report written to {out} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return results


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
