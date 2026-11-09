"""Training loop for ESB-16-DF (spec 10-ml section 10, contract 8.3 and 8.4).

    .venv/bin/python -m ml.train --config ml/configs/smoke.yaml       # smoke gate, about 0.5 h
    .venv/bin/python -m ml.train --config ml/configs/pilot.yaml       # pilot = main steps 0-6,000, about 1.9 h
    ml/tools/run_forever.sh ml/configs/main.yaml                      # main run (resumes the pilot), about 38.5 h
    .venv/bin/python -m ml.train --config ml/configs/smoke.yaml --run sanity_smoke --max-steps 40   # sanity check

How one run works:
  1. Read the YAML config (config.load_yaml merges `base:` files) and apply the command-line overrides.
  2. Build the model (init_train), the AAM speaker head, AdamW and the frozen WeSpeaker, all on MPS.
  3. With --resume, load a checkpoint: weights, EMA, optimizer, AAM head, completed steps, data seed.
  4. The batch for step s is batch index s of the stage's recipe (data.make_loader). Every example is seeded by
     (seed, example index), so a resumed run sees exactly the data an uninterrupted run would have seen.
  5. One step = step_batch (10-ml 10.2). Every 50 steps a train.csv row; every ckpt_every steps a checkpoint;
     every val_every steps a validation on the 400 fixed mixtures; gate files for the smoke and pilot runs.

Step counting: `step` always means COMPLETED steps. The update that completes step n uses the learning rate
lr_at(n), so update 2,000 runs at the peak and update 120,000 at the final rate. Checkpoint step_<n>.pt holds the
state after n steps, and a resumed run continues with batch index n.

Exit codes: 0 finished or stopped (stop_at, --max-steps, STOP file, smoke PASS), 1 crash, 2 gate failed,
3 NaN storm, 4 refused to start (a configuration problem a restart cannot fix; run_forever.sh does not restart).
"""
import os

# MPS and thread settings (contract 8.3, 10-ml 1.1). They must be set before torch is imported, and the spawn data
# workers inherit them. setdefault keeps any value the shell already exported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse          # noqa: E402
import collections       # noqa: E402
import copy              # noqa: E402
import csv               # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import re                # noqa: E402
import shutil            # noqa: E402
import subprocess        # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402
from importlib import metadata   # noqa: E402
from pathlib import Path         # noqa: E402
from types import SimpleNamespace   # noqa: E402

import numpy as np       # noqa: E402
import torch             # noqa: E402
import torch.nn as nn    # noqa: E402

from ml.earshot import data, metrics   # noqa: E402
from ml.earshot.config import DATA, ROOT, RUNS, SR, load_yaml   # noqa: E402
from ml.earshot.losses import TERMS, AAMHead, total_loss   # noqa: E402
from ml.earshot.model import CondInputs, ESB16DF, init_train   # noqa: E402
from ml.earshot.sources import get_sources   # noqa: E402

EXIT_OK, EXIT_CRASH, EXIT_GATE, EXIT_NAN, EXIT_REFUSED = 0, 1, 2, 3, 4
ENV_KEYS = ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO",
            "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")
LOG_EVERY = 50                 # one train.csv row per 50 steps (10-ml 10.3)
SMOKE_CHECK_EVERY = 100        # smoke gate: SI-SNRi on the 16 fixed examples every 100 steps (10-ml 10.5)
SMOKE_PASS_DB = 20.0           # smoke PASS: median SI-SNRi >= 20 dB
SMOKE_LE = 6 * SR              # smoke examples use 6 s target enrollments
PILOT_STEPS = (2000, 4000, 6000)   # pilot gate: val SI-SNRi median at 6,000 must beat both 2,000 and 4,000
PILOT_TIME_FROM = 500          # pilot gate, information only: median step time over steps 500-6,000 ...
MAIN_BUDGET_S = 40 * 3600      # ... turned into the main-run step count that fits 40 h ...
VAL_CKPT_FACTOR = 1.05         # ... with 5% added for validation and checkpoints (contract 8.4)
MAX_STEP_S = 1.25              # main run, contract 8.4 rule - if the first hour averages above this, the lead cuts
                               # stage B to main_steps_for_40h (the model is never shrunk)
BEST_MIN_DB = 20.0             # best checkpoint: absent attenuation and own-voice suppression both >= 20 dB
NAN_LIMIT, NAN_WINDOW = 10, 1000   # 10 non-finite steps within 1,000 steps stop the run (exit 3)
BATTERY_CHECK_S = 600          # `pmset -g batt` every 10 minutes
FIRST_HOUR_S = 3600            # step-time warning after the first hour
CKPT_RE = re.compile(r"^step_(\d+)\.pt$")
TRAIN_COLUMNS = ["step", "wall_s", "lr", "loss", *TERMS, "grad_norm", "step_s_median", "data_wait_s", "mps_gb",
                 "nan_skips", "short_enroll"]
VAL_COLUMNS = ["step", "weights", "sisnri_median", "sisnri_mean", "picked_sir0", "absent_db_median",
               "absent_under10_pct", "own_db_median", "tsos_pct", "eligible", "best_rank"]
VAL_PATH = ROOT / "ml" / "cache" / "val_v1.npz"      # written by `python -m ml.tools.render_val` (10-ml 11)
VAL_SUBSETS = ("A", "B", "C", "D", "E")             # subset codes 0..4 in val_v1.npz (render_val.SUBSETS order)


class Refused(Exception):
    """A start-up problem that a restart cannot fix (exit code 4)."""


# ---------------------------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------------------------
DEFAULTS = {
    "run": None, "seed": 1, "steps": 120000, "stop_at": None,
    "lr": {"peak": 5.0e-4, "final": 2.5e-5, "warmup": 2000, "decay_end": None},
    "optim": {"betas": [0.9, 0.999], "weight_decay": 1.0e-4, "clip": 5.0},
    "ema": None, "val_every": None, "ckpt_every": 1000, "keep_last": 3, "keep_best": 2, "workers": 6,
    "allow_missing_decim": False, "stages": [], "gate": None, "fixed_examples": None, "init": None,
}
MAIN_LR = {"peak": 5.0e-4, "final": 2.5e-5, "warmup": 2000, "decay_end": 120000}   # DECISIONS L 2026-11-08


def resolve_config(cfg):
    """Fill in defaults and check the config. Returns a new dict (plain Python types, safe to save in a checkpoint)."""
    out = copy.deepcopy(DEFAULTS)
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = dict(out[key], **value)
        else:
            out[key] = copy.deepcopy(value)
    if not out["run"]:
        raise Refused("the config has no `run` name")
    if out["lr"]["decay_end"] is None:
        out["lr"]["decay_end"] = out["steps"]
    if out["ema"] is not None:                         # `ema: null` turns the EMA off (smoke)
        out["ema"] = dict({"decay": 0.999, "start": 2000}, **out["ema"])
    if not out["stages"]:
        raise Refused("the config has no stages")
    out["stages"] = sorted(out["stages"], key=lambda st: st.get("start", 0))
    for st in out["stages"]:
        st.setdefault("start", 0)
        st.setdefault("oversup", False)
        if st["recipe"] not in data.RECIPES:
            raise Refused(f"unknown recipe {st['recipe']!r} (known: {', '.join(data.RECIPES)})")
        if int(st["batch"]) > data.EXAMPLE_STRIDE:
            raise Refused(f"stage batch {st['batch']} is larger than {data.EXAMPLE_STRIDE}")
    if out["stages"][0]["start"] != 0:
        raise Refused("the first stage must start at step 0")
    if out["gate"] not in (None, "smoke", "pilot"):
        raise Refused(f"unknown gate {out['gate']!r}")
    if out["gate"] == "pilot" and not out["stop_at"]:
        raise Refused("the pilot gate needs stop_at")
    return out


def lr_at(step, lr=None):
    """Learning rate of the update that completes step `step` (10-ml 10.1).

    Linear warm-up from 0 to the peak over `warmup` steps, then a cosine from the peak down to `final` at
    `decay_end`, flat after that. Main run: peak 5e-4 at 2,000, 2.5e-5 at 120,000. A pure function of the step,
    so nothing about the schedule needs saving in a checkpoint.
    """
    lr = lr or MAIN_LR
    peak, final, warmup, end = lr["peak"], lr["final"], lr["warmup"], lr["decay_end"]
    if step < warmup:
        return peak * step / warmup
    p = (step - warmup) / max(end - warmup, 1)
    p = min(max(p, 0.0), 1.0)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * p))


def stage_at(stages, step):
    """(stage dict, step where it ends or None) for the batch with index `step`."""
    current, end = stages[0], None
    for i, st in enumerate(stages):
        if st["start"] <= step:
            current = st
            end = stages[i + 1]["start"] if i + 1 < len(stages) else None
    return current, end


# ---------------------------------------------------------------------------------------------------------------
# Optimizer and EMA (10-ml 10.1)
# ---------------------------------------------------------------------------------------------------------------
def param_groups(model, aam, weight_decay):
    """Two AdamW groups: weight decay on Linear and LSTM weight matrices and the AAM weight; none on LayerNorms,
    biases, att.type_emb, att.null_tok and null.* (contract 8.3)."""
    decay, no_decay = [], []
    for module in list(model.modules()) + list(aam.modules()):
        for name, p in module.named_parameters(recurse=False):
            is_matrix = ((isinstance(module, nn.Linear) and name == "weight")
                         or (isinstance(module, nn.LSTM) and name.startswith("weight_"))
                         or (isinstance(module, AAMHead) and name == "weight"))
            (decay if is_matrix else no_decay).append(p)
    n_all = sum(1 for _ in model.parameters()) + sum(1 for _ in aam.parameters())
    assert len(decay) + len(no_decay) == n_all, "every parameter must be in exactly one group"
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model, aam, cfg):
    """AdamW with fused=True, or the plain version if the device refuses it. -> (optimizer, fused flag)."""
    groups = param_groups(model, aam, cfg["optim"]["weight_decay"])
    kwargs = {"lr": 0.0, "betas": tuple(cfg["optim"]["betas"]), "eps": 1e-8}
    try:
        return torch.optim.AdamW(groups, fused=True, **kwargs), True
    except (RuntimeError, ValueError):
        return torch.optim.AdamW(param_groups(model, aam, cfg["optim"]["weight_decay"]), **kwargs), False


class Ema:
    """Exponential moving average of the model weights (decay 0.999 from step 2,000).

    After the update that completes step `start` the EMA copies the weights; after every later update
    ema = ema + (1 - decay) (w - ema) = decay ema + (1 - decay) w, done for all tensors at once with _foreach_lerp_.
    The EMA is a full model copy, so validation and export can run it directly.
    """

    def __init__(self, decay=0.999, start=2000):
        self.decay = float(decay)
        self.start = int(start)
        self.model = None                  # None until step `start`

    def _copy(self, model):
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        for p in self.model.parameters():
            p.grad = None

    @torch.no_grad()
    def update(self, model, completed):
        """Call after every optimizer step with the number of completed steps."""
        if completed < self.start:
            return
        if self.model is None or completed == self.start:
            self._copy(model)
            return
        ema_params = list(self.model.parameters())
        live = [p.detach() for p in model.parameters()]
        torch._foreach_lerp_(ema_params, live, 1.0 - self.decay)

    def state_dict(self):
        return None if self.model is None else self.model.state_dict()

    def load(self, state, template):
        """Restore from a checkpoint's `ema` entry (None = not started) using `template` for the module layout."""
        if state is None:
            self.model = None
            return
        self._copy(template)
        self.model.load_state_dict(state)


# ---------------------------------------------------------------------------------------------------------------
# One step (10-ml 10.2)
# ---------------------------------------------------------------------------------------------------------------
def to_device(batch, device, keys=None):
    """numpy batch dict -> SimpleNamespace of torch tensors on `device` (float32 / int64 / bool as in the numpy
    arrays, see data.BATCH_KEYS). `info` (a list of dicts) is kept as it is."""
    keys = keys or list(data.BATCH_KEYS)
    out = {k: torch.from_numpy(np.ascontiguousarray(batch[k])).to(device) for k in keys}
    out["info"] = batch.get("info")
    return SimpleNamespace(**out)


def forward(model, wes, b, e_t=None):
    """Enrollments -> conditioning -> separation for one batch b (attributes as in data.BATCH_KEYS).

    e_t is computed with the frozen WeSpeaker from the target enrollment unless given (the validation set stores
    it). WeSpeaker and both token paths run for every example; the flags (has_t, has_o, use_wes, mode) decide in
    build_cond which results are used, so the code has no per-example branches.
    Returns (y [B, L], head_logits [B, T, 2], g_t [B, 128]).
    """
    if e_t is None:
        with torch.no_grad():
            e_t = wes(b.enr_t)                                         # [B, 256], frozen
    tok_t, g_t = model.encode_enrollment(b.enr_t, b.tok_idx_t, b.tok_val_t)
    tok_o, g_o = model.encode_enrollment(b.enr_o, b.tok_idx_o, b.tok_val_o)
    cond = model.build_cond(CondInputs(e_t, tok_t, b.tok_val_t, g_t, b.e_o, tok_o, b.tok_val_o, g_o,
                                       b.has_t, b.has_o, b.use_wes, b.mode))
    y, head_logits = model(b.mix, cond)
    return y, head_logits, g_t


def step_batch(model, wes, aam, opt, params, batch, step, lr_cfg, clip, oversup_on, ema=None):
    """Train on one batch: the batch with index `step` (= steps completed before this one).

    Returns a dict: loss and grad_norm (floats), ok (both finite, so the update was applied), lr, parts (the
    per-term batch means from total_loss, detached tensors still on the device) and rows (per term, how many
    examples it applied to, also on the device). A non-finite loss or gradient norm skips the update (no optimizer
    step, no EMA update); the caller counts it as a NaN skip.
    """
    device = next(model.parameters()).device
    b = to_device(batch, device)
    y, head_logits, g_t = forward(model, wes, b)
    rows = {}
    loss, parts = total_loss(y, head_logits, g_t, b, aam, oversup_on, rows)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, clip)
    loss_value, gn_value = torch.stack([loss.detach(), gn.detach()]).tolist()   # the one device sync per step
    ok = math.isfinite(loss_value) and math.isfinite(gn_value)
    lr = lr_at(step + 1, lr_cfg)
    if ok:
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        if ema is not None:
            ema.update(model, step + 1)
    return {"loss": loss_value, "grad_norm": gn_value, "ok": ok, "lr": lr, "parts": parts, "rows": rows}


# ---------------------------------------------------------------------------------------------------------------
# Validation on the 400 fixed mixtures (contract 8.4, 10-ml 11)
# ---------------------------------------------------------------------------------------------------------------
VAL_INPUT_KEYS = ("mix", "enr_t", "tok_idx_t", "tok_val_t", "enr_o", "tok_idx_o", "tok_val_o", "e_t", "e_o",
                  "has_t", "has_o", "use_wes", "mode")


def load_val(path=VAL_PATH):
    """The validation set as a dict of numpy arrays (info parsed from JSON). Refused if it was never rendered."""
    path = Path(path)
    if not path.exists():
        raise Refused(f"{path} is missing; render it with `.venv/bin/python -m ml.tools.render_val` (about 1 min)")
    with np.load(path) as z:
        val = {k: z[k] for k in z.files}
    val["info"] = json.loads(str(val["info"]))
    return val


@torch.no_grad()
def run_offline(model, val, device, rows=None, batch=8):
    """Offline forward of the validation rows (all by default) in batches of 8 -> y float32 [n, L] (numpy)."""
    rows = np.arange(len(val["mix"])) if rows is None else np.asarray(rows)
    out = []
    for lo in range(0, len(rows), batch):
        pick = rows[lo:lo + batch]
        b = SimpleNamespace(**{k: torch.from_numpy(np.ascontiguousarray(val[k][pick])).to(device)
                               for k in VAL_INPUT_KEYS})
        y, _, _ = forward(model, None, b, e_t=b.e_t)
        out.append(y.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, val["mix"].shape[1]), np.float32)


def val_metrics(y, val, rows=None):
    """The validation numbers of contract 8.4 from outputs y [n, L] for `rows` of the set (all by default).

    SI-SNRi median and mean over subsets A, B, D, E; picked rate at SIR 0 dB (subset B); absent attenuation
    (subset C, median and % under 10 dB); own-voice suppression (subset D, median); TSOS in percent (subset A).
    `eligible` = absent and own medians both >= 20 dB (the best-checkpoint rule). A subset with no rows gives NaN.
    """
    rows = np.arange(len(val["mix"])) if rows is None else np.asarray(rows)
    subset = val["subset"][rows]
    f64 = np.float64
    mix, ref = val["mix"][rows], val["ref"][rows]

    def pick(*names):
        return np.flatnonzero(np.isin(subset, [VAL_SUBSETS.index(n) for n in names]))

    out = {}
    sep = pick("A", "B", "D", "E")
    sisnri = metrics.si_snri(y[sep].astype(f64), ref[sep].astype(f64), mix[sep].astype(f64))
    out["sisnri_median"] = float(np.median(sisnri)) if len(sep) else float("nan")
    out["sisnri_mean"] = float(np.mean(sisnri)) if len(sep) else float("nan")
    for name in ("A", "B", "D", "E"):
        part = sisnri[subset[sep] == VAL_SUBSETS.index(name)]
        out[f"sisnri_{name}"] = float(np.median(part)) if len(part) else float("nan")
    counts = [metrics.picked(y[i], ref[i], val["talkers"][rows[i]].astype(f64), val["vad_t"][rows[i]],
                             val["vad_talkers"][rows[i]]) for i in pick("B")]
    out["picked_sir0"] = metrics.picked_rate(counts)
    absent = metrics.absent_summary([metrics.attenuation_db(mix[i], y[i]) for i in pick("C")])
    out["absent_db_median"] = absent["median"]
    out["absent_under10_pct"] = absent["under10_pct"]
    own = metrics.own_summary([metrics.own_suppression_db(val["own"][rows[i]].astype(f64), y[i],
                                                          val["vad_o"][rows[i]], val["vad_t"][rows[i]])
                               for i in pick("D")])
    out["own_db_median"] = own["median"]
    out["own_db_p05"] = own["own_db_p05"]
    out["own_n"] = own["n"]                        # D clips with >= 0.2 s of own-only audio (events.log)
    out["tsos_pct"] = 100.0 * metrics.tsos_rate([metrics.tsos(ref[i], y[i]) for i in pick("A")])
    out["eligible"] = bool(out["absent_db_median"] >= BEST_MIN_DB and out["own_db_median"] >= BEST_MIN_DB)
    return out


def validate(model, val, device, rows=None):
    """Offline forward on the device + metrics -> dict (see val_metrics), plus the seconds it took."""
    t0 = time.perf_counter()
    y = run_offline(model, val, device, rows)
    out = val_metrics(y, val, rows)
    out["seconds"] = time.perf_counter() - t0
    return out


def update_best(best, step, score, eligible, keep):
    """The best-checkpoint list after one validation: [(score, step)] sorted by score, highest first, at most
    `keep` entries. Only eligible validations enter (contract 8.4). -> (new list, rank of `step` or None)."""
    entries = [e for e in best if e[1] != step]
    if eligible and math.isfinite(score):
        entries.append((float(score), int(step)))
    entries = sorted(entries, key=lambda e: (-e[0], e[1]))[:keep]
    rank = next((i + 1 for i, e in enumerate(entries) if e[1] == step), None)
    return [(float(s), int(n)) for s, n in entries], rank


# ---------------------------------------------------------------------------------------------------------------
# Smoke run: 16 fixed examples (10-ml 10.5)
# ---------------------------------------------------------------------------------------------------------------
def smoke_batches(cfg):
    """Render the fixed examples once (recipe of the only stage, seed = config seed, Le 6 s) -> list of batches.
    Example i uses default_rng([seed, i]); batch k holds examples k*B .. k*B + B - 1. Training alternates batches."""
    stage = cfg["stages"][0]
    B, L = int(stage["batch"]), int(round(stage["segment_s"] * SR))
    n = int(cfg["fixed_examples"])
    if n % B:
        raise Refused(f"fixed_examples {n} is not a multiple of the batch size {B}")
    src = get_sources("train")
    examples = []
    for i in range(n):
        rng = np.random.default_rng([cfg["seed"], i])
        ex = data.make_example(src, stage["recipe"], rng, L, SMOKE_LE,
                               allow_missing_decim=cfg["allow_missing_decim"])
        ex["info"]["index"] = i
        examples.append(ex)
    return [data.stack_batch(examples[k:k + B]) for k in range(0, n, B)]


@torch.no_grad()
def smoke_sisnri(model, wes, batches, device):
    """SI-SNRi (dB) of every fixed example with the current weights -> numpy [n]."""
    values = []
    for batch in batches:
        b = to_device(batch, device)
        y, _, _ = forward(model, wes, b)
        y = y.float().cpu().numpy().astype(np.float64)               # MPS has no float64: convert on the CPU
        values.append(metrics.si_snri(y, batch["ref"].astype(np.float64),
                                      batch["mix"].astype(np.float64)))
    return np.concatenate(values)


# ---------------------------------------------------------------------------------------------------------------
# Checkpoints (10-ml 10.4)
# ---------------------------------------------------------------------------------------------------------------
def list_ckpts(run_dir):
    """[(step, path)] of the step_<n>.pt files in run_dir, oldest first."""
    found = []
    for path in Path(run_dir).glob("step_*.pt"):
        m = CKPT_RE.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def save_ckpt(path, state):
    """torch.save to <path>.tmp, then os.replace: a crash mid-write never leaves a broken step_<n>.pt."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_ckpt(path):
    """Load a checkpoint on the CPU (weights_only=True: plain tensors, numbers, strings, lists and dicts only)."""
    return torch.load(str(path), map_location="cpu", weights_only=True)


def make_state(model, ema, opt, aam, step, cfg, best, nan_skips, wall_s):
    """Everything a checkpoint holds (10-ml 10.4), plus the NaN-skip total and the training wall time so far."""
    return {"model": model.state_dict(), "ema": ema.state_dict() if ema is not None else None,
            "opt": opt.state_dict(), "aam": aam.state_dict(), "step": int(step), "seed": int(cfg["seed"]),
            "config": cfg, "recipe_version": data.RECIPE_VERSION, "best": [list(e) for e in best],
            "torch_version": str(torch.__version__), "nan_skips": int(nan_skips), "wall_s": float(wall_s)}


def prune(run_dir, keep_last):
    """Delete the step_<n>.pt files of this run except the newest `keep_last`. Never touches any other file
    (best_1.pt and best_2.pt are separate copies). Returns the deleted paths."""
    files = list_ckpts(run_dir)
    old = files[:-keep_last] if keep_last > 0 else files
    for _, path in old:
        path.unlink()
    return [p for _, p in old]


def read_best_json(run_dir):
    """{step: path} of the best_<rank>.pt copies on disk, from best.json (empty if none yet)."""
    path = Path(run_dir) / "best.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {int(e["step"]): Path(run_dir) / f"best_{e['rank']}.pt" for e in json.load(f)}


def refresh_best_files(run_dir, best):
    """Make best_<rank>.pt hold the checkpoint of best[rank - 1] (copy, then rename), then rewrite best.json.

    The source of each copy is step_<n>.pt if it still exists, else the best_<r>.pt copy that already holds step n.
    All copies go to .tmp files first and are renamed afterwards, so swapping best_1 and best_2 works.
    Returns warnings (a best entry whose checkpoint exists nowhere)."""
    run_dir = Path(run_dir)
    on_disk = read_best_json(run_dir)
    moves, warnings, records = [], [], []
    for rank, (score, step) in enumerate(best, start=1):
        dst = run_dir / f"best_{rank}.pt"
        records.append({"rank": rank, "step": int(step), "score": float(score)})
        if on_disk.get(step) == dst and dst.exists():
            continue                                      # already in place
        src = run_dir / f"step_{step}.pt"
        if not src.exists():
            src = on_disk.get(step)
        if src is None or not Path(src).exists():
            warnings.append(f"best entry step {step} has no checkpoint file to copy")
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        moves.append((tmp, dst))
    for tmp, dst in moves:
        os.replace(tmp, dst)
    tmp = run_dir / "best.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)
    os.replace(tmp, run_dir / "best.json")
    return warnings


def find_resume(run_dir, resume):
    """The checkpoint file --resume asks for: 'last' = the highest step_<n>.pt in run_dir (None if there is none),
    or a path."""
    if resume is None:
        return None
    if resume == "last":
        files = list_ckpts(run_dir)
        return files[-1][1] if files else None
    path = Path(resume)
    if not path.exists():
        raise Refused(f"--resume {resume}: no such file")
    return path


# ---------------------------------------------------------------------------------------------------------------
# Logs (10-ml 10.3): train.csv, val.csv, events.log, run.json
# ---------------------------------------------------------------------------------------------------------------
def fmt(value):
    """A CSV cell: floats with 6 significant digits, None as empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


class RunLog:
    """Plain-text logs in the run folder. Every write appends and flushes, so a crash loses nothing."""

    def __init__(self, run_dir, quiet=False):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet

    def event(self, text):
        """One timestamped line in events.log (also printed)."""
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
        with open(self.dir / "events.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if not self.quiet:
            print(line, flush=True)

    def row(self, name, columns, values):
        """Append one row to <name>.csv (header first if the file is new)."""
        path = self.dir / name
        new = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(columns)
            w.writerow([fmt(values.get(c)) for c in columns])

    def write_json(self, name, obj):
        """Write <name> atomically (tmp file, then rename)."""
        tmp = self.dir / (name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=str)
        os.replace(tmp, self.dir / name)


def read_csv(path):
    """Rows of a log CSV as dicts of floats (empty cells -> None). When a resumed run wrote a step twice, the last
    row of that step wins."""
    path = Path(path)
    if not path.exists():
        return []
    by_step = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v) if v not in ("", None) else None
                except ValueError:
                    row[k] = v
            by_step[int(row["step"])] = row
    return [by_step[s] for s in sorted(by_step)]


class Window:
    """Sums over the steps since the last train.csv row. The loss parts stay on the device (no sync per step) and
    are read with one .tolist() per row.

    Each loss term is averaged over the examples it applied to: the window adds (batch mean x rows it applied to)
    and the row counts, then logs sum / max(rows, 1). A sparse term such as absent (10% of recipe-B examples, so no
    absent row in 43% of 8-example batches) is then logged at its true mean instead of being pulled toward 0 by the
    batches where it did not apply. A term that applied to no row in the whole window logs 0.
    """

    def __init__(self):
        self.sums = None                        # per term: sum over steps of (batch mean x applying rows)
        self.rows = None                        # per term: applying rows summed over steps
        self.n_ok = 0
        self.loss = self.grad_norm = self.wait = 0.0
        self.step_times = []
        self.short = 0

    def add(self, out, step_s, wait_s, short):
        self.step_times.append(step_s)
        self.wait += wait_s
        self.short += short
        if out["ok"]:                           # a skipped (non-finite) step would turn every mean into NaN
            vec = torch.stack([out["parts"][n].float() for n in TERMS])
            if out.get("rows"):
                rows = torch.stack([out["rows"][n].float() for n in TERMS])
            else:                               # no counts given: every step weighs 1 (a plain mean over steps)
                rows = torch.ones_like(vec)
            self.sums = vec * rows if self.sums is None else self.sums + vec * rows
            self.rows = rows if self.rows is None else self.rows + rows
            self.n_ok += 1
            self.loss += out["loss"]
            self.grad_norm += out["grad_norm"]

    def means(self):
        n = max(self.n_ok, 1)
        if self.sums is not None:
            parts = (self.sums / self.rows.clamp(min=1)).tolist()
        else:
            parts = [0.0] * len(TERMS)
        out = {name: v for name, v in zip(TERMS, parts)}
        out.update(loss=self.loss / n, grad_norm=self.grad_norm / n,
                   step_s_median=float(np.median(self.step_times)) if self.step_times else None,
                   data_wait_s=self.wait / max(len(self.step_times), 1), short_enroll=self.short)
        return out


def mps_gb(device):
    """Memory the MPS driver holds for this process, in GB (0 on the CPU)."""
    if str(device).startswith("mps"):
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def on_battery():
    """True if `pmset -g batt` says the Mac runs on battery (False if pmset is missing or fails)."""
    try:
        text = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "Battery Power" in text


def run_info(cfg, device, fused, resumed_from):
    """The run.json record: config, versions, environment, data manifest sizes."""
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None

    def version(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    manifests = {}
    for path in sorted((DATA / "manifests").glob("*.jsonl")):
        with open(path, "rb") as f:
            manifests[path.name] = {"bytes": path.stat().st_size, "rows": sum(1 for _ in f)}
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg, "git_commit": commit,
            "torch": str(torch.__version__), "coremltools": version("coremltools"), "numpy": np.__version__,
            "python": sys.version.split()[0], "device": str(device), "fused_adamw": fused,
            "env": {k: os.environ.get(k) for k in ENV_KEYS}, "manifests": manifests,
            "recipe_version": data.RECIPE_VERSION, "resumed_from": resumed_from}


# ---------------------------------------------------------------------------------------------------------------
# Gates (10-ml 10.5): gate_<name>.json with PASS or FAIL and the numbers
# ---------------------------------------------------------------------------------------------------------------
def smoke_decision(step, median_db, max_steps, threshold=SMOKE_PASS_DB):
    """Smoke gate after a check at `step`: "PASS" as soon as the median SI-SNRi on the 16 examples reaches 20 dB,
    "FAIL" if it has not by the last step, else None (keep training)."""
    if median_db is not None and math.isfinite(median_db) and median_db >= threshold:
        return "PASS"
    if step >= max_steps:
        return "FAIL"
    return None


def main_steps_for_40h(step_s):
    """Main-run steps that fit 40 h at `step_s` seconds per training step, with 5% added for validation and
    checkpoints, rounded down to a multiple of 1,000: floor(40 x 3600 / (step_s x 1.05) / 1000) x 1000.
    Example - the main run's first hour, 1.101 s, gives 124,000; the 1.25 s rule line gives 109,000. None if
    step_s is missing or not a positive number."""
    if step_s is None or not math.isfinite(step_s) or step_s <= 0:
        return None
    return int(math.floor(MAIN_BUDGET_S / (step_s * VAL_CKPT_FACTOR) / 1000.0)) * 1000


def pilot_gate(val_rows, nan_skips, train_rows):
    """Pilot gate from val.csv and train.csv rows (as read_csv returns them) and the NaN-skip total.

    PASS needs both checks (DECISIONS L 2026-11-08):
      val_6k_above_2k_and_4k - the val SI-SNRi median at 6k is higher than at 4k and higher than at 2k, so the
                               model is still improving at the end of the pilot (2k vs 4k is not compared);
      no_nan_skips           - nan_skips == 0.
    There is no absolute dB target: the research phase never measured a learning curve, so the old "2 dB at 6k"
    had no basis. The step time is information, never a check: step_s_median_500_6000 is the median of the
    train.csv rows' 50-step medians with 500 < step <= 6,000, and main_steps_for_40h is the main-run step count it
    allows (main_steps_for_40h above). A missing or non-finite validation fails the first check.
    PASS means only "still rising at 6k", not "learns well": the spec-loss pilot v1 (-0.86 / -1.14 / -0.24 dB at
    2k / 4k / 6k, all negative) would pass it too.
    Returns the gate dict (result, checks, numbers).
    """
    sis = {}
    for r in val_rows:
        if int(r["step"]) in PILOT_STEPS and r.get("sisnri_median") is not None:
            sis[int(r["step"])] = float(r["sisnri_median"])
    values = [sis.get(s) for s in PILOT_STEPS]
    have_all = all(v is not None and math.isfinite(v) for v in values)
    rising = have_all and values[2] > values[1] and values[2] > values[0]
    times = [float(r["step_s_median"]) for r in train_rows
             if PILOT_TIME_FROM < int(r["step"]) <= PILOT_STEPS[-1] and r.get("step_s_median") is not None]
    step_s = float(np.median(times)) if times else None
    checks = {"val_6k_above_2k_and_4k": rising, "no_nan_skips": nan_skips == 0}
    return {"gate": "pilot", "result": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "sisnri_median": {str(s): sis.get(s) for s in PILOT_STEPS}, "nan_skips": int(nan_skips),
            "step_s_median_500_6000": step_s, "step_rows": len(times),
            "main_steps_for_40h": main_steps_for_40h(step_s)}


def write_gate(log, result):
    """gate_<name>.json in the run folder, plus an events.log line."""
    log.write_json(f"gate_{result['gate']}.json", result)
    log.event(f"gate {result['gate']}: {result['result']} {json.dumps(result.get('checks', {}))}")


# ---------------------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------------------
class Trainer:
    """One invocation of `python -m ml.train`: set up (fresh, resumed or fine-tune init), loop, stop.

    run() returns the exit code. Tests build a Trainer with a small config on the CPU and pass `history`, a list
    that receives one record per step: {"step": batch index, "indices": example indices, "loss", "ok", "step_s"
    (seconds for the whole step), "data_s" (of which waiting for the batch)}.
    """

    def __init__(self, cfg, run_dir=None, resume=None, device=None, workers=None, max_steps=None,
                 force_recipe=False, history=None, quiet=False):
        self.cfg = resolve_config(cfg)
        self.run_dir = Path(run_dir) if run_dir else RUNS / self.cfg["run"]
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.workers = self.cfg["workers"] if workers is None else int(workers)
        self.max_steps = max_steps
        self.resume = resume
        self.force_recipe = force_recipe
        self.history = history
        self.log = RunLog(self.run_dir, quiet)

    # ----- set-up -----------------------------------------------------------------------------------------------
    def setup(self):
        """Model, AAM head, optimizer, EMA, WeSpeaker; then resume, fine-tune init or a fresh start."""
        from ml.earshot.wespeaker import WeSpeaker         # imported here: the data workers never need it
        cfg = self.cfg
        torch.manual_seed(cfg["seed"])                     # model init only; the loop uses no torch randomness
        self.model = init_train(ESB16DF()).to(self.device)
        self.aam = AAMHead().to(self.device)
        self.opt, self.fused = build_optimizer(self.model, self.aam, cfg)
        self.params = list(self.model.parameters()) + list(self.aam.parameters())
        self.ema = Ema(cfg["ema"]["decay"], cfg["ema"]["start"]) if cfg["ema"] else None
        self.wes = WeSpeaker.load().to(self.device)
        self.step, self.best, self.nan_skips, self.wall_offset = 0, [], 0, 0.0

        ckpt = find_resume(self.run_dir, self.resume)
        if ckpt is not None:
            ck = load_ckpt(ckpt)
            if ck["recipe_version"] != data.RECIPE_VERSION and not self.force_recipe:
                raise Refused(f"{ckpt} has recipe_version {ck['recipe_version']}, the code has "
                              f"{data.RECIPE_VERSION}; pass --force-recipe to continue anyway")
            self.model.load_state_dict(ck["model"])
            self.aam.load_state_dict(ck["aam"])
            self.opt.load_state_dict(ck["opt"])            # moves the Adam moments to the parameters' device
            if self.ema is not None:
                self.ema.load(ck["ema"], self.model)
            self.step = int(ck["step"])
            self.best = [(float(s), int(n)) for s, n in ck["best"]]
            self.nan_skips = int(ck.get("nan_skips", 0))
            self.wall_offset = float(ck.get("wall_s", 0.0))
            if int(ck["seed"]) != cfg["seed"]:
                self.log.event(f"warning: config seed {cfg['seed']} differs from the checkpoint's {ck['seed']}; "
                               "using the checkpoint's so the data order stays the same")
                cfg["seed"] = int(ck["seed"])
            self.log.event(f"resume from {ckpt} at step {self.step}")
            name = f"run_resume_{self.step}.json"
        else:
            if self.resume is not None:
                self.log.event(f"resume {self.resume}: no checkpoint in {self.run_dir}, starting fresh")
            elif list_ckpts(self.run_dir):
                raise Refused(f"{self.run_dir} already holds checkpoints; continue with --resume last, or pick "
                              "a new name with --run")
            self.set_aside_old_logs()
            if cfg["init"]:
                init = Path(cfg["init"]) if Path(cfg["init"]).is_absolute() else ROOT / cfg["init"]
                if not init.exists():
                    raise Refused(f"init checkpoint {init} does not exist")
                ck = load_ckpt(init)
                self.model.load_state_dict(ck["ema"] if ck["ema"] is not None else ck["model"])
                self.aam.load_state_dict(ck["aam"])
                self.log.event(f"init from {init} ({'EMA' if ck['ema'] is not None else 'raw'} weights of step "
                               f"{ck['step']}), fresh optimizer")
            name = "run.json"
        self.log.write_json(name, run_info(cfg, self.device, self.fused, str(ckpt) if ckpt else None))

        self.end = cfg["steps"] if not cfg["stop_at"] else min(cfg["steps"], int(cfg["stop_at"]))
        self.forced_end = None
        if self.max_steps is not None and self.step + self.max_steps < self.end:
            self.end = self.step + int(self.max_steps)
            self.forced_end = self.end                     # --max-steps: checkpoint + one evaluation at the end
        self.val = load_val() if cfg["val_every"] else None
        self.fixed = smoke_batches(cfg) if cfg["fixed_examples"] else None
        self.log.event(f"start: run {cfg['run']}, device {self.device}, fused AdamW {self.fused}, steps "
                       f"{self.step} -> {self.end}, workers {self.workers}, seed {cfg['seed']}")

    def set_aside_old_logs(self):
        """A fresh start in a folder with logs but no checkpoint renames the old CSVs (never deletes them)."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in ("train.csv", "val.csv", "smoke.csv"):
            path = self.run_dir / name
            if path.exists():
                path.rename(self.run_dir / f"{name}.{stamp}.old")
                self.log.event(f"fresh start: kept the old {name} as {name}.{stamp}.old")

    # ----- the loop ---------------------------------------------------------------------------------------------
    def run(self):
        """Train until the end step, a gate decision, a STOP file or a NaN storm. Returns the exit code."""
        if (self.run_dir / "STOP").exists():
            self.log.event(f"STOP file present in {self.run_dir}; not starting (delete it to continue)")
            return EXIT_OK
        self.setup()
        if self.step >= self.end:
            self.log.event(f"nothing to do: step {self.step} is already at the end ({self.end})")
            return self.gate_at_end()
        self.window = Window()
        self.nan_steps = collections.deque()
        self.t0 = time.time()
        self.last_battery = -BATTERY_CHECK_S
        self.hour_sum, self.hour_n, self.hour_done = 0.0, 0, False
        self.last_short_warning = -NAN_WINDOW
        self.saved_at = None                               # step of the last checkpoint written by this process
        self.best_changed = False                          # set by validation(), used by checkpoint()
        loader = it = None
        try:
            while self.step < self.end:
                stage, stage_end = stage_at(self.cfg["stages"], self.step)
                stop = self.end if stage_end is None else min(self.end, stage_end)
                B, L = int(stage["batch"]), int(round(stage["segment_s"] * SR))
                if self.fixed is None:
                    loader = data.make_loader("train", stage["recipe"], self.cfg["seed"], self.step, stop, B, L,
                                              self.workers, allow_missing_decim=self.cfg["allow_missing_decim"])
                    it = iter(loader)
                self.log.event(f"stage {stage['recipe']}: steps {self.step} -> {stop}, batch {B} x "
                               f"{stage['segment_s']} s, oversup {stage['oversup']}")
                while self.step < stop:
                    t_begin = time.perf_counter()
                    batch = self.fixed[self.step % len(self.fixed)] if self.fixed is not None else next(it)
                    t_data = time.perf_counter()
                    out = step_batch(self.model, self.wes, self.aam, self.opt, self.params, batch, self.step,
                                     self.cfg["lr"], self.cfg["optim"]["clip"], stage["oversup"], self.ema)
                    step_s = time.perf_counter() - t_begin
                    if self.history is not None:
                        self.history.append({"step": self.step, "loss": out["loss"], "ok": out["ok"],
                                             "step_s": step_s, "data_s": t_data - t_begin,
                                             "indices": [int(i["index"]) for i in batch["info"]]})
                    self.step += 1
                    short = sum(1 for i in batch["info"] if i.get("short_t") or i.get("short_o"))
                    self.window.add(out, step_s, t_data - t_begin, short)
                    self.time_checks(step_s)
                    code = self.after_step(out)
                    if code is not None:
                        return code
                it = loader = None                         # drop the old loader (its workers exit)
            return EXIT_OK
        finally:
            it = loader = None

    def gate_at_end(self):
        """Exit code of a run that starts already at its end step: a re-run after it finished, or a restart after
        a crash between the last checkpoint and the gate file. A gate run returns its gate's result, so exit 0 never
        hides a failed or missing gate: gate_<name>.json if it exists, else the gate evaluated now (pilot - from
        val.csv and train.csv; smoke - one more check of the 16 fixed examples with the checkpoint's weights)."""
        gate = self.cfg["gate"]
        if gate is None:
            return EXIT_OK
        path = self.run_dir / f"gate_{gate}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            self.log.event(f"gate {gate} was already decided: {result['result']} ({path.name})")
        elif gate == "pilot":
            result = pilot_gate(read_csv(self.run_dir / "val.csv"), self.nan_skips,
                                read_csv(self.run_dir / "train.csv"))
            write_gate(self.log, result)
        else:                                              # smoke: the check itself writes the gate file
            self.t0, self.saved_at, self.best_changed = time.time(), self.step, False   # the checkpoint exists
            code = self.smoke_check()
            return EXIT_OK if code is None else code
        return EXIT_OK if result["result"] == "PASS" else EXIT_GATE

    # ----- after every step -------------------------------------------------------------------------------------
    def after_step(self, out):
        """Logs, NaN rule, validation, checkpoints, gates and the STOP file. Returns an exit code to stop, else None."""
        step, cfg = self.step, self.cfg
        at_end = step >= self.end
        forced = self.forced_end is not None and step == self.forced_end
        if not out["ok"]:
            self.nan_skips += 1
            self.nan_steps.append(step)
            while self.nan_steps and self.nan_steps[0] <= step - NAN_WINDOW:
                self.nan_steps.popleft()
            self.log.event(f"warning: non-finite loss ({out['loss']}) or grad norm ({out['grad_norm']}) at step "
                           f"{step}; update skipped ({len(self.nan_steps)} in the last {NAN_WINDOW} steps)")
            if len(self.nan_steps) >= NAN_LIMIT:
                self.log_row()
                self.log.event(f"NaN storm: {NAN_LIMIT} non-finite steps within {NAN_WINDOW}; stopping (exit 3)")
                return EXIT_NAN
        if step % LOG_EVERY == 0 or at_end:
            self.log_row()
        stop_file = step % LOG_EVERY == 0 and (self.run_dir / "STOP").exists()

        do_val = self.val is not None and (step % cfg["val_every"] == 0 or forced)
        if do_val:
            self.validation()
        if step % cfg["ckpt_every"] == 0 or do_val or at_end or stop_file:
            self.checkpoint()

        if cfg["gate"] == "smoke" and (step % SMOKE_CHECK_EVERY == 0 or at_end):
            code = self.smoke_check()
            if code is not None:
                return code
        if cfg["gate"] == "pilot" and step == int(cfg["stop_at"]):
            result = pilot_gate(read_csv(self.run_dir / "val.csv"), self.nan_skips,
                                read_csv(self.run_dir / "train.csv"))
            write_gate(self.log, result)
            return EXIT_OK if result["result"] == "PASS" else EXIT_GATE
        if stop_file:
            self.log.event(f"STOP file found at step {step}; checkpoint saved, stopping")
            return EXIT_OK
        if at_end:
            why = "--max-steps" if forced else ("stop_at" if step < cfg["steps"] else "last step")
            self.log.event(f"finished at step {step} ({why})")
            return EXIT_OK
        return None

    def log_row(self):
        """One train.csv row with the means since the last row (one device sync)."""
        means = self.window.means()
        row = dict(means, step=self.step, wall_s=self.wall_offset + time.time() - self.t0,
                   lr=lr_at(self.step, self.cfg["lr"]), mps_gb=mps_gb(self.device), nan_skips=self.nan_skips)
        self.log.row("train.csv", TRAIN_COLUMNS, row)
        if means["short_enroll"] and self.step - self.last_short_warning >= NAN_WINDOW:
            self.log.event(f"warning: {means['short_enroll']} examples with short enrollments (token fallback) "
                           f"in the {LOG_EVERY} steps before {self.step}")
            self.last_short_warning = self.step
        self.window = Window()

    def time_checks(self, step_s):
        """Battery check every 10 minutes; mean step time after the first hour of this process."""
        elapsed = time.time() - self.t0
        if elapsed - self.last_battery >= BATTERY_CHECK_S:
            self.last_battery = elapsed
            if on_battery():
                self.log.event("warning: running on battery power (pmset -g batt); training wants AC power")
        if not self.hour_done:
            self.hour_sum += step_s
            self.hour_n += 1
            if elapsed >= FIRST_HOUR_S:
                self.hour_done = True
                mean = self.hour_sum / self.hour_n
                self.log.event(f"first hour: mean step time {mean:.3f} s over {self.hour_n} steps")
                if mean > MAX_STEP_S:
                    fit = main_steps_for_40h(mean)
                    self.log.event(f"warning: mean step time {mean:.3f} s is above {MAX_STEP_S} s; at this speed "
                                   f"40 h fits {fit} steps (config {self.cfg['steps']}); contract 8.4 - the lead "
                                   "cuts stage B in main.yaml to that total (the model is never shrunk)")

    def validation(self):
        """Validate (EMA weights once they exist), write val.csv, update the best list."""
        use_ema = self.ema is not None and self.ema.model is not None
        net = self.ema.model if use_ema else self.model
        m = validate(net, self.val, self.device)
        old = list(self.best)
        self.best, rank = update_best(self.best, self.step, m["sisnri_median"], m["eligible"],
                                      self.cfg["keep_best"])
        self.best_changed = self.best != old
        row = dict(m, step=self.step, weights="ema" if use_ema else "raw", best_rank=rank)
        self.log.row("val.csv", VAL_COLUMNS, row)
        self.log.event(f"validation at step {self.step} ({row['weights']}, {m['seconds']:.1f} s): SI-SNRi median "
                       f"{m['sisnri_median']:.2f} dB (A {m['sisnri_A']:.2f}, B {m['sisnri_B']:.2f}, "
                       f"D {m['sisnri_D']:.2f}, E {m['sisnri_E']:.2f}), picked {m['picked_sir0']:.3f}, absent "
                       f"{m['absent_db_median']:.1f} dB, own {m['own_db_median']:.1f} dB (n {m['own_n']}), TSOS "
                       f"{m['tsos_pct']:.1f}%, eligible {m['eligible']}")
        if rank is not None:
            self.log.event(f"new best: step {self.step} is rank {rank} (SI-SNRi {m['sisnri_median']:.2f} dB)")
        return m

    def checkpoint(self):
        """Save step_<n>.pt, refresh best_1/best_2 if the best list changed, prune old step files."""
        if self.saved_at == self.step:
            return
        path = self.run_dir / f"step_{self.step}.pt"
        wall = self.wall_offset + time.time() - self.t0
        save_ckpt(path, make_state(self.model, self.ema, self.opt, self.aam, self.step, self.cfg, self.best,
                                   self.nan_skips, wall))
        self.saved_at = self.step
        self.log.event(f"checkpoint {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        if self.best_changed:
            for warning in refresh_best_files(self.run_dir, self.best):
                self.log.event(f"warning: {warning}")
            self.best_changed = False
        for old in prune(self.run_dir, self.cfg["keep_last"]):
            self.log.event(f"pruned {old.name}")

    def smoke_check(self):
        """SI-SNRi on the 16 fixed examples; PASS / FAIL gate file. Returns an exit code to stop, else None."""
        values = smoke_sisnri(self.model, self.wes, self.fixed, self.device)
        med = float(np.median(values))
        self.log.row("smoke.csv", ["step", "sisnri_median", "sisnri_mean", "sisnri_min"],
                     {"step": self.step, "sisnri_median": med, "sisnri_mean": float(np.mean(values)),
                      "sisnri_min": float(np.min(values))})
        self.log.event(f"smoke check at step {self.step}: SI-SNRi median {med:.2f} dB on {len(values)} examples")
        decision = smoke_decision(self.step, med, self.cfg["steps"])
        if decision is None:
            return None
        self.checkpoint()
        write_gate(self.log, {"gate": "smoke", "result": decision, "step": self.step, "sisnri_median": med,
                              "sisnri_per_example": [round(float(v), 3) for v in values],
                              "threshold_db": SMOKE_PASS_DB, "max_steps": self.cfg["steps"],
                              "checks": {"sisnri_median_ge_20db": med >= SMOKE_PASS_DB},
                              "pytest": "run separately: .venv/bin/python -m pytest ml/tests -q"})
        return EXIT_OK if decision == "PASS" else EXIT_GATE


# ---------------------------------------------------------------------------------------------------------------
# Command line (10-ml 10.7)
# ---------------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ESB-16-DF (10-ml 10). Exit codes: 0 done, 1 crash, "
                                            "2 gate failed, 3 NaN storm, 4 refused to start.")
    p.add_argument("--config", required=True, help="run config, e.g. ml/configs/main.yaml")
    p.add_argument("--resume", default=None, help="'last' (newest step_<n>.pt of the run, fresh if none) or a path")
    p.add_argument("--run", default=None, help="run name (folder ml/runs/<name>/), overrides the config's `run`")
    p.add_argument("--stop-at", type=int, default=None, help="stop after this many completed steps")
    p.add_argument("--workers", type=int, default=None, help="data workers (default: the config's, 6)")
    p.add_argument("--device", choices=["mps", "cpu"], default=None, help="default: mps if available")
    p.add_argument("--force-recipe", action="store_true", help="resume even if RECIPE_VERSION changed")
    p.add_argument("--max-steps", type=int, default=None,
                   help="sanity runs: stop after N steps of this process; a checkpoint and one evaluation "
                        "(validation, or the smoke check) run at that point")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log = None
    try:
        cfg = load_yaml(args.config)
        if args.run:
            cfg["run"] = args.run
        if args.stop_at is not None:
            cfg["stop_at"] = args.stop_at
        trainer = Trainer(cfg, resume=args.resume, device=args.device, workers=args.workers,
                          max_steps=args.max_steps, force_recipe=args.force_recipe)
        log = trainer.log
        code = trainer.run()
    except Refused as ex:
        print(f"ml.train refused to start: {ex}", file=sys.stderr, flush=True)
        if log is not None:
            log.event(f"refused: {ex}")
        code = EXIT_REFUSED
    except Exception:
        text = traceback.format_exc()
        print(text, file=sys.stderr, flush=True)
        if log is not None:
            log.event("crash:\n" + text.rstrip())
        code = EXIT_CRASH
    sys.exit(code)


if __name__ == "__main__":
    main()
