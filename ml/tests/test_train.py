"""Tests for the training loop ml/train.py (10-ml 10 and the test_train row of 10-ml 14). All on the CPU.

- the learning-rate schedule at 0, 2k, 60k and 120k steps, and the smoke schedule
- the EMA math (copy at the start step, then ema = 0.999 ema + 0.001 w)
- one optimizer step changes the parameters; a non-finite batch changes nothing (NaN skip)
- a tiny run (B = 2, 1 s, workers 0, recipe B so Everyone, no-own and absent examples can occur): 3 steps with a
  finite loss, and a run resumed from its step-2 checkpoint gives the same step-3 batch indices, the same loss and
  the same parameters as the uninterrupted run
- checkpoint pruning and the best_1 / best_2 copies, the best-list rule, refusals
- train.csv averages each loss term over the examples it applied to; a gate run restarted at its end step still
  returns its gate's exit code; run_forever.sh's restart rules (with a fake interpreter)
- the validation code on a few rows of the real validation set (skipped if it was not rendered)

The data tests read the real data pack and the own-voice cache, and skip without them.
"""
import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml import train
from ml.earshot import data, metrics, sources
from ml.earshot.losses import AAMHead
from ml.earshot.model import ESB16DF, init_train

needs_data = pytest.mark.skipif(not (sources.MANIFESTS / "speech_train.jsonl").exists()
                                or not sources.OWN_CACHE_PATH.exists(),
                                reason="data pack or ml/cache/own_v1.npz missing")


def tiny_cfg(**over):
    """A 3-step CPU run: batch 2, 1 s segments, recipe B (all example kinds), checkpoint every 2 steps."""
    cfg = {"run": "tiny", "seed": 5, "steps": 3,
           "lr": {"peak": 1.0e-3, "final": 1.0e-4, "warmup": 1, "decay_end": 3},
           "ema": None, "val_every": None, "ckpt_every": 2, "keep_last": 3, "keep_best": 2, "workers": 0,
           "allow_missing_decim": True,
           "stages": [{"start": 0, "recipe": "B", "batch": 2, "segment_s": 1.0, "oversup": True}]}
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------------------------------------------
# Schedule and EMA
# ---------------------------------------------------------------------------------------------------------------
def test_lr_schedule_values():
    assert train.lr_at(0) == 0.0                                    # warm-up starts at 0
    assert math.isclose(train.lr_at(1000), 2.5e-4, rel_tol=1e-12)   # half way up
    assert math.isclose(train.lr_at(2000), 5.0e-4, rel_tol=1e-12)   # peak at the end of the warm-up
    p = (60000 - 2000) / (120000 - 2000)                            # cosine written out independently
    expected = 2.5e-5 + 0.5 * (5.0e-4 - 2.5e-5) * (1 + math.cos(math.pi * p))
    assert math.isclose(train.lr_at(60000), expected, rel_tol=1e-12)
    assert abs(train.lr_at(60000) - 2.6882e-4) < 1e-8               # hand value - 2.5e-5 + 2.375e-4 * 1.026621
    assert math.isclose(train.lr_at(120000), 2.5e-5, rel_tol=1e-12)
    assert math.isclose(train.lr_at(150000), 2.5e-5, rel_tol=1e-12)  # flat after the end (stage C continues it)
    smoke = {"peak": 5.0e-4, "final": 5.0e-4, "warmup": 100, "decay_end": 1500}
    assert math.isclose(train.lr_at(50, smoke), 2.5e-4, rel_tol=1e-12)
    assert all(math.isclose(train.lr_at(s, smoke), 5.0e-4, rel_tol=1e-12) for s in (100, 700, 1500))


def test_lr_schedule_of_the_configs():
    main = train.resolve_config(train.load_yaml(train.ROOT / "ml/configs/main.yaml"))
    assert main["lr"] == train.MAIN_LR
    # 120,000 steps (DECISIONS L 2026-11-08, first hour of main 1.101 s x 1.05 -> 38.5 h); the cosine ends at the
    # last step.
    assert main["steps"] == 120000 == main["lr"]["decay_end"] == train.DEFAULTS["steps"]
    pilot = train.resolve_config(train.load_yaml(train.ROOT / "ml/configs/pilot.yaml"))
    assert pilot["lr"] == main["lr"] and pilot["steps"] == main["steps"] and pilot["stages"] == main["stages"]
    ft = train.resolve_config(train.load_yaml(train.ROOT / "ml/configs/finetune.yaml"))
    assert math.isclose(train.lr_at(500, ft["lr"]), 1.0e-4) and math.isclose(train.lr_at(10000, ft["lr"]), 1.0e-5)


def test_ema_math():
    torch.manual_seed(0)
    net = nn.Linear(3, 2)
    ema = train.Ema(decay=0.999, start=2)
    ema.update(net, 1)
    assert ema.model is None                                        # nothing before the start step
    ema.update(net, 2)
    assert all(torch.equal(a, b) for a, b in zip(ema.model.parameters(), net.parameters()))   # exact copy
    old = [p.detach().clone() for p in ema.model.parameters()]
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update(net, 3)
    for e, o, w in zip(ema.model.parameters(), old, net.parameters()):
        assert torch.allclose(e, 0.999 * o + 0.001 * w, atol=1e-7)
        assert not e.requires_grad
    state = ema.state_dict()
    again = train.Ema(0.999, 2)
    again.load(state, net)
    assert all(torch.equal(a, b) for a, b in zip(again.model.parameters(), ema.model.parameters()))


def fake_step(absent, n_absent, B=8):
    """A step_batch result where every term is 1.0 on all B rows, except absent (value, applying rows)."""
    parts = {n: torch.tensor(1.0) for n in train.TERMS}
    rows = {n: torch.tensor(B) for n in train.TERMS}
    parts["absent"], rows["absent"] = torch.tensor(float(absent)), torch.tensor(n_absent)
    return {"ok": True, "loss": 1.0, "grad_norm": 1.0, "parts": parts, "rows": rows}


def test_log_window_averages_each_term_over_the_rows_it_applied_to():
    # Three batches without an absent row (total_loss reports 0 for them), then 1 and 2 absent rows.
    w = train.Window()
    for out in [fake_step(0, 0), fake_step(0, 0), fake_step(0, 0), fake_step(9.0, 1), fake_step(12.0, 2)]:
        w.add(out, 1.0, 0.0, 0)
    m = w.means()
    assert m["absent"] == pytest.approx((9.0 + 2 * 12.0) / 3)    # 11 dB, not the step mean 33 / 5 = 6.6
    assert m["tsnr"] == pytest.approx(1.0) and m["loss"] == pytest.approx(1.0)
    empty = train.Window()
    empty.add(fake_step(0, 0), 1.0, 0.0, 0)
    assert empty.means()["absent"] == 0.0                         # applied to no row in the window: 0
    plain = train.Window()                                        # no row counts: a plain mean over steps
    for v in (2.0, 4.0):
        out = fake_step(v, 1)
        del out["rows"]
        plain.add(out, 1.0, 0.0, 0)
    assert plain.means()["absent"] == pytest.approx(3.0)


def test_parameter_groups_follow_the_contract():
    model, aam = init_train(ESB16DF()), AAMHead()
    decay, no_decay = train.param_groups(model, aam, 1e-4)
    decay_ids = {id(p) for p in decay["params"]}
    names = {name for name, p in model.named_parameters() if id(p) in decay_ids}
    assert decay["weight_decay"] == 1e-4 and no_decay["weight_decay"] == 0.0
    assert id(aam.weight) in decay_ids
    assert "blocks.0.t.weight_ih_l0" in names and "film.0.weight" in names and "enc.lin.0.weight" in names
    for name in names:          # never a bias, a LayerNorm, a type embedding, a null token or a null vector
        assert not name.endswith("bias") and ".ln" not in name and "tn." not in name and "bn." not in name
        assert not name.startswith(("null.", "att.type_emb", "att.null_tok"))


# ---------------------------------------------------------------------------------------------------------------
# One step
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def parts():
    """Model, AAM head, optimizer, WeSpeaker on the CPU, and batch 0 of the tiny config (B = 2, 1 s, recipe B)."""
    from ml.earshot.wespeaker import WeSpeaker
    cfg = train.resolve_config(tiny_cfg())
    torch.manual_seed(cfg["seed"])
    model = init_train(ESB16DF())
    aam = AAMHead()
    opt, _ = train.build_optimizer(model, aam, cfg)
    batch = data.MixDataset("train", "B", cfg["seed"], 2, 16000, allow_missing_decim=True)[0]
    return {"cfg": cfg, "model": model, "aam": aam, "opt": opt, "wes": WeSpeaker.load(), "batch": batch}


def snapshot(model, aam):
    return [p.detach().clone() for p in list(model.parameters()) + list(aam.parameters())]


@needs_data
def test_non_finite_batch_skips_the_update(parts):
    model, aam, cfg = parts["model"], parts["aam"], parts["cfg"]
    before = snapshot(model, aam)
    bad = dict(parts["batch"])
    bad["mix"] = bad["mix"].copy()
    bad["mix"][0, 100] = np.nan
    out = train.step_batch(model, parts["wes"], aam, parts["opt"], list(model.parameters()) + list(aam.parameters()),
                           bad, 0, cfg["lr"], 5.0, True)
    assert not out["ok"]
    assert all(torch.equal(a, b) for a, b in zip(before, snapshot(model, aam)))   # nothing changed


@needs_data
def test_one_optimizer_step_changes_parameters(parts):
    model, aam, cfg = parts["model"], parts["aam"], parts["cfg"]
    params = list(model.parameters()) + list(aam.parameters())
    before = snapshot(model, aam)
    out = train.step_batch(model, parts["wes"], aam, parts["opt"], params, parts["batch"], 0, cfg["lr"], 5.0, True)
    assert out["ok"] and math.isfinite(out["loss"]) and math.isfinite(out["grad_norm"])
    assert out["lr"] == train.lr_at(1, cfg["lr"]) == 1.0e-3        # the update completing step 1 uses lr_at(1)
    after = snapshot(model, aam)
    changed = [not torch.equal(a, b) for a, b in zip(before, after)]
    assert all(torch.isfinite(a).all() for a in after)
    # Every tensor with a gradient moves; only the null vectors of paths this batch did not use may stay.
    assert sum(changed) >= 0.9 * len(changed), f"only {sum(changed)} of {len(changed)} tensors changed"
    assert not torch.equal(before[-1], after[-1])                  # the AAM head trains too
    assert set(out["parts"]) == set(train.TERMS)


# ---------------------------------------------------------------------------------------------------------------
# A tiny run, checkpoint and resume
# ---------------------------------------------------------------------------------------------------------------
@needs_data
def test_resume_gives_the_same_next_step(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    hist_a = []
    assert train.Trainer(tiny_cfg(), run_dir=a, device="cpu", workers=0, history=hist_a, quiet=True).run() == 0
    assert [h["step"] for h in hist_a] == [0, 1, 2]
    assert [h["indices"] for h in hist_a] == [[0, 1], [8, 9], [16, 17]]   # example index 8 j + i
    assert all(h["ok"] and math.isfinite(h["loss"]) for h in hist_a)      # finite loss, no NaN skip
    assert sorted(p.name for p in a.glob("step_*.pt")) == ["step_2.pt", "step_3.pt"]
    ck2 = train.load_ckpt(a / "step_2.pt")
    for key in ("model", "ema", "opt", "aam", "step", "seed", "config", "recipe_version", "best", "torch_version"):
        assert key in ck2
    assert ck2["step"] == 2 and ck2["seed"] == 5 and ck2["ema"] is None
    assert ck2["recipe_version"] == data.RECIPE_VERSION
    rows = train.read_csv(a / "train.csv")
    assert [int(r["step"]) for r in rows] == [3] and math.isfinite(rows[0]["loss"])

    # Resume the step-2 checkpoint in a new folder: step 3 must repeat exactly.
    b.mkdir()
    shutil.copyfile(a / "step_2.pt", b / "step_2.pt")
    hist_b = []
    assert train.Trainer(tiny_cfg(), run_dir=b, resume="last", device="cpu", workers=0, history=hist_b,
                         quiet=True).run() == 0
    assert [h["step"] for h in hist_b] == [2]
    assert hist_b[0]["indices"] == hist_a[2]["indices"]                 # bit-identical next-batch indices
    assert hist_b[0]["loss"] == hist_a[2]["loss"]                       # identical loss on the CPU
    ck_a, ck_b = train.load_ckpt(a / "step_3.pt"), train.load_ckpt(b / "step_3.pt")
    for name in ck_a["model"]:
        assert (ck_a["model"][name] - ck_b["model"][name]).abs().max() <= 1e-6, name
    assert (ck_a["aam"]["weight"] - ck_b["aam"]["weight"]).abs().max() <= 1e-6


@needs_data
def test_fresh_start_refuses_a_folder_with_checkpoints_and_stop_file_stops(tmp_path):
    torch.save({"step": 1}, tmp_path / "step_1.pt")
    with pytest.raises(train.Refused):
        train.Trainer(tiny_cfg(), run_dir=tmp_path, device="cpu", workers=0, quiet=True).run()
    (tmp_path / "STOP").write_text("")
    assert train.Trainer(tiny_cfg(), run_dir=tmp_path, resume="last", device="cpu", quiet=True).run() == 0
    assert not (tmp_path / "train.csv").exists()                         # stopped before any step


def test_recipe_version_mismatch_is_refused(tmp_path, monkeypatch):
    model, aam = init_train(ESB16DF()), AAMHead()
    cfg = train.resolve_config(tiny_cfg())
    opt, _ = train.build_optimizer(model, aam, cfg)
    state = train.make_state(model, None, opt, aam, 2, cfg, [], 0, 0.0)
    state["recipe_version"] = data.RECIPE_VERSION + 1
    train.save_ckpt(tmp_path / "step_2.pt", state)
    with pytest.raises(train.Refused, match="recipe_version"):
        train.Trainer(tiny_cfg(), run_dir=tmp_path, resume="last", device="cpu", quiet=True).run()


# ---------------------------------------------------------------------------------------------------------------
# Pruning, best copies, best rule
# ---------------------------------------------------------------------------------------------------------------
def test_prune_keeps_the_last_three_and_nothing_else_is_touched(tmp_path):
    for n in (1000, 2000, 3000, 4000, 5000):
        torch.save({"step": n}, tmp_path / f"step_{n}.pt")
    (tmp_path / "best_1.pt").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "step_7.pt.tmp").write_text("x")
    deleted = train.prune(tmp_path, 3)
    assert sorted(p.name for p in deleted) == ["step_1000.pt", "step_2000.pt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best_1.pt", "notes.txt", "step_3000.pt", "step_4000.pt",
                                                          "step_5000.pt", "step_7.pt.tmp"]


def test_best_copies_follow_the_ranking(tmp_path):
    def ckpt(n):
        torch.save({"step": n}, tmp_path / f"step_{n}.pt")

    def held(rank):
        return torch.load(tmp_path / f"best_{rank}.pt", weights_only=True)["step"]

    ckpt(2000)
    ckpt(4000)
    assert train.refresh_best_files(tmp_path, [(5.0, 4000), (3.0, 2000)]) == []
    assert held(1) == 4000 and held(2) == 2000
    (tmp_path / "step_2000.pt").unlink()                     # pruned later: the copies must not depend on it
    (tmp_path / "step_4000.pt").unlink()
    ckpt(6000)
    assert train.refresh_best_files(tmp_path, [(6.0, 6000), (5.0, 4000)]) == []
    assert held(1) == 6000 and held(2) == 4000              # the old best_1 moved down to best_2


def test_update_best_rule():
    best, rank = train.update_best([], 2000, 3.0, eligible=False, keep=2)
    assert best == [] and rank is None                       # absent or own under 20 dB: never a best
    best, rank = train.update_best(best, 4000, 3.0, eligible=True, keep=2)
    assert best == [(3.0, 4000)] and rank == 1
    best, rank = train.update_best(best, 6000, 5.0, eligible=True, keep=2)
    assert best == [(5.0, 6000), (3.0, 4000)] and rank == 1
    best, rank = train.update_best(best, 8000, 4.0, eligible=True, keep=2)
    assert best == [(5.0, 6000), (4.0, 8000)] and rank == 2
    best, rank = train.update_best(best, 10000, 1.0, eligible=True, keep=2)
    assert best == [(5.0, 6000), (4.0, 8000)] and rank is None


def test_stage_lookup():
    cfg = train.resolve_config(train.load_yaml(train.ROOT / "ml/configs/main.yaml"))
    assert train.stage_at(cfg["stages"], 0) == (cfg["stages"][0], 20000)
    assert train.stage_at(cfg["stages"], 19999)[0]["recipe"] == "A"
    assert train.stage_at(cfg["stages"], 20000) == (cfg["stages"][1], 105000)   # stage B 20,000-105,000
    assert train.stage_at(cfg["stages"], 104999)[0]["recipe"] == "B"
    assert train.stage_at(cfg["stages"], 105000) == (cfg["stages"][2], None)    # stage C 105,000-120,000
    assert train.stage_at(cfg["stages"], 119999) == (cfg["stages"][2], None)
    assert cfg["steps"] - cfg["stages"][2]["start"] == 15000                    # stage C keeps 15,000 steps
    pilot = train.resolve_config(train.load_yaml(train.ROOT / "ml/configs/pilot.yaml"))
    assert pilot["run"] == "main" and pilot["stop_at"] == 6000 and pilot["gate"] == "pilot"


# ---------------------------------------------------------------------------------------------------------------
# Validation code (real validation set, a few rows, CPU)
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not train.VAL_PATH.exists(),
                    reason="ml/cache/val_v1.npz not rendered (python -m ml.tools.render_val)")
def test_validation_runs_on_a_few_rows():
    val = train.load_val()
    assert val["mix"].shape == (400, 64000) and len(val["info"]) == 400
    assert [int((val["subset"] == i).sum()) for i in range(5)] == [160, 80, 60, 60, 40]
    # Every subset D clip can be scored for own-voice suppression (>= 0.2 s where only the own voice talks).
    d = val["subset"] == 3
    own_only = 64 * ((val["vad_o"][d] > 0) & (val["vad_t"][d] == 0)).sum(axis=1)
    assert own_only.min() >= metrics.OWN_MIN_SAMPLES
    rows = np.concatenate([np.flatnonzero(val["subset"] == i)[:2] for i in range(5)])
    torch.manual_seed(0)
    model = init_train(ESB16DF())
    m = train.validate(model, val, torch.device("cpu"), rows)
    for key in ("sisnri_median", "sisnri_mean", "absent_db_median", "absent_under10_pct", "own_db_median",
                "tsos_pct"):
        assert math.isfinite(m[key]), key
    assert 0.0 <= m["tsos_pct"] <= 100.0 and isinstance(m["eligible"], bool)
    # An output equal to the reference scores a large SI-SNRi; the mixture itself scores exactly 0.
    sep = rows[np.isin(val["subset"][rows], [0, 1, 3, 4])]
    perfect = train.val_metrics(val["ref"][rows].copy(), val, rows)
    same = train.val_metrics(val["mix"][rows].copy(), val, rows)
    assert perfect["sisnri_median"] > 50.0 and abs(same["sisnri_median"]) < 1e-6 and len(sep) == 8


@needs_data
@pytest.mark.parametrize("gate", ["pilot", "smoke"])
def test_a_run_already_at_its_end_still_reports_its_gate(tmp_path, gate):
    # A gate run that restarts at its end step (crash between the last checkpoint and the gate file, or a re-run)
    # must return the gate's exit code, never a silent 0. Here the gate fails (no validation rows / far from 20 dB).
    over = {"gate": "pilot", "stop_at": 3} if gate == "pilot" else {"gate": "smoke", "steps": 2, "fixed_examples": 2}
    cfg = tiny_cfg(**over)
    gate_file = tmp_path / f"gate_{gate}.json"
    assert train.Trainer(cfg, run_dir=tmp_path, device="cpu", workers=0, quiet=True).run() == train.EXIT_GATE
    assert json.loads(gate_file.read_text())["result"] == "FAIL"
    gate_file.unlink()                                            # as if the process died before writing it
    again = train.Trainer(cfg, run_dir=tmp_path, resume="last", device="cpu", workers=0, quiet=True)
    assert again.run() == train.EXIT_GATE
    assert json.loads(gate_file.read_text())["result"] == "FAIL"  # evaluated and written at the restart
    third = train.Trainer(cfg, run_dir=tmp_path, resume="last", device="cpu", workers=0, quiet=True)
    assert third.run() == train.EXIT_GATE                         # re-run: the saved result, still not 0
    assert "was already decided: FAIL" in (tmp_path / "events.log").read_text()


@needs_data
def test_stop_file_mid_run_saves_and_stops(tmp_path, monkeypatch):
    # The STOP check runs every LOG_EVERY steps; make it every step and drop the file during step 1.
    monkeypatch.setattr(train, "LOG_EVERY", 1)

    class DropStop(list):
        def append(self, record):
            super().append(record)
            (tmp_path / "STOP").write_text("")

    hist = DropStop()
    assert train.Trainer(tiny_cfg(), run_dir=tmp_path, device="cpu", workers=0, history=hist, quiet=True).run() == 0
    assert len(hist) == 1 and (tmp_path / "step_1.pt").exists()          # checkpoint written, then stopped
    assert "STOP file found at step 1" in (tmp_path / "events.log").read_text()


def test_command_line_exit_code_for_a_refused_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: x\nstages:\n  - {start: 0, recipe: NOPE, batch: 2, segment_s: 1.0}\n")
    with pytest.raises(SystemExit) as stop:
        train.main(["--config", str(bad), "--device", "cpu"])
    assert stop.value.code == train.EXIT_REFUSED


# ---------------------------------------------------------------------------------------------------------------
# run_forever.sh restart rules (a fake interpreter plays ml.train; no training happens)
# ---------------------------------------------------------------------------------------------------------------
FAKE_PYTHON = """#!/bin/bash
# Fake .venv/bin/python for the run_forever.sh test. `-c ...` prints the run name; `-m ml.train ...` takes the next
# line "<exit code> <new checkpoint step or ->" of scenario.txt, writes that checkpoint and exits with that code.
root="$(cd "$(dirname "$0")/../.." && pwd)"
if [ "$1" = "-c" ]; then echo fake; exit 0; fi
line=$(head -n 1 "$root/scenario.txt")
tail -n +2 "$root/scenario.txt" > "$root/scenario.next" && mv "$root/scenario.next" "$root/scenario.txt"
echo "$line" >> "$root/calls.txt"
set -- $line
if [ "$2" != "-" ]; then touch "$root/ml/runs/fake/step_$2.pt"; fi
exit "$1"
"""


def run_forever(tmp_path, scenario, quick_s):
    """Run a copy of ml/tools/run_forever.sh in a fake repository -> (exit code, number of ml.train starts)."""
    (tmp_path / "ml" / "tools").mkdir(parents=True)
    (tmp_path / "ml" / "runs" / "fake").mkdir(parents=True)
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    shutil.copyfile(train.ROOT / "ml" / "tools" / "run_forever.sh", tmp_path / "ml" / "tools" / "run_forever.sh")
    fake = tmp_path / ".venv" / "bin" / "python"
    fake.write_text(FAKE_PYTHON)
    fake.chmod(0o755)
    (tmp_path / "ml" / "runs" / "fake" / "step_1000.pt").write_text("")     # the run already has a checkpoint
    (tmp_path / "scenario.txt").write_text("\n".join(scenario) + "\n")
    env = dict(os.environ, RUN_FOREVER_WAIT_S="0", RUN_FOREVER_QUICK_S=str(quick_s))
    done = subprocess.run(["bash", str(tmp_path / "ml" / "tools" / "run_forever.sh"), "x.yaml"], env=env,
                          capture_output=True, text=True, timeout=120)
    calls = (tmp_path / "calls.txt").read_text().split("\n")
    return done.returncode, len([c for c in calls if c])


@pytest.mark.skipif(shutil.which("caffeinate") is None, reason="run_forever.sh needs macOS caffeinate")
def test_run_forever_restart_rules(tmp_path):
    # A crash that repeats at the same batch, each attempt longer than the quick limit: stops after 5 (exit 5).
    assert run_forever(tmp_path / "a", ["1 -"] * 9, quick_s=0) == (5, 5)
    # Crashes that each wrote a newer checkpoint first are progress: restart until the run finishes.
    assert run_forever(tmp_path / "b", [f"1 {s}" for s in range(2000, 9000, 1000)] + ["0 -"], quick_s=0) == (0, 8)
    # Quick crashes: the old rule, 5 in a row give up with exit 1.
    assert run_forever(tmp_path / "c", ["1 -"] * 9, quick_s=600) == (1, 5)
    # Gate failed, NaN storm, refused: no restart.
    for code in (2, 3, 4):
        assert run_forever(tmp_path / f"d{code}", [f"{code} -"], quick_s=600) == (code, 1)
