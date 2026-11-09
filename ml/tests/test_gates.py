"""Gate tests (10-ml 10.5 and the test_gates row of 10-ml 14).

Part 1 checks the gate logic of ml/train.py on made-up numbers: smoke_decision and pilot_gate (the rule of
DECISIONS L 2026-11-08 - val SI-SNRi at 6k above 4k and 2k, no NaN skips, step time information only), including CSV
rows written twice by a resumed run, and the gate JSON file.
Part 2 reads the real gate files ml/runs/smoke/gate_smoke.json and ml/runs/main/gate_pilot.json and asserts PASS;
each is skipped until its run has happened.
"""
import json

import pytest

from ml import train
from ml.earshot.config import RUNS


# ---------------------------------------------------------------------------------------------------------------
# Part 1 - gate logic
# ---------------------------------------------------------------------------------------------------------------
def test_smoke_decision():
    assert train.smoke_decision(100, 25.0, 1500) == "PASS"         # PASS as soon as the median reaches 20 dB
    assert train.smoke_decision(300, 20.0, 1500) == "PASS"         # exactly 20 dB passes
    assert train.smoke_decision(100, 19.9, 1500) is None           # keep training
    assert train.smoke_decision(1500, 19.9, 1500) == "FAIL"        # not there by step 1,500
    assert train.smoke_decision(1500, 21.0, 1500) == "PASS"
    assert train.smoke_decision(700, float("nan"), 1500) is None   # a NaN median never passes ...
    assert train.smoke_decision(1500, float("nan"), 1500) == "FAIL"   # ... and fails at the end


def val_rows(s2=1.0, s4=1.8, s6=2.5):
    return [{"step": 2000.0, "sisnri_median": s2}, {"step": 4000.0, "sisnri_median": s4},
            {"step": 6000.0, "sisnri_median": s6}]


def train_rows(step_s=1.1):
    """train.csv rows every 50 steps up to 6,000; the first 500 steps are slow (warm-up) and must not count."""
    return [{"step": float(s), "step_s_median": (5.0 if s <= 500 else step_s)} for s in range(50, 6001, 50)]


# Pilot rule since DECISIONS L 2026-11-08: PASS = val SI-SNRi median at 6k above both 4k and 2k, and no NaN skips.
# No absolute dB target; the step time is information only.
def test_pilot_gate_pass():
    g = train.pilot_gate(val_rows(), 0, train_rows())
    assert g["result"] == "PASS" and all(g["checks"].values())
    assert set(g["checks"]) == {"val_6k_above_2k_and_4k", "no_nan_skips"}
    assert g["sisnri_median"] == {"2000": 1.0, "4000": 1.8, "6000": 2.5}
    assert g["step_s_median_500_6000"] == pytest.approx(1.1) and g["step_rows"] == 110   # rows 550 .. 6,000
    assert g["main_steps_for_40h"] == 124000                     # floor(144,000 / (1.1 x 1.05) / 1,000) x 1,000


@pytest.mark.parametrize("vals", [
    val_rows(-0.86, -1.14, -0.24),     # the failed spec-loss pilot's numbers: 6k beats both, all under 0 dB -> PASS
    val_rows(1.0, 0.9, 2.5),           # 4k below 2k does not matter, 6k is the highest
    val_rows(-3.0, -2.0, -1.5),        # rising but negative: no absolute target any more
    val_rows(0.0, 0.0, 0.01),          # just above both
])
def test_pilot_gate_pass_when_still_rising_at_6k(vals):
    g = train.pilot_gate(vals, 0, train_rows())
    assert g["result"] == "PASS" and g["checks"]["val_6k_above_2k_and_4k"] is True


@pytest.mark.parametrize("vals, nan_skips, failed", [
    (val_rows(1.0, 1.8, 1.8), 0, "val_6k_above_2k_and_4k"),       # flat: 6k equal to 4k is not rising
    (val_rows(1.0, 1.0, 1.0), 0, "val_6k_above_2k_and_4k"),       # flat everywhere
    (val_rows(1.0, 1.8, 1.5), 0, "val_6k_above_2k_and_4k"),       # falling after 4k
    (val_rows(2.0, 1.0, 1.5), 0, "val_6k_above_2k_and_4k"),       # above 4k but below 2k
    (val_rows(3.0, 2.0, 1.0), 0, "val_6k_above_2k_and_4k"),       # falling all the way
    (val_rows(), 1, "no_nan_skips"),                              # one NaN skip fails
    (val_rows(), 7, "no_nan_skips"),
    (val_rows()[:1] + val_rows()[2:], 0, "val_6k_above_2k_and_4k"),   # the 4k validation is missing
    (val_rows()[:2], 0, "val_6k_above_2k_and_4k"),                     # the 6k validation is missing
    (val_rows(1.0, float("nan"), 2.5), 0, "val_6k_above_2k_and_4k"),   # a NaN median never passes
])
def test_pilot_gate_fail(vals, nan_skips, failed):
    g = train.pilot_gate(vals, nan_skips, train_rows())
    assert g["result"] == "FAIL" and g["checks"][failed] is False


@pytest.mark.parametrize("step_s, steps_40h", [
    (1.1, 124000), (1.262, 108000), (1.34, 102000), (3.0, 45000), (0.5, 274000)])
def test_pilot_gate_step_time_never_fails_it(step_s, steps_40h):
    # Slow or fast, the step time only fills in the numbers.
    g = train.pilot_gate(val_rows(), 0, train_rows(step_s))
    assert g["result"] == "PASS" and set(g["checks"]) == {"val_6k_above_2k_and_4k", "no_nan_skips"}
    assert g["step_s_median_500_6000"] == pytest.approx(step_s) and g["main_steps_for_40h"] == steps_40h
    # No train.csv rows at all: still decided by the validation alone, with the step numbers empty.
    g = train.pilot_gate(val_rows(), 0, [])
    assert g["result"] == "PASS" and g["step_s_median_500_6000"] is None and g["main_steps_for_40h"] is None
    assert g["step_rows"] == 0


def test_main_steps_for_40h_by_hand():
    # floor(40 x 3600 / (step_s x 1.05) / 1000) x 1000
    assert train.main_steps_for_40h(1.101) == 124000             # 144,000 / 1.15605 = 124,562 -> 124,000 (main)
    assert train.main_steps_for_40h(1.25) == 109000              # 144,000 / 1.3125 = 109,714 -> 109,000 (rule)
    assert train.main_steps_for_40h(1.262) == 108000             # 144,000 / 1.3251 = 108,671 -> 108,000 (A/B)
    assert train.main_steps_for_40h(1.05) == 130000              # 144,000 / 1.1025 = 130,612 -> 130,000
    assert train.main_steps_for_40h(1.3251) == 103000            # 144,000 / 1.391355 = 103,496 -> 103,000
    for bad in (None, 0.0, -1.0, float("nan"), float("inf")):
        assert train.main_steps_for_40h(bad) is None


def test_pilot_gate_from_csv_files_with_a_resume(tmp_path):
    # A run that crashed after its 4k validation and resumed from the 3k checkpoint writes step 4000 twice;
    # read_csv keeps the last row of each step.
    log = train.RunLog(tmp_path, quiet=True)
    for step, value in [(2000, 1.0), (4000, 0.5), (4000, 1.5), (6000, 2.2)]:
        log.row("val.csv", train.VAL_COLUMNS, {"step": step, "weights": "ema", "sisnri_median": value})
    for row in train_rows(1.0):
        log.row("train.csv", train.TRAIN_COLUMNS, row)
    vals = train.read_csv(tmp_path / "val.csv")
    assert [r["sisnri_median"] for r in vals] == [1.0, 1.5, 2.2]
    g = train.pilot_gate(vals, 0, train.read_csv(tmp_path / "train.csv"))
    assert g["result"] == "PASS"
    train.write_gate(log, g)
    saved = json.loads((tmp_path / "gate_pilot.json").read_text())
    assert saved["result"] == "PASS" and saved["gate"] == "pilot" and saved["nan_skips"] == 0
    assert "gate pilot: PASS" in (tmp_path / "events.log").read_text()


# ---------------------------------------------------------------------------------------------------------------
# Part 2 - the real gate files (skipped until the runs exist)
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [RUNS / "smoke" / "gate_smoke.json", RUNS / "main" / "gate_pilot.json"],
                         ids=["smoke", "pilot"])
def test_gate_file_passed(path):
    if not path.exists():
        pytest.skip(f"{path} does not exist yet (the run has not happened)")
    gate = json.loads(path.read_text())
    assert gate["result"] == "PASS", f"{path.name}: {json.dumps(gate)}"
