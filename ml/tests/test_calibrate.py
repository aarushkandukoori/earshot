"""Tests for ml/tools/calibrate_thresholds.py (10-ml 13, contract 6 thresholds.json).

- The boundary rule on synthetic score sets with hand-checked answers, including the "constraint not met" fallback,
  and on random sets (the 95% / 80% promise, the 0.01 / 0.001 grid, the tightest boundary).
- locked_cos at 1% false accepts.
- The verdict logic (pass / warn / fail per check).
- The simulated captures are deterministic and match their set (no WeSpeaker needed).
- A tiny real run on random_v1 (3 captures per set, 2 pairs, CPU, about 30 s) writes a thresholds.json in the exact
  contract 6 format with the model_hash of the given .esw, which export.write_thresholds accepts.
"""
import csv
import json

import numpy as np
import pytest

from ml import export
from ml.earshot import fileio, quality
from ml.earshot.config import DATA, EXPORTS, RUNS, SR
from ml.earshot.wespeaker import DEFAULT_WEIGHTS
from ml.tools import calibrate_thresholds as cal

RANDOM_ESW = EXPORTS / "random_v1" / "earshot.esw"
VAL_MANIFEST = DATA / "manifests" / "speech_val.jsonl"


# ---------------------------------------------------------------------------------------------------------------
# The boundary rule on hand-made score sets
# ---------------------------------------------------------------------------------------------------------------
def test_higher_side_met():
    # good 100..199, bad 0..99, step 1: 95 good values are >= 105, and 80 bad values are < 80.
    r = cal.boundaries(np.arange(100) + 100, np.arange(100), "higher", 1)
    assert (r["pass"], r["fail"], r["met"], r["thresholds"]) == (105, 80, True, [80, 105])
    assert r["good_pass"] == 0.95 and r["bad_fail"] == 0.80


def test_higher_side_fallback():
    # bad 60..159: 80% of them fail only below 140, past the pass boundary 105 -> fail = pass, not met.
    r = cal.boundaries(np.arange(100) + 100, np.arange(100) + 60, "higher", 1)
    assert (r["pass"], r["fail"], r["met"], r["thresholds"]) == (105, 105, False, [105, 105])
    assert r["good_pass"] == 0.95 and r["bad_fail"] == 0.45


def test_lower_side_met():
    # own_cos style (pass below P, fail at or above F): good 0..99 -> 95 values < 95; bad 100..199 -> 80 values >= 120.
    r = cal.boundaries(np.arange(100), np.arange(100) + 100, "lower", 1)
    assert (r["pass"], r["fail"], r["met"], r["thresholds"]) == (95, 120, True, [95, 120])
    assert r["good_pass"] == 0.95 and r["bad_fail"] == 0.80


def test_lower_side_fallback():
    # bad 50..149: 80% of them are >= 70 only, below the pass boundary 95 -> fail = pass, not met.
    r = cal.boundaries(np.arange(100), np.arange(100) + 50, "lower", 1)
    assert (r["pass"], r["fail"], r["met"], r["thresholds"]) == (95, 95, False, [95, 95])
    assert r["bad_fail"] == 0.55


def test_fail_boundary_may_equal_pass_boundary():
    # 80% of the bad values sit just under the pass boundary 2.0: the lowest grid value that fails them is 2.0
    # itself, so F = P with the constraint met (no warn band, but no "constraint not met" either).
    good = np.full(20, 2.0)
    bad = np.array([1.999] * 16 + [5.0] * 4)
    r = cal.boundaries(good, bad, "higher", 0.01)
    assert r["pass"] == 2.0 and r["fail"] == 2.0 and r["met"] and r["bad_fail"] == 0.8


def test_small_sets_and_empty_bad_set():
    # 20 good values: 19 must pass (ceil(0.95 * 20)); no bad values -> fail = pass and not met.
    good = np.arange(20, dtype=float)
    r = cal.boundaries(good, [], "higher", 1)
    assert r["pass"] == 1 and r["fail"] == 1 and not r["met"] and r["n_bad"] == 0
    with pytest.raises(ValueError):
        cal.boundaries([], [1.0], "higher", 1)
    with pytest.raises(ValueError):
        cal.boundaries([1.0], [1.0], "sideways", 1)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("side", ["higher", "lower"])
def test_random_sets_keep_the_promise(seed, side):
    rng = np.random.default_rng(seed)
    good = rng.normal(0.65 if side == "higher" else 0.2, 0.08, int(rng.integers(20, 600)))
    bad = rng.normal(0.45 if side == "higher" else 0.6, 0.12, int(rng.integers(20, 600)))
    step = 0.001
    r = cal.boundaries(good, bad, side, step)
    p, f = r["pass"], r["fail"]
    assert round(p, 3) == p and round(f, 3) == f                       # on the grid
    if side == "higher":
        assert np.mean(good >= p) >= 0.95 and np.mean(good >= p + step) < 0.95     # tightest pass boundary
        assert f <= p
        if r["met"]:
            assert np.mean(bad < f) >= 0.80 and (f == p or np.mean(bad < f - step) < 0.80)
        else:
            assert f == p and np.mean(bad < p) < 0.80
    else:
        assert np.mean(good < p) >= 0.95 and np.mean(good < p - step) < 0.95
        assert f >= p
        if r["met"]:
            assert np.mean(bad >= f) >= 0.80 and (f == p or np.mean(bad >= f + step) < 0.80)
        else:
            assert f == p and np.mean(bad >= p) < 0.80


def test_locked_threshold_one_percent():
    value, above = cal.locked_threshold(np.arange(1000) / 1000.0)     # 0.000 .. 0.999
    assert value == 0.989 and above == 0.01
    rng = np.random.default_rng(3)
    imp = rng.normal(0.3, 0.15, 1000)
    value, above = cal.locked_threshold(imp)
    assert round(value, 3) == value and above <= 0.01 and np.mean(imp > value - 0.001) > 0.01
    with pytest.raises(ValueError):
        cal.locked_threshold([])


def test_status_and_verdicts():
    assert cal.status(6.0, [0, 6], "higher") == "pass"
    assert cal.status(5.99, [0, 6], "higher") == "warn"
    assert cal.status(-0.01, [0, 6], "higher") == "fail"
    assert cal.status(0.49, [0.5, 0.6], "lower") == "pass"
    assert cal.status(0.5, [0.5, 0.6], "lower") == "warn"
    assert cal.status(0.6, [0.5, 0.6], "lower") == "fail"
    assert cal.status(None, [0.5, 0.6], "lower") == "not checked"
    t = export.initial_thresholds("x")
    row = {"trim_fail": None, "speech_s": 5.5, "sbr_db": 10.0, "one_voice": 0.7, "own_cos": 0.1, "clip_pct": 0.0}
    assert cal.verdicts(row, t)["overall"] == "pass"
    assert cal.verdicts(dict(row, one_voice=0.5), t)["overall"] == "warn"
    assert cal.verdicts(dict(row, clip_pct=2.0), t)["overall"] == "fail"
    v = cal.verdicts(dict(row, trim_fail="too short", speech_s=None, one_voice=None, own_cos=None), t)
    assert v["speech_s"] == "fail" and v["one_voice"] == "not checked" and v["overall"] == "fail"


def test_refuses_live_run_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(cal, "training_running", lambda: True)
    with pytest.raises(SystemExit, match="refused"):
        cal.check_out_path(RUNS / "main" / "thresholds.json")
    cal.check_out_path(tmp_path / "thresholds.json")                  # anywhere else is fine
    monkeypatch.setattr(cal, "training_running", lambda: False)
    cal.check_out_path(RUNS / "main" / "thresholds.json")             # allowed once training has stopped


# ---------------------------------------------------------------------------------------------------------------
# Simulated captures and a tiny real run
# ---------------------------------------------------------------------------------------------------------------
needs_val = pytest.mark.skipif(not VAL_MANIFEST.exists(), reason=f"{VAL_MANIFEST} is missing")


@needs_val
def test_captures_are_deterministic_and_match_their_set():
    from ml.earshot.sources import get_sources
    src = get_sources("val")
    a, info_a, owner_a, clip_a = cal.make_capture(src, "good", cal.SEED, 0)
    b, info_b, owner_b, clip_b = cal.make_capture(src, "good", cal.SEED, 0)
    assert np.array_equal(a, b) and info_a == info_b and (owner_a, clip_a) == (owner_b, clip_b)
    c, _, _, _ = cal.make_capture(src, "good", cal.SEED, 1)
    assert not np.array_equal(a[:len(c)], c[:len(a)])
    for kind in cal.SETS:
        pcm, info, owner, clip = cal.make_capture(src, kind, cal.SEED, 2)
        assert pcm.dtype == np.float32 and np.all(np.isfinite(pcm)) and np.max(np.abs(pcm)) <= 1.0
        assert 6 * SR <= len(pcm) <= 15 * SR and 0.5 <= info["lead_s"] <= 3.0
        assert 0 <= clip < 8 and src.in_own_pool[owner]
        person = src.speakers.index(info["person"])
        if kind == "owner":
            assert owner == person                     # the phone owner enrolls themselves
        else:
            assert owner != person
        if kind == "noisy":
            assert info["noise"] in ("babble", "synthetic") and -6 <= info["snr_db"] <= 0
        elif kind != "short":
            assert 5 <= info["snr_db"] <= 25
        if kind == "two_talkers":
            assert info["second"] != info["person"] and 0 <= info["second_below_db"] <= 3
        if kind == "short":
            assert info["short_in_range"] == (2.0 <= info["short_vad_s"] <= 3.9)
        if kind == "clipped":
            assert quality.clip_pct(pcm) >= 100 * cal.CLIPPED_FRACTION - 0.01


@needs_val
def test_noisy_set_background_is_selectable():
    # Default "steady" (babble or synthetic noise); "recipe" renders the noisy set with the 8.3 noise draw instead,
    # at the same SNR range. Only the noisy set reads the option.
    from ml.earshot import render
    from ml.earshot.sources import get_sources
    src = get_sources("val")
    _, steady, _, _ = cal.make_capture(src, "noisy", cal.SEED, 0)
    _, recipe, _, _ = cal.make_capture(src, "noisy", cal.SEED, 0, noisy_background="recipe")
    assert steady["noise"] in ("babble", "synthetic") and recipe["noise"] in render.NOISE_KINDS
    assert -6 <= recipe["snr_db"] <= 0 and steady["person"] == recipe["person"]
    good_a, _, _, _ = cal.make_capture(src, "good", cal.SEED, 0)
    good_b, _, _, _ = cal.make_capture(src, "good", cal.SEED, 0, noisy_background="recipe")
    assert np.array_equal(good_a, good_b)


@needs_val
@pytest.mark.skipif(not (RANDOM_ESW.exists() and DEFAULT_WEIGHTS.exists()), reason="needs random_v1 and WeSpeaker")
def test_tiny_run_writes_contract_thresholds(tmp_path):
    out = tmp_path / "thresholds.json"
    obj = cal.main(["--esw", str(RANDOM_ESW), "--out", str(out), "--n", "3", "--pairs", "2", "--threads", "2"])
    with open(out, encoding="utf-8") as f:
        written = json.load(f)
    assert written == obj
    export.check_thresholds_shape(written)                     # exactly the contract 6 keys and nesting
    assert written["model_hash"] == fileio.model_hash(RANDOM_ESW)
    initial = export.initial_thresholds(written["model_hash"])
    # Fixed values stay at the contract's; the calibrated pairs are ordered [low, high].
    assert written["own"] == initial["own"] and written["own_window_cos"] == initial["own_window_cos"]
    for key in ["speech_s", "dup_cos", "clip_pct"]:
        assert written["target"][key] == initial["target"][key]
    for key in ["sbr_db", "one_voice", "own_cos"]:
        low, high = written["target"][key]
        assert isinstance(low, float) and isinstance(high, float) and low <= high
    assert isinstance(written["locked_cos"], float) and -1.0 <= written["locked_cos"] <= 1.0
    # export accepts it for these weights only.
    export.write_thresholds(tmp_path / "copy.json", written["model_hash"], source=out)
    with pytest.raises(ValueError, match="calibrated for"):
        export.write_thresholds(tmp_path / "copy.json", "0" * 64, source=out)
    # The CSVs and the summary.
    rows = list(csv.DictReader(open(tmp_path / "calibration_scores.csv", encoding="utf-8")))
    assert len(rows) == 3 * len(cal.SETS) and sorted({r["set"] for r in rows}) == sorted(cal.SETS)
    assert all(r["overall_status"] in ("pass", "warn", "fail") for r in rows)
    pairs = list(csv.DictReader(open(tmp_path / "calibration_pairs.csv", encoding="utf-8")))
    assert len(pairs) == 2
    summary = json.load(open(tmp_path / "calibration_summary.json", encoding="utf-8"))
    assert summary["thresholds"] == written and set(summary["checks"]) == {"sbr_db", "one_voice", "own_cos"}
    assert summary["noisy_background"] == "steady" and "DECISIONS" in summary["pending_decisions"]
