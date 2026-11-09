"""Tests of the evaluation (10-ml 12, contract 9): ml/evaluate.py and ml/earshot/{testsets, stability, dipco,
evalreport}.py. CPU only.

Proves: the restaurant grid has 1,080 distinct cells and item i < 1,080 takes cell i; every suite builder is
deterministic per seed; only held-out speakers (and test noise / rooms) are used; the picked / absent / own-voice /
TSOS wiring is right on synthetic signals; the report tables are well formed; a checkpoint gets the same weights id
as its .esw; the CSV resume cache works; and every suite runs end to end on 4 items with the random_v1 weights.
"""
import numpy as np
import pytest
import torch

from dataprep.common import read_manifest
from ml import evaluate
from ml.earshot import dipco, evalreport, stability, testsets
from ml.earshot.config import EXPORTS, SR
from ml.earshot.sources import MANIFESTS, person_id

from .helpers import synthetic_speech

RANDOM_ESW = EXPORTS / "random_v1" / "earshot.esw"
needs_data = pytest.mark.skipif(not (MANIFESTS / "eval_librispeech_test.jsonl").exists(), reason="no data pack")
needs_esw = pytest.mark.skipif(not RANDOM_ESW.exists(), reason="ml/exports/random_v1/earshot.esw not exported")


def manifest_paths(*names):
    """Every `path` of the given manifests (data/manifests/<name>.jsonl)."""
    return {r["path"] for n in names for r in read_manifest(MANIFESTS / f"{n}.jsonl")}


def manifest_speakers(*names):
    return {r["speaker"] for n in names for r in read_manifest(MANIFESTS / f"{n}.jsonl") if r.get("speaker")}


# ---------------------------------------------------------------------------------------------------------------
# Grid and plans (no data needed)
# ---------------------------------------------------------------------------------------------------------------
def test_grid_has_1080_distinct_cells_and_item_i_takes_cell_i():
    assert testsets.N_CELLS == 1080 == len(set(testsets.CELLS))
    assert testsets.cell_factors(0) == {"snr_db": -5.0, "sir_db": -8.0, "n_talkers": 1, "own": "off",
                                        "room": "sim_in_hand", "enroll": "clean"}
    assert testsets.cell_factors(1079) == {"snr_db": 10.0, "sir_db": 10.0, "n_talkers": 3, "own": "on",
                                           "room": "real", "enroll": "noisy_0"}
    for i in (0, 1, 539, 1079):
        assert testsets.item_plan(2027, i)["cell"] == i
    later = [testsets.item_plan(2027, i)["cell"] for i in range(1080, 1180)]
    assert later == [testsets.item_plan(2027, i)["cell"] for i in range(1080, 1180)]      # deterministic
    assert all(0 <= c < 1080 for c in later) and len(set(later)) > 50                   # random cells
    # Absent and accents never take cell i; their plans are deterministic too.
    assert testsets.item_plan(2028, 5, grid_first=False) == testsets.item_plan(2028, 5, grid_first=False)
    cells = {testsets.item_plan(2028, i, grid_first=False)["cell"] for i in range(40)}
    assert cells != set(range(40))


def test_overrides_follow_the_cell():
    ov = testsets.cell_overrides(1079, own_db=7.5, noise_kinds=("demand", "fsd50k"))
    assert ov["snr_db"] == 10.0 and ov["sir_db"] == 10.0 and ov["n_talkers"] == 3
    assert ov["own_in_mix"] and ov["own_given"] and ov["own_db"] == 7.5
    assert ov["room"] == "real" and "geometry" not in ov
    assert ov["enr_noise"] and ov["enr_snr_db"] == 0.0 and ov["enr_mask"] == "vad" and not ov["enr_weak"]
    assert ov["background"] == "noise" and not ov["quiet"] and ov["use_wes"] and not ov["absent"]
    ov = testsets.cell_overrides(1, 3.0, ("fsd50k", "fsd50k"), absent=True)
    assert ov["absent"] and ov["own_given"] and not ov["own_in_mix"]                    # own voice always enrolled
    assert ov["room"] == "sim" and ov["geometry"] == "in_hand" and ov["enr_noise"] and ov["enr_snr_db"] == 10.0


# ---------------------------------------------------------------------------------------------------------------
# Builders (data pack needed)
# ---------------------------------------------------------------------------------------------------------------
ARRAY_KEYS = ("mix", "ref", "enr_t", "enr_o", "talkers", "own", "vad_t", "tok_idx_t", "tok_val_t")


@needs_data
@pytest.mark.parametrize("job", [("restaurant", 7), ("absent", 3), ("accents", ("edacc", 2)),
                                 ("accents", ("voxpopuli", 1)), ("libri2mix", 5)])
def test_builders_are_deterministic(job):
    a, b = testsets.build_item(job), testsets.build_item(job)
    for key in ARRAY_KEYS:
        np.testing.assert_array_equal(a[key], b[key])
    a["row"].pop("render_s")
    b["row"].pop("render_s")
    assert a["row"] == b["row"]
    other = testsets.build_item((job[0], (job[1][0], job[1][1] + 1)) if job[0] == "accents" else (job[0], job[1] + 1))
    assert not np.array_equal(a["mix"][:len(other["mix"])], other["mix"][:len(a["mix"])])


@needs_data
@pytest.mark.parametrize("i", [0, 1079])
def test_restaurant_item_i_renders_cell_i(i):
    item = testsets.build_item(("restaurant", i))
    f = testsets.cell_factors(i)
    row = item["row"]
    assert row["cell"] == i and all(row[k] == v for k, v in f.items())
    assert item["talkers"].shape == (f["n_talkers"], testsets.L_EVAL)
    assert row["geometry"] == {"sim_in_hand": "in_hand", "sim_on_table": "on_table", "real": "none"}[f["room"]]
    assert (row["enr_noise"] is None) == (f["enroll"] == "clean")
    assert item["has_o"] and item["own_clean"] is not None                              # own voice always enrolled
    assert (np.any(item["own"] != 0) and np.any(item["vad_o"])) == (f["own"] == "on")
    assert abs(10 * np.log10(np.mean(item["mix"].astype(np.float64) ** 2)) + 26.0) < 0.1   # test-mode level


@needs_data
def test_only_held_out_speakers_and_test_material():
    src = testsets.get_eval_sources()
    held = manifest_speakers(*testsets.HELD_OUT_MANIFESTS)
    assert set(src.speakers) == held and len(held) == 48
    train_val = {person_id(s) for s in manifest_speakers("speech_train", "speech_val")}
    assert not train_val & {person_id(s) for s in held}
    speech_ok = manifest_paths(*testsets.HELD_OUT_MANIFESTS)
    noise_ok = manifest_paths("noise_test", "eval_wham_tt")
    # A busy cell: 3 talkers, own voice on, real room; every file the scene used is test material.
    plan = testsets.item_plan(2027, 1500)
    ov = testsets.cell_overrides(1079, plan["own_db"], ("demand", "demand"))
    ex = testsets.render_eval(src, 2027, 1500, ov)
    for entry in ex["scene"]["sources"]:
        is_noise = entry["role"].endswith(("noise", "music"))
        assert entry["path"] in (noise_ok if is_noise else speech_ok), entry
    # Accents: the target (mixture and enrollment) is an accent speaker, everyone else is held out.
    acc = testsets.get_accent_sources()
    for k in range(len(acc.accent["voxpopuli"])):          # the first speaker whose rows allow a target split
        try:
            ex = testsets.render_eval(acc.view("voxpopuli", k), 2032, 0, ov)
            break
        except ValueError:
            continue
    accent_ok = manifest_paths("eval_voxpopuli_accented")
    for entry in ex["scene"]["sources"]:
        if entry["role"] in ("target (mixture)", "target enrollment"):
            assert entry["path"] in accent_ok
        elif not entry["role"].endswith(("noise", "music")):
            assert entry["path"] in speech_ok, entry


@needs_data
def test_stability_stream_is_deterministic_and_leveled():
    src = testsets.get_eval_sources()
    a, b = stability.build_stream(src, 8), stability.build_stream(src, 8)
    np.testing.assert_array_equal(a["mix"], b["mix"])
    np.testing.assert_array_equal(a["ref"], b["ref"])
    assert len(a["mix"]) == 8 * SR and np.all(np.isfinite(a["mix"]))
    assert abs(dipco.dbfs(a["mix"]) + 26.0) < 0.2 and a["info"]["noise"].startswith("noise/demand/SCAFE/")
    assert stability.window_bounds(3600 * SR - 64) == ((0, 300 * SR), (3600 * SR - 64 - 300 * SR, 3600 * SR - 64))
    assert stability.window_bounds(64000) == ((0, 16000), (48000, 64000))


def test_loop_crossfade_repeats_with_an_overlap():
    x = np.arange(1, 1001, dtype=np.float32)
    out = stability.loop_crossfade(x, 2500, 100)
    assert len(out) == 2500
    np.testing.assert_array_equal(out[:900], x[:900])                  # first copy untouched before the fade
    np.testing.assert_array_equal(out[1000:1800], x[100:900])          # second copy, until the next fade
    ramp = np.sin(0.5 * np.pi * (np.arange(100) + 0.5) / 100)
    np.testing.assert_allclose(ramp ** 2 + ramp[::-1] ** 2, 1.0, atol=1e-12)   # equal power


# ---------------------------------------------------------------------------------------------------------------
# Metric wiring on synthetic signals (no data needed)
# ---------------------------------------------------------------------------------------------------------------
def synthetic_item():
    """4 s: target active in the first 2 s only, one talker throughout, own voice in the last 2 s."""
    n, half = 4 * SR, 2 * SR
    s = synthetic_speech(n, seed=1)
    s[half:] = 0.0
    r = synthetic_speech(n, seed=2)
    o = synthetic_speech(n, seed=3)
    o[:half] = 0.0
    hops = n // 64
    vad_t = np.zeros(hops, np.uint8)
    vad_t[:hops // 2] = 1
    vad_o = 1 - vad_t
    return {"mix": s + r + o, "ref": s, "talkers": r[None], "vad_talkers": np.ones((1, hops), np.uint8),
            "vad_t": vad_t, "own": o, "vad_o": vad_o, "row": {"own": "on", "snr_db": -5.0}}


def test_metric_wiring_on_synthetic_signals():
    item = synthetic_item()
    s, r, x = item["ref"], item["talkers"][0], item["mix"]
    good = evaluate.item_metrics("restaurant", item, s.copy())
    assert good["sisnri"] > 50 and good["nan_out"] == 0
    assert (good["picked"], good["eligible"]) == (1, 1)               # chunk 2 has no target: not eligible
    assert good["tsos_over"] == 0 and good["tsos_active"] > 0
    assert good["own_supp_db"] == float("inf")                         # silent output on own-only hops
    assert good.get("estoi_gain") is None and good.get("pesq_out") is None   # SNR -5 dB cell: neither runs
    wrong = evaluate.item_metrics("restaurant", item, r.copy())
    assert (wrong["picked"], wrong["eligible"]) == (0, 1)              # followed the talker: confusion
    half = evaluate.item_metrics("restaurant", item, 0.5 * s)
    assert half["tsos_over"] == half["tsos_active"]                    # 0.25 of the energy missing > 10%
    loud = evaluate.item_metrics("restaurant", item, s + 0.1 * item["own"])
    assert abs(loud["own_supp_db"] - 20.0) < 1e-6                      # own voice 20 dB down on own-only hops
    absent = evaluate.item_metrics("absent", item, 0.1 * x)
    assert abs(absent["atten_db"] - 20.0) < 1e-6
    lib = evaluate.item_metrics("libri2mix", item, s.copy())
    assert "picked" in lib and "tsos_over" not in lib


# ---------------------------------------------------------------------------------------------------------------
# Summaries and report formatting (no data needed)
# ---------------------------------------------------------------------------------------------------------------
def restaurant_rows():
    """CSV-like rows (strings): 3 in the '2-talker + noise' subset, SIR 0 and -5 rows, own on and off."""
    def row(n_t, snr, sir, enroll, own, sisnri, picked, eligible, own_db=""):
        return {"n_talkers": str(n_t), "snr_db": str(snr), "sir_db": str(sir), "enroll": enroll, "own": own,
                "room": "real", "sisnri": str(sisnri), "picked": str(picked), "eligible": str(eligible),
                "tsos_over": "1", "tsos_active": "100", "own_supp_db": own_db, "estoi_gain": "", "pesq_out": "",
                "pesq_in": "", "nan_out": "0"}
    return [row(1, 0, 0, "noisy_10", "off", 8.0, 4, 4), row(1, 5, 0, "noisy_0", "on", 9.0, 3, 4, "30"),
            row(1, 5, -5, "noisy_10", "on", 10.0, 2, 4, "26"), row(1, 5, -5, "clean", "off", 1.0, 1, 4),
            row(2, 0, 10, "noisy_0", "on", 2.0, 0, 0, "10"), row(3, -5, -8, "noisy_0", "off", 3.0, 0, 4)]


def test_restaurant_summary_selects_the_contract_subsets():
    S = evalreport.restaurant_summary(restaurant_rows())
    assert S["n"] == 6 and S["sisnri_all"] == 5.5
    assert S["n_two"] == 3 and S["sisnri_two"] == 9.0                 # 1 talker, SNR 0 or 5, noisy enrollment
    assert S["picked0"] == 7 / 8 and S["chunks0"] == 8                # pooled over the SIR 0 rows
    assert S["picked5"] == 3 / 8                                      # pooled over the SIR -5 rows
    assert S["own_n"] == 3 and S["own_median"] == 26.0                # own voice on rows only
    assert S["tsos_pct"] == 1.0
    marg = {(f, lv): n for f, lv, n, *_ in evalreport.marginals(restaurant_rows())}
    assert marg[("n_talkers", 1)] == 4 and marg[("enroll", "clean")] == 1 and marg[("room", "real")] == 6


def test_report_tables_are_well_formed():
    assert evalreport.table(["a", "b"], [("x|y", 1)])[2] == "| x\\|y | 1 |"
    assert evalreport.show(6.0, "dB") == "6.00 dB" and evalreport.show(90.04, "%") == "90.0%"
    assert evalreport.show(float("nan"), "dB") == "-" and evalreport.show(float("inf"), "dB") == "+inf"
    assert evalreport.show_thr(">=", 0.10, "+") == ">= +0.10" and evalreport.show_thr("<=", 10, "%") == "<= 10%"
    absent = [{"atten_db": str(v)} for v in (30.0, 28.0, 5.0)]
    parity = [{"file": "f", "case": "c", "check": "out", "result": "PASS"}]
    text = evalreport.build_report(
        info={"kind": "esw", "path": "x/earshot.esw", "weights": "ab" * 32, "meta": {}},
        results={"restaurant": restaurant_rows(), "absent": absent, "parity": parity}, cached={}, timings={},
        limit=None, device="cpu", workers=0, threads=1, metric_status={"estoi": "available", "pesq": "available"},
        inventory=[("thing", "data/x.jsonl", "present (1 rows, 1 audio files referenced, 0 missing)")],
        command="python -m ml.evaluate", wall_s=1.0)
    sections = [line for line in text.splitlines() if line.startswith("### ")]
    assert len(sections) == 14
    header = "| Measure | n | Value | Pass threshold | Goal | Result | Goal met |"
    assert text.count(header) == 14
    lines = text.splitlines()
    for k, line in enumerate(lines):
        if line.startswith("| ") and k + 1 < len(lines) and lines[k + 1].startswith("|---"):
            width = line.count(" | ")                                   # every row of a table has as many cells
            j = k + 2
            while j < len(lines) and lines[j].startswith("| "):
                assert lines[j].replace("\\|", "").count(" | ") == width, lines[j]
                j += 1
    banner = text.splitlines()[2]                                       # no --limit: the reasons are the counts
    assert banner.startswith("> **Partial run") and "restaurant has 6 of 2,000 items" in banner
    assert "--limit" not in banner and "stability not run" in banner
    assert "- Release gate (contract 9 model rows) - **NOT MET** - row(s) 1, " in text
    assert "| SI-SNRi median | 6 | 5.50 dB | >= 6 dB | >= 9 dB | **FAIL** | no |" in text
    assert "| SI-SNRi median | 3 | 9.00 dB | >= 7 dB | >= 9 dB | **PASS** | yes |" in text
    assert "| clips under 10 dB | 3 | 33.3% | <= 5% | - | **FAIL** | - |" in text
    assert "| failing checks | 1 | 0 | <= 0 | - | **PASS** | - |" in text
    assert "**NOT RUN**" in text and "Look Once to Hear" in text and "personal use only" in text
    assert "internal R&D use only" in text and "Missing - nothing" in text


def test_nan_outputs_fail_their_rows_instead_of_hiding():
    rows = [dict(r, sisnri="20") for r in restaurant_rows()]
    rows[0] = dict(rows[0], sisnri="nan", nan_out="1")                 # one item with a NaN output
    S = evalreport.batch_summaries({"restaurant": rows})
    assert S["restaurant"]["sisnri_all"] == 20.0 and S["restaurant"]["nan_rows"] == 1   # median of finite items
    crow = evalreport.contract_rows(S, {"estoi": "available", "pesq": "available"})
    _, _, judged, row_word = crow[0]
    assert judged[0][1] == "PASS"                                      # the median alone would pass ...
    assert judged[-1][0]["name"].startswith("items with NaN output") and judged[-1][1] == "FAIL"
    assert row_word == "FAIL"                                          # ... but the NaN item fails the row
    absent = [{"atten_db": "nan", "nan_out": "1"}, {"atten_db": "30", "nan_out": "0"}]
    A = evalreport.batch_summaries({"absent": absent})["absent"]
    assert A["median"] == 30.0 and A["nan"] == 1
    assert evalreport.values([{"x": "inf"}, {"x": "nan"}, {"x": ""}], "x") == [float("inf")]   # inf = silent output


def test_partial_flag_counts_short_cached_stability_and_dipco():
    full = {s: [{}] * n for s, (n, _) in evalreport.FULL_BATCH.items()}
    S = {"stability": {"seconds": 64.0}, "dipco": {"full": False}, "parity": {"checks": 1}}
    assert evalreport.partial_reasons(S, full, None) == ["stability stream is 64 s of 3600 s",
                                                         "dipco ran only some chunks or targets"]
    S = {"stability": {"seconds": 3600.0}, "dipco": {"full": True}, "parity": {"checks": 1}}
    assert evalreport.partial_reasons(S, full, None) == []
    assert evalreport.partial_reasons(S, full, 8) == ["--limit 8"]


def test_release_gate_needs_every_gated_measure_to_pass():
    m = evalreport.measure
    crow = [("a", "", [(m("x", 7.0, ">=", 6), "PASS"), (m("pesq", None, "report", None), "NOT AVAILABLE")], ""),
            ("b", "", [(m("listen", 10, "manual", None), "OWNER")], "OWNER")]
    assert evalreport.release_gate(crow, []) == ("**MET**", "")       # PESQ and the listening never block
    assert evalreport.release_gate(crow, ["--limit 8"]) == ("**NOT MET**", "partial run")
    crow.append(("c", "", [(m("estoi", None, ">=", 0.1), "NOT AVAILABLE")], "NOT AVAILABLE"))
    assert evalreport.release_gate(crow, []) == ("**NOT MET**", "row(s) 3 not PASS")


# ---------------------------------------------------------------------------------------------------------------
# Weights id, CSV cache
# ---------------------------------------------------------------------------------------------------------------
@needs_esw
def test_checkpoint_gets_the_weights_id_of_its_export(tmp_path):
    # The raw "model" entry differs from the "ema" entry, so the test fails if load_model used the raw weights.
    model, info = evaluate.load_model(esw=RANDOM_ESW)
    ema = {k: v.clone() for k, v in model.state_dict().items()}
    raw = {k: v.clone() for k, v in ema.items()}
    name = next(n for n, _ in model.named_parameters())                # a learned weight (always exported)
    raw[name] += 0.01
    torch.save({"model": raw, "ema": ema, "step": 3}, tmp_path / "c.pt")
    _, ck = evaluate.load_model(ckpt=tmp_path / "c.pt")
    assert ck["weights"] == info["weights"] and ck["ema"] and ck["step"] == 3
    torch.save({"model": raw, "step": 3}, tmp_path / "raw.pt")         # no EMA entry: the raw weights are used
    _, ck_raw = evaluate.load_model(ckpt=tmp_path / "raw.pt")
    assert not ck_raw["ema"] and ck_raw["weights"] != info["weights"]


def test_csv_log_keeps_rows_of_the_same_weights_and_recipe_only(tmp_path):
    path = tmp_path / "absent.csv"
    log = evaluate.CsvLog(path, "absent", "w1", "r1")
    log.add({"key": "0", "index": 0, "atten_db": 30.123456789})
    log.add({"key": "1", "index": 1, "atten_db": None})
    log.close()
    again = evaluate.CsvLog(path, "absent", "w1", "r1")
    assert again.done == {"0", "1"} and again.rows[0]["atten_db"] == "30.1235" and again.rows[1]["atten_db"] == ""
    assert again.rows[0]["recipe"] == "r1"
    assert evaluate.CsvLog(path, "absent", "w2", "r1").done == set()    # other weights: nothing reused
    assert evaluate.CsvLog(path, "absent", "w1", "r2").done == set()    # other code or device: nothing reused
    path.write_text("weights,key,old_column\nw1,0,1\n")
    assert evaluate.CsvLog(path, "absent", "w1", "r1").done == set()    # old layout: nothing reused


def test_recipe_id_depends_on_the_device_and_is_stable():
    assert evaluate.recipe_id("cpu") == evaluate.recipe_id("cpu") and len(evaluate.recipe_id("cpu")) == 16
    assert evaluate.recipe_id("cpu") != evaluate.recipe_id("mps")
    names = {p.name for p in evaluate.ROW_CODE}
    assert {"evaluate.py", "testsets.py", "render.py", "metrics.py", "model.py"} <= names
    assert "evalreport.py" not in names                                 # report wording never drops rows


# ---------------------------------------------------------------------------------------------------------------
# DiPCo window rules on a made-up session (no data needed)
# ---------------------------------------------------------------------------------------------------------------
def test_dipco_window_rules():
    assert dipco.talk_seconds([(0, 2), (1, 3), (5, 6)], 0, 10) == 4.0
    assert dipco.talk_seconds([(0, 2), (1, 3)], 2.5, 10) == 0.5
    ses = {"segments": {"T": [(0.0, 1.0), (10.0, 11.0), (20.0, 21.5)], "A": [(2.0, 5.0), (20.0, 20.6)],
                        "B": [(9.0, 9.5)]},
           "close": {"T": "t.flac", "A": "a.flac"}}
    span = (0.0, 1.5)
    # 2: no target within 0.25 s, A talks 2 s -> absent; 4: A talks 1 s -> absent; 0: enrollment span;
    # 8: target starts at 10 < 10.25 (guard); 6: nobody talks.
    assert dipco.absent_windows(ses, "T", 0, 30, span) == [2.0, 4.0]
    # Envelope: the target talks >= 1 s and A (a close-talk rival) >= 0.5 s -> only the window at 20 s.
    assert dipco.envelope_windows(ses, "T", 0, 30, span) == [(20.0, ["A"])]
    assert dipco.pick_listening({"a": [0, 10, 20, 30], "b": [5], "c": []}, n=3) == [("a", 0), ("a", 20), ("b", 5)]
    plan = dipco.chunk_plan(150 * SR)
    assert plan == [(0, 0, 60 * SR), (52 * SR, 60 * SR, 120 * SR), (112 * SR, 120 * SR, 150 * SR)]
    assert dipco.chunk_plan(150 * SR, first=1, count=1) == [(52 * SR, 60 * SR, 120 * SR)]


# ---------------------------------------------------------------------------------------------------------------
# End to end: every suite on 4 items, CPU, random_v1 weights
# ---------------------------------------------------------------------------------------------------------------
@needs_data
@needs_esw
def test_every_suite_runs_end_to_end_on_cpu(tmp_path, monkeypatch):
    # Smaller pieces so the test stays quick: 16 s DiPCo chunks with 4 s warm-up, a 4 x 2 s stability stream,
    # parity on the short test vector only. Every code path is the one a full run takes.
    monkeypatch.setattr(dipco, "CHUNK", 16 * SR)
    monkeypatch.setattr(dipco, "WARM", 4 * SR)
    monkeypatch.setattr(evaluate, "UNIT_S", 2)
    monkeypatch.setattr(evaluate, "PARITY_FILES", ("testvec_short.est",))
    threads = torch.get_num_threads()
    out = tmp_path / "report.md"
    args = ["--esw", str(RANDOM_ESW), "--device", "cpu", "--threads", "2", "--limit", "4",
            "--eval-dir", str(tmp_path), "--out", str(out)]
    try:
        results = evaluate.main(args)
        weights = evaluate.load_model(esw=RANDOM_ESW)[1]["weights"]
        counts = {"restaurant": 4, "absent": 4, "libri2mix": 4, "accents": 8}
        for suite, n in counts.items():
            rows = evaluate.read_rows(tmp_path / f"{suite}.csv")
            assert len(rows) == n == len(results[suite]), suite
            assert all(r["weights"] == weights and r["nan_out"] == "0" for r in rows), suite
        assert [r["key"] for r in results["restaurant"]] == ["0", "1", "2", "3"]
        assert sorted({r["set"] for r in results["accents"]}) == ["edacc", "voxpopuli"]
        kinds = [r["kind"] for r in results["stability"]]
        assert kinds.count("minute") == 1 and {"scene", "first", "last", "total"} <= set(kinds)
        total = [r for r in results["stability"] if r["kind"] == "total"][0]
        assert total["nan_samples"] == "0" and float(total["end_s"]) == 8.0 - 64 / SR
        enrolls = [r for r in results["dipco"] if r["kind"] == "enroll"]
        assert [r["target"] for r in enrolls] == [t for _, t in dipco.TARGETS]
        assert (tmp_path / "dipco_listen" / "clips.csv").exists()
        assert all('"full": false' in r["note"] for r in enrolls)       # a --limit run never counts as full
        assert results["parity"] and all(r["result"] == "PASS" for r in results["parity"])
        assert {"vad", "vad_o"} <= {r["check"] for r in results["parity"]}
        recipe = evaluate.recipe_id("cpu")
        assert all(r["recipe"] == recipe for s in counts for r in results[s])
        text = out.read_text()
        assert text.count("\n### ") == 14 and "Partial run - not a release result** (--limit 4;" in text
        assert "- Release gate (contract 9 model rows) - **NOT MET**" in text and f"`{recipe}`" in text
        # A rerun with the same weights reuses the cached rows (the CSV is not rewritten).
        before = (tmp_path / "restaurant.csv").stat().st_mtime_ns
        again = evaluate.main(args[:-4] + ["--suites", "restaurant", "--eval-dir", str(tmp_path),
                                           "--out", str(tmp_path / "again.md")])
        assert (tmp_path / "restaurant.csv").stat().st_mtime_ns == before
        assert [r["sisnri"] for r in again["restaurant"]] == [r["sisnri"] for r in results["restaurant"]]
        assert "this run + 4 cached rows" in (tmp_path / "again.md").read_text()
    finally:
        torch.set_num_threads(threads)
