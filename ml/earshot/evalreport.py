"""The evaluation report (10-ml 12.1, contract 9): summaries of the result rows and the markdown text.

Every number comes from the per-mixture result rows as they are stored in ml/runs/eval/<suite>.csv (dicts of
strings), so a report can always be rebuilt from the CSVs alone. Layout of the report:
  header (weights, device, run size, package status) -> overview -> one table per contract 9 row (value, pass
  threshold, goal, PASS / FAIL) -> restaurant factor marginals -> timing and full-run estimates -> data
  inventory -> notes on interpretations -> credits and license notes (contract 0).
"""
import datetime
import json
import math
import os
import subprocess

import numpy as np

from . import metrics
from .config import DATA, ROOT, SR
from .dipco import TARGETS as DIPCO_TARGETS
from .stability import STREAM_S
from .testsets import ACCENT_MANIFESTS, COUNTS, GRID

# MPS planning number for the offline forward: validation on MPS ran 400 mixtures x 4 s (1,600 s of audio) in
# 21.1 s including its metrics (ml/runs/sanity_main/events.log, 2026-11-07 23:17) -> 13.2 ms per audio second.
# Evaluation never ran on MPS while training was live, so the MPS estimates below are built on this number.
MPS_S_PER_AUDIO_S = 21.1 / 1600.0
LIBRI2MIX_AUDIO_S = 2 * 15074.0     # 3,000 definitions, 4.187 h of min-mode mixtures, each run for 2 targets
STREAM_HOPS = 3600 * SR // 64       # 60 min stability stream
NOT_RUN, NOT_AVAILABLE = "NOT RUN", "NOT AVAILABLE"


# ---------------------------------------------------------------------------------------------------------------
# Reading CSV cells
# ---------------------------------------------------------------------------------------------------------------
def num(value):
    """A CSV cell -> float, or None if it is empty or not a number ("inf" stays +inf: a silent output)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def values(rows, key):
    """The numeric values of one column. Empty cells and NaN are skipped, so one NaN item cannot turn a median
    into NaN ("NO DATA"); NaN outputs are counted separately (nan_items) and fail their contract rows. +inf stays
    (a silent output)."""
    out = [num(r.get(key)) for r in rows]
    return [v for v in out if v is not None and not math.isnan(v)]


def nan_items(rows):
    """How many items of a batch suite had a non-finite model output (CSV column nan_out = 1)."""
    return int(sum(values(rows, "nan_out")))


def median(v):
    return float(np.median(v)) if len(v) else float("nan")


def same(cell, level):
    """Does a CSV cell hold this grid level? Numbers are compared as numbers ("-5" equals -5.0)."""
    return cell == level if isinstance(level, str) else num(cell) == float(level)


def pooled(rows, part, whole):
    """sum(part) / sum(whole) over rows (picked / eligible, tsos_over / tsos_active); NaN if the sum is 0."""
    total = sum(values(rows, whole))
    return sum(values(rows, part)) / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------------------------------------------
# Suite summaries
# ---------------------------------------------------------------------------------------------------------------
def restaurant_summary(rows):
    """The restaurant numbers of contract 9 from the restaurant rows."""
    two = [r for r in rows if same(r["n_talkers"], 1) and num(r["snr_db"]) in (0.0, 5.0) and r["enroll"] != "clean"]
    sir0 = [r for r in rows if same(r["sir_db"], 0.0)]
    sir5 = [r for r in rows if same(r["sir_db"], -5.0)]
    own = metrics.own_summary([num(r["own_supp_db"]) for r in rows if r["own"] == "on"])
    estoi_rows = [r for r in rows if num(r["snr_db"]) in (0.0, 5.0)]
    pesq_rows = [r for r in rows if num(r["snr_db"]) is not None and num(r["snr_db"]) >= 5.0]
    estoi, pesq_out, pesq_in = values(estoi_rows, "estoi_gain"), values(pesq_rows, "pesq_out"), values(pesq_rows,
                                                                                                       "pesq_in")
    return {
        "n": len(rows), "sisnri_all": median(values(rows, "sisnri")),
        "n_two": len(two), "sisnri_two": median(values(two, "sisnri")),
        "picked0": pooled(sir0, "picked", "eligible"), "chunks0": int(sum(values(sir0, "eligible"))),
        "picked5": pooled(sir5, "picked", "eligible"), "chunks5": int(sum(values(sir5, "eligible"))),
        "own_median": own["median"], "own_p05": own["own_db_p05"], "own_n": own["n"],
        "own_rows": sum(1 for r in rows if r["own"] == "on"),
        "tsos_pct": 100.0 * pooled(rows, "tsos_over", "tsos_active"), "tsos_frames": int(sum(values(rows,
                                                                                                      "tsos_active"))),
        "estoi_mean": float(np.mean(estoi)) if estoi else float("nan"), "estoi_median": median(estoi),
        "estoi_n": len(estoi), "estoi_rows": len(estoi_rows),
        "pesq_mean": float(np.mean(pesq_out)) if pesq_out else float("nan"),
        "pesq_in_mean": float(np.mean(pesq_in)) if pesq_in else float("nan"), "pesq_n": len(pesq_out),
        "pesq_rows": len(pesq_rows), "nan_rows": nan_items(rows),
    }


def marginals(rows):
    """[(factor, level, n, median SI-SNRi, pooled picked rate, TSOS %)] for every level of every grid factor."""
    out = []
    for factor, levels in GRID:
        for level in levels:
            part = [r for r in rows if same(r[factor], level)]
            out.append((factor, level, len(part), median(values(part, "sisnri")), pooled(part, "picked", "eligible"),
                        100.0 * pooled(part, "tsos_over", "tsos_active")))
    return out


def stability_summary(rows):
    """NaN count and the first / last window SI-SNRi from the stability rows (None if incomplete)."""
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    if "first" not in by_kind or "last" not in by_kind:
        return None
    first, last = num(by_kind["first"][0]["sisnri"]), num(by_kind["last"][0]["sisnri"])
    total = by_kind.get("total", [{}])[0]
    scene = by_kind.get("scene", [{}])[0]
    return {"nan": int(num(total.get("nan_samples")) or 0), "first": first, "last": last,
            "diff": last - first if first is not None and last is not None else float("nan"),
            "window_s": num(by_kind["first"][0]["end_s"]) - num(by_kind["first"][0]["start_s"]),
            "seconds": num(scene.get("end_s")), "minutes": len(by_kind.get("minute", [])),
            "scene": scene.get("note", ""), "speed": total.get("note", "")}


def dipco_summary(rows):
    """Absent attenuation per target and pooled, the S08 envelope win rate and the listening clips."""
    absent = [r for r in rows if r["kind"] == "absent"]
    env = [r for r in rows if r["kind"] == "envelope"]
    per_target = {}
    for r in absent:
        per_target.setdefault(r["target"], []).append(num(r["atten_db"]))
    listening = [r for r in rows if r["kind"] == "listening"]
    note = json.loads(listening[0]["note"]) if listening and listening[0]["note"] else {}
    clips, listen_dir = note.get("clips", 0), os.path.dirname(note.get("csv", ""))
    wins = values(env, "win")
    enroll = [r for r in rows if r["kind"] == "enroll"]
    notes = [json.loads(r["note"]) if r.get("note") else {} for r in enroll]
    return {"absent_median": median(values(absent, "atten_db")), "absent_n": len(absent),
            "per_target": {t: (median(v), len(v)) for t, v in per_target.items()},
            "env_rate": 100.0 * float(np.mean(wins)) if wins else float("nan"), "env_n": len(wins), "clips": clips,
            "listen_dir": listen_dir, "enroll": enroll,
            # full = all 4 targets ran every chunk (a --limit run does not); nan = non-finite output samples
            "full": len(notes) == len(DIPCO_TARGETS) and all(n.get("full") for n in notes),
            "nan": sum(int(n.get("nan_samples") or 0) for n in notes)}


def parity_summary(rows):
    """Parity checks run, failed, and whether it ran at all."""
    if not rows or rows[0].get("result") == NOT_RUN:
        return None
    fails = [r for r in rows if r["result"] != "PASS"]
    return {"checks": len(rows), "fails": len(fails), "failed": [f"{r['file']} {r['case']} {r['check']}"
                                                                  for r in fails][:10]}


def batch_summaries(results):
    """Summaries of every suite that has rows -> {suite: summary or None}."""
    S = {}
    rows = results.get("restaurant")
    S["restaurant"] = restaurant_summary(rows) if rows else None
    rows = results.get("absent")
    S["absent"] = (dict(metrics.absent_summary(values(rows, "atten_db")), n=len(rows), nan=nan_items(rows))
                   if rows else None)
    rows = results.get("libri2mix")
    S["libri2mix"] = {"median": median(values(rows, "sisnri")), "n": len(rows), "nan": nan_items(rows)} if rows \
        else None
    rows = results.get("accents")
    S["accents"] = ({s: {"median": median(values([r for r in rows if r["set"] == s], "sisnri")),
                         "n": sum(1 for r in rows if r["set"] == s),
                         "nan": nan_items([r for r in rows if r["set"] == s])} for s in ACCENT_MANIFESTS}
                    if rows else None)
    S["stability"] = stability_summary(results["stability"]) if results.get("stability") else None
    S["dipco"] = dipco_summary(results["dipco"]) if results.get("dipco") else None
    S["parity"] = parity_summary(results["parity"]) if results.get("parity") else None
    return S


# ---------------------------------------------------------------------------------------------------------------
# Contract 9 rows: measures, verdicts and formatting
# ---------------------------------------------------------------------------------------------------------------
def measure(name, value, op, thr, goal=None, unit="dB", n=None, note=""):
    """One line of a contract-row table. op is ">=" or "<=" (gated), "report" (reported only) or "manual"."""
    return {"name": name, "value": value, "op": op, "thr": thr, "goal": goal, "unit": unit, "n": n, "note": note}


def missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def meets(value, op, thr):
    """True / False against a threshold, None when there is no value."""
    if missing(value) or thr is None:
        return None
    return value >= thr if op == ">=" else value <= thr


def verdict(m, ran, available=True):
    """PASS / FAIL / NOT RUN / NOT AVAILABLE / NO DATA / REPORTED / OWNER for one measure."""
    if not ran:
        return NOT_RUN
    if not available:
        return NOT_AVAILABLE
    if m["op"] == "manual":
        return "OWNER"
    if missing(m["value"]):
        return "NO DATA"
    if m["op"] == "report":
        return "REPORTED"
    return "PASS" if meets(m["value"], m["op"], m["thr"]) else "FAIL"


def row_verdict(verdicts):
    """A contract row's result from its measures' verdicts (any FAIL fails the row)."""
    for word in ("FAIL", NOT_RUN, NOT_AVAILABLE, "NO DATA"):
        if word in verdicts:
            return word
    if all(v == "PASS" for v in verdicts):
        return "PASS"
    return " / ".join(sorted(set(verdicts)))


def show(value, unit):
    """A value for the tables: dB with 2 decimals, % with 1, counts as integers, other numbers with 3 decimals."""
    if missing(value):
        return "-"
    if isinstance(value, float) and math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "dB":
        return f"{value:.2f} dB"
    if unit == "count":
        return f"{int(value)}"
    return f"{value:+.3f}" if unit == "+" else f"{value:.3f}"


def show_thr(op, thr, unit):
    """'>= 6 dB', '<= 10%', '<= 0' ..."""
    if thr is None or op in ("report", "manual"):
        return "-" if op != "report" else "reported"
    suffix = {"dB": " dB", "%": "%", "+": ""}.get(unit, "")
    text = f"+{thr:.2f}" if unit == "+" else f"{thr:g}"
    return f"{op} {text}{suffix}"


def contract_rows(S, status):
    """The 14 contract 9 rows -> [(title, condition, [(measure, verdict)], row verdict)]."""
    R, A, L2, AC, ST, DI, PA = (S["restaurant"], S["absent"], S["libri2mix"], S["accents"], S["stability"],
                                S["dipco"], S["parity"])
    rows = []

    def g(summary, key):
        """summary[key], or None when the suite did not run."""
        return summary[key] if summary else None

    def add(title, condition, ran, measures, available=True, nan=0, nan_what="items with NaN output"):
        """One contract row. nan > 0 (non-finite model outputs in the suites behind the row) adds a failing
        measure, so a NaN can never hide behind a median of the finite items."""
        if nan:
            measures = measures + [measure(f"{nan_what} (left out of the values above)", nan, "<=", 0, None,
                                           "count")]
        judged = [(m, verdict(m, ran, available)) for m in measures]
        rows.append((title, condition, judged, row_verdict([v for _, v in judged])))

    r_nan = g(R, "nan_rows") or 0                     # restaurant items with a NaN output
    add("SI-SNRi median", "restaurant sim, all cells", R is not None,
        [measure("SI-SNRi median", g(R, "sisnri_all"), ">=", 6, 9, "dB", g(R, "n"))], nan=r_nan)
    add("SI-SNRi median", "2-talker + noise, SNR 0-5 dB, noisy enrollment (1 competing talker, SNR 0 or 5 dB, "
        "enrollment noisy 10 or 0 dB)", R is not None,
        [measure("SI-SNRi median", g(R, "sisnri_two"), ">=", 7, 9, "dB", g(R, "n_two"))], nan=r_nan)
    add("SI-SNRi median", "Libri2Mix test clean (data/eval/libri2mix_test.jsonl), enrollment = another utterance "
        "of the target", L2 is not None,
        [measure("SI-SNRi median", g(L2, "median"), ">=", 10, 12, "dB", g(L2, "n"))], nan=g(L2, "nan") or 0)

    def pct(v):
        return None if missing(v) else 100.0 * v
    add("Picked rate (per 2 s chunk, output closer to target than to every talker)", "SIR 0 dB / -5 dB",
        R is not None,
        [measure("picked rate, SIR 0 dB", pct(g(R, "picked0")), ">=", 90, 95, "%", g(R, "chunks0")),
         measure("picked rate, SIR -5 dB", pct(g(R, "picked5")), ">=", 75, 85, "%", g(R, "chunks5"))], nan=r_nan)
    conf = None if R is None or missing(R["picked0"]) else 100.0 * (1.0 - R["picked0"])
    add("Target confusion rate (= 1 - picked)", "SIR 0 dB", R is not None,
        [measure("confusion rate, SIR 0 dB", conf, "<=", 10, 5, "%", g(R, "chunks0"))], nan=r_nan)
    add("Target-absent output energy (attenuation vs mixture)", "absent clips", A is not None,
        [measure("attenuation median", g(A, "median"), ">=", 25, 33, "dB", g(A, "n")),
         measure("clips under 10 dB", g(A, "under10_pct"), "<=", 5, None, "%", g(A, "n"))], nan=g(A, "nan") or 0)
    add("Own-voice suppression", "own-voice-only frames, own voice enrolled (cells with own voice on)",
        R is not None,
        [measure("median", g(R, "own_median"), ">=", 25, 30, "dB", g(R, "own_n")),
         measure("5th percentile (own_db_p05)", g(R, "own_p05"), ">=", 18, None, "dB", g(R, "own_n"))], nan=r_nan)
    add("Target over-suppression (TSOS)", "target-active frames, all restaurant mixtures", R is not None,
        [measure("TSOS", g(R, "tsos_pct"), "<=", 3, 1.5, "%", g(R, "tsos_frames"))], nan=r_nan)
    add("ESTOI gain (pystoi, MIT)", "SNR 0-5 dB cells", R is not None,
        [measure("ESTOI gain, mean over clips", g(R, "estoi_mean"), ">=", 0.10, 0.15, "+", g(R, "estoi_n"),
                 note="" if R is None or missing(R["estoi_median"]) else f"median {R['estoi_median']:+.3f}")],
        available=status.get("estoi") == "available", nan=r_nan)
    add("PESQ wideband (pesq package, internal R&D use only, reported not gated)", "SNR >= 5 dB cells",
        R is not None,
        [measure("PESQ wb of the output, mean", g(R, "pesq_mean"), "report", None, None, "", g(R, "pesq_n")),
         measure("PESQ wb of the mixture, mean", g(R, "pesq_in_mean"), "report", None, None, "", g(R, "pesq_n"))],
        available=status.get("pesq") == "available")
    acc = []
    for s in ACCENT_MANIFESTS:
        med, base = (AC[s]["median"] if AC else None), g(R, "sisnri_all")
        diff = None if missing(med) or missing(base) else med - base
        acc.append(measure(f"{s}: abs(median - all-cells median)", None if diff is None else abs(diff), "<=", 1.5, 0.5,
                           "dB", AC[s]["n"] if AC else None,
                           note="" if diff is None else f"{s} median {med:.2f} dB, all-cells {base:.2f} dB, "
                                                        f"difference {diff:+.2f} dB"))
    acc_nan = r_nan + (sum(AC[s]["nan"] for s in ACCENT_MANIFESTS) if AC else 0)
    add("Accents", "EdAcc test, VoxPopuli accented as targets (within 1.5 dB of the all-cells median)",
        AC is not None and R is not None, acc, nan=acc_nan)
    # "Within 1 dB" is read two-sided (see NOTES): a stream that gets better by more than 1 dB also fails.
    add("Stability", "60 min stream through StreamRef (CPU)", ST is not None,
        [measure("NaN output samples", g(ST, "nan"), "<=", 0, None, "count", g(ST, "minutes")),
         measure("abs(SI-SNRi last window - first window)", None if ST is None else abs(ST["diff"]), "<=", 1.0, None,
                 "dB", note="" if ST is None else f"first {ST['first']:.2f} dB, last {ST['last']:.2f} dB, windows "
                                                  f"of {ST['window_s']:.0f} s")])
    add("DiPCo real dinner party", "far-field channel; targets P09 (S03), P25, P26, P27 (S08)", DI is not None,
        [measure("absent-target attenuation, median over all 4", g(DI, "absent_median"), ">=", 15, 20, "dB",
                 g(DI, "absent_n")),
         measure("envelope win rate, S08 windows", g(DI, "env_rate"), ">=", 75, 85, "%", g(DI, "env_n")),
         measure("10 listening clips (owner listens)", g(DI, "clips"), "manual", None, None, "count",
                 note=f"input and output WAVs and clips.csv in {DI['listen_dir']}" if DI else "")],
        nan=g(DI, "nan") or 0, nan_what="NaN output samples")
    add("Parity", "PyTorch self-check (file integrity) - StreamRef rerun vs the stored .est outputs (contract "
        "3.3); the engine side of this row is WP-ENGINE's C run", PA is not None,
        [measure("failing checks", g(PA, "fails"), "<=", 0, None, "count", g(PA, "checks"),
                 note="" if not PA or not PA["failed"] else "; ".join(PA["failed"]))])
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Timing and full-run estimates
# ---------------------------------------------------------------------------------------------------------------
FULL_BATCH = {"restaurant": (COUNTS["restaurant"], 8.0 * COUNTS["restaurant"]),
              "absent": (COUNTS["absent"], 8.0 * COUNTS["absent"]),
              "accents": (2 * COUNTS["accents"], 8.0 * 2 * COUNTS["accents"]),
              "libri2mix": (COUNTS["libri2mix"], LIBRI2MIX_AUDIO_S)}


def partial_reasons(S, allrows, limit):
    """Why this report is not a release result ([] for a complete run). Cached rows count too: stability or DiPCo
    rows left in the CSVs by an earlier --limit run are a short stream or a few chunks, not the full suite."""
    reasons = [f"--limit {limit}"] if limit is not None else []
    for suite, (n_full, _) in FULL_BATCH.items():
        n = len(allrows.get(suite) or [])
        if n < n_full:
            reasons.append(f"{suite} has {n:,} of {n_full:,} items")
    ST, DI = S["stability"], S["dipco"]
    if ST is None:
        reasons.append("stability not run")
    elif missing(ST["seconds"]) or ST["seconds"] < STREAM_S:
        reasons.append(f"stability stream is {show(ST['seconds'], 'count')} s of {STREAM_S} s")
    if DI is None:
        reasons.append("dipco not run")
    elif not DI["full"]:
        reasons.append("dipco ran only some chunks or targets")
    if S["parity"] is None:
        reasons.append("parity not run")
    return reasons


def release_gate(crow, reasons):
    """(word, why) for the report header: MET only when the run is complete and every gated measure (>= or <=) of
    every contract 9 row is PASS. PESQ (reported only) and the DiPCo listening (the owner's job) never block; a
    gated measure that is NOT RUN, NOT AVAILABLE or NO DATA blocks like a FAIL."""
    blocking = [str(k + 1) for k, (_, _, judged, _) in enumerate(crow)
                if any(m["op"] in (">=", "<=") and v != "PASS" for m, v in judged)]
    why = ([f"row(s) {', '.join(blocking)} not PASS"] if blocking else []) + (["partial run"] if reasons else [])
    return ("**NOT MET**", "; ".join(why)) if why else ("**MET**", "")


def rows_from(suites, cached, timings):
    """The overview's 'Rows from' cell: this run, the cache of an earlier run with the same weights and recipe,
    or this run plus rows it resumed from the cache."""
    if any(s in cached for s in suites):
        return "cache (earlier run, same weights and recipe)"
    resumed = sum((timings.get(s) or {}).get("cached_rows") or 0 for s in suites)
    return f"this run + {resumed:,} cached rows" if resumed else "this run"


def dipco_audio_s():
    """Seconds of model input for a full DiPCo run: each target's session length plus 8 s of warm-up per chunk."""
    from dataprep.common import read_manifest
    from .dipco import CHUNK, TARGETS, WARM
    far = {r["group"]: r["duration"] for r in read_manifest(DATA / "manifests" / "eval_dipco.jsonl")
           if r["extra"]["kind"] == "far-field"}
    total = 0.0
    for session, _ in TARGETS:
        chunks = math.ceil(far[session] * SR / CHUNK)
        total += far[session] + (chunks - 1) * WARM / SR
    return total


def estimates(timings, device):
    """{suite: (seconds on CPU or None, seconds on MPS)} for a full run, from this run's measured rates.

    Batch suites: items x (render + WeSpeaker + metrics) + audio / 8 s x forward time per 8 s of audio. On MPS the
    forward uses MPS_S_PER_AUDIO_S (unless this run was on MPS) and WeSpeaker is left out (small on the GPU).
    Stability always streams on the CPU. pystoi / pesq time is included only if they ran in this run.
    """
    out = {}
    for suite, (n, audio) in FULL_BATCH.items():
        t = timings.get(suite)
        if not t or not t.get("items"):
            continue
        per = t["render_s_per_item"] + t["metrics_s_per_item"]
        if suite == "restaurant":                     # ESTOI on the SNR 0 / 5 cells, PESQ on 5 / 10: half each
            per += 0.5 * (t.get("estoi_s_per_call") or 0.0) + 0.5 * (t.get("pesq_s_per_call") or 0.0)
        measured = n * (per + t["embed_s_per_item"]) + audio / 8.0 * t["forward_s_per_8s"]
        cpu = measured if device == "cpu" else None
        mps = measured if device == "mps" else n * per + audio * MPS_S_PER_AUDIO_S
        out[suite] = (cpu, mps)
    t = timings.get("stability")
    if t and t.get("items"):
        full = STREAM_HOPS * t["ms_per_hop"] / 1000.0 + t["build_s"] * 3600.0 / t["seconds"]
        out["stability"] = (full, full)
    t = timings.get("dipco")
    if t and t.get("items"):
        audio = dipco_audio_s()
        rate = t["forward_s_per_8s"] / 8.0
        cpu = audio * rate if device == "cpu" else None
        out["dipco"] = (cpu, audio * rate if device == "mps" else audio * MPS_S_PER_AUDIO_S)
    t = timings.get("parity")
    if t and t.get("items"):
        out["parity"] = (t["wall_s"], t["wall_s"])
    return out


def hms(seconds):
    """Seconds -> '1 h 05 min', '12 min' or '40 s'."""
    if seconds is None:
        return "-"
    if seconds >= 3600:
        return f"{int(seconds // 3600)} h {int(seconds % 3600 // 60):02d} min"
    if seconds >= 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


# ---------------------------------------------------------------------------------------------------------------
# Data inventory (what the suites read, and whether it is on disk)
# ---------------------------------------------------------------------------------------------------------------
INVENTORY = [
    ("held-out speech, VCTK test (8 speakers)", "data/manifests/speech_test.jsonl"),
    ("held-out speech, LibriSpeech test-clean (40 speakers)", "data/manifests/eval_librispeech_test.jsonl"),
    ("accents, EdAcc test", "data/manifests/eval_edacc_test.jsonl"),
    ("accents, VoxPopuli accented", "data/manifests/eval_voxpopuli_accented.jsonl"),
    ("noise_test (DEMAND SCAFE, FSD50K test)", "data/manifests/noise_test.jsonl"),
    ("WHAM! test noise (eval only)", "data/manifests/eval_wham_tt.jsonl"),
    ("DiPCo far-field and close-talk files", "data/manifests/eval_dipco.jsonl"),
    ("Libri2Mix test definitions", "data/eval/libri2mix_test.jsonl"),
    ("DiPCo segments", "data/eval/dipco_segments.jsonl"),
    ("simulated test rooms", "data/rir/sim_test.npy"),
    ("real test rooms (MIT IR Survey)", "data/rir/real_test.npy"),
    ("WeSpeaker ResNet34 weights", "models/wespeaker-voxceleb-resnet34/avg_model.pt"),
    ("mic-chain decimation FIR", "ml/assets/decim16k_fir.npy"),
]


def inventory():
    """[(what, path, status)]: status 'present (N rows, M files missing)' or 'MISSING'."""
    from dataprep.common import read_manifest
    out = []
    for what, rel in INVENTORY:
        path = ROOT / rel
        if not path.exists():
            out.append((what, rel, "MISSING"))
            continue
        if path.suffix != ".jsonl":
            out.append((what, rel, f"present ({path.stat().st_size / 1e6:.1f} MB)"))
            continue
        rows = list(read_manifest(path))
        refs = []
        for r in rows:
            if "path" in r:
                refs.append(r["path"])
            refs += [s["path"] for s in r.get("sources", [])] + [e["path"] for e in r.get("enrollment", [])]
            refs += [p for p in (r.get("far_path"), r.get("close_path")) if p]
        refs = sorted(set(refs))
        gone = sum(1 for p in refs if not os.path.exists(DATA / p))
        out.append((what, rel, f"present ({len(rows):,} rows, {len(refs):,} audio files referenced, "
                               f"{gone} missing)"))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Fixed text: interpretations and credits
# ---------------------------------------------------------------------------------------------------------------
NOTES_INTRO = ("Only the first point below is a DECISIONS L row (2026-11-08). The others are this package's "
               "readings of 10-ml 12 where the spec is silent or cannot be followed literally; they are binding "
               "only once the lead records them as DECISIONS L rows, which must happen before docs/eval/"
               "report-v1.md counts as the release report.")

NOTES = [
    "Audio is rendered on the fly from each suite's seed and never cached; only the per-mixture result rows are "
    "kept, in ml/runs/eval/<suite>.csv (lead decision, DECISIONS L 2026-11-08, instead of the npz cache of 10-ml "
    "12.1). A cached row is reused only when its weights id and its recipe id (a hash of the code that makes the "
    "rows, RECIPE_VERSION and the device) both match this run.",
    "Restaurant / absent / accents noise - each noise clip is WHAM! test 50%, DEMAND SCAFE test 25%, FSD50K test "
    "25% (the 12.3 split noise_test 50% / WHAM! 50%; DEMAND vs FSD50K inside noise_test is an even split chosen "
    "here). 1 or 2 noise clips and babble in 70% come from recipe B's draws; no music, no quiet rooms.",
    "Real rooms - the 193 real_test rows of category bar, restaurant or indoor; the target takes a uniformly drawn "
    "row and the other 7 positions take other rows of the same category (real_test has no room_id).",
    "Every evaluation mixture enrolls the own voice (contract 9), uses a clean own enrollment, an es_vad target "
    "mask, no weaker talker in the target enrollment and WeSpeaker on; the enrollment room (same 50 / other 30 / "
    "dry 20) and the enrollment noise source are drawn as in training. Test mode 8.9 (level -26 dBFS, high-pass "
    "100 Hz, no EQ, no clipping, no Lombard, fir_decim on). e_o is embedded from the clean own clip (as in "
    "validation).",
    "Grid order - cell c is the c-th entry of the product SNR x SIR x talkers x own x room x enrollment "
    "(enrollment changes fastest). Mixtures 0 to 1,079 take cell i, the rest random cells (seed 2027); absent and "
    "accents mixtures take random cells.",
    "ESTOI gain is gated on its mean over the SNR 0-5 dB clips (the contract does not say mean or median; the "
    "median is shown next to it). PESQ is the mean over the SNR >= 5 dB clips; clips where pesq raises count as "
    "skipped.",
    "Accents - 'within 1.5 dB' is read literally as |accent median - all-cells median| <= 1.5 dB, per set. Seeds "
    "2031 (EdAcc) and 2032 (VoxPopuli); 12.3 gives none. Eligible accent targets have at least 2 utterances and at "
    "least 14.5 s, because render.py never cycles an enrollment; 12.3's 'cycled to 6 s' is therefore not "
    "implemented, and the speakers it would need are listed below. This is the target-pool rule of DECISIONS L "
    "2026-11-07 ('re-drawn instead of cycling', written for training) applied to the accent targets; a speaker "
    "left out with 2 or more utterances (the list gives each count) would be kept by 12.3 read literally.",
    "Libri2Mix - sources with the definition's gains cut to `length` (min mode, dataprep.eval_libri2mix.render), "
    "then the test-mode chain at -26 dBFS with the same filters and gain on both sources, cut to a multiple of 64 "
    "samples; enrollment = the first 10 s of the definition's enrollment utterance through the same chain.",
    "Stability scene - target in spells (about 40% active), 2 competing talkers about 50% active each at SIR 0 dB, "
    "one DEMAND SCAFE channel looped with 1 s equal-power crossfades at SNR 5 dB, one sim_test room, clean 6 s "
    "enrollment. The streamed output is aligned by one hop (y_stream[n + 64] = y_offline[n]) before scoring.",
    "Stability 'within 1 dB of the first 5 min' is read two-sided, |last-window SI-SNRi - first-window SI-SNRi| "
    "<= 1 dB, so a stream that improves by more than 1 dB also fails (the literal reading; the one-sided "
    "alternative is last >= first - 1 dB).",
    "NaN outputs - an item whose model output is not finite is left out of the medians and rates and counted "
    "instead; any such item (or any NaN DiPCo output sample) fails every contract row that uses its suite.",
    "DiPCo - music (S08 from 62 s, S03 from 2,025 s) is not excluded (12.5 does not say to). A --limit run starts "
    "at the chunk holding the end of the enrollment span so it has windows to score.",
    "Parity here is a PyTorch self-check (file integrity): it reruns StreamRef and export.py's conditioning code "
    "from each .est file's inputs and applies the file's own meta.tolerances, so it catches a wrong, damaged or "
    "mismatched file but not an error in the reference model itself. The engine-vs-PyTorch parity of contract 9 "
    "is WP-ENGINE's C run on the same files.",
]

CREDITS = [
    "Interaction idea (look at a person, then hear only them) from UW's **Look Once to Hear** (Veluri et al., "
    "CHI 2024); none of its code is used (UW non-commercial license). The network follows the band-split RNN / "
    "TF-GridNet line of work.",
    "Speaker encoder - WeSpeaker ResNet34 trained on VoxCeleb; the weights are granted for research purposes, so "
    "**this build and these numbers are for personal use only** and the weights are never distributed.",
    "PESQ - the `pesq` package implements ITU-T P.862 (OPTICOM / ITU licensing): **internal R&D use only**, "
    "reported, never a release gate.",
    "ESTOI - pystoi (MIT).",
    "Test data - LibriSpeech test-clean (CC BY 4.0), VCTK (CC BY 4.0), EdAcc (CC BY-SA 4.0), VoxPopuli (CC0), "
    "DEMAND (CC BY-SA 3.0), FSD50K (CC0 / CC BY), WHAM! noise (CC BY-NC 4.0, evaluation only, never in training), "
    "MIT IR Survey (Traer and McDermott, PNAS 2016; CC BY 4.0 as stated by mirrors, test rooms only), DiPCo "
    "(CDLA-Permissive-1.0), Libri2Mix definitions rebuilt with LibriMix's scripts (MIT).",
]


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def table(header, rows):
    """Markdown table lines."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]   # escape pipes
    return lines


# ---------------------------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------------------------
def build_report(info, results, cached, timings, limit, device, workers, threads, metric_status, inventory,
                 command, wall_s, accent_excluded=None):
    """The markdown report text. results / cached = {suite: rows}; cached rows come from earlier runs with the
    same weights and recipe id (marked in the overview)."""
    allrows = dict(cached)
    allrows.update(results)
    S = batch_summaries(allrows)
    crow = contract_rows(S, metric_status)
    meta = info.get("meta") or {}
    lines = ["# Earshot evaluation report", ""]
    reasons = partial_reasons(S, allrows, limit)
    if reasons:
        lines += [f"> **Partial run - not a release result** ({'; '.join(reasons)}). The PASS / FAIL words below "
                  "apply to these items only.", ""]
    gate, why = release_gate(crow, reasons)
    weights = f"`{info['path']}` ({info['kind']}"
    weights += f", step {info.get('step')}, {'EMA' if info.get('ema') else 'raw'} weights)" if info["kind"] == "ckpt" \
        else f", init {meta.get('init', '-')}, run {meta.get('run', '-')})"
    # release_v1.sh looks for the exact text "- Release gate (contract 9 model rows) - **MET**".
    lines += [f"- Release gate (contract 9 model rows) - {gate}" + (f" - {why}" if why else ""),
              "- Still open after any result - the owner listens to the 10 DiPCo clips (row 13); the C engine "
              "parity is WP-ENGINE's run",
              f"- Weights - {weights}", f"- Model id (`.esw` data_sha256) - `{info['weights']}`",
              f"- Recipe id (resume-cache key - row code, RECIPE_VERSION, device) - `{info.get('recipe', '-')}`",
              f"- Date - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}, git {git_commit()}",
              f"- Device - {device} (offline forward and WeSpeaker; StreamRef on the CPU), torch threads {threads}, "
              f"render workers {workers}",
              f"- Packages - ESTOI (pystoi) {metric_status.get('estoi')}, PESQ (pesq) {metric_status.get('pesq')}",
              f"- Command - `{command}`", f"- Wall time - {hms(wall_s)}", ""]
    lines += ["## Overview", ""]
    lines += table(["#", "Contract 9 row", "Condition", "Result", "Rows from"],
                   [(k + 1, t, c, f"**{v}**", rows_from(suites_of(t, c), cached, timings))
                    for k, (t, c, _, v) in enumerate(crow)])
    lines.append("")
    for k, (title, condition, judged, v) in enumerate(crow):
        lines += [f"### {k + 1}. {title} - {condition}", ""]
        body = []
        for m, mv in judged:
            goal = "-" if m["goal"] is None else show_thr(m["op"], m["goal"], m["unit"])
            goal_met = meets(m["value"], m["op"], m["goal"]) if m["goal"] is not None else None
            body.append((m["name"], "-" if m["n"] is None else m["n"], show(m["value"], m["unit"]),
                         show_thr(m["op"], m["thr"], m["unit"]), goal, f"**{mv}**",
                         "-" if goal_met is None else ("yes" if goal_met else "no")))
        lines += table(["Measure", "n", "Value", "Pass threshold", "Goal", "Result", "Goal met"], body)
        notes = [m["note"] for m, _ in judged if m["note"]]
        lines += ([""] + [f"Note - {n}" for n in notes] if notes else []) + [""]
    lines += marginal_section(allrows.get("restaurant"))
    lines += timing_section(timings, device, metric_status)
    lines += detail_section(S, accent_excluded)
    lines += ["## Data inventory", ""]
    lines += table(["Material", "Path", "Status"], [(w, f"`{p}`", s) for w, p, s in inventory])
    missing_items = [w for w, _, s in inventory if is_missing(s)]
    lines += ["", "Missing - " + (", ".join(missing_items) if missing_items else "nothing; every evaluation source "
                                  "listed above is on disk."), ""]
    lines += ["## Interpretations", "", NOTES_INTRO, ""] + [f"- {n}" for n in NOTES] + [""]
    lines += ["## Credits and license notes", ""] + [f"- {c}" for c in CREDITS] + [""]
    return "\n".join(lines)


def is_missing(status):
    """True for an inventory status 'MISSING' or 'present (..., N missing)' with N > 0."""
    return status == "MISSING" or (status.endswith("missing)") and not status.endswith(" 0 missing)"))


def suites_of(title, condition):
    """Which suites feed a contract row (for the overview's 'Rows from' column)."""
    text = (title + " " + condition).lower()
    for word, suites in (("libri2mix", ["libri2mix"]), ("accents", ["accents", "restaurant"]),
                         ("absent output", ["absent"]), ("stability", ["stability"]), ("dipco", ["dipco"]),
                         ("parity", ["parity"])):
        if word in text:
            return suites
    return ["restaurant"]


def marginal_section(rows):
    """Restaurant factor marginals: every level of every grid factor."""
    lines = ["## Restaurant factor marginals", ""]
    if not rows:
        return lines + ["Restaurant suite not run.", ""]
    body = [(f, lv, n, show(med, "dB"), show(100.0 * pk if not missing(pk) else None, "%"), show(ts, "%"))
            for f, lv, n, med, pk, ts in marginals(rows)]
    lines += ["Median SI-SNRi per level (other factors pooled), with the pooled picked rate and TSOS.", ""]
    return lines + table(["Factor", "Level", "n", "Median SI-SNRi", "Picked rate", "TSOS"], body) + [""]


def timing_section(timings, device, metric_status):
    """Measured rates of this run and the full-run estimates on CPU and MPS."""
    lines = ["## Timing", ""]
    body = []
    for suite in ("restaurant", "absent", "libri2mix", "accents", "dipco"):
        t = timings.get(suite)
        if t and t.get("items"):
            body.append((suite, t["items"], f"{t['render_s_per_item']:.3f}", f"{t['embed_s_per_item']:.3f}",
                         f"{t['forward_s_per_item']:.3f}", f"{t['forward_s_per_8s']:.3f}",
                         f"{t['metrics_s_per_item']:.3f}", hms(t.get("suite_wall_s"))))
    if body:
        lines += [f"Measured in this run on {device} (seconds; DiPCo items are 60 s chunks with 8 s warm-up; render "
                  "times are single-process CPU times, measured inside the renderer).", ""]
        lines += table(["Suite", "Items run", "Render s / item", "WeSpeaker s / item", "Forward s / item",
                        "Forward s per 8 s of audio", "Metrics s / item", "Suite wall"], body) + [""]
    t = timings.get("stability")
    if t and t.get("items"):
        lines += [f"Stability - {t['seconds']} s stream built in {t['build_s']} s, streamed at {t['ms_per_hop']} ms "
                  "per 4 ms hop with StreamRef on the CPU.", ""]
    est = estimates(timings, device)
    if est:
        body = [(s, hms(c), hms(m)) for s, (c, m) in est.items()]
        cpu_total = sum(c for c, _ in est.values()) if all(c is not None for c, _ in est.values()) else None
        body.append(("**total**", hms(cpu_total), hms(sum(m for _, m in est.values()))))
        lines += ["Estimated full run (2,000 restaurant, 500 absent, 6,000 Libri2Mix, 1,000 accents, 60 min stream, "
                  "all DiPCo chunks) from the rates above. The MPS column uses the validation forward rate "
                  f"({1000 * MPS_S_PER_AUDIO_S:.1f} ms per audio second, ml/runs/sanity_main) unless this run was on "
                  "MPS; render and metrics stay on the CPU and add up serially (--workers overlaps rendering with "
                  "the forward). Libri2Mix items run one per forward (their lengths differ), so its MPS figure, "
                  "built on a batch-8 rate, is a lower bound. Restaurant also adds ESTOI on half the clips and "
                  "PESQ on half the clips at the per-call cost below.", ""]
        t = timings.get("restaurant") or {}
        for name, pkg in (("estoi", "pystoi"), ("pesq", "pesq")):
            cost = t.get(f"{name}_s_per_call")
            how = "calibrated on 8 s of synthetic audio (no such cell ran)" if t.get(f"{name}_calibrated") \
                else "measured in this run"
            lines.append(f"- {name.upper()} ({pkg}, {metric_status.get(name)}) - "
                         + (f"{cost:.3f} s per clip, {how}" if cost is not None else "not timed"))
        lines.append("")
        lines += table(["Suite", "CPU", "MPS"], body) + [""]
    return lines


def detail_section(S, accent_excluded):
    """Per-target DiPCo numbers, the stability scene and the accent speakers left out."""
    lines = ["## Details", ""]
    D = S.get("dipco")
    if D:
        body = [(t, n, show(med, "dB")) for t, (med, n) in sorted(D["per_target"].items())]
        lines += ["DiPCo absent-target windows per target:", ""] + table(["Target", "Windows", "Median attenuation"],
                                                                       body) + [""]
        for r in D["enroll"]:
            lines.append(f"- {r['target']} enrollment span {num(r['start_s']):.1f} to {num(r['end_s']):.1f} s "
                         f"(excluded from scoring); {r['note']}")
        lines.append("")
    ST = S.get("stability")
    if ST:
        lines += [f"Stability scene - {ST['scene']}", "", f"Stability speed - {ST['speed']}", ""]
    if accent_excluded:
        for s, spk in accent_excluded.items():
            lines.append(f"- Accent speakers left out of {s} (under 2 utterances or under 14.5 s): "
                         + (", ".join(f"{name} ({n} utt, {sec} s)" for name, n, sec in spk) or "none"))
        lines.append("")
    return lines
