"""Enrollment threshold calibration (10-ml 13, contract 6 "Enrollment quality checks", DECISIONS D30).

    .venv/bin/python -m ml.tools.calibrate_thresholds --esw ml/exports/v1/earshot.esw
    .venv/bin/python -m ml.tools.calibrate_thresholds --esw ml/exports/random_v1/earshot.esw \\
        --out ml/runs/calib_test/thresholds.json --n 20 --pairs 20          # quick check, a few minutes

What it does, in plain words
----------------------------
When someone is enrolled, the app records a few seconds of them talking and runs the checks of contract 6 (enough
speech, speech louder than the background, only one voice, not the phone owner, no clipping). Each check says pass,
warn or fail, and the numbers that separate those live in thresholds.json. This tool picks three of them from data:

- the pass boundary so that 95% of good enrollments pass the check, and
- the fail boundary so that at least 80% of the matching bad enrollments fail it, but never beyond the pass
  boundary; if both cannot hold, fail = pass and the tool prints "constraint not met" for that check.

Only sbr_db, one_voice and own_cos are calibrated this way. speech_s [4, 5], dup_cos 0.75 and clip_pct [0.1, 1.0]
stay at the contract values (the short and clipped sets below only report how often those fixed rules fail).
locked_cos (the "might be following someone else" check on the model output) is set at the 99th percentile of
impostor cosines, so 1% of impostors are accepted.

Every enrollment is simulated as a full phone capture: 6 to 15 s long, with 0.5 to 3 s of background before the
person talks, from validation speakers (speech_val), validation noise (noise_val) and validation rooms (sim_val),
seed 2030. Each capture goes through ml/earshot/quality.py, the same check formulas the app runs, with WeSpeaker
(ml/earshot/wespeaker.py, the same PyTorch module training uses).

Sets (--n captures each, default 500). Capture i of a set always uses np.random.default_rng([seed, set code, i]),
so a small run is exactly the start of the full run.
  code  set          what is in the capture                                  used for
  0     good         one talker, noise at SNR U(5, 25) dB, room as 10-ml 8.5  every pass boundary
  1     short        one talker with only 2 to 3.9 s of es_vad speech         reports the fixed speech_s rule
  2     noisy        one talker, steady noise at SNR U(-6, 0) dB              sbr_db fail boundary
  3     two_talkers  a second talker 0 to 3 dB below the first               one_voice fail boundary
  4     owner        the phone owner talking, scored against the owner's e    own_cos fail boundary
  5     clipped      a good-style capture with 2% of samples at full scale    reports the fixed clip_pct rule
  6     impostor pairs (--pairs, default 1,000)                               locked_cos

Points 10-ml 13 leaves open, decided here (WP-ML choices):
- Room "as 8.5" without a mixture: the 'same room' (50%) and 'other room' (30%) cases both become a random sim_val
  room at position 0 (validation has no real rooms), dry 20%. Noise is one clip drawn as in 8.3 (validation has no
  DEMAND rows, so FSD50K takes that share), no babble, so a good capture holds exactly one voice.
- Noisy set: a steady background instead of the 8.3 draw - babble of 6 to 20 validation talkers (50%) or the
  loader's synthetic noise (colored noise or mains hum, 50%). 10-ml 13 expects these captures to score about 1 to
  3 dB ("for stationary noise"); with the 8.3 draw (70% FSD50K, mostly short sound events) the quiet gaps between
  events are the "other" frames, so sbr_db stays high (measured 2026-11-08 on 120 + 120 captures: noisy p80 34 dB,
  and only 37% of them failed even at the pass boundary). Measured with this background: p05 3.1 to 3.5 dB, p80
  5.3 to 5.9 dB. The good set keeps the 8.3 draw, the realistic mix of phone backgrounds.
  This choice sets the shipped sbr_db fail boundary and is not a DECISIONS L row yet (the lead records it before
  the release). `--noisy-background recipe` renders the noisy set with the 8.3 draw instead, so the lead can see
  both; the choice is written into calibration_summary.json ("noisy_background").
- SNR is set against the active power of the person's clean speech (as in 8.5), with the noise over the whole
  capture, lead-in included. Then the phone's mic chain (high-pass, EQ, fir_decim, level U(-40, -15) dBFS,
  self-noise, the headroom rule of DECISIONS L 2026-11-08, no clipping).
- Every capture also draws a phone owner (a different own-pool speaker) whose e is the WeSpeaker vector of one of
  the owner's 8 s own clips (sources.build_clip, as 6.3); targets are scored against it (own_cos and the own-window
  removal of contract 6 step 9). In the owner set the person IS the owner, talking from their mix rows (the rows not
  used for the own clips; they may repeat inside a long capture).
- Short set: the talking is cut into 0.4 to 1.2 s pieces with 0.5 to 1.5 s of silence between them; the amount of
  talking is adjusted (up to 20 tries, each a fresh draw) until es_vad of the whole capture holds 2.0 to 3.9 s.
- Each calibrated check uses only the captures where its value exists (a capture that fails trimming has no
  WeSpeaker values); the summary lists how many that were.
- Boundaries are rounded to 0.01 dB (sbr_db) and 0.001 (cosines), always in the direction that keeps the 95% / 80%
  promise.
- locked_cos pair: person A gives an enrollment and its token-encoder voiceprint g_A; person B gives an enrollment
  (tokens, g_B, WeSpeaker e_B, from half of B's rows) and a 4 s scene from B's other rows; the model, conditioned on
  B, separates B (so its output "follows B"); embed_g of the last 3 s of that output is compared with g_A (impostor)
  and with g_B (genuine, reported only).

Writes (next to --out): thresholds.json (contract 6 format, model_hash of the given .esw), calibration_scores.csv
(every capture and its scores and verdicts), calibration_pairs.csv (every locked pair) and calibration_summary.json
(the boundaries, pass and fail rates, timings). Never writes rendered audio. Refuses to write into ml/runs/main or
ml/runs/smoke while ml.train is running.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ml.earshot import dsp, quality, render
from ml.earshot.config import MAX_T_TOK, RUNS, SR
from ml.earshot.fileio import model_hash
from ml.earshot.model import CondInputs
from ml.earshot.vad import es_vad

SEED = 2030
N_DEFAULT = 500                 # captures per set (10-ml 13)
PAIRS_DEFAULT = 1000            # impostor pairs for locked_cos
PASS_RATE = 0.95                # 95% of good captures pass each calibrated check
FAIL_RATE = 0.80                # at least 80% of the matching bad set fails it
LOCKED_QUANTILE = 0.99          # locked_cos = 99th percentile of impostor cosines (1% false accepts)
OWN_WINDOW_COS = 0.60           # fixed in contract 6 (not calibrated)

CAPTURE_S = (6.0, 15.0)         # full capture length
LEAD_S = (0.5, 3.0)             # background only before the person talks
GOOD_SNR_DB = (5.0, 25.0)
NOISY_SNR_DB = (-6.0, 0.0)
SECOND_TALKER_DB = (0.0, 3.0)   # second talker this many dB below the first
SHORT_SPEECH_S = (2.0, 3.9)     # es_vad speech of a short capture
SHORT_TRIES = 20
BURST_S = (0.4, 1.2)            # short set: talking pieces ...
BURST_GAP_S = (0.5, 1.5)        # ... and the silence between them
CLIPPED_FRACTION = 0.02         # clipped set: 2% of samples at full scale
ROOM_SIM_P = 0.8                # 10-ml 8.5: same room 50% + other room 30% (a sim_val room here), dry 20%
OWN_CLIPS = 8                   # own clips per own-pool speaker (6.3)
LOCKED_MIX = 4 * SR             # the model's input in a locked pair (4 s) ...
LOCKED_TAIL = 3 * SR            # ... of which the last 3 s go to embed_g (contract 5.3 "Locked check")
LOCKED_ENR_S = 6.0              # B's enrollment rows hold at least 6.5 s (render.split_target_rows)

SETS = ["good", "short", "noisy", "two_talkers", "owner", "clipped"]
NOISY_BACKGROUNDS = ("steady", "recipe")   # noisy set: steady babble / synthetic noise (default), or the 8.3 draw
SET_CODE = {name: code for code, name in enumerate(SETS)}
PAIR_CODE = 6

# check -> (bad set, which side passes, rounding step). "higher": pass if v >= P, fail if v < F (thresholds.json
# stores [F, P]). "lower": pass if v < P, fail if v >= F (stored [P, F]).
CALIBRATED = {"sbr_db": ("noisy", "higher", 0.01),
              "one_voice": ("two_talkers", "higher", 0.001),
              "own_cos": ("owner", "lower", 0.001)}
DEFAULT_OUT = RUNS / "main" / "thresholds.json"
PROTECTED_RUNS = [RUNS / "main", RUNS / "smoke"]   # live training writes here


# ---------------------------------------------------------------------------------------------------------------
# The boundary rule (10-ml 13 "Rule per check")
# ---------------------------------------------------------------------------------------------------------------
def decimals(step):
    """Digits after the point for a rounding step (0.01 -> 2, 0.001 -> 3)."""
    return max(0, -int(math.floor(math.log10(step) + 1e-9)))


def on_grid(k, step):
    """The k-th multiple of step, rounded so that it prints cleanly (k = 599, step 0.01 -> 5.99)."""
    return round(k * step, decimals(step))


def needed(rate, count):
    """How many of `count` values a rate asks for (0.95 of 20 -> 19, 0.8 of 7 -> 6), rounded up."""
    return int(math.ceil(rate * count - 1e-9))


def boundaries(good, bad, side, step, pass_rate=PASS_RATE, fail_rate=FAIL_RATE):
    """Pass and fail boundaries of one check from good and bad values -> dict.

    side "higher" (sbr_db, one_voice): pass if v >= P, fail if v < F, warn in between.
      P = the highest multiple of `step` that at least pass_rate of the good values reach.
      F = the lowest multiple of `step` below which at least fail_rate of the bad values lie, but never above P.
    side "lower" (own_cos): pass if v < P, fail if v >= F.
      P = the lowest multiple of `step` that at least pass_rate of the good values stay under.
      F = the highest multiple of `step` that at least fail_rate of the bad values reach, but never below P.
    When F would cross P, F = P and met = False ("constraint not met"), unless F = P still fails enough bad values.
    Also returned: thresholds (the [low, high] pair as thresholds.json stores it), good_pass and bad_fail (the
    fractions reached), n_good, n_bad.
    """
    good = np.sort(np.asarray(good, dtype=np.float64))
    bad = np.sort(np.asarray(bad, dtype=np.float64))
    if len(good) == 0:
        raise ValueError("no good values: cannot set a pass boundary")
    need_pass = needed(pass_rate, len(good))
    need_fail = needed(fail_rate, len(bad))

    if side == "higher":
        def passes(values, p):
            return int(np.sum(values >= p))

        def fails(values, f):
            return int(np.sum(values < f))

        # P: round the value that leaves exactly need_pass good values at or above it down to the grid.
        k = int(math.floor(good[len(good) - need_pass] / step + 1e-9))
        while passes(good, on_grid(k, step)) < need_pass:       # guard against float rounding
            k -= 1
        p = on_grid(k, step)
        f, met = p, False
        if len(bad) > 0:
            # F: the first grid value above the need_fail-th smallest bad value.
            k = int(math.floor(bad[need_fail - 1] / step + 1e-9)) + 1
            while fails(bad, on_grid(k, step)) < need_fail:
                k += 1
            while fails(bad, on_grid(k - 1, step)) >= need_fail:  # lowest grid value that still works
                k -= 1
            f = on_grid(k, step)
            if f > p:                                            # would cross the pass boundary
                f = p
            met = fails(bad, f) >= need_fail
        pair = [f, p]
    elif side == "lower":
        def passes(values, p):
            return int(np.sum(values < p))

        def fails(values, f):
            return int(np.sum(values >= f))

        # P: the first grid value above the need_pass-th smallest good value.
        k = int(math.floor(good[need_pass - 1] / step + 1e-9)) + 1
        while passes(good, on_grid(k, step)) < need_pass:
            k += 1
        while passes(good, on_grid(k - 1, step)) >= need_pass:   # lowest grid value that still works
            k -= 1
        p = on_grid(k, step)
        f, met = p, False
        if len(bad) > 0:
            # F: the need_fail-th largest bad value rounded down to the grid.
            k = int(math.floor(bad[len(bad) - need_fail] / step + 1e-9))
            while fails(bad, on_grid(k, step)) < need_fail:
                k -= 1
            f = on_grid(k, step)
            if f < p:                                            # would cross the pass boundary
                f = p
            met = fails(bad, f) >= need_fail
        pair = [p, f]
    else:
        raise ValueError(f"side must be 'higher' or 'lower', got {side!r}")

    return {"pass": p, "fail": f, "met": bool(met), "thresholds": pair,
            "good_pass": passes(good, p) / len(good),
            "bad_fail": fails(bad, f) / len(bad) if len(bad) else float("nan"),
            "n_good": int(len(good)), "n_bad": int(len(bad))}


def locked_threshold(impostor, quantile=LOCKED_QUANTILE, step=0.001):
    """locked_cos from impostor cosines: the smallest grid value with at most 1 - quantile of them above it.

    With the values sorted, the ceil(0.99 M)-th smallest one leaves at most 1% strictly above; it is rounded up to
    the grid (rounding up can only lower the false accepts). Returns (value, fraction of impostors above it).
    """
    values = np.sort(np.asarray(impostor, dtype=np.float64))
    if len(values) == 0:
        raise ValueError("no impostor cosines: cannot set locked_cos")
    v = values[needed(quantile, len(values)) - 1]
    k = int(math.ceil(v / step - 1e-9))
    while on_grid(k, step) < v:
        k += 1
    value = on_grid(k, step)
    return value, float(np.mean(values > value))


# ---------------------------------------------------------------------------------------------------------------
# Simulated phone captures (validation speakers, noise and rooms)
# ---------------------------------------------------------------------------------------------------------------
def draw_room(src, rng):
    """The enrollment room of 10-ml 8.5 when there is no mixture: a random sim_val room (80%) or dry (20%)."""
    return render.pick_room(src, "sim" if rng.random() < ROOM_SIM_P else "dry", rng)


def bursts(src, rows, n, talk, rng):
    """n samples holding about `talk` samples of one talker, cut into pieces of 0.4 to 1.2 s with 0.5 to 1.5 s of
    silence between them (short set), so the speech span is longer than the speech in it."""
    stream = render.speech_stream(src, rows, int(talk) + SR, rng)     # the talking the pieces are cut from
    out = np.zeros(n, dtype=np.float32)
    pos, used = 0, 0
    while used < talk and pos < n:
        length = min(int(rng.uniform(*BURST_S) * SR), int(talk) - used, n - pos)
        if length <= 2 * render.FADE:
            break
        out[pos:pos + length] = dsp.fade(stream[used:used + length], render.FADE)
        used += length
        pos += length + int(rng.uniform(*BURST_GAP_S) * SR)
    return out


def stationary_noise(src, n, room, used, rng):
    """Noisy-set background (unit power): babble of 6 to 20 validation talkers (50%, positions 5-7 of the room) or
    the loader's synthetic noise, colored noise or mains hum (50%, no room). Both are steady, the kind of background
    sbr_db can measure (see the module notes)."""
    if rng.random() < 0.5:
        talkers = [render.pick_distinct(len(src.speakers), used, rng) for _ in range(int(rng.integers(6, 21)))]
        return render.set_power(render.babble_mix(src, talkers, n, room, rng), 1.0), "babble"
    return render.synthetic_noise(n, rng), "synthetic"


def scale_to(x, power):
    """x scaled so its active power (dsp.hop_vad hops) equals `power` (left alone if x is silent)."""
    p = dsp.active_power(x, dsp.hop_vad(x))
    return (x * np.float32(math.sqrt(power / p))).astype(np.float32) if p > 0 else x


def render_capture(src, kind, rng, person=None, rows=None, n=None, lead=None, talk=None, noisy_background="steady"):
    """One simulated full phone capture -> (pcm float32 [n], info dict).

    kind    - a set name; decides the SNR range, the second talker, the bursts and the clipping
    person  - speaker index of the person enrolled (drawn from the target pool when None)
    rows    - the rows that person talks from (all of their rows when None)
    n, lead - capture length and lead-in in samples (drawn when None)
    talk    - short set only: samples of talking
    noisy_background - noisy set only: "steady" (stationary_noise, the default) or "recipe" (the 8.3 noise draw
              that the good set uses)
    The phone owner is not part of the audio (make_capture draws the owner after the audio).
    """
    if n is None:
        n = int(round(rng.uniform(*CAPTURE_S) * SR))
    if lead is None:
        lead = int(round(rng.uniform(*LEAD_S) * SR))
    used = set()
    if person is None:
        person = render.pick_distinct(src.target_pool, used, rng)
    used.add(person)
    if rows is None:
        rows = src.rows_of[person]

    # The person's dry speech after the lead-in, placed in the room (position 0).
    dry = np.zeros(n, dtype=np.float32)
    if kind == "short":
        dry[lead:] = bursts(src, rows, n - lead, talk, rng)
    else:
        dry[lead:] = render.speech_stream(src, rows, n - lead, rng)
    room = draw_room(src, rng)
    clean = render.convolve(dry, room.h[0])
    p_clean = dsp.active_power(clean, dsp.hop_vad(clean))
    x = clean.copy()
    info = {"person": src.speakers[person], "n": n, "lead_s": round(lead / SR, 3), "room": room.kind,
            "room_index": room.index}

    # Two-talker set: someone else talking over the same stretch, 0 to 3 dB below (position 1 of the room).
    if kind == "two_talkers":
        other = render.pick_distinct(len(src.speakers), used, rng)
        second = np.zeros(n, dtype=np.float32)
        second[lead:] = render.speech_stream(src, src.rows_of[other], n - lead, rng)
        below_db = float(rng.uniform(*SECOND_TALKER_DB))
        x += scale_to(render.convolve(second, room.h[1]), p_clean * 10.0 ** (-below_db / 10.0))
        info.update(second=src.speakers[other], second_below_db=round(below_db, 3))

    # One noise clip over the whole capture at the drawn SNR against the clean speech.
    snr_db = float(rng.uniform(*(NOISY_SNR_DB if kind == "noisy" else GOOD_SNR_DB)))
    if kind == "noisy" and noisy_background == "steady":
        noise, noise_kind = stationary_noise(src, n, room, used, rng)
    else:
        noise, noise_kind = render.noise_clip(src, render.choose(rng, render.NOISE_KINDS, render.NOISE_PROBS), n,
                                              room, rng)
    x += render.set_power(noise, p_clean / 10.0 ** (snr_db / 10.0))
    info.update(snr_db=round(snr_db, 3), noise=noise_kind)

    # The phone's mic chain (10-ml 8.6 without clipping) and the headroom rule, then the converter's clip at +-1.
    chain = render.draw_mic_chain(rng, clipping=False, test_mode=False)
    x_f = render.chain_filters(chain, x)
    gain = render.headroom_gain(render.level_gain(chain.level_dbfs, x_f), x_f)
    pcm = np.clip(x_f * gain + render.self_noise(chain, n, rng), -1.0, 1.0).astype(np.float32)
    info["level_dbfs"] = round(chain.level_dbfs, 3)

    # Clipped set: gain so that 2% of the samples land past full scale, then the converter clips them.
    if kind == "clipped":
        c = float(np.quantile(np.abs(pcm), 1.0 - CLIPPED_FRACTION))
        if c > 0:
            pcm = np.clip(pcm / np.float32(c), -1.0, 1.0).astype(np.float32)
    return pcm, info


def render_short(src, seed, i):
    """Short-set capture i: retried (fresh draws) until es_vad of the whole capture holds 2.0 to 3.9 s of speech.

    The amount of talking is goal x factor, with a U(2.0, 3.9) s goal. After a miss the factor is multiplied by
    goal / measured, kept within [0.5, 1.5] (es_vad can also mark loud noise events, which would otherwise make the
    correction run away)."""
    goal = float(np.random.default_rng([seed, SET_CODE["short"], i]).uniform(*SHORT_SPEECH_S))
    factor = 1.0
    for t in range(SHORT_TRIES):
        rng = np.random.default_rng([seed, SET_CODE["short"], i, t])
        pcm, info = render_capture(src, "short", rng, talk=goal * factor * SR)
        got = quality.speech_seconds(es_vad(pcm))
        if SHORT_SPEECH_S[0] <= got <= SHORT_SPEECH_S[1]:
            break
        factor = min(1.5, max(0.5, factor * goal / max(got, 0.1)))
    info.update(short_goal_s=round(goal, 3), short_vad_s=round(got, 3), short_tries=t + 1,
                short_in_range=bool(SHORT_SPEECH_S[0] <= got <= SHORT_SPEECH_S[1]))
    return pcm, info, rng


class Owners:
    """WeSpeaker vectors of the phone owners' own clips (10-ml 6.3 recipe), computed once per (speaker, clip)."""

    def __init__(self, src, embed):
        self.src = src
        self.embed = embed
        self.cache = {}

    def e(self, spk, c):
        if (spk, c) not in self.cache:
            self.cache[(spk, c)] = np.asarray(self.embed(self.src.build_clip(spk, c)), dtype=np.float32)
        return self.cache[(spk, c)]


def make_capture(src, kind, seed, i, noisy_background="steady"):
    """Capture i of a set -> (pcm, info, owner speaker index, own clip index). noisy_background: see
    render_capture (only the noisy set uses it).

    The owner is drawn after the audio from the same generator: a different own-pool speaker for every set except
    owner, where the person is an own-pool speaker talking from their mix rows."""
    if kind == "short":
        pcm, info, rng = render_short(src, seed, i)
    elif kind == "owner":
        rng = np.random.default_rng([seed, SET_CODE[kind], i])
        owner = int(src.own_pool[rng.integers(len(src.own_pool))])
        pcm, info = render_capture(src, kind, rng, person=owner, rows=src.own_mix_rows[owner])
        return pcm, info, owner, int(rng.integers(OWN_CLIPS))
    else:
        rng = np.random.default_rng([seed, SET_CODE[kind], i])
        pcm, info = render_capture(src, kind, rng, noisy_background=noisy_background)
    person = src.speakers.index(info["person"])
    for _ in range(1000):
        owner = int(src.own_pool[rng.integers(len(src.own_pool))])
        if owner != person and src.speakers[owner] != info.get("second"):
            break
    return pcm, info, owner, int(rng.integers(OWN_CLIPS))


# ---------------------------------------------------------------------------------------------------------------
# Scoring one capture and judging it against a thresholds object
# ---------------------------------------------------------------------------------------------------------------
def score_capture(pcm, own_e, embed):
    """quality.score on the full capture as a target enrollment -> the values kept in the CSV."""
    s = quality.score(pcm, "target", embed=embed, own_e=own_e, own_window_cos=OWN_WINDOW_COS)
    return {"speech_s_full": round(s["speech_s_full"], 6), "sbr_db": s["sbr_db"], "sbr_fallback": s["sbr_fallback"],
            "clip_pct": s["clip_pct"], "trim_fail": s["fail"], "clip_len": s["clip_len"],
            "one_voice": s["one_voice"], "own_cos": s["own_cos"],
            "speech_s": None if s["speech_s"] is None else round(s["speech_s"], 6), "n_avail": s["n_avail"]}


def status(value, pair, side):
    """pass / warn / fail of one value against a thresholds.json [low, high] pair, or "not checked" (None)."""
    if value is None:
        return "not checked"
    low, high = pair
    if side == "higher":                       # pair = [fail below, pass at or above]
        return "pass" if value >= high else ("fail" if value < low else "warn")
    return "pass" if value < low else ("fail" if value >= high else "warn")   # pair = [pass below, fail at or above]


def verdicts(row, thresholds):
    """Every target check of contract 6 for one scored capture -> {check: status} plus "overall"."""
    t = thresholds["target"]
    out = {}
    if row["trim_fail"] in ("no speech", "too short"):
        out["speech_s"] = "fail"                # nothing to enroll
    else:
        out["speech_s"] = status(row["speech_s"], t["speech_s"], "higher")
    out["sbr_db"] = status(row["sbr_db"], t["sbr_db"], "higher")
    out["one_voice"] = status(row["one_voice"], t["one_voice"], "higher")
    out["own_cos"] = status(row["own_cos"], t["own_cos"], "lower")
    out["clip_pct"] = status(row["clip_pct"], t["clip_pct"], "lower")
    values = list(out.values())
    out["overall"] = "fail" if "fail" in values else ("warn" if "warn" in values else "pass")
    return out


# ---------------------------------------------------------------------------------------------------------------
# locked_cos: impostor pairs through the model
# ---------------------------------------------------------------------------------------------------------------
def trimmed_clip(pcm):
    """The enrollment clip of contract 6 steps 1-4 (target role), or None when the capture fails trimming."""
    t = quality.trim(pcm, "target")
    if t["fail"] is not None:
        return None
    return pcm[t["clip_start"]:t["clip_start"] + t["clip_len"]]


@torch.no_grad()
def separate(model, mix, tok, val, g, e):
    """Offline forward of one mixture in Person mode with only a target enrolled (no own voice, WeSpeaker on)."""
    device = next(model.parameters()).device
    zeros = torch.zeros
    ci = CondInputs(e_t=torch.from_numpy(e)[None].to(device), tok_t=tok[None], val_t=val[None], g_t=g[None],
                    e_o=zeros(1, 256, device=device), tok_o=zeros(1, 16, 64, device=device),
                    val_o=zeros(1, 16, dtype=torch.bool, device=device), g_o=zeros(1, 128, device=device),
                    has_t=torch.tensor([True], device=device), has_o=torch.tensor([False], device=device),
                    use_wes=torch.tensor([True], device=device),
                    mode=torch.tensor([[1.0, 0.0]], device=device))
    y, _ = model(torch.from_numpy(mix)[None].to(device), model.build_cond(ci))
    return y[0].float().cpu().numpy()


@torch.no_grad()
def locked_pair(model, src, embed, seed, i):
    """Impostor pair i -> CSV row with cos_impostor = cos(embed_g(output following B), g_A) and cos_genuine =
    cos(the same, g_B). skipped names the reason when a pair cannot be scored."""
    rng = np.random.default_rng([seed, PAIR_CODE, i])
    used = set()
    a = render.pick_distinct(src.target_pool, used, rng)
    b = render.pick_distinct(src.target_pool, used, rng)
    row = {"index": i, "a": src.speakers[a], "b": src.speakers[b], "skipped": "",
           "cos_impostor": None, "cos_genuine": None}

    # B's rows split as for a training target: enrollment rows E (>= 6.5 s) and the rest for the scene.
    _, enr_rows, scene_rows = render.split_target_rows(src, b, rng, LOCKED_MIX, int(LOCKED_ENR_S * SR))
    pcm_a, _ = render_capture(src, "good", rng, person=a)
    pcm_b, _ = render_capture(src, "good", rng, person=b, rows=enr_rows)
    scene, _ = render_capture(src, "good", rng, person=b, rows=scene_rows, n=LOCKED_MIX, lead=0)
    clip_a, clip_b = trimmed_clip(pcm_a), trimmed_clip(pcm_b)
    if clip_a is None or clip_b is None:
        row["skipped"] = "enrollment too short"
        return row
    _, _, g_a, n_a = model.enroll_clip(clip_a, es_vad(clip_a), MAX_T_TOK)
    tok_b, val_b, g_b, n_b = model.enroll_clip(clip_b, es_vad(clip_b), MAX_T_TOK)
    if n_a < 4 or n_b < 4:
        row["skipped"] = "fewer than 4 tokens"
        return row
    y = separate(model, scene, tok_b, val_b, g_b, np.asarray(embed(clip_b), dtype=np.float32))
    try:
        g_out = model.embed_g(y[-LOCKED_TAIL:])
    except ValueError:                       # ES_ERR_ENROLL_SHORT on the output
        row["skipped"] = "output has fewer than 4 tokens"
        return row
    g_out = g_out.cpu().numpy()
    row["cos_impostor"] = quality.cosine(g_out, g_a.cpu().numpy())
    row["cos_genuine"] = quality.cosine(g_out, g_b.cpu().numpy())
    return row


# ---------------------------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------------------------
def training_running():
    """True when `python -m ml.train` or run_forever.sh is running (pgrep on the full command lines)."""
    try:
        result = subprocess.run(["pgrep", "-f", r"ml\.train( |$)|run_forever\.sh"], capture_output=True, text=True,
                                timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def check_out_path(out):
    """Refuse an output inside a live run folder (ml/runs/main, ml/runs/smoke) while training runs."""
    out = Path(out).resolve()
    for run_dir in PROTECTED_RUNS:
        if run_dir.resolve() in out.parents and training_running():
            raise SystemExit(f"refused: ml.train is running and {out} is inside {run_dir}; wait for training to "
                             f"finish or pass --out somewhere else (for example ml/runs/calib_test/thresholds.json)")


def write_json(path, obj):
    """JSON with 2-space indent, written to path + '.tmp' and then renamed (never a half-written file)."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def write_csv(path, rows):
    """rows (dicts with possibly different keys) as one CSV; the columns are the union in first-seen order."""
    columns = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
    os.replace(tmp, path)


def fraction(rows, kind, check, value):
    """Share of the captures of a set whose verdict for `check` equals `value` (NaN for an empty set)."""
    picked = [r for r in rows if r["set"] == kind]
    return sum(r[f"{check}_status"] == value for r in picked) / len(picked) if picked else float("nan")


def clean_nan(obj):
    """NaN -> None everywhere (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate the enrollment check thresholds (10-ml 13, contract 6).")
    parser.add_argument("--esw", required=True, help="exported weights (model_hash and the locked-check model)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"thresholds.json to write (default {DEFAULT_OUT})")
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="captures per set (default 500)")
    parser.add_argument("--pairs", type=int, default=PAIRS_DEFAULT,
                        help="impostor pairs for locked_cos (default 1000; 0 keeps the contract value 0.0)")
    parser.add_argument("--seed", type=int, default=SEED, help="seed (default 2030)")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                        help="device for WeSpeaker and the model (default cpu; mps only when no training runs)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch (default 4)")
    parser.add_argument("--noisy-background", choices=NOISY_BACKGROUNDS, default="steady",
                        help="noisy-set background: steady babble / synthetic noise (default) or the 8.3 recipe "
                             "draw (see the module notes; not a DECISIONS row yet)")
    args = parser.parse_args(argv)
    if args.n < 1 or args.pairs < 0:
        parser.error("--n must be >= 1 and --pairs >= 0")
    if args.device == "mps" and training_running():
        parser.error("refused: ml.train is running; use --device cpu")
    out = Path(args.out)
    check_out_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    # Imported here so `--help` stays fast; ml.export pulls in the whole package.
    from ml.earshot.sources import get_sources
    from ml.earshot.wespeaker import WeSpeaker
    from ml.export import check_thresholds_shape, initial_thresholds, load_esw

    t_start = time.perf_counter()
    weights_hash = model_hash(args.esw)
    model, _ = load_esw(args.esw)
    model = model.to(args.device).eval().requires_grad_(False)
    embed = WeSpeaker.load().to(args.device).embed_numpy
    src = get_sources("val")
    owners = Owners(src, embed)
    print(f"model_hash {weights_hash}; {len(src.speakers)} validation speakers, {len(src.own_pool)} in the own pool; "
          f"{args.n} captures per set, {args.pairs} impostor pairs, seed {args.seed}, device {args.device}, "
          f"noisy-set background {args.noisy_background}")

    # 1. Every capture of every set, scored with quality.py.
    rows, seconds = [], {}
    for kind in SETS:
        t0 = time.perf_counter()
        for i in range(args.n):
            pcm, info, owner, clip = make_capture(src, kind, args.seed, i, args.noisy_background)
            row = {"set": kind, "index": i, **info, "owner": src.speakers[owner], "own_clip": clip}
            row.update(score_capture(pcm, owners.e(owner, clip), embed))
            rows.append(row)
        seconds[kind] = time.perf_counter() - t0
        print(f"  {kind:12s} {args.n} captures in {seconds[kind]:.1f} s")

    # 2. The three calibrated checks.
    thresholds = initial_thresholds(weights_hash)
    checks = {}
    for check, (bad_set, side, step) in CALIBRATED.items():
        good = [r[check] for r in rows if r["set"] == "good" and r[check] is not None]
        bad = [r[check] for r in rows if r["set"] == bad_set and r[check] is not None]
        if good:
            result = boundaries(good, bad, side, step)
        else:                                   # nothing to calibrate on (only possible in a tiny run)
            print(f"  {check}: no good capture has a value; keeping the contract value")
            pair = thresholds["target"][check]
            result = {"pass": None, "fail": None, "met": False, "thresholds": pair, "good_pass": float("nan"),
                      "bad_fail": float("nan"), "n_good": 0, "n_bad": len(bad)}
        result.update(bad_set=bad_set, side=side, step=step, contract=thresholds["target"][check])
        checks[check] = result
        thresholds["target"][check] = result["thresholds"]
        print(f"  {check:9s} {result['thresholds']} (contract {result['contract']}): good pass "
              f"{100 * result['good_pass']:.1f}% of {result['n_good']}, {bad_set} fail "
              f"{100 * result['bad_fail']:.1f}% of {result['n_bad']}")
        if not result["met"]:
            print(f"  {check}: constraint not met (fewer than {100 * FAIL_RATE:.0f}% of the {bad_set} set fail "
                  f"even at the pass boundary)")

    # 3. locked_cos from impostor pairs.
    t0 = time.perf_counter()
    pairs = [locked_pair(model, src, embed, args.seed, i) for i in range(args.pairs)]
    seconds["pairs"] = time.perf_counter() - t0
    impostor = [p["cos_impostor"] for p in pairs if p["cos_impostor"] is not None]
    genuine = [p["cos_genuine"] for p in pairs if p["cos_genuine"] is not None]
    locked = {"pairs": args.pairs, "scored": len(impostor), "skipped": args.pairs - len(impostor)}
    if impostor:
        value, false_accepts = locked_threshold(impostor)
        thresholds["locked_cos"] = value
        locked.update(locked_cos=value, impostor_above=false_accepts,
                      genuine_above=float(np.mean(np.asarray(genuine) > value)),
                      impostor_median=float(np.median(impostor)), genuine_median=float(np.median(genuine)))
        print(f"  locked_cos {value} from {len(impostor)} impostor pairs ({100 * false_accepts:.2f}% above it; "
              f"{100 * locked['genuine_above']:.1f}% of genuine pairs above it), {seconds['pairs']:.1f} s")
    else:
        locked["locked_cos"] = thresholds["locked_cos"]
        print(f"  locked_cos stays {thresholds['locked_cos']} (no impostor pair scored)")
    check_thresholds_shape(thresholds)

    # 4. Verdicts of every capture under the final thresholds, then the files.
    for row in rows:
        row.update({f"{k}_status": v for k, v in verdicts(row, thresholds).items()})
    fixed = {"short_speech_s_fail": fraction(rows, "short", "speech_s", "fail"),
             "short_in_vad_range": (sum(bool(r.get("short_in_range")) for r in rows if r["set"] == "short")
                                    / max(1, sum(r["set"] == "short" for r in rows))),
             "clipped_clip_pct_fail": fraction(rows, "clipped", "clip_pct", "fail"),
             "good_overall_pass": fraction(rows, "good", "overall", "pass"),
             "overall_fail_by_set": {kind: fraction(rows, kind, "overall", "fail") for kind in SETS}}
    total_s = time.perf_counter() - t_start
    per_capture = {kind: seconds[kind] / args.n for kind in SETS}
    per_pair = seconds["pairs"] / args.pairs if args.pairs else 0.0
    full_estimate = sum(per_capture.values()) * N_DEFAULT + per_pair * PAIRS_DEFAULT
    summary = {"model_hash": weights_hash, "esw": str(args.esw), "seed": args.seed, "n_per_set": args.n,
               "device": args.device, "threads": args.threads, "thresholds": thresholds, "checks": checks,
               "noisy_background": args.noisy_background,
               "pending_decisions": "the noisy-set background (noisy_background) sets the sbr_db fail boundary and "
                                    "is a WP-ML reading of 10-ml 13, not yet a DECISIONS L row",
               "locked": locked, "fixed_rules": fixed, "seconds": {**seconds, "total": total_s},
               "seconds_per_capture": per_capture, "seconds_per_pair": per_pair,
               "full_run_estimate_s": full_estimate,
               "note": "full_run_estimate_s = per-capture and per-pair times of this run x 500 per set and 1,000 "
                       "pairs (an estimate; own-clip vectors are cached, so a full run spends a little less per "
                       "capture)"}
    write_json(out, thresholds)
    write_csv(out.parent / "calibration_scores.csv", rows)
    write_csv(out.parent / "calibration_pairs.csv", pairs if pairs else [{"index": ""}])
    write_json(out.parent / "calibration_summary.json", clean_nan(summary))
    print(f"  short set: {100 * fixed['short_speech_s_fail']:.1f}% fail speech_s (fixed [4, 5]); clipped set: "
          f"{100 * fixed['clipped_clip_pct_fail']:.1f}% fail clip_pct (fixed [0.1, 1.0]); good set: "
          f"{100 * fixed['good_overall_pass']:.1f}% pass every check")
    print(f"wrote {out}, calibration_scores.csv, calibration_pairs.csv, calibration_summary.json in {out.parent}")
    print(f"took {total_s:.1f} s; a full run (500 per set, 1,000 pairs, this device and thread count) would take "
          f"about {full_estimate / 60:.1f} min")
    return thresholds


if __name__ == "__main__":
    main()
