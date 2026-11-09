"""DiPCo real dinner-party check: data, enrollment, chunks, scoring windows, listening clips (10-ml 12.5).

On disk (data/manifests/eval_dipco.jsonl, CDLA-Permissive-1.0): one far-field channel of sessions S01, S03, S06,
S07, S08 and close-talk files only for S03 P09 and S08 P25, P26, P27, so the targets are those 4 people. Segment
times are in data/eval/dipco_segments.jsonl; far-field and close-talk times are the same clock (checked: every
close_start equals its far_start).

- Enrollment - the target's close-talk segments that start and end in the first 5 minutes and overlap no other
  participant's segment, in time order, joined with 0.2 s gaps until es_vad counts 6 s of speech (at most 10 s of
  audio), scaled to -26 dBFS RMS, es_vad mask. The span from the first to the last used segment is excluded from
  scoring. No own voice (nobody at a DiPCo table owns the phone).
- Input - the session's far-field channel scaled to -26 dBFS RMS (one gain for the whole file), run through the
  offline model in 60 s chunks, each with the 8 s before it as warm-up (discarded).
- Absent-target windows - 2 s windows (on a 2 s grid) where the target has no segment within 0.25 s and the
  other participants (union of their segments) talk at least 1 s: attenuation 10 log10(sum x^2 / sum y^2).
- Envelope windows - 2 s windows where the target talks at least 1 s and at least one other participant with a
  close-talk file talks at least 0.5 s (only S08 has such pairs): a win if env_corr(y, close-talk of the target)
  beats env_corr(y, close-talk of j) for every such participant j.
- Listening - 10 clips of 10 s (input and output WAVs) from "mixed" windows (target and others both talk at
  least 2 s), spread over the 4 targets, plus clips.csv with each clip's numbers.
Music plays from a fixed time in every session (S08 from 62 s, S03 from 2,025 s); 12.5 does not exclude it, so
neither does this module.
"""
import csv
import math
import re

import numpy as np
import soundfile as sf

from dataprep.common import read_manifest   # frozen data-pack code, imported read-only

from . import metrics
from .config import DATA, MAX_T_TOK, SR
from .enroll import groups_or_ones
from .sources import MANIFESTS, read_checked, read_part, read_row
from .vad import TOK_HOP, es_vad

TARGETS = [("S03", "dipco:P09"), ("S08", "dipco:P25"), ("S08", "dipco:P26"), ("S08", "dipco:P27")]
ENR_FIRST_S = 300.0             # enrollment segments from the first 5 minutes
ENR_GAP_S = 0.2
ENR_SPEECH_S = 6.0              # stop once es_vad counts 6 s of speech
ENR_MAX = 10 * SR               # WeSpeaker and the token encoder take at most 10 s ...
ENR_MIN = 1 * SR                # ... and WeSpeaker at least 1 s
LEVEL_DBFS = -26.0
CHUNK = 60 * SR                 # 60 s chunks ...
WARM = 8 * SR                   # ... each with 8 s of warm-up before it
WIN_S = 2.0                     # scoring windows
GUARD_S = 0.25                  # absent: no target segment within 0.25 s of the window
OTHERS_MIN_S = 1.0              # absent: others talk at least 1 s
TARGET_MIN_S = 1.0              # envelope: target talks at least 1 s ...
OTHER_CLOSE_MIN_S = 0.5         # ... and a close-talk participant at least 0.5 s
LISTEN_N, LISTEN_S, LISTEN_MIN_S = 10, 10.0, 2.0


def load():
    """-> {session: {"far": path, "close": {speaker: path}, "segments": {speaker: [(start_s, end_s)]}}}.
    Manifest rows go through the license gate (split test)."""
    sessions = {}
    for row in read_checked(MANIFESTS / "eval_dipco.jsonl", "test"):
        ses = sessions.setdefault(row["group"], {"far": None, "close": {}, "segments": {}})
        if row["extra"]["kind"] == "far-field":
            ses["far"] = row["path"]
        else:
            ses["close"][row["speaker"]] = row["path"]
    for seg in read_manifest(DATA / "eval" / "dipco_segments.jsonl"):
        ses = sessions.get(seg["session"])
        if ses is not None:
            ses["segments"].setdefault(seg["speaker"], []).append((float(seg["far_start"]), float(seg["far_end"])))
    for ses in sessions.values():
        for spk in ses["segments"]:
            ses["segments"][spk].sort()
    return sessions


def talk_seconds(intervals, lo, hi):
    """Seconds of [lo, hi) covered by the union of the (start, end) intervals."""
    clipped = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in clipped:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def others_of(ses, target):
    """All segments of every participant except the target, as one list."""
    return [iv for spk, ivs in ses["segments"].items() if spk != target for iv in ivs]


def scale_to_dbfs(x, dbfs=LEVEL_DBFS):
    """x scaled so its RMS over all samples is `dbfs` (left alone if silent)."""
    rms = math.sqrt(float(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if len(x) else 0.0
    return (x * np.float32(10.0 ** (dbfs / 20.0) / rms)).astype(np.float32) if rms > 0 else x


def enrollment(ses, target):
    """The target's enrollment (module docstring) -> dict enr f32, tok_idx, tok_val, short, span (s, s), segments."""
    others = others_of(ses, target)
    clean = [(a, b) for a, b in ses["segments"][target]
             if b <= ENR_FIRST_S and not any(oa < b and ob > a for oa, ob in others)]
    pieces, used = [], []
    enr = np.zeros(0, dtype=np.float32)
    for a, b in clean:
        piece = read_part(ses["close"][target], int(round(a * SR)), int(round((b - a) * SR)))
        if pieces:
            pieces.append(np.zeros(int(ENR_GAP_S * SR), dtype=np.float32))
        pieces.append(piece)
        used.append((a, b))
        enr = np.concatenate(pieces)[:ENR_MAX]
        if np.sum(es_vad(enr)) * TOK_HOP / SR >= ENR_SPEECH_S or len(enr) >= ENR_MAX:
            break
    if not used:
        raise ValueError(f"{target}: no clean close-talk segment in the first 5 minutes")
    if len(enr) < ENR_MIN:                          # WeSpeaker takes at least 1 s: pad a very short one with zeros
        enr = np.concatenate([enr, np.zeros(ENR_MIN - len(enr), dtype=np.float32)])
    enr = scale_to_dbfs(enr)
    idx, valid, short = groups_or_ones(es_vad(enr), MAX_T_TOK)
    return {"enr": enr, "tok_idx": idx, "tok_val": valid, "short": short, "span": (used[0][0], used[-1][1]),
            "segments": len(used), "speech_s": round(float(np.sum(es_vad(enr)) * TOK_HOP / SR), 2)}


def far_input(ses):
    """The session's far-field channel, cut to a multiple of 64 samples, scaled to -26 dBFS RMS."""
    x = read_row(ses["far"])
    return scale_to_dbfs(x[:(len(x) // 64) * 64])


def chunk_plan(n, first=0, count=None):
    """[(in_lo, out_lo, out_hi)] sample ranges: output [out_lo, out_hi) is kept from a forward over
    [in_lo, out_hi), in_lo = out_lo - 8 s (0 for the first chunk). Chunks first .. first + count - 1."""
    plan = []
    for k, out_lo in enumerate(range(0, n, CHUNK)):
        if k >= first and (count is None or k < first + count):
            plan.append((max(0, out_lo - WARM), out_lo, min(n, out_lo + CHUNK)))
    return plan


def overlaps(lo, hi, span):
    return lo < span[1] and hi > span[0]


def absent_windows(ses, target, lo_s, hi_s, span):
    """Start times (s) of the absent-target windows inside [lo_s, hi_s), outside the enrollment span."""
    mine, others = ses["segments"][target], others_of(ses, target)
    out = []
    for k in range(int(math.ceil(lo_s / WIN_S)), int(hi_s // WIN_S)):
        t0 = k * WIN_S
        if t0 + WIN_S > hi_s or overlaps(t0, t0 + WIN_S, span):
            continue
        if talk_seconds(mine, t0 - GUARD_S, t0 + WIN_S + GUARD_S) > 0.0:
            continue
        if talk_seconds(others, t0, t0 + WIN_S) >= OTHERS_MIN_S:
            out.append(t0)
    return out


def envelope_windows(ses, target, lo_s, hi_s, span):
    """[(start s, [other close-talk speakers talking >= 0.5 s])] of the envelope windows (empty without pairs)."""
    rivals = [spk for spk in ses["close"] if spk != target]
    out = []
    for k in range(int(math.ceil(lo_s / WIN_S)), int(hi_s // WIN_S)):
        t0 = k * WIN_S
        if t0 + WIN_S > hi_s or overlaps(t0, t0 + WIN_S, span):
            continue
        if talk_seconds(ses["segments"][target], t0, t0 + WIN_S) < TARGET_MIN_S:
            continue
        talking = [j for j in rivals if talk_seconds(ses["segments"].get(j, []), t0, t0 + WIN_S) >= OTHER_CLOSE_MIN_S]
        if talking:
            out.append((t0, talking))
    return out


def envelope_win(y_w, close_t_w, close_others_w):
    """(win, corr to the target, best corr to another participant) for one window."""
    c_t = metrics.env_corr(y_w, close_t_w)
    c_o = [metrics.env_corr(y_w, c) for c in close_others_w]
    return bool(all(c_t > c for c in c_o)), c_t, max(c_o)


def listen_candidates(ses, target, lo_s, hi_s, span):
    """Start times of 10 s "mixed" windows (target and others both talk >= 2 s) inside [lo_s, hi_s)."""
    mine, others = ses["segments"][target], others_of(ses, target)
    out = []
    for k in range(int(math.ceil(lo_s / LISTEN_S)), int(hi_s // LISTEN_S)):
        t0 = k * LISTEN_S
        if t0 + LISTEN_S > hi_s or overlaps(t0, t0 + LISTEN_S, span):
            continue
        if (talk_seconds(mine, t0, t0 + LISTEN_S) >= LISTEN_MIN_S
                and talk_seconds(others, t0, t0 + LISTEN_S) >= LISTEN_MIN_S):
            out.append(t0)
    return out


def pick_listening(candidates, n=LISTEN_N):
    """candidates = {target: [start s]} -> up to n (target, start s) picks.

    The n picks are handed out one per target in turn (a target with no candidates left is skipped), then each
    target's picks are spread evenly over its candidates in time. Deterministic.
    """
    targets = [t for t, c in candidates.items() if c]
    quota = {t: 0 for t in targets}
    left = n
    while left > 0 and any(quota[t] < len(candidates[t]) for t in targets):
        for t in targets:
            if left > 0 and quota[t] < len(candidates[t]):
                quota[t] += 1
                left -= 1
    picks = []
    for t in targets:
        c, q = candidates[t], quota[t]
        picks += [(t, c[(j * len(c)) // q]) for j in range(q)]
    return picks


CLIP_NAME = re.compile(r"^\d\d_P\d\d_\d+s_(in|out)\.wav$")     # e.g. 03_P25_290s_in.wav


def write_listening(out_dir, clips):
    """Write <k>_<target>_<start>s_in.wav / _out.wav (float WAV, 16 kHz) and clips.csv, after removing the clip
    WAVs an earlier run left there. clips = list of dicts with x, y (numpy) and the CSV numbers. Returns the CSV
    path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():                     # clips of an earlier run (exactly this file name pattern)
        if CLIP_NAME.match(old.name):
            old.unlink()
    rows = []
    for k, c in enumerate(clips):
        stem = f"{k:02d}_{c['target'].replace('dipco:', '')}_{int(c['start_s'])}s"
        sf.write(str(out_dir / f"{stem}_in.wav"), c["x"], SR, subtype="FLOAT")
        sf.write(str(out_dir / f"{stem}_out.wav"), c["y"], SR, subtype="FLOAT")
        rows.append({"clip": stem, **{key: v for key, v in c.items() if key not in ("x", "y")}})
    path = out_dir / "clips.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["clip"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def dbfs(x):
    """RMS level in dBFS (-inf for silence)."""
    p = float(np.mean(np.asarray(x, dtype=np.float64) ** 2)) if len(x) else 0.0
    return 10.0 * math.log10(p) if p > 0 else float("-inf")

