"""Enrollment check formulas of contract 6, written once and shared by threshold calibration and the quality vectors.

The app's ClipTools.swift and EnrollmentService implement the same steps; `quality_vectors.json` (made by
`python -m ml.export`) holds 5 fixed captures with the values computed here, and the app's ClipToolsTests and
ml/tests/test_metrics.py check them. x is always the full 16 kHz capture of n samples (float32).

Token frame f (contract 2.4 framing, hop 512 after 448 zeros of padding) covers raw samples [512f - 448, 512f + 64)
and its midpoint is 512f - 192. mask0 = es_vad(x).

Steps (contract 6 "Check formulas"):
 1-2. span       - first to last frame with mask0 = 1: [max(0, 512 f1 - 448), min(n, 512 f2 + 64))
 3.   10 s cap   - longer than 10 s: keep the 10 s window (starts on 512-sample steps) holding the most mask0 frame
                   midpoints, earliest on ties. Own voice always takes this window.
 4.   rounding   - length rounded down to a multiple of 8,000 samples (0.5 s), start kept; under 64,000 fails.
 5.   windows    - WeSpeaker windows start 8000 w, length 24,000, w = 0 .. floor((len - 24000) / 8000); a clip frame
                   is inside window w when its midpoint lies in [8000 w, 8000 w + 24000).
 6.   sbr_db     - 10 log10(mean x^2 over mask0 speech frames / mean x^2 over the other frames), full capture.
 7.   clip_pct   - 100 x count(|x| >= 0.98) / n, full capture.
 8.   one_voice, own_cos, dup - cosines of WeSpeaker vectors (need an embed function).
 9.   mask       - targets: es_vad(clip) with every frame inside a window whose cos(e_w, own e) > own_window_cos set
                   to 0; own voice: es_vad(clip).
 10.  speech_s   - kept frames x 0.032 (es_voiceprint_speech_seconds), after es_enroll with that mask.

Points the contract leaves open, decided here (WP-ML choices, written into quality_vectors.json "definitions"):
- 10 s window starts: region_start + 512 i, where the region is the speech span for a target and the whole capture
  for own voice ("own voice always takes this window"); a window must lie inside its region. An own-voice capture of
  10 s or less is used whole.
- sbr "mean x^2 over frames": sum of x^2 over the raw samples [max(0, 512f - 448), 512f + 64) of those frames,
  divided by the number of those samples (padding zeros are not samples). Both means get + 1e-12 so digital silence
  gives a large finite number instead of infinity.
- sbr fallback (no frame with mask0 = 0): the other frames are the max(1, F // 10) frames with the lowest es_vad
  energy E_f (stable sort, so ties go to the lower frame index); the speech frames are all the rest.
"""
import numpy as np

from .vad import es_vad, frame_energies_db

FRAME_HOP = 512          # token frame hop and length
PAD = 448                # left padding of the token framing
CLIP_MAX = 160000        # 10 s
ROUND_STEP = 8000        # 0.5 s
MIN_CLIP = 64000         # 4 s
WIN_LEN = 24000          # 1.5 s WeSpeaker windows
WIN_HOP = 8000           # every 0.5 s
CLIP_LEVEL = 0.98        # a sample counts as clipped at |x| >= 0.98
FRAME_SECONDS = 0.032    # speech seconds per kept frame
POWER_FLOOR = 1e-12      # added to both sbr means


# ---------------------------------------------------------------------------------------------------------------
# Frame map (step 1)
# ---------------------------------------------------------------------------------------------------------------
def frame_samples(f, n):
    """Raw sample range [a, b) of token frame f in a capture of n samples (padding removed)."""
    return max(0, FRAME_HOP * f - PAD), min(n, FRAME_HOP * f + 64)


def frame_mid(f):
    """Midpoint of token frame f in raw samples (may be negative for f = 0)."""
    return FRAME_HOP * f - 192


# ---------------------------------------------------------------------------------------------------------------
# Trimming (steps 2 to 4)
# ---------------------------------------------------------------------------------------------------------------
def frame_span(mask0, n):
    """Step 2: (start, end) samples from the first to the last frame with mask0 = 1, or None if there is none."""
    kept = np.flatnonzero(np.asarray(mask0))
    if len(kept) == 0:
        return None
    f1, f2 = int(kept[0]), int(kept[-1])
    return max(0, FRAME_HOP * f1 - PAD), min(n, FRAME_HOP * f2 + 64)


def best_window(mask0, region_start, region_end, length=CLIP_MAX):
    """Step 3: start of the `length`-sample window inside [region_start, region_end) holding the most mask0 frame
    midpoints. Candidate starts are region_start + 512 i; the earliest start wins a tie.

    The caller only uses this when the region is longer than `length`.
    """
    mids = [frame_mid(f) for f in np.flatnonzero(np.asarray(mask0))]
    best_start, best_count = region_start, -1
    start = region_start
    while start + length <= region_end:
        count = 0
        for m in mids:
            if start <= m < start + length:
                count += 1
        if count > best_count:            # strictly more, so the earliest start keeps a tie
            best_start, best_count = start, count
        start += FRAME_HOP
    return best_start


def round_length(length):
    """Step 4: round a length down to a multiple of 8,000 samples (0.5 s)."""
    return length - length % ROUND_STEP


def trim(x, role, mask0=None):
    """Steps 1 to 4 for one capture -> dict:
    span (start, end) or None, region (start, end) searched for the 10 s window, window_start (or None when no
    window was needed), clip_start, clip_len (after rounding), fail (None, "no speech" or "too short").

    role - "target" or "own". mask0 may be passed if es_vad(x) is already known.
    """
    if role not in ("target", "own"):
        raise ValueError(f"role must be 'target' or 'own', got {role!r}")
    n = len(x)
    if mask0 is None:
        mask0 = es_vad(x)
    span = frame_span(mask0, n)
    out = {"span": span, "region": None, "window_start": None, "clip_start": None, "clip_len": 0, "fail": None}
    if span is None:
        out["fail"] = "no speech"
        return out

    region = span if role == "target" else (0, n)
    out["region"] = region
    start, end = region
    if end - start > CLIP_MAX:
        start = best_window(mask0, region[0], region[1])
        out["window_start"] = start
        end = start + CLIP_MAX
    out["clip_start"] = start
    out["clip_len"] = round_length(end - start)
    if out["clip_len"] < MIN_CLIP:
        out["fail"] = "too short"
    return out


# ---------------------------------------------------------------------------------------------------------------
# WeSpeaker windows (step 5)
# ---------------------------------------------------------------------------------------------------------------
def windows(clip_len):
    """Step 5: list of (start, length) WeSpeaker windows of a clip; empty if the clip is shorter than 1.5 s."""
    if clip_len < WIN_LEN:
        return []
    return [(WIN_HOP * w, WIN_LEN) for w in range((clip_len - WIN_LEN) // WIN_HOP + 1)]


def frames_inside(window_start, n_frames, length=WIN_LEN):
    """Step 5: (first, last) clip frame indices inside the window, inclusive, i.e. midpoints in
    [window_start, window_start + length); None if no frame is inside."""
    inside = [f for f in range(n_frames) if window_start <= frame_mid(f) < window_start + length]
    if not inside:
        return None
    return inside[0], inside[-1]


# ---------------------------------------------------------------------------------------------------------------
# Level checks on the full capture (steps 6 and 7)
# ---------------------------------------------------------------------------------------------------------------
def frames_power(x, frames):
    """Mean of x^2 over the raw samples of the given token frames (float64)."""
    x = np.asarray(x, dtype=np.float64)
    total, count = 0.0, 0
    for f in frames:
        a, b = frame_samples(int(f), len(x))
        total += float(np.sum(x[a:b] ** 2))
        count += b - a
    return total / count if count > 0 else 0.0


def sbr_split(x, mask0=None):
    """Which frames count as speech and which as "other" for sbr -> (speech frames, other frames, fallback used).

    Normally speech = mask0 frames, other = the rest. With no frame at mask0 = 0 the other frames are the
    max(1, F // 10) frames with the lowest es_vad energy (stable sort) and speech is all the rest.
    """
    if mask0 is None:
        mask0 = es_vad(x)
    mask0 = np.asarray(mask0)
    # Any nonzero mask byte counts as speech (DECISIONS L 2026-11-07 enrollment mask rule, same as the C engine's
    # es_token_groups and vad.token_groups); es_vad masks are 0/1, so the values do not change.
    speech = np.flatnonzero(mask0 != 0)
    other = np.flatnonzero(mask0 == 0)
    if len(other) > 0 or len(mask0) == 0:
        return speech, other, False
    energies = frame_energies_db(x)
    n_low = max(1, len(energies) // 10)
    lowest = np.sort(np.argsort(energies, kind="stable")[:n_low])
    low_set = set(lowest.tolist())
    speech = np.array([f for f in range(len(energies)) if f not in low_set], dtype=np.int64)
    return speech, lowest, True


def sbr_db(x, mask0=None):
    """Step 6: speech-to-background ratio in dB on the full capture, or None when there is no speech frame."""
    speech, other, _ = sbr_split(x, mask0)
    if len(speech) == 0:
        return None
    p_speech = frames_power(x, speech)
    p_other = frames_power(x, other)
    return float(10.0 * np.log10((p_speech + POWER_FLOOR) / (p_other + POWER_FLOOR)))


def clip_pct(x):
    """Step 7: percentage of samples with |x| >= 0.98 (compared in float32, like the app)."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return 0.0
    count = int(np.count_nonzero(np.abs(x) >= np.float32(CLIP_LEVEL)))
    return 100.0 * count / len(x)


# ---------------------------------------------------------------------------------------------------------------
# Mask and speech seconds (steps 9 and 10)
# ---------------------------------------------------------------------------------------------------------------
def cosine(a, b):
    """Cosine similarity in float64."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def own_window_mask(mask, window_list, window_cos_own, threshold):
    """Step 9 (targets): copy of `mask` with every frame inside a window w with window_cos_own[w] > threshold set to
    0. window_list is the output of windows(); window_cos_own[w] = cos(e_w, own e)."""
    out = np.array(mask, dtype=np.uint8, copy=True)
    for (start, length), c in zip(window_list, window_cos_own):
        if c > threshold:
            for f in range(len(out)):
                if start <= frame_mid(f) < start + length:
                    out[f] = 0
    return out


def speech_seconds(mask):
    """Step 10: es_voiceprint_speech_seconds = kept frames x 0.032 (kept = frames with mask = 1)."""
    return int(np.count_nonzero(np.asarray(mask))) * FRAME_SECONDS


# ---------------------------------------------------------------------------------------------------------------
# The whole check for one capture
# ---------------------------------------------------------------------------------------------------------------
def score(x, role, embed=None, own_e=None, saved_e=(), own_window_cos=0.60):
    """Run every contract 6 step on one full capture -> dict of results (used by calibration and the vectors).

    x        - float32 [n], the full 16 kHz capture
    role     - "target" or "own" (own voice runs steps 1, 3 and 8 of the app's list: sbr, trim, speech_s)
    embed    - optional function float32 clip -> WeSpeaker vector [256]; without it one_voice, own_cos and dup are
               "not checked" (None) and the mask keeps es_vad(clip) (the no-WeSpeaker rule of contract 6)
    own_e    - the user's own WeSpeaker vector (targets only), or None
    saved_e  - WeSpeaker vectors of saved and tonight people (targets only)

    Keys: mask0_frames, speech_s_full, sbr_db, sbr_fallback, clip_pct, span, region, window_start, clip_start,
    clip_len, fail, windows, clip_frames, frames_inside, vad_frames (es_vad(clip) kept count), e, one_voice,
    own_cos, dup, window_cos_own, mask, speech_frames, speech_s, n_avail.
    fail is None, "no speech", "too short" or "ES_ERR_ENROLL_SHORT" (fewer than 4 tokens after the mask).
    """
    x = np.asarray(x, dtype=np.float32)
    mask0 = es_vad(x)
    out = {"role": role, "n": len(x), "mask0_frames": int(np.count_nonzero(mask0)),
           "speech_s_full": speech_seconds(mask0)}
    out["sbr_db"] = sbr_db(x, mask0)
    out["sbr_fallback"] = sbr_split(x, mask0)[2]
    out["clip_pct"] = clip_pct(x)
    out.update(trim(x, role, mask0))
    for key in ["windows", "clip_frames", "frames_inside", "vad_frames", "e", "one_voice", "own_cos", "dup",
                "window_cos_own", "mask", "speech_frames", "speech_s", "n_avail"]:
        out[key] = None
    if out["fail"] is not None:
        return out

    clip = x[out["clip_start"]:out["clip_start"] + out["clip_len"]]
    wins = windows(len(clip))
    mask = es_vad(clip)
    out["windows"] = wins
    out["clip_frames"] = len(mask)
    out["frames_inside"] = [frames_inside(start, len(mask)) for start, _ in wins]
    out["vad_frames"] = int(np.count_nonzero(mask))

    if embed is not None:
        e = np.asarray(embed(clip), dtype=np.float32)
        out["e"] = e
        if role == "target":
            e_w = [np.asarray(embed(clip[s:s + n]), dtype=np.float32) for s, n in wins]
            if e_w:
                out["one_voice"] = float(np.median([cosine(ew, e) for ew in e_w]))
            if own_e is not None:
                out["own_cos"] = cosine(e, own_e)
                out["window_cos_own"] = [cosine(ew, own_e) for ew in e_w]
                mask = own_window_mask(mask, wins, out["window_cos_own"], own_window_cos)
            if len(saved_e) > 0:
                out["dup"] = max(cosine(e, s) for s in saved_e)

    out["mask"] = mask
    out["speech_frames"] = int(np.count_nonzero(mask))
    out["speech_s"] = speech_seconds(mask)
    out["n_avail"] = out["speech_frames"] // 3
    if out["n_avail"] < 4:
        out["fail"] = "ES_ERR_ENROLL_SHORT"
    return out


def vector_values(x):
    """The weight-independent values stored in quality_vectors.json for one capture (no WeSpeaker), as plain JSON
    types: shared full-capture values plus one block per role."""
    def r6(value):
        """Speech seconds rounded to 6 decimals (kept * 0.032 is exact to 3 decimals; this drops float noise)."""
        return None if value is None else round(value, 6)

    shared = None
    roles = {}
    for role in ("target", "own"):
        s = score(x, role)
        if shared is None:
            shared = {"n": s["n"], "mask0_frames": s["mask0_frames"], "speech_s_full": r6(s["speech_s_full"]),
                      "sbr_db": s["sbr_db"], "sbr_fallback": s["sbr_fallback"], "clip_pct": s["clip_pct"],
                      "span": list(s["span"]) if s["span"] else None}
        roles[role] = {
            "region": list(s["region"]) if s["region"] else None,
            "window_start": s["window_start"],
            "clip_start": s["clip_start"],
            "clip_len": s["clip_len"],
            "fail": s["fail"],
            "windows": [list(w) for w in s["windows"]] if s["windows"] is not None else None,
            "clip_frames": s["clip_frames"],
            "frames_inside": ([list(r) if r else None for r in s["frames_inside"]]
                              if s["frames_inside"] is not None else None),
            "speech_frames": s["speech_frames"],
            "speech_s": r6(s["speech_s"]),
            "n_avail": s["n_avail"],
        }
    shared.update(roles)
    return shared


DEFINITIONS = {
    "frame": "token frame f covers raw samples [512f - 448, 512f + 64), midpoint 512f - 192; mask0 = es_vad(full)",
    "span": "[max(0, 512 f1 - 448), min(n, 512 f2 + 64)) for the first and last frames with mask0 = 1",
    "region": "where the 10 s window is searched: the span for target, the whole capture [0, n) for own",
    "window_start": "set only when the region is longer than 160000: start region_start + 512 i with the most "
                    "mask0 midpoints inside [start, start + 160000), earliest on ties; null otherwise",
    "clip": "clip_start = window_start or region start; clip_len = length rounded down to a multiple of 8000; "
            "fail 'too short' under 64000, 'no speech' without a mask0 frame",
    "windows": "[8000 w, 24000] for w = 0 .. floor((clip_len - 24000) / 8000)",
    "frames_inside": "per window, [first, last] es_vad(clip) frames (inclusive) whose midpoint is in the window",
    "sbr_db": "10 log10((P_speech + 1e-12) / (P_other + 1e-12)), P = sum of x^2 over the raw samples of the frames "
              "/ number of those samples; speech = mask0 frames, other = the rest; if no frame has mask0 = 0, "
              "other = the max(1, F // 10) frames with the lowest es_vad energy (stable sort) and speech = the rest",
    "clip_pct": "100 * count(|x| >= 0.98 in float32) / n on the full capture",
    "speech_frames": "kept frames of es_vad(clip) (before any own-window removal, no WeSpeaker here)",
    "speech_s": "speech_frames * 0.032 (es_voiceprint_speech_seconds)",
    "speech_s_full": "kept frames of es_vad(full capture) * 0.032 (the capture ring and the own-voice 8 s stop rule)",
    "n_avail": "speech_frames // 3; fail 'ES_ERR_ENROLL_SHORT' when under 4",
}
