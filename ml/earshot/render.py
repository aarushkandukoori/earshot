"""Render one training or test mixture (10-ml 8.3 steps 1-12, 8.4 labels, 8.6 mic chain; contract 8.1).

`render_example(src, recipe, rng, L, Le, ...)` follows 8.3 in its exact order:
  1 scalars (the Draws dataclass)   2 room   3-4 speakers and the target's enrollment / stream rows
  5 dry streams   6 Lombard   7 convolutions and backgrounds   8 activity labels   9 levels
  10 mixture   11 reference   12 mic chain.
The enrollments (8.5) are rendered afterwards by enroll.py from the scene this function returns, with the same
generator, so one (seed, example index) always gives the same mixture AND the same enrollments.

All randomness comes from the one numpy generator `rng` passed in (data.py makes it from (seed, example index)).
L and Le are sample counts (L = segment, Le = target enrollment length). Every signal is float32 at 16 kHz.
"""
import math
from dataclasses import dataclass, fields

import numpy as np
import scipy.signal

from . import dsp
from .config import HOP, SR
from .sources import read_crop, read_part, read_row

FADE = 80                   # 5 ms fades on every utterance or noise piece
ACTIVITY_FADE = 160         # 10 ms fades at the ends of an activity window
LEVEL_1M_DBFS = -26.0       # each talker's dry stream: active power -26 dBFS + N(0, 4) dB ("level at 1 m")
LEVEL_1M_STD_DB = 4.0
STREAM_GAP_S = (0.1, 0.5)   # gaps between utterances in a talker stream
P_NOBODY = 0.10             # Everyone examples with nobody talking (WP-ML choice, DECISIONS L)
P_OWN_GIVEN = 0.90          # own enrollment given (the recipe may say otherwise)
P_TWO_NOISE = 0.30          # 2 noise clips instead of 1
NOISE_KINDS = ("demand", "fsd50k", "musan", "synthetic")
NOISE_PROBS = (0.30, 0.40, 0.20, 0.10)
P_TARGET_PARTIAL = 0.25     # target active only U(0.3, 0.9) of the segment in 25% of examples
LOUD_SNR_DB = 5.0           # a "loud scene" has a drawn SNR <= 5 dB (Lombard applies to 30% of those)
TEST_LEVEL_DBFS = -26.0     # test mode (8.9): fixed level, high-pass 100 Hz, no EQ, no clipping, no Lombard
TEST_HIGHPASS_HZ = 100.0


# ---------------------------------------------------------------------------------------------------------------
# Step 1 - every per-example scalar, in draw order
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Draws:
    """Every per-example scalar, drawn in exactly this field order (resume-exact determinism depends on it).

    Every field is drawn for every example, even when it ends up unused, so the generator always advances by the
    same amount here. Content (rooms, speakers, offsets, filter values) is drawn by the later steps.
    """
    person: bool            # Person mode with recipe.p_person, else Everyone
    absent: bool            # Person: target absent with recipe.p_absent; Everyone: nobody talking with 0.10
    room: str               # "sim" / "real" / "dry" with recipe.rooms probabilities
    n_talkers: int          # competing talkers 0..3 with recipe.talkers probabilities
    own_given: bool         # own enrollment given, recipe.p_own_given (0.90)
    own_clip: int           # which of the speaker's 8 cached own clips, U{0..7}
    own_in_mix: bool        # own talker in the mix: own_given and u < recipe.p_own_mix / p_own_given
    babble: bool            # babble present, recipe.p_babble
    n_babble: int           # babble talkers, U{6..20}
    background: str         # "noise" or "music" (recipe.p_music), exactly one ("none" only by override, tests)
    n_noise: int            # noise clips when the background is noise: 1 (70%) or 2 (30%)
    noise_kinds: tuple      # 2 kinds, each demand 30 / fsd50k 40 / musan 20 / synthetic 10; the first n_noise used
    sir_db: float           # target vs loudest competing talker, U(recipe.sir)
    quiet: bool             # quiet room, recipe.p_quiet
    snr_db: float           # quiet: U(20, 40); else recipe.snr (uniform, or a normal clipped to its range)
    lombard: bool           # loud scene (snr_db <= 5) and u < recipe.p_lombard
    lombard_db: float       # U(3, 8) dB added to the SNR goal when lombard
    lombard_shelf_db: float  # U(2, 4) dB high shelf at 1 kHz on every talker when lombard
    own_u: float            # U(0, 1); own level above the target = 18 own_u dB in sim in_hand rooms, else 12 own_u
    enr_room: str           # target enrollment room: "same" 50 / "other" 30 / "dry" 20
    enr_other_real: bool    # an "other" room is a real row (20%) instead of a sim room (80%)
    enr_weak: bool          # weaker talker in the enrollment, 30%
    enr_weak_from_mix: bool  # the weaker talker is one of this example's competing talkers (50%, when one exists)
    enr_weak_db: float      # weaker talker U(5, 15) dB below the clean enrollment
    enr_noise: bool         # enrollment noise, 80%
    enr_noise_babble: bool  # babble on top of the enrollment noise, 50% of the noisy ones
    enr_n_babble: int       # enrollment babble talkers, U{6..12}
    enr_snr_db: float       # enrollment SNR, U(0, 25) dB
    enr_mask: str           # target mask: "oracle" 40 / "vad" 40 / "ones" 20
    use_wes: bool           # WeSpeaker vectors used (u >= 0.2); else both fall back to the null vectors
    own_noise: bool         # noise on the own enrollment's token path, 30%
    own_snr_db: float       # its SNR, U(15, 30) dB


# Overrides that are not Draws fields (test mode, 8.9): fixed content values.
EXTRA_OVERRIDES = {"own_db", "room_index", "geometry", "target_frac", "own_frac"}


def choose(rng, names, probs):
    """One name with the given probabilities, from a single uniform draw."""
    u = rng.random()
    total = 0.0
    for name, p in zip(names, probs):
        total += p
        if u < total:
            return name
    return names[-1]


def draw_snr(spec, rng):
    """SNR from a recipe spec: ("uniform", lo, hi) or ("normal", mean, std, lo, hi) clipped to [lo, hi]."""
    if spec[0] == "uniform":
        return float(rng.uniform(spec[1], spec[2]))
    _, mean, std, lo, hi = spec
    return float(np.clip(rng.normal(mean, std), lo, hi))


def draw_scalars(recipe, rng, overrides=None):
    """Step 1: draw the Draws fields in order. An override replaces a value right after it is drawn (the draw
    still happens, so the generator stays aligned), and later fields that depend on it see the override."""
    ov = overrides or {}
    v = {}

    def put(name, value):
        v[name] = ov.get(name, value)

    put("person", bool(rng.random() < recipe.p_person))
    u = rng.random()
    put("absent", bool(u < (recipe.p_absent if v["person"] else P_NOBODY)))
    put("room", choose(rng, ("sim", "real", "dry"), recipe.rooms))
    put("n_talkers", int(choose(rng, (0, 1, 2, 3), recipe.talkers)))
    put("own_given", bool(rng.random() < recipe.p_own_given))
    put("own_clip", int(rng.integers(0, 8)))
    u = rng.random()
    # P(own in mix | own given) = p_own_mix / p_own_given; 0 if a recipe never gives an own enrollment (no 1/0).
    p_mix_given = recipe.p_own_mix / recipe.p_own_given if recipe.p_own_given > 0 else 0.0
    put("own_in_mix", bool(v["own_given"] and u < p_mix_given))
    put("babble", bool(rng.random() < recipe.p_babble))
    put("n_babble", int(rng.integers(6, 21)))
    put("background", "music" if rng.random() < recipe.p_music else "noise")
    put("n_noise", 2 if rng.random() < P_TWO_NOISE else 1)
    put("noise_kinds", (choose(rng, NOISE_KINDS, NOISE_PROBS), choose(rng, NOISE_KINDS, NOISE_PROBS)))
    put("sir_db", float(rng.uniform(recipe.sir[0], recipe.sir[1])))
    put("quiet", bool(rng.random() < recipe.p_quiet))
    normal = draw_snr(recipe.snr, rng)
    quiet = float(rng.uniform(20.0, 40.0))
    put("snr_db", quiet if v["quiet"] else normal)
    u = rng.random()
    put("lombard", bool(u < recipe.p_lombard and v["snr_db"] <= LOUD_SNR_DB))
    put("lombard_db", float(rng.uniform(3.0, 8.0)))
    put("lombard_shelf_db", float(rng.uniform(2.0, 4.0)))
    put("own_u", float(rng.random()))
    put("enr_room", choose(rng, ("same", "other", "dry"), (0.5, 0.3, 0.2)))
    put("enr_other_real", bool(rng.random() < 0.2))
    put("enr_weak", bool(rng.random() < 0.3))
    put("enr_weak_from_mix", bool(rng.random() < 0.5))
    put("enr_weak_db", float(rng.uniform(5.0, 15.0)))
    put("enr_noise", bool(rng.random() < 0.8))
    put("enr_noise_babble", bool(rng.random() < 0.5))
    put("enr_n_babble", int(rng.integers(6, 13)))
    put("enr_snr_db", float(rng.uniform(0.0, 25.0)))
    put("enr_mask", choose(rng, ("oracle", "vad", "ones"), (0.4, 0.4, 0.2)))
    put("use_wes", bool(rng.random() >= 0.2))
    put("own_noise", bool(rng.random() < 0.3))
    put("own_snr_db", float(rng.uniform(15.0, 30.0)))
    return Draws(**v)


def check_overrides(overrides):
    """Raise on an override name that is neither a Draws field nor one of EXTRA_OVERRIDES."""
    known = {f.name for f in fields(Draws)} | EXTRA_OVERRIDES
    unknown = set(overrides or {}) - known
    if unknown:
        raise ValueError(f"unknown override(s) {sorted(unknown)}; allowed: {sorted(known)}")


# ---------------------------------------------------------------------------------------------------------------
# Step 2 - room
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Room:
    """The 8 RIRs of one example. Position k: 0 target, 1-3 talkers, 4 own voice, 5-7 babble and noise."""
    kind: str               # "sim", "real" or "dry"
    index: int              # sim room index, or the real row r0 of the target (-1 for dry)
    h: list                 # 8 float32 RIRs
    dp: list                # 8 direct-path sample indices
    geometry: str           # "in_hand" / "on_table" for sim rooms, else "none"
    rt60: float             # seconds (0 for dry)


def real_rows(src, r0, rng):
    """Rows of the real bank for positions 0..7 (10-ml 8.3 step 2): r0 for the target; every other source takes a
    row from the same room_id without replacement while any remain, then a random row of the same db."""
    same = [r for r in src.real_room_rows[src.real_room[r0]] if r != r0]
    order = rng.permutation(len(same))
    db_rows = src.real_db_rows[src.real_db[r0]]
    rows = [r0]
    for k in range(1, 8):
        if k - 1 < len(same):
            rows.append(same[order[k - 1]])
        else:
            rows.append(db_rows[int(rng.integers(len(db_rows)))])
    return rows


def pick_room(src, kind, rng, index=None, geometry=None):
    """Draw (or fix, with index / geometry) a room of the given kind -> Room."""
    if kind == "dry":
        one = np.ones(1, dtype=np.float32)
        return Room("dry", -1, [one] * 8, [0] * 8, "none", 0.0)
    if kind == "sim":
        if index is None:
            if geometry is None:
                index = int(rng.integers(src.n_sim))
            else:
                candidates = np.flatnonzero(src.sim_in_hand == (geometry == "in_hand"))
                index = int(candidates[rng.integers(len(candidates))])
        h = [np.asarray(src.sim[index, k], dtype=np.float32) for k in range(8)]
        geo = "in_hand" if src.sim_in_hand[index] else "on_table"
        return Room("sim", index, h, [int(d) for d in src.sim_dp[index]], geo, float(src.sim_rt60[index]))
    if kind == "real":
        r0 = int(rng.integers(src.n_real)) if index is None else int(index)
        rows = real_rows(src, r0, rng)
        h = [np.asarray(src.real[r], dtype=np.float32) for r in rows]
        return Room("real", r0, h, [int(src.real_dp[r]) for r in rows], "none", float(src.real_rt60[r0]))
    raise ValueError(f"unknown room kind {kind!r}")


# ---------------------------------------------------------------------------------------------------------------
# Steps 3 and 4 - speakers, and the target's enrollment rows vs stream rows
# ---------------------------------------------------------------------------------------------------------------
def pick_distinct(pool, used, rng):
    """One speaker index from `pool` that is not in `used` (redraw until it is new); adds it to `used`.

    pool is an int32 array of speaker indices, or an int n meaning all speakers 0..n-1.
    """
    for _ in range(10000):
        spk = int(rng.integers(pool)) if isinstance(pool, int) else int(pool[rng.integers(len(pool))])
        if spk not in used:
            used.add(spk)
            return spk
    raise RuntimeError("could not find a new speaker")


def split_target_rows(src, spk, rng, L, Le):
    """Step 4 for one candidate target -> (ok, enrollment rows E, stream rows).

    Shuffle the speaker's rows; E = the shortest prefix whose durations reach Le + 0.5 s, capped at all rows but
    one; the stream uses only the remaining rows. ok is False when E is shorter than Le + 0.5 s or the remaining
    rows hold less than the segment L, so the enrollment never cycles and never shares audio with the stream.
    """
    rows = src.rows_of[spk][rng.permutation(len(src.rows_of[spk]))]
    need = Le / SR + 0.5
    enr = []
    total = 0.0
    while total < need and len(enr) < len(rows) - 1:
        enr.append(int(rows[len(enr)]))
        total += float(src.dur[enr[-1]])
    rest = [int(r) for r in rows[len(enr):]]
    rest_total = float(sum(src.dur[r] for r in rest))
    return (total >= need and rest_total >= L / SR), enr, rest


def pick_target(src, rng, L, Le):
    """Steps 3-4 for the target -> (speaker, enrollment rows, stream rows, tries).

    Draw from the target pool; re-draw (same generator) while split_target_rows fails, at most 20 tries, then use
    the pool speaker with the most audio.
    """
    for tries in range(1, 21):
        spk = int(src.target_pool[rng.integers(len(src.target_pool))])
        ok, enr, rest = split_target_rows(src, spk, rng, L, Le)
        if ok:
            return spk, enr, rest, tries
    spk = src.target_fallback
    ok, enr, rest = split_target_rows(src, spk, rng, L, Le)
    if not ok:
        raise ValueError(f"no target can give a {Le / SR:.1f} s enrollment plus a {L / SR:.1f} s segment")
    return spk, enr, rest, 21


# ---------------------------------------------------------------------------------------------------------------
# Step 5 - dry talker streams
# ---------------------------------------------------------------------------------------------------------------
def speech_stream(src, rows, n, rng, record=None):
    """n samples of one talker: random windows of the allowed rows joined with U(0.1, 0.5) s gaps (8.3 step 5).

    The rows are shuffled; the first piece starts at a random offset in the first half of its row, the next rows
    are read from their start; each piece gets 5 ms fades; rows cycle if they run out. Only the samples needed are
    decoded. record (a list) collects (row, start, length) of every piece, for the tests.
    """
    order = [int(rows[i]) for i in rng.permutation(len(rows))]
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    k = 0
    while pos < n and k < 1000:
        row = order[k % len(order)]
        start = int(rng.integers(0, max(1, int(src.dur[row] * SR) // 2))) if k == 0 else 0
        piece = read_part(src.paths[row], start, n - pos)
        if len(piece) > 0:
            out[pos:pos + len(piece)] = dsp.fade(piece, FADE)
            if record is not None:
                record.append((row, start, len(piece)))
            pos += len(piece)
        pos += int(round(rng.uniform(STREAM_GAP_S[0], STREAM_GAP_S[1]) * SR))
        k += 1
    return out


def scale_to_active(x, dbfs):
    """Scale x so dsp.active_power(x, dsp.hop_vad(x)) equals dbfs (left alone if x is silent)."""
    power = dsp.active_power(x, dsp.hop_vad(x))
    if power <= 0.0:
        return x
    return (x * np.sqrt(10.0 ** (dbfs / 10.0) / power)).astype(np.float32)


def activity_window(x, frac, rng):
    """Keep one active span of round(frac * n) samples at a random start, zeros elsewhere, 10 ms fades.

    frac >= 1 keeps the whole signal unchanged (no fades at the segment edges).
    """
    n = len(x)
    length = int(round(frac * n))
    if length >= n:
        return x
    start = int(rng.integers(0, n - length + 1))
    out = np.zeros(n, dtype=np.float32)
    out[start:start + length] = dsp.fade(x[start:start + length], ACTIVITY_FADE)
    return out


def talker(src, rows, n, rng, frac, record=None):
    """One talker's dry stream at its level at 1 m (-26 dBFS + N(0, 4) dB) inside its activity window."""
    stream = speech_stream(src, rows, n, rng, record)
    stream = scale_to_active(stream, LEVEL_1M_DBFS + rng.normal(0.0, LEVEL_1M_STD_DB))
    return activity_window(stream, frac, rng)


def convolve(x, h):
    """Place a dry signal in the room: causal convolution cut to len(x). A 1-tap RIR (dry room) is a plain gain."""
    if len(h) == 1:
        return (x * h[0]).astype(np.float32)
    return scipy.signal.fftconvolve(x, h)[:len(x)].astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Step 7 helpers - backgrounds (noise, music, synthetic noise)
# ---------------------------------------------------------------------------------------------------------------
def mean_power(x):
    """Mean of x^2 over every sample (float64)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x * x)) if len(x) else 0.0


def set_power(x, power):
    """Scale x to the given mean power (left alone if x is silent)."""
    p = mean_power(x)
    if p <= 0.0:
        return x
    return (x * np.sqrt(power / p)).astype(np.float32)


def colored_noise(n, alpha, rng):
    """Unit-power noise with a spectrum shaped by f^(-alpha/2) above 20 Hz (nothing below 20 Hz).

    alpha 0 is white, 1 is pink, 2 is brown. Made by shaping white noise in the frequency domain.
    """
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.zeros(len(freqs))
    above = freqs >= 20.0
    shape[above] = freqs[above] ** (-alpha / 2.0)
    y = np.fft.irfft(spectrum * shape, n)
    return set_power(y.astype(np.float32), 1.0)


def mains_hum(n, rng):
    """Unit-power mains hum: 50 or 60 Hz, harmonics 1..10 with amplitudes U(0.2, 1) / k and random phases,
    plus colored noise (alpha U(0, 2)) 20 dB below the hum."""
    f0 = 50.0 if rng.random() < 0.5 else 60.0
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, 11):
        amp = rng.uniform(0.2, 1.0) / k
        y += amp * np.sin(2.0 * math.pi * k * f0 * t + rng.uniform(0.0, 2.0 * math.pi))
    y = set_power(y.astype(np.float32), 1.0)
    hiss = colored_noise(n, rng.uniform(0.0, 2.0), rng) * np.float32(10.0 ** (-20.0 / 20.0))
    return set_power(y + hiss, 1.0)


def synthetic_noise(n, rng):
    """The synthetic 10% noise source (contract 8.1): 70% colored noise (alpha U(0, 2)), 30% mains hum. No RIR."""
    if rng.random() < 0.7:
        return colored_noise(n, rng.uniform(0.0, 2.0), rng)
    return mains_hum(n, rng)


def tile(audio, n, gap_s, rng):
    """Repeat a short clip (5 ms fades) with gaps of U(gap_s) seconds of zeros until n samples are filled."""
    out = np.zeros(n, dtype=np.float32)
    piece = dsp.fade(audio, FADE)
    pos = 0
    while pos < n and len(piece) > 0:
        take = min(len(piece), n - pos)
        out[pos:pos + take] = piece[:take]
        pos += take + int(round(rng.uniform(gap_s[0], gap_s[1]) * SR))
    return out


def crop_or_tile(src, row, n, rng, gap_s):
    """A random n-sample crop of noise row `row`, or the whole clip tiled (gaps U(gap_s) s) if it is shorter."""
    length = int(src.noise_dur[row] * SR)
    if length >= n:
        start = int(rng.integers(0, length - n + 1))
        return dsp.fade(read_crop(src.noise_paths[row], start, n), FADE)
    return tile(read_row(src.noise_paths[row]), n, gap_s, rng)


def noise_clip(src, kind, n, room, rng, record=None):
    """One unit-power noise clip of the given kind, already placed in the room (8.3 step 7).

    demand    - random channel file and crop, no RIR (already ambient)
    fsd50k    - the clip tiled with gaps U(0, 1) s (a random crop if long), RIR of position 5, 6 or 7
    musan     - a crop (tiled without gaps if short), RIR of position 5, 6 or 7
    synthetic - colored noise or mains hum generated here, no RIR
    A kind with no rows in this split (val has no DEMAND, the CC0 test filter has no MUSAN) uses FSD50K instead.
    """
    if kind != "synthetic" and len(src.noise_rows[kind]) == 0:
        kind = "fsd50k"
    if kind == "synthetic":
        return synthetic_noise(n, rng), kind
    rows = src.noise_rows[kind]
    row = int(rows[rng.integers(len(rows))])
    if record is not None:
        record.append(("noise", src.noise_paths[row], src.noise_lic[row]))
    if kind == "demand":
        clip = crop_or_tile(src, row, n, rng, (0.0, 0.0))
    else:
        clip = crop_or_tile(src, row, n, rng, (0.0, 1.0) if kind == "fsd50k" else (0.0, 0.0))
        clip = convolve(clip, room.h[int(rng.integers(5, 8))])
    if mean_power(clip) <= 1e-12:          # a silent crop cannot carry an SNR; use colored noise instead
        return colored_noise(n, rng.uniform(0.0, 2.0), rng), "synthetic"
    return set_power(clip, 1.0), kind


def music_clip(src, n, room, rng, record=None):
    """One unit-power music clip: a crop of a music row (tiled if short), RIR of position 5, 6 or 7."""
    rows = src.noise_rows["music"]
    row = int(rows[rng.integers(len(rows))])
    if record is not None:
        record.append(("music", src.noise_paths[row], src.noise_lic[row]))
    clip = convolve(crop_or_tile(src, row, n, rng, (0.0, 0.0)), room.h[int(rng.integers(5, 8))])
    if mean_power(clip) <= 1e-12:
        return colored_noise(n, rng.uniform(0.0, 2.0), rng)
    return set_power(clip, 1.0)


def babble_mix(src, speakers, n, room, rng, record=None):
    """Babble: each talker's dry stream (level at 1 m, always active), summed per position (round robin over
    5, 6, 7 from a random start), then convolved once per position."""
    first = int(rng.integers(0, 3))
    per_position = [np.zeros(n, dtype=np.float32) for _ in range(3)]
    for b, spk in enumerate(speakers):
        rec = [] if record is not None else None
        per_position[(first + b) % 3] += talker(src, src.rows_of[spk], n, rng, 1.0, rec)
        if record is not None:
            record.extend(("babble", src.paths[r], src.lic[r]) for r, _, _ in rec)
    out = np.zeros(n, dtype=np.float32)
    for p in range(3):
        if np.any(per_position[p]):
            out += convolve(per_position[p], room.h[5 + p])
    return out


# ---------------------------------------------------------------------------------------------------------------
# Step 12 - mic chain (10-ml 8.6): the same linear filters on the mixture and every reference
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class MicChain:
    """The phone's microphone path for one example (also reused by its enrollments: same phone)."""
    highpass_hz: float      # 2nd-order Butterworth high-pass, U(60, 200) Hz (100 Hz in test mode)
    eqs: tuple              # 0, 1 or 2 peaking EQs (f0 log-uniform 300 Hz..6 kHz, gain U(-4, 4) dB, Q U(0.7, 2))
    level_dbfs: float       # mixture RMS after the gain, U(-40, -15) dBFS (-26 in test mode)
    self_noise_dbfs: float  # self-noise RMS, U(-95, -75) dBFS (None = off, test mode)
    self_noise_pink: bool   # pink or white self-noise, equal odds
    clip: str               # "none", "hard" (u < 0.03) or "soft" (0.03 <= u < 0.06); "none" without clipping
    clip_q: float           # quantile of |x| that sets the clip level c
    decim: bool = True      # fir_decim applied (False only with allow_missing_decim and no asset file)


def draw_mic_chain(rng, clipping, test_mode):
    """Draw the mic chain values (all drawn every time; test mode then fixes them)."""
    highpass = float(rng.uniform(60.0, 200.0))
    n_eq = int(rng.integers(0, 3))
    eqs = []
    for _ in range(n_eq):
        f0 = float(math.exp(rng.uniform(math.log(300.0), math.log(6000.0))))
        eqs.append((f0, float(rng.uniform(-4.0, 4.0)), float(rng.uniform(0.7, 2.0))))
    level = float(rng.uniform(-40.0, -15.0))
    noise_db = float(rng.uniform(-95.0, -75.0))
    pink = bool(rng.random() < 0.5)
    u = rng.random()
    q_hard = float(rng.uniform(0.995, 0.9995))
    q_soft = float(rng.uniform(0.99, 0.999))
    clip, q = "none", 1.0
    if clipping and u < 0.03:
        clip, q = "hard", q_hard
    elif clipping and u < 0.06:
        clip, q = "soft", q_soft
    if test_mode:
        return MicChain(TEST_HIGHPASS_HZ, (), TEST_LEVEL_DBFS, None, False, "none", 1.0)
    return MicChain(highpass, tuple(eqs), level, noise_db, pink, clip, q)


def chain_filters(chain, x):
    """The linear part of the mic chain: high-pass, the peaking EQs, then fir_decim (all causal)."""
    if not np.any(x):
        return np.zeros(len(x), dtype=np.float32)
    y = dsp.apply_sos(dsp.sos_highpass(chain.highpass_hz), x)
    for f0, gain_db, q in chain.eqs:
        y = dsp.apply_sos(dsp.sos_peaking(f0, gain_db, q), y)
    if chain.decim:
        y = dsp.fir_decim(y)
    return y


def level_gain(level_dbfs, x):
    """Gain that brings x to level_dbfs RMS (1 if x is silent)."""
    rms = math.sqrt(mean_power(x))
    return np.float32(10.0 ** (level_dbfs / 20.0) / rms) if rms > 0 else np.float32(1.0)


PEAK_MAX = 0.99   # ADC headroom (DECISIONS L 2026-11-08): no mixture or enrollment sample past this before clipping


def headroom_gain(gain, x):
    """Lower `gain` so that max |x * gain| stays at or below PEAK_MAX (a phone mic never delivers samples past full
    scale). Returns the gain unchanged when it already fits. Callers use the same gain on the reference."""
    peak = float(np.max(np.abs(x))) * float(gain) if x.size else 0.0
    return np.float32(float(gain) * PEAK_MAX / peak) if peak > PEAK_MAX else np.float32(gain)


def self_noise(chain, n, rng):
    """Microphone self-noise at chain.self_noise_dbfs RMS (zeros when off)."""
    if chain.self_noise_dbfs is None:
        return np.zeros(n, dtype=np.float32)
    base = colored_noise(n, 1.0, rng) if chain.self_noise_pink else set_power(
        rng.standard_normal(n).astype(np.float32), 1.0)
    return base * np.float32(10.0 ** (chain.self_noise_dbfs / 20.0))


def clip_signal(chain, x):
    """Mixture-only clipping: hard clip at c, or soft clip c tanh(x / c), with c = quantile(|x|, chain.clip_q)."""
    if chain.clip == "none":
        return x
    c = float(np.quantile(np.abs(x), chain.clip_q))
    if c <= 0.0:
        return x
    if chain.clip == "hard":
        return np.clip(x, -c, c).astype(np.float32)
    return (c * np.tanh(x / c)).astype(np.float32)


def decim_available(allow_missing_decim):
    """True if ml/assets/decim16k_fir.npy exists. Missing: raise with the command that makes it, unless allowed
    (smoke config and unit tests only, 10-ml 8.6), in which case fir_decim is skipped."""
    if dsp.DECIM_FIR_PATH.exists():
        return True
    if allow_missing_decim:
        print(f"note: {dsp.DECIM_FIR_PATH} is missing; fir_decim skipped (allow_missing_decim)")
        return False
    raise FileNotFoundError(f"{dsp.DECIM_FIR_PATH} is missing. The lead makes it with "
                            "`.venv/bin/python engine/tools/make_resampler_tables.py` and commits it (contract 11).")


# ---------------------------------------------------------------------------------------------------------------
# One example (10-ml 8.3 steps 1-12)
# ---------------------------------------------------------------------------------------------------------------
def source_entries(role, rows, src):
    """meta.sources lines for speech pieces [(row, start, length)] (one line per distinct row)."""
    seen = []
    for row, _, _ in rows:
        if row not in seen:
            seen.append(row)
    return [{"role": role, "path": src.paths[r], "license": src.lic[r], "row": int(r)} for r in seen]


def render_example(src, recipe, rng, L, Le, overrides=None, test_mode=False, metric_signals=False,
                   record_sources=False, return_parts=False, allow_missing_decim=False):
    """Render one mixture and its reference -> scene dict (enroll.py adds the enrollments from it).

    src: sources.Sources; recipe: data.Recipe; rng: the example's numpy generator; L, Le: samples.
    overrides: fixed values for any Draws field or EXTRA_OVERRIDES (test mode, 8.9).
    test_mode: level -26 dBFS, high-pass 100 Hz, no EQ, no clipping, no Lombard, no self-noise, fir_decim on.
    metric_signals: also return the target and talker early refs, the own image and their VADs after the mic chain.
    record_sources: record every row used (tests and test-vector meta). return_parts: the scaled signals before
    the mic chain (tests check the realized levels on them).
    Scene keys: mix, ref (f32 [L]), vad_t, vad_o (u8 [L/64]), present, own_in_mix, person, info, plus what
    enroll.py needs (draws, room, chain, target, talkers, own, used, enr_rows) and the optional extras.
    """
    if L % HOP:
        raise ValueError(f"segment length {L} is not a multiple of {HOP}")
    check_overrides(overrides)
    ov = dict(overrides or {})
    rec = {"target": [], "talkers": [], "own_mix": []} if record_sources else None
    bg_rec = [] if record_sources else None

    # 1. Scalars.
    d = draw_scalars(recipe, rng, ov)
    if test_mode:
        d.lombard = False
    if d.own_in_mix and not d.own_given:
        raise ValueError("an own talker in the mix needs an own enrollment (own_given)")

    # 2. Room.
    room = pick_room(src, d.room, rng, ov.get("room_index"), ov.get("geometry"))
    own_span_db = 18.0 if room.geometry == "in_hand" else 12.0
    own_db = float(ov.get("own_db", d.own_u * own_span_db))

    # 3-4. Speakers: the target first (with its re-draws), then the others, never one speaker twice.
    spk_t, enr_rows, stream_rows, tries = pick_target(src, rng, L, Le)
    used = {spk_t}
    n_all = len(src.speakers)
    talkers = [pick_distinct(n_all, used, rng) for _ in range(d.n_talkers)]
    own_spk = pick_distinct(src.own_pool, used, rng) if d.own_given else -1
    babble_spk = [pick_distinct(n_all, used, rng) for _ in range(d.n_babble)] if d.babble else []

    # 5. Dry streams at their level at 1 m, inside their activity windows.
    u = rng.random()
    partial = float(rng.uniform(0.3, 0.9))
    frac_t = float(ov.get("target_frac", partial if u < P_TARGET_PARTIAL else 1.0))
    dry_t = talker(src, stream_rows, L, rng, frac_t, rec["target"] if rec else None)
    dry_i = []
    for spk in talkers:
        frac = float(rng.uniform(0.5, 1.0))
        dry_i.append(talker(src, src.rows_of[spk], L, rng, frac, rec["talkers"] if rec else None))
    dry_o = None
    if d.own_in_mix:
        frac_o = float(ov.get("own_frac", rng.uniform(0.2, 0.6)))
        dry_o = talker(src, src.own_mix_rows[own_spk], L, rng, frac_o, rec["own_mix"] if rec else None)

    # 6. Lombard: a high shelf on every talker's dry stream (the SNR goal rises in step 9).
    if d.lombard:
        shelf = dsp.sos_high_shelf(1000.0, d.lombard_shelf_db)
        dry_t = dsp.apply_sos(shelf, dry_t)
        dry_i = [dsp.apply_sos(shelf, x) for x in dry_i]
        if dry_o is not None:
            dry_o = dsp.apply_sos(shelf, dry_o)

    # 7. Into the room: reverberant images, early references (direct path + 50 ms), backgrounds.
    t = convolve(dry_t, room.h[0])
    s = convolve(dry_t, dsp.early_rir(room.h[0], room.dp[0]))
    img = [convolve(x, room.h[1 + i]) for i, x in enumerate(dry_i)]
    r = [convolve(x, dsp.early_rir(room.h[1 + i], room.dp[1 + i])) for i, x in enumerate(dry_i)]
    o = o_e = None
    if dry_o is not None:
        o = convolve(dry_o, room.h[4])
        o_e = convolve(dry_o, dsp.early_rir(room.h[4], room.dp[4]))
    babble = babble_mix(src, babble_spk, L, room, rng, bg_rec) if babble_spk else None
    noise, music, kinds_used = None, None, []
    if d.background == "noise":
        noise = np.zeros(L, dtype=np.float32)
        for k in range(d.n_noise):
            clip, kind = noise_clip(src, d.noise_kinds[k], L, room, rng, bg_rec)
            noise += clip
            kinds_used.append(kind)
    elif d.background == "music":
        music = music_clip(src, L, room, rng, bg_rec)

    # 8. Activity labels on the clean early references, before any level change (8.4).
    vad_s = dsp.hop_vad(s)
    vad_i = [dsp.hop_vad(x) for x in r]
    vad_o = dsp.hop_vad(o_e) if o_e is not None else np.zeros(L // HOP, dtype=np.uint8)

    # 9. Levels: SIR, own level and SNR, all against the target's active power P_t.
    p_t = dsp.active_power(t, vad_s)
    if img:
        p_max = max(max(dsp.active_power(x, v) for x, v in zip(img, vad_i)), 1e-20)
        g = np.float32(math.sqrt(p_t / (p_max * 10.0 ** (d.sir_db / 10.0))))
        img = [x * g for x in img]
        r = [x * g for x in r]
    if o is not None:
        g = np.float32(math.sqrt(p_t * 10.0 ** (own_db / 10.0) / max(dsp.active_power(o, vad_o), 1e-20)))
        o, o_e = o * g, o_e * g
    snr_goal = d.snr_db + (d.lombard_db if d.lombard else 0.0)
    p_b = p_t / 10.0 ** (snr_goal / 10.0)
    if music is not None and babble is not None:
        p_music = min(p_t / 10.0 ** (rng.uniform(5.0, 20.0) / 10.0), 0.5 * p_b)
        music, babble = set_power(music, p_music), set_power(babble, p_b - p_music)
    elif noise is not None and babble is not None:
        w = float(rng.uniform(0.2, 0.8))
        noise, babble = set_power(noise, w * p_b), set_power(babble, (1.0 - w) * p_b)
    else:
        noise = set_power(noise, p_b) if noise is not None else None
        music = set_power(music, p_b) if music is not None else None
        babble = set_power(babble, p_b) if babble is not None else None
    parts_bg = [x for x in (babble, noise, music) if x is not None]
    background = np.sum(parts_bg, axis=0).astype(np.float32) if parts_bg else np.zeros(L, dtype=np.float32)
    if parts_bg:
        # Cross terms make the power of the sum differ slightly from the sum of the powers; one last gain on every
        # background part makes the realized SNR (target vs the summed background) equal the goal exactly.
        g = np.float32(math.sqrt(p_b / mean_power(background)))
        background = background * g
        babble, noise, music = [x * g if x is not None else None for x in (babble, noise, music)]

    # 10. Mixture. Absent (Person): no target. Nobody talking (Everyone): no target and no competing talkers.
    nobody = (not d.person) and d.absent
    x = background.copy()
    if not d.absent:
        x += t
    if not nobody:
        for img_k in img:
            x += img_k
    if o is not None:
        x += o

    # 11. Reference and its activity: Person = target early ref; Everyone = target + talkers early refs (talkers
    # 0-3, as scaled in the mix); babble, noise and the own voice are never in it.
    if d.absent:
        ref = np.zeros(L, dtype=np.float32)
        vad_t = np.zeros(L // HOP, dtype=np.uint8)
    elif d.person:
        ref = s.copy()
        vad_t = vad_s.copy()
    else:
        ref = s.copy()
        vad_t = vad_s.copy()
        for r_k, v_k in zip(r, vad_i):
            ref += r_k
            vad_t |= v_k

    # 12. Mic chain: the same filters and gain on the mixture and the reference; self-noise and clipping on the
    # mixture only.
    chain = draw_mic_chain(rng, recipe.clipping, test_mode)
    chain.decim = decim_available(allow_missing_decim)
    x_f = chain_filters(chain, x)
    gain = level_gain(chain.level_dbfs, x_f)
    # Headroom: loud scenes with a high crest factor would peak past full scale (about 11% of recipe-B mixtures,
    # up to +11 dB). Lower the gain instead (same gain on the reference), then apply the intended 3% + 3% clipping.
    # The final clip at +-1 acts like the phone's converter (self-noise could add a hair past PEAK_MAX).
    gain = headroom_gain(gain, x_f)
    mix = np.clip(clip_signal(chain, x_f * gain + self_noise(chain, L, rng)), -1.0, 1.0).astype(np.float32)
    ref = chain_filters(chain, ref) * gain

    # The drawn SNR holds before the mic chain (8.3 step 9 sets levels before step 12). The high-pass and EQs then
    # remove more of a low-frequency background than of speech, so the SNR the model sees can be several dB higher.
    # Log that value too, for analysis only: it changes no signal and draws nothing from rng.
    snr_chain = None
    if not d.absent and parts_bg:
        p_bf = mean_power(chain_filters(chain, background))
        if p_bf > 0:
            p_tf = max(dsp.active_power(chain_filters(chain, t), vad_s), 1e-20)   # target image after the filters
            snr_chain = round(10.0 * math.log10(p_tf / p_bf), 4)

    info = {
        "person": d.person, "absent": d.absent, "present": not d.absent, "room": room.kind,
        "room_index": room.index, "geometry": room.geometry, "rt60": round(room.rt60, 4),
        "target": src.speakers[spk_t], "target_tries": tries, "target_frac": round(frac_t, 4),
        "n_talkers": d.n_talkers, "sir_db": round(d.sir_db, 4) if d.n_talkers else None,
        "own_given": d.own_given, "own_in_mix": d.own_in_mix, "own_db": round(own_db, 4) if d.own_in_mix else None,
        "babble": len(babble_spk), "background": d.background, "noise_kinds": kinds_used,
        "quiet": d.quiet, "lombard": d.lombard, "snr_drawn": round(d.snr_db, 4), "snr_db": round(snr_goal, 4),
        "snr_chain_db": snr_chain,
        "level_dbfs": round(chain.level_dbfs, 3), "clip": chain.clip, "decim": chain.decim,
        "level_real_dbfs": round(10.0 * math.log10(max(mean_power(x_f * gain), 1e-20)), 3),   # after headroom
        "L": L, "Le": Le,
    }
    scene = {"mix": mix.astype(np.float32), "ref": ref.astype(np.float32), "vad_t": vad_t, "vad_o": vad_o,
             "present": not d.absent, "own_in_mix": d.own_in_mix, "person": d.person, "info": info,
             "draws": d, "room": room, "chain": chain, "target": spk_t, "talkers": talkers, "own": own_spk,
             "used": used, "enr_rows": enr_rows, "stream_rows": stream_rows}
    if metric_signals:
        scene["metric"] = {
            "target": chain_filters(chain, s) * gain,
            "talkers": [chain_filters(chain, r_k) * gain for r_k in r],
            "own": chain_filters(chain, o) * gain if o is not None else np.zeros(L, dtype=np.float32),
            "vad_target": vad_s, "vad_talkers": vad_i, "vad_own": vad_o}
    if return_parts:
        scene["parts"] = {"t": t, "s": s, "img": img, "r": r, "o": o, "o_e": o_e, "babble": babble,
                          "noise": noise, "music": music, "background": background, "vad_s": vad_s,
                          "vad_i": vad_i, "vad_o": vad_o, "p_t": p_t, "own_db": own_db, "gain": float(gain)}
    if record_sources:
        scene["rows"] = rec
        scene["sources"] = (source_entries("target (mixture)", rec["target"], src)
                            + source_entries("talkers (mixture)", rec["talkers"], src)
                            + source_entries("own voice (mixture)", rec["own_mix"], src)
                            + [{"role": role, "path": path, "license": lic} for role, path, lic in bg_rec])
    return scene
