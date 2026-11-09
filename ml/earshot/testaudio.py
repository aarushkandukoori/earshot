"""Fixed, license-clean audio for the test vectors and the quality vectors (10-ml 7.4, contract 3.3 and 6).

Everything here is deterministic (seed 7 by default): the same call always returns the same samples, on any machine
with the same data pack. Sources, all from the test splits of the data pack:
- speech - VCTK test speakers (data/manifests/speech_test.jsonl, CC-BY-4.0)
- noise  - FSD50K noise_test clips with license CC0-1.0 (data/manifests/noise_test.jsonl)
- room   - simulated room 0 of data/rir/sim_test.npy (made by dataprep/rir_build.py, no outside license)

10-ml 7.4 says the long test-vector mixture comes from render.render_example in test mode. Two builders exist:
- `testvec_inputs` (scene "handbuilt") builds the scene by hand with the same ingredients and test-mode rules (8.3
  and 8.9): target + 1 talker at SIR 0 dB + own voice at +6 dB + noise at SNR 5 dB, sim room 0, high-pass 100 Hz,
  no EQ, fir_decim, no clipping, mixture at -26 dBFS. ml/exports/random_v1/ was made with it (before render.py
  existed), and it stays the default of `ml.export` so a re-export reproduces random_v1 bit for bit.
- `testvec_inputs_render` (scene "render", `python -m ml.export ... --scene render`) renders the same scene with
  render.render_example in test mode, as 10-ml 7.4 asks. Switching only changes the samples, never the file format.

When the lead switches random_v1 to the rendered scene (DECISIONS L row "random_v1 long test vector uses a
hand-built stand-in mixture until ML-1 render.py exists"), re-export with --scene render and tell WP-ENGINE to rerun
parity, because the `in` and `out` samples change.
"""
import json

import numpy as np
import scipy.signal
import soundfile as sf

from . import dsp
from .config import DATA, SR

SPEECH_TEST = DATA / "manifests" / "speech_test.jsonl"
NOISE_TEST = DATA / "manifests" / "noise_test.jsonl"
SIM_TEST = DATA / "rir" / "sim_test.npy"
SIM_TEST_JSONL = DATA / "rir" / "sim_test.jsonl"
FADE = 80              # 5 ms fades at the ends of every utterance piece
LEVEL_DBFS = -26.0     # test-mode level


# ---------------------------------------------------------------------------------------------------------------
# Reading the data pack
# ---------------------------------------------------------------------------------------------------------------
def read_jsonl(path):
    """All rows of a JSON-lines file, in file order."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def vctk_test_speakers():
    """{speaker id: [rows sorted by path]} for the VCTK test speakers, speakers in sorted order."""
    by_speaker = {}
    for row in read_jsonl(SPEECH_TEST):
        if row["source"] == "vctk":
            by_speaker.setdefault(row["speaker"], []).append(row)
    return {spk: sorted(by_speaker[spk], key=lambda r: r["path"]) for spk in sorted(by_speaker)}


def cc0_noise_rows(min_seconds):
    """FSD50K noise_test rows with license CC0-1.0, use "noise" and at least min_seconds long, sorted by path."""
    rows = []
    for row in read_jsonl(NOISE_TEST):
        if row["source"] == "fsd50k" and row["license"] == "CC0-1.0" and "noise" in row["use"]:
            if row["duration"] >= min_seconds:
                rows.append(row)
    return sorted(rows, key=lambda r: r["path"])


def read_row(row):
    """One manifest row -> mono float32 audio at 16 kHz."""
    audio, sr = sf.read(str(DATA / row["path"]), dtype="float32")
    if sr != SR:
        raise ValueError(f"{row['path']}: sample rate {sr}, expected {SR}")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def sim_room(room):
    """(h [8, 17600] float32, dp list of 8 ints) for simulated test room `room`."""
    bank = np.load(SIM_TEST, mmap_mode="r")
    meta = read_jsonl(SIM_TEST_JSONL)[room]
    return np.asarray(bank[room], dtype=np.float32), [int(d) for d in meta["dp"]]


def source_entry(role, row):
    """The meta.sources line for one audio file (path and license, both plain ASCII)."""
    return {"role": role, "path": row["path"], "license": row["license"], "source": row["source"],
            "speaker": row.get("speaker")}


# ---------------------------------------------------------------------------------------------------------------
# Small signal helpers
# ---------------------------------------------------------------------------------------------------------------
def db_to_amp(db):
    return 10.0 ** (db / 20.0)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def concat_with_gaps(pieces, n, gap_range, rng):
    """Join audio pieces with gaps of U(gap_range) seconds of zeros (each piece faded 5 ms at both ends),
    cycling through the pieces if needed, and return exactly n samples."""
    out = []
    total = 0
    i = 0
    while total < n:
        piece = dsp.fade(pieces[i % len(pieces)], FADE)
        out.append(piece)
        total += len(piece)
        gap = int(round(rng.uniform(gap_range[0], gap_range[1]) * SR))
        out.append(np.zeros(gap, dtype=np.float32))
        total += gap
        i += 1
    return np.concatenate(out)[:n].astype(np.float32)


def speech_stream(audios, n, rng):
    """10-ml 8.3 step 5 in its simple form: the first utterance from a random offset, then whole utterances, with
    gaps of U(0.1, 0.5) s and 5 ms fades, cycling if needed -> n samples."""
    first = audios[0]
    start = int(rng.integers(0, max(1, len(first) // 2)))
    pieces = [first[start:]] + list(audios[1:])
    return concat_with_gaps(pieces, n, (0.1, 0.5), rng)


def set_active_level(x, level_dbfs):
    """Scale x so its active power (dsp.active_power with dsp.hop_vad of x itself) is level_dbfs."""
    power = dsp.active_power(x, dsp.hop_vad(x))
    return (x * np.sqrt(10.0 ** (level_dbfs / 10.0) / power)).astype(np.float32)


def activity_window(x, frac, rng):
    """Keep one active span of round(frac * len(x)) samples at a random start, zeros elsewhere, 10 ms fades."""
    n = len(x)
    length = int(round(frac * n))
    if length >= n:
        return x.astype(np.float32)
    start = int(rng.integers(0, n - length + 1))
    out = np.zeros(n, dtype=np.float32)
    out[start:start + length] = dsp.fade(x[start:start + length], 160)
    return out


def tile_noise(audio, n, rng):
    """FSD50K rule of 8.3 step 7: repeat the clip with random gaps of U(0, 1) s until n samples are filled."""
    out = []
    total = 0
    while total < n:
        out.append(dsp.fade(audio, FADE))
        total += len(audio)
        gap = int(round(rng.uniform(0.0, 1.0) * SR))
        out.append(np.zeros(gap, dtype=np.float32))
        total += gap
    return np.concatenate(out)[:n].astype(np.float32)


def convolve(x, h):
    """Place a dry signal in the room: causal convolution, cut to the input length."""
    return scipy.signal.fftconvolve(x, h)[:len(x)].astype(np.float32)


def test_mode_chain(x):
    """Test-mode mic chain (10-ml 8.6 and 8.9): high-pass 100 Hz, no EQ, fir_decim, level -26 dBFS RMS, no
    clipping. Returns float32."""
    y = dsp.apply_sos(dsp.sos_highpass(100.0), x)
    y = dsp.fir_decim(y)
    return (y * (db_to_amp(LEVEL_DBFS) / rms(y))).astype(np.float32)


def read_rows_until(rows, seconds):
    """Read rows in order until at least `seconds` of audio are loaded (all rows if they are shorter)."""
    audios = []
    total = 0.0
    for row in rows:
        audios.append(read_row(row))
        total += len(audios[-1]) / SR
        if total >= seconds:
            break
    return audios, rows[:len(audios)]


def join_all(pieces, gap_range, rng):
    """Join every piece in order with gaps of U(gap_range) s of zeros between them, 5 ms fades on each piece."""
    out = []
    for i, piece in enumerate(pieces):
        if i > 0:
            gap = int(round(rng.uniform(gap_range[0], gap_range[1]) * SR))
            out.append(np.zeros(gap, dtype=np.float32))
        out.append(dsp.fade(piece, FADE))
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Test-vector inputs (contract 3.3, 10-ml 7.4)
# ---------------------------------------------------------------------------------------------------------------
TV_LONG = 160000      # 10 s mixture (testvec_long.est); the short file uses the first 32,000 samples
TV_ENR_T = 96000      # 6 s target enrollment
TV_ENR_O = 128000     # 8 s own-voice enrollment
TV_SIR_DB = 0.0       # competing talker at the target's level
TV_OWN_DB = 6.0       # own voice 6 dB above the target
TV_SNR_DB = 5.0       # noise 5 dB below the target
TV_ENR_SNR_DB = 10.0  # target enrollment noise


def testvec_inputs(seed=7):
    """Build the fixed test-vector audio -> dict with
    in_long f32 [160000], enr_t f32 [96000], enr_o f32 [128000], sources (list for meta.sources), scene (dict).

    Draw order (one generator, np.random.default_rng([seed, 0])): speakers, target row shuffle, talker row shuffle,
    noise clips, then the signals in the order written below.
    """
    rng = np.random.default_rng([seed, 0])
    speakers = vctk_test_speakers()
    ids = list(speakers)
    order = rng.permutation(len(ids))
    spk_t, spk_1, spk_o = ids[order[0]], ids[order[1]], ids[order[2]]

    # Target rows: shuffle; the enrollment uses the shortest prefix reaching Le + 0.5 s, the mixture the rest, so
    # the enrollment is always from different utterances than the mixture (contract 8.1).
    rows_t = [speakers[spk_t][i] for i in rng.permutation(len(speakers[spk_t]))]
    enr_rows = []
    total = 0.0
    while total < TV_ENR_T / SR + 0.5:
        enr_rows.append(rows_t[len(enr_rows)])
        total += enr_rows[-1]["duration"]
    mix_rows_t = rows_t[len(enr_rows):]
    rows_1 = [speakers[spk_1][i] for i in rng.permutation(len(speakers[spk_1]))]

    # Own voice, split as sources.own_split (6.3): cache rows = the shortest prefix (manifest order) reaching 8 s,
    # never the last row; they feed only the own enrollment. The own talker in the mixture uses the other rows.
    rows_o = speakers[spk_o]
    cache_rows = []
    total = 0.0
    while total < 8.0 and len(cache_rows) < len(rows_o) - 1:
        cache_rows.append(rows_o[len(cache_rows)])
        total += cache_rows[-1]["duration"]
    mix_rows_o = rows_o[len(cache_rows):]

    noise_rows = cc0_noise_rows(5.0)
    pick = rng.choice(len(noise_rows), size=2, replace=False)
    noise_mix, noise_enr = noise_rows[int(pick[0])], noise_rows[int(pick[1])]
    h, dp = sim_room(0)
    seconds = TV_LONG / SR + 2.0

    # Dry streams (levels at 1 m: -26 dBFS active power) and activity windows (8.3 step 5).
    audios, used_t = read_rows_until(mix_rows_t, seconds)
    dry_t = set_active_level(speech_stream(audios, TV_LONG, rng), LEVEL_DBFS)
    audios, used_1 = read_rows_until(rows_1, seconds)
    frac_1 = float(rng.uniform(0.5, 1.0))
    dry_1 = activity_window(set_active_level(speech_stream(audios, TV_LONG, rng), LEVEL_DBFS), frac_1, rng)
    audios, used_o = read_rows_until(mix_rows_o, seconds)
    frac_o = float(rng.uniform(0.2, 0.6))
    dry_o = activity_window(set_active_level(speech_stream(audios, TV_LONG, rng), LEVEL_DBFS), frac_o, rng)
    noise_dry = tile_noise(read_row(noise_mix), TV_LONG, rng)

    # Room: sim_test room 0, positions 0 target, 1 talker, 4 own voice, 5 noise (data/rir/README.md).
    t = convolve(dry_t, h[0])
    s = convolve(dry_t, dsp.early_rir(h[0], dp[0]))
    img_1 = convolve(dry_1, h[1])
    ref_1 = convolve(dry_1, dsp.early_rir(h[1], dp[1]))
    o = convolve(dry_o, h[4])
    o_early = convolve(dry_o, dsp.early_rir(h[4], dp[4]))
    noise = convolve(noise_dry, h[5])

    # Levels (8.3 steps 8 and 9): VAD on the early references, then SIR, own level and SNR against P_t.
    p_t = dsp.active_power(t, dsp.hop_vad(s))
    img_1 = img_1 * np.sqrt(p_t / (dsp.active_power(img_1, dsp.hop_vad(ref_1)) * 10.0 ** (TV_SIR_DB / 10.0)))
    o = o * np.sqrt(p_t * 10.0 ** (TV_OWN_DB / 10.0) / dsp.active_power(o, dsp.hop_vad(o_early)))
    noise = noise * np.sqrt(p_t / 10.0 ** (TV_SNR_DB / 10.0) / np.mean(noise.astype(np.float64) ** 2))
    in_long = test_mode_chain(t + img_1 + o + noise)

    # Target enrollment: the enrollment rows, gaps U(0.1, 0.4) s, same room (position 0), a second CC0 noise clip
    # at position 6 with SNR 10 dB against the clean component, test-mode mic chain.
    audios = [read_row(r) for r in enr_rows]
    c_e = convolve(concat_with_gaps(audios, TV_ENR_T, (0.1, 0.4), rng), h[0])
    noise_e = convolve(tile_noise(read_row(noise_enr), TV_ENR_T, rng), h[6])
    p_e = dsp.active_power(c_e, dsp.hop_vad(c_e))
    noise_e = noise_e * np.sqrt(p_e / 10.0 ** (TV_ENR_SNR_DB / 10.0) / np.mean(noise_e.astype(np.float64) ** 2))
    enr_t = test_mode_chain(c_e + noise_e)

    # Own enrollment (6.3): cache rows joined with gaps U(0.10, 0.30) s, one 8 s window at a random start, scaled
    # to -26 dBFS active power. Clean: no room, no mic chain.
    joined = join_all([read_row(r) for r in cache_rows], (0.10, 0.30), rng)
    start = int(rng.integers(0, len(joined) - TV_ENR_O + 1))
    enr_o = set_active_level(joined[start:start + TV_ENR_O], LEVEL_DBFS)

    for name, x in [("in_long", in_long), ("enr_t", enr_t), ("enr_o", enr_o)]:
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) >= 1.0:
            raise ValueError(f"test-vector signal {name} is not finite or would clip")

    sources = ([source_entry("target (mixture)", r) for r in used_t]
               + [source_entry("talker 1 (mixture)", r) for r in used_1]
               + [source_entry("own voice (mixture)", r) for r in used_o]
               + [source_entry("noise (mixture)", noise_mix)]
               + [source_entry("target enrollment", r) for r in enr_rows]
               + [source_entry("target enrollment noise", noise_enr)]
               + [source_entry("own enrollment", r) for r in cache_rows]
               + [{"role": "room", "path": "rir/sim_test.npy", "room": 0, "positions": [0, 1, 4, 5, 6],
                   "license": "generated by dataprep/rir_build.py (pyroomacoustics simulation)"}])
    scene = {"seed": seed, "builder": "ml/earshot/testaudio.py testvec_inputs (render.py not built yet)",
             "speakers": {"target": spk_t, "talker1": spk_1, "own": spk_o}, "room": "sim_test room 0",
             "sir_db": TV_SIR_DB, "own_db": TV_OWN_DB, "snr_db": TV_SNR_DB, "enr_t_snr_db": TV_ENR_SNR_DB,
             "talker1_active_frac": round(frac_1, 4), "own_active_frac": round(frac_o, 4),
             "mic_chain": "highpass 100 Hz, no EQ, fir_decim, -26 dBFS RMS, no clipping (test mode)",
             "enr_o": "clean, 6.3 recipe, -26 dBFS active power"}
    return {"in_long": in_long, "enr_t": enr_t, "enr_o": enr_o, "sources": sources, "scene": scene}


# ---------------------------------------------------------------------------------------------------------------
# Quality-vector captures (contract 6, 10-ml 7.4): 5 fixed 16 kHz captures for the check formulas
# ---------------------------------------------------------------------------------------------------------------
BED_DBFS = -50.0        # white noise bed under the speech captures


def noise_bed(n, rng):
    return (rng.standard_normal(n) * db_to_amp(BED_DBFS)).astype(np.float32)


def speech_block(rows, n, rng, gap_range=(0.1, 0.5)):
    """n samples of one speaker's utterances (gaps U(gap_range) s), at -26 dBFS active power -> (audio, rows used)."""
    audios, used = read_rows_until(rows, n / SR + 2.0)
    first = audios[0]
    start = int(rng.integers(0, max(1, len(first) // 4)))
    x = concat_with_gaps([first[start:]] + audios[1:], n, gap_range, rng)
    return set_active_level(x, LEVEL_DBFS), used


def no_nonspeech_capture(n):
    """A voiced tone (pitch around 140 Hz, 5 harmonics, -26 dBFS RMS) where every 5th token frame is quieter by
    20 + 0.5 i dB (i = 0, 1, 2, ...). Every quiet frame sits between loud ones, so es_vad marks all frames after its
    1-frame dilation: no non-speech frame, so sbr uses the lowest-10% fallback. The 0.5 dB steps keep the energy
    order of the quiet frames unambiguous."""
    t = np.arange(n) / SR
    f0 = 140.0 + 20.0 * np.sin(2.0 * np.pi * 0.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / SR
    voice = np.zeros(n)
    for harmonic in range(1, 6):
        voice += np.sin(harmonic * phase) / harmonic
    voice *= db_to_amp(LEVEL_DBFS) / rms(voice)
    n_frames = (n - 64) // 512 + 1
    for f in range(0, n_frames, 5):
        a, b = max(0, 512 * f - 448), min(n, 512 * f + 64)
        voice[a:b] *= db_to_amp(-(20.0 + 0.5 * (f // 5)))
    return voice.astype(np.float32)


def clip_to_fraction(x, fraction):
    """Scale x so that exactly round(fraction * n) samples reach |x| >= 0.98, then hard clip at full scale (+-1).

    The gain puts 0.98 halfway between the k-th and (k+1)-th largest |x|, so exactly k samples land at or above it.
    """
    n = len(x)
    k = int(round(fraction * n))
    mags = np.sort(np.abs(x.astype(np.float64)))[::-1]
    gain = 0.98 / (0.5 * (mags[k - 1] + mags[k]))
    y = np.clip(x.astype(np.float64) * gain, -1.0, 1.0).astype(np.float32)
    count = int(np.count_nonzero(np.abs(y) >= np.float32(0.98)))
    if count != k or not np.any(np.abs(y) == 1.0):
        raise ValueError(f"clip_to_fraction: got {count} samples at |x| >= 0.98, wanted {k}")
    return y


def quality_captures(seed=7):
    """The 5 fixed captures of quality_vectors.json -> list of dicts {name, description, pcm f32 [n], sources}.

    1. lead_noise_10s   - 2 s of noise only, then a clean 8 s talker (noise bed -50 dBFS throughout)
    2. long_14s         - 14 s capture (longer than 10 s, with a 1.5 s pause), so the best 10 s window is used
    3. no_nonspeech_6s  - no frame with mask0 = 0, so sbr uses the lowest-10% fallback
    4. rounds_to_4s     - 4.2 s capture whose target clip rounds down to exactly 64,000 samples
    5. clipped_8s       - 8 s with exactly 1.5% of samples at |x| >= 0.98 (hard clipped at full scale)
    """
    from .quality import MIN_CLIP, trim     # imported here: quality.py does not depend on this file

    rng = np.random.default_rng([seed, 1])
    speakers = vctk_test_speakers()
    ids = list(speakers)
    order = rng.permutation(len(ids))
    captures = []

    # 1. Two seconds of noise only, then 8 s of one talker.
    n = 160000
    speech, used = speech_block(speakers[ids[order[0]]], 128000, rng)
    x = noise_bed(n, rng)
    x[32000:] += speech
    captures.append({"name": "lead_noise_10s", "description": "2 s of noise (-50 dBFS) then a clean 8 s talker",
                     "pcm": x, "sources": [source_entry("talker", r) for r in used]})

    # 2. 14 s: speech from 0.5 s to 13.5 s with a 1.5 s pause at 6.0 to 7.5 s.
    n = 224000
    speech, used = speech_block(speakers[ids[order[1]]], n, rng)
    speech[:8000] = 0.0
    speech[96000:120000] = 0.0
    speech[216000:] = 0.0
    x = noise_bed(n, rng) + speech
    captures.append({"name": "long_14s", "description": "14 s capture, speech 0.5-6.0 s and 7.5-13.5 s, noise bed",
                     "pcm": x, "sources": [source_entry("talker", r) for r in used]})

    # 3. Synthetic: every frame counts as speech.
    captures.append({"name": "no_nonspeech_6s", "description": "voiced tone, every 5th frame 20+ dB quieter: "
                     "es_vad marks every frame, sbr uses the lowest-10% fallback",
                     "pcm": no_nonspeech_capture(96000), "sources": []})

    # 4. 4.2 s cut from a dense speech stream at the first 512-sample step whose target clip is exactly 64,000.
    n = 67200
    stream, used = speech_block(speakers[ids[order[2]]], 20 * SR, rng, gap_range=(0.1, 0.25))
    bed = noise_bed(n, rng)
    x = None
    for start in range(0, len(stream) - n, 512):
        candidate = (stream[start:start + n] + bed).astype(np.float32)
        result = trim(candidate, "target")
        if result["fail"] is None and result["clip_len"] == MIN_CLIP:
            x = candidate
            break
    if x is None:
        raise ValueError("no 4.2 s excerpt rounds to exactly 4 s")
    captures.append({"name": "rounds_to_4s", "description": "4.2 s capture whose target clip rounds to exactly 4 s",
                     "pcm": x, "sources": [source_entry("talker", r) for r in used]})

    # 5. 8 s of loud speech with exactly 1.5% of the samples at |x| >= 0.98.
    n = 128000
    speech, used = speech_block(speakers[ids[order[3]]], n, rng)
    x = clip_to_fraction(noise_bed(n, rng) + speech, 0.015)
    captures.append({"name": "clipped_8s", "description": "8 s talker, 1.5% of samples at |x| >= 0.98, hard clipped "
                     "at full scale", "pcm": x, "sources": [source_entry("talker", r) for r in used]})

    for c in captures:
        c["pcm"] = np.asarray(c["pcm"], dtype=np.float32)
        if not np.all(np.isfinite(c["pcm"])) or np.max(np.abs(c["pcm"])) > 1.0:
            raise ValueError(f"quality capture {c['name']} is not finite or exceeds full scale")
    return captures


# ---------------------------------------------------------------------------------------------------------------
# Test-vector inputs rendered by render.render_example in test mode (10-ml 7.4 and 8.9)
# ---------------------------------------------------------------------------------------------------------------
# The scene of 10-ml 7.4 written as render overrides: sim_test room 0, target (always active) + 1 talker at SIR
# 0 dB + own voice at +6 dB + one CC0 FSD50K noise clip at SNR 5 dB, no babble; target enrollment in the same room
# with noise at 10 dB and no weaker talker. Test mode adds: -26 dBFS, high-pass 100 Hz, no EQ, fir_decim, no
# clipping, no Lombard, no self-noise.
TV_OVERRIDES = {
    "person": True, "absent": False, "room": "sim", "room_index": 0, "n_talkers": 1, "sir_db": TV_SIR_DB,
    "own_given": True, "own_in_mix": True, "own_db": TV_OWN_DB, "babble": False, "background": "noise",
    "n_noise": 1, "noise_kinds": ("fsd50k", "fsd50k"), "quiet": False, "snr_db": TV_SNR_DB, "lombard": False,
    "target_frac": 1.0, "enr_room": "same", "enr_weak": False, "enr_noise": True, "enr_noise_babble": False,
    "enr_snr_db": TV_ENR_SNR_DB, "use_wes": True, "own_noise": False,
}


def testvec_inputs_render(seed=7):
    """Same return value as testvec_inputs, but the audio comes from data.make_example (render.render_example in
    test mode) on the test split: speech_test (VCTK test speakers, CC BY 4.0), noise_test rows with license
    CC0-1.0 only (FSD50K), sim_test room 0. Generator default_rng([seed, 0]).

    enr_t is the rendered target enrollment (6 s, other utterances of the target, noise at 10 dB); enr_o is the
    clean own clip of 6.3 (sources.build_clip, -26 dBFS active power), as the phone stores it.
    """
    from . import data                         # imported here: the hand-built path above does not need them
    from .sources import Sources

    src = Sources("test", noise_licenses={"CC0-1.0"})
    rng = np.random.default_rng([seed, 0])
    ex = data.make_example(src, "B", rng, TV_LONG, TV_ENR_T, overrides=TV_OVERRIDES, test_mode=True,
                           record_sources=True)
    scene, info = ex["scene"], ex["info"]
    in_long, enr_t, enr_o = ex["mix"], ex["enr_t"], ex["own_clean"]
    for name, x in [("in_long", in_long), ("enr_t", enr_t), ("enr_o", enr_o)]:
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) >= 1.0:
            raise ValueError(f"test-vector signal {name} is not finite or would clip")
    bad = [s for s in scene["sources"] if s.get("license") not in (None, "CC-BY-4.0", "CC0-1.0")]
    if bad:
        raise ValueError(f"test-vector sources with an unexpected license: {bad}")
    sources = scene["sources"] + [{"role": "room", "path": "rir/sim_test.npy", "room": 0,
                                   "license": "generated by dataprep/rir_build.py (pyroomacoustics simulation)"}]
    scene_meta = {"seed": seed, "builder": "ml/earshot/render.py render_example, test mode (10-ml 8.9)",
                  "speakers": {"target": info["target"],
                               "talker1": src.speakers[scene["talkers"][0]], "own": info["own"]},
                  "own_clip": info["own_clip"], "room": "sim_test room 0", "sir_db": TV_SIR_DB,
                  "own_db": TV_OWN_DB, "snr_db": TV_SNR_DB, "enr_t_snr_db": TV_ENR_SNR_DB,
                  "enr_t_noise": info["enr_noise"], "noise_kinds": info["noise_kinds"],
                  "mic_chain": "highpass 100 Hz, no EQ, fir_decim, -26 dBFS RMS, no clipping, no self-noise "
                               "(test mode)",
                  "enr_o": "clean, 6.3 recipe (sources.build_clip), -26 dBFS active power"}
    return {"in_long": in_long, "enr_t": enr_t, "enr_o": enr_o, "sources": sources, "scene": scene_meta}
