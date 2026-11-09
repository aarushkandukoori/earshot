"""The stability suite's long stream (10-ml 12.3 row "stability", contract 9 row "Stability").

One 60 min scene, seed 2029, built from the held-out test material (testsets.EvalSources) with render.py's
building blocks:
- one sim_test room (random, either geometry);
- the target (a held-out speaker from the target pool) talks in spells of U(2, 10) s separated by pauses of
  U(1, 2) times the spell, so it is active about 1 / (1 + 1.5) = 40% of the time;
- 2 competing talkers, each in spells of U(2, 10) s with pauses of U(0.5, 1.5) times the spell (active about
  50%), each at SIR 0 dB against the target (active power at the mic, as render.py measures it);
- DEMAND SCAFE (test) noise: one channel file (300 s), looped with 1 s equal-power crossfades to the full length,
  no room (an ambient recording), at SNR 5 dB against the target's active power;
- the test-mode mic chain (8.9: high-pass 100 Hz, fir_decim, -26 dBFS, headroom rule);
- a clean 6 s target enrollment in the same room and a clean own-voice enrollment (enroll.py, es_vad masks).

The suite itself (StreamRef on the CPU, NaN check, minutes 0-5 vs 55-60) runs in ml/evaluate.py.
Quick runs (--limit) build a shorter stream with exactly the same recipe.
"""
import math
import time

import numpy as np
import scipy.signal

from . import dsp, enroll, render
from .config import SR
from .testsets import EVAL_RECIPE, LE_EVAL, L_EVAL, SEEDS, eval_chain

STREAM_S = 3600                 # 60 min
SPELL_S = (2.0, 10.0)           # one talk spell
TARGET_PAUSE = (1.0, 2.0)       # target pause = spell x U(1, 2)   -> active about 40%
TALKER_PAUSE = (0.5, 1.5)       # talker pause = spell x U(0.5, 1.5) -> active about 50%
N_TALKERS = 2
SIR_DB = 0.0
SNR_DB = 5.0
XFADE_S = 1.0                   # noise loop crossfade
NOISE_PREFIX = "noise/demand/SCAFE/"


def convolve_long(x, h):
    """Causal convolution cut to len(x), by overlap-add (memory stays small for an hour of audio)."""
    if len(h) == 1:
        return (x * h[0]).astype(np.float32)
    return scipy.signal.oaconvolve(x, h)[:len(x)].astype(np.float32)


def spells(src, rows, n, rng, pause):
    """n samples of one talker in talk spells and pauses (dry, before any level change).

    It starts with a pause of U(0, 2) s; each spell is a fresh render.speech_stream window of U(2, 10) s (10 ms fades
    at both ends) and is followed by a pause of U(pause) times the spell length.
    """
    out = np.zeros(n, dtype=np.float32)
    pos = int(rng.uniform(0.0, 2.0) * SR)
    while pos < n:
        spell = int(rng.uniform(*SPELL_S) * SR)
        take = min(spell, n - pos)
        out[pos:pos + take] = dsp.fade(render.speech_stream(src, rows, take, rng), render.ACTIVITY_FADE)
        pos += take + int(spell * rng.uniform(*pause))
    return out


def loop_crossfade(x, n, xfade):
    """Repeat x until n samples; each repeat overlaps the previous one by `xfade` samples with an equal-power
    crossfade (gains cos and sin of a quarter period, so the two squared gains always sum to 1)."""
    ramp = np.sin(0.5 * math.pi * (np.arange(xfade) + 0.5) / xfade).astype(np.float32)   # rising gain
    out = np.zeros(n + len(x), dtype=np.float32)
    out[:len(x)] = x
    pos = len(x)                                  # end of the audio written so far
    step = len(x) - xfade
    while pos < n:
        start = pos - xfade
        out[start:pos] = out[start:pos] * ramp[::-1] + x[:xfade] * ramp
        out[pos:pos + step] = x[xfade:]
        pos += step
    return out[:n]


def build_stream(src, seconds=STREAM_S, seed=SEEDS["stability"]):
    """The stability scene -> dict with mix, ref (f32 [n]), the enrollments and token groups, own_clean, info.

    src is a testsets.EvalSources. All randomness comes from default_rng([seed, 0]), so the scene is the same on
    every run; `seconds` only sets the length (a multiple of 64 samples).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng([seed, 0])
    n = (int(seconds * SR) // 64) * 64
    room = render.pick_room(src, "sim", rng)
    spk_t, enr_rows, stream_rows, _ = render.pick_target(src, rng, L_EVAL, LE_EVAL)
    used = {spk_t}
    talkers = [render.pick_distinct(len(src.speakers), used, rng) for _ in range(N_TALKERS)]
    own_spk = render.pick_distinct(src.own_pool, used, rng)

    # Target: spells at -26 dBFS active power, reverberant image t (mixture) and early reference s.
    dry = render.scale_to_active(spells(src, stream_rows, n, rng, TARGET_PAUSE), render.LEVEL_1M_DBFS)
    t = convolve_long(dry, room.h[0])
    s = convolve_long(dry, dsp.early_rir(room.h[0], room.dp[0]))
    del dry
    vad_s = dsp.hop_vad(s)
    p_t = dsp.active_power(t, vad_s)
    x = t.copy()
    del t

    # Competing talkers, each scaled to SIR 0 dB against the target.
    for k, spk in enumerate(talkers):
        dry = spells(src, src.rows_of[spk], n, rng, TALKER_PAUSE)
        img = convolve_long(dry, room.h[1 + k])
        p_i = max(dsp.active_power(img, dsp.hop_vad(dry)), 1e-20)
        x += img * np.float32(math.sqrt(p_t / (p_i * 10.0 ** (SIR_DB / 10.0))))
        del dry, img

    # DEMAND SCAFE noise looped with 1 s crossfades, SNR 5 dB.
    files = [i for i, p in enumerate(src.noise_paths) if p.startswith(NOISE_PREFIX)]
    noise_row = files[int(rng.integers(len(files)))]
    clip = render.read_row(src.noise_paths[noise_row])
    noise = loop_crossfade(dsp.fade(clip, render.FADE), n, int(XFADE_S * SR))
    x += render.set_power(noise, p_t / 10.0 ** (SNR_DB / 10.0))
    del noise

    # Test-mode mic chain: same filters and gain on the reference.
    chain = eval_chain()
    x_f = render.chain_filters(chain, x)
    del x
    gain = render.headroom_gain(render.level_gain(chain.level_dbfs, x_f), x_f)
    mix = np.clip(x_f * gain, -1.0, 1.0).astype(np.float32)
    del x_f
    ref = (render.chain_filters(chain, s) * gain).astype(np.float32)
    del s

    # Enrollments with enroll.py: clean target enrollment in the same room, clean own enrollment, es_vad masks.
    d = render.draw_scalars(EVAL_RECIPE, rng, {"person": True, "absent": False, "own_given": True,
                                                "enr_room": "same", "enr_weak": False, "enr_noise": False,
                                                "enr_mask": "vad", "use_wes": True, "own_noise": False})
    scene = {"draws": d, "chain": chain, "room": room, "enr_rows": enr_rows, "used": used, "talkers": talkers,
             "own": own_spk}
    tgt = enroll.render_enrollment(src, scene, rng, LE_EVAL, test_mode=True)
    own = enroll.render_own_enrollment(src, scene, rng, test_mode=True)
    info = {"seconds": n / SR, "target": src.speakers[spk_t], "talkers": "|".join(src.speakers[k] for k in talkers),
            "own_speaker": src.speakers[own_spk], "room_index": room.index, "geometry": room.geometry,
            "rt60": round(room.rt60, 3), "noise": src.noise_paths[noise_row],
            "target_active": round(float(np.mean(vad_s)), 3), "build_s": round(time.perf_counter() - t0, 1)}
    return {"mix": mix, "ref": ref, "enr_t": tgt["enr_t"], "tok_idx_t": tgt["tok_idx_t"],
            "tok_val_t": tgt["tok_val_t"], "enr_o": own["enr_o"], "tok_idx_o": own["tok_idx_o"],
            "tok_val_o": own["tok_val_o"], "own_clean": own["clean"], "has_t": True, "has_o": True,
            "use_wes": True, "mode": np.array([1.0, 0.0], np.float32), "info": info}


def window_bounds(n, window_s=300.0, stream_s=STREAM_S):
    """Sample ranges (first, last) of the comparison windows of an aligned stream of n samples: minutes 0-5 and
    55-60 of the 60 min stream. A shorter (quick) stream compares its first and last quarter instead, so the
    windows stay long enough to hold target speech."""
    if n >= stream_s * SR - 64:                       # the full stream (one hop shorter after alignment)
        w = int(window_s * SR)
    else:
        w = (n // 4 // 64) * 64
    return (0, w), (n - w, n)
