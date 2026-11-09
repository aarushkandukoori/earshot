"""Tests for the enrollment check formulas (ml/earshot/quality.py, contract 6) and quality_vectors.json.

10-ml 14 puts these in test_metrics.py. ML-0b wrote the quality part below; ML-1 adds the metrics.py tests
(si_snr, picked, TSOS, envelope correlation) to this same file.

Most tests pass mask0 by hand so every expected number can be worked out on paper:
token frame f covers raw samples [512f - 448, 512f + 64) and its midpoint is 512f - 192.
"""
import json

import numpy as np
import pytest

from ml import export
from ml.earshot import metrics, quality, testaudio
from ml.earshot.config import EXPORTS
from ml.earshot.vad import num_frames

VECTORS = EXPORTS / "random_v1" / "quality_vectors.json"


# ---------------------------------------------------------------------------------------------------------------
# quality.py, piece by piece
# ---------------------------------------------------------------------------------------------------------------
def test_frame_map():
    assert quality.frame_samples(0, 96000) == (0, 64)          # frame 0 is 448 padding zeros + 64 samples
    assert quality.frame_samples(1, 96000) == (64, 576)
    assert quality.frame_samples(187, 96000) == (95296, 95808)  # last frame of a 6 s clip (F = 188)
    assert quality.frame_mid(1) == 320 and quality.frame_mid(0) == -192


def test_frame_span():
    mask = np.zeros(20, dtype=np.uint8)
    assert quality.frame_span(mask, 10304) is None
    mask[3] = mask[10] = 1
    assert quality.frame_span(mask, 10304) == (512 * 3 - 448, 512 * 10 + 64)     # (1088, 5184)
    mask[0] = 1
    assert quality.frame_span(mask, 10304) == (0, 5184)


def test_best_window_counts_midpoints():
    # Speech frames 2, 3, 4 and 10..13; midpoints 832, 1344, 1856 and 4928, 5440, 5952, 6464.
    mask = np.zeros(20, dtype=np.uint8)
    mask[[2, 3, 4, 10, 11, 12, 13]] = 1
    # 2048-sample windows starting on 512-sample steps: [4608, 6656) holds all 4 midpoints of the second group.
    assert quality.best_window(mask, 0, 8000, length=2048) == 4608
    # A tie (2 midpoints in several windows) goes to the earliest start; steps are counted from the region start.
    mask = np.zeros(20, dtype=np.uint8)
    mask[[2, 3, 10, 11]] = 1
    assert quality.best_window(mask, 0, 8000, length=2048) == 0
    assert quality.best_window(mask, 64, 8000, length=2048) == 64


def test_trim_target_and_own():
    n = 200000                                   # 12.5 s, F = 391
    x = np.zeros(n, dtype=np.float32)
    mask0 = np.zeros(num_frames(n), dtype=np.uint8)
    mask0[20:361] = 1                            # speech in frames 20..360
    target = quality.trim(x, "target", mask0)
    assert target["span"] == (9792, 184384)      # (512*20 - 448, 512*360 + 64), 174,592 samples > 10 s
    assert target["region"] == (9792, 184384)
    # Every window from the span start holds 312 midpoints at most, so the earliest (the span start) wins.
    assert target["window_start"] == 9792 and target["clip_start"] == 9792 and target["clip_len"] == 160000
    own = quality.trim(x, "own", mask0)
    assert own["region"] == (0, n)
    # Own voice searches the whole capture on 512-sample steps from 0: [9728, 169728) is the first with 312.
    assert own["window_start"] == 9728 and own["clip_len"] == 160000 and own["fail"] is None

    # Short spans: 67,072 samples round down to 64,000 (pass); 61,952 round to 56,000 (fail).
    mask0[:] = 0
    mask0[20:151] = 1
    result = quality.trim(x[:100000], "target", mask0[:num_frames(100000)])
    assert result["span"] == (9792, 76864) and result["clip_len"] == 64000 and result["fail"] is None
    mask0[:] = 0
    mask0[20:141] = 1
    result = quality.trim(x[:100000], "target", mask0[:num_frames(100000)])
    assert result["clip_len"] == 56000 and result["fail"] == "too short"
    # Own voice under 10 s uses the whole capture, rounded.
    result = quality.trim(x[:70000], "own", mask0[:num_frames(70000)])
    assert result["window_start"] is None and (result["clip_start"], result["clip_len"]) == (0, 64000)
    assert quality.trim(x[:70000], "own", np.zeros(num_frames(70000), np.uint8))["fail"] == "no speech"


def test_windows_and_frames_inside():
    wins = quality.windows(160000)
    assert len(wins) == 18 and wins[0] == (0, 24000) and wins[-1] == (136000, 24000)
    assert len(quality.windows(64000)) == 6
    assert quality.windows(24000) == [(0, 24000)] and quality.windows(23999) == []
    n_frames = num_frames(160000)                # 313
    assert quality.frames_inside(0, n_frames) == (1, 47)        # midpoints 320 .. 23872
    assert quality.frames_inside(8000, n_frames) == (16, 62)
    assert quality.frames_inside(136000, n_frames) == (266, 312)


def frame_signal(amplitudes, n):
    """A capture whose raw samples in token frame f all equal amplitudes[f] (frame-aligned constant levels)."""
    x = np.zeros(n, dtype=np.float32)
    for f, a in enumerate(amplitudes):
        lo, hi = quality.frame_samples(f, n)
        x[lo:hi] = a
    return x


def test_sbr_db_by_hand():
    n = 10304                                    # F = 21, the frames cover every sample
    amplitudes = [0.01] * 21
    for f in range(5, 15):
        amplitudes[f] = 0.1
    x = frame_signal(amplitudes, n)
    mask0 = np.zeros(21, dtype=np.uint8)
    mask0[5:15] = 1
    expected = 10 * np.log10((0.1 ** 2 + 1e-12) / (0.01 ** 2 + 1e-12))
    assert abs(quality.sbr_db(x, mask0) - expected) < 1e-6        # 20 dB
    assert quality.sbr_db(x, np.zeros(21, np.uint8)) is None      # no speech frame


def test_sbr_db_fallback_uses_lowest_energy_frames():
    n = 10304
    amplitudes = [0.01 * (f + 1) for f in range(21)]
    amplitudes[7] = 0.001                        # the quietest full frame
    x = frame_signal(amplitudes, n)
    mask0 = np.ones(21, dtype=np.uint8)
    speech, other, fallback = quality.sbr_split(x, mask0)
    # F // 10 = 2 lowest es_vad energies: frame 0 (only 64 real samples next to 448 padding zeros) and frame 7.
    assert fallback and list(other) == [0, 7] and len(speech) == 19
    p_other = (64 * 0.01 ** 2 + 512 * 0.001 ** 2) / (64 + 512)
    p_speech = sum(512 * amplitudes[f] ** 2 for f in range(21) if f not in (0, 7)) / (19 * 512)
    assert abs(quality.sbr_db(x, mask0) - 10 * np.log10((p_speech + 1e-12) / (p_other + 1e-12))) < 1e-6


def test_clip_pct():
    x = np.zeros(1000, dtype=np.float32)
    x[:15] = 0.98
    x[15:20] = -1.0
    x[20] = 0.9799999
    assert quality.clip_pct(x) == 2.0


def test_own_window_mask_and_speech_seconds():
    mask = np.ones(num_frames(160000), dtype=np.uint8)
    wins = quality.windows(160000)
    cos = [0.1] * len(wins)
    cos[3] = 0.61                                # window 3 = [24000, 48000): frames 48..94 inside
    cos[5] = 0.60                                # equal to the threshold: kept (the rule is a strict >)
    out = quality.own_window_mask(mask, wins, cos, 0.60)
    assert list(np.flatnonzero(out == 0)) == list(range(48, 95))
    assert quality.speech_seconds(out) == pytest.approx((313 - 47) * 0.032)
    assert np.all(mask == 1)                     # the input is not changed


def test_score_with_a_fake_encoder():
    """The WeSpeaker steps with a stand-in encoder: clip -> e, window 2 -> the user's own vector."""
    x = testaudio.no_nonspeech_capture(96000)    # synthetic, no data pack needed; target clip = 88,000 samples
    e, own = np.eye(256, dtype=np.float32)[0], np.eye(256, dtype=np.float32)[1]
    calls = []

    def embed(clip):
        calls.append(len(clip))
        return own if len(calls) == 4 else e     # call 1 = clip, calls 2.. = windows 0, 1, 2, ...

    s = quality.score(x, "target", embed=embed, own_e=own, saved_e=[e])
    assert calls[0] == 88000 and calls[1:] == [24000] * len(s["windows"])
    assert s["one_voice"] == 1.0 and s["own_cos"] == 0.0 and s["dup"] == 1.0
    lo, hi = s["frames_inside"][2]
    assert np.all(s["mask"][lo:hi + 1] == 0)
    assert s["speech_frames"] == s["vad_frames"] - (hi - lo + 1)
    plain = quality.score(x, "target")           # no WeSpeaker: nothing checked, nothing removed
    assert plain["one_voice"] is None and plain["speech_frames"] == plain["vad_frames"]


# ---------------------------------------------------------------------------------------------------------------
# quality_vectors.json (contract 6; the app's ClipToolsTests checks the same file)
# ---------------------------------------------------------------------------------------------------------------
def load_vectors():
    if not VECTORS.exists():
        pytest.skip(f"{VECTORS} not written yet (python -m ml.export --random --seed 0 --out ml/exports/random_v1/)")
    with open(VECTORS, encoding="utf-8") as f:
        return json.load(f)


def test_quality_vectors_reproduce():
    obj = load_vectors()
    assert obj["version"] == 1 and obj["kind"] == "earshot-quality-vectors"
    names = [v["name"] for v in obj["vectors"]]
    assert names == ["lead_noise_10s", "long_14s", "no_nonspeech_6s", "rounds_to_4s", "clipped_8s"]
    by_name = {}
    for v in obj["vectors"]:
        pcm = export.decode_pcm(v)
        assert len(pcm) == v["n"] == v["expected"]["n"]
        assert quality.vector_values(pcm) == v["expected"], v["name"]
        by_name[v["name"]] = v["expected"]
    # Each capture shows the case it was made for.
    assert by_name["lead_noise_10s"]["span"][0] >= 32000 - 512
    assert by_name["long_14s"]["n"] > 160000 and by_name["long_14s"]["target"]["window_start"] is not None
    assert by_name["long_14s"]["own"]["window_start"] is not None
    assert by_name["no_nonspeech_6s"]["sbr_fallback"] is True
    assert by_name["no_nonspeech_6s"]["mask0_frames"] == num_frames(96000)
    assert by_name["rounds_to_4s"]["n"] == 67200 and by_name["rounds_to_4s"]["target"]["clip_len"] == 64000
    assert by_name["clipped_8s"]["clip_pct"] == 1.5


def test_quality_captures_are_fixed():
    if not testaudio.SPEECH_TEST.exists():
        pytest.skip("data pack not present")
    first = testaudio.quality_captures(7)
    again = testaudio.quality_captures(7)
    for a, b in zip(first, again):
        assert np.array_equal(a["pcm"], b["pcm"])
    if VECTORS.exists():
        stored = load_vectors()["vectors"]
        for capture, v in zip(first, stored):
            assert np.array_equal(capture["pcm"], export.decode_pcm(v)), v["name"]


# ---------------------------------------------------------------------------------------------------------------
# metrics.py (10-ml 12.2): SI-SNR, picked rate, attenuation, own-voice suppression, TSOS, envelope correlation
# ---------------------------------------------------------------------------------------------------------------
def gauss(n, seed, scale=1.0):
    return scale * np.random.default_rng([seed, 44]).standard_normal(n)


def orthogonal_noise(r, seed, snr_db):
    """Zero-mean noise orthogonal to the zero-mean r, scaled so that |r|^2 / |n|^2 = 10^(snr_db / 10)."""
    n = gauss(len(r), seed)
    n -= n.mean()
    n -= (n @ r) / (r @ r) * r                       # remove the part along r (stays zero-mean)
    return n * np.sqrt((r @ r) / (n @ n) / 10 ** (snr_db / 10))


def test_si_snr_known_values():
    r = gauss(16000, 1)
    r -= r.mean()
    assert metrics.si_snr(3.7 * r, r) > 100                  # a scaled copy: only rounding error is left
    for snr_db in (10.0, 0.0, -5.0):
        e = r + orthogonal_noise(r, 2, snr_db)
        assert abs(metrics.si_snr(e, r) - snr_db) < 1e-9
    # Nothing along r at all (exactly orthogonal, or a silent reference): |a r|^2 = 0, so the formula gives -inf.
    assert metrics.si_snr(np.array([1.0, 1.0, -1.0, -1.0]), np.array([1.0, -1.0, 1.0, -1.0])) == -np.inf
    assert metrics.si_snr(r, np.zeros_like(r)) == -np.inf
    assert metrics.si_snr(orthogonal_noise(r, 3, 0.0), r) < -200        # orthogonal up to rounding


def test_si_snr_is_scale_and_offset_invariant():
    r = gauss(8000, 4)
    e = r + gauss(8000, 5, 0.5)
    base = metrics.si_snr(e, r)
    for c in (0.01, 5.0, -2.0):
        assert abs(metrics.si_snr(c * e, r) - base) < 1e-9
        assert abs(metrics.si_snr(e, c * r) - base) < 1e-9
    assert abs(metrics.si_snr(e + 0.3, r - 0.1) - base) < 1e-9   # means are removed first
    # Works along the last axis of a stack, and si_snri is the difference of two si_snr values.
    stack = metrics.si_snr(np.stack([e, 2 * r]), np.stack([r, r]))
    assert abs(stack[0] - base) < 1e-9 and stack[1] > 100
    x = r + gauss(8000, 6, 2.0)
    assert abs(metrics.si_snri(e, r, x) - (base - metrics.si_snr(x, r))) < 1e-12


def chunk_vad(fracs, hops=500):
    """Hop VAD of len(fracs) chunks: chunk c is active in its first fracs[c] share of hops."""
    out = []
    for f in fracs:
        v = np.zeros(hops, dtype=np.uint8)
        v[:int(round(f * hops))] = 1
        out.append(v)
    return np.concatenate(out)


def test_picked_logic():
    n = 4 * 32000 + 1000                             # 4 full chunks and a tail that is dropped
    s, r1, r2 = gauss(n, 7), gauss(n, 8), gauss(n, 9)
    y = np.zeros(n)
    c = [slice(i * 32000, (i + 1) * 32000) for i in range(4)]
    y[c[0]] = s[c[0]] + 0.1 * r1[c[0]]               # chunk 0: follows the target            -> picked
    y[c[1]] = r1[c[1]] + 0.1 * s[c[1]]               # chunk 1: follows talker 1              -> not picked
    y[c[2]] = s[c[2]]                                # chunk 2: target active only 20%        -> not eligible
    y[c[3]] = s[c[3]] + 3.0 * r2[c[3]]               # chunk 3: closer to talker 2 than to s
    vad_t = np.concatenate([chunk_vad([1.0, 0.6, 0.2, 0.5]), np.ones(15, np.uint8)])
    vad_1 = np.concatenate([chunk_vad([0.5, 1.0, 1.0, 0.25]), np.ones(15, np.uint8)])
    # Talker 2 active in 20% of chunk 3: below 25%, so it is not compared and chunk 3 counts as picked.
    vad_2 = np.concatenate([chunk_vad([0.0, 0.0, 0.0, 0.2]), np.ones(15, np.uint8)])
    assert metrics.picked(y, s, [r1, r2], vad_t, [vad_1, vad_2]) == (2, 3)
    # At 30% talker 2 is compared, and the output is closer to it than to the target: chunk 3 is not picked.
    vad_2 = np.concatenate([chunk_vad([0.0, 0.0, 0.0, 0.3]), np.ones(15, np.uint8)])
    assert metrics.picked(y, s, [r1, r2], vad_t, [vad_1, vad_2]) == (1, 3)
    # No competing talker active anywhere: nothing is eligible.
    silent = np.zeros_like(vad_1)
    assert metrics.picked(y, s, [r1, r2], vad_t, [silent, silent]) == (0, 0)
    assert metrics.picked_rate([(1, 3), (3, 3)]) == pytest.approx(4 / 6)
    assert np.isnan(metrics.picked_rate([(0, 0)]))


def test_absent_attenuation():
    x = gauss(16000, 10)
    assert metrics.attenuation_db(x, 0.1 * x) == pytest.approx(20.0, abs=1e-9)
    assert metrics.attenuation_db(x, x) == pytest.approx(0.0, abs=1e-9)
    assert metrics.attenuation_db(x, np.zeros_like(x)) == np.inf
    summary = metrics.absent_summary([5.0, 15.0, 25.0, 30.0])
    assert summary == {"median": 20.0, "under10_pct": 25.0}


def test_own_voice_suppression():
    n_hops = 100
    o = np.full(64 * n_hops, 0.5)
    y = np.ones(64 * n_hops)
    y[:64 * 50] = 0.005                              # quiet on the own-only hops 0..49
    vad_o = np.zeros(n_hops, np.uint8)
    vad_o[:70] = 1
    vad_t = np.zeros(n_hops, np.uint8)
    vad_t[50:] = 1                                   # hops 50..69 have both talking: not own-only
    # 50 own-only hops = 3,200 samples = 0.2 s, just enough. 10 log10(0.25 / 0.005^2) = 40 dB.
    assert metrics.own_suppression_db(o, y, vad_o, vad_t) == pytest.approx(40.0, abs=1e-9)
    vad_t[49] = 1                                    # 49 hops = 3,136 samples: under 0.2 s, skipped
    assert metrics.own_suppression_db(o, y, vad_o, vad_t) is None
    summary = metrics.own_summary([None, 10.0, 20.0, 30.0, 40.0, 50.0])
    # 5th percentile with linear interpolation: index 0.05 * 4 = 0.2 -> 10 + 0.2 * 10 = 12.
    assert summary == {"median": 30.0, "own_db_p05": pytest.approx(12.0), "n": 5}


def test_tsos():
    s = gauss(32000, 11, 0.1)
    s[16000:24000] *= 1e-3                           # 60 dB quieter: those frames are not target-active
    n_frames = 1 + 32000 // 128                      # 512 / 2 zeros on both sides, hop 128
    over, active = metrics.tsos(s, 0.5 * s)          # half gain: 0.25 |S|^2 missing > 0.1 |S|^2 in every frame
    assert over == active
    fully_quiet = (24000 - 256) // 128 - (16000 + 256) // 128 + 1   # frames that lie inside the quiet part
    assert active <= n_frames - fully_quiet and active >= n_frames - fully_quiet - 8
    assert metrics.tsos(s, 0.8 * s) == (0, active)   # 0.04 |S|^2 missing: under 10%
    assert metrics.tsos(s, 2.0 * s) == (0, active)   # louder is never over-suppression
    assert metrics.tsos(np.zeros(8000), gauss(8000, 12)) == (0, 0)
    assert metrics.tsos_rate([(1, 4), (0, 4)]) == pytest.approx(1 / 8)
    assert np.isnan(metrics.tsos_rate([(0, 0)]))


def modulated_noise(n, seed):
    """Noise switched between loud and quiet in random 100 ms steps, so its envelope has a clear shape."""
    rng = np.random.default_rng([seed, 55])
    steps = rng.choice([0.05, 1.0], size=n // 1600 + 1)
    return rng.standard_normal(n) * np.repeat(steps, 1600)[:n]


def test_envelope_correlation_lag_search():
    a = modulated_noise(48000, 13)
    late = np.concatenate([np.zeros(800), a[:-800]])           # a delayed by 50 ms
    corr, lag_ms = metrics.env_corr_with_lag(a, late)
    assert lag_ms == 50.0 and corr > 0.95
    early = np.concatenate([a[480:], np.zeros(480)])           # a advanced by 30 ms
    corr, lag_ms = metrics.env_corr_with_lag(a, early)
    assert lag_ms == -30.0 and corr > 0.95
    other = modulated_noise(48000, 14)                          # a different envelope
    assert metrics.env_corr(a, other) < metrics.env_corr(a, late) - 0.3
    too_late = np.concatenate([np.zeros(3200), a[:-3200]])     # 200 ms: outside the +-100 ms search
    assert metrics.env_corr(a, too_late) < metrics.env_corr(a, late) - 0.3
    assert metrics.pearson(np.ones(10), np.arange(10)) == 0.0  # a constant has nothing to correlate


def test_optional_packages_report_not_available():
    s = gauss(32000, 15)
    for name, value in (("estoi", metrics.estoi_gain(s, s, s)), ("pesq", metrics.pesq_wb(s, s))):
        if metrics.available(name):
            assert value is None or np.isfinite(value)
        else:
            assert value is None and metrics.status()[name] == metrics.NOT_AVAILABLE
