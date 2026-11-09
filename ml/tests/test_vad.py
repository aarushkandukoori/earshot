"""Tests for ml/earshot/vad.py: frame counts, the es_vad rule and token grouping (10-ml 5 and 14, contract 2.4)."""
import numpy as np

from ml.earshot.vad import dilate1, es_vad, num_frames, threshold_mask, token_groups


def test_num_frames():
    expected = {575: 0, 576: 2, 1088: 3, 32000: 63, 96000: 188, 128000: 250, 160000: 313}
    for n, F in expected.items():
        assert num_frames(n) == F, n


def test_es_vad_short_clip_returns_empty():
    out = es_vad(np.ones(575, dtype=np.float32))
    assert out.dtype == np.uint8 and out.shape == (0,)


def test_es_vad_all_zeros():
    out = es_vad(np.zeros(32000, dtype=np.float32))
    assert out.dtype == np.uint8 and out.shape == (63,)
    assert out.sum() == 0     # E = -100 everywhere, thr = max(-94, -140) = -94, and -100 > -94 is false


def test_es_vad_tone_burst():
    # 2 s of silence with a 0.5 s tone. Frame f covers samples 512f - 448 .. 512f + 63.
    # The tone starts at sample 4672 (= the first sample of frame 10) and lasts 8000 samples, so frames 10..24 are
    # full of tone and frame 25 holds 320 tone samples (2 dB quieter). Tone frames are near -9 dB, silent frames
    # are -100 dB: P10 = -100 (index (63 - 1) // 10 = 6), thr = max(-94, about -49) = about -49.
    # Speech frames 10..25, then the 1-frame dilation adds 9 and 26.
    x = np.zeros(32000, dtype=np.float32)
    n = np.arange(8000)
    x[4672:12672] = 0.5 * np.sin(2 * np.pi * 300 * n / 16000)
    out = es_vad(x)
    expected = np.zeros(63, dtype=np.uint8)
    expected[9:27] = 1
    np.testing.assert_array_equal(out, expected)


# Frame energies (dB) for the ramp tests. F = 30, so P10 is the sorted value at index (30 - 1) // 10 = 2, which is
# -50 (sorted: -100, -70, -50, -48, ...). The max is -20, so thr = max(-50 + 6, -20 - 40) = -44.
# Frame 7 (-43) is speech. Frame 12 sits exactly on the threshold and must NOT be speech (strict >).
# A wrong P10 index of 3 would give thr = -42 and drop frame 7; an index of 1 would give thr = -60.
RAMP_DB = [-100.0, -70.0, -50.0] + [-48.0] * 4 + [-43.0] + [-48.0] * 4 + [-44.0] + [-48.0] * 4 + [-20.0] + [-48.0] * 12


def test_threshold_rule_p10_index_and_strict():
    energies = np.array(RAMP_DB, dtype=np.float64)
    assert len(energies) == 30
    mask = threshold_mask(energies)
    assert list(np.flatnonzero(mask)) == [7, 17]
    assert list(np.flatnonzero(dilate1(mask))) == [6, 7, 8, 16, 17, 18]


def test_es_vad_ramp_signal():
    # The same ramp built as a signal: every frame f >= 1 is a constant of amplitude 10^(dB/20), frame 0 is silent.
    # Frame 12 is set to -44.5 dB here (just under the threshold, since an exact float tie is not reachable).
    levels = list(RAMP_DB)
    levels[12] = -44.5
    n = 64 + 512 * 29
    assert num_frames(n) == 30
    x = np.zeros(n, dtype=np.float32)
    for f in range(1, 30):
        x[512 * f - 448: 512 * f + 64] = 10.0 ** (levels[f] / 20.0)
    out = es_vad(x)
    assert list(np.flatnonzero(out)) == [6, 7, 8, 16, 17, 18]


def test_token_groups_pool_runs_into_one_list():
    # Two runs of speech frames (4 and 6 frames): 10 kept frames make 3 groups across the run boundary; frame 14 is
    # a leftover and dropped. 3 tokens is fewer than 4, which the caller flags as too short.
    mask = np.zeros(20, dtype=np.uint8)
    mask[2:6] = 1
    mask[9:15] = 1
    idx, valid, n_avail = token_groups(mask, 48)
    assert n_avail == 3
    assert idx.shape == (48, 3) and idx.dtype == np.int64 and valid.dtype == bool
    np.testing.assert_array_equal(idx[:3], [[2, 3, 4], [5, 9, 10], [11, 12, 13]])
    assert valid.sum() == 3 and np.all(idx[3:] == 0)
    assert n_avail < 4


def test_token_groups_subsample():
    # 312 kept frames make 104 groups; with 48 slots we keep groups floor(i * 104 / 48).
    mask = np.zeros(400, dtype=np.uint8)
    mask[:312] = 1
    idx, valid, n_avail = token_groups(mask, 48)
    assert n_avail == 104
    groups = np.arange(312).reshape(104, 3)
    for i in range(48):
        np.testing.assert_array_equal(idx[i], groups[(i * 104) // 48])
    assert valid.all()
    # The same mask with 16 own-voice slots.
    idx16, valid16, _ = token_groups(mask, 16)
    for i in range(16):
        np.testing.assert_array_equal(idx16[i], groups[(i * 104) // 16])
    assert valid16.all()


def test_token_groups_short():
    mask = np.zeros(50, dtype=np.uint8)
    mask[10:21] = 1                     # 11 frames -> 3 groups
    _, valid, n_avail = token_groups(mask, 16)
    assert n_avail == 3 and valid.sum() == 3
    _, valid, n_avail = token_groups(np.zeros(50, dtype=np.uint8), 48)
    assert n_avail == 0 and valid.sum() == 0
