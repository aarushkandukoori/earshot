"""Tests for ml/earshot/config.py and ml/earshot/dsp.py (10-ml 2.1, 2.2 and 14)."""
import math

import numpy as np
import pytest
import scipy.signal
import torch

from ml.earshot import config, dsp
from ml.earshot.vad import num_frames

from .helpers import noise, speech


# ---------------------------------------------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------------------------------------------
def test_config_hash_is_pinned():
    assert len(config.CONFIG) == 29
    assert config.config_hash() == "089d72c06d244e1650d209453c91ddffde990b9589dd7a9c228f9e1ea3b93aec"


def test_band_tables():
    assert sum(config.WIDTHS) == 257 and len(config.WIDTHS) == 16
    assert config.OFFSETS[12] == 128                      # bands 0-11 cover bins 0-127
    assert config.TAPS == [3] * 12 + [1] * 4
    assert sum(config.COEF_LEN) == 1026


def test_load_yaml_base_merge(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nnested: {x: 1, y: 2}\nlst: [1, 2]\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nnested: {y: 5}\nlst: [9]\nb: 3\n")
    cfg = config.load_yaml(tmp_path / "child.yaml")
    assert cfg == {"a": 1, "nested": {"x": 1, "y": 5}, "lst": [9], "b": 3}


# ---------------------------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------------------------
def test_windows_match_contract_formulas():
    wa, ws = dsp.windows()
    assert wa.dtype == torch.float32 and wa.shape == (512,) and ws.shape == (512,)
    # Recompute from the contract text with plain math in float64.
    for n in range(512):
        if n < 448:
            expect_a = math.sqrt(0.5 - 0.5 * math.cos(2 * math.pi * n / 896))
        else:
            expect_a = math.sqrt(0.5 - 0.5 * math.cos(2 * math.pi * (n - 384) / 128))
        if n < 384:
            expect_s = 0.0
        else:
            expect_s = (0.5 - 0.5 * math.cos(2 * math.pi * (n - 384) / 128)) / expect_a
        assert abs(wa[n].item() - expect_a) <= 1e-7
        assert abs(ws[n].item() - expect_s) <= 1e-7


def test_window_product_overlap_adds_to_one():
    wa, ws = dsp.windows_np()
    prod = wa.astype(np.float64) * ws.astype(np.float64)
    # Each output sample gets index 384 + n of one frame and index 448 + n of the previous frame.
    for n in range(64):
        assert abs(prod[384 + n] + prod[448 + n] - 1.0) <= 1e-6


# ---------------------------------------------------------------------------------------------------------------
# STFT and inverse
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["noise", "speech"])
def test_perfect_reconstruction(kind):
    L = 10 * 16000
    if kind == "noise":
        x = noise(L, seed=5, level=0.3)
    else:
        x = speech(L, speaker_index=3)
    xt = torch.from_numpy(x)[None]
    y = dsp.istft(dsp.stft(xt), L)[0].numpy()
    assert y.shape == (L,)
    err = np.abs(y[:L - 64] - x[:L - 64]).max()
    assert err <= 1e-6, f"reconstruction error {err}"


def test_stft_frame_counts_and_alignment():
    for L in [64, 640, 6400, 32000]:
        X = dsp.stft(torch.zeros(2, L))
        assert X.shape == (2, L // 64, 257) and X.dtype == torch.complex64
    with pytest.raises(ValueError):
        dsp.stft(torch.zeros(1, 100))
    # An impulse at sample s shows up in frame t exactly when 64t - 448 <= s <= 64t + 63 (frame t ends at 64t + 63).
    L, s = 6400, 3000
    x = torch.zeros(1, L)
    x[0, s] = 1.0
    energy = dsp.stft(x)[0].abs().sum(dim=-1)
    for t in range(L // 64):
        inside = 64 * t - 448 <= s <= 64 * t + 63
        assert (energy[t].item() > 0) == inside, t


def test_stft_tok_frame_counts():
    for L in [575, 576, 1088, 32000, 96000]:
        X = dsp.stft_tok(torch.zeros(1, L))
        assert X.shape == (1, num_frames(L), 257)


# ---------------------------------------------------------------------------------------------------------------
# Room impulse response helper
# ---------------------------------------------------------------------------------------------------------------
def test_early_rir():
    rng = np.random.default_rng(0)
    h = rng.standard_normal(8000).astype(np.float32)
    dp = 123
    e = dsp.early_rir(h, dp)
    assert len(e) == dp + 880
    end = dp + 800
    np.testing.assert_array_equal(e[:end], h[:end])        # delay and direct path kept unchanged
    assert e[dp] == h[dp]
    curve = 0.5 * (1 + np.cos(np.linspace(0, np.pi, 80)))
    np.testing.assert_allclose(e[end:], h[end:end + 80] * curve, atol=1e-7)
    assert e[-1] == 0.0                                    # the fade ends at exactly zero


# ---------------------------------------------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------------------------------------------
def gain_db_at(sos, freq):
    _, h = scipy.signal.sosfreqz(sos, worN=[freq], fs=16000)
    return 20 * np.log10(np.abs(h[0]))


def test_highpass_minus_3db_at_cutoff():
    for fc in [60.0, 120.0, 200.0]:
        assert abs(gain_db_at(dsp.sos_highpass(fc), fc) - (-3.0103)) <= 0.1


def test_peaking_gain_at_center():
    for f0, gain, q in [(300.0, -4.0, 0.7), (1000.0, 3.0, 1.2), (6000.0, 4.0, 2.0)]:
        sos = dsp.sos_peaking(f0, gain, q)
        assert abs(gain_db_at(sos, f0) - gain) <= 0.1
        assert abs(gain_db_at(sos, 20.0)) <= 0.1            # far below: flat


def test_high_shelf_gains():
    sos = dsp.sos_high_shelf(4000.0, 6.0)
    assert abs(gain_db_at(sos, 7999.0) - 6.0) <= 0.1       # near Nyquist: full shelf gain
    assert abs(gain_db_at(sos, 20.0)) <= 0.1               # DC: 0 dB


def test_apply_sos_is_causal():
    x = np.zeros(1000, dtype=np.float32)
    x[500] = 1.0
    y = dsp.apply_sos(dsp.sos_highpass(100.0), x)
    assert y.dtype == np.float32 and np.all(y[:500] == 0.0)


def test_fir_decim_or_clear_error():
    x = noise(1000, seed=9)
    if dsp.DECIM_FIR_PATH.exists():
        h = np.load(dsp.DECIM_FIR_PATH)
        np.testing.assert_allclose(dsp.fir_decim(x), np.convolve(x, h)[:1000], atol=1e-6)
    else:
        with pytest.raises(FileNotFoundError, match="make_resampler_tables"):
            dsp.fir_decim(x)


# ---------------------------------------------------------------------------------------------------------------
# Activity, levels, fades
# ---------------------------------------------------------------------------------------------------------------
def test_hop_vad_burst():
    # Silence with a sine burst over hops 100..199 (samples 6400..12799).
    s = np.zeros(400 * 64, dtype=np.float32)
    n = np.arange(6400, 12800)
    s[6400:12800] = 0.1 * np.sin(2 * np.pi * 440 * n / 16000)
    vad = dsp.hop_vad(s)
    assert vad.dtype == np.uint8 and vad.shape == (400,)
    # The 16 ms window of hop j covers hops j-3..j, so hops 100..202 contain burst energy (hop 202 sees 1/4 of a
    # window, -6 dB, well inside the 35 dB range); hop 203 sees none. Dilation by 4 gives hops 96..206.
    expected = np.zeros(400, dtype=np.uint8)
    expected[96:207] = 1
    np.testing.assert_array_equal(vad, expected)
    assert dsp.hop_vad(np.zeros(6400, dtype=np.float32)).sum() == 0   # e > 1e-9 rule


def test_active_power():
    x = np.zeros(640, dtype=np.float32)
    x[128:256] = 0.5                       # hops 2 and 3
    vad = np.zeros(10, dtype=np.uint8)
    vad[2:4] = 1
    assert abs(dsp.active_power(x, vad) - 0.25) <= 1e-9
    assert abs(dsp.active_power(x, np.zeros(10, dtype=np.uint8)) - 0.25 * 128 / 640) <= 1e-9


def test_fade():
    x = np.ones(1000, dtype=np.float32)
    y = dsp.fade(x, 80)
    assert y[0] == 0.0 and y[-1] == 0.0
    np.testing.assert_array_equal(y[80:920], x[80:920])
    assert np.all(np.diff(y[:80]) > 0) and np.all(np.diff(y[920:]) < 0)
    assert x[0] == 1.0                      # input untouched


def test_istft_equals_the_bench_fold_version():
    # The bench prototype's istft used F.fold for the overlap-add; ours adds the two halves directly.
    # Both make the same two-term float sums, so the results must be bit-identical.
    import torch.nn.functional as F

    def istft_fold(Y, L):
        _, ws = dsp.windows()
        fr = torch.fft.irfft(Y, n=512, dim=-1) * ws
        seg = fr[..., 384:].transpose(1, 2)
        T = seg.shape[-1]
        out = F.fold(seg, output_size=(1, (T + 1) * 64), kernel_size=(1, 128), stride=(1, 64))
        return out.reshape(out.shape[0], -1)[:, 64:64 + L]

    gen = torch.Generator().manual_seed(0)
    Y = torch.complex(torch.randn(2, 50, 257, generator=gen), torch.randn(2, 50, 257, generator=gen))
    assert torch.equal(dsp.istft(Y, 3200), istft_fold(Y, 3200))


def test_irfft_ignores_imaginary_part_of_dc_and_nyquist():
    # Contract 2.1 FFT convention, which the C engine copies: the imaginary parts of bins 0 and 256 are ignored.
    gen = torch.Generator().manual_seed(1)
    Y = torch.complex(torch.randn(3, 257, generator=gen), torch.randn(3, 257, generator=gen))
    Y_clean = Y.clone()
    Y_clean[:, 0] = Y[:, 0].real.to(torch.complex64)
    Y_clean[:, 256] = Y[:, 256].real.to(torch.complex64)
    assert torch.equal(torch.fft.irfft(Y, n=512), torch.fft.irfft(Y_clean, n=512))
