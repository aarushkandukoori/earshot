"""Signal helpers shared by the model, the data loader and the tests (spec 10-ml section 2.2).

Torch functions take float32 tensors shaped [B, L] on any device. Numpy functions take 1-D float32 arrays.

Framing in one paragraph (contract 2.1). Every 64 samples (one hop, 4 ms) we look at the newest 512 samples (32 ms).
The analysis window `wa` rises slowly over 448 samples and falls fast over the last 64, so the newest samples count
most. After the network, the synthesis window `ws` keeps only the last 128 samples of the 512-sample frame, and
neighbouring frames overlap-add at a 64-sample hop. Because wa * ws on those 128 samples is a plain Hann window of
length 128, and two Hann halves 64 samples apart sum to exactly 1, an untouched spectrum comes back as the input.
"""
import math

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

from . import config
from .vad import num_frames

SR = config.SR            # 16,000 Hz
N_FFT = config.N_FFT      # 512
HOP = config.HOP          # 64
PAD = config.ANA_RISE     # 448 zeros of left padding


# ---------------------------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------------------------
def hann_np(L):
    """Periodic Hann window of length L in float64: H_L[n] = 0.5 - 0.5 cos(2 pi n / L)."""
    n = np.arange(L, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * math.pi * n / L)


def windows_np():
    """(wa, ws) as float32 numpy arrays [512], computed in float64 exactly as contract 2.1 writes them.

    wa[n] = sqrt(H_896[n])        for 0 <= n < 448   (slow rise)
    wa[n] = sqrt(H_128[n - 384])  for 448 <= n < 512 (fast fall: the second half of a 128-point Hann)
    ws[n] = 0                     for n < 384
    ws[n] = H_128[n - 384] / wa[n] for 384 <= n < 512
    """
    h896 = hann_np(896)
    h128 = hann_np(128)
    wa = np.zeros(512, dtype=np.float64)
    for n in range(512):
        if n < 448:
            wa[n] = math.sqrt(h896[n])
        else:
            wa[n] = math.sqrt(h128[n - 384])
    ws = np.zeros(512, dtype=np.float64)
    for n in range(384, 512):
        ws[n] = h128[n - 384] / wa[n]
    return wa.astype(np.float32), ws.astype(np.float32)


def windows():
    """(wa, ws) as float32 torch tensors [512] on the CPU (see windows_np for the formulas)."""
    wa, ws = windows_np()
    return torch.from_numpy(wa), torch.from_numpy(ws)


# One CPU copy plus a small cache of copies per device, so stft/istft do not rebuild the windows every call.
_WINDOW_CACHE = {}


def windows_on(device):
    """(wa, ws) on the given torch device (cached)."""
    key = str(torch.device(device))
    if key not in _WINDOW_CACHE:
        wa, ws = windows()
        _WINDOW_CACHE[key] = (wa.to(device), ws.to(device))
    return _WINDOW_CACHE[key]


# ---------------------------------------------------------------------------------------------------------------
# STFT and inverse STFT (offline form; the streaming form lives in stream_ref.py)
# ---------------------------------------------------------------------------------------------------------------
def stft(x):
    """[B, L] float32 -> [B, T, 257] complex64 with T = L / 64.

    The signal is left-padded with 448 zeros and cut into 512-sample frames at hop 64, so frame t covers input
    samples 64t - 448 .. 64t + 63 (it ends at sample 64t + 63). Each frame is multiplied by wa and goes through
    torch.fft.rfft (unnormalized).
    """
    L = x.shape[-1]
    if L % HOP != 0:
        raise ValueError(f"stft needs a length that is a multiple of {HOP}, got {L}")
    wa, _ = windows_on(x.device)
    frames = F.pad(x, (PAD, 0)).unfold(-1, N_FFT, HOP)   # [B, T, 512]
    return torch.fft.rfft(frames * wa, dim=-1)


def istft(Y, L):
    """[B, T, 257] complex -> [B, L] float32, aligned with the input of stft (same result as bench_merged.py istft).

    Each frame goes through irfft (norm "backward", so 1/512), is multiplied by ws, and only its last 128 samples
    (indices 384..511) are kept. Those 128-sample pieces overlap-add at hop 64: frame t covers output samples
    64t - 64 .. 64t + 63, so its first half (indices 384..447) lands on output block t - 1 and its second half
    (indices 448..511) on output block t. Output block n (samples 64n .. 64n + 63) is therefore
    second_half[n] + first_half[n + 1]. The last block has no frame T, so the last 64 output samples are not
    perfectly reconstructed.

    The bench prototype did the same overlap-add with F.fold and dropped the fold's first 64 samples; the sums are the
    same two-term float additions, so the outputs are bit-identical (test_dsp checks this). F.fold is avoided because
    its kernel (col2im) has no MPS version and silently runs on the CPU during training.
    """
    _, ws = windows_on(Y.device)
    frames = torch.fft.irfft(Y, n=N_FFT, dim=-1) * ws         # [B, T, 512]
    first_half = frames[..., N_FFT - 2 * HOP:N_FFT - HOP]      # [B, T, 64], indices 384..447
    second_half = frames[..., N_FFT - HOP:]                    # [B, T, 64], indices 448..511
    B, T = frames.shape[0], frames.shape[1]
    zeros = torch.zeros(B, 1, HOP, dtype=frames.dtype, device=frames.device)
    next_first = torch.cat([first_half[:, 1:], zeros], dim=1)  # first half of frame n + 1 (none after the last)
    out = (second_half + next_first).reshape(B, T * HOP)
    return out[:, :L]


def stft_tok(x):
    """Token-encoder frames (contract 2.4): [B, L] -> [B, F, 257] complex64 with F = vad.num_frames(L).

    Same window and padding as stft, but at hop 512 (no overlap). Frame f covers padded samples 512f .. 512f + 511.
    Below 576 samples F is 0 on purpose (the token encoder needs at least two frames), so an empty tensor is returned.
    """
    B, L = x.shape
    n_frames = num_frames(L)
    if n_frames == 0:
        return torch.zeros(B, 0, config.N_BINS, dtype=torch.complex64, device=x.device)
    wa, _ = windows_on(x.device)
    frames = F.pad(x, (PAD, 0)).unfold(-1, N_FFT, config.TOK_HOP)[:, :n_frames]   # [B, F, 512]
    return torch.fft.rfft(frames * wa, dim=-1)


# ---------------------------------------------------------------------------------------------------------------
# Room impulse responses
# ---------------------------------------------------------------------------------------------------------------
def early_rir(h, dp):
    """Early part of a room impulse response (data/rir/README.md recipe): 50 ms after the direct path plus a 5 ms fade.

    h is a 1-D RIR at 16 kHz and dp the index of its direct-path sample. We keep h[0 : dp + 880] (the delay before the
    direct path is kept, so the early reference stays time-aligned with the reverberant signal) and fade the last 80
    samples with a half cosine from 1 down to 0.
    """
    end = dp + 800                               # 800 samples = 50 ms after the direct path
    e = np.array(h[:end + 80], dtype=np.float32, copy=True)
    fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, 80)))   # 1 -> 0 over 80 samples (5 ms)
    n_fade = max(0, min(80, len(e) - end))       # a short RIR may end inside the fade
    e[end:end + n_fade] *= fade_curve[:n_fade].astype(np.float32)
    return e


# ---------------------------------------------------------------------------------------------------------------
# Biquad filters (second-order sections, scipy "sos" rows [b0, b1, b2, a0=1, a1, a2])
# ---------------------------------------------------------------------------------------------------------------
def sos_highpass(fc):
    """2nd-order Butterworth high-pass at fc Hz (-3 dB at fc)."""
    return scipy.signal.butter(2, fc, "highpass", fs=SR, output="sos")


def sos_peaking(f0, gain_db, q):
    """RBJ audio-EQ-cookbook peaking EQ: gain_db at f0, bandwidth set by q. Returns one sos row, shape [1, 6]."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / SR
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def sos_high_shelf(f0, gain_db):
    """RBJ audio-EQ-cookbook high shelf with slope S = 1: about gain_db above f0, 0 dB far below. Shape [1, 6]."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / SR
    cos_w0 = math.cos(w0)
    # alpha = sin(w0)/2 * sqrt((A + 1/A) * (1/S - 1) + 2); with S = 1 the square root is sqrt(2).
    alpha = math.sin(w0) / 2.0 * math.sqrt(2.0)
    two_sqrt_a_alpha = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_a_alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def apply_sos(sos, x):
    """Run a causal IIR filter (scipy.signal.sosfilt), so mixture and references stay sample-aligned. float32 out."""
    return scipy.signal.sosfilt(sos, x).astype(np.float32)


# The 64-tap filter that models the phone's 48 kHz -> 16 kHz decimation (generated by the lead on day 0).
DECIM_FIR_PATH = config.ASSETS / "decim16k_fir.npy"
_DECIM_FIR = None


def fir_decim(x):
    """Filter x with the engine's 64-tap decimation FIR, causally: np.convolve(x, h64)[:len(x)].

    Raises FileNotFoundError (with the command that makes the file) if ml/assets/decim16k_fir.npy is missing;
    render.py decides whether a missing file is allowed (only with allow_missing_decim, contract 11).
    """
    global _DECIM_FIR
    if _DECIM_FIR is None:
        if not DECIM_FIR_PATH.exists():
            raise FileNotFoundError(
                f"{DECIM_FIR_PATH} is missing. The lead makes it with "
                "`.venv/bin/python engine/tools/make_resampler_tables.py` and commits it (contract 11).")
        _DECIM_FIR = np.load(DECIM_FIR_PATH).astype(np.float64)
    return np.convolve(np.asarray(x, dtype=np.float64), _DECIM_FIR)[:len(x)].astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Activity labels and levels
# ---------------------------------------------------------------------------------------------------------------
def hop_vad(s):
    """Speech activity per 4 ms hop of a clean signal (10-ml 8.4) -> uint8 [len(s) // 64].

    For hop j: e[j] = mean of s^2 over the 256 samples (16 ms) that end at the end of hop j, i.e.
    s[64j - 192 : 64j + 64], with zeros before sample 0. The hop is active if its level is within 35 dB of the
    loudest hop and e[j] > 1e-9. Then every active hop also marks the 4 hops on each side (dilation).
    Hop j's label describes input samples 64j .. 64j + 63, the newest hop of STFT frame j.
    """
    s = np.asarray(s, dtype=np.float64)
    n_hops = len(s) // HOP
    if n_hops == 0:
        return np.zeros(0, dtype=np.uint8)
    # Energy of each hop on its own (sum of squares of its 64 samples).
    hop_energy = (s[:n_hops * HOP].reshape(n_hops, HOP) ** 2).sum(axis=1)
    # 16 ms energy = this hop plus the 3 hops before it (missing hops before the start count as 0), divided by 256.
    e = np.zeros(n_hops, dtype=np.float64)
    for j in range(n_hops):
        total = 0.0
        for back in range(4):
            if j - back >= 0:
                total += hop_energy[j - back]
        e[j] = total / 256.0
    e_db = 10.0 * np.log10(e + 1e-12)
    active = (e_db > e_db.max() - 35.0) & (e > 1e-9)
    # Dilate by 4 hops on each side.
    out = np.zeros(n_hops, dtype=np.uint8)
    for j in np.flatnonzero(active):
        lo = max(0, j - 4)
        hi = min(n_hops, j + 5)
        out[lo:hi] = 1
    return out


def active_power(x, vad):
    """Mean of x^2 over the samples of hops with vad = 1 (hop j = samples 64j .. 64j+63).

    Falls back to the mean over all samples when no hop is active.
    """
    x = np.asarray(x, dtype=np.float64)
    vad = np.asarray(vad)
    n_hops = min(len(vad), len(x) // HOP)
    active_hops = np.flatnonzero(vad[:n_hops])
    if len(active_hops) == 0:
        return float(np.mean(x ** 2)) if len(x) > 0 else 0.0
    total = 0.0
    for j in active_hops:
        seg = x[j * HOP:(j + 1) * HOP]
        total += float(np.sum(seg ** 2))
    return total / (len(active_hops) * HOP)


def fade(x, n):
    """Half-cosine fade-in over the first n samples and fade-out over the last n samples. Returns a new array.

    The fade-in gain is 0.5 - 0.5 cos(pi * i / n) for i = 0..n-1 (starts at exactly 0), the fade-out is its mirror
    image (ends at exactly 0 on the last sample).
    """
    y = np.array(x, dtype=np.float32, copy=True)
    n = min(n, len(y) // 2)
    if n <= 0:
        return y
    ramp = (0.5 - 0.5 * np.cos(math.pi * np.arange(n) / n)).astype(np.float32)
    y[:n] *= ramp
    y[len(y) - n:] *= ramp[::-1]
    return y
