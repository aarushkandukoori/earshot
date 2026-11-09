"""Evaluation metrics (spec 10-ml section 12.2, contract 9). Plain numpy in float64, one clip at a time.

Signals are 1-D arrays at 16 kHz, all aligned with each other (the offline model output is aligned with its input):
  x = mixture, s = target reference, y = model output, o = own voice image at the mic, r_i = competing talker i.
VAD labels are per 64-sample hop (hop j = samples 64j .. 64j + 63), as made by dsp.hop_vad.

Per-clip functions return numbers (or counts); the small summary helpers turn a list of per-clip results into the
numbers the report and val.csv show (medians, percentiles, pooled rates).

ESTOI (pystoi) and PESQ (pesq) are imported only when used. Neither is installed yet (10-ml 12.4): their functions
then return None and `available(name)` / `status()` say "not available".
"""
import importlib

import numpy as np
import scipy.signal

from .config import HOP, SR
from .dsp import hann_np

NOT_AVAILABLE = "not available"
CHUNK = 32000                 # picked rate works on non-overlapping 2 s chunks
ACTIVE_FRAC = 0.25            # a talker counts in a chunk if active in >= 25% of its hops
OWN_MIN_SAMPLES = 3200        # own-voice suppression skips clips with under 0.2 s of own-only audio
ABSENT_LOW_DB = 10.0          # contract 9: at most 5% of absent clips under 10 dB
TSOS_FFT, TSOS_HOP = 512, 128
TSOS_RANGE_DB = 30.0          # target-active frames: within 30 dB of the loudest frame
TSOS_RATIO = 0.1              # over-suppressed: missing energy > 10% of the frame's target energy
ENV_FRAME, ENV_HOP = 320, 160     # 20 ms RMS frames every 10 ms
ENV_MAX_LAG = 10              # +-100 ms in 10 ms envelope hops
ENV_BAND = (100.0, 4000.0)    # band-pass before the envelope


def _f64(a):
    return np.asarray(a, dtype=np.float64)


# ---------------------------------------------------------------------------------------------------------------
# SI-SNR
# ---------------------------------------------------------------------------------------------------------------
def si_snr(e, r):
    """Scale-invariant SNR of estimate e against reference r, in dB (works on the last axis, so [..., L] too).

    Remove both means; project e onto r: a = <e, r> / (<r, r> + 1e-12); the target part is a r and the rest
    e - a r is error. SI-SNR = 10 log10(|a r|^2 / (|e - a r|^2 + 1e-12)). Scaling e or r by any non-zero number gives
    the same value. If e has no part along r (or r is all zeros) the result is -inf, exactly as the formula says.
    """
    e = _f64(e)
    r = _f64(r)
    e = e - e.mean(axis=-1, keepdims=True)
    r = r - r.mean(axis=-1, keepdims=True)
    a = (e * r).sum(axis=-1, keepdims=True) / ((r * r).sum(axis=-1, keepdims=True) + 1e-12)
    target = a * r
    noise = e - target
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10((target * target).sum(axis=-1) / ((noise * noise).sum(axis=-1) + 1e-12))


def si_snri(y, s, x):
    """SI-SNR improvement: si_snr(y, s) - si_snr(x, s). How much closer to the target the output is than the mix."""
    return si_snr(y, s) - si_snr(x, s)


# ---------------------------------------------------------------------------------------------------------------
# Picked rate (did the model follow the right person?)
# ---------------------------------------------------------------------------------------------------------------
def picked(y, s, talkers, vad_t, talker_vads, chunk=CHUNK, frac=ACTIVE_FRAC):
    """Picked chunks of one clip -> (n_picked, n_eligible).

    The clip is cut into non-overlapping chunks of 32,000 samples (a shorter tail is dropped). A chunk is eligible
    if the target is active in at least 25% of its hops and at least one competing talker is too. It is picked if
    the output is closer to the target than to every competing talker active in at least 25% of its hops:
    si_snr(y_c, s_c) > si_snr(y_c, r_ic). talkers is a list (or [n, L] array) of talker early references r_i and
    talker_vads the matching hop VADs.
    """
    y = _f64(y)
    s = _f64(s)
    vad_t = np.asarray(vad_t)
    hops = chunk // HOP                                    # 500 hops per chunk
    n_chunks = len(y) // chunk
    n_picked = 0
    n_eligible = 0
    for c in range(n_chunks):
        lo, hi = c * chunk, (c + 1) * chunk
        h_lo, h_hi = c * hops, (c + 1) * hops
        if np.mean(vad_t[h_lo:h_hi]) < frac:
            continue                                       # target not active enough
        active = [i for i in range(len(talkers)) if np.mean(np.asarray(talker_vads[i])[h_lo:h_hi]) >= frac]
        if not active:
            continue                                       # nobody to confuse the target with
        n_eligible += 1
        score_target = si_snr(y[lo:hi], s[lo:hi])
        if all(score_target > si_snr(y[lo:hi], _f64(talkers[i])[lo:hi]) for i in active):
            n_picked += 1
    return n_picked, n_eligible


def picked_rate(counts):
    """Pooled picked rate over clips: sum(picked) / sum(eligible) from a list of (n_picked, n_eligible).
    Returns NaN when no chunk was eligible. The confusion rate is 1 - picked rate."""
    total_picked = sum(c[0] for c in counts)
    total_eligible = sum(c[1] for c in counts)
    if total_eligible == 0:
        return float("nan")
    return total_picked / total_eligible


# ---------------------------------------------------------------------------------------------------------------
# Absent attenuation and own-voice suppression
# ---------------------------------------------------------------------------------------------------------------
def attenuation_db(x, y):
    """Absent attenuation of one clip: 10 log10(sum x^2 / sum y^2). A silent output gives +inf."""
    x = _f64(x)
    y = _f64(y)
    with np.errstate(divide="ignore"):
        return float(10.0 * np.log10(np.sum(x * x) / np.sum(y * y)))


def absent_summary(values):
    """-> {"median": median dB, "under10_pct": percent of clips under 10 dB} over a list of attenuation_db values."""
    v = _f64(values)
    if len(v) == 0:
        return {"median": float("nan"), "under10_pct": float("nan")}
    return {"median": float(np.median(v)), "under10_pct": float(100.0 * np.mean(v < ABSENT_LOW_DB))}


def hop_samples(hop_mask, n):
    """Expand a per-hop bool mask to a per-sample bool mask of length n (hop j covers samples 64j .. 64j + 63)."""
    m = np.repeat(np.asarray(hop_mask, dtype=bool), HOP)[:n]
    if len(m) < n:
        m = np.concatenate([m, np.zeros(n - len(m), dtype=bool)])
    return m


def own_suppression_db(o, y, vad_o, vad_t, min_samples=OWN_MIN_SAMPLES):
    """Own-voice suppression of one clip, or None if it has under 0.2 s of own-only audio.

    Own-only hops: vad_o = 1 and vad_t = 0 (the user talks, the target does not). Over those hops' samples,
    10 log10(sum o^2 / sum y^2) with o the own voice image at the mic.
    """
    o = _f64(o)
    y = _f64(y)
    own_only = (np.asarray(vad_o) > 0) & (np.asarray(vad_t) == 0)
    m = hop_samples(own_only, len(y))
    if m.sum() < min_samples:
        return None
    with np.errstate(divide="ignore"):
        return float(10.0 * np.log10(np.sum(o[m] ** 2) / np.sum(y[m] ** 2)))


def own_summary(values):
    """-> {"median", "own_db_p05", "n"} over per-clip own_suppression_db values (None entries are skipped).
    own_db_p05 is the 5th percentile: the contract wants 95% of clips at or above 18 dB."""
    v = _f64([val for val in values if val is not None])
    if len(v) == 0:
        return {"median": float("nan"), "own_db_p05": float("nan"), "n": 0}
    return {"median": float(np.median(v)), "own_db_p05": float(np.percentile(v, 5)), "n": int(len(v))}


# ---------------------------------------------------------------------------------------------------------------
# Target over-suppression (TSOS)
# ---------------------------------------------------------------------------------------------------------------
def stft_mag_np(x, n_fft, hop):
    """Magnitude STFT in float64, same framing as losses.stft_mag: n_fft / 2 zeros on both sides, frames of n_fft
    every hop, periodic Hann, rfft -> [frames, n_fft / 2 + 1]."""
    pad = n_fft // 2
    xp = np.concatenate([np.zeros(pad), _f64(x), np.zeros(pad)])
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * hann_np(n_fft), axis=-1))


def tsos(s, y):
    """Target over-suppression counts of one clip -> (n_over, n_active).

    STFT 512 / 128 (Hann) of s and y. A frame is target-active if its energy sum |S|^2 is within 30 dB of the loudest
    frame (and above 0). An active frame is over-suppressed if the energy the output is missing,
    sum relu(|S| - |Y|)^2, is more than 10% of sum |S|^2. A silent reference has no active frames: (0, 0).
    """
    S = stft_mag_np(s, TSOS_FFT, TSOS_HOP)
    Y = stft_mag_np(y, TSOS_FFT, TSOS_HOP)
    frame_energy = np.sum(S * S, axis=1)
    loudest = frame_energy.max() if len(frame_energy) else 0.0
    if loudest <= 0.0:
        return 0, 0
    active = (frame_energy > 0.0) & (frame_energy >= loudest * 10.0 ** (-TSOS_RANGE_DB / 10.0))
    missing = np.sum(np.maximum(S - Y, 0.0) ** 2, axis=1)
    over = active & (missing > TSOS_RATIO * frame_energy)
    return int(over.sum()), int(active.sum())


def tsos_rate(counts):
    """Pooled TSOS over clips: sum(over) / sum(active) from a list of (n_over, n_active); NaN with no active frame.
    Multiply by 100 for the percent that contract 9 gates (<= 3%)."""
    total_over = sum(c[0] for c in counts)
    total_active = sum(c[1] for c in counts)
    if total_active == 0:
        return float("nan")
    return total_over / total_active


# ---------------------------------------------------------------------------------------------------------------
# Optional packages: ESTOI (pystoi, MIT) and PESQ (pesq, internal R&D use only, reported not gated)
# ---------------------------------------------------------------------------------------------------------------
_MODULES = {}
_PACKAGE = {"estoi": "pystoi", "pesq": "pesq"}


def _module(name):
    """Import pystoi or pesq on first use; None if it is not installed (or fails to load)."""
    if name not in _MODULES:
        try:
            _MODULES[name] = importlib.import_module(_PACKAGE[name])
        except Exception:                                  # ImportError, or a broken compiled extension
            _MODULES[name] = None
    return _MODULES[name]


def available(name):
    """True if the package behind metric `name` ("estoi" or "pesq") can be imported."""
    return _module(name) is not None


def status():
    """{"estoi": "available" or "not available", "pesq": ...} for the report."""
    return {name: ("available" if available(name) else NOT_AVAILABLE) for name in _PACKAGE}


def estoi_gain(s, y, x):
    """ESTOI gain: stoi(s, y, extended) - stoi(s, x, extended) at 16 kHz. None when pystoi is not installed."""
    pystoi = _module("estoi")
    if pystoi is None:
        return None
    s, y, x = _f64(s), _f64(y), _f64(x)
    return float(pystoi.stoi(s, y, SR, extended=True) - pystoi.stoi(s, x, SR, extended=True))


def pesq_wb(s, y):
    """Wideband PESQ of output y against reference s. None when pesq is not installed or raises on this clip
    (the caller counts those clips as skipped; use available("pesq") to tell the two apart)."""
    pesq = _module("pesq")
    if pesq is None:
        return None
    try:
        return float(pesq.pesq(SR, _f64(s), _f64(y), "wb"))
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------------------------
# Envelope correlation (DiPCo real dinner-party check, 10-ml 12.5)
# ---------------------------------------------------------------------------------------------------------------
_ENV_SOS = None


def envelope(x):
    """RMS envelope: band-pass 100 to 4,000 Hz (Butterworth, scipy order 4, causal), then the RMS of 20 ms frames
    (320 samples) every 10 ms (160 samples) -> [frames]."""
    global _ENV_SOS
    if _ENV_SOS is None:
        _ENV_SOS = scipy.signal.butter(4, ENV_BAND, "bandpass", fs=SR, output="sos")
    band = scipy.signal.sosfilt(_ENV_SOS, _f64(x))
    if len(band) < ENV_FRAME:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(band, ENV_FRAME)[::ENV_HOP]
    return np.sqrt(np.mean(frames * frames, axis=1))


def pearson(a, b):
    """Pearson correlation of two equal-length arrays; 0 if either one is constant (no variation to correlate)."""
    a = _f64(a) - np.mean(a)
    b = _f64(b) - np.mean(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a * b) / denom)


def env_corr_with_lag(a, b, max_lag=ENV_MAX_LAG):
    """Best envelope correlation of a and b over lags of -100 to +100 ms -> (corr, lag_ms).

    Lag L (in 10 ms envelope hops) compares envelope a at hop k with envelope b at hop k + L, so a positive lag means
    b is late. Only the overlapping hops are used.
    """
    ea = envelope(a)
    eb = envelope(b)
    n = min(len(ea), len(eb))
    best_corr, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            pa, pb = ea[:n - lag], eb[lag:n]
        else:
            pa, pb = ea[-lag:n], eb[:n + lag]
        if len(pa) < 2:
            continue
        c = pearson(pa, pb)
        if c > best_corr:
            best_corr, best_lag = c, lag
    if best_corr == -np.inf:
        return 0.0, 0.0                                   # clips too short for any envelope comparison
    return best_corr, best_lag * 1000.0 * ENV_HOP / SR


def env_corr(a, b):
    """Best envelope correlation of a and b over lags of -100 to +100 ms (see env_corr_with_lag)."""
    return env_corr_with_lag(a, b)[0]
