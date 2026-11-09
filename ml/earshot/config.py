"""Constants, paths and YAML config loading for Earshot ML (spec 10-ml section 2.1).

CONFIG is the architecture object of contract 3.1, copied verbatim. The C engine compares every field of the
`config` object in a weights file with its own compiled constants, so this dict must never drift from the contract.
config_hash() is the SHA-256 the contract defines; a test pins it to CONFIG_HASH below.
"""
import hashlib
import json
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------------------------------------------
# The architecture object (contract 3.1). 29 keys. Do not edit without a DECISIONS row and a model_version bump.
# ---------------------------------------------------------------------------------------------------------------
CONFIG = {
    "sr": 16000,
    "n_fft": 512,
    "hop": 64,
    "synth_len": 128,
    "ana_rise": 448,
    "band_widths": [8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 33],
    "D": 64,
    "blocks": 4,
    "time_hidden": 128,
    "band_hidden": 64,
    "film_after": [0, 2],
    "att_after": 0,
    "att_heads": 4,
    "att_dim": 64,
    "tok_dim": 64,
    "max_target_tokens": 48,
    "max_own_tokens": 16,
    "null_tokens": 4,
    "g_dim": 128,
    "wespk_dim": 256,
    "df_taps": 3,
    "df_bands": 12,
    "mask_bound": 1.5,
    "feat_pow": 0.3,
    "feat_log_scale": 0.1,
    "ln_eps": 1e-05,
    "tok_hop": 512,
    "tok_pool": 3,
    "tok_hidden": 32,
}

# The value 10-ml 2.1 states for config_hash(). test_dsp.py checks config_hash() == CONFIG_HASH.
CONFIG_HASH = "089d72c06d244e1650d209453c91ddffde990b9589dd7a9c228f9e1ea3b93aec"


def config_hash():
    """Lowercase hex SHA-256 of the canonical JSON of CONFIG (sorted keys, no spaces), as in contract 3.1."""
    text = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Short names for the numbers used all over the code (all read from CONFIG so there is one source of truth).
# ---------------------------------------------------------------------------------------------------------------
SR = CONFIG["sr"]                    # 16,000 samples per second
N_FFT = CONFIG["n_fft"]              # 512-sample analysis frame (32 ms)
HOP = CONFIG["hop"]                  # 64 samples per hop (4 ms)
N_BINS = N_FFT // 2 + 1              # 257 frequency bins of 31.25 Hz
SYNTH_LEN = CONFIG["synth_len"]      # 128-sample synthesis window (8 ms)
ANA_RISE = CONFIG["ana_rise"]        # 448 zeros of left padding = the long rising part of the analysis window
D = CONFIG["D"]                      # 64 features per band
N_BLOCKS = CONFIG["blocks"]          # 4 dual-path blocks
TIME_HIDDEN = CONFIG["time_hidden"]  # 128 hidden units in the time LSTM
BAND_HIDDEN = CONFIG["band_hidden"]  # 64 hidden units per direction in the band BiLSTM
ATT_HEADS = CONFIG["att_heads"]      # 4 attention heads
ATT_DIM = CONFIG["att_dim"]          # 64 = 4 heads x 16
HEAD_DIM = ATT_DIM // ATT_HEADS      # 16 numbers per head
TOK_DIM = CONFIG["tok_dim"]          # 64 numbers per voice token
MAX_T_TOK = CONFIG["max_target_tokens"]  # 48 target token slots
MAX_O_TOK = CONFIG["max_own_tokens"]     # 16 own-voice token slots
NULL_TOK = CONFIG["null_tokens"]         # 4 learned null tokens, always present
N_TOKENS = MAX_T_TOK + MAX_O_TOK + NULL_TOK  # 68 token slots in total
G_DIM = CONFIG["g_dim"]              # 128-number token-encoder voiceprint g
WESPK_DIM = CONFIG["wespk_dim"]      # 256-number WeSpeaker voiceprint e
FILM_IN = 2 * WESPK_DIM + 2 * G_DIM + 2   # 770 = [e_t, g_t, e_o, g_o, mode]
FILM_OUT = 2 * 2 * 16 * D                 # 4,096 = (point, gamma/beta, band, d)
DF_TAPS = CONFIG["df_taps"]          # 3 deep-filter taps in the low bands
DF_BANDS = CONFIG["df_bands"]        # bands 0..11 use the deep filter
MASK_BOUND = CONFIG["mask_bound"]    # coefficients are 1.5 * tanh(...)
FEAT_POW = CONFIG["feat_pow"]        # |X|^0.3 compression
FEAT_LOG_SCALE = CONFIG["feat_log_scale"]  # 0.1 * ln(|X|^2 + 1e-8)
LN_EPS = CONFIG["ln_eps"]            # LayerNorm epsilon
TOK_HOP = CONFIG["tok_hop"]          # token encoder hop: 512 samples (32 ms)
TOK_POOL = CONFIG["tok_pool"]        # 3 speech frames are averaged into one token
TOK_HIDDEN = CONFIG["tok_hidden"]    # 32 hidden units per direction in the token encoder LSTMs

# Band layout (contract 2.2). WIDTHS[k] = bins in band k, OFFSETS[k] = first bin of band k,
# TAPS[k] = deep-filter taps of band k, COEF_LEN[k] = decoder outputs of band k (real and imaginary parts).
WIDTHS = [8] * 8 + [16] * 4 + [32] * 3 + [33]
K = len(WIDTHS)                      # 16 bands
OFFSETS = []
_start = 0
for _w in WIDTHS:
    OFFSETS.append(_start)
    _start += _w
TAPS = [DF_TAPS] * DF_BANDS + [1] * (K - DF_BANDS)
COEF_LEN = [2 * w * t for w, t in zip(WIDTHS, TAPS)]   # sums to 1,026
DF_BINS = sum(WIDTHS[:DF_BANDS])     # 128: bins 0..127 (0 to 3.97 kHz) carry the 2-hop X history

assert sum(WIDTHS) == N_BINS and WIDTHS == CONFIG["band_widths"]
assert sum(COEF_LEN) == 1026 and DF_BINS == 128

# ---------------------------------------------------------------------------------------------------------------
# Paths. ROOT is the repository root (this file is ROOT/ml/earshot/config.py).
# DATA is the same folder dataprep/common.py calls DATA; it is written here directly so importing the config does not
# import the frozen dataprep package.
# ---------------------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "ml" / "runs"
EXPORTS = ROOT / "ml" / "exports"
CACHE = ROOT / "ml" / "cache"
ASSETS = ROOT / "ml" / "assets"
CONFIGS = ROOT / "ml" / "configs"
# WeSpeaker self-test inputs (clip + PyTorch embedding), written by ml.tools.convert_wespeaker and read by
# ml.export to make wespk_selftest.est. It sits next to wespeaker_check.json, outside the release folders
# (random_v1/, v1/), so a release folder holds only its six contract files and v1 does not depend on random_v1.
WESPK_SELFTEST_NPZ = EXPORTS / "wespeaker_selftest_inputs.npz"


def _merge(base, over):
    """Overlay dict `over` on dict `base`: nested dicts merge key by key, every other value (lists too) replaces."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path):
    """Read a run config. If it has `base: other.yaml`, load that file first (same folder) and overlay this one."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("base", None)
    if base_name is None:
        return cfg
    base = load_yaml(path.parent / base_name)
    return _merge(base, cfg)
