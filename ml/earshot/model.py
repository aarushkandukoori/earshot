"""ESB-16-DF, the Earshot separation network, in its offline (whole-clip, batched) form (spec 10-ml section 3).

Contract 2.2 to 2.4 define every layer; this file follows them line by line. In short, for every 4 ms hop:
  1. features   - 257 complex bins -> 3 numbers per bin (compressed real, compressed imaginary, log power)
  2. enc        - each of the 16 bands gets its own Linear + LayerNorm -> z [16 bands][64]
  3. blocks     - 4 dual-path blocks; each runs an LSTM over time per band, then a BiLSTM across the 16 bands
  4. film, att  - after block 0: FiLM 0 and cross-attention to the voice tokens; after block 2: FiLM 1
  5. heads      - p_target and p_self (is the target talking, is the user talking)
  6. dec        - per band, complex filter coefficients (3 taps below 4 kHz, 1 tap above), bounded by 1.5 tanh
  7. deep filter + inverse STFT -> 64 output samples

Module names are chosen so that state_dict keys equal the export names of contract 3.2 (before the LSTM rename of
10-ml 7.2: PyTorch's weight_ih_l0 / weight_hh_l0 / bias_ih_l0 + bias_hh_l0 become w_ih / w_hh / bias at export).
The streaming (one hop at a time) form of the same network is ml/earshot/stream_ref.py.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (ATT_DIM, ATT_HEADS, BAND_HIDDEN, COEF_LEN, D, FILM_IN, FILM_OUT, G_DIM, HEAD_DIM, K, LN_EPS,
                     MASK_BOUND, MAX_T_TOK, N_BLOCKS, N_TOKENS, NULL_TOK, OFFSETS, TAPS, TIME_HIDDEN,
                     TOK_DIM, TOK_HIDDEN, WESPK_DIM, WIDTHS)
from .dsp import istft, stft, stft_tok
from .vad import es_vad, token_groups


# ---------------------------------------------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class CondInputs:
    """Everything that says who to keep, for a batch of B examples (all tensors on one device).

    Values that are not used (e_t when has_t is False, and so on) may be anything finite: build_cond replaces them
    with the learned null vectors. Token rows with val = False are ignored by attention.
    """
    e_t: torch.Tensor      # [B, 256] WeSpeaker voiceprint of the target enrollment
    tok_t: torch.Tensor    # [B, 48, 64] target voice tokens from encode_enrollment
    val_t: torch.Tensor    # [B, 48] bool, which target token rows are real
    g_t: torch.Tensor      # [B, 128] token-encoder voiceprint of the target
    e_o: torch.Tensor      # [B, 256] WeSpeaker voiceprint of the user's own voice
    tok_o: torch.Tensor    # [B, 16, 64] own-voice tokens
    val_o: torch.Tensor    # [B, 16] bool
    g_o: torch.Tensor      # [B, 128]
    has_t: torch.Tensor    # [B] bool, False = Everyone mode (no target person)
    has_o: torch.Tensor    # [B] bool, False = no own voice enrolled
    use_wes: torch.Tensor  # [B] bool, False = WeSpeaker path dropped (both e vectors become nulls)
    mode: torch.Tensor     # [B, 2] float, [1, 0] = Person, [0, 1] = Everyone


@dataclass
class Cond:
    """The conditioning set, computed once per person / mode change (contract 2.3)."""
    film: torch.Tensor     # [B, 2, 2, 16, 64] (FiLM point p, s = 0 gamma / 1 beta, band k, d)
    k: torch.Tensor        # [B, 4, 68, 16] attention keys per head
    v: torch.Tensor        # [B, 4, 68, 16] attention values per head
    valid: torch.Tensor    # [B, 68] bool; slot order: 48 target slots, 16 own slots, 4 null tokens


# ---------------------------------------------------------------------------------------------------------------
# Input features (contract 2.2 row 1)
# ---------------------------------------------------------------------------------------------------------------
def features(X):
    """Complex spectrum [..., 257] -> features [..., 257, 3].

    c = X * (|X| + 1e-8)^(-0.7) keeps the phase of X and compresses its magnitude to about |X|^0.3.
    The 3 features per bin are [Re c, Im c, 0.1 ln(|X|^2 + 1e-8)]. Flattening the last two axes of a band gives the
    contract index 3 * j + f (bin-major).
    """
    mag = X.abs()
    c = X * torch.pow(mag + 1e-8, -0.7)
    log_power = 0.1 * torch.log(mag * mag + 1e-8)
    return torch.stack([c.real, c.imag, log_power], dim=-1)


def masked_mean(tok, valid):
    """Mean of the valid rows: [B, M, C] and bool [B, M] -> [B, C]. With no valid rows it returns 0, never NaN."""
    weights = valid[..., None].to(tok.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp(min=1).to(tok.dtype)
    return (tok * weights).sum(dim=1) / count


# ---------------------------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------------------------
class BandEncoder(nn.Module):
    """Per band k: Linear(3 W_k -> 64) then LayerNorm(64), each band with its own weights (contract 2.2 row 2)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.ModuleList([nn.Linear(3 * w, D) for w in WIDTHS])
        self.ln = nn.ModuleList([nn.LayerNorm(D, eps=LN_EPS) for _ in WIDTHS])

    def forward(self, f):
        """f [..., 257, 3] (from features) -> z [..., 16, 64]."""
        outs = []
        for k in range(K):
            band = f[..., OFFSETS[k]:OFFSETS[k] + WIDTHS[k], :].flatten(-2)   # [..., 3 W_k], index 3j + f
            outs.append(self.ln[k](self.lin[k](band)))
        return torch.stack(outs, dim=-2)


class DualPath(nn.Module):
    """One dual-path block: a time path then a band path, each with pre-LayerNorm, Linear and a residual add.

    Time path: every band of every example is one sequence over the T frames (LSTM, carried state).
    Band path: every frame is one sequence over the 16 bands (BiLSTM, zero initial state every frame).
    The separation blocks use Ht = 128 (one direction), Hb = 64; the token encoder uses Ht = Hb = 32 and a
    bidirectional time LSTM (it sees the whole enrollment clip at once).
    """

    def __init__(self, Ht, Hb, time_bidir):
        super().__init__()
        self.tn = nn.LayerNorm(D, eps=LN_EPS)
        self.t = nn.LSTM(D, Ht, batch_first=True, bidirectional=time_bidir)
        self.tf = nn.Linear(Ht * (2 if time_bidir else 1), D)
        self.bn = nn.LayerNorm(D, eps=LN_EPS)
        self.b = nn.LSTM(D, Hb, batch_first=True, bidirectional=True)
        self.bf = nn.Linear(2 * Hb, D)

    def forward(self, z):
        """z [B, T, 16, 64] -> [B, T, 16, 64]."""
        B, T, n_bands, d = z.shape
        # Time path: reorder to [B * 16 sequences, T steps, 64].
        seq = self.tn(z).permute(0, 2, 1, 3).reshape(B * n_bands, T, d)
        y, _ = self.t(seq)
        z = z + self.tf(y).reshape(B, n_bands, T, d).permute(0, 2, 1, 3)
        # Band path: [B * T sequences, 16 steps, 64].
        seq = self.bn(z).reshape(B * T, n_bands, d)
        y, _ = self.b(seq)
        z = z + self.bf(y).reshape(B, T, n_bands, d)
        return z


class Attention(nn.Module):
    """Cross-attention from every frame to the voice tokens (contract 2.2 row 6 and 2.3 token list).

    q comes from the whole frame (all 16 bands flattened, index k * 64 + d). There are 4 heads of 16 numbers.
    kv turns each 64-number token into a key (first 64) and a value (last 64). type_emb marks a token as target (0),
    own voice (1) or null (2); null_tok are 4 learned tokens that are always present, so softmax always has something
    to attend to (for example in Everyone mode with no own voice).
    """

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(K * D, ATT_DIM)
        self.o = nn.Linear(ATT_DIM, K * D)
        self.kv = nn.Linear(TOK_DIM, 2 * ATT_DIM)
        self.type_emb = nn.Parameter(torch.zeros(3, TOK_DIM))
        self.null_tok = nn.Parameter(torch.zeros(NULL_TOK, TOK_DIM))

    def forward(self, z, k, v, valid):
        """z [B, T, 16, 64]; k, v [B, 4, n, 16]; valid [B, n] bool -> z + attention output, [B, T, 16, 64].

        Written out by hand (no scaled_dot_product_attention) so the C code can be read against it.
        """
        B, T = z.shape[0], z.shape[1]
        q = self.q(z.reshape(B, T, K * D)).reshape(B, T, ATT_HEADS, HEAD_DIM)     # head h = q[16h : 16h + 16]
        scores = torch.einsum("bthd,bhnd->bthn", q, k) * 0.25                    # 0.25 = 1 / sqrt(16)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))      # invalid tokens get weight 0
        p = torch.softmax(scores, dim=-1)
        a = torch.einsum("bthn,bhnd->bthd", p, v).reshape(B, T, ATT_DIM)          # a[16h + d]
        return z + self.o(a).reshape(B, T, K, D)


class Nulls(nn.Module):
    """Learned stand-ins for missing voiceprints (contract 2.3): used in Everyone mode, with no own voice, or with
    the WeSpeaker path dropped."""

    def __init__(self):
        super().__init__()
        self.e_t = nn.Parameter(torch.zeros(WESPK_DIM))
        self.g_t = nn.Parameter(torch.zeros(G_DIM))
        self.e_o = nn.Parameter(torch.zeros(WESPK_DIM))
        self.g_o = nn.Parameter(torch.zeros(G_DIM))


class BandDecoder(nn.Module):
    """Per band: LayerNorm, Linear(64 -> 128), tanh, Linear(128 -> 2 W_k T_k), then 1.5 tanh (contract 2.2 row 9).

    The output vector holds the W_k T_k real parts first, then the W_k T_k imaginary parts, each laid out tau-major
    (index tau * W_k + j).
    """

    def __init__(self, out_len):
        super().__init__()
        self.ln = nn.LayerNorm(D, eps=LN_EPS)
        self.fc1 = nn.Linear(D, 2 * D)
        self.fc2 = nn.Linear(2 * D, out_len)

    def forward(self, zk):
        """zk [..., 64] -> coefficients [..., 2 W_k T_k], each in (-1.5, 1.5)."""
        return MASK_BOUND * torch.tanh(self.fc2(torch.tanh(self.fc1(self.ln(zk)))))


class TokenEncoder(nn.Module):
    """Enrollment token encoder after the shared band encoder (contract 2.4 steps 3 to 6).

    It does not own a band encoder: ESB16DF.encode_enrollment runs the shared `enc` and passes z in.
    """

    def __init__(self):
        super().__init__()
        self.blk = DualPath(TOK_HIDDEN, TOK_HIDDEN, time_bidir=True)
        self.proj = nn.Linear(K * D, TOK_DIM)
        self.g = nn.Linear(TOK_DIM, G_DIM)

    def forward(self, z, idx, valid):
        """z [B, F, 16, 64]; idx [B, M, 3] int64 frame indices; valid [B, M] bool -> (tok [B, M, 64], g [B, 128]).

        Each token is the mean of its 3 frames (after the dual-path block), projected to 64 numbers. Invalid rows are
        set to 0. g is the L2-normalized projection of the mean of the valid tokens.
        """
        B, n_frames = z.shape[0], z.shape[1]
        M = idx.shape[1]
        h = self.blk(z).reshape(B, n_frames, K * D)                     # flatten index k * 64 + d
        flat_idx = idx.reshape(B, M * 3, 1).expand(B, M * 3, K * D)
        frames = torch.gather(h, 1, flat_idx).reshape(B, M, 3, K * D)   # the 3 frames of every group
        pooled = frames.mean(dim=2)                                     # [B, M, 1024]
        tok = self.proj(pooled)                                         # [B, M, 64]
        tok = torch.where(valid[..., None], tok, torch.zeros_like(tok))
        g = F.normalize(self.g(masked_mean(tok, valid)), dim=-1)
        return tok, g


# ---------------------------------------------------------------------------------------------------------------
# The whole network
# ---------------------------------------------------------------------------------------------------------------
class ESB16DF(nn.Module):
    """ESB-16-DF (model_version 1.0.0). See the module docstring for the data flow.

    The submodules are created in module-tree order (enc, blocks, att, film, null, dec, heads, tok), which is also
    the order in which PyTorch's default initialization draws random numbers.
    """

    def __init__(self):
        super().__init__()
        self.enc = BandEncoder()
        self.blocks = nn.ModuleList([DualPath(TIME_HIDDEN, BAND_HIDDEN, time_bidir=False) for _ in range(N_BLOCKS)])
        self.att = Attention()
        # FiLM generator: film.0 = Linear(770 -> 256), film.1 = SiLU, film.2 = Linear(256 -> 4096).
        self.film = nn.Sequential(nn.Linear(FILM_IN, 256), nn.SiLU(), nn.Linear(256, FILM_OUT))
        self.null = Nulls()
        self.dec = nn.ModuleList([BandDecoder(COEF_LEN[k]) for k in range(K)])
        self.heads = nn.Linear(D, 2)
        self.tok = TokenEncoder()

    # ----- per-hop pieces, shared with the streaming reference -----
    def encode(self, X):
        """Complex spectrum [..., 257] -> band features z [..., 16, 64] (features, then the band encoder)."""
        return self.enc(features(X))

    def decode(self, z):
        """z [..., 16, 64] -> list of 16 coefficient tensors, band k shaped [..., 2 W_k T_k] (after 1.5 tanh)."""
        return [self.dec[k](z[..., k, :]) for k in range(K)]

    @staticmethod
    def coef_complex(c, k):
        """Band k's coefficient vector [..., 2 W_k T_k] -> complex [..., T_k, W_k] (index tau * W_k + j)."""
        w, taps = WIDTHS[k], TAPS[k]
        n = w * taps
        re = c[..., :n].reshape(*c.shape[:-1], taps, w)
        im = c[..., n:].reshape(*c.shape[:-1], taps, w)
        return torch.complex(re, im)

    def deep_filter(self, coefs, X):
        """Offline deep filter (contract 2.2 row 10): Y(t, f) = sum over tau < T_k of C_tau(t, f) X(t - tau, f).

        coefs - list from decode, band k [B, T, 2 W_k T_k]; X [B, T, 257] complex. X before the first frame is 0.
        """
        B, T = X.shape[0], X.shape[1]
        bands = []
        for k in range(K):
            C = self.coef_complex(coefs[k], k)                     # [B, T, T_k, W_k]
            Xk = X[:, :, OFFSETS[k]:OFFSETS[k] + WIDTHS[k]]        # [B, T, W_k]
            Yk = C[:, :, 0] * Xk
            for tau in range(1, TAPS[k]):
                # X shifted tau frames later in time: shifted[t] = X[t - tau], zero for t < tau.
                # n_zero = min(tau, T) keeps shifted exactly T frames long even when the clip is shorter than
                # tau frames (T = 1 or 2); then shifted is all zeros.
                n_zero = min(tau, T)
                zeros = torch.zeros(B, n_zero, WIDTHS[k], dtype=Xk.dtype, device=Xk.device)
                shifted = torch.cat([zeros, Xk[:, :T - n_zero]], dim=1)
                Yk = Yk + C[:, :, tau] * shifted
            bands.append(Yk)
        return torch.cat(bands, dim=-1)

    # ----- enrollment and conditioning -----
    def encode_enrollment(self, wav, idx, valid):
        """Token encoder (contract 2.4). wav [B, N] float32, idx [B, M, 3] int64, valid [B, M] bool (both from
        vad.token_groups, M = 48 for a target or 16 for own voice) -> (tok [B, M, 64], g [B, 128])."""
        X = stft_tok(wav)                                          # [B, F, 257], hop 512
        if X.shape[1] == 0:
            raise ValueError("enrollment clip is shorter than 576 samples (token frame count F = 0)")
        z = self.encode(X)                                         # shared band encoder -> [B, F, 16, 64]
        return self.tok(z, idx, valid)

    def film_input(self, ci):
        """u = [e_t', g_t', e_o', g_o', mode] (770) after the null rules of contract 2.3 -> [B, 770].

        e_t' = e_t if (has_t and use_wes) else null.e_t;  g_t' = g_t if has_t else null.g_t
        e_o' = e_o if (has_o and use_wes) else null.e_o;  g_o' = g_o if has_o else null.g_o
        torch.where picks per example; both branches must be finite (torch.where passes NaN gradients through).
        """
        use_e_t = (ci.has_t & ci.use_wes)[:, None]
        use_e_o = (ci.has_o & ci.use_wes)[:, None]
        e_t = torch.where(use_e_t, ci.e_t, self.null.e_t)
        g_t = torch.where(ci.has_t[:, None], ci.g_t, self.null.g_t)
        e_o = torch.where(use_e_o, ci.e_o, self.null.e_o)
        g_o = torch.where(ci.has_o[:, None], ci.g_o, self.null.g_o)
        return torch.cat([e_t, g_t, e_o, g_o, ci.mode.to(e_t.dtype)], dim=-1)

    def token_list(self, ci):
        """Token list of contract 2.3 -> (tokens [B, 68, 64], valid [B, 68] bool).

        Order: 48 target slots + type_emb[0], 16 own slots + type_emb[1], 4 null tokens + type_emb[2].
        Target slots are valid only with has_t, own slots only with has_o; null tokens are always valid.
        """
        B = ci.tok_t.shape[0]
        nulls = self.att.null_tok[None].expand(B, NULL_TOK, TOK_DIM)
        tokens = torch.cat([ci.tok_t + self.att.type_emb[0],
                            ci.tok_o + self.att.type_emb[1],
                            nulls + self.att.type_emb[2]], dim=1)
        always = torch.ones(B, NULL_TOK, dtype=torch.bool, device=ci.val_t.device)
        valid = torch.cat([ci.val_t & ci.has_t[:, None], ci.val_o & ci.has_o[:, None], always], dim=1)
        return tokens, valid

    def build_cond(self, ci):
        """CondInputs -> Cond: FiLM gamma/beta for both points, and the attention keys and values."""
        B = ci.e_t.shape[0]
        film = self.film(self.film_input(ci)).reshape(B, 2, 2, K, D)   # index ((p*2 + s)*16 + k)*64 + d
        tokens, valid = self.token_list(ci)
        kv = self.att.kv(tokens)                                       # [B, 68, 128]
        k = kv[..., :ATT_DIM].reshape(B, N_TOKENS, ATT_HEADS, HEAD_DIM).transpose(1, 2)
        v = kv[..., ATT_DIM:].reshape(B, N_TOKENS, ATT_HEADS, HEAD_DIM).transpose(1, 2)
        return Cond(film=film, k=k, v=v, valid=valid)

    # ----- separation -----
    def forward(self, mix, cond):
        """mix [B, L] float32 (L a multiple of 64), cond a Cond -> (y [B, L], head_logits [B, T, 2]).

        y is aligned with mix (the streaming engine plays the same samples 64 samples later). head_logits are
        before the sigmoid: [p_target, p_self] logits per frame.
        """
        L = mix.shape[-1]
        X = stft(mix)                                              # [B, T, 257]
        z = self.encode(X)                                         # [B, T, 16, 64]
        for b in range(N_BLOCKS):
            z = self.blocks[b](z)
            if b == 0:
                # FiLM point 0, then cross-attention to the voice tokens.
                z = z * (1 + cond.film[:, 0, 0, None]) + cond.film[:, 0, 1, None]
                z = self.att(z, cond.k, cond.v, cond.valid)
            if b == 2:
                # FiLM point 1.
                z = z * (1 + cond.film[:, 1, 0, None]) + cond.film[:, 1, 1, None]
        head_logits = self.heads(z.mean(dim=2))                    # mean over the 16 bands
        coefs = self.decode(z)
        Y = self.deep_filter(coefs, X)
        y = istft(Y, L)
        return y, head_logits

    # ----- CPU helpers for single clips -----
    def enroll_clip(self, pcm, mask, max_tokens):
        """One clip through the token encoder with a given speech mask (uint8 [F]).

        Returns (tok [max_tokens, 64], valid bool [max_tokens], g [128], n_avail), tensors on the model's device.
        The caller decides what n_avail < 4 means (the engine fails with ES_ERR_ENROLL_SHORT).
        """
        device = next(self.parameters()).device
        idx, valid, n_avail = token_groups(mask, max_tokens)
        wav = torch.from_numpy(np.asarray(pcm, dtype=np.float32)).to(device)[None]
        idx_t = torch.from_numpy(idx).to(device)[None]
        valid_t = torch.from_numpy(valid).to(device)[None]
        tok, g = self.encode_enrollment(wav, idx_t, valid_t)
        return tok[0], valid_t[0], g[0], n_avail

    def embed_g(self, pcm):
        """The locked check's voiceprint (contract 2.4 step 8): es_vad mask, 48-token limit -> g [128].

        Raises ValueError if fewer than 4 tokens are available (ES_ERR_ENROLL_SHORT in the engine).
        """
        mask = es_vad(pcm)
        _, _, n_avail = token_groups(mask, MAX_T_TOK)
        if n_avail < 4:
            raise ValueError(f"ES_ERR_ENROLL_SHORT: only {n_avail} tokens (need at least 4)")
        with torch.no_grad():
            _, _, g, _ = self.enroll_clip(pcm, mask, MAX_T_TOK)
        return g


# ---------------------------------------------------------------------------------------------------------------
# Initialization (10-ml 3.4)
# ---------------------------------------------------------------------------------------------------------------
# Tensors that training starts at zero (and the random parity export fills with N(0, 0.1^2)), in this order.
ZERO_INIT_NAMES = ["att.o.weight", "att.o.bias", "att.type_emb", "att.null_tok",
                   "null.e_t", "null.g_t", "null.e_o", "null.g_o"]


def set_forget_bias(model):
    """For every LSTM direction: bias_ih[H:2H] = 1 and bias_hh[H:2H] = 0, so the exported sum is +1 (gate order
    i, f, g, o: the forget gate is the second block of H rows)."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.LSTM):
                H = module.hidden_size
                for name, p in module.named_parameters():
                    if name.startswith("bias_ih"):
                        p[H:2 * H] = 1.0
                    elif name.startswith("bias_hh"):
                        p[H:2 * H] = 0.0


def init_train(model):
    """Training init: PyTorch defaults (already done by the constructor), forget bias +1, and zeros for the output
    of the attention, the type embeddings, the null tokens and the null vectors. Changes the model in place and
    returns it."""
    set_forget_bias(model)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name in ZERO_INIT_NAMES:
            params[name].zero_()
    return model


def init_random_for_parity(seed):
    """A new model for parity tests and `export --random` (contract 3.2): torch.manual_seed(seed), PyTorch default
    init, forget bias +1, then the zero-init tensors filled with N(0, 0.1^2) in ZERO_INIT_NAMES order, so every
    path of the network is exercised."""
    torch.manual_seed(seed)
    model = ESB16DF()
    set_forget_bias(model)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name in ZERO_INIT_NAMES:
            params[name].normal_(0.0, 0.1)
    return model


def parameter_counts(model):
    """Parameter count per part of 10-ml 3.1 (PyTorch counts both LSTM bias vectors) plus the total."""
    def count(prefixes):
        total = 0
        for name, p in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                total += p.numel()
        return total

    return {
        "enc": count(["enc."]),
        "blocks": count(["blocks."]),
        "att_qo": count(["att.q.", "att.o."]),
        "dec": count(["dec."]),
        "heads": count(["heads."]),
        "film": count(["film."]),
        "kv": count(["att.kv."]),
        "type_emb": count(["att.type_emb"]),
        "null_tok": count(["att.null_tok"]),
        "nulls": count(["null."]),
        "tok": count(["tok."]),
        "total": sum(p.numel() for p in model.parameters()),
    }
