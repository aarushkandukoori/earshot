"""Candidate causal target-speaker extraction (TSE) models for the Earshot benchmarks.

Every model maps
    mix: (batch, samples) waveform at 16 kHz
    emb: (batch, 192) speaker embedding from the enrollment clip
to the estimated target waveform (batch, samples).

Candidates
    tasnet32   Conv-TasNet style TCN, encoder kernel 32 / stride 16 (1 ms frames)
    tasnet64   same, encoder kernel 64 / stride 32 (2 ms frames)
    stftgru    STFT 512 window / 64 hop, 2-layer GRU 384, complex ratio mask
    stftlstm   same with LSTM instead of GRU (MPS has a native LSTM kernel)
    bsrnn      small band-split dual-path RNN (16 bands, D=64, 3 layers), GRU
    bsrnnlstm  same with LSTM

All models are causal: output sample n never depends on input after sample n plus
the frame lookahead (window length). streaming.py runs the same weights chunk by chunk.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
EMB_DIM = 192


class FiLM(nn.Module):
    """Feature-wise linear modulation from the speaker embedding: x * (1 + gamma) + beta."""

    def __init__(self, emb_dim, channels):
        super().__init__()
        self.proj = nn.Linear(emb_dim, 2 * channels)

    def params(self, emb):
        gamma, beta = self.proj(emb).chunk(2, dim=-1)
        return 1.0 + gamma, beta


# ----------------------------------------------------------------------------------
# Conv-TasNet style TCN
# ----------------------------------------------------------------------------------


class CumLayerNorm(nn.Module):
    """Cumulative layer norm (causal norm from Conv-TasNet). x is (batch, channels, frames).
    Frame t is normalized with the mean/var of all channels of frames 0..t. Done in fp32."""

    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        _, C, T = x.shape
        xf = x.float()
        cum_sum = xf.sum(1).cumsum(-1)  # (batch, T)
        cum_pow = (xf * xf).sum(1).cumsum(-1)
        count = torch.arange(1, T + 1, device=x.device, dtype=torch.float32) * C
        mean = cum_sum / count
        var = (cum_pow / count - mean * mean).clamp_min(0.0)
        y = (xf - mean.unsqueeze(1)) * torch.rsqrt(var.unsqueeze(1) + self.eps)
        return (y * self.gamma + self.beta).to(x.dtype)


class TCNBlock(nn.Module):
    """1x1 conv -> PReLU -> cLN -> causal dilated depthwise conv -> PReLU -> cLN -> 1x1 (res + skip)."""

    def __init__(self, B, H, P, dilation):
        super().__init__()
        self.B, self.dilation, self.pad = B, dilation, (P - 1) * dilation
        self.inp = nn.Conv1d(B, H, 1)
        self.act1 = nn.PReLU()
        self.norm1 = CumLayerNorm(H)
        self.dconv = nn.Conv1d(H, H, P, dilation=dilation, groups=H)
        self.act2 = nn.PReLU()
        self.norm2 = CumLayerNorm(H)
        self.out = nn.Conv1d(H, 2 * B, 1)  # residual and skip 1x1 convs fused (skip channels = B)

    def forward(self, x):
        y = self.norm1(self.act1(self.inp(x)))
        y = F.pad(y, (self.pad, 0))  # left pad only = causal
        y = self.norm2(self.act2(self.dconv(y)))
        res, skip = self.out(y).split(self.B, dim=1)
        return x + res, skip


class ConvTasNetTSE(nn.Module):
    def __init__(self, L=32, N=256, B=128, H=256, P=3, X=8, R=3, emb_dim=EMB_DIM):
        super().__init__()
        self.L, self.S, self.N, self.B, self.H, self.P, self.X, self.R = L, L // 2, N, B, H, P, X, R
        self.encoder = nn.Conv1d(1, N, L, stride=self.S, bias=False)
        self.norm = CumLayerNorm(N)
        self.bottleneck = nn.Conv1d(N, B, 1)
        self.films = nn.ModuleList([FiLM(emb_dim, B) for _ in range(R)])
        self.repeats = nn.ModuleList(
            [nn.ModuleList([TCNBlock(B, H, P, 2**i) for i in range(X)]) for _ in range(R)]
        )
        self.mask_act = nn.PReLU()
        self.mask = nn.Conv1d(B, N, 1)
        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=self.S, bias=False)

    def forward(self, mix, emb):
        n = mix.shape[-1]
        pad = self.L - self.S  # frame t covers samples [t*S - pad, (t+1)*S): causal framing
        w = F.relu(self.encoder(F.pad(mix.unsqueeze(1), (pad, 0))))  # (batch, N, frames)
        y = self.bottleneck(self.norm(w))
        skip_sum = None
        for film, blocks in zip(self.films, self.repeats):
            g, b = film.params(emb)
            y = y * g.unsqueeze(-1).to(y.dtype) + b.unsqueeze(-1).to(y.dtype)  # FiLM at start of repeat
            for blk in blocks:
                y, s = blk(y)
                skip_sum = s if skip_sum is None else skip_sum + s
        m = torch.sigmoid(self.mask(self.mask_act(skip_sum)))
        out = self.decoder(w * m)  # overlap-add back to samples
        return out[:, 0, pad : pad + n]

    def macs_per_frame(self):
        N, B, H, P, L = self.N, self.B, self.H, self.P, self.L
        block = B * H + H * P + H * 2 * B
        return N * L + N * B + self.X * self.R * block + B * N + N * L

    def frames_per_second(self):
        return SR / self.S

    def algorithmic_latency_samples(self, chunk):
        return chunk + (self.L - self.S)


# ----------------------------------------------------------------------------------
# STFT front end (causal framing, sqrt-Hann analysis and synthesis)
# ----------------------------------------------------------------------------------


class STFT(nn.Module):
    """Causal STFT with rfft. Frames end at the current sample (left pad n_fft - hop).
    The inverse uses overlap-add with the same window and a constant normalization.
    Compute is identical to a low-latency asymmetric window pair (only the window values differ)."""

    def __init__(self, n_fft=512, hop=64):
        super().__init__()
        assert n_fft % hop == 0
        self.n_fft, self.hop, self.pad = n_fft, hop, n_fft - hop
        win = torch.hann_window(n_fft, periodic=True).sqrt()
        self.register_buffer("win", win, persistent=False)
        self.ola_norm = float((win * win).sum() / hop)  # sum of hann over overlaps (= 4 here)

    def forward(self, x):  # (batch, samples) -> complex (batch, frames, bins)
        x = F.pad(x.float(), (self.pad, 0))
        frames = x.unfold(-1, self.n_fft, self.hop) * self.win
        return torch.fft.rfft(frames, dim=-1)

    def inverse(self, X, length):  # complex (batch, frames, bins) -> (batch, length)
        frames = torch.fft.irfft(X, n=self.n_fft, dim=-1) * self.win  # (batch, T, n_fft)
        bsz, T, _ = frames.shape
        k = self.n_fft // self.hop
        blocks = frames.reshape(bsz, T, k, self.hop)
        out = frames.new_zeros(bsz, T + k - 1, self.hop)
        for i in range(k):  # overlap-add: sub-block i of frame t lands on block t + i
            out[:, i : i + T] += blocks[:, :, i]
        y = out.reshape(bsz, -1) / self.ola_norm
        return y[:, self.pad : self.pad + length]


def compress(X, power=0.3, eps=1e-8):
    """Power-law compressed complex spectrum |X|^p * e^(j angle X) as real features (..., 2*bins)."""
    re, im = X.real, X.imag
    scale = (re * re + im * im + eps) ** ((power - 1.0) / 2.0)
    return torch.cat([re * scale, im * scale], dim=-1)


def apply_crm(X, m):
    """Complex ratio mask. m is real (..., 2*bins) = [real part, imag part]."""
    mr, mi = m.float().chunk(2, dim=-1)
    re, im = X.real, X.imag
    return torch.complex(re * mr - im * mi, re * mi + im * mr)


# ----------------------------------------------------------------------------------
# STFT + GRU (or LSTM) with complex ratio mask
# ----------------------------------------------------------------------------------


class STFTRNN(nn.Module):
    def __init__(self, n_fft=512, hop=64, hidden=384, layers=2, emb_dim=EMB_DIM, rnn="gru"):
        super().__init__()
        self.stft = STFT(n_fft, hop)
        self.bins, self.hidden, self.layers, self.rnn_type = n_fft // 2 + 1, hidden, layers, rnn
        self.inp = nn.Linear(2 * self.bins, hidden)
        self.inp_norm = nn.LayerNorm(hidden)
        self.inp_act = nn.PReLU()
        self.film = FiLM(emb_dim, hidden)
        rnn_cls = nn.GRU if rnn == "gru" else nn.LSTM
        self.rnn = rnn_cls(hidden, hidden, num_layers=layers, batch_first=True)
        self.out1 = nn.Linear(hidden, hidden)
        self.out_act = nn.PReLU()
        self.out2 = nn.Linear(hidden, 2 * self.bins)

    def body(self, feat, g, b, state=None):
        h = self.inp_act(self.inp_norm(self.inp(feat)))
        h = h * g.to(h.dtype) + b.to(h.dtype)
        h, state = self.rnn(h.float(), state)  # nn.GRU/LSTM weights stay fp32 under MPS autocast
        return self.out2(self.out_act(self.out1(h))), state

    def forward(self, mix, emb):
        X = self.stft(mix)  # (batch, T, bins) complex fp32
        g, b = self.film.params(emb)
        m, _ = self.body(compress(X), g.unsqueeze(1), b.unsqueeze(1))
        return self.stft.inverse(apply_crm(X, m), mix.shape[-1])

    def macs_per_frame(self):
        Fb, Hd = self.bins, self.hidden
        gates = 3 if self.rnn_type == "gru" else 4
        rnn = self.layers * gates * (Hd * Hd + Hd * Hd)
        return 2 * Fb * Hd + rnn + Hd * Hd + Hd * 2 * Fb

    def frames_per_second(self):
        return SR / self.stft.hop

    def algorithmic_latency_samples(self, chunk):
        # with a low-latency asymmetric window pair (synthesis window = 2 hops) the output lags
        # one extra hop; with the plain symmetric window used here it lags n_fft - hop samples.
        return chunk + self.stft.hop


# ----------------------------------------------------------------------------------
# Small band-split dual-path RNN (BSRNN style)
# ----------------------------------------------------------------------------------


BAND_WIDTHS = [8] * 8 + [16] * 4 + [32] * 3 + [33]  # 16 bands, 257 bins (31.25 Hz per bin)


class BandSplitRNN(nn.Module):
    def __init__(self, n_fft=512, hop=64, D=64, layers=3, emb_dim=EMB_DIM, widths=BAND_WIDTHS, rnn="gru"):
        super().__init__()
        self.stft = STFT(n_fft, hop)
        self.bins, self.D, self.layers, self.widths = n_fft // 2 + 1, D, layers, list(widths)
        self.rnn_type = rnn
        rnn_cls = nn.GRU if rnn == "gru" else nn.LSTM
        assert sum(widths) == self.bins
        K, W = len(widths), max(widths)
        self.K, self.W = K, W
        # gather index: band k slot j -> bin (or the zero bin `bins` for padding)
        idx = torch.full((K, W), self.bins, dtype=torch.long)
        valid = torch.zeros(K, W, dtype=torch.bool)
        start = 0
        for k, w in enumerate(widths):
            idx[k, :w] = torch.arange(start, start + w)
            valid[k, :w] = True
            start += w
        self.register_buffer("gather_idx", idx.reshape(-1), persistent=False)
        self.register_buffer("valid_pos", valid.reshape(-1).nonzero().squeeze(1), persistent=False)
        # per-band linear maps (padded to the widest band so one einsum handles all bands)
        self.split_w = nn.Parameter(torch.randn(K, 2 * W, D) / math.sqrt(2 * W))
        self.split_b = nn.Parameter(torch.zeros(K, D))
        self.film = FiLM(emb_dim, D)
        self.t_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(layers)])
        self.t_rnn = nn.ModuleList([rnn_cls(D, D, batch_first=True) for _ in range(layers)])
        self.t_fc = nn.ModuleList([nn.Linear(D, D) for _ in range(layers)])
        self.b_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(layers)])
        self.b_rnn = nn.ModuleList(
            [rnn_cls(D, D // 2, batch_first=True, bidirectional=True) for _ in range(layers)]
        )
        self.b_fc = nn.ModuleList([nn.Linear(D, D) for _ in range(layers)])
        self.m_w1 = nn.Parameter(torch.randn(K, D, 2 * D) / math.sqrt(D))
        self.m_b1 = nn.Parameter(torch.zeros(K, 2 * D))
        self.m_w2 = nn.Parameter(torch.randn(K, 2 * D, 2 * W) / math.sqrt(2 * D) * 0.1)
        self.m_b2 = nn.Parameter(torch.zeros(K, 2 * W))

    def split(self, X):  # complex (batch, T, bins) -> (batch, T, K, D)
        feat = compress(X)  # (batch, T, 2*bins) = [re..., im...]
        bsz, T, _ = feat.shape
        re, im = feat.chunk(2, dim=-1)
        zero = feat.new_zeros(bsz, T, 1)
        re = torch.cat([re, zero], -1)[..., self.gather_idx].reshape(bsz, T, self.K, self.W)
        im = torch.cat([im, zero], -1)[..., self.gather_idx].reshape(bsz, T, self.K, self.W)
        z = torch.cat([re, im], dim=-1)  # (batch, T, K, 2W)
        return torch.einsum("btkw,kwd->btkd", z, self.split_w.to(z.dtype)) + self.split_b.to(z.dtype)

    def mask(self, z):  # (batch, T, K, D) -> real CRM (batch, T, 2*bins)
        h = torch.tanh(torch.einsum("btkd,kde->btke", z, self.m_w1.to(z.dtype)) + self.m_b1.to(z.dtype))
        m = torch.einsum("btke,kew->btkw", h, self.m_w2.to(h.dtype)) + self.m_b2.to(h.dtype)
        bsz, T = m.shape[:2]
        mr, mi = m.reshape(bsz, T, self.K, 2, self.W).unbind(3)
        mr = mr.reshape(bsz, T, -1)[..., self.valid_pos]
        mi = mi.reshape(bsz, T, -1)[..., self.valid_pos]
        return torch.cat([mr, mi], dim=-1)

    def time_layer(self, i, z, state=None):  # causal RNN along time, shared across bands
        bsz, T, K, D = z.shape
        x = z.permute(0, 2, 1, 3).reshape(bsz * K, T, D)
        y, state = self.t_rnn[i](self.t_norm[i](x).float(), state)
        y = self.t_fc[i](y).reshape(bsz, K, T, D).permute(0, 2, 1, 3)
        return z + y, state

    def band_layer(self, i, z):  # bidirectional RNN across the 16 bands of each frame
        bsz, T, K, D = z.shape
        x = z.reshape(bsz * T, K, D)
        y, _ = self.b_rnn[i](self.b_norm[i](x).float())
        return z + self.b_fc[i](y).reshape(bsz, T, K, D)

    def forward(self, mix, emb):
        X = self.stft(mix)
        z = self.split(X)
        g, b = self.film.params(emb)
        z = z * g[:, None, None, :].to(z.dtype) + b[:, None, None, :].to(z.dtype)
        for i in range(self.layers):
            z, _ = self.time_layer(i, z)
            z = self.band_layer(i, z)
        return self.stft.inverse(apply_crm(X, self.mask(z)), mix.shape[-1])

    def macs_per_frame(self):
        K, D, Fb = self.K, self.D, self.bins
        gates = 3 if self.rnn_type == "gru" else 4
        split = 2 * Fb * D
        t_gru = K * gates * (D * D + D * D)
        b_gru = K * 2 * gates * (D * (D // 2) + (D // 2) ** 2)
        layer = t_gru + K * D * D + b_gru + K * D * D
        mask = K * D * 2 * D + 2 * D * 2 * Fb
        return split + self.layers * layer + mask

    def frames_per_second(self):
        return SR / self.stft.hop

    def algorithmic_latency_samples(self, chunk):
        return chunk + self.stft.hop


# ----------------------------------------------------------------------------------


def build(name):
    if name == "tasnet32":
        return ConvTasNetTSE(L=32)
    if name == "tasnet64":
        return ConvTasNetTSE(L=64)
    if name == "stftgru":
        return STFTRNN(rnn="gru")
    if name == "stftlstm":
        return STFTRNN(rnn="lstm")
    if name == "bsrnn":
        return BandSplitRNN(rnn="gru")
    if name == "bsrnnlstm":
        return BandSplitRNN(rnn="lstm")
    raise ValueError(name)


ALL = ["tasnet32", "tasnet64", "stftgru", "stftlstm", "bsrnn", "bsrnnlstm"]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def neg_si_sdr(est, ref, eps=1e-8):
    est = est.float() - est.float().mean(-1, keepdim=True)
    ref = ref.float() - ref.float().mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps) * ref
    noise = est - proj
    ratio = proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps)
    return -(10 * torch.log10(ratio + eps)).mean()


def hook_macs(model, seconds=4.0):
    """Count Conv1d / ConvTranspose1d / Linear MACs with forward hooks (RNNs and einsums excluded).
    Used only to cross-check the analytic formulas."""
    total = [0]

    def conv_hook(mod, inp, out):
        total[0] += out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0]

    def convt_hook(mod, inp, out):
        total[0] += inp[0].numel() * (mod.out_channels // mod.groups) * mod.kernel_size[0]

    def lin_hook(mod, inp, out):
        total[0] += out.numel() * mod.in_features

    hs = []
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose1d):
            hs.append(m.register_forward_hook(convt_hook))
        elif isinstance(m, nn.Conv1d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear) and not any(m is f.proj for f in model.modules() if isinstance(f, FiLM)):
            hs.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(torch.zeros(1, int(SR * seconds)), torch.zeros(1, EMB_DIM))
    for h in hs:
        h.remove()
    return total[0] / seconds
