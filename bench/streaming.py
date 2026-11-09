"""Streaming (chunk-by-chunk) versions of the candidates, with explicit state.

forward(chunk, *state) -> (out_chunk, *new_state). The same weights as the offline model are used,
and check_equivalence() proves the streamed output equals the offline output (shifted by the
model's built-in lookahead), so the streaming code is causal and correct.

The FiLM scale/shift is computed once from the enrollment embedding and kept as a constant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import models as M


def frames_of(x, size, step, n):
    """n overlapping frames of a 1-D tensor (like x.unfold(0, size, step), which Core ML can't convert)."""
    return torch.stack([x[j * step : j * step + size] for j in range(n)])


def cln_step(norm, x, st, ar):
    """Streaming cumulative layer norm. x (1, C, T), st = [running sum, running sum sq, frames]."""
    C = x.shape[1]
    cs = st[0] + x.sum(1).cumsum(-1)  # (1, T)
    cp = st[1] + (x * x).sum(1).cumsum(-1)
    cnt = (st[2] + ar) * C
    mean = cs / cnt
    var = (cp / cnt - mean * mean).clamp_min(0.0)
    y = (x - mean.unsqueeze(1)) * torch.rsqrt(var.unsqueeze(1) + norm.eps) * norm.gamma + norm.beta
    new = torch.stack([cs[0, -1], cp[0, -1], st[2] + ar[-1]])
    return y, new


class TasNetStream(nn.Module):
    def __init__(self, m: M.ConvTasNetTSE, emb, chunk):
        super().__init__()
        assert chunk % m.S == 0
        self.m, self.chunk, self.T = m, chunk, chunk // m.S
        with torch.no_grad():
            gb = [f.params(emb) for f in m.films]
        self.register_buffer("g", torch.stack([g for g, _ in gb]).reshape(m.R, 1, m.B, 1))
        self.register_buffer("b", torch.stack([b for _, b in gb]).reshape(m.R, 1, m.B, 1))
        self.register_buffer("ar", torch.arange(1, self.T + 1, dtype=torch.float32))
        self.blocks = [blk for blocks in m.repeats for blk in blocks]
        self.n_norms = 1 + 2 * len(self.blocks)
        self.lookahead = m.L - m.S  # output is delayed by this many samples

    def init_state(self):
        m = self.m
        st = [torch.zeros(1, 1, m.L - m.S), torch.zeros(self.n_norms, 3)]
        st += [torch.zeros(1, m.H, blk.pad) for blk in self.blocks]
        st += [torch.zeros(1, 1, m.L - m.S)]
        return st

    def forward(self, x, *st):
        m = self.m
        enc_tail, cln, dec_tail = st[0], st[1], st[-1]
        caches = st[2:-1]
        xin = torch.cat([enc_tail, x.view(1, 1, -1)], dim=-1)
        w = F.relu(F.conv1d(xin, m.encoder.weight, stride=m.S))  # (1, N, T)
        y, s = cln_step(m.norm, w, cln[0], self.ar)
        new_cln = [s]
        y = m.bottleneck(y)
        skip = None
        new_caches = []
        k = 0
        for r, blocks in enumerate(m.repeats):
            y = y * self.g[r] + self.b[r]
            for blk in blocks:
                z, s1 = cln_step(blk.norm1, blk.act1(blk.inp(y)), cln[1 + 2 * k], self.ar)
                zc = torch.cat([caches[k], z], dim=-1)
                new_caches.append(zc[:, :, zc.shape[-1] - blk.pad :])
                z = blk.act2(F.conv1d(zc, blk.dconv.weight, blk.dconv.bias, dilation=blk.dilation, groups=m.H))
                z, s2 = cln_step(blk.norm2, z, cln[2 + 2 * k], self.ar)
                res, sk = blk.out(z).split(m.B, dim=1)
                y = y + res
                skip = sk if skip is None else skip + sk
                new_cln += [s1, s2]
                k += 1
        mask = torch.sigmoid(m.mask(m.mask_act(skip)))
        out = F.conv_transpose1d(w * mask, m.decoder.weight, stride=m.S)  # (1, 1, T*S + L - S)
        ov = m.L - m.S
        n = self.T * m.S
        head = out[:, :, :ov] + dec_tail
        y_out = torch.cat([head, out[:, :, ov:n]], dim=-1).view(-1)
        return (y_out, xin[:, :, xin.shape[-1] - ov :], torch.stack(new_cln), *new_caches, out[:, :, n:])


def cln_rows(norm, x, st, ar):
    """Streaming cumulative layer norm on time-major rows. x (T, C). Sizes are Python ints so the
    traced graph has no shape ops (Core ML conversion needs that)."""
    C = norm.n_ch
    cs = st[0] + x.sum(1).cumsum(0)  # (T,)
    cp = st[1] + (x * x).sum(1).cumsum(0)
    cnt = (st[2] + ar) * C
    mean = cs / cnt
    var = (cp / cnt - mean * mean).clamp_min(0.0)
    y = (x - mean.unsqueeze(1)) * torch.rsqrt(var.unsqueeze(1) + norm.eps) * norm.g_row + norm.b_row
    return y, torch.stack([cs[-1], cp[-1], st[2] + ar[-1]])


class TasNetStreamTM(nn.Module):
    """Same math as TasNetStream, written time-major (frames x channels) with F.linear and an explicit
    3-tap depthwise conv. PyTorch's CPU depthwise conv1d loops over the 256 channels one by one
    (aten::slow_conv_dilated2d), which made TasNetStream about 10x slower than the math needs."""

    def __init__(self, m: M.ConvTasNetTSE, emb, chunk):
        super().__init__()
        assert chunk % m.S == 0
        self.m, self.chunk, self.T = m, chunk, chunk // m.S
        with torch.no_grad():
            gb = [f.params(emb) for f in m.films]
        self.register_buffer("g", torch.stack([g[0] for g, _ in gb]))  # (R, B)
        self.register_buffer("b", torch.stack([b[0] for _, b in gb]))
        self.register_buffer("ar", torch.arange(1, self.T + 1, dtype=torch.float32))
        self.blocks = [blk for blocks in m.repeats for blk in blocks]
        self.n_norms = 1 + 2 * len(self.blocks)
        self.lookahead = m.L - m.S
        for norm in [m.norm] + [n for blk in self.blocks for n in (blk.norm1, blk.norm2)]:
            norm.g_row = norm.gamma.detach().view(1, -1)
            norm.b_row = norm.beta.detach().view(1, -1)
            norm.n_ch = int(norm.gamma.numel())
        # (out, in, 1) conv weights -> (out, in) linear weights; depthwise taps -> (P, H)
        self.enc_w = m.encoder.weight.detach().view(m.N, m.L)
        self.dec_w = m.decoder.weight.detach().view(m.N, m.L)
        self.bott = (m.bottleneck.weight.detach().view(m.B, m.N), m.bottleneck.bias.detach())
        self.mask_lin = (m.mask.weight.detach().view(m.N, m.B), m.mask.bias.detach())
        self.lin = []
        for blk in self.blocks:
            self.lin.append(
                (
                    blk.inp.weight.detach().view(m.H, m.B),
                    blk.inp.bias.detach(),
                    blk.dconv.weight.detach().view(m.H, m.P).t().contiguous(),  # (P, H)
                    blk.dconv.bias.detach(),
                    blk.out.weight.detach().view(2 * m.B, m.H),
                    blk.out.bias.detach(),
                )
            )

    def init_state(self):
        m = self.m
        st = [torch.zeros(m.L - m.S), torch.zeros(self.n_norms, 3)]
        st += [torch.zeros(blk.pad, m.H) for blk in self.blocks]
        st += [torch.zeros(m.L - m.S)]
        return st

    def forward(self, x, *st):
        m, T = self.m, self.T
        enc_tail, cln, dec_tail = st[0], st[1], st[-1]
        caches = st[2:-1]
        xin = torch.cat([enc_tail, x])
        w = F.relu(F.linear(frames_of(xin, m.L, m.S, T), self.enc_w))  # (T, N)
        y, s = cln_rows(m.norm, w, cln[0], self.ar)
        new_cln = [s]
        y = F.linear(y, *self.bott)  # (T, B)
        skip = None
        new_caches = []
        for k, blk in enumerate(self.blocks):
            if k % m.X == 0:
                r = k // m.X
                y = y * self.g[r] + self.b[r]
            w_in, b_in, taps, b_dw, w_out, b_out = self.lin[k]
            z, s1 = cln_rows(blk.norm1, blk.act1(F.linear(y, w_in, b_in)), cln[1 + 2 * k], self.ar)
            zc = torch.cat([caches[k], z])  # (2d + T, H)
            d = blk.dilation
            new_caches.append(zc[T:])  # last blk.pad rows
            u = zc[0:T] * taps[0] + zc[d : d + T] * taps[1] + zc[2 * d : 2 * d + T] * taps[2] + b_dw
            u, s2 = cln_rows(blk.norm2, blk.act2(u), cln[2 + 2 * k], self.ar)
            o = F.linear(u, w_out, b_out)
            y = y + o[:, : m.B]
            skip = o[:, m.B :] if skip is None else skip + o[:, m.B :]
            new_cln += [s1, s2]
        mask = torch.sigmoid(F.linear(m.mask_act(skip), *self.mask_lin))
        frames = (w * mask) @ self.dec_w  # (T, L) per-frame contributions, L = 2 * S
        S = m.S
        first, second = frames[:, :S], frames[:, S:]
        # 50% overlap-add: output block j = first half of frame j + second half of frame j - 1
        prev = torch.cat([dec_tail.view(1, S), second[: T - 1]], dim=0)
        y_out = (first + prev).reshape(-1)
        return (y_out, xin[T * S :], torch.stack(new_cln), *new_caches, second[T - 1])


def ola_step(stft, frames, tail, nh):
    """Overlap-add nh new frames onto the carried tail. Returns (finished samples, new tail)."""
    k = stft.n_fft // stft.hop
    blocks = frames.reshape(nh, k, stft.hop)
    buf = torch.cat([tail.view(k - 1, stft.hop), frames.new_zeros(nh, stft.hop)], dim=0)  # (k-1+nh, hop)
    for j in range(nh):  # frame j covers blocks j .. j+k-1 (written without in-place ops so it exports)
        buf = buf + F.pad(blocks[j], (0, 0, j, nh - 1 - j))
    done = buf[:nh].reshape(-1) / stft.ola_norm
    return done, buf[nh:].reshape(-1)


class STFTRNNStream(nn.Module):
    def __init__(self, m: M.STFTRNN, emb, chunk):
        super().__init__()
        self.m, self.chunk = m, chunk
        self.nh = chunk // m.stft.hop
        assert self.nh * m.stft.hop == chunk
        with torch.no_grad():
            g, b = m.film.params(emb)
        self.register_buffer("g", g.reshape(1, 1, -1))
        self.register_buffer("b", b.reshape(1, 1, -1))
        self.lookahead = m.stft.n_fft - m.stft.hop  # symmetric window; see models.STFT docstring

    def init_state(self):
        s = self.m.stft
        st = [torch.zeros(s.n_fft - s.hop), torch.zeros(s.n_fft - s.hop)]
        h = torch.zeros(self.m.layers, 1, self.m.hidden)
        st += [h] if self.m.rnn_type == "gru" else [h, h.clone()]
        return st

    def forward(self, x, *st):
        m, s = self.m, self.m.stft
        xin = torch.cat([st[0], x])
        X = torch.fft.rfft(frames_of(xin, s.n_fft, s.hop, self.nh) * s.win)  # (nh, bins)
        state = st[2] if m.rnn_type == "gru" else (st[2], st[3])
        mm, state = m.body(M.compress(X).unsqueeze(0), self.g, self.b, state)
        frames = torch.fft.irfft(M.apply_crm(X, mm[0]), n=s.n_fft) * s.win
        y, tail = ola_step(s, frames, st[1], self.nh)
        new_state = [state] if m.rnn_type == "gru" else list(state)
        return (y, xin[self.nh * s.hop :], tail, *new_state)


class BandSplitStream(nn.Module):
    def __init__(self, m: M.BandSplitRNN, emb, chunk):
        super().__init__()
        self.m, self.chunk = m, chunk
        self.nh = chunk // m.stft.hop
        assert self.nh * m.stft.hop == chunk
        with torch.no_grad():
            g, b = m.film.params(emb)
        self.register_buffer("g", g.reshape(1, 1, 1, -1))
        self.register_buffer("b", b.reshape(1, 1, 1, -1))
        self.lookahead = m.stft.n_fft - m.stft.hop

    def init_state(self):
        s, m = self.m.stft, self.m
        st = [torch.zeros(s.n_fft - s.hop), torch.zeros(s.n_fft - s.hop)]
        for _ in range(m.layers):
            h = torch.zeros(1, m.K, m.D)
            st += [h] if m.rnn_type == "gru" else [h, h.clone()]
        return st

    def forward(self, x, *st):
        m, s = self.m, self.m.stft
        xin = torch.cat([st[0], x])
        X = torch.fft.rfft(frames_of(xin, s.n_fft, s.hop, self.nh) * s.win).unsqueeze(0)  # (1, nh, bins)
        z = m.split(X) * self.g + self.b
        per = 1 if m.rnn_type == "gru" else 2
        new_state = []
        for i in range(m.layers):
            hs = st[2 + per * i : 2 + per * (i + 1)]
            z, state = m.time_layer(i, z, hs[0] if per == 1 else (hs[0], hs[1]))
            new_state += [state] if per == 1 else list(state)
            z = m.band_layer(i, z)
        frames = torch.fft.irfft(M.apply_crm(X[0], m.mask(z)[0]), n=s.n_fft) * s.win
        y, tail = ola_step(s, frames, st[1], self.nh)
        return (y, xin[self.nh * s.hop :], tail, *new_state)


def make_stream(model, emb, chunk, tasnet_impl="tm"):
    if isinstance(model, M.ConvTasNetTSE):
        return TasNetStreamTM(model, emb, chunk) if tasnet_impl == "tm" else TasNetStream(model, emb, chunk)
    if isinstance(model, M.STFTRNN):
        return STFTRNNStream(model, emb, chunk)
    if isinstance(model, M.BandSplitRNN):
        return BandSplitStream(model, emb, chunk)
    raise TypeError(type(model))


@torch.no_grad()
def check_equivalence(name, chunk, seconds=2.0, seed=0, tasnet_impl="tm"):
    """Max abs difference between streamed and offline output (after the lookahead shift)."""
    torch.manual_seed(seed)
    model = M.build(name).eval()
    emb = torch.randn(1, M.EMB_DIM)
    n = int(M.SR * seconds)
    x = torch.randn(1, n) * 0.1
    ref = model(x, emb)[0]
    stream = make_stream(model, emb, chunk, tasnet_impl).eval()
    st = stream.init_state()
    outs = []
    for i in range(n // chunk):
        y, *st = stream(x[0, i * chunk : (i + 1) * chunk], *st)
        outs.append(y)
    z = torch.cat(outs)
    d = stream.lookahead
    a, b = z[d:], ref[: z.shape[0] - d]
    return float((a - b).abs().max()), float(b.abs().max())
