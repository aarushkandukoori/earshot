"""Training-step timing for the merged Earshot model (ESB-16 + deep filter + WeSpeaker in the loop), MPS, torch 2.7.1.
Random data, random weights, real shapes. Prints one JSON line per config."""
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

dev = torch.device("mps")
SR, NFFT, HOP = 16000, 512, 64
WIDTHS = [8] * 8 + [16] * 4 + [32] * 3 + [33]
K = len(WIDTHS)


def hann(L):
    k = torch.arange(L, dtype=torch.float64)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * k / L)


def windows():
    wa = torch.zeros(512, dtype=torch.float64)
    wa[:448] = hann(896)[:448].sqrt()
    wa[448:] = hann(128)[64:].sqrt()
    ws = torch.zeros(512, dtype=torch.float64)
    ws[384:] = hann(128) / wa[384:]
    return wa.float(), ws.float()


WA, WS = [w.to(dev) for w in windows()]


def stft(x):  # (b, L) -> (b, T, 257) complex, frame t ends at sample (t+1)*64
    fr = F.pad(x, (448, 0)).unfold(-1, 512, 64)
    return torch.fft.rfft(fr * WA, dim=-1)


def istft(Y, L):  # (b, T, 257) -> (b, L)
    fr = torch.fft.irfft(Y, n=512, dim=-1) * WS
    seg = fr[..., 384:].transpose(1, 2)  # (b, 128, T)
    T = seg.shape[-1]
    out = F.fold(seg, output_size=(1, (T + 1) * 64), kernel_size=(1, 128), stride=(1, 64))
    return out.reshape(out.shape[0], -1)[:, 64:64 + L]


class BandEnc(nn.Module):
    def __init__(s, D=64, nf=3):
        super().__init__()
        s.nf = nf
        s.lin = nn.ModuleList([nn.Linear(nf * w, D) for w in WIDTHS])
        s.ln = nn.ModuleList([nn.LayerNorm(D) for _ in WIDTHS])

    def forward(s, X):
        mag = X.abs()
        c = torch.polar(mag.pow(0.3), X.angle()) if False else X * (mag + 1e-8).pow(-0.7)
        feats = [c.real, c.imag] + ([0.1 * torch.log(mag.pow(2) + 1e-8)] if s.nf == 3 else [])
        f = torch.stack(feats, -1)  # (b, T, 257, nf)
        outs, i = [], 0
        for k, w in enumerate(WIDTHS):
            outs.append(s.ln[k](s.lin[k](f[:, :, i:i + w].flatten(2))))
            i += w
        return torch.stack(outs, 2)  # (b, T, K, D)


class DualPath(nn.Module):
    def __init__(s, D=64, Ht=128, Hb=64, bidir_time=False):
        super().__init__()
        s.tn, s.bn = nn.LayerNorm(D), nn.LayerNorm(D)
        s.t = nn.LSTM(D, Ht, batch_first=True, bidirectional=bidir_time)
        s.tf = nn.Linear(Ht * (2 if bidir_time else 1), D)
        s.b = nn.LSTM(D, Hb, batch_first=True, bidirectional=True)
        s.bf = nn.Linear(2 * Hb, D)

    def forward(s, z):
        b, T, Kk, D = z.shape
        y, _ = s.t(s.tn(z).permute(0, 2, 1, 3).reshape(b * Kk, T, D))
        z = z + s.tf(y).reshape(b, Kk, T, -1).permute(0, 2, 1, 3)
        y, _ = s.b(s.bn(z).reshape(b * T, Kk, D))
        return z + s.bf(y).reshape(b, T, Kk, D)


class BandDec(nn.Module):
    def __init__(s, D=64, taps_low=3, df=True):
        super().__init__()
        s.taps = [taps_low if (df and k < 12) else 1 for k in range(K)]
        s.net = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.Tanh(),
                                             nn.Linear(2 * D, 2 * w * t)) for w, t in zip(WIDTHS, s.taps)])

    def forward(s, z, X):
        outs, i = [], 0
        b, T, _ = X.shape
        for k, (w, t) in enumerate(zip(WIDTHS, s.taps)):
            c = 1.5 * torch.tanh(s.net[k](z[:, :, k]))  # (b, T, 2*w*t)
            c = torch.complex(c[..., : w * t], c[..., w * t:]).reshape(b, T, t, w)
            Xb = X[:, :, i:i + w]
            y = c[:, :, 0] * Xb
            for tau in range(1, t):
                y = y + c[:, :, tau] * F.pad(Xb, (0, 0, tau, 0))[:, :T]
            outs.append(y)
            i += w
        return torch.cat(outs, -1)


class TokenEnc(nn.Module):
    def __init__(s, enc, D=64, ntok=48, dtok=64, g_dim=128):
        super().__init__()
        s.enc, s.ntok = enc, ntok
        s.blk = DualPath(D, 32, 32, bidir_time=True)
        s.tok = nn.Linear(K * D, dtok)
        s.g = nn.Linear(dtok, g_dim)

    def forward(s, wav, ntok):
        fr = F.pad(wav, (448, 0)).unfold(-1, 512, 512)
        X = torch.fft.rfft(fr * WA, dim=-1)
        z = s.blk(s.enc(X))  # (b, T, K, D)
        b, T, Kk, D = z.shape
        t = F.avg_pool1d(z.reshape(b, T, Kk * D).transpose(1, 2), 3, 3).transpose(1, 2)[:, :ntok]
        tok = s.tok(t)
        g = F.normalize(s.g(tok.mean(1)), dim=-1)
        return tok, g


class ESB(nn.Module):
    def __init__(s, nf=3, df=True, film2=True, D=64, B=4, Ht=128, Hb=64):
        super().__init__()
        s.enc = BandEnc(D, nf)
        s.blocks = nn.ModuleList([DualPath(D, Ht, Hb) for _ in range(B)])
        s.tokenc = TokenEnc(s.enc, D)
        s.film2 = film2
        npts = 2 if film2 else 1
        s.film = nn.Sequential(nn.Linear(256 + 128 + 256 + 128 + 2, 256), nn.SiLU(), nn.Linear(256, npts * 2 * K * D))
        s.kv = nn.Linear(64, 128)
        s.type_emb = nn.Parameter(torch.zeros(3, 64))
        s.null_tok = nn.Parameter(torch.zeros(4, 64))
        s.q, s.o = nn.Linear(K * D, 64), nn.Linear(64, K * D)
        nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)
        s.dec = BandDec(D, 3, df)
        s.heads = nn.Linear(D, 2)

    def forward(s, mix, enr_t, enr_o, wes_t, wes_o):
        X = stft(mix)
        tok_t, g_t = s.tokenc(enr_t, 48)
        tok_o, g_o = s.tokenc(enr_o, 16)
        b = mix.shape[0]
        toks = torch.cat([tok_t + s.type_emb[0], tok_o + s.type_emb[1], s.null_tok.expand(b, -1, -1) + s.type_emb[2]], 1)
        k, v = s.kv(toks).chunk(2, -1)
        k = k.reshape(b, -1, 4, 16).transpose(1, 2); v = v.reshape(b, -1, 4, 16).transpose(1, 2)
        cond = torch.cat([wes_t, g_t, wes_o, g_o, torch.zeros(b, 2, device=mix.device)], -1)
        fl = s.film(cond).reshape(b, -1, 2, K, 64)
        z = s.enc(X)
        T = z.shape[1]
        for i, blk in enumerate(s.blocks):
            z = blk(z)
            if i == 0 or (i == 2 and s.film2):
                p = 0 if i == 0 else 1
                z = z * (1 + fl[:, p, 0, None]) + fl[:, p, 1, None]
            if i == 0:
                q = s.q(z.reshape(b, T, -1)).reshape(b, T, 4, 16).transpose(1, 2)
                a = F.scaled_dot_product_attention(q, k, v)
                z = z + s.o(a.transpose(1, 2).reshape(b, T, 64)).reshape(b, T, K, 64)
        Y = s.dec(z, X)
        return istft(Y, mix.shape[-1]), torch.sigmoid(s.heads(z.mean(2))), g_t


# ---- WeSpeaker ResNet34 (random weights, real shapes) with in-graph Kaldi fbank
class KaldiFbank(nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer("kern", torch.randn(514, 1, 400) * 0.01)
        s.register_buffer("mel", torch.rand(80, 257))

    def forward(s, wav):
        spec = F.conv1d((wav * 32768.0).unsqueeze(1), s.kern, stride=160)
        re, im = spec[:, :257], spec[:, 257:]
        fb = torch.log(torch.clamp(torch.matmul(s.mel, re * re + im * im), min=1.19e-7))
        return (fb - fb.mean(2, keepdim=True)).transpose(1, 2)


class BB(nn.Module):
    def __init__(s, cin, c, st=1):
        super().__init__()
        s.c1 = nn.Conv2d(cin, c, 3, st, 1, bias=False); s.b1 = nn.BatchNorm2d(c)
        s.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(c)
        s.sc = nn.Sequential() if (st == 1 and cin == c) else nn.Sequential(nn.Conv2d(cin, c, 1, st, bias=False), nn.BatchNorm2d(c))

    def forward(s, x):
        return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sc(x))


class ResNet34(nn.Module):
    def __init__(s, m=32):
        super().__init__()
        s.fb = KaldiFbank()
        s.c1 = nn.Conv2d(1, m, 3, 1, 1, bias=False); s.b1 = nn.BatchNorm2d(m)
        cin, L = m, []
        for c, n, st in [(m, 3, 1), (2 * m, 4, 2), (4 * m, 6, 2), (8 * m, 3, 2)]:
            for j in range(n):
                L.append(BB(cin, c, st if j == 0 else 1)); cin = c
        s.layers = nn.Sequential(*L)
        s.seg = nn.Linear(10 * m * 8 * 2, 256)

    def forward(s, wav):
        x = s.fb(wav).permute(0, 2, 1).unsqueeze(1)
        x = s.layers(F.relu(s.b1(s.c1(x))))
        mean = x.mean(-1).flatten(1); std = torch.sqrt(torch.var(x, dim=-1) + 1e-7).flatten(1)
        return F.normalize(s.seg(torch.cat([mean, std], 1)), dim=-1)


def tsnr(est, ref, tau=1e-3):
    p = ref.pow(2).sum(-1)
    return (-10 * torch.log10(p / ((ref - est).pow(2).sum(-1) + tau * p + 1e-8) + 1e-8)).mean()


def mrstft(est, ref):
    l = 0
    for n in (256, 512, 1024):
        w = torch.hann_window(n, device=est.device)
        E = torch.stft(est, n, n // 4, window=w, return_complex=True).abs()
        R = torch.stft(ref, n, n // 4, window=w, return_complex=True).abs()
        l = l + (E - R).abs().mean() + (torch.log(E + 1e-5) - torch.log(R + 1e-5)).abs().mean()
    return l


def oversup(est, ref):
    E = torch.stft(est, 512, 128, window=torch.hann_window(512, device=est.device), return_complex=True).abs()
    R = torch.stft(ref, 512, 128, window=torch.hann_window(512, device=est.device), return_complex=True).abs()
    return F.relu(R.pow(0.3) - E.pow(0.3)).pow(2).mean()


def step_time(nf, df, film2, wes, bsz, secs, enr_t_s=5.0, enr_o_s=6.0, steps=10, warm=3):
    net = ESB(nf, df, film2).to(dev)
    aam = nn.Linear(128, 3758, bias=False).to(dev)
    ps = list(net.parameters()) + list(aam.parameters())
    try:
        opt = torch.optim.AdamW(ps, lr=5e-4, fused=True)
    except Exception:
        opt = torch.optim.AdamW(ps, lr=5e-4)
    spk = ResNet34().to(dev).eval() if wes else None
    L = int(SR * secs)
    mix = torch.randn(bsz, L, device=dev) * 0.1
    ref = torch.randn(bsz, L, device=dev) * 0.05
    et = torch.randn(bsz, int(SR * enr_t_s), device=dev) * 0.1
    eo = torch.randn(bsz, int(SR * enr_o_s), device=dev) * 0.1
    lab = torch.randint(0, 3758, (bsz,), device=dev)
    ts = []
    for i in range(warm + steps):
        torch.mps.synchronize(); t0 = time.time()
        if wes:
            with torch.no_grad():
                wt = spk(et)
        else:
            wt = torch.randn(bsz, 256, device=dev)
        wo = torch.randn(bsz, 256, device=dev)  # own-voice WeSpeaker vector precomputed offline
        est, act, g = net(mix, et, eo, wt, wo)
        loss = tsnr(est, ref) + mrstft(est, ref) + 0.3 * oversup(est, ref) \
            + 0.1 * F.binary_cross_entropy(act, torch.full_like(act, 0.5)) + 0.05 * F.cross_entropy(30 * aam(g), lab)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ps, 5.0); opt.step()
        torch.mps.synchronize(); ts.append(time.time() - t0)
    t = sorted(ts[warm:])
    nparam = sum(p.numel() for p in net.parameters())
    return {"median_s": round(t[len(t) // 2], 3), "min_s": round(t[0], 3), "max_s": round(t[-1], 3),
            "mps_pool_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2), "params_M": round(nparam / 1e6, 3)}


if __name__ == "__main__":
    cfgs = {
        "replica_product_esb16": dict(nf=2, df=False, film2=False, wes=False, bsz=8, secs=4.0),
        "merged_no_wes": dict(nf=3, df=True, film2=True, wes=False, bsz=8, secs=4.0),
        "merged_full": dict(nf=3, df=True, film2=True, wes=True, bsz=8, secs=4.0),
        "merged_full_4x8s": dict(nf=3, df=True, film2=True, wes=True, bsz=4, secs=8.0),
    }
    which = sys.argv[1:] or list(cfgs)
    for name in which:
        load = os.getloadavg()[0]
        r = step_time(**cfgs[name])
        r.update(name=name, load_start=round(load, 2), torch=torch.__version__)
        print(json.dumps(r), flush=True)
        torch.mps.empty_cache()
