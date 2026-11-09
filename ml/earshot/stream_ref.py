"""Per-hop streaming reference of ESB-16-DF, CPU float32 (spec 10-ml section 4, contract 2.1 and 2.6).

This is the Python twin of the C engine's `es_stream`: it keeps exactly the state of contract 2.6 and processes one
64-sample hop at a time. The test vectors (`.est` files) that the C engine must match are made with this class.

State (all float32, zeroed by reset):
  ana   [512]          analysis history, newest sample last
  xhist [2, 128]       complex X of the previous 2 hops for bins 0..127 (index 0 = t-1, index 1 = t-2)
  h, c  [4, 16, 128]   time-LSTM hidden and cell state per block per band
  ola   [128]          overlap-add buffer

Identity with the offline model (tested): y_stream[n + 64] == y_offline[n] within 1e-5, because the streamed
output is the offline output delayed by one hop (the engine has to wait for the hop to arrive).
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch

from .config import ATT_DIM, ATT_HEADS, D, DF_BINS, HEAD_DIM, HOP, K, N_FFT, N_TOKENS, OFFSETS, TAPS, WIDTHS
from .dsp import windows
from .model import features


@dataclass
class PackedCond:
    """One instance's conditioning set in the engine layout (contract 2.6 `es_cond`)."""
    film: torch.Tensor   # [4096] flatten of [2, 2, 16, 64]: index ((p*2 + s)*16 + k)*64 + d
    k: torch.Tensor      # [4, 68, 16] keys: the n valid tokens first (target, own, null order), zeros after
    v: torch.Tensor      # [4, 68, 16] values, same layout
    n: int               # number of valid tokens (n_t + n_o + 4)


def pack_cond(cond, i):
    """Example i of a batched model.Cond -> PackedCond (CPU float32), exactly what es_cond_build makes in C."""
    film = cond.film[i].detach().float().cpu().reshape(-1).clone()
    valid = cond.valid[i].detach().cpu()
    order = torch.nonzero(valid).flatten()       # indices of valid slots, ascending, so the order is kept
    n = int(order.numel())
    k = torch.zeros(ATT_HEADS, N_TOKENS, HEAD_DIM, dtype=torch.float32)
    v = torch.zeros(ATT_HEADS, N_TOKENS, HEAD_DIM, dtype=torch.float32)
    k[:, :n] = cond.k[i].detach().float().cpu()[:, order]
    v[:, :n] = cond.v[i].detach().float().cpu()[:, order]
    return PackedCond(film=film, k=k, v=v, n=n)


class StreamRef:
    """Streaming reference. StreamRef(model).run(pc, x) gives the samples the engine must produce for input x.

    With allpass=True the network is skipped (Y = X), which must give the input back delayed by 64 samples.
    """

    def __init__(self, model, allpass=False):
        # Work on a private CPU copy in eval mode. model.to() and model.eval() change a module in place, so using
        # the caller's object directly would silently move a live MPS training model to the CPU and out of
        # training mode. The copy also means later changes to the caller's model do not reach this reference.
        self.model = copy.deepcopy(model).to("cpu").eval()
        self.allpass = allpass
        self.wa, self.ws = windows()
        self.reset()

    def reset(self):
        """Zero the whole streaming state (contract 2.6)."""
        n_blocks = len(self.model.blocks)
        hidden = self.model.blocks[0].t.hidden_size
        self.ana = torch.zeros(N_FFT, dtype=torch.float32)
        self.xhist = torch.zeros(2, DF_BINS, dtype=torch.complex64)
        self.h = torch.zeros(n_blocks, K, hidden, dtype=torch.float32)
        self.c = torch.zeros(n_blocks, K, hidden, dtype=torch.float32)
        self.ola = torch.zeros(2 * HOP, dtype=torch.float32)

    # -----------------------------------------------------------------------------------------------------------
    def _attend(self, pc, z):
        """Cross-attention over the first pc.n tokens of the packed keys and values (no masking needed)."""
        att = self.model.att
        n = pc.n
        q = att.q(z.reshape(K * D)).reshape(ATT_HEADS, HEAD_DIM)           # head h = q[16h : 16h + 16]
        keys = pc.k[:, :n]                                                 # [4, n, 16]
        values = pc.v[:, :n]
        scores = torch.einsum("hd,hnd->hn", q, keys) * 0.25                # [4, n]
        p = torch.softmax(scores, dim=-1)
        a = torch.einsum("hn,hnd->hd", p, values).reshape(ATT_DIM)         # a[16h + d]
        return z + att.o(a).reshape(K, D)

    def _network(self, pc, X, taps):
        """Steps 2 to 6 of 10-ml section 4 for one hop: X [257] complex -> (Y [257] complex, heads2 [2])."""
        m = self.model
        film = pc.film.reshape(2, 2, K, D)

        # 2. Features and band encoder.
        f = features(X)                       # [257, 3]
        z = m.enc(f)                          # [16, 64]
        if taps is not None:
            taps["feat"] = f.reshape(-1).clone()          # 771 floats, index 3 * bin + f
            taps["z_enc"] = z.clone()

        # 3. The four dual-path blocks, with FiLM 0 + attention after block 0 and FiLM 1 after block 2.
        for b in range(len(m.blocks)):
            blk = m.blocks[b]
            # Time path: the 16 bands are a batch of 16 sequences of length 1, carrying (h, c) from the last hop.
            out, (hb, cb) = blk.t(blk.tn(z)[:, None, :], (self.h[b][None], self.c[b][None]))
            self.h[b] = hb[0]
            self.c[b] = cb[0]
            z = z + blk.tf(out[:, 0])
            # Band path: one sequence over the 16 bands, zero initial state.
            z = z + blk.bf(blk.b(blk.bn(z)[None])[0][0])
            if b == 0:
                z = z * (1 + film[0, 0]) + film[0, 1]
                z = self._attend(pc, z)
            if b == 2:
                z = z * (1 + film[1, 0]) + film[1, 1]
            if taps is not None:
                taps[f"z_blk{b}"] = z.clone()

        # 4. Heads (with the sigmoid here, unlike the offline forward which returns logits).
        heads2 = torch.sigmoid(m.heads(z.mean(dim=0)))

        # 5. Decoder: 16 coefficient vectors, concatenated in band order = 1,026 floats.
        coefs = m.decode(z)
        if taps is not None:
            taps["coef"] = torch.cat(coefs).clone()

        # 6. Deep filter: tap 0 uses this hop's X, tap 1 the previous hop (xhist[0]), tap 2 the one before (xhist[1]).
        history = [X, self.xhist[0], self.xhist[1]]
        bands = []
        for k in range(K):
            lo, hi = OFFSETS[k], OFFSETS[k] + WIDTHS[k]
            C = m.coef_complex(coefs[k], k)            # [T_k, W_k] complex
            Yk = C[0] * history[0][lo:hi]
            for tau in range(1, TAPS[k]):
                Yk = Yk + C[tau] * history[tau][lo:hi]
            bands.append(Yk)
        Y = torch.cat(bands)
        self.xhist[1] = self.xhist[0]                 # index assignment copies the values
        self.xhist[0] = X[:DF_BINS]
        return Y, heads2

    def hop(self, pc, x64, taps=None):
        """Process one hop of 64 new input samples -> (y64 [64], heads2 [2]) as float32 tensors.

        pc is a PackedCond (ignored with allpass). If `taps` is a dict, the debug tensors of contract 3.3
        (feat, z_enc, z_blk0..3, coef) for this hop are stored in it.
        """
        with torch.no_grad():
            x64 = torch.as_tensor(x64, dtype=torch.float32)
            # 1. Shift the new samples into the analysis history, window, FFT.
            self.ana = torch.cat([self.ana[HOP:], x64])
            X = torch.fft.rfft(self.ana * self.wa)    # [257] complex
            if self.allpass:
                Y = X
                heads2 = torch.zeros(2, dtype=torch.float32)
            else:
                Y, heads2 = self._network(pc, X, taps)
            # 7. Synthesis: irfft, window, overlap-add the last 128 samples, emit the oldest 64.
            frame = torch.fft.irfft(Y, n=N_FFT) * self.ws
            self.ola = self.ola + frame[N_FFT - 2 * HOP:]
            y64 = self.ola[:HOP].clone()
            self.ola = torch.cat([self.ola[HOP:], torch.zeros(HOP, dtype=torch.float32)])
        return y64, heads2

    def run(self, pc, x, debug=0, reset=True):
        """Stream a whole signal: x [N] -> (y [N'], heads [N'/64, 2]) as float32 numpy, N' = 64 * (N // 64).

        Hop j emits samples 64j .. 64j + 63. Starts from a reset state unless reset=False. With debug = 8 it also
        returns a dict of the first 8 hops' debug tensors: feat [8, 771], z_enc [8, 16, 64], z_blk0..3 [8, 16, 64],
        coef [8, 1026].
        """
        if debug > 0 and self.allpass:
            raise ValueError("debug taps need the network (allpass=False)")
        if reset:
            self.reset()
        x = np.asarray(x, dtype=np.float32)
        n_hops = len(x) // HOP
        y = np.zeros(n_hops * HOP, dtype=np.float32)
        heads = np.zeros((n_hops, 2), dtype=np.float32)
        dbg_rows = []
        for j in range(n_hops):
            taps = {} if j < debug else None
            y64, heads2 = self.hop(pc, torch.from_numpy(x[j * HOP:(j + 1) * HOP]), taps)
            y[j * HOP:(j + 1) * HOP] = y64.numpy()
            heads[j] = heads2.numpy()
            if taps is not None:
                dbg_rows.append(taps)
        if debug <= 0:
            return y, heads
        dbg = {}
        for name in ["feat", "z_enc", "z_blk0", "z_blk1", "z_blk2", "z_blk3", "coef"]:
            dbg[name] = np.stack([row[name].numpy() for row in dbg_rows]).astype(np.float32)
        return y, heads, dbg
