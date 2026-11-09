"""MPS microbenchmarks and correctness checks for building blocks of a causal TSE model.

Usage: python -I bench_mps.py <out_json>
Times one training step (forward + backward + AdamW) for conv / GRU / LSTM blocks on
CPU, MPS fp32, MPS autocast fp16 and bf16, and checks MPS-vs-CPU numerical agreement.
"""
import json
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
CPU = torch.device("cpu")
MPS = torch.device("mps")
B = 8          # batch
T_WAV = 64000  # 4 s at 16 kHz


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


class CausalTCN(nn.Module):
    """Conv-TasNet style separator stack with causal depthwise convs (1 repeat of 8 blocks)."""

    def __init__(self, C=256, H=512, layers=8):
        super().__init__()
        self.enc = nn.Conv1d(1, C, 32, stride=16, bias=False)       # 2 ms frames, 1 ms hop
        self.dec = nn.ConvTranspose1d(C, 1, 32, stride=16, bias=False)
        self.inp = nn.ModuleList([nn.Conv1d(C, H, 1) for _ in range(layers)])
        self.dw = nn.ModuleList([nn.Conv1d(H, H, 3, dilation=2 ** i, groups=H) for i in range(layers)])
        self.out = nn.ModuleList([nn.Conv1d(H, C, 1) for _ in range(layers)])
        self.act = nn.PReLU()
        self.dil = [2 ** i for i in range(layers)]

    def forward(self, wav):
        x = self.enc(wav)
        for inp, dw, out, d in zip(self.inp, self.dw, self.out, self.dil):
            h = self.act(inp(x))
            h = self.act(dw(F.pad(h, (2 * d, 0))))
            x = x + out(h)
        return self.dec(x)


class FrameRNN(nn.Module):
    """STFT-domain style: per-frame RNN over time (B, T=500 frames, F=257 features)."""

    def __init__(self, cell, F_in=257, H=256, layers=2):
        super().__init__()
        self.rnn = cell(F_in, H, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(H, F_in)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.proj(y)


CASES = {
    "tcn_wave_4s": (lambda: CausalTCN(), lambda: torch.randn(B, 1, T_WAV)),
    "gru2x256_500frames": (lambda: FrameRNN(nn.GRU), lambda: torch.randn(B, 500, 257)),
    "lstm2x256_500frames": (lambda: FrameRNN(nn.LSTM), lambda: torch.randn(B, 500, 257)),
    # Band-split style: many short sequences (B*T sequences of length 32 bands)
    "lstm_band_axis_4000seq_len32": (lambda: FrameRNN(nn.LSTM, F_in=64, H=64, layers=1),
                                     lambda: torch.randn(B * 500, 32, 64)),
}


def bench_train(make_model, make_input, dev, dtype=None, steps=8, warmup=3):
    torch.manual_seed(0)
    model = make_model().to(dev)
    x = make_input().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for i in range(warmup + steps):
        if i == warmup:
            sync(dev)
            t = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            y = model(x)
            loss = y.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync(dev)
    ms = (time.perf_counter() - t) / steps * 1000
    return round(ms, 1), bool(torch.isfinite(loss).item())


def agreement(make_model, make_input):
    """Same weights and input on CPU and MPS (fp32): compare outputs and gradients."""
    torch.manual_seed(0)
    m_cpu = make_model()
    x = make_input()
    m_mps = make_model()
    m_mps.load_state_dict(m_cpu.state_dict())
    m_mps = m_mps.to(MPS)
    y_cpu = m_cpu(x)
    y_cpu.pow(2).mean().backward()
    y_mps = m_mps(x.to(MPS))
    y_mps.pow(2).mean().backward()
    out_err = ((y_mps.cpu() - y_cpu).abs().max() / y_cpu.abs().max()).item()
    g_err = 0.0
    for (n, p_c), (_, p_m) in zip(m_cpu.named_parameters(), m_mps.named_parameters()):
        e = ((p_m.grad.cpu() - p_c.grad).abs().max() / (p_c.grad.abs().max() + 1e-12)).item()
        g_err = max(g_err, e)
    return {"max_rel_out_err": out_err, "max_rel_grad_err": g_err}


def op_checks():
    res = {}
    x = torch.randn(4, 64000, device=MPS, requires_grad=True)
    win = torch.hann_window(512, device=MPS)
    try:
        X = torch.stft(x, 512, 128, window=win, return_complex=True)
        X.abs().pow(2).mean().backward()
        y = torch.istft(X.detach(), 512, 128, window=win, length=64000)
        res["stft_istft_complex_mps"] = f"ok, roundtrip max err {((y - x.detach()).abs().max()).item():.2e}"
    except Exception as e:  # noqa: BLE001
        res["stft_istft_complex_mps"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
    try:
        z = torch.randn(8, 257, 500, dtype=torch.complex64, device=MPS)
        w = torch.randn(8, 257, 500, dtype=torch.complex64, device=MPS)
        (z * w).abs().mean().item()
        res["complex_mul_mps"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["complex_mul_mps"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
    try:
        s = torch.amp.GradScaler("mps")
        res["GradScaler_mps"] = f"constructed, enabled={s.is_enabled()}"
    except Exception as e:  # noqa: BLE001
        res["GradScaler_mps"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
    try:
        with torch.autocast("mps", dtype=torch.bfloat16):
            a = torch.randn(64, 64, device=MPS) @ torch.randn(64, 64, device=MPS)
        res["autocast_bf16_matmul_dtype"] = str(a.dtype)
    except Exception as e:  # noqa: BLE001
        res["autocast_bf16_matmul_dtype"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
    try:
        lstm = nn.LSTM(64, 64, batch_first=True).to(MPS)
        with torch.autocast("mps", dtype=torch.float16):
            o, _ = lstm(torch.randn(2, 10, 64, device=MPS))
        res["autocast_fp16_lstm_out_dtype"] = str(o.dtype)
    except Exception as e:  # noqa: BLE001
        res["autocast_fp16_lstm_out_dtype"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
    return res


def main():
    out = {"torch": torch.__version__, "mps": torch.backends.mps.is_available(),
           "threads": torch.get_num_threads(), "batch": B, "ms_per_train_step": {}, "agreement": {}}
    for name, (mk_m, mk_x) in CASES.items():
        row = {}
        for label, dev, dt in [("cpu_fp32", CPU, None), ("mps_fp32", MPS, None),
                               ("mps_fp16_autocast", MPS, torch.float16),
                               ("mps_bf16_autocast", MPS, torch.bfloat16)]:
            try:
                row[label] = bench_train(mk_m, mk_x, dev, dt, steps=4 if dev.type == "cpu" else 8)
            except Exception as e:  # noqa: BLE001
                row[label] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
                traceback.print_exc()
        out["ms_per_train_step"][name] = row
        try:
            out["agreement"][name] = agreement(mk_m, mk_x)
        except Exception as e:  # noqa: BLE001
            out["agreement"][name] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
        print(name, row, out["agreement"][name], flush=True)
    out["ops"] = op_checks()
    out["mps_driver_allocated_GB"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    print(json.dumps(out, indent=1))
    with open(sys.argv[1], "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
