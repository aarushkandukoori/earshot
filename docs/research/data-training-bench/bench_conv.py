import time, numpy as np, scipy.signal as ss, torch
rng = np.random.default_rng(0)
x = rng.standard_normal(64000).astype(np.float32)          # 4 s @ 16 kHz
for rir_s in (0.3, 0.8, 1.5):
    h = (rng.standard_normal(int(rir_s * 16000)) * np.exp(-np.arange(int(rir_s*16000)) / 2000)).astype(np.float32)
    for name, f in (("fftconvolve", ss.fftconvolve), ("oaconvolve", ss.oaconvolve)):
        f(x, h); t = time.perf_counter(); n = 200
        for _ in range(n): f(x, h)
        print(f"{name:12s} rir={rir_s}s  {1000*(time.perf_counter()-t)/n:.2f} ms per 4 s source")
# batched FFT convolution on MPS: 8 examples x 5 sources
dev = torch.device("mps")
B, S, T, L = 8, 5, 64000, 12800
xs = torch.randn(B, S, T, device=dev); hs = torch.randn(B, S, L, device=dev)
def fftconv(a, b):
    n = a.shape[-1] + b.shape[-1] - 1; nfft = 1 << (n - 1).bit_length()
    return torch.fft.irfft(torch.fft.rfft(a, nfft) * torch.fft.rfft(b, nfft), nfft)[..., :a.shape[-1]]
for _ in range(3): fftconv(xs, hs)
torch.mps.synchronize(); t = time.perf_counter()
for _ in range(20): y = fftconv(xs, hs)
torch.mps.synchronize(); print(f"MPS batched FFT conv 8x5 sources (0.8 s RIR): {1000*(time.perf_counter()-t)/20:.1f} ms per batch")
