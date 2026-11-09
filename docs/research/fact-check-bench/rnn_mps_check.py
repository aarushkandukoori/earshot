import time, torch, os
torch.manual_seed(0)
def bench(cell, dev, B=8, T=500, H=256, L=2, reps=8, warm=3):
    m = cell(H, H, num_layers=L, batch_first=True).to(dev)
    x = torch.randn(B, T, H, device=dev, requires_grad=True)
    def step():
        y, _ = m(x)
        y.square().mean().backward()
    for _ in range(warm): step()
    if dev == 'mps': torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); step()
        if dev == 'mps': torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]
print('torch', torch.__version__, 'load', os.getloadavg())
torch.set_num_threads(4)
for dev in ['mps', 'cpu']:
    for name, cell in [('LSTM', torch.nn.LSTM), ('GRU', torch.nn.GRU)]:
        med = bench(cell, dev)
        print(f'{dev} {name} 2x256 B8 T500 fwd+bwd median {med*1000:.0f} ms')
print('load after', os.getloadavg())
