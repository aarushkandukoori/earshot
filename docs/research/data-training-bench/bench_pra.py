import time, numpy as np, pyroomacoustics as pra
fs = 16000
def make(rt60, dim, hybrid, n_src=5, seed=0):
    rng = np.random.default_rng(seed)
    e_abs, max_order = pra.inverse_sabine(rt60, dim)
    if hybrid:
        room = pra.ShoeBox(dim, fs=fs, materials=pra.Material(e_abs), max_order=3, ray_tracing=True, air_absorption=True)
        room.set_ray_tracing(n_rays=5000, receiver_radius=0.5)
    else:
        room = pra.ShoeBox(dim, fs=fs, materials=pra.Material(e_abs), max_order=min(max_order, 40), use_rand_ism=True, max_rand_disp=0.05, air_absorption=True)
    mic = np.array([dim[0] * 0.5, dim[1] * 0.5, 1.2])
    room.add_microphone(mic)
    for _ in range(n_src):
        p = np.array([rng.uniform(0.5, dim[0]-0.5), rng.uniform(0.5, dim[1]-0.5), rng.uniform(1.1, 1.7)])
        room.add_source(p)
    t = time.perf_counter(); room.compute_rir(); dt = time.perf_counter() - t
    lens = [len(r) for r in room.rir[0]]
    rt = pra.experimental.measure_rt60(room.rir[0][0], fs=fs)
    return dt, max_order, np.mean(lens) / fs, rt
for rt60, dim in ((0.4, [6, 5, 3]), (0.8, [10, 8, 3.5]), (1.2, [15, 12, 4])):
    for hybrid in (False, True):
        dt, mo, L, rt = make(rt60, dim, hybrid)
        print(f"rt60={rt60} dim={dim} hybrid={hybrid}: {dt:.2f} s for 5 RIRs (sabine max_order={mo}, used {'3+rays' if hybrid else min(mo,40)}), rir len {L:.2f} s, measured RT60 {rt:.2f} s")
