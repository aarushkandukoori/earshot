"""Simulated room impulse responses - pyroomacoustics image source model (MIT license), plan section 3.2.

Every room has one omnidirectional microphone (the phone) and 8 source positions, always in this order:
    0  target     the person the user wants to hear, 0.5 to 2.5 m from the phone (log-uniform, so mostly near)
    1-3 talker    competing talkers, 1 to 5 m
    4  own_voice  the phone owner's mouth, 0.25 to 0.8 m
    5-7 extra     more positions for babble talkers and point noises, 2 to 8 m (shorter in small rooms)

Two ways of using the phone (plan 3.2):
    in_hand  (60 %) - mic 1.0 to 1.4 m high. Talkers seated (mouth 1.1 to 1.3 m) or standing (1.45 to 1.75 m).
                      Own voice 0.25 to 0.5 m away and 0.15 to 0.4 m above the mic, behind the phone.
    on_table (40 %) - phone lying on a 0.72 to 0.78 m table. A phone on a table is close to a boundary microphone,
                      so every source gets a second image mirrored about the table plane, scaled by a reflection
                      factor of 0.6 to 0.9, simulated in the same room (pyroomacoustics has no table object).
                      All sources sit above the table and at most 2 x table - 5 cm high, so the mirror image stays
                      inside the room. Talkers are seated; own voice is 0.4 to 0.8 m away.

RT60 - drawn per room (90 % log-uniform 0.2 to 1.3 s, 10 % uniform 1.3 to 2.0 s). The plan measured that inverse
Sabine misses its target (0.8 s became 1.07 s), so we simulate the target source alone, measure its RT60 (Schroeder
T30) and correct the absorption up to 4 times, then simulate the whole room and store the measured value of each RIR.

ISM - max_order from the Sabine formula capped at 40, use_rand_ism with 5 cm jitter (fewer sweeping echoes), air
absorption on, fs 16 kHz, frequency-flat wall absorption.

Level - kept physical, as pyroomacoustics gives it - a direct path of amplitude 1/r, so a source 1 m away in free
field gives a unit impulse and the mixer can let geometry set the received level (plan 5.2).
"""

from __future__ import annotations

import math

import numpy as np
import pyroomacoustics as pra

FS = 16000
C = 343.0
L = 17600             # samples kept per RIR (1.1 s); 3,800 + 100 + 100 rooms x 8 x 17,600 x 2 bytes = 1.13 GB
FADE = 1600           # 100 ms half-cosine fade-out where a longer RIR is cut
MAX_ORDER = 40
RAND_DISP = 0.05
PRA_DELAY = 40        # pyroomacoustics delays every arrival by frac_delay_length // 2 = 40 samples (2.5 ms)
ROLES = ("target", "talker1", "talker2", "talker3", "own_voice", "extra1", "extra2", "extra3")
SPLITS = {"train": (3800, 1001), "val": (100, 2002), "test": (100, 3003)}   # rooms, seed (one seed per split)


def loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_rt60(rng: np.random.Generator) -> float:
    return loguniform(rng, 0.2, 1.3) if rng.random() < 0.9 else float(rng.uniform(1.3, 2.0))


def sample_dims(rng: np.random.Generator) -> list[float]:
    """Length 4-20 m, width 3-15 m (never wider than long), height 2.5-5 m with high ceilings only in big rooms."""
    lx = loguniform(rng, 4.0, 20.0)
    ly = loguniform(rng, 3.0, min(15.0, lx))
    lz = float(rng.uniform(2.5, min(5.0, 2.7 + 2.3 * (lx - 4.0) / 16.0)))
    return [lx, ly, lz]


def ism_reach(dims: list[float]) -> float:
    """The pyroomacoustics distance unit behind max_order - min over wall pairs of a*b/sqrt(a^2 + b^2), in metres."""
    lx, ly, lz = dims
    return min(a * b / math.hypot(a, b) for a, b in ((lx, ly), (lx, lz), (ly, lz)))


def sabine(rt60: float, dims: list[float]) -> tuple[float, int]:
    """Energy absorption and image-source order for a wanted RT60 - the formula of pyroomacoustics.inverse_sabine,
    but clamped (absorption 0.01 to 0.99, order 1 to 40) instead of raising when the room is too big for the RT60."""
    lx, ly, lz = dims
    vol = lx * ly * lz
    surf = 2 * (lx * ly + lx * lz + ly * lz)
    e_abs = 24 * math.log(10) * vol / (C * surf * rt60)
    order = math.ceil(C * rt60 / ism_reach(dims) - 1)
    return float(min(max(e_abs, 0.01), 0.99)), int(min(max(order, 1), MAX_ORDER))


def mouth_height(rng: np.random.Generator, geometry: str) -> float:
    if geometry == "on_table" or rng.random() < 0.5:
        return float(rng.uniform(1.1, 1.3))      # seated
    return float(rng.uniform(1.45, 1.75))        # standing


def _inside_xy(p, dims, margin) -> bool:
    return margin <= p[0] <= dims[0] - margin and margin <= p[1] <= dims[1] - margin


def place(rng, mic, dims, others, d_lo, d_hi, z, log_d=False, az=None, az_spread=math.pi, margin=0.25,
          min_sep=0.3) -> tuple[list[float], float]:
    """A position at 3D distance d from the mic (d drawn from [d_lo, d_hi]) and height z, inside the room with a wall
    margin, at least min_sep from the other sources. az (optional) centres the direction. When the room is too small
    for the wanted distance, the upper bound shrinks step by step. Returns (position, distance)."""
    dz = z - mic[2]
    need = abs(dz) + 0.05          # the horizontal part must be positive
    for shrink in range(14):
        hi = max(d_hi * 0.85**shrink, need)
        lo = min(max(d_lo, need), hi)
        for _ in range(40):
            d = loguniform(rng, lo, hi) if (log_d and hi > lo) else float(rng.uniform(lo, hi))
            r = math.sqrt(max(d * d - dz * dz, 0.0))
            a = rng.uniform(0, 2 * math.pi) if az is None else az + rng.uniform(-az_spread, az_spread)
            p = [mic[0] + r * math.cos(a), mic[1] + r * math.sin(a), z]
            if _inside_xy(p, dims, margin) and all(math.dist(p, o) >= min_sep for o in others):
                return p, math.dist(p, mic)
    raise ValueError("could not place a source")


def sample_room(rng: np.random.Generator) -> dict:
    """Room, phone and the 8 source positions (see the module docstring). Raises ValueError on a rare impossible
    draw; build_room() simply draws again."""
    geometry = "in_hand" if rng.random() < 0.6 else "on_table"
    rt60 = sample_rt60(rng)
    # Keep the wall absorption the RT60 needs between 0.06 and 0.95 - a big room cannot be that dead, and a small
    # room that reverberant would need mirror-like walls. Also, with max_order 40 the image sources only reach about
    # RT60 = 0.42 x ism_reach (measured on test rooms), so long RT60s get big, tall rooms, as in real buildings.
    best = None
    for _ in range(300):
        dims = sample_dims(rng)
        e_abs = sabine(rt60 / 1.25, dims)[0]
        if 0.06 <= e_abs <= 0.95 and rt60 <= 0.42 * ism_reach(dims):
            break
        if e_abs <= 0.95 and (best is None or ism_reach(dims) > ism_reach(best)):
            best = dims
    else:
        dims = best or dims
    table = None
    if geometry == "in_hand":
        mic = [float(rng.uniform(0.8, dims[0] - 0.8)), float(rng.uniform(0.8, dims[1] - 0.8)),
               float(rng.uniform(1.0, 1.4))]
        zmax = dims[2] - 0.3
    else:
        h = float(rng.uniform(0.72, 0.78))
        table = {"height": round(h, 4), "reflection": round(float(rng.uniform(0.6, 0.9)), 4)}
        mic = [float(rng.uniform(0.8, dims[0] - 0.8)), float(rng.uniform(0.8, dims[1] - 0.8)), h + 0.01]
        zmax = 2 * h - 0.05        # mirror image (2h - z) stays at least 5 cm above the floor
    pos, dist = [], []

    def add(p_d):
        pos.append(p_d[0])
        dist.append(p_d[1])

    # 0 - target
    lo = 0.5 if geometry == "in_hand" else 0.6
    add(place(rng, mic, dims, pos, lo, 2.5, min(mouth_height(rng, geometry), zmax), log_d=True))
    az_t = math.atan2(pos[0][1] - mic[1], pos[0][0] - mic[0])
    # 1-3 - competing talkers
    for _ in range(3):
        add(place(rng, mic, dims, pos, 1.0, 5.0, min(mouth_height(rng, geometry), zmax)))
    # 4 - own voice, behind the phone (the phone points at the target)
    if geometry == "in_hand":
        dz = float(rng.uniform(0.15, 0.4))
        add(place(rng, mic, dims, pos, max(0.25, dz + 0.05), 0.5, mic[2] + dz, az=az_t + math.pi,
                  az_spread=math.pi / 4, margin=0.1, min_sep=0.2))
    else:
        z = float(rng.uniform(1.1, 1.3))
        add(place(rng, mic, dims, pos, max(0.4, z - mic[2] + 0.05), 0.8, z, az=az_t + math.pi,
                  az_spread=math.pi / 3, margin=0.1, min_sep=0.2))
    # 5-7 - extra positions for babble talkers and point noises
    for _ in range(3):
        z = float(rng.uniform(0.6, min(1.9, zmax))) if geometry == "in_hand" else float(
            rng.uniform(table["height"] + 0.15, zmax))
        add(place(rng, mic, dims, pos, 2.0, 8.0, z))
    return {"geometry": geometry, "rt60_target": rt60, "dims": dims, "mic": mic, "table": table,
            "pos": pos, "dist": dist}


def measure_rt60(h: np.ndarray) -> float:
    """Schroeder T30 (fit from -5 to -35 dB, extrapolated to 60 dB) with pyroomacoustics; T20 if that fails."""
    for decay in (30, 20):
        try:
            v = float(pra.experimental.measure_rt60(h, fs=FS, decay_db=decay))
        except Exception:
            v = 0.0
        if np.isfinite(v) and v > 0:
            return v
    return float("nan")


def _simulate(dims, e_abs, order, sources, mic) -> list[np.ndarray]:
    room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(e_abs), max_order=order, use_rand_ism=True,
                       max_rand_disp=RAND_DISP, air_absorption=True)
    room.add_microphone(np.array(mic, dtype=float))
    for p in sources:
        room.add_source(np.array(p, dtype=float))
    room.compute_rir()
    return [np.asarray(h, dtype=np.float64) for h in room.rir[0]]


def cut(h: np.ndarray, n: int = L, fade: int = FADE) -> np.ndarray:
    """First n samples; when h is longer, the last `fade` samples get a half-cosine fade so the cut is not a click."""
    out = np.zeros(n, dtype=np.float64)
    if len(h) > n:
        out[:] = h[:n]
        out[n - fade:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade)))
    else:
        out[:len(h)] = h
    return out


def direct_index(h: np.ndarray, dist: float, window: int = 8) -> int:
    """Sample index of the direct-path peak - the largest |h| within +-window samples of the expected arrival."""
    n0 = int(round(dist / C * FS)) + PRA_DELAY
    lo = max(0, n0 - window)
    return lo + int(np.argmax(np.abs(h[lo:n0 + window + 1])))


def simulate_room(spec: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Calibrate the absorption on the target alone, then compute all 8 RIRs. Returns (float32 [8, L], meta)."""
    pra.random.seed(int(rng.integers(2**31 - 1)))       # use_rand_ism draws from pyroomacoustics' own generator
    dims, mic, target = spec["dims"], spec["mic"], spec["rt60_target"]
    req, best = target / 1.25, None
    for _ in range(4):
        e_abs, order = sabine(req, dims)
        rt = measure_rt60(_simulate(dims, e_abs, order, [spec["pos"][0]], mic)[0])
        err = abs(math.log(rt / target)) if np.isfinite(rt) else 9.0
        if best is None or err < best[0]:
            best = (err, e_abs, order, rt)
        if err < math.log(1.1):
            break
        req = float(np.clip(req * target / rt, 0.05, 6.0)) if np.isfinite(rt) else req * 0.8
    _, e_abs, order, rt_probe = best
    sources = list(spec["pos"])
    if spec["table"]:
        h_t = spec["table"]["height"]
        sources += [[p[0], p[1], 2 * h_t - p[2]] for p in spec["pos"]]     # mirror images about the table
    raw = _simulate(dims, e_abs, order, sources, mic)
    rirs, rt_src, dps, amps, lens = [], [], [], [], []
    for k in range(len(ROLES)):
        h = raw[k]
        if spec["table"]:
            m = raw[k + len(ROLES)] * spec["table"]["reflection"]
            n = max(len(h), len(m))
            h = np.pad(h, (0, n - len(h))) + np.pad(m, (0, n - len(m)))
        rt_src.append(measure_rt60(h))
        lens.append(len(h))
        hc = cut(h)
        dp = direct_index(hc, spec["dist"][k])
        rirs.append(hc)
        dps.append(dp)
        amps.append(float(hc[dp]))
    meta = {
        "geometry": spec["geometry"], "dims": [round(v, 3) for v in dims], "mic": [round(v, 3) for v in mic],
        "table": spec["table"], "rt60_target": round(target, 3), "rt60": round(float(np.nanmedian(rt_src)), 3),
        "rt60_src": [round(v, 3) for v in rt_src], "rt60_probe": round(rt_probe, 3),
        "e_absorption": round(e_abs, 4), "max_order": order, "air_absorption": True, "rand_ism_disp": RAND_DISP,
        "roles": list(ROLES), "pos": [[round(v, 3) for v in p] for p in spec["pos"]],
        "dist": [round(v, 3) for v in spec["dist"]], "dp": dps, "dp_amp": [round(v, 4) for v in amps],
        "raw_len": lens,
    }
    return np.stack(rirs).astype(np.float32), meta


def build_room(split: str, idx: int) -> tuple[int, np.ndarray, dict]:
    """Room idx of a split - same seed, same room, on every machine and in any process order."""
    _, seed = SPLITS[split]
    rng = np.random.default_rng([seed, idx])
    for _ in range(50):
        try:
            spec = sample_room(rng)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError(f"could not sample room {split}/{idx}")
    rirs, meta = simulate_room(spec, rng)
    meta = {"room": idx, "split": split, "seed": [seed, idx], **meta}
    return idx, rirs.astype(np.float16), meta
