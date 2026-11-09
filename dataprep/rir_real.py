"""Measured room impulse responses (plan sections 3.1 and 7, step 1).

- OpenSLR 28 (RIRS_NOISES, Apache 2.0) - the 325 real RIRs listed in real_rirs_isotropic_noises/rir_list (RWCP,
  REVERB 2014, Aachen AIR), 16 kHz. Read straight out of the 1.3 GB zip with range requests (about 52 MB). Channel 0
  of multichannel files. For training (real_train).
- MIT IR Survey (Traer and McDermott 2016, CC BY 4.0 as stated by mirrors) - 270 IRs from places people visit in daily
  life, 1.5 m source-mic distance, 32 kHz, resampled to 16 kHz. Test only (real_test), with bar, restaurant, indoor,
  vehicle and outdoor flags parsed from the file names.

Every RIR is processed the same way -
  1. find the direct path - the first sample above 30 % of the peak, then the largest sample within the next 1 ms;
  2. drop the leading silence (measurement latency, not physics), keeping PRE = 32 samples before the direct path;
  3. normalise so the direct-path sample is +1.0 (sign flipped if needed);
  4. keep L_REAL = 24,000 samples (1.5 s), with a 100 ms half-cosine fade where a longer RIR is cut;
  5. store float16, and the direct-path index "dp" in the JSON line.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile

import numpy as np
import soundfile as sf

from .common import resample
from .noise_remote import RemoteZip, fetch_bytes, head_size, make_session
from .rir_sim import FADE, FS, cut

SLR28_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
SLR28_LIST = "RIRS_NOISES/real_rirs_isotropic_noises/rir_list"
MIT_URL = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
L_REAL = 24000
PRE = 32

BAR = re.compile(r"(^|_)(bar|winebar)(_|$)|pub\b|sportsbar", re.I)
RESTAURANT = re.compile(r"restaurant|coffeeshop|cafe|cafeteria|pizzeria|foodcourt|dininghall|icecreamparlor|"
                        r"sa[nm]dwichshop|winebar", re.I)
OUTDOOR = re.compile(r"^(outside|outdoor)|streetsof|parkinglot|backyard|porch|balcony|fron[ty]yard|field|forest|"
                     r"driveway|bikepath|playground|skatepark|courtyard", re.I)
VEHICLE = re.compile(r"^(car|train)(_|$)", re.I)


def find_direct(h: np.ndarray) -> int:
    a = np.abs(h)
    onset = int(np.argmax(a >= 0.3 * a.max()))
    return onset + int(np.argmax(a[onset:onset + 16]))


def prepare(h: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """Steps 1-4 of the module docstring. Returns (float32 [L_REAL], info)."""
    h = np.asarray(h, dtype=np.float64)
    if sr != FS:
        h = resample(h, sr, FS).astype(np.float64)
    dp = find_direct(h)
    start = max(0, dp - PRE)
    h = h[start:]
    dp -= start
    rt = measure_rt60_t20(h)
    h = h / h[dp]
    return cut(h, L_REAL, FADE).astype(np.float32), {"dp": dp, "trimmed": start, "rt60": round(rt, 3),
                                                     "len_s": round(len(h) / FS, 3)}


def measure_rt60_t20(h: np.ndarray) -> float:
    """Noise-compensated Schroeder T20 for a measured RIR - a rough label, not a calibrated measurement.

    Measured RIRs end in a noise floor, which flattens the Schroeder curve and inflates the RT60 (plain T20 gave
    values near 8 s). So, in the spirit of Lundeby's method - estimate the floor from the last 10 % of the response,
    find where the 10 ms energy envelope falls to 5 dB above it, re-estimate the floor from 50 ms after that point
    (3 rounds), subtract the floor, integrate only up to the crossing, then fit the decay from -5 to -25 dB."""
    p = np.asarray(h, dtype=np.float64) ** 2
    n = len(p)
    noise = float(np.mean(p[int(0.9 * n):])) if n >= 1000 else 0.0
    w = max(1, int(0.01 * FS))
    env = np.convolve(p, np.ones(w) / w, mode="same")
    pk = int(np.argmax(env))
    end = n
    for _ in range(3):
        below = np.nonzero(env[pk:] < noise * 10 ** 0.5)[0]
        end = pk + int(below[0]) if len(below) else n
        tail = p[end + int(0.05 * FS):]
        if len(tail) < w:
            break
        noise = float(np.mean(tail))
    pc = np.clip(p[:end] - noise, 0.0, None)
    edc = np.cumsum(pc[::-1])[::-1]
    if edc[0] <= 0:
        return float("nan")
    edc_db = 10 * np.log10(np.maximum(edc / edc[0], 1e-30))
    i5 = np.nonzero(edc_db < -5)[0]
    i25 = np.nonzero(edc_db < -25)[0]
    if not len(i5) or not len(i25) or i25[0] - i5[0] < 10:
        return float("nan")
    t = np.arange(i5[0], i25[0]) / FS
    slope = np.polyfit(t, edc_db[i5[0]:i25[0]], 1)[0]      # dB per second, negative
    return float(-60.0 / slope) if slope < 0 else float("nan")


def parse_rir_list(text: str) -> list[dict]:
    """Lines like '--rir-id 00001 --room-id RVB2014_largeroom1 RIRS_NOISES/real_rirs_isotropic_noises/x.wav'."""
    out = []
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        d = {"path": tok[-1]}
        for k in range(len(tok) - 1):
            if tok[k] == "--rir-id":
                d["rir_id"] = tok[k + 1]
            elif tok[k] == "--room-id":
                d["room_id"] = tok[k + 1]
        out.append(d)
    return out


def slr28_db(name: str) -> str:
    n = name.rsplit("/", 1)[-1]
    return "RWCP" if n.startswith("RWCP") else "AIR" if n.startswith("air") else "REVERB2014" if n.startswith(
        "RVB2014") else "other"


def mit_place(member: str) -> tuple[str, str]:
    """'Audio/h035_Bar_LargeSportsBar_5txts.wav' -> ('h035', 'Bar_LargeSportsBar')."""
    stem = member.rsplit("/", 1)[-1].removesuffix(".wav")
    hid, _, rest = stem.partition("_")
    return hid, re.sub(r"_(?:\d+|s)?tx(?:ts|t|s)$", "", rest)


def mit_flags(place: str) -> dict:
    bar = bool(BAR.search(place))
    restaurant = bool(RESTAURANT.search(place))
    outdoor = bool(OUTDOOR.search(place))
    vehicle = bool(VEHICLE.search(place))
    cat = "bar" if bar else "restaurant" if restaurant else "outdoor" if outdoor else "vehicle" if vehicle else "indoor"
    return {"category": cat, "bar": bar, "restaurant": restaurant, "indoor": not outdoor, "vehicle": vehicle}


def fetch_slr28(log: logging.Logger) -> tuple[np.ndarray, list[dict]]:
    sess = make_session()
    rz = RemoteZip([(SLR28_URL, head_size(sess, SLR28_URL))], sess, log=log)
    _, text = next(rz.iter_members([SLR28_LIST]))
    rows = parse_rir_list(text.decode("utf-8"))
    by_path = {r["path"]: r for r in rows}
    arrs, metas = {}, {}
    for name, data in rz.iter_members(sorted(by_path), max_gap=1 << 20):
        if data is None:
            log.error("could not read %s", name)
            continue
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        h, info = prepare(x[:, 0], sr)
        r = by_path[name]
        arrs[name] = h
        metas[name] = {"name": name.rsplit("/", 1)[-1].removesuffix(".wav"), "rir_id": r.get("rir_id"),
                       "room_id": r.get("room_id"), "db": slr28_db(name), "channels": x.shape[1],
                       "sr_original": sr, **info, "source": "openslr28", "license": "Apache-2.0"}
    names = [r["path"] for r in rows if r["path"] in arrs]     # keep rir_list order
    log.info("OpenSLR 28 - %d of %d RIRs read, %.1f MB downloaded", len(names), len(rows), rz.bytes / 1e6)
    return np.stack([arrs[n] for n in names]).astype(np.float16), [
        {"row": i, **metas[n]} for i, n in enumerate(names)]


def fetch_mit(log: logging.Logger) -> tuple[np.ndarray, list[dict]]:
    data = fetch_bytes(make_session(), MIT_URL, log)
    arrs, metas = [], []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = sorted(n for n in z.namelist() if n.startswith("Audio/h") and n.endswith(".wav"))
        for n in names:
            x, sr = sf.read(io.BytesIO(z.read(n)), dtype="float32", always_2d=True)
            h, info = prepare(x[:, 0], sr)
            hid, place = mit_place(n)
            arrs.append(h)
            metas.append({"row": len(metas), "name": n.rsplit("/", 1)[-1].removesuffix(".wav"), "id": hid,
                          "place": place, **mit_flags(place), "channels": x.shape[1], "sr_original": sr, **info,
                          "distance_m": 1.5, "source": "mit_ir_survey", "license": "CC-BY-4.0"})
    log.info("MIT IR Survey - %d IRs (%d bar, %d restaurant, %d indoor)", len(metas), sum(m["bar"] for m in metas),
             sum(m["restaurant"] for m in metas), sum(m["indoor"] for m in metas))
    return np.stack(arrs).astype(np.float16), metas
