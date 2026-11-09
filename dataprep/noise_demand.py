"""DEMAND - real multichannel noise recordings, 300 s per scene (plan sections 2.2 and 2.3).

Source - Zenodo record 1227121. One zip per scene and sample rate, each holding 16 mono WAVs (ch01 .. ch16). We read
only 4 channels per scene out of each zip with HTTP range requests (about 28 MB instead of 100 MB per scene).

Channels - the 16 microphones sit in 4 staggered rows of 4, 5 cm apart (DEMAND.pdf, figure 1). We keep the four
corner microphones (1, 4, 13, 16), the pairs that are farthest apart and so least alike.

Splits - the scenes the plan lists are training noise; SCAFE (cafe terrace, only published at 48 kHz) is held out as
test noise, as the plan says. Nine scenes are too few to carve a validation split from, so DEMAND has no val files
(validation noise comes from FSD50K's own val split and MUSAN).

License - the PDF bundled with the dataset says CC BY-SA 3.0 (the Zenodo page says CC BY 4.0); we use the stricter one.
"""

from __future__ import annotations

import logging

from . import common
from .common import entry, flac_ok, write_flac
from .noise_remote import RateLimiter, RemoteZip, head_size, make_session
from .noise_util import NoiseManifests, decode_audio, to_16k_mono

ZENODO = "https://zenodo.org/records/1227121/files/"
LICENSE = "CC-BY-SA-3.0"
CHANNELS = (1, 4, 13, 16)

# scene -> (split, category, description from DEMAND.pdf)
SCENES = {
    "PCAFETER": ("train", "restaurant", "busy office cafeteria"),
    "PRESTO": ("train", "restaurant", "university restaurant at lunchtime"),
    "OMEETING": ("train", "meeting", "meeting room while the microphone array is discussed"),
    "OHALLWAY": ("train", "hallway", "office building hallway, people passing by"),
    "PSTATION": ("train", "station", "main transfer area of a busy subway station"),
    "SPSQUARE": ("train", "crowd", "public town square with many tourists"),
    "STRAFFIC": ("train", "traffic", "busy traffic intersection"),
    "TBUS": ("train", "vehicle", "inside a public transit bus"),
    "TMETRO": ("train", "vehicle", "inside a subway train"),
    "SCAFE": ("test", "restaurant", "terrace of a cafe at a public square"),
}


def zip_name(scene: str) -> str:
    return f"{scene}_48k.zip" if scene == "SCAFE" else f"{scene}_16k.zip"


def rel_path(scene: str, ch: int) -> str:
    return f"noise/demand/{scene}/ch{ch:02d}.flac"


def make_entry(scene: str, ch: int, seconds: float) -> dict:
    split, cat, desc = SCENES[scene]
    return entry(rel_path(scene, ch), "demand", seconds, split, LICENSE, ["noise"], group=scene, category=cat,
                 scene=scene, channel=ch, desc=desc, sr_original=48000 if scene == "SCAFE" else 16000)


def run(log: logging.Logger) -> dict:
    sess = make_session()
    lim = RateLimiter(0.55)
    stats = {"written": 0, "already_done": 0}
    with NoiseManifests() as man:
        for scene in SCENES:
            want = {}
            for ch in CHANNELS:
                rel = rel_path(scene, ch)
                secs = flac_ok(rel)
                if secs and rel in man:
                    stats["already_done"] += 1
                elif secs:
                    man.add(make_entry(scene, ch, secs))
                    stats["already_done"] += 1
                else:
                    want[f"{scene}/ch{ch:02d}.wav"] = ch
            if not want:
                continue
            url = f"{ZENODO}{zip_name(scene)}?download=1"
            rz = RemoteZip([(url, head_size(sess, url))], sess, lim, log)
            for name, data in rz.iter_members(sorted(want), max_gap=0):
                if data is None:
                    log.error("could not read %s from %s", name, zip_name(scene))
                    continue
                x, sr, _ = decode_audio(data)
                y = to_16k_mono(x, sr)
                ch = want[name]
                write_flac(rel_path(scene, ch), y)
                man.add(make_entry(scene, ch, y.size / common.SR))
                stats["written"] += 1
            log.info("DEMAND %s done (%d channels, %.1f MB read)", scene, len(want), rz.bytes / 1e6)
    log.info("DEMAND finished %s", stats)
    return stats
