"""Libri2Mix test definitions (16 kHz, "min" mode) - parsing, enrollment choice and rendering.

We never store rendered Libri2Mix audio. data/eval/libri2mix_test.jsonl keeps one mixture per line: the two
LibriSpeech test-clean sources with their LibriMix gains, the WHAM! noise clip with its gain, the lengths and one
enrollment utterance per source. render() rebuilds the audio the same way LibriMix's
scripts/create_librimix_from_metadata.py does (github.com/JorisCos/LibriMix, MIT):

    s_i     = source_i (float32, as read) * gain_i
    noise   = WHAM! channel 0, first max(len s_1, len s_2) samples, extended with LibriMix's Hann cross-fade
              (extend_noise) when the clip is shorter, then * noise_gain
    "min"   = every signal cut to min(len s_1, len s_2) samples; all offsets are 0
    mix_clean = s_1 + s_2,  mix_both = s_1 + s_2 + noise

Enrollment - BUT/NTT's SpeakerBeam list (used by WeSep and others) comes with a click-through software evaluation
agreement (binding on access, indemnity, arbitration in Tokyo), which only the user can accept. So we build our own
deterministic list, rule "earshot-v1": another LibriSpeech test-clean utterance of the same speaker, never one of the
mixture's own sources, at least 3 s long when the speaker has one, chosen by the smallest
sha1("<mixture id>|<target utterance>|<candidate>"). Numbers measured with it are close to, but not identical
with, numbers measured with the SpeakerBeam list.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000                      # LibriMix's RATE constant, used by extend_noise
ENROLL_RULE = "earshot-v1"
ENROLL_MIN_S = 3.0


def utt_id(path: str) -> str:
    """'test-clean/4077/13754/4077-13754-0001.flac' -> '4077-13754-0001'"""
    return Path(path).name.split(".")[0]


def speaker_of(utt: str) -> str:
    return utt.split("-")[0]


def parse_mixture_csv(text: str) -> list[dict]:
    """Rows of LibriMix's metadata/Libri2Mix/libri2mix_<subset>.csv as dicts with typed gains."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        rows.append({
            "id": r["mixture_ID"],
            "sources": [(r["source_1_path"], float(r["source_1_gain"])), (r["source_2_path"], float(r["source_2_gain"]))],
            "noise": (r["noise_path"], float(r["noise_gain"])),
        })
    return rows


def parse_info_csv(text: str) -> dict[str, list[tuple[str, str]]]:
    """libri2mix_<subset>_info.csv -> {mixture_ID: [(speaker, sex), (speaker, sex)]}. Speaker ids come as '4077.0'."""
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        out[r["mixture_ID"]] = [(str(int(float(r["speaker_1_ID"]))), r["speaker_1_sex"]),
                                (str(int(float(r["speaker_2_ID"]))), r["speaker_2_sex"])]
    return out


def _rank(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def pick_enrollment(mix_id: str, target_utt: str, speaker_utts: dict[str, float], exclude: set[str],
                    min_s: float = ENROLL_MIN_S) -> str | None:
    """Deterministic enrollment utterance for target_utt in mixture mix_id.

    speaker_utts - {utterance id: duration in s} for every utterance of the target's speaker.
    exclude      - utterance ids that may not be used (the mixture's own sources).
    Prefers utterances of at least min_s seconds; returns None if the speaker has no other utterance."""
    cands = [u for u in speaker_utts if u not in exclude and u != target_utt]
    long_enough = [u for u in cands if speaker_utts[u] >= min_s]
    pool = long_enough or cands
    if not pool:
        return None
    return min(pool, key=lambda u: _rank(mix_id, target_utt, u))


def extend_noise(noise: np.ndarray, max_length: int) -> np.ndarray:
    """LibriMix's extend_noise - repeat a too-short noise clip with a Hann cross-fade until it has max_length
    samples. Copied in behaviour from create_librimix_from_metadata.py (MIT) so renders match the official ones."""
    noise_ex = noise
    window = np.hanning(RATE + 1)
    i_w = window[:len(window) // 2 + 1]          # rising half, 8001 samples
    d_w = window[len(window) // 2::-1]           # falling half, 8001 samples
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate((noise_ex[:len(noise_ex) - len(d_w)],
                                   noise_ex[len(noise_ex) - len(d_w):] * d_w + noise[:len(i_w)] * i_w,
                                   noise[len(i_w):]))
    return noise_ex[:max_length]


def build_definition(row: dict, info: list[tuple[str, str]] | None, lengths: dict[str, int],
                     utts_by_speaker: dict[str, dict[str, float]], noise_rel: str, noise_scale: float = 1.0,
                     noise_meta: dict | None = None) -> dict:
    """One line of data/eval/libri2mix_test.jsonl.

    lengths         - {utterance id: samples} for LibriSpeech test-clean (from the files on disk).
    noise_rel       - path of our stored WHAM! clip, relative to data/.
    noise_scale     - factor our stored clip was multiplied by to stay below full scale (1.0 = untouched); the
                      definition's noise gain is divided by it so renders still match LibriMix."""
    srcs = []
    utts = [utt_id(p) for p, _ in row["sources"]]
    for k, (path, gain) in enumerate(row["sources"]):
        u = utts[k]
        spk = info[k][0] if info else speaker_of(u)
        srcs.append({"path": f"LibriSpeech/{path}", "utt": u, "speaker": f"librispeech:{spk}",
                     "sex": info[k][1] if info else None, "gain": gain, "offset": 0, "samples": lengths[u]})
    length = min(s["samples"] for s in srcs)
    enroll = []
    for k, s in enumerate(srcs):
        spk = speaker_of(s["utt"])
        e = pick_enrollment(row["id"], s["utt"], utts_by_speaker.get(spk, {}), set(utts))
        enroll.append(None if e is None else {
            "path": f"LibriSpeech/test-clean/{spk}/{e.split('-')[1]}/{e}.flac", "utt": e,
            "duration": round(utts_by_speaker[spk][e], 4), "rule": ENROLL_RULE})
    wham_path, wham_gain = row["noise"]
    noise = {"path": noise_rel, "wham_id": wham_path, "gain": wham_gain / noise_scale, "gain_librimix": wham_gain,
             "offset": 0}
    if noise_meta:
        noise.update(noise_meta)
    return {"id": row["id"], "set": "libri2mix_test_min_16k", "sr": RATE, "mode": "min", "length": length,
            "max_length": max(s["samples"] for s in srcs), "sources": srcs, "noise": noise, "enrollment": enroll,
            "split": "test", "use": ["eval"],
            "license": {"sources": "CC-BY-4.0", "noise": "CC-BY-NC-4.0", "definition": "MIT"}}


def render(defn: dict, data_root: Path, noisy: bool = True) -> tuple[np.ndarray, list[np.ndarray], np.ndarray | None]:
    """Rebuild (mixture, [s_1, s_2], noise) for one definition. Arrays are float32 at 16 kHz, `length` samples."""
    srcs = []
    for s in defn["sources"]:
        x, sr = sf.read(data_root / s["path"], dtype="float32")
        assert sr == RATE, f"{s['path']} is {sr} Hz"
        srcs.append(x[s["offset"]:] * np.float32(s["gain"]))
    n_min, n_max = defn["length"], defn["max_length"]
    srcs = [x[:n_min] for x in srcs]
    mix = np.sum(srcs, axis=0).astype(np.float32)
    noise = None
    if noisy:
        nz = defn["noise"]
        x, sr = sf.read(data_root / nz["path"], dtype="float32", frames=n_max)
        assert sr == RATE, f"{nz['path']} is {sr} Hz"
        if x.ndim > 1:
            x = x[:, 0]
        if len(x) < n_max:
            x = extend_noise(x, n_max)
        noise = (x[:n_min] * np.float32(nz["gain"])).astype(np.float32)
        mix = mix + noise
    return mix, srcs, noise
