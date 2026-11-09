"""Plain helper functions shared by the ML-0 tests (signals and the 4 contract conditioning cases).

- Speech comes from the VCTK test speakers in data/ (CC BY 4.0) when the data pack is present; otherwise a
  synthetic speech-like signal (a voiced tone with pitch glides, syllable envelope and pauses) stands in, so the
  tests also run on a machine without the data.
"""
import json

import numpy as np
import soundfile as sf
import torch

from ml.earshot import config
from ml.earshot.model import CondInputs
from ml.earshot.vad import es_vad

CASES = ["person", "everyone", "noown", "nowespk"]   # contract 3.3 test cases


# ---------------------------------------------------------------------------------------------------------------
# Test signals
# ---------------------------------------------------------------------------------------------------------------
def synthetic_speech(n, seed):
    """Speech-like float32 signal of n samples: a voiced tone with a gliding pitch and 5 harmonics, switched on and
    off like syllables (with pauses), at roughly -26 dBFS when active, plus a little noise."""
    rng = np.random.default_rng([seed, 11])
    t = np.arange(n) / config.SR
    f0 = 110.0 + 80.0 * rng.random() + 25.0 * np.sin(2 * np.pi * (0.4 + 0.3 * rng.random()) * t)
    phase = 2 * np.pi * np.cumsum(f0) / config.SR
    voice = np.zeros(n)
    for h in range(1, 6):
        voice += np.sin(h * phase) / h
    # Syllables: 4 per second, about 70% of them voiced, half-cosine edges.
    env = np.zeros(n)
    syl = config.SR // 4
    for start in range(0, n, syl):
        if rng.random() < 0.7:
            stop = min(n, start + int(syl * 0.8))
            length = stop - start
            env[start:stop] = np.sin(np.pi * np.arange(length) / max(length, 1))
    x = 0.05 * voice * env + 0.002 * rng.standard_normal(n)
    return x.astype(np.float32)


_SPEECH_ROWS = None


def _speech_rows_by_speaker():
    """VCTK test rows grouped by speaker (sorted speaker ids), or None if the data pack is missing."""
    global _SPEECH_ROWS
    if _SPEECH_ROWS is None:
        manifest = config.DATA / "manifests" / "speech_test.jsonl"
        if not manifest.exists():
            _SPEECH_ROWS = {}
        else:
            rows = {}
            with open(manifest, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if (config.DATA / row["path"]).exists():
                            rows.setdefault(row["speaker"], []).append(row)
            _SPEECH_ROWS = {spk: rows[spk] for spk in sorted(rows)}
    return _SPEECH_ROWS or None


def speech(n, speaker_index, first_row=0):
    """n samples of speech: utterances of one VCTK test speaker joined end to end (scaled to peak 0.5),
    or the synthetic stand-in when the data pack is missing."""
    by_speaker = _speech_rows_by_speaker()
    if by_speaker is None:
        return synthetic_speech(n, seed=100 * speaker_index + first_row)
    speakers = list(by_speaker)
    rows = by_speaker[speakers[speaker_index % len(speakers)]]
    pieces, total, r = [], 0, first_row
    while total < n:
        audio, sr = sf.read(str(config.DATA / rows[r % len(rows)]["path"]), dtype="float32")
        assert sr == config.SR
        if audio.ndim > 1:
            audio = audio[:, 0]
        pieces.append(audio)
        total += len(audio)
        r += 1
    x = np.concatenate(pieces)[:n]
    return (0.5 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def noise(n, seed, level=0.02):
    """White Gaussian noise, float32."""
    return (level * np.random.default_rng([seed, 22]).standard_normal(n)).astype(np.float32)


def unit_vector(dim, seed):
    v = np.random.default_rng([seed, 33]).standard_normal(dim)
    return torch.from_numpy((v / np.linalg.norm(v)).astype(np.float32))


# ---------------------------------------------------------------------------------------------------------------
# Conditioning for the 4 contract cases
# ---------------------------------------------------------------------------------------------------------------
def make_cond_inputs(net, enr_t, enr_o, cases):
    """Batched CondInputs, one row per case name in `cases` (contract 3.3):
    person = target + own + WeSpeaker; everyone = no target, mode [0, 1]; noown = no own voice;
    nowespk = WeSpeaker path off. The WeSpeaker vectors are fixed random unit vectors (WeSpeaker is not under test).
    """
    with torch.no_grad():
        tok_t, val_t, g_t, n_t = net.enroll_clip(enr_t, es_vad(enr_t), config.MAX_T_TOK)
        tok_o, val_o, g_o, n_o = net.enroll_clip(enr_o, es_vad(enr_o), config.MAX_O_TOK)
    assert n_t >= 4 and n_o >= 4, "test enrollments must give at least 4 tokens"
    e_t, e_o = unit_vector(256, 1), unit_vector(256, 2)
    B = len(cases)
    has_t = torch.tensor([c != "everyone" for c in cases])
    has_o = torch.tensor([c != "noown" for c in cases])
    use_wes = torch.tensor([c != "nowespk" for c in cases])
    mode = torch.tensor([[0.0, 1.0] if c == "everyone" else [1.0, 0.0] for c in cases])
    return CondInputs(
        e_t=e_t.expand(B, -1).clone(), tok_t=tok_t.expand(B, -1, -1).clone(), val_t=val_t.expand(B, -1).clone(),
        g_t=g_t.expand(B, -1).clone(), e_o=e_o.expand(B, -1).clone(), tok_o=tok_o.expand(B, -1, -1).clone(),
        val_o=val_o.expand(B, -1).clone(), g_o=g_o.expand(B, -1).clone(),
        has_t=has_t, has_o=has_o, use_wes=use_wes, mode=mode)


def cond_inputs_to(ci, device):
    """Copy of a CondInputs with every tensor moved to `device`."""
    fields = {name: getattr(ci, name).to(device) for name in ci.__dataclass_fields__}
    return CondInputs(**fields)
