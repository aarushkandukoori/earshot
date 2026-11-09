"""Robustness of pretrained speaker encoders to short, noisy enrollment (LibriSpeech test-clean, 40 speakers).

Usage: python -I bench_spk.py <librispeech_test_clean_dir> <models_dir> <out_json>
For each encoder: EER of (noisy enrollment, clean 4 s test) trials and cosine drift between the
noisy and clean embedding of the same enrollment crop. Enrollment conditions mimic pointing a phone
at a talker in a loud room (babble + pink noise, a competing talker, reverberation).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi

SR = 16000
rng = np.random.default_rng(0)
torch.set_num_threads(4)


# ---------------------------------------------------------------- data
def load_corpus(root):
    spk = {}
    for f in sorted(glob.glob(f"{root}/**/*.flac", recursive=True)):
        s = os.path.basename(f).split("-")[0]
        spk.setdefault(s, []).append(f)
    return spk


def crop(x, n, r):
    if len(x) <= n:                      # tile (never zero-pad: zeros break log-fbank + CMN)
        x = np.resize(x, n + 1)
    s = r.integers(len(x) - n)
    return x[s:s + n]


def concat_to(files, audio, n, r):
    """Concatenate utterances of one speaker until >= n samples, then take a random n-sample crop."""
    parts, total = [], 0
    for f in r.permutation(files):
        parts.append(audio[f]); total += len(audio[f])
        if total >= n:
            break
    return crop(np.concatenate(parts), n, r)


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def pink(n, r):
    X = np.fft.rfft(r.standard_normal(n))
    f = np.arange(len(X)); f[0] = 1
    return np.fft.irfft(X / np.sqrt(f), n)


def synth_rir(rt60, r, n=int(0.8 * SR)):
    t = np.arange(n) / SR
    tail = r.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.003 * SR)] = 0
    h = tail / np.sqrt(np.sum(tail ** 2)) * 10 ** (-1 / 20)   # DRR about +1 dB
    h[0] = 1.0
    return h


def corrupt(x, cond, others, r):
    """others: list of arrays from non-trial speakers (for babble / competing talker)."""
    if cond.startswith("clean"):
        return x
    y = x.copy()
    if "rev" in cond:
        y = np.convolve(y, synth_rir(0.6, r))[: len(y)]
    sig = rms(y)
    out = y.copy()
    if "talker" in cond:  # one competing talker 5 dB below the target
        c = crop(others[0], len(y), r)
        out += c / rms(c) * sig * 10 ** (-5 / 20)
    snr = float(cond.split("snr")[1].split("_")[0])
    bab = sum(crop(o, len(y), r) / rms(crop(o, len(y), r)) for o in others[1:7])
    noise = bab / rms(bab) + 0.5 * pink(len(y), r) / rms(pink(len(y), r))
    out += noise / rms(noise) * sig * 10 ** (-snr / 20)
    return out


# ---------------------------------------------------------------- encoders
class WeSpeakerOnnx:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, wav):
        w = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0) * (1 << 15)
        f = kaldi.fbank(w, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                        sample_frequency=SR, window_type="hamming", use_energy=False)
        f = f - f.mean(dim=0, keepdim=True)
        e = self.sess.run(None, {self.inp: f.unsqueeze(0).numpy()})[0][0]
        return e / np.linalg.norm(e)


class SpeechBrainEcapa:
    def __init__(self, ckpt):
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        self.fbank = Fbank(n_mels=80)
        self.model = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                                dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, wav):
        f = self.fbank(torch.from_numpy(wav.astype(np.float32)).unsqueeze(0))
        f = f - f.mean(dim=1, keepdim=True)          # sentence mean norm, std_norm=False
        e = self.model(f).squeeze().numpy()
        return e / np.linalg.norm(e)


class Resemblyzer:
    def __init__(self):
        from resemblyzer import VoiceEncoder
        self.enc = VoiceEncoder("cpu", verbose=False)

    def __call__(self, wav):
        e = self.enc.embed_utterance(wav.astype(np.float32))
        return e / np.linalg.norm(e)


def n_params_onnx(path):
    import onnx  # noqa: F401  (optional)
    return None


# ---------------------------------------------------------------- eval
def eer(scores, labels):
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    tp = np.cumsum(l); fp = np.cumsum(1 - l)
    fnr = 1 - tp / l.sum(); fpr = fp / (1 - l).sum()
    i = np.argmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2 * 100)


def main():
    root, mdir, out_json = sys.argv[1:4]
    spk = load_corpus(root)
    names = sorted(spk)
    audio = {f: sf.read(f, dtype="float32")[0] for s in names for f in spk[s]}
    # trial design: 15 enrollments per speaker (600), each vs 1 target + 2 non-target clean 4 s tests
    plan = []
    for s in names:
        for _ in range(11):
            ft = rng.choice(spk[s])
            fe = [f for f in spk[s] if f != ft]
            imp = rng.choice([x for x in names if x != s], 2, replace=False)
            fi = [rng.choice(spk[x]) for x in imp]
            oth_spk = rng.choice([x for x in names if x != s and x not in imp], 7, replace=False)
            oth = [audio[rng.choice(spk[x])] for x in oth_spk]
            plan.append((s, fe, ft, fi, oth, int(rng.integers(1 << 30))))
    conds = ["clean_3s", "clean_5s", "snr10_3s", "snr5_3s", "snr0_3s", "snr5_5s", "talker_snr10_3s", "rev_snr5_3s"]
    encoders = {
        "wespeaker_resnet34": lambda: WeSpeakerOnnx(f"{mdir}/wespeaker/voxceleb_resnet34.onnx"),
        "wespeaker_resnet34_LM": lambda: WeSpeakerOnnx(f"{mdir}/wespeaker/voxceleb_resnet34_LM.onnx"),
        "wespeaker_campplus": lambda: WeSpeakerOnnx(f"{mdir}/wespeaker/voxceleb_CAM++.onnx"),
        "wespeaker_ecapa512": lambda: WeSpeakerOnnx(f"{mdir}/wespeaker/voxceleb_ECAPA512.onnx"),
        "speechbrain_ecapa1024": lambda: SpeechBrainEcapa(f"{mdir}/speechbrain_ecapa/embedding_model.ckpt"),
        "resemblyzer_ge2e": lambda: Resemblyzer(),
    }
    res = {"n_enroll": len(plan), "trials_per_cond": len(plan) * 3, "conditions": conds, "eer_pct": {}, "cos_to_clean": {}, "ms_per_5s_embed": {}}
    for name, mk in encoders.items():
        enc = mk()
        t0 = time.perf_counter(); enc(np.zeros(5 * SR, np.float32) + 1e-3 * rng.standard_normal(5 * SR).astype(np.float32))
        for _ in range(5):
            enc(rng.standard_normal(5 * SR).astype(np.float32) * 0.05)
        res["ms_per_5s_embed"][name] = round((time.perf_counter() - t0) / 6 * 1000, 1)
        test_cache = {}

        def test_emb(f, seed):
            if (f, seed) not in test_cache:
                test_cache[(f, seed)] = enc(crop(audio[f], 4 * SR, np.random.default_rng(seed)))
            return test_cache[(f, seed)]
        res["eer_pct"][name], res["cos_to_clean"][name] = {}, {}
        clean_ref = {}
        for cond in conds:
            dur = 5 if cond.endswith("5s") else 3
            scores, labels, cosd = [], [], []
            for (s, fe, ft, fi, oth, seed) in plan:
                r = np.random.default_rng(seed)
                x = concat_to(fe, audio, dur * SR, r)
                key = (seed, dur)
                if key not in clean_ref:
                    clean_ref[key] = enc(x)
                y = corrupt(x, cond, oth, np.random.default_rng(seed + 1))
                e = enc(y.astype(np.float32))
                cosd.append(float(e @ clean_ref[key]))
                scores.append(float(e @ test_emb(ft, seed))); labels.append(1)
                for f in fi:
                    scores.append(float(e @ test_emb(f, seed))); labels.append(0)
            res["eer_pct"][name][cond] = round(eer(np.array(scores), np.array(labels)), 2)
            res["cos_to_clean"][name][cond] = round(float(np.mean(cosd)), 3)
        print(name, res["ms_per_5s_embed"][name], res["eer_pct"][name], res["cos_to_clean"][name], flush=True)
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
