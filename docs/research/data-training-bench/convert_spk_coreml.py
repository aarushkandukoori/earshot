"""Can a speaker encoder with its filterbank front-end inside the graph be converted to Core ML?

Usage: python -I convert_spk_coreml.py <models_dir> <librispeech_test_clean_dir> <out_dir> [resnet34|resnet34_LM]
1. WeSpeaker ResNet34-LM + a Kaldi-compatible fbank written as one Conv1d (DFT basis) -> Core ML.
   Checks: fbank vs torchaudio.compliance.kaldi.fbank, embedding vs the official ONNX, Core ML vs PyTorch.
2. SpeechBrain ECAPA with its stock torch.stft-based Fbank -> Core ML (does it convert as-is?).
"""
import glob
import json
import sys
import time
import traceback

import coremltools as ct
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi

SR = 16000


class KaldiFbank(nn.Module):
    """Kaldi fbank (25 ms hamming, 10 ms hop, preemph 0.97, DC removal, 512-pt FFT, 80 mel, log) + CMN.
    DC removal, pre-emphasis, window and DFT are all linear, so they fold into a single Conv1d kernel."""

    def __init__(self, n_mels=80, win=400, hop=160, nfft=512, preemph=0.97):
        super().__init__()
        n = torch.arange(win, dtype=torch.float64)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / (win - 1))
        eye = torch.eye(win, dtype=torch.float64)
        dc = eye - 1.0 / win
        pre = eye.clone()
        pre[1:, :-1] -= preemph * torch.eye(win - 1, dtype=torch.float64)
        pre[0, 0] = 1 - preemph
        lin = torch.diag(window) @ pre @ dc
        k = torch.arange(nfft // 2 + 1, dtype=torch.float64)[:, None]
        ang = 2 * torch.pi * k * n[None, :] / nfft
        kern = torch.cat([torch.cos(ang) @ lin, -torch.sin(ang) @ lin], 0)
        self.register_buffer("kern", kern.float().unsqueeze(1))
        mel = kaldi.get_mel_banks(n_mels, nfft, float(SR), 20.0, 0.0, 100.0, -500.0, 1.0)[0]
        self.register_buffer("mel", F.pad(mel, (0, 1)).float())
        self.hop, self.nb = hop, nfft // 2 + 1

    def forward(self, wav):                       # (B, N) in [-1, 1]
        spec = F.conv1d((wav * 32768.0).unsqueeze(1), self.kern, stride=self.hop)
        re, im = spec[:, : self.nb], spec[:, self.nb:]
        fb = torch.matmul(self.mel, re * re + im * im)          # (B, 80, T)
        fb = torch.log(torch.clamp(fb, min=1.1920929e-07))
        fb = fb - fb.mean(dim=2, keepdim=True)
        return fb.transpose(1, 2)                                # (B, T, 80)


class BasicBlock(nn.Module):
    def __init__(self, cin, c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(c)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != c:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, c, 1, stride, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet34(nn.Module):
    """WeSpeaker ResNet34-TSTP-emb256 (same parameter names as wespeaker/models/resnet.py)."""

    def __init__(self, m=32, feat_dim=80, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, m, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(m)
        cin, layers = m, []
        for i, (c, nblk, st) in enumerate([(m, 3, 1), (2 * m, 4, 2), (4 * m, 6, 2), (8 * m, 3, 2)]):
            blocks = []
            for j in range(nblk):
                blocks.append(BasicBlock(cin, c, st if j == 0 else 1)); cin = c
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))
        self.seg_1 = nn.Linear(int(feat_dim / 8) * m * 8 * 2, embed_dim)

    def forward(self, feats):                     # (B, T, 80)
        x = feats.permute(0, 2, 1).unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        mean = x.mean(dim=-1).flatten(1)
        std = torch.sqrt(torch.var(x, dim=-1) + 1e-7).flatten(1)
        return self.seg_1(torch.cat([mean, std], 1))


class Embedder(nn.Module):
    def __init__(self, front, net):
        super().__init__(); self.front, self.net = front, net

    def forward(self, wav):
        e = self.net(self.front(wav))
        return e / torch.linalg.vector_norm(e, dim=1, keepdim=True)


def cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    mdir, ls_dir, out_dir = sys.argv[1:4]
    variant = sys.argv[4] if len(sys.argv) > 4 else "resnet34"
    res = {"variant": variant}
    wavs = [sf.read(f, dtype="float32")[0][: 5 * SR] for f in sorted(glob.glob(f"{ls_dir}/**/*.flac", recursive=True))[:20]]
    wavs = [w for w in wavs if len(w) == 5 * SR][:10]
    # ---- 1. WeSpeaker ResNet34-LM
    front = KaldiFbank().eval()
    w = torch.from_numpy(wavs[0]).unsqueeze(0)
    ref = kaldi.fbank(w * 32768, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                      sample_frequency=SR, window_type="hamming", use_energy=False)
    ref = ref - ref.mean(0, keepdim=True)
    mine = front(w)[0]
    res["fbank_max_abs_err_vs_torchaudio_kaldi"] = float((mine - ref).abs().max())
    net = ResNet34()
    sd = torch.load(f"{mdir}/wespeaker_pt/{variant}_avg_model.pt", map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict({k: v for k, v in sd.items() if not k.startswith("projection")}, strict=False)
    res["state_dict_missing"], res["state_dict_unexpected"] = list(missing), list(unexpected)
    model = Embedder(front, net.eval()).eval()
    res["params_M"] = round(sum(p.numel() for p in net.parameters()) / 1e6, 2)
    onnx = ort.InferenceSession(f"{mdir}/wespeaker/voxceleb_{variant}.onnx", providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt = [model(torch.from_numpy(x).unsqueeze(0)).numpy()[0] for x in wavs]
    on = [onnx.run(None, {"feats": front(torch.from_numpy(x).unsqueeze(0)).numpy()})[0][0] for x in wavs]
    res["cos_pytorch_vs_official_onnx_min"] = min(cos(a, b) for a, b in zip(pt, on))
    traced = torch.jit.trace(model, torch.from_numpy(wavs[0]).unsqueeze(0))
    for prec_name, prec in (("fp32", ct.precision.FLOAT32), ("fp16", ct.precision.FLOAT16)):
        try:
            t = time.perf_counter()
            ml = ct.convert(traced, inputs=[ct.TensorType(name="wav", shape=(1, ct.RangeDim(16000, 16 * SR, default=5 * SR)))],
                            outputs=[ct.TensorType(name="embedding")], convert_to="mlprogram",
                            minimum_deployment_target=ct.target.iOS17, compute_precision=prec)
            conv_s = time.perf_counter() - t
            ml.save(f"{out_dir}/wespeaker_{variant}_{prec_name}.mlpackage")
            outs, lat = [], []
            for x in wavs:
                t = time.perf_counter(); o = ml.predict({"wav": x[None, :]})["embedding"][0]; lat.append(time.perf_counter() - t)
                outs.append(o)
            o3 = ml.predict({"wav": wavs[1][None, : 3 * SR]})["embedding"][0]
            with torch.no_grad():
                p3 = model(torch.from_numpy(wavs[1][: 3 * SR]).unsqueeze(0)).numpy()[0]
            res[f"coreml_{prec_name}"] = {"convert_s": round(conv_s, 1), "cos_vs_pytorch_min": min(cos(a, b) for a, b in zip(outs, pt)),
                                          "cos_vs_pytorch_3s_input": cos(o3, p3), "mac_predict_ms_median": round(1000 * float(np.median(lat[1:])), 1)}
        except Exception as e:  # noqa: BLE001
            res[f"coreml_{prec_name}"] = f"FAIL {type(e).__name__}: {str(e)[:300]}"
            traceback.print_exc()
    # ---- 2. SpeechBrain ECAPA with stock torch.stft Fbank
    try:
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

        class SB(nn.Module):
            def __init__(self):
                super().__init__()
                self.fb = Fbank(n_mels=80)
                self.net = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                                      dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192)

            def forward(self, wav):
                f = self.fb(wav); f = f - f.mean(dim=1, keepdim=True)
                return self.net(f).squeeze(1)
        sb = SB().eval()
        sb.net.load_state_dict(torch.load(f"{mdir}/speechbrain_ecapa/embedding_model.ckpt", map_location="cpu", weights_only=True))
        res["speechbrain_params_M"] = round(sum(p.numel() for p in sb.net.parameters()) / 1e6, 2)
        tr = torch.jit.trace(sb, torch.from_numpy(wavs[0]).unsqueeze(0))
        ml = ct.convert(tr, inputs=[ct.TensorType(name="wav", shape=(1, 5 * SR))], convert_to="mlprogram",
                        minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT32)
        o = list(ml.predict({"wav": wavs[2][None, :]}).values())[0].ravel()
        with torch.no_grad():
            p = sb(torch.from_numpy(wavs[2]).unsqueeze(0)).numpy().ravel()
        res["speechbrain_ecapa_stock_fbank_coreml"] = {"converted": True, "cos_vs_pytorch": cos(o, p)}
    except Exception as e:  # noqa: BLE001
        res["speechbrain_ecapa_stock_fbank_coreml"] = f"FAIL {type(e).__name__}: {str(e)[:400]}"
        traceback.print_exc()
    print(json.dumps(res, indent=1, default=str))
    with open(f"{out_dir}/coreml_results_{variant}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
