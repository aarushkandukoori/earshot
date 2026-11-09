"""WeSpeaker ResNet34 speaker encoder: the one definition used everywhere (contract 6, 10-ml 6.1).

What it does, in one line: raw 16 kHz audio [B, N] in [-1, 1] -> one 256-number voice fingerprint per clip,
scaled to unit length (so the cosine of two fingerprints is just their dot product).

The same module is used in three places, so all three compute exactly the same function:
  1. training (frozen, eval mode, fp32, no grad, on MPS) to compute the target vector e_t,
  2. the own-voice cache tool and the export tools (CPU fp32),
  3. ml/tools/convert_wespeaker.py, which turns it into the Core ML package the iPhone runs.

Pipeline (each step is a class below):
  KaldiFbank  - Kaldi-style log mel filterbank (80 mel bands, 25 ms frames every 10 ms), then mean
                normalization over time. DC removal, pre-emphasis, the Hamming window and the DFT are all
                linear, so they are folded into ONE fixed Conv1d (514 output channels x 400 taps, stride 160).
                That keeps the front end a plain convolution, which Core ML converts without custom ops.
  ResNet34    - the WeSpeaker network (channel base m = 32), stats pooling (mean and std over time), and a
                Linear layer to 256 numbers. Parameter names match WeSpeaker's checkpoint exactly.
  WeSpeaker   - front end + network + L2 normalization. WeSpeaker.load() reads avg_model.pt.

Ported unchanged in its math from docs/research/data-training-bench/convert_spk_coreml.py (that code matched
torchaudio's kaldi fbank within 8.1e-5 and the official ONNX model at cosine 1.0, measured,
docs/research/data-training.md 4.3). The mel filter matrix comes from torchaudio's Kaldi helper, the same matrix
WeSpeaker used in training; ml/tests/test_wespeaker.py rebuilds it from the plain Kaldi formula as a check.

License note: the weights are VoxCeleb research-terms weights, so anything built with them is personal use only.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi

# ----------------------------------------------------------------------------------------------------------
# Fixed numbers (contract 6)
# ----------------------------------------------------------------------------------------------------------
SR = 16000                  # sample rate (Hz)
WIN = 400                   # frame length: 25 ms at 16 kHz
HOP = 160                   # frame shift: 10 ms at 16 kHz
NFFT = 512                  # DFT size (400 samples zero padded to 512, like Kaldi's round_to_power_of_two)
NBINS = NFFT // 2 + 1       # 257 frequency bins from 0 Hz to 8 kHz
N_MELS = 80                 # mel bands
PREEMPH = 0.97              # pre-emphasis coefficient
LOW_HZ = 20.0               # lowest mel band edge (Kaldi default)
LOG_FLOOR = 1.1920929e-07   # float32 machine epsilon; Kaldi clamps the mel energy to this before the log
EMBED_DIM = 256             # output size
MIN_SAMPLES = 16000         # 1 s, shortest input allowed (contract 6)
MAX_SAMPLES = 160000        # 10 s, longest input allowed (contract 6)

# repo root = two folders above this file (ml/earshot/wespeaker.py -> repo)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "wespeaker-voxceleb-resnet34" / "avg_model.pt"


# ----------------------------------------------------------------------------------------------------------
# Front end helpers (run once when the module is built)
# ----------------------------------------------------------------------------------------------------------
def kaldi_mel_banks():
    """Kaldi's 80 triangular mel filters as a float32 tensor [80, 257].

    Each row is one triangle on the mel scale (mel = 1127 ln(1 + f / 700)), with 80 + 2 equally spaced edges from
    20 Hz to 8 kHz. torchaudio's helper returns them for bins 0..255 only (Kaldi leaves out the 8 kHz Nyquist bin),
    so one zero column is appended to make the matrix fit the full 257-bin spectrum.
    Arguments: 80 bins, 512-point DFT, 16 kHz, low 20 Hz, high 0 (= Nyquist), then three VTLN settings
    (100, -500, warp 1.0) that mean "no vocal tract length warping".
    We use torchaudio's own matrix (computed in float32) because it is exactly what WeSpeaker was trained with;
    a float64 rebuild differs by up to 1.4e-5 per weight, which moves quiet low bands by about 1.6e-4 in log
    units (measured on VCTK, 2026-11-07).
    """
    banks, _ = ta_kaldi.get_mel_banks(N_MELS, NFFT, float(SR), LOW_HZ, 0.0, 100.0, -500.0, 1.0)  # [80, 256]
    return F.pad(banks.float(), (0, 1))                                                      # [80, 257]


def fbank_kernel(win=WIN, nfft=NFFT, preemph=PREEMPH):
    """The fixed Conv1d kernel [2 * NBINS, 1, win] that does DC removal, pre-emphasis, Hamming and the DFT.

    Every one of those steps is a matrix acting on a 400-sample frame x (a column vector):
      dc   - subtract the frame mean:          dc = I - (1 / win) * ones
      pre  - y[i] = x[i] - 0.97 x[i - 1], and y[0] = x[0] - 0.97 x[0] (Kaldi repeats the first sample)
      win  - multiply by the Hamming window:  diag(0.54 - 0.46 cos(2 pi n / (win - 1)))
      dft  - real part cos(2 pi k n / 512), imaginary part -sin(2 pi k n / 512), k = 0..256
             (the zero padding from 400 to 512 samples just means the DFT sums stop at n = 399)
    Applied in that order, the product dft @ win @ pre @ dc is one [514, 400] matrix. Each row is one Conv1d
    filter: rows 0..256 give the real parts, rows 257..513 the imaginary parts.
    """
    n = np.arange(win, dtype=np.float64)
    eye = np.eye(win, dtype=np.float64)

    dc = eye - 1.0 / win

    pre = eye.copy()
    for i in range(1, win):
        pre[i, i - 1] = -preemph                # y[i] = x[i] - 0.97 x[i - 1]
    pre[0, 0] = 1.0 - preemph                   # y[0] = x[0] - 0.97 x[0]

    hamming = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (win - 1))
    window = np.diag(hamming)

    linear = window @ pre @ dc                  # [400, 400]: everything before the DFT

    k = np.arange(nfft // 2 + 1, dtype=np.float64)[:, None]   # [257, 1] frequency bin numbers
    angle = 2.0 * np.pi * k * n[None, :] / nfft               # [257, 400]
    real_rows = np.cos(angle) @ linear
    imag_rows = -np.sin(angle) @ linear
    kernel = np.concatenate([real_rows, imag_rows], axis=0)   # [514, 400]
    return kernel[:, None, :]                                 # Conv1d wants [out_channels, in_channels, taps]


# ----------------------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------------------
class KaldiFbank(nn.Module):
    """Raw audio [B, N] in [-1, 1] -> mean-normalized log mel filterbank [B, T, 80].

    T = 1 + (N - 400) // 160 frames (Kaldi's snip_edges: only whole frames, no padding).
    Matches torchaudio.compliance.kaldi.fbank(x * 32768, num_mel_bins=80, dither=0, window_type="hamming")
    followed by subtracting the per-band mean over time.
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        # Fixed buffers, never trained. persistent=False keeps them out of state_dict (they are rebuilt from the
        # formulas above), but .to(device) still moves them to MPS or CPU together with the module.
        # dtype is always float32 in use; the tests build a float64 copy to check the formulas without rounding.
        kernel = torch.from_numpy(fbank_kernel()).to(dtype)               # [514, 1, 400]
        mel = kaldi_mel_banks().to(dtype)                                 # [80, 257]
        self.register_buffer("kern", kernel, persistent=False)
        self.register_buffer("mel", mel, persistent=False)

    def forward(self, wav):
        # WeSpeaker was trained on int16-scaled audio, so scale [-1, 1] up to [-32768, 32768] first.
        x = (wav * 32768.0).unsqueeze(1)                                  # [B, 1, N]
        spec = F.conv1d(x, self.kern, stride=HOP)                         # [B, 514, T]: real then imaginary parts
        real = spec[:, :NBINS]                                            # [B, 257, T]
        imag = spec[:, NBINS:]                                            # [B, 257, T]
        power = real * real + imag * imag                                 # power spectrum per frame
        fb = torch.matmul(self.mel, power)                                # [B, 80, T] mel band energies
        fb = torch.log(torch.clamp(fb, min=LOG_FLOOR))                    # log, with Kaldi's floor
        fb = fb - fb.mean(dim=2, keepdim=True)                            # mean normalization over time (CMN)
        return fb.transpose(1, 2)                                         # [B, T, 80]


class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 convs with batch norm, plus a shortcut (1x1 conv when the shape changes)."""

    def __init__(self, cin, c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.shortcut = nn.Sequential()                                   # identity when the shape is unchanged
        if stride != 1 or cin != c:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, c, 1, stride, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet34(nn.Module):
    """WeSpeaker ResNet34-TSTP-emb256 (parameter names identical to wespeaker/models/resnet.py).

    Input fbank [B, T, 80] -> output [B, 256] (not yet normalized).
    Four stages of 3, 4, 6, 3 basic blocks with 32, 64, 128, 256 channels; stages 2 to 4 halve both the
    frequency and the time axis. Then "TSTP" stats pooling: mean and standard deviation over time, which turns a
    clip of any length into one fixed-size vector, and a Linear layer to 256.
    """

    def __init__(self, m=32, feat_dim=N_MELS, embed_dim=EMBED_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(1, m, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(m)
        # (channels, number of blocks, stride of the first block) for layer1..layer4
        stages = [(m, 3, 1), (2 * m, 4, 2), (4 * m, 6, 2), (8 * m, 3, 2)]
        cin = m
        for i, (c, n_blocks, stride) in enumerate(stages):
            blocks = []
            for j in range(n_blocks):
                first_stride = stride if j == 0 else 1
                blocks.append(BasicBlock(cin, c, first_stride))
                cin = c
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))       # names layer1..layer4 as in WeSpeaker
        # after 3 halvings the 80 mel bands become 10; 256 channels x 10 bands = 2560 numbers, x2 for mean and std
        pooled = (feat_dim // 8) * (8 * m) * 2                            # 5120
        self.seg_1 = nn.Linear(pooled, embed_dim)

    def forward(self, feats):
        x = feats.permute(0, 2, 1).unsqueeze(1)                           # [B, 1, 80, T] an "image": freq x time
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                                                # [B, 256, 10, T / 8]
        mean = x.mean(dim=-1).flatten(1)                                  # [B, 2560] mean over time
        std = torch.sqrt(torch.var(x, dim=-1) + 1e-7).flatten(1)          # [B, 2560] std over time (unbiased var)
        return self.seg_1(torch.cat([mean, std], dim=1))                  # [B, 256]


class WeSpeaker(nn.Module):
    """Raw audio [B, N] in [-1, 1] (16 kHz, 16,000 <= N <= 160,000) -> unit-length embedding [B, 256].

    Build it with WeSpeaker.load(), which returns the frozen, eval-mode model on the CPU. Move it with .to("mps")
    for training. It is never trained: call it inside torch.no_grad().
    """

    def __init__(self):
        super().__init__()
        self.front = KaldiFbank()
        self.net = ResNet34()

    def forward(self, wav):
        # Shape check in plain Python. It is skipped while torch.jit.trace records the graph for Core ML, because
        # the Core ML input has its own length range (ct.RangeDim) and a Python check would be baked in as a constant.
        if not torch.jit.is_tracing():
            if wav.dim() != 2:
                raise ValueError(f"WeSpeaker expects audio shaped [B, N], got {tuple(wav.shape)}")
            n = wav.shape[1]
            if n < MIN_SAMPLES or n > MAX_SAMPLES:
                raise ValueError(f"WeSpeaker input length {n} is outside [{MIN_SAMPLES}, {MAX_SAMPLES}] samples")
        e = self.net(self.front(wav))                                     # [B, 256]
        return e / torch.linalg.vector_norm(e, dim=1, keepdim=True)       # L2 normalize: unit length

    @classmethod
    def load(cls, path=DEFAULT_WEIGHTS):
        """Load avg_model.pt (weights_only=True), drop the training-only 'projection' head, load strictly."""
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        kept = {}
        for name, tensor in state.items():
            if name.startswith("projection"):       # the speaker-classification head used only in training
                continue
            kept[name] = tensor
        model = cls()
        # strict=True raises if any network parameter is missing from the file or the file has an unknown key
        result = model.net.load_state_dict(kept, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        model.eval()
        model.requires_grad_(False)
        return model

    @torch.no_grad()
    def embed_numpy(self, audio):
        """Convenience for tools and tests: one 1-D float32 numpy clip -> numpy float32 [256], on the model's device."""
        device = self.front.kern.device
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0).to(device)
        return self(x)[0].cpu().numpy()


# The research script called the front end + network + normalization "Embedder"; same class.
Embedder = WeSpeaker


def cosine(a, b):
    """Cosine similarity of two vectors (numpy float64 math)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
