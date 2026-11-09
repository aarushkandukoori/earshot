"""Tests for the WeSpeaker port (ml/earshot/wespeaker.py) and its Core ML package (10-ml 14, contract 6).

Run from the repo root:
    .venv/bin/python -m pytest ml/tests/test_wespeaker.py -q

What is checked
  - strict load of avg_model.pt (every network weight present, nothing unknown), frozen and in eval mode
  - output shape [B, 256], unit length, the 1 s and 10 s length limits, batch = one-by-one
  - the mel matrix equals the plain Kaldi formula
  - the folded Conv1d fbank equals torchaudio.compliance.kaldi.fbank and a straightforward per-frame numpy fbank
  - MPS (training device) equals CPU
  - the Core ML package matches PyTorch (cosine >= 0.9999) and has the contract 6 input, output and precision
  - the self-test inputs file holds the clip and its PyTorch embedding
Core ML and self-test checks are skipped with a message when ml/tools/convert_wespeaker.py has not been run yet.
"""

import json
import math

import numpy as np
import pytest
import torch
import torchaudio.compliance.kaldi as ta_kaldi

from ml.earshot import wespeaker as W
from ml.tools import convert_wespeaker as C

torch.set_num_threads(4)

COSINE_MIN = 0.9999


# ----------------------------------------------------------------------------------------------------------
# Fixtures (built once for the whole file)
# ----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    return W.WeSpeaker.load()


@pytest.fixture(scope="module")
def clips():
    """5 VCTK test clips, one per length 1, 3, 5, 8, 10 s (different speakers)."""
    return C.check_clips(seconds_list=[1, 3, 5, 8, 10])


def embed(model, audio):
    with torch.no_grad():
        return model(torch.from_numpy(audio).unsqueeze(0))[0].numpy()


# ----------------------------------------------------------------------------------------------------------
# Plain reference implementations (slow loops, float64; only used here)
# ----------------------------------------------------------------------------------------------------------
def mel_formula(n_mels=80, nfft=512, sr=16000, low_hz=20.0):
    """Kaldi's triangular mel filters from the textbook formula, float64 [80, 257] (last column 0)."""
    def mel(hz):
        return 1127.0 * math.log(1.0 + hz / 700.0)

    mel_low, mel_high = mel(low_hz), mel(sr / 2.0)
    step = (mel_high - mel_low) / (n_mels + 1)
    banks = np.zeros((n_mels, nfft // 2 + 1))
    for m in range(n_mels):
        left, center, right = mel_low + m * step, mel_low + (m + 1) * step, mel_low + (m + 2) * step
        for k in range(nfft // 2):
            f = mel(k * sr / nfft)
            banks[m, k] = max(0.0, min((f - left) / (center - left), (right - f) / (right - center)))
    return banks


def reference_fbank(audio, mel):
    """Kaldi fbank done the obvious way, one frame at a time, in float64.

    For each 400-sample frame every 160 samples: subtract the mean, pre-emphasis, Hamming window, zero pad to 512,
    FFT, power, mel filters, log with the 1.19e-7 floor. Then subtract each band's mean over time.
    """
    x = audio.astype(np.float64) * 32768.0
    n_frames = 1 + (len(x) - 400) // 160
    hamming = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(400) / 399)
    out = np.zeros((n_frames, mel.shape[0]))
    for t in range(n_frames):
        frame = x[160 * t: 160 * t + 400].copy()
        frame = frame - frame.mean()
        emph = np.zeros(400)
        emph[0] = frame[0] - 0.97 * frame[0]
        for i in range(1, 400):
            emph[i] = frame[i] - 0.97 * frame[i - 1]
        spectrum = np.fft.rfft(emph * hamming, n=512)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        out[t] = np.log(np.maximum(mel @ power, 1.1920929e-07))
    return out - out.mean(axis=0, keepdims=True)


def torchaudio_fbank(audio, dtype):
    """torchaudio's Kaldi fbank with WeSpeaker's settings (dither 0 at inference), then mean normalization."""
    w = torch.from_numpy(audio).to(dtype).unsqueeze(0) * 32768.0
    fb = ta_kaldi.fbank(w, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                        sample_frequency=16000, window_type="hamming", use_energy=False)
    return fb - fb.mean(dim=0, keepdim=True)


# ----------------------------------------------------------------------------------------------------------
# PyTorch port
# ----------------------------------------------------------------------------------------------------------
def test_strict_load(model):
    state = torch.load(str(W.DEFAULT_WEIGHTS), map_location="cpu", weights_only=True)
    expected = sorted(k for k in state if not k.startswith("projection"))
    assert sorted(model.net.state_dict().keys()) == expected          # same names, nothing missing or extra
    for name in expected:
        if name.endswith("num_batches_tracked"):
            # a training-step counter, unused in eval mode; avg_model.pt stores it as an averaged float
            # (e.g. 619921.625) and PyTorch keeps it as an integer, so it is not compared
            continue
        assert torch.equal(model.net.state_dict()[name], state[name]), name
    assert sum(p.numel() for p in model.net.parameters()) == 6_634_336   # 6.63 M (contract 6)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    # strictness: a file missing one weight must be refused
    partial = {k: state[k] for k in expected[1:]}
    with pytest.raises(RuntimeError):
        W.ResNet34().load_state_dict(partial, strict=True)


def test_unit_length_shapes_and_limits(model, clips):
    three = clips[1]["audio"]                                         # 3 s clip
    five = clips[2]["audio"][: len(three)]
    batch = torch.from_numpy(np.stack([three, five]))
    with torch.no_grad():
        out = model(batch).numpy()
    assert out.shape == (2, 256) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    # a batch gives the same vectors as one clip at a time
    assert np.abs(out[0] - embed(model, three)).max() < 1e-5
    assert np.abs(out[1] - embed(model, five)).max() < 1e-5
    # the shortest and longest allowed inputs work; one sample outside the range is refused
    long_audio = clips[4]["audio"]                                    # exactly 10 s = 160,000 samples
    assert len(long_audio) == W.MAX_SAMPLES
    assert abs(np.linalg.norm(embed(model, long_audio)) - 1.0) < 1e-5
    assert abs(np.linalg.norm(embed(model, long_audio[: W.MIN_SAMPLES])) - 1.0) < 1e-5
    with pytest.raises(ValueError):
        embed(model, long_audio[: W.MIN_SAMPLES - 1])
    with pytest.raises(ValueError):
        embed(model, np.concatenate([long_audio, long_audio[:1]]))


def test_mel_banks_match_kaldi_formula():
    mine = W.kaldi_mel_banks().double().numpy()
    formula = mel_formula()
    assert mine.shape == (80, 257)
    assert np.all(mine[:, 256] == 0.0)                                # Nyquist bin unused, as in Kaldi
    # torchaudio builds the triangles in float32: measured max difference 1.4e-5 per weight
    assert np.abs(mine - formula).max() < 5e-5


def test_fbank_matches_torchaudio_kaldi(clips):
    """The folded Conv1d front end equals torchaudio.compliance.kaldi.fbank.

    Math check in float64 (a float64 copy of KaldiFbank vs torchaudio in float64): within 1e-4 (10-ml 14; measured
    about 2e-11). The shipped float32 module vs the exact float64 fbank: within 5e-4 max and 1e-5 mean. The float32
    gap comes from quiet low mel bands (pre-emphasis leaves little energy there, so rounding shows after the log):
    measured 1.8e-4 for this module, while torchaudio's own float32 fbank is 1.1e-3 away from its float64 one on the
    same VCTK clips, so a float32-vs-float32 comparison at 1e-4 would only measure torchaudio's rounding.
    """
    front32 = W.KaldiFbank()
    front64 = W.KaldiFbank(torch.float64)
    for clip in clips[:3]:                                            # 1, 3 and 5 s
        audio = clip["audio"]
        exact = torchaudio_fbank(audio, torch.float64)
        with torch.no_grad():
            mine64 = front64(torch.from_numpy(audio).double().unsqueeze(0))[0]
            mine32 = front32(torch.from_numpy(audio).unsqueeze(0))[0].double()
        assert mine64.shape == exact.shape == (1 + (len(audio) - 400) // 160, 80)
        assert (mine64 - exact).abs().max().item() < 1e-4, clip["name"]
        err32 = (mine32 - exact).abs()
        assert err32.max().item() < 5e-4, clip["name"]
        assert err32.mean().item() < 1e-5, clip["name"]


def test_fbank_matches_plain_reference(clips):
    """The Conv1d trick (DC removal, pre-emphasis, window and DFT folded into one kernel) equals doing those steps
    one by one on each frame."""
    audio = clips[1]["audio"]                                         # 3 s
    mel = W.kaldi_mel_banks().double().numpy()
    ref = reference_fbank(audio, mel)
    with torch.no_grad():
        mine64 = W.KaldiFbank(torch.float64)(torch.from_numpy(audio).double().unsqueeze(0))[0].numpy()
    assert mine64.shape == ref.shape
    assert np.abs(mine64 - ref).max() < 1e-6


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps_matches_cpu(model, clips):
    """Training runs the frozen encoder on MPS; it must give the same vectors as the CPU."""
    mps_model = W.WeSpeaker.load().to("mps")
    for clip in clips[1:3]:
        x = torch.from_numpy(clip["audio"]).unsqueeze(0)
        with torch.no_grad():
            e_mps = mps_model(x.to("mps"))[0].cpu().numpy()
        assert W.cosine(e_mps, embed(model, clip["audio"])) >= COSINE_MIN


# ----------------------------------------------------------------------------------------------------------
# Core ML package (built by: .venv/bin/python -m ml.tools.convert_wespeaker)
# ----------------------------------------------------------------------------------------------------------
def require_package():
    if not C.PACKAGE.exists():
        pytest.skip(f"{C.PACKAGE} not built yet; run .venv/bin/python -m ml.tools.convert_wespeaker")


@pytest.mark.parametrize("unit", ["CPU_AND_GPU", "CPU_ONLY"])
def test_coreml_matches_pytorch(model, clips, unit):
    require_package()
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(C.PACKAGE), compute_units=getattr(ct.ComputeUnit, unit))
    for clip in clips:                                                # 1, 3, 5, 8 and 10 s
        e_ml = C.coreml_embed(mlmodel, clip["audio"])
        assert e_ml.shape == (256,)
        assert abs(np.linalg.norm(e_ml) - 1.0) < 1e-4
        cos = W.cosine(e_ml, embed(model, clip["audio"]))
        assert cos >= COSINE_MIN, f"{unit} {clip['name']}: cosine {cos}"


def test_coreml_package_contract():
    """Contract 6: input "wav" float32 [1, N] with N 16,000..160,000; output "embedding" float32 [1, 256];
    ML program for iOS 17 (spec version 8); fp32 everywhere (no float16 tensor anywhere in the program)."""
    require_package()
    import coremltools as ct

    spec = ct.models.MLModel(str(C.PACKAGE), skip_model_load=True).get_spec()
    float32 = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    assert spec.WhichOneof("Type") == "mlProgram"
    assert spec.specificationVersion == 8                             # iOS 17 minimum deployment target

    assert [i.name for i in spec.description.input] == ["wav"]
    wav = spec.description.input[0].type.multiArrayType
    assert wav.dataType == float32
    assert list(wav.shape) == [1, 80000]                              # default length 5 s
    ranges = [(r.lowerBound, r.upperBound) for r in wav.shapeRange.sizeRanges]
    assert ranges == [(1, 1), (W.MIN_SAMPLES, W.MAX_SAMPLES)]

    assert [o.name for o in spec.description.output] == ["embedding"]
    emb = spec.description.output[0].type.multiArrayType
    assert emb.dataType == float32
    assert list(emb.shape) == [1, 256]

    assert "FLOAT16" not in str(spec.mlProgram)


# ----------------------------------------------------------------------------------------------------------
# Self-test inputs (ml/exports/wespeaker_selftest_inputs.npz, read by export.py for wespk_selftest.est)
# ----------------------------------------------------------------------------------------------------------
def test_selftest_inputs(model):
    if not C.SELFTEST_NPZ.exists():
        pytest.skip(f"{C.SELFTEST_NPZ} not written yet; run .venv/bin/python -m ml.tools.convert_wespeaker")
    data = np.load(C.SELFTEST_NPZ)                                    # no pickle needed
    clip, emb = data["clip"], data["embedding"]
    meta = json.loads(str(data["meta_json"]))
    assert clip.dtype == np.float32 and clip.shape == (80000,)        # 5 s (contract 6)
    assert emb.dtype == np.float32 and emb.shape == (256,)
    assert meta["kind"] == "wespeaker-selftest"
    assert meta["tolerances"] == {"cosine_min": 0.999}
    assert meta["sources"][0]["license"] == "CC-BY-4.0"
    level_db = 10.0 * np.log10(np.mean(clip.astype(np.float64) ** 2))
    assert abs(level_db - (-26.0)) < 0.01 and np.max(np.abs(clip)) < 1.0
    # the stored embedding is the PyTorch CPU fp32 embedding of the stored clip
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5
    assert np.abs(embed(model, clip) - emb).max() < 1e-5
    # and it is the clip the tool picks (so re-running the tool reproduces the file)
    expected_clip, _ = C.selftest_clip()
    assert np.array_equal(expected_clip, clip)
