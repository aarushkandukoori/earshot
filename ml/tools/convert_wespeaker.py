"""Convert the PyTorch WeSpeaker model to Core ML, check it, and save the self-test inputs (10-ml 6.2, contract 6).

Run from the repo root:
    .venv/bin/python -m ml.tools.convert_wespeaker              # convert + check + self-test inputs
    .venv/bin/python -m ml.tools.convert_wespeaker --skip-convert   # reuse the existing package

Steps
  1. Convert - torch.jit.trace ml/earshot/wespeaker.py on a 5 s clip, then ct.convert to an fp32 ML program with
     input "wav" float32 [1, N] (N from 16,000 to 160,000, default 80,000) and output "embedding" float32 [1, 256].
     Saved to ml/exports/wespeaker_resnet34_fp32.mlpackage.
  2. Check - 10 clips from speech_test (VCTK test speakers, 1, 3, 5, 8 and 10 s, two of each length) through the
     SAVED package with CPU_AND_GPU and CPU_ONLY, compared with the PyTorch CPU fp32 embedding. Writes
     ml/exports/wespeaker_check.json (cosines and timings). Exits with code 1 if any cosine is below 0.9999.
  3. Self-test inputs - ml/exports/wespeaker_selftest_inputs.npz, which export.py (ML-0b) turns into
     wespk_selftest.est with the shared container writer. It holds:
        clip       float32 [80000]  5 s of one VCTK test utterance, scaled to -26 dBFS RMS
        embedding  float32 [256]    PyTorch CPU fp32 embedding of clip (unit length)
        meta_json  str              JSON text: {"kind": "wespeaker-selftest", "tolerances": {"cosine_min": 0.999},
                                    "sources": [...], ...} - copy "kind" and "tolerances" into meta of the .est
     Load it with np.load(path) (no pickle needed) and json.loads(str(data["meta_json"])).

Why fp32 and CPU_AND_GPU: an fp16 front end overflows (cosine 0.01 to 0.06) and the Neural Engine returned wrong
embeddings (cosine 0.57 for the mixed-precision package), measured in docs/research/data-training.md 4.3.
The package holds VoxCeleb research-terms weights: personal use only, never distributed.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import coremltools as ct

from ml.earshot.config import WESPK_SELFTEST_NPZ
from ml.earshot.wespeaker import DEFAULT_WEIGHTS, MAX_SAMPLES, MIN_SAMPLES, SR, WeSpeaker, cosine

# ----------------------------------------------------------------------------------------------------------
# Paths and fixed numbers
# ----------------------------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
SPEECH_TEST = DATA / "manifests" / "speech_test.jsonl"
EXPORTS = REPO_ROOT / "ml" / "exports"
PACKAGE = EXPORTS / "wespeaker_resnet34_fp32.mlpackage"
CHECK_JSON = EXPORTS / "wespeaker_check.json"
SELFTEST_NPZ = WESPK_SELFTEST_NPZ    # ml/exports/wespeaker_selftest_inputs.npz (one definition, config.py)

DEFAULT_SAMPLES = 80000          # 5 s: the trace example and the Core ML default length
CHECK_SECONDS = [1, 1, 3, 3, 5, 5, 8, 8, 10, 10]   # 10-ml 6.2: lengths 1, 3, 5, 8, 10 s, two each
COSINE_MIN = 0.9999              # the conversion must match PyTorch at least this well (10-ml 6.2)
SELFTEST_COSINE_MIN = 0.999      # what the phone requires at launch (contract 6 Self-test row)
SELFTEST_LEVEL_DBFS = -26.0      # level of the self-test clip (10-ml 7.4)
COMPUTE_UNITS = ["CPU_AND_GPU", "CPU_ONLY"]   # never ALL or CPU_AND_NE (contract 6)
TIMING_RUNS = 10                 # predictions per timing measurement (after one warm-up)


# ----------------------------------------------------------------------------------------------------------
# Test audio (VCTK test speakers, CC BY 4.0)
# ----------------------------------------------------------------------------------------------------------
def read_speech_test():
    """All rows of data/manifests/speech_test.jsonl, in file order."""
    rows = []
    with open(SPEECH_TEST) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_audio(row):
    """One manifest row -> float32 mono audio at 16 kHz."""
    audio, sr = sf.read(str(DATA / row["path"]), dtype="float32")
    assert sr == SR, f"{row['path']} is {sr} Hz, expected {SR}"
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def speaker_clip(rows, speaker, n_samples):
    """Concatenate one speaker's utterances in manifest order until n_samples are reached; keep the first n_samples.

    Returns (audio float32 [n_samples], list of the file paths used).
    """
    pieces = []
    paths = []
    total = 0
    for row in rows:
        if row["speaker"] != speaker:
            continue
        audio = read_audio(row)
        pieces.append(audio)
        paths.append(row["path"])
        total += len(audio)
        if total >= n_samples:
            break
    if total < n_samples:
        raise ValueError(f"speaker {speaker} has only {total} samples, need {n_samples}")
    return np.concatenate(pieces)[:n_samples].astype(np.float32), paths


def check_clips(rows=None, seconds_list=CHECK_SECONDS):
    """The conversion check clips: clip i has length seconds_list[i] and uses speaker number i (mod 8).

    Neighbouring clips (the two clips of one length) therefore come from different speakers.
    Returns a list of dicts {"name", "seconds", "speaker", "paths", "audio"}.
    """
    if rows is None:
        rows = read_speech_test()
    speakers = sorted(set(row["speaker"] for row in rows))
    clips = []
    for i, seconds in enumerate(seconds_list):
        speaker = speakers[i % len(speakers)]
        audio, paths = speaker_clip(rows, speaker, int(seconds * SR))
        clips.append({"name": f"clip{i}_{seconds}s", "seconds": seconds, "speaker": speaker,
                      "paths": paths, "audio": audio})
    return clips


def selftest_clip(rows=None):
    """The phone's launch self-test clip: 5 s from the middle of the first speech_test utterance (in manifest order)
    that is at least 5.5 s long, scaled so its RMS over the 5 s is -26 dBFS.

    Returns (clip float32 [80000], info dict for meta_json).
    """
    if rows is None:
        rows = read_speech_test()
    for row in rows:
        if row["duration"] >= 5.5:
            audio = read_audio(row)
            start = (len(audio) - DEFAULT_SAMPLES) // 2            # centered crop skips the silent edges
            piece = audio[start:start + DEFAULT_SAMPLES].astype(np.float64)
            rms = np.sqrt(np.mean(piece ** 2))
            gain = 10.0 ** (SELFTEST_LEVEL_DBFS / 20.0) / rms
            clip = (piece * gain).astype(np.float32)
            assert np.max(np.abs(clip)) < 1.0, "self-test clip would clip after scaling"
            info = {"path": row["path"], "license": row["license"], "speaker": row["speaker"],
                    "start_sample": int(start), "num_samples": DEFAULT_SAMPLES, "gain": float(gain),
                    "level": "RMS over the 5 s clip = -26 dBFS"}
            return clip, info
    raise ValueError("no speech_test utterance of at least 5.5 s")


# ----------------------------------------------------------------------------------------------------------
# Conversion and Core ML helpers
# ----------------------------------------------------------------------------------------------------------
def convert(model, example):
    """Trace the PyTorch model on one 5 s example [1, 80000] and convert it to an fp32 Core ML ML program."""
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    wav_shape = (1, ct.RangeDim(MIN_SAMPLES, MAX_SAMPLES, default=DEFAULT_SAMPLES))   # N is flexible
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="wav", shape=wav_shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,       # fp32 everywhere (an fp16 front end is broken)
    )
    # Human-readable descriptions, shown in Xcode's model viewer.
    mlmodel.short_description = ("WeSpeaker ResNet34 (VoxCeleb) speaker embedding with a Kaldi fbank front end; "
                                 "Earshot enrollment checks")
    mlmodel.license = "VoxCeleb research-terms weights: personal use only, never distributed"
    mlmodel.author = "Earshot WP-ML (ml/earshot/wespeaker.py)"
    mlmodel.input_description["wav"] = "raw 16 kHz mono audio in [-1, 1], shape [1, N], N 16000..160000"
    mlmodel.output_description["embedding"] = "unit-length speaker embedding, shape [1, 256]"
    mlmodel.user_defined_metadata["compute_units"] = "cpuAndGPU in the foreground, cpuOnly in the background; never all"
    return mlmodel


def coreml_embed(mlmodel, audio):
    """One 1-D float32 clip -> Core ML embedding as a numpy float32 [256]."""
    out = mlmodel.predict({"wav": audio[None, :].astype(np.float32)})
    return np.asarray(out["embedding"], dtype=np.float32).reshape(-1)


def median_ms(fn, runs=TIMING_RUNS):
    """Call fn() once to warm up, then `runs` more times; return the median wall time in milliseconds."""
    fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(1000.0 * (time.perf_counter() - t0))
    return float(np.median(times))


def folder_mb(path):
    """Total size of all files under a folder (an .mlpackage is a folder), in MB."""
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1e6


def file_sha256(path):
    """Lowercase hex SHA-256 of a file, read in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def pytorch_timings(clip):
    """Median ms per 5 s clip for the PyTorch model on the CPU and (if present) on MPS."""
    result = {}
    cpu_model = WeSpeaker.load()
    x = torch.from_numpy(clip).unsqueeze(0)
    with torch.no_grad():
        result["pytorch_cpu_ms_per_5s"] = round(median_ms(lambda: cpu_model(x)), 2)
        if torch.backends.mps.is_available():
            mps_model = WeSpeaker.load().to("mps")
            xm = x.to("mps")

            def run_mps():
                mps_model(xm)
                torch.mps.synchronize()          # wait for the GPU so the time is real

            result["pytorch_mps_ms_per_5s"] = round(median_ms(run_mps), 2)
            result["pytorch_mps_vs_cpu_cos"] = cosine(mps_model(xm).cpu().numpy(), cpu_model(x).numpy())
    return result


# ----------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert WeSpeaker to Core ML and check it (10-ml 6.2)")
    parser.add_argument("--skip-convert", action="store_true", help="reuse the existing .mlpackage")
    args = parser.parse_args()

    model = WeSpeaker.load()                     # frozen, eval mode, CPU fp32
    rows = read_speech_test()
    clip, clip_info = selftest_clip(rows)        # 5 s VCTK clip: trace example and self-test clip
    report = {"date": date.today().isoformat(), "package": str(PACKAGE.relative_to(REPO_ROOT)),
              "torch": torch.__version__, "coremltools": ct.__version__, "cosine_min_required": COSINE_MIN}

    # ---- 1. Convert
    if args.skip_convert:
        if not PACKAGE.exists():
            sys.exit(f"{PACKAGE} does not exist; run without --skip-convert first")
        print(f"reusing {PACKAGE}")
    else:
        t0 = time.perf_counter()
        mlmodel = convert(model, torch.from_numpy(clip).unsqueeze(0))
        PACKAGE.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(PACKAGE))               # replaces an older package of the same name
        report["convert_s"] = round(time.perf_counter() - t0, 1)
        print(f"saved {PACKAGE} ({report['convert_s']} s)")
    report["package_mb"] = round(folder_mb(PACKAGE), 1)

    # ---- 2. Check the SAVED package (what the app ships), per compute unit, against PyTorch CPU fp32
    clips = check_clips(rows)
    with torch.no_grad():
        reference = [model.embed_numpy(c["audio"]) for c in clips]
        clip_embedding = model.embed_numpy(clip)

    report["clips"] = [{"name": c["name"], "seconds": c["seconds"], "speaker": c["speaker"], "paths": c["paths"]}
                       for c in clips]
    all_pass = True
    for unit in COMPUTE_UNITS:
        mlmodel = ct.models.MLModel(str(PACKAGE), compute_units=getattr(ct.ComputeUnit, unit))
        cosines = []
        norms = []
        for c, ref in zip(clips, reference):
            e = coreml_embed(mlmodel, c["audio"])
            cosines.append(cosine(e, ref))
            norms.append(float(np.linalg.norm(e)))
        selftest_cos = cosine(coreml_embed(mlmodel, clip), clip_embedding)
        ms = median_ms(lambda: coreml_embed(mlmodel, clip))
        unit_ok = min(cosines) >= COSINE_MIN and selftest_cos >= COSINE_MIN
        all_pass = all_pass and unit_ok
        report[unit] = {"cos_per_clip": [round(v, 7) for v in cosines], "cos_min": round(min(cosines), 7),
                        "one_minus_cos_max": float(1.0 - min(cosines)),
                        "selftest_clip_cos": round(selftest_cos, 7), "norm_min": round(min(norms), 6),
                        "norm_max": round(max(norms), 6), "predict_ms_per_5s_median": round(ms, 2),
                        "pass": unit_ok}
        print(f"{unit}: min cos {min(cosines):.7f}, self-test clip cos {selftest_cos:.7f}, "
              f"{ms:.1f} ms per 5 s clip, {'PASS' if unit_ok else 'FAIL'}")
    report.update(pytorch_timings(clip))
    report["pass"] = all_pass
    with open(CHECK_JSON, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {CHECK_JSON}")

    # ---- 3. Self-test inputs for export.py (wespk_selftest.est)
    meta = {
        "kind": "wespeaker-selftest",
        "tolerances": {"cosine_min": SELFTEST_COSINE_MIN},
        "sources": [clip_info],
        "embedding_from": "ml/earshot/wespeaker.py, PyTorch CPU fp32, eval mode",
        "wespeaker_weights": str(Path(DEFAULT_WEIGHTS).relative_to(REPO_ROOT)),
        "wespeaker_weights_sha256": file_sha256(DEFAULT_WEIGHTS),
        "coreml_selftest_clip_cos": {unit: report[unit]["selftest_clip_cos"] for unit in COMPUTE_UNITS},
        "note": "WeSpeaker VoxCeleb research-terms weights - personal use only",
    }
    SELFTEST_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SELFTEST_NPZ, clip=clip.astype(np.float32), embedding=clip_embedding.astype(np.float32),
             meta_json=np.array(json.dumps(meta)))
    print(f"wrote {SELFTEST_NPZ}")

    if not all_pass:
        print(f"FAIL: a Core ML cosine is below {COSINE_MIN}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
