# Third-party material

Earshot's own code, docs and assets are MIT (see [LICENSE](LICENSE)). This file lists everything Earshot builds
on, where it comes from and its license. Unless a line says otherwise, none of it is copied into this repo.

## What is not in this repo

- **Audio datasets** - `dataprep/` downloads every dataset into `data/`, which is gitignored. The repo holds the
  download and check scripts only, so no third-party audio is committed.
- **Trained Earshot weights and test vectors** - `ml/exports/`, `ml/runs/`, `ml/cache/` and
  `ios/Earshot/Resources/Models/` are gitignored. You make them with the ML export, then `make models` copies them
  into the app.
- **WeSpeaker voiceprint weights** - not in the repo, and never should be. See the next section.

## Speaker encoder (WeSpeaker ResNet34)

- **Code** - WeSpeaker, https://github.com/wenet-e2e/wespeaker, Apache-2.0. `ml/earshot/wespeaker.py` is
  written for Earshot. It follows WeSpeaker's ResNet34 and Kaldi fbank front end, with parameter names that match
  the official checkpoint so that checkpoint loads as is. No WeSpeaker source file is copied.
- **Weights** - `voxceleb_resnet34` (`avg_model.pt`), trained on VoxCeleb, from
  https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34. WeSpeaker's docs say a pretrained model follows its
  training data's license, and VoxCeleb is granted under CC BY 4.0 "for research purposes" (the Hugging Face card
  says Apache-2.0, which conflicts). So the weights and their Core ML copy stay out of this repo, and any build that
  bundles them is for personal use only. You download them yourself into `models/wespeaker-voxceleb-resnet34/`
  (gitignored). The full check is in `docs/research/fact-check.md` section 10 and `docs/DECISIONS.md` D24.

## Training data

Hours and counts are from `docs/DECISIONS.md` D56. Full URLs, attribution and per-file rules are in
[`dataprep/LICENSES.md`](dataprep/LICENSES.md). The data loader refuses any non-commercial row outside the test
split, and `python -m dataprep.verify` fails if one shows up.

| Dataset | License | Used for |
|---|---|---|
| LibriTTS-R train (Koizumi et al.) | CC BY 4.0 | training speech, 35.8 h |
| GLOBE V2 (Wang et al., from Mozilla Common Voice) | CC0 1.0 | training speech, 11.0 h |
| CSTR VCTK 0.92 (Yamagishi, Veaux, MacDonald) | CC BY 4.0 | training speech, 3.2 h; held-out speakers are test only |
| VoxPopuli en (Wang et al.) | CC0 1.0 | training speech, 3.4 h |
| FSD50K, CC0 and CC BY clips only (Fonseca et al.) | per clip CC0 1.0, CC BY 3.0 or CC BY 4.0 | training noise; CC BY clips keep their Freesound uploader and title in the manifest |
| DEMAND (Thiemann, Ito, Vincent) | CC BY-SA 3.0 | training noise; the SCAFE scene is held out for test |
| MUSAN noise and music, commercial-use files only (Snyder, Chen, Povey) | per file CC0, public domain, CC BY or CC BY-SA | training noise and music |
| OpenSLR 28 real room impulse responses (Ko et al.) | Apache-2.0 | room echo |
| Simulated rooms | made by Earshot's code with pyroomacoustics (MIT) | room echo |

Credits line for the app - Speech from LibriTTS-R (Koizumi et al., CC BY 4.0), VCTK (Yamagishi, Veaux, MacDonald,
CC BY 4.0), VoxPopuli (Wang et al., CC0) and GLOBE (Wang et al., CC0, from Mozilla Common Voice). Noise from FSD50K
(Fonseca et al.; CC BY clips credited to their Freesound uploaders), DEMAND (Thiemann, Ito, Vincent, CC BY-SA 3.0)
and MUSAN (Snyder, Chen, Povey). Real room impulse responses from OpenSLR 28 (Ko et al., Apache 2.0).

## Evaluation data

| Dataset | License | Note |
|---|---|---|
| LibriSpeech test-clean (Panayotov et al.) | CC BY 4.0 | held-out speakers |
| LibriMix metadata, Libri2Mix test lists (Cosentino et al.) | MIT | mixture definitions |
| WHAM! noise, test split (Wichern et al.) | CC BY-NC 4.0 | **evaluation only, never used for training** |
| EdAcc test (Sanabria et al.) | CC BY-SA 4.0 | accents |
| VoxPopuli en_accented test (Wang et al.) | CC0 1.0 | accents |
| DiPCo, part (Van Segbroeck et al.) | CDLA-Permissive-1.0 | real dinner-party recordings |
| MIT IR Survey (Traer and McDermott) | CC BY 4.0 as stated by mirrors (the lab page states none) | held-out real restaurant and bar rooms |

Not fetched on purpose - the BUT/NTT SpeakerBeam enrollment list (a click-through evaluation agreement), Emilia
(gated or non-commercial) and VOiCES (archive-only). See `dataprep/LICENSES.md`.

## Apple frameworks

Part of iOS and the Xcode SDK, used under Apple's SDK terms. Nothing from them is copied into the repo.

- **Accelerate** - vDSP FFTs and BLAS in the C engine (`engine/src/`).
- **Core ML** - runs the voiceprint encoder on the phone (never on the audio thread).
- **AVFoundation and AudioToolbox** - the audio session and the RemoteIO audio unit.
- **CoreMotion** - AirPods head tracking.
- **MediaPlayer** - AirPods stem press commands.
- **SoundAnalysis** - the on-device alarm listener.
- **SwiftUI, UIKit, Combine, CryptoKit, UserNotifications, XCTest** - the app, hashing, notifications and tests.

## Python packages

Installed into `.venv`, not vendored. Pinned versions are in `docs/spec/10-ml.md` 1.1 and
`ml/requirements-eval.txt`.

| Package | Version | License | Used for |
|---|---|---|---|
| torch, torchaudio | 2.7.1 | BSD-3-Clause | model, training, export, Kaldi mel matrix |
| coremltools | 9.0 | BSD-3-Clause | converting the voiceprint encoder to Core ML |
| numpy | 2.5.4 | BSD-3-Clause | everywhere |
| scipy | 1.18.1 | BSD-3-Clause | filters, the generated resampler tables |
| soundfile | 0.14.0 | BSD-3-Clause (loads libsndfile, LGPL-2.1, as a shared library) | reading and writing audio |
| soxr | 1.1.0 | LGPL-2.1-or-later | resampling in `dataprep/` |
| PyYAML | 6.0.3 | MIT | training configs |
| pyroomacoustics | 0.10.1 | MIT | simulated rooms |
| requests | 2.34.2 | Apache-2.0 | dataset downloads |
| pyarrow | 26.0.1 | Apache-2.0 | reading parquet shards in `dataprep/` |
| huggingface_hub | 2.4.0 | Apache-2.0 | dataset downloads |
| fsspec | 2026.9.0 | BSD-3-Clause | remote file reads in `dataprep/` |
| pytest | 9.2.0 | MIT | tests |
| pystoi | 0.4.1 | MIT | ESTOI score in evaluation |
| pesq | 0.0.4 | MIT wrapper around the ITU-T P.862 reference code, which has its own ITU terms | PESQ score in evaluation |

The one-off comparison scripts in `docs/research/data-training-bench/` also import onnx, onnxruntime,
resemblyzer and speechbrain. Nothing else needs them.

## Tools

- **XcodeGen** (MIT) - makes `ios/Earshot.xcodeproj` from `ios/project.yml`.
- **uv** (Apache-2.0 or MIT) - makes the Python venv.
- **ffmpeg** (LGPL or GPL, depending on the build) - only `bench/bench_decode.py`, to make Opus test copies.

## Made for Earshot

- **Resampler and decimation filters** - `engine/src/resampler_tables.c` and `ml/assets/decim16k_fir.npy` are
  generated by `engine/tools/make_resampler_tables.py` with scipy. They are Earshot's own (MIT).
- **App icon and reading passage** - `ios/Earshot/Resources/Assets.xcassets/` and
  `ios/Earshot/Resources/passage.txt` were made for this project (MIT).

## Ideas and prior work

- **Look Once to Hear** (Veluri et al., University of Washington, CHI 2024) - the interaction idea, where you look at
  someone for a few seconds and then hear only them. Earshot uses none of its code or weights (its code has a UW
  non-commercial license).
- **Band-split RNN and TF-GridNet** - the model design follows this line of research papers. No code from
  those projects is vendored in this repo.
- More prior work, with links, is in `docs/research/prior-art.md`.
