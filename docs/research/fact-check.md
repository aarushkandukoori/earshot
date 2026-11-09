# Earshot - adversarial fact check of the research findings

Date - 2026-11-06 (checks run 23:30-00:00 local).
Inputs - the findings in `prior-art.md`, `ios-platform.md`, `data-training.md` and `bench/RESULTS.md`.
Method - I picked the 12 claims that most drive the architecture and tried to break each one from a primary
source: the paper PDF, Apple's documentation (the JSON behind developer.apple.com), the license file itself,
the dataset page, the app source code, or a fresh measurement on this Mac. When no primary source settled a
claim, the verdict is "unclear".

## Bottom line

- **The wired latency picture is worse than reported.** The Superpowered table does contain iPhone 15/16
  measurements. They are filed under the plain name "iPhone" because the app's model list stops at the
  iPhone 14 Pro Max. Those rows are mostly **15-17 ms round trip (range 9-27 ms)**, not the 9 ms of the
  iPhone 13. At 15-17 ms the 8 ms model profile lands around 24-27 ms, over the 20 ms target.
- **The AirPods planning range "120-220 ms extra" is not supported for AirPods Pro 2.** The cited Pro 2
  number (126 ms) is touch-to-sound on an iPad, and the built-in speaker took 83 ms in the same test. So the
  Bluetooth part was **about 43 ms (Pro 2) and 84 ms (Pro 1)** in that test. Nobody has measured Earshot's
  real path (iPhone mic, app, AirPods over A2DP).
- **Speaker-encoder license gap.** WeSpeaker's code is Apache-2.0, but its docs say VoxCeleb-trained weights
  follow the VoxCeleb license, and the VoxCeleb page grants it "for research purposes" with video copyright
  kept by the owners. This affects every VoxCeleb-trained encoder in the plan.
- **GRU on MPS is confirmed slow** (re-measured 11x slower than LSTM). This conflicts with the GRU-based
  Earshot-BSR design in `prior-art.md`.
- **The other 8 claims hold up**, some with corrected details. One example is the asymmetric-window result:
  the 1.5 dB case was speaker-dependent separation on Danish HINT, not WSJ0-2mix.

| # | Claim (short) | Verdict | Confidence |
|---|---|---|---|
| 1 | Wired round trip - iPhone 13 Pro 9-10 ms, 14 Pro 12-17 ms, no public iPhone 15/16 number | **refuted** (iPhone 15/16 part) | medium |
| 2 | AirPods add about 120-220 ms | **unclear** | medium |
| 3 | Asymmetric 32/8 ms window beats 8/8 ms by 1.4-1.5 dB | confirmed, with corrections | high |
| 4 | BNNS Graph is real-time safe (no allocation, single thread, Audio Unit demo), Builder has GRU/LSTM, no FFT | confirmed, with caveats | high |
| 5 | Built-in mic in plus AirPods out with `.playAndRecord` + `.allowBluetoothA2DP` only | confirmed (documented), one unverified counter-report | high |
| 6 | `CMHeadphoneMotionManager` needs only `NSMotionUsageDescription`, no entitlement | confirmed | high |
| 7 | Free Personal Team allows Background Modes and HealthKit, not Head Pose or Sustained Execution | confirmed | high |
| 8 | PyTorch 2.7.1 has no native MPS GRU, GRU about 14x slower than LSTM | confirmed (11-14x) | high |
| 9 | LOTH code UW non-commercial and no weights, SpeakerBeam evaluation-only, USEF-TSE and WHAM! CC BY-NC 4.0 | confirmed | high |
| 10 | WeSpeaker ResNet34 (VoxCeleb) is a commercially safe frozen encoder | **unclear** | medium |
| 11 | LibriTTS-R - CC BY 4.0, 2,311 train speakers, 555 h, tarballs 8.1/28/46 GB | confirmed | high |
| 12 | Cross-attention 8.85 dB vs FiLM 6.70 dB in a mono causal 8/4 ms noisy-enrollment extractor | confirmed, with confounds | high |

---

## 1. Wired round-trip latency (Superpowered table)

- **Claim** - Crowd-measured round trip (built-in mic to speaker, measurement mode, 64 frames at 48 kHz):
  iPhone 13/13 Pro 9-10 ms, iPhone 12 family 12-14 ms, iPhone 14/14 Plus 15 ms, iPhone 14 Pro 12-17 ms,
  14 Pro Max 16-17 ms, SE 3 8-9 ms. The table has no iPhone 15 or 16 rows, so no public number exists
  for iPhone 15/16 (`ios-platform.md`, finding "1. Latency - measured round trip" and summary).
- **Verdict** - **Refuted in part.** Every per-model number is right. The conclusion that no iPhone 15/16
  data exists is wrong.
- **What I checked** - I downloaded the live table (header - 2,898,400 measurements on 40,681 models/builds)
  and parsed every iPhone row. I also read the app's `iOS/ViewController.mm` and `iOS/machines.h`
  through the GitHub API. The repo's last commit is 2024-08-06.
- **What holds** - The rows match the claim exactly. For example the iPhone 13 Pro has 9, 9, 9, 9 ms and one
  10-200 ms row. The iPhone 14 Pro has 12, 16, 17, 17, 17 ms. The iPhone 14 and 14 Plus show 15 ms and the SE 3
  shows 8-9 ms. The source confirms `.playAndRecord` with `.defaultToSpeaker`, `.measurement` mode,
  48 kHz, a 64/48000 s buffer request, RemoteIO, and input and output handled in one callback.
- **What is wrong** - The app names a phone by looking its machine id up in `machines.h`. That list ends at
  `iPhone15,3` (iPhone 14 Pro Max), and any phone not in it is labeled with the plain name "iPhone". Every older
  iPhone that can run iOS 17 is in the list. So a plain "iPhone" row on iOS 17 or 18 can only come from an
  iPhone 15 or 16 family phone. The table has 9 such rows at 64 frames and 48 kHz - **9, 10, 15, 16, 17, 17,
  17 (17-198), 17 (17-69) and 27 ms**. The median of the best values is **17 ms**. One more row, at 24 kHz,
  shows 257 ms and is probably a Bluetooth route.
- **Architecture impact** - `ios-platform.md` estimates the total as about X + 8 ms for the 8 ms profile, so
  20 ms needs X of at most 12 ms. If the iPhone 16 Pro behaves like most of these rows (X = 15-17 ms), the
  8 ms profile gives about 23-25 ms before resampling and slack. The 4 ms profile (16/4/2 ms windows) gives
  about 19-21 ms. Only phones in the 9-10 ms group make 20 ms with the 8 ms profile. These are unlabeled crowd
  rows on the speaker path, so Phase 0 measurement on the real phone with USB-C output is still step one.
  Treat the 4 ms profile and a 32-frame buffer as the likely wired setup, not a rare fallback.
- **Confidence** - medium (crowd data, model inferred from the naming logic, iOS 17-18 only, speaker path).

## 2. AirPods (A2DP) added latency

- **Claim** - For a live app with AirPods, plan on about 120-220 ms extra mouth-to-ear latency. The supporting
  data are AirPods 2 (reported `outputLatency` 160.667 ms, measured 215-220 ms falling to 155-160 ms) and
  AirPods Pro 2 at 126 ms vs Pro 1 at 167 ms tap-to-sound (speaker baseline 83 ms). A macOS log showed 151 ms
  normally and 60 ms in Low Latency Game mode (`ios-platform.md`, "2. Bluetooth - A2DP output latency").
- **Verdict** - **Unclear.** Each quoted number is in its source, but the planning range does not follow from
  them for the user's AirPods Pro 2.
- **What I checked** - Coyle's 2022 post, 9to5mac's write-up of his 2019 test, Apple forum threads 679274 and
  764070, and searches for any other AirPods Pro 2 measurement (none found).
- **What is wrong** - Coyle's numbers are one-way, from an Apple Pencil tap on an iPad Pro to the keyboard click
  sound. They include touch and system processing, and he notes that the Pencil alone adds about 20 ms. In
  the same test the built-in speakers took 83 ms, so the Bluetooth part was **about 43 ms (Pro 2) and
  84 ms (Pro 1)**. The 155-220 ms figures are for AirPods 2 (not Pro) on an iPhone 12 mini in 2021, measured by
  one developer. The 151/60 ms log is from macOS with an unnamed headset, and the log also mentions a JBL device.
  No source measures Earshot's path: built-in mic, app, then AirPods Pro over A2DP on iOS 26.
- **Architecture impact** - Do not build in a 120 ms floor. Design the Bluetooth mode and its UI for a
  measured value anywhere from about 40 to 220 ms. Calibrate each time AirPods connect, and re-check over time
  (the AirPods 2 report saw it drift by about 60 ms in 20-30 minutes). Expect Pro 2 to beat Pro 1.
- **Confidence** - medium.

## 3. Asymmetric analysis/synthesis windows

- **Claim** - At 8 ms algorithmic latency a 32 ms analysis / 8 ms synthesis window pair beat a symmetric
  8/8 ms pair by 1.4-1.5 dB SDR in online two-speaker separation. Evidence - online deep clustering 5.7 to
  7.1 dB, online mask inference 7.3 to 8.8 dB, 32/32 ceiling 10.2 dB "(WSJ0-2mix, 8 kHz)" (`prior-art.md`).
- **Verdict** - **Confirmed, with corrections.**
- **What I checked** - The full PDF of Wang, Naithani, Politis and Virtanen (EUSIPCO 2021), Tables 1-2 and
  Section 3.2, and the full text of Wu and Braun (ICASSP 2025).
- **What holds** - Online DC improves from 5.7 to 7.1 dB and online MI from 7.3 to 8.8 dB, and the online 32/32
  result is 10.2 dB. Latency stays at 8 ms.
- **What is wrong** - The mask-inference result and the 10.2 dB ceiling are **not WSJ0-2mix**. They come from
  **speaker-dependent** separation on Danish HINT: three fixed speaker pairs, trained and tested on the same
  talkers, at 16 kHz. Only the DC result is speaker-independent (WSJ0 at 8 kHz), and its "online" mode
  estimates cluster centres from the first 0.6 s of each mixture. Both models are 2021-era LSTM stacks (4x600 and
  3x512 units) scoring 5-9 dB. **Wu and Braun found the asymmetric-window gain not significant for a strong
  CRUSE model**. It mattered only for a weaker U-Net.
- **Architecture impact** - Keep the 32/8 ms pair as the default, since it costs no extra compute. But assume
  about 0 dB gain for a strong TSE model until our own ablation shows otherwise. It is already the first
  planned ablation.

## 4. BNNS Graph real-time safety

- **Claim** - Apple recommends BNNSGraph for audio and latency-sensitive inference. It compiles from
  `.mlmodelc` (`BNNSGraphCompileFromFile`, iOS 18) with a single-thread option, and
  `BNNSGraphContextExecute` does not allocate when the workspace and outputs come from the caller and the shapes
  are fixed. One context runs on one thread at a time, and Apple demonstrated it in an Audio Unit. The iOS 26
  `BNNSGraph.Builder` has GRU/LSTM and similar ops but no FFT.
- **Verdict** - **Confirmed, with caveats.**
- **What I checked** - Apple documentation JSON for `BNNSGraphContextExecute`,
  `BNNSGraphCompileOptionsSetTargetSingleThread`, `BNNSGraphCompileFromFile`,
  `BNNSGraphContextMakeStreaming`, `BNNSGraphContextSetStreamingAdvanceCount`, `BNNSGraph.Builder`,
  `Builder.Tensor` (all 120 methods) and `Builder.Direction`, plus the WWDC24 session 10211 transcript.
- **What holds** - The execute doc lists four no-allocation conditions: a non-nil workspace, all tensors
  bounded before the call, no new shape information, and all outputs supplied by the caller. A context may be
  used on only one thread at a time, and single-thread execution is an option (the default is
  multi-threaded). In WWDC24 10211 Apple builds an Audio Unit extension on BNNS Graph and explains that real time
  means no allocation or multithreading during execute. Builder has `gru`, `lstm`, `bidirectionalLSTM`, `rnn`,
  `conv`, `convTranspose`, `matmul`, `linear`, `layerNorm`, `rmsNorm`, `softmax`, `gelu` and `silu`, and no
  FFT/DFT/STFT op. Streaming contexts are iOS 18.0 and need a `BNNSOptions` `StateMode=Streaming` attribute.
  `SetStreamingAdvanceCount` is iOS 18.4.
- **Caveats** - (a) The WWDC demo model is a bitcrusher, not a recurrent network. Apple publishes no timing for a
  model of TSE size. (b) Builder's `gru` takes only `forward` or `reverse` (bidirectional exists only for LSTM),
  and coremltools' MIL `gru` is the same. A BiGRU across bands therefore becomes two GRUs plus a concat, while a
  BiLSTM maps directly. (c) Apple publishes no list of the Core ML ops that `CompileFromFile` accepts. (d) Streaming
  contexts do not support models whose states advance by different amounts.
- **Architecture impact** - The plan stands. Compile the real exported graph on the phone early.

## 5. Built-in mic in, AirPods out

- **Claim** - `.playAndRecord` with only `.allowBluetoothA2DP` routes output to AirPods (A2DP) while input
  stays on the built-in mic. With both options set, HFP wins, and HFP moves input and output to the headset.
- **Verdict** - **Confirmed (documented behavior).**
- **What I checked** - Apple docs for `allowBluetoothA2DP`, `allowBluetooth` and `allowBluetoothHFP`, and forum
  thread 784318.
- **What holds** - Since iOS 10, `.playAndRecord` may route output to A2DP with this option. When both options are
  set and a device supports both profiles, hands-free ports get higher routing priority. Choosing an HFP input
  also moves output to HFP.
- **Risk** - One unanswered report (iOS 18.4.1, iPhone 12 Pro Max) says the route listed the built-in mic but
  the audio sounded like the AirPods mic. The poster also tried combinations with `.allowBluetooth` and
  `.defaultToSpeaker`, so it may be a configuration mix-up. Apple did not reply.
- **Architecture impact** - Keep the Phase 0 tap test: play a click near the phone's mic and confirm it in
  the captured input.

## 6. Head tracking without an entitlement

- **Claim** - `CMHeadphoneMotionManager` (iOS 14) works with AirPods Pro, needs only `NSMotionUsageDescription`
  (the app crashes without it) and no entitlement. The Head Pose entitlement is only for system spatial mixing.
- **Verdict** - **Confirmed.**
- **What I checked** - Apple docs for `CMHeadphoneMotionManager` and `com.apple.developer.coremotion.head-pose`,
  and Apple Support article 102469.
- **What holds** - The class is iOS 14.0+. The doc requires `NSMotionUsageDescription`, says the app crashes
  without it, and mentions no entitlement. The Head Pose entitlement (iOS 18) covers head tracking in
  `AVAudioEnvironmentNode`, `AUSpatialMixer` and `PHASEListener`, and none of these is the motion manager. Apple
  Support lists "AirPods Pro", with no generation excluded, among models with head tracking.
- **Not verified** - The update rate (about 25 Hz per community reports), behavior with the screen locked, and
  behavior inside a `.playAndRecord` + A2DP session. I found no primary source for any of these.

## 7. Free Personal Team capabilities

- **Claim** - A free account allows Background Modes, HealthKit, App Groups, Keychain sharing and Data
  protection. It does not allow Head Pose, Spatial Audio Profile, Sustained Execution, Push, Time Sensitive
  Notifications, Siri, iCloud, Extended Virtual Addressing or Increased Debugging Memory Limit. Limits are
  10 App IDs per 7 days, 3 devices, 3 apps per device, and profiles that last 7 days.
- **Verdict** - **Confirmed.**
- **What I checked** - I parsed Apple's capability table (58 rows) and read the compare-memberships page.
- **What holds** - The free "Apple Developer" column marks App groups, Background modes, Data protection,
  HealthKit, HomeKit, Inter-App Audio, Keychain sharing, Maps and Wireless Accessory Configuration. All the
  capabilities listed as unavailable are unmarked, and so are Push to Talk and Access WiFi Information. The
  membership page gives the 10 App ID, 3 device, 3 apps per device and 7-day limits.

## 8. GRU vs LSTM on the M4 GPU (MPS)

- **Claim** - PyTorch 2.7.1 has no native MPS GRU. `nn.GRU` (2x256, 500 frames, batch 8) took 657-727 ms per
  training step on MPS, against 51 ms for LSTM (about 14x) and 125-135 ms for GRU on the CPU. Use LSTM on MPS
  (`data-training.md`, `bench/RESULTS.md`).
- **Verdict** - **Confirmed (11-14x).**
- **What I checked** - The PyTorch source at v2.7.1 and at main. In `native_functions.yaml` the only MPS RNN
  kernels are `_lstm_mps` and `lstm_mps_backward`. `RnnOps.mm` mentions LSTM 12 times and GRU 0 times. I also
  re-ran the same shapes twice in the project venv (torch 2.7.1, load 4.7-6.4):

  | Run | MPS LSTM | MPS GRU | CPU LSTM (4 threads) | CPU GRU (4 threads) |
  |---|---|---|---|---|
  | 1 | 59 ms | 661 ms | 140 ms | 132 ms |
  | 2 | 55 ms | 623 ms | 125 ms | 110 ms |

- **Architecture impact** - This **conflicts with the GRU-based Earshot-BSR** in `prior-art.md` (a BiGRU across
  bands and a GRU across time). Either switch both to LSTM, which has a native MPS kernel, a native
  `bidirectionalLSTM` in BNNS Builder and Core ML support, at about 4/3 the recurrent MACs. Or keep the GRUs and
  train them on the CPU (faster than LSTM there) or with torch 2.15 (the bench measured GRU 2.4-3.2x faster
  there), then export from the 2.7.1 environment.

## 9. Licenses of code and data to avoid

- **Claim** - The LOTH code is under a UW non-commercial license and no weights are released. BUT/NTT SpeakerBeam
  is evaluation-only and reserves patent rights. USEF-TSE and WHAM! noise are CC BY-NC 4.0.
- **Verdict** - **Confirmed.**
- **What I checked** - I read each license file through the GitHub API, plus the LOTH README, repo tree and
  releases, and the official WHAM! page.
- **What holds** - LOTH's license is a University of Washington (2023) non-commercial license, with commercial
  rights through UW CoMotion. The README links only datasets (Google Drive), and the repo has no weight files
  and no releases. SpeakerBeam's file is a software evaluation agreement with BUT and NTT. It allows internal
  testing and evaluation only, and they assert all rights, including patents. USEF-TSE's LICENSE is the CC BY-NC
  4.0 text. The WHAM! page states CC BY-NC 4.0, and its noise zip is 17 GB, unzipping to 35 GB.

## 10. Speaker encoder license (WeSpeaker ResNet34 on VoxCeleb)

- **Claim** - WeSpeaker is Apache-2.0, so the frozen WeSpeaker ResNet34 (VoxCeleb) encoder is a commercially
  usable choice (`prior-art.md` license list plus the `data-training.md` recommendation).
- **Verdict** - **Unclear.** The code license is fine. The weights' license is not clearly commercial.
- **What I checked** - WeSpeaker's LICENSE and the "Model License" section of `docs/pretrained.md`, the VoxCeleb
  license page that WeSpeaker links (KAIST mirror), the Oxford VGG page (no license text found today), the
  Hugging Face card for `Wespeaker/wespeaker-voxceleb-resnet34`, and the SpeechBrain ECAPA card.
- **What I found** - The code is Apache-2.0. WeSpeaker's docs say each pretrained model follows its training
  dataset's license, and that VoxCeleb models follow CC BY 4.0. The VoxCeleb page grants CC BY 4.0
  "for research purposes" and says copyright stays with the owners of the original videos. The Hugging Face card
  says apache-2.0, which conflicts with WeSpeaker's own docs. SpeechBrain ECAPA (apache-2.0 card, trained on
  VoxCeleb) and CAM++ have the same issue.
- **Architecture impact** - This is fine for the user's personal build. For anything distributed, plan one of
  two options. (a) Train our own encoder on the commercial-friendly speech already in the pack: LibriTTS-R 2,311,
  VoxPopuli 1,313, GLOBE 23,519 and VCTK 110 speakers. (b) Use the jointly learned internal embedding that
  `prior-art.md` cites from arXiv 2406.09928 (that result was not re-checked here). Keep WeSpeaker only as a
  benchmark, and get a legal review before release.

## 11. LibriTTS-R size and license

- **Claim** - LibriTTS-R is CC BY 4.0, 24 kHz and Miipher-restored, with 2,311 training speakers (247 + 904 +
  1,160) and 555 h. Its train tarballs are 8.1, 28 and 46 GB, and the server supports range requests.
- **Verdict** - **Confirmed.**
- **What I checked** - The OpenSLR 141 page, Table 2 of the LibriTTS paper (arXiv 1904.02882), and a HEAD request
  on `train_clean_360.tar.gz` to openslr.org and its trmal mirror.
- **What holds** - The license is CC BY 4.0 at 24 kHz, and the samples are identical to LibriTTS apart from the
  Miipher restoration. Speakers are 247 + 904 + 1,160 = 2,311, and hours are 53.78 + 191.29 + 310.08 = 555.15.
  Tarball sizes are 8.1G, 28G and 46G. `train_clean_360` returns Content-Length 28,953,983,271 bytes with
  Accept-Ranges: bytes on both hosts.
- **Notes** - A `.tar.gz` cannot be read at random positions, so "stream and filter" still reads all of
  about 82 GB in order (about 2 h at 11 MB/s). Range support only helps resume a broken download. LibriTTS has
  the same speakers and splits as LibriSpeech, so do not count LibriSpeech train speakers twice, and keep the
  test speakers out of training.

## 12. Cross-attention vs FiLM conditioning

- **Claim** - In a mono causal TF-GridNet extractor (8 ms window, 4 ms hop) with noisy enrollments,
  cross-attention reached 8.85 dB SI-SNRi against 6.70 dB for FiLM (2 talkers plus WHAM! noise). A USEF-TFGridNet
  trained with clean enrollment fell to -0.03 dB with noisy enrollment (arXiv 2502.16611, NeurIPS 2025).
- **Verdict** - **Confirmed, with confounds.**
- **What I checked** - The full PDF (v3, Xu, Yang, Trigoni and Markham): Table 1, Table 6, Section 3 and
  Appendices B-C.
- **What holds** - In the mono rows with 2 speakers in the mixture and 2 in the enrollment, cross-attention
  scores 8.85 ± 3.67 dB, FiLM 6.70 ± 3.43 dB and USEF-TFGridNet -0.03 ± 5.97 dB. The audio is 16 kHz with an STFT
  window of 128 and hop of 64 (8/4 ms). There are three causal TF-GridNet blocks with unidirectional LSTMs and
  causal attention, following LOTH. Noise is from WHAM!, and the cross-attention model has 1.88 M params.
- **Confounds** - (a) The FiLM variant is larger (3.94 M params vs 1.88 M) and squeezes a token sequence (the
  enrollment features pooled to T/40 tokens) into one vector. The 2.15 dB gap therefore mixes fusion type with
  "sequence vs single vector". (b) Both variants use the paper's positive plus negative enrollment pair (clips
  with and without the target) and a two-stage training with a distilled encoder. Earshot's plan (a d-vector plus
  tokens, with own voice as the negative) is different. (c) USEF-TFGridNet was run as published, with a
  clean-enrollment design. Its collapse shows that training needs noisy enrollment. It does not show that USEF
  cues fail.
- **Architecture impact** - This supports keeping cross-attention over enrollment tokens. Keep the FiLM vs
  FiLM + cross-attention ablation, and do not expect the full 2.15 dB to carry over.

---

## Extra spot checks (not in the 12)

| Claim | Result | Source |
|---|---|---|
| LOTH - 7.40 / 7.01 / 4.53 dB SI-SNRi, 2.04 M params, 4.63 GMAC, 12 ms algorithmic, 6.24 ms compute, 18.24 ms total on Orange Pi 5B | confirmed | arXiv 2405.06289 (table and Section 5) |
| Wu and Braun - A1 20 ms 10.52 dB, F6 3 ms large (10.95 GMAC/s) 10.34 dB, D3 3 ms small 9.03 dB; asymmetric windows significant only for the weak U-Net | confirmed | arXiv 2409.10358 |
| AV-PVQE - PVQE picks the target 98.63% (louder), 61.08% (equal, within 2 dB), 6.25% (quieter); LRS3 SI-SNRi -2.31 to 9.03 dB | confirmed | arXiv 2609.30631 |
| WeSep causal BSRNN (32 ms) - 6.87 dB / 79.93% vs USEF 13.05 / 95.70 vs USEF + context 14.15 / 95.93 | numbers confirmed, **label wrong** - "Accuracy" is the share of samples with SI-SDRi above 1 dB, not correct-speaker accuracy, and the row says "Speaker Emb." without naming ECAPA | arXiv 2607.27436, Table 2 |
| CHiME-9 ECHI Aria track - baseline 55.28%, best ranked system (SFU) 60.31%, close-talk oracle 86.93%; baseline is a causal TF-GridNet (8 ms window, 4 ms hop) with FiLM; 20 ms rule; two teams added own-voice suppression | confirmed (Waseda-NTT reached 63.54% but was excluded from the ranking) | arXiv 2609.26306 |
| Apple DTS - no recording session can activate from the background; any incoming call interrupts `.playAndRecord` | confirmed (Kevin Elliott, Feb 2025) | developer.apple.com/forums/thread/774784 |
| Apple DTS - `MPRemoteCommandCenter` receives A2DP (AVRCP) commands; no API receives HFP commands | confirmed (Aug 2024) | developer.apple.com/forums/thread/761549 |
| coremltools 9.0 - `_TORCH_MAX_VERSION = "2.7.0"`, a `torch.stft` converter, no `istft` converter | confirmed in the installed source | project venv |
| torch 2.15.0 exists (uploaded 2026-10-28); latest stable coremltools is 9.0, with a 9.1.dev1 pre-release | confirmed | pypi.org JSON API |
| A18 Pro - 2 performance + 4 efficiency cores, 16-core Neural Engine, CPU ML accelerators for low-latency work | confirmed | Apple newsroom, Sept 2024 |
| `setPreferredIOBufferDuration` minimum "at least 0.005 s (256 frames)", may be lower; read back `ioBufferDuration` | confirmed (doc text) | Apple doc |
| `.measurement` mode - minimal system processing, primary mic, lower output level | confirmed | Apple doc |
| All cited post-2025 arXiv IDs exist with matching titles (2609.30631, 2607.27436, 2609.26306, 2511.11473, 2505.12288, 2406.09928, 2407.11055) | confirmed | arxiv.org abstract pages |

## What to change in the plan

1. **Wired latency** - Until Phase 0 measures the iPhone 16 Pro, assume a round trip of about 15-17 ms. Make the
   4 ms profile (16/4/2 ms windows) with a 32-frame I/O buffer the main wired candidate, and keep resampler
   delay small. Even then 20 ms is borderline, so report only measured numbers.
2. **AirPods** - Design for a measured 40-220 ms, calibrate on every connect, and re-check over time. Never
   hard-code 120 ms.
3. **Recurrent cells** - Move Earshot-BSR from GRU to LSTM if training on MPS, or train the GRU version on the
   CPU or torch 2.15. Plan BiGRU as forward plus reverse GRUs for BNNS, or use `bidirectionalLSTM`.
4. **Asymmetric window** - Keep it (free), but budget about 0 dB for it until the ablation.
5. **Speaker encoder** - VoxCeleb-trained weights are fine for the personal build. A distributed build needs
   our own encoder trained on CC BY / CC0 data, or the internal jointly trained embedding.
6. **Conditioning** - Keep cross-attention. FiLM was not the cheaper option in the closest paper (3.94 M vs
   1.88 M params).
7. **Fix two labels in `prior-art.md`** - The asymmetric-window mask-inference result is Danish HINT,
   speaker-dependent, at 16 kHz (not WSJ0-2mix). WeSep's "accuracy" means SI-SDRi above 1 dB (not correct
   speaker).

## Reproduce

- `fact-check-bench/superpowered_iphone_rows.py` - parses a saved copy of the Superpowered table and prints the
  labeled iPhone rows and the plain "iPhone" rows on iOS 17 and later. Output from 2026-11-06 is in
  `fact-check-bench/superpowered_iphone_rows_2026-11-06.txt`.
- `fact-check-bench/rnn_mps_check.py` - GRU vs LSTM training-step timing on MPS and CPU. Output is in
  `fact-check-bench/rnn_mps_check_2026-11-06.txt`. Run it with the project venv:
  `../../../.venv/bin/python -I rnn_mps_check.py` (from this folder).

## Sources

Latency and Bluetooth
- https://superpowered.com/latencydata/data.html (retrieved 2026-11-06)
- https://github.com/superpoweredSDK/SuperpoweredLatency/blob/master/iOS/ViewController.mm
- https://github.com/superpoweredSDK/SuperpoweredLatency/blob/master/iOS/machines.h
- https://stephencoyle.net/airpods-pro-2
- https://9to5mac.com/2019/12/22/airpods-latency-test/
- https://developer.apple.com/forums/thread/679274
- https://developer.apple.com/forums/thread/764070
- https://developer.apple.com/forums/thread/784318

Apple platform
- https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetootha2dp
- https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetooth
- https://developer.apple.com/documentation/avfaudio/avaudiosession/setpreferrediobufferduration(_:)
- https://developer.apple.com/documentation/avfaudio/avaudiosession/mode-swift.struct/measurement
- https://developer.apple.com/documentation/accelerate/bnnsgraphcontextexecute(_:_:_:_:_:_:)
- https://developer.apple.com/documentation/accelerate/bnnsgraphcompileoptionssettargetsinglethread(_:_:)
- https://developer.apple.com/documentation/accelerate/bnnsgraphcontextmakestreaming(_:_:_:_:)
- https://developer.apple.com/documentation/accelerate/bnnsgraphcontextsetstreamingadvancecount(_:_:)
- https://developer.apple.com/documentation/accelerate/bnnsgraph/builder/tensor
- https://developer.apple.com/documentation/accelerate/bnnsgraph/builder/direction
- https://developer.apple.com/videos/play/wwdc2024/10211/
- https://developer.apple.com/documentation/coremotion/cmheadphonemotionmanager
- https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.coremotion.head-pose
- https://support.apple.com/en-us/102469
- https://developer.apple.com/help/account/reference/supported-capabilities-ios
- https://developer.apple.com/support/compare-memberships/
- https://developer.apple.com/forums/thread/774784
- https://developer.apple.com/forums/thread/761549
- https://www.apple.com/newsroom/2024/09/apple-debuts-iphone-16-pro-and-iphone-16-pro-max/

Papers
- https://arxiv.org/abs/2106.11794 (asymmetric windows, EUSIPCO 2021)
- https://arxiv.org/abs/2409.10358 (Wu and Braun, ultra-low-latency SE study)
- https://arxiv.org/abs/2502.16611 (noisy positive/negative enrollment, NeurIPS 2025)
- https://arxiv.org/abs/2405.06289 (Look Once to Hear)
- https://arxiv.org/abs/2609.30631 (AV-PVQE)
- https://arxiv.org/abs/2607.27436 (WeSep)
- https://arxiv.org/abs/2609.26306 (CHiME-9 ECHI)
- https://arxiv.org/abs/1904.02882 (LibriTTS)

Training stack
- https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/native_functions.yaml
- https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/mps/operations/RnnOps.mm
- https://pypi.org/project/torch/2.15.0/
- https://github.com/apple/coremltools (installed 9.0 source, `_deps/__init__.py`, `frontend/torch/ops.py`)

Licenses and data
- https://github.com/vb000/LookOnceToHear/blob/main/LICENSE
- https://github.com/BUTSpeechFIT/speakerbeam/blob/main/LICENSE.txt
- https://github.com/ZBang/USEF-TSE/blob/main/LICENSE
- http://wham.whisper.ai/
- https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md
- https://mm.kaist.ac.kr/datasets/voxceleb/
- https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34
- https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- https://www.openslr.org/141/
