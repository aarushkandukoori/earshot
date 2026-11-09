# Earshot - system design, daily-use product first

Date - 2026-11-07. Angle - what makes Earshot usable every day at restaurants and parties (picking people by looking,
switching, AirPods realities, battery, safety, onboarding), with a model and engine that are measured, not hoped for.

Inputs - `docs/BRIEF.md`, `docs/research/prior-art.md`, `ios-platform.md`, `data-training.md`, `fact-check.md`,
`bench/RESULTS.md`, plus new measurements made for this design on the M4 (Appendix B) and web sources (References).

How to read it
- Confidence tags - **(H)** primary source or our own measurement, **(M)** secondary source or a primary source that
  is ambiguous, **(L)** estimate, inference or marketing. Numbers marked "estimate" are not measurements.
- "Measured here" - measured for this design on 2026-11-07 on the shared M4 (load average 3 to 9 from other jobs).
  Method and raw numbers are in Appendix B.
- Writing rules from the brief apply (no em dashes, "Label - explanation" bullets).

Prior work this design builds on, and credits in the app - UW's **Look Once to Hear** (look at a talker for a few
seconds, then hear only them) [8], Oxford's noisy positive/negative enrollment [9], Microsoft's personalized speech
enhancement work (PVQE, TSOS) [14][15], NTT/BUT's target-absent training [16], Google's VoiceFilter-Lite, WeSep/WeSpeaker
[18]. We implement from papers; we copy no code from non-commercial repos.

---

## 0. One-page summary

**What Earshot is** - an iPhone app that plays one chosen person (or a small table of chosen people) into AirPods while
pushing everything else down. The phone's microphone hears the room, a small causal neural network on the phone's CPU
keeps the enrolled voice(s), and AirPods noise cancellation hides most of the direct room sound.

**The honest promises**

| Promise | AirPods Pro 2 (primary) | Wired sealed USB-C earbuds (secondary) |
|---|---|---|
| Delay mouth to ear | Earshot adds about **17 ms**; Bluetooth adds an unknown **45 to 200 ms** that we measure at every connect and show (C.2) | **21 to 29 ms** expected with the v1 model; under 20 ms only if Phase 0 finds a fast phone path (C.1) |
| Hides the room | ANC plus seal attenuate the direct room sound by about 21 to 25 dB; Earshot cannot go below that leak [24] (M) | Depends on the earbud seal; Apple EarPods isolate about 3 dB, so they are not supported for focus use [25] (M) |
| Own voice | Never played back (negative voiceprint, own-voice detector, "mute while I talk" duck) | Same, duck optional |
| Works with no enrollment | "Everyone" mode - all speech except yours, noise removed | Same |
| Battery | iPhone about 3 to 4 % per hour (estimate, B.6 and D.9); AirPods Pro 2 about 6 h with ANC [44] is the real limit | iPhone only |

**Key decisions**

1. **Model - ESB-16**, a causal band-split dual-path **LSTM** network in the STFT domain: 16 kHz, asymmetric 32 ms
   analysis / 8 ms synthesis windows, 4 ms hop, **8 ms algorithmic latency**, 16 bands, width 64, 4 blocks, time LSTM 128,
   band BiLSTM 64 per direction, FiLM plus cross-attention conditioning, complex mask, target and own-voice activity
   heads. **1.94 M parameters, 2.98 GMAC/s** (A.3).
2. **Measured cost** - training **0.98 s per step** (batch 8 x 4 s plus two 5 s enrollments, MPS, torch 2.7.1), so
   **140k steps take about 42 h**; inference written in C with Accelerate takes **175 to 187 us per 4 ms hop** on an M4
   performance core (4.4 to 4.7 % of real time) and 0.59 to 0.64 ms on an efficiency core (Appendix B) (H, Mac proxy).
   GRU was rejected because PyTorch 2.7.1 has no MPS GRU kernel [5][6].
3. **People are separate warm instances** - each enrolled person at the table runs their own extractor state (up to 3),
   so switching is an instant 150 ms crossfade and "Table mode" is the sum. Compute makes this affordable (3 x 0.19 ms per
   4 ms hop on a P-core proxy).
4. **"The person you're looking at" is learned, not sensed** - a single phone mic cannot localize talkers, and head
   tracking only says where the user faces. Earshot stores the head direction at which each person was enrolled, refines
   it while that person talks, and a stem double-press selects the stored person nearest the current head direction
   (E.2). Head movements undershoot targets [47], so selection is relative, with hysteresis.
5. **Enrollment you can do at a table** - point the camera at the person for about 5 s; Vision lip movement marks when
   they talk, and only those frames build the voiceprint. A "quick add" grabs the last 8 s of audio for whoever just
   spoke, with a playback check (B.1).
6. **Engine** - RemoteIO at 48 kHz with 64-frame buffers, a C render callback that only moves samples, one real-time
   worker thread joined to the audio workgroup that runs hand-written C with Accelerate, adaptive output slack with
   concealment, safety DSP (fades, limiter, cap, mute) in the callback (D).
7. **Honest latency** - measured in the app with an acoustic chirp loop on every route change and AirPods connect,
   displayed as "Earshot 17 ms + Bluetooth N ms" (C.4). Never claim under 20 ms over Bluetooth.
8. **Safety outside the network** - own-voice duck, activity gate, slow AGC, zero-latency limiter, hard gain cap,
   400 ms fade-ins, speaker-route mute, and an alarm/siren passthrough using Apple's built-in SoundAnalysis classifier
   (measured here: 303 labels including `smoke_detector`, `siren`, `car_horn`) (E.5).

**What is impossible, stated plainly**

- **Under 20 ms over Bluetooth** - impossible; the A2DP path alone is tens of ms or more [22][23].
- **Removing the room entirely** - impossible; ANC leaves about 21 to 25 dB of direct sound [24].
- **Knowing where talkers are from head tracking alone** - impossible; it is learned from enrollment and speech activity.
- **Using AirPods microphones for direction while listening through the phone mic** - not possible; it forces HFP, which
  moves output to the hands-free link and loses stem presses [3].
- **Restarting after a phone call while the app is in the background** - not possible (Apple DTS) [31]; the user taps a
  notification.
- **Reading real output loudness (dB SPL) in real time** - no public API [3]; Earshot relies on its own caps plus iOS
  Reduce Loud Audio.
- **Medical claims** - none. Earshot is a conversation-focus tool, not a hearing aid. FDA OTC hearing aids are a separate
  regulated category (output limit 111 dB SPL, 117 dB SPL with input-controlled compression) [36].

---

## 1. The daily-use story (what has to work)

A Friday dinner, four friends, a loud restaurant (70 to 80 dBA, SNR near +3 to +5 dB at 1 m [4]).

1. **Open Earshot** (or the Action button). AirPods connect; Earshot checks ANC guidance, re-measures delay in 3 s if the
   AirPods were reconnected, and starts in **Everyone** mode with a 400 ms fade-in. The user already hears less clatter.
2. **Add people** - tap "+", hold the phone up like a photo toward Maya while she talks. A ring fills as lip-active
   speech accumulates (5 s target). Green check, "Maya, just for tonight". Repeat for Sam. Each enrollment also stores
   the user's current head direction.
3. **Put the phone on the table**, bottom edge toward the group, screen locked. Earshot keeps running (background audio).
4. **Listen to the table** - triple-press switches Focus/Table. In Table mode Maya and Sam are both kept; the neighbors'
   table is pushed down.
5. **Switch by looking** - the user turns toward Sam and double-presses: a short tone, then only Sam. Switching is
   instant because Sam's extractor was already warm.
6. **Talk** - the user's own voice never comes back delayed (negative voiceprint plus duck).
7. **Something urgent** - a siren or smoke alarm passes through within about 2 to 5 s, with a tone and banner.
8. **A call comes in** - Earshot fades out. After the call it resumes if it can; if iOS refuses (background), a
   notification says "Tap to resume Earshot".
9. **Leave** - "just for tonight" people are deleted after 12 h unless saved.

Everything below exists to make these nine steps reliable.

---

## A. Model

### A.1 Signal path and framing

| Item | Value | Why / source |
|---|---|---|
| Model sample rate | 16 kHz mono | Speech band; all chosen corpora exist at 16 kHz or higher [4] (H) |
| I/O sample rate | 48 kHz (44.1 kHz supported), resampled in the app | Running RemoteIO at a non-hardware rate added large latency in crowd data [3] (M) |
| STFT | N = 512 analysis window (32 ms, 31.25 Hz bins), M = 128 synthesis window (8 ms), hop R = 64 (4 ms), 257 bins | Asymmetric pair keeps frequency resolution at low latency [10][11]; budget about 0 dB gain for it until our ablation [5][12] (H) |
| Algorithmic latency | **8.0 ms**, no lookahead | Verified in simulation: perfect reconstruction (max error 1.3e-15), constant delay of 2 hops = 128 samples (Appendix B.3) (H) |
| Input features | power-law compressed complex spectrum, X_c = abs(X)^0.3 * exp(j angle X), real and imaginary parts | As in [11] and bench models [6] |

**Window pair (exact, so P3 and P4 build the same thing)** - with periodic Hann `H_L[n] = 0.5 - 0.5 cos(2 pi n / L)`:

- Analysis - `w_a[n] = sqrt(H_896[n])` for `0 <= n < 448`, and `w_a[n] = sqrt(H_128[n - 384])` for `448 <= n < 512`
  (a 448-sample rise, then the falling half of a sqrt-Hann over the last 64 samples).
- Synthesis - `w_s[n] = 0` for `n < 384`, and `w_s[n] = H_128[n - 384] / w_a[n]` for `384 <= n < 512`.
- The product is a Hann of length 128 on the last 128 samples, which sums to 1 at hop 64 (checked: 1 +/- 4e-16).
- **Streaming synthesis** - keep a 128-sample overlap-add buffer; each hop add the last 128 samples of the windowed inverse
  FFT, emit the first 64, shift by 64. Emitting from the start of a 512-sample buffer instead gives 28 ms of delay (a bug
  I hit in the simulation; Appendix B.3).

### A.2 Architecture (ESB-16, exact sizes)

K = 16 bands, D = 64, B = 4 blocks. Band widths in bins: `[8]*8 + [16]*4 + [32]*3 + [33]` (0-2 kHz in 250 Hz bands,
2-4 kHz in 500 Hz bands, 4-7 kHz in 1 kHz bands, last band 7-8 kHz plus Nyquist). Every bin keeps its own input feature
and its own mask value; bands only set how many "tokens" the recurrent paths see.

| Stage | Layers (per hop unless noted) | Output |
|---|---|---|
| Band encoder | per band k: Linear(2 w_k -> 64), then LayerNorm(64) | z: 16 x 64 |
| Block 1 | time path: LayerNorm, **LSTM(64 -> 128)** shared across bands (16 parallel states), Linear(128 -> 64), residual. Band path: LayerNorm, **BiLSTM(64 -> 64 per direction)** across the 16 bands, Linear(128 -> 64), residual | 16 x 64 |
| Conditioning (after block 1) | FiLM: z = z * (1 + gamma) + beta, gamma and beta 16 x 64 from MLP([g_target 128, g_self 128, mode 2] -> 256 -> SiLU -> 2048), computed once per voiceprint change. Cross-attention: q = Linear(1024 -> 64) of the flattened frame, 4 heads x 16 over up to 64 tokens (48 target + 16 own-voice, each 64-d, K/V precomputed by Linear(64 -> 128)), out = Linear(64 -> 1024) **zero-initialized**, residual | 16 x 64 |
| Blocks 2-4 | same as block 1 | 16 x 64 |
| Heads | mean over bands, Linear(64 -> 2), sigmoid: p_target (target talking), p_self (user talking) | 2 per hop |
| Decoder | per band: LayerNorm, Linear(64 -> 128), tanh, Linear(128 -> 2 w_k), mask = 1.5 tanh(.) for real and imaginary parts | complex mask, 257 bins |
| Output | S_hat = mask * X (complex multiply), inverse FFT, synthesis window, overlap-add | 64 samples per hop |

**Enrollment encoder (runs once per voiceprint, off the audio thread)** - same STFT size with a 32 ms hop
(non-overlapping), the **shared** band encoder, one non-causal dual-path block (BiLSTM over time 32 per direction, BiLSTM
over bands 32 per direction), then a **masked** average pool of the frames where the person is talking (mask from lip
movement or VAD, B.1) into at most 48 tokens (16 for the user's own voice), token = Linear(1024 -> 64). The voiceprint
vector g = L2-normalize(Linear(64 -> 128)(mean of tokens)). Own-voice tokens get a learned "negative" type embedding.

**Null and absent conventions** - "Everyone" mode uses a learned null g_target and 4 learned null tokens. "No own voice
enrolled" uses a learned null g_self and no self tokens.

### A.3 Size and compute

| Quantity | Value | Source |
|---|---|---|
| Parameters, training graph | 1.94 M total; FiLM generator 0.59 M (runs once per voiceprint); enrollment encoder 0.12 M; speaker-ID head (training only) 128 x number of training speakers, about 0.58 M, not counted | measured here (H) |
| Real-time weights | 4.37 MB fp32 (fits in L2; one copy shared by all instances) | measured here (H) |
| MACs | 11.91 M per 4 ms hop = **2.98 GMAC/s** per instance | analytic, Appendix B.1 |
| C/Accelerate compute per hop, M4 P-core | median 175 to 187 us, p99 0.21 to 0.63 ms, p99.9 0.34 to 1.05 ms (tails are desktop preemption at default QoS under load 5 to 9) | measured here (H, Mac proxy) |
| Same on M4 E-core (background QoS) | median 0.59 to 0.64 ms (15 to 16 % of a hop) | measured here (H, Mac proxy) |
| Wider is nearly free on CPU | width 96/48 took 160 to 172 us vs 175 to 187 us for 128/64; cost is dominated by per-call overhead | measured here (H) |
| Training step, M4 MPS, torch 2.7.1 | **0.98 s/step** (batch 8 x 4 s, two 5 s enrollments, tSNR + 3-resolution STFT + BCE losses); MPS pool 12.1 GB | measured here (H) |
| Same, batch 4 x 8 s (stage 3) | 1.08 s/step | measured here (H) |

Why 16 bands and not 32 - the 32-band version (prior-art's M1 converted to LSTM) measured 1.43 s/step for the
extractor alone and 2.54 s/step with enrollment (MPS pool 19.9 GB, at the memory limit), which would allow only about
60k steps in 42 h. Cost scaled linearly with batch, length, bands and width (Appendix B.2), so 16 bands is the size the
M4 can train well. 20 bands (finer below 1 kHz) is ablation A4 (+20 % cost).

### A.4 Conditioning details

- **Why FiLM plus cross-attention** - in the closest published setup (mono, causal, 8/4 ms framing, noisy enrollment),
  cross-attention reached 8.85 dB SI-SNRi against 6.70 dB for FiLM, with confounds (the FiLM variant was larger and pooled
  to one vector) [9][5] (H). We keep both and ablate (A2).
- **Why a jointly trained internal voiceprint** - PVQE's internal embedding matched a separate 15 M-parameter speaker
  model, was better for small models and over-suppressed 2.7x fewer frames [14] (H); EER does not predict extraction
  quality [18] (H); and VoxCeleb-trained encoder weights are "for research purposes" [5][63], which blocks distribution.
- **Optional frozen WeSpeaker ResNet34** (personal build only) - most robust of 6 encoders on short noisy enrollments
  (EER 4.3 % at 5 dB SNR vs 16.4 % for Resemblyzer) [4] (H). Ablation A3 adds its 256-d vector to the FiLM input with path
  dropout 0.3. Keep it only if it adds at least 0.5 dB SI-SDRi or 3 points of picked rate at SIR 0 dB.
- **Speaker-ID loss on g** (AAM-softmax over training speakers) makes g usable for the app's quality, distinctness and
  "locked" checks without a second model [18] (M).

### A.5 Streaming state (per instance, preallocated)

| State | Size |
|---|---|
| Time-LSTM h and c, 4 blocks x 16 bands x 128 x 2 | 16,384 floats (64 KB) |
| Analysis history 512, overlap-add buffer 128, delayed input for room sound 128 | 768 floats |
| FiLM gamma, beta (16 x 64 x 2); attention K, V (64 x 64 x 2) | 10,240 floats |
| Head smoothing, gate, duck, AGC state | under 64 floats |
| Total | about **110 KB per instance**; the band BiLSTM restarts every frame and needs no state |

Resampler states (128 taps each direction) and the 48 kHz rings belong to the engine, not the instance.

### A.6 Modes

| Mode | Conditioning | Output | Training share (A.6 to F.2) |
|---|---|---|---|
| **Focus** | one person's voiceprint + own voice as negative | that person only | 65 % of examples (55 % target present, 10 % target absent) |
| **Table** | 2 to 3 warm instances, one per person | sum of the instances | none extra (reuses Focus) |
| **Everyone** | learned null + own voice as negative | every talker except the user, noise removed | 25 % |
| **Off / paused** | - | silence (fade 150 ms) | - |

The remaining 10 % of examples are Focus clips with a mid-clip **zero-state start** (state reset at a random point), which
is exactly what a cold switch does in battery-saver mode (E.3).

Why Table mode is separate instances and not one multi-target network - it needs no new training, each person keeps
single-target quality, and switching is instant. Cost is 3x compute (still about 14 % of a P-core). A true multi-target
model (union of token sets) is a v2 experiment.

### A.7 Output post-processing (after the network, per hop at 16 kHz)

Order for each hop, `x_d` = input delayed 128 samples so it lines up with the output:

1. **Activity gate** per instance - gain goes from 0 dB at p_target >= 0.5 down to **-12 dB** at p_target <= 0.2 (linear in
   dB), attack 1 hop, release 200 ms. A safety net for leakage when the person is silent; the network is already trained
   to output silence [16][17].
2. **Sum instances** with crossfade weights (switches use a 150 ms equal-power crossfade).
3. **Own-voice duck** - when p_self > 0.5, -15 dB with 1-hop attack and 150 ms release. Default on for AirPods (hearing
   your own voice 50 to 200 ms late is very disruptive; 200 ms delayed feedback caused the most disfluency in normal
   speakers [28]), default off for wired.
4. **Slow voice AGC** - RMS over 400 ms, measured only while some p_target > 0.5, gain toward -26 dBFS plus the user's
   "voice level" offset (-10 to +6 dB), slew at most 6 dB/s, gain frozen during silence so noise is never pumped up.
   The maximum gain is set from Phase 0's measured mic level for 60 dB SPL speech at 1 m; hard cap 6 dB above that.
5. **Room sound** - y = agc * (s_hat + g_room * (x_d - sum of s_hat_i)), g_room in {off, -24, -15, -8 dB} (DeepFilterNet's
   attenuation-limit idea [59]). Default off on AirPods (the ANC leak already supplies ambience), -24 dB on wired.
6. **Upsample** to 48 kHz and hand to the I/O callback, where the safety chain runs (D.6, E.5).

### A.8 The wired low-latency profile (deferred, and why)

A 16 ms / 4 ms / 2 ms window profile would cut algorithmic latency to 4 ms, but it doubles frames per second (about 2x
training cost, about 2 s per step) and costs about 1 to 1.5 dB at fixed compute [12] (H). The user owns only AirPods, where
4 ms is 2 to 6 % of the total delay. So v1 ships one model (8 ms). The low-latency model is trained only if Phase 0 shows
the phone's wired round trip is at most about 15.9 ms at 64 frames (C.1) **and** the user gets sealed wired earbuds.

### A.9 Alternatives considered and rejected

| Option | Why not (evidence) |
|---|---|
| GRU dual-path (prior-art M1) | No MPS GRU kernel in torch 2.7.1; GRU 11 to 14x slower than LSTM on MPS [5][6] (H) |
| 32-band LSTM | 2.54 s/step measured here, MPS memory at the limit (H) |
| Conv-TasNet TCN | 0.64 to 1.3 s/step [6]; worse quality per FLOP at equal latency in [11]; PyTorch-style CPU inference 1.4 to 1.8 ms per hop [6] (H) |
| Full-band STFT-LSTM | Cheapest (0.23 s/step [6]) but full-band RNNs separate voices poorly [2] (M); kept as a debugging baseline |
| Core ML on the audio thread | Not real-time safe; on these graphs Core ML put 100 % of ops on the CPU anyway and ran 5 to 17x slower than C [6] (H) |
| BNNS Graph as v1 runtime | Apple's recommended real-time CPU path [33] (H) but op coverage of our exported graph and timing at our size are unpublished; evaluated in Phase 0 as an optional backend, adopted only if it saves at least 30 % energy at equal p99.9 |

---

## B. Enrollment

### B.1 Capture flows

| Flow | When | How it works |
|---|---|---|
| **Camera point** (default) | adding someone at the table | Back camera preview (video only, no audio input, so Earshot's audio session is untouched). Vision face landmarks (`VNFaceLandmarks2D` inner/outer lips) [51] on the most central face at 15 to 30 fps; "talking" = lip-opening variance over 200 ms above a threshold. Audio comes from Earshot's own mic stream (same mic as listening, which reduces enrollment/listening mismatch [2]); video and audio are aligned by host time (CMSampleBuffer PTS vs AudioTimeStamp.mHostTime). Capture stops at 5 s of lip-active speech or 12 s total. |
| **Phone point** (no camera permission, or dark room) | same | Hold the phone toward the person; an energy/spectral VAD makes the speech mask. Weaker: interferers that talk during the clip get in. |
| **Quick add** | "add who just spoke" | Earshot keeps an 8 s RAM-only ring of mic audio (never written to disk). The user taps "Add who just spoke"; Earshot finds the dominant talker segment, plays 2 s of it back ("Is this them?"), then enrolls. Rejected if the 1.5 s window embeddings form two clusters. |
| **Own voice** | onboarding, once | Read a 20 to 30 s passage in a quiet place with the phone held normally (at least 10 s of speech). Mandatory before AirPods listening. |

At every enrollment Earshot also records the **current head yaw** (if AirPods motion is available) as that person's
direction for this table session (E.2).

### B.2 Encoder and training match

- The voiceprint is the internal encoder's tokens plus g (A.2), computed in the C engine on a background thread (157 frames
  for 5 s, about a millisecond of CPU, estimate).
- Training matches the app - the enrollment speech mask is 40 % oracle (target VAD dilated 80 ms, 10 % frames flipped),
  30 % energy VAD, 30 % none; enrollments are 80 % noisy (0 to 25 dB SNR), 30 % with one weaker competing talker 5 to 15 dB
  below, and use the same mic-chain augmentation as the mixture (F.2). Models trained only on clean enrollment collapsed on
  noisy enrollment (-0.03 dB) [9] (H).

### B.3 Length and quality checks (initial thresholds, calibrated on validation in G)

| Check | Pass (green) | Warn (yellow) | Fail (red) | Basis |
|---|---|---|---|---|
| Lip/VAD-active speech | >= 5 s | 3 to 5 s | < 3 s | 5 s roughly halved speaker-encoder EER vs 3 s at 5 dB SNR [4] (H); 89 % of LOTH users found 5 s acceptable [8] (H) |
| Speech-to-background ratio (masked vs unmasked frames) | >= 6 dB | 0 to 6 dB | < 0 dB | every encoder degraded sharply at 0 dB [4] (H) |
| Self-consistency (mean cosine of 1.5 s window g to clip g) | >= 0.6 | 0.4 to 0.6 | < 0.4 | initial guess, calibrate |
| Not you | cos(g, g_self) < 0.5 | 0.5 to 0.6 | >= 0.6 | own voice must stay a negative |
| Distinct from saved people | cos < 0.75 | >= 0.75: "sounds like Sam, keep both?" | - | similar voices cause confusion [13] |
| Clipping | < 0.1 % of samples | - | >= 1 % | mic overload |

Yellow results are accepted with "will improve as they talk" (B.5).

### B.4 People library and storage

- **Kinds** - "Just for tonight" (default; deleted 12 h after the session ends) and "Saved" (named, persistent).
- **Tables** - a table session holds up to 3 warm people plus their head directions; more people can be enrolled and are
  switched in cold (E.3).
- **Voiceprint file (version 1)** - little-endian: magic `ESVP`, version, `model_id[32]` (hash of the export bundle;
  voiceprints are model-specific), `g[128]`, `n_tokens` (uint16), `tokens[n][64]`, quality, speech seconds, SNR dB,
  created and updated times. About 13 KB per person.
- **Enrollment clips** - for "Saved" people only, keep up to 5 clips of 6 s (16-bit 16 kHz, about 190 KB each) so
  voiceprints can be rebuilt when the model changes. Off switch in Settings ("Keep voice samples").
- **Privacy** - everything on-device; Data Protection class "complete until first user authentication" so a locked phone
  can still load them; nothing leaves the phone. Before saving someone, the app asks the user to get their OK. A voiceprint
  is a biometric identifier under Illinois BIPA [39] (H), which is one reason "just for tonight" is the default. Earshot
  never stores conversation audio unless the user turns on a debug recording, and recording others needs their consent in
  all-party-consent states such as Pennsylvania [40] (M; not legal advice).

### B.5 Refreshing a voiceprint over time

- **Within a session** - every 30 s, if an instance produced at least 8 s of output with p_target > 0.8 and its "locked"
  check passed (E.3), recompute tokens and g from that cleaned output and blend: keep 32 original tokens plus 16 new ones,
  g = normalize(0.8 g_orig + 0.2 g_new). **Anchor rule** - the blended g must keep cosine >= 0.9 with the original
  enrollment g, otherwise the update is discarded (prevents drifting onto another person).
- **Across sessions (Saved people)** - after a session with at least 20 s of locked speech, store a new 6 s clip (keep the
  best 5 by quality). The voiceprint uses the best 2 clips' tokens and the mean g of all 5.
- **Re-enroll prompts** - voiceprint older than 90 days, or the median session lock score below threshold in 3 sessions.
- **Model upgrade** - rebuild from stored clips; without clips, mark "re-enroll needed".

---

## C. Latency budgets (mic to ear)

### C.1 Wired sealed USB-C earbuds, v1 model

| # | Stage | ms | Kind and source |
|---|---|---|---|
| 1 | Phone audio hardware plus both 64-frame I/O buffers (mic in to headphone out round trip) | **17** planning (crowd range 9 to 27) | measured by others: the 9 iPhone 15/16-class rows of the Superpowered table read 9, 10, 15, 16, 17, 17, 17, 17, 27 ms, speaker path, iOS 17-18 [21][5] (M). USB-C converter delay is not in it; Phase 0 replaces this line |
| 1a | of which the two I/O buffers | 2.67 | doc arithmetic, 2 x 64 / 48,000 [30] (H) |
| 2 | Decimator 48 -> 16 kHz, 128-tap linear-phase Kaiser FIR | 1.32 | analytic: group delay 63.5 samples; designed here, flat to 6.5 kHz (+/- 0.01 dB), -3 dB at 7.25 kHz, at least 51 dB down above 8 kHz (Appendix B.4) (H) |
| 3 | STFT framing (asymmetric windows, includes the 4 ms hop wait) | 8.00 | analytic, verified in simulation (A.1) (H) |
| 4 | Compute slack (worker deadline = 1 I/O buffer) | 1.33 | design; compute itself is 0.18 ms median on the M4 proxy (A.3) |
| 5 | Interpolator 16 -> 48 kHz | 1.32 | analytic (same filter) |
| | **Total** | **29.0** at the 17 ms planning value; **21.0** at the best crowd row (9 ms) | sum |

**Reading it** - the v1 model meets 20 ms only if line 1 is at most 8 ms, which no iPhone 15/16-class crowd row shows.
The deferred low-latency profile (4 ms framing, 32-frame buffers saving 1.33 ms, 0.67 ms slack, about 0.5 ms of
minimum-phase resamplers, estimate) gives about **21 ms at the 17 ms planning value and 13 ms at 9 ms**. So **20 ms wired
is possible only on a fast unit and is not promised**; Phase 0 decides. With sealed earbuds (at least about 20 dB passive
isolation) the direct sound is far below the processed sound, so comb filtering at 21 to 29 ms is weak; with open EarPods
(about 3 dB isolation [25]) it is not.

### C.2 AirPods Pro 2 over A2DP, v1 model

| # | Stage | ms | Kind and source |
|---|---|---|---|
| 1 | Built-in mic, ADC and iOS input path | 2.4 | reported `inputLatency` for the built-in mic with AirPods output, iPhone 12 mini [23] (M); measured inside the loop in Phase 0 |
| 2 | Input I/O buffer (64 frames) | 1.33 | doc arithmetic [30] (H); granted value is read back |
| 3 | Decimator | 1.32 | analytic (C.1 line 2) |
| 4 | STFT framing | 8.00 | analytic (A.1) |
| 5 | Compute slack (adaptive, starts at 1 I/O buffer, may grow to 4 ms after underruns) | 1.33 (max 4.0) | design (D.3) |
| 6 | Interpolator | 1.32 | analytic |
| 7 | Output I/O buffer | 1.33 | doc arithmetic |
| | **Earshot's share (lines 1-7)** | **17.0** (19.7 at maximum slack) | sum |
| 8 | Bluetooth A2DP (iOS encode, radio, AirPods decode and playout) | **unknown, planning 45 to 200** | measured by others: AirPods Pro 2 were 43 ms slower than the built-in speaker in a tap-to-sound test (126 vs 83 ms), Pro 1 84 ms slower [22][5] (M); AirPods 2 on an iPhone measured 215-220 ms mic-to-ear when fresh, drifting to 155-160 ms after 20-30 min [23] (M). No one has measured this exact path |
| | **Total** | **about 62 to 217 ms planning; shown as the measured value** | sum |

What the numbers mean for daily use

- **Lip sync** - broadcast guidance puts the detectability threshold for late audio at about 125 ms and the acceptability
  limit at about 185 ms [26] (M), and audiovisual speech still fuses up to about 200 ms of audio lag [27] (M). Expect
  "slightly off lips" on AirPods Pro 2 and more on Pro 1. Earshot shows the number and recommends Pro 2.
- **Own voice** - delayed own voice is the most disruptive effect [28] (M), so own-voice suppression plus the duck are
  mandatory on AirPods.
- **Why not shave Earshot's 17 ms further** - the 4 ms framing profile would save 4 ms (2 to 6 % of the total) at a cost of
  about 1 to 1.5 dB [12]; minimum-phase resamplers could save about 2 ms and are a v1.1 option once measured.

### C.3 Choices that keep the AirPods number low

- Session `.playAndRecord`, mode `.measurement`, options `[.allowBluetoothA2DP]` only (no HFP, no `.mixWithOthers`) [3][30]
  (H). HFP and the iOS 26 high-quality Bluetooth recording option both add latency or move the mic [3] (H).
- Ask the user to set **Spatial Audio Off** for Earshot (auto-enabled for Now Playing AURemoteIO apps; latency effect
  unknown) and **Conversation Awareness Off** (it lowers media volume when the user talks and may pause audio) [3][46] (M).
- Game Mode (`LSSupportsGameMode`) may lower AirPods latency but needs a full-screen foreground game [3] (M); it is a
  personal-build experiment only, measured in Phase 0, never a default.

### C.4 In-app latency measurement

- **Loop** - the worker replaces the model output with a 0.3 s exponential sweep (300 Hz to 6 kHz, -30 dBFS) injected at
  48 kHz after the interpolator; the mic signal before the decimator is cross-correlated (GCC-PHAT) on a background
  thread. 5 repeats, median, spread must be under 2 ms (wired) or 10 ms (Bluetooth). Mouth to ear = measured loop + 8.0 ms
  framing + 2.65 ms resampling.
- **Wired** - "Hold an earbud against the bottom of the phone for 3 seconds."
- **AirPods** - preferred: "Hold the bottom of the phone against your right AirPod (keep it in your ear) for 3 seconds";
  the in-ear leakage of the sweep should be detectable thanks to the sweep's processing gain (unverified, Phase 0 test P1-3).
  Fallback: Automatic Ear Detection off, hold a removed AirPod to the mic. Removing one AirPod normally pauses playback
  [54] (M), which is why the in-ear method comes first.
- **When** - every new route, every AirPods connect, on demand, and a reminder after 30 min (AirPods 2 drifted by about
  60 ms in 20-30 min [23]).
- **Display** - "Delay 74 ms (measured 3 min ago): Earshot 17 + Bluetooth 57". If the loop fails, show the reported
  `inputLatency + outputLatency + 2 x ioBufferDuration` labeled "estimate" [3]. Log both.

---

## D. iOS real-time engine

### D.1 Audio session (main thread, foreground only)

```swift
let s = AVAudioSession.sharedInstance()
try s.setCategory(.playAndRecord, mode: .measurement, options: [.allowBluetoothA2DP])   // never HFP, speaker or mixable
try s.setPreferredSampleRate(48_000)
try s.setPreferredIOBufferDuration(64.0 / 48_000.0)
try s.setPrefersNoInterruptionsFromSystemAlerts(true)
try s.setAllowHapticsAndSystemSoundsDuringRecording(true)   // switch confirmations
try s.setActive(true)
if let mic = s.availableInputs?.first(where: { $0.portType == .builtInMic }) { try s.setPreferredInput(mic) }
// read back: sampleRate, ioBufferDuration, inputLatency, outputLatency, currentRoute; warn if buffer > requested
```

- If the granted buffer is larger than requested, warn: iOS 18 silently used 1104-frame buffers when Sound Recognition or
  Vocal Shortcuts was on [56] (M). Earshot has its own alarm detector, so the user can turn Sound Recognition off.
- Phase 0 decides whether `.measurement` coexists with A2DP and whether a back-mic data source is honored [3] (L); until
  then Earshot uses the primary mic, which is also the training assumption.

### D.2 RemoteIO and the render callback (C, no Objective-C or Swift inside)

- `kAudioUnitSubType_RemoteIO`, input enabled on bus 1, Float32 mono non-interleaved at the session rate on both sides,
  `kAudioUnitProperty_MaximumFramesPerSlice = 4096` so buffer inflation never fails.
- Callback, in order - `AudioUnitRender` bus 1 into a preallocated buffer; write to the input SPSC ring (16,384 frames);
  `semaphore_signal` the worker; read up to n frames from the output ring; if short, **conceal** (repeat the last
  64-frame block at -6 dB, then fade to zero over 2 blocks, count an underrun); run the safety chain (D.6); copy to the
  second channel if the output is stereo.
- Workgroup - read `kAudioOutputUnitProperty_OSWorkgroup` (pass `sizeof(os_workgroup_t)` and check the result [3]).

### D.3 Worker thread (the only place the network runs)

- One pthread, `THREAD_TIME_CONSTRAINT_POLICY` with period 4 ms, computation 1 ms, constraint equal to the slack, joined to
  the RemoteIO workgroup with `os_workgroup_join` [34] (H). `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)` on the
  thread [34] (H). Flush-to-zero for denormals (`fesetenv(FE_DFL_DISABLE_DENORMS_ENV)` on arm64, verify in P4 tests) (M).
- Loop - wait on the semaphore; while at least 192 input frames are queued, run `es_process_hop` (decimate, STFT, every
  active instance, post-processing, interpolate) and push 192 output frames.
- **Adaptive slack** - the output ring is pre-filled with `slack` frames (default 64 = 1.33 ms). Two underruns within 60 s
  raise slack by 64 frames (to at most 192 = 4 ms), applied during a moment with all p_target below 0.2 by inserting a
  64-frame crossfaded repeat; 10 quiet minutes lower it again. Every change is logged.
- **Budget rule** - p99.9 of `es_process_hop` must stay below 50 % of a hop (2 ms) on the phone (G.5). Measured headroom on
  the M4 proxy is about 20x on a P-core and 6x on an E-core for one instance.

### D.4 Resampling

- Rational polyphase FIR in C; 48 kHz (ratio 1/3) and 44.1 kHz (160/441) tables precomputed at setup. 128-tap Kaiser
  (beta 5.0, cutoff 7.4 kHz), group delay 1.32 ms each way (Appendix B.4).
- Training applies the same decimator magnitude response to mixtures and targets, so the model learns a neutral mapping
  for the 6.5 to 8 kHz roll-off (F.2).

### D.5 Inference runtime - hand-written C with Accelerate

- **Why** - the measured per-hop cost is tiny (A.3), every line is readable, there is no allocation after setup, and parity
  with PyTorch is checked by golden vectors. Core ML stays off the live path [6]; BNNS Graph is a Phase 0 comparison (A.9).
- **Kernels** - `cblas_sgemm` with weights stored pre-transposed `[in, out]` (the PyTorch layout was 1.1 to 4x slower at
  these shapes [6]); the 16 band rows of the time LSTM in one GEMM; band BiLSTM input projections batched, recurrence as
  16 small GEMVs per direction; `vDSP_fft_zrip` for the 512-point FFT; vForce `vvexpf`/`vvtanhf` for gates; one-call-per-row
  rule for 2 to 3 rows with large matrices [6].
- **Size target** - engine core under about 1,500 lines of C (P4).

### D.6 Real-time safety rules and how they are tested

| Rule | Test (P4/P5) |
|---|---|
| No malloc/free, no locks, no Objective-C/Swift, no logging, no file I/O on the callback or worker | macOS harness runs 100k hops under a malloc-counting interposer: must count 0; code review checklist; `os_signpost` only outside RT threads |
| Bounded time | per-hop timing histogram on device (G.5) |
| Lock-free communication | C11 atomics for parameters (double-buffered structs swapped at hop boundaries), SPSC rings for audio and stats |
| Voiceprint changes are atomic | new FiLM/K/V buffers are written into the inactive slot off-thread, then one atomic index swap |
| Safety chain always runs last, in the callback | applies to concealment output and to every route state |

### D.7 Route changes

1. `routeChangeNotification` (main thread) sets an atomic mute; the callback ramps to zero in 5 ms.
2. On `.oldDeviceUnavailable` (unplug, AirPods out of range) stay muted. Allow output only to `.headphones`, `.usbAudio`
   or `.bluetoothA2DP`; never `.builtInSpeaker` or `.builtInReceiver` (feedback howl risk) [3] (H).
3. Re-assert `setPreferredInput(builtInMic)` (USB-C earbuds bring their own mic), re-read rate, buffer and latencies; if
   the rate changed, rebuild RemoteIO and the resampler tables.
4. Re-measure latency (prompt on AirPods, automatic on wired), update the display, unmute with a 400 ms fade-in.
5. AirPods taken by another device (automatic switching) look like a route loss; onboarding sets "Connect to This iPhone -
   When Last Connected" [45] (H).

### D.8 Interruptions, background, lifecycle

- `UIBackgroundModes` `audio`; the session starts only in the foreground and keeps running with the screen locked; with
  input running the I/O size does not grow on lock [3] (H).
- Interruption began - mute and mark inactive. Ended with `.shouldResume` - reactivate and fade in. If reactivation fails
  in the background (expected per Apple DTS [31]), post a local notification "Earshot paused by a call. Tap to resume."
- `mediaServicesWereResetNotification` - rebuild session, unit, worker and engine.
- iOS 27 lifecycle notifications replace the interruption notification later [3]; handle both when the SDK moves.

### D.9 Thermal, battery and heat

- **Estimate (L)** - one instance runs about 4.7 % of a P-core-equivalent; with Geekerwan's A18 Pro figures (P-core about
  6 W at full SPEC load, E-core about 0.24 W, secondary source [42]) the network costs roughly 0.05 to 0.15 W. The audio
  path baseline is about 0.16 W (Apple rates 85 h of audio playback [41] on a 3,582 mAh, about 13.8 Wh battery [41]).
  Total about 0.3 to 0.5 W, so **about 3 to 4 % of battery per hour**. Must be measured (G.5).
- **Tools** - Xcode 26 Power Profiler (system and per-app power impact on a connected iPhone) [43] (H), plus battery level
  over 2 h with the screen locked.
- **Downshift** - on `ProcessInfo.thermalState == .serious` or Low Power Mode: keep only the selected person warm, run the
  alarm classifier every 3 s instead of continuously; on `.critical` fade to pause with a notification.

### D.10 Head tracking and remote commands plumbing

- `CMHeadphoneMotionManager` on its own `OperationQueue`; needs only `NSMotionUsageDescription` (the app crashes without
  it) and no entitlement [32][5] (H). The handler writes timestamp, yaw, pitch and rotation rate into an atomic struct;
  the update rate is computed from timestamps (about 25 Hz reported, undocumented) [3] (L).
- `MPRemoteCommandCenter`: `togglePlayPauseCommand`, `nextTrackCommand`, `previousTrackCommand`; `MPNowPlayingInfoCenter`
  title "Earshot - Maya", playback rate 1. A2DP stem presses arrive only this way; HFP presses never reach apps [31] (H).
- **Controls and Action button** - a `ControlWidget` (iOS 18 Controls API) offers "Add person in front" and "Earshot
  on/off" in Control Center, the Lock Screen and the Action button [48] (H); needs a widget extension and an App Group
  (both allowed on a free Personal Team [3]).

### D.11 Telemetry and logs (no audio)

Per-minute JSONL in the app's Documents (schema in H, I7): route, granted buffer, rates, measured and reported latency,
mode, number of warm instances, compute p50/p99/max, underruns, slack, thermal state, battery level, output RMS,
limiter gain reduction, mean p_target and lock score per person, own-voice duck time, alarms fired. A user-enabled
"debug recording" saves 60 s of input and output locally, with a consent reminder (B.4).

---

## E. UX

### E.1 Principles

- **Hands-free first** - every common action works from the AirPods stem with the phone locked.
- **Always say what it is doing** - spoken name or tone on switches, a "locked on" indicator, measured delay.
- **Fail quiet, never loud** - any uncertainty mutes or fades; nothing ever jumps in level.
- **Honest copy** - "focus", "push down", never "remove", "hearing aid" or "under 20 ms" on Bluetooth.

### E.2 Picking the person you're looking at

What exists - per person, a stored head direction `dir_i` (yaw at enrollment, AirPods only). What does not exist - any
way to know talker positions from the phone mic.

**Look-select (double-press)**

1. Read the current yaw `y` (smoothed over 200 ms).
2. Candidate = the person with the smallest angular distance `abs(wrap(y - dir_i))` among the table's people.
3. If that distance is at most 35 degrees and the candidate differs from the current target by at least 10 degrees more
   than the current one (hysteresis), switch to the candidate. Otherwise cycle to the next person in table order.
4. Confirmation - a 60 ms tone pattern (1, 2 or 3 short beeps for table slots 1 to 3) and, if enabled, the person's name
   pre-rendered with `AVSpeechSynthesizer.write(_:toBufferCallback:)` [52] and mixed from a preallocated buffer at -12 dB.

**Learning directions (so drift and undershoot do not matter)**

- Head movements undershoot talker directions, especially beyond 60 degrees, and people look away in groups [47] (M), so
  `dir_i` is the user's habitual head yaw, not the true angle.
- While person i talks (p_i > 0.7 for 1 s) and the head is steady (rotation rate under 20 degrees/s), update
  `dir_i += 0.1 * wrap(y - dir_i)`.
- If yaw jumps more than 45 degrees in under 50 ms (reference reset, AirPods reconnect), mark directions stale: double-press
  cycles until the user double-presses while facing a talking person, which re-anchors that person and shifts the others
  by the same offset.

**Follow my gaze (auto, opt-in, off by default)** - switch when the head stays within 15 degrees of another person's
direction for 1.2 s and that person's warm instance shows speech (p > 0.5) within the last 2 s, and never while the current
target is talking (p > 0.6 in the last 300 ms). Off by default because looking away is normal in conversation [47]; the
pilot (G.6) decides.

**Without AirPods motion** (wired, or motion unavailable when locked - unverified [3]) - switching is by tapping the
person's card, the Lock Screen skip button or the earbud remote, cycling in table order.

### E.3 Switching people

| Situation | What happens | Expected settle |
|---|---|---|
| Person is in the table's warm set (up to 3) | 150 ms equal-power crossfade between instance outputs | instant (state already adapted) |
| Person not warm (more than 3 people, battery saver, thermal downshift) | load voiceprint into the selected instance, reset its state, 300 ms fade-in | about 0.3 to 0.5 s (estimate; measured in G) |
| "Locked on" check | every 1 s the enrollment encoder runs on the last 3 s of each instance's output; cosine to the person's g at or above the threshold shows a check mark | threshold set at 1 % false accepts on validation |
| Lock lost for 5 s while p_target is high | banner "Might be following someone else - double-press or re-add" | - |

### E.4 Controls

| Control | Where | Values (default) |
|---|---|---|
| Mode | main screen, triple-press | Focus / Table / Everyone (Everyone until someone is enrolled) |
| Person | cards, double-press, Lock Screen skip | table order |
| Pause | stem single press, Lock Screen | fade 150 ms |
| Voice level | slider | -10 to +6 dB around the AGC target (0) |
| Room sound | segmented | Off / Low -24 dB / Medium -15 dB / High -8 dB (AirPods Off, wired Low) |
| Mute while I talk | toggle | AirPods on, wired off |
| Alerts passthrough | toggle | on |
| Delay | chip on main screen | measured value, tap to re-measure |

**Stem and remote mapping**

| Gesture | AirPods Pro 1/2 | Wired USB-C remote | Lock Screen | Earshot action |
|---|---|---|---|---|
| Single press | play/pause | single click | play/pause | pause / resume |
| Double press | next track | double click | skip forward | look-select (E.2) |
| Triple press | previous track | triple click | skip back | Focus <-> Table (Focus <-> Everyone with fewer than 2 people) |
| Press and hold | system noise control (not available to apps) [3] | - | - | tip: hold to switch to Transparency, single-press to pause Earshot first |

### E.5 Safety

| Layer | Spec | Where |
|---|---|---|
| Fade-in | 400 ms raised cosine at start, resume, route change, interruption end | I/O callback |
| Mute | 5 ms ramp to zero on route loss, speaker/receiver route, failed calibration, NaN/Inf from the engine | I/O callback |
| Gain cap | total digital gain from mic to output never exceeds the Phase 0 calibrated maximum (60 dB SPL speech at 1 m to target level, plus 6 dB) | engine |
| AGC | slow (6 dB/s max slew), frozen in silence | engine (A.7) |
| Limiter | zero-lookahead peak limiter (no added latency), ceiling -3 dBFS, release 50 ms, then a tanh soft clip above -1 dBFS | I/O callback |
| Howl guard | a narrowband peak growing more than 6 dB per 100 ms for 300 ms ducks the output by 12 dB (wired open earbuds held near the phone) | engine |
| Alarm passthrough | `SNClassifySoundRequest(.version1)` on a background queue fed from a ring copy of the 16 kHz mic signal; labels `smoke_detector`, `siren`, `police_siren`, `ambulance_siren`, `fire_engine_siren`, `civil_defense_siren`, `emergency_vehicle`, `car_horn`, `air_horn`, `train_horn`, `screaming`, `glass_breaking`, `baby_crying` (all present in the 303-label list, measured here); 3 s windows with 50 % overlap, so detection takes about 2 to 5 s; confidence at least 0.6 in 2 consecutive windows mixes the raw mic in at -6 dB with a tone, haptic and banner | background thread + engine flag |
| Exposure | onboarding asks to turn on Reduce Loud Audio (85 dB) [38]; WHO-ITU safe listening is 80 dB(A) for 40 h per week for adults [37] (H) | iOS |
| No medical claims | copy reviewed against the FDA OTC distinction [36] | product |

The alarm detector is a helper, not a safety device; the copy says so.

### E.6 Onboarding (about 4 minutes, once)

1. **What it is** - one screen: what Earshot does, that it is not a hearing aid, that AirPods add a delay.
2. **Permissions** - microphone, motion (head tracking), notifications (resume after calls), camera (optional, for
   enrollment).
3. **AirPods setup checklist**, each with a "how" sheet - Noise Control: **Noise Cancellation** (not Transparency or
   Adaptive; optional Shortcuts automation "When AirPods connect, Set Noise Control Mode" [55] (M)); **Conversation
   Awareness off** [46]; **Personalized Volume off**; **Connect to This iPhone: When Last Connected** [45]; **Spatial Audio
   off** for Earshot (Control Center); **Sound Recognition off** while using Earshot [56]; **Reduce Loud Audio on** [38].
   Optional - if the user took the AirPods Pro 2 Hearing Test, Media Assist personalizes music, video and calls [50] (H);
   whether it applies to Earshot's audio is unverified.
4. **Your voice** - read the passage (B.1). "So Earshot never plays your own voice back to you."
5. **Measure the delay** - C.4. Result screen with the number and one sentence on lip sync.
6. **First listen** - 10 s of Everyone mode with an A/B toggle to hear the difference.
7. **Add a person** - practice on a friend or a podcast on a laptop.
8. **Controls card** - the stem mapping table, also kept in Help.

### E.7 What differs on AirPods and wired

| Topic | AirPods Pro 2 / Pro 1 | Wired sealed USB-C earbuds |
|---|---|---|
| Delay | 17 ms + measured Bluetooth (45 to 200 ms planning); Pro 2 better than Pro 1 [22] | 21 to 29 ms expected (C.1) |
| Room leak | about -24.5 dB (Pro 2), -21 dB (Pro 1) isolation [24] | earbud-dependent; EarPods (open fit, about -3 dB [25]) not supported for focus |
| Own voice | negative voiceprint + duck on | negative voiceprint, duck optional |
| Room sound default | Off | Low (-24 dB) |
| Head tracking | yes (Pro 1 and Pro 2) [32] | no; tap or remote cycling |
| Stem/remote | AVRCP via Now Playing | inline remote, same mapping |
| Calibration | in-ear leakage loop (unverified) or ear detection off | earbud to bottom mic |
| Battery limit | AirPods about 6 h with ANC [44] | phone only |
| Extra guidance | ANC, Conversation Awareness, automatic switching, Spatial Audio | force built-in mic (earbud mic ignored) |

### E.8 Screens

- **Main** - big card for who is playing (name, color, lip/voice activity dot per table person, "locked" check), mode
  switch, delay chip, voice level, room sound. Large touch targets; readable at arm's length on a table.
- **Add person** - camera viewfinder with a face box, lip-activity ring, speech-seconds counter, quality result.
- **People** - saved people, table history, delete, rename, "keep voice samples" toggle.
- **Help** - stem mapping, AirPods checklist, what the delay means.

### E.9 Accessibility

Dynamic Type, VoiceOver labels on every control (but note VoiceOver speech plays into the same output), high-contrast
colors, haptics for switches when the phone is in hand.

---

## F. Data and training plan (fits the M4 and the disk)

Disk now - 41 GiB free on 2026-11-07 (measured with `df`), so the full v1 pack (about 9.3 GB) fits while keeping well
above the 10 GB floor. The `dataprep/` package (manifest schema in `dataprep/common.py`) is already being built.

### F.1 Datasets (from `data-training.md` [4], licenses checked there and in [5])

| Part | Source | License | Keep | Disk | Use |
|---|---|---|---|---|---|
| Speech | LibriTTS-R train (2,311 speakers, streamed from 82 GB of tarballs) | CC BY 4.0 | 60 s per speaker, at least 2 chapters | 2.6 GB | targets 60 %, interferers, babble |
| Speech | VoxPopuli en, 2 shards (about 800 speakers) | CC0 | 60 s per speaker | 0.9 GB | targets 15 %, accents |
| Speech | GLOBE V2, 8 shards (about 1,300 speakers) | CC0 | all non-US rows, US capped at 30 s | 0.9 GB | targets 15 %; listen to 20 clips first (Adobe-enhanced) |
| Speech | VCTK 0.92 mic1, 102 train speakers (8 held out) | CC BY 4.0 | 2 min per speaker by range request | 0.25 GB | targets 5 %, accents |
| Speech (optional) | Emilia-YODAS EN, 1 tar | CC BY 4.0, gated (user accepts terms) | DNSMOS >= 3.2 | 0.4 GB | interferers, babble |
| Noise | FSD50K CC0 + CC BY clips: chatter, crowd, cutlery, dishes, glass, laughter, applause, music | CC0 / CC BY (attribution list in credits) | restaurant classes + 4 h music | 1.0 GB | noise |
| Noise | DEMAND 8 scenes (PCAFETER, PRESTO, OMEETING, PSTATION, SPSQUARE, OHALLWAY, TBUS, TMETRO), 4 channels | CC BY-SA 3.0 per the PDF | all | 0.3 GB | noise; SCAFE held out |
| Noise | MUSAN noise + 6 h music | CC BY 4.0 / commercial-safe per paper | noise, music subset | 0.9 GB | noise, music |
| Babble bank | 1,000 x 10 s renders of 6 to 20 training talkers, each through its own RIR | inherits speech licenses | generated | 0.2 GB | babble (cheaper than 20 crops per example) |
| RIRs | pyroomacoustics bank, 4,000 rooms x 7 sources, realized RT60 measured | MIT tool | generated | 0.7 GB | 75 % of examples |
| RIRs | OpenSLR 28 real RIRs | Apache-2.0 | 417 files, 1.5 s, float16 | 0.3 GB | 15 % of examples |
| Test only | LibriSpeech test-clean (on disk), WHAM! tt (CC BY-NC, eval only), EdAcc test (CC BY-SA), VoxPopuli en_accented test (CC0), VOiCES subset (CC BY 4.0), Lombard Grid (CC BY 4.0), MIT IR Survey bar/restaurant IRs (CC BY per mirrors), DEMAND SCAFE | as listed | subsets | about 1.0 GB | evaluation |
| **Total** | | | | **about 9.3 GB kept, about 119 GB downloaded (about 3 h at 11 MB/s)** | |

Speaker-disjoint splits - 3 % of training speakers per source for validation; official test speakers for testing; test
speakers never appear as interferers or babble. The Mixer refuses any manifest row whose license is non-commercial unless
the split is `test`.

### F.2 Mixture recipe (one 4 s example, dynamic mixing, seeded per index)

| Element | Value |
|---|---|
| Mode | Focus 65 % (target present 55 %, absent 10 %), Everyone 25 %, Focus with mid-clip zero-state start 10 % |
| Competing talkers | 0 (10 %), 1 (35 %), 2 (30 %), 3 (25 %), each active 50 to 100 % of the clip |
| Levels | SIR target vs loudest interferer uniform -8 to +12 dB (quieter targets are where personalized models fail: 6 % picked [13]) |
| Own voice ("user") | present in 50 % of examples, a different speaker 0.25 to 0.6 m from the mic (in hand) or 0.4 to 0.8 m (on table), 0 to +12 dB above the target at the mic, active 20 to 60 %; never in the reference |
| Own-voice enrollment given | 85 % of examples (also when the user is silent); 15 % null-self, and those never contain an own-voice talker |
| Babble | 70 %, from the babble bank |
| Noise | 80 %, 1 or 2 clips (DEMAND 30 %, FSD50K 40 %, MUSAN 20 %, other 10 %) |
| Music | 20 %, 5 to 20 dB below the target |
| SNR | N(5, 5) dB clipped to -8 to +25 dB; 10 % quiet rooms at 20 to 40 dB [4] |
| Geometry | phone in hand 40 %, on table 40 % (mirrored table image source, reflection 0.6 to 0.9), held up and pointing 10 %, shirt pocket 10 % (low-pass 3 to 5 kHz, -6 to -15 dB above 4 kHz) |
| Rooms | 75 % simulated bank, 15 % real RIRs, 10 % dry |
| Target reference | direct path + first 50 ms of the RIR, 5 ms half-cosine fade, propagation delay kept (no lookahead needed) [4] |
| Mic chain (mixture and reference get the same linear filters) | high-pass 60 to 200 Hz, 0 to 2 peaking EQs +/- 4 dB, the app's decimator response (D.4), self-noise -95 to -75 dBFS, level -40 to -15 dBFS RMS (re-centered on Phase 0's measured iPhone level), 6 % clipped (mixture only) |
| Lombard | 30 % of loud scenes: +3 to +8 dB level and +2 to +4 dB high shelf above 1 kHz on talkers |
| Similarity curriculum | from 40k steps, 30 % of interferers share the target's gender and corpus |
| Enrollment | 1 to 2 other utterances of the target, 3 to 8 s; 80 % noisy (0 to 25 dB SNR); 30 % with a weaker talker 5 to 15 dB below; room same 50 % / other 30 % / dry 20 %; speech mask as in B.2 |
| Own-voice enrollment | 4 to 8 s, 70 % clean, 30 % noisy at 10 to 30 dB SNR |
| Everyone-mode reference | sum of all non-user talkers (direct + early); if none, treated as absent |

The **Mixer output contract** is interface I2 (H). `python -m dataprep.mixer.preview --n 20` renders WAVs for listening
every time the recipe changes.

### F.3 Losses

| Term | Formula | Weight | Applies to |
|---|---|---|---|
| Thresholded SNR | -10 log10(sum s^2 / (sum (s - s_hat)^2 + tau sum s^2)), tau = 1e-3 (30 dB cap) [17][16] | 1 | target present, Everyone |
| Multi-resolution STFT magnitude | L1 on magnitude + L1 on log magnitude at 256/512/1024-point FFTs | 0.5 (check term magnitudes in the first 2k steps) | same |
| Target absent | 10 log10(sum s_hat^2 + tau_abs sum x^2), tau_abs = 10^-2.5 [4][17] | 1 | absent clips |
| Target-silent frames | same as absent, over frames where the target VAD is 0, if at least 0.1 s | 0.1 | present clips |
| Over-suppression | mean ReLU(abs(S)^0.3 - abs(S_hat)^0.3)^2 [15] | 0 until 20k steps, then 0.5 | present clips |
| Activity heads | BCE(p_target, target VAD) + BCE(p_self, own-voice VAD) | 0.1 each | all |
| Speaker ID | AAM-softmax on g (margin 0.2, scale 30) | 0.05 | Focus clips |

All losses in float32. Silence collapse is guarded by the thresholded SNR and by watching mean output energy.

### F.4 Schedule, steps and wall clock (from the measured 0.98 s/step)

| Phase | Steps | Segment x batch | Contents | Wall clock |
|---|---|---|---|---|
| Sanity | - | - | overfit 50 mixtures above 25 dB SI-SDRi in under 15 min; causality test (perturb after t, outputs before t unchanged); streaming = offline to 1e-5 | 1 h |
| Ablations A1-A4 (F.5) | 18k each | 4 s x 8, width 96/48 model (0.86 s/step measured) | short cosine schedule | 4 x 4.3 h = 17 h |
| Stage 1 | 0 to 20k | 4 s x 8 | SNR +5 dB, at most 2 interferers, no absent/Everyone/own-voice talker yet (self enrollment still given) | 5.5 h |
| Stage 2 | 20k to 120k | 4 s x 8 | full recipe, over-suppression term on, similarity curriculum from 40k | 27 h |
| Stage 3 | 120k to 140k | 8 s x 4 (1.08 s/step) | long context for recurrent state; real RIRs 30 %; LR tail | 6 h |
| Validation | every 2k steps | 400 fixed mixtures | 70 runs of about 1 min | 1.2 h |
| **Main run total** | **140k** | | about 1.12 M examples, about 1,250 h of mixtures | **about 40 h + 10 % margin = 44 h** |

- Optimizer - AdamW, peak LR 7e-4 (attention present), weight decay 1e-4, 2,000 warmup steps, cosine to 2.5e-5, gradient
  clip 5, fp32 (LSTM gains nothing from autocast on MPS [6]); EMA of weights (decay 0.999).
- Environment - torch 2.7.1 + coremltools 9.0 (the supported pair [6]); `PYTORCH_ENABLE_MPS_FALLBACK=0`; setting
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` alone fails ("invalid low watermark ratio 1.4", measured here), so also set
  `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8`; `caffeinate -dimsu`; AC power; DataLoader with spawn-safe entry point,
  6 workers, `persistent_workers`, per-worker numpy generators [4].
- Memory - 12.1 GB MPS pool measured, inside the 19 GB recommended working set [6].
- If the first hour of the real run is more than 15 % slower than 1.0 s/step (data loading, other jobs), cut stage 2 to
  80k steps rather than shrink the model.

### F.5 Ablations (run before the main run, decided on the validation set)

| ID | Question | Arms | Decide by |
|---|---|---|---|
| A1 | Does the asymmetric window help this model? | 32/8 ms vs symmetric 8/8 ms at equal compute | SI-SDRi, picked rate |
| A2 | Is cross-attention worth it? | FiLM only vs FiLM + attention | picked rate at SIR -5 and 0 dB |
| A3 | Does a frozen WeSpeaker vector help? | internal only vs internal + WeSpeaker (dropout 0.3) | +0.5 dB or +3 points picked |
| A4 | More bands? | 16 vs 20 bands | SI-SDRi per hour of training |
| Later | over-suppression weight, tau_abs 1e-2 vs 1e-2.5, enrollment mask on/off, gate on/off (listening) | | |

### F.6 Checkpoints and validation

- Save every 2k steps (model, optimizer, scheduler, RNG and sampler state; about 24 MB each); keep the last 5 and the best 3.
- Validation (400 fixed mixtures from validation speakers, rendered once with a fixed seed) reports SI-SDRi, picked rate at
  SIR 0 dB, absent attenuation, TSOS and own-voice suppression. Best checkpoint = highest SI-SDRi among those with absent
  attenuation >= 20 dB and TSOS <= 5 %; export the EMA weights of that checkpoint (and compare with an average of the last 5).

### F.7 Machine plan

Downloads and prep (about 3 h download, about 2 h processing) can overlap with P1/P4/P5 work. Training runs one job at a
time on the GPU (two concurrent runs would fight for the 24 GB). Total machine time for v1 - about 3 days (17 h
ablations + 44 h main run + exports and evaluation).

---

## G. Evaluation plan

### G.1 Metrics

SI-SDRi and SNRi; ESTOI (pystoi); DNSMOS P.835 and personalized DNSMOS; PESQ for internal use only (licensing [4]); Whisper
WER against transcripts; speaker similarity with an independent encoder (SpeechBrain ECAPA, never the conditioning model);
**picked rate** per 2 s chunk (output closer to the target than to any interferer) [13]; **TSOS** (frames where removed
target energy exceeds 0.1 of the reference) [15]; **absent attenuation** 10 log10(sum s_hat^2 / sum x^2); **own-voice
suppression**; **switch settle time** (time until a cold-switched instance reaches 90 % of its steady SI-SDRi);
**stability** over 60 min streams.

### G.2 Test conditions

| Test set | What it checks |
|---|---|
| Earshot restaurant simulation - 2,000 mixtures from held-out speakers, DEMAND SCAFE and WHAM! tt noise, MIT IR Survey bar/restaurant RIRs; grid over SNR (-5, 0, 5, 10 dB), 1 to 3 interferers, own voice on/off, hand/table, clean/noisy enrollment | the main gate |
| Libri2Mix 16 kHz min test, clean, BUT enrollment list (evaluation only) | comparison with the literature |
| Accent sets - EdAcc test, VoxPopuli en_accented | works on every accent |
| Lombard Grid | loud-room speaking style |
| Streaming conversations - 20 to 60 s, target talking 30 to 60 %, interferers overlapping | picked rate over time, gaps, switching |
| Real replayed - VOiCES room with babble; DiPCo 1 session | real acoustics |
| Our loudspeaker rig on the iPhone (G.4) | the real phone, mic and rooms |

### G.3 Acceptance targets (v1 gates; estimates of what is reachable, not predictions)

| Metric | Condition | v1 gate | Stretch | Reference point |
|---|---|---|---|---|
| SI-SDRi median | restaurant sim, all cells | >= 6 dB | >= 8 dB | mono causal TF-GridNet 8.85 dB on 2 talkers + noise [9] |
| SI-SDRi median | restaurant sim, SNR 0 to 5 dB, 1 to 2 interferers, noisy enrollment | >= 7 dB | >= 9 dB | same |
| SI-SDRi median | Libri2Mix clean, causal | >= 9 dB | >= 12 dB | causal BSRNN 6.87 (vector) to 13.05 dB (frame cues) at 32 ms [5] |
| Picked rate | SIR 0 dB | >= 90 % | >= 95 % | PVQE 61 % at equal level [13] |
| Picked rate | SIR -5 dB | >= 75 % | >= 85 % | PVQE 6 % when quieter [13] |
| TSOS | target-active frames | <= 3 % | <= 1.5 % | pDCATTUNET 1.2 % [15] |
| Absent attenuation | median | >= 25 dB | >= 35 dB | PVQE 33-34 dB [14] |
| Absent false alarms | share with attenuation < 10 dB | <= 5 % | <= 2 % | NTT/BUT miss/false trade-off [16] |
| Own-voice suppression | median, own voice enrolled | >= 20 dB | >= 30 dB | prior-art goal [2] |
| Own-voice false pass | 1 s windows with own voice above -10 dB | <= 5 % | <= 2 % | - |
| WER | Whisper, SNR 0 to 5 dB with interferers | >= 30 % relative reduction vs mixture | >= 50 % | DNS 2023 ranks by word accuracy [4] |
| DNSMOS OVRL | restaurant sim, SNR >= 5 dB | >= +0.5 over the mixture | >= +0.8 | Oxford model about 2.1 absolute [9] |
| Switch settle (cold) | streaming | <= 0.5 s | <= 0.3 s | - |
| Stability | 60 min stream | no NaN/Inf; last-5-min SI-SDRi within 1 dB of first 5 min; level drift <= 3 dB | - | - |

### G.4 Real-recording test (loudspeaker rig, no recording of bystanders)

- **Rig** - 3 to 5 small powered speakers: target at mouth height 1 m from the phone; 2 to 3 interferers at 1.5 to 3 m;
  1 to 2 noise speakers aimed at walls; an "own voice" speaker 0.3 m from the phone. Levels: target 65 to 70 dBA at 1 m,
  noise 65 to 75 dBA at the phone (phone SPL app) [4].
- **Signals** - held-out LibriSpeech/VCTK/EdAcc/VoxPopuli-accented speech, WHAM! tt and DEMAND SCAFE noise.
- **Method** - record each source alone with the iPhone in measurement mode (through Earshot's own capture path), then all
  together; the solo recordings are real references; check that their sum matches the joint recording (residual below
  -25 dB) before trusting them [4]. Enroll with the real app flow (camera point at a speaker is not possible, so phone point
  plus VAD mask) while noise plays.
- **Rooms** - quiet room, living room, and a cafe at a quiet hour with permission.
- **Gates** - SI-SDRi median >= 4 dB, picked rate at SIR 0 dB >= 85 %, absent attenuation >= 20 dB, own-voice suppression
  >= 15 dB. The rig also gives the iPhone mic response (log sweep) and the speech level in dBFS used for F.2 and the AGC cap.
- **Listening** - the user plus 3 to 5 friends, informal blind A/B (Earshot vs unprocessed vs Everyone) at SNR <= 5 dB;
  gate: Earshot preferred in >= 70 % of trials.

### G.5 On-device acceptance (iPhone 16 Pro)

| Test | Gate |
|---|---|
| Parity - engine output vs PyTorch golden vectors | max abs error <= 2e-4 on output, 1e-3 on heads |
| Compute - `es_process_hop` over 60 min, screen locked, 1 and 3 instances | p99.9 <= 2 ms (50 % of hop), max logged |
| Dropouts - 3 x 60 min locked sessions with AirPods, including 10 route changes, 2 calls, 1 Siri, 1 alarm | 0 audible glitches outside route changes; <= 1 concealed underrun per hour |
| Latency - wired loop and AirPods loop, 5 repeats, at 0, 10, 30 min | reported as measured; Earshot's own share <= 20 ms |
| Battery - Power Profiler + battery level over 2 h, screen locked | <= 5 % per hour with 1 instance, <= 8 % with 3 |
| Thermal - 60 min in a pocket at room temperature | thermalState never above `.fair` |
| Alarm passthrough - 20 played siren/smoke-alarm clips in babble | >= 18 detected within 5 s; <= 1 false alarm per hour of restaurant noise |

### G.6 Daily-use pilot (the product gate)

10 real sessions (restaurants, parties, a hallway) of at least 20 minutes each, logs only (no audio). Gates - zero crashes,
100 % recovery from route changes and calls (with at most one tap), look-select picks the intended person in >= 90 % of
double-presses (user marks misses with a triple-press-then-double-press pattern logged as a correction), median user
rating >= 4/5 for "I could follow the person I chose", zero reports of hearing their own voice delayed.

---

## H. Work breakdown (6 packages, parallel, separate directories)

Rules - each package owns only its directory; shared contracts are the interfaces below (owned by the lead and changed
only by a versioned note in `docs/`); each package builds and tests alone with stubs.

| Package | Directory | Owns | Inputs | Outputs | Done when |
|---|---|---|---|---|---|
| **P1 Probe** | `probe/` | small iOS app measuring the platform | - | `docs/measurements/probe-<date>.json` (I6) and a summary | every Phase 0 question in J has a number |
| **P2 Data** | `dataprep/` (fetchers exist) + `dataprep/mixer/` | data pack, RIR and babble banks, Mixer, preview tool | sources in F.1 | manifests (I1), MixExample batches (I2) | 9.3 GB pack built, license filter tested, 20 previews listened to |
| **P3 Model** | `model/` | `esb.py`, `stft.py`, `losses.py`, `train.py`, `export.py`, `ref_stream.py` | I2 | export bundles (I3) incl. golden vectors; an early random-weight bundle in week 1 | sanity tests pass; main run exported |
| **P4 Engine** | `engine/` | C99 + Accelerate: model, STFT, resampler, enrollment encoder, post-processing, safety, rings; macOS test harness | I3 | static library for iOS and macOS, header I4, voiceprints I5 | golden parity, malloc count 0, timing bench |
| **P5 App** | `app/` | SwiftUI app, session, RemoteIO host, worker, calibration, motion, remote commands, enrollment UI (camera, quick add), people store, alarm listener, logs | I4, I5, I6 | the app, session logs I7 | G.5 and G.6 pass |
| **P6 Eval** | `eval/` | test-set renderer, metrics, streaming eval, rig protocol and scripts, device-log analyzer, report | I1, I3, I7 | `docs/eval/report-<date>.md` | G.3 table filled automatically |

### Interface contracts

**I1 - manifest row** (exists in `dataprep/common.py`) - JSON lines with `path`, `source`, `speaker`, `gender`, `accent`,
`group`, `duration`, `sr` (16000), `split` (train/val/test), `license`, `use`, `extra`.

**I2 - MixExample** (one dict per example, numpy float32 at 16 kHz; deterministic for `(seed, index)`)
- `mix[T]`, `target[T]` (zeros if absent), `present` (bool), `mode` (0 Focus, 1 Everyone)
- `enroll[Te]`, `enroll_mask[Te // 512]` (uint8, 32 ms frames), `self_enroll[Ts]` (length 0 means null-self)
- `target_vad[T // 64]`, `self_vad[T // 64]` (uint8, 4 ms frames), `zero_state_at` (hop index or -1)
- `spk_index` (training speaker id or -1), `meta` (sources, SNR, SIR, room, geometry)

**I3 - export bundle** `model/export/<name>/`
- `model.json` - version, sr, n_fft 512, hop 64, synthesis 128, band widths, D, blocks, Ht, Hb, token limits, g_dim, window
  formulas (A.1) and arrays' checksums, and the tensor table (name, shape, byte offset, layout).
- `weights.bin` - little-endian float32, each tensor 64-byte aligned, Linear weights stored `[in, out]` row-major.
- `golden/` - `mix.f32`, `enroll.f32`, `self.f32`, `out.f32`, `heads.f32`, `state_hop1000.f32`, `voiceprint.esvp`,
  `tolerance.json`.

**I4 - engine C API** (`engine/include/earshot_engine.h`)

```c
typedef struct es_engine es_engine;                    // opaque, everything allocated in es_create
typedef struct { int io_rate; int max_instances; int max_io_frames; } es_config;
typedef struct { float voice_db, room_db; int mute_while_talking, gate_on; } es_params;
typedef struct { float p_target[3], p_self, lock[3], out_rms_db, gain_db; uint32_t hops, underruns; } es_stats;

es_engine *es_create(const char *bundle_dir, const es_config *cfg, char *err, int errlen);   // not RT
void es_destroy(es_engine *e);                                                                 // not RT
int  es_enroll(es_engine *e, const float *pcm16k, int n, const uint8_t *mask32ms, int self,
               es_voiceprint *out);                                                            // not RT
int  es_set_instance(es_engine *e, int slot, const es_voiceprint *vp /* NULL = Everyone */, int warm); // not RT, atomic swap
int  es_set_self(es_engine *e, const es_voiceprint *vp /* NULL = none */);                    // not RT
void es_select(es_engine *e, const float weights[3], int crossfade_ms);                       // RT-safe
void es_set_params(es_engine *e, const es_params *p);                                         // RT-safe
void es_process_hop(es_engine *e, const float *in_io, float *out_io);  // RT: exactly one hop (192 frames at 48 kHz)
void es_get_stats(const es_engine *e, es_stats *s);                    // RT-safe snapshot
// safety chain, used by the app's render callback
typedef struct es_safety es_safety;
void es_safety_process(es_safety *s, float *buf, int n);               // RT: fades, mute, limiter, soft clip
```

**I5 - voiceprint** - the ESVP v1 binary in B.4; produced only by `es_enroll`; rejected if `model_id` differs from the
loaded bundle.

**I6 - probe measurement JSON** - per test: device, iOS, route, requested and granted buffer, rate, reported latencies,
measured loop latencies (all repeats), data source and polar pattern honored, motion rate and gaps (screen on/locked),
remote commands received, compute timing histograms (C engine and, if built, BNNS Graph), Power Profiler summary,
Conversation Awareness and Spatial Audio observations, SoundAnalysis cost.

**I7 - session log** - JSON lines per minute with the fields in D.11 plus app version and bundle name.

### Order and milestones

- **Week 1** - P1 probe running on the phone; P2 fetchers + Mixer; P3 model code + random-weight bundle; P4 engine against
  that bundle; P5 app shell with a passthrough engine, session, calibration, remote commands; P6 metrics + test sets.
- **Week 2** - ablations and main training (P3); device integration of the real bundle (P4, P5); rig recordings (P6).
- **Week 3** - full evaluation, pilot sessions, fixes; decision on the wired low-latency profile from P1's numbers.

---

## I. Top risks and mitigations

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| 1 | **AirPods delay is high or drifts** (Pro 1 especially), so lips look off and the leak sounds like an echo | high / high | Measure and show; require ANC; own-voice suppression + duck; recommend Pro 2; Game Mode experiment in the personal build; honest copy; wired sealed earbuds for long conversations |
| 2 | **Target confusion** with similar or quieter voices | high / high | SIR down to -8 dB in training; similarity curriculum; cross-attention tokens; lip-masked enrollment; per-person "locked" check; instant re-add; Table mode avoids picking between table members |
| 3 | **Sim-to-real gap** (phone mic, real rooms, Lombard speech); CHiME-9 real gains were modest [19] | high / high | Measured mic response and level in training; real RIRs; rig recordings as references; stage-3 fine-tune; pilot gates; report real numbers separately |
| 4 | **Training budget** - 0.98 s/step means about 140k steps in 44 h, possibly under-trained | medium / medium | Measured before committing; narrow model for ablations; extra calendar time for a second fine-tune run; never shrink below 16 bands |
| 5 | **iOS unknowns** - measurement mode with A2DP, motion while locked, remote commands, buffer inflation, Conversation Awareness ducking | medium / high | Phase 0 probe first; fallbacks for each (default mode, cycling, on-screen controls, user checklist) |
| 6 | **Wired 20 ms is likely out of reach** (crowd round trip about 15-17 ms) | high / low (user owns only AirPods) | Never promise it; low-latency model only if Phase 0 shows a fast path |
| 7 | **Own voice leaks** into the output | medium / high | Mandatory own-voice enrollment; negative conditioning; p_self duck; G.3 gates |
| 8 | **Dropouts** from preemption or thermal throttling | low / high | Worker thread with adaptive slack and concealment; 6 to 20x compute headroom (Mac proxy); telemetry; thermal downshift |
| 9 | **Free Personal Team** - the app stops opening 7 days after each install; 3 apps per device [49] | certain / medium | Weekly re-install from Xcode (2 minutes); or the paid Apple Developer Program, whose profiles last about a year [49] (M) |
| 10 | **Licenses and patents** - VoxCeleb-trained weights research-only; WHAM! and LOTH non-commercial; UW/Hearvana commercializing target speech hearing [2] | medium / high for any release | Personal build only; internal encoder by default; NC data only in test; clean-room implementation; patent search before any distribution |
| 11 | **Privacy and law** - voiceprints of others are biometric data [39]; recording others needs consent in PA [40] | medium / medium | On-device only; "just for tonight" default; consent prompt for saving; no conversation audio stored by default |
| 12 | **Hearing safety** | low / high | Cap, AGC, limiter, fades, Reduce Loud Audio guidance, alarm passthrough, no medical claims |

---

## J. Phase 0 decision gates (P1 probe, first week)

| # | Question | Decision it drives |
|---|---|---|
| P1-1 | Built-in mic to USB-C wired round trip at 32/64 frames, `.measurement` | wired low-latency model yes/no (C.1) |
| P1-2 | Granted buffers 16 to 192 frames, `.measurement` and `.default`; with Sound Recognition on | buffer and slack defaults; warning copy |
| P1-3 | AirPods Pro 2 and Pro 1 loop latency: in-ear leakage method vs ear detection off; at 0, 10, 30 min | calibration UX, displayed numbers |
| P1-4 | `.measurement` + A2DP coexist; built-in mic really used (tap test); back data source or polar patterns honored | session config; enrollment mic |
| P1-5 | `CMHeadphoneMotionManager` rate, gaps and drift in `.playAndRecord` + A2DP, screen on and locked | look-select availability when locked |
| P1-6 | Remote commands (single/double/triple) delivered; effect of Conversation Awareness and Spatial Audio | control mapping, checklist |
| P1-7 | C engine (random-weight ESB-16) p50/p99.9 on the worker joined to the workgroup, 60 min, locked, 1 and 3 instances; BNNS Graph comparison | warm-instance count, runtime choice |
| P1-8 | Power Profiler and battery over 2 h | battery copy, downshift thresholds |
| P1-9 | iPhone mic level (dBFS) for 60 dB SPL speech at 1 m; frequency response | AGC cap, training level range |
| P1-10 | SoundAnalysis CPU cost and detection delay on device | alarm passthrough defaults |
| P1-11 | Ignoring the remote pause sent on AirPod removal: does the removed bud keep playing? | calibration fallback |

---

## Appendix A - reference model (PyTorch, condensed)

This is the architecture that was timed in Appendix B (training graph). P3 grows it into `model/esb.py` with the
asymmetric windows of A.1, masked pooling, null embeddings and the loss terms of F.3.

```python
class DualPath(nn.Module):                      # time LSTM shared across bands + BiLSTM across bands
    def __init__(s, D=64, Ht=128, Hb=64, bidir_time=False):
        super().__init__()
        s.tn, s.bn = nn.LayerNorm(D), nn.LayerNorm(D)
        s.t = nn.LSTM(D, Ht, batch_first=True, bidirectional=bidir_time)
        s.tf = nn.Linear(Ht * (2 if bidir_time else 1), D)
        s.b = nn.LSTM(D, Hb, batch_first=True, bidirectional=True)
        s.bf = nn.Linear(2 * Hb, D)
    def forward(s, z):                          # z: (batch, T, K, D)
        b, T, K, D = z.shape
        y, _ = s.t(s.tn(z).permute(0, 2, 1, 3).reshape(b * K, T, D))
        z = z + s.tf(y).reshape(b, K, T, D).permute(0, 2, 1, 3)
        y, _ = s.b(s.bn(z).reshape(b * T, K, D))
        return z + s.bf(y).reshape(b, T, K, D)

class ESB16(nn.Module):
    WIDTHS = [8] * 8 + [16] * 4 + [32] * 3 + [33]             # 257 bins
    def __init__(s, D=64, B=4, Ht=128, Hb=64, d_att=64, g_dim=128):
        super().__init__()
        K = len(s.WIDTHS)
        s.enc = BandEncoder(s.WIDTHS, D)                       # per-band Linear(2w -> D) + LayerNorm
        s.blocks = nn.ModuleList([DualPath(D, Ht, Hb) for _ in range(B)])
        s.enr = DualPath(D, 32, 32, bidir_time=True)           # enrollment encoder, 32 ms hop
        s.tok, s.g = nn.Linear(K * D, d_att), nn.Linear(d_att, g_dim)
        s.type_emb = nn.Parameter(torch.zeros(2, d_att))        # 0 = target, 1 = own voice (negative)
        s.null_g, s.null_self_g = nn.Parameter(torch.zeros(g_dim)), nn.Parameter(torch.zeros(g_dim))
        s.null_tok = nn.Parameter(torch.zeros(4, d_att))
        s.film = nn.Sequential(nn.Linear(2 * g_dim + 2, 256), nn.SiLU(), nn.Linear(256, 2 * K * D))
        s.q, s.kv, s.o = nn.Linear(K * D, d_att), nn.Linear(d_att, 2 * d_att), nn.Linear(d_att, K * D)
        nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)
        s.dec = BandDecoder(s.WIDTHS, D)                        # LN, Linear(D -> 2D), tanh, Linear(2D -> 2w), 1.5 tanh
        s.heads = nn.Linear(D, 2)                               # p_target, p_self

    def forward(s, X, gam, bet, keys, vals):                    # X: causal STFT (asymmetric window), (b, T, 257)
        z = s.enc(X)
        b, T, K, D = z.shape
        for i, blk in enumerate(s.blocks):
            z = blk(z)
            if i == 0:                                          # conditioning after block 1
                z = z * (1 + gam[:, None]) + bet[:, None]
                q = s.q(z.reshape(b, T, K * D)).reshape(b, T, 4, 16).transpose(1, 2)
                a = F.scaled_dot_product_attention(q, keys, vals)   # keys/vals: (b, 4, n_tok, 16)
                z = z + s.o(a.transpose(1, 2).reshape(b, T, 64)).reshape(b, T, K, D)
        mask = s.dec(z)                                         # complex mask per bin
        return mask, torch.sigmoid(s.heads(z.mean(2)))
```

## Appendix B - measurements made for this design (2026-11-07, shared M4, torch 2.7.1)

Scratch scripts (kept outside the repo) followed the protocols of `bench/bench_train.py` and `bench/chunk_skeleton.c`:
random tensors, forward + loss + backward + clip + fused AdamW, median after warm-up; C per-hop compute with random
weights, real shapes, pre-transposed weights, `VECLIB_MAXIMUM_THREADS=1`, 20,000 timed hops. Load average 3 to 9 from
other jobs, so treat tails as pessimistic and medians as about 10 % slow.

### B.1 Training step on MPS (batch 8 x 4 s unless noted)

| Config (bands, width D, time LSTM, band BiLSTM per dir, blocks 4) | Enrollment encoder | s/step | MPS pool GB | GMAC/s |
|---|---|---|---|---|
| 32, 64, 96, 48 | 2 blocks, 16 ms hop | 2.54 | 19.9 | 3.90 |
| 32, 64, 96, 48 | none | 1.43 | 17.6 | 3.90 |
| 32, 64, 96, 48, batch 4 | none | 0.68 | 9.4 | 3.90 |
| 32, 64, 96, 48, 2 s segments | none | 0.66 | 9.4 | 3.90 |
| 32, 64, 64, 32 | none | 0.98 | 13.3 | 2.26 |
| 20, 64, 96, 48 | none | 0.92 | 11.7 | 2.45 |
| 20, 64, 80, 40 | none | 0.83 | 9.6 | 1.90 |
| 16, 64, 96, 48 | none | 0.76 | 9.4 | 1.96 |
| 16, 64, 128, 64 | none | 0.89 | 11.8 | 2.98 |
| 16, 64, 96, 48 | 2 blocks, 16 ms hop | 1.10 | 11.3 | 1.96 |
| 16, 64, 96, 48 | 1 block, 32 ms hop | 0.86 to 0.90 | 9.6 | 1.96 |
| **16, 64, 128, 64 (ESB-16)** | **1 block, 32 ms hop** | **0.98** | **12.1** | **2.98** |
| 16, 64, 128, 64, batch 4 x 8 s | 1 block, 32 ms hop | 1.08 | 12.2 | 2.98 |

Losses in these runs - thresholded SNR + 3-resolution STFT + BCE heads (the STFT term cost about 0.15 s at 32 bands and
nothing measurable at 16). Time grew linearly with batch and length, so the cost is compute- or bandwidth-bound, not launch
overhead. A batch of 16 at 32 bands did not finish (memory).

### B.2 Per-hop compute in C with Accelerate (one 4 ms hop, 16 bands)

| Config | Weights MB | MMAC/hop | P-core median us | P-core p99 / p99.9 / max us | E-core median us |
|---|---|---|---|---|---|
| D 64, Ht 128, Hb 64 (ESB-16) | 4.37 | 11.90 | 175 to 187 | 212 to 629 / 339 to 1,053 / 493 to 1,829 | 591 to 638 |
| D 64, Ht 96, Hb 48 | 3.35 | 7.84 | 160 to 172 | 185 to 196 / 280 to 369 / 538 to 2,655 | 546 to 635 |
| D 48, Ht 64, Hb 32 | 1.90 | 3.79 | 133 | 156 / 206 / 392 | 315 |

E-core runs used `taskpolicy -c background`, which also lowers priority, so their tails (up to 0.1 to 1 s) are not
meaningful for a real-time thread.

### B.3 Window pair check (numpy)

- Product window sum at hop 64 - min 0.9999999999999998, max 1.0000000000000004.
- Streaming reconstruction with the 128-sample overlap-add buffer - max error 1.3e-15, delay 2 hops (128 samples, 8 ms).
- Smallest analysis-window value on the synthesis span - 0.0245 (so w_s stays bounded, max 1.0).

### B.4 Resampler filter (scipy `firwin`, Kaiser)

| Taps | beta | Cutoff | Passband to 6.5 kHz | -3 dB | Stopband above 8 kHz | Group delay |
|---|---|---|---|---|---|---|
| **128** | **5.0** | **7.4 kHz** | **+/- 0.01 dB** | **7.25 kHz** | **<= -51.4 dB** | **63.5 samples = 1.32 ms** |
| 96 | 4.5 | 7.2 kHz | -0.06 to +0.03 dB | 7.01 kHz | <= -50.1 dB | 0.99 ms |

### B.5 SoundAnalysis built-in classifier (macOS 26.5, `SNClassifySoundRequest(.version1)`)

303 labels; includes `smoke_detector`, `siren`, `police_siren`, `ambulance_siren`, `fire_engine_siren`,
`civil_defense_siren`, `emergency_vehicle`, `car_horn`, `air_horn`, `train_horn`, `screaming`, `shout`, `glass_breaking`,
`baby_crying`, `door_bell`, `alarm_clock`, `telephone_bell_ringing`, `speech`. Default window 3.0 s, overlap 0.5. The iOS
list should match but is checked on device in P1-10.

---

## References

Local (this repo)
1. `docs/BRIEF.md`
2. `docs/research/prior-art.md`
3. `docs/research/ios-platform.md`
4. `docs/research/data-training.md`
5. `docs/research/fact-check.md`
6. `bench/RESULTS.md`
7. Appendix B of this document (measured here)

Papers
8. Veluri et al., Look Once to Hear, CHI 2024 - https://arxiv.org/abs/2405.06289 ; license https://github.com/vb000/LookOnceToHear
9. Xu, Yang, Trigoni, Markham, noisy positive/negative enrollment, NeurIPS 2025 - https://arxiv.org/abs/2502.16611
10. Wang, Naithani, Politis, Virtanen, asymmetric windows, 2021 - https://arxiv.org/abs/2106.11794
11. Wang, Wichern, Watanabe, Le Roux, very low latency STFT enhancement, TASLP 2023 - https://arxiv.org/abs/2204.09911
12. Wu and Braun, ultra-low latency SE study, ICASSP 2025 - https://arxiv.org/abs/2409.10358
13. AV-PVQE (picked rates 98.6 / 61.1 / 6.3 %), 2026 - https://arxiv.org/abs/2609.30631
14. Pärnamaa and Saabas, PSE without a separate speaker model, Interspeech 2024 - https://arxiv.org/abs/2406.09928
15. Eskimez et al., personalized SE and TSOS, ICASSP 2022 - https://arxiv.org/abs/2110.09625
16. Delcroix et al., Listen only to me (absent targets), Interspeech 2022 - https://arxiv.org/abs/2204.04811
17. Wisdom et al., MixIT (thresholded SNR), NeurIPS 2020 - https://arxiv.org/abs/2006.12701
18. WeSep, Interspeech 2024 - https://arxiv.org/abs/2409.15799 ; causal results https://arxiv.org/abs/2607.27436
19. CHiME-9 ECHI, 2026 - https://arxiv.org/abs/2609.26306
20. Knowledge boosting, Interspeech 2024 - https://arxiv.org/abs/2407.11055

Latency, perception, devices
21. Superpowered latency data (retrieved 2026-11-06; iPhone 15/16 rows parsed in `docs/research/fact-check-bench/`) - https://superpowered.com/latencydata/data.html
22. Stephen Coyle, AirPods Pro 2 latency (126 vs 167 vs 83 ms, iPad Pro, keyboard clicks) - https://stephencoyle.net/airpods-pro-2
23. Apple Developer Forums 679274, AirPods actual vs reported latency - https://developer.apple.com/forums/thread/679274
24. RTINGS AirPods Pro 2 and Pro isolation - https://www.rtings.com/headphones/tools/compare/apple-airpods-pro-2nd-generation-apple-airpods-4/29490/74581 ; https://rtings.com/headphones/tools/compare/apple-airpods-pro-apple-airpods-pro-2nd-generation/1625/29490
25. RTINGS EarPods vs AirPods Pro isolation - https://www.rtings.com/headphones/tools/compare/apple-airpods-pro-apple-earpods/1625/406
26. ITU-R BT.1359-1 (lip-sync detectability and acceptability) - https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1359-1-199811-I!!PDF-E.pdf ; summary https://en.wikipedia.org/wiki/Audio-to-video_synchronization
27. van Wassenhove, Grant, Poeppel 2007, temporal window of AV speech integration, doi:10.1016/j.neuropsychologia.2006.01.001 - https://scholar.usuhs.edu/en/publications/temporal-window-of-integration-in-auditory-visual-speech-percepti/
28. Stuart et al. 2002, delayed auditory feedback at 0/25/50/200 ms, JASA doi:10.1121/1.1466868 - https://labs.sonicfield.org/library/effect-of-delayed-auditory-feedback-on-normal-speakers-at-two-speech-rates
29. Stone and Moore, Tolerable hearing aid delays I-IV, Ear and Hearing - https://labs.sonicfield.org/library/tolerable-hearing-aid-delays-iv-effects-on-subjective-disturbance-during-speech-

Apple platform
30. AVAudioSession docs (setPreferredIOBufferDuration, measurement mode, allowBluetoothA2DP) - https://developer.apple.com/documentation/avfaudio/avaudiosession/setpreferrediobufferduration(_:) ; https://developer.apple.com/documentation/avfaudio/avaudiosession/mode-swift.struct/measurement ; https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetootha2dp
31. Apple DTS: no recording session from background - https://developer.apple.com/forums/thread/774784 ; remote commands over A2DP only - https://developer.apple.com/forums/thread/761549
32. CMHeadphoneMotionManager - https://developer.apple.com/documentation/coremotion/cmheadphonemotionmanager
33. BNNS Graph, WWDC24 10211 and WWDC25 276 - https://developer.apple.com/videos/play/wwdc2024/10211/ ; https://developer.apple.com/videos/play/wwdc2025/276/
34. BLASSetThreading and audio workgroups - https://developer.apple.com/documentation/accelerate/blassetthreading(_:) ; https://developer.apple.com/documentation/audiotoolbox/adding-parallel-real-time-threads-to-audio-workgroups
35. SoundAnalysis SNClassifySoundRequest - https://developer.apple.com/documentation/soundanalysis/snclassifysoundrequest
43. Power Profiler, WWDC25 226 - https://developer.apple.com/videos/play/wwdc2025/226/
45. Switch AirPods between devices (Connect to This iPhone) - https://support.apple.com/en-ng/104988
46. Adaptive Audio and Conversation Awareness - https://support.apple.com/en-us/104979
48. Controls API (Control Center, Lock Screen, Action button), WWDC24 10157 - https://developer.apple.com/videos/play/wwdc2024/10157/
49. Membership comparison (7-day free profiles, limits) - https://developer.apple.com/support/compare-memberships/ ; paid profiles about one year (forum) - https://developer.apple.com/forums/thread/712540
50. Hearing Aid feature and Media Assist - https://support.apple.com/en-us/120992
51. Vision face landmarks (lips) - https://developer.apple.com/documentation/vision/vnfacelandmarks2d
52. AVSpeechSynthesizer write to buffer - https://developer.apple.com/documentation/avfaudio/avspeechsynthesizer/write(_:tobuffercallback:)
53. MPRemoteCommandCenter and MPNowPlayingInfoCenter - https://developer.apple.com/documentation/mediaplayer/mpremotecommandcenter ; https://developer.apple.com/documentation/mediaplayer/mpnowplayinginfocenter
54. AirPods ear detection pause behavior (Apple text quoted in community threads) - https://discussions.apple.com/thread/253460699 ; gestures https://support.apple.com/en-us/102628
55. Shortcuts AirPods noise-control automations - https://www.imore.com/great-shortcuts-and-automations-airpods-pro ; https://9to5mac.com/2025/03/28/my-only-airpods-pro-problem-was-fixed-by-apples-shortcuts-app/
56. Buffer inflation with Sound Recognition / Vocal Shortcuts (iOS 18) - https://developer.apple.com/forums/thread/769245

Safety, regulation, privacy
36. FDA OTC hearing aid final rule summaries (111 / 117 dB SPL, no gain limit) - https://www.asha.org/news/2022/fda-releases-final-rule-for-otc-hearing-aids/ ; https://www.entnet.org/advocacy/regulatory-advocacy/over-the-counter-sale-of-hearing-aids/aao-hns-summary-fda-over-the-counter-hearing-aids-final-rule
37. WHO-ITU H.870 safe listening (80 dB(A) for 40 h per week) - https://www.who.int/publications/i/item/9789241515276
38. Apple headphone safety (Reduce Loud Audio, notifications) - https://support.apple.com/en-us/108968
39. Illinois BIPA, 740 ILCS 14/10 (voiceprint is a biometric identifier) - https://lig.ilga.gov/legislation/ilcs/documents/074000140K10.htm
40. Pennsylvania recording law (all-party consent) - https://www.dmlp.org/node/1302

Battery and hardware
41. iPhone 16 Pro battery 3,582 mAh (regulatory filing via MacRumors) - https://www.macrumors.com/2024/09/19/iphone-16-battery-capacities-revealed/ ; Apple specs (audio playback up to 85 h) - https://www.apple.com/mt/iphone-16-pro/specs
42. A18 Pro core power (Geekerwan SPEC figures via a Bilibili summary; low confidence) - https://www.bilibili.com/opus/978640085494792198
44. AirPods Pro 2 listening time 6 h with ANC (Apple spec via retailer listing) - https://www.bestbuy.com/site/apple-airpods-pro-2nd-gen/6447382.p

Behavior and prior products
47. Head orientation undershoots talkers; looking away is normal (Frontiers 2024) - https://www.frontiersin.org/articles/10.3389/fpsyg.2024.1425972/full ; eye-gaze steering of hearing aids (Favre-Félix et al. 2018, Trends in Hearing, doi:10.1177/2331216518814388) - https://doaj.org/article/3c7a5d69cf68469baffd54ae0302bead
59. DeepFilterNet attenuation limit - https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/enhance.py
63. WeSpeaker model license and VoxCeleb terms - https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md ; https://mm.kaist.ac.kr/datasets/voxceleb/
