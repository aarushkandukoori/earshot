# Earshot - quality-first system design (Earshot-Q)

Date - 2026-11-07. Designer angle - quality first: the best voice isolation the iPhone 16 Pro can run in real time,
using the Neural Engine or GPU only where the latency and real-time-safety math allows it.

Inputs - `docs/BRIEF.md` [D1], `docs/research/prior-art.md` [D2], `ios-platform.md` [D3], `data-training.md` [D4],
`fact-check.md` [D5], `bench/RESULTS.md` [D6], new web checks (references [1]-[45] at the end), and new measurements made
for this design on the M4 (Appendix A).

How to read this document

- **Confidence** - [H] high: read in a primary source (paper text, Apple documentation JSON, license file) or measured here.
  [M] medium: secondary source, a single report, or my arithmetic on primary numbers. [L] low: estimate or inference.
- **Measured** - measured on the training Mac (M4, 4P+6E, 10-core GPU, 24 GB) on 2026-11-07 for this design, unless another
  source is named. The Mac was shared with other jobs (dataset downloads, another GPU benchmark; load average
  5 to 10), so I give ranges and the load. **Nothing has run on the iPhone yet.** Every phone number below is a Mac proxy
  until Phase 0 measures it.
- **Credit** - the interaction ("look at a person for a few seconds, then hear only them") is UW's Look Once to Hear
  (Veluri et al., CHI 2024) [7]. The network is from the TF-GridNet / band-split RNN lineage. No code from
  non-commercial or evaluation-only sources is used (section F.1, [D2], [D5]).

---

## 0. Decision summary

- **The quality ceiling is set by the M4 training budget, not by the iPhone CPU.** Measured - the chosen model costs
  0.39 ms per 4 ms hop on one M4 performance core in plain C with Accelerate (real-time factor 0.10), while one training
  step of 4 x 4 s costs 1.0 to 1.9 s on the M4 GPU, and MPS memory caps the batch at 4 x 4 s (Appendix A). So the model is
  sized to what the Mac can train well in 3 to 6 days, not to what the phone could run.
- **Primary model (AirPods path) - Earshot-Q-M.** Causal band-split dual-path LSTM in the STFT domain. 16 kHz, asymmetric
  32 ms analysis / 8 ms synthesis window, 4 ms hop, **8 ms algorithmic latency**, no lookahead. 32 bands, width 64,
  5 blocks (BiLSTM 48 per direction across bands, LSTM 96 across time). FiLM plus cross-attention at blocks 1 and 3 over
  enrollment tokens. The user's own voice is a negative token set, and learned null tokens give the "Everyone" mode.
  3-tap deep filter below 4 kHz, complex mask above, target-activity and own-voice-activity heads.
  **4.92 GMAC/s, 1.6 M streaming parameters (5.2 MB fp32)**, plus a 2.4 M FiLM generator and a 0.9 M token encoder that
  run once per enrollment.
- **Wired model (secondary) - Earshot-LL.** The same network and code with a 16 ms analysis window and either a 2 ms hop
  (4 ms algorithmic, "LL-2") or a 1 ms hop (2 ms algorithmic, "LL-1"). Which one is trained depends on the hardware
  latency Phase 0 measures on this phone (section C.4).
- **Runtime - hand-written C with Accelerate** (cblas, vDSP, vForce) on one real-time worker thread joined to the RemoteIO
  audio workgroup. BNNS Graph is benchmarked as plan B. Core ML runs only off the audio path (enrollment embedding,
  identity monitor).
- **Neural Engine and GPU are not on the live path.** The real-time math fails on safety, not on speed (section 1.2).
  The Neural Engine gets an optional v2 job, a delayed "context booster" whose late or missing result is harmless.
- **Latency** - over AirPods Pro 2, Earshot adds **9.6 ms over a plain pass-through app on the same route** (12.3 ms of
  app-side path in total). The Bluetooth part is unknown on this phone, with a planning range of about 45 to 220 ms, and it
  is measured in the app at every connect. Wired - under 20 ms needs the phone's hardware latency X (everything outside
  the app's buffers) to be at most about 13.4 ms for LL-2 or 15.7 ms for LL-1. Crowd data for iPhone 15/16 puts X near
  14.3 ms (range 6.3 to 24.3 ms) [D5], so 20 ms is borderline and only a measurement can settle it.
- **Enrollment** - 5 to 8 s with the phone pointed at the person. WeSpeaker ResNet34 global embedding (personal build only,
  VoxCeleb research license [D5]) plus jointly trained frame-level tokens. Numeric quality checks, own voice enrolled once
  at setup, up to 8 saved people, guarded refresh during use.
- **Training** - the about 9.3 GB data pack that `dataprep/` is already fetching (fetchers seen running on 2026-11-07),
  dynamic mixing with own voice, absent targets, target switches and all-speech mode, 3 stages. Baseline **about 60 h of
  M4 GPU time (650k examples)**; quality plan - keep stage 2 running until validation stops improving (expected 4 to 6
  days, about 1.2 M examples). The wired model needs another 35 to 40 h after Phase 0.
- **Acceptance (selected, section G.3)** - restaurant simulation median SI-SDRi at least 7 dB (goal 9), picked rate at
  least 95% at equal level, own voice at least 25 dB down, target-absent output at least 25 dB down, DNSMOS OVRL at least
  2.8, 0 audible dropouts in 2 h with the screen locked, battery drain at most 8% per hour.

What is impossible, and what the design does instead

- **Under 20 ms over Bluetooth** - impossible. The A2DP link alone is tens to hundreds of ms (section C.3). Earshot
  measures and shows the number, relies on ANC to hide the direct sound, and suppresses the user's own voice.
- **Choosing a person by direction with one phone mic** - impossible acoustically. Earshot selects by voice identity.
  Head tracking only chooses among people already enrolled (each has a stored seat direction), and a stem press confirms.
- **Restarting after a phone call while in the background** - impossible on iOS; a recording session cannot activate from
  the background (Apple DTS) [D3][D5]. Earshot posts a local notification asking the user to reopen it.
- **Knowing the true sound level at the eardrum** - no public API reports real-time output SPL [D3]. Earshot uses a
  conservative digital ceiling and leans on iOS headphone safety, which measures Apple headphones best [33].
- **Running the network on the Neural Engine every 4 ms with real-time safety** - not possible with public APIs (1.2).

---

## 1. Where quality is won or lost on this hardware

### 1.1 The phone CPU has room (M4 measurements as a proxy)

I wrote a C skeleton of the exact Earshot-Q per-hop work (random weights, real shapes, real memory): STFT, band encoder,
5 dual-path LSTM blocks, FiLM and cross-attention, per-band decoder, deep filter, inverse FFT and overlap-add, with
pre-transposed weights and single-threaded Accelerate (Appendix A.1). One thread, 20,000 hops after 500 warm-up hops.

| Model | Hop | MMAC per hop (analytic) | GMAC/s | P-core median | P-core p99 | P-core p99.9 | P-core max | E-core median | E-core p99 |
|---|---|---|---|---|---|---|---|---|---|
| Q-S (D48, Hb32, Ht64, 4 blocks) | 4 ms | 7.79 | 1.95 | 261 us | 445 us | 702 us | 4.8 ms | 460 us | 956 us |
| **Q-M (D64, Hb48, Ht96, 5 blocks)** | 4 ms | 19.68 | **4.92** | **385 us** | 1090 us | 1657 us | 7.4 ms | 1886 us | 2623 us |
| Q-L (D96, Hb64, Ht128, 6 blocks) | 4 ms | 44.15 | 11.04 | 557 us | 1485 us | 2595 us | 12.1 ms | 1491 us | 3352 us |
| LL-S (D48, Hb32, Ht64, 4 blocks, 256-pt FFT) | 2 ms | 7.71 | 3.86 | 254 us | 460 us | 750 us | 0.9 ms | 522 us | 1260 us |
| LL-M (D64, Hb48, Ht96, 5 blocks, 256-pt FFT) | 2 ms | 19.58 | 9.79 | 365 us | 589 us | 905 us | 1.3 ms | 1192 us | 2329 us |

[H, measured; P-core = default QoS at load about 5, E-core = `taskpolicy -c background` at load about 4, which also lowers
priority, so E-core tails and the Q-M/Q-L inversion are scheduling noise. Max values are desktop preemptions; the phone
runs this thread at real-time priority.]

What it means

- **Real time fits easily on a performance core** - Q-M uses about 10% of one M4 P-core at the median. The A18 Pro has the
  same core generation as the M4 [D6], but lower clocks and a phone's thermal limits, so plan for 1.1 to 1.5x these times
  until Phase 0 [L].
- **Cost is set by sequential steps, not MACs** - each hop runs 5 blocks x 2 directions x 32 bands = 320 small recurrent
  steps across the band axis. That is why Q-L has 2.2x the MACs of Q-M but takes only 1.45x the time. A fused gate loop
  (no vForce call per step) should cut this further [L].
- **E-cores are marginal for Q-M** - if iOS schedules the worker on an efficiency core, Q-M may need 1.9 ms per 4 ms hop.
  Phase 0 must confirm where the workgroup-joined thread runs. Q-S is the fallback (460 us on an E-core).

### 1.2 Neural Engine and GPU - the real-time math

| Requirement for the live path | CPU (C + Accelerate) | Neural Engine via Core ML | GPU via Metal / Core ML |
|---|---|---|---|
| Speed - result within one I/O period (1.33 ms) after each 4 ms hop | yes, 0.39 ms median measured (1.1) [H] | probably on average - small Core ML models measured 0.43 to 0.71 ms on an iPhone 15 Pro ANE, with 0.1 to 0.23 ms fixed dispatch overhead (secondary sources) [D3][M]. Apple's own example - DistilBERT at 3.47 ms and 0.454 W on an iPhone 13 ANE [1][H] | not measured here |
| No allocation, no locks, no Objective-C on the audio path [D1] | yes (preallocated C) | no - prediction goes through Objective-C or Swift objects (`MLFeatureProvider`, `MLMultiArray`) [D3][M] | no - command buffers are Objective-C objects |
| Bounded tail latency | yes, own thread at real-time priority in the audio workgroup [D3] | no guarantee found - the ANE is shared with system features and Apple documents no priority or deadline control [L]; synchronous prediction is not thread safe and stateful predictions must be serialized [D3][H] | no guarantee found; shared with UI rendering [L] |
| Actually used for this graph | - | no - Core ML placed 100% of the ops of the bench's streaming steps on the CPU, even when allowed the ANE [D6][H] | - |
| Numerically trustworthy | yes | not always - the ANE returned wrong speaker embeddings (cosine 0.57) for a mixed-precision model that was exact on CPU and GPU [D4][H] | yes for that model [D4] |

Verdict - **the live path runs on the CPU only.** The Neural Engine is used only where a result may arrive 50 ms or more
late, or not at all, without harm. That is the v2 context booster (A.9). Apple's own guidance for latency-sensitive audio
inference is BNNS Graph on the CPU [2][H].

### 1.3 The M4 GPU is the binding constraint (measured)

Training-step benchmark of the exact architecture (Appendix A.2): forward, thresholded-SNR plus multi-resolution STFT loss,
backward, gradient clipping, AdamW, with the token encoder run on a target and an own-voice enrollment of 4 s each.
PyTorch 2.7.1, fp32 (LSTMs stay fp32 under MPS autocast [D6]), random data, data loading excluded.

| Model | Batch x segment | Seconds per step | ms of GPU time per second of training audio | 4 s examples per 24 h | MPS pool |
|---|---|---|---|---|---|
| Q-S | 4 x 4 s | 0.85 | 53 | 407k | 9.0 GB |
| **Q-M** | 4 x 4 s | **1.03 to 1.18 (1.88 once at load 9)** | 64 to 74 (118) | 290k to 340k | 12.4 to 14.0 GB |
| Q-M | 2 x 6 s | 1.01 | 84 | - | 9.3 GB |
| Q-M | 8 x 2 s | 1.58 | 99 | - | 15.3 GB |
| Q-L | 2 x 4 s | 1.12 | 140 | 154k | 9.9 GB |
| LL-S (2 ms hop) | 4 x 4 s | 1.17 | 73 | 296k | 13.9 GB |

[H, measured; GPU idle at the start of each run, load 5 to 10, another designer's MPS benchmark ran at times. Batch 8 x 4 s
failed for Q-M (out of memory, with 11.7 to 13.8 GB of "other allocations" inside the MPS LSTM kernels) and thrashed for
Q-S (2.1 to 65 s per step).]

Decision - **Q-M.** In the 60 h baseline it sees about 650k examples (about 720 h of audio), and in the 4 to 6 day quality
plan about 1.2 M. Q-L would see less than half as many examples in the same time. Q-S is trained as the thermal fallback and
as the ablation workhorse. Q-L stays an option only if the user can give the Mac 8 to 10 days for one run.

---

## A. Model

### A.1 Signal path and framing (Q-M)

1. RemoteIO delivers 48 kHz mono Float32 from the built-in mic (measurement mode, section D.1).
2. Minimum-phase polyphase FIR decimator 48 to 16 kHz (177 taps at 48 kHz, passband 7 kHz, stopband 8 kHz, 60 dB).
   Group delay 0.11 to 0.15 ms below 4 kHz and 0.28 ms at 6 kHz, against 1.83 ms for the linear-phase version
   [H, computed with scipy, Appendix A.3].
3. Every hop of M = 64 samples (4 ms), take the newest K = 512 samples (32 ms), multiply by the asymmetric analysis window
   A(n), 512-point real FFT, 257 bins of 31.25 Hz.
4. The network outputs a complex filter per bin, applied to the current and two previous spectra below 4 kHz.
5. Inverse FFT, multiply by the synthesis window S(n), overlap-add with hop 64. Output is final 8 ms after its input
   arrived.
6. Minimum-phase polyphase interpolator 16 to 48 kHz (same group delay), then the safety chain at 48 kHz (D.5, E.4).

Window pair (Mauler and Martin's scheme as written in Wang, Naithani, Politis and Virtanen 2021, Eq. 1-5 [3][H]), with
K = 512, M = 64 and d leading zeros:

- Hann prototype - H_2M(n) = 0.5 (1 - cos(pi n / M)), n = 0 ... 2M - 1.
- Analysis - A(n) = 0 for n < d; A(n) = sqrt(H_2(K-M-d)(n - d)) for d <= n < K - M; A(n) = sqrt(H_2M(n - K + 2M)) for
  K - M <= n < K.
- Synthesis - S(n) = 0 for n < K - 2M; S(n) = H_2M(n - K + 2M) / A(n) for K - 2M <= n < K - M;
  S(n) = sqrt(H_2M(n - K + 2M)) for K - M <= n < K.
- Check - A(n) S(n) equals the Hann prototype on the last 2M samples, so overlap-add at hop M reconstructs the input
  exactly. Unit test - an all-ones mask must reproduce the input delayed by 128 samples to within 1e-6.
- d starts at 0; d = 64 is an ablation (Mauler and Martin add leading zeros against time aliasing [3][4]).
- Algorithmic latency = synthesis window = 2M = 8 ms [3][H].
- Expected gain over a symmetric 8/8 ms window - budget 0 dB. The 1.5 dB gain in [3] was a speaker-dependent LSTM on
  Danish HINT, and Wu and Braun found no significant gain for a strong model [D5][5][H]. The long analysis window is kept
  because it is free and gives 31 Hz resolution for pitch harmonics.

### A.2 Layer-by-layer specification (Q-M)

Band layout - 32 bands over 257 bins: 16 bands of 4 bins (0 to 2 kHz, 125 Hz each), 8 of 8 bins (2 to 4 kHz), 8 of 16 bins
(4 to 8 kHz, the last one also takes the Nyquist bin, 17 bins). Same split as Earshot-BSR in [D2].

| # | Stage | Exact specification | MACs per 4 ms hop |
|---|---|---|---|
| 1 | Features | per bin - compressed complex value (|X|^0.3 e^(j angle X), 2 numbers) and 0.1 ln(|X|^2 + 1e-8) (1 number); 3 numbers per bin | about 1 k |
| 2 | Band encoder | per band k - Linear(3 w_k to 64) with its own weights, then LayerNorm(64). Output z is 32 x 64 | 49 k |
| 3 | Dual-path block (x5) - band path | LayerNorm(64), BiLSTM 64 to 48 per direction across the 32 bands (zero initial state every hop), Linear(96 to 64), residual add | 1.57 M per block |
| 4 | Dual-path block (x5) - time path | LayerNorm(64), LSTM 64 to 96 across time, one LSTM shared by all 32 bands (32 states carried between hops), Linear(96 to 64), residual add | 2.16 M per block |
| 5 | Conditioning after block 1 and after block 3 | FiLM z = z (1 + gamma_l) + beta_l with gamma_l, beta_l of shape 32 x 64 (from the FiLM generator, once per enrollment), then cross-attention - query Linear(2048 to 64) on the flattened frame, 4 heads of 16, keys and values precomputed from at most 100 tokens, output Linear(64 to 2048) **zero-initialized**, residual add | 0.27 M each |
| 6 | Decoder per band | LayerNorm(64), Linear(64 to 128), tanh, Linear(128 to 2 w_k T_k), coefficients bounded by 2 tanh(x / 2). T_k = 3 taps for the 24 bands below 4 kHz (bins 0 to 127), 1 tap above | 0.39 M |
| 7 | Filter | Y(t, f) = sum over tau = 0, 1, 2 of C_tau(t, f) X(t - tau, f) below 4 kHz; Y = C_0 X above | 2 k |
| 8 | Heads | target activity p_t = sigmoid(Linear(64 to 1)) and own-voice activity p_o = sigmoid(Linear(64 to 1)), both on the band-mean of the last block | 0.1 k |
| 9 | Synthesis | 512-point inverse FFT, synthesis window, overlap-add | about 12 k |
| | **Total** | | **19.68 M per hop = 4.92 GMAC/s** [H, analytic, Appendix A.4] |

LSTM details - PyTorch gate order (i, f, g, o), forget-gate bias initialized to 1, no peepholes, no projection. LSTM
instead of the GRU in [D2] because PyTorch 2.7.1 has a native MPS LSTM kernel and no GRU kernel (GRU measured 11 to 14x
slower to train) [D5][D6][H], and BNNS Graph has a native `bidirectionalLSTM` [D5][H]. Cost - 4/3 of the GRU's recurrent
MACs.

Parameters (analytic) - band encoder 51 k, 5 blocks 593 k, 2 attention modules 529 k, decoder 399 k, heads 0.1 k,
**about 1.57 M streaming parameters, 5.2 MB in fp32** (the C skeleton allocates 5.21 MB) [H].

### A.3 Conditioning

Inputs computed once per enrollment (section B), never on the audio thread:

| Input | Shape | Source |
|---|---|---|
| e_t, e_o | 256 each | WeSpeaker ResNet34 embedding of the target's and the user's enrollment (frozen) [D4] |
| g_t, g_o | 256 each | pooled vector from the jointly trained token encoder (below) |
| target tokens | up to 64 x 64 (keys) and 64 x 64 (values) per attention module | token encoder over the target enrollment |
| own-voice tokens | up to 32 | token encoder over the user's own enrollment, plus a learned "negative" type vector added to the keys |
| null tokens | 4 | learned, always appended (a "none of these" sink) |
| mode flag | 1 | 0 = one person, 1 = Everyone |

- **FiLM generator** - MLP [e_t, g_t, e_o, g_o, mode] (1025) to 256, PReLU, to 2 x 2 x 32 x 64 (gamma and beta for both
  conditioning points). 2.37 M parameters, run once per enrollment and per mode change.
- **Token encoder** (non-causal, offline) - STFT 512 / hop 128 (sqrt-Hann, 8 ms hop), the same 32-band split, Linear(3w to
  48) and LayerNorm, one dual-path block with a BiLSTM across bands (48 to 32 per direction) and a BiLSTM across time
  (48 to 48 per direction), mean-pooled every 10 frames (80 ms) into tokens of 32 x 48 = 1536, then per attention module
  Linear(1536 to 64) for keys and for values, and Linear(1536 to 256) on the token mean for g. Only frames that a simple
  energy VAD marks as speech make tokens. About 0.9 M parameters (0.74 M measured with one key/value pair).
- **Why both a global vector and tokens** - frame-level cues beat a single embedding by a wide margin in small causal
  extractors (causal BSRNN on Libri2Mix 6.87 dB with a speaker embedding vs 13.05 dB with frame-level USEF cues [D5];
  8.85 vs 6.70 dB for cross-attention vs FiLM with noisy enrollment, with confounds [D5]). The frozen WeSpeaker vector was
  the most robust of 6 encoders on short noisy enrollments [D4]. Fusion right after a shallow first block follows [D2].
- **Dropout so either path works** - during training replace (e_t, e_o) with learned nulls in 30% of examples and
  (g_t, tokens) in 15%, never both. This lets a distributed build drop WeSpeaker (license, B.7) without retraining.
- **Speaker-ID auxiliary** - additive-angular-margin softmax over the training speakers on g_t (training only, weight
  0.1); TD-SpeakerBeam gained up to about 1 dB from this [D2][H].

### A.4 Streaming state and memory

| State | Size |
|---|---|
| Time-LSTM h and c, 5 blocks x 32 bands x 96 x 2 | 30,720 floats (123 KB) |
| Analysis history | 512 samples |
| Two past spectra for the deep filter | 2 x 257 complex |
| Overlap-add buffer | 512 samples |
| Conditioning, double-buffered (D.6) | 2 x (8,192 FiLM floats + 2 modules x 100 x 128 key/value floats) = 2 x 33,792 floats (270 KB) |

Everything is allocated once at engine start. A hop never allocates.

### A.5 "Everyone" (all speech, no noise) mode

- Same network and weights. e_t, g_t and the target tokens are replaced by learned nulls, the mode flag is 1, and the own
  voice stays as the negative.
- Training target - the sum of every talker except the user (direct path plus 50 ms of early reflections), noise and music
  removed. 25% of training examples (choice from [D2]).
- Precedents - USEF-PNet trains one checkpoint for personalized and plain enhancement with a dummy enrollment and kept both
  scores (13.00 vs 13.00 dB personalized) [D2][H]; PVQE with an empty embedding still removed most noise but let
  background speech through (5.4 dB vs 33-34 dB suppression), which is exactly the Everyone behavior [D2][H].

### A.6 Output post-processing (in the I/O callback, after upsampling)

1. **Mix-back floor** - y = g_voice s_hat + g_bg x_d, where x_d is the 48 kHz input delayed by exactly the model's processing
   delay and g_bg is the "Background" setting (Off, -30, -20, -12 dB). g_bg is forced to 0 while the own-voice head says
   the user is talking (p_o above 0.5, 20 ms attack, 300 ms release), so the floor never replays the user's voice. The same
   idea as DeepFilterNet's `atten_lim_db` and the mix ratio in phone-streamed hearing aids [D2].
2. **Absence gate** - when p_t stays below 0.2 for 300 ms, add -10 dB (release 300 ms, attack 5 ms when p_t rises above
   0.5). The network is trained to go quiet by itself (F.4); the gate is a safety net.
3. **Mute-while-I-talk** (AirPods default on) - when p_o rises above 0.6, duck the whole output by 12 dB (10 ms attack,
   150 ms release). Delayed own voice is most disruptive near 200 ms of delay (2 to 3 times more disfluencies than no delay,
   17 talkers) [6][M], which is the AirPods delay range.
4. **Identity gain** from the background identity monitor (D.11), -10 dB when the output stops matching the target.
5. **Limiter, gain cap, fades** (E.4).

### A.7 Wired siblings (Earshot-LL)

| Profile | Analysis / synthesis / hop | Bins and bands | Algorithmic latency | Size trained | GMAC/s | M4 P-core per hop |
|---|---|---|---|---|---|---|
| LL-2 | 16 ms / 4 ms / 2 ms (K = 256, M = 32) | 129 bins, 32 bands of 2, 4 and 8(+1) bins | 4 ms | LL-S (D48, Hb32, Ht64, 4 blocks) | 3.86 | 254 us median, 750 us p99.9 [H, measured] |
| LL-1 | 16 ms / 2 ms / 1 ms (K = 256, M = 16) | same | 2 ms | LL-S | 7.7 (analytic) | not measured; about 250 us per 1 ms hop expected [L] |

- Same code as Q-M with a config change. Window pair from the same formulas. Dual-window STFT at 4 ms latency beat
  Conv-TasNet at half the compute in Wang, Wichern, Watanabe and Le Roux [D2][H]. Expect lower quality than Q-M - at 3 ms a
  0.23 GMAC/s model lost about 1.5 dB SI-SDR, while a 10.95 GMAC/s model matched the 20 ms baseline [D5][5][H].
- **Only one LL profile is trained**, chosen in C.4 from the measured hardware latency. If the phone is fast enough for Q-M
  under 20 ms, no LL model is needed at all.
- **Distillation** - LL training uses the finished Q-M as a teacher (add an L1 term to Q-M's output on the same input,
  weight 0.5), since Q-M's output is time-aligned with the target reference [L, design choice; distillation helped small
  personalized models in E3Net [D2]].

### A.8 Choices considered and rejected

| Option | Why not |
|---|---|
| Q-L (11 GMAC/s) | Fits the phone (557 us per hop) but gets fewer than half the training examples of Q-M in the same Mac time (1.3). |
| GRU blocks (Earshot-BSR as in [D2]) | 11 to 14x slower to train on MPS in torch 2.7.1 [D5][H]. |
| Band-axis transformer or time-axis TCN | Measured slower to train on MPS than LSTMs at these shapes - transformer layer across bands 203 vs 133 ms, 5-layer dilated TCN 311 vs 121 ms per layer pair, fwd + bwd [H, measured, Appendix A.2]. |
| Lookahead or a 16 ms synthesis window on AirPods | Would buy quality (a 32/32 ms window was the 10.2 dB ceiling in [3]), but the user asked for the lowest processing latency on top of Bluetooth [D1]. Kept as an A/B experiment, not the default. |
| 24 kHz now | The data pack, DNSMOS and the encoder study are 16 kHz [D4]. Moved to v2 (A.9). |
| Neural Engine or GPU on the live path | 1.2. |

### A.9 Upgrades after v1 passes acceptance (in priority order)

1. **Group mode (v1.5)** - enroll 2 or 3 people and hear all of them. Tokens of all chosen people are concatenated (the
   attention takes a union naturally) and their global vectors averaged. Needs a fine-tune with 10% two-target examples.
   Conversation-partner extraction is shown to work in [D2] (8.19 dB for 2 talkers).
2. **Camera-assisted enrollment (v1.5)** - while enrolling, Vision face landmarks (`VNFaceLandmarks2D.innerLips`, iOS 11+)
   [8][H] mark frames where the pointed-at person's mouth moves; only those frames make tokens, and frames where they are
   silent become a negative clip, as in the positive/negative enrollment of [D2][D5]. Visual cues fix most target confusion
   (PVQE SI-SNRi -2.31 to 9.03 dB with the target's mouth video) [D5][H]. Camera use is limited to enrollment, not the live
   path (power).
3. **24 kHz super-wideband (v2)** - add 4 bands of 32 bins for 8 to 12 kHz (+12% compute), with a 24 kHz data pack
   (LibriTTS-R, VCTK, GLOBE, DEMAND 48 kHz, FSD50K originals). Target-talker content above 8 kHz helps speech-in-speech
   recognition while masker content above 8 kHz does not [9][M, abstract]. Check the UIUC patent on extended-high-frequency
   extraction first [10][L].
4. **Neural Engine context booster (v2)** - a larger causal model (about 5 to 10 M parameters) runs every 64 ms on the
   Neural Engine over the last 1 s, and sends a compressed per-band hint that Q-M reads through an extra zero-initialized
   cross-attention between blocks 2 and 3, 128 ms late. Knowledge boosting showed this pattern lifts a small causal TSE
   model from 3.95 to 7.53-7.70 dB SI-SDR with hints 48 ms late (large model alone 8.52 dB), with joint training and the
   delay simulated by shifting the hints [11][H]. Real-time safety holds because the audio path never waits - a missing
   hint means "reuse the last one", and the model is trained with 10% missing hints. Ship only if it adds at least 1 dB
   SI-SDRi or 5 points of picked rate at SIR 0 dB or below. The gain on a 1.6 M-parameter fast model is unknown; the
   published gains are for a 23 k-parameter one.
5. **Two-channel input (v2)** - only if Phase 0 shows the built-in stereo polar pattern works in measurement mode at the same
   latency [D3]. Multi-mic gains were large in [D2] (mono 2.89, 2-mic 4.75 dB).

---

## B. Enrollment

### B.1 Capture flow

1. The user taps "Add person" (or double-presses an AirPods stem, E.1) while the person talks.
2. The app records from the same built-in mic and session as the live path, so the enrollment channel matches the live
   channel (over-suppression is worst when enrollment and live conditions differ [D2][H]). If Phase 0 shows measurement mode
   honors a back-facing cardioid data source, use it for enrollment only.
3. A ring on screen fills with detected voiced speech. Capture stops at 6 s of voiced speech, or after 15 s.
4. The user's own speech is removed from the clip - frames whose 1.5 s WeSpeaker window matches the own-voice embedding
   (cosine above 0.6, to be calibrated) are dropped before the checks.
5. Checks (B.3) run in about 0.2 s. On success the person gets a name (default "Person 1") and becomes the target.

Length - 5 s instead of 3 s at 5 dB SNR roughly halved WeSpeaker's EER (4.3% to 2.4%) [D4][H]; 89% of Look Once to Hear
users found 5 s acceptable [D2][H]. Earshot asks for 6 s of voiced speech and accepts 4 s.

### B.2 Encoders

| Part | Runs on | Cost | Output |
|---|---|---|---|
| WeSpeaker ResNet34 (VoxCeleb, plain), Kaldi filterbank inside the graph as one Conv1d [D4] | Core ML, `computeUnits = .cpuOnly` (the Neural Engine gave wrong embeddings [D4]); launch check against a stored clip, cosine at least 0.999 | 7.7 to 18 ms per 5 s on the Mac [D4][H]; phone not measured | e (256) |
| Token encoder (A.3) | the C runtime, same weight file as Q-M | about 1 ms [L] | tokens, g (256) |
| FiLM generator (A.3) | the C runtime | 0.6 M MACs, well under 1 ms [L] | gamma and beta for the 2 conditioning points |

### B.3 Quality checks (initial thresholds, then calibrated on the validation set)

| Check | Rule | Message if it fails |
|---|---|---|
| Enough speech | voiced speech at least 4 s | "Keep listening - I need a few more seconds of them talking" |
| SNR | median voiced-frame energy at least 5 dB above the 10th-percentile frame energy (warn below 10 dB) | "Too noisy - move the phone closer to them" |
| One voice | median pairwise cosine of WeSpeaker embeddings over 1.5 s windows (0.5 s hop) at least 0.55 | "I heard more than one voice - try when only they are talking" |
| Not the user | cosine to the own-voice embedding below 0.5 | "That sounded like your voice" |
| Not a duplicate | cosine to every saved person below 0.7 | "This sounds like Maya - update Maya instead?" |
| No clipping | under 0.1% of samples at full scale | "Too loud - hold the phone a bit farther away" |

Calibration rule - set each threshold so that 95% of good single-talker enrollments in the validation set pass and at least
80% of the matching bad cases (two talkers, own voice, SNR below 0 dB) fail. Report the achieved rates in the eval report.

### B.4 The user's own voice

- Enrolled once in onboarding - 12 s of reading a short passage in a quiet spot, SNR at least 20 dB, same mic and session.
- Gives e_o, g_o and up to 32 negative tokens. Re-recordable in Settings.
- Used in every mode, because hearing yourself 50 to 200 ms late is disruptive [6][M] and because phone-streamed hearing aid
  users rated their own voice worse with 65-75 ms delay [D2][H].

### B.5 Multiple saved people

- Up to 8 people. Per person the app stores the enrollment audio (16 kHz FLAC, about 0.15 MB), e, the token keys and values
  for each attention module, g, the seat direction (head yaw at enrollment, E.1), a creation date and quality scores.
- **The audio is kept** so tokens can be recomputed when a new model ships. Tokens are tied to a model version, and the app
  re-encodes all people at the first launch after an update.
- Storage - app container with file protection `complete` (iOS Data Protection), never uploaded. The user can delete a
  person, which deletes the audio. Onboarding asks the user to tell people they are enrolling them.
- Switching between saved people costs only the FiLM generator run (well under 1 ms) and a double-buffer flip (D.6).

### B.6 Refreshing a voiceprint over time

- **Collect** - during use, keep input audio segments (not the processed output, so model artifacts never enter the
  voiceprint) where p_t is above 0.9, p_o below 0.1, and the identity monitor's cosine is above its high threshold, up to
  6 s per session.
- **Update** - at the end of a session, recompute tokens from the original enrollment plus the collected audio. The original
  always keeps at least half the token slots. The global vector becomes a normalized weighted mean with at most 0.3 weight on
  new audio.
- **Guard** - reject an update whose new global vector has cosine below 0.7 to the original (likely drift onto someone
  else), keep the last 3 versions for undo, and never update in Everyone mode.
- **Prompt** - suggest re-enrolling when a saved person is older than 60 days or the monitor's session mean cosine falls 0.1
  below its first-week mean.

### B.7 Licenses

- WeSpeaker code is Apache-2.0, but its VoxCeleb-trained weights follow the VoxCeleb terms, granted "for research purposes"
  [D5][H]. Fine for the user's personal build. A distributed build turns the WeSpeaker inputs into the learned nulls the
  model was trained with (A.3), or trains its own encoder on the CC BY / CC0 speech in the pack (about 4,500 speakers [D4]).

---

## C. Latency budgets (mic to ear)

### C.1 How latency is counted

- X - everything outside the app's two I/O buffers: ADC, input driver and safety offset, output driver, DAC, and for
  Bluetooth the A2DP encoder, radio, jitter buffer and decoder. It is unknown for this phone and route until Phase 0.
- A same-callback pass-through measures X + 2B, where B is the I/O buffer duration (Superpowered's method) [D3][H].
- On top of that, a frame model adds its synthesis window 2M when the hop runs on a worker thread whose result is used in
  the next I/O callback. I verified this with a sample-exact simulation of callbacks at B frames, 3:1 resampling and
  overlap-add (Appendix A.5) [H, computed]:

| Profile | Extra delay over pass-through, inference inside the callback | Extra delay, worker thread (result used one callback later) |
|---|---|---|
| Q (hop 4 ms) | 4.00 ms at B = 4 ms, 6.67 ms at B = 1.33 ms, 7.33 ms at B = 0.67 ms | 8.00 ms for every B tested (0.67 to 4 ms) |
| LL-2 (hop 2 ms) | 2.00 ms at B = 2 ms, 3.33 ms at B = 1.33 ms | 4.00 ms (4.67 at B = 1.33 ms) |
| LL-1 (hop 1 ms) | 1.00 ms at B = 1 ms, 1.67 ms at B = 0.67 ms | 2.00 ms (2.33 at B = 0.67 ms) |

- So total = X + 2B + 2M + R + W B, where R = resampler group delay (0.26 ms both ways at 1-4 kHz, computed) and W = extra
  I/O periods of deadline slack given to the worker (W = 1 means the worker may take up to 2 periods).
- The worker design is chosen over in-callback inference (even though in-callback can be 0.7 to 4 ms faster) because a
  slow hop then becomes a concealed hop, not a dropout, and it keeps working when iOS changes the buffer size [D3].

### C.2 Wired USB-C earbuds (LL-2 shown; LL-1 and Q-M in the last rows)

| Line | ms | Source and confidence |
|---|---|---|
| Hardware latency X (mic ADC, drivers, DAC, USB device) | 14.3 (range 6.3 to 24.3) | Superpowered crowd round trips for iPhone 15/16-family rows, median 17 ms at 64 frames, minus 2 x 1.33 ms buffers [D5][M]. Speaker path, not USB-C; no public USB-C number exists [D3]. **Measure in Phase 0.** |
| Input I/O buffer, 32 frames at 48 kHz | 0.67 | requested; iOS may grant more, so read `ioBufferDuration` back [12][H] |
| Decimation 48 to 16 kHz (minimum-phase FIR) | 0.13 | computed group delay at 1-4 kHz [H, Appendix A.3] |
| Synthesis window 2M (LL-2) | 4.00 | window pair definition [3][H] and simulation (A.5) |
| Worker deadline slack, W = 1 | 0.67 | design choice; LL-S measured 254 us median, 750 us p99.9 on an M4 P-core (1.1) [H] |
| Interpolation 16 to 48 kHz | 0.13 | computed [H] |
| Output I/O buffer, 32 frames | 0.67 | [12][H] |
| **Total, LL-2** | **20.6 (12.6 to 30.6)** | sum; at the crowd median it misses 20 ms |
| Total, LL-2 with W = 0 | 19.9 | only if the phone's p99.9 compute fits in 0.67 ms |
| Total, LL-1 (2M = 2 ms), W = 1 | 18.6 (10.6 to 28.6) | sum |
| Total, Q-M on wired (2M = 8 ms, B = 64, W = 1) | 26.6 (18.6 to 36.6) | sum; meets 20 ms only if X is at most 7.7 ms |

### C.3 AirPods Pro 2 over A2DP (Q-M, primary path)

| Line | ms | Source and confidence |
|---|---|---|
| Mic ADC and input driver (part of X) | not separable | reported `inputLatency` was 2.4 ms on an iPhone 12 mini with A2DP output [D3][M] |
| Input I/O buffer, 64 frames | 1.33 | 64 frames were granted with A2DP output in a developer test [D3][M]; read back on device |
| Decimation | 0.13 | computed [H] |
| Synthesis window 2M (Q-M) | 8.00 | [3][H], simulation (A.5) |
| Worker deadline slack, W = 1 | 1.33 | Q-M measured 385 us median, 1.66 ms p99.9 on a loaded M4 P-core (1.1) [H]; W = 1 costs 1.33 ms, which is small next to Bluetooth |
| Interpolation | 0.13 | computed [H] |
| Output I/O buffer, 64 frames | 1.33 | [12][H] |
| **App-side subtotal** | **12.3** | sum; of this, 9.6 ms is added over a plain pass-through app on the same route |
| A2DP output path (iOS encoder, radio, AirPods jitter buffer, decoder, DAC) | about 43 to 220 | Pro 2 tap-to-sound 126 ms vs 83 ms for the built-in speaker in the same test, so about 43 ms of extra output latency [13][D5][M]; AirPods 2 measured 215-220 ms falling to 155-160 ms over 20-30 min [14][M]. Nobody has measured mic in, app, AirPods Pro 2 out [D5]. **Measure at every connect.** |
| **Total, mouth to ear** | **about 55 to 235** | sum; AirPods Pro 1 expected worse (167 vs 126 ms in the same tap test) [13][M] |

Consequences on AirPods

- **Comb filtering is not the problem** - ANC isolation is about 24.5 dB for Pro 2 and 21 dB for Pro 1 [15][M], so the
  leaked direct sound sits 20 to 25 dB under the processed voice.
- **Lip-sync is** - audio lagging video becomes detectable at about 125 ms and unacceptable at about 185 ms for broadcast
  material (ITU-R BT.1359) [16][M]. The app shows the measured delay and explains it.
- **Own voice must be removed** (B.4, A.6), because 200 ms is the most disruptive delayed-feedback range [6][M].

### C.4 Choosing the wired profile from Phase 0

1. Phase 0 measures the wired pass-through round trip at B = 16, 32 and 64 frames (loopback, D.10), giving X.
2. Predicted totals - Q-M: X + 2B + 8 + 0.26 + W B; LL-2: X + 2B + 4 + 0.26 + W B; LL-1: X + 2B + 2 + 0.26 + W B.
3. Pick the highest-quality profile whose prediction is at most 19 ms (1 ms margin): Q-M if X is at most about 7.7 ms,
   LL-2 up to about 13.4 ms, LL-1 up to about 15.7 ms (B = 32, W = 1).
4. If X is above about 15.7 ms, no frame-based model meets 20 ms on this phone. Ship LL-1, display the measured number
   (for example "21 ms"), and say so plainly.
5. Train only the chosen LL profile (35 to 40 h, F.5).

---

## D. iOS real-time engine

### D.1 Session and route configuration

- `.playAndRecord`, mode `.measurement`, options `[.allowBluetoothA2DP]` only. Never `.allowBluetoothHFP` (old name
  `.allowBluetooth`), `.defaultToSpeaker` or `.mixWithOthers` [D3][D5][H].
- Preferred sample rate 48 kHz. Preferred I/O buffer 32 frames on wired, 64 on AirPods. Read back `ioBufferDuration`,
  `sampleRate`, `inputLatency`, `outputLatency` after activation and after every route change, and also check the actual
  `inNumberFrames` in the callback (the request is a hint [12][H]; Sound Recognition or Vocal Shortcuts silently forced
  1104-frame buffers on iOS 18 [17][M]).
- `setPrefersNoInterruptionsFromSystemAlerts(true)`. Select the built-in mic after activation and re-assert it after every
  route change (USB-C EarPods bring their own mic) [D3].
- Output is allowed only on `.headphones`, `.usbAudio` and `.bluetoothA2DP`. Any other output route mutes immediately
  (feedback and loudness risk) [D3].
- Measurement mode lowers output level and uses the primary mic [18][H], so Earshot has its own gain stage (E.4).

### D.2 Audio I/O, sample rates and resampling

- One RemoteIO unit, input enabled on bus 1, 48 kHz Float32 mono. One render callback in C pulls the mic with
  `AudioUnitRender` and writes the output in the same callback, the layout Superpowered's latency app uses [D3].
- Decimator and interpolator - minimum-phase polyphase FIR, 177 taps at 48 kHz (designed in Python, embedded as a C array),
  group delay 0.11-0.15 ms at 0.5-4 kHz [H, computed]. Cost about 5.6 M MAC/s for both directions [L, arithmetic]. An
  8th-order elliptic IIR has similar delay (0.09-0.13 ms) and is the alternative; the FIR is chosen because block
  processing has no feedback state to get wrong.
- Training applies the same decimator's 0-8 kHz response to the mixture as part of the mic-chain augmentation (F.3).

### D.3 Threads and scheduling

| Thread | Work per period | Rules |
|---|---|---|
| RemoteIO I/O thread (system, real time) | copy mic samples, decimate, push to the input ring, signal the worker when a hop is complete, pull the output ring, interpolate, safety chain, meters | constant time, never waits, never calls the model |
| Inference worker (1 thread) | wait on a semaphore, run one hop of the model, push 64 output samples and the two head values, write the latest 2 s of output into the identity-monitor ring | `THREAD_TIME_CONSTRAINT_POLICY` with period 4 ms (2 ms for LL-2), computation 1.5 ms, constraint 3 ms [19][H for the API, L for the values]; joined with `os_workgroup_join` to the RemoteIO workgroup from `kAudioOutputUnitProperty_OSWorkgroup` [20][H]; `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)` on this thread (iOS 18+, per thread) [21][H]; flush-to-zero set in FPCR |
| Motion queue | `CMHeadphoneMotionManager` updates, smoothed yaw into an atomic | not real time |
| Identity monitor (background, QoS utility) | every 0.5 s, speaker embedding of the last 1.5 s of output (D.11) | may allocate; never touches audio buffers except by reading the SPSC ring |
| Main | session, routes, interruptions, enrollment encoders, UI | may allocate |

- **Deadline and concealment** - the I/O callback looks for the hop that should have finished W + 1 periods after it
  completed. If it is missing, the callback plays the delayed input times the floor gain for that hop, crossfaded over 2 ms,
  and counts a late hop. More than 1 late hop per minute triggers a model downshift (D.9).
- **Rings** - single-producer single-consumer lock-free ring buffers with C11 atomics, power-of-two sizes, preallocated.

### D.4 Inference runtime - hand-written C with Accelerate

- **Engine** - one C file per concern: `stft.c` (window pair, vDSP FFT), `lstm.c` (batched gate math), `qnet.c` (blocks,
  conditioning, decoder, deep filter), `weights.c` (load and verify). Weights are float32, pre-transposed to [K, N] because
  the PyTorch layout was 1.1 to 4x slower in sgemm at these shapes [D6][H]. Expected size 600 to 900 lines [L].
- **Why not BNNS Graph first** - Apple recommends it for audio and it is real-time safe when the workspace and outputs are
  caller-provided [2][D5][H], but which Core ML ops compile from file and how LSTM state is carried are undocumented, and
  Apple publishes no timing for a model of this kind [D5][H]. The C engine gives exact parity tests against a NumPy
  reference and was already measured (1.1). BNNS Graph (FP16, which Apple calls "significantly faster" [2][H]) is
  benchmarked against it in Phase 2 and replaces it only if it is at least 1.5x faster at the p99.9 with bit-close output.
- **Not Core ML on the audio path** - 1.2.
- **Parity** - the training export writes golden vectors (F.7). The C engine must match the PyTorch streaming forward to
  max abs error 1e-4 on output samples and 1e-3 on head values, run at app launch on a stored 2 s clip (about 0.2 s).

### D.5 Real-time safety rules (checked in code review and by a test harness)

- No `malloc`, `free`, locks, Objective-C or Swift calls, logging, file I/O or `dispatch` in the callback or the worker.
- Every buffer is allocated at engine creation. Parameter changes arrive through atomics or the conditioning double buffer.
- The callback's work is constant per frame. The worker's work is constant per hop.
- Denormals flushed to zero on the worker thread (LSTM cell states decay toward tiny values).
- Fuzz harness on macOS - random buffer sizes from 16 to 1104 frames, NaN and Inf inputs, full-scale noise, silence for an
  hour. Outputs must stay finite and under the ceiling. NaN in the model output resets the model state and conceals one hop.

### D.6 Conditioning hand-off

Two conditioning slots (A.4). The main thread computes a new person's FiLM and tokens into the inactive slot, then flips
an atomic index. The worker reads the index once at the start of each hop, so a switch takes effect at the next hop
boundary with no lock. The LSTM state is not reset on a switch; the model is trained for mid-sentence switches (F.3).

### D.7 Route changes

On `routeChangeNotification` (main thread):

1. Set the mute atomic (20 ms fade in the callback).
2. Validate the new output port type (D.1). Re-assert the built-in mic.
3. Re-read latencies and the granted buffer. Select the model profile for the route (Q-M on AirPods, the chosen LL profile
   on wired) and the deadline slack W.
4. Reset the model state if the profile changed. Both profiles are allocated at engine start, so no allocation happens here.
5. Show the cached measured latency for this route, and ask for a new measurement if none exists or it is older than 10 min.
6. Fade in over 400 ms.

### D.8 Interruptions, background and lifecycle

- `UIBackgroundModes` `audio`; start only in the foreground; keep running with the screen locked [D3][H].
- Interruption began - mute, mark stopped. Ended with `.shouldResume` - try `setActive(true)` and restart; if it fails
  (it will from the background [D5][H]), post a local notification "Tap to resume Earshot".
- `mediaServicesWereResetNotification` - rebuild the session, the unit and the engine.
- iOS 27 lifecycle notifications replace the interruption notification after an SDK update [D3]; the handler is written
  behind one small adapter.

### D.9 Thermal, battery and downshift

- Observe `ProcessInfo.thermalState` [22][H] and Low Power Mode. `.serious` or Low Power Mode - switch to Q-S at a hop
  boundary (its conditioning is precomputed at enrollment), 30 ms fade. `.critical` - stop processing with a notice.
- Late hops above 1 per minute - downshift as above.
- Battery acceptance is measured with the Instruments Power Profiler (WWDC25 session 226 shows per-subsystem CPU, GPU,
  display and networking power on iPhone) [23][H for the tool].
- Rough expectation, not a measurement - the 3,582 mAh battery [24][M] is about 13.8 Wh at an assumed 3.85 V. At 0.6 to
  1.2 W total with the screen off, that is 9 to 4.5% per hour [L].

### D.10 In-app latency measurement

- **Loopback test** - the user holds an AirPod (or a wired earbud) against the phone's bottom mic. The app plays ten 50 ms
  logarithmic chirps (300 Hz to 6 kHz) at 0.5 s intervals through the current output, records the mic in the same callback
  timeline, and cross-correlates. The median peak lag is the pass-through round trip X + 2B. Reject the run if the peak is
  less than 10 dB above the side lobes.
- **Display** - mouth to ear = measured pass-through + 2M + R + W B (known exactly in code), shown as for example "AirPods -
  141 ms (measured 3 min ago)". Without a fresh measurement, show the `inputLatency + outputLatency + 2B` estimate marked
  "estimated" (reported values have been wrong by tens of ms [D3][M]).
- **When** - at first connect of each device, then suggested again at 10 and 30 min on AirPods (the delay drifted by about
  60 ms in 20-30 min in one report [14][M]).
- Phase 0 uses the same code.

### D.11 Identity monitor

- Every 0.5 s, if the mean of p_t over the last 1.5 s is above 0.6, embed the last 1.5 s of model output with WeSpeaker
  (Core ML, CPU) and compute cosine to the target (c_t) and to the user (c_o).
- If c_t stays below a low threshold for 2 checks in a row, set the identity gain to -10 dB (200 ms ramp) and show "Lost
  them? Tap to re-lock". If c_o is above its threshold, duck 10 dB more. Thresholds are calibrated in G so that 95% of
  correct-target windows pass.
- Basis - in NTT/BUT's study a verification step on the output detected an absent target better than the extractor alone
  (EER 8.9% with 10 s of enrollment) [D4][H].
- Limits - it reacts in 1 to 2 s, so it only catches sustained confusion. The distributed build uses the token encoder's g
  instead of WeSpeaker.

---

## E. User experience

### E.1 Picking the person you are looking at

| Way | What happens | Limits |
|---|---|---|
| Point the phone (primary) | "Add person", phone top or back toward them for about 6 s while they talk (B.1) | needs the person to talk; phone within about 1.5 m works best (enrollment EER rises steeply at 0 dB SNR [D4]) |
| Stem double press while facing them | starts the same capture from the phone mic; the current head yaw becomes their seat direction | phone on the table must be near the person |
| Look to switch (saved people) | turning the head to within 20 degrees of a saved person's seat direction for 1 s highlights them; a double press switches (default). A "look only" setting switches without the press | yaw is relative and drifts; it is re-zeroed on every enrollment and every lock [D3]. Motion updates in a `.playAndRecord` + A2DP session and with the screen locked are unverified, so this is a Phase 0 gate |
| Tap in the list | switches immediately | - |

AirPods stem mapping (A2DP remote commands, only with the A2DP-only session [D5][H])

| Press | Command | Earshot action |
|---|---|---|
| Single | `togglePlayPauseCommand` | pause or resume Earshot (fade) |
| Double | `nextTrackCommand` | lock onto the saved person in the look direction; if none is within 20 degrees, enroll whoever talks next |
| Triple | `previousTrackCommand` | toggle Person and Everyone |
| Press and hold | reserved by iOS for noise control [D3] | not available to apps |

### E.2 Switching

- Switching flips the conditioning at the next hop (D.6). The model is trained with mid-segment switches so it follows the
  new person within about 0.5 s (acceptance in G.3). A streaming AV extractor without such training needed about 1 s to
  recover after a switch [25][H].
- A 30 ms crossfade to the floor and back masks the transient.

### E.3 Controls

| Control | Range | AirPods default | Wired default |
|---|---|---|---|
| Voice boost | 0 to +18 dB in 3 dB steps | +6 dB | +6 dB |
| Background (mix-back of the delayed room) | Off, -30, -20, -12 dB | Off | -20 dB |
| Mute my voice (duck while I talk) | on, off | on | off |
| Mode | Person, Everyone, Off | Person | Person |
| People | list, add, rename, re-enroll, delete | - | - |
| Latency | measured value, "measure again" | shown | shown |

### E.4 Safety

- **Limiter** - feed-forward peak limiter at -3 dBFS, 0.5 ms attack, 50 ms release, no lookahead (lookahead would add
  latency), followed by a soft clipper above -1 dBFS.
- **Gain cap** - total digital gain from mic to output at most +24 dB; Voice boost and the model's own gain are both inside
  this cap.
- **Fades** - 400 ms raised-cosine fade-in on start, after route changes and after interruptions; 20 ms fade-out on any
  mute.
- **Runaway guard** - if output RMS exceeds input RMS by more than 30 dB for 200 ms (feedback or a bug), duck to -20 dB and
  warn.
- **Hearing safety** - onboarding asks the user to turn on Reduce Loud Audio (Settings - Sounds and Haptics - Headphone
  Safety), which iOS measures most accurately with Apple headphones [33][M]. ITU-T H.870 sets the adult weekly allowance at
  1.6 Pa^2 h, equal to 80 dBA for 40 h [26][H]. The app can show the HealthKit headphone exposure history (read only).
- **No medical claims** - Earshot is a listening tool, not a hearing aid; the app never says it treats hearing loss.

### E.5 Onboarding (about 4 minutes)

1. What Earshot does and does not do, in plain words (including "over AirPods your friend's voice arrives about 0.1 to 0.2 s
   late").
2. Microphone and motion permissions.
3. Output check - AirPods or wired.
4. AirPods checklist - Noise Cancellation on (offer a Shortcuts automation "When Earshot opens - Set Noise Control Mode to
   Noise Cancellation" [D3]), Conversation Awareness off [D3], Spatial Audio Off for Earshot in Control Center [D3],
   Reduce Loud Audio on.
5. Latency check (D.10), with the result shown.
6. Own-voice enrollment (B.4).
7. Practice - add a person with a demo voice played into the AirPods from a stored file, and hear before and after.
8. Safety notes.

### E.6 What differs on AirPods and wired

| Aspect | AirPods Pro 1 and 2 (primary) | Wired USB-C (secondary) |
|---|---|---|
| Model | Q-M, 8 ms algorithmic | LL-2 or LL-1, 4 or 2 ms (C.4) |
| Latency label | "AirPods - high delay - 141 ms (measured)" | "Wired - 19 ms (measured)", or the honest number if over 20 |
| Direct sound | hidden by ANC (21 to 24.5 dB) [15] | sealed in-ear earbuds recommended; open EarPods let the direct sound through, so a 20 ms copy colors it |
| Own voice | network suppression plus 12 dB duck | network suppression only (delay under 20 ms) |
| Background mix-back | off by default, never while the user talks | -20 dB by default |
| Head tracking | look to switch | none (no sensors) |
| Buttons | single, double, triple press | EarPods center button play and pause, if delivered as remote commands (Phase 0) |
| Pro 1 vs Pro 2 | same features; Pro 1 has less isolation and more delay [13][15] | - |

---

## F. Data and training plan

### F.1 Datasets (the pack in [D4], already being fetched by `dataprep/`)

| Part | Source | License | Kept on disk |
|---|---|---|---|
| Speech | LibriTTS-R train, 2,311 speakers, about 60 s each (streamed from 82 GB of tarballs) | CC BY 4.0 [D4][D5][H] | 2.6 GB |
| | VoxPopuli en, 2 shards, about 800 speakers | CC0 [D4][H] | 0.9 GB |
| | GLOBE V2, 8 shards, about 1,300 speakers (66-79% US rows; denoised with Adobe Podcast Enhance - listen first) | CC0 [D4][H] | 0.9 GB |
| | VCTK 0.92 mic1, 102 training speakers by HTTP range reads | CC BY 4.0 [D4][H] | 0.25 GB |
| | Emilia-YODAS EN, 1 tar (optional, the user must accept the terms) | CC BY 4.0, gated [D4][H] | 0.4 GB |
| Noise | FSD50K CC0 and CC BY clips of restaurant classes plus 4 h of music | per clip; CC BY needs credits [D4][M] | 1.0 GB |
| | DEMAND, 8 scenes x 4 channels (SCAFE held out) | CC BY-SA 3.0 in its PDF vs CC BY 4.0 on Zenodo; treat as CC BY-SA [D4][H] | 0.35 GB |
| | MUSAN noise plus 6 h of music | "only content that allows commercial use" [D4][H] | 0.9 GB |
| Rooms | 4,000 simulated rooms (pyroomacoustics, MIT), measured RT60 kept per room; phone in hand 60%, on table 40% with a mirrored table image | own | 0.7 GB |
| | OpenSLR 28 real RIRs | Apache-2.0 [D4][H] | 0.3 GB |
| Test only | MIT IR Survey bar and restaurant IRs | CC BY 4.0 per mirrors only [D4][M] | under 0.1 GB |
| | WHAM! test split, 1,000 clips | CC BY-NC 4.0, evaluation only [D5][H] | 0.3 GB |
| | Libri2Mix test (from LibriSpeech test-clean, on disk) with BUT's enrollment list | list is evaluation-only [D4][H] | rendered on the fly |
| | EdAcc test, VoxPopuli en_accented, VOiCES subset, DiPCo 1-2 sessions, Lombard GRID | CC BY-SA 4.0, CC0, CC BY 4.0, CDLA-Permissive, CC BY 4.0 [D4][M] | about 1.0 GB |
| Own recordings | loudspeaker rig on the iPhone (G.4), phone mic sweep | own | about 0.5 GB |
| **Total** | about 4,500 training speakers, about 75 h of speech | | **about 9.8 GB** |

Disk - 41 GiB free on 2026-11-07 at 02:47 (`df`), so the pack fits with the 10 GB floor [H, measured]. Checkpoints add about
0.4 GB (10 resumable Q-M checkpoints of about 35 MB with optimizer state) [L, arithmetic].

Never used for training - WHAM! (non-commercial), EARS and Expresso (non-commercial), Look Once to Hear code (UW
non-commercial), BUT SpeakerBeam (evaluation-only), USEF-TSE (CC BY-NC) [D5][H].

### F.2 Mixture recipe (rendered on the fly, 16 kHz)

| Element | Value |
|---|---|
| Mode | 75% Person, 25% Everyone (A.5) |
| Target presence (Person mode) | 90% present (25% of these start or stop inside the segment), 10% absent - enrollment from a speaker not in the mix, reference all zeros [D4] |
| Target switch | 10% of Person examples switch the conditioning from speaker A to B at a random time between 1 and 3 s (both present); the reference follows with a 50 ms crossfade |
| Competing talkers | 0 (10%), 1 (35%), 2 (30%), 3 (25%), each active 50-100% of the segment [D4] |
| Levels | each talker N(0, 4) dB at 1 m, geometry sets the level at the mic, then rescale so the SIR of target vs loudest interferer lies in -8 to +12 dB. Quieter targets are required, since a production personalized model picked the target only 6.25% of the time when it was quieter [D5][H] |
| Own voice | present in 50% of examples, a different speaker 0.25-0.8 m from the mic, active 20-60%, 0 to 12 dB louder than the target at the mic, never in the reference; its enrollment (cleaner, SNR 15-40 dB, 6-10 s) is always given as the negative, except 10% where it is replaced by the learned "no own voice" null |
| Babble | 70% of examples, 6 to 20 talkers from the pool, each with its own RIR |
| Noise | 80% of examples; DEMAND 30%, FSD50K 40%, MUSAN 20%, other 10% |
| Music | 20% of examples, 5 to 20 dB below the target |
| SNR | target vs babble + noise + music N(5, 5) dB clipped to -8 to 25 dB; 10% quiet rooms at 20-40 dB. Real listening situations averaged about 4-5 dB in noise [D4][H] |
| Rooms | 75% simulated bank, 15% OpenSLR 28 real, 10% dry |
| Reference | direct path plus the first 50 ms of the target's RIR, 5 ms fade, no time alignment (keeps the model causal) [D4] |
| Mic chain | high-pass 60-200 Hz, 0-2 peaking EQs of +-4 dB, self-noise -95 to -75 dBFS, level -40 to -15 dBFS RMS, 3% hard and 3% soft clipping; the app decimator's response (D.2); the same linear EQ on the reference [D4] |
| Lombard | 30% of targets and interferers in loud scenes get +3 to +8 dB level and a +2 to +4 dB shelf above 1 kHz [D4] |
| Enrollment | 1 or 2 other utterances of the target, 3-6 s; 80% noisy (babble + noise at 0-25 dB SNR); 30% with one weaker talker 5-15 dB below; 50% same room, 30% another room, 20% dry; same mic chain [D4] |
| Splits | speaker-disjoint; 3% of speakers held out for validation; official test speakers never appear as interferers or babble [D4] |

### F.3 Losses (all computed in fp32)

| Term | Formula | Weight | When |
|---|---|---|---|
| Thresholded SNR, target present | -10 log10( ||s||^2 / (||s - y||^2 + tau ||s||^2) ), tau = 1e-3 (30 dB cap) | 1 | always [D4][27][H] |
| Absent target | 10 log10( ||y||^2 + tau_a ||x||^2 ), tau_a = 10^-2.5 | 1 | absent examples [D4][27][H] |
| Target-silent frames | same form on frames where the target's VAD is 0 | 0.1 | present examples with at least 0.1 s of target silence [D4] |
| Own-voice frames | same form on frames where the own voice is active and the target is silent | 0.3 | when own voice is in the mix (design choice for AirPods) |
| Multi-resolution STFT | spectral convergence plus log-magnitude L1 at FFT sizes 256, 512, 1024, hop 1/4 | 1 | always [D2] |
| Over-suppression | mean of ReLU(|S|^0.3 - |S_hat|^0.3)^2 | 0.3 | stage 2 onward, ablation 4; cut target over-suppression from 14.5% to 6.8% of frames in [D2][H] |
| Heads | binary cross-entropy for p_t and p_o against energy VAD labels of the clean sources | 0.1 each | always |
| Speaker ID | additive-angular-margin softmax on g_t | 0.1 | always |
| Distillation (LL only) | L1 to Q-M's output | 0.5 | LL training |

The `tsnr_loss` in [D4] section 5.1 implements the first three terms in 15 lines and is the starting code.

### F.4 Optimizer and schedule

- AdamW, peak learning rate 5e-4 (the model has attention [D4]), betas 0.9 and 0.999, weight decay 1e-4 except norms and
  biases, 2,000 warm-up steps, cosine decay to 2.5e-5, gradient clip 5 [D4].
- Batch 4 x 4 s with 2-step gradient accumulation (effective 8 x 4 s); batch 8 x 4 s does not fit (1.3).
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` and `LOW=0.8` so an oversized batch fails fast instead of swapping (the default
  low ratio of 1.4 rejects a high ratio of 1.0) [H, measured]. Run under `caffeinate` on AC power.
- Data loader - 6 workers, spawn-safe, per-worker numpy generator, FLAC crops, CPU convolution with the RIR bank [D4]. The
  loader needs well under 1 core at this step rate [D6][H].

### F.5 Stages, steps and wall-clock (from the measured step times in 1.3, plus 5% for validation and checkpoints)

| Stage | Recipe | Segment x batch | Steps | Step time | Wall-clock |
|---|---|---|---|---|---|
| 0 - sanity | overfit 50 fixed mixtures to above 25 dB SI-SDRi; causality test (changing samples after t must not change outputs before t) | 4 s x 4 | 2k | 1.2 s | 0.7 h |
| 1 - warm start | SNR +5 dB, at most 2 interferers, no absent, no Everyone, no switches, no own voice | 4 s x 4 | 20k | 1.2 s | 6.7 h |
| 2 - main | full recipe (F.2), over-suppression term on | 4 s x 4 | 130k (baseline) to 300k (quality plan) | 1.2 s | 43 h to 100 h |
| 3 - long context and realism | 6 s segments, 30% real RIRs, rig recordings mixed in at 10%, learning rate restarted at 1e-4 | 6 s x 2 | 25k | 1.0 s | 7 h |
| **Q-M total** | | | **about 175k to 345k** | | **about 60 h to 120 h** |
| Q-S (thermal fallback) | stages 1-3 | 4 s x 4 | 120k | 0.85 s | 30 h |
| LL (one profile, after Phase 0) | stages 1-3 with distillation from Q-M | 4 s x 4 | 110k | 1.17 s | 38 h |
| Ablations (Q-S size, 20k steps each, 7 runs) | A table in F.8 | 4 s x 4 | 7 x 20k | 0.85 s | 7 x 4.7 h = 33 h |

Examples seen by Q-M - about 650k (baseline) to 1.3 M (quality plan), 720 to 1,450 h of audio. Step times were measured
under shared load and varied up to 1.6x between runs (1.3), so check the real rate in the first hour and rescale. Runs are
resumable and are split into chunks of at most 48 h.

### F.6 Checkpoints and validation

- Checkpoint every 2,000 steps (weights, EMA weights, optimizer, data-loader seed state); keep the last 5 and the best 3.
- EMA decay 0.999; export the EMA weights, or the average of the last 5 checkpoints if that validates better [D4].
- Validation every 2,000 steps on 400 fixed mixtures (rendered once, speaker-disjoint) in streaming mode - SI-SDRi, picked
  rate, absent attenuation, own-voice attenuation, TSOS. Selection metric - mean SI-SDRi minus a penalty of 0.2 dB per
  point of picked-rate loss below 95%.

### F.7 Export and parity

- `exports/<run>/weights.eqw` - header with magic, version, JSON config (sizes, band widths, window parameters, tensor table)
  and 64-byte-aligned float32 tensors in the C engine's layouts, plus a SHA-256.
- `exports/<run>/golden.npz` - a 2 s test mixture, target and own enrollments, expected conditioning tensors, expected output
  waveform and head values per hop.
- `exports/<run>/speaker_encoder.mlpackage` - WeSpeaker ResNet34 with the in-graph filterbank, mixed precision, with its own
  parity clip [D4].
- `exports/<run>/model_card.json` - data, steps, validation scores, licenses of everything used.

### F.8 Ablations (run in this order, Q-S size, 20k steps, decided on the validation set)

1. Conditioning - FiLM only vs FiLM + cross-attention (expect a gain, [D5] reported 2.15 dB with confounds).
2. Own-voice share 25% vs 50% and the own-voice frame weight 0 vs 0.3 (own-voice suppression vs target quality).
3. Over-suppression term on vs off.
4. Window - asymmetric 32/8 vs symmetric 8/8 ms, and d = 0 vs 64.
5. Deep filter vs complex mask below 4 kHz.
6. WeSpeaker vs CAM++ vs no external encoder (tokens only).
7. Target-switch augmentation on vs off (switch recovery time).

---

## G. Evaluation plan

### G.1 Metrics

SI-SDRi and SNRi; ESTOI; DNSMOS P.835 (SIG, BAK, OVRL) and personalized DNSMOS; PESQ for internal use only (licensing);
Whisper WER; "picked" rate per 2 s chunk (output closer to the target than to every interferer); failure rate (SI-SDRi below
1 dB); TSOS (frames where removed target energy exceeds 0.1 of the reference); target-absent attenuation; own-voice
attenuation on own-voice-only frames; switch recovery time; speaker similarity with an independent encoder (SpeechBrain
ECAPA, never the conditioning model) [D4][H].

### G.2 Test suites and conditions

- **Restaurant simulation** - 2,000 mixtures from held-out speakers, DEMAND SCAFE and WHAM! test noise, MIT IR Survey
  bar and restaurant RIRs; grid over SNR (-5, 0, 5, 10, 20 dB), SIR (-8, -5, 0, 5, 10 dB), 0-3 interferers, own voice on and
  off, in hand and on table, enrollment clean / noisy 10 dB / noisy 0 dB, 3 s vs 8 s.
- **Libri2Mix clean** with BUT's enrollment list, for comparison with literature.
- **Accents and style** - EdAcc and VoxPopuli accented targets in the restaurant simulation; Lombard GRID targets and
  enrollments.
- **Streaming conversations** - 50 conversations of 60 s where the target talks 30-60% of the time, with turn-taking and
  overlaps, including one target switch.
- **Real rooms** - the rig (G.4), VOiCES, DiPCo.

### G.3 Acceptance targets (v1, Q-M on the restaurant simulation unless stated)

| Metric | Pass | Goal | Basis |
|---|---|---|---|
| SI-SDRi, median | at least 7 dB | 9 dB | closest published causal mono 8/4 ms result 8.85 dB on an easier set [D5][H] |
| SI-SDRi, SIR at most 0 dB, median | at least 5 dB | 7 dB | - |
| Libri2Mix clean, SI-SDRi | at least 11 dB | 13 dB | causal BSRNN 13.05 dB with frame-level cues at 32 ms latency [D5][H] |
| Picked rate per 2 s chunk | at least 95% at SIR 0 dB, 85% at -5 dB, 70% at -8 dB | 98 / 92 / 80% | a production personalized model reached 61% at equal level [D5][H] |
| Failure rate | at most 5% | 2% | 1.7-3.4% in [D4][H] (non-causal) |
| TSOS | at most 3% of target frames | 1.5% | best published 1.20% [D2][H] |
| Target-absent attenuation | median at least 25 dB; at most 5% of clips above -10 dB | 33 dB | PVQE 33-34 dB [D2][H] |
| Own voice | median at least 25 dB down, 95th percentile at least 18 dB | 30 dB | AirPods requirement (C.3) |
| DNSMOS P.835 OVRL / SIG | at least 2.8 / 3.2 | 3.1 / 3.4 | the closest mono TSE paper scored about 2.1 [D2][H] |
| ESTOI gain | at least +0.10 | +0.15 | - |
| Whisper WER, SNR 0-5 dB | at least 30% relative reduction vs the unprocessed mic | 50% | - |
| Switch recovery | within 1 dB of steady state 0.5 s after a switch | 0.25 s | [25] |
| Streaming conversations | picked rate at least 90%; no wrong-person run longer than 2 s in more than 2% of conversations | - | - |
| Rig recordings (G.4) | SI-SDRi at least 5 dB, picked at least 90% at SIR 0 dB, own voice at least 20 dB down | 7 dB | CHiME-9 showed real rooms are much harder than simulation [D2][H] |
| Listening test (8 listeners x 20 rig clips, blind A/B) | Earshot preferred over the raw mic in at least 75% of trials | 85% | - |
| Latency, AirPods | app-side path at most 12.5 ms; measured total displayed | - | C.3 |
| Latency, wired | measured mouth to ear at most 20 ms when C.4 says it is reachable; otherwise the measured number shown | - | C.2 |
| Real time | 0 audible dropouts and at most 1 concealed hop per hour in a 2 h test with the screen locked; worker p99.9 at most 50% of the hop | - | [D1][D3] |
| Battery and heat | at most 8% per hour (AirPods, screen off, 2 h); thermal state at most `.fair` at 25 C | - | [D1] |
| Safety | output never above the ceiling in the fuzz test; fade-in on every start; mute within 50 ms after an output route change | - | E.4 |

Any pass-column failure blocks release; goal columns guide the next training run.

### G.4 Real-recording test

- **Rig** (from [D4] 6.3) - 3 to 5 small powered speakers; the iPhone on a tripod or flat on a table; "target" at mouth
  height 1 m from the phone, 2-3 "interferers" 1.5-3 m away, 1-2 speakers playing restaurant noise into the walls, an
  "own voice" speaker 0.3 m from the phone. Levels about 65-70 dBA for the target at 1 m and 65-75 dBA of noise at the phone.
- **Record every source alone, then all together.** The solo recordings are real references at the phone mic. Check that
  the sum of solos matches the joint recording to below -25 dB residual before trusting them.
- **Enroll with the real app** while noise plays.
- **Rooms** - a quiet room, a living room, a cafe at a quiet hour (with permission).
- **Live check** - 2-3 consenting friends at a cafe table, the user wearing AirPods Pro 2; the user rates each 2-minute block
  (Earshot on or off, randomized) for effort and understanding on a 1-5 scale; the phone keeps the raw mic (with consent) for
  DNSMOS and pDNSMOS.

### G.5 On-device tests

- Phase 0 probe results (D.10, [D3] section 9.8).
- 1 h worker timing histogram on the phone, screen locked, joined and not joined to the workgroup.
- 2 h AirPods session for dropouts, late hops, thermal state and battery (Power Profiler).
- Route fuzz - plug and unplug, AirPods in and out of the case, Bluetooth out of range, a call, Siri, an alarm.

---

## H. Work breakdown (6 packages that can be built in parallel)

Contracts frozen on day 1 - `ear_model.h` (model vtable), the conditioning layout (A.4), `weights.eqw` and `golden.npz`
(F.7), the dataprep manifest schema (exists in `dataprep/common.py`), and the eval report JSON schema.

| Package (directory) | Owns | Interface contract | Done when |
|---|---|---|---|
| 1. `engine/` - C core with a thin Swift wrapper; macOS test host | session and route policy, RemoteIO, resamplers, rings, worker thread and workgroup, deadline and concealment, safety chain, latency loopback, Phase 0 probe screens | `engine/include/ear_engine.h` - `ear_engine_create(cfg)`, `start(model)`, `stop`, `set_params(EarParams)` (atomic), `get_status(EarStatus)`, `measure_latency(EarLatency*)`; `engine/include/ear_model.h` - `struct EarModel { void *ctx; int hop16, synth16; int (*process_hop)(void*, const float *in, float *out, float aux[2]); void (*reset)(void*); }` | Phase 0 report filled; 2 h on the phone with a stub model (pass-through and a deliberately slow one) shows 0 dropouts and correct concealment |
| 2. `runtime/` - C with Accelerate | streaming Q-M, Q-S and LL inference, token encoder, FiLM generator, weight loading and checks | `runtime/include/qnet.h` - `qnet_load(path)`, `qnet_stream_new`, `qnet_encode_person(pcm, n, QnetPerson*)`, `qnet_make_conditioning(target, own, mode, slot)`, and an `EarModel` adapter; reads `weights.eqw` | golden vectors match (D.4) for random and trained weights; Mac timing within 20% of Appendix A.1 |
| 3. `training/` - Python, PyTorch 2.7.1 | model, streaming forward, mixing loader, losses, training loop, export, WeSpeaker Core ML conversion | reads `data/manifests/*.jsonl`; writes `exports/<run>/` (F.7) | stage 0 passes; exports load in runtime; validation curve logged |
| 4. `dataprep/` - Python (in progress) | downloads, FLAC, manifests, RIR bank, fixed-seed eval renders | manifest schema in `dataprep/common.py`; `data/eval/<suite>/` with a JSON index | pack complete with licenses recorded per file |
| 5. `eval/` - Python | metrics, suite runner, rig capture and alignment tools, listening-test kit, acceptance checker | `eval/run.py --export exports/<run> --suite <name> --backend torch or c` writes `reports/<run>/<suite>.json`; `eval/accept.py` prints pass or fail per row of G.3 | runs all suites on a random-weight export and on the first trained one |
| 6. `app/` - Swift and SwiftUI | onboarding, enrollment capture and checks, people store, Core ML speaker encoder, identity monitor, head tracking, remote commands, settings, latency display | uses `engine` and `runtime` only through the Swift wrapper `EarshotCore`; person store = one folder per person with `person.json` and `enroll.flac` | the flows in E work end to end on the phone with a stub model, then with Q-M |

Order - week 1: contracts, Phase 0 probe (1), training stage 0 (3), dataprep continues (4). Weeks 2-3: engine with stub model
(1), runtime parity (2), stage 1-2 training (3), eval suites (5), app flows with stub (6). Then trained Q-M on the phone,
rig recordings, fine-tune (stage 3), acceptance, and the LL model after Phase 0.

---

## I. Top risks and mitigations

| # | Risk | Evidence | Mitigation |
|---|---|---|---|
| 1 | Wired under 20 ms is impossible on this phone | crowd median X about 14.3 ms makes LL-2 about 20.6 ms (C.2) [D5][M] | measure first (C.4); LL-1 profile; 32- or 16-frame buffers; minimum-phase resampling (saves 3.4 ms vs linear phase); report the measured number, never a promise |
| 2 | The Mac cannot train the model well enough in the time | Q-M 1.0-1.9 s per 4 x 4 s step, batch capped by MPS memory (1.3) [H] | model sized to the budget; 4-6 day quality plan with resumable runs; ablations at Q-S size; check the rate in the first hour |
| 3 | Target confusion with similar or quieter voices | 61% picked at equal level, 6.25% when quieter, for a production model [D5][H] | frame-level tokens, SIR down to -8 dB in training, switch augmentation, identity monitor with re-lock, camera-assisted enrollment (v1.5), context booster (v2) |
| 4 | Own voice leaks back 50-200 ms late on AirPods | 200 ms is the most disruptive delay [6][M] | negative enrollment, 50% own-voice training with extra weight, own-voice head and 12 dB duck, mix-back disabled while talking, 25 dB acceptance |
| 5 | Simulation does not transfer to real rooms | real cafeteria gains were small in CHiME-9 (55.3 to 60.3% vs 86.9% oracle) [D2][H] | real RIRs, the phone mic chain, rig recordings in stage 3, real-room acceptance rows and listening tests |
| 6 | The worker runs on an efficiency core or misses deadlines on the phone | Q-M took 1.9 ms per hop on an M4 E-core (1.1) [H] | Phase 0 timing on the phone; W = 1 slack on AirPods; concealment; Q-S fallback; fused gate code; BNNS Graph FP16 benchmark |
| 7 | Battery and heat over hours | no phone measurement yet; estimate 4.5-9% per hour [L] | Power Profiler in Phase 2; thermal downshift to Q-S; acceptance row in G.3 |
| 8 | Noisy enrollment in loud rooms | WeSpeaker EER 8.4% at 0 dB SNR with 3 s [D4][H] | 6 s of voiced speech, numeric checks with guidance, own-voice removal from the clip, refresh during use, camera gating (v1.5) |
| 9 | Untested iOS behaviors | motion updates in `.playAndRecord` + A2DP and when locked, remote commands with EarPods, measurement-mode mic selection, the Sound Recognition buffer bug on iOS 26.1 [D3] | Phase 0 probe before any UX depends on them; every feature has a fallback (list taps instead of look-to-switch) |
| 10 | Bluetooth latency unknown and drifting | 43 to 220 ms planning range, 60 ms drift in one report [D5][14][M] | measure at connect, re-check at 10 and 30 min, show the number, never claim under 20 ms |
| 11 | Licenses and patents | VoxCeleb weights are research-only [D5][H]; UW's commercialization (Hearvana) and NTT patents not searched [D2][M]; UIUC EHF patent [10][L] | personal build only for WeSpeaker; null-trained model for distribution; patent search before any release; credit Look Once to Hear |
| 12 | C engine and PyTorch disagree | hand-written numerics | golden vectors at export and at app launch, NumPy reference, fuzz tests |

---

## Appendix A - measurements made for this design

All on the M4 MacBook Pro, 2026-11-07 between 02:45 and 03:20 local, project venv (Python 3.12.15, torch 2.7.1, Apple clang 17).
Load average 5 to 10 from other jobs (dataset fetchers, another designer's MPS benchmark). The scripts ran from a temporary
folder; their full text is below so the numbers can be reproduced.

### A.1 Per-hop compute skeleton (C, Accelerate, one thread)

Build `clang -O3 -DACCELERATE_NEW_LAPACK -framework Accelerate qskel.c -o qskel`; run `VECLIB_MAXIMUM_THREADS=1 ./qskel` (P-core)
and `VECLIB_MAXIMUM_THREADS=1 taskpolicy -c background ./qskel` (E-cores). Results are in section 1.1. The skeleton has one
attention module and one FiLM (the design has two; the second adds 1.4% of the MACs).

```c
// qskel.c - timing skeleton for Earshot-Q (random weights, real shapes). Prints JSON lines.
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static size_t g_w = 0;
static float *buf(size_t n, float s) { float *p; if (posix_memalign((void **)&p, 64, (n ? n : 1) * 4)) exit(1);
    for (size_t i = 0; i < n; i++) p[i] = s * ((float)rand() / RAND_MAX - 0.5f); return p; }
static float *W(size_t n, int fan) { g_w += n; return buf(n, 2.0f / sqrtf((float)fan)); }
static float *Z(size_t n) { return buf(n, 0.0f); }
static void lin(const float *X, const float *Wt, const float *b, float *Y, int M, int N, int K) {  // Y = X Wt (+b)
    if (b) for (int m = 0; m < M; m++) memcpy(Y + (size_t)m * N, b, (size_t)N * 4);
    float beta = b ? 1.0f : 0.0f;
    if (M == 1) { cblas_sgemv(CblasRowMajor, CblasTrans, K, N, 1.0f, Wt, N, X, 1, beta, Y, 1); return; }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, X, K, Wt, N, beta, Y, N); }
static void ln_rows(float *X, int R, int C, const float *g, const float *b) {
    for (int r = 0; r < R; r++) { float *x = X + (size_t)r * C, mean, msq;
        vDSP_meanv(x, 1, &mean, C); vDSP_measqv(x, 1, &msq, C);
        float inv = 1.0f / sqrtf(fmaxf(msq - mean * mean, 0) + 1e-5f), nm = -mean * inv;
        vDSP_vsmsa(x, 1, &inv, &nm, x, 1, C); vDSP_vma(x, 1, g, 1, b, 1, x, 1, C); } }
static void lstm_gates(int R, int H, float *g, float *h, float *c) {  // g = R x [i f c o] preactivations
    int n = 4 * H * R; float one = 1.0f, two = 2.0f, m1 = -1.0f;
    for (int r = 0; r < R; r++) vDSP_vsmul(g + (size_t)r * 4 * H + 2 * H, 1, &two, g + (size_t)r * 4 * H + 2 * H, 1, H);
    vDSP_vneg(g, 1, g, 1, n); vvexpf(g, g, &n); vDSP_vsadd(g, 1, &one, g, 1, n); vvrecf(g, g, &n);
    for (int r = 0; r < R; r++) { float *gi = g + (size_t)r * 4 * H, *cr = c + (size_t)r * H;
        for (int k = 0; k < H; k++) cr[k] = gi[H + k] * cr[k] + gi[k] * (2.0f * gi[2 * H + k] - 1.0f); }
    int m = R * H;
    vDSP_vsmul(c, 1, &two, h, 1, m); vDSP_vneg(h, 1, h, 1, m); vvexpf(h, h, &m); vDSP_vsadd(h, 1, &one, h, 1, m);
    vvrecf(h, h, &m); vDSP_vsmsa(h, 1, &two, &m1, h, 1, m);
    for (int r = 0; r < R; r++) vDSP_vmul(h + (size_t)r * H, 1, g + (size_t)r * 4 * H + 3 * H, 1, h + (size_t)r * H, 1, H); }
typedef struct { int nfft, hop, bins, K, D, Hb, Ht, B, ntok, datt, lowbins, log2n, wid[64], st[64];
    FFTSetup fs; float *awin, *swin, *hist, *ola, *frame, *re, *im, *xr, *xi, *pr[2], *pi[2], *mag, *feat;
    float **ew, **eb, *eln_g, *eln_b, **bln_g, **bln_b, **fwi, **fwh, **fb, **bwi, **bwh, **bb, **bow, **bob;
    float **tln_g, **tln_b, **twi, **twh, **tb, **tow, **tob, **th, **tc, *film_g, *film_b, *qw, *qb, *kt, *vt, *ow, *ob;
    float **d1w, **d1b, **d2w, **d2b, *actw, *z, *zn, *gxf, *gxb, *hf, *cf, *hb, *cb, *ybi, *y, *gt, *q, *sc, *att, *hid, *coef; } Q;
static Q *q_new(int nfft, int hop, int lowbins, const int *wid, int K, int D, int Hb, int Ht, int B) {
    Q *m = calloc(1, sizeof(Q)); g_w = 0;
    m->nfft = nfft; m->hop = hop; m->bins = nfft / 2 + 1; m->K = K; m->D = D; m->Hb = Hb; m->Ht = Ht; m->B = B;
    m->ntok = 72; m->datt = 64; m->lowbins = lowbins; m->log2n = (int)lrint(log2(nfft));
    m->fs = vDSP_create_fftsetup(m->log2n, kFFTRadix2);
    int s = 0; for (int k = 0; k < K; k++) { m->wid[k] = wid[k]; m->st[k] = s; s += wid[k]; }
    m->awin = buf(nfft, 1); m->swin = buf(nfft, 1); m->hist = Z(nfft); m->ola = Z(2 * nfft); m->frame = Z(nfft);
    m->re = Z(nfft / 2); m->im = Z(nfft / 2); m->xr = Z(m->bins); m->xi = Z(m->bins); m->mag = Z(m->bins); m->feat = Z(192);
    for (int i = 0; i < 2; i++) { m->pr[i] = Z(m->bins); m->pi[i] = Z(m->bins); }
    m->ew = calloc(K, 8); m->eb = calloc(K, 8); m->d1w = calloc(K, 8); m->d1b = calloc(K, 8); m->d2w = calloc(K, 8); m->d2b = calloc(K, 8);
    for (int k = 0; k < K; k++) { int w = wid[k], taps = m->st[k] < lowbins ? 3 : 1;
        m->ew[k] = W((size_t)3 * w * D, 3 * w); m->eb[k] = W(D, 1); m->d1w[k] = W((size_t)D * 2 * D, D); m->d1b[k] = W(2 * D, 1);
        m->d2w[k] = W((size_t)2 * D * 2 * w * taps, 2 * D); m->d2b[k] = W((size_t)2 * w * taps, 1); }
    m->eln_g = buf(D, 1); m->eln_b = buf(D, 1);
#define A(x) m->x = calloc(B, 8)
    A(bln_g); A(bln_b); A(fwi); A(fwh); A(fb); A(bwi); A(bwh); A(bb); A(bow); A(bob); A(tln_g); A(tln_b); A(twi); A(twh); A(tb); A(tow); A(tob); A(th); A(tc);
    for (int l = 0; l < B; l++) { m->bln_g[l] = buf(D, 1); m->bln_b[l] = buf(D, 1);
        m->fwi[l] = W((size_t)D * 4 * Hb, D); m->fwh[l] = W((size_t)Hb * 4 * Hb, Hb); m->fb[l] = W(4 * Hb, 1);
        m->bwi[l] = W((size_t)D * 4 * Hb, D); m->bwh[l] = W((size_t)Hb * 4 * Hb, Hb); m->bb[l] = W(4 * Hb, 1);
        m->bow[l] = W((size_t)2 * Hb * D, 2 * Hb); m->bob[l] = W(D, 1); m->tln_g[l] = buf(D, 1); m->tln_b[l] = buf(D, 1);
        m->twi[l] = W((size_t)D * 4 * Ht, D); m->twh[l] = W((size_t)Ht * 4 * Ht, Ht); m->tb[l] = W(4 * Ht, 1);
        m->tow[l] = W((size_t)Ht * D, Ht); m->tob[l] = W(D, 1); m->th[l] = Z((size_t)K * Ht); m->tc[l] = Z((size_t)K * Ht); }
    m->film_g = buf((size_t)K * D, 1); m->film_b = buf((size_t)K * D, 1);
    m->qw = W((size_t)K * D * m->datt, K * D); m->qb = W(m->datt, 1); m->kt = buf((size_t)m->ntok * m->datt, 1);
    m->vt = buf((size_t)m->ntok * m->datt, 1); m->ow = W((size_t)m->datt * K * D, m->datt); m->ob = W((size_t)K * D, 1); m->actw = W(D, D);
    int H4 = 4 * (Hb > Ht ? Hb : Ht);
    m->z = Z((size_t)K * D); m->zn = Z((size_t)K * D); m->gxf = Z((size_t)K * 4 * Hb); m->gxb = Z((size_t)K * 4 * Hb);
    m->hf = Z(Hb); m->cf = Z(Hb); m->hb = Z(Hb); m->cb = Z(Hb); m->ybi = Z((size_t)K * 2 * Hb); m->y = Z((size_t)K * D);
    m->gt = Z((size_t)K * H4); m->q = Z(m->datt); m->sc = Z(m->ntok * 4); m->att = Z(m->datt); m->hid = Z(2 * D); m->coef = Z(204);
    return m; }
static void q_hop(Q *m, const float *in, float *out) {
    int n = m->nfft, h = n / 2, F = m->bins, K = m->K, D = m->D, Hb = m->Hb, Ht = m->Ht;
    memmove(m->hist, m->hist + m->hop, (size_t)(n - m->hop) * 4); memcpy(m->hist + n - m->hop, in, (size_t)m->hop * 4);
    vDSP_vmul(m->hist, 1, m->awin, 1, m->frame, 1, n);
    DSPSplitComplex sc = {m->re, m->im};
    vDSP_ctoz((DSPComplex *)m->frame, 2, &sc, 1, h); vDSP_fft_zrip(m->fs, &sc, 1, m->log2n, kFFTDirection_Forward);
    m->xr[0] = m->re[0]; m->xi[0] = 0; m->xr[h] = m->im[0]; m->xi[h] = 0;
    memcpy(m->xr + 1, m->re + 1, (size_t)(h - 1) * 4); memcpy(m->xi + 1, m->im + 1, (size_t)(h - 1) * 4);
    float e = 1e-8f; vDSP_vmma(m->xr, 1, m->xr, 1, m->xi, 1, m->xi, 1, m->mag, 1, F); vDSP_vsadd(m->mag, 1, &e, m->mag, 1, F);
    for (int k = 0; k < K; k++) { int w = m->wid[k], a = m->st[k];       // band encoder
        for (int j = 0; j < w; j++) { float p = m->mag[a + j], s = powf(p, -0.35f);
            m->feat[j] = m->xr[a + j] * s; m->feat[w + j] = m->xi[a + j] * s; m->feat[2 * w + j] = 0.1f * logf(p); }
        lin(m->feat, m->ew[k], m->eb[k], m->z + (size_t)k * D, 1, D, 3 * w); }
    ln_rows(m->z, K, D, m->eln_g, m->eln_b);
    for (int l = 0; l < m->B; l++) {
        memcpy(m->zn, m->z, (size_t)K * D * 4); ln_rows(m->zn, K, D, m->bln_g[l], m->bln_b[l]);   // band BiLSTM
        lin(m->zn, m->fwi[l], m->fb[l], m->gxf, K, 4 * Hb, D); lin(m->zn, m->bwi[l], m->bb[l], m->gxb, K, 4 * Hb, D);
        memset(m->hf, 0, Hb * 4); memset(m->cf, 0, Hb * 4); memset(m->hb, 0, Hb * 4); memset(m->cb, 0, Hb * 4);
        for (int k = 0; k < K; k++) { int kb = K - 1 - k; float *gf = m->gxf + (size_t)k * 4 * Hb, *gb = m->gxb + (size_t)kb * 4 * Hb;
            cblas_sgemv(CblasRowMajor, CblasTrans, Hb, 4 * Hb, 1.0f, m->fwh[l], 4 * Hb, m->hf, 1, 1.0f, gf, 1);
            lstm_gates(1, Hb, gf, m->hf, m->cf); memcpy(m->ybi + (size_t)k * 2 * Hb, m->hf, Hb * 4);
            cblas_sgemv(CblasRowMajor, CblasTrans, Hb, 4 * Hb, 1.0f, m->bwh[l], 4 * Hb, m->hb, 1, 1.0f, gb, 1);
            lstm_gates(1, Hb, gb, m->hb, m->cb); memcpy(m->ybi + (size_t)kb * 2 * Hb + Hb, m->hb, Hb * 4); }
        lin(m->ybi, m->bow[l], m->bob[l], m->y, K, D, 2 * Hb); vDSP_vadd(m->z, 1, m->y, 1, m->z, 1, K * D);
        memcpy(m->zn, m->z, (size_t)K * D * 4); ln_rows(m->zn, K, D, m->tln_g[l], m->tln_b[l]);   // time LSTM, 32 rows
        lin(m->zn, m->twi[l], m->tb[l], m->gt, K, 4 * Ht, D);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, 4 * Ht, Ht, 1.0f, m->th[l], Ht, m->twh[l], 4 * Ht, 1.0f, m->gt, 4 * Ht);
        lstm_gates(K, Ht, m->gt, m->th[l], m->tc[l]); lin(m->th[l], m->tow[l], m->tob[l], m->y, K, D, Ht);
        vDSP_vadd(m->z, 1, m->y, 1, m->z, 1, K * D);
        if (l == 0) {  // FiLM + 4-head cross-attention over ntok tokens
            vDSP_vma(m->z, 1, m->film_g, 1, m->film_b, 1, m->z, 1, K * D); lin(m->z, m->qw, m->qb, m->q, 1, m->datt, K * D);
            int dh = m->datt / 4, T = m->ntok;
            for (int hd = 0; hd < 4; hd++) { float *s = m->sc + hd * T, mx = -1e30f, sum = 0;
                for (int t = 0; t < T; t++) { float a; vDSP_dotpr(m->q + hd * dh, 1, m->kt + (size_t)t * m->datt + hd * dh, 1, &a, dh); s[t] = a * 0.25f; mx = fmaxf(mx, s[t]); }
                for (int t = 0; t < T; t++) { s[t] = expf(s[t] - mx); sum += s[t]; }
                memset(m->att + hd * dh, 0, dh * 4);
                for (int t = 0; t < T; t++) { float wv = s[t] / sum; vDSP_vsma(m->vt + (size_t)t * m->datt + hd * dh, 1, &wv, m->att + hd * dh, 1, m->att + hd * dh, 1, dh); } }
            lin(m->att, m->ow, m->ob, m->y, 1, K * D, m->datt); vDSP_vadd(m->z, 1, m->y, 1, m->z, 1, K * D); } }
    float act; vDSP_dotpr(m->z, 1, m->actw, 1, &act, D); (void)act;
    for (int k = 0; k < K; k++) { int w = m->wid[k], a = m->st[k], taps = a < m->lowbins ? 3 : 1;   // decoder + deep filter
        lin(m->z + (size_t)k * D, m->d1w[k], m->d1b[k], m->hid, 1, 2 * D, D); int nh = 2 * D; vvtanhf(m->hid, m->hid, &nh);
        lin(m->hid, m->d2w[k], m->d2b[k], m->coef, 1, 2 * w * taps, 2 * D);
        for (int j = 0; j < w; j++) { int b = a + j; float yr = 0, yi = 0; const float *cr = m->coef + j * taps, *ci = m->coef + (w + j) * taps;
            float xr[3] = {m->xr[b], m->pr[0][b], m->pr[1][b]}, xi[3] = {m->xi[b], m->pi[0][b], m->pi[1][b]};
            for (int t = 0; t < taps; t++) { yr += cr[t] * xr[t] - ci[t] * xi[t]; yi += cr[t] * xi[t] + ci[t] * xr[t]; }
            if (b < h) { m->re[b] = yr; m->im[b] = yi; } else m->im[0] = yr; } }
    float *t0r = m->pr[1], *t0i = m->pi[1]; m->pr[1] = m->pr[0]; m->pi[1] = m->pi[0]; m->pr[0] = t0r; m->pi[0] = t0i;
    memcpy(m->pr[0], m->xr, F * 4); memcpy(m->pi[0], m->xi, F * 4);
    vDSP_fft_zrip(m->fs, &sc, 1, m->log2n, kFFTDirection_Inverse); vDSP_ztoc(&sc, 1, (DSPComplex *)m->frame, 2, h);
    vDSP_vmul(m->frame, 1, m->swin, 1, m->frame, 1, n); vDSP_vadd(m->ola, 1, m->frame, 1, m->ola, 1, n);
    memcpy(out, m->ola, (size_t)m->hop * 4); memmove(m->ola, m->ola + m->hop, (size_t)(n - m->hop) * 4); memset(m->ola + n - m->hop, 0, (size_t)m->hop * 4); }
static int cmp(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static void bench(const char *name, Q *m, double macs, double hop_ms) {
    float *in = buf(m->hop, 0.2f), *out = Z(m->hop); for (int i = 0; i < 500; i++) q_hop(m, in, out);
    int N = 20000; uint64_t *ts = malloc(N * 8);
    for (int i = 0; i < N; i++) { uint64_t t = now_ns(); q_hop(m, in, out); ts[i] = now_ns() - t; }
    qsort(ts, N, 8, cmp); double med = ts[N / 2] * 1e-3;
    printf("{\"model\": \"%s\", \"weights_MB\": %.2f, \"MMAC_per_hop\": %.2f, \"median_us\": %.1f, \"p99_us\": %.1f, \"p999_us\": %.1f, \"max_us\": %.1f, \"rtf\": %.4f}\n",
           name, g_w * 4.0 / 1e6, macs / 1e6, med, ts[(int)(N * 0.99)] * 1e-3, ts[(int)(N * 0.999)] * 1e-3, ts[N - 1] * 1e-3, med / (hop_ms * 1e3)); }
int main(void) {
    srand(1); int w32[32], w32l[32];
    for (int k = 0; k < 32; k++) { w32[k] = k < 16 ? 4 : (k < 24 ? 8 : 16); w32l[k] = k < 16 ? 2 : (k < 24 ? 4 : 8); }
    w32[31] += 1; w32l[31] += 1;
    struct { const char *n; int nfft, hop, low, D, Hb, Ht, B; double macs; } c[] = {
        {"Q-S", 512, 64, 128, 48, 32, 64, 4, 7.79e6}, {"Q-M", 512, 64, 128, 64, 48, 96, 5, 19.68e6},
        {"Q-L", 512, 64, 128, 96, 64, 128, 6, 44.15e6}, {"LL-S", 256, 32, 64, 48, 32, 64, 4, 7.71e6},
        {"LL-M", 256, 32, 64, 64, 48, 96, 5, 19.58e6} };
    for (int i = 0; i < 5; i++) bench(c[i].n, q_new(c[i].nfft, c[i].hop, c[i].low, c[i].nfft == 512 ? w32 : w32l, 32,
                                                   c[i].D, c[i].Hb, c[i].Ht, c[i].B), c[i].macs, c[i].hop / 16.0);
    return 0; }
```

### A.2 Training-step benchmark (PyTorch on MPS)

Run with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 .venv/bin/python -I -W ignore`. Each
configuration waited for GPU utilization under 20% (read from `ioreg`), then timed 5 steps after 2 warm-up steps. Results are
in section 1.3. The per-layer numbers in A.8 (BiLSTM across bands 133 ms, time LSTM 121 ms, band transformer layer 203 ms,
5-layer dilated TCN 311 ms, fwd + bwd at batch 8 x 4 s, D = 64) came from a separate micro-benchmark of single layers.

```python
import math, time, torch, torch.nn as nn, torch.nn.functional as F
dev, SR = torch.device("mps"), 16000
def widths(nfft):
    w = ([4] * 16 + [8] * 8 + [16] * 8) if nfft == 512 else ([2] * 16 + [4] * 8 + [8] * 8); w[-1] += 1; return w
class STFT(nn.Module):  # timing stand-in: sqrt-Hann; the asymmetric pair has the same cost
    def __init__(s, n_fft=512, hop=64):
        super().__init__(); s.n_fft, s.hop, s.pad = n_fft, hop, n_fft - hop
        win = torch.hann_window(n_fft, periodic=True).sqrt(); s.register_buffer("win", win, persistent=False)
        s.norm = float((win * win).sum() / hop)
    def forward(s, x): return torch.fft.rfft(F.pad(x, (s.pad, 0)).unfold(-1, s.n_fft, s.hop) * s.win, dim=-1)
    def inverse(s, X, n):
        fr = torch.fft.irfft(X, n=s.n_fft, dim=-1) * s.win; b, T, _ = fr.shape; k = s.n_fft // s.hop
        blocks = fr.reshape(b, T, k, s.hop); out = fr.new_zeros(b, T + k - 1, s.hop)
        for i in range(k): out[:, i:i + T] += blocks[:, :, i]
        return (out.reshape(b, -1) / s.norm)[:, s.pad:s.pad + n]
class BandSplit(nn.Module):
    def __init__(s, D, w):
        super().__init__(); s.K, s.W = len(w), max(w); bins = sum(w)
        idx = torch.full((s.K, s.W), bins, dtype=torch.long); st = 0
        for k, wi in enumerate(w): idx[k, :wi] = torch.arange(st, st + wi); st += wi
        s.register_buffer("idx", idx.reshape(-1), persistent=False)
        s.w = nn.Parameter(torch.randn(s.K, 3 * s.W, D) / math.sqrt(3 * s.W)); s.b = nn.Parameter(torch.zeros(s.K, D)); s.norm = nn.LayerNorm(D)
    def forward(s, X):
        mag2 = X.real ** 2 + X.imag ** 2 + 1e-8; sc = mag2 ** (-0.35)
        f = torch.stack([X.real * sc, X.imag * sc, 0.1 * torch.log(mag2)], -1); b, T = f.shape[:2]
        f = torch.cat([f, f.new_zeros(b, T, 1, 3)], 2)[:, :, s.idx].reshape(b, T, s.K, s.W * 3)
        return s.norm(torch.einsum("btkw,kwd->btkd", f, s.w) + s.b)
class Block(nn.Module):  # BiLSTM across bands + LSTM across time (BiLSTM in the offline token encoder)
    def __init__(s, D, Hb, Ht, causal=True):
        super().__init__(); s.bn, s.brnn, s.bfc = nn.LayerNorm(D), nn.LSTM(D, Hb, batch_first=True, bidirectional=True), nn.Linear(2 * Hb, D)
        s.tn = nn.LayerNorm(D); s.trnn = nn.LSTM(D, Ht, batch_first=True, bidirectional=not causal); s.tfc = nn.Linear(Ht * (1 if causal else 2), D)
    def forward(s, z):
        b, T, K, D = z.shape; y, _ = s.brnn(s.bn(z).reshape(b * T, K, D)); z = z + s.bfc(y).reshape(b, T, K, D)
        y, _ = s.trnn(s.tn(z).permute(0, 2, 1, 3).reshape(b * K, T, D)); return z + s.tfc(y).reshape(b, K, T, D).permute(0, 2, 1, 3)
class EnrollEnc(nn.Module):
    def __init__(s, D=48, Hb=32, Ht=48, datt=64, pool=10, emb=256):
        super().__init__(); w = widths(512); s.stft = STFT(512, 128); s.split = BandSplit(D, w); s.K = len(w)
        s.block = Block(D, Hb, Ht, causal=False); s.pool = pool
        s.k = nn.Linear(s.K * D, datt); s.v = nn.Linear(s.K * D, datt); s.g = nn.Linear(s.K * D, emb)
    def forward(s, enr):
        z = s.block(s.split(s.stft(enr))); b, T, K, D = z.shape; T2 = T // s.pool
        tok = z[:, :T2 * s.pool].reshape(b, T2, s.pool, K * D).mean(2); return s.k(tok), s.v(tok), s.g(tok.mean(1))
class QNet(nn.Module):
    def __init__(s, D, Hb, Ht, B, nfft=512, hop=64, datt=64, heads=4, emb=256):
        super().__init__(); s.w = widths(nfft); s.K = len(s.w); s.W = max(s.w); s.D = D
        s.stft = STFT(nfft, hop); s.split = BandSplit(D, s.w); s.blocks = nn.ModuleList([Block(D, Hb, Ht) for _ in range(B)])
        s.film = nn.Sequential(nn.Linear(4 * emb + 1, 256), nn.PReLU(), nn.Linear(256, 2 * 2 * s.K * D))
        s.q = nn.ModuleList([nn.Linear(s.K * D, datt) for _ in range(2)]); s.o = nn.ModuleList([nn.Linear(datt, s.K * D) for _ in range(2)])
        s.heads = heads
        s.m1 = nn.Parameter(torch.randn(s.K, D, 2 * D) / math.sqrt(D)); s.m1b = nn.Parameter(torch.zeros(s.K, 2 * D))
        s.m2 = nn.Parameter(torch.randn(s.K, 2 * D, 2 * s.W * 3) / math.sqrt(2 * D) * 0.1); s.m2b = nn.Parameter(torch.zeros(s.K, 2 * s.W * 3))
        s.act = nn.Linear(D, 2); valid = torch.zeros(s.K, s.W, dtype=torch.bool)
        for k, wi in enumerate(s.w): valid[k, :wi] = True
        s.register_buffer("valid", valid.reshape(-1).nonzero().squeeze(1), persistent=False)
    def attend(s, j, z, kt, vt):
        b, T, K, D = z.shape; h = s.heads; q = s.q[j](z.reshape(b, T, K * D)).reshape(b, T, h, -1).transpose(1, 2)
        kk = kt.reshape(b, -1, h, kt.shape[-1] // h).transpose(1, 2); vv = vt.reshape(b, -1, h, vt.shape[-1] // h).transpose(1, 2)
        return z + s.o[j](F.scaled_dot_product_attention(q, kk, vv).transpose(1, 2).reshape(b, T, -1)).reshape(b, T, K, D)
    def forward(s, mix, cond, kt, vt):
        X = s.stft(mix); z = s.split(X); gb = s.film(cond).reshape(-1, 1, 2, s.K, 2, s.D)
        for i, blk in enumerate(s.blocks):
            z = blk(z)
            if i in (0, 2): j = 0 if i == 0 else 1; z = s.attend(j, z * (1 + gb[:, :, j, :, 0]) + gb[:, :, j, :, 1], kt, vt)
        h = torch.tanh(torch.einsum("btkd,kde->btke", z, s.m1) + s.m1b); m = torch.einsum("btke,kew->btkw", h, s.m2) + s.m2b
        b, T = m.shape[:2]; m = 2 * torch.tanh(0.5 * m.reshape(b, T, s.K * s.W, 2, 3)[:, :, s.valid])
        coef = torch.complex(m[..., 0, :], m[..., 1, :]); Xp = F.pad(X, (0, 0, 2, 0))
        return s.stft.inverse(sum(coef[..., t] * Xp[:, 2 - t:2 - t + T] for t in range(3)), mix.shape[-1]), torch.sigmoid(s.act(z.mean(2)))
def tsnr(est, ref, tau=1e-3):
    p = ref.pow(2).sum(-1); return (-10 * torch.log10(p / ((ref - est).pow(2).sum(-1) + tau * p + 1e-8) + 1e-8)).mean()
def mrstft(est, ref):
    l = 0
    for n in (256, 512, 1024):
        w = torch.hann_window(n, device=est.device)
        E = torch.stft(est, n, n // 4, window=w, return_complex=True).abs(); R = torch.stft(ref, n, n // 4, window=w, return_complex=True).abs()
        l = l + (E - R).abs().mean() + (torch.log(E + 1e-5) - torch.log(R + 1e-5)).abs().mean()
    return l
def step_time(cfg, bsz, secs, enr_secs=4.0, steps=5, warm=2):
    net = QNet(**cfg).to(dev); enc = EnrollEnc().to(dev); ps = list(net.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(ps, lr=1e-4); mix = torch.randn(bsz, int(SR * secs), device=dev) * 0.1; ref = torch.randn_like(mix) * 0.05
    enr = torch.randn(2 * bsz, int(SR * enr_secs), device=dev) * 0.1; wes = torch.randn(bsz, 512, device=dev); ts = []
    for i in range(warm + steps):
        torch.mps.synchronize(); t0 = time.time()
        k, v, g = enc(enr); kt = torch.cat([k[:bsz], k[bsz:]], 1); vt = torch.cat([v[:bsz], v[bsz:]], 1)
        est, act = net(mix, torch.cat([wes, g[:bsz], g[bsz:], torch.ones(bsz, 1, device=dev)], -1), kt, vt)
        loss = tsnr(est, ref) + mrstft(est, ref) + F.binary_cross_entropy(act, torch.full_like(act, 0.5))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ps, 5.0); opt.step()
        torch.mps.synchronize(); ts.append(time.time() - t0)
    return sorted(ts[warm:])[len(ts[warm:]) // 2]
for name, cfg, bsz, secs in [("Q-S", dict(D=48, Hb=32, Ht=64, B=4), 4, 4.0), ("Q-M", dict(D=64, Hb=48, Ht=96, B=5), 4, 4.0),
                             ("Q-M 6 s", dict(D=64, Hb=48, Ht=96, B=5), 2, 6.0), ("Q-L", dict(D=96, Hb=64, Ht=128, B=6), 2, 4.0),
                             ("LL-S", dict(D=48, Hb=32, Ht=64, B=4, nfft=256, hop=32), 4, 4.0)]:
    print(name, f"{step_time(cfg, bsz, secs):.3f} s/step"); torch.mps.empty_cache()
```

### A.3 Resampler group delay (scipy 1.18.1)

Kaiser FIR for 48 kHz with passband 7 kHz, stopband 8 kHz, 60 dB - 177 taps, 1.83 ms constant delay. Its minimum-phase version
(`scipy.signal.minimum_phase(h, method="homomorphic", half=False)`) - 0.11, 0.11, 0.12, 0.15, 0.28 ms at 0.5, 1, 2, 4, 6 kHz.
8th-order elliptic IIR (0.1 dB ripple, passband 7 kHz) - 0.09, 0.09, 0.10, 0.13, 0.26 ms, 53 dB at 8 kHz. With a 6.5 kHz
passband the FIR shrinks to 119 taps (1.23 ms linear phase, 0.12-0.16 ms minimum phase below 4 kHz).

### A.4 MAC and parameter counts

Per hop - band encoder 3 x 257 x D; per block 2 x 32 x 4 Hb (D + Hb) (band BiLSTM) + 32 x 2 Hb x D + 32 x 4 Ht (D + Ht)
(time LSTM) + 32 x Ht x D; FiLM 2 x 2 x 32 x D; attention 2 x (2 x 32 x D x 64 + 2 x 100 x 64); decoder 32 x D x 2D + sum over
bands of 2D x 2 w T; deep filter 4 x (128 x 3 + 129); FFT and inverse FFT about 12k. Q-M gives 19.68 M per hop and 4.92 GMAC/s;
Q-S 7.79 M and 1.95 GMAC/s; Q-L 44.15 M and 11.04 GMAC/s; LL-S at a 2 ms hop 7.71 M and 3.86 GMAC/s.

### A.5 Latency simulation (in-callback vs worker)

Callbacks every B frames at 48 kHz; 16 kHz samples available = floor(k B / 3) after callback k; frame h is processed when its
last sample is available; after n complete hops the outputs before index (n - 1) M are final (synthesis window 2M). Find the
smallest constant delay E (in 48 kHz frames) such that every output sample due in callback k is final by callback k (inside
the callback) or k - 1 (worker), checked over 1,200 callbacks. Results are in C.1.

```python
def extra_delay(B, M, mode):                # B frames at 48 kHz, M samples at 16 kHz
    for E in range(0, 48 * 40):
        ok = True
        for k in range(200, 1400):
            kk = k - (1 if mode == "worker" else 0)
            finals = ((kk * B) // 3 // M - 1) * M
            if (k * B - E - 1) // 3 >= finals: ok = False; break
        if ok: return E / 48.0
```

---

## References

Local documents

- [D1] `docs/BRIEF.md` - product brief and hard constraints.
- [D2] `docs/research/prior-art.md` - prior art and model approach (78 references).
- [D3] `docs/research/ios-platform.md` - iOS 26 platform facts (93 references).
- [D4] `docs/research/data-training.md` - data, speaker encoder, training recipe, evaluation (67 references, local measurements).
- [D5] `docs/research/fact-check.md` - adversarial check of 12 key claims.
- [D6] `bench/RESULTS.md` - M4 benchmarks of training, streaming CPU, Accelerate and Core ML.

Web sources (read for this design on 2026-11-07 unless noted as via a local document)

1. Apple Machine Learning Research. Deploying Transformers on the Apple Neural Engine (DistilBERT 3.47 ms at 0.454 W on iPhone 13). https://machinelearning.apple.com/research/neural-engine-transformers [H]
2. Apple WWDC25 session 276, What's new in BNNS Graph (BNNS Graph recommended for audio; FP16 "significantly faster"; no timings published). https://developer.apple.com/videos/play/wwdc2025/276/ [H]
3. Wang, Naithani, Politis, Virtanen. Deep neural network based low-latency speech separation with asymmetric analysis-synthesis window pair, 2021 (window equations 1-5 read in the PDF). https://arxiv.org/abs/2106.11794 [H]
4. Mauler, Martin. A low delay, variable resolution, perfect reconstruction spectral analysis-synthesis system for speech enhancement, EUSIPCO 2007 (via [D2] and [3]). https://ieeexplore.ieee.org/document/7098797 [M]
5. Wu, Braun. Ultra-Low Latency Speech Enhancement - A Comprehensive Study, ICASSP 2025 (via [D5]). https://arxiv.org/abs/2409.10358 [H]
6. Stuart, Kalinowski, Rastatter, Lynch. Effect of delayed auditory feedback on normal speakers at two speech rates, JASA 111, 2237 (2002), abstract. https://doi.org/10.1121/1.1466868 [M]
7. Veluri, Itani, Chen, Yoshioka, Gollakota. Look Once to Hear, CHI 2024 (via [D2]); license https://github.com/vb000/LookOnceToHear/blob/main/LICENSE. https://arxiv.org/abs/2405.06289 [H]
8. Apple. VNFaceLandmarks2D.innerLips (iOS 11.0+), documentation JSON. https://developer.apple.com/documentation/vision/vnfacelandmarks2d/innerlips [H]
9. Monson et al. Differential benefits of unmasking extended high-frequency content of target or background speech, JASA 2023, abstract. https://doi.org/10.1121/10.0020175 [M]
10. US patent 12444429, Speech identification and extraction from noise using extended high frequency information (search result only). https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12444429 [L]
11. Srinivas, Itani, Chen, Eskimez, Yoshioka, Gollakota. Knowledge boosting during low-latency inference, Interspeech 2024 (HTML read - merge modules with FiLM and cross-attention, delays 0/8/24/48 ms, Table 1 and 4). https://arxiv.org/html/2407.11055v3 [H]
12. Apple. setPreferredIOBufferDuration(_:) documentation JSON ("requests a change"; minimum "at least 0.005 seconds (256 frames) but might be lower"). https://developer.apple.com/documentation/avfaudio/avaudiosession/setpreferrediobufferduration(_:) [H]
13. Stephen Coyle. AirPods Pro 2 latency (126 ms Pro 2, 167 ms Pro 1, 83 ms built-in speaker, tap-to-sound on iPad). https://stephencoyle.net/airpods-pro-2 [M]
14. Apple Developer Forums 679274, AirPods actual latency vs reported (via [D3]). https://developer.apple.com/forums/thread/679274 [M]
15. RTINGS, AirPods Pro 2 vs AirPods 4 and AirPods Pro vs Pro 2 isolation (via [D3]). https://www.rtings.com/headphones/tools/compare/apple-airpods-pro-2nd-generation-apple-airpods-4/29490/74581 [M]
16. NTIA ITS report on audio-video synchronization citing ITU-R BT.1359 thresholds (sign conventions differ between sources). https://its.ntia.gov/publications/download/11-474.pdf [M]
17. Apple Developer Forums 769245, buffer sizes with Sound Recognition or Vocal Shortcuts (via [D3]). https://developer.apple.com/forums/thread/769245 [M]
18. Apple. AVAudioSession.Mode.measurement (via [D3][D5]). https://developer.apple.com/documentation/avfaudio/avaudiosession/mode-swift.struct/measurement [H]
19. Apple TN2169, High Precision Timers in iOS / OS X (via [D3]). https://developer.apple.com/library/archive/technotes/tn2169/_index.html [H]
20. Apple. Adding parallel real-time threads to audio workgroups; kAudioOutputUnitProperty_OSWorkgroup (via [D3]). https://developer.apple.com/documentation/audiotoolbox/adding-parallel-real-time-threads-to-audio-workgroups [H]
21. Apple. BLASSetThreading(_:) documentation JSON (iOS 18.0+, "This setting is per thread"). https://developer.apple.com/documentation/accelerate/blassetthreading(_:) [H]
22. Apple. ProcessInfo.ThermalState (via [D3]). https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.enum [H]
23. Apple WWDC25 session 226, Profile and optimize power usage in your app (Power Profiler). https://developer.apple.com/videos/play/wwdc2025/226 [H for the tool]
24. MacRumors. iPhone 16 battery capacities revealed (Anatel filing, iPhone 16 Pro 3,582 mAh). https://macrumors.com/2024/09/19/iphone-16-battery-capacities-revealed [M]
25. Tap2Listen - instant target switching for streaming audio-visual speaker extraction, 2026 (baseline needs about 1 s to recover after a switch). https://arxiv.org/html/2610.36643v1 [H]
26. ITU-T H.870 (08/2018), Guidelines for safe listening devices/systems (Mode 1 - 1.6 Pa^2 h per 7 days, from 80 dBA for 40 h). https://efhoh.org/wp-content/uploads/2025/05/H.870-2018.pdf ; current edition summary https://www.itu.int/dms_pubrec/itu-t/rec/h/T-REC-H.870-202203-I!!SUM-HTM-E.htm [H]
27. Wisdom et al. Unsupervised Sound Separation Using Mixture Invariant Training (thresholded SNR, via [D4]). https://arxiv.org/abs/2006.12701 [H]
28. Xu, Yang, Trigoni, Markham. Target speaker extraction through comparing noisy positive and negative audio enrollments, NeurIPS 2025 (via [D5]). https://arxiv.org/abs/2502.16611 [H]
29. WeSep, causal BSRNN with different cues (via [D5]). https://arxiv.org/abs/2607.27436 [H]
30. AV-PVQE (picked rate 98.63 / 61.08 / 6.25%; video lifts SI-SNRi -2.31 to 9.03 dB) (via [D5]). https://arxiv.org/abs/2609.30631 [H]
31. CHiME-9 ECHI (via [D5]). https://arxiv.org/abs/2609.26306 [H]
32. Østergaard et al. Don't Listen to Me - lightweight, low-latency own-voice cancellation, Interspeech 2026 (enrollment-conditioned own-voice removal at 2 ms algorithmic latency; cites Stone and Moore for 15-20 ms own-voice delay limits). https://arxiv.org/html/2606.23332v1 [H]
33. Apple Support. Headphone notifications and headphone audio levels (via [D3]). https://support.apple.com/en-us/108968 [M]
34. Goehring et al. 2018, delay tolerance with hearing loss (summarizes Stone and Moore limits of about 14-30 ms). https://eprints.soton.ac.uk/414900/ [M]
35. Delcroix et al. "Listen only to me!" - target-absent training and verification (via [D4]). https://arxiv.org/abs/2204.04811 [H]
36. Eskimez et al. Personalized speech enhancement - new models and comprehensive evaluation (TSOS) (via [D2]). https://arxiv.org/abs/2110.09625 [H]
37. Pärnamaa, Saabas. Personalized speech enhancement without a separate speaker embedding model (via [D2]). https://arxiv.org/abs/2406.09928 [H]
38. Superpowered latency data table (via [D5]). https://superpowered.com/latencydata/data.html [M]
39. Apple Developer Forums 774784 - recording sessions cannot activate from the background (Apple DTS, via [D5]). https://developer.apple.com/forums/thread/774784 [H]
40. Apple Developer Forums 761549 - MPRemoteCommandCenter handles A2DP commands, no API for HFP commands (Apple DTS, via [D5]). https://developer.apple.com/forums/thread/761549 [H]
41. Apple. CMHeadphoneMotionManager documentation JSON (iOS 14.0+, NSMotionUsageDescription required or the app crashes). https://developer.apple.com/documentation/coremotion/cmheadphonemotionmanager [H]
42. WeSpeaker pretrained model licenses and the VoxCeleb license page (via [D5]). https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md ; https://mm.kaist.ac.kr/datasets/voxceleb/ [H]
43. Hu, Itani, Chen, Gollakota. Proactive hearing assistants that isolate egocentric conversations, EMNLP 2025 (via [D2]). https://arxiv.org/abs/2511.11473 [H]
44. Schröter et al. Low latency speech enhancement for hearing aids using deep filtering, TASLP 2022 (via [D2]). https://doi.org/10.1109/TASLP.2022.3198548 [H]
45. Apple Newsroom, iPhone 16 Pro (A18 Pro with 2 performance and 4 efficiency cores, 16-core Neural Engine, four mics) (via [D3]). https://www.apple.com/newsroom/2024/09/apple-debuts-iphone-16-pro-and-iphone-16-pro-max/ [H]
