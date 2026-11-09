# Earshot v1 - binding contract (00-contract)

Date - 2026-11-07. Author - chief architect. Status - **binding**. The ML, engine and iOS section writers
(10-ml.md, 20-engine.md, 30-ios.md) expand this file and must not change any number, name, shape or signature in it.
A change needs a dated note in docs/DECISIONS.md signed off by the lead, and a bump of the affected version field.
Pre-build amendments (made before any code or export exists) are logged in DECISIONS L without a version bump.

Inputs - docs/BRIEF.md, docs/designs/quality.md, docs/designs/product.md, docs/research/fact-check.md (overrides refuted
claims), prior-art.md, ios-platform.md, data-training.md, bench/RESULTS.md, bench/merged/bench_merged.py and
bench/merged/bench_quiet.jsonl (merged model measured on a quiet M4), data/manifests/SUMMARY.md, data/rir/README.md.

Labels used everywhere - **measured** (on this Mac or the phone, file named), **benchmark** (measured by others),
**doc** (Apple or paper text), **analytic** (arithmetic from shapes), **estimate** (not measured, may be wrong),
**target** (a goal, not a result). Nothing has run on the iPhone yet, so every phone number is an estimate.

---

## 0. Plain-English summary (2-minute read)

**What it does.** You sit in a loud restaurant wearing AirPods Pro. Earshot listens through the iPhone's own microphone,
keeps the voice of the one person you chose, pushes everyone and everything else down, and plays the result into your
AirPods. Noise cancellation in the AirPods hides most of the real room. Everything runs on the phone; no audio leaves it.

**How you pick the person.**
- Add someone - tap "Add person" and hold the phone toward them for about 6 seconds while they talk. Earshot learns a
  voiceprint from that sample. With nobody added yet, a stem double-press starts the same capture hands-free.
- Look to switch - when you add someone, Earshot remembers where your head was pointing (AirPods head tracking). Later,
  face that person and double-press the AirPods stem to switch to them. Single press pauses, triple press changes mode.
- Modes - Focus (one person), Table (up to 3 added people at once), Everyone (all voices except yours, noise removed).

**The model in 5 lines.**
1. Every 4 ms the phone takes the newest 32 ms of sound (16 kHz) and turns it into 257 frequency bins.
2. The bins are grouped into 16 bands; 4 stacked LSTM blocks track each band over time and compare bands with each other.
3. After the first block the network is told who to keep: the person's voiceprint (two vectors plus up to 48 short
   "voice tokens") and your own voiceprint as a "never keep this" example.
4. A small decoder outputs a filter for every bin (it also looks 2 frames back below 4 kHz) and the clean voice is rebuilt.
5. It has 2.58 M parameters (1.18 M used every hop) and costs 3.0 GMAC/s, about 0.2 ms of one CPU core per 4 ms (estimate).

**The latency truth.** Earshot itself adds about **11 ms** on top of a plain pass-through app (8 ms is the model's
window, the rest is resampling and a safety margin). Bluetooth adds a delay no app can remove. Planning numbers from
other people's tests are about **43 ms for AirPods Pro 2 and 84 ms for AirPods Pro 1** (estimate, docs/research/
fact-check.md section 2), so expect roughly **60 ms (Pro 2) and 100 ms (Pro 1)** from mouth to ear, with a possible range
of about 56 to 236 ms. These are estimates until Earshot measures them on this phone. The app offers a chirp
measurement at every AirPods connect and shows the result. **Earshot never claims under 20 ms over Bluetooth.** Because the voice
arrives late, ANC must be on (it hides the direct sound by about 21 to 25 dB, benchmark) and **your own voice is always
suppressed**, since hearing yourself 100 ms late is very disruptive.

**v1 includes** - AirPods Pro 1 and 2 over Bluetooth, built-in iPhone mic, Focus / Table / Everyone, phone-aim enrollment
and "add who just spoke", own-voice enrollment, look-to-switch with stem presses, in-app latency measurement, output
limiter and fades, alarm passthrough (helper only), people library on the phone, screen-locked use, recovery after calls
and route changes, diagnostics screen and per-minute logs (no audio).

**v1 excludes** - wired low-latency profile (later work, costs v1 nothing), camera lip-based enrollment, automatic
"follow my gaze", automatic voiceprint refresh, 24 kHz audio, AirPods microphones, any server, App Store or TestFlight.

**Expected quality (estimate, not a result).** On a simulated restaurant with 1 to 3 other talkers and noise: about
6 to 9 dB SI-SNR improvement, the right person picked about 90 to 95% of the time when voices are equally loud (less when
the target is quieter), own voice and absent-target output pushed down about 20 to 30 dB. Real rooms will be worse than
simulation; section 9 has the targets that decide "done".

**Credits and licenses.** The interaction (look at a person for a few seconds, then hear only them) comes from UW's
**Look Once to Hear** (Veluri et al., CHI 2024); we use none of its code (UW non-commercial license). The network follows
the band-split RNN / TF-GridNet line of work. The speaker encoder is WeSpeaker ResNet34 trained on VoxCeleb, whose
weights are granted for research purposes: **this build is for personal use only**. No non-commercial code or data is
used in training (WHAM! noise is evaluation only).

---

## 1. Design scoring

Weights - total = 0.3 quality + 0.25 latency_rt + 0.2 feasibility + 0.15 product + 0.1 simplicity (scores 1 to 10).

| Design | quality | latency_rt | feasibility | product | simplicity | total |
|---|---|---|---|---|---|---|
| quality.md (Earshot-Q) | 9 | 8 | 5 | 7 | 5 | **7.25** |
| product.md (ESB-16 daily use) | 7 | 7 | 9 | 9 | 7 | **7.70** |
| Merged (this contract) | 8 | 9 | 8 | 9 | 7 | **8.30** |

Why - quality.md has the stronger network ideas (deep filter, 3 input features, two FiLM points, frozen WeSpeaker in the
FiLM input, own-voice frame loss, minimum-phase resamplers, exact latency simulation) but its 32-band Q-M does not fit
batch 8 on the M4, needs 60 to 120 h of training, and adds two more models (Q-S, LL). product.md fits the Mac
(ESB-16, 0.98 s/step measured, docs/designs/product.md Appendix B.1) and has the best daily-use story (warm instances,
Table mode, learned head directions, people library, alarm passthrough) but uses only a complex mask and 2.6 ms of
linear-phase resampling.

What the merge takes
- From product.md - ESB-16 backbone (16 bands, D 64, 4 blocks, time LSTM 128, band BiLSTM 64), shared enrollment encoder
  at a 32 ms hop, up to 3 warm instances and Table mode, look-select with learned directions, stem mapping, people
  library and privacy rules, alarm passthrough, onboarding checklist, slow AGC, safety layers, Phase 0 question list.
- From quality.md - 3-tap deep filter below 4 kHz, |X|^0.3 complex plus log-power features, a second FiLM point after
  the third block, frozen WeSpeaker vectors in the FiLM input with path dropout, own-voice frame loss, minimum-phase FIR
  resamplers, worker-plus-concealment timing model, fuzz and parity discipline, latency accounting.
- The lost real-time-robustness design is covered here by - the 16 kHz ring architecture with resampling in the
  callback (works for any callback size), a fixed 2.69 ms deadline slack on AirPods that only grows (never shrinks)
  after underruns, NaN reset, malloc-count and fuzz harnesses, conditioning double buffers with acknowledge, launch-time
  parity self-test, one engine swap path for every route or buffer change that keeps RemoteIO running (5.3
  Lifecycle), and a worker thread policy sized from the granted callback (4.2).
- The merged model was prototyped and measured - merged_full 1.025 to 1.086 s/step, 4 x 8 s 1.03 s/step, MPS pool 9.6 to
  10.5 GB, 2.574 M parameters (measured, bench/merged/bench_quiet.jsonl).

---

## 2. Model contract (ESB-16-DF, model_version 1.0.0)

### 2.1 Signal and framing

| Item | Value |
|---|---|
| Sample rate | 16,000 Hz mono float32, full scale = 1.0 |
| FFT | N = 512 (32 ms), 257 bins of 31.25 Hz |
| Hop | 64 samples (4 ms); 250 hops per second |
| Synthesis window | 128 samples (8 ms) |
| Algorithmic latency | **128 samples = 8.0 ms** including the hop wait, no lookahead (analytic, verified in docs/designs/product.md B.3) |
| FFT convention | torch.fft.rfft (unnormalized) and torch.fft.irfft (norm "backward", 1/512); irfft ignores the imaginary part of bins 0 and 256 |

Windows (periodic Hann `H_L[n] = 0.5 - 0.5 cos(2 pi n / L)`, computed in float64, stored float32)
- Analysis `wa[n] = sqrt(H_896[n])` for 0 <= n < 448, `wa[n] = sqrt(H_128[n - 384])` for 448 <= n < 512.
- Synthesis `ws[n] = 0` for n < 384, `ws[n] = H_128[n - 384] / wa[n]` for 384 <= n < 512.
- Offline STFT - left-pad the signal with 448 zeros, frames of 512 at hop 64, so frame t ends at input sample (t+1)*64 - 1.
- Offline iSTFT - irfft, multiply by ws, overlap-add the last 128 samples of each frame at hop 64 (frame t covers output
  samples t*64 - 64 to t*64 + 63), giving an output aligned with the input (exactly bench/merged/bench_merged.py istft).
- Streaming - per hop: shift 64 new samples into `ana[512]`, window, FFT, network, irfft, multiply by ws,
  `ola[0:128] += frame[384:512]`, emit `ola[0:64]`, shift ola left by 64, zero `ola[64:128]`.
- Streaming identity - **y_stream[n + 64] = y_offline[n]**, max abs error <= 1e-5 on CPU fp32 (test in WP-ML).
- Unit test - an all-pass network (Y = X) reproduces the input delayed by 64 samples in streaming, max error 1e-5.

### 2.2 Architecture (exact)

K = 16 bands, widths in bins `W = [8]*8 + [16]*4 + [32]*3 + [33]` (sum 257; bands 0-11 cover bins 0-127 = 0 to 3.97 kHz).
D = 64. Blocks B = 4. LayerNorm eps 1e-5 everywhere. LSTM gate order (i, f, g, o), no peepholes, no projection,
forget-gate bias initialized to +1.

| # | Stage | Exact specification | Output |
|---|---|---|---|
| 1 | Features | per bin: `c = X * (|X| + 1e-8)^(-0.7)`; features `[Re c, Im c, 0.1 ln(|X|^2 + 1e-8)]`; per band flattened bin-major (index = 3*j + f) | 771 floats |
| 2 | Band encoder | per band k: `Linear(3 W_k -> 64)` then `LayerNorm(64)` (own weights per band) | z [16][64] |
| 3 | Block b, time path | `LN(64)`, one `LSTM(64 -> 128)` shared by the 16 bands (16 states carried across hops), `Linear(128 -> 64)`, residual add | z |
| 4 | Block b, band path | `LN(64)`, `BiLSTM(64 -> 64 per direction)` over the 16 bands as a sequence (zero state every hop), `Linear(128 -> 64)` on [fwd, bwd], residual add | z |
| 5 | FiLM point 0 (after block 0) | `z = z * (1 + gamma0) + beta0`, gamma0, beta0 [16][64] | z |
| 6 | Cross-attention (after FiLM 0) | `q = Linear(1024 -> 64)(flatten z)` (flatten index k*64 + d); 4 heads x 16 (head h = q[16h : 16h+16]); scores `q_h . K_h[n] * 0.25`, softmax over the n valid tokens; `a[16h + d] = sum_n p_n V_h[n][d]`; `z += reshape(Linear(64 -> 1024)(a))`; output Linear zero-initialized in training | z |
| 7 | FiLM point 1 (after block 2) | `z = z * (1 + gamma1) + beta1` | z |
| 8 | Heads (on block-3 output) | `sigmoid(Linear(64 -> 2)(mean over bands of z))` = [p_target, p_self] | 2 floats |
| 9 | Decoder per band k | `LN(64)`, `Linear(64 -> 128)`, `tanh`, `Linear(128 -> 2 W_k T_k)`, then `1.5 tanh`; T_k = 3 for k < 12, else 1; first W_k T_k values are real parts, the rest imaginary, each reshaped [T_k][W_k] (index tau*W_k + j) | coefficients |
| 10 | Deep filter | `Y(t, f) = sum_{tau < T_k} C_tau(t, f) X(t - tau, f)` (complex multiply); X before the first hop is 0 | Y [257] complex |
| 11 | Synthesis | section 2.1 | 64 samples |

Order inside block b - time path, then band path, then (b = 0) FiLM 0 and attention, or (b = 2) FiLM 1.

### 2.3 Conditioning

Who to keep is given by one **conditioning set** per instance, built off the audio thread by the C engine (Core ML is
used only for WeSpeaker).

| Input | Shape | Source |
|---|---|---|
| e_t, e_o | 256 each, L2-normalized | WeSpeaker ResNet34 (section 6) on the target's and the user's enrollment clips |
| g_t, g_o | 128 each, L2-normalized | token encoder (2.4) on the same clips |
| target tokens | n_t x 64, n_t <= 48 | token encoder |
| own tokens | n_o x 64, n_o <= 16 | token encoder |
| null tokens | 4 x 64 | learned (`att.null_tok`), always present |
| mode | 2, one-hot | [1, 0] = Person, [0, 1] = Everyone |

- **FiLM generator** - `film = Linear(256 -> 4096)(SiLU(Linear(770 -> 256)(u)))` with
  `u = [e_t, g_t, e_o, g_o, mode]` (770). Output index `((p*2 + s)*16 + k)*64 + d`, p = FiLM point (0, 1),
  s = 0 gamma / 1 beta.
- **Token list** - target tokens + `type_emb[0]`, own tokens + `type_emb[1]`, null tokens + `type_emb[2]`, in that order,
  n = n_t + n_o + 4 <= 68. `[K | V] = Linear(64 -> 128)(token)`: K = first 64, V = last 64, head h uses [16h, 16h+16).
- **Everyone mode** - e_t = `null.e_t`, g_t = `null.g_t`, n_t = 0, mode [0, 1]. Own voice stays a negative.
- **No own voice enrolled** - e_o = `null.e_o`, g_o = `null.g_o`, n_o = 0.
- **No WeSpeaker** (encoder self-test failed, or a future distributed build) - e_t = `null.e_t` and e_o = `null.e_o`,
  everything else unchanged. Trained by 20% path dropout, so it degrades gracefully. With `use_wespk = 1` but a given
  voiceprint that lacks its vector, both vectors also fall back to the nulls ("both null" is the only no-WeSpeaker
  state the model saw in training).
- **Table mode** - one instance per person (up to 3), each with its own conditioning set and state; outputs are summed.
- The model is never given more than one target person per instance in v1.

### 2.4 Token encoder (enrollment, non-causal, C engine and PyTorch identical)

1. Frames - left-pad the clip with 448 zeros, frames of 512 at hop **512**, window wa, rfft. Frame count
   `F = 0` for n < 576 (a deliberate minimum, so F starts at 2), else `F = floor((n - 64) / 512) + 1`. `es_vad`
   returns at once when F = 0.
2. Features and the **shared** band encoder of 2.2 (same weights) give z [F][16][64].
3. One dual-path block - time path `LN`, `BiLSTM(64 -> 32 per direction)` over the F frames, `Linear(64 -> 64)`, residual;
   band path `LN`, `BiLSTM(64 -> 32 per direction)` over bands, `Linear(64 -> 64)`, residual.
4. Speech mask m[F] (uint8) - the frames with m = 1 form one ordered list (not one list per contiguous run); group
   that list in threes from its start (a final group of 1 or 2 is dropped), mean each group over its 3 frames, giving
   n_avail tokens of 1024 (flatten k*64 + d).
5. If n_avail > max (48 target, 16 own) keep indices `floor(i * n_avail / max)` for i = 0 .. max-1.
6. `token = Linear(1024 -> 64)`; `g = L2normalize(Linear(64 -> 128)(mean of tokens))`.
7. Fail with ES_ERR_ENROLL_SHORT if n_avail < 4.
8. `es_embed_g` (the "locked" check on model output) runs steps 1-6 with the `es_vad` mask and the target limit and
   returns only g.

**Energy VAD** (`es_vad`, also `ml/earshot/vad.py`, bit-identical rule) - per token frame f, the 512 raw samples of the
padded clip `xp[512f : 512f + 512]`; `E_f = 10 log10(mean(x^2) + 1e-10)`; sort E, `P10 = E_sorted[floor(0.1 (F - 1))]`,
`thr = max(P10 + 6, max(E) - 40)`; `m_f = E_f > thr`; then dilate by 1 frame on each side.

### 2.5 Size and compute

| Quantity | Value | Label |
|---|---|---|
| Parameters, whole model | 2,575,108 (bench prototype 2,574,340 + 768 for the four null vectors) | measured with bench_merged.py on CPU, plus analytic |
| Used every hop | 1,182,852 (4.73 MB fp32) - band encoder 52,416, blocks 730,624, attention q/o 132,160, decoder 267,522, heads 130 | measured |
| Used once per conditioning change | FiLM generator 1,250,048; kv 8,320; type_emb 192; null_tok 256; null vectors 768 | measured |
| Used once per enrollment | token encoder 132,672 (plus the shared band encoder) | measured |
| Training only | AAM speaker head 128 x 3,758 = 481,024 (not exported) | analytic |
| MACs per hop | 12.00 M (blocks 11.53 M, decoder 0.26 M, attention 0.14 M, encoder 0.05 M, FFTs about 0.01 M) = **3.00 GMAC/s** | analytic |
| MACs per enrollment | token encoder 0.97 M per 32 ms frame (0.30 G for 10 s); FiLM 1.25 M; kv 0.56 M | analytic |
| C per-hop time, M4 P-core | about 0.19 ms median (ESB-16 measured 175 to 187 us, docs/designs/product.md B.2, plus about 1% MACs) | estimate |
| C per-hop time, iPhone 16 Pro | 0.2 to 0.3 ms median (A18 Pro at 1.1 to 1.5x the M4 time) | estimate |

### 2.6 Streaming state per instance (`es_stream`, float32, allocated once, zeroed by reset)

| Buffer | Shape | Floats | Meaning |
|---|---|---|---|
| ana | [512] | 512 | analysis history, newest sample last |
| xhist | [2][128][2] | 512 | X of the previous 2 hops, bins 0-127, (re, im); index 0 = t-1 |
| h_t, c_t | [4][16][128] each | 16,384 | time-LSTM hidden and cell per block per band |
| ola | [128] | 128 | overlap-add buffer |
| **Total state** | | **17,536 (70 KB)** | |

Scratch (reused, not state) - X[257] complex, feat[771], z and zn [16][64], gates [16][512], band h/c fwd and bwd
[16][64], q[64], scores[4][68], a[64], dec hidden [16][128], coef[1026], Y[257] complex, frame[512].

**Conditioning set** (`es_cond`, float32, double-buffered per instance in the engine) - film [2][2][16][64] (4,096),
K [4][68][16] (4,352), V [4][68][16] (4,352), int32 n_tokens, int32 mode: 12,800 floats (51 KB).

**Computed once per enrollment** - token encoder (tokens, g) in C, WeSpeaker e in Core ML.
**Once per conditioning change** (person switch, mode change, own voice re-enrolled) - FiLM gamma/beta for both points,
token list with type embeddings, K and V. **Per hop** - everything in 2.2. Nothing per hop allocates.

---

## 3. Weight export and test-vector formats

### 3.1 Container (shared by weights `.esw` and test vectors `.est`)

All little-endian. One file = fixed preamble + JSON header + 64-byte aligned arrays.

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic - ASCII `ESW1` (weights) or `EST1` (test vectors) |
| 4 | 4 | uint32 format_version = 1 |
| 8 | 4 | uint32 header_len - bytes of UTF-8 JSON that follow (no NUL) |
| 12 | 4 | uint32 reserved = 0 |
| 16 | header_len | JSON header |
| data_start | ... | arrays; `data_start = round_up(16 + header_len, 64)`, zero padding before it |

JSON header fields
- `format` - "earshot-weights" or "earshot-testvec"; `format_version` - 1.
- `model` - "esb16-df"; `model_version` - "1.0.0".
- `config` - exactly this object (the engine compares every field with its compiled constants)
  `{"sr":16000,"n_fft":512,"hop":64,"synth_len":128,"ana_rise":448,"band_widths":[8,8,8,8,8,8,8,8,16,16,16,16,32,32,32,33],
  "D":64,"blocks":4,"time_hidden":128,"band_hidden":64,"film_after":[0,2],"att_after":0,"att_heads":4,"att_dim":64,
  "tok_dim":64,"max_target_tokens":48,"max_own_tokens":16,"null_tokens":4,"g_dim":128,"wespk_dim":256,"df_taps":3,
  "df_bands":12,"mask_bound":1.5,"feat_pow":0.3,"feat_log_scale":0.1,"ln_eps":1e-05,"tok_hop":512,"tok_pool":3,
  "tok_hidden":32}`.
- `config_hash` - lowercase hex SHA-256 of `json.dumps(config, sort_keys=True, separators=(",", ":"))` in UTF-8.
- `arrays` - list of `{name, dtype ("f32" | "u8" | "i32"), shape [..], offset, nbytes}`; offset is relative to
  data_start and a multiple of 64; arrays appear in list order.
- `data_bytes`, `data_sha256` - size and lowercase hex SHA-256 of everything from data_start to the end of the file.
- **model_hash** = the `data_sha256` of the loaded `earshot.esw` (64 lowercase hex chars). It identifies the trained
  weights (`config_hash` identifies only the architecture, so it is the same for random_v1 and v1), does not change
  when only `meta` changes, and is the value written into `thresholds.json`, `person.json`, telemetry and every `.est`
  (`meta.weights_sha256`). The engine returns it from `es_model_data_hash`.
- `meta` - free object: training run, step, EMA flag, git commit, date, data licenses, and for weights the line
  "WeSpeaker conditioning trained with VoxCeleb research-terms weights - personal use only".

Loader rules (C engine, `engine/src/json_min.c` parser of at most about 200 lines, no external library)
- Reject on wrong magic, format_version, model_version major, any `config` field different from the compiled
  constants, a missing or extra tensor, a wrong shape or dtype, an offset out of bounds or not 64-aligned, or a
  data_sha256 mismatch (CommonCrypto CC_SHA256). Error strings name the first failing field.
- Tensors are copied into the engine's own 64-byte aligned memory, so the caller may free the input buffer after loading.

### 3.2 Weight tensors (`earshot.esw`, all f32, 2,570,500 floats = 10.28 MB of data; the folder random_v1/ or v1/ carries the version)

Layout rules - every Linear weight is stored **[in, out] row-major** (PyTorch weight transposed). LSTM weights are stored
`w_ih [in, 4H]` and `w_hh [H, 4H]` (columns in gate order i, f, g, o) and `bias [4H] = bias_ih + bias_hh`. A reverse
direction uses the suffix `_rev`. k = 0..15, b = 0..3, W_k and T_k as in 2.2. Order in the file = the table below
read row by row with the index loop outermost (for each k the 4 `enc` tensors, then for each b its 17 block tensors,
then for each k its 6 `dec` tensors); the loader matches tensors by name, so order is not load-bearing.

| Names | Shape |
|---|---|
| `enc.lin.{k}.weight`, `enc.lin.{k}.bias`, `enc.ln.{k}.weight`, `enc.ln.{k}.bias` | [3 W_k, 64], [64], [64], [64] |
| `blocks.{b}.tn.weight`, `.tn.bias` | [64], [64] |
| `blocks.{b}.t.w_ih`, `.t.w_hh`, `.t.bias` | [64, 512], [128, 512], [512] |
| `blocks.{b}.tf.weight`, `.tf.bias` | [128, 64], [64] |
| `blocks.{b}.bn.weight`, `.bn.bias` | [64], [64] |
| `blocks.{b}.b.w_ih`, `.b.w_hh`, `.b.bias`, `.b.w_ih_rev`, `.b.w_hh_rev`, `.b.bias_rev` | [64, 256], [64, 256], [256] (x2) |
| `blocks.{b}.bf.weight`, `.bf.bias` | [128, 64], [64] |
| `att.q.weight`, `att.q.bias`, `att.o.weight`, `att.o.bias` | [1024, 64], [64], [64, 1024], [1024] |
| `att.kv.weight`, `att.kv.bias`, `att.type_emb`, `att.null_tok` | [64, 128], [128], [3, 64], [4, 64] |
| `film.0.weight`, `film.0.bias`, `film.2.weight`, `film.2.bias` | [770, 256], [256], [256, 4096], [4096] |
| `null.e_t`, `null.g_t`, `null.e_o`, `null.g_o` | [256], [128], [256], [128] |
| `dec.{k}.ln.weight`, `.ln.bias`, `.fc1.weight`, `.fc1.bias`, `.fc2.weight`, `.fc2.bias` | [64], [64], [64, 128], [128], [128, 2 W_k T_k], [2 W_k T_k] |
| `heads.weight`, `heads.bias` | [64, 2], [2] |
| `tok.blk.tn.*`, `tok.blk.bn.*` (weight, bias) | [64] each |
| `tok.blk.t.w_ih`, `.t.w_hh`, `.t.bias` and `_rev` versions | [64, 128], [32, 128], [128] (x2) |
| `tok.blk.b.w_ih`, `.b.w_hh`, `.b.bias` and `_rev` versions | [64, 128], [32, 128], [128] (x2) |
| `tok.blk.tf.weight`, `.tf.bias`, `tok.blk.bf.weight`, `.bf.bias` | [64, 64], [64], [64, 64], [64] |
| `tok.proj.weight`, `tok.proj.bias`, `tok.g.weight`, `tok.g.bias` | [1024, 64], [64], [64, 128], [128] |

270 tensors in total. The PyTorch module tree in `ml/earshot/model.py` uses these names (LSTM params are renamed at
export). The training-only `aam.weight` is never exported.

**Random-init export** (`python -m ml.export --random --seed 0 --out ml/exports/random_v1/`) - PyTorch default init,
forget bias +1, then every tensor that training initializes to zero (`att.o.*`, `att.type_emb`, `att.null_tok`,
`null.*`) is overwritten with N(0, 0.1^2) so every path is exercised by parity tests.

### 3.3 Test vectors (`.est`, made by `python -m ml.export --testvec`, CPU fp32, PyTorch streaming reference)

| Array | dtype, shape | Meaning |
|---|---|---|
| `in` | f32 [N] | mixture at 16 kHz; `testvec_short.est` N = 32,000 (2 s), `testvec_long.est` N = 160,000 (10 s) |
| `enr_t`, `mask_t`, `enr_o`, `mask_o` | f32 [Lt], u8 [Ft], f32 [Lo], u8 [Fo] | enrollment clips and speech masks (Lt 96,000, Lo 128,000) |
| `vad_t` | u8 [Ft] | expected `es_vad(enr_t)` |
| `e_t`, `e_o` | f32 [256] | WeSpeaker vectors given to the engine as inputs |
| `tok_t`, `g_t`, `tok_o`, `g_o` | f32 [n_t, 64], [128], [n_o, 64], [128] | expected token encoder outputs |
| `cond.{case}.film`, `.k`, `.v`, `.n` | f32 [4096], [4, 68, 16], [4, 68, 16] (zero beyond n), i32 [1] | for case in `person`, `everyone`, `noown`, `nowespk` |
| `out.{case}` | f32 [N] | streaming output samples (hop j emits samples 64j .. 64j+63), from a reset state |
| `heads.{case}` | f32 [N/64, 2] | p_target, p_self per hop |
| `dbg.feat`, `dbg.z_enc`, `dbg.z_blk{b}`, `dbg.coef` | f32 [8, 771], [8, 16, 64], [8, 16, 64], [8, 1026] | first 8 hops of case `person`, short file only (definitions below the tolerance table) |

JSON `meta.tolerances` (the parity test must read and apply these)

| Check | Max abs error | Also |
|---|---|---|
| vad | 0 mismatches | - |
| tokens, g, film, k, v | 1e-4 | - |
| dbg (debug tensors) | 1e-4 | - |
| `out` short (2 s) | 2e-4 | SNR(engine vs reference) >= 60 dB |
| `out` long (10 s) | 1e-3 | SNR >= 50 dB |
| heads | 1e-3 | - |

- **meta.tolerances keys** (exactly these, in both files; a missing or extra key fails the parity test, never a
  default) - `vad` 0, `tokens` 1e-4, `g` 1e-4, `film` 1e-4, `k` 1e-4, `v` 1e-4, `dbg` 1e-4, `out` (short 2e-4, long
  1e-3), `out_snr_db` (short 60, long 50), `heads` 1e-3. The long file carries `dbg` although it has no `dbg` arrays.
- **dbg definitions** - `dbg.z_enc` = z after the band encoder; `dbg.z_blk{b}` = z after everything that belongs to
  block b (after FiLM 0 and attention for b = 0, after FiLM 1 for b = 2); `dbg.coef` = the 16 bands in band order,
  each `[re: T_k x W_k][im: T_k x W_k]`, after `1.5 tanh`.
- **Cases** - `person` = PERSON(target from `enr_t`, own from `enr_o`, use_wespk 1); `everyone` = EVERYONE(own, 1);
  `noown` = PERSON(target, no own, 1); `nowespk` = PERSON(target, own, 0).
- `meta.weights_sha256` = the model_hash (3.1) of the `earshot.esw` the vectors were made from.

The app runs `testvec_short.est` once per launch on the phone through `es_selftest`, which first requires
`meta.weights_sha256` to equal `es_model_data_hash` (else it fails with "test vectors are for other weights"), then
checks `vad_t`, tokens, g, `cond.person` and `out.person` / `heads.person` only (500 hops, about 0.1 s, estimate), and
the app refuses to process audio if it fails. The engine's own parity test checks every array and case.

---

## 4. C API (`engine/include/earshot.h`, API version 1)

C11, Apple Accelerate and CommonCrypto only. WP-ENGINE copies this header verbatim into `engine/include/earshot.h`.

```c
#ifndef EARSHOT_H
#define EARSHOT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define ES_API_VERSION      1
#define ES_SR               16000
#define ES_HOP              64        /* samples per hop at 16 kHz (4 ms) */
#define ES_NFFT             512
#define ES_NBINS            257
#define ES_NBANDS           16
#define ES_D                64
#define ES_WESPK_DIM        256
#define ES_G_DIM            128
#define ES_TOK_DIM          64
#define ES_MAX_TGT_TOKENS   48
#define ES_MAX_OWN_TOKENS   16
#define ES_NULL_TOKENS      4
#define ES_MAX_TOKENS       68
#define ES_MAX_INSTANCES    3
#define ES_ALGO_LATENCY     128       /* samples at 16 kHz */
#define ES_HISTORY_SAMPLES  262144    /* 16.4 s ring, 15.36 s readable (16,384-sample overrun guard) */
#define ES_PROBE_DBFS       (-30)     /* latency probe level; Phase 0 picks -30, -24 or -18 (DECISIONS row) */

typedef enum {
  ES_OK = 0, ES_ERR_ARG = -1, ES_ERR_FORMAT = -2, ES_ERR_VERSION = -3, ES_ERR_CONFIG = -4,
  ES_ERR_NOMEM = -5, ES_ERR_ENROLL_SHORT = -6, ES_ERR_BUSY = -7, ES_ERR_STATE = -8,
  ES_ERR_RATE = -9, ES_ERR_PROBE_WEAK = -10, ES_ERR_SELFTEST = -11, ES_ERR_THREAD = -12
} es_status;
typedef enum { ES_MODE_PERSON = 0, ES_MODE_EVERYONE = 1 } es_mode;
typedef enum { ES_ROLE_TARGET = 0, ES_ROLE_OWN = 1 } es_role;

const char *es_status_str(es_status s);                 /* static string, never NULL */

/* ---------- 1. Model weights (immutable after load, shareable) ---------- */
typedef struct es_model es_model;
es_status   es_model_load(const void *esw, size_t nbytes, es_model **out, char *err, size_t errlen);
void        es_model_free(es_model *m);
const char *es_model_config_hash(const es_model *m);    /* 64 hex chars, owned by m; architecture only */
const char *es_model_data_hash(const es_model *m);      /* data_sha256 = model_hash (3.1), 64 hex chars, owned by m */
int         es_model_latency_samples(const es_model *m);/* 128 */
es_status   es_selftest(const es_model *m, const void *est, size_t nbytes, float *max_err_out);

/* ---------- 2. Enrollment and conditioning (never on an RT thread) ---------- */
typedef struct es_voiceprint es_voiceprint;
es_voiceprint *es_voiceprint_new(void);
void        es_voiceprint_free(es_voiceprint *v);
int         es_enroll_num_frames(int n_samples);         /* 0 if n < 576 (so F starts at 2), else floor((n-64)/512)+1 */
void        es_vad(const float *pcm, int n, uint8_t *mask_out);  /* es_enroll_num_frames(n) entries; F = 0 writes nothing */
es_status   es_enroll(const es_model *m, const float *pcm, int n, const uint8_t *mask /* NULL = es_vad */,
                      es_role role, const float *e_wespk /* 256 or NULL */, es_voiceprint *out);
int         es_voiceprint_num_tokens(const es_voiceprint *v);
void        es_voiceprint_get_g(const es_voiceprint *v, float g_out[ES_G_DIM]);
int         es_voiceprint_has_wespk(const es_voiceprint *v);
float       es_voiceprint_speech_seconds(const es_voiceprint *v);   /* kept frames x 0.032 */
es_status   es_embed_g(const es_model *m, const float *pcm, int n, float g_out[ES_G_DIM]); /* "locked" check */

typedef struct es_cond es_cond;
es_cond    *es_cond_new(void);
void        es_cond_free(es_cond *c);
es_status   es_cond_build(const es_model *m, es_mode mode, const es_voiceprint *target /* NULL iff EVERYONE */,
                          const es_voiceprint *own /* NULL = none */, int use_wespk, es_cond *out);

/* ---------- 3. One stream = one instance (the model, hop by hop) ---------- */
typedef struct es_stream es_stream;
es_stream  *es_stream_new(const es_model *m);           /* allocates state and scratch */
void        es_stream_free(es_stream *s);
void        es_stream_reset(es_stream *s);               /* RT-safe: zero all state */
int         es_stream_hop(es_stream *s, const es_cond *c, const float in[ES_HOP],
                          float out[ES_HOP], float heads[2]);   /* RT-safe; returns 1 if a NaN reset happened */

/* ---------- 4. Engine = rings + resampling + worker + instances + post + safety ---------- */
typedef struct es_engine es_engine;
typedef struct {
  int io_rate;        /* 48000 or 44100, else ES_ERR_RATE */
  int io_frames;      /* granted callback size (frames), 16..4096 */
  int max_frames;     /* largest callback accepted, 4096 */
  int n_instances;    /* 1..3 */
  int slack16;        /* extra deadline slack in 16 kHz samples: 22, 43 or 64 */
} es_engine_config;
typedef struct {
  float voice_db;     /* -10..+6, added to the AGC target */
  float room_db;      /* -60 = off; -24, -15, -8 */
  float max_gain_db;  /* AGC ceiling 0..18; voice_db + max_gain_db is clamped to 24 */
  float out_gain_db;  /* final trim -40..0 */
  float slot_gain[ES_MAX_INSTANCES];  /* 0..1 per instance; changes crossfade over xfade_ms */
  float xfade_ms;     /* default 150 */
  int   duck_own;     /* 1 = -15 dB while p_self > 0.5 */
  int   gate;         /* 1 = activity gate on */
  int   agc;          /* 1 = slow voice AGC on */
  int   bypass;       /* 1 = play the delayed input instead of the model (A/B, diagnostics) */
} es_params;
typedef struct {
  uint64_t hops, underruns, nan_resets, probe_runs;
  int32_t  slack16, last_frames, howl;      /* howl = runaway guard active */
  float    p_target[ES_MAX_INSTANCES], p_self;
  float    in_rms_db, out_rms_db, agc_gain_db, limiter_gr_db;
  float    compute_us_last, compute_us_max;
} es_stats;
typedef struct { float loop_ms, spread_ms, psr_db; int n_ok; } es_probe_result;

void       es_params_default(es_params *p);  /* voice_db 0, room_db -60, max_gain_db 18, out_gain_db 0, slot_gain {1,0,0},
                                                xfade_ms 150, duck_own 1, gate 1, agc 1, bypass 0 (= Swift EngineParams) */
es_status  es_engine_new(const es_model *m, const es_engine_config *cfg, es_engine **out); /* starts muted (gain 0) */
void       es_engine_free(es_engine *e);                /* worker must be stopped */
void       es_engine_render(es_engine *e, const float *in, float *out, int n); /* RT: audio callback only */
int        es_engine_run_hops(es_engine *e);            /* RT: worker body, runs all complete hops */
es_status  es_engine_start_worker(es_engine *e, void *os_workgroup /* os_workgroup_t or NULL */);
void       es_engine_stop_worker(es_engine *e);
es_status  es_engine_set_cond(es_engine *e, int slot, const es_cond *c, int reset_state); /* may return BUSY;
             c == NULL empties the slot (zero output, no compute); the next non-NULL cond always resets (4.3) */
void       es_engine_set_params(es_engine *e, const es_params *p);
void       es_engine_set_mute(es_engine *e, int mute);   /* RT-safe; 5 ms ramp down; the first unmute and every
                                                            later one fade in over 400 ms (raised cosine) */
void       es_engine_cue(es_engine *e, int beeps);       /* RT-safe; 1..3 beeps of 60 ms at 880 Hz, -18 dBFS */
void       es_engine_get_stats(const es_engine *e, es_stats *s);   /* RT-safe snapshot */
float      es_engine_latency_ms(const es_engine *e);     /* Earshot's delay over a pass-through */
uint64_t   es_engine_input_position(const es_engine *e); /* RT-safe atomic load: samples written to the history */
int        es_engine_read_input(es_engine *e, uint64_t *cursor, float *dst, int max);  /* 16 kHz mic history from
             *cursor; a cursor older than the oldest held sample starts there; skipped = new - old cursor - returned */
int        es_engine_read_output(es_engine *e, int slot, float *dst, int n);  /* last n <= 48000 samples */
int        es_engine_read_timings(es_engine *e, uint16_t *us, int max);      /* per-hop compute, microseconds */
es_status  es_engine_probe_start(es_engine *e);
es_status  es_engine_probe_result(es_engine *e, es_probe_result *r);  /* STATE until done */

/* ---------- 5. DSP chain blocks (used inside render; public for tests) ---------- */
typedef struct { float gain, start, target, step, phase, sr; int shape; } es_ramp;  /* shape 0 linear, 1 raised cosine;
   gain = start + (target - start) * w(phase), phase 0..1, w = phase or 0.5 - 0.5 cos(pi phase) */
void es_ramp_init(es_ramp *r, float sr, float gain);
void es_ramp_to(es_ramp *r, float target, float ms, int shape);
void es_ramp_process(es_ramp *r, float *buf, int n);
typedef struct { float ceiling, rel_coef, gain, min_gain; } es_limiter;
void es_limiter_init(es_limiter *l, float sr, float ceiling_dbfs /* -3 */, float release_ms /* 50 */);
void es_limiter_process(es_limiter *l, float *buf, int n);   /* zero lookahead, instant attack */
void es_softclip(float *buf, int n);                        /* knee at -1 dBFS, output within [-1, 1] */

#ifdef __cplusplus
}
#endif
#endif
```

### 4.1 Ownership and threading

| Calls | Thread | RT-safe | Notes |
|---|---|---|---|
| `es_model_*` (incl. `es_model_data_hash`), `es_selftest`, `es_voiceprint_*`, `es_vad`, `es_enroll`, `es_embed_g`, `es_cond_*` | any non-RT thread | no (may allocate) | reentrant for different output objects; `es_model` is read-only after load and may be shared |
| `es_stream_new/free`, `es_engine_new/free`, `start_worker`, `stop_worker`, `probe_*`, `read_*` | non-RT | no | |
| `es_engine_set_cond`, `es_engine_set_params` | **one** non-RT writer (the Swift main actor) | no | lock-free hand-off, see 4.2 |
| `es_engine_render` | the audio I/O thread only | **yes** | constant time per frame, never waits |
| `es_engine_run_hops` | the worker only | **yes** | constant time per hop |
| `es_stream_hop`, `es_stream_reset`, `set_mute`, `cue`, `get_stats`, `es_engine_input_position`, DSP blocks | any | **yes** | |

Every object is created and freed by its own `_new` / `_free`; the caller owns what `_new` returns. Inputs passed by
pointer are only read during the call. RT-safe means no malloc/free, locks, syscalls other than `semaphore_signal`,
Objective-C or Swift, logging, file I/O or unbounded loops. Denormals are flushed to zero on the worker
(FPCR FZ bit) and BLAS is set single-threaded on that thread (`BLASSetThreading`). Build flags never include
`-ffast-math`.

### 4.2 Engine behavior (fixed numbers)

- **Rings** - single-producer single-consumer, C11 atomics, power-of-two sizes: in16 and out16 4,096 samples each,
  input history `ES_HISTORY_SAMPLES`, per-instance output history 65,536 (reads up to 48,000), timings 1,024 x uint16.
- **Render (audio thread), in order** - (1) decimate `in` to 16 kHz and push to in16 and the history ring; (2) if in16
  holds >= 64 samples, `semaphore_signal` the worker; (3) pull the 16 kHz samples the interpolator needs from out16; if
  short, use zeros for the missing part, ramp a conceal gain to 0 over 1 ms and back to 1 over 1 ms when data returns,
  count one underrun; samples that arrive after their time has passed are skipped, so the delay stays 64 + P (20-engine
  8.3 step 3); planned zeros (slack growth, payback for input lost to a full in16, which also counts as an underrun)
  get the same 1 ms ramps before and after them; (4) interpolate to `n` frames; (5) safety chain at the I/O rate (below); (6) if a latency probe
  is running, replace the output with the probe signal and copy `in` into the probe buffer.
- **Worker** - wait on the semaphore; `es_engine_run_hops`: while in16 >= 64, pop one hop, read each instance's
  conditioning index (4.3), run `es_stream_hop` per active instance, post-process, push 64 samples to out16, write
  per-hop compute time to the timings ring. Thread policy `THREAD_TIME_CONSTRAINT_POLICY`, computed by
  `es_engine_new` from the config so a wake that runs several hops is never an overrun by design -
  `period = max(4 ms, io_frames / io_rate)`, `computation = min(period / 2, ceil(period / 4 ms) x 1.0 ms)`,
  `constraint = 0.625 x period`, preemptible. At the default 64 frames this is period 4 ms, computation 1.0 ms,
  constraint 2.5 ms; at 1,104 frames, 23 ms / 6 ms / 14.4 ms (analytic; the 1.0 ms per hop is an estimate until
  Phase 0). Joined to the RemoteIO workgroup with `os_workgroup_join`.
- **Prefill and latency** - out16 starts with `P = P_min(io_rate, io_frames) + slack16` zeros, where `P_min` is the
  smallest prefill with zero starvation for an instant worker whose result is first used by the next callback,
  found by a 1,200-callback simulation inside `es_engine_new` (table in 20-engine 9.4, analytic: 64 at 48 kHz for
  16 to 96 and 192 frames, 86 at 128, 128 at 256; 69 to 132 at 44.1 kHz for 16 to 192 frames).
  `es_engine_latency_ms = 1000 (64 + P) / 16000 + rs_delay_ms`; with P_min 64, slack16 43 and rs 0.2575 ms (48 kHz)
  this is 10.95 ms; a 128-frame grant gives 12.3 ms and 44.1 kHz routes 11.3 to 15.2 ms (analytic).
  A test feeds an impulse and must measure this delay within 1 sample at 16 kHz.
- **Slack growth** - AirPods default slack16 = 43 (2.69 ms), wired default 22. After 2 underruns within 60 s the
  engine raises slack16 one step (22, 43, 64) by inserting zeros at a moment when all p_target < 0.2. It never lowers
  slack during a session; a new engine (route change) starts at the default.
- **NaN guard** - any non-finite stream output resets that stream, outputs zeros for the hop, counts `nan_resets`.
- **Resamplers** - one rational polyphase FIR implementation. 48 kHz tables - 177-tap Kaiser lowpass (cutoff
  7,500 Hz, beta 5.65) made minimum-phase (`scipy.signal.minimum_phase(h, "homomorphic", half=False)`); computed
  here with scipy 1.18.1 - passband ripple +/-0.004 dB to 6.5 kHz, -3 dB at 7,385 Hz, >= 60.0 dB stopband from
  8 kHz, group delay 0.115 to 0.154 ms at 1 to 4 kHz (mean 0.129 ms per direction; `ES_RS_DELAY_MS` = 0.2575 ms for
  both directions, used in budgets). 44.1 kHz tables (ratio 160/441) use the same design at the higher rate; their
  figures are printed by the tool (group delay 0.135 to 0.175 ms per direction, `ES_RS_DELAY_MS` 0.2987 ms, analytic,
  20-engine 9.1) and the 44.1 kHz tests compare with those. Tables are generated by
  `engine/tools/make_resampler_tables.py` into `engine/src/resampler_tables.c`; the same tool writes
  `ml/assets/decim16k_fir.npy` (the 0 to 8 kHz response at 16 kHz, 64 taps) for training augmentation. The lead runs
  the tool on day 0 and commits both outputs (section 11). Any other I/O rate returns ES_ERR_RATE.
- **Post-processing per hop at 16 kHz** (only active slots count; an active slot has conditioning with n_tokens >= 4) -
  (1) per instance activity gate, gain linear in dB from 0 dB at p_target >= 0.5 to -12 dB at p_target <= 0.2; it
  opens within 1 hop (onsets are never cut) and closes with a 200 ms time constant; (2) sum instances with slot_gain
  crossfades (equal-power, xfade_ms); (3) own-voice duck -15 dB while max p_self > 0.5, engaging within 1 hop and
  releasing with 150 ms; (4) slow AGC - RMS over 400 ms of the crossfaded sum of (2) (before the duck), measured only
  while max p_target > 0.5, target -26 dBFS + voice_db, slew at most 6 dB/s, gain within [-12, max_gain_db], frozen
  otherwise; (5) gain cap - 50 ms one-pole mean squares of the output of (4) and of x_d; any excess of their ratio over
  +24 dB is removed at once and released with a 200 ms time constant (the deep filter alone can raise a bin by up to
  6.36x, +16.1 dB, since each tap's real and imaginary parts reach 1.5); (6) room mix-back `+ 10^(room_db/20) x_d`,
  forced to 0 at once while max p_self > 0.5 whatever duck_own is, recovering with 150 ms; (7) bypass replaces (1) to
  (6) by x_d. **x_d** = the 16 kHz input of the previous hop (a 64-sample delay inside the hop stream), so x_d and the
  model output share the same 128-sample algorithmic latency measured from capture (64 of stream lag plus the 64-sample
  hop wait, 2.1).
- **Safety chain at the I/O rate** - mute ramp (5 ms linear down, 400 ms raised-cosine fade-in on every unmute; a new
  engine starts muted, so the app's first unmute is the start fade-in), out_gain_db, cue beeps mixed, zero-lookahead
  peak limiter (ceiling -3 dBFS, instant attack, 50 ms release), soft clip above -1 dBFS, hard clip to [-1, 1]. Runaway guard - output RMS above input RMS + 30 dB for 200 ms ducks
  the output by 20 dB and sets `howl`. Mic-to-ear gain (50 ms RMS) never exceeds +24 dB, enforced by post step (5);
  the -3 dBFS limiter bounds peaks.
- **Latency probe** - 5 exponential sweeps 300 Hz to 6 kHz, 0.3 s each, at `ES_PROBE_DBFS` (-30 dBFS until Phase 0
  picks -30, -24 or -18 and records it in DECISIONS), 0.5 s apart (4.0 s), mic captured at the I/O rate into a
  preallocated 4.5 s buffer. `probe_result` (non-RT) runs GCC-PHAT per sweep, returns the median
  lag (`loop_ms` = round trip X + 2B of a pass-through), the spread, and the peak-to-sidelobe ratio. Fails with
  ES_ERR_PROBE_WEAK only if fewer than 3 sweeps have psr >= 10 dB, and always fills `spread_ms`; the app, which knows
  the route, rejects a spread over 10 ms on Bluetooth or 2 ms wired (the engine is never told the route).

### 4.3 Conditioning hand-off (lock-free, no waiting on the audio side)

Per instance the engine holds `cond[2]`, `atomic int active`, `atomic int ack`, and a reset flag per buffer.
`es_engine_set_cond` (single writer) returns ES_ERR_BUSY if `active != ack` (the last change is not yet picked up);
otherwise it copies `c` into `cond[1 - active]`, stores the reset flag, then `active = 1 - active` (release). The
worker at each hop start does `a = active` (acquire), uses `cond[a]`, resets the stream first if that buffer's reset
flag is set and not yet applied, then `ack = a`. When the worker is stopped, set_cond also sets `ack = active`.
The Swift wrapper retries BUSY every 4 ms for up to 200 ms. A switch takes effect at the next hop boundary.
Warm switch (instance already running that person) - no reset. Cold switch (new person into a slot) - reset.
**Empty slot** - `es_engine_set_cond(e, slot, NULL, any)` (same BUSY rule) stores an empty buffer (n_tokens = 0): the
worker skips that stream and outputs zeros for it, and the next non-NULL cond always resets the stream. Every slot of
a new engine starts empty. **Active slot** = current conditioning with n_tokens >= 4. The app first sets that slot's
`slot_gain` to 0 and waits `xfade_ms` before emptying it (thermal downshift, deleting a person, 5.3).
`es_engine_set_params` uses a seqlock; the worker reads it once per hop and keeps the previous params on a torn read
(never spins).

---

## 5. Swift API (`ios/Earshot/Core/`, Swift 5 language mode, iOS 26.0 target)

Rule - Swift never runs on the audio thread. The RemoteIO render callback is C (`ios/Earshot/Audio/AudioHost.c`) and
calls only `es_engine_render`. All `es_engine_set_*` calls come from the main actor.

```swift
// EarshotCore.swift - thin wrappers, one file
public struct EarshotError: Error { public let status: Int32; public let message: String }
public enum ListenMode: Int32 { case person = 0, everyone = 1 }
public enum VoiceRole: Int32 { case target = 0, own = 1 }

public final class EarshotModel: @unchecked Sendable {            // immutable after init
    public init(weights: Data) throws                              // es_model_load
    public var configHash: String { get }
    public var modelHash: String { get }                           // es_model_data_hash = model_hash (3.1)
    public func selfTest(testVector: Data) throws -> Float         // es_selftest, returns max error
    public func vad(_ pcm16k: [Float]) -> [UInt8]                  // es_vad, es_enroll_num_frames(n) entries
    public func embedG(_ pcm16k: [Float]) throws -> [Float]        // es_embed_g, 128, unit length (locked check)
}
public final class Voiceprint: @unchecked Sendable {               // es_voiceprint, immutable after init
    public init(model: EarshotModel, pcm16k: [Float], mask: [UInt8]?, role: VoiceRole, weSpeaker: [Float]?) throws
    public var tokenCount: Int { get }
    public var g: [Float] { get }                                  // 128, unit length
    public var speechSeconds: Float { get }
    public var hasWeSpeaker: Bool { get }
}
public final class Conditioning: @unchecked Sendable {             // es_cond
    public init(model: EarshotModel, mode: ListenMode, target: Voiceprint?, own: Voiceprint?, useWeSpeaker: Bool) throws
}
public struct EngineParams {                                        // mirrors es_params, defaults = es_params_default
    public var voiceDB: Float = 0, roomDB: Float = -60, maxGainDB: Float = 18, outGainDB: Float = 0
    public var slotGain: [Float] = [1, 0, 0], crossfadeMS: Float = 150
    public var duckOwnVoice = true, gate = true, agc = true, bypass = false
}
public struct EngineStats { /* mirrors es_stats */ }
public struct ProbeResult { public let loopMS: Float, spreadMS: Float, psrDB: Float, okSweeps: Int }

public final class EarshotEngine: @unchecked Sendable {            // owns es_engine*
    public init(model: EarshotModel, ioRate: Int, ioFrames: Int, instances: Int, slack16: Int) throws
    public var cEngine: OpaquePointer { get }                       // passed to AudioHost as refCon
    public func startWorker(workgroup: UnsafeMutableRawPointer?) throws   // value of eh_workgroup(), may be nil
    public func stopWorker()
    @MainActor public func setConditioning(slot: Int, _ c: Conditioning, resetState: Bool) async throws  // retries BUSY
    @MainActor public func clearConditioning(slot: Int) async throws    // es_engine_set_cond(NULL), same BUSY retry
    @MainActor public func setParams(_ p: EngineParams)
    public func setMuted(_ muted: Bool)
    public func cue(beeps: Int)
    public func stats() -> EngineStats
    public var latencyMS: Float { get }                             // es_engine_latency_ms
    public func makeInputTap() -> InputTap                          // starts at es_engine_input_position (new audio only)
    public func recentInput(samples: Int) -> [Float]                // newest n <= 245,760 samples (quick add)
    public func recentOutput(slot: Int, samples: Int) -> [Float]
    public func drainTimings() -> [UInt16]
    public func startLatencyProbe() throws
    public func latencyProbeResult() -> Result<ProbeResult, EarshotError>?   // nil while running
}
public struct InputTap {                                            // reads the 16 kHz mic history ring
    public mutating func read(maxSamples: Int) -> [Float]           // new samples since the last read, in order
    public private(set) var droppedSamples: Int                     // samples skipped because the reader fell behind
}
```

```c
// AudioHost.h - C, owned by WP-IOS
typedef struct eh_host eh_host;
OSStatus eh_create(void *es_engine, double io_rate, int io_frames, eh_host **out); // RemoteIO, input bus 1, Float32 mono
OSStatus eh_start(eh_host *h);
OSStatus eh_stop(eh_host *h);
void    *eh_workgroup(eh_host *h);    // kAudioOutputUnitProperty_OSWorkgroup, size checked, NULL on failure
OSStatus eh_set_engine(eh_host *h, void *es_engine); // atomic swap of the engine the callback uses; returns once the
                                      // callback acknowledged it (or at once if I/O is stopped), error after 200 ms
void     eh_destroy(eh_host *h);
```

Input taps - `makeInputTap()` starts at `es_engine_input_position` (new audio only, never audio from before the call).
`read(maxSamples:)` returns new samples in order; when the reader fell behind, `es_engine_read_input` starts at the
oldest held sample and the skipped count (new cursor - old cursor - returned) is added to `droppedSamples`. A tap
belongs to one engine; after an engine swap every tap is recreated. `recentInput(samples:)` reads the newest n samples
(at most 245,760 = 15.36 s) for quick add.

App-level Swift types (signatures fixed, bodies by WP-IOS)
- `@MainActor final class AudioController: ObservableObject` - `func start() async throws`, `func stop()`,
  `@Published var route: RouteInfo`, `@Published var latency: LatencyDisplay`, `func measureLatency() async throws`.
  Owns AVAudioSession, `eh_host`, `EarshotEngine`; swaps in a new engine on every route or buffer change (5.3
  Lifecycle).
- `actor SpeakerEncoder` - `init(modelURL: URL) throws`, `func embed(_ pcm16k: [Float]) throws -> [Float]` (256, unit
  length), `func selfTest() -> Bool`; holds a `.cpuAndGPU` and a `.cpuOnly` instance (section 6).
- `actor EnrollmentService` - `func enroll(pcm16k: [Float], role: VoiceRole, own: Person?, saved: [Person]) async throws
  -> EnrollmentResult` (voiceprint, quality report, pass / warn / fail with message).
- `@MainActor final class PeopleStore: ObservableObject` - `people: [Person]`, `own: Person?`, `func save(_:)`,
  `func delete(_:)`, `func purgeExpired()`.
- `@MainActor final class Selector` - `func lookSelect()`, `func cycle()`, `func setMode(_:)`; uses `HeadTracker`
  (`CMHeadphoneMotionManager`, smoothed yaw in degrees) and `RemoteCommands` (`MPRemoteCommandCenter`).

### 5.1 Person storage (`Application Support/People/<uuid>/`, file protection complete-until-first-auth)

`person.json` - `{id, name, kind ("tonight" | "saved" | "own"), created, updated, expires (ISO 8601 or null),
yaw_deg (or null), model_hash, e_wespk [256] (or null), mask (base64 of the uint8 token-frame mask passed to
es_enroll), quality {speech_s, snr_db, consistency, own_cos}}` plus `enroll.wav` (16 kHz mono 16-bit, at most 10 s).
`model_hash` is the 3.1 value of the weights in use when the person was saved. Voiceprints and conditioning are rebuilt
from `enroll.wav` and the stored `mask` at launch (a few ms of C each, estimate; no WeSpeaker call), so a new model
needs no migration; only a file without `mask` gets one recomputed (8.3 of 30-ios). `yaw_deg` is a hint from the last
session: AirPods yaw has no fixed reference across sessions or reconnects, so it always loads as stale and is
re-anchored by the first look-select while exactly one table person is talking. "Tonight" people expire 12 h after the session ends.
Nothing is uploaded. Saving someone asks the user to confirm the person agreed.

### 5.2 Session telemetry (`Documents/logs/YYYY-MM-DD.jsonl`, one line per minute, no audio)

`t, route, io_rate, io_frames, slack16, mode, n_instances, compute_us_p50/p99/max, underruns, nan_resets, howl,
thermal, battery, low_power, out_rms_db, limiter_gr_db, p_target_mean[], p_self_mean, latency_measured_ms,
latency_reported_ms, app_version, model_hash`.

### 5.3 App behavior constants (AirPods defaults)

| Item | Value |
|---|---|
| Audio session | `.playAndRecord`, mode `.measurement`, options `[.allowBluetoothA2DP]` only; preferred 48 kHz, 64-frame buffer; built-in mic re-selected after every route change; `setPrefersNoInterruptionsFromSystemAlerts(true)`, `setAllowHapticsAndSystemSoundsDuringRecording(true)` |
| Allowed outputs | `.bluetoothA2DP`, `.headphones`, `.usbAudio`; anything else mutes at once |
| Start | Everyone mode until a person exists; the engine starts muted and the first unmute is the 400 ms fade-in |
| Enrollment (target) | stop at 6 s of VAD speech or 15 s total; accept >= 4 s; clip trimmed to first..last speech frame, at most 10 s |
| Enrollment (own voice) | read a passage; stop at 8 s of speech (max 20 s), checked on the full capture; `snr_db` (the sbr formula of section 6) >= 20 dB pass, < 15 dB fail; stored clip = the 10 s window with the most es_vad frame midpoints (512-sample steps, earliest on ties), rounded down to 0.5 s; the same clip feeds es_enroll, WeSpeaker and enroll.wav |
| Quick add | "Add who just spoke" uses the last 8 s of the engine's input history (`recentInput`), plays 2 s back for confirmation; the app keeps a RAM copy of the last 8 s only across an engine swap |
| Stem single / double / triple | single - pause (all slot gains to 0 and room off over the 150 ms crossfade, then set_mute(1)); resume = set_mute(0) (400 ms fade) and restore gains and room. double - look-select; with no person on the table, hands-free target capture (1 beep at start, 2 on success, 3 on failure). triple - Focus <-> Table if >= 2 people, else Focus <-> Everyone |
| Look-select | yaw smoothed 200 ms; nearest stored direction within 35 degrees, must beat the current one by 10 degrees; else cycle in table order; 1-3 beeps = slot |
| Direction learning | while a person talks (p_target > 0.7 for 1 s) and head rate < 20 deg/s, `dir += 0.1 * wrap(yaw - dir)`; yaw jump > 45 deg in < 50 ms marks directions stale |
| Warm instances | up to 3 (Table); a 4th person is a cold switch (reset). Everyone goes into a free slot (reset), else into the selected person's slot (that person returns cold). Deleting a person on the table fades its slot to 0, then `clearConditioning` |
| Locked check | every 1 s, `es_embed_g` on the last 3 s of an instance's output, cosine to the person's g; lost for 5 s while p_target > 0.5 shows "Might be following someone else"; no automatic gain change in v1 |
| Thermal | `.serious` or Low Power Mode - set every non-selected slot gain to 0, wait 150 ms, then `clearConditioning` on those slots (no engine rebuild), alarm classifier every 3 s; recovery (`.fair` and Low Power off for 60 s) refills them cold; `.critical` - fade to pause with a notice |
| Alarm passthrough | `SNClassifySoundRequest(.version1)` on the 16 kHz history, 3 s windows, 50% overlap; smoke_detector, siren family, car_horn, screaming, glass_breaking, baby_crying with confidence >= 0.6 in 2 windows - bypass on with out_gain_db -6 for 5 s, tone, banner; labeled a helper, not a safety device |
| Latency display | "Delay 59 ms (measured 3 min ago) - Earshot 11 + route 48"; without a measurement "about N ms (estimate)" from inputLatency + outputLatency + 2 x ioBufferDuration + Earshot; measure prompt at connect; re-checks at 10 and 30 min are offered, never required; a valid measurement is stored per output port UID and shown with its age when a new probe fails; after PROBE_WEAK the app offers method B (ear detection off) |
| Lifecycle | background audio mode; start only in the foreground; interruption ends with shouldResume - reactivate, else local notification "Tap to resume Earshot"; mediaServicesWereReset - rebuild everything. **Engine swap** (route signature or callback size change, interruption end) - RemoteIO keeps running: the main actor builds the new engine (muted, same io_rate, max_frames 4096, conditioning set, worker started with `eh_workgroup`), mutes the old one, calls `eh_set_engine` (waits for the callback's acknowledge), stops the old worker, frees the old engine, then unmutes the new one (400 ms fade-in). RemoteIO is stopped and recreated only when the hardware rate changes or after mediaServicesWereReset; if that happens in the background, Earshot mutes and posts "Tap to resume Earshot" instead (a recording session cannot start from the background). Whether a locked swap keeps audio alive is a Phase 0 row |

---

## 6. Speaker encoder contract (WeSpeaker ResNet34)

| Item | Value |
|---|---|
| Weights | `models/wespeaker-voxceleb-resnet34/avg_model.pt` (on disk, 6.63 M params, VoxCeleb research terms - **personal use only**, never distributed) |
| One definition | `ml/earshot/wespeaker.py` - Kaldi fbank as one fixed Conv1d (514 x 400 kernel, stride 160: DC removal, pre-emphasis 0.97, Hamming, 512-point DFT), 80 mel, log floor 1.19e-7, mean normalization over time, ResNet34 (m = 32), stats pooling, Linear to 256, **L2 normalize**. Matches torchaudio kaldi fbank within 8.1e-5 and the official ONNX at cosine 1.0 (measured, docs/research/data-training.md 4.3) |
| Input | Core ML input `wav`, float32 shape [1, N], raw 16 kHz audio in [-1, 1] (x 32768 happens inside), N flexible 16,000..160,000 (`ct.RangeDim`) |
| Output | `embedding`, float32 [1, 256], unit length |
| Package | `ml/exports/wespeaker_resnet34_fp32.mlpackage`, mlprogram, fp32 everywhere (the fp16 front end is broken, measured), minimum deployment iOS 17, about 26 MB |
| Compute units | `.cpuAndGPU` in the foreground; never `.all` (the mixed-precision package returned cosine 0.57 on the Neural Engine and the fp32 package 0.9995 with `.all`, measured, docs/research/data-training.md 4.3). `SpeakerEncoder` holds two compiled instances, `.cpuAndGPU` and `.cpuOnly` (fp32 CPU_ONLY cosine 1.0000, measured, same file), and uses `.cpuOnly` whenever `UIApplication.shared.applicationState != .active` (background apps cannot submit GPU work; hands-free capture runs locked) |
| Self-test | `wespk_selftest.est` - EST1 container, format earshot-testvec, the esb16-df config, `meta.kind` "wespeaker-selftest", `meta.tolerances` {"cosine_min": 0.999}, arrays `clip` f32 [80000] (5 s) and `embedding` f32 [256]. At launch embed `clip`; cosine >= `meta.tolerances.cosine_min` passes; else retry with `.cpuOnly`; else run without WeSpeaker (`use_wespk = 0`) and show a notice |
| Phone usage | enrollment clips (target, own voice), the one-voice check windows (1.5 s, hop 0.5 s); never on the audio thread, never in the live path |
| Training match | the same `wespeaker.py` module, frozen, eval mode, fp32, no grad, on MPS: target e_t computed in the training loop on the noisy augmented enrollment (measured config, merged_full); own-voice e_o read from a cache of clean 8 s clips (`ml/tools/cache_own_embeddings.py`: per own-pool speaker, 8 seeded 8 s windows of the speaker's concatenated cache rows, overlap allowed, section 8.1; about 31 MB), matching the phone where own-voice enrollment must pass snr_db >= 20 dB |

Enrollment quality checks (initial thresholds in `thresholds.json`; WP-ML calibrates sbr_db, one_voice, own_cos and
locked_cos on validation so that 95% of good single-talker enrollments pass and at least 80% of bad ones fail, using the
check formulas below; speech_s [4, 5], dup_cos 0.75 and clip_pct [0.1, 1.0] stay fixed, since speech_s is the 5.3
accept limit; the app reads the file, never hard-codes values)

| Check | Pass | Warn | Fail |
|---|---|---|---|
| VAD speech seconds (target) | >= 5 | 4 to 5 | < 4 |
| Speech-to-background (speech vs non-speech frame energy) | >= 6 dB | 0 to 6 dB | < 0 dB |
| One voice (median cosine of 1.5 s window e to clip e) | >= 0.60 | 0.45 to 0.60 | < 0.45 |
| Not the user (cosine e to own e) | < 0.50 | 0.50 to 0.60 | >= 0.60 |
| Distinct from saved people (cosine e) | < 0.75 | >= 0.75 asks "sounds like X - update X?" | - |
| Clipping (samples at full scale) | < 0.1% | 0.1 to 1% | >= 1% |

**Check formulas** (shared by `ml/earshot/quality.py`, used by `calibrate_thresholds.py`, and by the app's ClipTools and
EnrollmentService; x = the full 16 kHz capture of n samples)
1. Frame map - token frame f covers raw samples [512f - 448, 512f + 64), midpoint 512f - 192 (2.4 framing);
   `mask0 = es_vad(x)`.
2. Span - from the first to the last frame with mask0 = 1: [max(0, 512 f1 - 448), min(n, 512 f2 + 64)).
3. Longer than 10 s - keep the 10 s window (start on a 512-sample step) holding the most mask0 frame midpoints,
   earliest on ties. Own voice always takes this window.
4. Round the length down to a multiple of 8,000 samples (0.5 s), keeping the start; under 64,000 samples (4 s) fails.
   The result is the clip (stored as enroll.wav).
5. WeSpeaker windows - start 8000 w, length 24,000, w = 0 .. floor((len - 24000) / 8000); a clip frame is inside
   window w when its midpoint lies in [8000 w, 8000 w + 24000).
6. `sbr_db = 10 log10(mean x^2 over mask0 speech frames / mean x^2 over the other frames)` on the full capture; with
   no non-speech frame, the lowest-10%-energy frames are the other frames. The own-voice `snr_db` is this formula.
7. `clip_pct = 100 x count(|x| >= 0.98) / n` on the full capture.
8. e = embed(clip), e_w = embed(window w); `one_voice = median_w cos(e_w, e)`; `own_cos = cos(e, own e)`;
   `dup = max cos(e, e of saved and tonight people)`.
9. Mask - targets only: `es_vad(clip)`, then zero every frame inside a window with cos(e_w, own e) > own_window_cos.
   Own voice: `es_vad(clip)`.
10. `speech_s = es_voiceprint_speech_seconds` after `es_enroll` with that mask (the own-voice 8 s stop rule counts the
    full capture).

Without WeSpeaker (self-test failed) one_voice, own_cos and dup show "not checked", step 9 keeps `es_vad(clip)`
unchanged, and `use_wespk = 0`. `ml/exports/<run>/quality_vectors.json` (5 fixed clips with the expected sbr_db,
clip_pct, window list and speech_s) is checked by `ml/tests/test_metrics.py` and by the app's `ClipToolsTests`. If
`thresholds.json` `model_hash` differs from `EarshotModel.modelHash`, the app still uses the file, shows "Thresholds
were calibrated for another model" in Diagnostics and logs it.
`thresholds.json` - `{"version": 1, "model_hash": "...", "target": {"speech_s": [4, 5], "sbr_db": [0, 6],
"one_voice": [0.45, 0.60], "own_cos": [0.50, 0.60], "dup_cos": 0.75, "clip_pct": [0.1, 1.0]}, "own": {"speech_s": 8,
"snr_db": [15, 20]}, "own_window_cos": 0.60, "locked_cos": 0.0}` (`locked_cos` set by calibration at 1% false accepts).

---

## 7. Repository layout and ownership

```
earshot/
  docs/        BRIEF.md, spec/ (00-contract lead; 10-ml, 20-engine, 30-ios by their writers), DECISIONS.md,
               measurements/phase0-YYYY-MM-DD.json (WP-IOS), eval/report-v1.md (WP-ML)        lead
  dataprep/    existing data pack code (frozen; bugs fixed by the lead only)                  lead
  data/, models/   gitignored                                                                 -
  ml/                                                                                         WP-ML
    __init__.py, earshot/__init__.py, tools/__init__.py   (empty, so python -m works)
    earshot/   config.py dsp.py model.py stream_ref.py wespeaker.py vad.py fileio.py losses.py metrics.py quality.py
               sources.py (manifests, pools, license gate) render.py (one example) enroll.py (enrollments)
               data.py (recipes, Dataset, loader)
    train.py export.py evaluate.py        (run as python -m ml.train etc. from the repo root)
    requirements-eval.txt   pystoi==0.4.1, pesq==0.0.4 (the only packages added to .venv, one online install)
    configs/   smoke.yaml pilot.yaml main.yaml finetune.yaml
    tools/     convert_wespeaker.py cache_own_embeddings.py calibrate_thresholds.py render_val.py run_forever.sh
    tests/     test_dsp.py test_vad.py test_model.py test_stream.py test_causal.py test_export.py test_wespeaker.py
               (the 7 ML-0 files) test_data.py test_losses.py test_metrics.py test_train.py test_gates.py
    assets/    decim16k_fir.npy (committed; generated by engine/tools/make_resampler_tables.py, run and committed
               by the lead only)              runs/, exports/, cache/  gitignored
  engine/                                                                                     WP-ENGINE
    include/earshot.h
    src/       internal.h (private structs) model.c json_min.c nn.c stft.c stream.c enroll.c cond.c resample.c
               resampler_tables.c (committed, lead regenerates) ring.h pipeline.c post.c safety.c rt_thread.c probe.c
               selftest.c
    tools/     make_resampler_tables.py esrun.c (offline WAV through the full engine) wav.c wav.h
    tests/     test_units.c test_parity.c test_rt.c rt_guard.c (malloc and lock interposer dylib) test_fuzz.c
               test_latency.c bench.c
    bench-results/   YYYY-MM-DD.jsonl (bench output, committed)
    Makefile   targets: all test parity fuzz bench esrun ios-check clean
  ios/                                                                                        WP-IOS
    project.yml          XcodeGen spec (xcodegen at /opt/homebrew/bin); Earshot.xcodeproj is generated, gitignored
    Makefile             project, build, install (xcodebuild + devicectl), test
    Earshot/   App/ (EarshotApp, Launch, Earshot-Bridging-Header.h) Core/ (EarshotCore.swift, ContainerFile,
               Thresholds) Audio/ (AudioHost.c/.h, AudioController.swift) Enrollment/ People/
               Control/ (HeadTracker, RemoteCommands, Selector) Safety/ (AlarmListener, LockedCheck)
               Telemetry/ (SessionLog, Phase0) UI/ Resources/
    Earshot/Resources/Models/   gitignored; filled by `make models` (lead)
    Stubs/earshot_stub.c        pass-through implementation of earshot.h for day-one work; removed at integration
    EarshotTests/
  bench/       existing benchmarks (read-only)
  Makefile     lead: models (copy exports into ios Resources), check (all tests), clean
```

The app target compiles `engine/src/*.c` directly (no static library). Bundle id `com.aarushkandukoori.earshot`,
the owner's free Personal Team, automatic signing, deployment target iOS 26.0, Swift 5 mode, `-O2 -std=c11 -DACCELERATE_NEW_LAPACK` C
(the same flags as `make ios-check`; engine sources also `-Wall -Wextra -Werror`), no `-ffast-math`. `ios/project.yml`
is specified only in 30-ios 2.1.
Info.plist keys - NSMicrophoneUsageDescription, NSMotionUsageDescription, UIBackgroundModes [audio].

---

## 8. Training (M4 MPS, torch 2.7.1 fp32, `.venv`)

### 8.1 Data and mixture recipe (dynamic mixing at 16 kHz, deterministic per (seed, example index))

Sources - speech_train (53.4 h, 3,758 speakers) for targets, interferers, babble and own voice; noise_train (23.6 h:
FSD50K, DEMAND, MUSAN noise and music); RIRs sim_train (3,800 rooms x 8 positions: 0 target, 1-3 talkers, 4 own voice,
5-7 babble and noise) and real_train (325). Validation - speech_val, noise_val, sim_val. Test speakers never appear in
training. The loader refuses any row whose license is non-commercial unless the split is test.

Speaker pools (counts measured 2026-11-07 from data/manifests/speech_train.jsonl) - target pool = speakers with >= 2
utterances and >= 14.5 s in total (3,118); a target is re-drawn when its enrollment rows are shorter than the
enrollment length or the remaining rows are shorter than the segment, so the enrollment is never cycled and never
shares audio with the target stream.
Own-voice pool = speakers whose rows, in manifest order, split into cache rows (the shortest prefix reaching 8 s, never
the last row) and mix rows (the rest) holding >= 4 s (3,135). Interferer and babble pools = all speakers. The AAM head
keeps all 3,758 classes.

| Element | Value |
|---|---|
| Mode | Person 75% (target present 90%, absent 10%), Everyone 25% |
| Competing talkers | 0 (10%), 1 (35%), 2 (30%), 3 (25%), each active 50-100% of the segment |
| Target vs loudest talker (SIR) | uniform -8 to +12 dB |
| Own voice | in 50% of examples (RIR position 4), U(0, 18) dB above the target at the mic in sim in_hand rooms and U(0, 12) dB otherwise, active 20-60%, never in the reference; the own talker uses only its mix rows, never its cache rows; own enrollment given 90% (null 10%, and then no own talker) |
| Babble | 70%, 6-20 talkers at positions 5-7 |
| Noise / music | each example has exactly one of noise (80%) or music (20%), plus babble in 70%. Noise sources - DEMAND 30, FSD50K 40, MUSAN 20, synthetic 10% (generated in the loader, no RIR: 70% colored noise shaped by f^(-alpha/2), alpha U(0, 2); 30% 50 or 60 Hz hum, harmonics 1-10 at amplitude U(0.2, 1)/n, plus colored noise 20 dB below). The background budget P_b = P_t / 10^(SNR/10) goes to the backgrounds present: music alone takes all of P_b; music with babble takes min(P_t 10^(-U(5, 20)/10), 0.5 P_b) and babble the rest; noise and babble split P_b as w / 1 - w with w U(0.2, 0.8); one background alone takes all |
| SNR (target vs babble + noise + music) | N(5, 5) dB clipped to [-8, 25]; 10% quiet rooms at 20-40 dB |
| Rooms | 75% sim_train, 15% real_train, 10% dry |
| Reference | direct path + first 50 ms of the target RIR, 5 ms half-cosine fade, propagation delay kept |
| Everyone reference | sum of target and competing talkers (positions 0-3) as above; babble, noise, own voice excluded |
| Mic chain (same linear filters on mixture and reference) | high-pass 60-200 Hz, 0-2 peaking EQs +/-4 dB, `decim16k_fir`, self-noise -95 to -75 dBFS, level -40 to -15 dBFS RMS, 3% hard and 3% soft clipping (mixture only) |
| Lombard | 30% of loud scenes (drawn SNR <= 5 dB): +3 to +8 dB and +2 to +4 dB shelf above 1 kHz on talkers |
| Target enrollment | other utterances of the target (never cycled), length per batch uniform 3 to 10 s (0.5 s steps); 80% noisy at 0-25 dB SNR; 30% with one weaker talker 5-15 dB below; room same 50% / other 30% / dry 20%; mask oracle VAD 40%, `es_vad` rule 40%, all ones 20% |
| Own enrollment | one of the speaker's 8 cached clean 8 s clips (8 seeded 8 s windows of its concatenated cache rows, overlap allowed); token path gets noise at 15-30 dB in 30% |
| WeSpeaker dropout | 20% of examples use `null.e_t` and `null.e_o` |
| Segment, batch | 4 s x 8 (stages A, B); 8 s x 4 (stage C) |

### 8.2 Losses (fp32)

| Term | Formula | Weight | Applies |
|---|---|---|---|
| Thresholded SNR | `-10 log10(|s|^2 / (|s - y|^2 + 1e-3 |s|^2))` | 1 | target present, Everyone |
| Multi-resolution STFT | L1 magnitude + L1 log magnitude at FFT 256 / 512 / 1024, hop 1/4, Hann | 1 | same |
| Absent | `10 log10(|y|^2 + 10^-2.5 |x|^2)` | 1 | absent examples |
| Target-silent frames | absent form on frames where the target VAD is 0 (>= 0.1 s) | 0.1 | present |
| Own-voice frames | absent form on frames where own voice is active and the target silent | 0.3 | own voice in mix |
| Over-suppression | `mean(ReLU(|S|^0.3 - |Y|^0.3)^2)` (STFT 512 / 128) | 0.3 | present, stage B on |
| Heads | BCE(p_target, target VAD) + BCE(p_self, own VAD) at 4 ms resolution | 0.1 each | all |
| Speaker ID | AAM-softmax on g_t, margin 0.2, scale 30, 3,758 classes | 0.05 | Person examples |

MR-STFT keeps weight 1 (DECISIONS L 2026-11-08, Loss A/B outcome). After the spec-loss pilot failed its old gate, a
paired A/B from scratch (same data and seed, fixed held-out set) tried two other losses - tsnr_only (MR-STFT at
weight 0) and comp03 (a compressed spectral loss). At step 4,000 both were about 0.2 dB SI-SNRi ahead of the spec
loss, but the acceptance test at step 6,000 tied (tsnr_only against the spec-loss pilot, pooled +0.06 dB, 95%
bootstrap CI -0.06 to +0.13), so the spec loss stays (measured).

Labels - `hop_vad` (10-ml 8.4) at 4 ms: 16 ms energy ending at each hop, active if within 35 dB of the clip maximum and
above 1e-9, dilated by 4 hops; the Everyone target VAD is the OR over talkers 0-3. Every term is selected per example
with `torch.where(applies, term, 0)` (never multiplied by a 0/1 flag), masked means divide by `max(count, 1)`, and the
AAM loss runs only on Person rows, so a batch with Everyone, no-own and absent examples has a finite loss and finite
gradients (10-ml 3.3 and 9).

### 8.3 Optimizer and schedule

AdamW, peak LR 5e-4, betas (0.9, 0.999), weight decay 1e-4 (none on norms, biases, embeddings, nulls), linear warm-up
2,000 steps, cosine decay to 2.5e-5 at step 120,000, gradient clip 5.0, EMA decay 0.999 from step 2,000 (exported).
`PYTORCH_ENABLE_MPS_FALLBACK=0`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8`,
`caffeinate -dimsu`, AC power, DataLoader 6 spawn workers with persistent workers and per-worker numpy generators.

### 8.4 Stages and wall clock

Planning step time - **1.101 s/step**, the measured first hour of the main run itself (`events.log` 2026-11-08 20:52,
mean over 3,232 steps at 8 x 4 s with data loading; train.csv 50-step medians 1.10 to 1.14 s) **x 1.05** for
validation and checkpoints = **1.156 s per step**. Hours = steps x 1.156 / 3600; that is compute time, and the wall
time is longer while the run yields the Mac to the owner's other builds. Stage C (4 x 8 s) is planned at the same
step time, as the merged_full benchmark showed (1.03 s at 4 x 8 s, 1.025 to 1.086 s at 8 x 4 s). History - the first
plan used 1.05 s/step (merged_full, a benchmark without data loading, bench/merged/bench_quiet.jsonl); an A/B run
measured 1.262 s/step while the Mac was loaded and briefly cut the plan to 108,000 steps, undone once the main run's
own first hour was measured (DECISIONS L 2026-11-08).

| Stage | Steps | Batch x segment | Recipe | Gate to pass | Wall clock |
|---|---|---|---|---|---|
| Smoke | 1,500 max | 8 x 4 s, 16 fixed examples | no augmentation randomness | training SI-SNRi on the 16 examples >= 20 dB; tests in 11 pass | 0.5 h |
| Pilot (= main steps 0-6k) | 6,000 | 8 x 4 s | stage A | val SI-SNRi median at 6k above both 4k and 2k (still rising at the end); no NaN skips. No absolute dB target; step time is reported, not checked (DECISIONS L 2026-11-08) | 1.9 h |
| Main A (incl. pilot) | 0-20,000 | 8 x 4 s | SNR N(8,4) in [0, 25], <= 2 talkers, SIR -3..12, no absent, no Everyone, no own talker (own enrollment given) | val rising | 6.4 h |
| Main B | 20,000-105,000 | 8 x 4 s | full recipe 8.1, over-suppression on | - | 27.3 h |
| Main C | 105,000-120,000 | 4 x 8 s | full recipe, real RIRs 30%, LR continues the cosine | - | 4.8 h |
| **Main total** | **120,000** | | about 0.9 M examples, about 1,070 h of mixtures | section 9 | **38.5 h** |
| Fine-tune (optional) | 10,000 | 4 x 8 s | 10% phone rig recordings, real RIRs 30%, LR 1e-4 cosine to 1e-5 | rig rows of 9 improve | 3.2 h |

- The first hour of main decides. If it averages more than 1.25 s/step, `events.log` warns and the lead cuts stage B
  to the total that fits 40 h (`main_steps_for_40h` in train.py). The model is never shrunk. The measured first hour
  was 1.101 s/step, so the rule did not trigger and the plan is the full 120,000 steps (DECISIONS L 2026-11-08).
- **Checkpoints** every 1,000 steps (about 19 min) to `ml/runs/<run>/step_<n>.pt` (written to a temp file, then
  renamed) holding model, EMA, optimizer, schedule step, AAM head and sampler seed; keep the last 3 and the best 2 (about
  50 MB each). `ml/tools/run_forever.sh` restarts `python -m ml.train --resume last` after any crash under caffeinate.
  Because examples are seeded by (seed, index), a resumed run sees exactly the data it would have seen.
- **Validation** every 2,000 steps on 400 fixed mixtures (`ml/tools/render_val.py`, seed 2026, speech_val + noise_val
  + sim_val), offline forward on MPS (offline equals streaming on CPU, and MPS equals CPU offline within 1e-4, both by
  test): SI-SNRi, picked rate at SIR 0 dB, absent
  attenuation, own-voice suppression. Best = highest SI-SNRi among checkpoints with absent attenuation >= 20 dB and
  own-voice suppression >= 20 dB.

---

## 9. Acceptance targets (targets, not results)

Model, offline = streaming, on the test sets of WP-ML (`python -m ml.evaluate`, report in docs/eval/report-v1.md).
"Restaurant sim" = 2,000 mixtures from held-out speakers (VCTK test 8, LibriSpeech test-clean 40), noise_test and WHAM!
test noise (eval only), real_test and sim_test RIRs, grid over SNR (-5, 0, 5, 10 dB), SIR (-8, -5, 0, 5, 10 dB),
1-3 talkers, own voice on/off, room (sim_test in hand, sim_test on table, real_test rows with category bar, restaurant
or indoor; real_test has no hand/table geometry), enrollment clean / noisy 10 dB / noisy 0 dB: 4 x 5 x 3 x 2 x 3 x 3 =
1,080 cells; mixtures 0 to 1,079 take one cell each, the other 920 take random cells.

| Metric | Condition | Pass (blocks release) | Goal |
|---|---|---|---|
| SI-SNRi median | restaurant sim, all cells | >= 6 dB | >= 9 dB |
| SI-SNRi median | 2-talker + noise, SNR 0-5 dB, noisy enrollment | >= 7 dB | >= 9 dB |
| SI-SNRi median | Libri2Mix test clean (data/eval/libri2mix_test.jsonl), enrollment = another utterance of the target, fixed seed | >= 10 dB | >= 12 dB |
| Picked rate (per 2 s chunk, output closer to target than to every talker) | SIR 0 dB / -5 dB | >= 90% / >= 75% | 95% / 85% |
| Target confusion rate (= 1 - picked) | SIR 0 dB | <= 10% | <= 5% |
| Target-absent output energy (attenuation vs mixture) | absent clips | median >= 25 dB; <= 5% of clips under 10 dB | 33 dB |
| Own-voice suppression | own-voice-only frames, own voice enrolled | median >= 25 dB, 5th percentile >= 18 dB (95% of clips at or above 18 dB; report column `own_db_p05`) | 30 dB |
| Target over-suppression (TSOS) | target-active frames | <= 3% | <= 1.5% |
| ESTOI gain (pystoi, MIT) | SNR 0-5 dB | >= +0.10 | +0.15 |
| PESQ wideband (`pesq` package, internal use only, reported not gated) | SNR >= 5 dB | reported | - |
| Accents | EdAcc test, VoxPopuli accented as targets | SI-SNRi within 1.5 dB of the all-cells median | within 0.5 dB |
| Stability | 60 min stream | no NaN; last-5-min SI-SNRi within 1 dB of the first 5 min | - |
| DiPCo real dinner party | far-field channel, enrollment from the target's own close-talk segment elsewhere in the session; targets P09 (S03), P25, P26, P27 (S08), the only close-talk audio on disk | absent-target segments (target silent, others talking) attenuated >= 15 dB median over all four; output envelope correlates best with the target's close-talk envelope in >= 75% of 2 s S08 windows where the target talks and another close-talk participant talks >= 0.5 s; owner listens to 10 clips | 20 dB / 85% |
| Parity | engine vs PyTorch, section 3.3 | all tolerances | - |

Phone (iPhone 16 Pro, AirPods Pro 2 unless stated)

| Metric | Pass | Label |
|---|---|---|
| Per-hop compute, 1 instance, worker in the workgroup, 60 min screen locked | p99 <= 1.0 ms, p99.9 <= 2.0 ms (50% of a hop), max logged | target |
| Same with 3 instances (Table) | p99.9 <= 2.0 ms per hop | target |
| Large buffer | 1,104-frame callback (Sound Recognition on), 3 instances, 30 min screen locked: per-hop p99.9 <= 2.0 ms, 0 underruns, worker policy of 4.2 accepted | target |
| Underruns | 0 in a 60 min locked session with 1 instance; <= 1 with 3 instances | target |
| Route and call robustness | 10 route changes, 2 calls, 1 Siri, 1 alarm in a session: no crash, recovery with at most one tap | target |
| Battery drain | <= 5% per hour (1 instance, screen off, 2 h); 3 to 6% per hour expected | target; expectation is an estimate |
| Thermal | `thermalState` never above `.fair` after 60 min in a pocket at room temperature (1 instance) | target |
| Warm instances | Focus with 3 warm instances, screen off, 60 min: battery <= 8% per hour and `thermalState` never above `.fair`; if missed, Focus keeps only the selected and the last selected person warm | target |
| Latency | measured loop + Earshot share shown; Earshot share <= 11.5 ms at 48 kHz with a 64-frame grant; on other grants and on 44.1 kHz routes `es_engine_latency_ms` is shown and logged (expected 12.3 ms at 48 kHz / 128 frames, 11.3 to 15.2 ms at 44.1 kHz, analytic) but not gated | target |
| Safety | output never above the ceiling in fuzz; fade-in on every start; mute within 50 ms of a bad route | target |
| Daily use | 10 real sessions of >= 20 min: zero own-voice-delay reports, look-select picks the intended person in >= 90% of double presses, owner rating >= 4/5 | target |

---

## 10. Latency budgets (mouth to ear)

Earshot share = decimator + model framing + slack + interpolator = 0.13 + 8.00 + 2.69 + 0.13 = **10.95 ms** (analytic,
48 kHz, 64-frame grant) over a plain pass-through on the same route. The 8.00 ms framing includes the hop wait and the
one-callback hand-off only where P_min = 64 (20-engine 9.4 exact count simulation: 48 kHz with 16 to 96 or 192 frames);
a 128-frame grant adds 1.4 ms and 44.1 kHz routes add 0.3 to 4.3 ms, and the app shows whatever
`es_engine_latency_ms` returns.

### 10.1 AirPods Pro 2 over A2DP (primary)

| Line | ms | Source |
|---|---|---|
| Mic, ADC, iOS input path | 2.4 | benchmark (reported inputLatency, iPhone 12 mini, ios-platform.md 2.1); measured inside the loop on device |
| Input I/O buffer, 64 frames at 48 kHz | 1.33 | doc arithmetic; granted value read back |
| Decimator (minimum phase) | 0.13 | analytic, mean group delay 1 to 4 kHz (4.2, 20-engine 9.1) |
| Model framing (window + hop wait + hand-off) | 8.00 | analytic, simulated (product.md B.3, quality.md A.5) |
| Deadline slack | 2.69 | design (slack16 = 43) |
| Interpolator | 0.13 | analytic |
| Output I/O buffer | 1.33 | doc arithmetic |
| **App side** | **16.0** | sum |
| Bluetooth A2DP (encode, radio, AirPods buffer, decode) | about 43 (range 40 to 220) | benchmark - Pro 2 tap-to-sound 126 ms vs built-in speaker 83 ms (fact-check.md 2); never measured on this path |
| **Total** | **about 59 (56 to 236)** | estimate until measured in the app |

### 10.2 AirPods Pro 1 over A2DP (primary)

Same app side **16.0 ms**. Bluetooth about **84 ms** (benchmark - 167 vs 83 ms tap-to-sound, fact-check.md 2; range 60 to
220). **Total about 100 ms** (estimate). Pro 1 also isolates less (about 21 dB vs 24.5 dB ANC, benchmark), so the app
recommends Pro 2 for long conversations.

### 10.3 Wired USB-C (later, not built in v1)

| Line | v1 model on wired (ms) | Later LL profile (ms) | Source |
|---|---|---|---|
| Hardware round trip X + 2B (64 / 32 frames) | 17 (9 to 27) | 15.7 (7.7 to 25.7) | benchmark - 9 iPhone 15/16 crowd rows, median 17 ms (fact-check.md 1); not USB-C |
| Resamplers | 0.26 | 0.26 | analytic (48 kHz) |
| Framing | 8.00 | 4.00 (16 ms / 4 ms windows, 2 ms hop) | analytic |
| Slack | 1.38 (slack16 22) | 0.69 | design |
| **Total** | **26.6 (18.6 to 36.6)** | **20.6 (12.6 to 30.6)** | estimate |

Under 20 ms on wired is possible only if the phone's measured X is small; it is never promised. The LL profile needs its
own model (about 35 to 40 h of training, estimate) and is decided only after Phase 0 measures X and only if the owner
gets sealed wired earbuds. Nothing in v1 depends on it.

---

## 11. Work packages

Three parallel engineers plus the lead. Each owns only its directory (section 7). Shared artifacts flow through
`ml/exports/` (read by the others, written only by WP-ML) and `docs/measurements/` (written by WP-IOS).

### WP-ML (ml/)

- **Inputs** - this contract, data pack and manifests, RIR banks, `models/wespeaker-voxceleb-resnet34/avg_model.pt`.
- **ML-0, first and blocking for parity (do before anything else)** - `model.py` (offline), `stream_ref.py` (per-hop
  reference with the exact state of 2.6), `vad.py`, `fileio.py` (container 3.1), `export.py --random` and `--testvec`,
  `convert_wespeaker.py`, `quality.py`, and the 7 ML-0 test files (section 7). Outputs
  `ml/exports/random_v1/{earshot.esw, testvec_short.est, testvec_long.est, thresholds.json, wespk_selftest.est,
  quality_vectors.json}` and `ml/exports/wespeaker_resnet34_fp32.mlpackage`.
- **ML-1** - data loader (8.1), losses (8.2), train loop with resume (8.3), `render_val.py`, smoke and pilot runs.
- **ML-2** - main run (about 38.5 h of compute), `evaluate.py` over section 9, `calibrate_thresholds.py`, final export
  `ml/exports/v1/` (same file names as random_v1), docs/eval/report-v1.md.
- **Tests that prove done** - pytest: window perfect reconstruction (1e-6); streaming identity y_stream[n+64] =
  y_offline[n] (1e-5, CPU fp32) on 10 s; MPS vs CPU offline forward on random_v1, 4 s, max abs diff <= 1e-4 (skipped
  without MPS); causality - perturbing x[t:] leaves y_offline[: 64 floor(t/64) - 64] and y_stream[: 64 floor(t/64)]
  unchanged (max diff <= 1e-6) and changes the samples right after (> 1e-3), for t in {6400, 6437, 32000};
  VAD rule on fixed vectors; export round trip (write, read, compare, hash); the `meta.tolerances` key set equals
  exactly the 3.3 list; quality_vectors.json against `quality.py` (section 6); a batch holding one Everyone, one
  no-own and one absent example gives a finite loss and finite gradients for every parameter; data loader determinism per (seed, index)
  and license filter; smoke and pilot gates of 8.4; Core ML vs PyTorch WeSpeaker cosine >= 0.9999 on CPU_AND_GPU.
- **Depends on** - the lead's day-0 commit of `ml/assets/decim16k_fir.npy` before the ML-1 pilot (a missing file is
  allowed only for smoke and unit tests); nothing else.

### WP-ENGINE (engine/)

- **Inputs** - this contract (header verbatim), `ml/exports/random_v1/` (from ML-0).
- **E-0, start immediately (no weights needed)** - windows, FFT wrapper (vDSP_fft_zrip; forward scaled by 0.5, inverse
  by 1/512 to match torch, checked by test), LSTM and BiLSTM kernels against naive C, rings, resampler tables and
  polyphase resampler, ramps, limiter, soft clip, VAD, json_min, container reader.
- **E-1** - `es_model_load`, `es_stream_hop`, `es_enroll`, `es_cond_build`, `es_embed_g`, `es_selftest`; parity
  against random_v1 test vectors.
- **E-2** - `es_engine_*` pipeline, worker thread (pthread, time-constraint policy, workgroup join, FTZ, single-threaded
  BLAS), post-processing, safety chain, cues, probe, stats, timings; `esrun` offline tool.
- **E-3** - bench on the M4 (1 and 3 instances), `make ios-check` (compile all sources for arm64-apple-ios26.0).
- **Tests that prove done** - every tolerance of 3.3 on random_v1 and later v1; window and FFT round trip 1e-6;
  resampler passband, stopband and delay - 48 kHz tables equal the 4.2 numbers and 44.1 kHz tables the figures the
  table tool prints (ES_RS_DELAY_MS 0.2987), within 0.01 dB / 1 dB / 0.01 ms; malloc interposer
  counts 0 allocations and 0 frees inside render and run_hops over 100,000 hops; fuzz - random callback sizes 16 to
  1,104 frames, NaN / Inf / full-scale / silence inputs for 1 simulated hour, output always finite and <= -3 dBFS plus
  soft-clip margin; latency test (impulse delay = `es_engine_latency_ms` within 1 sample, in bypass and through the
  synthetic all-pass model); alignment - the synthetic all-pass model's output equals x_d within 1e-6 every hop after
  the second; gain cap - a synthetic model with every coefficient 1.5 + 1.5j and the AGC at its maximum keeps the 50 ms
  output RMS <= input RMS + 24.1 dB; slack-step zeros get the same 1 ms ramps as an underrun; set_cond BUSY protocol
  under a racing writer thread; bench median per hop <= 0.25 ms for 1 instance on an M4 P-core (target).
- **Depends on** - ML-0 for E-1 parity only.

### WP-IOS (ios/)

- **Inputs** - this contract, `Stubs/earshot_stub.c` until E-2 lands, the mlpackage and random_v1 files from ML-0.
- **I-0, start immediately** - XcodeGen project, app shell, AVAudioSession per 5.3, `AudioHost.c` RemoteIO with the
  stub engine (pass-through), route and interruption handling, diagnostics screen.
- **I-1 Phase 0 on the phone** (writes `docs/measurements/phase0-YYYY-MM-DD.json`) - granted buffers 32 to 192 frames
  and session rates on AirPods Pro 1, Pro 2 and built-in mic; tap test that the built-in mic is really used with A2DP
  output; loop latency with the probe (in-ear leak method and ear-detection-off method) at 0, 10, 30 min for Pro 1 and
  Pro 2; motion update rate and gaps with the screen on and locked; stem single/double/triple delivery with
  Conversation Awareness and Spatial Audio on and off; engine compute histogram (random_v1, 1 and 3 instances, 60 min
  locked, plus 1,104 frames with 3 instances for 30 min); battery and thermal over 2 h with 1 instance and 60 min with
  3 warm instances; mic level in dBFS for speech at about 1 m; probe methods A and B success rate at about 45 dB and
  about 70 dB (played babble) ambient for Pro 1 and Pro 2, at -30, -24 and -18 dBFS (picks `ES_PROBE_DBFS`); with the
  screen locked, 10 route changes (AirPods out and in, case open and close) and 1 forced engine swap, recording whether
  audio resumes without a tap; with the screen locked, embed a 10 s clip and run the alarm classifier for 60 s,
  recording success, time and compute units.
- **I-2** - SpeakerEncoder with self-test, EnrollmentService with the 6 checks, own-voice onboarding, PeopleStore,
  quick add.
- **I-3** - listening flows: modes, Table, Selector with look-select and learned directions, stem commands and Now
  Playing, cues, latency measurement and display, settings.
- **I-4** - safety and lifecycle polish, alarm passthrough, thermal downshift, telemetry, onboarding (what it is and is
  not, credits to Look Once to Hear, WeSpeaker personal-use note, AirPods checklist - ANC on, Conversation Awareness
  off, Personalized Volume off, Spatial Audio off for Earshot, Connect "When Last Connected", Sound Recognition off,
  Reduce Loud Audio on).
- **Tests that prove done** - XCTest for Selector math, quality-check thresholds read from thresholds.json, the check
  formulas of section 6 against quality_vectors.json, person.json round trip (with `mask`) and expiry, latency display
  strings, every launch failure branch with the stub (30-ios 9.4), InputTap starting at the current position; on device - the phone rows of section 9 and the Phase 0 file.
- **Depends on** - E-1 for real enrollment, E-2 for real audio, ML-0 for the encoder and weights.

### Lead integration

Day 0 - run `.venv/bin/python engine/tools/make_resampler_tables.py` (scipy 1.18.1) as soon as WP-ENGINE delivers it
(E-0 day 1) and commit both outputs, `engine/src/resampler_tables.c` and `ml/assets/decim16k_fir.npy`; only the lead
regenerates them, and any re-run gets a DECISIONS row. Debug build only - install a counting malloc zone or
`malloc_logger` hook armed for the worker and render thread IDs; the 60 min locked compute run must log 0 allocations
on those threads (DYLD interposition is impossible on the phone). Copy exports into `ios/Earshot/Resources/Models/`
(`make models`), remove the stub, run the app with random_v1 then v1,
check `es_selftest` on device, run the phone acceptance rows, review every RT path against 4.1, keep docs/DECISIONS.md,
and show the owner this summary before the 38.5 h main run starts.

---

## 12. Out of scope for v1, and top risks

Out of scope (later work, nothing in v1 depends on it) - wired low-latency model; camera lip-gated enrollment;
automatic follow-my-gaze switching; automatic voiceprint refresh during use; a single multi-target network; 24 kHz;
two-mic input; Neural Engine context booster; BNNS Graph backend; a smaller thermal-fallback model; spoken name
confirmations; Control Center / Action button widget; HealthKit exposure view; DNSMOS gating; Game Mode experiments;
any distributed build (it would need the no-WeSpeaker path, legal review and a patent search).

| # | Risk | Mitigation |
|---|---|---|
| 1 | AirPods delay higher than planned or drifting (Pro 1 worst) | measure at connect and at 10 / 30 min, show it, ANC required, own voice suppressed and ducked, recommend Pro 2 |
| 2 | Target confusion with similar or quieter voices | SIR down to -8 dB, tokens plus WeSpeaker, own voice as negative, quality checks, locked check with banner, Table mode |
| 3 | Simulation does not transfer to the real phone and rooms | real RIRs, mic-chain augmentation with the app's decimator, Phase 0 mic level, DiPCo check, optional rig fine-tune |
| 4 | Shared Mac slows training (first-hour mean above 1.25 s/step, the contract 8.4 line) | measured budget (main run's first hour 1.101 s/step x 1.05, 38.5 h of compute) under 40 h, resumable checkpoints, the run yields to other builds, stage B is the only thing cut |
| 5 | iOS unknowns - measurement mode with A2DP, motion while locked, stem delivery, a 44.1 kHz route | Phase 0 first; fallbacks: default mode, tap/cycle selection, on-screen controls, 44.1 kHz tables |
| 6 | Worker misses deadlines (efficiency core, thermal) | 0.2 ms vs 4 ms budget (estimate), 2.69 ms slack that grows, concealment, workgroup join, downshift to 1 instance |
| 7 | Own voice leaks back late | mandatory own-voice enrollment, own-voice loss weight, p_self duck, 25 dB acceptance row |
| 8 | C engine and PyTorch disagree | one container format, test vectors with debug tensors, launch self-test |
| 9 | WeSpeaker on the phone gives wrong vectors | `.cpuAndGPU` in the foreground, `.cpuOnly` in the background, launch self-test, `.cpuOnly` retry, no-WeSpeaker path trained by dropout |
| 10 | Free Personal Team profiles expire every 7 days | weekly re-install via `make -C ios install` (about 2 min) |
| 11 | Licenses and privacy (VoxCeleb weights, voiceprints are biometric data) | personal build only, no NC data in training, on-device storage, "tonight" default, consent prompt |
| 12 | Hearing safety | +24 dB gain cap (post step 5), limiter, soft clip, fades, runaway guard, Reduce Loud Audio guidance, no medical claims |
