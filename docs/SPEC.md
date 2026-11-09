# Earshot SPEC

Assembled from docs/spec/ (00-contract binds; sections 10/20/30 implement it). Edit the files in docs/spec/, then rebuild from the repo root with `{ head -4 docs/SPEC.md; for f in 00-contract 10-ml 20-engine 30-ios; do cat docs/spec/$f.md; printf '\n\n'; done; } > docs/SPEC.md.new && mv docs/SPEC.md.new docs/SPEC.md`.

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


# Earshot v1 - WP-ML build spec (10-ml)

Date - 2026-11-07. Owner - WP-ML engineer. Status - expands docs/spec/00-contract.md (binding). Every number, name,
shape and signature below agrees with the contract; where this file adds detail the contract leaves open, the detail
is marked **WP-ML choice**. Labels as in the contract - **measured** (file named), **benchmark**, **analytic**,
**estimate**, **target**. Nothing here has been run yet; every time and size below is an estimate unless it cites a file.

Read first - docs/BRIEF.md, docs/spec/00-contract.md (sections 2, 3, 6, 8, 9, 11), data/manifests/SUMMARY.md,
data/rir/README.md, dataprep/common.py (manifest schema), bench/merged/bench_merged.py (the measured prototype; its
`stft`, `istft`, `BandEnc`, `DualPath`, `BandDec` are the starting code, but its token encoder and attention ignore
masks and must be rewritten as in section 3).

---

## 0. What WP-ML delivers

| Milestone | Delivers | Blocks |
|---|---|---|
| ML-0 | `config.py dsp.py model.py stream_ref.py vad.py fileio.py wespeaker.py quality.py export.py`, `tools/convert_wespeaker.py`, the first 7 test files of section 14; files `ml/exports/random_v1/{earshot.esw, testvec_short.est, testvec_long.est, thresholds.json, wespk_selftest.est, quality_vectors.json}` and `ml/exports/wespeaker_resnet34_fp32.mlpackage` | WP-ENGINE E-1 parity, WP-IOS I-2 |
| ML-1 | `sources.py render.py enroll.py data.py losses.py metrics.py train.py`, `tools/cache_own_embeddings.py render_val.py run_forever.sh`, configs, smoke run (gate 8.4), pilot run (gate 8.4) | main run |
| ML-2 | main run (38.5 h of compute planned, contract 8.4), `evaluate.py`, `tools/calibrate_thresholds.py`, `ml/exports/v1/` (same 6 file names), `docs/eval/report-v1.md` | release |

Credits that must appear in `meta` of every export and in the report - the interaction idea is UW's Look Once to Hear
(Veluri et al., CHI 2024; none of its code is used); the WeSpeaker ResNet34 VoxCeleb weights are research terms, so
the build is **personal use only**; WHAM! noise is evaluation only.

---

## 1. Environment and file plan

### 1.1 Environment (fixed)

- Python - `.venv/bin/python` in the repo root (3.12). Installed and pinned - torch 2.7.1,
  torchaudio 2.7.1, coremltools 9.0, numpy 2.5.4, scipy 1.18.1, soundfile 0.14.0, soxr 1.1.0, pyyaml 6.0.3,
  pytest 9.2.0, pyroomacoustics 0.10.1 (checked 2026-11-07 by import). Never upgrade or add packages except the two
  evaluation packages of 12.4.
- Always run from the repo root as modules - `python -m ml.train`, `python -m ml.export`, `python -m ml.evaluate`,
  `python -m ml.tools.render_val`, `python -m pytest ml/tests`. So `ml/__init__.py`, `ml/earshot/__init__.py` and
  `ml/tools/__init__.py` exist (empty). `dataprep` (frozen) is imported read-only for `read_manifest` and `DATA`.
- Environment variables - `train.py`, `evaluate.py` and `render_val.py` set these with `os.environ.setdefault` before
  importing torch, and the shell commands also export them - `PYTORCH_ENABLE_MPS_FALLBACK=0`,
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8`, `VECLIB_MAXIMUM_THREADS=1`,
  `OMP_NUM_THREADS=1` (the last two keep the 6 data workers from oversubscribing the 10 cores).
- Precision - float32 everywhere (MPS has no float64). float64 only inside numpy for window construction, filter
  design and the VAD energy sum (5.1). Losses are computed after `.float()`.
- Devices - training, validation and evaluation forward passes on MPS; streaming reference, export, test vectors and
  all pytest tests on CPU (`torch.set_num_threads(4)` in tests).
- Randomness - data randomness comes only from `np.random.default_rng([seed, ...])` generators created per example
  or per batch (8.8). No `np.random.*` global calls, no `random` module, no torch RNG in the training loop (the model
  has no dropout; WeSpeaker path dropout is drawn in the loader).

### 1.2 Files (sizes are estimates of clear, commented code)

| File | Purpose | Main contents | Lines |
|---|---|---|---|
| `ml/earshot/config.py` | constants and run configs | `CONFIG` (the contract 3.1 object, verbatim), `config_hash()`, `WIDTHS`, `TAPS`, paths (`ROOT`, `DATA`, `RUNS`, `EXPORTS`, `CACHE`), `load_yaml(path)` with `base:` merging | 120 |
| `ml/earshot/dsp.py` | signal helpers shared by everything | `windows()`, `stft()`, `istft()`, `stft_tok()`, `early_rir()`, biquads, `fir_decim()`, `hop_vad()`, `active_power()`, fades | 220 |
| `ml/earshot/model.py` | ESB-16-DF, offline (batch) form | `BandEncoder`, `DualPath`, `Attention`, `Nulls`, `BandDecoder`, `TokenEncoder`, `CondInputs`, `Cond`, `ESB16DF`, `init_train()`, `init_random_for_parity()` | 420 |
| `ml/earshot/stream_ref.py` | per-hop reference with the exact state of contract 2.6 | `pack_cond()`, `StreamRef.reset/hop/run` | 220 |
| `ml/earshot/vad.py` | `es_vad` rule and token grouping | `num_frames()`, `es_vad()`, `token_groups()` | 90 |
| `ml/earshot/wespeaker.py` | the one WeSpeaker definition | `KaldiFbank`, `BasicBlock`, `ResNet34`, `WeSpeaker.load()` | 150 |
| `ml/earshot/sources.py` | manifests, pools, license gate, audio reads | `Sources`, `check_row()`, `read_crop()`, `own_split()`, `build_clip()` | 200 |
| `ml/earshot/render.py` | one training or test example (8.3, 8.4, 8.6) | `Draws` (every per-example scalar, in draw order), `render_example()`, `mic_chain()` | 300 |
| `ml/earshot/enroll.py` | target and own enrollments (8.5) | `render_enrollment()` | 150 |
| `ml/earshot/data.py` | recipes, Dataset, loader (8.2, 8.7, 8.8) | `Recipe`, `RECIPES`, `MixDataset`, `make_loader()`, `worker_init()` | 150 |
| `ml/earshot/quality.py` | the contract 6 check formulas (trim, best 10 s window, 0.5 s rounding, windows, sbr, clip, speech_s) shared by calibration and the quality vectors | `frame_span()`, `best_window()`, `windows()`, `sbr_db()`, `clip_pct()`, `own_window_mask()`, `score()` | 150 |
| `ml/earshot/losses.py` | all loss terms | `tsnr()`, `absent()`, `masked_absent()`, `mrstft()`, `oversup()`, `heads_bce()`, `AAMHead`, `total_loss()` | 180 |
| `ml/earshot/metrics.py` | evaluation metrics | `si_snr()`, `si_snri()`, `attenuation_db()`, `own_suppression_db()`, `picked()`, `tsos()`, `estoi_gain()`, `pesq_wb()`, `env_corr()` | 220 |
| `ml/earshot/fileio.py` | `.esw` / `.est` container | `write_container()`, `read_container()` | 120 |
| `ml/train.py` | training loop | `main()`, `lr_at()`, `build_optimizer()`, `Ema`, `step_batch()`, `validate()`, `save_ckpt()`, `load_ckpt()`, `prune()`, gates | 400 |
| `ml/export.py` | weights, test vectors, thresholds | `to_export()`, `from_export()`, `write_esw()`, `load_esw()`, `make_testvecs()`, `write_wespk_selftest()`, `write_thresholds()` | 320 |
| `ml/evaluate.py` | test suites of contract 9 and the report | suite builders, `run_offline()`, `run_stream()`, `dipco()`, `write_report()` | 480 |
| `ml/configs/*.yaml` | smoke, pilot, main, finetune | section 10.6 | 20-40 each |
| `ml/tools/convert_wespeaker.py` | PyTorch WeSpeaker to Core ML plus check | 6.2 | 120 |
| `ml/tools/cache_own_embeddings.py` | own-voice clip recipes and e_o cache | 6.3 | 130 |
| `ml/tools/calibrate_thresholds.py` | enrollment check thresholds | section 13 | 220 |
| `ml/tools/render_val.py` | 400 fixed validation mixtures | section 11 | 150 |
| `ml/tools/run_forever.sh` | crash-restart wrapper | 10.7 | 30 |
| `ml/requirements-eval.txt` | `pystoi==0.4.1`, `pesq==0.0.4` | 12.4 | 2 |
| `ml/tests/*.py` | pytest suite | section 14 | 60-200 each |

Generated, gitignored - `ml/runs/<run>/`, `ml/exports/`, `ml/cache/` (own-voice cache, validation set, evaluation
renders). `ml/assets/decim16k_fir.npy` is committed: WP-ENGINE writes `engine/tools/make_resampler_tables.py`, and the
lead runs it on day 0 and commits its two outputs (contract 11); see 8.6 for what happens if it is missing.

---

## 2. config.py and dsp.py

### 2.1 config.py

- `CONFIG` - a dict literal identical to contract 3.1 (29 keys). `config_hash()` returns
  `sha256(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest()`. Computed for this spec
  with Python 3 json (analytic) - **`089d72c06d244e1650d209453c91ddffde990b9589dd7a9c228f9e1ea3b93aec`**. A test pins it.
- `WIDTHS = [8]*8 + [16]*4 + [32]*3 + [33]`, `OFFSETS` (cumulative bin start per band), `TAPS = [3]*12 + [1]*4`,
  `COEF_LEN = [2*W*T for W, T in zip(WIDTHS, TAPS)]` (sum 1,026).
- `load_yaml(path)` - reads YAML; if it has `base: other.yaml`, loads that first (same folder) and overlays this
  file's keys (dicts merge recursively, lists replace). 20 lines.

### 2.2 dsp.py (torch functions accept [B, L] float32 on any device; numpy functions work on 1-D float32)

| Function | Exact behavior |
|---|---|
| `windows()` | `(wa, ws)` float32 [512] from contract 2.1, computed in float64 (`hann(L) = 0.5 - 0.5 cos(2 pi n / L)`, periodic) |
| `stft(x)` | requires `L % 64 == 0`; `F.pad(x, (448, 0)).unfold(-1, 512, 64) * wa`, `torch.fft.rfft` -> [B, T, 257] complex64, T = L / 64; frame t ends at sample 64t + 63 |
| `istft(Y, L)` | `irfft(Y, n=512) * ws`, keep samples 384..511 of each frame, `F.fold` at stride 64, output `[:, 64 : 64 + L]` (bench_merged.py `istft`, unchanged); frame t covers output samples 64t - 64 .. 64t + 63 |
| `stft_tok(x)` | token frames - `F.pad(x, (448, 0)).unfold(-1, 512, 512) * wa`, rfft -> [B, F, 257], F = `vad.num_frames(L)` |
| `early_rir(h, dp)` | `end = dp + 800`; `e = h[:end + 80].copy()`; `e[end:end + 80] *= 0.5 (1 + cos(linspace(0, pi, 80)))` (data/rir/README.md recipe; 50 ms plus 5 ms fade, delay kept) |
| `sos_highpass(fc)` | `scipy.signal.butter(2, fc, "highpass", fs=16000, output="sos")` |
| `sos_peaking(f0, gain_db, q)` | RBJ cookbook peaking biquad as one sos row |
| `sos_high_shelf(f0, gain_db)` | RBJ cookbook high shelf, slope 1 |
| `apply_sos(sos, x)` | `scipy.signal.sosfilt` (causal, so mixture and references stay aligned) |
| `fir_decim(x)` | `np.convolve(x, h64)[:len(x)]` with `h64 = np.load("ml/assets/decim16k_fir.npy")` (64 taps) |
| `hop_vad(s)` | activity per hop of a clean signal (8.4) -> uint8 [L / 64] |
| `active_power(x, vad)` | mean of x^2 over the samples of hops with vad = 1; mean over all samples if no hop is active |
| `fade(x, n)` | half-cosine fade-in and fade-out of n samples at the ends |

Perfect reconstruction - `istft(stft(x), L)[: L - 64]` equals `x[: L - 64]` within 1e-6 (the last 64 samples lack the
next frame). The streaming form is in section 4.

---

## 3. model.py (ESB-16-DF, contract 2.2 to 2.4)

### 3.1 Module tree (state_dict names are the export names before the LSTM rename of 7.2)

```
ESB16DF
  enc        BandEncoder   enc.lin.{k} = Linear(3*W_k, 64), enc.ln.{k} = LayerNorm(64)        k = 0..15
  blocks     ModuleList(4) DualPath(Ht=128, Hb=64, time_bidir=False)
               tn LayerNorm(64) | t LSTM(64, 128, batch_first) | tf Linear(128, 64)
               bn LayerNorm(64) | b LSTM(64, 64, batch_first, bidirectional) | bf Linear(128, 64)
  att        Attention     q Linear(1024, 64), o Linear(64, 1024), kv Linear(64, 128),
                           type_emb Parameter[3, 64], null_tok Parameter[4, 64]
  film       Sequential(Linear(770, 256), SiLU(), Linear(256, 4096))      -> names film.0.*, film.2.*
  null       Nulls         e_t[256], g_t[128], e_o[256], g_o[128]  (Parameters)
  dec        ModuleList(16) BandDecoder  ln LayerNorm(64), fc1 Linear(64, 128), fc2 Linear(128, 2*W_k*T_k)
  heads      Linear(64, 2)
  tok        TokenEncoder  blk DualPath(Ht=32, Hb=32, time_bidir=True)  (tf, bf are Linear(64, 64))
                           proj Linear(1024, 64), g Linear(64, 128)
```

`TokenEncoder` does not own a band encoder; `ESB16DF.encode_enrollment` runs the shared `enc` and passes z in, so no
`tok.enc.*` keys exist. All LayerNorms use eps 1e-5. The training-only `AAMHead` lives in losses.py, not in the model.

Parameter count (PyTorch counts two LSTM bias vectors) - **2,575,108** total; enc 52,416, blocks 730,624, att q+o
132,160, dec 267,522, heads 130, film 1,250,048, kv 8,320, type_emb 192, null_tok 256, nulls 768, tok 132,672
(analytic, equals contract 2.5). A test checks every part.

### 3.2 Data classes

```python
@dataclass
class CondInputs:                 # all tensors on one device, batch B
    e_t: Tensor   # [B, 256]  WeSpeaker of the target enrollment (any value when has_t is False)
    tok_t: Tensor # [B, 48, 64] from encode_enrollment;  val_t: Tensor  # [B, 48] bool
    g_t: Tensor   # [B, 128]
    e_o: Tensor   # [B, 256];  tok_o: Tensor  # [B, 16, 64];  val_o: Tensor  # [B, 16] bool;  g_o: Tensor  # [B, 128]
    has_t: Tensor # [B] bool  False = Everyone mode
    has_o: Tensor # [B] bool  False = no own voice enrolled
    use_wes: Tensor  # [B] bool  False = WeSpeaker path dropped
    mode: Tensor  # [B, 2] float  [1, 0] Person, [0, 1] Everyone

@dataclass
class Cond:
    film: Tensor  # [B, 2, 2, 16, 64]  (point p, gamma/beta s, band k, d) = contract index ((p*2+s)*16+k)*64+d
    k: Tensor     # [B, 4, 68, 16]
    v: Tensor     # [B, 4, 68, 16]
    valid: Tensor # [B, 68] bool  token order: 48 target slots, 16 own slots, 4 null
```

### 3.3 Forward functions

- `features(X)` - `mag = X.abs()`; `c = X * (mag + 1e-8) ** -0.7`; stack `[c.real, c.imag, 0.1 * log(mag**2 + 1e-8)]`
  on the last axis -> [B, T, 257, 3]. Band k input is `f[:, :, o_k : o_k + W_k].reshape(B, T, 3 W_k)` (index 3j + f).
  The mixture and enrollment never require grad, so the non-differentiable point of `abs` at 0 is never hit.
- `encode(X)` - per band `ln_k(lin_k(...))`, stacked -> z [B, T, 16, 64].
- `DualPath.forward(z)` - exactly bench_merged.py `DualPath` - time path over T per band (B*16 sequences), then band
  path over the 16 bands per frame (B*T sequences, zero initial state), each with pre-LayerNorm, Linear and residual.
- `encode_enrollment(wav, idx, valid)` - wav [B, N], idx [B, M, 3] int64 frame indices, valid [B, M] bool
  (M = 48 or 16, both from `vad.token_groups`, built by the loader). Steps - `stft_tok`, `features`, `enc`,
  `tok.blk` (time BiLSTM over all F frames, then band BiLSTM) -> z [B, F, 1024] (flatten k*64 + d); gather the 3
  frames of each group and average -> [B, M, 1024]; `tok = tok.proj(...)` -> [B, M, 64]; set invalid rows to 0;
  `g = normalize(tok.g(masked_mean(tok, valid)), dim=-1)` with
  `masked_mean(tok, valid) = (tok * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)`, so an example
  with no valid rows (Everyone target, no own voice) gets a finite g instead of 0 / 0 = NaN. Returns `(tok, g)`.
- `build_cond(ci)` - the null rules of contract 2.3 with `torch.where`, per example -
  `e_t' = e_t if (has_t and use_wes) else null.e_t`; `g_t' = g_t if has_t else null.g_t` (every input to `torch.where`
  is finite, because `torch.where` passes NaN gradients from the unused branch);
  `e_o' = e_o if (has_o and use_wes) else null.e_o`; `g_o' = g_o if has_o else null.g_o`;
  `u = cat([e_t', g_t', e_o', g_o', mode])` (770) -> `film(u).reshape(B, 2, 2, 16, 64)`.
  Tokens `cat([tok_t + type_emb[0], tok_o + type_emb[1], null_tok + type_emb[2]], 1)` [B, 68, 64];
  `valid = cat([val_t & has_t, val_o & has_o, ones(4)])`; `kv = att.kv(tokens)`; K = `kv[..., :64]`,
  V = `kv[..., 64:]`, each `.reshape(B, 68, 4, 16).transpose(1, 2)`.
- `forward(mix, cond)` -> `(y [B, L], head_logits [B, T, 2])`, in this order.
  1. `X = stft(mix)`; `z = encode(X)`.
  2. For b in 0..3 - `z = blocks[b](z)`; if b == 0 - FiLM 0 `z = z * (1 + film[:, 0, 0, None]) + film[:, 0, 1, None]`
     then attention (below); if b == 2 - FiLM 1 with `film[:, 1]`.
  3. Attention - `q = att.q(z.reshape(B, T, 1024)).reshape(B, T, 4, 16)`;
     `s = einsum("bthd,bhnd->bthn", q, K) * 0.25`; `s.masked_fill_(~valid[:, None, None, :], -inf)`;
     `p = softmax(s, -1)`; `a = einsum("bthn,bhnd->bthd", p, V).reshape(B, T, 64)`;
     `z = z + att.o(a).reshape(B, T, 16, 64)`. Written out by hand (no `scaled_dot_product_attention`) so the C code
     can be read against it. Null tokens are always valid, so no row is fully masked.
  4. `head_logits = heads(z.mean(2))` (sigmoid is applied by the loss and by the streaming reference).
  5. Decoder per band - `c = 1.5 * tanh(fc2(tanh(fc1(ln(z[:, :, k])))))`; real part `c[..., :W T]`, imaginary
     `c[..., W T:]`, each `.reshape(B, T, T_k, W_k)` (index tau*W_k + j).
  6. Deep filter - `Y_k = sum_tau C_tau * shift(X_k, tau)`, `shift(X, tau)[t] = X[t - tau]`, zero for t < tau.
  7. `y = istft(cat(Y_k), L)`.
- `embed_g(pcm)` (CPU helper used by calibration and the locked check) - `es_vad`, `token_groups(mask, 48)`,
  `encode_enrollment`, return g. Raises if fewer than 4 tokens (the engine's `ES_ERR_ENROLL_SHORT`).

### 3.4 Initialization

- `init_train(model)` - PyTorch defaults, then for every LSTM direction `bias_ih[H:2H] = 1`, `bias_hh[H:2H] = 0`
  (exported sum = +1); zeros for `att.o.weight`, `att.o.bias`, `att.type_emb`, `att.null_tok`, `null.*`.
- `init_random_for_parity(seed)` - returns a new model - `torch.manual_seed(seed)`, construct `ESB16DF()` (PyTorch
  default init), forget bias as above, then fill `att.o.*`, `att.type_emb`, `att.null_tok`, `null.*` with N(0, 0.1^2) in that order (contract 3.2).
  Used by `export --random` and by every parity test.

---

## 4. stream_ref.py (per-hop reference, CPU fp32)

`pack_cond(cond, i)` turns example i of a batched `Cond` into the engine layout of contract 2.6 - `film` [4096]
(flatten of [2, 2, 16, 64]), `k`, `v` [4, 68, 16] with the valid tokens moved to the front in order (target, own,
null) and zeros after them, `n` = number of valid tokens. The C engine builds exactly this from `es_cond_build`.

`StreamRef(model, allpass=False)` holds the model (eval, CPU) and the state of contract 2.6, all float32 -
`ana` [512], `xhist` [2, 128] complex (index 0 = t-1), `h`, `c` [4, 16, 128], `ola` [128]. `reset()` zeroes them.
`hop(pc, x64) -> (y64, heads2)` does the following, in this order.

1. `ana = cat(ana[64:], x64)`; `X = rfft(ana * wa)` (257 bins).
2. `f = features(X)` -> 771 floats (index 3*bin + f); `z = encode(X)` -> [16, 64].
3. For b in 0..3 - time path `out, (hb, cb) = blocks[b].t(tn(z)[:, None, :], (h[b][None], c[b][None]))`, store
   `h[b], c[b]`, `z += tf(out[:, 0])`; band path `z += bf(blocks[b].b(bn(z)[None])[0][0])`; then FiLM 0 and
   attention after b = 0 (attention over the first n tokens of the packed K, V), FiLM 1 after b = 2.
4. `heads2 = sigmoid(heads(z.mean(0)))`.
5. Decoder -> coefficient vector [1026] (concatenation over bands of each band's 2 W_k T_k outputs).
6. Deep filter with X, `xhist[0]`, `xhist[1]` for bands 0-11 (bins 0-127); then `xhist[1] = xhist[0]`,
   `xhist[0] = X[:128]`.
7. `fr = irfft(Y, n=512) * ws`; `ola += fr[384:]`; `y64 = ola[:64].copy()`; `ola[:64] = ola[64:]`; `ola[64:] = 0`.

With `allpass=True` steps 2 to 6 are replaced by `Y = X`. `run(pc, x)` loops `hop` over `len(x) // 64` hops and
returns `y` [N] and `heads` [N/64, 2]. With `debug=8` it also returns, for the first 8 hops, `feat` [8, 771],
`z_enc` [8, 16, 64] (after the band encoder), `z_blk{b}` [8, 16, 64] (z after everything that belongs to block b,
that is after FiLM 0 and attention for b = 0 and after FiLM 1 for b = 2) and `coef` [8, 1026].

Identity (contract 2.1) - `y_stream[n + 64] == y_offline[n]` within 1e-5 on CPU fp32, and
`heads_stream[t] == sigmoid(head_logits[t])` within 1e-5. Speed of the reference (estimate) - 1 to 2 ms per hop in
Python, so 10 s takes about 5 s and 60 min about 20 to 30 min.

---

## 5. vad.py (bit-identical to the engine's `es_vad`)

### 5.1 `num_frames(n)` and `es_vad(pcm)`

- `num_frames(n)` - `0 if n < 576 else (n - 64) // 512 + 1` (same as `es_enroll_num_frames`). `es_vad` returns an
  empty uint8 array at once when F = 0 (contract 2.4; Python's `(-1) // 10` would otherwise index an empty array).
- `es_vad(pcm)` (float32 [n]) -> uint8 [F], in four steps.
  1. `xp = concat(zeros(448), pcm)` as float64.
  2. For f in 0..F-1 - `seg = xp[512 f : 512 f + 512]`; `ss = cumsum(seg * seg)[-1]` (sequential float64 sum,
     index ascending, which the C code reproduces with a plain `double` loop); `E_f = 10 log10(ss / 512 + 1e-10)`.
  3. `Es = sort(E)`; `P10 = Es[(F - 1) // 10]` (equal to `floor(0.1 (F - 1))` for every F); `thr = max(P10 + 6,
     max(E) - 40)`; `m = E > thr` (strict).
  4. Dilate by one frame - `m2[f] = m[f-1] | m[f] | m[f+1]` with out-of-range neighbors false.

### 5.2 `token_groups(mask, max_tokens)`

```python
kept = np.flatnonzero(mask)                 # frames with m = 1, in order
n_avail = len(kept) // 3                    # drop a remainder under 3
groups = kept[: 3 * n_avail].reshape(n_avail, 3)
if n_avail > max_tokens:
    groups = groups[(np.arange(max_tokens) * n_avail) // max_tokens]   # floor(i * n_avail / max)
idx = zeros((max_tokens, 3), int64); valid = zeros(max_tokens, bool)
idx[: len(groups)] = groups; valid[: len(groups)] = True
return idx, valid, n_avail                  # caller treats n_avail < 4 as ES_ERR_ENROLL_SHORT
```

---

## 6. WeSpeaker (wespeaker.py and two tools)

### 6.1 `ml/earshot/wespeaker.py`

Port `KaldiFbank`, `BasicBlock`, `ResNet34` and `Embedder` from docs/research/data-training-bench/
convert_spk_coreml.py unchanged (that code matched torchaudio kaldi fbank within 8.1e-5 and the official ONNX at
cosine 1.0, measured, data-training.md 4.3). `WeSpeaker.load(path="models/wespeaker-voxceleb-resnet34/avg_model.pt")`
loads with `weights_only=True`, drops keys starting with `projection`, loads strictly otherwise (assert no missing
keys), sets eval, `requires_grad_(False)`. Input [B, N] raw audio in [-1, 1], 16,000 <= N <= 160,000; output [B, 256]
unit length. In training it runs on MPS under `torch.no_grad()`; it is never trained.

### 6.2 `ml/tools/convert_wespeaker.py`

`torch.jit.trace` on a 5 s clip, `ct.convert(..., inputs=[ct.TensorType(name="wav", shape=(1, ct.RangeDim(16000,
160000, default=80000)))], outputs=[ct.TensorType(name="embedding")], convert_to="mlprogram",
minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT32)`; save to
`ml/exports/wespeaker_resnet34_fp32.mlpackage`. Then check 10 clips from speech_test (VCTK, lengths 1, 3, 5, 8 and
10 s, two each) with `CPU_AND_GPU` and `CPU_ONLY` against the PyTorch CPU embedding; write
`ml/exports/wespeaker_check.json`; exit 1 if any cosine < 0.9999. The `.mlpackage` is never distributed (personal use).

### 6.3 `ml/tools/cache_own_embeddings.py` -> `ml/cache/own_v1.npz` (about 31 MB)

Own-voice pool (contract 8.1) - `sources.own_split(spk)` walks the speaker's rows in manifest order; cache rows = the
shortest prefix whose durations reach 8 s (never the last row), mix rows = the rest. The speaker is in the pool when
its mix rows hold >= 4 s (3,135 of 3,758 speakers, measured 2026-11-07 from speech_train.jsonl). Only cache rows ever
feed own enrollments; only mix rows ever feed the own talker in a mixture (8.3 step 5), so the model can never match
the own voice by identical waveform.

For each pool speaker (sorted speaker ids; this order over all 3,758 is also the AAM class index) and clip c in 0..7,
with `rng = default_rng([2026, spk_index, c])` - concatenate the cache rows in order with gaps of U(0.10, 0.30) s of
zeros and 5 ms fades, then take one 8.0 s (128,000-sample) window at a random start (the 8 clips may overlap, contract
6); scale so `active_power` (with `hop_vad`) is -26 dBFS. Embed in batches of 32 on MPS. Saved arrays - `emb` f32
[3758, 8, 256] (zeros outside the pool), `in_pool` bool [3758], `speakers` (str ids), the recipe as flat
int32 arrays `seg_clip` (spk*8 + c), `seg_row` (row in speech_train.jsonl), `seg_src` (start sample in the file),
`seg_len`, `seg_dst` (start sample in the clip), and `gain` f32 [3758, 8]. `sources.build_clip(src, spk, c)`
rebuilds the exact audio from the recipe (the loader needs it for the token path). Run time 10 to 20 min (estimate).

---

## 7. fileio.py and export.py

### 7.1 Container (contract 3.1)

- `write_container(path, magic, header, arrays)` - `arrays` is an ordered list of `(name, np.ndarray)` with dtypes
  float32 (`"f32"`), uint8 (`"u8"`), int32 (`"i32"`), C-contiguous little-endian. Build the data blob first (each
  array at the next 64-byte boundary, zero padding between arrays, no padding after the last), fill `arrays`,
  `data_bytes`, `data_sha256` in the header, serialize the header with `json.dumps(header, separators=(",", ":"))`,
  write the 16-byte preamble (`magic`, `uint32 1`, `uint32 header_len`, `uint32 0`), the header, zeros up to
  `data_start = round_up(16 + header_len, 64)`, then the blob. Write to `path + ".tmp"`, then `os.replace`.
- `read_container(path)` -> `(magic, header, {name: array})`; validates magic, version, offsets (inside the data,
  multiples of 64), sizes against shape and dtype, and `data_sha256`; raises `ValueError` naming the first bad field.

### 7.2 Weight layout (`to_export(model)` and its inverse `from_export(tensors)`)

- Linear - `weight.T` ([in, out]), bias unchanged. LayerNorm and Parameters unchanged.
- LSTM direction - `w_ih = weight_ih_l0.T` [in, 4H], `w_hh = weight_hh_l0.T` [H, 4H], `bias = bias_ih_l0 +
  bias_hh_l0` [4H]; the reverse direction uses `*_l0_reverse` and the suffix `_rev`. Gate order i, f, g, o is
  PyTorch's own, so columns need no reordering.
- `from_export` puts the summed bias into `bias_ih` and zeros into `bias_hh` (outputs are equal).
- File order (contract 3.2 table, read row by row with the index loop outermost) - for each k `enc.lin.{k}.weight`,
  `enc.lin.{k}.bias`, `enc.ln.{k}.weight`, `enc.ln.{k}.bias`; for each b `tn.weight, tn.bias, t.w_ih, t.w_hh, t.bias,
  tf.weight, tf.bias, bn.weight, bn.bias, b.w_ih, b.w_hh, b.bias, b.w_ih_rev, b.w_hh_rev, b.bias_rev, bf.weight,
  bf.bias` (prefix `blocks.{b}.`); `att.q.weight, att.q.bias, att.o.weight, att.o.bias, att.kv.weight, att.kv.bias,
  att.type_emb, att.null_tok`; `film.0.weight, film.0.bias, film.2.weight, film.2.bias`; `null.e_t, null.g_t,
  null.e_o, null.g_o`; for each k `dec.{k}.ln.weight, .ln.bias, .fc1.weight, .fc1.bias, .fc2.weight, .fc2.bias`;
  `heads.weight, heads.bias`; `tok.blk.tn.weight, tok.blk.tn.bias, tok.blk.bn.weight, tok.blk.bn.bias`;
  `tok.blk.t.{w_ih, w_hh, bias, w_ih_rev, w_hh_rev, bias_rev}`; `tok.blk.b.{same six}`; `tok.blk.tf.weight,
  tok.blk.tf.bias, tok.blk.bf.weight, tok.blk.bf.bias`; `tok.proj.weight, tok.proj.bias, tok.g.weight, tok.g.bias`.
  270 tensors, 2,570,500 floats (10.28 MB, analytic). `export.py` holds this list as one function, `EXPORT_ORDER()`.
- Header - `format "earshot-weights"`, `format_version 1`, `model "esb16-df"`, `model_version "1.0.0"`, `config`,
  `config_hash`, and `meta` = `{run, step, ema, git_commit, date, init ("random-seed-0" or "trained"),
  data_licenses (sorted unique license strings of the training manifests), wespeaker_note: "WeSpeaker conditioning
  trained with VoxCeleb research-terms weights - personal use only", credits: "interaction idea from Look Once to
  Hear (UW, CHI 2024); no LOTH code used", eval_only_data: ["WHAM!"]}`.
- `model_hash` (used in thresholds.json, person.json, telemetry and `meta.weights_sha256`) - the `data_sha256` of
  `earshot.esw` (contract 3.1; identifies trained weights, `config_hash` only the architecture). It does not change
  when only `meta` changes, so the export, calibrate, re-export with `--thresholds` sequence of 7.3 and 13 keeps it.

### 7.3 Commands

| Command | Writes |
|---|---|
| `python -m ml.export --random --seed 0 --out ml/exports/random_v1/` | all six files - `earshot.esw` from `init_random_for_parity(seed 0)`, the three `.est` files, `thresholds.json` (contract 6 initial values), `quality_vectors.json` (13) |
| `python -m ml.export --ckpt ml/runs/main/best_1.pt --out ml/exports/v1/ [--raw] [--thresholds ml/runs/main/thresholds.json]` | same six files from the EMA weights (raw weights with `--raw`); thresholds from calibration |
| `python -m ml.export --testvec --out DIR` | re-creates the three `.est` files from `DIR/earshot.esw`, loaded through `load_esw` (so the reference really uses the exported numbers) |

### 7.4 Test vectors (`make_testvecs`, CPU fp32, contract 3.3)

Inputs (fixed, license-clean, written once per export with seed 7) -
- `in` long - 10 s rendered by `render.render_example` in test mode (8.9) from speech_test (VCTK test speakers,
  CC BY 4.0) with a CC0 FSD50K noise_test clip, sim_test room 0: target + 1 talker (SIR 0 dB) + own voice (+6 dB) +
  noise (SNR 5 dB), mixture -26 dBFS. `in` short = the first 32,000 samples of the long one.
- `enr_t` - 96,000 samples, other utterances of the target, noisy at 10 dB; `enr_o` - 128,000 samples, a clean
  clip of the own-voice speaker built like 6.3.
- `vad_t = es_vad(enr_t)`; `mask_t = vad_t` with frames 40 to 49 forced to 0 (proves the engine uses the passed mask);
  `mask_o = es_vad(enr_o)`. `e_t`, `e_o` - PyTorch WeSpeaker on CPU.
- `tok_t, g_t` = `encode_enrollment(enr_t, token_groups(mask_t, 48))`, only valid rows kept ([n_t, 64]);
  `tok_o, g_o` likewise with 16.

Cases - `person` (target, own, WeSpeaker on), `everyone` (has_t False, own on), `noown` (has_o False), `nowespk`
(use_wes False). For each - `cond.{case}.film|k|v|n` from `pack_cond`, then `out.{case}` and `heads.{case}` from a
reset `StreamRef`. `dbg.*` - first 8 hops of `person`, short file only.

Array order in the file - `in, enr_t, mask_t, enr_o, mask_o, vad_t, e_t, e_o, tok_t, g_t, tok_o, g_o`, then per case
(person, everyone, noown, nowespk) `cond.{case}.film, .k, .v, .n, out.{case}, heads.{case}`, then
`dbg.feat, dbg.z_enc, dbg.z_blk0, dbg.z_blk1, dbg.z_blk2, dbg.z_blk3, dbg.coef`.

Header - same fields as the weights with `format "earshot-testvec"`, magic `EST1`, plus `meta.weights_sha256` (the
`data_sha256` of the `earshot.esw` used), `meta.sources` (file paths and licenses of the audio) and
`meta.tolerances` with exactly the contract 3.3 key names and values -
`{"vad": 0, "tokens": 1e-4, "g": 1e-4, "film": 1e-4, "k": 1e-4, "v": 1e-4, "dbg": 1e-4, "out": 2e-4,
"out_snr_db": 60.0, "heads": 1e-3}` in the short file and `"out": 1e-3, "out_snr_db": 50.0` in the long file (the
long file keeps `"dbg"` although it has no `dbg` arrays). `test_export` asserts the key set equals this list.

`wespk_selftest.est` (contract 6 Self-test row) - same container and header (magic `EST1`, the esb16-df config), `meta.kind
"wespeaker-selftest"`, `meta.tolerances {"cosine_min": 0.999}`, arrays `clip` f32 [80000] (5 s of a VCTK test
utterance at -26 dBFS) and `embedding` f32 [256] (PyTorch CPU fp32).

`thresholds.json` - exactly the contract 6 object; `model_hash` from 7.2; random_v1 carries the initial values.

`quality_vectors.json` (contract 6) - 5 fixed 16 kHz captures written as base64 float32 (seed 7: a clean 8 s talker
with 2 s of leading noise, a 14 s capture longer than 10 s, a capture with no non-speech frame, a 4.2 s capture that
rounds to exactly 4 s, and one with 1.5% clipped samples), each with the expected `sbr_db`, `clip_pct`, span, best
window start, rounded length, window list and the `es_vad(clip)` speech seconds, all from `quality.py`. The values do
not depend on the weights. `test_metrics.py` and the app's `ClipToolsTests` check them.

---

## 8. Data loader (sources.py, render.py, enroll.py, data.py; contract 8.1)

### 8.1 Sources and the license gate

`sources.Sources(split)` loads the following once per process with `dataprep.common.read_manifest`.
- speech - `speech_{split}.jsonl`; noise - `noise_{split}.jsonl` rows with `use` noise (by `source` - demand, fsd50k,
  musan) and rows with `use` music; RIRs - `data/rir/sim_{split}.npy` (+ `.jsonl` for `dp`) and, for train only,
  `real_train.npy` (+ `.jsonl` for `dp`, `room_id`, `db`), opened lazily with `np.load(..., mmap_mode="r")` inside
  the worker (never pickled).
- Per-row arrays kept small for spawn pickling - relative paths (list of str), durations (float32), speaker index
  (int32), per-speaker lists of row indices.
- License gate - `check_row(row, split)` raises `LicenseError` if the row's license is empty, or contains the token
  `NC` (split on `-` and spaces, case-insensitive) and the split is not `test`, or if `"eval"` is in `use` and the
  split is not `test`, or if `row["split"] != split`. Evaluation code calls the same function with split `test`.
- Disjointness - at load, assert that the train, val and test speaker sets do not intersect (speech_test,
  eval_librispeech_test, eval_edacc_test, eval_voxpopuli_accented).
- Pools (train, contract 8.1; counts measured 2026-11-07 from speech_train.jsonl) - target pool = speakers with at
  least 2 utterances and at least 14.5 s in total (3,118 of 3,758; 640 fail, among them the 122 single-utterance
  speakers); own-voice pool = speakers whose `own_split` (6.3) leaves at least 4 s of mix rows (3,135); interferer and
  babble pools = all speakers. AAM label = index in the sorted 3,758 speaker ids (contract 8.2 class count); speakers
  outside the target pool simply never appear as a label.
- Audio read - `read_crop(row, start, n)` = `soundfile.read(DATA / path, start=start, frames=n, dtype="float32")`,
  zero-padded to n. A random 4 s FLAC crop costs 0.53 ms (measured, bench/RESULTS.md section 8).

### 8.2 Recipes (one table in code, `RECIPES`; YAML only names a recipe)

| Field | SMOKE | A (0-20k) | B (20k-105k) | C (105k-120k) | FT (fine-tune) |
|---|---|---|---|---|---|
| Person / Everyone | 100 / 0 | 100 / 0 | 75 / 25 | 75 / 25 | 75 / 25 |
| Target absent (of Person) | 0 | 0 | 10% | 10% | 10% |
| Competing talkers 0/1/2/3 | 0/100/0/0 | 15/45/40/0 | 10/35/30/25 | same as B | same as B |
| SIR target vs loudest talker | U(0, 6) | U(-3, 12) | U(-8, 12) | same | same |
| Own enrollment given | 90% | 90% | 90% | 90% | 90% |
| Own talker in mix | 0 | 0 | 50% of all examples (= 0.5/0.9 of those with own enrollment) | same | same |
| Babble | 0 | 70% | 70% | 70% | 70% |
| Noise / music | 100 / 0 | 80 / 20 | 80 / 20 | 80 / 20 | 80 / 20 |
| SNR | U(5, 10) | N(8, 4) in [0, 25] | N(5, 5) in [-8, 25]; 10% quiet U(20, 40) | same | same |
| Rooms sim / real / dry | 100 / 0 / 0 | 75 / 15 / 10 | 75 / 15 / 10 | 60 / 30 / 10 | 60 / 30 / 10 |
| Lombard (of loud scenes) | 0 | 30% | 30% | 30% | 30% |
| Mic chain, clipping | on, no clipping | on | on | on | on |
| Segment x batch | 4 s x 8 | 4 s x 8 | 4 s x 8 | 8 s x 4 | 8 s x 4 |
| Rig recordings | 0 | 0 | 0 | 0 | 10% (8.10) |

Stage A talker counts are the contract's "<= 2 talkers" spread over 0-2 (15/45/40, adopted in DECISIONS L). Everything
not in the table is the same in every recipe and follows 8.3 to 8.6. `RECIPE_VERSION = 1` is saved in checkpoints;
changing any number bumps it.

### 8.3 Rendering one example (`render.render_example(src, recipe, rng, L, Le)`), in this exact order

1. Scalars first, drawn into one `Draws` dataclass whose field order is the draw order and whose field comments give
   each distribution (that dataclass is the table; resume-exact determinism depends only on it), in this order - mode (Person or Everyone; in Person, absent with the recipe's probability; in
   Everyone, the 10% "nobody talking" case of step 11), room kind, talker count, own enrollment given, own clip index
   and own talker, babble and its talker count, background kind (noise 80% or music 20%, exactly one), noise clip
   count and sources (only when noise), SIR, SNR (quiet room or normal),
   Lombard, own level. Content (rooms, speakers, offsets, filter values) is drawn by the later steps.
2. Room - sim, real or dry. Sim - room r ~ U{0..n_rooms-1}, `h_k = sim[r, k]` (float16 to float32), `dp_k` from the
   jsonl, positions as in data/rir/README.md (0 target, 1-3 talkers, 4 own voice, 5-7 babble and noise). Real - target
   row r0 ~ U{0..324}; every other source takes a row from the same `room_id` without replacement while any remain,
   then a random row of the same `db`. Dry - `h = [1.0]`, `dp = 0`.
3. Speakers - target (from the target pool), then distinct competing talkers, own-voice speaker (from the own pool) and babble speakers
   (no speaker twice in one example). Everyone mode still draws talker 0 as "target" (its slot is just talker 0).
4. Target and enrollment utterances - shuffle the target's rows; the enrollment set E is the shortest prefix whose
   total duration is >= Le + 0.5 s, capped at all rows but one; the target stream uses only the remaining rows. If E
   is shorter than Le + 0.5 s or the remaining rows are shorter than the segment L, the target is re-drawn from the
   target pool (same generator, at most 20 tries, then the longest-audio speaker of the pool). So the enrollment is
   always from **different utterances** than the mixture and is never cycled (contract 8.1).
5. Dry streams - `speech_stream(spk, n, rows, rng)` concatenates random windows of the allowed rows (first one from a
   random offset) with gaps of U(0.1, 0.5) s, 5 ms fades, cycling rows if needed. Each talker's stream is scaled to
   active power -26 dBFS + N(0, 4) dB ("level at 1 m"; the sim RIR then applies 1/r). Activity window - an active
   span of `round(frac * L)` samples at a random start, zeros elsewhere, 10 ms fades; `frac` = 1 for the target in 75%
   of present examples and U(0.3, 0.9) in 25% (**WP-ML choice** from quality.md F.2), U(0.5, 1.0) for competing
   talkers, U(0.2, 0.6) for the own voice (the own talker uses only its mix rows from `own_split`, never its cache
   rows, 6.3), 1 for babble talkers.
6. Lombard - loud scene = drawn SNR <= 5 dB; in 30% of loud scenes apply `sos_high_shelf(1000, U(2, 4) dB)` to every
   talker's dry stream (target, talkers, own) and add U(3, 8) dB to the SNR goal of step 9.
7. Convolve (`scipy.signal.fftconvolve(...)[:L]`, about 0.4 ms per source, measured, bench/RESULTS.md section 9) -
   target image `t = dry_t * h_0`, target reference `s = dry_t * early_rir(h_0, dp_0)`; talker images
   `img_i = dry_i * h_i` and early refs `r_i`; own image `o = dry_o * h_4` and early ref `o_e`; babble talkers summed
   per position (round robin over 5, 6, 7 from a random start) then convolved once per position. Noise - 1 clip (70%)
   or 2 clips (30%) when the background kind is noise, each from DEMAND 30% / FSD50K 40% / MUSAN 20% / synthetic 10%: DEMAND = random channel file and crop,
   no RIR (already ambient); FSD50K = clip tiled with random gaps U(0, 1) s to fill L, convolved with a position 5-7
   RIR; MUSAN noise = crop (tiled if short), position 5-7 RIR; synthetic = generated in the loader, no RIR
   (contract 8.1; the data pack has no fourth noise source) - 70% colored noise (white noise shaped by
   `f^(-alpha/2)` above 20 Hz, alpha ~ U(0, 2)), 30% mains hum (50 or 60 Hz, harmonics 1-10 with amplitudes
   `U(0.2, 1) / n`, plus colored noise 20 dB below). Music - crop of a music row, position 5-7 RIR.
8. VAD labels (8.4) on the clean early references, before any level change.
9. Levels - `P_t = active_power(t, vad_t)`. SIR (if any talkers) - with the drawn `sir`,
   `P_max = max_i active_power(img_i, vad_i)`, scale all talkers (images and early refs) by
   `sqrt(P_t / (P_max 10^(sir/10)))`. Own (if in the mix) - with the drawn level `own_db` (U(0, 18) dB in sim rooms
   with geometry `in_hand`, U(0, 12) dB otherwise, contract 8.1), scale o and o_e so
   `active_power(o) = P_t 10^(own_db/10)`. SNR - budget `P_b = P_t / 10^(snr/10)` goes to the backgrounds present
   (contract 8.1): music alone takes all of `P_b`; music with babble - music (mean power over the segment)
   `min(P_t / 10^(U(5, 20)/10), 0.5 P_b)` and babble the rest; noise with babble - split as `w` / `1 - w` with
   `w ~ U(0.2, 0.8)`; one background alone takes all. So the realized SNR always equals the drawn one (test_data).
   Quiet-room examples use the quiet SNR.
10. Mixture `x = t + sum(img_i) + o + babble + noise + music`. Absent - drop t, `s = 0`, `vad_t = 0` (levels were set
    with the would-be target, and the enrollment still comes from that speaker, who is not in the mix).
11. References - Person present - `ref = s`. Everyone - `ref = s + sum(r_i)` (talkers 0-3, scaled as in the mix;
    babble, noise and own voice excluded), `vad_t = OR` of those talkers' VADs; in 10% of Everyone examples talker 0
    and all competing talkers are dropped (**WP-ML choice**), giving `ref = 0` and an absent-type example.
12. Mic chain (8.6) on x and ref (and on the metric-only signals o and r_i when requested).

### 8.4 Activity labels (`dsp.hop_vad`, 4 ms resolution)

For a clean signal s of length L - `e[j] = mean(s[64j - 192 : 64j + 64]^2)` (16 ms ending at the end of hop j, zeros
before 0); `e_db = 10 log10(e + 1e-12)`; active if `e_db > max(e_db) - 35` and `e > 1e-9`; then dilate by 4 hops on
each side (**WP-ML choice**). Used for `vad_t` (target or Everyone reference), `vad_o` (own early ref), talker VADs
(levels, picked rate) and the oracle enrollment mask (via `es_vad`, 8.5). Hop j's label describes input samples
64j .. 64j+63, the newest hop of STFT frame j, which is what heads predict.

### 8.5 Target enrollment and own enrollment (`render_enrollment`)

Target (Person examples, present or absent) - Le from the batch (U{3.0, 3.5, ..., 10.0} s per batch).
- Dry - rows E of 8.3 step 4 concatenated with gaps U(0.1, 0.4) s, cropped to Le (step 4 guarantees enough audio, so
  no source sample appears twice).
- Room - same room 50% (sim - `h_0` of the same room; real - row r0; dry - dry), other room 30% (random sim room 80%
  or real row 20%, position 0), dry 20%. Clean component `c_e = dry * h`.
- Weaker talker 30% - one of this example's competing talkers when one exists (50%), else a fresh speaker; position 1
  RIR of the enrollment room (dry if the room is dry); active power 5 to 15 dB below `c_e`.
- Noise 80% - one noise clip drawn as in 8.3 plus, in half of those, babble of 6 to 12 talkers; SNR U(0, 25) dB
  against `active_power(c_e)`.
- Mic chain - the mixture's high-pass, EQ and `fir_decim` (same phone), level U(-40, -15) dBFS drawn independently,
  self-noise, no clipping.
- Mask - oracle 40% (`es_vad` of the clean component after the mic chain), `es_vad` of the final clip 40%, all ones
  20%. `token_groups(mask, 48)`; if `n_avail < 4` the all-ones mask is used instead and `stats.short_enroll` counts it.

Own voice (own enrollment given, 90%) - clip c ~ U{0..7} of the own speaker, rebuilt with `sources.build_clip`; `e_o` from
the cache. Token path only - 30% add one noise clip at SNR U(15, 30) dB; mixture mic chain; level U(-40, -15) dBFS;
mask = `es_vad` (as on the phone); `token_groups(mask, 16)`. Not given (10%) - `has_o = False`, zero arrays, and no
own talker in the mix.

WeSpeaker dropout - `use_wes = rng.random() >= 0.2` per example (both e_t and e_o fall back to nulls).

### 8.6 Mic chain (same linear filters on mixture and references)

High-pass `sos_highpass(U(60, 200) Hz)`; 0, 1 or 2 (equal odds) peaking EQs with f0 log-uniform 300 Hz to 6 kHz,
gain U(-4, 4) dB, Q U(0.7, 2.0); `fir_decim`. Then gain `g = 10^(U(-40, -15)/20) / rms(x)` on x and every reference.
Mixture only - self-noise (white or pink, equal odds) at U(-95, -75) dBFS RMS; clipping with u ~ U(0, 1) - u < 0.03
hard clip at `quantile(|x|, U(0.995, 0.9995))`, 0.03 <= u < 0.06 soft clip `c tanh(x / c)` with
`c = quantile(|x|, U(0.99, 0.999))`. If `ml/assets/decim16k_fir.npy` is missing, `render.py` raises with the command
that makes it (`.venv/bin/python engine/tools/make_resampler_tables.py`, run and committed by the lead on day 0,
contract 11), except when `allow_missing_decim: true` (smoke config
and unit tests only), which skips `fir_decim` and logs it.

### 8.7 Batch schema (`MixDataset.__getitem__(batch_index)` returns one whole batch as numpy)

| Key | dtype, shape | Meaning |
|---|---|---|
| `mix`, `ref` | f32 [B, L] | mixture; target or Everyone reference (zeros if absent) |
| `vad_t`, `vad_o` | f32 [B, L/64] | per-hop activity of ref and of the own voice (zeros if none) |
| `present`, `own_in_mix`, `person` | bool [B] | ref has energy; own talker in mix; Person mode |
| `enr_t` | f32 [B, Le] | target enrollment (zeros in Everyone) |
| `tok_idx_t`, `tok_val_t` | i64 [B, 48, 3], bool [B, 48] | token groups (5.2) |
| `enr_o`, `tok_idx_o`, `tok_val_o` | f32 [B, 128000], i64 [B, 16, 3], bool [B, 16] | own enrollment and groups |
| `e_o` | f32 [B, 256] | cached own WeSpeaker vector |
| `has_t`, `has_o`, `use_wes` | bool [B] | conditioning flags (3.2) |
| `mode` | f32 [B, 2] | [1, 0] Person, [0, 1] Everyone |
| `spk_t` | i64 [B] | AAM class of the target speaker, -1 in Everyone |
| `info` | list of B dicts | drawn values (room, rt60, snr, sir, own_db, counts, Le) for logging and tests |

`e_t` is not in the batch; the training loop computes it on MPS with WeSpeaker from `enr_t` (contract 6, merged_full
config). Loader cost per example 40 to 80 ms on one core (estimate from the measured 0.4 ms convolutions and 0.53 ms
crops, about 15 to 30 of each), so 6 workers give about 10 batches per second against the 1 per second needed.

### 8.8 Seeding, workers, determinism (macOS spawn)

- Example i of batch j has global index `i_ex = 8 j + i` (8 even when B = 4, so indices never collide across
  stages) and its own generator `rng = np.random.default_rng([seed, i_ex])`; batch-level values (Le, and L from the
  stage) use `np.random.default_rng([seed, j, 7])`. The training batch for step s is batch index s, so a resumed run
  sees exactly the data it would have seen (contract 8.4).
- `make_loader(src_split, recipe, seed, start_batch, end_batch, B, L, workers=6)` returns
  `DataLoader(MixDataset(...), batch_size=None, sampler=range(start_batch, end_batch), num_workers=workers,
  multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2, pin_memory=False,
  worker_init_fn=worker_init)`. The DataLoader yields in sampler order even with 6 workers.
- `MixDataset`, `worker_init` and the collate logic are module-level (picklable, no lambdas); `train.py` keeps its
  entry point under `if __name__ == "__main__":`. `worker_init` calls `torch.set_num_threads(1)` and opens the
  memmaps. The per-worker generator the contract mentions is this per-example generator; no other RNG is used.
- At a stage boundary (20,000 and 105,000) the loop drops the old loader and builds a new one for the new recipe.

### 8.9 Test mode (used by render_val, evaluate and the test vectors)

`render_example(..., overrides={...})` accepts fixed values for any drawn quantity (mode, absent, talker count, sir,
snr, own on/off and dB, room kind and index, geometry, enrollment kind, Le) and a `metric_signals=True` flag that
also returns the own image `o`, the talker early refs `r_i` and their VADs after the mic chain. Test mode fixes the
level at -26 dBFS, high-pass 100 Hz, no EQ, no clipping, no Lombard, `fir_decim` on.

### 8.10 Rig recordings (fine-tune only, optional)

If `data/rig/manifest.jsonl` exists (rows `{mix, target, enroll, own, license: "own"}`, 16 kHz, solo recordings at
the phone, made with the procedure of quality.md G.4), FT examples use a rig row 10% of the time - `ref` = the target
solo recording, `enr_t` = the enrollment recording, `enr_o` = the own-voice recording, no further augmentation. Rows
are split 80/20 into fine-tune and evaluation by recording session. Without the file the FT recipe has 0% rig.

---

## 9. Losses (losses.py, contract 8.2, all fp32)

Per example i (sums over samples of the 4 or 8 s segment; `eps = 1e-8`; `x` = mixture, `s` = ref, `y` = output).

| Term | Code | Weight | Applies when |
|---|---|---|---|
| `tsnr` | `-10 log10(sum s^2 / (sum (s - y)^2 + 1e-3 sum s^2 + eps) + eps)` | 1 | `present` (Person present or Everyone with talkers) |
| `mrstft` | for n in (256, 512, 1024), hop n/4, Hann, with `A = abs(stft_n(y))`, `R = abs(stft_n(s))` - `mean(abs(A - R)) + mean(abs(log(A + 1e-5) - log(R + 1e-5)))`, summed over n | 1 | `present` |
| `absent` | `10 log10(sum y^2 + 10^-2.5 sum x^2 + eps)` | 1 | not `present` |
| `gap` | the absent form over samples whose hop has `vad_t = 0` | 0.1 | `present` and at least 1,600 such samples (0.1 s) |
| `own` | the absent form over samples with `vad_o = 1` and `vad_t = 0` | 0.3 | `own_in_mix` and at least 1,600 such samples (**WP-ML choice**, same 0.1 s floor as `gap`) |
| `oversup` | `mean(relu(abs(S)**0.3 - abs(Y)**0.3)**2)`, S and Y = STFT 512 / 128, Hann, of s and y | 0.3 | `present`, recipes B, C, FT |
| `bce_t`, `bce_o` | `binary_cross_entropy_with_logits(head_logits[..., 0], vad_t)` and `[..., 1]` vs `vad_o`, mean over hops | 0.1 each | all |
| `aam` | AAM-softmax on g_t, margin 0.2, scale 30, 3,758 classes | 0.05 | `person` |

`total = mean_i(sum of weight x torch.where(applies, term, 0))` - each term is selected with `torch.where`, never
multiplied by a 0/1 flag (0 x NaN = NaN), and unused inputs are made finite first (masked means divide by
`max(count, 1)`; `g_t` of an Everyone row is replaced by `null.g_t` before `normalize`). The AAM term is computed only
on the rows with `person == True` (index select), so `spk_t = -1` never reaches `cross_entropy`; with no Person row in
the batch the term is 0. `AAMHead` holds `weight` [3758, 128] (481,024 parameters, training
only); `cos = normalize(g) @ normalize(weight).T`; for the true class `phi = cos cos(m) - sin sin(m)` with
`sin = sqrt(clamp(1 - cos^2, 0, 1))`, and where `cos < cos(pi - m)` use `cos - sin(pi - m) m` instead (standard
ArcFace); logits `30 x` (phi on the label, cos elsewhere); cross-entropy. `total_loss` returns the scalar and a dict
of every term's batch mean (detached) for logging.

`mrstft` keeps weight 1 (DECISIONS L 2026-11-08, Loss A/B outcome; contract 8.2). In the A/B, tsnr_only (`mrstft`
at weight 0) and comp03 (a compressed spectral loss) were both about 0.2 dB ahead at step 4,000, but the acceptance
test at step 6,000 tied, so the spec loss stays. `total_loss` keeps a weight-0 path for the A/B harness's tsnr_only
variant - when `WEIGHTS["mrstft"]` is 0 it computes the term under `torch.no_grad()`, so the value is still logged in
train.csv but no graph is built and the backward pass skips its three STFTs. The parameter updates are bit-identical
to keeping it in the graph at weight 0 (checked against the in-graph weight-0 version the A/B first ran with, CPU
only, batch 2 - loss, every part and grad_norm equal at each step;
model, AAM and Adam state equal after 3 steps from a fresh start, where no EMA exists yet; model, EMA, AAM and Adam
state equal after 3 steps from the A/B step-4,000 checkpoint).

---

## 10. Training (train.py, contract 8.3 and 8.4)

### 10.1 Optimizer, schedule, EMA

- AdamW (`fused=True`, falling back to the plain version if MPS refuses), betas (0.9, 0.999), eps 1e-8. Two groups -
  weight decay 1e-4 on Linear and LSTM weight matrices and the AAM weight; 0 on LayerNorm weights and biases, all
  biases, `att.type_emb`, `att.null_tok`, `null.*`.
- `lr_at(step)` - `5e-4 * step / 2000` for step < 2,000; then `2.5e-5 + 0.5 (5e-4 - 2.5e-5)(1 + cos(pi p))` with
  `p = (step - 2000) / (120000 - 2000)`, clamped to 1. Stage C simply continues it. Fine-tune - 500 warm-up steps
  (**WP-ML choice**) then cosine from 1e-4 to 1e-5 over 10,000 steps.
- Gradient clipping - `clip_grad_norm_(model and AAM parameters, 5.0)`.
- EMA - at step 2,000 copy the weights; after every later optimizer step
  `torch._foreach_lerp_(ema_params, params, 0.001)` (decay 0.999). Validation, best selection and export use the EMA
  weights once they exist.

### 10.2 One step (`step_batch`)

```python
b = to_mps(batch)                                         # numpy -> torch.float32 / int64 / bool on MPS
with torch.no_grad():
    e_t = wes(b.enr_t)                                    # [B, 256], frozen WeSpeaker (6.1)
tok_t, g_t = model.encode_enrollment(b.enr_t, b.tok_idx_t, b.tok_val_t)
tok_o, g_o = model.encode_enrollment(b.enr_o, b.tok_idx_o, b.tok_val_o)
cond = model.build_cond(CondInputs(e_t, tok_t, b.tok_val_t, g_t, b.e_o, tok_o, b.tok_val_o, g_o,
                                   b.has_t, b.has_o, b.use_wes, b.mode))
y, head_logits = model(b.mix, cond)
loss, parts = total_loss(y, head_logits, g_t, b, aam, oversup_on)
opt.zero_grad(set_to_none=True); loss.backward()
gn = clip_grad_norm_(params, 5.0)
if finite(loss) and finite(gn): set lr; opt.step(); ema.update(step)
else: nan_skips += 1                                       # no update, logged
```

Ten non-finite steps within 1,000 stop the run with exit code 3 (a human looks; `run_forever.sh` does not restart).
The finiteness check reads two scalars, so there is one MPS sync per step; the measured bench step synchronized every
step too, so the 1.03 s/step figure already includes it.
WeSpeaker and the own-token path run for every example; flags mask the unused results, which keeps the code
branch-free. Enrollments of up to 10 s (the bench used 5 s and 6 s) add an estimated 2 to 5% to the measured
1.03 s/step. The real number, with data loading, is 1.101 s/step at 8 x 4 s (mean over the first hour of the main
run, 3,232 steps), which sets the main-run length (contract 8.4). Earlier numbers came from a busy Mac - 1.262 s in
the A/B run tsnr_only, 1.34 s in the spec-loss pilot.

### 10.3 Logs (plain files in `ml/runs/<run>/`, no TensorBoard)

- `train.csv` - every 50 steps, one row - `step, wall_s, lr, loss, tsnr, mrstft, absent, gap, own, oversup, bce_t,
  bce_o, aam, grad_norm, step_s_median, data_wait_s, mps_gb, nan_skips, short_enroll`. Values are means over the 50
  steps, accumulated on the device and read with one `.item()` sync per row.
- `val.csv` - every validation - `step, weights (ema|raw), sisnri_median, sisnri_mean, picked_sir0, absent_db_median,
  absent_under10_pct, own_db_median, tsos_pct, eligible, best_rank`.
- `events.log` - timestamped text - start, resume, stage change, checkpoint, new best, gate results, warnings
  (on battery from `pmset -g batt` every 10 min, step time over 1.25 s averaged over the first hour, short enrollments).
- `run.json` at start - resolved config, git commit, torch and coremltools versions, env vars, data manifest sizes,
  `RECIPE_VERSION`.

### 10.4 Checkpoints and resume

- Every 1,000 steps - `step_<n>.pt` written as `step_<n>.pt.tmp` then `os.replace`. Contents - `model`, `ema` (or
  None), `opt`, `aam`, `step` (completed steps), `seed`, `config`, `recipe_version`, `best` (list of
  `(score, step)`), `torch_version`. About 47 MB (analytic - model + EMA + AAM + two Adam moments).
- Keep the last 3 step files and the best 2 as separate copies `best_1.pt`, `best_2.pt` (copy, then rename).
  Pruning deletes only `step_<n>.pt` files of this run that are neither among the last 3 nor in use.
- Best - after each validation, eligible if `absent_db_median >= 20` and `own_db_median >= 20`; score =
  `sisnri_median`; the top 2 eligible are kept (contract 8.4).
- `--resume last` - load the highest `step_<n>.pt` (start fresh if none, and say so in `events.log`); refuse if
  `recipe_version` differs (unless `--force-recipe`); move optimizer state to MPS; continue at batch index `step`.
  The schedule is a pure function of step, so nothing else needs restoring.

### 10.5 Gates (written as `gate_<name>.json` with PASS or FAIL and the numbers)

- Smoke (contract 8.4) - 16 fixed examples (SMOKE recipe, seed 1234, Le 6 s, rendered once and reused); batches
  alternate between examples 0-7 and 8-15. Every 100 steps compute SI-SNRi on all 16 with the current weights; PASS
  as soon as the median is >= 20 dB; FAIL at 1,500 steps (exit code 2). Plus `pytest ml/tests` must pass.
- Pilot (= main steps 0-6,000) - PASS needs the val SI-SNRi median at 6k above both 4k and 2k (still rising at the
  end of the pilot; 2k and 4k are not compared) and `nan_skips == 0` (DECISIONS L 2026-11-08). There is no absolute
  dB target, because the research phase never measured a learning curve. Step time is information, never a check -
  `gate_pilot.json` reports `step_s_median_500_6000` (median of the train.csv 50-step medians over steps 500-6,000)
  and `main_steps_for_40h` = floor(40 x 3600 / (step_s x 1.05) / 1000) x 1000, the main-run length that step time
  allows. The run stops at 6,000 with a checkpoint; the main run resumes it. PASS means only "still rising at 6k",
  not "learns well" - the spec-loss pilot v1 (-0.86 / -1.14 / -0.24 dB at 2k / 4k / 6k, all negative) would pass it.
- Main - after the first hour, if the mean step time is above 1.25 s, `events.log` warns and gives the step count
  that fits 40 h at that speed (`main_steps_for_40h`), and the lead cuts stage B to that count (edit `stages` in
  main.yaml; the model is never shrunk), contract 8.4. The main run's first hour measured 1.101 s, so main.yaml
  keeps 120,000 steps.

### 10.6 Configs

```yaml
# ml/configs/main.yaml - main run (10-ml 10.6, contract 8.4), 120,000 steps, about 38.5 h of compute.
# The first hour of this run measured 1.101 s/step (events.log 2026-11-08 20:52, 3,232 steps, data loading included)
# x 1.05 for validation and checkpoints = 1.156 s -> 120,000 steps = 38.5 h (wall time is longer while the guard
# yields to other builds). The contract 8.4 rule was not triggered, so this is the original plan. The 108,000-step
# plan was a premature cut from a loaded-Mac A/B number, 1.262 s/step (DECISIONS L 2026-11-08).
# Started by `ml/tools/run_forever.sh ml/configs/main.yaml`, which resumes the pilot (pilot.yaml stops this same run
# at step 6,000). Contract 8.4 rule - if the first hour averages above 1.25 s/step, events.log warns and the lead cuts
# stage B here (edit `stages` to the total main_steps_for_40h gives); the model is never shrunk.
run: main
seed: 1                      # data seed - example i of batch j uses default_rng([seed, 8 j + i])
steps: 120000
stop_at: null
lr: {peak: 5.0e-4, final: 2.5e-5, warmup: 2000, decay_end: 120000}
optim: {betas: [0.9, 0.999], weight_decay: 1.0e-4, clip: 5.0}
ema: {decay: 0.999, start: 2000}
val_every: 2000              # validation on the 400 fixed mixtures (ml/cache/val_v1.npz)
ckpt_every: 1000
keep_last: 3
keep_best: 2
workers: 6
allow_missing_decim: false
stages:
  - {start: 0,      recipe: A, batch: 8, segment_s: 4.0, oversup: false}
  - {start: 20000,  recipe: B, batch: 8, segment_s: 4.0, oversup: true}    # 20,000-105,000 (85,000 steps)
  - {start: 105000, recipe: C, batch: 4, segment_s: 8.0, oversup: true}    # 105,000-120,000 (15,000 steps)
```

- `pilot.yaml` - `base: main.yaml`, `stop_at: 6000`, `gate: pilot` (same run directory `ml/runs/main/`).
- `smoke.yaml` - `run: smoke`, `steps: 1500`, `lr: {peak: 5.0e-4, final: 5.0e-4, warmup: 100}`, `ema: null`,
  `fixed_examples: 16`, one stage `{recipe: SMOKE, batch: 8, segment_s: 4.0}`, `gate: smoke`,
  `allow_missing_decim: true`.
- `finetune.yaml` - `base: main.yaml`, `run: finetune`, `init: ml/runs/main/best_1.pt` (EMA weights become the
  starting weights; fresh optimizer), `steps: 10000`, `lr: {peak: 1.0e-4, final: 1.0e-5, warmup: 500,
  decay_end: 10000}`, one stage `{recipe: FT, batch: 4, segment_s: 8.0, oversup: true}`.

### 10.7 Commands (from the repo root, on AC power)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
caffeinate -dimsu .venv/bin/python -m ml.train --config ml/configs/smoke.yaml          # 0.5 h
caffeinate -dimsu .venv/bin/python -m ml.train --config ml/configs/pilot.yaml          # 1.9 h
ml/tools/run_forever.sh ml/configs/main.yaml                                           # 38.5 h compute, resumes pilot
caffeinate -dimsu .venv/bin/python -m ml.train --config ml/configs/finetune.yaml       # optional, 3.2 h
```

Flags of `ml.train` - `--config`, `--resume last|PATH`, `--run NAME`, `--stop-at N`, `--workers N`,
`--device mps|cpu`, `--force-recipe`. Exit codes - 0 finished or stopped at `stop_at`, 1 crash, 2 gate failed,
3 NaN storm.

`ml/tools/run_forever.sh CONFIG` - exports the three PyTorch variables, then loops
`caffeinate -dimsu .venv/bin/python -m ml.train --config "$CONFIG" --resume last`; exits on code 0, 2 or 3; on any
other code waits 30 s and restarts; stops after 5 crashes in a row that each lasted under 10 minutes. It never
deletes files.

---

## 11. Validation set (`python -m ml.tools.render_val`, contract 8.4)

400 fixed mixtures, seed 2026, from speech_val (102 speakers), noise_val (FSD50K and MUSAN; the pack has no DEMAND
val rows, so DEMAND's share goes to FSD50K) and sim_val (90%) or dry (10%) rooms; no real RIRs (the only real train
bank is used for training). Recipe B values, segment 4 s, Le 6 s, `use_wes` always true.

| Subset | n | Overrides | Measures |
|---|---|---|---|
| A | 160 | none (recipe B, Person present) | SI-SNRi, TSOS |
| B | 80 | Person present, 1-3 talkers, SIR 0 dB, SNR 5 dB | picked rate at SIR 0 dB, SI-SNRi |
| C | 60 | Person absent | absent attenuation |
| D | 60 | Person present, own talker on at +6 dB, active 40%, own enrolled | own-voice suppression, SI-SNRi |
| E | 40 | Everyone (with talkers) | SI-SNRi |

Stored in `ml/cache/val_v1.npz` (about 0.8 GB) - `mix, ref` f32 [400, 64000]; `talkers` f16 [400, 3, 64000] and
`own` f16 [400, 64000] (metric signals, 8.9); `vad_t, vad_o` u8 [400, 1000]; talker VADs u8 [400, 3, 1000];
`enr_t` f32 [400, 96000], `tok_idx_t, tok_val_t`; `enr_o` f32 [400, 128000], `tok_idx_o, tok_val_o`;
`e_t, e_o` f32 [400, 256] (WeSpeaker computed once at render time; validation speakers are not in the own cache);
flags `has_t, has_o, use_wes, mode, present, person`; `subset` u8; `info` JSON. `validate()` runs the offline model
on MPS in batches of 8 (about 1 min, estimate) and computes the 12.2 metrics. Val SI-SNRi = median over subsets A,
B, D, E.

---

## 12. Evaluation (`python -m ml.evaluate`, metrics.py, contract 9)

### 12.1 Command and outputs

`python -m ml.evaluate --esw ml/exports/v1/earshot.esw --suites all --out docs/eval/report-v1.md` evaluates the
exported weights (loaded with `load_esw`), so the numbers belong to the file that ships. `--ckpt PATH` evaluates a
checkpoint during development. Rendered suites are cached in `ml/cache/eval/<suite>_v1.npz` (keyed by
`RECIPE_VERSION`). Per-mixture results go to `ml/runs/eval/<suite>.csv`; the report has one table per contract 9 row
with value, pass threshold, goal and PASS / FAIL, plus the credits and license notes of section 0.

### 12.2 Metric definitions (metrics.py, numpy float64)

- `si_snr(e, r)` - remove means; `a = <e, r> / (<r, r> + 1e-12)`; `10 log10(|a r|^2 / (|e - a r|^2 + 1e-12))`.
  `si_snri = si_snr(y, s) - si_snr(x, s)`, all signals aligned (the offline output is aligned with the input).
- Picked (per 2 s chunk) - non-overlapping 32,000-sample chunks where the target is active in at least 25% of hops
  and at least one competing talker is too; picked if `si_snr(y_c, s_c) > si_snr(y_c, r_ic)` for every competing
  talker i active in at least 25% of hops. Picked rate = picked / eligible chunks; confusion rate = 1 - picked rate.
- Absent attenuation - `10 log10(sum x^2 / sum y^2)` per absent clip; report median and percent under 10 dB.
- Own-voice suppression - over hops with `vad_o = 1` and `vad_t = 0`, `10 log10(sum o^2 / sum y^2)` with o the own
  image at the mic; clips with under 0.2 s of such hops are skipped; report median and 5th percentile (the contract's
  row reads "5th percentile >= 18 dB", that is 95% of clips reach 18 dB; report column `own_db_p05`).
- TSOS - STFT 512 / 128 Hann of s and y; target-active frames have energy within 30 dB of the loudest frame; a frame
  is over-suppressed if `sum relu(|S| - |Y|)^2 > 0.1 sum |S|^2`; TSOS = over-suppressed / active frames.
- ESTOI gain - `pystoi.stoi(s, y, 16000, extended=True) - pystoi.stoi(s, x, 16000, extended=True)`.
- PESQ - `pesq.pesq(16000, s, y, "wb")` on SNR >= 5 dB cells, reported only, internal R&D use (ITU / OPTICOM
  licensing, data-training.md 6.1); clips where pesq raises are skipped and counted.
- Envelope correlation (DiPCo) - 20 ms RMS frames at a 10 ms hop of each signal band-passed 100 to 4,000 Hz (4th-order
  Butterworth); Pearson correlation, maximized over lags of -100 to +100 ms.

### 12.3 Suites

| Suite | Build (all seeded, test mode 8.9, offline forward on MPS unless noted) | Reported rows |
|---|---|---|
| restaurant | 2,000 mixtures of 8 s, seed 2027. Targets, talkers, babble and own voice from the 48 held-out speakers (VCTK test 8 + LibriSpeech test-clean 40). Noise - noise_test (DEMAND SCAFE, FSD50K test) 50% and WHAM! test (eval only) 50%; babble in 70%. Grid - SNR (-5, 0, 5, 10), SIR (-8, -5, 0, 5, 10), talkers (1, 2, 3), own voice (off, on at U(0, 12) dB), room (sim_test in_hand, sim_test on_table, real_test restaurant / bar / indoor rows), enrollment (clean, noisy 10 dB, noisy 0 dB; 6 s). 1,080 cells; mixture i < 1,080 takes cell i, the remaining 920 take random cells. Own voice is always enrolled. | SI-SNRi median all cells; median for "2-talker + noise" = 1 competing talker, SNR 0 or 5, noisy enrollment; picked at SIR 0 / -5; confusion at SIR 0; own-voice suppression (own on); TSOS; ESTOI gain (SNR 0-5); PESQ (SNR >= 5); every factor's marginal medians |
| absent | 500 mixtures from the same generator, random cells, target removed, seed 2028 | attenuation median, % under 10 dB |
| libri2mix | data/eval/libri2mix_test.jsonl, 3,000 definitions x 2 targets; sources with the given gains, offsets and `length` (min mode), **no noise** (clean); enrollment = the definition's enrollment utterance for that target, first 10 s; no own voice | SI-SNRi median |
| accents | 500 mixtures each with EdAcc test and VoxPopuli accented speakers as targets (speakers with at least 2 utterances; enrollment from their other utterances, cycled to 6 s), everything else as restaurant | median vs the all-cells median |
| stability | one 60 min stream (seed 2029) - target active about 40% with pauses, 2 talkers, DEMAND SCAFE noise looped with 1 s crossfades at SNR 5 dB, one sim_test room; run with `StreamRef` on CPU (20 to 30 min, estimate) | any NaN; SI-SNRi of minutes 0-5 vs 55-60 |
| dipco | 12.5 | absent-segment attenuation, envelope win rate, 10 listening clips |
| parity | reload each `.est` next to the `.esw`, rerun `StreamRef` from its inputs, compare with its stored outputs and tolerances | PASS / FAIL (the C side is checked by WP-ENGINE) |

### 12.4 PESQ and STOI packages (checked 2026-11-07)

Neither is installed in the venv, and neither is in the uv or pip cache, so neither can be installed with no
network today. `uv pip install --dry-run --python .venv/bin/python pystoi pesq` resolved to exactly two new packages,
`pystoi==0.4.1` and `pesq==0.0.4`, with every existing pin unchanged (measured dry run). pystoi 0.4.1 is a pure-Python
wheel (MIT, needs only numpy and scipy); pesq 0.0.4 exists on PyPI only as a source package, so it compiles with
Xcode's clang during install. Plan - one online `uv pip install --python .venv/bin/python -r ml/requirements-eval.txt`,
then confirm the pins (torch 2.7.1, torchaudio 2.7.1, coremltools 9.0, numpy 2.5.4, scipy 1.18.1) with a one-line
import check; afterwards both run offline and uv can reinstall them from its cache with `--offline`. ESTOI is a
release row, so pystoi is required for ML-2. PESQ is reported only; if pesq fails to build, the report says "PESQ not
available" and nothing else changes. Both are imported lazily inside metrics.py.

### 12.5 DiPCo real dinner-party check

- Data on disk - far-field files for S01, S03, S06, S07, S08 and close-talk files only for S03 P09 and S08 P25, P26,
  P27; segment times in data/eval/dipco_segments.jsonl (3,405 lines). Targets are therefore P09, P25, P26 and P27.
- Enrollment - from the target's close-talk file, segments in the first 5 minutes where no other participant's
  segment overlaps, concatenated with 0.2 s gaps up to 6 s of `es_vad` speech; that span is excluded from scoring.
  Own voice not enrolled (no phone owner in DiPCo).
- Input - the session's far-field channel, scaled to -26 dBFS RMS, processed in 60 s chunks with 8 s of warm-up
  discarded per chunk (offline model; long-state behavior is covered by the stability suite).
- Absent-target windows - 2 s windows where the target has no segment within 0.25 s and other participants talk for
  at least 1 s; attenuation `10 log10(sum x^2 / sum y^2)`; median per target and pooled.
- Envelope win rate - 2 s windows where the target talks at least 1 s and at least one other participant with a
  close-talk file talks at least 0.5 s (only S08 has such pairs); a win if `env_corr(y, close_target)` beats
  `env_corr(y, close_j)` for every such participant j.
- Listening - 10 clips of 10 s (input and output WAVs, mixed windows) written to `ml/runs/eval/dipco_listen/` for the
  owner, plus a CSV of each clip's numbers.

---

## 13. Threshold calibration (`python -m ml.tools.calibrate_thresholds --esw ml/exports/v1/earshot.esw`)

Built from validation speakers and sim_val rooms, seed 2030. Every enrollment is a simulated full capture of 6 to 15 s
with 0.5 to 3 s of lead-in before the person speaks (background only), passed through `ml/earshot/quality.py`, which
implements the contract 6 check formulas exactly as the app does - trim to the speech span, best 10 s window, round
down to 0.5 s, windows of 24,000 samples every 8,000, `sbr_db` and `clip_pct` on the full capture (lowest-10% fallback),
own-window removal for targets, `speech_s` from the voiceprint after that removal. WeSpeaker runs through the same
PyTorch module.
- Good set - 500 single-talker enrollments, SNR U(5, 25) dB, room as 8.5, no weaker talker.
- Bad sets (500 each) - speech under 4 s (VAD speech 2 to 3.9 s; checks that the fixed speech_s rule rejects them);
  noisy captures with mixing SNR U(-6, 0) dB (for stationary noise `sbr_db` is roughly `10 log10(1 + SNR)`, so these
  score about 1 to 3 dB; the calibrated sbr fail boundary lands there); two equal
  talkers (second talker 0 to 3 dB below); the phone owner's voice (enrollment made from the own-voice speaker, scored
  against that speaker's own e); 2% of samples clipped.
- Rule per check - the pass boundary is set so 95% of the good set passes; the fail boundary is set so at least 80%
  of the matching bad set fails, but never beyond the pass boundary; if both cannot hold, the fail boundary equals the
  pass boundary and the tool prints "constraint not met" for that check. Only `sbr_db`, `one_voice` and `own_cos` are
  calibrated; `speech_s` ([4, 5], the contract 5.3 accept limit), `dup_cos` (0.75) and `clip_pct` ([0.1, 1.0]) stay at
  the contract values.
- `locked_cos` - cosine of `embed_g(model output, last 3 s)` to the enrolled g when the output follows a different
  speaker (impostor set, 1,000 pairs); value = 99th percentile, so 1% false accepts.
- Writes `ml/runs/main/thresholds.json` (contract 6 format, `model_hash` of the given `.esw`) and a CSV of all scores;
  `export --thresholds` copies it into `ml/exports/v1/`. Run time about 15 min (estimate).

---

## 14. Tests (`.venv/bin/python -m pytest ml/tests -q`, CPU, about 3 to 5 min, estimate)

Parity tests use `init_random_for_parity(seed 0)` so the zero-initialized paths are exercised. Tolerances are the
contract's where it gives one.

| File | Proves |
|---|---|
| `test_dsp.py` | windows equal the contract formulas (1e-7); `sum` of `wa * ws` over 64-sample shifts on the synthesis span equals 1 (1e-6); `istft(stft(x))[:L-64] == x[:L-64]` within 1e-6 on 10 s of noise and speech; frame counts (`stft` T = L/64, `stft_tok` F = `num_frames`); `early_rir` length, fade shape and direct-path sample; high-pass -3 dB at fc and peaking gain at f0 within 0.1 dB |
| `test_vad.py` | `num_frames` for n = 575, 576, 1088, 32000, 96000, 128000, 160000 (0, 2, 3, 63, 188, 250, 313); `es_vad` of 575 samples returns an empty array; `es_vad` on fixed vectors (all zeros gives all 0; silence with a 0.5 s tone burst gives exactly the hand-computed frames incl. dilation; a ramp checks the P10 index and the strict `>`); `token_groups` (10 kept frames give 3 groups; 104 groups subsample to `floor(i * 104 / 48)`; fewer than 4 flagged) |
| `test_model.py` | parameter total 2,575,108 and each part of 3.1; output shapes; null rules (Everyone, no own, no WeSpeaker each change only the intended inputs, and gradients reach the used null parameters); attention masking (changing an invalid token's K or V changes nothing, max diff 0); forget-gate bias sums to +1 |
| `test_stream.py` | streaming identity `y_stream[n+64] == y_offline[n]` within 1e-5 on 10 s (CPU fp32), heads within 1e-5, for all 4 cases; all-pass `StreamRef` reproduces the input delayed by 64 samples within 1e-5; `pack_cond` token order and zero padding; MPS vs CPU offline forward on random_v1, 4 s, all 4 cases, max abs diff <= 1e-4 (skipped without MPS; training, validation and evaluation run on MPS, contract 11) |
| `test_causal.py` | for t in {6400, 6437, 32000} add N(0, 1) noise to x[t:]; offline output unchanged before `64 floor(t/64) - 64` and streaming output unchanged before `64 floor(t/64)` (max diff <= 1e-6), and changed right after (> 1e-3). This is the precise form of "input after t never changes output before t": the offline output is aligned with the input and may use up to 127 samples ahead (the 8 ms algorithmic latency); the streamed output is causal at hop granularity |
| `test_export.py` | container round trip for f32, u8 and i32 (values, shapes, 64-byte offsets, `data_sha256`); a flipped data byte and a wrong magic are rejected; `CONFIG` hash equals 2.1; 270 tensors in `EXPORT_ORDER` with contract shapes; model -> `.esw` -> model gives equal outputs (1e-6) and a byte-identical re-export; random init has the zero-init tensors filled with std near 0.1; `make_testvecs` writes every array of 7.4 with the right dtype and shape, `meta.tolerances` whose key set equals exactly the contract 3.3 list (both files), and `meta.weights_sha256` equal to the `.esw` `data_sha256`; `export --testvec` reproduces the same `out.*` from the loaded `.esw` (1e-6) |
| `test_wespeaker.py` | strict load of `avg_model.pt`; unit-length output; fbank vs `torchaudio.compliance.kaldi.fbank` within 1e-4 (8.1e-5 measured before); Core ML `CPU_AND_GPU` vs PyTorch cosine >= 0.9999 on 5 clips (skipped with a message if the `.mlpackage` is not built yet) |
| `test_data.py` | the same (seed, index) renders bit-identical arrays in two separate spawn processes, a different index differs; license gate (synthetic NC train row raises, NC test row passes, eval row in train raises); train / val / test speakers disjoint; enrollment rows never overlap target rows; realized SIR, SNR and own level within 0.1 dB of the drawn values (no clipping, `info`); own voice never in `ref`; absent gives `ref == 0` and `vad_t == 0`; Everyone `ref` equals the sum of talker refs; dry room without noise gives `ref` equal to the target part of `mix` (1e-6); every token group has `n_avail >= 4`; no enrollment contains the same source sample twice or shares a row with the target stream (8.3 step 4); the own talker never uses a cache row (6.3); music-only and noise-only examples hit the drawn SNR within 0.1 dB; batch keys, dtypes and shapes of 8.7 |
| `test_losses.py` | `tsnr` of a perfect output is -30 dB (the cap); `absent` of a zero output is `10 log10(10^-2.5 sum x^2)`; `mrstft` and `oversup` are 0 for identical signals and `oversup` is 0 when the output is louder; the weights equal the section 9 table (`mrstft` 1); with its weight set to 0 in the test (the A/B harness's tsnr_only path), `mrstft` is logged with a finite value but changes neither the total nor any gradient and runs without autograd, while at weight 1 it runs with autograd and changes the gradient of y; masks switch terms on and off as in section 9; AAM logit of the true class equals `30 cos(theta + 0.2)`; a batch holding one Everyone, one no-own and one absent example gives a finite loss and finite gradients for every parameter (`tok.*`, `enc.*` included), and AAM ignores the Everyone row |
| `test_metrics.py` | `si_snr` known values (scaled copy gives a very large value, orthogonal noise at a known level gives its SNR); picked logic on synthetic chunks; TSOS on a half-gain output; envelope correlation lag search; `quality.py` reproduces every value of `quality_vectors.json` (contract 6) |
| `test_train.py` | 3 steps on CPU (B = 2, 1 s, workers 0, tiny steps config, stage B recipe so Everyone, no-own and absent examples occur) run with a finite loss and no NaN skip; save at step 2, resume, and step 3 equals the uninterrupted run's step 3 (same batch indices, parameters within 1e-6) |
| `test_gates.py` | the smoke and pilot gate logic on made-up rows (pilot - rising at 6k passes, flat or falling fails, a NaN skip fails, a missing or NaN validation fails, step time never fails it, `main_steps_for_40h` by hand); reads `ml/runs/smoke/gate_smoke.json` and `ml/runs/main/gate_pilot.json` and asserts PASS (skipped if the run has not happened) |

---

## 15. Order of work and hand-offs

1. ML-0, about 3 days (estimate) - config, dsp, vad, model, stream_ref, fileio, export (random and test vectors),
   wespeaker and convert tool, quality.py, tests 1-7 of section 14. Hand `ml/exports/random_v1/` and the `.mlpackage` to WP-ENGINE
   and WP-IOS, and post the config hash and file sizes in docs/DECISIONS.md through the lead.
2. ML-1, about 4 days (estimate) - the data modules (sources, render, enroll, data) with their tests and 20 listened-to WAV renders
   (`python -m ml.tools.render_val --preview 20` writes mixture, reference and enrollment WAVs to
   `ml/runs/preview/`), own cache (10 to 20 min), validation set, losses, train.py, smoke (0.5 h), pilot (1.9 h).
   Needs `ml/assets/decim16k_fir.npy` before the pilot; the lead runs WP-ENGINE's table tool on day 0 and commits it
   (contract 11).
3. ML-2 - the lead shows the owner the contract summary, then the main run (38.5 h of compute planned), evaluation,
   calibration, `export --ckpt`, report. Re-export test vectors from v1 and tell WP-ENGINE to rerun parity.

## 16. Points this spec had to interpret (all resolved by the lead 2026-11-07, DECISIONS L)

Each point below was raised to the lead and is now contract text, so the contract governs it.
- Weights file name - `earshot.esw` in versioned folders (contract 3.2 heading fixed).
- Synthetic "other" noise (8.3 step 7), noise / music exclusivity and the background budget rule (8.3 step 9) -
  contract 8.1.
- `meta.tolerances` keys (with `dbg`), the `wespk_selftest.est` header, `model_hash`, the `dbg` tap points, the test
  cases and the export order reading - contract 3.1, 3.2, 3.3 and 6.
- Restaurant grid room factor (sim in hand, sim on table, real; 1,080 cells) and the 4 DiPCo targets with the envelope
  comparison on S08 only - contract 9.
- `ml/assets/decim16k_fir.npy` - the lead runs the table tool on day 0 and commits it (contract 11).
- Speaker pools (target >= 2 utterances and >= 14.5 s; own pool by `own_split`; AAM 3,758 classes), own cache recipe,
  own level U(0, 18) dB in in_hand rooms - contract 6 and 8.1.
- Head labels (`hop_vad`, 8.4), finite masked means and `torch.where` term selection (3.3, 9) - contract 8.2.
- Check formulas and calibration on full captures through `quality.py` (13) - contract 6.
- Stage A talker split 15/45/40, loud scene = SNR <= 5 dB, 0.1 s floor for `gap` and `own`, 10% of Everyone examples
  with nobody talking, target partially active in 25% of present examples, validation sources (11), the data module
  split and the `Draws` dataclass, `ml/requirements-eval.txt` (12.4) - DECISIONS L rows.


# Earshot v1 - WP-ENGINE specification (20-engine)

Date - 2026-11-07. Owner - WP-ENGINE (the C engineer). Parent - docs/spec/00-contract.md (binding). This file expands
contract sections 2, 3, 4, 7, 9 and 11 for the engine. Every number, name, shape and signature here is copied from the
contract; where the contract is silent this file decides, and section 16 lists each such decision so the lead can check
it. If this file and the contract ever disagree, the contract wins and the disagreement is a bug in this file.

Labels - **measured** (file named), **analytic** (arithmetic or a script run while writing this file, script named),
**estimate**, **target**, as in the contract. Nothing here has run on the iPhone.

Reading order for a new engineer - section 1 (what files exist), 3 (numeric rules), 5 and 6 (the model hop by hop),
8 (the real-time pipeline), 14 (tests). Everything else is reference.

---

## 1. File plan (`engine/`)

The file list is the contract's (section 7, which now includes `src/internal.h`, `tests/rt_guard.c`, `tools/wav.h` and
`bench-results/`). Each engine source file stays under about 400 lines. WP-ENGINE writes only inside `engine/`; the
table tool's second output, `ml/assets/decim16k_fir.npy`, is generated and committed by the lead on day 0 together
with `src/resampler_tables.c` (contract 11), and only the lead regenerates either file.

| File | Owns | About lines |
|---|---|---|
| `include/earshot.h` | the public API, copied verbatim from contract section 4 (never edited here) | 150 |
| `src/internal.h` | private structs (`es_model`, `es_stream`, `es_cond`, `es_voiceprint`, `es_engine`), kernel prototypes, test hooks | 250 |
| `src/json_min.c` | JSON tokenizer (jsmn-style flat token array) and lookups by dotted path | 200 |
| `src/model.c` | container reader (`ESW1` / `EST1`), header validation, SHA-256 checks, tensor binding, `es_model_load/free`, `es_model_config_hash`, `es_model_data_hash`, `es_model_latency_samples`, synthetic test models | 400 |
| `src/nn.c` | `es_linear`, `es_layernorm`, `es_lstm_cell`, `es_lstm_seq` (uni and bidirectional), helpers | 250 |
| `src/stft.c` | windows, vDSP FFT wrapper (forward and inverse), feature computation | 150 |
| `src/stream.c` | `es_stream_new/free/reset`, `es_stream_hop` and its debug twin | 350 |
| `src/enroll.c` | `es_enroll_num_frames`, `es_vad`, token encoder, `es_enroll`, `es_embed_g`, `es_voiceprint_*` | 350 |
| `src/cond.c` | `es_cond_new/free/build` (FiLM generator, token list, K and V) | 150 |
| `src/resample.c` | rational polyphase FIR (`es_rs`), count-only stepping used by the prefill simulation | 150 |
| `src/resampler_tables.c` | generated by `tools/make_resampler_tables.py`, committed | generated |
| `src/ring.h` | SPSC rings (float and uint16) and the overwrite history ring, header-only | 150 |
| `src/pipeline.c` | `es_engine_new/free`, `render`, `run_hops`, `set_cond`, `set_params`, `set_mute`, `cue`, stats, `input_position`, reads | 400 |
| `src/post.c` | per-hop post-processing at 16 kHz (gate, crossfade sum, duck, AGC, gain cap, room, bypass) | 200 |
| `src/safety.c` | `es_ramp_*`, `es_limiter_*`, `es_softclip`, runaway guard, cue generator, `es_params_default`, `es_status_str` | 250 |
| `src/rt_thread.c` | worker thread (pthread, time-constraint policy, workgroup join, FTZ, single-threaded BLAS), start and stop | 150 |
| `src/probe.c` | latency probe (sweep table, capture, GCC-PHAT) | 250 |
| `src/selftest.c` | `es_selftest` | 150 |
| `tools/make_resampler_tables.py` | designs the FIR tables, writes `src/resampler_tables.c` and `ml/assets/decim16k_fir.npy` | 150 |
| `tools/wav.c`, `tools/wav.h` | WAV read (PCM16, PCM24, float32; channel 0) and write (float32 or PCM16) | 150 |
| `tools/esrun.c` | offline WAV through the stream or the full engine | 250 |
| `tests/test_units.c` | kernel, DSP, ring, loader and pipeline unit tests (needs no export) | 600 |
| `tests/test_parity.c` | every array and case of the `.est` files against the engine | 300 |
| `tests/test_rt.c`, `tests/rt_guard.c` | malloc and lock interposer (a dylib) plus the 100,000-hop RT test | 250 |
| `tests/test_fuzz.c` | one simulated hour of hostile input | 200 |
| `tests/test_latency.c` | impulse delay vs `es_engine_latency_ms`, prefill table | 200 |
| `tests/bench.c` | per-hop and per-callback timing, p50 / p99 / p99.9 / max | 150 |
| `Makefile` | targets `all test parity fuzz bench esrun ios-check clean` | 100 |

The engine core (src/ without the generated table) is about 3,600 lines. A sophomore can read one file at a time: each
file has a header comment saying what it owns, which thread may call it, and which contract section it implements.

---

## 2. Build, flags and how the iOS app compiles the engine

### 2.1 Flags (Makefile, macOS host)

```make
CC      = xcrun clang
CFLAGS  = -std=c11 -O2 -g -Wall -Wextra -Wshadow -Wstrict-prototypes -Werror \
          -DACCELERATE_NEW_LAPACK -Iinclude -Isrc
LDLIBS  = -framework Accelerate          # CommonCrypto is part of libSystem
SAN     = -fsanitize=address,undefined -fno-omit-frame-pointer   # used when SAN=1
```

- Never `-ffast-math`, `-Ofast`, `-ffinite-math-only` or `-funsafe-math-optimizations` (they reorder sums and let the
  compiler delete the `isfinite` checks the NaN guard relies on). Clang's default `-ffp-contract=on` (fused
  multiply-add inside one expression) stays; it is deterministic for a given binary.
- `make test SAN=1` builds test_units and test_parity with sanitizers. test_rt and bench never use sanitizers.
- Minimum OS - macOS 15 and iOS 18 for `BLASSetThreading`; the app targets iOS 26.0 so this is always met.

### 2.2 Makefile targets

| Target | Does | Needs an export |
|---|---|---|
| `all` | builds `build/libengine.a` (host, for tests only), `build/esrun`, all test binaries | no |
| `test` | test_units, test_latency, test_rt (10,000 hops), test_fuzz with `FUZZ_SECONDS=60`, and test_parity if `$(EXPORT)` exists | parity only |
| `parity` | `build/test_parity $(EXPORT)` with `EXPORT ?= ../ml/exports/random_v1` | yes |
| `fuzz` | `FUZZ_SECONDS=3600 build/test_fuzz` (one simulated hour, about 5 min on the M4, estimate) | no (synthetic model) or yes |
| `bench` | `VECLIB_MAXIMUM_THREADS=1 build/bench $(EXPORT)`, prints JSON lines | no (synthetic model) |
| `esrun` | builds `build/esrun` | no |
| `ios-check` | compiles every `src/*.c` with `xcrun --sdk iphoneos clang -target arm64-apple-ios26.0 $(CFLAGS) -c`, no link | no |
| `clean` | removes `build/` | no |

`libengine.a` exists only for the host tests; the app never links it.

### 2.3 iOS (owned by WP-IOS)

The app target compiles `engine/src/*.c` directly. WP-IOS owns `ios/project.yml`, and 30-ios 2.1 is its only
specification. It must compile `engine/src/*.c` with `-O2`, `-std=c11` and `-DACCELERATE_NEW_LAPACK` (the same flags
as `make ios-check`; without the define the iOS 26.0 SDK's `vecLib.h` selects the deprecated CBLAS interface, checked
in the SDK) and `-Wall -Wextra -Werror` for the engine sources, never `-ffast-math`.

- The `-O2` (GCC_OPTIMIZATION_LEVEL 2 in Debug too) keeps the engine fast in Debug builds (at `-O0` the hop would be
  several times slower, estimate).
- `ios/Stubs/earshot_stub.c` must be removed from the sources at integration; it defines the same symbols.
- `src/internal.h` is never included by Swift or by `AudioHost.c`; only `earshot.h` is public.

---

## 3. Numeric and real-time rules (apply to every file)

1. **float32 everywhere** on the per-hop path. Window design, VAD energies, the probe and the resampler design use
   double. No half precision anywhere in v1.
2. **No allocation after create.** `es_stream_new` and `es_engine_new` allocate every buffer (64-byte aligned,
   `posix_memalign`) and zero it. Enrollment, conditioning, loading, self-test and probe analysis may allocate (they
   are non-RT by contract 4.1).
3. **Linear layers.** Weights are stored `[in, out]` row-major by the exporter, so the engine never transposes. One
   helper, used for every Linear in the model:

```c
/* Y[M][out] = X[M][in] * W[in][out] + b. b may be NULL. Y must not alias X. */
static inline void es_linear(const float *X, int M, int in, int out,
                             const float *W, const float *b, float *Y) {
  if (b) for (int m = 0; m < M; m++) memcpy(Y + (size_t)m * out, b, sizeof(float) * (size_t)out);
  else   memset(Y, 0, sizeof(float) * (size_t)M * (size_t)out);
  if (M == 1) cblas_sgemv(CblasRowMajor, CblasTrans, in, out, 1.0f, W, out, X, 1, 1.0f, Y, 1);
  else        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, out, in,
                          1.0f, X, in, W, out, 1.0f, Y, out);
}
```

   The `[in, out]` layout was 1.1 to 4x faster than the PyTorch layout at these shapes (measured, bench/RESULTS.md 5).
   An accumulate variant `es_linear_acc` (beta 1, no bias copy) is used for the LSTM `h * w_hh` term and for residuals.
4. **Transcendentals.** vForce for arrays of 64 or more (`vvtanhf`, `vvexpf`, `vvlogf`, `vvsqrtf`); libm `expf`,
   `tanhf`, `logf` for scalars and short loops. Sigmoid is computed as `0.5f * tanhf(0.5f * x) + 0.5f` (exact identity,
   one vForce call covers all four LSTM gates). The parity tolerances (1e-4 per tensor) absorb the last-bit
   differences from PyTorch.
5. **Determinism.** Same binary, same inputs, same thread setup gives bit-identical outputs (test 14.2 runs every case
   twice and compares with `memcmp`). BLAS is single-threaded on the worker (`BLASSetThreading`) and in bench and
   parity (`VECLIB_MAXIMUM_THREADS=1`). Results may differ in the last bits between the M4 and the A18 Pro; the
   tolerances cover that.
6. **Denormals.** The worker sets FPCR.FZ (bit 24) once at thread start:
   `uint64_t r; __asm__ volatile("mrs %0, fpcr" : "=r"(r)); r |= 1ull << 24; __asm__ volatile("msr fpcr, %0" :: "r"(r));`
   inside `#if defined(__aarch64__)`. Parity tests run both with and without FZ; both must pass.
7. **RT-safe** means what contract 4.1 says. In this engine the only syscall on an RT thread is `semaphore_signal`
   (render) and `semaphore_wait` (worker idle, outside `run_hops`). Time is read with `mach_absolute_time()`
   (commpage, no syscall), converted with a `mach_timebase_info` ratio cached at create.
8. **Bounded loops.** Every loop on an RT thread has a compile-time or create-time bound (callback frames at most
   4,096 per chunk, at most 64 hops per `run_hops` call, at most 68 tokens).

---

## 4. Weight and test-vector loader (`model.c`, `json_min.c`)

### 4.1 Container reader (shared by `.esw` and `.est`)

```c
typedef struct { const char *name; int dtype; int ndim; int shape[4]; const void *data; size_t nbytes; } es_array;
typedef struct { const uint8_t *base; size_t n; const char *json; uint32_t json_len; size_t data_start;
                 es_array *arrays; int n_arrays; es_jtok *tok; int n_tok; } es_container;
es_status es_container_open(const void *buf, size_t n, const char magic[4], es_container *ct, char *err, size_t errlen);
const es_array *es_container_find(const es_container *ct, const char *name);
void es_container_close(es_container *ct);          /* frees arrays and tokens, never the input buffer */
```

Checks, in this order; the first failure returns and writes one line naming the field into `err`
(for example `"array blocks.2.t.w_hh: shape [128,256], expected [128,512]"`).

| # | Check | Status |
|---|---|---|
| 1 | `n >= 16`, magic equals `ESW1` or `EST1` as requested | ES_ERR_FORMAT |
| 2 | uint32 `format_version == 1` (little-endian read with `memcpy`, never a cast) | ES_ERR_VERSION |
| 3 | `16 + header_len <= n`, reserved == 0, header parses as one JSON object | ES_ERR_FORMAT |
| 4 | `format` is `earshot-weights` / `earshot-testvec`; `format_version` 1; `model` `esb16-df` | ES_ERR_FORMAT |
| 5 | `model_version` major == 1 (`"1.x.y"`) | ES_ERR_VERSION |
| 6 | `config` has exactly the 29 keys of contract 3.1, each equal to the compiled constant (ints exact; floats within 1e-12 relative; `band_widths` and `film_after` element by element) | ES_ERR_CONFIG |
| 7 | `config_hash` equals SHA-256 of the compiled canonical string (4.2) | ES_ERR_CONFIG |
| 8 | `data_start = round_up(16 + header_len, 64)`; `data_bytes == n - data_start` | ES_ERR_FORMAT |
| 9 | `data_sha256` equals `CC_SHA256(base + data_start, data_bytes)` | ES_ERR_FORMAT |
| 10 | every array - dtype in {f32, u8, i32}; `offset % 64 == 0`; `nbytes == prod(shape) * size(dtype)`; `offset + nbytes <= data_bytes`; offsets increasing in list order | ES_ERR_FORMAT |

`json_min.c` - a flat tokenizer (types object, array, string, number, true, false, null; each token holds start, end,
child count and parent index). Token capacity `header_len / 2 + 16` (each token needs at least two characters), so one
`malloc` per file. Lookups - `es_json_get(ct, parent, "key")`, `es_json_num`, `es_json_str_eq`, `es_json_arr_at`.
Strings are compared raw (the exporter writes only ASCII names); `\uXXXX` escapes are rejected with ES_ERR_FORMAT.
Numbers parse with `strtod` on a NUL-terminated copy of at most 64 characters.

### 4.2 Canonical config and its hash

`model.c` holds the compiled config as the exact string Python writes with
`json.dumps(config, sort_keys=True, separators=(",", ":"))`:

```c
static const char ES_CONFIG_CANON[] =
  "{\"D\":64,\"ana_rise\":448,\"att_after\":0,\"att_dim\":64,\"att_heads\":4,\"band_hidden\":64,"
  "\"band_widths\":[8,8,8,8,8,8,8,8,16,16,16,16,32,32,32,33],\"blocks\":4,\"df_bands\":12,\"df_taps\":3,"
  "\"feat_log_scale\":0.1,\"feat_pow\":0.3,\"film_after\":[0,2],\"g_dim\":128,\"hop\":64,\"ln_eps\":1e-05,"
  "\"mask_bound\":1.5,\"max_own_tokens\":16,\"max_target_tokens\":48,\"n_fft\":512,\"null_tokens\":4,"
  "\"sr\":16000,\"synth_len\":128,\"time_hidden\":128,\"tok_dim\":64,\"tok_hidden\":32,\"tok_hop\":512,"
  "\"tok_pool\":3,\"wespk_dim\":256}";
/* 470 bytes. SHA-256 = 089d72c06d244e1650d209453c91ddffde990b9589dd7a9c228f9e1ea3b93aec
   (analytic, computed with Python 3.12 json.dumps + hashlib while writing this file). */
```

A unit test hashes `ES_CONFIG_CANON` and compares with the literal above. `es_model_config_hash` returns this 64-char
hex string (owned by the model). Check 6 still runs field by field so the error message names the field. The loader
also keeps the verified header `data_sha256` (`char data_hash[65]` next to `config_hash`); `es_model_data_hash`
returns it. It is the contract 3.1 model_hash.

### 4.3 Binding the 270 tensors

`es_model_load` copies the whole data region into one 64-byte aligned arena with a single `memcpy` (so every tensor
keeps its 64-aligned offset and the caller may free the file buffer), then binds pointers by name:

```c
static int bind(es_model *m, const es_container *ct, const char *name, int d0, int d1, const float **dst);
/* finds the array, requires dtype f32 and shape [d0] (d1 == 0) or [d0, d1], sets *dst = arena + offset,
   marks the array used. Returns 0 or writes err and returns -1. */
```

`bind` is called once per tensor in the contract 3.2 order, generated by loops in this order: `enc` for k = 0..15
(lin.weight, lin.bias, ln.weight, ln.bias); `blocks` for b = 0..3 (tn, t, tf, bn, b with `_rev`, bf); `att`; `film`;
`null`; `dec` for k = 0..15 (ln, fc1, fc2); `heads`; `tok` (blk.tn, blk.bn, blk.t, blk.t `_rev`, blk.b, blk.b `_rev`,
blk.tf, blk.bf, proj, g). Lookup is by name, so the loader does not depend on file order, but the exporter writes this
order. After binding, any array not marked used is an extra tensor (ES_ERR_FORMAT, named); the count must be 270.
Shapes come from contract 3.2 with `W_k = [8]*8 + [16]*4 + [32]*3 + [33]` and `T_k = 3` for k < 12, else 1.

The model also precomputes, at load, `band_start[16]` (0, 8, ..., 224), `band_width[16]`, `band_taps[16]`,
`coef_off[16]` (prefix sums of `2 W_k T_k`, total 1,026), and the windows `wa[512]`, `ws[512]` (section 5).

```c
typedef struct { const float *w, *b; } es_lin;                 /* w [in][out], b [out] */
typedef struct { const float *w, *b; } es_ln;                  /* [64], [64] */
typedef struct { const float *wih, *whh, *bias; } es_lstmw;   /* [in][4H], [H][4H], [4H], gates i f g o */
struct es_model {
  void *arena; size_t arena_bytes; char config_hash[65], data_hash[65];
  int band_start[16], band_width[16], band_taps[16], coef_off[16];
  float wa[512], ws[512];
  es_lin enc_lin[16]; es_ln enc_ln[16];
  struct { es_ln tn, bn; es_lstmw t, b, b_rev; es_lin tf, bf; } blk[4];
  es_lin att_q, att_o, att_kv; const float *type_emb, *null_tok;           /* [3][64], [4][64] */
  es_lin film0, film2; const float *null_e_t, *null_g_t, *null_e_o, *null_g_o;
  struct { es_ln ln; es_lin fc1, fc2; } dec[16];
  es_lin heads;
  struct { es_ln tn, bn; es_lstmw t, t_rev, b, b_rev; es_lin tf, bf; } tok;
  es_lin tok_proj, tok_g;
};
```

### 4.4 Synthetic models (test hook, no export needed)

`es_status es_model_synthetic(uint32_t seed, int allpass, es_model **out)` (declared in internal.h, test-only) builds a
model with the same arena layout and pointers. Values come from xorshift32 (`x ^= x << 13; x ^= x >> 17; x ^= x << 5`)
mapped to uniform [-1, 1) and scaled by `1 / sqrt(fan_in)` for weights, 0.1 for biases, `1 + 0.1 u` for LayerNorm
weights, 0.1 for embeddings and null vectors. With `allpass = 1`, every `dec.{k}.fc2.weight` is zero and
`dec.{k}.fc2.bias` is `atanh(2/3) = 0.80471896` on the real tau = 0 entries (index `j` for j < W_k) and 0 elsewhere, so
every coefficient is `1.5 tanh(atanh(2/3)) = 1`: C_0 = 1, other taps 0, Y = X. This lets E-0 and E-2 (RT, fuzz,
latency, bench) run before ML-0 delivers `random_v1`. Parity always needs the PyTorch export.

---

## 5. STFT and iSTFT (`stft.c`)

### 5.1 Windows (computed once per model in double, stored float)

```c
static double hann(int n, int L) { return 0.5 - 0.5 * cos(2.0 * M_PI * n / L); }
for (int n = 0; n < 512; n++) {
  double a = n < 448 ? sqrt(hann(n, 896)) : sqrt(hann(n - 384, 128));
  m->wa[n] = (float)a;
  m->ws[n] = n < 384 ? 0.0f : (float)(hann(n - 384, 128) / a);
}
```

Facts the unit test checks - `wa[448..511]` equals `sqrt(H_128[64..127])`; the product `wa * ws` summed at hop 64 is 1
within 1e-6 (numpy gave 1 within 4e-16, docs/designs/product.md B.3); the smallest `wa` on the synthesis span is 0.0245
so `ws <= 1.0`.

### 5.2 FFT wrapper (vDSP, N = 512, log2n = 9)

Each `es_stream` (and each enrollment call) owns its own `FFTSetup` from `vDSP_create_fftsetup(9, kFFTRadix2)`, made
at create time and never shared between threads. Split buffers `re[256]`, `im[256]`.

```c
/* forward: x[512] real -> Xr[257], Xi[257], equal to torch.fft.rfft (unnormalized) */
DSPSplitComplex sp = { re, im };
vDSP_ctoz((const DSPComplex *)x, 2, &sp, 1, 256);
vDSP_fft_zrip(setup, &sp, 1, 9, kFFTDirection_Forward);
Xr[0] = 0.5f * re[0];  Xi[0] = 0;  Xr[256] = 0.5f * im[0];  Xi[256] = 0;
for (k = 1; k < 256; k++) { Xr[k] = 0.5f * re[k]; Xi[k] = 0.5f * im[k]; }

/* inverse: Yr, Yi [257] -> y[512], equal to torch.fft.irfft(n=512); Im of bins 0 and 256 ignored */
re[0] = Yr[0]; im[0] = Yr[256];
for (k = 1; k < 256; k++) { re[k] = Yr[k]; im[k] = Yi[k]; }
vDSP_fft_zrip(setup, &sp, 1, 9, kFFTDirection_Inverse);
vDSP_ztoc(&sp, 1, (DSPComplex *)y, 2, 256);
vDSP_vsmul(y, 1, &(float){1.0f / 512.0f}, y, 1, 512);
```

vDSP's real forward transform returns twice the DFT and its inverse returns N times the inverse DFT (Apple vDSP
packing convention, doc), hence 0.5 and 1/512 as contract E-0 says. The unit test compares against a float64 naive
DFT; if it fails, fix the scale factors, never the tolerance.

### 5.3 Features (one frame, 257 bins, written into `feat[771]` at index `3*bin + f`)

```c
vDSP_zvmags(&(DSPSplitComplex){Xr, Xi}, 1, p, 1, 257);          /* p = |X|^2 */
vvsqrtf(mag, p, &n257);                                            /* |X| */
vDSP_vsadd(mag, 1, &(float){1e-8f}, t, 1, 257); vvlogf(t, t, &n257);
vDSP_vsmul(t, 1, &(float){-0.7f}, t, 1, 257);   vvexpf(t, t, &n257); /* s = (|X| + 1e-8)^-0.7 */
vDSP_vmul(Xr, 1, t, 1, feat + 0, 3, 257);                          /* Re c */
vDSP_vmul(Xi, 1, t, 1, feat + 1, 3, 257);                          /* Im c */
vDSP_vsadd(p, 1, &(float){1e-8f}, t, 1, 257); vvlogf(t, t, &n257);
vDSP_vsmul(t, 1, &(float){0.1f}, feat + 2, 3, 257);                /* 0.1 ln(|X|^2 + 1e-8) */
```

Because the 16 bands are contiguous bin ranges and the band input is flattened bin-major, band k's encoder input is
simply `feat + 3 * band_start[k]`, length `3 * W_k`. The same function is used by the hop and by the token encoder.

### 5.4 Streaming synthesis (per hop, after the deep filter)

```c
es_irfft(s, Yr, Yi, frame);                                    /* 5.2 */
for (n = 0; n < 128; n++) s->ola[n] += frame[384 + n] * m->ws[384 + n];
memcpy(out, s->ola, 64 * sizeof(float));
memmove(s->ola, s->ola + 64, 64 * sizeof(float));
memset(s->ola + 64, 0, 64 * sizeof(float));
```

Identity this must satisfy (contract 2.1) - with an all-pass model, `out` of hop j equals input samples
`64j - 64 .. 64j - 1`, so the stream reproduces the input delayed by 64 samples (max error 1e-5; the engine achieves
about 1e-7 with the synthetic all-pass model, estimate). Offline helpers `es_stft_offline` and `es_istft_offline`
(test-only, in stft.c) implement contract 2.1's offline definitions so test_units can check
`y_stream[n + 64] = y_offline[n]` in C as well.

---

## 6. One hop of the model (`stream.c`, `nn.c`)

### 6.1 State and scratch (`es_stream`, all allocated in `es_stream_new`)

State (zeroed by `es_stream_reset`, contract 2.6) - `ana[512]`, `xhist[2][128][2]` (index 0 = t-1, (re, im)),
`h_t[4][16][128]`, `c_t[4][16][128]`, `ola[128]`: 17,536 floats. Scratch (never needs zeroing) - `Xr, Xi [257]`,
`feat[771]`, `z, zn [16][64]`, `gates [16][512]`, `gxf, gxb [16][256]` (band LSTM input projections), `lw[448]` (band
LSTM work - h, c, gates, tanh c), `bcat [16][128]`, `tmp [16][128]`, `q[64]`, `sc[4][68]`, `a[64]`, `hid [16][128]`,
`coef[1026]`, `Yr, Yi [257]`,
`frame[512]`, `re, im [256]`, `p, mag, t [257]`, plus the stream's own `FFTSetup`. `es_stream_reset` uses `memset` only
(RT-safe).

### 6.2 Kernels (`nn.c`)

```c
/* LayerNorm over C = 64, R rows, two-pass like PyTorch: mean, then mean of squared deviations. */
void es_layernorm(const float *X, int R, const es_ln *ln, float *Y) {
  for (int r = 0; r < R; r++) {
    const float *x = X + 64 * r; float *y = Y + 64 * r, mean, var;
    vDSP_meanv(x, 1, &mean, 64);
    float nm = -mean; vDSP_vsadd(x, 1, &nm, y, 1, 64);
    vDSP_measqv(y, 1, &var, 64);
    float inv = 1.0f / sqrtf(var + 1e-5f);
    vDSP_vsmul(y, 1, &inv, y, 1, 64);
    vDSP_vma(y, 1, ln->w, 1, ln->b, 1, y, 1, 64);
  }
}

/* R independent LSTM cells, hidden H. g = R x 4H preactivations (i, f, g, o), overwritten.
   h, c = R x H, updated in place. th = R x H scratch. Sigmoid via tanh (rule 3.4). */
void es_lstm_cell(int R, int H, float *g, float *h, float *c, float *th) {
  int n = R * 4 * H, m = R * H;
  for (int r = 0; r < R; r++) {                      /* halve i, f, o; keep g */
    float *gr = g + (size_t)r * 4 * H;
    vDSP_vsmul(gr, 1, &(float){0.5f}, gr, 1, 2 * H);
    vDSP_vsmul(gr + 3 * H, 1, &(float){0.5f}, gr + 3 * H, 1, H);
  }
  vvtanhf(g, g, &n);
  for (int r = 0; r < R; r++) {
    float *gr = g + (size_t)r * 4 * H, *cr = c + (size_t)r * H;
    for (int j = 0; j < H; j++) {
      float i = 0.5f * gr[j] + 0.5f, f = 0.5f * gr[H + j] + 0.5f, gg = gr[2 * H + j];
      cr[j] = f * cr[j] + i * gg;
    }
  }
  vvtanhf(th, c, &m);
  for (int r = 0; r < R; r++)
    for (int j = 0; j < H; j++)
      h[(size_t)r * H + j] = (0.5f * g[(size_t)r * 4 * H + 3 * H + j] + 0.5f) * th[(size_t)r * H + j];
}
```

```c
/* B independent sequences of length T, zero initial state. Row (t, b) of the input projections (x * w_ih + bias,
   computed beforehand for all steps with one es_linear) is gx + t*gx_t + b*gx_b (4H floats); the hidden output of
   row (t, b) goes to hs + t*hs_t + b*hs_b (H floats). work holds B*7*H floats (h, c, gates, tanh c). */
void es_lstm_seq(const float *gx, int gx_t, int gx_b, float *hs, int hs_t, int hs_b,
                 int T, int B, int H, const float *whh, int reverse, float *work);
```

Each step (t = 0..T-1, or T-1..0 when `reverse`) copies the B rows of step t into `gates`, adds `h * whh`
(`es_linear_acc`, M = B), runs `es_lstm_cell(B, H, ...)` and copies h out. Strides used - stream band path B = 1,
T = 16, H = 64, `gx_t = 256`, `hs_t = 128` (`hs = bcat` fwd, `bcat + 64` rev); token time path B = 16 (bands), T = F,
H = 32, `gx_t = 16*128`, `gx_b = 128`, `hs_t = 16*64`, `hs_b = 64`; token band path B = F, T = 16, `gx_t = 128`,
`gx_b = 16*128`, `hs_t = 64`, `hs_b = 16*64` (rev outputs at offset 32).

### 6.3 `es_stream_hop(s, c, in, out, heads)` in exact order

Precondition - `c != NULL` and `c->n_tokens >= 4`; otherwise `out` and `heads` are zeroed, state is untouched, return 0.
`m` is `s->model`. Shapes are in brackets.

| # | Step | Calls | Buffers |
|---|---|---|---|
| 1 | Shift input - `memmove(ana, ana + 64, 448 floats)`, copy `in[64]` to `ana[448..511]` | memmove, memcpy | ana [512] |
| 2 | Window - `frame[n] = ana[n] * wa[n]` | vDSP_vmul | frame [512] |
| 3 | FFT (5.2) | vDSP_ctoz, vDSP_fft_zrip | Xr, Xi [257] |
| 4 | Features (5.3) | vDSP, vForce | feat [771] |
| 5 | Band encoder - for k: `es_linear(feat + 3 s_k, 1, 3 W_k, 64, enc_lin[k].w, enc_lin[k].b, z[k])`, then `es_layernorm(z[k], 1, &enc_ln[k], z[k])` | 16 sgemv | z [16][64] |
| 6 | Blocks b = 0..3, each 6a to 6e | | |
| 6a | Time path - `zn = LN(z, blk[b].tn)`; `gates = zn * t.wih + t.bias` (M = 16); `gates += h_t[b] * t.whh` (M = 16); `es_lstm_cell(16, 128, gates, h_t[b], c_t[b], tmp)`; `z += h_t[b] * tf.w + tf.b` | 3 sgemm | gates [16][512], h_t[b], c_t[b] [16][128] |
| 6b | Band path - `zn = LN(z, blk[b].bn)`; `gxf = zn * b.wih + b.bias`, `gxb = zn * b.wih_rev + b.bias_rev` (M = 16); forward LSTM over k = 0..15 from zero state into `bcat[k][0..63]`; reverse over k = 15..0 into `bcat[k][64..127]`; `z += bcat * bf.w + bf.b` (M = 16) | 3 sgemm, 32 sgemv | gxf, gxb [16][256], bcat [16][128] |
| 6c | If b == 0, FiLM 0 - `z[i] = z[i] * (1 + film[i]) + film[1024 + i]` for i < 1024 (point 0: gamma at offset 0, beta at 1024) | plain loop | z |
| 6d | If b == 0, attention (6.4) | | q, sc, a |
| 6e | If b == 2, FiLM 1 - same with gamma at offset 2048, beta at 3072 | plain loop | z |
| 7 | Heads - `mu[d] = (sum_k z[k][d]) / 16`; `h2 = mu * heads.w + heads.b`; `heads[i] = 1 / (1 + expf(-h2[i]))` | sgemv | 2 floats |
| 8 | Decoder - for k: `zn[k] = LN(z[k], dec[k].ln)`; `hid[k] = zn[k] * fc1.w + fc1.b`; then `vvtanhf(hid, 2048)`; for k: `coef + coef_off[k] = hid[k] * fc2.w + fc2.b`; then `vvtanhf(coef, 1026)` and `coef *= 1.5` | 32 sgemv | hid [16][128], coef [1026] |
| 9 | Deep filter (6.5) and history update | plain loops | Yr, Yi [257], xhist |
| 10 | iFFT, synthesis window, overlap-add, emit (5.4) | vDSP | ola [128], out [64] |
| 11 | NaN guard - if any of `out[0..63]`, `heads[0..1]` is not finite: `es_stream_reset(s)`, zero `out` and `heads`, return 1 | isfinite loop | |

The residual adds in 6a and 6b are done by passing `z` itself as the accumulate target of `es_linear_acc` after adding
the bias (`vDSP_vadd` of the bias rows), so no extra copy is needed. The FiLM index rule of contract 2.3,
`((p*2 + s)*16 + k)*64 + d`, gives exactly the offsets 0 / 1024 / 2048 / 3072 used in 6c and 6e.

### 6.4 Cross-attention (after FiLM 0)

```c
es_linear(z, 1, 1024, 64, m->att_q.w, m->att_q.b, q);    /* z flattened k*64 + d */
int n = c->n_tokens;                                       /* 4..68 */
for (int h = 0; h < 4; h++) {
  const float *K = &c->k[h][0][0], *V = &c->v[h][0][0], *qh = q + 16 * h;
  float mx = -INFINITY, sum = 0.0f, *s = sc[h];
  for (int j = 0; j < n; j++) { float d = 0; for (int e = 0; e < 16; e++) d += qh[e] * K[j * 16 + e];
                                s[j] = 0.25f * d; mx = fmaxf(mx, s[j]); }
  for (int j = 0; j < n; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
  float inv = 1.0f / sum;
  for (int e = 0; e < 16; e++) { float acc = 0; for (int j = 0; j < n; j++) acc += s[j] * V[j * 16 + e];
                                 a[16 * h + e] = acc * inv; }
}
vDSP_vadd(z, 1, m->att_o.b, 1, z, 1, 1024);              /* z += o.bias */
cblas_sgemv(CblasRowMajor, CblasTrans, 64, 1024, 1.0f, m->att_o.w, 1024, a, 1, 1.0f, z, 1);  /* z += a * o.w */
```

Plain loops are used on purpose (8,704 MACs, a few microseconds, estimate); they read like the formula.

### 6.5 Deep filter and history

```c
for (int k = 0; k < 16; k++) {
  int s0 = m->band_start[k], W = m->band_width[k], T = m->band_taps[k];
  const float *cr = coef + m->coef_off[k], *ci = cr + W * T;
  for (int j = 0; j < W; j++) {
    int f = s0 + j;
    float yr = cr[j] * Xr[f] - ci[j] * Xi[f], yi = cr[j] * Xi[f] + ci[j] * Xr[f];       /* tau = 0 */
    for (int tau = 1; tau < T; tau++) {                                                    /* f < 128 here */
      float xr = xhist[tau - 1][f][0], xi = xhist[tau - 1][f][1];
      float a = cr[tau * W + j], b = ci[tau * W + j];
      yr += a * xr - b * xi;  yi += a * xi + b * xr;
    }
    Yr[f] = yr; Yi[f] = yi;
  }
}
memcpy(xhist[1], xhist[0], 128 * 2 * sizeof(float));                 /* t-1 becomes t-2 */
for (int f = 0; f < 128; f++) { xhist[0][f][0] = Xr[f]; xhist[0][f][1] = Xi[f]; }
```

Bands 0 to 11 cover bins 0 to 127 exactly, so the history holds bins 0 to 127 only. At reset the history is zero, which
is contract 2.2 row 10 ("X before the first hop is 0").

### 6.6 Debug twin and cost

`int es_stream_hop_dbg(es_stream *s, const es_cond *c, const float in[64], float out[64], float heads[2],
const es_dbg *d)` (internal.h) is the real implementation; `es_stream_hop` calls it with `d = NULL`. When `d` is set it
copies `feat` after step 4, `z` after step 5 (`z_enc`), `z` at the end of each block b including any FiLM or attention
that follows it (`z_blk[b]`, the value handed to block b + 1 or to the heads and decoder), and `coef` after step 8.
These are the `dbg.*` arrays of contract 3.3.

Cost - 12.00 M MAC per hop (analytic, contract 2.5), about 200 BLAS calls; expected 0.19 ms median on an M4 P-core and
0.2 to 0.3 ms on the A18 Pro (estimate, contract 2.5). Bench gate - median at most 0.25 ms for one instance on the M4
(target, contract 11).

---

## 7. Enrollment and conditioning (`enroll.c`, `cond.c`; non-RT, may allocate)

### 7.1 `es_vad(pcm, n, mask)` (contract 2.4, bit-for-bit rule)

```c
int F = es_enroll_num_frames(n);                  /* n < 576 ? 0 : (n - 64) / 512 + 1 */
if (F == 0) return;                               /* contract 2.4: nothing read or written */
for f < F: sum = 0.0 (double); for i < 512: { long j = 512L * f + i - 448; double x = j >= 0 ? pcm[j] : 0.0; sum += x * x; }
           E[f] = 10.0 * log10(sum / 512.0 + 1e-10);
sorted = copy of E, qsort ascending; P10 = sorted[(F - 1) / 10];      /* integer division == floor(0.1 (F - 1)) */
thr = fmax(P10 + 6.0, max(E) - 40.0);
raw[f] = E[f] > thr;  mask[f] = raw[f-1] | raw[f] | raw[f+1]   (out-of-range neighbours count as 0)
```

All energy math is double, sums in index order. `(F - 1) / 10` equals `floor(0.1 * (F - 1))` for every F (0.1 in
double is slightly above 0.1, so the product never rounds below an integer); WP-ML's `vad.py` must use the same.

### 7.2 Token encoder (`es_tokens`, shared by `es_enroll` and `es_embed_g`)

Inputs - `pcm[n]`, `mask[F]`, limit (48 target, 16 own). `n` must be at most 960,000 (60 s), else ES_ERR_ARG.

1. Frames - for f < F: `frame[i] = wa[i] * xp[512 f + i]` with `xp` = 448 zeros then `pcm` (zero past the end is never
   reached because F is computed from n). FFT (5.2), features (5.3), band encoder (step 5 of 6.3) give `z[F][16][64]`.
2. Time path - `zn = LN(z, tok.tn)` over all `F*16` rows; `gx = zn * tok.t.wih + bias` and `gxr = zn * tok.t.wih_rev +
   bias_rev` (one sgemm each, M = 16 F); `es_lstm_seq` with B = 16 sequences (the bands) of length T = F, H = 32,
   forward into `y[f][k][0..31]`, reverse into `y[f][k][32..63]`; `z += y * tok.tf.w + tok.tf.b`.
3. Band path - `zn = LN(z, tok.bn)`; projections as above with `tok.b`; `es_lstm_seq` with B = F sequences of length
   16 (bands), H = 32, fwd and rev into `y`; `z += y * tok.bf.w + tok.bf.b`.
4. Pooling - walk f = 0..F-1, collect indices with `mask[f] == 1` in order; `kept` = their count. Take them three at a
   time from the start of that list; a final group of 1 or 2 is dropped. `n_avail = kept / 3`. Token pre-vector i is
   `(z[a] + z[b] + z[c]) / 3.0f` over the 1,024 floats (flatten k*64 + d) of the group's frames a, b, c.
5. If `n_avail < 4` return ES_ERR_ENROLL_SHORT.
6. Selection - if `n_avail > limit`, keep pre-vectors `(int64_t)i * n_avail / limit` for i < limit;
   `n_tok = min(n_avail, limit)`.
7. `tokens = pre * tok.proj.w + tok.proj.b` ([n_tok][64]); `gbar = mean over tokens` (sum / n_tok);
   `g = gbar * tok.g.w + tok.g.b`; `g /= max(||g||_2, 1e-12)` (PyTorch `F.normalize`).

Work memory (`z`, `zn`, the two projections, `y`) is one `malloc` of about `7,168 F` floats (9 MB for 10 s, analytic),
freed before return. Cost about 0.97 M MAC per frame (contract 2.5), a few ms for 10 s (estimate).

### 7.3 Public calls

- `es_enroll(m, pcm, n, mask, role, e_wespk, out)` - role TARGET uses limit 48, OWN uses 16; `mask == NULL` means run
  `es_vad`. Stores `tokens`, `n_tok`, `g`, `role`, `speech_s = kept * 0.032f`, and a copy of `e_wespk` (256 floats,
  stored as given; `has_wespk = 1`) or `has_wespk = 0` when NULL. Returns ES_ERR_ARG for NULL pointers, bad role, or
  any non-finite sample in `pcm` or `e_wespk`.
- `es_embed_g(m, pcm, n, g_out)` - `es_vad` mask, limit 48, steps 1 to 7, returns only `g` (the "locked" check).
- `es_voiceprint` holds `float tokens[48][64]`, `int n_tok`, `float g[128]`, `float e[256]`, `int has_wespk`, `int role`,
  `float speech_s` (fixed size, about 14 KB).

### 7.4 `es_cond_build(m, mode, target, own, use_wespk, out)` (contract 2.3)

1. Validate - PERSON needs `target` with role TARGET; EVERYONE needs `target == NULL`; `own`, if given, must have role
   OWN. Otherwise ES_ERR_ARG.
2. WeSpeaker switch - `w = use_wespk && (target == NULL || target->has_wespk) && (own == NULL || own->has_wespk)`.
   If `use_wespk` is 1 but a given voiceprint has no vector, both vectors fall back to the nulls (the only no-WeSpeaker
   state the model saw in training is "both null", contract 8.1).
3. `u[770]` = `[e_t (256), g_t (128), e_o (256), g_o (128), mode (2)]` with
   - e_t - `w && target ? target->e : null.e_t`; g_t - `target ? target->g : null.g_t`;
   - e_o - `w && own ? own->e : null.e_o`; g_o - `own ? own->g : null.g_o`;
   - mode - PERSON `[1, 0]`, EVERYONE `[0, 1]`.
4. FiLM - `hdn = u * film0.w + film0.b` (256); `hdn = hdn / (1 + expf(-hdn))` (SiLU); `out->film = hdn * film2.w +
   film2.b` (4,096).
5. Tokens - list rows in order: target tokens + `type_emb[0]`, own tokens + `type_emb[1]`, `null_tok[i] + type_emb[2]`
   for i < 4; `n = n_t + n_o + 4` (at most 68). `kv = list * att_kv.w + att_kv.b` (M = n, 128 wide).
6. Scatter - `out->k[h][j][e] = kv[j][16h + e]`, `out->v[h][j][e] = kv[j][64 + 16h + e]`; rows j >= n are zero.
   `out->n_tokens = n`, `out->mode = mode`.

```c
struct es_cond { float film[4096]; float k[4][68][16]; float v[4][68][16]; int32_t n_tokens; int32_t mode; };
/* 12,800 floats + 2 ints, 64-byte aligned. n_tokens == 0 marks an empty slot inside the engine (8.4). */
```

Test-vector case mapping (contract 3.3; used by test_parity and es_selftest) - `person` = PERSON,
target from `enr_t`, own from `enr_o`, use_wespk 1; `everyone` = EVERYONE, own from `enr_o`, use_wespk 1; `noown` =
PERSON, target, no own, use_wespk 1; `nowespk` = PERSON, target and own, use_wespk 0.

---

## 8. The real-time pipeline (`pipeline.c`, `ring.h`, `rt_thread.c`)

### 8.1 Rings (`ring.h`, header-only, C11 atomics)

```c
typedef struct {
  _Alignas(64) _Atomic uint64_t w;     /* total items written; stored only by the producer (release) */
  _Alignas(64) _Atomic uint64_t r;     /* total items read; stored only by the consumer (release) */
  _Alignas(64) float *buf; uint32_t size, mask;                     /* size is a power of two */
} es_ring;
uint32_t ring_count(es_ring *q);   /* consumer side: acquire w, relaxed r */
uint32_t ring_space(es_ring *q);   /* producer side: relaxed w, acquire r */
void     ring_write(es_ring *q, const float *x, uint32_t n);   /* caller checked space; at most two memcpy */
void     ring_read(es_ring *q, float *x, uint32_t n);          /* caller checked count */
void     ring_skip(es_ring *q, uint32_t n);
```

The same code with `uint16_t` items is `es_ring16` (timings). The **history ring** (`es_hist`: `buf`, `size`, `mask`,
`_Atomic uint64_t w`) is written by one thread and read non-destructively by readers holding their own cursor:
`hist_read(h, &cursor, dst, max)` starts at `max(cursor, w - (size - 16384))`, copies `min(max, w - start)` items,
then re-reads `w`; if `w_after - start > size - 16384` the copy was overrun (amended 2026-11-07: the writer fills
slots before it publishes `w`, so the guard band counts; a cursor ahead of `w` returns 0 and leaves the cursor unchanged) and it returns 0 with `cursor = w_after - (size -
16384)`. Otherwise `cursor = start + copied`. A cursor of 0 therefore means "the oldest sample still held" (15.36 s
readable); the caller computes skipped samples as `new cursor - old cursor - returned`. `es_engine_input_position(e)`
is one acquire load of `w` (RT-safe); the app's `makeInputTap()` starts its cursor there, so a new tap never returns
audio from before it was made (contract 5). Sizes are
contract 4.2 - in16 and out16 4,096; input history 262,144; per-slot output history 65,536 (reads up to 48,000);
timings 1,024.

| Ring | Producer | Consumer |
|---|---|---|
| in16 | render | worker |
| out16 | worker (and `es_engine_new` for the prefill, before any thread runs) | render |
| input history | render | `es_engine_read_input` (any non-RT thread) |
| output history, one per slot | worker (raw `es_stream_hop` output of that slot, before gate and gains) | `es_engine_read_output` |
| timings | worker | `es_engine_read_timings` |

### 8.2 `es_engine_new` (non-RT)

1. Validate - `io_rate` 48000 or 44100 (else ES_ERR_RATE); `16 <= io_frames <= max_frames <= 4096`;
   `1 <= n_instances <= 3`; `slack16` in {22, 43, 64} (else ES_ERR_ARG).
2. Allocate and zero everything - `n_instances` streams (`es_stream_new`), per slot `cond[2]` with `n_tokens = 0`,
   rings, decimator and interpolator (`es_rs`, 9.2), render scratch (`xin[4096]`, `d16[1600]`, `u16[1600]`,
   `y48[4096]`), post state, safety state, probe buffers (4.5 s capture plus the 0.3 s sweep table, section 12),
   `semaphore_create(mach_task_self(), &sem, SYNC_POLICY_FIFO, 0)`, cached `mach_timebase_info`.
3. `P_min` - the 1,200-callback count simulation of 9.4 for (`io_rate`, `io_frames`).
4. Prefill - write `P = P_min + slack16` zeros into out16. Record `P_min`, `slack16`, `ins = 0`, `din = 0`, `S = 0`.
5. Safety state - the engine starts muted (contract 4): mute flag 1, mute ramp at gain 0 with nothing armed; the app's
   first `es_engine_set_mute(e, 0)` arms the 400 ms raised-cosine fade-in;
   limiter `es_limiter_init(sr, -3, 50)`; params = `es_params_default`.

### 8.3 `es_engine_render(e, in, out, n)` (audio thread)

Callbacks larger than `max_frames` are processed in chunks of at most `max_frames`. Per chunk:

1. **Input** - `xin[i] = isfinite(in[i]) ? clamp(in[i], -1, 1) : 0`. Decimate `xin` to 16 kHz (`es_rs_process`, 0 to
   1,366 samples at 48 kHz). Write them to in16 (if `ring_space` is short, write what fits, add the rest to
   `din_pending`, moved into `din` by step 5, and count one underrun) and to the input history.
2. **Wake** - if the worker is running and `ring_count(in16) >= 64`, `semaphore_signal(sem)`.
3. **Pull 16 kHz output** with one accounting rule that covers prefill, slack growth and concealment.
   `need = es_rs_need(interp, n)`. The out16 read index R (items read so far) should equal `want = S - ins - din`,
   where S counts every 16 kHz sample the interpolator has consumed (real or zero).
   - if `R > want` - put `z = min(R - want, need)` planned zeros first (slack growth or dropped input);
   - if `R < want` - `ring_skip(out16, min(want - R, count))` (samples whose time has passed after an underrun);
   - copy `min(remaining, count)` real samples; any remainder is **unplanned zeros** = an underrun (counted only while
     the worker is running; at most one count per callback); `S += need`.
4. **Interpolate** `u16[0..need)` to exactly `n` samples (`es_rs_process` with `max_out = n`).
5. **Conceal ramp** - an `es_ramp` at the I/O rate. Unplanned zeros (an underrun cannot be predicted) - on that
   callback `es_ramp_to(0, 1 ms, linear)`, otherwise `es_ramp_to(1, 1 ms, linear)`; `es_ramp_process` over the chunk.
   Planned zeros (slack growth, dropped-input payback) are known in advance, so they wait for the ramp: a pending
   `ins` or `din` step is first held back while `es_ramp_to(0, 1 ms, linear)` runs (over as many callbacks as 1 ms
   takes), is applied on the callback after the gain reached 0, and the gain then returns to 1 over 1 ms once real
   samples flow again. A slack step (21 zeros, 1.3 ms) therefore never makes a hard gap (contract 4.2).
6. **Slack growth** - underrun times are kept as I/O-sample timestamps (two-entry array). When an underrun arrives
   within 60 s of the previous one and `slack16 < 64`, set `grow_pending = 21`. While `grow_pending > 0` and every
   active slot's `p_target < 0.2` (atomics written by the worker), `ins_pending += 21`, `slack16 += 21` (22 to 43 to
   64), `grow_pending = 0`; step 5 moves `ins_pending` into `ins` after its ramp, and rule 3 then inserts the zeros.
   Slack is never lowered; a new engine starts at the default.
7. **Safety chain** (section 11) on `y48[0..n)`, then meters (`in_rms_db` from `xin`, `out_rms_db`, `limiter_gr_db`,
   `last_frames = n`).
8. **Probe** - if the probe state is RUNNING, replace `out` by the probe signal times the mute-ramp gain and copy `xin`
   into the capture buffer (section 12).

`es_engine_latency_ms(e) = 1000 * (64 + P_min + slack16) / 16000 + ES_RS_DELAY_MS[rate]`. With `P_min = 64`,
`slack16 = 43` at 48 kHz this is 10.6875 + 0.2575 = **10.95 ms** (analytic, contract 4.2 and 10). The impulse test
(14.5) proves the formula within one 16 kHz sample.

### 8.4 Worker (`rt_thread.c`) and `es_engine_run_hops`

`es_engine_start_worker(e, wg)` creates a pthread and waits (non-RT, on a startup semaphore) for its setup result. The
thread, before its loop, in this order - `pthread_setname_np("earshot.worker")`; `thread_policy_set` with
`THREAD_TIME_CONSTRAINT_POLICY` computed from the config (contract 4.2) - `period = max(4 ms, io_frames / io_rate)`,
`computation = min(period / 2, ceil(period / 4 ms) x 1.0 ms)`, `constraint = 0.625 x period` (4.0 / 1.0 / 2.5 ms at
64 frames, 23 / 6 / 14.4 ms at 1,104 frames), preemptible 1, converted to Mach ticks with the cached timebase (failure - report ES_ERR_THREAD and exit); if `wg != NULL`, `os_workgroup_join(wg,
&token)` (a non-zero result is recorded in a debug flag and the worker continues unjoined); FPCR.FZ (rule 3.6);
`BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)`; set `worker_running = 1`. Loop - `semaphore_wait(sem)`; if `stop`
is set, leave the workgroup (if joined) and exit; else `es_engine_run_hops(e)`.
`es_engine_stop_worker` sets `stop`, signals, joins, clears `worker_running`, then sets `ack[s] = active[s]` for every
slot. `es_engine_free` stops a running worker first, then frees.

```c
int es_engine_run_hops(es_engine *e) {                       /* worker only, RT-safe */
  int hops = 0;
  while (hops < 64 && ring_count(&e->in16) >= 64) {
    uint64_t t0 = mach_absolute_time();
    ring_read(&e->in16, e->x, 64);
    params_read(e, &e->wp);                                    /* seqlock, keeps previous on a torn read */
    memcpy(e->xd, e->dl, 64 * 4); memcpy(e->dl, e->x, 64 * 4); /* xd = input of the previous hop (8.6) */
    for (int s = 0; s < e->n_inst; s++) {
      int a = atomic_load_explicit(&e->active[s], memory_order_acquire);
      es_cond *c = &e->cond[s][a];
      if (e->reset_pending[s][a]) { es_stream_reset(e->st[s]); e->reset_pending[s][a] = 0; }
      atomic_store_explicit(&e->ack[s], a, memory_order_release);
      if (c->n_tokens >= 4) e->nan_resets += es_stream_hop(e->st[s], c, e->x, e->ys[s], e->hd[s]);
      else { memset(e->ys[s], 0, 64 * 4); e->hd[s][0] = e->hd[s][1] = 0; }
      hist_write(&e->outhist[s], e->ys[s], 64);
    }
    post_hop(e, e->y);                                         /* section 10 */
    if (ring_space(&e->out16) >= 64) ring_write(&e->out16, e->y, 64);
    uint32_t us = ticks_to_us(e, mach_absolute_time() - t0);
    if (ring16_space(&e->timings) >= 1) ring16_write1(&e->timings, us > 65535 ? 65535 : (uint16_t)us);
    stats_after_hop(e, us);                                    /* hops, p_target[], p_self, compute_us_last/max */
    hops++;
  }
  return hops;
}
```

### 8.5 Conditioning and parameter hand-off (contract 4.3, exactly)

- `es_engine_set_cond(e, slot, c, reset_state)` - single writer. ES_ERR_ARG for a bad slot or a non-NULL `c` with
  `n_tokens < 4`. If `active[slot] != ack[slot]` (ack loaded with acquire) return ES_ERR_BUSY.
  Else `b = 1 - active`; copy `*c` into
  `cond[slot][b]` (51 KB `memcpy`); `reset_pending[slot][b] = reset_state || (cond[slot][active].n_tokens == 0)` (a
  slot waking up from empty always resets); `atomic_store(active[slot], b, release)`; if the worker is not running,
  also store `ack[slot] = b`.
- `c == NULL` (contract 4.3) empties the slot - the buffer gets `n_tokens = 0`, the worker skips that stream and
  outputs zeros for it (thermal downshift and person deletion without an engine rebuild; Swift `clearConditioning`).
- `es_engine_set_params` - clamp every field to its contract range (non-finite fields take the default), then if
  `voice_db + max_gain_db > 24` set `max_gain_db = 24 - voice_db`, then publish through a seqlock: `seq` odd (release),
  store the 12 four-byte words of `es_params` with relaxed atomics, `seq` even (release). Readers (worker once per
  hop, render once per callback) load `seq` (acquire), copy the 12 words with relaxed loads, fence acquire, re-load
  `seq`; if it was odd or changed they keep their previous copy. No reader ever spins.
- `es_engine_set_mute` and `es_engine_cue` store one atomic int each; the render picks them up at the next callback.

### 8.6 Why `xd` is the previous hop's input

Post-processing mixes the input back in (room sound, bypass) and must line it up with the model output. Inside the
hop stream the model output lags the input by 64 samples (5.4), so `xd` for hop j is the input of hop j - 1. Measured
from capture, both paths then have the same 128-sample algorithmic latency (64 of stream lag plus the 64-sample hop
wait), as contract 4.2 states ("x_d = the 16 kHz input of the previous hop"). The unit test runs the
synthetic all-pass model and requires `model output == xd` within 1e-6 for every hop after the first two.

### 8.7 Reads and stats

`es_engine_read_input`, `read_output`, `read_timings` are the ring reads of 8.1 (non-RT callers). `es_engine_get_stats`
copies fields that are each `_Atomic` (64-bit counters and 32-bit floats stored with relaxed order; `test_units`
asserts `atomic_is_lock_free` for both), so it is RT-safe and never torn per field. `p_self` is the maximum over
active slots; `howl` and `limiter_gr_db` come from the render, the rest from the worker.

---

## 9. Resampling (`resample.c`, `resampler_tables.c`, `tools/make_resampler_tables.py`)

### 9.1 Tables (generated, committed)

`make_resampler_tables.py` (numpy, scipy 1.18.1 from `.venv`) writes `src/resampler_tables.c` and
`ml/assets/decim16k_fir.npy`, and prints the figures below; it exits non-zero if any check fails.

| Family | Prototype (rate) | Design | Up L / down M | Taps per phase Q |
|---|---|---|---|---|
| 48 kHz | 177 taps at 48 kHz | `firwin(177, 7500, window=("kaiser", 5.65), fs=48000)` then `minimum_phase(h, "homomorphic", half=False)`, scaled to sum 1 | decim 1 / 3, interp 3 / 1 | 177, 59 |
| 44.1 kHz | 26,080 taps at 7.056 MHz (= 44,100 x 160 = 16,000 x 441) | same call with `fs=7056000` and 26,080 taps (the same 3.7 ms span) | decim 160 / 441, interp 441 / 160 | 163, 60 (last tap of some phases 0) |

Polyphase table for (L, M) - `taps[p][j] = L * h[p + j L]` (0 past the end), stored **reversed** within each phase
(`rev[p][Q-1-j] = taps[p][j]`) so each output is one `vDSP_dotpr` over ascending memory. The file also defines
`ES_RS_DELAY_MS[2]` = decimator plus interpolator mean group delay over 1 to 4 kHz.

Figures computed while writing this file (analytic, scipy 1.18.1 in `.venv`; the table tool must reproduce them) -

| | 48 kHz family | 44.1 kHz family |
|---|---|---|
| Passband ripple to 6.5 kHz | -0.0031 to +0.0042 dB | -0.0031 to +0.0043 dB |
| -3 dB point | 7,385 Hz | printed by the tool (not computed here) |
| Stopband from 8 kHz | -60.0 dB | -60.0 dB |
| Group delay 1 to 4 kHz, one direction | 0.115 to 0.154 ms, mean 0.1288 | 0.135 to 0.175 ms, mean 0.1494 |
| `ES_RS_DELAY_MS` (both directions) | 0.2575 | 0.2987 |

The 48 kHz row reproduces contract 4.2. `decim16k_fir.npy` (for training) - the decimator's complex response sampled
at the 257 bins of a 512-point grid at 16 kHz, `irfft` to 512 taps, first 64 kept with a half-Hann taper over taps 48
to 63; the tool asserts its magnitude matches the decimator within 0.05 dB below 6.5 kHz and prints the phase error.

### 9.2 Polyphase resampler

```c
typedef struct { int L, M, Q; const float *rev; float *hist; int64_t base, consumed; int p; } es_rs;
/* hist holds Q old samples followed by up to 4,096 new ones (decimator) or 1,600 (interpolator).
   Q, not Q-1: the interpolator can start a call with base = consumed - 1, whose output reads Q old samples
   (amended 2026-11-07 after the E-0 review; DECISIONS L).
   base = absolute index of the newest input sample the next output needs; p = its phase (0..L-1).
   Both start at 0 with a zero history. */

int es_rs_process(es_rs *r, const float *in, int n_in, float *out, int max_out) {
  memcpy(r->hist + r->Q, in, sizeof(float) * n_in);
  int64_t end = r->consumed + n_in; int k = 0;
  while (k < max_out && r->base < end) {
    int64_t local = r->base - r->consumed + r->Q;                 /* index of x[base] in hist */
    vDSP_dotpr(r->rev + (size_t)r->p * r->Q, 1, r->hist + local - (r->Q - 1), 1, &out[k++], r->Q);
    r->p += r->M; r->base += r->p / r->L; r->p %= r->L;
  }
  r->consumed = end;
  memmove(r->hist, r->hist + n_in, sizeof(float) * r->Q);
  return k;
}
int es_rs_need(const es_rs *r, int n_out) {                         /* new inputs needed for n_out outputs */
  int64_t last = r->base + (r->p + (int64_t)(n_out - 1) * r->M) / r->L;
  int64_t need = last + 1 - r->consumed;
  return need > 0 ? (int)need : 0;
}
```

The decimator is called with `max_out` large enough for all outputs; the interpolator with `max_out = n` after feeding
exactly `es_rs_need(n)` inputs, so it always returns n. The arithmetic is exact integers, so the delay is
`(64 + P) / 16000 s` plus the filter delays for every callback size. A count-only twin (`es_rs_step_count`) shares the
same three stepping lines for the simulation in 9.4.

### 9.3 Cost

48 kHz - decimator 177 MACs per 16 kHz output, interpolator 59 MACs per 48 kHz output: about 5.6 M MAC/s for both
(analytic). Per 64-frame callback about 7,600 MACs, a few microseconds (estimate).

### 9.4 Prefill simulation (`P_min`)

```
for P = 0, 1, 2, ...:                            (first P that survives is P_min)
  fresh count-only decimator and interpolator; in_avail = 0; avail = P; pending = 0
  for k = 1..1200:
    avail += 64 * pending; pending = 0           # hops computed after the previous callback are usable now
    in_avail += decim outputs for io_frames inputs
    need = es_rs_need(interp, io_frames); if avail < need: P fails
    avail -= need; step the interpolator by need inputs and io_frames outputs
    pending = in_avail / 64; in_avail -= 64 * pending
```

Results of this exact procedure (analytic, a 40-line Python transcription run while writing this file) - the engine
computes its own value; test_units compares with this table.

| io_frames | 16 | 32 | 48 | 64 | 96 | 128 | 192 | 256 | 480 | 512 | 1024 | 1104 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P_min at 48 kHz | 64 | 64 | 64 | 64 | 64 | 86 | 64 | 128 | 192 | 214 | 384 | 416 | 1408 |
| P_min at 44.1 kHz | 69 | 75 | 80 | 86 | 98 | 110 | 132 | 156 | 237 | 249 | 435 | 463 | 1549 |

So the planned 10.95 ms holds for the requested 64-frame buffer at 48 kHz. A 128-frame grant adds 1.4 ms, a 44.1 kHz
route adds 1.4 to 4.3 ms, and the 1,104-frame buffers iOS can force when Sound Recognition is on add 22 ms; the
latency display shows whatever `es_engine_latency_ms` returns.

---

## 10. Post-processing per hop at 16 kHz (`post.c`, worker)

Inputs - `ys[s][64]`, `hd[s][2]` per slot, `xd[64]`, params `wp`. Gains below are per-hop targets; every linear gain is
applied with a per-sample linear ramp from the previous hop's value to the new one (no zipper noise). Constants -
`a200 = 1 - exp(-4/200) = 0.0198013`, `a150 = 1 - exp(-4/150) = 0.0263143`, `a400 = 1 - exp(-4/400) = 0.0099502`,
`a50 = 1 - exp(-4/50) = 0.0768837`. `post.c` computes these with `expf` at init; the literals are for reading only. "Active slot" means `n_tokens >= 4`. `pt_max` and `ps_max` are the maxima of
`p_target` and `p_self` over active slots (0 if none).

1. **Activity gate** (per slot, `gate` on) - `gt = p >= 0.5 ? 0 : p <= 0.2 ? -12 : -12 * (0.5 - p) / 0.3` dB. If
   `gt > g_db` then `g_db = gt` (opens within 1 hop, so onsets are never cut) else `g_db += a200 * (gt - g_db)`
   (closes with a 200 ms time constant). `gate` off - `g_db = 0`.
2. **Crossfade sum** - each slot has a power `P_s` that moves toward `slot_gain[s]^2` by at most `4 / xfade_ms` per hop
   (`xfade_ms` clamped to 4..2,000); amplitude `sqrt(P_s)`. During a switch A to B the powers sum to 1 (equal power).
   `sum = sum_s amp_s * gate_s * ys[s]`.
3. **Own-voice duck** - `own = ps_max > 0.5`. Target `-15` dB if `duck_own && own`, else 0. Lower target - jump
   (1 hop attack); higher - `d_db += a150 * (target - d_db)`.
4. **Slow AGC** (`agc` on) - on hops with `pt_max > 0.5`: `ms_hop = mean(sum^2)`; first such hop sets `agc_ms =
   ms_hop`, later ones `agc_ms += a400 * (ms_hop - agc_ms)`; `want = clamp(-26 + voice_db - 10 log10(agc_ms + 1e-12),
   -12, max_gain_db)`; `agc_db += clamp(want - agc_db, -0.024, +0.024)` (6 dB/s). Other hops - frozen. `agc` off -
   `agc_db` slews to 0 at the same rate. The level is measured on `sum` (after 2, before the duck) so ducking never
   pumps the AGC.
5. **Gain cap (+24 dB mic to ear, contract 4.2)** - `y = sum * duck * agc`; `in_ms += a50 * (mean(xd^2) - in_ms)`,
   `out_ms += a50 * (mean(y^2) - out_ms)`; `excess = 10 log10(out_ms + 1e-10) - 10 log10(in_ms + 1e-10) - 24`; cap
   target `-max(excess, 0)` dB, lower - jump, higher - `a200`. Needed because the deep filter alone can raise a bin by
   up to 6.36x (three taps of magnitude at most 1.5 sqrt(2), since real and imaginary parts are each bounded by 1.5;
   +16.1 dB below 4 kHz, and 2.12x, +6.5 dB, above), so the AGC clamp does not bound the total (contract 4.2 step 5).
6. **Room mix-back** - `room = room_db <= -59.5 ? 0 : 10^(room_db / 20)`; envelope `r_env` drops to 0 at once when
   `own` and recovers with `a150`, independent of `duck_own` (the delayed room path must never replay the user's
   voice). `y += room * r_env * xd`.
7. **Bypass** - `bm` moves toward `bypass` by 0.2 per hop (20 ms crossfade); `y = (1 - bm) * y + bm * xd`. With
   `bm == 1` the output is exactly `xd` (contract 4.2 "bypass replaces (1) to (6)", the same steps 1 to 6 here).

---

## 11. Safety chain at the I/O rate (`safety.c`, render)

Order (contract 4.2), on the interpolated chunk `y[0..n)` at `sr = io_rate`:

| # | Stage | Exact behavior |
|---|---|---|
| 0 | Sanitize | non-finite to 0, clamp to [-1000, 1000] (defensive; normal values are untouched) |
| 1 | Conceal ramp | 8.3 step 5 |
| 2 | Mute ramp | `es_ramp`; mute - `es_ramp_to(0, 5, 0)` (5 ms linear); every unmute - `es_ramp_to(1, 400, 1)` (400 ms raised cosine); the engine starts muted, so the app's first unmute is the start fade-in |
| 3 | Output trim | linear gain ramps to `10^(out_gain_db/20)` over 20 ms (`es_ramp`, linear) |
| 4 | Runaway guard | measure, then a duck ramp (below) |
| 5 | Cue beeps | added after the trim, so they are heard while paused (muted) |
| 6 | Limiter | `es_limiter_process`, ceiling -3 dBFS, instant attack, 50 ms release |
| 7 | Soft clip | `es_softclip`, knee -1 dBFS |
| 8 | Hard clip | `fminf(fmaxf(y, -1), 1)` |

**`es_ramp`** (contract struct `{gain, start, target, step, phase, sr, shape}`).
- `es_ramp_init(r, sr, gain)` - `gain = start = target = gain`, `step = 0`, `phase = 1`, `shape = 0`.
- `es_ramp_to(r, target, ms, shape)` - `ms <= 0` jumps. Otherwise `start = gain` (so the ramp continues from the
  current gain, and an unmute during a mute ramp never jumps), `phase = 0`, `step = 1000 / (ms * sr)` per sample.
- `es_ramp_process(r, buf, n)` - per sample, `phase = min(phase + step, 1)`, `w = phase` (shape 0, linear) or
  `w = 0.5 - 0.5 cos(pi * phase)` (shape 1, raised cosine), `gain = start + (target - start) * w`, multiply. Both
  shapes work in either direction; the engine uses shape 1 only for fade-ins. No `acos` is needed.

**Limiter** (zero lookahead, so zero added latency).

```c
void es_limiter_init(es_limiter *l, float sr, float ceiling_dbfs, float release_ms) {
  l->ceiling = powf(10.0f, ceiling_dbfs / 20.0f);            /* -3 dBFS = 0.7079458 */
  l->rel_coef = expf(-1.0f / (release_ms * 0.001f * sr));    /* 48 kHz: 0.9995834, 44.1 kHz: 0.9995466 */
  l->gain = 1.0f; l->min_gain = 1e-6f;
}
void es_limiter_process(es_limiter *l, float *x, int n) {
  for (int i = 0; i < n; i++) {
    float a = fabsf(x[i]), gt = a > l->ceiling ? l->ceiling / a : 1.0f;
    if (gt < l->gain) l->gain = gt;                                 /* instant attack */
    else l->gain = gt + (l->gain - gt) * l->rel_coef;               /* release toward gt, never above it */
    if (l->gain < l->min_gain) l->gain = l->min_gain;
    x[i] *= l->gain;
  }
}
```

Because the gain is never above `ceiling / |x|` when `|x| > ceiling`, every output sample is at most the ceiling (plus
one float rounding step); with stage 0 limiting `|x|` to 1,000, the `min_gain` floor of 1e-6 can never break that.
The release recovers 63% of the way to unity in 50 ms. Gain reduction is reported as `-20 log10(min gain in the
chunk)`.

**Soft clip** - `k = 0.8912509` (-1 dBFS), `r = 1 - k`; `|x| <= k` passes; otherwise
`y = copysignf(k + r * tanhf((fabsf(x) - k) / r), x)`. Continuous with slope 1 at the knee, output strictly inside
(-1, 1). After the limiter it only acts if the ceiling were raised; it is the second net.

**Runaway guard** - per chunk, one-pole mean squares with `a = 1 - exp(-n / (0.05 sr))` of the sanitized input `xin`
and of the signal after stage 3. Condition - `out_ms > 1000 * max(in_ms, 1e-8)` (30 dB) and `out_ms > 1e-6` (-60 dBFS).
Held for 0.2 s cumulative I/O samples - `howl = 1`, duck ramp to -20 dB (5 ms linear). Condition false for 2 s -
`howl = 0`, ramp back over 400 ms (raised cosine).

**Cues** - `es_engine_cue(e, beeps)` with beeps clamped to 1..3. Each beep 60 ms of `0.1258925 * sinf(phase)` (880 Hz,
-18 dBFS peak), 5 ms raised-cosine edges, 90 ms of silence between beeps; phase accumulator
`phase += 2 pi 880 / sr`, wrapped at 2 pi. A new request restarts the sequence.

**DC block and high-pass** - none in v1. The contract chain has no such stage; the model input is AC-coupled by iOS and
the training mic chain high-passes at 60 to 200 Hz, so the model gives no DC of its own; a mic offset in bypass or
room sound is bounded by the limiter. A 10 Hz DC blocker before the limiter is proposed to the lead (section 16).

---

## 12. Latency probe (`probe.c`)

- **Signal** (built in `es_engine_new` at `io_rate`) - one exponential sweep, 300 Hz to 6 kHz in 0.3 s,
  `x(t) = A sin(2 pi f1 T / ln(f2/f1) * (exp(t ln(f2/f1) / T) - 1))`, `A = 10^(ES_PROBE_DBFS/20)` (0.0316228 at the
  default -30 dBFS; Phase 0 may change the constant to -24 or -18 with a DECISIONS row), 5 ms raised-cosine
  edges. Schedule - sweep k (k = 0..4) starts at `0.8 k` s; playback lasts 4.0 s; capture lasts 4.5 s
  (216,000 floats at 48 kHz), both preallocated.
- **Run** - `es_engine_probe_start` returns ES_ERR_BUSY if a probe is running, else zeroes the indices and stores
  state RUNNING (release). The render plays `sweep * mute_gain` into `out` and copies `xin` into the capture at the
  same index (stage 8 of 8.3), so the measured lag is the pass-through round trip X + 2B. At 4.5 s the state becomes
  CAPTURED and `probe_runs` increments.
- **Result** (`es_engine_probe_result`, non-RT, allocates) - RUNNING or never started - ES_ERR_STATE. Otherwise, per
  sweep k, GCC-PHAT between the sweep `r` (14,400 samples at 48 kHz) and the capture window `c` from the sweep start for
  0.75 s: FFT size 65,536 (vDSP, log2n 16); `G = C conj(R) / (|C conj(R)| + 1e-12)`; `g = IFFT(G)`; peak over lags 0 to
  0.45 s with parabolic interpolation; `psr_db = 20 log10(|g_peak| / max |g|)` over the same lag range excluding
  +/- 2 ms around the peak. Valid sweeps - `psr_db >= 10`. `loop_ms` = median lag of valid sweeps, `spread_ms` =
  max - min of valid lags, `psr_db` = median, `n_ok` = count. ES_ERR_PROBE_WEAK only if `n_ok < 3`; `spread_ms` is
  always filled. The engine is never told the route: the app's LatencyMeter rejects a spread over 10 ms on Bluetooth
  or 2 ms wired (contract 4.2).
  The result is cached; later calls return it until the next start.
- **Test** - test_units feeds the probe through a simulated loop (output delayed by D samples, scaled by -20 dB, plus
  white noise at -50 dBFS, for D = 480, 2,400 and 9,600 at 48 kHz) and requires `|loop_ms - D / 48| <= 0.05 ms` and
  `n_ok == 5`; with noise only, ES_ERR_PROBE_WEAK.

---

## 13. Self-test (`selftest.c`, `es_selftest`)

Runs on any non-RT thread (the app calls it at every launch with `testvec_short.est`). Steps, failing fast:

1. Open the container with magic `EST1` (4.1 checks 1 to 10, including the config hash); format errors return
   ES_ERR_FORMAT, version and config errors return their own codes. Then `meta.weights_sha256` must equal
   `es_model_data_hash(m)`, else ES_ERR_SELFTEST with the reason "test vectors are for other weights" (so a random_v1
   test vector bundled with v1 weights fails clearly, contract 3.3).
2. `es_vad(enr_t)` equals `vad_t` exactly.
3. `es_enroll(enr_t, mask_t, TARGET, e_t)` and `es_enroll(enr_o, mask_o, OWN, e_o)`; tokens and g match `tok_t, g_t,
   tok_o, g_o` (token counts equal, max abs error at most `tolerances.tokens` / `.g`).
4. `es_cond_build(PERSON, target, own, 1)` matches `cond.person.*` (`n` exact; film, k, v at most their tolerances).
5. A fresh `es_stream` runs all `N / 64` hops of `in` with that engine-built conditioning; `out.person` within
   `tolerances.out` and SNR at least `tolerances.out_snr_db`; `heads.person` within `tolerances.heads`.

Returns ES_OK or ES_ERR_SELFTEST; `*max_err_out` = largest absolute error seen on `out`. Expected run time about 0.1 s
on the phone (estimate, contract 3.3). `SNR = 10 log10(sum ref^2 / sum (eng - ref)^2)`; a zero error gives +inf (pass).

---

## 14. Tests, tools and benchmarks

Every test binary prints one line per check (`PASS name value limit` or `FAIL ...`) and exits non-zero on any failure.
Tests marked "synthetic" use `es_model_synthetic` and need nothing from WP-ML.

### 14.1 `test_units.c` (synthetic; runs in seconds)

| Area | Check | Limit |
|---|---|---|
| Config | SHA-256 of `ES_CONFIG_CANON` equals the literal in 4.2 | exact |
| Windows | formulas of 5.1; hop-64 sum of `wa * ws` | 1e-6 |
| FFT | forward vs float64 naive DFT on random frames, error / max abs X | 1e-6 |
| FFT | inverse(forward(x)) vs x | 1e-6 |
| STFT | streaming all-pass (synthetic all-pass model) reproduces input delayed 64 | 1e-5 |
| STFT | C streaming vs C offline (`es_stft_offline` / `es_istft_offline`), `y_stream[n+64] = y_offline[n]` | 1e-5 |
| Kernels | `es_linear` (M = 1 and 16), `es_layernorm`, `es_lstm_cell`, `es_lstm_seq` (both directions) vs naive double C | 1e-5 |
| Attention | 6.4 vs a naive double softmax-attention for n = 4, 37, 68 | 1e-5 |
| Deep filter | 6.5 vs naive complex double over 3 hops (history shift included) | 1e-6 |
| Determinism | 200 hops twice from reset, outputs equal | `memcmp` |
| NaN guard | inject NaN into `ana`, hop returns 1, output zero, state zero | exact |
| VAD | synthetic bursts (tone bursts in -60 dBFS noise) give the hand-computed mask; n = 575, 576, 1088, 4672, 5184 give F = 0 (nothing read or written), 2, 3, 10, 11 | exact |
| Pooling | masks with gaps and remainders give the expected groups; selection rule for n_avail 50, limit 48 and 16 | exact |
| json_min | 20 valid and 20 malformed documents (truncated, bad escapes, deep nesting, huge numbers) | expected result |
| Container | truncated file, bad magic, version 2, header_len past end, misaligned offset, extra tensor, missing tensor, wrong shape, flipped data byte; each returns the right status and names the field | exact |
| Rings | wrap-around; a 2-thread test pushes 50 M samples of a counter sequence through in16 with random chunk sizes; consumer checks the sequence | exact |
| Resampler | impulse response FFT - passband ripple, stopband, mean group delay 1 to 4 kHz vs 9.1, both families | 0.01 dB / 1 dB / 0.01 ms |
| Resampler | random block sizes give bit-identical output to one big block | `memcmp` |
| Prefill | engine `P_min` equals the 9.4 table for all listed sizes | exact |
| Ramps | linear 5 ms ends at sample 240 (48 kHz); raised cosine 400 ms ends at 19,200; unmute at gain 0.5 continues without a jump (start field); a falling raised cosine is monotonic | 1e-6 |
| Limiter | 10 M random samples up to 1,000 never exceed the ceiling; release from a 0.1 gain reaches `1 - 0.9 e^-1` at 50 ms | 1e-6 / 1 sample |
| Soft clip | identity below the knee, continuity and slope at the knee, bound | 1e-6 |
| Post | gate, duck, AGC slew (never above 0.024 dB per hop), room envelope zero while own (also with `duck_own = 0`); gain cap - a synthetic model with every coefficient 1.5 + 1.5j and the AGC held at its maximum keeps the 50 ms output RMS <= input RMS + 24.1 dB | formulas of 10 |
| Alignment | synthetic all-pass engine with gates and gains off - model output equals `xd` (8.6) | 1e-6 |
| Hand-off | writer thread calls `set_cond` 1 M times against a worker running `run_hops`; BUSY seen, every accepted change applied exactly once, resets applied only for their buffer | exact |
| Probe | 12 simulated loop; spread is filled and PROBE_WEAK depends only on `n_ok` | 0.05 ms |
| Input taps | `es_engine_input_position` after 10 s of tone, then a read from that cursor returns 0 samples; an overrun read returns 0 and moves the cursor so `new - old - returned` equals the skipped count | exact |
| Start state | a new engine outputs exact zeros until the first `set_mute(e, 0)`, then the 400 ms fade-in | exact |
| Atomics | `atomic_is_lock_free` for the stats fields | true |

### 14.2 `test_parity.c` (needs an export; the random-weights parity runs before training finishes)

`build/test_parity ../ml/exports/random_v1` (default) or `../ml/exports/v1`. Loads `earshot.esw`, then for
`testvec_short.est` and `testvec_long.est`:

1. Read `meta.tolerances` (exactly the contract 3.3 keys `vad`, `tokens`, `g`, `film`, `k`, `v`, `dbg`, `out`,
   `out_snr_db`, `heads`; a missing or extra key is a failure, never a default). Check `meta.weights_sha256` equals
   `es_model_data_hash`.
2. VAD, tokens, g as in 13.
3. For each case of 7.4 - build the conditioning from the engine's own voiceprints and compare with `cond.{case}.*`.
4. For each case, run the stream twice: (a) with the **reference** conditioning loaded from the file into an `es_cond`
   (isolates the hop), (b) with the engine-built conditioning (end to end). Both must meet `out`, `out_snr_db` and
   `heads`. Run (a) with FPCR.FZ off and on.
5. Short file only - `es_stream_hop_dbg` for the first 8 hops of `person` vs `dbg.feat`, `dbg.z_enc`, `dbg.z_blk0..3`,
   `dbg.coef` within `tolerances.dbg`. The first failing debug tensor tells which stage disagrees.
6. Bit determinism - case `person` run twice, `memcmp` equal.
7. Print max abs error and SNR for every array (kept in the test log for the lead).

Tolerances (contract 3.3) - vad 0 mismatches; tokens, g, film, k, v and debug 1e-4; out 2e-4 with SNR >= 60 dB
(short) and 1e-3 with SNR >= 50 dB (long); heads 1e-3.

### 14.3 `test_rt.c` with `rt_guard.dylib` (synthetic or export)

- `tests/rt_guard.c` builds a dylib that interposes (`__DATA,__interpose`) `malloc`, `calloc`, `realloc`, `free`,
  `posix_memalign`, `aligned_alloc`, `malloc_zone_malloc`, `malloc_zone_calloc`, `malloc_zone_realloc`,
  `malloc_zone_memalign`, `malloc_zone_free`, `pthread_mutex_lock`, `os_unfair_lock_lock`. Each hook increments a
  global atomic counter only when `pthread_self()` is in a 4-entry table of guarded threads (no TLS, so the hook never
  allocates), then calls the real function.
- The test runs as `DYLD_INSERT_LIBRARIES=build/rt_guard.dylib build/test_rt` (the Makefile sets it). An "audio" thread
  calls `es_engine_render` with callback sizes drawn from {16, 64, 128, 480, 1104} and signals a "worker" thread that
  calls `es_engine_run_hops` (the worker body); both are guarded, along with `set_mute`, `cue`, `get_stats`, and
  `es_stream_hop` called directly. Warm-up of 100 hops runs unguarded (first-call setup inside Accelerate is allowed
  only at create time; if a warm-up allocation shows up, move that call into `_new`).
- Pass - 0 allocations, 0 frees, 0 lock calls over 100,000 hops with 3 instances (`make test` runs 10,000).

### 14.4 `test_fuzz.c` (synthetic by default, `--export DIR` for real weights)

Single-threaded, deterministic (seed printed): render then `run_hops` alternate, so it is reproducible. One simulated
hour at 48 kHz (`FUZZ_SECONDS`, default 3,600), callback sizes uniform 16 to 1,104. Every 0.5 to 5 s the input switches
among silence, white noise at -60 to 0 dBFS, full-scale square waves, DC +/-1, denormals (1e-40), NaN bursts, +/-Inf
bursts, sweeps, and the test-vector mixture. Every 1 to 10 s it randomizes params (including out-of-range and NaN
values), toggles mute, sends cues, and swaps conditioning among the four cases (and NULL). Checks on every output
sample - finite and `|y| <= 0.7079458 + 1e-6`. Also - the engine keeps producing hops, `nan_resets` is reported, and
no hop exceeds 64 per `run_hops` call.

### 14.5 `test_latency.c` (synthetic)

For io_rate in {48000, 44100}, io_frames in {16, 32, 64, 128, 192, 256, 480, 1104} and slack16 in {22, 43, 64} -
engine with `bypass = 1`, gate, duck and AGC off, room off; run 0.6 s of silence (past the 400 ms fade-in), then an
impulse of 0.5 at a known I/O sample, then 0.5 s of silence. Delay estimate - FFT of the 8,192 output samples starting
at the impulse, unwrapped phase, mean group delay over 1 to 4 kHz. Pass - `|estimate - es_engine_latency_ms| <=
0.0625 ms` (one 16 kHz sample) and 0 underruns. A second run uses the synthetic all-pass model with bypass off and must
give the same delay (the model path and bypass are aligned). A third run forces one slack step (two underruns within
60 s) and requires the output around the 21 inserted zeros to carry the same 1 ms ramps down and up as an underrun
(no sample-to-sample step above the ramp slope).

### 14.6 `bench.c` (M4, run when `bench/wait_quiet.sh` reports a quiet machine)

- `hop_1inst`, `hop_3inst` - `es_stream_hop` on the test-vector mixture (or noise), 500 warm-up hops, 20,000 timed
  hops, `mach_absolute_time` around each hop, `VECLIB_MAXIMUM_THREADS=1` plus `BLASSetThreading` single-threaded,
  default QoS (P-core). `--ecore` re-runs under `taskpolicy -c background` for an E-core view (tails not meaningful,
  docs/designs/product.md B.2).
- `render_64`, `render_1104` - one render call plus the hops it triggers, per callback.
- `enroll_10s`, `cond_build`, `selftest` - one-shot timings.
- Output - one JSON line per row: `{"name", "n", "p50_us", "p90_us", "p99_us", "p999_us", "max_us", "load_start",
  "model": "synthetic|random_v1|v1"}`. Gate - `hop_1inst` p50 <= 250 us (target, contract 11). Prior measurement for
  this shape - 175 to 187 us median for ESB-16 without the deep filter and attention (measured,
  docs/designs/product.md B.2).

### 14.7 `esrun` (offline WAV through the engine)

```
build/esrun --weights ../ml/exports/v1/earshot.esw --in mix.wav --out out.wav
            [--target enroll.wav] [--own own.wav] [--wespk-t e_t.f32 --wespk-o e_o.f32]
            [--mode person|everyone] [--engine --io-rate 48000 --frames 64] [--param room_db=-24 ...]
```

- Default (stream mode) - input must be 16 kHz; runs `es_stream_hop` hop by hop and writes the stream output (64-sample
  lag, contract 2.1). `--engine` - input at 48 or 44.1 kHz through `es_engine_render` + `run_hops` synchronously with
  the full post and safety chain, as on the phone.
- WeSpeaker vectors are raw little-endian float32 files (256 values) written by Python; without them `use_wespk = 0`.
- Prints model hash, enrollment report (speech seconds, tokens), latency, stats and per-hop timing percentiles.

---

## 15. Work order and done criteria

| Step | Contents | Done when |
|---|---|---|
| E-0 (day 1, no weights) | header copy, `internal.h`, `json_min.c`, container reader, windows, FFT, features, `nn.c`, rings, table tool and resampler, ramps, limiter, soft clip, VAD, synthetic model | test_units rows for these pass |
| E-1 (needs `random_v1`) | `es_model_load`, `es_stream_hop`, `es_enroll`, `es_cond_build`, `es_embed_g`, `es_selftest` | `make parity` passes on random_v1 (short and long, all cases, debug tensors) |
| E-2 | pipeline, worker, post, safety, cues, probe, stats, timings, `esrun` | test_rt (100,000 hops), test_latency, `make fuzz` (one hour) pass |
| E-3 | bench on the M4 (1 and 3 instances), `make ios-check`, hand-off to the lead | bench JSON committed to `engine/bench-results/YYYY-MM-DD.jsonl` (docs/measurements/ belongs to WP-IOS); ios-check clean with `-Werror` |
| After ML-2 | re-run `make parity EXPORT=../ml/exports/v1` and bench with v1 | all pass |

Review checklist for every RT function (the lead re-checks at integration) - no `malloc`/`free`/`calloc`, no locks, no
`printf`/`os_log`, no Objective-C or Swift, no file I/O, no `dispatch_*`, no unbounded loop, every buffer from create
time, every shared field atomic or behind the 4.3 hand-off or the seqlock.

---

## 16. Decisions this file makes where the contract is silent (for the lead)

Each item is binding for WP-ENGINE unless the lead overrides it in docs/DECISIONS.md. Items marked **(ML)** must be
mirrored by WP-ML in `export.py` / `stream_ref.py`; items marked **(iOS)** by WP-IOS. Items marked **(now contract)**
were adopted into docs/spec/00-contract.md on 2026-11-07 (DECISIONS L), so the contract text governs them.

1. **(now contract) Private header** - `src/internal.h`, `tests/rt_guard.c`, `tools/wav.h`, `bench-results/`.
2. **(now contract, ML) `dbg.z_blk{b}`** - z at the end of block b including FiLM 0 and attention (b = 0) or FiLM 1 (b = 2).
   `dbg.coef` - the 16 bands concatenated in band order, each `[re: T_k x W_k][im: T_k x W_k]`.
3. **(now contract, ML) Test cases** - the mapping at the end of 7.4.
4. **(now contract, ML) `meta.tolerances` keys** - `vad, tokens, g, film, k, v, dbg, out, out_snr_db, heads` (14.2).
5. **(now contract, ML) Token pooling** - kept frames form one ordered list, grouped in threes from its start; one final remainder of
   1 or 2 frames is dropped (7.2 step 4).
6. **(now contract, ML) Tensor order** - outer loop over the block or band index, inner over the names of the row (4.3). The loader
   matches by name, so order is not load-bearing.
7. **(now contract) WeSpeaker partially missing** - `use_wespk = 1` with a voiceprint that lacks a vector uses both null vectors.
8. **(now contract) `x_d` alignment** - the previous hop's input (8.6), which equals the model output timing.
9. **(now contract, iOS) Empty slot** - `es_engine_set_cond(e, slot, NULL, any)` empties a slot; refilling it always
   resets; Swift `clearConditioning(slot:)`.
10. **(now contract) Probe route** - withdrawn: the engine never infers the route. PROBE_WEAK means `n_ok < 3`; the
    app rejects a spread over 10 ms (Bluetooth) or 2 ms (wired).
11. **(now contract) Gain cap stage** - post step 5 enforces the contract's +24 dB mic-to-ear rule.
12. **(now contract) Envelope directions** - gate opens in 1 hop and closes with 200 ms; duck engages in 1 hop and releases with
    150 ms; the room-sound envelope follows own voice even when `duck_own = 0`.
13. **Crossfade** - linear power ramps (equal power during a switch); bypass changes crossfade over 20 ms.
14. **(now contract) Underrun accounting** - concealment zeros are paid back by skipping the late samples, so latency never drifts
    (8.3 step 3); input lost to a full in16 counts as an underrun; underruns count only while the worker runs.
15. **(now contract) Resampler delay constant** - mean group delay over 1 to 4 kHz, 0.2575 ms (48 kHz) and 0.2987 ms (44.1 kHz), so
    the reported Earshot share is 10.95 ms at 48 kHz / 64 frames (contract 4.2 and 10).
16. **(now contract) Ramp** - `es_ramp` has a `start` field; both shapes run from `start` to `target` in either
    direction (11); the engine uses the raised cosine only for fade-ins.
17. **Small numbers** - cue gap 90 ms with 5 ms edges; runaway guard clears after 2 s; output trim smoothed over
    20 ms; `es_enroll` accepts at most 60 s; input samples are clamped to [-1, 1] and non-finite ones zeroed.
18. **(now contract, iOS) Reads** - `es_engine_input_position` gives the current write position (new taps start
    there); `read_input` with an old cursor starts at the oldest held audio and the caller counts the skip;
    `read_output` returns the slot's raw model output before gate and gains.
19. **Proposed, not built** - a 10 Hz DC blocker before the limiter (11). The earlier proposal of a `wired` field in
    `es_engine_config` is dropped, since the app now applies the spread limit (item 10).


# Earshot v1 - iOS app (30-ios, WP-IOS)

Date - 2026-11-07. Owner - WP-IOS. Status - expands docs/spec/00-contract.md (the contract wins on any conflict; every
number, name and signature below is copied from it). Labels as in the contract - **measured**, **benchmark**, **doc**,
**analytic**, **estimate**, **target**. Nothing has run on the iPhone yet, so every phone number here is an estimate.

Read first - contract sections 4 (C API), 5 (Swift API, storage, app constants), 6 (speaker encoder and checks), 7 (layout),
9 (phone acceptance rows), 11 (WP-IOS packages). This file says how to build all of it on the phone.

## 0. What WP-IOS builds, and the three rules

The `ios/` app - an Xcode project generated from `project.yml`, a C RemoteIO host that calls only `es_engine_render`,
Swift around it (session, routes, interruptions, enrollment, people, head tracking, stem presses, latency, safety,
telemetry, UI), and the on-phone checks that prove it works (Phase 0 file and the phone rows of contract section 9).

1. **Swift never runs on the audio thread.** The render callback is C (`AudioHost.c`) and calls `AudioUnitRender`,
   `es_engine_render`, `memset` and `memcpy` only. The model runs on the engine's worker thread (WP-ENGINE).
2. **Only the main actor writes to the engine** (`setConditioning`, `setParams`). Everything slow (Core ML, `es_enroll`,
   file I/O, logs) runs in actors or utility queues, never in the callback or worker.
3. **Fail quiet, never loud.** Any doubt about the route, the engine or the self-test mutes. Nothing ever jumps in level.

## 1. File plan (`ios/`)

Every Swift file stays under about 400 lines, so a sophomore can read each one in a sitting.

| Path | Job | Lines (estimate) |
|---|---|---|
| `project.yml` | XcodeGen spec (section 2.1) | 70 |
| `Makefile` | `project`, `build`, `install`, `test`, `test-device`, `logs`, `phase0`, `privacy-check` | 60 |
| `.gitignore` | `Earshot.xcodeproj/`, `build/`, `.device`, `Earshot/Resources/Models/*` | 5 |
| `Earshot/App/EarshotApp.swift` | `@main` SwiftUI app; creates the model, stores and controllers once; scene phase hooks | 150 |
| `Earshot/App/Launch.swift` | launch sequence (section 9.4): load weights, `es_selftest`, thresholds, encoder self-test, people rebuild | 150 |
| `Earshot/App/Earshot-Bridging-Header.h` | `#include "earshot.h"` and `#include "AudioHost.h"` | 3 |
| `Earshot/Core/EarshotCore.swift` | contract 5 wrappers, exact signatures | 350 |
| `Earshot/Core/ContainerFile.swift` | reads `.est` / `.esw` containers (contract 3.1) for Swift-side test data; SHA-256 with CryptoKit | 120 |
| `Earshot/Core/Thresholds.swift` | decodes `thresholds.json`, classifies pass / warn / fail | 100 |
| `Earshot/Audio/AudioHost.h`, `AudioHost.c` | RemoteIO unit, render callback, workgroup (section 4) | 180 |
| `Earshot/Audio/SessionConfig.swift` | AVAudioSession setup and read-back (section 3) | 120 |
| `Earshot/Audio/RouteInfo.swift` | route snapshot, allowed-output rule, signature | 120 |
| `Earshot/Audio/AudioController.swift` | contract `AudioController`; start, stop, the one engine swap path, interruptions, thermal | 400 |
| `Earshot/Audio/LatencyMeter.swift` | probe flow, stored measurements, `LatencyDisplay` strings (section 7) | 200 |
| `Earshot/Enrollment/SpeakerEncoder.swift` | contract `actor SpeakerEncoder`, Core ML on `.cpuAndGPU` in the foreground and `.cpuOnly` in the background | 150 |
| `Earshot/Enrollment/CaptureSession.swift` | records from an `InputTap` with live VAD progress | 150 |
| `Earshot/Enrollment/ClipTools.swift` | the contract 6 check formulas (trim, best window, rounding, windows, SBR, clipping), WAV encode | 200 |
| `Earshot/Enrollment/EnrollmentService.swift` | contract `actor EnrollmentService`, the 6 checks | 300 |
| `Earshot/Enrollment/QuickAdd.swift` | "Add who just spoke" from the engine's input history (RAM copy only across an engine swap) | 120 |
| `Earshot/People/Person.swift`, `PeopleStore.swift` | contract 5.1 files, expiry, purge, consent | 300 |
| `Earshot/Control/HeadTracker.swift` | `CMHeadphoneMotionManager`, smoothed yaw, rate, stale detection | 200 |
| `Earshot/Control/TableState.swift` | 3 warm slots, mode, slot gains | 200 |
| `Earshot/Control/Selector.swift` | contract `Selector`: look-select, cycle, setMode, direction learning | 250 |
| `Earshot/Control/RemoteCommands.swift` | `MPRemoteCommandCenter` and Now Playing | 120 |
| `Earshot/Safety/AlarmListener.swift` | SoundAnalysis alarm passthrough | 150 |
| `Earshot/Safety/LockedCheck.swift` | 1 s "following the right person" check | 120 |
| `Earshot/Telemetry/SessionLog.swift` | per-minute JSONL (contract 5.2) | 150 |
| `Earshot/Telemetry/Phase0.swift` | guided Phase 0 runs and the JSON file (section 15.3) | 350 |
| `Earshot/UI/*.swift` | `OnboardingView`, `ListenView`, `AddPersonView`, `PeopleView`, `SettingsView`, `DiagnosticsView`, `HelpView`, `Copy.swift` (every user-facing string) | 1,500 total |
| `Earshot/Resources/` | `Assets.xcassets`, `passage.txt` (own-voice reading text), `Models/` (gitignored, filled by `make models`) | - |
| `Stubs/earshot_stub.c` | pass-through implementation of all of `earshot.h` for day-one work (section 2.5) | 400 |
| `EarshotTests/*.swift` | XCTest (section 15.1) | 800 |

`Earshot/Resources/Models/` holds exactly the ML-0 / ML-2 outputs - `earshot.esw`, `testvec_short.est`,
`thresholds.json`, `wespk_selftest.est`, `wespeaker_resnet34_fp32.mlpackage`.

## 2. Project and command-line build

### 2.1 `ios/project.yml` (XcodeGen 2.46.0 at /opt/homebrew/bin/xcodegen, measured present)

```yaml
name: Earshot
options:
  deploymentTarget: { iOS: "26.0" }
  createIntermediateGroups: true
settings:
  base:
    DEVELOPMENT_TEAM: ${EARSHOT_TEAM}        # from TEAM in the gitignored ios/local.mk
    CODE_SIGN_STYLE: Automatic
    SWIFT_VERSION: "5.0"                    # Swift 5 language mode (contract 5)
    SWIFT_DEFAULT_ACTOR_ISOLATION: nonisolated
    TARGETED_DEVICE_FAMILY: "1"
    GCC_C_LANGUAGE_STANDARD: c11
    GCC_OPTIMIZATION_LEVEL: "2"             # -O2 for C in Debug and Release, the engine must stay real time
    GCC_FAST_MATH: NO                       # never -ffast-math (contract 4.1)
    GCC_PREPROCESSOR_DEFINITIONS: ["$(inherited)", "ACCELERATE_NEW_LAPACK=1"]   # same as the engine Makefile and make ios-check
    HEADER_SEARCH_PATHS: ["$(SRCROOT)/../engine/include", "$(SRCROOT)/Earshot/Audio"]
    SWIFT_OBJC_BRIDGING_HEADER: Earshot/App/Earshot-Bridging-Header.h
    COREML_CODEGEN_LANGUAGE: None           # we load the compiled model by URL, no generated class
targets:
  Earshot:
    type: application
    platform: iOS
    sources:
      - path: Earshot
      - path: ${EARSHOT_ENGINE_SRC}         # set by the Makefile: ../engine/src (real) or Stubs (stub)
        compilerFlags: ["-Wall", "-Wextra", "-Werror"]   # -O2 and -std=c11 come from the settings above
      - path: ../engine/include
        buildPhase: none
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.aarushkandukoori.earshot
        MARKETING_VERSION: "0.1"
        CURRENT_PROJECT_VERSION: "1"
    info:
      path: Earshot/App/Info.plist
      properties:
        CFBundleDisplayName: Earshot
        NSMicrophoneUsageDescription: "Earshot listens through the iPhone microphone to bring the person you choose forward. Audio never leaves this iPhone."
        NSMotionUsageDescription: "Earshot uses AirPods head tracking so you can switch to the person you are facing."
        UIBackgroundModes: [audio]
        UILaunchScreen: {}
        UISupportedInterfaceOrientations: [UIInterfaceOrientationPortrait]
    scheme:
      testTargets: [EarshotTests]
  EarshotTests:
    type: bundle.unit-test
    platform: iOS
    sources: [EarshotTests]
    dependencies: [{ target: Earshot }]
```

- **No entitlements file** - background audio is the `UIBackgroundModes` key, file protection is set per file, and head
  tracking needs only `NSMotionUsageDescription` (fact-check.md 6, 7). Nothing here needs a paid account.
- **Not in v1** - camera, HealthKit, `LSSupportsGameMode`, widgets and App Groups (contract 0 and 12).
- **Engine sources** - the app compiles `engine/src/*.c` directly (contract 7) with `-O2 -std=c11
  -DACCELERATE_NEW_LAPACK -Wall -Wextra -Werror`, the flags of `make ios-check` (without the define the iOS 26.0 SDK's
  `vecLib.h` selects the deprecated CBLAS interface). Until E-2 lands it compiles `Stubs/earshot_stub.c` instead. Both
  build against the same `engine/include/earshot.h`. This file is the only `project.yml` specification (20-engine 2.3
  points here); `make build ENGINE=real` must finish with zero warnings (I-0).
- **Core ML package** - the `.mlpackage` sits in `Earshot/Resources/Models/`, so Xcode compiles it to
  `wespeaker_resnet34_fp32.mlmodelc` in the bundle. I-0 checks the built `.app` contains that folder. If XcodeGen ever
  puts the package in the wrong phase, `SpeakerEncoder` falls back to `MLModel.compileModel(at:)` once and caches the
  result in `Application Support/ModelCache/`.

### 2.2 `ios/Makefile`

```make
# ENGINE is stub until E-2 lands, then: make project ENGINE=real
# DEVICE is the phone UDID, saved once with: make devices; echo <udid> > .device
ENGINE ?= stub
SRC     = $(if $(filter real,$(ENGINE)),../engine/src,Stubs)
DEVICE ?= $(shell cat .device 2>/dev/null)
BUNDLE  = com.aarushkandukoori.earshot
CONFIG ?= Release
APP     = build/Build/Products/$(CONFIG)-iphoneos/Earshot.app
XC      = xcodebuild -project Earshot.xcodeproj -scheme Earshot -derivedDataPath build

project:      ; EARSHOT_ENGINE_SRC=$(SRC) xcodegen generate --spec project.yml
build: project; $(XC) -configuration $(CONFIG) -destination 'generic/platform=iOS' -allowProvisioningUpdates build
install: build
	xcrun devicectl device install app --device $(DEVICE) $(APP)
	xcrun devicectl device process launch --device $(DEVICE) $(BUNDLE)
test: project ; $(XC) -destination 'platform=iOS Simulator,name=iPhone 17 Pro,OS=26.0' test
test-device: project ; $(XC) -destination 'id=$(DEVICE)' -allowProvisioningUpdates test
devices:      ; xcrun devicectl list devices
logs:         ; xcrun devicectl device copy from --device $(DEVICE) --domain-type appDataContainer \
                  --domain-identifier $(BUNDLE) --source Documents/logs --destination build/logs
phase0:       ; xcrun devicectl device copy from --device $(DEVICE) --domain-type appDataContainer \
                  --domain-identifier $(BUNDLE) --source Documents/phase0 --destination ../docs/measurements
privacy-check: ; ! grep -rnE 'URLSession|NWConnection|import Network|WKWebView' Earshot
```

- The simulator runtime on this Mac is iOS 26.0 with an "iPhone 17 Pro" device (measured, `xcrun simctl list`). No
  iPhone 16 Pro simulator exists, which does not matter for logic tests.
- `CONFIG=Release` is the default because Phase 0 and acceptance timings must come from optimized Swift. Use
  `make install CONFIG=Debug` for debugging with the Xcode debugger.

### 2.3 Signing with the free Personal Team

1. **Once, by the owner in the Xcode GUI** - Xcode, Settings, Accounts, add the Apple ID that owns the Personal Team.
   The command line cannot sign in; after this, `-allowProvisioningUpdates` lets `xcodebuild` register the phone and
   create the 7-day profile with automatic signing.
2. **First install** - on the phone, Settings, General, VPN & Device Management, trust the developer certificate.
   Developer Mode is already on (BRIEF.md).
3. **Every 7 days** - `make -C ios install` (about 2 min, contract risk 10). The app shows "Profile expires in N days"
   in Diagnostics from the embedded profile's expiration date, read at launch.
4. **Limits** - a free team allows 3 apps per device and 10 App IDs per 7 days (fact-check.md 7). If install fails with
   an app-limit error, the owner deletes another free-team app from the phone.
5. **Xcode 26.0.1 with iOS 26.1** - if Xcode reports missing device support, connect the phone with Xcode open once so it
   prepares the device, then retry the command.

### 2.4 Build settings that matter for real time

- `GCC_OPTIMIZATION_LEVEL 2` in every configuration, `GCC_FAST_MATH NO`, C11.
- Swift `-Onone` in Debug and `-O` in Release (Xcode defaults). Timing numbers are only valid from Release.
- No sanitizers in Release. Thread Sanitizer may be used in Debug for the Swift side only.

### 2.5 The stub engine (`Stubs/earshot_stub.c`, until E-2)

It implements every function of `earshot.h` so the app links and runs on day one.

- **Model and enrollment** - `es_model_load` accepts any buffer, `es_selftest` returns ES_OK with max error 0,
  `es_vad` implements the real contract 2.4 rule (it is 30 lines), `es_enroll` sets g to a unit vector derived from the
  clip's band energies and counts tokens with the real grouping rule, `es_cond_build` copies nothing,
  `es_model_data_hash` returns 64 zeros.
- **Engine** - starts muted like the real one; `es_engine_render` decimates by plain 3:1 averaging (nearest-sample
  pick at 44.1 kHz) into a 16 kHz history ring (for enrollment and alarm work; `es_engine_input_position` and the
  real read and cursor rules), plays the input straight through with the real mute ramp, 400 ms fade-in and a -3 dBFS
  hard ceiling, and fills `es_stats` (`last_frames`, `in_rms_db`, `out_rms_db`). Stats `p_target` stay 0;
  `es_engine_set_cond` accepts NULL.
- **Probe** - implements the contract 4.2 probe signal (5 sweeps 300 Hz to 6 kHz, 0.3 s, `ES_PROBE_DBFS`, 0.5 s apart) and a
  plain FFT GCC-PHAT, so Phase 0 can measure route latency before E-2. Phase 0 repeats the latency rows with the real
  engine and records both (`"engine": "stub" | "real"`).
- **Removed at integration** by the lead (contract 11); `make project ENGINE=real` switches the sources.

## 3. Audio session, microphones and sample rates

### 3.1 Session setup (`SessionConfig.swift`, main actor, foreground only)

```swift
func activateSession() throws -> RouteInfo {
    let s = AVAudioSession.sharedInstance()
    try s.setCategory(.playAndRecord, mode: .measurement, options: [.allowBluetoothA2DP]) // never HFP, speaker, mixable
    try s.setPreferredSampleRate(48_000)
    try s.setPreferredIOBufferDuration(64.0 / 48_000.0)          // 64 frames = 1.33 ms, a hint
    try s.setPrefersNoInterruptionsFromSystemAlerts(true)
    try s.setAllowHapticsAndSystemSoundsDuringRecording(true)
    try s.setActive(true)
    try selectBuiltInMic(s)                                         // after activation, and after every route change
    return RouteInfo(session: s)                                    // reads everything back
}
func selectBuiltInMic(_ s: AVAudioSession) throws {
    guard let mic = s.availableInputs?.first(where: { $0.portType == .builtInMic }) else { throw AppError.noBuiltInMic }
    try s.setPreferredInput(mic)        // no data source or polar pattern in v1: measurement mode uses the primary mic
}
```

- **Never set** `.allowBluetoothHFP` (old name `.allowBluetooth`), `.defaultToSpeaker`, `.mixWithOthers`,
  `.bluetoothHighQualityRecording`, voice processing or echo-cancelled input. HFP wins over A2DP when both are allowed
  and moves input and output to the headset (fact-check.md 5).
- **Read back** after activation and after every route change - `sampleRate`, `ioBufferDuration`, `inputLatency`,
  `outputLatency`, `currentRoute` inputs and outputs, and the input's `selectedDataSource` (logged, never set).
- **io_frames** = `Int((ioBufferDuration * sampleRate).rounded())`, clamped to 16...4096. One second after start the
  controller compares it with `stats.last_frames` (the real callback size). If they differ, it runs the engine swap once
  with the real value. iOS 18 silently granted 1,104 frames with Sound Recognition or Vocal Shortcuts on
  (ios-platform.md 1.5); if `last_frames` > 128 the app shows "iOS gave Earshot a large audio buffer (N frames). Turn off
  Sound Recognition and Vocal Shortcuts for the lowest delay."
- **Mode fallback** - `.measurement` is the contract mode. If Phase 0 shows it breaks A2DP output (output lands on the
  receiver) the single constant `SessionConfig.mode` becomes `.default` for A2DP routes, with a dated note in
  docs/DECISIONS.md (contract risk 5).

### 3.2 Which microphone listens and which enrolls (decision)

**Both listening and enrollment use the iPhone's built-in microphone, in the same session, through the same engine
input.** The AirPods microphones are never used in v1 (contract 0 excludes them).

Why, from the research
- **A2DP is output only.** AirPods audio input exists only over HFP (`.allowBluetoothHFP`). HFP moves both input and
  output to the hands-free link, takes priority over A2DP, loses AVRCP stem presses (no API receives HFP commands, Apple
  DTS) and was reported to stop `CMHeadphoneMotionManager` updates (ios-platform.md 2.3, fact-check.md extra checks).
- **iOS 26 high-quality AirPods recording** (`.bluetoothHighQualityRecording`) is mode `.default` only, "may increase
  input latency" per Apple, is not for real-time use, is not supported on AirPods Pro 1, and still moves the mic to the
  ears (ios-platform.md 2.4). `.farFieldInput` and `Mode.dualRoute` are iOS 26.2 APIs, not in the 26.0 SDK of Xcode
  26.0.1, and both require HFP (ios-platform.md 2.5).
- **Ear mics hear the wrong person.** They sit next to the user's mouth, so the user's own voice would dominate. The
  phone on the table is closer to the other people.
- **Enrollment must match listening.** Enrollment audio is read from the engine's 16 kHz input history
  (`makeInputTap()`), the exact samples the model sees later (same mic, same decimator, same gain), with no session
  change and no gap in listening. Mismatched enrollment and live channels make target extraction worse (quality.md B.1).
- **Own-voice enrollment** is also from the phone mic, because the model must suppress the user's voice as the phone
  mic hears it.

Check that it is really the phone mic - one report on iOS 18.4.1 said the route listed the built-in mic but sounded like
the AirPods mic (fact-check.md 5). Phase 0 runs the tap test (section 15.3) with AirPods in, and the app repeats a
one-line version in onboarding: "Tap the bottom edge of the phone" must move the input meter by at least 20 dB.

### 3.3 Sample rates and resampling

- RemoteIO client formats use the **hardware rate** (`session.sampleRate`), so RemoteIO itself never resamples (a rate
  mismatch can add latency, ios-platform.md 1.2).
- Accepted rates are 48,000 and 44,100 Hz (`es_engine_config.io_rate`). Anything else returns ES_ERR_RATE and the app
  shows "This audio route runs at N Hz, which Earshot does not support" and stays muted.
- All conversion to and from 16 kHz happens inside `es_engine_render` with the contract 4.2 minimum-phase polyphase FIR
  (48 kHz ratio 1/3, 44.1 kHz ratio 160/441; mean group delay 0.13 ms per direction at 48 kHz and 0.15 ms at 44.1 kHz,
  analytic, 20-engine 9.1). The app never uses
  `AVAudioConverter`.

### 3.4 Routes (`RouteInfo.swift`)

```swift
struct Port: Equatable { let type: AVAudioSession.Port; let uid: String; let name: String }
struct RouteInfo: Equatable {
    var outputs: [Port]
    var inputType: AVAudioSession.Port?          // must be .builtInMic
    var sampleRate: Double, ioBufferDuration: Double, ioFrames: Int
    var inputLatency: Double, outputLatency: Double   // seconds, as reported
    var allowed: Bool        // every output is .bluetoothA2DP, .headphones or .usbAudio, and input is .builtInMic
    var isBluetooth: Bool    // output .bluetoothA2DP -> slack16 43, else 22 (contract 4.2)
    var signature: String    // output uids + input type + sampleRate + ioFrames; a change triggers the engine swap (6)
}
```

- **Allowed outputs** - `.bluetoothA2DP`, `.headphones`, `.usbAudio` only. `.builtInSpeaker`, `.builtInReceiver`,
  `.bluetoothHFP`, `.airPlay`, `.carAudio` and anything else mute at once (contract 5.3).
- **Which AirPods** - iOS gives no model name, only the user-set port name. The first time a new A2DP port UID appears,
  the app asks "Which AirPods are these? AirPods Pro (1st generation) / AirPods Pro 2" and stores the answer in
  UserDefaults by port UID. It sets the default latency estimate text and the "Pro 2 is better for long conversations"
  tip (contract 10.2).
- **Wired** (`.headphones`, `.usbAudio`) works with the v1 model and slack16 22, and shows its measured delay honestly.
  The low-latency wired profile is later work and nothing in this file depends on it.

## 4. Audio I/O - RemoteIO and the C render callback

### 4.1 Why RemoteIO and not AVAudioEngine

- One C callback pulls the mic (`AudioUnitRender` on bus 1) and writes the output in the same call. That is the
  shortest deterministic path (input buffer plus output buffer), the structure of Superpowered's latency app
  (ios-platform.md 9.2), and it is what the contract fixes (`AudioHost.c`, contract 5).
- AVAudioEngine source and sink nodes take Swift or Objective-C blocks on the audio thread; real-time-safe block variants
  arrive only in iOS 27 (ios-platform.md 4.6). RemoteIO also gives the workgroup property directly.

### 4.2 `AudioHost.c`

`eh_create(void *es_engine, double io_rate, int io_frames, eh_host **out)`, in order
1. Find and instantiate `kAudioUnitType_Output` / `kAudioUnitSubType_RemoteIO` / `kAudioUnitManufacturer_Apple`.
2. `kAudioOutputUnitProperty_EnableIO` = 1 on input scope bus 1 and on output scope bus 0.
3. Formats, linear PCM Float32, packed, non-interleaved, at `io_rate` - mic side (output scope, bus 1) 1 channel;
   speaker side (input scope, bus 0) **2 channels** (the callback copies the mono output into both, so both AirPods play).
4. `kAudioUnitProperty_MaximumFramesPerSlice` = 4096, global (matches `es_engine_config.max_frames`).
5. Render callback on input scope bus 0, refCon = the host. Allocate `in_buf[4096]` floats and a one-buffer
   `AudioBufferList` with `calloc` here, never later.
6. `AudioUnitInitialize`, then read the workgroup once - `os_workgroup_t wg = NULL; UInt32 sz = sizeof(wg);`
   `AudioUnitGetProperty(unit, kAudioOutputUnitProperty_OSWorkgroup, kAudioUnitScope_Global, 0, &wg, &sz)`; keep it
   only if the status is noErr, `sz == sizeof(wg)` and `wg != NULL` (an uninitialized size gave a bad value in one
   report, ios-platform.md 4.4). `eh_workgroup` returns it or NULL. Release it in `eh_destroy` as the AudioUnit header
   comment for this property says (check the comment in the 26.0 SDK during I-0).

```c
static OSStatus eh_render(void *ref, AudioUnitRenderActionFlags *flags, const AudioTimeStamp *ts,
                          UInt32 bus, UInt32 n, AudioBufferList *io) {
    eh_host *h = (eh_host *)ref;
    float *left = (float *)io->mBuffers[0].mData;
    if (n > EH_MAX_FRAMES) {                                   /* cannot happen with MaximumFramesPerSlice 4096 */
        for (UInt32 b = 0; b < io->mNumberBuffers; b++) memset(io->mBuffers[b].mData, 0, n * sizeof(float));
        return noErr;
    }
    h->abl.mBuffers[0].mData = h->in_buf;
    h->abl.mBuffers[0].mDataByteSize = n * sizeof(float);
    if (AudioUnitRender(h->unit, flags, ts, 1, n, &h->abl) != noErr)
        memset(h->in_buf, 0, n * sizeof(float));               /* mic not ready yet: process silence */
    es_engine_render(h->engine, h->in_buf, left, (int)n);      /* the only engine call on this thread */
    if (io->mNumberBuffers > 1) memcpy(io->mBuffers[1].mData, left, n * sizeof(float));
    return noErr;
}
```

Callback rules (code review checklist, every PR touching `Audio/`) - no `malloc`/`free`, locks, `dispatch_*`, Swift,
Objective-C, `printf`/`os_log`, file I/O, `AudioUnitSetProperty`, or loops that depend on anything but `n`. Allowed calls
are exactly `AudioUnitRender`, `es_engine_render`, `memset`, `memcpy`. Mute, fades, limiter and probe all live inside
`es_engine_render` (contract 4.2), so the host stays this small.

## 5. Worker thread and lock-free hand-offs (what the app must do)

The engine owns the worker and the rings (contract 4.2, details in 20-engine.md). The app's job is to start and stop
them in the right order and to talk to them only through the lock-free calls.

- **Rings** - single-producer single-consumer with C11 atomics: in16 and out16 (4,096 samples), input history
  (262,144 samples, 15.36 s readable), per-instance output history (65,536), timings (1,024 x uint16). The callback is the only
  producer of in16 and the history; the worker is the only producer of out16 and timings. App readers
  (`InputTap`, `recentOutput`, `drainTimings`) only read.
- **Worker** - one pthread created by `es_engine_start_worker`, `THREAD_TIME_CONSTRAINT_POLICY` sized from the granted
  callback by the engine (contract 4.2: period 4 ms, computation 1.0 ms, constraint 2.5 ms at 64 frames; 23 / 6 /
  14.4 ms at 1,104 frames), preemptible, joined to the RemoteIO workgroup that the app passes in
  (`eh_workgroup`), FTZ on, single-threaded BLAS. It sleeps on a semaphore that the callback signals when in16 holds a
  full 64-sample hop. If the workgroup is NULL the worker still runs with the policy; Diagnostics shows "workgroup: no".
- **Conditioning** - each slot has two `es_cond` buffers and an atomic index (contract 4.3). The main actor calls
  `setConditioning(slot:_:resetState:)`, which copies into the idle buffer and flips the index; the worker picks it up at
  the next hop boundary and acknowledges. BUSY (previous flip not yet picked up) is retried every 4 ms for up to 200 ms
  by the wrapper. Because that wrapper suspends while retrying, `AudioController` chains every conditioning call through
  one task so two quick stem presses can never interleave:

```swift
private var condChain: Task<Void, Never>? = nil
func applyConditioning(slot: Int, _ c: Conditioning, reset: Bool) {
    let previous = condChain
    condChain = Task { @MainActor in
        await previous?.value
        do { try await engine?.setConditioning(slot: slot, c, resetState: reset) }
        catch { diagnostics.note("set_cond slot \(slot) failed: \(error)") }   // after 200 ms of BUSY
    }
}
```

- **Emptying a slot** - `clearConditioning(slot:)` (`es_engine_set_cond(NULL)`, same BUSY retry, chained through the
  same task) after the slot's gain has been 0 for 150 ms; the worker then skips that slot, and the next
  `setConditioning` into it always resets (contract 4.3).
- **Building conditioning** - `Conditioning(model:mode:target:own:useWeSpeaker:)` (`es_cond_build`, FiLM 1.25 M MACs)
  runs in `EnrollmentService` or a detached task, never in the callback or worker. The engine copies it, so the Swift
  object may be dropped after the call.
- **Params** - `setParams` (seqlock inside the engine) from the main actor only. The worker reads once per hop and keeps
  the old params on a torn read.
- **Reading state** - `stats()` at 10 Hz on the main actor (meters, `p_target` for the Selector and LockedCheck),
  `drainTimings()` once per second (telemetry histogram), `latencyMS` when the latency chip refreshes.

## 6. Lifecycle (`AudioController.swift`)

States - `idle`, `running`, `paused` (I/O running, output muted), `badRoute` (I/O running, muted), `interrupted`,
`failed(message)`. One function, `swapEngine(reason:)`, is the only way a running engine is replaced, and it never stops
RemoteIO (contract 5.3 Lifecycle). `recreateIO(reason:)` (RemoteIO and engine) runs only for a hardware rate change or
media services reset.

**start()** (foreground only; the Start button is disabled unless `scenePhase == .active` and both self-tests have run;
`es_selftest` failing blocks audio entirely, contract 3.3)
1. `activateSession()`. A disallowed route still starts I/O, muted, with "Connect AirPods to listen" (the input meter,
   enrollment and Phase 0 need the mic).
2. `EarshotEngine(model:ioRate:ioFrames:instances:slack16:)` with instances 3 (thermal downshift empties slots instead
   of changing this), slack16 43 on Bluetooth or 22 wired. The engine starts muted (contract 4).
3. `setParams`, then `setConditioning` for every filled slot with `resetState: true` (worker not running yet, so no
   BUSY); slots with nobody stay empty.
4. `eh_create(engine.cEngine, rate, frames, &host)`, `engine.startWorker(workgroup: eh_workgroup(host))`, `eh_start`.
5. Unmute last, and only if the route is allowed and an own voice exists - `setMuted(false)` (400 ms fade-in). Then Now
   Playing, head tracking, alarm listener, locked check, session log.

**Setup mode** - before the own voice is enrolled, the engine runs in Everyone mode with no own voice (contract 2.3) and
stays muted, so onboarding can show the input meter and record the own voice without the user hearing themselves late.
Output is also muted during every own-voice recording.

**stop()** - `setMuted(true)`, wait 10 ms (5 ms ramp), `eh_stop`, `stopWorker`, `eh_destroy`, release the engine,
`setActive(false, options: .notifyOthersOnDeactivation)`, mark "tonight" expiry (section 9).

**swapEngine(reason:)** - RemoteIO keeps running the whole time, so it also works with the screen locked (a recording
session cannot start from the background, Apple DTS). Re-read the route; build the new engine (step 2, muted, same
io_rate, `max_frames` 4096, the new io_frames); step 3; `newEngine.startWorker(workgroup: eh_workgroup(host))`; copy
`old.recentInput(samples: 128000)` into QuickAdd's RAM buffer; `old.setMuted(true)` and wait 10 ms; `eh_set_engine(host,
new.cEngine)` (returns once the callback acknowledged); `old.stopWorker()`; release the old engine; recreate every
`InputTap` (CaptureSession fails a running capture with "Audio was interrupted", AlarmListener re-taps); step 5
(unmute with the 400 ms fade-in). Reasons - route signature changed with the same hardware rate, callback size differs
from io_frames, interruption ended. A new engine starts at the default slack (contract 4.2). Every swap is logged with
its reason and duration.

**recreateIO(reason:)** - only when the hardware rate changed (the RemoteIO client format must follow it, 3.3) or
after media services reset. Foreground - mute, wait 10 ms, `eh_stop`, `stopWorker`, `eh_destroy`, free the engine, then
steps 2 to 5 of start. Background - mute, keep the session, state `paused`, post "Tap to resume Earshot"; opening the app
runs it.

**Route changes** - `AVAudioSession.routeChangeNotification` arrives off the main thread; hop to the main actor.
1. `engine.setMuted(true)` at once (RT-safe, 5 ms ramp). Target - muted within 50 ms of a bad route (contract 9).
2. Debounce 300 ms (the change from `setPreferredInput` itself fires a notification), then `selectBuiltInMic`.
3. New `RouteInfo`. Not allowed - stay `badRoute` with I/O running (the app must keep its session alive while locked).
   Allowed and signature changed - `swapEngine(reason: .route)`, or `recreateIO(reason: .rate)` if the hardware rate
   changed. Allowed and unchanged - unmute.
4. A new A2DP connect starts the latency prompts (section 7). AirPods taken by another Apple device look like a route
   loss; onboarding sets "Connect to This iPhone - When Last Connected".

**Interruptions** (calls, Siri, Clock alarms, other non-mixable apps; `interruptionNotification`)
- **Began** - mute, state `interrupted`. iOS has already stopped I/O.
- **Ended with `.shouldResume`** - `setActive(true)`; on success `eh_start` (iOS stopped I/O) and
  `swapEngine(reason: .interruption)` (the route may have changed during the call) with the 400 ms fade-in. On failure (expected in the background, Apple DTS) post a local
  notification "Earshot paused by a call. Tap to resume." Opening the app from it resumes automatically.
- **Ended without `.shouldResume`**, or reason `.routeDisconnected` (iOS 17+ interrupts Now Playing apps when the route
  drops) - stay paused; foreground shows "Resume"; background posts "Tap to resume Earshot".
- `setPrefersNoInterruptionsFromSystemAlerts(true)` keeps banner-style incoming calls from interrupting until answered.

**Media services reset** - `mediaServicesWereResetNotification`: drop host and engine without further calls, keep
`EarshotModel` (plain C memory), reload `SpeakerEncoder`, then `start()` if in the foreground, else notify.

**Background, screen locked** - `UIBackgroundModes audio` keeps a running session alive; with input running the I/O size
does not grow on lock (ios-platform.md 1.1). A recording session cannot be activated from the background, so pause never
stops I/O (it mutes), a route or buffer change swaps only the C engine (RemoteIO keeps running), and nothing calls
`setActive(true)` from the background except the `.shouldResume` attempt above. Phase 0 checks that 10 locked route
changes and 1 forced swap keep audio alive without a tap (15.3).

**Thermal and Low Power** (`ProcessInfo.thermalStateDidChangeNotification`, `NSProcessInfoPowerStateDidChange`)
- `.serious` or Low Power Mode - no rebuild (contract 5.3): set every non-selected slot gain to 0, wait 150 ms, then
  `clearConditioning` on those slots, so only the selected slot (the Everyone slot in Everyone mode) is computed; alarm
  classifier every 3 s; banner "Saving power - only the selected person is ready". The selected person keeps playing
  with no gap.
- `.critical` - fade to pause with the notice "iPhone is hot - Earshot paused" (local notification when locked).
- Recovery - state `.fair` or better and Low Power off for 60 s, then `setConditioning` refills the table slots (other
  people come back cold, reset).

## 7. Latency measurement and display (`LatencyMeter.swift`)

**What is measured.** `es_engine_probe_*` replaces the output with 5 exponential sweeps (300 Hz to 6 kHz, 0.3 s each,
at `ES_PROBE_DBFS`, -30 dBFS until Phase 0 picks -30, -24 or -18, 0.5 s apart, 4.0 s) and records the mic at the I/O rate; GCC-PHAT gives the median lag `loop_ms` = the
round trip of a plain pass-through on this route (input path, both I/O buffers, Bluetooth, AirPods). The probe signal
is injected after the safety chain, so Earshot's own processing is not in the loop. **Mouth to ear = loop_ms +
`es_engine_latency_ms`** (10.95 ms at 48 kHz with a 64-frame grant and slack16 43, analytic; about 12.3 ms on a
128-frame grant and 11.3 to 15.2 ms on a 44.1 kHz route, and it grows if slack grows).

**How the user does it** (`measureLatency()`, needs state `running`)
- **Method A, in-ear (default)** - "Keep your AirPods in. Hold the bottom edge of the phone against the outside of your
  right AirPod for 5 seconds. Volume about half." No settings change.
- **Method B, ear detection off** - "Turn off Automatic Ear Detection (Settings, Bluetooth, your AirPods), take out the
  right AirPod and hold its speaker to the bottom of the phone." Offered when A fails. Phase 0 measures both and decides
  which is the default (contract 11, I-1).
- Poll `latencyProbeResult()` every 100 ms, timeout 8 s. ES_ERR_PROBE_WEAK (fewer than 3 sweeps with psr >= 10 dB),
  or a spread over 10 ms on Bluetooth / 2 ms wired, which LatencyMeter checks itself because the engine is never told
  the route (contract 4.2), shows "Couldn't hear the test sound. Turn the volume up a little and try
  again" and offers method B.
- A result is stored per output port UID with time, method, spread and psr (UserDefaults, a few bytes, no audio).
  When a new probe fails, the last valid result for that port stays on screen with its age.

**When** - prompt (never automatic, since it needs the user's hands) at every AirPods connect, then banners at 10 and 30
minutes after connect ("Re-check the delay - it can drift"), offered and never required; wired routes measure once per connect. Planning numbers are
about 43 ms Bluetooth for Pro 2 and 84 ms for Pro 1 (benchmark, fact-check.md 2), so expect about 59 and 100 ms mouth to
ear (estimate) with a possible range of about 56 to 236 ms.

**What is shown** (delay chip on Listen, Diagnostics, Now Playing artist line; strings are unit-tested)
- Measured - `"Delay \(E + R) ms (measured \(age)) - Earshot \(E) + route \(R)"` with `E = round(latencyMS)`,
  `R = round(loopMS)`, so the parts always add up. Example "Delay 59 ms (measured 3 min ago) - Earshot 11 + route 48".
  Age is "just now" under 60 s, "N min ago", then "N h ago".
- Not measured on this route - `"about \(N) ms (estimate)"` with
  `N = round(1000 * (inputLatency + outputLatency + 2 * ioBufferDuration) + latencyMS)`.
- Never a total under 20 ms on Bluetooth - a Bluetooth `loop_ms` under 20 ms is a method error (the sweep leaked
  acoustically from somewhere else) and is rejected like a weak probe. Never the word "real time". Tap the chip to
  measure again.
- Diagnostics shows both the measured and the reported value; telemetry logs both (`latency_measured_ms` = loop + E, or
  null; `latency_reported_ms` = the estimate).

## 8. Enrollment

### 8.1 Capture (`CaptureSession.swift`)

- **Target** - tap "Add person" (or a stem double-press with an empty table, section 11). Screen - "Hold the phone toward
  them while they talk". `tap = engine.makeInputTap()` (it starts at the current write position, so nothing from
  before the press, contract 5); every 250 ms read new 16 kHz samples into a preallocated buffer (15 s), run
  `model.vad` on everything so far and show `speech_s = kept frames x 0.032` on a ring. If `tap.droppedSamples` grows
  or the engine is swapped, the capture fails with "Audio was interrupted - try again". Stop at 6 s of VAD
  speech or 15 s total (contract 5.3); "Done" is allowed from 4 s. Hands-free cues - 1 beep at start, 2 at success,
  3 at failure (`engine.cue`).
- **Own voice** - onboarding or Settings; the user reads `passage.txt` in a quiet place with the phone held normally;
  stop at 8 s of speech or 20 s.
- **Head direction** - while capturing, keep every yaw sample from `HeadTracker`. If motion was live for at least half
  the capture, the person's direction = circular mean of the yaw samples taken during speech frames; else `yaw_deg` null.
- Listening continues during capture (Everyone mode or the current person); no session change.

### 8.2 Clip preparation (`ClipTools.swift`, all pure functions, unit-tested)

Steps 1-5 here and the formulas of 8.3 are the contract 6 check formulas, copied there word for word; the contract
governs. `ClipToolsTests` checks them against `quality_vectors.json` from WP-ML. `es_vad` below means `model.vad`
(contract 5).

Token frame f covers raw samples `[512f - 448, 512f + 64)`; its midpoint is `512f - 192` (contract 2.4 framing).
1. `mask0 = es_vad(full)`. Span = `[max(0, 512 f1 - 448), min(n, 512 f2 + 64))` for the first and last kept frames.
2. Longer than 10 s - keep the 10 s window (start on a 512-sample step) holding the most `mask0` frame midpoints,
   earliest on ties. For own voice, always take this 10 s best window.
3. Round the length down to a multiple of 8,000 samples (0.5 s), keeping the start. Training enrollments are 3 to 10 s
   in 0.5 s steps (contract 8.1), and Core ML sees at most 13 distinct input lengths. Under 64,000 samples (4 s) fails.
4. 1.5 s windows for WeSpeaker - start `8000 w`, length 24,000, `w = 0 .. floor((len - 24000) / 8000)`.
5. A frame is "inside" window w when its midpoint lies in `[8000 w, 8000 w + 24000)`.

### 8.3 Checks and voiceprint (`EnrollmentService.enroll(pcm16k:role:own:saved:)`)

Target, in order (thresholds from `thresholds.json`, never hard-coded; boundaries inclusive on the pass and warn side)

| Step | Computation | Pass / warn / fail |
|---|---|---|
| 1 | `sbr_db = 10 log10(mean x^2 over mask0 speech frames / mean x^2 over the other frames)` on the full capture (if no non-speech frame exists, use the lowest 10% energy frames) | >= 6 / 0 to 6 / < 0 dB |
| 2 | `clip_pct = 100 x count(abs(x) >= 0.98) / n` on the full capture | < 0.1 / 0.1 to 1 / >= 1 % |
| 3 | trim (8.2), `e = encoder.embed(clip)`, `e_w` for every window | - |
| 4 | `one_voice = median_w cos(e_w, e)` | >= 0.60 / 0.45 to 0.60 / < 0.45 |
| 5 | `own_cos = cos(e, own.e_wespk)` | < 0.50 / 0.50 to 0.60 / >= 0.60 |
| 6 | `mask = es_vad(clip)`, then zero every frame inside a window with `cos(e_w, own e) > own_window_cos` (0.60) | - |
| 7 | `Voiceprint(model:, pcm16k: clip, mask: mask, role: .target, weSpeaker: e)`; ES_ERR_ENROLL_SHORT is a fail | - |
| 8 | `speech_s = voiceprint.speechSeconds` | >= 5 / 4 to 5 / < 4 s |
| 9 | `dup = max cos(e, p.e_wespk)` over saved and tonight people | >= 0.75 asks "Sounds like Maya - update Maya?" |

- **Verdict** - any fail is a fail (first failing message shown); else any warn is "accepted, will be OK" in yellow;
  else pass. Messages - "Keep the phone on them a few more seconds", "Too noisy - move the phone closer", "I heard more
  than one voice - try when only they are talking", "That sounded like your voice", "Too loud - hold the phone a bit
  farther away".
- **Own voice** - steps 1, 3 and 8 only (role `.own`, mask `es_vad(clip)`); `speech_s` must reach 8 s on the full
  capture; `snr_db` = step 1 formula, >= 20 pass, 15 to 20 warn, < 15 fail. After a new own voice, every slot's
  conditioning is rebuilt with `resetState: false`.
- **Without WeSpeaker** (encoder self-test failed, contract 6) - step 3 trims without embedding; steps 4, 5 and 9 are
  shown as "not checked"; step 6 keeps `es_vad(clip)` with no own-window removal; `Voiceprint(... weSpeaker: nil)`,
  `useWeSpeaker: false`.
- `quality` stored in person.json - `speech_s`, `snr_db` (= sbr_db for targets), `consistency` (= one_voice),
  `own_cos`.
- Cost - the fp32 package shipped took 18 ms median per 5 s clip on the M4 CPU and GPU (measured, data-training.md 4.3;
  the 7.7 ms figure is the mixed-precision package); a 10 s target is 1 clip plus up to 18 windows, so about 0.2 to
  0.5 s on the phone (estimate, logged by Phase 0).

### 8.4 `SpeakerEncoder` (actor)

- `init(modelURL:)` loads `wespeaker_resnet34_fp32.mlmodelc` twice, with `computeUnits = .cpuAndGPU` and `.cpuOnly`
  (contract 6; the mixed-precision package gave cosine 0.57 on the Neural Engine and the fp32 package 0.9995 with
  `.all`, measured; never `.all`). `embed` uses `.cpuOnly` whenever `UIApplication.shared.applicationState != .active`
  (read on the main actor before the call), because background apps cannot submit GPU work and hands-free capture runs
  with the screen locked.
- `embed(_:)` requires 16,000 to 160,000 samples, builds an `MLMultiArray` [1, N] float32 named `wav` from raw samples in
  [-1, 1], reads `embedding` [1, 256], checks every value is finite, L2-normalizes, returns 256 floats.
- `selfTest()` reads `wespk_selftest.est` (`clip` [80000], `embedding` [256]) with `ContainerFile`; cosine >=
  `meta.tolerances.cosine_min` (0.999) passes; else reload with `.cpuOnly` and retry; else the app runs with `use_wespk = 0` and shows "Voice check
  unavailable - Earshot still works, but picks people less reliably".
- Never called from the callback or worker, never in the live path (contract 6).

### 8.5 Quick add (`QuickAdd.swift`)

"Add who just spoke" takes the last 8 s with `engine.recentInput(samples: 128000)` (the engine already holds 15.36 s of
history, contract 5), or, right after an engine swap, from the RAM copy `swapEngine` made (never written to disk). It
runs the target pipeline, and plays back the 2 s window with the most speech frames: engine muted
(`setMuted(true)`), clip normalized to -26 dBFS RMS with peaks at most -6 dBFS, played with `AVAudioPlayer(data:)` from
an in-memory WAV (same session, foreground only), then unmuted. "Is this them?" Yes saves a "tonight" person; No
discards everything.

## 9. People store and privacy

### 9.1 Files (`PeopleStore`, main actor; file work on a utility queue)

- `Application Support/People/<uuid>/person.json` and `enroll.wav` exactly as contract 5.1: `{id, name, kind
  ("tonight" | "saved" | "own"), created, updated, expires, yaw_deg, model_hash, e_wespk, mask, quality {speech_s,
  snr_db, consistency, own_cos}}`, ISO 8601 dates, `mask` = base64 of the uint8 token-frame mask passed to `es_enroll`, explicit `CodingKeys` (no automatic snake case). WAV - 16 kHz mono 16-bit,
  44-byte header, `Int16(clamp(x, -1, 1) * 32767)`, at most 10 s.
- Written with `.atomic` and `.completeFileProtectionUntilFirstUserAuthentication`, so a locked phone can still load
  them. The `People/` and `Documents/logs/` folders get `isExcludedFromBackup = true`, so voiceprints never reach an
  iCloud backup.
- `model_hash` = `model.modelHash` (`es_model_data_hash`, the `.esw` header's `data_sha256`, contract 3.1), the same
  value WP-ML writes into `thresholds.json` and `meta.weights_sha256`; the app never hashes the file itself.
- **Launch rebuild** - for each person, read the WAV and the stored `mask` and build `Voiceprint` with the stored
  `e_wespk` (no WeSpeaker call, a few ms of C per person, estimate). Only a file without `mask` gets it recomputed
  (8.3 step 6 with the stored own e when WeSpeaker works, else `es_vad(clip)`) and saved. Runs in the background, one
  person at a time; a person's card shows "getting ready" until done.

### 9.2 Kinds and expiry

- "Tonight" is the default kind. On creation `expires = now + 12 h`; while listening, every minute sets
  `expires = now + 12 h`; `stop()` sets it to stop time + 12 h. So people expire 12 h after the session ends even if the
  app was killed. `purgeExpired()` runs at launch, on every foreground and every 10 minutes.
- "Saved" needs consent - "Did Maya agree to have her voice saved on this phone? Voiceprints count as biometric data."
  Buttons "She agreed - save" and "Tonight only". `expires` null.
- "Own" - one person, `expires` null, re-recordable in Settings. Delete removes the folder at once; "Delete everyone"
  is in People. Deleting a person who sits on the table first sets that slot's gain to 0, waits 150 ms and calls
  `clearConditioning(slot:)` (contract 5.3), so a deleted voiceprint is never computed again.

### 9.3 Privacy rules

- No audio, voiceprint or log ever leaves the phone. The app contains no networking code (`make privacy-check` fails the
  build otherwise), no analytics, no third-party SDKs.
- The only audio written to disk is each person's `enroll.wav` (at most 10 s). Conversations, quick-add rings and probe
  recordings stay in RAM. Logs hold numbers only (contract 5.2).
- iOS shows the orange microphone indicator while Earshot runs; the app never tries to hide it.

### 9.4 Launch sequence (`Launch.swift`, before the Start button is enabled)

Runs once per launch off the main actor (utility queue), in this order; each failure has one fixed outcome.

| Step | Does | On failure |
|---|---|---|
| 1 | Load `earshot.esw` into `EarshotModel` | blocking error screen "Earshot's model is missing or damaged - reinstall", no audio ever starts |
| 2 | `model.selfTest(testVector: testvec_short.est)` (`es_selftest`, includes the `meta.weights_sha256` check, contract 3.3) | Start stays disabled; Diagnostics shows the status and max error (contract 3.3: no audio if it fails) |
| 3 | Decode `thresholds.json` | missing or `version != 1` - blocking error; `model_hash != model.modelHash` - use the file, Diagnostics warning "Thresholds were calibrated for another model" (contract 6) |
| 4 | `SpeakerEncoder.selfTest()` - `.cpuAndGPU`, then `.cpuOnly`, then `use_wespk = 0` with the notice of 8.4 | never blocks; the result sets `useWeSpeaker` for every conditioning |
| 5 | `purgeExpired()`, then rebuild people one at a time from `enroll.wav` and the stored `mask` (9.1) | a person whose files fail to load is shown as "needs re-adding" and skipped |
| 6 | Enable Start after steps 1 to 4; a person becomes selectable once rebuilt | - |

`LaunchTests` (XCTest with the stub) covers every failure branch - missing weights, a self-test failure, missing
thresholds, a thresholds hash mismatch, an encoder self-test failure, and a person without `mask`.

## 10. Picking by looking (`HeadTracker`, `TableState`, `Selector`)

### 10.1 Head tracking

- `CMHeadphoneMotionManager` (AirPods Pro 1 and 2 both supported, no entitlement, `NSMotionUsageDescription` required or
  the app crashes; fact-check.md 6). `startDeviceMotionUpdates(to:)` on a serial `OperationQueue`; the handler hops to
  the main actor with `(timestamp, yaw = attitude.yaw in degrees)`. Delegate connect and disconnect events are observed.
- The rate is computed from timestamps (about 25 Hz is a community report, undocumented). Only differences of yaw are
  used, so its sign and zero do not matter.
- **Smoothed yaw** - circular mean `atan2(mean sin, mean cos)` of the samples in the last 200 ms (the newest sample if
  fewer than 2). **Head rate** - `abs(wrap(y_now - y_200ms_ago)) / 0.2 s`. `wrap(a)` maps to [-180, 180).
- **Live** - newest sample under 0.5 s old. **Unavailable** - no sample for 2 s (for example screen locked, if Phase 0
  shows updates stop); the Listen screen shows "Head tracking paused" and look-select falls back to cycling.
- **Stale directions** - two raw samples with `abs(wrap(delta)) > 45` degrees less than 50 ms apart (contract 5.3), or
  any disconnect / connect event (the reference frame may reset).

### 10.2 Table, slots and modes (`TableState`)

- The **table** is up to 3 people for this session, one per engine slot (0, 1, 2), all kept warm (running) so switching
  between them is a 150 ms crossfade. Adding a 4th replaces the least recently selected slot (cold switch, reset).
- **Focus** - `slotGain[selected] = 1`, other people's slots 0 (still running). **Table** (needs >= 2 people) - every
  person slot 1, outputs summed. **Everyone** - the Everyone conditioning goes into a free slot (reset), or into the
  selected person's slot if all 3 are full (that person returns cold later); only that slot has gain 1. With no person
  yet the app starts in Everyone mode in slot 0 (contract 5.3).
- Every person conditioning is `Conditioning(mode: .person, target: vp, own: ownVP, useWeSpeaker: u)` and Everyone is
  `Conditioning(mode: .everyone, target: nil, own: ownVP, useWeSpeaker: u)`, where `u` = encoder self-test passed and
  every voiceprint involved has WeSpeaker vectors.

### 10.3 Look-select (stem double-press or the "Look" button)

```swift
func lookSelect() {
    let people = table.filledPersonSlots                     // [(slot, person, dir?)]
    guard !people.isEmpty else { return startHandsFreeCapture() }
    let current = (table.mode == .focus) ? table.selected : nil     // in Table or Everyone mode nobody is "current"
    if head.live, !head.stale, let y = head.smoothedYaw {
        let withDir = people.filter { $0.dir != nil }
        if let best = withDir.min(by: { abs(wrap(y - $0.dir!)) < abs(wrap(y - $1.dir!)) }) {
            let dBest = abs(wrap(y - best.dir!))
            let dCur = current.flatMap { c in people.first { $0.slot == c }?.dir }.map { abs(wrap(y - $0)) } ?? .infinity
            if dBest <= 35, best.slot != current, dBest + 10 <= dCur { return focus(on: best.slot) }  // contract 5.3
        }
    } else if head.live, head.stale, let y = head.smoothedYaw, let talker = table.onlyPersonTalking(last: 1.0) {
        table.reanchor(slot: talker, to: y)          // that person's dir = y, every other dir shifts by the same offset
        return focus(on: talker)
    }
    cycle()                                          // next filled slot after `current` in slot order, then Focus
}
```

- `focus(on:)` sets Focus, applies slot gains (warm, no reset), updates Now Playing, plays `slot + 1` beeps.
- `onlyPersonTalking(last:)` - exactly one table person whose `p_target` stayed above 0.7 for the last 1 s (10 Hz stats).
- Tapping a person's card does the same as `focus(on:)`.

### 10.4 Direction learning (main actor, 10 Hz with the stats poll)

For each table person: if `p_target[slot] > 0.7` has held for 1 s and head rate < 20 deg/s, apply
`dir += 0.1 * wrap(yaw - dir)` (or `dir = yaw` if dir is null) and restart that person's 1 s timer, so at most one update
per second (contract 5.3). Learned values are written to `yaw_deg` when the session stops. Head yaw has no fixed
reference between sessions, so every direction loaded at launch starts stale and is re-anchored by the first look-select
while facing a talking person.

## 11. Stem presses and Now Playing (`RemoteCommands.swift`)

What is possible (ios-platform.md 6, fact-check.md extra checks)
- With the A2DP-only session, AirPods presses arrive as AVRCP commands through `MPRemoteCommandCenter`, but only while
  Earshot is the Now Playing app - a non-mixable `.playAndRecord` session active, audio running, at least one command
  handler, `nowPlayingInfo` set. The Lock Screen buttons send the same commands.

| Press | Command(s) | Earshot action |
|---|---|---|
| Single | `togglePlayPauseCommand`, `playCommand`, `pauseCommand` | pause / resume (`play` only resumes, `pause` only pauses) |
| Double | `nextTrackCommand` | `lookSelect()`; with an empty table, hands-free target capture |
| Triple | `previousTrackCommand` | Focus <-> Table if >= 2 people, else Focus <-> Everyone; 1 / 2 / 3 beeps = Focus / Table / Everyone |

What is not possible
- Press and hold is reserved by iOS for noise control. Apps cannot read or set ANC / Transparency.
- No HFP button events reach apps (Earshot never uses HFP anyway). `AVCaptureEventInteraction` stem clicks need H2
  AirPods and a capture session, so they are not used.
- Removing an AirPod with ear detection on sends `pause`; Earshot pauses (safe). Conversation Awareness may duck or
  pause; onboarding asks to turn it off and Phase 0 tests it.

Details
- All other commands (skip intervals, seek, position, rate, like, bookmark) get `isEnabled = false`.
- **Pause** never stops I/O (restart from the background is impossible). It sets every slot gain to 0 and room sound
  off (`roomDB = -60`) together (150 ms crossfade, contract 5.3), then after 150 ms `setMuted(true)`, so room sound and
  bypass never stay at full level. **Resume** calls `setMuted(false)` (400 ms fade-in) and restores gains and room.
- `nowPlayingInfo` - title "Earshot - Focus: Maya" (or "Table: 2 people", "Everyone", "Paused"), artist = latency chip
  text, `MPNowPlayingInfoPropertyIsLiveStream = true`, playback rate 1. Rate 0 while paused is used only if Phase 0 shows
  a single press still arrives while paused.

## 12. Safety, alarm passthrough and the locked check

### 12.1 Output safety (engine does the DSP, the app enforces the rules)

- **In the engine** (contract 4.2) - 5 ms mute ramp, 400 ms raised-cosine fade-in at start and on unmute, peak limiter
  at -3 dBFS (zero lookahead, 50 ms release), soft clip above -1 dBFS, hard clip, runaway guard (output RMS over input
  RMS + 30 dB for 200 ms ducks 20 dB and sets `howl`), mic-to-ear gain at most +24 dB (50 ms RMS, post step 5 gain
  cap).
- **In the app** - output only on allowed routes; mute first, ask questions later; fade-in on every start, resume,
  engine swap and interruption end; UI ranges equal the `es_params` ranges (voice -10 to +6 dB, room Off / -24 / -15 /
  -8 dB with AirPods default Off, max gain 0 to 18 dB under Settings, Advanced; trim -40 to 0 dB); `stats.howl` shows
  "Feedback detected - output lowered".
- **Volume** - the app cannot set system volume. If `AVAudioSession.outputVolume > 0.75` at start it shows "Start with
  the volume at about half". Onboarding asks for Reduce Loud Audio (Settings, Sounds & Haptics, Headphone Safety, 85 dB).
- **No medical claims** - Earshot is "a listening tool, not a hearing aid". Every user-facing string lives in
  `Copy.swift`; a unit test fails on "hearing aid" (outside that exact disclaimer), "treat", "cure", "hearing loss",
  "medical", "real time", "under 20 ms", "removes all" and on any em dash.

### 12.2 Alarm passthrough (`AlarmListener.swift`, a helper, not a safety device)

- Own `InputTap` (recreated after every engine swap), read every 250 ms on a serial utility queue into `SNAudioStreamAnalyzer` (16 kHz mono format) with
  `SNClassifySoundRequest(classifierIdentifier: .version1)`, window 3 s, overlap 0.5 (overlap 0, every 3 s, under
  thermal downshift).
- Labels - `smoke_detector`, the siren family (`siren`, `police_siren`, `ambulance_siren`, `fire_engine_siren`,
  `civil_defense_siren`, `emergency_vehicle`, all present in the built-in label list, measured in product.md B.5),
  `car_horn`, `screaming`, `glass_breaking`, `baby_crying`. Confidence >= 0.6 in 2 consecutive windows triggers.
- Action (main actor) - `bypass = true`, `outGainDB = -6` for 5 s, `cue(beeps: 3)`, banner "Alarm-like sound - letting
  the room through", then restore the previous params. While paused it also unmutes for those 5 s. Toggle in Settings
  (default on).

### 12.3 Locked check (`LockedCheck.swift`, actor, every 1 s)

For each person slot with gain > 0 whose mean `p_target` over the last 3 s is above 0.5 - `recentOutput(slot:samples:
48000)`, `model.embedG` (`es_embed_g`, contract 5), cosine to the person's g. Below
`locked_cos` counts as lost; lost for 5 s while `p_target > 0.5` shows "Might be following someone else - double-press or
re-add". No automatic gain change in v1 (contract 5.3). The card shows a check mark while locked.

## 13. Screens and onboarding (SwiftUI)

- **Onboarding** (first launch, about 4 minutes, every step skippable except 2 and 5)
  1. What it is - "Earshot brings one person forward and pushes the rest down. It is a listening tool, not a hearing aid.
     Over AirPods their voice reaches you late, about 0.06 s on AirPods Pro 2 and 0.1 s on Pro 1 (estimate until Earshot
     measures it), so keep Noise Cancellation on." Credits - the
     interaction comes from UW's Look Once to Hear (Veluri et al., CHI 2024), no code used; the voice-check model is
     WeSpeaker trained on VoxCeleb, research terms, so this build is for personal use only.
  2. Permissions - microphone, motion, notifications (resume after calls).
  3. Which AirPods (3.4), plus the tap test (3.2).
  4. AirPods checklist, each with a "how" sheet - Noise Cancellation on (optional Shortcuts automation "When AirPods
     connect, Set Noise Control Mode"), Conversation Awareness off, Personalized Volume off, Spatial Audio off for Earshot
     (Control Center), Connect to This iPhone "When Last Connected", Sound Recognition off, Reduce Loud Audio on.
  5. Your voice (8.1) - "So Earshot never plays your own voice back to you." Mandatory before listening.
  6. Measure the delay (section 7), result with one sentence on why it matters.
  7. First listen - 10 s in Everyone mode with a hold-to-compare button (`bypass`).
  8. Controls card - the stem table and beep meanings (also in Help).
- **Listen** (main) - who is playing (name, mode), up to 3 table chips with a talking dot (`p_target > 0.5`) and a locked
  check, Look / Add person / Add who just spoke buttons, mode switch, voice level slider, room sound control, pause,
  hold-to-compare, delay chip, banners (route, thermal, alarm, lost lock, large buffer, head tracking paused).
- **Add person** - instruction, speech ring with seconds, live input meter, result card with each check in green /
  yellow / red / grey, name field (default "Person N"), Tonight / Save with the consent prompt.
- **People** - You, Tonight, Saved; rename, re-record, add to table, delete, delete everyone.
- **Settings** - voice level default, room sound, mute while I talk (`duck_own`, default on), activity gate, voice
  leveling (AGC), alarm passthrough, Advanced (max gain), re-record my voice, measure delay, About (credits, licenses,
  personal-use note, privacy summary).
- **Diagnostics** - section 14. **Help** - stem table, beeps, checklist, what the delay means.
- Accessibility - Dynamic Type, VoiceOver labels on every control (VoiceOver speech plays into the same AirPods), large
  touch targets readable at arm's length.

## 14. Telemetry and diagnostics

- **Session log** (`SessionLog.swift`) - every 60 s while listening, one JSON line appended on a utility queue to
  `Documents/logs/YYYY-MM-DD.jsonl` with exactly the contract 5.2 fields: `t, route, io_rate, io_frames, slack16, mode,
  n_instances, compute_us_p50/p99/max, underruns, nan_resets, howl, thermal, battery, low_power, out_rms_db,
  limiter_gr_db, p_target_mean[], p_self_mean, latency_measured_ms, latency_reported_ms, app_version, model_hash`.
  Counters are per-minute deltas; percentiles come from that minute's `drainTimings()`. Logs older than 30 days are
  deleted. No audio, no names. `make logs` copies them to `ios/build/logs`.
- **Diagnostics screen** - route and port UIDs, rates, requested / granted / callback frames, reported latencies, slack16,
  workgroup yes / no, compute last / max and session p50 / p99 / p99.9, underruns, NaN resets, probe history,
  `es_selftest` max error, WeSpeaker self-test cosine and compute units, `model_hash`, thresholds version, motion rate
  and newest-sample age, stem press counters, thermal, battery, profile expiry, input meter (peak and RMS dBFS), Phase 0
  runner.

## 15. Tests and on-device checks

### 15.1 XCTest (`make test`, simulator; pure Swift logic, no audio hardware)

| File | Proves |
|---|---|
| `SelectorTests` | `wrap`; nearest within 35 degrees; hysteresis (current 20 off, candidate 12 off cycles; candidate 9 off switches); Table / Everyone have no current; cycle order; stale re-anchor shifts every direction by one offset; learning step 0.1 with the 1 s and 20 deg/s gates; jump detector 45 degrees / 50 ms |
| `ThresholdsTests` | decodes the contract `thresholds.json` example; every boundary (speech 5.0 pass, 4.0 warn, 3.99 fail; own_cos 0.50 warn, 0.60 fail; sbr 0 warn, -0.01 fail; clip 0.1 warn, 1.0 fail; dup 0.75 asks) |
| `ClipToolsTests` | frame span and midpoint mapping, 10 s best window, 0.5 s rounding, window list, inside-window rule, sbr and clip_pct; every value of WP-ML's `quality_vectors.json` (copied into the test bundle by `make models`) |
| `PersonStoreTests` | person.json round trip with exactly the contract key set; WAV round trip within 1/32768; expiry refresh and purge; consent required for "saved"; backup exclusion flag set |
| `LatencyDisplayTests` | E 10.95 + loop 48.2 gives "Delay 59 ms (measured 3 min ago) - Earshot 11 + route 48"; estimate string; Bluetooth loop under 20 ms rejected; spread over 10 ms (Bluetooth) or 2 ms (wired) rejected; last valid result kept with its age |
| `RouteRulesTests` | allowed and forbidden port types; slack16 43 / 22; signature changes |
| `CoreSelfTestTests` | with `Resources/Models` present, `EarshotModel.selfTest` passes (meaningful only with `ENGINE=real`); `ContainerFile` rejects a corrupted hash |
| `CopyLintTests` | banned words and em dashes in `Copy.swift` (12.1) |
| `LaunchTests` | every failure branch of 9.4 with the stub |
| `InputTapTests` | with the stub, a tap made after 10 s of input returns only audio written after it; a reader that falls behind reports `droppedSamples` |
| `PersonStoreTests` (addition) | `mask` round trip; launch rebuild uses the stored mask and makes no WeSpeaker call |

### 15.2 On-device acceptance (contract 9 phone rows, iPhone 16 Pro, AirPods Pro 2 unless stated)

- Compute - 60 min screen locked, 1 instance, worker in the workgroup - p99 <= 1.0 ms, p99.9 <= 2.0 ms; 3 instances
  p99.9 <= 2.0 ms (from `drainTimings`, all targets).
- Underruns - 0 in 60 min locked with 1 instance, <= 1 with 3.
- Robustness - 10 route changes (AirPods in and out of the case, out of range, wired plug), 2 phone calls, 1 Siri,
  1 Clock alarm, AirPods reconnect while locked: no crash, recovery with at most one tap.
- Battery <= 5% per hour (1 instance, screen off, 2 h); thermal never above `.fair` after 60 min in a pocket.
- Warm instances - Focus with 3 warm instances, screen off, 60 min: battery <= 8% per hour, thermal never above
  `.fair`; if missed, Focus keeps only the selected and the last selected person warm (contract 9).
- Large buffer - 1,104-frame callback (Sound Recognition on), 3 instances, 30 min locked: p99.9 <= 2.0 ms, 0 underruns.
- Latency - measured loop plus Earshot share shown; Earshot share <= 11.5 ms at 48 kHz with a 64-frame grant; other
  grants and 44.1 kHz routes are shown and logged, not gated (contract 9).
- Safety - fade-in heard on every start; mute within 50 ms of a bad route. The app logs the time from the route-change
  (or interruption) notification to the `setMuted(true)` call (target under 10 ms, plus the 5 ms ramp); iOS's own
  delivery delay is invisible to the app, so the owner also checks by ear that no sound ever comes from the phone's
  speaker or receiver when AirPods go back in the case, run out of battery or move out of range.
- Daily use - 10 sessions of >= 20 min: zero own-voice-delay reports, look-select picks the intended person in >= 90% of
  double presses (the owner says "right" or "wrong" into a tally on the Listen screen), rating >= 4/5.

### 15.3 Phase 0 (I-1) - `docs/measurements/phase0-YYYY-MM-DD.json`

Run from Diagnostics, guided step by step, written to `Documents/phase0/` and copied with `make phase0`. Rows marked
(E-2) need the real engine; everything else runs with the stub.

```json
{"date": "2026-11-..", "device": "iPhone17,1", "ios": "26.1", "app_version": "0.1 (1)", "engine": "stub|real",
 "model_hash": "..",
 "buffers":   [{"route": "airpods_pro2", "mode": "measurement", "requested": 32, "granted": 64, "callback": 64,
                "rate": 48000, "in_lat_ms": 0, "out_lat_ms": 0, "underruns_60s": 0}],
 "tap_test":  [{"route": "airpods_pro2", "phone_tap_peak_db": 0, "airpod_tap_peak_db": 0, "data_source": "Bottom",
                "pass": true}],
 "latency":   [{"route": "airpods_pro1", "method": "in_ear|ear_detection_off", "minute": 0, "ambient_db": 45,
                "probe_dbfs": -30, "loop_ms": 0,
                "spread_ms": 0, "psr_db": 0, "ok_sweeps": 5, "reported_ms": 0}],
 "motion":    [{"route": "airpods_pro2", "screen": "on|locked", "minutes": 10, "rate_hz": 0, "gaps_over_200ms": 0,
                "max_gap_ms": 0}],
 "stem":      [{"route": "airpods_pro2", "conv_awareness": "on|off", "spatial": "off|on", "screen": "locked",
                "single": true, "double": true, "triple": true, "single_while_paused_rate0": true}],
 "compute":   [{"instances": 1, "minutes": 60, "screen": "locked", "p50_us": 0, "p99_us": 0, "p999_us": 0,
                "max_us": 0, "underruns": 0, "workgroup": true}],
 "power":     [{"minutes": 120, "instances": 1, "pct_per_hour": 0, "max_thermal": "nominal"}],
 "locked":    [{"route_changes": 10, "forced_swaps": 1, "resumed_without_tap": 0, "needed_tap": 0}],
 "mic_level": [{"distance_m": 1.0, "speech_rms_dbfs": 0, "speech_peak_dbfs": 0}],
 "wespeaker": [{"units": "cpuAndGPU|cpuOnly", "screen": "on|locked", "selftest_cos": 0, "ms_1p5s": 0, "ms_10s": 0,
                "first_call_ms": 0, "ok": true}],
 "alarm":     [{"screen": "locked", "seconds": 60, "windows": 0, "ok": true}],
 "notes": ""}
```

- **Buffers** - request 32, 48, 64, 96, 128, 192 frames (session off, set, on), 60 s each, for AirPods Pro 1, Pro 2 and
  no headphones (output muted); then 10 min at 64 frames screen locked.
- **Tap test** - AirPods in, the user taps the phone's bottom edge, then the right AirPod; pass when the phone tap peak is
  at least 20 dB above the AirPod tap peak.
- **Latency** - both methods, Pro 1 and Pro 2, at 0, 10 and 30 min after connect.
- **Motion** - 10 min screen on and 10 min locked inside the running session.
- **Stem** - single, double, triple with Conversation Awareness and Spatial Audio on and off, screen locked; also whether
  a single press arrives while paused with playback rate 0.
- **Latency (addition)** - methods A and B at about 45 dB and about 70 dB (played babble) ambient, Pro 1 and Pro 2,
  at -30, -24 and -18 dBFS; the lowest level with at least 80% success in both rooms becomes `ES_PROBE_DBFS` (a
  DECISIONS row; the louder levels are checked for comfort at half volume).
- **Locked** - with the screen locked, 10 route changes (AirPods out and in, case open and close) and 1 forced engine
  swap; record whether audio resumes without a tap (contract 11).
- **WeSpeaker and alarm, locked** - embed a 10 s clip with the screen locked (`.cpuOnly`) and run the alarm classifier
  for 60 s; record success, time and compute units.
- **Compute (E-2)** - random_v1 weights, voiceprints built from the test-vector clips `enr_t` / `enr_o`, 1 and 3
  instances, 60 min locked, plus 1,104 frames (Sound Recognition on) with 3 instances for 30 min. **Power** - 2 h
  screen off with 1 instance and 60 min with 3 warm instances (Focus), battery per minute from the session log.
  Debug builds also arm the lead's counting malloc hook for the worker and render threads (contract 11).
- **Mic level** - a person talking at about 1 m, VAD speech frames only.

## 16. Order of work and done criteria

| Step | Contents | Done when |
|---|---|---|
| I-0 | project, Makefile, session, `AudioHost.c` with the stub, routes, interruptions, Diagnostics | stub pass-through plays mic to AirPods with fade-in; route and call tests pass; `make test` green |
| I-1 | Phase 0 runner and file | stub rows recorded for Pro 1 and Pro 2; (E-2) rows added when the engine lands |
| I-2 | `SpeakerEncoder`, `EnrollmentService`, own voice, `PeopleStore`, quick add | self-test passes on the phone; all enrollment XCTests green; a real enrollment shows the 6 checks |
| I-3 | modes, table, Selector, stem, Now Playing, cues, latency flow and display, Settings | look-select and stem rows of Phase 0 work; latency strings tested |
| I-4 | safety polish, alarm passthrough, thermal downshift, telemetry, onboarding, Help | contract 9 phone rows run and logged |

Depends on - E-1 for real enrollment, E-2 for real audio, ML-0 for the encoder and weights (contract 11).


