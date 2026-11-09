# Earshot benchmarks on this Mac (measured 2026-11-06)

Every number below was measured on the training Mac (MacBook Pro M4, 4 performance + 6 efficiency cores,
10-core GPU, 24 GB, macOS 26.5.2, on AC power, no thermal warning recorded) unless it is marked
**analytic**. Raw results are the JSON lines in `bench/results/`. The scripts are in `bench/`
(see "Reproduce" at the end).

Caveat - this Mac was shared with other jobs all evening (speaker-model benchmarks, an iOS Simulator,
zstd, gitleaks; the load average peaked above 300). Every run first waited for 1-minute load < 4 and
GPU utilization < 15 % (`wait_quiet.sh`) and records the load it started at. Training runs all started
at load 1.6 to 3.7. The PyTorch streaming run on the performance cores waited 30 minutes, never got a
quiet machine and ran at load 5 to 9, so its tails (p99, max) are pessimistic.

## Bottom line

Training, batch 8 x 4 s at 16 kHz, MPS, torch 2.7.1
- **STFT + LSTM (stftlstm) is easily affordable** - 0.22 to 0.23 s/step, so 200k steps take about 13 h.
  That leaves 2 to 3x headroom for a bigger model, batch or longer segments inside 24 to 36 h.
- **Band-split RNN with LSTMs (bsrnnlstm) is affordable** - 0.40 s/step, 200k steps in about 22 h.
- **Conv-TasNet kernel 64 / stride 32 (tasnet64) is affordable for 100k to 150k steps** - 0.64 s/step with
  bf16 or fp16 autocast (0.66 fp32), 100k steps in 18 h; 200k needs 35 h, which leaves no slack in 36 h.
- **Conv-TasNet kernel 32 / stride 16 (tasnet32) is not** - 1.2 to 1.3 s/step, 100k steps need 33 to 36 h,
  200k need 66 to 72 h.
- **GRU models do not train well on MPS with torch 2.7.1** - 1.3 s/step (stftgru) and 2.2 s/step (bsrnn),
  and 2 to 6x worse while the CPU is busy, because there is no native MPS GRU kernel. The same GRU trains
  at 0.40 s/step on the 4 CPU performance cores, or at 0.54 s/step on MPS with torch 2.15, which
  coremltools does not support (it would need a second environment just for training).
- **Mixed precision is not worth much here** - 3 to 8 % faster for the TCNs, no gain for the RNNs.
- **Memory is not a constraint** - the worst case (tasnet32 fp32) uses 8.7 GB of the 19.1 GB working set MPS recommends.

Real time on the phone, from the CPU numbers
- **Compute is not the limit for any candidate.** Written with Accelerate in C (`chunk_skeleton.c`), one
  M4 performance core needs 27 to 180 us per 4 ms chunk (0.7 to 4.5 % of real time). Clamped to the
  efficiency cores it needs 0.3 to 0.9 ms (8 to 22 %). The heaviest model, tasnet32 at 2.46 GMAC per second
  of audio, uses 3.2 % of one performance core.
- **How the inference is written decides it.** PyTorch on one performance core needs 0.10 to 0.13 ms per
  4 ms chunk for the STFT models (2.4 to 3.3 %), 0.8 to 0.9 ms for the band-split models (20 to 23 %) and
  1.4 to 1.8 ms for the TCNs (34 to 46 %), because a TCN chunk is about 2,000 tiny ops at about 1 us of
  dispatch each. Written with `nn.Conv1d` the TCN takes 14.5 to 17 ms per 4 ms chunk (4x too slow) because
  PyTorch's CPU depthwise dilated conv loops over the 256 channels one at a time. On an efficiency core
  PyTorch dispatch costs 7 us per op and the TCN misses real time at 4 ms chunks (1.3x too slow).
- **Core ML keeps these streaming steps on the CPU.** All four converted models (tasnet64, tasnet32,
  stftlstm, stftgru) convert with coremltools 9.0, match PyTorch to 1.2e-6 or better over 300 chunks with
  state carried between calls, and take 0.29 to 0.81 ms per 4 ms chunk on the Mac. MLComputePlan puts
  100 % of their ops on the CPU, none on the Neural Engine, so battery cost is CPU time.
- **Conclusion for the phone** - every candidate fits one performance core with 20x or more headroom if the
  audio thread runs native code (C / Swift with Accelerate or BNNS, pre-transposed weights, ring buffers,
  no allocation). 4 ms chunks are affordable, so there is no need to pay 4 ms more latency for 8 ms chunks.
  The STFT-RNN needs a fifth to a half of the TCN's CPU time but its algorithmic latency is 8 ms (with a
  low-latency asymmetric window) against 5 to 6 ms for the TCNs. These are Mac proxies; the A18 Pro has
  the same core generation but its clocks, thermal limits and audio-thread scheduling must be checked on
  the phone.

Data pipeline
- **Decoding is cheap** - random 4 s crops decode at 7,000x real time per core from FLAC and 780x from Ogg
  Opus 32 kbit/s. The fastest training run needs about 560 to 830 s of decoded audio per second (4 to 6
  sources per mixture), so FLAC costs about 0.1 core and Opus 32k about 1 core.
- **Opus saves disk** - 14 MB per hour of audio (Opus 32k) against 67 MB (LibriSpeech FLAC) and 115 MB
  (WAV). 10 GB holds about 700 h of Opus 32k or 150 h of FLAC. Opus is lossy, which may matter for a
  separation model's targets.
- **Simulate rooms offline** - one pyroomacoustics RIR costs 67 ms at RT60 0.6 s and 210 ms at 0.9 s, too
  slow to make per example. Convolving a 4 s source with a 1 s RIR costs 0.4 ms, so a precomputed RIR bank
  plus on-the-fly convolution is cheap.

## 1. Environment and package versions

| item | version |
|---|---|
| Python | 3.12.15 (uv-managed, `uv python install 3.12`) at `.venv` |
| torch / torchaudio | 2.7.1 / 2.7.1 |
| coremltools | 9.0 (newest stable, released 2025-11-10) |
| numpy / scipy | 2.5.4 / 1.18.1 |
| soundfile / libsndfile | 0.14.0 / 1.2.2 (reads and writes Ogg Opus) |
| pyroomacoustics | 0.10.1 |
| protobuf (coremltools dependency) | 7.36.2 |
| ffmpeg / libopus | 8.1.2 / 1.6.1 (Homebrew) |
| uv / clang | 0.12.23 / Apple clang 17.0.0 |

Why torch 2.7.1 and not the newest torch (2.15.0, released 2026-10-28)
- coremltools 9.0 release notes - "Support for PyTorch 2.7"; its source sets `_TORCH_MAX_VERSION = "2.7.0"`.
- coremltools 9.1.dev1 (a dev pre-release, 2026-08-03) is tested up to torch 2.8.0. No coremltools
  release supports torch 2.9 to 2.15.
- 2.7.1 is the 2.7 bug-fix release. It fixes an MPS correctness bug ("binary kernels produce incorrect
  results when one of the tensor arguments is from a wrapped scalar on MPS devices"), which matters for
  MPS training. coremltools prints a harmless "Torch 2.7.1 has not been tested" warning because its check
  compares against 2.7.0 exactly. Converted models matched PyTorch to 1.2e-6 or better (section 7).
- No package failed to install.

torch 2.15.0 was installed in a temporary venv only to compare MPS training speed (section 3b), then
deleted with its uv cache entry to save disk. Recreate with
`uv venv --python 3.12 .venv-torch215 && uv pip install --python .venv-torch215/bin/python torch==2.15.0 numpy`.

Disk used by this task - `.venv` 0.57 GB, uv-managed Python 72 MB, `data/LibriSpeech/dev-clean`
348 MB, `data/bench_formats` 62 MB (WAV and Opus test copies of 200 files, safe to delete), plus the
wheels uv keeps in `~/.cache/uv`. About 1.4 GB in total.

## 2. Candidates (all causal, FiLM on a 192-d speaker embedding)

| name | what | params | MACs per second of audio (**analytic**) | frame hop | algorithmic latency (**analytic**) with 4 ms / 8 ms chunks |
|---|---|---|---|---|---|
| tasnet32 | Conv-TasNet TCN, encoder kernel 32 stride 16, N=256 B=128 H=256 P=3 X=8 R=3, cumulative LN, FiLM at the start of each repeat, skip channels = B | 2.65 M | 2.46 G | 1 ms | 5 ms / 9 ms |
| tasnet64 | same, kernel 64 stride 32 | 2.67 M | 1.24 G | 2 ms | 6 ms / 10 ms |
| stftgru | STFT 512 window / 64 hop, 257 bins, power-law (0.3) compressed real+imag input, Linear 514-384 + LayerNorm + PReLU, FiLM, 2-layer GRU 384, Linear 384-384-514, complex ratio mask | 2.47 M | 0.58 G (+ FFTs, about 0.006 G) | 4 ms | 8 ms / 12 ms with a low-latency asymmetric window pair; 32 ms / 36 ms with a plain 512 symmetric window |
| stftlstm | same with a 2-layer LSTM 384 | 3.06 M | 0.73 G | 4 ms | same as stftgru |
| bsrnn | band-split dual-path RNN, 16 bands (8x8, 4x16, 3x32, 1x33 bins), D=64, 3 layers of causal GRU over time + bidirectional GRU over bands, per-band mask MLP, complex ratio mask | 0.52 M | 0.67 G | 4 ms | same as stftgru |
| bsrnnlstm | same with LSTMs | 0.56 M | 0.84 G | 4 ms | same as stftgru |

MAC counts come from the layer shapes (`macs_per_frame()` in `models.py`). For the TCNs, a forward-hook
counter over every Conv1d / ConvTranspose1d gives exactly the same numbers. Algorithmic latency = chunk +
model lookahead (L - stride for the TCN, one hop for an STFT model whose synthesis window is two hops).
It excludes audio I/O, resampling and Bluetooth. The benchmarks use the symmetric window; the asymmetric
pair changes only the window values, not the compute. The LSTM variants were added because PyTorch has a
native MPS kernel for LSTM but not for GRU.

## 3. Training step time (batch 8 x 4 s at 16 kHz)

One step = forward + negative SI-SDR loss + backward + grad-norm clip (5.0) + fused AdamW, random inputs,
synchronized every step, median of 20 timed steps after 5 warm-up steps (GRU models on MPS - 5 after 2).
"MPS pool" = `torch.mps.driver_allocated_memory()` (the allocator's pool, an upper bound of the peak).
"peak alloc" = largest `current_allocated_memory()` seen after forward, backward or the optimizer step.
CPU rows use torch's default 4 threads (the performance cores). Step time excludes data loading, which
runs in parallel worker processes.

### 3a. torch 2.7.1 (the shipping environment)

| model | device | precision | s/step | fwd s | bwd+opt s | peak alloc GB | MPS pool GB | steps in 24 h | steps in 36 h | h per 100k steps | load |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tasnet32 | mps | fp32 | 1.299 | 0.381 | 0.912 | 8.68 | 9.80 | 66k | 100k | 36.1 | 2.1 |
| tasnet32 | mps | fp16 autocast | 1.207 | 0.372 | 0.829 | 8.57 | 8.75 | 72k | 107k | 33.5 | 2.6 |
| tasnet32 | mps | bf16 autocast | 1.195 | 0.372 | 0.820 | 8.57 | 8.75 | 72k | 108k | 33.2 | 3.2 |
| tasnet32 | mps | fp32 + activation checkpointing | 1.516 | 0.359 | 1.155 | 1.21 | 2.29 | 57k | 85k | 42.1 | 1.9 |
| tasnet32 | mps | bf16 + activation checkpointing | 1.492 | 0.369 | 1.123 | 1.22 | 2.29 | 58k | 87k | 41.4 | 1.6 |
| tasnet64 | mps | fp32 | 0.664 | 0.201 | 0.462 | 4.36 | 5.49 | 130k | 195k | 18.4 | 2.3 |
| tasnet64 | mps | fp16 autocast | 0.642 | 0.205 | 0.435 | 4.32 | 4.44 | 135k | 202k | 17.8 | 2.1 |
| tasnet64 | mps | bf16 autocast | 0.637 | 0.208 | 0.427 | 4.32 | 4.44 | 136k | 203k | 17.7 | 1.6 |
| stftlstm | mps | fp32 | 0.231 | 0.109 | 0.122 | 0.49 | 1.36 | 374k | 561k | 6.4 | 2.2 |
| stftlstm | mps | fp16 autocast | 0.235 | 0.107 | 0.126 | 0.31 | 1.35 | 368k | 552k | 6.5 | 1.6 |
| stftlstm | mps | bf16 autocast | 0.223 | 0.103 | 0.119 | 0.41 | 1.36 | 387k | 581k | 6.2 | 1.9 |
| bsrnnlstm | mps | fp32 | 0.401 | 0.149 | 0.250 | 1.95 | 3.73 | 216k | 324k | 11.1 | 1.6 |
| bsrnnlstm | mps | fp16 autocast | 0.453 | 0.164 | 0.284 | 1.93 | 3.73 | 191k | 286k | 12.6 | 2.2 |
| bsrnnlstm | mps | bf16 autocast | 0.426 | 0.157 | 0.269 | 1.93 | 3.86 | 203k | 304k | 11.8 | 2.5 |
| stftgru | mps | fp32 | 1.321 | 0.461 | 0.858 | 0.53 | 1.37 | 65k | 98k | 36.7 | 1.8 |
| stftgru | mps | fp16 autocast | 1.542 | 0.510 | 1.026 | 0.41 | 1.35 | 56k | 84k | 42.8 | 1.6 |
| stftgru | mps | bf16 autocast | 1.552 | 0.508 | 1.047 | 0.41 | 1.35 | 56k | 83k | 43.1 | 2.6 |
| bsrnn | mps | fp32 | 2.151 | 0.771 | 1.366 | 2.72 | 3.69 | 40k | 60k | 59.8 | 2.7 |
| bsrnn | mps | fp16 autocast | 2.512 | 0.850 | 1.650 | 2.06 | 3.48 | 34k | 52k | 69.8 | 2.6 |
| bsrnn | mps | bf16 autocast | 2.539 | 0.862 | 1.660 | 2.06 | 3.47 | 34k | 51k | 70.5 | 2.3 |
| tasnet32 | cpu | fp32 | 6.614 | 2.720 | 3.894 | - | - | 13k | 20k | 183.7 | 2.9 |
| tasnet64 | cpu | fp32 | 2.539 | 0.958 | 1.588 | - | - | 34k | 51k | 70.5 | 2.9 |
| stftgru | cpu | fp32 | 0.400 | 0.176 | 0.224 | - | - | 216k | 324k | 11.1 | 1.9 |
| stftlstm | cpu | fp32 | 0.494 | 0.235 | 0.258 | - | - | 175k | 263k | 13.7 | 3.0 |
| bsrnn | cpu | fp32 | 0.964 | 0.485 | 0.475 | - | - | 90k | 134k | 26.8 | 3.0 |
| bsrnnlstm | cpu | fp32 | 1.124 | 0.574 | 0.556 | - | - | 77k | 115k | 31.2 | 3.0 |

CPU peak RSS - tasnet32 8.9 GB, tasnet64 4.7 GB, band-split 3.1 GB, STFT models 0.9 GB.
MPS recommended maximum working set - 19.07 GB. tasnet32 fp32 was measured twice, 1.299 s at load 2.1
(in the table) and 1.512 s at load 5.4.

What these numbers say
- fp16 on MPS needs a workaround in torch 2.7.1 - `torch.amp.GradScaler("mps")` crashes ("Cannot convert
  a MPS Tensor to float64") because `unscale_` calls `self._scale.double()`. The fp16 rows use a static
  loss scale of 1024 and skip the step on overflow (same cost). torch 2.15 removed the float64 cast and
  GradScaler works there.
- Mixed precision barely helps - 3 to 8 % for the TCNs, nothing or worse for the RNNs. `nn.LSTM` / `nn.GRU`
  keep fp32 weights under MPS autocast, so the recurrence stays fp32, and cumulative LN is computed in fp32
  on purpose (cumulative sums in fp16 are not safe). That is also why tasnet32 memory stays at 8.6 GB.
- GRU has no native MPS kernel in 2.7.1 - PyTorch runs it as a loop of thousands of tiny kernels, so it is
  bound by CPU dispatch. The first GRU measurements, taken while another job used 880 % CPU, were 8.19 s/step
  for stftgru and 5.06 s/step for bsrnn (6x and 2.4x slower than quiet). On a busy Mac the GRU models are
  unpredictable. LSTM uses the native kernel and is 5.7x (STFT) and 5.4x (band-split) faster than GRU.
- GRU trains faster on the CPU than on MPS in 2.7.1 - stftgru 0.40 s/step on the 4 performance cores,
  but then those cores are not free for data loading.
- `torch.compile` on MPS fails in 2.7.1 - "InductorError: SyntaxError: failed to compile" in a generated
  Metal kernel (PyTorch warns it is an early prototype).
- Activation checkpointing per TCN block cuts tasnet32 memory from 8.7 to 1.2 GB for 17 % more time.

### 3b. torch 2.15.0 on MPS, same code (temporary venv)

| model | precision | torch 2.15.0 s/step | torch 2.7.1 s/step | 2.15 vs 2.7.1 |
|---|---|---|---|---|
| tasnet32 | fp32 | 1.545 | 1.299 | 19 % slower |
| tasnet64 | fp32 | 0.807 (A/B repeats 0.799, 0.806) | 0.664 (A/B repeats 0.678, 0.680) | 18 to 21 % slower |
| tasnet64 | bf16 | 0.759 | 0.637 | 19 % slower |
| tasnet64 | fp16 (GradScaler on 2.15, static scale on 2.7.1) | 0.774 | 0.642 | 21 % slower |
| stftgru | fp32 | 0.542 | 1.321 | 2.4x faster |
| stftlstm | fp32 | 0.216 | 0.231 | 7 % faster |
| bsrnn | fp32 | 0.677 | 2.151 | 3.2x faster |
| bsrnnlstm | fp32 | 0.375 | 0.401 | 7 % faster |
| stftgru on CPU | fp32 | 0.400 | 0.400 | same |

The A/B test alternated the two versions twice on a quiet machine, so the TCN slowdown in 2.15 is real.
Weights (`state_dict`) move between versions, so training a GRU on 2.15 and exporting from 2.7.1 is possible.

### 3c. Sustained runs (thermal check)

| run | steps | median s/step | median of each tenth of the run, in order |
|---|---|---|---|
| stftlstm fp32 MPS | 500 (2 min) | 0.234 | 0.225 0.225 0.225 0.228 0.227 0.252 0.238 0.242 0.252 0.254 |
| tasnet64 bf16 MPS | 300 (3.5 min) | 0.666 | 0.650 0.648 0.656 0.649 0.691 0.669 0.665 0.728 0.675 0.681 |
| tasnet64 bf16 MPS | 900 (11 min) | 0.690 | 0.680 0.678 0.655 0.674 0.784 0.655 0.726 0.739 0.729 0.800 |
| stftgru fp32 CPU | 200 (1.4 min) | 0.400 | 0.414 0.410 0.400 0.400 0.398 0.398 0.398 0.397 0.399 0.396 |

`pmset -g therm` recorded no thermal or performance warning. The slower end of the 11-minute run
coincided with an iOS Simulator boot by another job (load average 329 when the run ended), so it is not
clean evidence of throttling. Plan with about 10 % margin over the short-run medians, plus validation and
checkpoint time, and confirm with the first hour of the real run.

## 4. Streaming inference in PyTorch on CPU, one thread (`torch.set_num_threads(1)`)

Each candidate runs chunk by chunk with explicit state (`streaming.py`), 10 s of audio per configuration
(2,500 chunks of 4 ms or 1,250 of 8 ms) after warm-up, timed per chunk. "eager" = plain PyTorch under
`torch.inference_mode()`; "jit" = the same step through `torch.jit.trace` + `torch.jit.freeze`.
"tm" = time-major TCN (frames x channels, `F.linear`, explicit 3-tap depthwise conv); "conv" = the same
math with `nn.Conv1d` calls. RTF = time per chunk / chunk duration (must stay well below 1).
Calibration - one tiny eager op (add on a 1x256x4 tensor) costs 1.16 us here.

Performance cores (default QoS), load 5 to 9 during the run

| model | impl | chunk | eager median us | eager p99 us | jit median us | jit p99 us | best median RTF | aten ops per chunk |
|---|---|---|---|---|---|---|---|---|
| tasnet32 | tm | 4 ms | 1835 | 5448 | 1406 | 2086 | 0.35 | 1979 |
| tasnet32 | tm | 8 ms | 1913 | 4626 | 1580 | 5870 | 0.20 | 1983 |
| tasnet32 | conv | 4 ms | 17263 | 36099 | 15742 | 27370 | 3.94 | 2008 |
| tasnet32 | conv | 8 ms | 16946 | 30637 | 15030 | 29616 | 1.88 | 2008 |
| tasnet64 | tm | 4 ms | 1836 | 3954 | 1353 | 2415 | 0.34 | 1977 |
| tasnet64 | tm | 8 ms | 1785 | 2212 | 1455 | 2583 | 0.18 | 1979 |
| tasnet64 | conv | 4 ms | 16501 | 20538 | 14513 | 16960 | 3.63 | 2008 |
| tasnet64 | conv | 8 ms | 16212 | 18153 | 14687 | 17958 | 1.84 | 2008 |
| stftgru | - | 4 ms | 114 | 131 | 96 | 108 | 0.024 | 51 |
| stftgru | - | 8 ms | 247 | 314 | 226 | 252 | 0.028 | 55 |
| stftlstm | - | 4 ms | 130 | 147 | 112 | 128 | 0.028 | 51 |
| stftlstm | - | 8 ms | 320 | 353 | 304 | 340 | 0.038 | 55 |
| bsrnn | - | 4 ms | 788 | 830 | 793 | 833 | 0.197 | 163 |
| bsrnn | - | 8 ms | 1103 | 1143 | 1233 | 1292 | 0.138 | 167 |
| bsrnnlstm | - | 4 ms | 929 | 987 | 923 | 998 | 0.231 | 166 |
| bsrnnlstm | - | 8 ms | 1162 | 2120 | 1291 | 1529 | 0.145 | 170 |

Efficiency cores only (`taskpolicy -c background`, eager, time-major TCN), load 3.1 to 3.4.
Calibration - one tiny eager op costs 7.05 us here.

| model | chunk | median us | p99 us | max us | median RTF |
|---|---|---|---|---|---|
| tasnet32 | 4 ms | 5505 | 13464 | 28314 | 1.38 |
| tasnet32 | 8 ms | 5448 | 12119 | 12976 | 0.68 |
| tasnet64 | 4 ms | 5196 | 12540 | 13919 | 1.30 |
| tasnet64 | 8 ms | 5415 | 6520 | 7335 | 0.68 |
| stftgru | 4 ms | 1144 | 1472 | 1665 | 0.29 |
| stftgru | 8 ms | 2126 | 3057 | 7474 | 0.27 |
| stftlstm | 4 ms | 1186 | 2232 | 7710 | 0.30 |
| stftlstm | 8 ms | 2370 | 3389 | 7513 | 0.30 |
| bsrnn | 4 ms | 2156 | 4166 | 25066 | 0.54 |
| bsrnn | 8 ms | 2678 | 3639 | 4886 | 0.33 |
| bsrnnlstm | 4 ms | 2434 | 5832 | 6518 | 0.61 |
| bsrnnlstm | 8 ms | 2812 | 3572 | 3890 | 0.35 |

What these numbers say
- The TCN's cost in PyTorch is dispatch, not math - about 2,000 aten ops per chunk whatever the chunk
  size, so 8 ms chunks cost the same as 4 ms chunks. The C version of the same math (section 6) is 10 to
  19x faster.
- `nn.Conv1d` depthwise on CPU is the trap - the profiler shows `aten::slow_conv_dilated2d` called once per
  channel (268,800 calls in 50 chunks). Rewriting the depthwise conv as three shifted multiply-adds made
  the TCN step 9 to 11x faster with identical output.
- The first streaming run (`results/stream_cpu_1thread_firstrun_conv_only_loaded.jsonl`, conv impl only,
  load 4 to 7) agrees - tasnet 15 to 18.5 ms, stftgru 121 to 139 us, stftlstm 163 to 178 us, bsrnn 811 to
  817 us per 4 ms chunk.

## 5. Accelerate sgemm throughput, one thread (`VECLIB_MAXIMUM_THREADS=1`)

`sgemm_bench.c`, built with `clang -O3 -DACCELERATE_NEW_LAPACK -framework Accelerate`. Y[M x N] =
X[M x K] times W. "W[K,N]" = weights stored pre-transposed (CblasNoTrans). "W^T" = PyTorch layout
W[N,K] (CblasTrans). Best of 7 repeats of a loop of about 50 ms, data hot in cache. The M4 reports
FEAT_SME and FEAT_SME2; Accelerate reaches 1.7 TFLOPS on one thread for large shapes.

Performance core

| what | shape (M x K) * (K x N) | W[K,N] us | GFLOPS | W^T us | GFLOPS | sgemv us | GFLOPS |
|---|---|---|---|---|---|---|---|
| task example | 8x128 * 128x256 | 0.74 | 711 | 2.17 | 242 | - | - |
| task example | 1x256 * 256x768 | 1.01 | 391 | 1.83 | 215 | 1.83 | 215 |
| tasnet32 4 ms - B to H 1x1 conv | 4x128 * 128x256 | 0.72 | 365 | 2.07 | 127 | - | - |
| tasnet32 4 ms - H to 2B res+skip | 4x256 * 256x256 | 1.34 | 391 | 4.10 | 128 | - | - |
| tasnet32 4 ms - encoder | 4x32 * 32x256 | 0.29 | 224 | 0.75 | 87 | - | - |
| tasnet32 4 ms - decoder | 4x256 * 256x32 | 0.55 | 120 | 1.18 | 56 | - | - |
| tasnet32 8 ms - B to H | 8x128 * 128x256 | 0.71 | 739 | 2.09 | 251 | - | - |
| tasnet32 8 ms - H to 2B | 8x256 * 256x256 | 1.35 | 776 | 4.09 | 257 | - | - |
| tasnet64 4 ms - B to H | 2x128 * 128x256 | 0.68 | 192 | 2.05 | 64 | - | - |
| tasnet64 4 ms - H to 2B | 2x256 * 256x256 | 1.33 | 197 | 4.24 | 62 | - | - |
| STFT RNN - input layer | 1x514 * 514x384 | 1.14 | 345 | 2.71 | 146 | 2.76 | 143 |
| STFT RNN - GRU gates | 1x384 * 384x1152 | 2.25 | 394 | 3.19 | 277 | 3.19 | 277 |
| STFT RNN - LSTM gates | 1x384 * 384x1536 | 2.83 | 417 | 4.35 | 271 | 4.34 | 272 |
| STFT RNN - output layer | 1x384 * 384x514 | 1.37 | 289 | 1.45 | 271 | 1.46 | 271 |
| STFT RNN 8 ms - 2-frame input proj (GRU) | 2x384 * 384x1152 | 7.62 | 232 | 25.86 | 68 | - | - |
| STFT RNN 8 ms - 2-frame input proj (LSTM) | 2x384 * 384x1536 | 11.55 | 204 | 34.38 | 69 | - | - |
| band-split - time GRU gates, 16 bands | 16x64 * 64x192 | 0.35 | 1123 | 0.91 | 434 | - | - |
| band-split - time LSTM gates, 16 bands | 16x64 * 64x256 | 0.48 | 1101 | 1.97 | 266 | - | - |
| band-split - band GRU step | 1x32 * 32x96 | 0.09 | 67 | 0.11 | 55 | 0.11 | 57 |
| band-split - fc, 16 bands | 16x64 * 64x64 | 0.28 | 473 | 0.47 | 278 | - | - |
| probe | 3x384 * 384x1152 | 7.78 | 341 | 26.27 | 101 | - | - |
| probe | 4x384 * 384x1152 | 8.72 | 406 | 26.96 | 131 | - | - |
| probe | 8x384 * 384x1152 | 9.46 | 748 | 30.49 | 232 | - | - |
| probe | 16x384 * 384x1152 | 9.28 | 1525 | 30.90 | 458 | - | - |
| probe | 2x384 * 384x512 | 4.72 | 167 | 11.89 | 66 | - | - |
| probe | 2x384 * 384x768 | 5.39 | 219 | 17.47 | 68 | - | - |
| probe | 2x384 * 384x1024 | 8.82 | 178 | 23.14 | 68 | - | - |
| probe | 2x514 * 514x384 | 3.92 | 202 | 14.77 | 53 | - | - |
| probe | 2x384 * 384x514 | 4.14 | 191 | 12.58 | 63 | - | - |
| reference | 64x256 * 256x256 | 4.85 | 1729 | 8.59 | 977 | - | - |
| reference | 256x256 * 256x256 | 18.88 | 1778 | 25.50 | 1316 | - | - |
| reference | 512x512 * 512x512 | 151.67 | 1770 | 170.07 | 1578 | - | - |
| reference | 1024x1024 * 1024x1024 | 1293.34 | 1660 | 1402.91 | 1531 | - | - |

Efficiency core (`taskpolicy -c background`), selected shapes

| shape (M x K) * (K x N) | W[K,N] us | GFLOPS | W^T us | GFLOPS |
|---|---|---|---|---|
| 8x128 * 128x256 | 6.11 | 86 | 9.46 | 55 |
| 1x256 * 256x768 | 4.59 | 86 | 8.75 | 45 |
| 4x256 * 256x256 | 8.36 | 63 | 18.78 | 28 |
| 8x256 * 256x256 | 7.35 | 143 | 13.74 | 76 |
| 1x384 * 384x1152 | 7.81 | 113 | 13.60 | 65 |
| 1x384 * 384x1536 | 11.13 | 106 | 17.51 | 67 |
| 2x384 * 384x1152 | 56.27 | 31 | 121.35 | 15 |
| 16x64 * 64x192 | 1.86 | 211 | 2.99 | 131 |
| 512x512 * 512x512 | 962.71 | 279 | 1282.56 | 209 |

What these numbers say
- Store weights pre-transposed - the PyTorch layout (W^T) is 1.1 to 4x slower at streaming shapes.
- At M >= 2 the time hardly grows with M - 4x256 and 8x256 rows cost the same 1.34 us, and 2 to 16 rows
  against a 384x1152 matrix all cost 7.6 to 9.5 us. Accelerate appears to repack the weight matrix on every
  call, so the cost tracks weight size, not rows. Batching frames is nearly free; for M = 2 or 3 with a
  large matrix, one M = 1 call per row is faster (2 x 2.25 us against 7.62 us).
- Streaming shapes reach 70 to 1100 GFLOPS on a performance core and 15 to 210 on an efficiency core.

## 6. Per-chunk compute written with Accelerate (`chunk_skeleton.c`, one thread)

A C program that runs each candidate's per-chunk work with random weights but the real shapes and memory.
It does every matmul (cblas_sgemm, pre-transposed weights, one call per row when that is faster), ring
buffers for the dilated depthwise convs, cumulative LN, PReLU, sigmoid / tanh (vForce), GRU / LSTM cells,
512-point real FFTs (vDSP), compression, complex mask and overlap-add. 20,000 chunks timed after 500
warm-up chunks. It is not a working model and leaves out resampling and I/O; it gives the compute cost of
a straightforward native implementation. P = performance core (default QoS), E = efficiency cores
(`taskpolicy -c background`).

| model | chunk | weights MB (fp32) | MMAC per chunk | P median us | P p99 us | P max us | P RTF | P GFLOPS | E median us | E p99 us | E RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tasnet32 | 4 ms | 10.01 | 9.84 | 129 | 188 | 1506 | 0.032 | 153 | 881 | 1189 | 0.220 |
| tasnet64 | 4 ms | 10.08 | 4.95 | 96 | 157 | 2374 | 0.024 | 103 | 788 | 1442 | 0.197 |
| stftgru | 4 ms | 9.27 | 2.31 | 27 | 33 | 280 | 0.007 | 170 | 344 | 703 | 0.086 |
| stftlstm | 4 ms | 11.64 | 2.90 | 48 | 62 | 441 | 0.012 | 120 | 413 | 707 | 0.103 |
| bsrnn | 4 ms | 1.56 | 2.69 | 75 | 82 | 180 | 0.019 | 72 | 315 | 355 | 0.079 |
| bsrnnlstm | 4 ms | 1.74 | 3.38 | 179 | 191 | 267 | 0.045 | 38 | 465 | 528 | 0.116 |
| tasnet32 | 8 ms | 10.01 | 19.68 | 159 | 173 | 446 | 0.020 | 248 | 1075 | 1483 | 0.134 |
| tasnet64 | 8 ms | 10.08 | 9.90 | 131 | 143 | 464 | 0.016 | 151 | 875 | 1291 | 0.109 |
| stftgru | 8 ms | 9.27 | 4.62 | 53 | 60 | 448 | 0.007 | 173 | 406 | 980 | 0.051 |
| stftlstm | 8 ms | 11.64 | 5.80 | 73 | 121 | 2099 | 0.009 | 159 | 502 | 1003 | 0.063 |
| bsrnn | 8 ms | 1.56 | 5.37 | 125 | 151 | 287 | 0.016 | 86 | 631 | 730 | 0.079 |
| bsrnnlstm | 8 ms | 1.74 | 6.75 | 212 | 220 | 395 | 0.026 | 64 | 931 | 1111 | 0.116 |

A second performance-core run at load 3.3 (`results/skeleton_1thread_run2.jsonl`) reproduced the medians
within 4 % except stftlstm (36 / 67 us instead of 48 / 73 us) and bsrnn at 8 ms (154 instead of 125 us).

What these numbers say
- The math is small - even tasnet32 (9.8 MMAC per 4 ms chunk) takes 129 us on one performance core.
- Tail latency comes from the OS - the max values (0.2 to 2.4 ms on P, up to 72 ms on E under background
  QoS) are preemptions on a busy desktop. On the phone the audio thread runs at real-time priority, so tails
  must be measured there.
- The band-split LSTM is slow here because its 96 small cells per frame call vForce on 32-element vectors
  (about 10 calls per cell); a fused gate loop would bring it near the GRU version.
- Weight memory is 10 to 12 MB fp32 for the TCN and STFT-RNN models (fits the 16 MB performance-cluster
  L2 of the M4), 1.6 to 1.7 MB for the band-split models.

## 7. Core ML export check (`check_coreml.py`, `coreml_plan.py`)

The streaming step (chunk + all state tensors in, output + new state out) was traced and converted with
coremltools 9.0 to an ML Program (iOS 18 target), then run for 300 chunks with the state fed back and
compared with PyTorch. predict() timings are medians on the Mac from Python; predict() on a trivial
one-add model costs 24 us, so almost all of the time is Core ML executing the model.

| model (4 ms chunk) | convert s | max abs error vs PyTorch (output size) | CPU fp32 us | CPU fp16 us | CPU + Neural Engine fp16 us | all units fp16 us | ops placed by MLComputePlan |
|---|---|---|---|---|---|---|---|
| tasnet64 | 4.5 | 7.5e-8 (0.15) | 700 | 523 | 537 | 525 | 1979 CPU, 0 Neural Engine |
| tasnet32 | 4.6 | 9.3e-8 (0.21) | 809 | 612 | 612 | 612 | 1981 CPU, 0 Neural Engine |
| stftlstm | 0.4 | 4.5e-7 (0.014) | 338 | 449 | 439 | 444 | 74 CPU, 0 Neural Engine |
| stftgru | 0.4 | 1.2e-6 (0.023) | 287 | 455 | 448 | 454 | 76 CPU, 0 Neural Engine |

What these numbers say
- The export path works with torch 2.7.1 + coremltools 9.0, including `torch.fft.rfft` / `irfft`,
  `nn.LSTM` / `nn.GRU` with state inputs and the complex mask.
- Two things in the streaming code failed to convert and were rewritten - `Tensor.unfold` (not implemented
  in coremltools 9.0) and shape arithmetic on traced tensors (the "int" op fails). The overlap-add was also
  written without in-place slice updates before conversion. `streaming.py` now follows those rules, and
  its output is still identical to the offline model.
- Core ML chose the CPU for every op of these small per-chunk graphs, for "all units" and for
  "CPU + Neural Engine". Do not plan on the Neural Engine for the streaming path without testing a different
  graph layout on the phone.
- Core ML on CPU is 5 to 17x slower than the C skeleton (for example 523 us against 96 us for tasnet64) but
  still uses only 7 to 20 % of a 4 ms chunk.

## 8. Audio decode throughput (`bench_decode.py`, LibriSpeech dev-clean subset)

200 random dev-clean utterances (0.392 h). WAV = PCM 16-bit written by soundfile. Opus = ffmpeg libopus,
VBR, `-application audio`, Ogg container. libsndfile decodes the Opus files at 16 kHz with the same length
as the FLAC originals. "full" = decode each file completely (best of 3 passes, files in page cache).
"crop" = open, seek to a random offset, read 4 s (what a training loader does), once per file. Multi-process
rows run N worker processes over the same files.

| format | kbit/s | MB per audio hour | full decode, x real time, 1 core | 4 s crop, ms | crops, x real time, 1 core | 4 processes | 8 processes | load |
|---|---|---|---|---|---|---|---|---|
| FLAC (LibriSpeech) | 149 | 67 | 8106 | 0.53 | 7035 | 22771 | 31296 | 3.6 |
| WAV PCM16 | 256 | 115 | 34384 | 0.12 | 30111 | 84230 | 114354 | 3.6 |
| Opus 32 kbit/s | 32 | 14 | 776 | 4.74 | 781 | 3111 | 4410 | 5.7 |
| Opus 64 kbit/s | 72 | 32 | 475 | 8.56 | 433 | 1631 | 2401 | 5.4 |

What these numbers say
- Seeking is cheap in every format - a 4 s crop costs about the same as decoding 4 s. Opus decoding itself
  is the slow part - 9x slower than FLAC at 32 kbit/s and 16x at 64 kbit/s.
- Loader budget - the fastest training run (0.23 s/step, 8 mixtures of 4 s) needs about 140 s of output
  audio per second; with 4 to 6 decoded sources per mixture that is 560 to 830 s of decoded audio per
  second. That is about 0.1 core for FLAC, 0.7 to 1.1 cores for Opus 32k and 1.3 to 1.9 cores for Opus 64k.
  All fit in the 6 efficiency cores.
- Disk - 10 GB holds about 150 h of FLAC, 310 h of Opus 64k or 700 h of Opus 32k.

## 9. Data augmentation cost (`bench_augment.py`, one core)

| RIR (pyroomacoustics ShoeBox, image source method, random 3-10 m rooms) | max order | ms per RIR | RIR length s |
|---|---|---|---|
| RT60 0.3 s, order from `inverse_sabine` | 35 | 6.7 | 0.92 |
| RT60 0.6 s, order from `inverse_sabine` | 77 | 67.2 | 1.78 |
| RT60 0.9 s, order from `inverse_sabine` | 113 | 210.5 | 2.47 |
| RT60 0.3 / 0.6 / 0.9 s, order capped at 12 | 12 | 0.7 / 0.8 / 0.8 | 0.26 / 0.29 / 0.28 (tail cut off) |

| convolve 4 s of audio with an RIR (scipy 1.18.1) | 0.25 s RIR | 0.5 s RIR | 1.0 s RIR |
|---|---|---|---|
| `scipy.signal.fftconvolve` ms | 0.36 | 0.38 | 0.42 |
| `scipy.signal.oaconvolve` ms | 0.87 | 0.37 | 0.42 |

What these numbers say
- Making a reverberant RIR per training example is too slow (67 to 210 ms each, and each source needs its
  own). Capping the image order is fast but truncates the reverb tail, so it is wrong for long RT60.
- A precomputed RIR bank is cheap to make offline (for example 20,000 RIRs at about 0.1 s each is about
  30 core-minutes) and on-the-fly convolution costs about 0.4 ms per source, about 2 ms per mixture.

## 10. Correctness checks

- MPS vs CPU, fp32 - for every model, output relative error 3.9e-7 to 5.4e-7, gradient relative error
  7e-7 (band-split) to 1.2e-3 (tasnet32, long cumulative sums added in a different order). fp16 / bf16
  autocast gradients are within 0.1 to 11 % of fp32 (`check_mps.py`).
- Streaming equals offline - every streaming implementation was run chunk by chunk (64 and 128 samples) on
  2 s of audio and compared with the offline model shifted by its lookahead. Max difference 1e-7 for outputs
  of size 0.2 (TCN) and 9e-9 for outputs of size 0.02 (STFT models) (`streaming.check_equivalence`). So the
  streaming code is causal and uses the same weights as training.
- Core ML equals PyTorch over 300 streamed chunks (section 7).
- LibriSpeech dev-clean download - md5 42e2234ba48799c1f50f24a7926300a1 matches openslr's md5sum.txt;
  2,703 FLAC files, 97 transcripts, 40 speakers, 348 MB after extraction, in `data/LibriSpeech/dev-clean`.
  The 337.9 MB archive downloaded in 27 s (12.5 MB/s) and was deleted after the extract. The top-level
  LibriSpeech text files in the archive were byte-identical to the ones already in `data/LibriSpeech`
  (from test-clean), so only `dev-clean/` was extracted.

## Reproduce

```
cd earshot   # the repo root
uv python install 3.12
uv venv --python 3.12 --managed-python .venv
uv pip install --python .venv/bin/python torch==2.7.1 torchaudio==2.7.1 numpy scipy soundfile pyroomacoustics coremltools==9.0
cd bench
../.venv/bin/python check_mps.py                                   # MPS vs CPU, autocast
../.venv/bin/python -c "import streaming as S; print(S.check_equivalence('tasnet64', 64))"
./run_train_sweep.sh                                               # section 3a  -> results/train.jsonl
./run_train_extra.sh                                               # 3a re-runs, 3b, 3c -> results/train_extra.jsonl (needs .venv-torch215 for 3b)
clang -O3 -DACCELERATE_NEW_LAPACK -framework Accelerate sgemm_bench.c -o sgemm_bench
clang -O3 -DACCELERATE_NEW_LAPACK -framework Accelerate chunk_skeleton.c -o chunk_skeleton
VECLIB_MAXIMUM_THREADS=1 ./sgemm_bench > results/sgemm_1thread.jsonl
VECLIB_MAXIMUM_THREADS=1 taskpolicy -c background ./sgemm_bench > results/sgemm_1thread_ecore.jsonl
VECLIB_MAXIMUM_THREADS=1 ./chunk_skeleton > results/skeleton_1thread.jsonl
VECLIB_MAXIMUM_THREADS=1 taskpolicy -c background ./chunk_skeleton > results/skeleton_1thread_ecore.jsonl
./run_cpu_benches.sh                                               # sections 4, 6 (run 2), 7, 8, 9
../.venv/bin/python coreml_plan.py --models tasnet64 tasnet32 stftlstm stftgru > results/coreml_plan.jsonl
../.venv/bin/python -I summarize.py                                # markdown tables from results/
```

Files - `models.py` (candidates, MAC formulas), `streaming.py` (stateful streaming steps + equivalence
check), `bench_train.py`, `bench_stream.py`, `sgemm_bench.c`, `chunk_skeleton.c`, `bench_decode.py`,
`bench_augment.py`, `check_mps.py`, `check_coreml.py`, `coreml_plan.py`, `summarize.py`,
`wait_quiet.sh`, `run_train_sweep.sh`, `run_train_extra.sh`, `run_cpu_benches.sh`, `results/`.
