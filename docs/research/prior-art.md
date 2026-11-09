# Earshot - prior art and the model approach for real-time target speaker extraction

Research date 2026-11-06. Scope - the five questions in the research brief (closest systems, best architecture family under the Earshot budget, speaker conditioning, failure modes and fixes, one network for "only this person" plus "all speech"). Constraints come from `docs/BRIEF.md` - single phone microphone, 16 kHz, algorithmic latency of at most about 8-10 ms, roughly 0.5-3 GMAC per second of audio on one iPhone 16 Pro CPU core, under 20 ms mouth-to-ear on the wired path.

How to read confidence (used in every table and in the findings list)

- **H (high)** - I read the number or statement in the primary source itself (paper text, official Apple/FDA page, license file).
- **M (medium)** - taken from a primary abstract or a reputable summary, or it is my arithmetic on primary numbers.
- **L (low)** - manufacturer marketing, third-party summaries, or my own estimate. Anything called "estimate" below is not a measurement.

Citations are numbered like [12] and resolve in the reference list at the end.

---

## 0. Decision summary

- **What to build** - a causal STFT-domain **band-split dual-path GRU** network ("Earshot-BSR"). It is conditioned on the enrolled voice through **FiLM plus a small cross-attention over enrollment tokens**, takes a second (negative) voice print of the **user's own voice** so it never plays the user back to themselves, and has a **learned null embedding** so the same network runs an "all speech, no noise" mode.
- **Framing and latency** - 16 kHz; **asymmetric 32 ms analysis window** (512-point FFT, 257 bins), **8 ms synthesis window**, **4 ms hop**. Algorithmic latency **8 ms**, no lookahead. The long analysis window keeps the 31 Hz frequency resolution needed to tell voices apart by pitch harmonics; the short synthesis window keeps the latency.
- **Default size (M1)** - 32 bands, width 64, 4 blocks (BiGRU with 48 units per direction across bands, GRU with 80 units across time). **About 10.7 M MACs per 4 ms hop, about 2.7 GMAC/s, about 2.1 M parameters** (half of the parameters sit in a FiLM generator that runs once per enrollment, not per hop). Lite **S** is about 1.2 GMAC/s, large **L** about 8.6 GMAC/s (my counts, script in Appendix A).
- **Why this family** - dual-path time-frequency models lead speech separation and extraction quality (TF-GridNet beat SepFormer, Waveformer and DCCRN backbones in [29][1]); at the same 8 ms latency an asymmetric 32/8 ms window pair beat a symmetric 8/8 ms window by 1.4-1.5 dB SDR in online separation [14]; band-splitting cuts the frequency sequence from 257 bins to 32 bands so the model fits the budget; STFT models beat time-domain Conv-TasNet at equal latency with about half the FLOPs [13]; in the closest published setup to Earshot (mono, causal, 8 ms window, 4 ms hop, noisy enrollment) cross-attention conditioning beat FiLM by 2.1 dB [31].
- **Expected quality (estimate, not measured)** - about 6-8.5 dB SI-SNRi on simulated mono two-talker plus noise mixtures with noisy enrollment, about 8-11 dB on clean two-talker mixtures, and clearly less in real restaurants. The nearest published reference is 8.85 dB for a heavier (my estimate about 6 GMAC/s) causal TF-GridNet in the same kind of test [31].
- **Biggest risks** - (1) target confusion, the network following the louder voice instead of the enrolled one [27][30]; (2) real iPhone CPU efficiency, prior iPhone deployments with generic runtimes were far below peak [2][8]; (3) GRU training speed on the M4 GPU (MPS); (4) the gap between simulated and real rooms, which CHiME-9 showed is large [44].
- **Licensing** - do not copy code from Look Once to Hear (UW non-commercial license) [1], BUT/NTT SpeakerBeam (evaluation-only license that reserves patent rights) [72], or USEF-TSE (CC BY-NC 4.0) [29], and do not train shipping models on WHAM! noise (CC BY-NC 4.0) [64]. Write the model from the papers and credit them. MIT/Apache sources are listed in section 7.

---

## 1. Systems closest to Earshot

### 1.1 What "latency" means in this report

For a frame-based model, **algorithmic latency** is the delay forced by the math alone - window length on the output side plus any lookahead - before compute time and audio hardware. Earshot's 20 ms mouth-to-ear budget must also cover compute time (up to one hop if each hop is processed after it arrives), the 48 kHz to 16 kHz resampling filters (my estimate about 1-1.5 ms for linear-phase polyphase filters, measure it), and the iPhone's input and output buffers and converters. UW's Look Once to Hear reports 18.24 ms as 12 ms algorithmic plus 6.24 ms compute, which leaves out the audio hardware [1] (H). This is why Earshot targets 8 ms algorithmic, not 12.

### 1.2 UW Gollakota lab (closest prior art; credit it)

| System | What it does | Input | Latency | Compute and hardware | Reported quality | Code, weights, license | Conf. |
|---|---|---|---|---|---|---|---|
| **Look Once to Hear / Target Speech Hearing**, CHI 2024 [1] | Wearer looks at a talker for a few seconds; a noisy binaural clip enrolls them; the headset then plays only that voice | Binaural headset mics | 8 ms chunks + 4 ms lookahead = 12 ms algorithmic; 18.24 ms with 6.24 ms compute (hardware I/O not included) | Streaming TF-GridNet (D=64, B=3, H=64), 2.04 M params, 4.63 GMAC (text says per second); 5.47 ms per 8 ms chunk on an Orange Pi 5B with ONNX Runtime | 7.40 dB SI-SNRi with clean enrollment, 7.01 dB with noisy enrollment (knowledge-distilled encoder), 4.53 dB with beamformer enrollment; Waveformer 4.94 dB, DCCRN 6.71 dB on the same test; in-the-wild overall MOS 2.09 raw to 3.40 (21 raters) | github.com/vb000/LookOnceToHear under a **UW non-commercial license**; no pretrained weights listed | H |
| **Semantic Hearing**, UIST 2023 [2] | Keep or block 20 sound classes, binaural output | Wired binaural mics into an iPhone | 10.1 ms algorithmic, about 20 ms end-to-end; a 1.4 ms variant still gave 6.59 dB | 0.52 M params; **6.56 ms per 10 ms chunk on iPhone 11** with ONNX Runtime for iOS | 7.17 dB SI-SNRi | MIT | H |
| **Waveformer**, ICASSP 2023 [3] | Real-time target sound extraction | Mono | about 10 ms chunks | Dilated causal conv encoder + one transformer decoder layer; 1.6 M params, 2.43 GMAC in the LOTH comparison [1] | 2.2-3.3 dB SI-SNRi over prior models | MIT | H/M |
| **Sound Bubbles**, Nature Electronics 2024 [4] | Hear only talkers within a 1-2 m bubble | Up to 6 mics on a headband | 8 ms chunks processed in 6.36 ms on an embedded CPU | not given in the abstract | Sounds outside the bubble reduced by 49 dB | Code said to be available; license not verified. Founders started Hearvana (USD 6 M pre-seed, 2025) to commercialize this and Target Speech Hearing [59] | M |
| **Target Conversation Extraction**, Interspeech 2024 [6] | Extract conversation partners from turn-taking | Mono/binaural sims | not a latency paper | - | 8.19 dB (2 talkers), 7.92 dB (2-4 talkers) | MIT | M |
| **Knowledge boosting**, Interspeech 2024 [7] | A tiny on-device model receives delayed hints from a large remote model | Binaural sims | 8 ms chunks (12 ms window), hints up to 48 ms late | Causal TF-GridNet; small TSE 23 K params and 3.94 M MACs per chunk, large 516 K params and 44.12 M MACs per chunk | TSE small 3.95 dB, small + 48 ms-late hints 7.53 dB, large 8.52 dB. **TSE needs far more capacity than denoising** (SE small 9.01 vs large 11.28 dB) | Project page | H |
| **Proactive hearing assistants**, EMNLP 2025 [5] | Finds conversation partners automatically, using the wearer's own voice as the anchor | Binaural | Fast model every 12.5 ms (2 ms look-back/lookahead) + slow model every 1 s | Fast TF-GridNet D=32, H=32, 6 blocks, 491 K params, 8.9 ms per 12.5 ms chunk on Orange Pi 5B; slow model 986 K params, 41.3 ms per 1 s on Apple M2; slow embedding multiplied into the fast model between blocks 1 and 2 | 7.22-11.95 dB SI-SDRi; real recordings 7.84 dB (2 talkers), 6.00 dB (3 talkers); **fast model alone 1.45 dB**; output muted during the wearer's own speech | not stated | H |
| **ClearBuds**, MobiSys 2022 [8] | The wearer's own voice for calls | Two synchronized custom earbuds into an iPhone | 109 ms end-to-end (telephony) | 131 MFLOPs per 22.4 ms packet, **21.4 ms per packet on iPhone 12 Pro** | 9.35-10.56 dB SI-SDRi on synthetic data | Designs open-sourced per the authors; license not checked | H |

### 1.3 Hearing-aid and very-low-latency enhancement research (not speaker selective)

| System | Idea | Latency | Compute | Quality | License | Conf. |
|---|---|---|---|---|---|---|
| **Schröter et al., deep filtering for hearing aids**, TASLP 2022 [9] and multi-frame follow-up, Interspeech 2023 [10] | A DNN predicts a complex FIR filter across frames per frequency, applied to the noisy spectrum; better than a complex ratio mask at hearing-aid frequency resolution | Hearing aids need **6-10 ms**; their 48-band 24 kHz hearing-aid filter bank is "roughly an STFT with 4 ms window and 1 ms hop"; models had **8 ms including 2 frames of lookahead** | MF-DF 0.51 M params, RTF 0.18 on a notebook Core i5 | 18.21 dB SI-SDR, PESQ 2.85 on VoiceBank-DEMAND | Code (minus the HA filter bank) in DeepFilterNet repo, MIT/Apache-2.0 | H |
| **DeepFilterNet2 / DeepFilterNet3** [11][12] | ERB-band gains + deep filter below about 5 kHz | 20 ms window, 10 ms hop, 2 frames lookahead = **40 ms** | DFN2 2.31 M params, **0.36 GMAC/s**, RTF 0.04 (i5-8250U), RTF 0.42 on a Raspberry Pi 4; DFN3 RTF 0.19 single thread | PESQ 3.08 (DFN2), 3.17 (DFN3) on VoiceBank-DEMAND | MIT/Apache-2.0, code and weights | H |
| **Wang, Wichern, Watanabe, Le Roux**, TASLP 2023 [13] | Dual-window STFT - 16 ms analysis window for resolution, 4 ms synthesis window for latency, 2 ms hop | **4 ms**, or 2 ms by predicting one frame ahead | LSTM-ResUNet 2.32 M params, 27.8 GFLOPs per 4 s of audio | Mono, very hard noisy-reverberant set (input -6.2 dB): SI-SDR 2.89 dB at 4 ms vs Conv-TasNet 2.42 dB with about 2x the FLOPs; 6-mic version 7.90 dB | No code noted | H |
| **Wang, Naithani, Politis, Virtanen**, 2021 [14] | Asymmetric analysis-synthesis window pair for low-latency **separation** | 8 ms (synthesis window) | LSTM separators | Online WSJ0-2mix SDR: deep clustering 5.7 to 7.1 dB, mask inference 7.3 to 8.8 dB (+1.4/+1.5 dB) with 32 ms analysis / 8 ms synthesis; 32/32 ceiling 10.2 dB | - | H |
| **Google "Deep FIR"**, 2024-2025 [15] | LSTM predicts a 128-tap time-domain FIR filter | 0.32-1.25 ms | 626 K params; real time at 376 MIPS on a 600 MHz HiFi4 DSP | CHiME-2 SI-SDR - Deep FIR 6.21 dB at 1.25 ms; dual-window LSTW baselines 11.10 dB (2 ms), 11.43 dB (4 ms); 16 ms overlap-add 12.98 dB. Notes **noise leakage during speech** at low latency | - | H |
| **Wu and Braun**, ICASSP 2025 [16] | Controlled study of low-latency tricks on CRUSE | 20 / 10 / 5 / 3 ms | Baseline 0.625 M params, 0.23 GMAC/s | Learnable windows best at 5-10 ms; asymmetric windows not significant for the strong model (helped a weaker U-Net); future-frame prediction no clear gain; Mamba fine at 20 ms but worse at low latency; **a 10.95 GMAC/s model at 3 ms matched the 20 ms baseline** while a 0.23 GMAC/s model at 3 ms lost about 1.5 dB SI-SDR | - | H |

### 1.4 Target speaker extraction and personalized enhancement

| System | Conditioning | Latency | Compute | Reported quality | License | Conf. |
|---|---|---|---|---|---|---|
| **TD-SpeakerBeam**, ICASSP 2020 [18] | Jointly trained enrollment network, multiplicative fusion, optional speaker-ID loss | Non-causal | Conv-TasNet-like (N=256, B=256, H=512) | CSJ-2mix SDR 17.24 dB, 17.81 dB with speaker-ID loss; more training speakers helped SpeakerBeam a lot but not TasNet | BUT/NTT **evaluation-only** license [72] | H |
| **SpEx+**, Interspeech 2020 [19] | Shared ("tied") encoder for mixture and enrollment + multi-scale | Non-causal | 11.1 M params | WSJ0-2mix SI-SDR 17.4 dB; tied encoder 18.20 vs untied 17.78 dB on WSJ0-2mix-extr; longer enrollment 7.5 s to 60 s: 17.2 to 17.6 dB SDRi | Code exists, license not checked | H |
| **VoiceFilter-Lite**, Google, Interspeech 2020 [20] | Frame-wise concatenation of a d-vector, LSTM stack, works on ASR filterbank features (no waveform output) | Streaming, frame by frame | 3 LSTM layers; 2.2 MB int8 TFLite version | Big WER gains under overlap; **asymmetric L2 loss** against over-suppression; **adaptive suppression strength** from an overlap detector; bypassed when no d-vector exists | No code | H |
| **pDCCRN / pDCATTUNET**, Microsoft, ICASSP 2022 [21] | d-vector concatenated before the recurrent layer | 32 ms window, 16 ms hop | - | Introduced the **TSOS** (target speaker over-suppression) metric. Personalized models over-suppress more than unconditional DCCRN; asymmetric loss cut TSOS from 14.51% to 6.80%, plus ASR multi-task to 3.13%; pDCATTUNET with both 1.20% | No code | H |
| **E3Net**, Microsoft, Interspeech 2022 [22] | Learnable encoder; embedding concatenated with encoder features | 20 ms window, 10 ms hop | 6.61 M params, RTF 0.065 single Xeon thread (3x faster than pDCCRN) | Lower TSOS than pDCCRN; distilled students 2-4x faster | No code | H |
| **Personalized PercepNet**, Amazon, Interspeech 2021 [23] | Embedding from its own feature space, appended to every frame; also predicts target voice activity | 10 ms frames + 30 ms lookahead (40 ms) | 8.5 M params, 4.7% of one i7-8565U core | PESQ 1.46 to 2.36 | No code | H |
| **pDeepFilterNet2**, 2024 [24] | Frozen ECAPA-TDNN (192-dim) concatenated in the encoder | 40 ms | 2.31 M params, 0.33 GMAC/s | PESQ 2.10 (DFN2) to 2.36 on target + interferer tests; DNS5 blind OVRL 3.04 vs TEA-PSE 3.0 3.65 (TEA-PSE 3.0 is 22.24 M params, 19.66 GMAC/s [25]) | No code noted | H |
| **PVQE with internal embedding**, Microsoft, Interspeech 2024 [26] | The enhancement model's own GRU state, averaged over enrollment frames, replaces a separate 15 M-param speaker model | 20 ms window, 10 ms hop | PVQE-S 1.07 M params, 0.135 ms per 10 ms frame on one i7 thread | Internal embedding as good as the separate model, better for small models, **2.7x fewer over-suppressed frames**; background-speech suppression 33-34 dB with an embedding vs 5.4 dB with a zero ("empty") embedding | No code | H |
| **AV-PVQE**, 2026 [27] | Adds the target's mouth video to PVQE | 20 ms (no lookahead) | 12.79 M params | **PVQE picked the target 98.6% of the time when it was louder, 61.1% at equal level, 6.3% when quieter**; with video SI-SNRi rose from -2.31 to 9.03 dB on LRS3 | not stated | H |
| **pBSRNN**, DNS 2023 [28] | Band-split RNN + "speaker attentive module" | 20 ms window, 10 ms hop | 5.97 M params, 5.54 GMAC | 4th/5th place in DNS 2023 personalized tracks | - | H |
| **USEF-TSE**, TASLP 2025 [29] | No speaker embedding; mixture frames query enrollment frames via cross multi-head attention | Non-causal | USEF-TFGridNet 15.2 M params, **125 GMAC/s** | SI-SDRi WSJ0-2mix 23.3, WHAM! 17.6, WHAMR! 16.1, Libri2Mix 16.8 dB; FiLM vs concatenation identical for TF-GridNet (23.3 / 23.3) | **CC BY-NC 4.0** | H |
| **WeSep**, 2026 [30] | Toolkit; causal BSRNN with different speaker cues | 32 ms | not stated | Clean Libri2Mix, **causal**: ECAPA embedding 6.87 dB (79.9% correct-speaker accuracy), USEF cues 13.05 dB (95.7%), USEF + context 14.15 dB; non-causal 13.17-16.56 dB | "will be public" | H |
| **Positive/negative noisy enrollment**, Oxford, NeurIPS 2025 [31] | Enrollment = a noisy clip where the target talks plus one where the target is silent (user taps to mark); Siamese TF-GridNet encoder; cross-attention fusion | **Causal extraction, 8 ms window, 4 ms hop** | Causal TF-GridNet, 3 blocks, 64 hidden units, 1.88 M params | Mono, 2 talkers + WHAM! noise, 2-talker enrollment: **8.85 dB SI-SNRi (cross-attention) vs 6.70 dB (FiLM)**; prior art TCE 6.67 dB; USEF-TFGridNet trained on clean enrollment **-0.03 dB** on noisy enrollment; DNSMOS only about 2.1 (artifacts) | GitHub, no license file (all rights reserved by default) | H |

### 1.5 Products, challenges and apps

| Item | What is known | Conf. |
|---|---|---|
| **Phonak Audéo Sphere Infinio** (Aug 2024) | Separate DEEPSONIC chip runs a denoising DNN "from any direction"; manufacturer says 4.5 M parameters, 7.7 billion operations per second, more than 22 M training samples, SNR improvement 9 dB (default) to 10.2 dB (max) together with a fixed directional mic; latency and power not disclosed [48] | L (manufacturer) |
| **Oticon Intent, Starkey Edge AI** | On-chip DNNs (Oticon "DNN 2.0", trained on 12 M sound scenes per Oticon; Starkey G2 Neuro processor) - denoising, not enrolled-speaker extraction [49] | L |
| **Apple Conversation Boost** | Computational audio and beamforming mics focus AirPods Pro on the voice in front of you [50]; an independent test reported about 5.5 dB SNR improvement [52] | H (feature) / L (5.5 dB) |
| **Apple Hearing Aid feature** | FDA De Novo authorization 12 Sept 2024, first OTC hearing aid software, 118-adult study [51]; supported on AirPods Pro 2 and AirPods Pro 3 [50] (so not on the user's AirPods Pro 1st gen) | H |
| **Apple Live Listen** | Streams the iPhone mic to AirPods / MFi hearing devices [50] - the same hardware path Earshot uses, without speaker selection | H |
| **Apple Voice Isolation** | Mic mode for calls that isolates the phone user's own voice [73] - opposite target from Earshot | M |
| **Google Sound Amplifier Conversation mode** | Pixel 3+: point the phone's camera at your partner to amplify their voice; recommends wired noise-canceling headphones [53]; mechanism and latency not disclosed | M |
| **Meta Conversation Focus** (Dec 2025) | Ray-Ban/Oakley Meta glasses boost the voice in front using directional mics [54] | M |
| **Chatable (2020), HeardThat, clir Hearing** | iPhone apps that separate speech from noise and ask you to point the phone at the talker; clir claims "8 ms latency wired, 98 ms wireless" [55][56][57]. None publishes an enrolled-voice method or measurements | L |
| **Audatic phone-streamed denoising**, Frontiers 2023 [58] | iPhone 7 (one core) runs a U-Net denoiser with a 25 ms budget, streams to Phonak hearing aids; total delay 65-75 ms; speech reception threshold improved about 1.6 dB with each user's best mix ratio; **own-voice ratings got worse, which the authors attribute to latency** | H |
| **CHiME-9 ECHI** (2026) [44] | Real four-party cafeteria conversations recorded on Aria glasses and hearing aids; rule of 20 ms theoretical latency; baseline causal TF-GridNet with **8 ms window, 4 ms hop, FiLM** conditioning on a rainbow-passage enrollment. Real-world gains were modest - Aria-track listener transcription correctness 55.3% for the baseline, 60.3% for the most intelligible ranked system, 86.9% for the close-talk oracle. Two teams added **own-voice suppression**; one also used non-target talkers' enrollments to disambiguate | H |
| **Clarity CEC2** rules [46] | Hearing-aid challenge allows at most 5 ms lookahead because converters add more delay; the organizers describe under about 10 ms total delay as the hearing-aid target | M |
| **SLT 2026 REAL-TSE** [45] | Online track allows 100 ms; many teams injected **target-absent** segments in training | H |

**What Earshot can and cannot borrow**

- **Interaction** - "point or look for a few seconds, then hear only that person" is UW's Look Once to Hear idea [1]. Credit it in the app and docs.
- **Physics gap** - LOTH, Sound Bubbles and the proactive system use binaural or multi-mic inputs, so direction helps both enrollment and extraction. A single phone mic gives identity cues only. Expect lower numbers than their binaural results.
- **No shipping equivalent found** - I found no consumer product that does enrollment-based single-person extraction on a phone (searched Apple, Google, Meta, Chatable, HeardThat, clir, hearing-aid makers). This is an absence of evidence, not proof.

---

## 2. Which architecture family wins under the Earshot budget

Budget recap - mono, 16 kHz, at most about 8-10 ms algorithmic latency, 0.5-3 GMAC/s on one A18 Pro core (more if the Neural Engine or a second core can be used).

### 2.1 Evidence by family

| Family | Best low-latency evidence | Compute | Strengths | Weaknesses | Conf. |
|---|---|---|---|---|---|
| **Time-domain TCN (Conv-TasNet style)** | Causal Conv-TasNet on WSJ0-2mix (8 kHz, 2 ms window): **10.6 dB SI-SNRi** vs 15.3 dB non-causal, 5.1 M params [43]. At 4 ms latency it lost to a dual-window STFT model using about half the FLOPs [13]. Waveformer (TCN + transformer) 4.94 dB vs streaming TF-GridNet 7.40 dB in LOTH's binaural TSE test [1]. SkiM (time-domain dual-path LSTM) reported 17.1 dB SDRi at under 1 ms latency on simulated meeting data, 75% cheaper than DPRNN [68] (abstract only) | Conv-TasNet 4/4/1 ms: 54.5 GFLOPs per 4 s [13]; Waveformer 2.43 GMAC [1] | Tiny latency (1-2 ms) comes for free; simple | Worse quality per FLOP at equal latency [13]; STFT reported more robust under reverberation [32]; collapsed to silent output when trained with plain SNR loss on silence-heavy data [2] | H |
| **STFT mask or complex mask with GRUs (CRN / CRUSE / DCCRN style)** | DCCRN 6.71 dB in LOTH test at 6.6 GMAC [1]; CRUSE study shows small windows cost about 1-1.5 dB unless the model is made much larger [16] | CRUSE 0.23 GMAC/s; DCCRN 6.6-14.4 GMAC/s [1][11] | Cheap, mature, easy streaming state | Full-band GRUs at low resolution separate voices poorly; frequency resolution of an 8 ms window is 125 Hz | H |
| **Dual-window (asymmetric) STFT** | 4 ms latency beat Conv-TasNet at half the FLOPs [13]; 32/8 ms beat 8/8 ms by 1.4-1.5 dB for online separation [14]; dual-window LSTW at 4 ms lost only 1.55 dB vs 16 ms overlap-add [15]. Counter-evidence - not significant for a strong denoiser [16] | Same as the core network; FFT size grows but the network cost does not have to | Keeps 31-62 Hz resolution at 4-8 ms latency; zero lookahead | Needs a careful window pair for perfect reconstruction [13][17] | H |
| **Deep filtering (complex multi-frame filter)** | Beat complex ratio masks at hearing-aid resolution [9]; 8 ms latency hearing-aid models at 0.5 M params [10] | DFN2 0.36 GMAC/s [11] | Recovers harmonics that a coarse filter bank smears; cheap output stage | DeepFilterNet's standard config is 40 ms; ERB-gain stage is coarse for voice-vs-voice; personalized DFN2 lags heavy PSE models (OVRL 3.04 vs 3.65) [24] | H |
| **Dual-path / band-split RNN (TF-GridNet, BSRNN)** | Streaming TF-GridNet best of three backbones in LOTH (7.40 dB) [1]; knowledge-boosting large TF-GridNet 8.52 dB [7]; CHiME-9 baseline [44]; Oxford mono causal TF-GridNet with 8 ms window 8.85 dB [31]; causal BSRNN in WeSep [30]; a multi-channel TF-GridNet system ran at 4 ms algorithmic latency in Clarity CEC2 (18.8 dB SI-SDRi, search snippet) [47] | LOTH 4.63 GMAC/s at 8 ms hop [1]; Oxford config about 6 GMAC/s (my estimate from its sizes); full TF-GridNet TSE 125 GMAC/s [29] | Best separation quality per parameter; natural streaming state; easy to scale | Cost scales with frequency rows x frames per second; per-bin TF-GridNet at 4 ms hop exceeds 3 GMAC/s | H (quality) / M (cost estimates) |
| **Mamba / state space** | Competitive at 20 ms, degrades faster at lower latency [16] | - | Parallel training | Weaker at low latency in the only controlled study | H |
| **Generative / LM-based TSE** | StarTSE (streaming, autoregressive LM) uses 560 ms chunks, RTF 0.248 on an RTX 4090 [78] | Large | Clean-sounding output offline | Not usable at ms latency | M |

Note on Libri2Mix / WHAM! / WHAMR! - I found **no published causal TSE result at 10 ms or less on WHAM! or WHAMR!**. The closest numbers are the causal Libri2Mix results at 32 ms [30], the mono LibriSpeech + WHAM!-noise results at 8 ms window [31], and LOTH's binaural LibriSpeech + WHAMR! noise results at 12 ms [1].

### 2.2 Verdict for Q2

- **Best family** - a dual-path time-frequency recurrent model (TF-GridNet / BSRNN lineage), made affordable by **band-splitting** and made low-latency by an **asymmetric analysis/synthesis window pair**, with a **complex mask plus short deep filter** output. Confidence M - no paper tests this exact combination under our budget, but each piece has direct evidence above.
- **Why not per-bin streaming TF-GridNet as in LOTH** - at a 4 ms hop it lands around 5-6 GMAC/s by my estimate (LOTH's 4.63 GMAC/s was at an 8 ms hop and 12 ms latency), about 2x the CPU budget. Keep it as the "L" option if the Neural Engine or a second core is usable.
- **Why not time-domain TCN** - simplest latency story, but worse quality per FLOP at the same latency [13] and less robust to reverberation [32].
- **Why not DeepFilterNet as the backbone** - great noise remover, weak at voice-vs-voice; keep its deep-filter output idea [9] and its attenuation-limit trick [70].
- **Capacity matters for TSE** - in the knowledge-boosting study the small TSE model (about 0.5 GMAC/s) got 3.95 dB and the large one (about 5.5 GMAC/s, my unit reading) 8.52 dB [7], and Wu and Braun show more compute buys back latency losses [16]. So spend the full budget, and benchmark whether a second P-core or the Neural Engine can host a slow helper model (section 6.9).

---

## 3. Conditioning - how to inject the enrolled voice

### 3.1 Fusion type

- **Large models** - choice barely matters. The TSE overview found multiplication and FiLM usually work well but the impact of the fusion choice "seems rather insignificant" [32]; USEF-TFGridNet scored 23.3 dB with FiLM and with concatenation [29].
- **Small causal models** - cross-attention to frame-level enrollment features wins clearly. Causal BSRNN on Libri2Mix: 6.87 dB with an ECAPA vector vs 13.05 dB with USEF frame-level cues [30]. Mono causal TF-GridNet with noisy enrollment: 8.85 dB cross-attention vs 6.70 dB FiLM, and cross-attention beat FiLM in every scenario the authors tested [31]. Frame-level attention also added 0.1-0.9 dB in-domain for E3Net and Conv-TasNet (largest with FBANK features), with mixed results cross-dataset [33].
- **FiLM parameter cost** - per-band FiLM generators are large (the Oxford FiLM variant had 3.94 M vs 1.88 M params) [31], but they run once per enrollment, not per hop.

### 3.2 Where to inject

- **Early, after one shallow block** - the overview reports best results with a shallow mixture encoder and the fusion layer low in the extractor [32]. LOTH and the proactive system multiply the speaker vector between blocks 1 and 2 [1][5]; CHiME-9's baseline repeats FiLM after GridNet blocks [44]; FiLM is also used to steer a 10 ms hearing-aid network at test time [69]. Recommendation - fuse after block 1; optionally repeat a light FiLM after block 3 (ablation).

### 3.3 Frozen pretrained encoder vs jointly trained

| Evidence | Finding | Conf. |
|---|---|---|
| Liu, Li, Serrà [33] | Speaker-ID embeddings can throw away TSE-relevant information; **simple FBANK mean/std (160-dim) was competitive**; d-vector beat ECAPA ("e-vector") for TSE (E3Net 11.5 vs 10.3 dB); fine-tuning helped in-domain but **not cross-dataset** (PASE+ 12.1 to 12.6 dB in-domain, 9.9 to 9.6 dB on VCTK) | H |
| Pärnamaa and Saabas [26] | None of the top DNS-challenge teams fine-tuned the speaker model; the PSE model's own internal embedding matched a separate 15 M-param model and was **better for small models**, with 2.7x fewer over-suppressed frames | H |
| TD-SpeakerBeam [18] | Jointly trained enrollment network far better than x-vector-based selection; speaker-ID auxiliary loss added up to about 1 dB; more training speakers mattered a lot | H |
| SpEx+ [19] | Sharing the encoder between mixture and enrollment improved SI-SDR (18.20 vs 17.78 dB) | H |
| LOTH [1] | Frozen Resemblyzer d-vector as target space; a separate encoder distilled to map **noisy** enrollment to the clean d-vector lost only 0.4 dB (7.01 vs 7.40) | H |
| Overview [32] | Pretrained = robust (big corpora); joint = optimized for TSE; combining them was untested as of 2022 | H |

**Recommendation** - use both, each with dropout during training. (a) A frozen, Apache-2.0 d-vector (Resemblyzer, as LOTH did [1][71]) as a robust global anchor, fed through a small noisy-enrollment student trained with LOTH's cosine distillation. (b) Jointly trained frame-level tokens from the extractor's own band encoder (shared weights, SpEx+/PVQE idea [19][26]) for cross-attention. Add a speaker-ID loss on the pooled tokens [18].

### 3.4 Noisy enrollment

- **Train on noisy enrollments** - models trained only on clean enrollment can collapse on noisy ones (USEF-TFGridNet -0.03 dB) [31]. PVQE trained with 10 s enrollments, 50% noisy at SNR 0-40 dB [26].
- **Positive/negative contrast** - a clip where the target talks plus a clip where they are silent lets the model subtract the interferers; it fails when an interferer talks as long as the target in the positive clip (about 3 s) and works below about 2.7 s; user labels about 0.3 s off barely hurt [31]. For Earshot the negative clip can come from the camera (lips still) or an AirPods stem press. Related - a per-frame "is the target talking" signal (oracle or from diarization) can replace enrollment speech as the clue [75], and denoising the mixture before it meets the enrollment helps in noise [76] (abstract only).
- **Directional capture** - iPhones support selectable polar patterns on front/back built-in mics [63], and stereo capture from built-in mics since iOS 14 [77]. Use the back mic's directional pattern during enrollment if `supportedPolarPatterns` offers it (check on device).
- **Length** - longer enrollment helps a little (SpEx+ 7.5 s to 60 s: 17.2 to 17.6 dB) [19]; LOTH users found 5 s acceptable (89%) [1].

### 3.5 Unseen speakers

- Use as many training speakers as possible (strong effect for SpeakerBeam [18]), test only on held-out speakers, and use a speaker-similarity curriculum (dissimilar pairs first, similar later; about 1 dB gain) [36].

---

## 4. Known failure modes and the fixes that worked

| Failure mode | Evidence | Fixes with evidence | How to measure | Conf. |
|---|---|---|---|---|
| **Target confusion** (follows the wrong talker, often the louder one) | PVQE picked the target only 61% at equal level and 6% when quieter [27]; causal ECAPA-conditioned BSRNN 79.9% accuracy [30]; LOTH quality falls almost linearly with voice similarity [1]; many confusions come from speaker similarity [34] | Train SIR down to negative values (target quieter); similarity curriculum, about +1 dB [36]; speaker-ID loss, up to +1 dB [18]; metric learning plus post-filter, >1 dB [34]; chunk-level confusion-aware loss, 14.8% relative fewer confusions [35]; frame-level cross-attention [30][31]; longer-context slow model [5][7]; a visual cue fixes most of it [27] | "Picked" rate per 2 s chunk (output closer to target than to any interferer) at SIR -10, -5, 0, +5 dB | H |
| **Leakage when the target is silent** | Most TSE systems are not trained to output silence [32]; standard SI-SDR undefined for silent references [37] | Train target-absent mixtures with a thresholded inactive loss 10 log10(‖ŝ‖² + τ‖x‖²), τ from a 30 dB cap [40][39]; Borsdorf's silence-evaluating SI-SDR [37]; note absent-target training slightly lowers present-target quality, a detection branch helps [38]; target-activity head and gating (Personalized PercepNet predicts target VAD [23]) | Output energy reduction (dB) on target-absent clips (PVQE reports 33-34 dB [26]) | H |
| **Over-suppression of the target (TSOS)** | Personalized models over-suppress more than plain denoisers [21]; worst when enrollment and live audio come from different devices or rooms [26] | Asymmetric loss (TSOS 14.5% to 6.8%) [21]; VoiceFilter-Lite asymmetric L2 + adaptive suppression strength [20]; ASR multi-task (to 3.1%) [21]; distillation [22]; internal embedding (2.7x fewer) [26]; attenuation floor by mixing back input (DeepFilterNet `atten_lim_db` [70], Audatic mix ratio [58], Phonak adjustable strength [48]) | TSOS frame rate (pDCCRN definition, γ = 0.1) [21] | H |
| **Artifacts and noise leakage at low latency / low SNR** | Low-latency denoisers leak noise during speech [15]; 3 ms windows cost about 1-1.5 dB at fixed compute [16]; Oxford mono model DNSMOS about 2.1 [31] | Larger model [16]; asymmetric or learnable windows [14][16]; deep filtering [9]; multi-resolution spectral loss [11][24]; attenuation limit [70] | DNSMOS P.835 (SIG/BAK/OVRL), PESQ, listening tests | H |
| **Reverberation** | Same TSE model: WSJ0-2mix 23.3 vs WHAMR! 16.1 dB [29]; x-vector selection hurt by reverberant enrollment [18] | Train with RIRs on every source; STFT front ends are more robust [32]; choose the reverberant target carefully - recent work prefers a time-shifted dry target [67], other work keeps early reflections for short decay [66]; ablate | WHAMR!-style reverberant test set + real rooms | M |
| **Silence collapse during training** | Causal Conv-TasNet converged to all-zero output on silence-heavy data with SNR loss [2] | Mix in SI-SNR (90% SNR + 10% SI-SNR in [2]) or thresholded SNR [40]; keep target-absent share moderate | Watch output energy during training | H |
| **Slow onset** | Proactive system - 4.77 dB in the first 2 s of a turn vs over 8 dB later [5] | Fast/slow two-rate design [5][7]; enrollment-based conditioning (not turn-taking) avoids most of this | SI-SDRi vs time since target onset | H |
| **Own voice played back late** | Phone-streamed denoising made own-voice ratings worse [58]; CHiME-9 teams added own-voice suppression [44]; proactive system mutes output while the wearer talks [5] | Enroll the user's voice once; condition on it as a negative; train with a loud near-field "user" talker | Own-voice energy in the output (dB) | H |
| **Enrollment / channel mismatch** | Over-suppression mostly when target speech differs from enrollment conditions [26]; speaker similarity metrics sensitive to device mismatch [45] | Enroll on the same phone mic used for listening (Earshot does); augment enrollment with EQ, reverb, noise | Cross-device tests | M |

---

## 5. One network for "only this person" and "all speech, no noise"

Precedents

- **USEF-PNet (2025)** [41] - mixes personalized samples and plain enhancement samples in each mini-batch; enhancement samples get a **zero-valued dummy enrollment**; one checkpoint replaces both models. Plain enhancement on VoiceBank-DEMAND improved from 19.21 to 19.69 dB SI-SDR; personalized Libri2Mix stayed equal (13.00 vs 13.00 dB). Not shown to be causal. (H)
- **MUSE (2023)** [42] - one model does target extraction when a cue is given and separation otherwise; denoising was unaffected (11.5 vs 11.6 dB SI-SNR). (H)
- **PVQE (2024)** [26] - zeros for the speaker embedding gives the "closest talker" behaviour; background-speech removal drops from 33-34 dB to 5.4 dB, which is exactly what an "all speech" mode wants. (H)
- **VoiceFilter-Lite (2020)** [20] - simply bypassed when no d-vector exists. (H)

**Recommendation** - a **learned null embedding** (a trainable vector, plus learned null tokens for the attention) rather than literal zeros, with mode-specific training targets.

- **Focus mode** - conditioning = target voice + user's own voice as negative; target = the enrolled talker only.
- **All-speech mode** - conditioning = null + user's own voice as negative; target = every other talker, noise removed.
- **Mix** - about 25-30% of training batches in all-speech mode (my choice; USEF-PNet does not state its ratio [41]).
- **Why learned null** - a zero vector passed through FiLM means "no modulation", which is a fine default, but a learned vector lets the network choose its own neutral point. Low risk either way; ablate.

---

## 6. Recommended architecture (decision-ready)

### 6.1 Signal path and latency budget (wired path)

1. Mic at 48 kHz, decimate to 16 kHz (low-delay polyphase filter).
2. Every 4 ms (64 samples) - take the newest 512 samples (32 ms), multiply by the asymmetric analysis window, 512-point FFT (257 bins).
3. Network outputs a complex filter per bin; apply it; inverse FFT; multiply by the 8 ms synthesis window; overlap-add with hop 64.
4. Upsample to 48 kHz, then safety chain (attenuation floor, gate, limiter, gain cap, fade-in).

Budget - 8 ms algorithmic + about 1-1.5 ms resampling (estimate) + compute slack (up to 4 ms if processed one hop behind) + iPhone I/O (measure). If the measured total exceeds 20 ms, switch to the 4 ms profile - 16 ms analysis, 4 ms synthesis, 2 ms hop as in Wang et al. [13] - with model S (about 2.3 GMAC/s at 2 ms hop by my count).

**Window pair** - analysis window of N = 512 samples with a slow rise over the first N - M = 384 samples and a fast fall over the last M = 128; synthesis window nonzero only on the last 128 samples; the pair must give perfect reconstruction at hop M/2 = 64. Use the exact construction from Wang et al. (synthesis window derived from the last output-window samples of the analysis window, their Eq. 16) [13] or Mauler-Martin [17]; the Tampere paper shows the same idea for separation [14]. Algorithmic latency = M = 8 ms.

### 6.2 Layer-by-layer specification (default M1)

| Stage | Spec | Notes |
|---|---|---|
| Input features | Power-law compressed complex spectrum (exponent 0.3) + log power; causal running normalization per band | Compression as in [13][26]; running norm in the DeepFilterNet style [11] |
| Band split | **K = 32 bands** over 257 bins - 16 bands x 4 bins (0-2 kHz, 125 Hz each), 8 x 8 bins (2-4 kHz), 8 x 16 bins (4-8 kHz, Nyquist bin added to the last) | Band widths follow the BSRNN idea of finer bands at low frequency [28] |
| Band encoder | Per band - LayerNorm, Linear(3w to D = 64), separate weights per band | 51 K params |
| Block 1 | Across bands - LayerNorm, **BiGRU 64 to 48 per direction** over the 32 bands, Linear(96 to 64), residual. Across time - LayerNorm, **GRU 64 to 80** per band (weights shared across bands, 32 states), Linear(80 to 64), residual | Dual-path pattern from TF-GridNet/BSRNN [1][28]; GRU instead of LSTM saves 25% |
| **Conditioning** (after block 1) | **FiLM** z = z(1 + γ) + β with γ, β of shape 32 x 64 from a small MLP (input = target vector 256 + user vector 256 + mode flag, hidden 256, output 2 x 32 x 64) - computed once per enrollment. **Cross-attention** - query = Linear(flattened frame 2048 to 64), 4 heads of 16, over at most 64 enrollment tokens (keys/values precomputed), output Linear(64 to 2048) added to z, **output layer zero-initialized** so training starts as pure FiLM | Placement as LOTH/proactive [1][5]; attention per [29][31]; zero-init idea from AV-PVQE [27]; negative (user) tokens carry a learned "negative" type embedding, like the Oxford segment embeddings [31] |
| Blocks 2-4 | Same as block 1 | 4 blocks total |
| Decoder | Per band - LayerNorm, Linear(64 to 128), Tanh, Linear(128 to 2·w·taps). **3-tap deep filter** (frames t, t-1, t-2) for bands below 4 kHz, 1-tap complex mask above; coefficients bounded with tanh | Deep filtering beat complex masks at hearing-aid resolution [9] |
| Target-activity head | Mean over bands of the last block, Linear(64 to 1), sigmoid | Drives gating and a "target speaking" indicator; idea from [23][38] |
| State per stream | 4 blocks x 32 x 80 GRU states, 512-sample input ring, 2 past spectra for the deep filter | About 0.1 MB; no allocation after setup |

### 6.3 Size and compute (my counts)

Counted with the script in Appendix A (GRU = 3H(I+H) MACs per step; FFTs, attention, FiLM, decoder and filter application included; layer norms and activations ignored). 250 hops per second.

| Variant | Width D | BiGRU (per dir) | Time GRU | Blocks | MACs per hop | GMAC/s | Params | Intended use |
|---|---|---|---|---|---|---|---|---|
| S | 48 | 32 | 64 | 3 | 4.6 M | 1.16 | 1.4 M | E-core or 2 ms-hop low-latency profile |
| **M1 (default)** | 64 | 48 | 80 | 4 | **10.7 M** | **2.68** | **2.1 M** | One P-core |
| L | 96 | 64 | 128 | 6 | 34.3 M | 8.58 | 3.9 M | Two P-cores or teacher for distillation |

About 0.8-1.2 M of each model's parameters are the FiLM generator, which runs only when a voice is enrolled (the script counts a single 256-to-FiLM layer; the 512-256-FiLM MLP in 6.2 adds about 0.13 M). Real iPhone speed must be measured - prior iPhone deployments with generic runtimes were slow (Semantic Hearing 6.56 ms per 10 ms chunk for a 0.52 M-param model on an iPhone 11 [2]; ClearBuds 21.4 ms per 22.4 ms packet for 131 MFLOPs on an iPhone 12 Pro [8]). Apple's BNNS Graph is built for real-time CPU inference without allocation and with forced single-threading, demonstrated inside an Audio Unit [61][62], and the A18 Pro CPU has ML accelerators Apple describes as handling "low-latency computations on the CPU" [60]. Batch the 32 bands into one matrix multiply per GRU step.

### 6.4 Enrollment encoder (runs once, no latency limit)

- **Capture** - 3-8 s from the back mic while the phone points at the person (directional pattern if available [63]); the camera's view of the target's mouth can label "talking" vs "silent" frames for the positive/negative scheme [31] (idea, not tested).
- **Path A, global anchor** - frozen Resemblyzer d-vector (Apache-2.0) [71] plus a small student trained to map noisy enrollments to the clean-enrollment d-vector with a cosine loss, as LOTH did [1].
- **Path B, tokens** - the extractor's band encoder (shared weights) + 2 non-causal dual-path blocks over the enrollment, pooled every 40 ms to at most 64 tokens (Oxford pooled 40 frames [31]); speaker-ID auxiliary loss on their mean [18].
- **User's own voice** - enrolled once at setup through the same encoder; passed as negative FiLM input and negative tokens.
- **Dropout** - drop path A or path B with probability about 0.15 each during training so the network uses both (my choice).

### 6.5 Losses

- **Signal** - thresholded SNR on the waveform, 10 log10(‖s - ŝ‖² + τ‖s‖²) - 10 log10‖s‖² with τ = 10^-3 (30 dB cap) [39][40], plus a multi-resolution STFT magnitude loss [11][24], plus a small SI-SNR term to avoid gain drift and silence collapse [2].
- **Over-suppression** - mean of ReLU(|S|^0.3 - |Ŝ|^0.3)², weight about 1 (pDCCRN's asymmetric term [21]).
- **Target absent** - 10 log10(‖ŝ‖² + τ‖x‖²) on target-absent clips (FUSS inactive loss [40]).
- **Confusion** - on 0.5 s chunks, a hinge that fires when the output is closer (SI-SDR) to an interferer than to the target, in the spirit of X-SepFormer's chunk-level loss [35] (my simplification; ablate).
- **Activity head** - binary cross-entropy against target voice activity.
- **Speaker ID** - classification loss on path-B pooled embedding [18].

### 6.6 Training data recipe (short; the data plan is a separate document)

- **Mixture types** - target + noise 20%; target + 1-3 interferers (+ noise) 45%; target absent (interferers + noise) 10%; all-speech mode 25% (my starting split).
- **Levels** - SIR of target vs loudest interferer uniform -8 to +12 dB (must include quieter targets because of [27]); SNR -5 to +20 dB; a loud near-field "user" talker in about half the mixtures.
- **Rooms and channel** - RIRs on every source (T60 about 0.2-1.0 s); random mic EQ and level changes to cover pocket/table placement.
- **Enrollment** - a different utterance of the target, 3-8 s, 50% noisy (SNR 0-20 dB, interferers in 0-40% of its duration), reverb and EQ augmentation [26][31].
- **Speakers** - as many as the disk allows; hold out test speakers; curriculum from dissimilar to similar pairs [36].
- **License watch** - WHAM! noise is CC BY-NC 4.0 and 17 GB [64]; do not use it for a shipping model.

### 6.7 Training schedule

1. **Stage 1** - clean enrollment, 2 talkers + noise, no absent/null samples (fast convergence; the Oxford two-stage schedule reached its target in 60% fewer steps [31]).
2. **Stage 2** - full recipe (noisy enrollment, 3-4 talkers, absent targets, null mode, own voice), similarity curriculum.
3. **Stage 3** - fine-tune with real RIRs, phone-mic recordings and motion-like gain changes (the UW systems used similar staged fine-tuning [1][5]).
4. **Distill** - train L as a teacher, distill to M1 and S (distillation improved over-suppression and speed for E3Net [22]).

**M4 feasibility (estimate)** - M1 costs about 64 GFLOP per 4 s training example (forward + backward). At an effective 0.25-1 TFLOP/s on the M4 GPU that is about 340 K-1.35 M examples (about 375-1,500 hours of audio) per 24 h. The sequential time-GRU (1,000 steps per 4 s example) may run far slower on MPS than these numbers. Measure the step time in the first hour; if it is too slow, train on 2 s segments or swap the time GRU for a causal dilated depthwise convolution (kernel 3, dilations 1-16), which trains in parallel.

### 6.8 Expected quality (estimates, then verify)

| Test condition | Nearest published reference | M1 estimate | Conf. |
|---|---|---|---|
| Mono, LibriSpeech 2 talkers + WHAM!-like noise, noisy enrollment, causal 8 ms | 8.85 dB, causal TF-GridNet at about 6 GMAC/s (my estimate) with cross-attention; 6.70 dB with FiLM [31] | 6-8.5 dB SI-SNRi | L |
| Clean 2-talker (Libri2Mix-like), clean enrollment | 13.05 dB causal BSRNN with USEF cues at 32 ms latency; 6.87 dB with an ECAPA vector [30] | 8-11 dB | L |
| Real restaurant, phone on the table | CHiME-9 multi-mic systems gained only a few points of intelligibility in real cafeterias [44]; LOTH in-the-wild MOS 2.09 to 3.40 [1] | Clearly below simulation; judge by listening tests and MOS | L |

Targets to hold the model to (goals, not predictions) - "picked" rate at least 90-95% at equal level (PVQE baseline 61% [27]); TSOS under 2% of frames (pDCATTUNET best 1.20% [21]); at least 20-30 dB output reduction on target-absent clips (PVQE 33-34 dB [26]); own-voice reduction of at least 20 dB.

### 6.9 Ablations, in order of expected value

1. **Window** - asymmetric 32/8 ms vs symmetric 8/8 ms (CHiME-9 baseline framing [44]) at equal compute; expect about +1-1.5 dB [14].
2. **Conditioning** - FiLM only vs FiLM + cross-attention (+2.1 dB in [31]).
3. **Model size** - S vs M1 vs L on device (capacity effect in [7][16]).
4. **Output** - complex mask vs 3-tap deep filter below 4 kHz [9].
5. **Losses** - with and without the asymmetric, absent-target and confusion terms.
6. **Enrollment** - clean-only vs noisy training; positive/negative vs positive only.
7. **Time module** - GRU vs causal TCN (training speed on MPS vs quality).

### 6.10 Upgrades after v1 works

- **Two-rate model** - keep M1 on the CPU every 4 ms; run a larger "context" model every 64 ms on the Neural Engine over the last 1-2 s, feeding delayed hints through the FiLM/attention inputs; train jointly with the delay. Evidence - delayed hints lifted small-model TSE from 3.95 to 7.53 dB [7]; the proactive system's fast model alone got 1.45 dB vs 12.48 dB with its slow model [5]. The slow path tolerates Neural Engine scheduling jitter because it is never on the audio thread. Keep identification off the playback path in general - streaming diarization pipelines take 500 ms or more, far beyond a hearing-device budget [74].
- **Two-mic input** - if stereo capture from the built-in mics [77] works at low latency, a 2-channel version adds spatial selectivity for about 5% more compute; multi-mic gains in [13] were large (mono 2.89 dB vs 2-mic 4.75 dB vs 6-mic 7.90 dB).
- **Camera** - the phone camera fixes most confusion when the face is visible (AV-PVQE 9.03 vs -2.31 dB [27]); at minimum use it to label enrollment frames.
- **Several targets** - a conversation mode that enrolls 2-3 people (not researched in depth here).

---

## 7. Licensing and IP notes

| Source | License | Use in Earshot |
|---|---|---|
| Look Once to Hear code [1] | UW non-commercial; commercial rights via UW CoMotion | Read the paper, do not copy code |
| BUT/NTT SpeakerBeam [72] | Evaluation-only; owners reserve patent rights | Do not use |
| USEF-TSE [29] | CC BY-NC 4.0 | Do not use code |
| Oxford positive/negative enrollment code [31] | No license file | Treat as all rights reserved |
| WHAM! / WHAMR! noise [64] | CC BY-NC 4.0 | Not for shipping models |
| Semantic Hearing, Waveformer, Target Conversation Extraction [2][3][6] | MIT | OK with attribution |
| DeepFilterNet code and weights [12] | MIT or Apache-2.0 | OK |
| ESPnet (has TF-GridNet), Asteroid, GTCRN [71][65] | Apache-2.0 / MIT / MIT | OK |
| Resemblyzer, WeSpeaker, SpeechBrain [71] | Apache-2.0 code | OK; check the training-data terms of any pretrained weights |

Patents - UW's founders are commercializing Target Speech Hearing through Hearvana [59], and BUT/NTT's license mentions patent rights [72]. I did not search patent databases; check before any commercial release (open question).

---

## 8. Open questions and things I could not verify

- **iPhone throughput** - real GMAC/s per A18 Pro P-core and E-core for batched GRU steps with BNNS Graph vs Accelerate vs Core ML; power draw over hours. No public measurement found.
- **Training speed** - PyTorch MPS speed for long sequential GRUs on the M4; decides GRU vs TCN for the time axis.
- **Asymmetric windows for TSE** - the +1.5 dB evidence is for LSTM separation [14]; a strong denoiser saw no significant gain [16]. Needs our own ablation.
- **Stereo built-in mics** - whether two raw, time-aligned channels are available at low latency on iPhone 16 Pro.
- **Reverberant target** - dry vs early-reflection target for natural listening [66][67].
- **Patents** - UW, NTT and others in this space.
- **Unverified numbers** - Phonak, Oticon, Starkey and app latency claims are manufacturer statements; Apple Conversation Boost's 5.5 dB comes from a third-party test; X-SepFormer, Zhao et al. and curriculum-learning gains are from abstracts or summaries [34][35][36].
- **Absence claims** - "no consumer phone app does enrolled single-speaker extraction" is based on searches, not exhaustive review.

---

## Appendix A - compute estimate script (section 6.3)

Plain Python, no dependencies. Output on 2026-11-06:

```
S                4.6 M MACs/hop   1.16 GMAC/s  1.40 M params
S at 2 ms hop    4.6 M MACs/hop   2.32 GMAC/s  1.40 M params
M1              10.7 M MACs/hop   2.68 GMAC/s  2.07 M params
L               34.3 M MACs/hop   8.58 GMAC/s  3.87 M params
```

```python
# Earshot-BSR compute estimate: MACs per 4 ms hop and per second, plus parameter count.
# A GRU step costs 3*H*(I+H) MACs. Layer norms and activations are ignored.

def band_widths():
    w = [4]*16 + [8]*8 + [16]*8   # bins per band: 0-2 kHz, 2-4 kHz, 4-8 kHz (31.25 Hz per bin)
    w[-1] += 1                    # Nyquist bin joins the last band -> 257 bins total
    return w

def gru(i, h):
    return 3*h*(i + h)

def model(D, Hb, Ht, B, hop_ms=4.0, taps_low=3, tokens=64, d_att=64, emb=256):
    w = band_widths(); K = len(w)
    macs = sum(3*wi*D for wi in w)                       # band encoder Linear(3w -> D)
    params = sum(3*wi*D + D for wi in w)
    for _ in range(B):                                   # dual-path blocks
        macs += K*(2*gru(D, Hb) + 2*Hb*D)                #   BiGRU across bands + Linear(2Hb -> D)
        macs += K*(gru(D, Ht) + Ht*D)                    #   GRU across time per band + Linear(Ht -> D)
        params += 2*(gru(D, Hb) + 6*Hb) + 2*Hb*D + gru(D, Ht) + 6*Ht + Ht*D
    macs += 2*K*D + 2*K*D*d_att + 2*tokens*d_att         # FiLM apply + cross-attention (K/V precomputed)
    params += 2*K*D*d_att + emb*2*K*D                    # attention projections + FiLM generator
    start = 0
    for wi in w:                                         # decoder per band
        taps = taps_low if start < 128 else 1            # 3-tap deep filter below 4 kHz
        macs += D*2*D + 2*D*2*wi*taps
        params += D*2*D + 2*D*2*wi*taps
        start += wi
    macs += K*D + 4*(128*taps_low + 129) + 2*4*256*9//2  # activity head, filter apply, FFT + iFFT
    fps = 1000.0/hop_ms
    return macs/1e6, macs*fps/1e9, params/1e6

for name, cfg in [("S", dict(D=48, Hb=32, Ht=64, B=3)),
                  ("S at 2 ms hop", dict(D=48, Hb=32, Ht=64, B=3, hop_ms=2.0)),
                  ("M1", dict(D=64, Hb=48, Ht=80, B=4)),
                  ("L", dict(D=96, Hb=64, Ht=128, B=6))]:
    m, g, p = model(**cfg)
    print(f"{name:14s} {m:5.1f} M MACs/hop  {g:5.2f} GMAC/s  {p:4.2f} M params")
```

---

## References

1. Veluri, Itani, Chen, Yoshioka, Gollakota. Look Once to Hear: Target Speech Hearing with Noisy Examples. CHI 2024. https://arxiv.org/abs/2405.06289 (HTML v3 https://arxiv.org/html/2405.06289v3); project https://tsh.cs.washington.edu/; code https://github.com/vb000/LookOnceToHear; license https://raw.githubusercontent.com/vb000/LookOnceToHear/main/LICENSE
2. Veluri, Itani, Chan, Yoshioka, Gollakota. Semantic Hearing: Programming Acoustic Scenes with Binaural Hearables. UIST 2023. https://arxiv.org/abs/2311.00320; code https://github.com/vb000/SemanticHearing
3. Veluri et al. Real-Time Target Sound Extraction (Waveformer). ICASSP 2023. https://arxiv.org/abs/2211.02250; code https://github.com/vb000/Waveformer
4. Chen, Itani, Eskimez, Yoshioka, Gollakota. Hearable devices with sound bubbles. Nature Electronics 2024. https://www.microsoft.com/en-us/research/?p=1105107; UW news https://www.washington.edu/news/2024/11/14/ai-headphones-sound-bubble-noise-cancelling/
5. Hu, Itani, Chen, Gollakota. Proactive Hearing Assistants that Isolate Egocentric Conversations. EMNLP 2025. https://arxiv.org/abs/2511.11473; https://aclanthology.org/2025.emnlp-main.1289
6. Chen et al. Target conversation extraction: Source separation using turn-taking dynamics. Interspeech 2024. https://arxiv.org/abs/2407.11277; https://www.microsoft.com/en-us/research/?p=1060068; code https://github.com/chentuochao/Target-Conversation-Extraction
7. Srinivas, Itani, Chen, Eskimez, Yoshioka, Gollakota. Knowledge boosting during low-latency inference. Interspeech 2024. https://arxiv.org/abs/2407.11055
8. Chatterjee, Kim et al. ClearBuds: Wireless Binaural Earbuds for Learning-Based Speech Enhancement. MobiSys 2022. https://arxiv.org/abs/2206.13611; https://clearbuds.cs.washington.edu
9. Schröter, Rosenkranz, Escalante-B., Maier. Low Latency Speech Enhancement for Hearing Aids Using Deep Filtering. IEEE/ACM TASLP 30, 2022. https://doi.org/10.1109/TASLP.2022.3198548; https://cris.fau.de/publications/281182597
10. Schröter et al. Deep Multi-Frame Filtering for Hearing Aids. Interspeech 2023. https://arxiv.org/abs/2305.08225
11. Schröter et al. DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio. IWAENC 2022. https://arxiv.org/abs/2205.05474
12. Schröter et al. DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement. Interspeech 2023. https://arxiv.org/abs/2305.08227; code https://github.com/Rikorose/DeepFilterNet
13. Wang, Wichern, Watanabe, Le Roux. STFT-Domain Neural Speech Enhancement with Very Low Algorithmic Latency. IEEE/ACM TASLP 2023. https://arxiv.org/abs/2204.09911; https://ar5iv.labs.arxiv.org/html/2204.09911; MERL TR https://merl.com/publications/docs/TR2022-166.pdf
14. Wang, Naithani, Politis, Virtanen. Deep neural network based low-latency speech separation with asymmetric analysis-synthesis window pair. 2021. https://arxiv.org/abs/2106.11794
15. Hershey, Lyon, Dementyev, Reddy, Wisdom, Chatlani. Towards Sub-millisecond Latency Real-Time Speech Enhancement Models on Hearables. 2024/2025. https://arxiv.org/abs/2409.18239
16. Wu, Braun. Ultra-Low Latency Speech Enhancement - A Comprehensive Study. ICASSP 2025. https://arxiv.org/abs/2409.10358; https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/Microsoft___LowLatencySE___ICASSP_2025.pdf
17. Mauler, Martin. A low delay, variable resolution, perfect reconstruction spectral analysis-synthesis system for speech enhancement. EUSIPCO 2007. https://ieeexplore.ieee.org/document/7098797
18. Delcroix et al. Improving speaker discrimination of target speech extraction with time-domain SpeakerBeam. ICASSP 2020. https://arxiv.org/abs/2001.08378
19. Ge, Xu, Wang, Chng, Dang, Li. SpEx+: A Complete Time Domain Speaker Extraction Network. Interspeech 2020. https://arxiv.org/abs/2005.04686
20. Wang et al. VoiceFilter-Lite: Streaming Targeted Voice Separation for On-Device Speech Recognition. Interspeech 2020. https://arxiv.org/abs/2009.04323; https://google.github.io/speaker-id/publications/VoiceFilter-Lite
21. Eskimez, Yoshioka, Wang, Wang, Chen, Huang. Personalized Speech Enhancement: New Models and Comprehensive Evaluation. ICASSP 2022. https://arxiv.org/abs/2110.09625
22. Thakker, Eskimez, Yoshioka, Wang. Fast Real-time Personalized Speech Enhancement: E3Net and Knowledge Distillation. Interspeech 2022. https://arxiv.org/abs/2204.00771
23. Giri, Venkataramani, Valin, Isik, Krishnaswamy. Personalized PercepNet. Interspeech 2021. https://arxiv.org/abs/2106.04129
24. Serre, Fontaine, Benhaim, Dutour, Essid. A lightweight dual-stage framework for personalized speech enhancement based on DeepFilterNet2. ICASSP 2024 workshops. https://arxiv.org/abs/2404.08022
25. TEA-PSE 3.0 (Tencent Ethereal Audio Lab), ICASSP 2023 DNS Challenge. https://arxiv.org/abs/2303.07704
26. Pärnamaa, Saabas. Personalized Speech Enhancement Without a Separate Speaker Embedding Model. Interspeech 2024. https://arxiv.org/abs/2406.09928
27. Adapting Personalized Speech Enhancement for Low-Latency Audio-Visual Target-Speaker Extraction (AV-PVQE). 2026. https://arxiv.org/abs/2609.30631
28. Personalized speech enhancement combining band-split RNN and speaker attentive module (pBSRNN). 2023. https://arxiv.org/abs/2302.09953
29. Zeng, Li. USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction. IEEE/ACM TASLP. https://arxiv.org/abs/2409.02615; code https://github.com/ZBang/USEF-TSE (license https://raw.githubusercontent.com/ZBang/USEF-TSE/main/LICENSE)
30. WeSep: A Modular and Cue-Composable Framework for Target Speaker Extraction. 2026. https://arxiv.org/abs/2607.27436
31. Xu, Yang, Trigoni, Markham. Target Speaker Extraction through Comparing Noisy Positive and Negative Audio Enrollments. NeurIPS 2025. https://arxiv.org/abs/2502.16611; code https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll
32. Zmolikova, Delcroix, Ochiai, Kinoshita, Černocký, Yu. Neural Target Speech Extraction: An Overview. IEEE Signal Processing Magazine 2023. https://arxiv.org/abs/2301.13341
33. Liu, Li, Serrà. Quantitative Evidence on Overlooked Aspects of Enrollment Speaker Embeddings for Target Speaker Separation. 2022. https://arxiv.org/abs/2210.12635
34. Zhao, Yang, Gu, Zhang, Zou. Target Confusion in End-to-end Speaker Extraction: Analysis and Approaches. Interspeech 2022. https://arxiv.org/abs/2204.01355
35. Liu, Du, Wan, Zhou. X-SepFormer: End-to-end Speaker Extraction Network with Explicit Optimization on Speaker Confusion. ICASSP 2023. https://arxiv.org/abs/2303.05023
36. Liu, Liu, Miao, Yamagishi. Target Speaker Extraction with Curriculum Learning. Interspeech 2024. https://arxiv.org/abs/2406.07845
37. Borsdorf, Xu, Li, Schultz. Universal speaker extraction in the presence and absence of target speakers for speech of one and two talkers. Interspeech 2021. https://www.isca-archive.org/interspeech_2021/borsdorf21_interspeech.html
38. Zhang, Borsdorf et al. Speaker Extraction with Detection of Presence and Absence of Target Speakers (TSEJoint). Interspeech 2023. https://www.isca-archive.org/interspeech_2023/zhang23k_interspeech.html
39. Wisdom et al. Unsupervised Sound Separation Using Mixture Invariant Training. NeurIPS 2020. https://arxiv.org/abs/2006.12701
40. Wisdom et al. What's All the FUSS About Free Universal Sound Separation Data? 2020. https://arxiv.org/abs/2011.00803
41. Huang, Guan, Long. Unified Architecture and Unsupervised Speech Disentanglement for Speaker Embedding-Free Enrollment in Personalized Speech Enhancement (USEF-PNet). 2025. https://arxiv.org/abs/2505.12288
42. Saijo, Zhang, Wang, Watanabe, Kobayashi, Ogawa. A Single Speech Enhancement Model Unifying Dereverberation, Denoising, Speaker Counting, Separation, and Extraction (MUSE). 2023. https://arxiv.org/abs/2310.08277
43. Luo, Mesgarani. Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation. IEEE/ACM TASLP 2019. https://arxiv.org/abs/1809.07454
44. Sutherland et al. CHiME-9 ECHI: A Machine Learning Challenge for Enhancing Conversations to Address Hearing Impairment. 2026. https://arxiv.org/abs/2609.26306; challenge code https://github.com/CHiME9-ECHI/CHiME9-ECHI
45. SLT 2026 REAL-TSE Challenge: Real-world Target Speaker Extraction from Conversational Recordings. https://arxiv.org/abs/2607.15198
46. Clarity CEC2 rules https://claritychallenge.github.io/docs/cec2/taking_part/cec2_rules; Clarity latency note https://claritychallenge.github.io/blog/tags/latency
47. Cornell et al. Multi-channel target speaker extraction with refinement: the WavLab submission to CEC2. 2023. https://arxiv.org/abs/2302.07928
48. Phonak Audéo Sphere material - https://audiologyblog.phonakpro.com/?p=834954; https://audiologyblog.phonakpro.com/?p=835364; AudiologyOnline interview https://www.audiologyonline.com/interviews/next-level-sound-processing-with-29099
49. HearingTracker - Oticon Intent launch https://www.hearingtracker.com/de/news/oticon-launches-oticon-intent-a-sensor-driven-hearing-aid-designed-to-understand-your-listening-needs; AI hearing aids review https://legacy.hearingtracker.com/resources/ai-in-hearing-aids-a-review-of-brands-and-models
50. Apple Accessibility - Hearing. https://www.apple.com/accessibility/hearing/
51. FDA. FDA Authorizes First Over-the-Counter Hearing Aid Software (12 Sept 2024). https://www.fda.gov/news-events/press-announcements/fda-authorizes-first-over-counter-hearing-aid-software
52. HearingTracker. What Is Conversation Boost for AirPods Pro (cites Noopl test). https://www.hearingtracker.com/news/enable-conversation-boost
53. Google. Sound Amplifier help (Conversation mode). https://support.google.com/accessibility/android/answer/9157755
54. Engadget. Meta is rolling out Conversation Focus... (Dec 2025). https://www.engadget.com/wearables/meta-is-rolling-out-conversation-focus-and-ai-powered-spotify-features-to-its-smart-glasses-192133928.html
55. TechCrunch. ChatableApps launches its hearing assistance app (May 2020). https://techcrunch.com/2020/05/27/chatableapps-ios/amp/; HearingTracker interview https://www.hearingtracker.com/news/chatableapps-launch
56. App Store - clir Hearing: AI Noise Remover. https://apps.apple.com/us/app/id6476435772
57. App Store - HeardThat. https://apps.apple.com/us/app/-/id1526360165
58. Diehl et al. Deep learning-based denoising streamed from mobile phones improves speech-in-noise understanding for hearing aid users. Frontiers in Medical Engineering 2023. https://www.frontiersin.org/journals/medical-engineering/articles/10.3389/fmede.2023.1281904/full; related Sci. Rep. 2023 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9932078/
59. GeekWire. Seattle startup Hearvana raises USD 6M for AI-powered sound enhancement (2025). https://www.geekwire.com/2025/seattle-startup-hearvana-raises-6m-for-ai-powered-sound-enhancement/
60. Apple Newsroom. Apple debuts iPhone 16 Pro and iPhone 16 Pro Max (A18 Pro). https://www.apple.com/newsroom/2024/09/apple-debuts-iphone-16-pro-and-iphone-16-pro-max/
61. Apple WWDC24 session 10211, Support real-time ML inference on the CPU (BNNS Graph). https://developer.apple.com/videos/play/wwdc2024/10211/
62. Apple WWDC25 session 276, What's new in BNNS Graph. https://developer.apple.com/videos/play/wwdc2025/276/
63. Apple. Audio Session Programming Guide - Configuring Device Hardware (polar patterns). https://developer.apple.com/library/content/documentation/Audio/Conceptual/AudioSessionProgrammingGuide/OptimizingForDeviceHardware/OptimizingForDeviceHardware.html
64. WHAM! noise license (CC BY-NC 4.0) - DCASE datalist https://dcase-repo.github.io/dcase_datalist/datasets/scenes/whisper.html; audEERING catalog https://audeering.github.io/datasets/datasets/wham.html; paper https://arxiv.org/abs/1907.01160
65. GTCRN (MIT). https://github.com/Xiaobin-Rong/gtcrn
66. Venkatesh et al. How much to Dereverberate? Low-Latency Single-Channel Speech Enhancement in Distant Microphone Scenarios. ICASSP 2025. https://arxiv.org/abs/2505.01338
67. Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement. 2026. https://arxiv.org/abs/2603.02641
68. Li, Yang, Wang, Qian. SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech Separation. ICASSP 2022. https://arxiv.org/abs/2201.10800
69. Uphaus, Merboldt, Hofbauer, Gerkmann. Directivity-Conditioned Low-Latency Neural Filtering for Speech Enhancement in Hearing Aids (FiLM, 10 ms). IWAENC 2026. https://arxiv.org/abs/2609.15760
70. DeepFilterNet attenuation limit (`atten_lim_db`, mixes enhanced and noisy signal). https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/enhance.py
71. Toolkits - ESPnet (Apache-2.0) https://github.com/espnet/espnet; Asteroid (MIT) https://github.com/asteroid-team/asteroid; Resemblyzer (Apache-2.0) https://github.com/resemble-ai/Resemblyzer; WeSpeaker (Apache-2.0) https://github.com/wenet-e2e/wespeaker; SpeechBrain (Apache-2.0) https://github.com/speechbrain/speechbrain
72. BUT/NTT SpeakerBeam evaluation license. https://github.com/BUTSpeechFIT/speakerbeam/blob/main/LICENSE.txt
73. Apple Support. Use Voice Isolation, Wide Spectrum, or Automatic Mic Mode. https://support.apple.com/101993
74. Burke, Raj, Kinahan. Target Speaker Identification: A Low-Latency Streaming Pipeline. 2026. https://arxiv.org/abs/2608.17972
75. Delcroix et al. Speaker activity driven neural speech extraction. ICASSP 2021. https://arxiv.org/abs/2101.05516
76. Lightweight Speech Enhancement Guided Target Speech Extraction in Noisy Multi-Speaker Scenarios. 2025. https://arxiv.org/abs/2508.19583
77. Apple. Capturing stereo audio from built-in microphones (iOS 14+). https://developer.apple.com/documentation/avfaudio/capturing-stereo-audio-from-built-in-microphones; `AVAudioSession.PolarPattern.stereo` https://developer.apple.com/documentation/avfaudio/avaudiosession/polarpattern/stereo
78. StarTSE: Towards Streaming Target Speaker Extraction via Chunk-wise Interleaved Splicing of Autoregressive Language Model. 2026. https://arxiv.org/abs/2604.19635 (RTF figure seen in a search snippet only, not in the full text)
