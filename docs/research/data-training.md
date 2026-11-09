# Earshot - data, speaker encoder, training recipe and evaluation plan

Research date 2026-11-06. Scope - everything needed to train a causal target-speaker extraction (TSE) model for Earshot
on one M4 MacBook Pro (MPS, 24-48 h of training), with at most about 10 GB of datasets on disk and about 11 MB/s of
download bandwidth.

How to read this document

- Every factual claim carries a reference number like [12] and a confidence tag - [H] high (read in a primary source),
  [M] medium (secondary source, or primary source that is ambiguous), [L] low (inference or unverified).
- "Measured" means measured by me on this Mac on 2026-11-06 with the scripts in
  `docs/research/data-training-bench/` (results JSON next to them). The machine was shared with other jobs, so some
  CPU numbers were taken under background load; I give ranges where runs disagreed.
- Things I could not verify are labelled "not verified".

Prior work this plan builds on - UW's Look Once to Hear (Veluri et al., CHI 2024) [42], NTT/BUT's work on inactive
target speakers [43], Microsoft's personalized speech enhancement work and DNS challenges [45][63], Google's
VoiceFilter-Lite [46], WeSep/WeSpeaker [34][41]. Look Once to Hear's code is under a UW non-commercial license [42], so
we reuse ideas, not code.

---

## 0. Decision summary (the plan in one page)

**Sample rate** - store and train everything at 16 kHz mono for v1. Every commercially usable corpus below exists at
16 kHz or higher, and 16 kHz keeps compute and latency low. If the model design moves to 24 kHz, swap sources as noted
in section 1.6.

**Storage format** - 16-bit FLAC, one file per utterance, plus one manifest (JSON lines). Noise and RIR banks also as
FLAC / float16 `.npy`. Measured on this Mac [66] - FLAC costs 68 MB per hour of 16 kHz speech (59 % of raw int16) and
serves 1,100 to 1,700 random 4 s crops per second per core (about 5,000 per second with 6 to 8 loader processes, roughly
8x what training needs). Opus is 6x smaller but about 8x slower to random-access and lossy. MP3 random seeks produced
decoder errors. Raw int16 memmap is fastest but 1.7x bigger than FLAC.

**Data pack v1 (about 9 GB on disk)**

| Part | Sources (license) | What we keep | Disk |
|---|---|---|---|
| Speech, targets and interferers | LibriTTS-R train (CC BY 4.0), VoxPopuli en (CC0), GLOBE V2 (CC0), VCTK 0.92 (CC BY 4.0), optional Emilia-YODAS EN (CC BY 4.0, gated) | about 4,500 speakers (about 5,000 with Emilia-YODAS), 30 s to 2 min each, about 75 h | about 5.0 GB |
| Noise and music | FSD50K CC0 + CC BY clips only, DEMAND restaurant/cafe/station scenes, MUSAN noise + music | about 25 h | about 2.0 GB |
| RIRs | 4,000 simulated rooms (pyroomacoustics, MIT), OpenSLR 28 real RIRs (Apache 2.0), MIT IR Survey (CC BY 4.0, test only) | about 30k RIRs | about 1.0 GB |
| Evaluation | LibriSpeech test-clean (already on disk), EdAcc test (CC BY-SA 4.0), VoxPopuli en_accented test (CC0), WHAM! noise test split (CC BY-NC 4.0, eval only), VOiCES subset (CC BY 4.0) | rendered on the fly with fixed seeds | about 1.0 GB |

Download volume is about 120 GB in total (most of it the LibriTTS-R tarballs, which we stream and filter so only the
kept 3 % reaches the disk). At 11 MB/s that is about 3 h of downloading, which can run overnight.

**Disk check (measured 2026-11-06, 21:30)** - the data volume had 17 GiB free. The brief's rule (never below 10 GB free)
means the full 9.3 GB pack needs about 19.5 GB free first, so it needs the cleanup the user may approve. Without it, use
the **v1-lite pack (about 6 GB)** - LibriTTS-R at 45 s per speaker (2.0 GB), 1 VoxPopuli shard (0.45 GB), 4 GLOBE shards
(0.45 GB), VCTK (0.25 GB), FSD50K subset without music (0.7 GB), DEMAND (0.3 GB), MUSAN noise only (0.6 GB), 2,000
simulated rooms plus real RIRs (0.65 GB), eval sets (0.6 GB). The lite pack keeps about 3,500 speakers.

**Speaker encoder** - WeSpeaker ResNet34 (VoxCeleb, plain version, 6.63 M parameters, published VoxCeleb1-O EER
0.87 %, CC BY 4.0 per WeSpeaker, Apache 2.0 per its model card). In my test of 6 encoders on short, noisy enrollments it
was the most robust (EER 0.45 % clean, 4.3 % at 5 dB SNR, 8.4 % at 0 dB with 3 s of speech; Resemblyzer, used by Look
Once to Hear, was 3.9 / 16.4 / 27.7 %). With a Kaldi-exact filterbank written as one Conv1d inside the model, it converts
to Core ML from a single trace and matches PyTorch exactly (cosine 1.0000) on CPU and GPU. The Neural Engine gave wrong
embeddings (cosine 0.57), so run it with `computeUnits = .cpuAndGPU` (8 ms per 5 s clip on this Mac).

**Training recipe** - scale-dependent thresholded SNR loss (SNR clamp 30 dB) on target-present examples, a
mixture-referenced energy loss on target-absent examples and target-silent frames (10 % of examples have no target),
dynamic on-the-fly mixing of 0 to 3 competing talkers plus 6 to 20 talker babble, restaurant noise and music at SNR
around 5 dB (clipped to -8 to 25 dB), simulated phone-in-hand and phone-on-table rooms with RT60 mostly 0.2 to 1.3 s, an own-voice
interferer close to the phone, and noisy, reverberant 3 to 6 s enrollment from a different utterance. AdamW at 1e-3
with warmup and cosine decay, gradient clipping 5, bf16 autocast, batch 8 to 16 of 4 s. Use LSTM, not GRU, on MPS
(measured 14x faster in PyTorch 2.7.1).

**Evaluation** - SI-SDRi and SNRi (primary), ESTOI, DNSMOS P.835 and personalized DNSMOS, PESQ for internal use only,
Whisper WER, speaker-confusion rate, failure rate, target-over-suppression rate, target-absent attenuation and
false-alarm rate. Test sets - Libri2Mix 16 kHz min test with the standard SpeakerBeam enrollment list (for comparison
with literature), an Earshot restaurant simulation built only from held-out speakers, noises and real RIRs, accent sets
(EdAcc, VoxPopuli accented), real replayed recordings (VOiCES, DiPCo) and our own loudspeaker playback rig recorded on
the iPhone.

**Things only the user can do** - accept the Emilia dataset terms on Hugging Face if we want Emilia-YODAS (optional);
everything else in the v1 pack downloads without accounts or gates. Also - approve about 9.3 GB of disk and about
119 GB of download traffic, and provide 3 to 5 small powered speakers for the real-room test rig (section 6.3).

---

## 1. Speech corpora

### 1.1 What the extractor needs from speech data

- **Many speakers** - in NTT/BUT's study, moving the same TSE model from 251 to 921 training speakers raised SDRi by
  1.2 dB and halved the failure rate (3.4 % to 1.7 %) [43] [H]. WeSep trained on VoxCeleb1 (in-the-wild speech) got
  16.18 dB on Libri2Mix versus 13.32 dB when trained on Libri2Mix train-100, and 10.18 versus 5.54 dB on a
  cross-language test [41] [H].
- **Several utterances per speaker** - enrollment must come from a different utterance than the target [43] [H]. A
  minute of speech per speaker gives about 10 utterances, which is enough for target/enrollment pairs.
- **Clean targets** - the target reference is what the model learns to output, so noisy or reverberant "clean" speech
  teaches the model to keep that noise. Prefer restored or studio speech for targets; noisier conversational speech is
  fine for interferers and babble.
- **Accent and style variety** - read audiobook English (LibriSpeech family) is mostly North American; the product
  must work on every accent, so add accented corpora (VoxPopuli, GLOBE, VCTK, EdAcc for eval).

My reasoning (not a measured result) - with a fixed disk budget, more speakers at about 1 minute each beats fewer
speakers at many minutes each, because the model's hardest job is following an embedding it has never seen. This should
be checked with a small ablation (section 5.9) [L].

### 1.2 Corpus table

Sizes are the download size unless marked. "Partial" says whether you can fetch only part of the corpus.

| Corpus | Speakers / hours | Variety | License / gate | Download and URL | Rate | Partial download |
|---|---|---|---|---|---|---|
| LibriSpeech [1][2] | train 251 + 921 + 1,166 = 2,338 speakers (counted from SPEAKERS.TXT), about 960 h | read audiobooks, mostly US English; train-other-500 is noisier | CC BY 4.0 [H], no gate | train-clean-100 6.3 GB, train-clean-360 23 GB, train-other-500 30 GB at `https://www.openslr.org/resources/12/` | 16 kHz FLAC | monolithic tar.gz, but can be streamed and filtered so only kept files hit the disk; HF parquet mirror `openslr/librispeech_asr` has about 20 speakers per 480 MB shard (measured) |
| LibriTTS-R [3][4] | train 247 + 904 + 1,160 = 2,311 speakers, 555 h (from LibriTTS paper Table 2) | same speakers and books as LibriSpeech, sentence-split, restored with Miipher | CC BY 4.0 [H], no gate | train_clean_100 8.1 GB, train_clean_360 28 GB (29.0 GB measured), train_other_500 46 GB at `https://www.openslr.org/resources/141/` | 24 kHz | stream and filter (server supports range requests, measured); HF mirror `mythicinfinity/libritts_r` has about 10 speakers per 530 MB shard (measured) |
| MLS English [5] | 5,490 train speakers (2,742 M, 2,748 F), 44,660 h | LibriVox audiobooks | CC BY 4.0 [H], no gate | 2.4 TB FLAC, 651 GB Opus at `https://dl.fbaipublicfiles.com/mls/`; HF `parler-tts/mls_eng` 705 GB in 1,416 shards | 16 kHz | HF shards are sorted by speaker, 9 to 11 speakers per 500 MB shard (measured), so collecting many speakers means reading row groups from hundreds of shards; not worth it for v1 |
| VCTK 0.92 [6] | 110 speakers, about 400 sentences each | accents from `speaker-info.txt` (read remotely) - English 33, American 22, Scottish 19, Irish 9, Canadian 8, Northern Irish 6, South African 4, Indian 3, Australian 2, Welsh / New Zealand / British / unknown 1 each; hemi-anechoic studio | CC BY 4.0 (README "COPYING" section) [H], no gate | `VCTK-Corpus-0.92.zip` 11.75 GB, direct link `https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content`; the mic1 FLACs alone are 44,455 files, 6.26 GB (measured from the zip directory) | 48 kHz FLAC | yes, the server supports HTTP range requests (measured), so single files can be pulled out of the zip without downloading it |
| Common Voice [7] | very large | crowd-sourced, every accent, phone and laptop mics | CC0, but since October 2025 only through Mozilla Data Collective with an account and new terms [H] - gated | `https://datacollective.mozillafoundation.org` | 48 kHz MP3 | gated, so v1 uses GLOBE instead |
| GLOBE V2 [8] | 23,519 speakers, 164 accents (paper), 535 h | Common Voice clips, filtered (SNR, bandwidth, Whisper alignment) and denoised with Adobe Podcast Enhance (paper) [H] | CC0 [H], no gate | `MushanW/GLOBE_V2` on HF, 76.1 GB in 168 parquet shards of about 453 MB | 44.1 kHz in V2 (24 kHz in the paper) | yes, per shard; one shard holds 3.2 h, 152 to 197 speakers, but 66 to 79 % of rows are "United States English" (measured on 2 shards) |
| VoxPopuli en [9] | 1,313 speakers, 543 h transcribed | European Parliament speeches by members from across the EU (the card does not give the share of non-native speakers; I expect many [L]); `en_accented` test split has 15 L2 accents, 29 h [H] | CC0 [H], no gate | `facebook/voxpopuli` on HF, `en` 98.8 GB in 30 shards of 3.2 GB | 16 kHz | yes, and shards are shuffled - one 3.2 GB shard holds about 470 speakers (measured on 2 shards) |
| AMI headset [10] | about 100 h of meetings | real conversations, mostly non-native speakers, headset mics with crosstalk | CC BY 4.0 [H], no gate | `edinburghcstr/ami` config `ihm` 15.2 GB in 51 parquet shards; originals at `https://groups.inf.ed.ac.uk/ami/download/` | 16 kHz | yes, per shard |
| DNS Challenge 5 clean speech [11] | read_speech about 299 GB (LibriVox), VCTK 27 GB, emotional 2.4 GB, others | read speech, singing, emotional; mostly re-packaged public corpora | "provided under the original terms" of each source [H]; repo LICENSE is CC BY 4.0 [H] | `https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/Track1_Headset/...` multi-part tarballs | 48 kHz | only sequentially (first parts of a split tarball) [M]; not needed since we already have the LibriVox and VCTK originals |
| Emilia (original) [12] | 46.8k h English | in-the-wild video and podcast speech [M], run through source separation, pyannote diarization and DNSMOS filtering [H] | CC BY-NC 4.0 [H], gated | `amphion/Emilia-Dataset` folder `Emilia/EN`, 1,140 tars, 1.16 TB (measured listing) | 24 kHz MP3 | per tar | 
| Emilia-YODAS [12] | 92.2k h English | YouTube CC-BY videos, conversational | CC BY 4.0 [H], gated (accept conditions on HF) | `amphion/Emilia-Dataset` folder `Emilia-YODAS/EN`, 1,362 tars of about 1.09 GB (1.49 TB, measured listing) | 24 kHz MP3 (paper says Emilia is 24 kHz) [M] | yes, per tar (one tar is about 68 h by arithmetic) [L]; YODAS part used a lower DNSMOS filter (2.4 instead of 3.0) [H]; Emilia metadata has a per-video diarized `speaker` field, not verified for the YODAS part |
| EARS [13] | 107 speakers, 100 h | anechoic, 7 reading styles, 22 emotions, about 18 min of free monologue per speaker | CC BY-NC 4.0 (LICENSE file) [H] | 108 GitHub release zips, 70.6 GB total, about 650 MB per speaker (measured via GitHub API) | 48 kHz | yes, per speaker; non-commercial, so eval only if at all |
| Expresso [14] | 4 speakers, about 40 to 46 h | expressive, improvised dialogues | CC BY-NC 4.0 [H] | `expresso.tar` 36 GB | 48 kHz | no; 4 speakers and non-commercial, skip |
| EdAcc [15] | 40 h; test + validation only | dyadic Zoom conversations, 25 accent and 29 L1 categories | CC BY-SA 4.0 [H], no gate | `edinburghcstr/edacc` 6.95 GB in 16 parquet shards | 32 kHz | yes, per shard; eval only |

Notes

- **LibriTTS-R versus LibriSpeech** - same speakers. LibriTTS-R is sentence-segmented, restored by Miipher and 24 kHz
  [3] [H], so it makes cleaner targets. Exclude the files listed in
  `libritts_r_failed_speech_restoration_examples.tar.gz` [3] [H]. LibriSpeech's raw audiobook audio contains room tone
  and mic noise that would become part of the "target".
- **GLOBE caveat** - the paper says GLOBE was denoised with Adobe Podcast Enhance [8] [H]. Listen to 20 random clips
  before trusting it as targets; if it sounds over-processed, use GLOBE only for interferers and enrollment variety.
- **HF mirror license mismatch** - the `mythicinfinity/libritts_r` card's BibTeX links CC BY-NC-ND (that is the paper's
  license), while OpenSLR lists the dataset as CC BY 4.0 [3] [H]. Download from OpenSLR to keep the provenance clean.
- **VoxCeleb** - not needed for TSE training in v1 (the pretrained speaker encoder already learned from it). Its
  license page wording varies between mirrors (CC BY 4.0 "for research purposes" on one, CC BY-SA 4.0 on others) [39]
  [M], and audio needs a request form [39] [H].

### 1.3 Recommended disk-efficient mix (speech part, about 5.0 GB)

| Source | Use | Selection rule | Speakers | Hours kept | Disk (16 kHz FLAC at 68 MB/h) |
|---|---|---|---|---|---|
| LibriTTS-R train (all three subsets) | targets (60 %), interferers, babble | stream the three tarballs (82 GB), keep up to 60 s per speaker spread over at least 2 chapters, drop failed-restoration files | 2,311 | about 38 | about 2.6 GB |
| VoxPopuli en, 2 train shards | targets (15 %), interferers | keep up to 60 s per speaker, skip clips under 2 s | about 800 (estimate from 470 per shard with overlap) [L] | about 13 | about 0.9 GB |
| GLOBE V2, 8 shards | targets (15 %), interferers, accent variety | keep every non-US-accent row, cap US speakers at 30 s | about 1,300 | about 14 | about 0.9 GB |
| VCTK 0.92 mic1 | targets (5 %), accents | pull `*_mic1.flac` files for 102 training speakers by range request, keep 2 min each; hold out 8 speakers for test | 102 | about 3.4 | about 0.25 GB |
| Emilia-YODAS EN, 1 tar (optional, gated) | interferers, babble, 5 % of targets | DNSMOS OVRL from metadata at least 3.2, 3 to 15 s clips | several hundred (not verified) | about 6 | about 0.4 GB |
| **Total** | | | **about 4,500 (about 5,000 with Emilia-YODAS)** | **about 75** | **about 5.0 GB** |

Speaker-disjoint splits - hold out 3 % of speakers from every source for validation, and use official test speakers
(LibriSpeech test-clean, VoxPopuli test, GLOBE test, 8 VCTK speakers, EdAcc) for testing. Never let a test speaker
appear as an interferer or in babble during training.

### 1.4 Storage format, measured

Benchmark - 400 LibriSpeech test-clean utterances (0.88 h), Python 3.12, soundfile 0.14 / libsndfile 1.2.2, this M4
[66].

| Format | MB per hour of 16 kHz mono speech | Full decode speed, 1 core | Random 4 s crop, 1 core | Random 4 s crop, 6 to 10 processes | Notes |
|---|---|---|---|---|---|
| FLAC (as shipped by LibriSpeech) | 67.9 | 7,900x real time | 1,100 to 1,700 per s | 5,000 to 9,200 per s | lossless, seekable |
| Opus 24 kbps (Ogg) | 10.7 | 940x | 160 to 210 per s | 600 to 1,200 per s | lossy; a random seek costs about as much as decoding from the file start |
| Opus 32 kbps | 14.2 | 850x | 190 per s | 560 to 700 per s | lossy |
| MP3 48 kbps | 22.0 | 7,500x | 1,650 per s | 4,200 to 4,500 per s | libmpg123 printed about 25,000 "part2_3_length too large" errors during random seeks (bit-reservoir frames), so crops can start with glitches |
| int16 raw memmap | 115.2 | not applicable | 20,000 to 27,000 per s (warm) | 50,000 to 110,000 per s | no decode; biggest on disk |

What training needs - a batch of 16 examples uses about 10 to 12 crops per example (target, enrollment, 0 to 3
interferers, babble talkers, noise), so 160 to 200 crops per step. At 3 steps per second that is about 600 crops per
second, well inside FLAC's 5,000 or more with 6 to 8 workers. Parallel scaling was noisy because other jobs were
using the CPU, and the M4 has 4 performance and 6 efficiency cores, so do not count on more than about 5,000 crops per
second.

Decision - FLAC for all speech and noise, float16 `.npy` for the RIR bank. Keep targets lossless: an Opus-coded target
would teach the model to reproduce codec artifacts.

Other measured costs that matter for the loader [66]

- FFT convolution of a 4 s source with a 0.3 / 0.8 / 1.5 s RIR (scipy `fftconvolve`, 1 core) - 0.45 / 0.61 / 1.0 ms.
- Batched FFT convolution on MPS (8 examples x 5 sources, 0.8 s RIRs) - 6.6 ms per batch.
- pyroomacoustics ISM, 5 RIRs per room, `max_order` capped at 40 - about 0.2 s per room on one core; the hybrid
  ray tracer took 0.3 to 0.9 s.

### 1.5 How to download each speech source (exact steps)

- **LibriTTS-R** - stream `https://www.openslr.org/resources/141/train_clean_100.tar.gz` (8.1 GB),
  `.../train_clean_360.tar.gz` (29.0 GB measured), `.../train_other_500.tar.gz` (46 GB) through Python
  `tarfile.open(fileobj=response, mode="r|gz")`, keep members by the per-speaker budget, resample 24 to 16 kHz with
  `scipy.signal.resample_poly(x, 2, 3)`, write FLAC. About 2.1 h at 11 MB/s. Peak disk is only what you keep.
- **VoxPopuli** - `hf_hub_download("facebook/voxpopuli", "en/train-00000-of-00030.parquet", repo_type="dataset")` and
  shard 00015 (3.2 GB each), convert, delete the parquet. Do one shard at a time to keep peak disk under 4 GB.
- **GLOBE V2** - shards `data/train-000NN-of-00164.parquet` (about 453 MB each) from `MushanW/GLOBE_V2`. Pick 8 spread-out
  shard numbers. Resample 44.1 to 16 kHz (`resample_poly(x, 160, 441)`).
- **VCTK** - open the zip remotely (Python `zipfile` over an `fsspec` HTTP file, which reads only the central directory
  and the members you ask for; this worked on the VCTK, WHAM!, OpenSLR 28 and MIT IR zips in my tests) and pull
  `wav48_silence_trimmed/pXXX/pXXX_NNN_mic1.flac` (layout confirmed from the zip directory) plus `speaker-info.txt`. No
  11.75 GB temporary file. The README says p280 and p315 had problems with the MKH 800 microphone [6] [H]; Hugging Face
  loader code says p280 has no mic2 files and p315 has no transcripts [M]. We only use mic1, and transcripts are not
  needed for training.
- **Emilia-YODAS (optional)** - user accepts the terms at `https://huggingface.co/datasets/amphion/Emilia-Dataset`, logs in
  with `hf auth login`, then download one tar from `Emilia-YODAS/EN/`. Read the JSON next to each MP3 for `dnsmos` and
  speaker fields.

### 1.6 If the model moves to 24 kHz

LibriSpeech, VoxPopuli, AMI and the MUSAN/DEMAND 16 kHz files would be band-limited, and training on a mix of 16 kHz and
24 kHz targets would teach the model that missing highs are normal. A 24 kHz pack would use LibriTTS-R, GLOBE (44.1 kHz),
VCTK (48 kHz), Emilia-YODAS (24 kHz), DEMAND 48 kHz files and FSD50K originals (44.1 kHz, 24.7 GB Zenodo split zip).
FLAC at 24 kHz will be somewhat bigger than 68 MB/h (not measured).

---

## 2. Noise and interference for restaurants, bars and parties

### 2.1 What real loud rooms look like (sets the SNR ranges)

- **Real SNRs** - 20 older adults wore recorders for 5 to 6 weeks; across 894 listening situations SNR fell from about
  20 dB at 40 dBA of noise to 0 dB at about 74 dBA, and only 7.5 % of situations were below 0 dB. "Noisy" clusters
  averaged 4.2 dB, indoor crowds had a median of 5.3 dB [30] [H].
- **Talkers raise their voice** - speech level rose 0.34 dB per dB of noise below 59 dBA and 0.54 dB per dB above it
  [30] [H] (Lombard effect, section 3.4).
- **Conversational SNR goes negative above about 69 dB SPL** of background at 1 m talker distance (75 dB SPL at 0.5 m)
  in a lab study with real conversations [31] [M] (abstract only).
- **Restaurant levels** - a Hong Kong survey found 66.7 to 82.6 dBA, mean 73.9 dBA [32] [M] (via search summary).
- So a phone pointed at a talker about 1 m away in a busy restaurant sees roughly -5 to +10 dB SNR, centred near
  +3 to +5 dB [L, inference from the above].

### 2.2 Noise corpus table

| Corpus | Content | Size and URL | License | Use |
|---|---|---|---|---|
| Babble from our own speech pool | 6 to 20 talkers summed, each with its own RIR | no disk | inherits speech licenses | main babble source (train); held-out speakers only for test |
| DEMAND [16] | 16-channel recordings, 300 s per scene. Relevant - PCAFETER (busy office cafeteria), PRESTO (university restaurant at lunch), SCAFE (cafe terrace, 48 kHz only), OMEETING (meeting room), PSTATION (subway station), SPSQUARE (town square), OHALLWAY (office hallway), plus TBUS, TMETRO, STRAFFIC | per-scene zips at `https://zenodo.org/records/1227121/files/<SCENE>_16k.zip`, 78 to 130 MB each (16 kHz), SCAFE_48k 326 MB | the bundled PDF says CC BY-SA 3.0, the Zenodo panel says CC BY 4.0 [H for both statements]; treat as CC BY-SA 3.0 | train on 8 scenes, keep 4 of 16 channels (about 0.3 GB); hold out SCAFE for test |
| MUSAN [17] | music 42.5 h (Jamendo, Free Music Archive, Incompetech, HD Classical), speech 60 h (LibriVox 20 h, US government hearings 40 h), noise 929 files about 6 h (Free Sound, Sound Bible; DTMF, rain, crowds, footsteps etc.) | `https://www.openslr.org/resources/17/musan.tar.gz`, 11.09 GB (measured) | CC BY 4.0 overall, per-file CC or US public domain; "we only include content that allows for commercial use" (paper) [H] | stream the tarball, keep noise/ and about 6 h of music (about 0.9 GB) |
| FSD50K [18] | 51,197 clips, 108.6 h, 200 AudioSet classes | Zenodo split zip 24.7 GB at `https://zenodo.org/records/4060432`; 16 kHz parquet mirror `quinnlue/FSD50K-16k` 6.9 GB with per-clip `labels` and `license` columns (measured) | per clip - CC0 42.4 h, CC BY 47.6 h, CC BY-NC 14.9 h, Sampling+ 3.7 h (my scan of the mirror's metadata) [H for the scan, M for the mirror's fidelity] | keep CC0 + CC BY only; restaurant-relevant classes (hours available in that subset) - Music 19.5, Chatter 1.7, Crowd 1.1, Laughter 1.5, Applause 1.4, Glass 1.6, Chink and clink 0.8, Dishes 0.6, Cutlery 0.4, Walk/footsteps 1.2, Door 2.3, Water tap 1.5, Typing 1.1 |
| DNS Challenge 5 noise [11] | AudioSet (CC BY 4.0) and Freesound (CC0 only) clips, about 60k clips, 181 h (DNS 2023 paper [63]) | 9 archives `noise_fullband/datasets_fullband.noise_fullband.audioset_00X.tar.bz2` and `...freesound_00X...`, about 39 GB total | source licenses as stated [H] | optional - one archive (about 4.3 GB) for extra variety |
| WHAM! noise [19] | binaural recordings in San Francisco Bay Area restaurants, cafes, bars and parks, intelligible speech removed; 28,000 clips, 81.7 h; tr 20,000 / cv 5,000 / tt 3,000 | `wham_noise.zip` 18.19 GB (measured) at `https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip`; supports range requests, `tt/` alone is 4.15 GB uncompressed float32 stereo 16 kHz (measured) | CC BY-NC 4.0 [H] - non-commercial | test only - the best match to Earshot's rooms; pull `tt/` by range request and keep one channel as FLAC |
| ACE corpus [23] | ambient, fan and babble noise recorded in 7 rooms with each mic array, including a 3-channel "mobile" array | `ACE_Corpus_RIRN_Mobile.tbz2` 1.59 GB at Zenodo record 6257551 | Zenodo says CC BY 4.0, Imperial's page says CC BY-ND 4.0 [H for both statements] | optional test material; ND wording means do not redistribute modified versions |

Music licenses - MUSAN music (commercial use allowed per the paper) and FSD50K Music clips filtered to CC0 / CC BY are
enough. Note the CC BY clips need attribution in the app credits (FSD50K metadata has uploader names).

### 2.3 Recommended noise pack (about 2.0 GB)

FSD50K CC0 + CC BY restaurant classes and 4 h of music (about 1.0 GB), DEMAND 8 scenes x 4 channels (about 0.3 GB),
MUSAN noise + 6 h music (about 0.9 GB, check after conversion), babble on the fly. Test-only - 1,000 WHAM! `tt` clips
(about 3 h, about 0.3 GB as mono FLAC, estimate) and DEMAND SCAFE.

---

## 3. Room impulse responses, phone microphone, Lombard speech

### 3.1 Real RIR sets

| Set | Content | Size, URL | License | Use |
|---|---|---|---|---|
| MIT IR Survey [21] | 270 IRs from places 7 volunteers visited in daily life, 1.5 m source-mic distance; my count of file names - 14 bars, 4 restaurants, 2 coffee shops, 2 fast food, 2 pizzerias, wine bar, cafeteria, food court, dining halls, kitchens (about 30 food/drink venues) | `https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip`, 11.7 MB, 32 kHz 24-bit mono (measured) | CC BY 4.0 per mirrors and the Silicon Labs MLTK loader [M]; the lab page itself states no license | test only (held-out real restaurant and bar rooms) |
| OpenSLR 28 RIRS_NOISES [20] | real RIRs from RWCP, REVERB 2014 and Aachen AIR (417 wav, 0.84 GB uncompressed, 16 kHz, some 2-channel), 60,600 simulated RIRs (small/medium/large rooms), 846 point-source noises from MUSAN | `https://www.openslr.org/resources/28/rirs_noises.zip`, 1.31 GB (measured listing) | Apache 2.0 [H] | real RIRs for 15 % of training examples |
| OpenSLR 26 [20] | simulated RIRs | `sim_rir_16k.zip` 178 MB | Apache 2.0 [H] | optional; our own simulation is more targeted |
| BUT ReverbDB [22] | 9 rooms (offices, stairwell, hotel room, conference, lecture, meeting rooms), 31 mics each, about 4,700 RIRs, plus room noise | RIR-only package 8.7 GB at `https://speech.fit.vut.cz/software/but-speech-fit-reverb-database` | CC BY 4.0 [H] | optional; too big for v1 unless 1 or 2 rooms are kept |
| ACE [23] | 7 rooms, mic configs from single to 32-channel, including a 3-channel "mobile" config; RIRs plus ambient/fan/babble noise | 24.5 GB total, Mobile 1.59 GB | CC BY 4.0 vs CC BY-ND 4.0 conflict (see above) | optional test RIRs for a phone-like device; the device is not described on the pages I read (not verified) |

### 3.2 Simulated rooms with pyroomacoustics (MIT license [24])

pyroomacoustics provides `ShoeBox(..., materials, max_order, air_absorption, use_rand_ism, max_rand_disp)`,
`inverse_sabine(rt60, dims)`, an experimental ISM plus ray-tracing hybrid and frequency-independent cardioid-family mic
and source directivities (ISM only) [24] [H]. `use_rand_ism` reduces sweeping echoes [24] [H].

Measured [66] - `inverse_sabine` targets did not land where asked: 0.8 s became 1.07 s and 1.2 s became 1.71 s (Schroeder
measurement, order capped at 40). So always measure the realized RT60 of each generated room with
`pra.experimental.measure_rt60` and sample rooms by the measured value.

Room distribution (my proposal, based on 2.1)

| Parameter | Distribution |
|---|---|
| Room size | length 4 to 20 m, width 3 to 15 m, height 2.5 to 5 m (restaurants, bars, halls, living rooms) |
| RT60 (measured) | 0.2 to 1.3 s, roughly log-uniform, with 10 % of rooms up to 2 s; a search summary of one study of four dining venues reported 1.3 to 2.2 s at 500 Hz (paper not opened) [L], and an acoustics vendor's design target for restaurants is 0.6 to 1.0 s [67] [L] |
| ISM | `max_order` from `inverse_sabine` capped at 40, `use_rand_ism=True`, `max_rand_disp=0.05`, `air_absorption=True`, `fs=16000` |
| Bank | 4,000 rooms x 7 sources (target, 3 interferers, own voice, 2 noise points), truncated where energy is 60 dB down (cap 1.0 s), float16 - about 0.7 GB; generation about 4,000 x 0.25 s / 10 cores, a few minutes |

Geometry for the two ways Earshot is used (my proposal)

- **Phone in hand (60 % of rooms)** - mic height 1.0 to 1.4 m. Target mouth 0.5 to 1.5 m away horizontally, at 1.1 to
  1.3 m (seated) or 1.45 to 1.75 m (standing). Interferers 1.0 to 6 m away in random directions. Own voice (the user)
  0.25 to 0.5 m from the mic and 0.15 to 0.4 m above it. Noise points 2 to 8 m away.
- **Phone on table (40 %)** - mic height 0.72 to 0.78 m. A phone lying on the table is close to a boundary microphone, so
  add a second image of every source mirrored about the table plane, scaled by a reflection factor of 0.6 to 0.9, in the
  same RIR computation (pyroomacoustics has no table object). Target 0.6 to 1.5 m away at seated mouth height. Own voice
  0.4 to 0.8 m.
- **Target reference** - target convolved with the direct path plus the first 50 ms of its RIR, with a 5 ms half-cosine
  fade. Early reflections within about 50 ms help intelligibility about as much as direct sound [29] [M], and asking a
  low-latency causal model to remove them too would cost quality. Keep the propagation delay in the reference (do not
  time-align), so the model never needs look-ahead.
- **Faster alternative** - FRA-RIR (Apache 2.0) approximates ISM by random sampling of reflection paths, runs without a
  GPU and is used for on-the-fly RIRs in WeSep [25][41] [H]. Worth trying if the precomputed bank shows overfitting.

### 3.3 Phone microphone simulation

- **Use measurement mode on the phone** - Apple says `AVAudioSession.Mode.measurement` minimizes system-supplied signal
  processing and uses the primary microphone on multi-mic devices [28] [H]. So the model should see the raw mic signal,
  and training should simulate a raw mic, not a voice-processed one.
- **No public measured iPhone 16 Pro mic response found** - I did not find a dataset of smartphone mic impulse responses
  with a clear license (not verified as absent). Measure ours once (log sweep from a flat speaker at 1 m in a quiet
  room) and centre the augmentation on it.
- **Augmentation chain (applied to the mixture; the same linear EQ is applied to the target reference so the model is
  not asked to undo random EQ)**
  - High-pass - 2nd order, corner 60 to 200 Hz.
  - Colour - 0 to 2 peaking filters, plus or minus 4 dB, 300 Hz to 6 kHz; high shelf plus or minus 3 dB at 4 kHz.
  - Self-noise - white or pink at -95 to -75 dBFS.
  - Level - mixture RMS -40 to -15 dBFS (AGC is off in measurement mode).
  - Clipping - 3 % of examples hard-clipped, 3 % soft-clipped (`tanh`), mixture only, never the reference.
  - Handling and wind noise - not covered by any dataset I verified; FSD50K has "Rub", "Scratching" style classes that
    could be checked later (not verified).
  - Optional AGC simulation (10 %) - slow gain changes of plus or minus 6 dB, time constant 0.5 to 2 s, applied to both
    mixture and reference, in case the app ever runs without measurement mode.

### 3.4 Lombard speech

- **Effect** - in noise, talkers raise level, raise F0, lengthen vowels and flatten spectral tilt [27] [H]; level rises
  0.34 to 0.54 dB per dB of noise in real life [30] [H].
- **Data** - the Audio-Visual Lombard Grid corpus has 54 talkers (30 F, 24 M), 5,400 utterances, half Lombard, half
  plain; audio zip 652.6 MB at `https://zenodo.org/records/3736465/files/lombardgrid_audio.zip`; CC BY 4.0 [26] [H]. Its
  sentences are short GRID commands, so use it as a Lombard test set (plain versus Lombard enrollment and target), not
  as bulk training data. EARS has a "loud" reading style but is non-commercial [13] [H].
- **Augmentation (proposal)** - for 30 % of targets and interferers in loud scenes, add +3 to +8 dB of level and a
  +2 to +4 dB high shelf above 1 kHz (spectral flattening). Pitch and duration changes are optional; I found no
  measured magnitudes in the sources I read (not verified).

---

## 4. Speaker encoder for enrollment

### 4.1 Candidates (published numbers)

| Model | Params | VoxCeleb1-O EER (cleaned list) | Weights license | Front-end | Notes |
|---|---|---|---|---|---|
| WeSpeaker ResNet34 [34] | 6.63 M | 0.867 % plain, 0.797 % with large-margin fine-tune (LM), 0.723 % LM + AS-norm | CC BY 4.0 (WeSpeaker docs; one HF card says Apache 2.0) [H] | Kaldi fbank 80 x 25 ms / 10 ms, mean normalization | ONNX provided; the same model family is used by pyannote |
| WeSpeaker CAM++ [34] | 7.18 M | 0.803 % plain, 0.707 % LM | CC BY 4.0 [H] | same | TDNN-based, cheap |
| WeSpeaker ECAPA-TDNN c512 [34] | 6.19 M | 1.069 % plain, 0.878 % LM | CC BY 4.0 [H] | same | |
| SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb) [33] | 20.8 M (3D-Speaker table [35]) | 0.80 % | Apache 2.0 [H] | SpeechBrain `Fbank` (torch.stft) + sentence mean norm | most used ECAPA checkpoint |
| 3D-Speaker CAM++ / ERes2NetV2 [35] | 7.2 M / 17.8 M | 0.65 % / 0.61 % | code Apache 2.0; model license on ModelScope not stated on the page [M] | fbank | |
| NVIDIA TitaNet-Large [36] | about 23 M | 0.66 % | CC BY 4.0 [H] | NeMo mel front-end | trained on VoxCeleb, Fisher, Switchboard, LibriSpeech, SRE; needs NeMo to load |
| ReDimNet B2 [38][34] | 4.89 M | 0.782 % (WeSpeaker reproduction, no LM) | code MIT [H]; weights trained on VoxCeleb2 (and VoxBlink2 for some) | raw waveform in, mel inside | newest family; not tested here |
| Resemblyzer (GE2E d-vector) [37] | small LSTM (not stated) | not reported | Apache 2.0 [H] | 40-mel | used by Look Once to Hear [42] [H] |

Evidence that EER does not decide TSE quality - with a frozen encoder in the same TSE model on Libri2Mix, WeSep got
13.32 dB (ECAPA), 13.18 dB (ResNet34), 12.41 dB (TDNN) and 12.29 dB (CAM++, despite its better EER); training the
encoder jointly with the extractor added 0.55 dB (ECAPA) and 0.78 dB (ResNet34) [41] [H]. `docs/research/prior-art.md`
also reports that in WeSep's newer causal results a single ECAPA vector reached only 6.87 dB while frame-level enrollment
cues reached 13.05 dB [M, via that report] - so the global embedding is one input, not the whole conditioning story.

### 4.2 Measured robustness to short, noisy enrollment

Setup [66] (`bench_spk.py`)

- **Speakers** - LibriSpeech test-clean, 40 speakers that none of these encoders were trained on (they were trained on
  VoxCeleb).
- **Trials** - 440 enrollments (11 per speaker), each scored by cosine against 1 same-speaker and 2 other-speaker clean
  4 s clips, so 1,320 trials per condition. Enrollments are built by concatenating the speaker's other utterances (never
  zero-padded; a first version of the script zero-padded short clips and gave nonsense, so it was fixed and re-run).
- **Noise** - babble of 6 other test-clean speakers plus pink noise, at the stated SNR relative to the enrollment
  speech. "Talker" adds one competing talker 5 dB below the target on top of noise at 10 dB. "Reverb" uses a synthetic
  exponential RIR (RT60 0.6 s, direct-to-reverberant ratio about +1 dB) plus noise at 5 dB.
- **Resolution** - one trial is 0.08 % of a condition, but with only 40 speakers I would not trust differences under
  about 1 % absolute [L].

EER in percent (lower is better)

| Encoder | clean 3 s | clean 5 s | 10 dB, 3 s | 5 dB, 3 s | 0 dB, 3 s | 5 dB, 5 s | talker + 10 dB, 3 s | reverb + 5 dB, 3 s |
|---|---|---|---|---|---|---|---|---|
| WeSpeaker ResNet34 | **0.45** | **0.45** | **2.05** | **4.32** | **8.41** | 2.44 | 5.23 | 8.18 |
| WeSpeaker ResNet34-LM | 0.68 | **0.45** | 2.27 | **4.32** | 10.23 | **2.05** | 5.91 | 9.49 |
| WeSpeaker CAM++ | 0.68 | 0.68 | 2.27 | 4.55 | 11.36 | 2.95 | **4.77** | 9.32 |
| WeSpeaker ECAPA c512 | 1.14 | 0.68 | 2.50 | 4.72 | 12.27 | 2.73 | 6.36 | **7.27** |
| SpeechBrain ECAPA c1024 | 1.76 | 0.91 | 3.18 | 5.45 | 13.41 | 3.64 | 9.32 | 10.17 |
| Resemblyzer GE2E | 3.86 | 3.41 | 10.00 | 16.42 | 27.73 | 15.68 | 18.41 | 23.86 |

Cosine between the embedding of a noisy enrollment and the embedding of the same crop without noise (1.0 means
unaffected) - about 0.80 / 0.68 / 0.49 at 10 / 5 / 0 dB for all four WeSpeaker models, 0.77 / 0.63 / 0.41 for
SpeechBrain ECAPA and 0.76 / 0.68 / 0.60 for Resemblyzer (Resemblyzer drifts less, but its clean separation is about 8x
worse to begin with).

What this means

- **WeSpeaker ResNet34 (plain)** was best or within 0.25 % of the best in 6 of 8 conditions and clearly best at 0 dB.
- **Longer enrollment helps a lot** - 5 s instead of 3 s at 5 dB SNR roughly halved the EER (4.3 to 2.4 %). Ask for
  about 5 s and keep refining the embedding while the target talks.
- **Every encoder struggles at 0 dB and with reverb** (7 to 14 % EER). Enrollment should be captured at close range,
  and the extractor must be trained on noisy enrollments, since its conditioning vector will be this noisy in use.
- **Limits of this test** - read speech, synthetic reverb, 40 speakers. Run it again on EdAcc / VoxPopuli accented
  speakers and real phone enrollments once those exist.

### 4.3 Core ML conversion with the filterbank inside the model

Measured with coremltools 9.0 and PyTorch 2.7.1 on this Mac [66] (`convert_spk_coreml.py`, `debug_coreml*.py`)

- **Front-end as one Conv1d** - DC removal, pre-emphasis 0.97, Hamming window and the 512-point DFT are all linear, so
  they fold into one fixed 514 x 400 kernel with stride 160; then a mel matrix multiply, `log` and mean normalization.
  This matched `torchaudio.compliance.kaldi.fbank` within 8.1e-5 (log units), and the PyTorch ResNet34 with it matched
  WeSpeaker's official ONNX exactly (cosine 1.0). About 40 lines of code.
- **Conversion** - `torch.jit.trace` then `ct.convert(..., convert_to="mlprogram", minimum_deployment_target=iOS17)`
  with a flexible input of 1 to 16 s (`ct.RangeDim`). About 1 s, no custom ops.
- **fp32 package (26 MB)** - cosine to PyTorch 1.0000 with `CPU_ONLY` and `CPU_AND_GPU`, 0.9995 with `ALL`; median
  18 ms per 5 s clip on this Mac.
- **Pure fp16 is broken** - cosine 0.01 to 0.06, because the power spectrum of int16-scaled audio overflows float16.
  Rescaling the input to fit float16 forces a higher log floor on quiet bins, which changed the embedding (cosine 0.74),
  so that is not usable without retraining.
- **Mixed precision works on CPU and GPU only** - front-end in fp32 and the ResNet trunk in fp16 (coremltools
  `FP16ComputePrecision(op_selector=...)`, 16 MB) gave cosine 1.0000 on `CPU_ONLY` (9.2 ms) and `CPU_AND_GPU` (7.7 ms),
  but 0.57 on `ALL` and `CPU_AND_NE`. The Neural Engine path returns wrong embeddings for this model. One in-memory fp32
  run right after conversion also returned a wrong embedding once (min cosine 0.59); I could not reproduce it in three
  later runs.
- **SpeechBrain ECAPA with its stock `torch.stft` front-end also converted** (fixed 5 s input, cosine 1.0), which
  confirms coremltools 9.0's STFT converter [40] works for analysis-only front-ends.

Practical recipe for the app - ship the mixed or fp32 package, set `MLModelConfiguration.computeUnits = .cpuAndGPU`,
and at launch run one stored 5 s clip and compare with its stored embedding (cosine at least 0.999; fall back to
`.cpuOnly` if not). Enrollment runs once per person, so 8 to 18 ms does not matter. All numbers above are on the Mac,
not yet on the iPhone.

Difficulty rating - low for WeSpeaker ResNet34 or CAM++ (front-end above plus the 60-line ResNet definition), low for
SpeechBrain ECAPA (stock front-end converts), medium for TitaNet (needs NeMo to load and a re-implemented NeMo front-end;
not tested), unknown for ReDimNet (not tested).

### 4.4 Recommendation

**WeSpeaker ResNet34, VoxCeleb, plain (not LM)** [34]

- **License** - WeSpeaker states its VoxCeleb models follow CC BY 4.0; this checkpoint's Hugging Face card says
  Apache 2.0 [H]. Both allow commercial use; credit WeSpeaker and VoxCeleb.
- **Accuracy** - published VoxCeleb1-O EER 0.867 % [34] [H]; most robust of 6 encoders on short noisy enrollment in my
  test (section 4.2).
- **Size and conversion** - 6.63 M parameters, 26 MB fp32 or 16 MB mixed Core ML, converts with the filterbank inside
  (section 4.3).
- **Download** - `https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34` (`avg_model.pt` 45 MB, `config.yaml`,
  `voxceleb_resnet34.onnx` 26.5 MB).
- **Runner-up** - WeSpeaker CAM++ (best with a competing talker in my test, cheapest to run) uses the same front-end, so
  it is a 1-line swap for ablation 3 in section 5.9.

Where it plugs in

1. The conditioning vector for the extractor (frozen in v1).
2. The slow causal "is the output still the enrolled voice" gate suggested by NTT/BUT's verification results [43].
3. An enrollment quality check in the UI (agreement of embeddings from successive 1 s windows).
4. Not for grading - evaluate speaker similarity with SpeechBrain ECAPA so we never grade with the conditioning model.

Relation to `docs/research/prior-art.md` - that report recommends a frozen Resemblyzer d-vector as the global anchor
plus frame-level cross-attention cues from the extractor's own encoder. My measurements say ResNet34 is a much stronger
global anchor than Resemblyzer under noise (EER at 5 dB 4.3 % versus 16.4 %), so I suggest ResNet34 for the global
path and keeping the frame-level cue path. A better EER does not guarantee better extraction (WeSep found CAM++'s better
EER gave worse TSE [41]), so ablation 3 decides. If noisy enrollment still hurts, add Look Once to Hear's trick - a
small student network trained with a cosine loss to map noisy enrollment to the clean-enrollment embedding, which cost
them only 0.4 dB versus clean enrollment [42] [H].

---

## 5. Training recipe for a causal TSE model

The recipe is architecture-agnostic; the model itself is decided elsewhere. Numbers marked "proposal" are my choices
built from the cited work, to be tuned on the validation set.

### 5.1 Loss

- **Why not plain SI-SDR** - SI-SDR is scale-invariant [48] [H], so the output level is arbitrary, which is bad for a
  hearing app (level jumps) and makes "output silence" undefined. NTT/BUT switched to a scale-dependent SNR loss for
  exactly this reason [43] [H].
- **Target present** - negative thresholded SNR, `L = -10 log10( |s|^2 / (|s - y|^2 + tau |s|^2) )` with
  `tau = 10^(-30/10)`, so very good examples stop dominating the gradient [47][43] [H].
- **Target absent** - `L = 10 log10( |y|^2 + tau_abs |x|^2 )` where x is the mixture [47][43] [H]. MixIT clamps 30 dB
  below the mixture [47] [H]; NTT/BUT used tau_abs = 1e-2 (20 dB) [43] [H]. Proposal - 1e-2.5 (25 dB), because Earshot
  must go quiet, then check the false-alarm and miss rates.
- **Target silent inside a segment (proposal)** - add the same absent-style term over the frames where the target's
  voice-activity mask is 0, so the model learns to close the gate between the target's sentences, not only on whole
  silent clips.
- **Known trade-off** - training with 10 % absent-target examples taught the model to output silence (about -100 dB
  attenuation on most absent cases) but also produced misses on about 10 % of present-target cases; SDRi fell from
  12.4 to 10.8 dB. A separate verification step (compare the output's speaker embedding to the enrollment) did better,
  EER 8.9 % with 10 s enrollment, about 5 % with 15 s or more [43] [H]. For Earshot this suggests a slow, causal
  "is the output still the enrolled voice" gate on top of the extractor (decided in the app design, not here).
- **Over-suppression of the target** - Microsoft's personalized speech enhancement work found target over-suppression
  is the main practical failure of causal models; an asymmetric loss that penalizes removed target energy (`beta = 1`,
  on power-law compressed spectra, `p = 0.3`) plus ASR multi-task training cut it from 9.5 % to 1.2 % of frames for
  their best model [45] [H]. VoiceFilter-Lite
  used the same idea with an asymmetric L2 penalty (`alpha > 1`) [46] [H]. Proposal - add it as a second-stage
  ablation (`L_total = L_tSNR + 0.3 x L_asym`), not on day one.

The whole v1 loss fits in a few lines (PyTorch, all in float32):

```python
def tsnr_loss(est, ref, mix, present, frame_vad, tau=1e-3, tau_abs=10 ** -2.5, eps=1e-8):
    """est, ref, mix: (B, T) waveforms. present: (B,) bool. frame_vad: (B, T) 1 where the target talks."""
    est, ref, mix = est.float(), ref.float(), mix.float()
    p_ref = ref.pow(2).sum(-1)
    # target present: negative thresholded SNR (clamps at 30 dB)
    l_on = -10 * torch.log10(p_ref / ((ref - est).pow(2).sum(-1) + tau * p_ref + eps) + eps)
    # target absent (whole clip): push output energy down to 25 dB below the mixture
    l_off = 10 * torch.log10(est.pow(2).sum(-1) + tau_abs * mix.pow(2).sum(-1) + eps)
    # target silent inside a present clip: same idea, only on frames where the target is quiet
    quiet = 1.0 - frame_vad
    l_gap = 10 * torch.log10((est * quiet).pow(2).sum(-1) + tau_abs * (mix * quiet).pow(2).sum(-1) + eps)
    has_gap = quiet.sum(-1) > 1600  # at least 0.1 s of target silence at 16 kHz
    loss = torch.where(present, l_on + 0.1 * torch.where(has_gap, l_gap, torch.zeros_like(l_gap)), l_off)
    return loss.mean()
```

The 0.1 weight on the gap term is a starting guess to tune (proposal).

### 5.2 Mixture recipe for loud rooms (proposal, built from 2.1, [41], [43], [45], [46])

| Element | Value |
|---|---|
| Segment length | 4 s (Conv-TasNet used 4 s [49] [H], WeSep 3 s [41] [H]); last 15 % of steps at 6 s so recurrent state sees longer context |
| Target presence | 90 % present, 10 % absent (enrollment from a speaker not in the mix, reference all zeros) [43] |
| Partial target | 25 % of present examples start or stop inside the segment |
| Competing talkers | 0 (10 %), 1 (35 %), 2 (30 %), 3 (25 %); each active over 50 to 100 % of the segment; WeSep used 1/2/3+ speakers with probabilities 0.1/0.7/0.2 and SIR -5 to 10 dB [41] [H] |
| Talker loudness | per-talker level at 1 m N(0, 4) dB, then geometry sets the received level; rescale if the total SIR leaves -8 to +12 dB |
| Own voice | 25 % of examples add a different speaker 0.25 to 0.8 m from the mic, active 20 to 60 % of the segment, 0 to 12 dB louder than the target (the user's mouth is closer); never in the reference |
| Babble | 70 % of examples, 6 to 20 talkers from the pool (not the target, not test speakers), each through its own RIR |
| Noise | 80 % of examples, 1 or 2 clips from DEMAND (30 %), FSD50K restaurant classes (40 %), MUSAN noise (20 %), other (10 %) |
| Music | 20 % of examples, at 5 to 20 dB below the target |
| SNR (target vs babble + noise + music, at the mic) | N(5, 5) dB clipped to -8 to 25 dB; 10 % "quiet room" examples at 20 to 40 dB |
| Rooms | 75 % simulated bank, 15 % OpenSLR 28 real RIRs, 10 % dry |
| Mic chain | section 3.3 |
| Gain | input at its raw level (-40 to -15 dBFS RMS); reference scaled identically |

Enrollment (proposal)

| Element | Value |
|---|---|
| Content | 1 or 2 other utterances of the target speaker, concatenated, 3 to 6 s total |
| Noise | 80 % noisy - babble + noise at 0 to 25 dB SNR; 20 % clean |
| Competing talker | 30 % add one weaker talker 5 to 15 dB below the target (the phone points at the target, others are farther) |
| Room | 50 % same room as the mixture (different position), 30 % another room, 20 % dry |
| Mic chain | same distribution as the mixture |

### 5.3 Optimizer, schedule, steps

| Setting | Value | Basis |
|---|---|---|
| Optimizer | AdamW, peak LR 1e-3 (5e-4 if the model has attention), weight decay 1e-4 | Adam 1e-3 in Conv-TasNet [49] and WeSep [41] [H] |
| Schedule | 2,000 warmup steps, cosine (or exponential) decay to 2.5e-5 | WeSep decays exponentially to 2.5e-5 [41] [H] |
| Gradient clipping | L2 norm 5 | Conv-TasNet [49], WeSep [41] [H] |
| Batch | 8 to 16 segments of 4 s | memory on 24 GB unified RAM; 8 x 4 s TCN step used 6.5 GB of MPS memory (measured) |
| Weights for eval/export | EMA of weights (decay 0.999) and/or average of the last 5 to 10 checkpoints | WeSep averages 10 [41] [H]; EMA is my proposal |
| Validation | 400 fixed mixtures (rendered once with a fixed seed, speaker-disjoint), every 2,000 steps | proposal |
| Steps | set total = 0.9 x (training hours x 3600) / measured seconds per step | proposal |

Step budget example from measured MPS speed - a Conv-TasNet-style causal conv stack (8 blocks, 512 hidden channels,
1 ms hop) with batch 8 x 4 s took 300 ms per training step with bf16 autocast [66]. 36 h of training would be about
430k steps, about 3.5 M examples, more than the 200 epochs x 13,900 mixtures (2.8 M) that NTT/BUT used [43] [H]. A model
twice as heavy still gets more than 1 M examples.

### 5.4 Mixed precision on MPS (measured, PyTorch 2.7.1, macOS 26.5)

| Block (one training step, batch 8) | CPU fp32 | MPS fp32 | MPS fp16 autocast | MPS bf16 autocast |
|---|---|---|---|---|
| Causal TCN, 4 s waveform, 8 blocks x 512 ch | 2,656 to 3,182 ms | 404 to 450 ms | 299 to 306 ms | 301 to 308 ms |
| GRU 2 x 256 over 500 frames | 125 to 135 ms | 657 to 727 ms | 773 to 810 ms | 721 to 767 ms |
| LSTM 2 x 256 over 500 frames | 130 to 140 ms | 51 ms | 51 ms | 51 ms |
| LSTM over 4,000 short sequences (band-axis style) | 93 to 146 ms | 53 to 63 ms | 44 ms | 45 to 49 ms |

- **bf16 autocast works on MPS** and gives about 1.3x on conv-heavy blocks; it does not cover LSTM (LSTM output stays
  float32 under fp16 autocast), so recurrent parts run in fp32 anyway. bf16 needs no GradScaler; fp16 would need one
  (`torch.amp.GradScaler("mps")` constructs in 2.7.1) [66].
- **Keep the loss in fp32** - compute energies and `log10` after `.float()`.
- **MPS and CPU agree** - same weights and input gave max relative output error 1e-6 and gradient error at most 1.3e-4
  for all four blocks, so the old LSTM correctness bugs [52] did not show up in 2.7.1 [66].
- **Upgrading PyTorch does not fix GRU** - in a scratch venv with PyTorch 2.15.0, GRU on MPS improved to 244 ms but was
  still slower than CPU (189 ms) and 4.4x slower than LSTM (56 ms); the conv stack was not faster (368 to 480 ms) [66].
  Stay on 2.7.1 (the project venv's version; the newest release is 2.15.0 from 2026-10-28 [65] [H]). coremltools 9.0
  supports PyTorch 2.7 [40] [H] and warns that 2.7.0 is the newest version it was tested with, so one environment can
  both train and convert.

### 5.5 Data loader design on macOS

- **spawn is the default on macOS** - workers start a new interpreter and receive the dataset by pickling; put the
  training entry point under `if __name__ == "__main__":` and define datasets, `collate_fn` and `worker_init_fn` at
  module top level (no lambdas) [50] [H].
- **Seeding** - each worker's torch seed is `base_seed + worker_id`, but other libraries' seeds may repeat [50] [H]. In
  `worker_init_fn`, build `np.random.default_rng(torch.initial_seed() % 2**32)` and use only that generator.
- **Keep the dataset object small** - store manifests as numpy arrays (paths, offsets, speaker ids), open files and
  memmaps lazily inside `__getitem__` (or `worker_init_fn`), so each spawned worker does not receive a copy of big
  arrays.
- **Workers** - 6 to 8 workers, `persistent_workers=True`, `prefetch_factor=4`, `pin_memory=False`; call
  `torch.set_num_threads(1)` in each worker so numpy/scipy do not oversubscribe the 10 cores.
- **Where the mixing happens** - read crops, apply RIRs and EQ in workers (CPU, measured 0.5 to 1 ms per source
  convolution), or move the convolutions to MPS as one batched FFT (6.6 ms per batch measured). Start on CPU.
- **Do not sync every step** - `.item()` forces a GPU sync; log every 50 steps.
- **Memory** - `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (default 1.7 of the recommended working set) is a hard cap on MPS
  allocations [51] [H]. Set it to about 1.0 so a too-big batch fails fast instead of pushing macOS into swap.
- **Keep the Mac awake** - run training under `caffeinate -dimsu` and on AC power.

### 5.6 Known MPS pitfalls (short list)

- nn.GRU is slow on MPS (above); use nn.LSTM or a hand-written GRU cell [66].
- Set `PYTORCH_ENABLE_MPS_FALLBACK=0` (the default) during development so unsupported ops fail loudly instead of
  silently running on CPU [51] [H].
- Historical LSTM `batch_first` and multi-layer bugs were fixed in 2022/2023 [52] [H]; my agreement test passed [66].
- Complex tensors, `torch.stft` and `torch.istft` work on MPS in 2.7.1 (round-trip error 2.4e-6) [66], but coremltools
  has a `torch.stft` converter and no `istft` converter in 9.0 (read in the installed source, `ops.py`) [40] [H]. Write
  the model's STFT and inverse STFT as real Conv1d / ConvTranspose1d with fixed DFT kernels so it trains and converts the
  same way.
- float64 is not supported on MPS; keep everything float32.

### 5.7 Curriculum (proposal)

First 10 % of steps - SNR shifted up 5 dB and at most 2 competing talkers, no absent targets. Introduce absent targets
and own-voice after that. Last 15 % - 6 s segments. NTT/BUT's finding that absent-target training can cost some
present-target quality [43] is the reason to delay it.

### 5.8 Checks before a long run

- Overfit 50 mixtures to above 25 dB SI-SDRi in under 15 minutes.
- Causality test - change samples after time t and confirm outputs before t do not change.
- Listen to 20 rendered training mixtures, references and enrollments every time the recipe changes.

### 5.9 First ablations worth running (each about 4 to 6 h)

1. 60 s versus 180 s per speaker at a fixed disk budget (more speakers versus more minutes).
2. tau_abs 1e-2 versus 1e-2.5 and absent-target rate 10 versus 20 %.
3. Encoder - WeSpeaker ResNet34 versus CAM++ versus Resemblyzer (prior-art.md's choice), each frozen, plus ResNet34
   with its last block fine-tuned.
4. With and without the asymmetric over-suppression term.

---

## 6. Evaluation

### 6.1 Metrics and tools

| Metric | What it tells us | Tool, license |
|---|---|---|
| SI-SDR and SI-SDRi [48] | separation quality ignoring gain | 10-line numpy function (or torchmetrics, Apache 2.0) |
| SNR and SNRi | quality including gain (matters for a hearing app) | same |
| ESTOI | intelligibility | `pystoi`, MIT [56] [H] |
| PESQ (P.862.2 wideband) | perceived quality, comparable with literature | `pesq` PyPI wrapper is MIT [55] [H], but PESQ itself is covered by copyright and patents; OPTICOM licenses commercial use and the ITU reference code notice limits use [55] [M]. Use for internal R&D numbers only, never ship it |
| DNSMOS P.835 (SIG, BAK, OVRL) and P.808 | no-reference quality, works on real recordings | ONNX models `sig_bak_ovr.onnx` and `model_v8.onnx` in the DNS-Challenge repo, 16 kHz input in 9.01 s windows [53] [H]; repo license CC BY 4.0 [H]; torchmetrics wraps it [54] |
| Personalized DNSMOS (pDNSMOS) | same, but penalizes leftover interfering talkers | `pDNSMOS/sig_bak_ovr.onnx`, `dnsmos_local.py -p` [53] [H] |
| Word error rate | intelligibility of the extracted voice; DNS 2023 ranked by word accuracy plus P.835 [63] [H] | Whisper (MIT) on outputs versus transcripts (LibriSpeech, VoxPopuli, EdAcc, VCTK have transcripts) |
| Speaker similarity | does the output still sound like the target | cosine between output and clean target using a different encoder than the conditioning one (for example SpeechBrain ECAPA) |
| UTMOS (optional) | second no-reference quality opinion | UTMOS22, MIT [64] [H] |
| Speaker-confusion rate | output follows the wrong talker, a known TSE failure caused by ambiguous embeddings [44] [H] | share of present-target items where SI-SDR(output, any interferer) > SI-SDR(output, target); also per 2 s chunk in streaming tests |
| Failure rate | extraction failed | share with SI-SDRi below 1 dB, as in [43] [H] |
| Target over-suppression (TSOS) | target voice removed | share of frames where removed target energy exceeds 0.1 of the reference energy, as defined in [45] [H] |
| Target-absent attenuation | does it go quiet | `10 log10(|y|^2 / |x|^2)` on absent items; report median and the share above -10 dB and -20 dB (false alarms) |
| Present/absent EER | how separable the two cases are | attenuation as the score, as in [43] [H] |

Speaker-confusion and absent-target behaviour should also be measured in streaming form (20 to 60 s simulated
conversations where the target talks 30 to 60 % of the time), because that is how the app runs.

### 6.2 Test sets

| Test set | Why | Build |
|---|---|---|
| Libri2Mix 16 kHz min, test (clean) with the fixed SpeakerBeam enrollment mapping | standard TSE benchmark, used by WeSep [41] [H]; lets us compare with published numbers (WeSep's BSRNN reached 13.3 dB SI-SDR trained on train-100 and 16.6 dB on train-360 [41] [H]; causal versions reported in prior-art.md are 6.9 to 14.2 dB [M]) | LibriMix scripts (MIT) [57] from LibriSpeech test-clean, which is already on disk; the enrollment mapping comes from BUT's SpeakerBeam repo, whose license is evaluation-only [58] [H] - fine for evaluation, do not redistribute; the noisy variant needs WHAM! noise (non-commercial) |
| Earshot restaurant simulation | the real target condition | 2,000 mixtures from held-out speakers, DEMAND SCAFE and WHAM! `tt` noise, MIT IR Survey bar/restaurant RIRs, grid over SNR (-5, 0, 5, 10 dB), 1 to 3 interferers, own voice on/off, in-hand/on-table, enrollment clean/noisy; rendered from a fixed seed |
| Accent sets | works on every accent | EdAcc test (CC BY-SA 4.0) and VoxPopuli `en_accented` test (CC0, 15 L2 accents; one 2.5 GB shard holds 313 speakers, measured) as targets in the same simulation |
| Lombard set | loud-room speaking style | Lombard Grid (CC BY 4.0) targets and enrollments |
| VOiCES subset [61] | real rooms, LibriSpeech played from a loudspeaker on a rotating platform with TV, music or babble distractors, 12 mics | S3 `s3://lab41openaudiocorpus/` (no AWS account needed with `--no-sign-request`), CC BY 4.0 [H]; pull one room with babble |
| DiPCo [62] | real dinner parties, 4 headset mics + 5 far-field devices, 16 kHz | Zenodo 13.4 GB, CDLA-Permissive-1.0 [M]; keep 1 or 2 sessions; headset channels give approximate references and enrollment |
| DNS 2023 personalized test sets [63] | real crowd-sourced recordings with enrollment clips, 1 to 3 m distances | download scripts in the DNS-Challenge repo [11]; check the size before downloading (not verified) |
| REAL-M [59] | real two-talker mixtures from 50 speakers, about 3 h | license not stated in what I read (not verified) |

Comparison targets - Look Once to Hear reported 7.40 dB SI-SNRi with clean enrollment and 7.01 dB with its
noisy-enrollment distillation on its own binaural benchmark, and MOS gains in a 21-person study [42] [H]. Those numbers
are binaural and on different data, so only use them as a rough reference.

### 6.3 Real acoustic recordings without human subjects

Playing speech through loudspeakers in a real room and recording with the target device is how LibriCSS [60], BUT
ReverbDB [22] and VOiCES [61] were made [H for the general method].

Rig (proposal)

- **Hardware** - 3 to 5 small full-range powered speakers, the iPhone 16 Pro on a tripod or held by the operator, a
  phone SPL meter app for level checks.
- **Layout** - "target" speaker at mouth height 1 m from the phone; 2 to 3 "interferer" speakers 1.5 to 3 m away at
  other angles; 1 to 2 speakers playing restaurant noise or babble, pointed at walls for a diffuse field; an "own voice"
  speaker 0.3 m from the phone.
- **Levels** - target about 65 to 70 dBA at 1 m, noise 65 to 75 dBA at the phone (section 2.1).
- **Signals** - held-out LibriSpeech test-clean, VCTK held-out speakers, EdAcc and VoxPopuli accented clips; noise from
  WHAM! `tt` and DEMAND SCAFE.
- **Record each source alone, then all together** - with nothing moving, room acoustics are linear and time-invariant,
  so each solo recording is that source's reference at the phone mic. Compute SI-SDRi, SNRi, ESTOI, PESQ and confusion
  on real acoustics; check that the sum of the solo recordings matches the joint recording (residual below -25 dB)
  before trusting the references.
- **Enrollment** - record a 3 to 5 s "pointing at the target" clip with the real app, with noise playing.
- **Rooms** - a quiet room, a living room, and a real cafe or restaurant at a quiet hour (ask the venue).
- **No-reference checks in live use** - DNSMOS and pDNSMOS on the app's own recordings, plus Whisper WER against the
  played text.

---

## 7. Download and preparation order (about 3 h of downloading)

| Step | Action | Download | Kept on disk |
|---|---|---|---|
| 1 | MIT IR Survey zip, OpenSLR 28 zip (keep `real_rirs_isotropic_noises`, RIRs cut to 1.5 s, first channel, float16) | 1.3 GB | 0.3 GB |
| 2 | Generate the pyroomacoustics bank (4,000 rooms) | 0 | 0.7 GB |
| 3 | DEMAND 8 scenes 16 kHz + SCAFE 48 kHz, keep 4 channels | 1.2 GB | 0.35 GB |
| 4 | MUSAN stream, keep noise + 6 h music | 11.1 GB | 0.9 GB |
| 5 | FSD50K 16 kHz mirror, keep CC0 + CC BY restaurant classes and 4 h music | 6.9 GB | 1.0 GB |
| 6 | LibriTTS-R three train tarballs, streamed | 82 GB | 2.6 GB |
| 7 | VoxPopuli en 2 shards, one at a time | 6.5 GB | 0.9 GB |
| 8 | GLOBE V2 8 shards | 3.6 GB | 0.9 GB |
| 9 | VCTK mic1 files by range request (2 min per speaker is about 1/12 of each speaker's 57 MB of mic1 FLAC) | about 0.6 GB | 0.25 GB |
| 10 | Eval - 1,000 of the 3,000 WHAM! `tt` clips by range request, EdAcc 2 shards, part of one VoxPopuli accented shard, VOiCES subset | about 5 GB | about 1.0 GB |
| 11 | Optional - Emilia-YODAS 1 tar (after the user accepts the terms) | 1.1 GB | 0.4 GB |
| | **Total** | **about 119 GB** | **about 9.3 GB** |

Peak extra disk during any step stays under 4 GB if shards are processed one at a time. Check `df` before each step and
stop if free space would drop below 10 GB (brief rule). With 17 GiB free today, steps 1 to 10 at full size do not fit
(9.3 GB kept + 4 GB peak); run the v1-lite sizes from section 0 unless the cleanup happens first.

---

## 8. Risks and open questions

- **Accent coverage is still thin for some groups** - VoxPopuli covers European English speakers, GLOBE is mostly US,
  Australian, England and Canadian English by row count (measured), VCTK covers UK and a few others. African and South
  and East Asian English are under-represented. Options - more GLOBE shards filtered by accent, AfriSpeech or Svarah
  (licenses not checked in this pass).
- **GLOBE's Adobe enhancement** - may add a "processed" timbre to targets; listen first.
- **Mirror provenance** - the FSD50K 16 kHz parquet mirror is third-party; spot-check 20 clips' licenses against the
  official `dev_clips_info_FSD50K.json` from Zenodo before relying on the `license` column.
- **License conflicts to resolve by reading the files themselves** - DEMAND (CC BY-SA 3.0 vs CC BY 4.0), ACE (CC BY vs
  CC BY-ND), MIT IR Survey (license stated by mirrors, not by the lab page).
- **Non-commercial material used only for evaluation** - WHAM! noise, EARS, the Look Once to Hear code, BUT SpeakerBeam
  enrollment lists. If Earshot is ever sold, re-check whether evaluation use is acceptable or replace them.
- **Real phone mic response** - unknown until we measure it.

---

## 9. Measurement scripts (this document's own evidence)

All in `docs/research/data-training-bench/` with their JSON outputs:

- `bench_decode.py`, `bench_parallel.py` - storage format sizes and decode speed.
- `bench_mps.py` - conv / GRU / LSTM training-step speed on CPU and MPS, autocast, MPS-vs-CPU agreement, op checks.
- `bench_conv.py`, `bench_pra.py` - RIR convolution and pyroomacoustics costs.
- `bench_spk.py` - speaker encoder robustness to short noisy enrollment.
- `convert_spk_coreml.py`, `debug_coreml.py`, `debug_coreml2.py`, `debug_coreml_fp16.py`, `debug_coreml_mixed.py` - speaker
  encoder plus in-graph filterbank to Core ML, parity per compute unit, fp16 and mixed precision.
- `hfsize.py`, `shardstats.py`, `fsd_stats.py`, `remote_zip_ls.py` - dataset sizes, speakers per shard, FSD50K license
  hours, remote zip listings (metadata only, no audio downloaded).

---

## 10. References

1. LibriSpeech, OpenSLR 12 - https://www.openslr.org/12/
2. LibriSpeech `README.TXT`, `SPEAKERS.TXT`, `LICENSE.TXT` (local copy in `data/LibriSpeech/`)
3. LibriTTS-R, OpenSLR 141 - https://www.openslr.org/141/
4. Zen et al., LibriTTS (2019) - https://arxiv.org/abs/1904.02882
5. MLS, OpenSLR 94 - https://www.openslr.org/94/ ; HF `parler-tts/mls_eng` - https://huggingface.co/datasets/parler-tts/mls_eng ; Pratap et al. (2020) - https://arxiv.org/abs/2012.03411
6. CSTR VCTK 0.92, Edinburgh DataShare - https://datashare.ed.ac.uk/handle/10283/3443 (README read via the DSpace API)
7. Mozilla Common Voice notice on HF - https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 ; Mozilla Foundation HF README - https://huggingface.co/spaces/mozilla-foundation/README/blob/main/README.md
8. GLOBE paper (Wang et al., 2024) - https://arxiv.org/abs/2406.14875 ; GLOBE V2 - https://huggingface.co/datasets/MushanW/GLOBE_V2 ; GLOBE V3 - https://huggingface.co/datasets/MushanW/GLOBE_V3
9. VoxPopuli on HF - https://huggingface.co/datasets/facebook/voxpopuli
10. AMI license - https://groups.inf.ed.ac.uk/ami/corpus/license.shtml ; HF AMI - https://huggingface.co/datasets/edinburghcstr/ami
11. Microsoft DNS Challenge repo - https://github.com/microsoft/DNS-Challenge ; scripts `download-dns-challenge-5-headset-training.sh`, `download-dns-challenge-5-noise-ir.sh`
12. Emilia dataset card - https://huggingface.co/datasets/amphion/Emilia-Dataset ; He et al., Emilia (2025) - https://arxiv.org/abs/2501.15907
13. EARS - https://github.com/facebookresearch/ears_dataset
14. Expresso - https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset
15. EdAcc - https://huggingface.co/datasets/edinburghcstr/edacc
16. DEMAND, Zenodo 1227121 - https://zenodo.org/records/1227121 (and its DEMAND.pdf)
17. MUSAN, OpenSLR 17 - https://www.openslr.org/17/ ; Snyder et al. (2015) - https://arxiv.org/abs/1510.08484
18. FSD50K, Zenodo 4060432 - https://zenodo.org/records/4060432 ; 16 kHz mirror - https://huggingface.co/datasets/quinnlue/FSD50K-16k
19. WHAM! - Wichern et al. (2019) https://arxiv.org/abs/1907.01160 ; audEERING listing https://audeering.github.io/datasets/datasets/wham.html ; lhotse recipe (download URL) https://lhotse.readthedocs.io/en/latest/_modules/lhotse/recipes/wham.html
20. OpenSLR 26 - https://www.openslr.org/26/ ; OpenSLR 28 - https://www.openslr.org/28/
21. MIT IR Survey - https://mcdermottlab.mit.edu/Reverb/IR_Survey.html ; Traer and McDermott, PNAS 2016, doi:10.1073/pnas.1612524113 ; license as stated by https://huggingface.co/datasets/benjamin-paine/mit-impulse-response-survey and https://siliconlabs.github.io/mltk/_modules/mltk/datasets/audio/mit_ir_survey.html
22. BUT ReverbDB - https://speech.fit.vut.cz/software/but-speech-fit-reverb-database
23. ACE corpus - https://zenodo.org/records/6257551 ; https://www.imperial.ac.uk/speech-audio-processing/projects/ace-challenge/
24. pyroomacoustics docs - https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html ; repo (MIT) - https://github.com/LCAV/pyroomacoustics
25. Luo and Yu, FRA-RIR (Interspeech 2023) - https://arxiv.org/abs/2208.04101 ; code (Apache 2.0) - https://github.com/tencent-ailab/FRA-RIR
26. Audio-Visual Lombard Grid corpus - https://zenodo.org/records/3736465 ; Alghamdi et al., JASA 2018 - https://eprints.whiterose.ac.uk/131924/
27. Lu and Cooke, JASA 124(5) 2008, doi:10.1121/1.2990705
28. Apple, AVAudioSession.Mode.measurement - https://developer.apple.com/documentation/avfaudio/avaudiosession/mode-swift.struct/measurement
29. Bradley, Sato and Picard, "On the importance of early reflections for speech in rooms", JASA 2003 - https://labs.sonicfield.org/library/on-the-importance-of-early-reflections-for-speech-in-rooms
30. Wu et al., Ear and Hearing 2018 - https://pmc.ncbi.nlm.nih.gov/articles/PMC5824438/
31. Weisser and Buchholz, JASA 145(1) 2019, doi:10.1121/1.5087567
32. To and Chung, "Restaurant noise - levels and temporal characteristics" (2015) - https://research.mpu.edu.mo/en/publications/restaurant-noise-levels-and-temporal-characteristics/ (numbers via search summary)
33. SpeechBrain ECAPA - https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
34. WeSpeaker pretrained models - https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md ; results - https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md
35. 3D-Speaker - https://github.com/modelscope/3D-Speaker
36. TitaNet-Large - https://huggingface.co/nvidia/speakerverification_en_titanet_large
37. Resemblyzer - https://github.com/resemble-ai/Resemblyzer
38. ReDimNet - https://github.com/IDRnD/ReDimNet
39. VoxCeleb - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ ; KAIST mirror license text - https://mm.kaist.ac.kr/datasets/voxceleb/files/license.txt
40. coremltools releases - https://github.com/apple/coremltools/releases ; installed source `coremltools/converters/mil/frontend/torch/ops.py` (stft converter) and `.../passes/defs/lower_complex_dialect_ops.py`
41. Wang et al., WeSep (Interspeech 2024) - https://arxiv.org/abs/2409.15799 ; configs - https://github.com/wenet-e2e/wesep
42. Veluri et al., Look Once to Hear (CHI 2024) - https://arxiv.org/abs/2405.06289 ; code and license - https://github.com/vb000/LookOnceToHear
43. Delcroix et al., "Listen only to me!" (Interspeech 2022) - https://arxiv.org/abs/2204.04811
44. Zhao et al., "Target confusion in end-to-end speaker extraction" (Interspeech 2022) - https://arxiv.org/abs/2204.01355
45. Eskimez et al., "Personalized speech enhancement - new models and comprehensive evaluation" - https://arxiv.org/abs/2110.09625
46. Wang et al., VoiceFilter-Lite (Interspeech 2020) - https://arxiv.org/abs/2009.04323
47. Wisdom et al., MixIT (NeurIPS 2020) - https://arxiv.org/abs/2006.12701
48. Le Roux et al., "SDR - half-baked or well done?" (ICASSP 2019) - https://arxiv.org/abs/1811.02508
49. Luo and Mesgarani, Conv-TasNet - https://arxiv.org/abs/1809.07454
50. PyTorch data loading docs - https://docs.pytorch.org/docs/2.15/data.html
51. PyTorch MPS environment variables - https://docs.pytorch.org/docs/2.15/mps_environment_variables.html
52. PyTorch MPS LSTM issues and fixes - https://github.com/pytorch/pytorch/issues/80306 , https://github.com/pytorch/pytorch/issues/83144 , https://github.com/pytorch/pytorch/pull/95137
53. DNSMOS in the DNS-Challenge repo - https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
54. torchmetrics DNSMOS - https://lightning.ai/docs/torchmetrics/stable/audio/deep_noise_suppression_mean_opinion_score.html
55. `pesq` Python wrapper - https://github.com/ludlows/PESQ ; PESQ licensing discussion (OPTICOM) - https://auditory.org/mhonarc/2013/msg00482.html
56. pystoi - https://github.com/mpariente/pystoi
57. LibriMix - https://github.com/JorisCos/LibriMix
58. BUT SpeakerBeam (evaluation license) - https://github.com/BUTSpeechFIT/speakerbeam
59. Subakan et al., REAL-M - https://arxiv.org/abs/2110.10812
60. Chen et al., LibriCSS - https://arxiv.org/abs/2001.11482 ; https://github.com/chenzhuo1011/libri_css
61. VOiCES on AWS Open Data - https://registry.opendata.aws/lab41-sri-voices/
62. DiPCo - https://zenodo.org/record/8122551 ; https://arxiv.org/abs/1909.13447
63. Dubey et al., ICASSP 2023 DNS Challenge - https://arxiv.org/abs/2303.11510
64. UTMOS22 (MIT) - https://github.com/sarulab-speech/UTMOS22
65. PyPI release history for torch (2.15.0 on 2026-10-28), coremltools (9.0), torchaudio (2.11.1) - https://pypi.org/project/torch/ , https://pypi.org/project/coremltools/
66. Local measurements on this M4 MacBook Pro, 2026-11-06 - `docs/research/data-training-bench/`
67. Commercial Acoustics, restaurant reverberation case study (vendor page, seen only as a search result) - https://commercial-acoustics.com/case-studies/restaurant-reverberation-time-reduction/
