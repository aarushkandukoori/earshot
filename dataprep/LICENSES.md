# Earshot data pack - licenses and allowed use

Every source named in the data plan (`docs/research/data-training.md`), with its license, where it comes from, the
attribution to show in the app credits or a paper, and what Earshot may do with it. Checked 2026-11-07.

**Evaluation only, never training** - WHAM! noise is CC BY-NC 4.0 (non-commercial). Its manifest lines carry
`"split": "test"` and `"use": ["eval"]`, and `python -m dataprep.verify` fails if any non-commercial or
evaluation-only license appears on a train or val line. If Earshot is ever sold, re-check whether evaluating on
non-commercial data is acceptable, or drop WHAM! from the eval pack. The other eval sets (LibriSpeech test-clean,
EdAcc, VoxPopuli en_accented, DiPCo) allow commercial use; they are held out by design, not by license.

**Not fetched on purpose** - the BUT/NTT SpeakerBeam enrollment list (a click-through software evaluation agreement
that binds whoever accesses it; only the user can accept it), Emilia-YODAS (gated, the user must accept terms on
Hugging Face), VOiCES (only as multi-GB tar.gz archives, see `README.md`).

## License ids used in manifests

`verify.py` reads this table. A manifest or RIR-bank line whose `license` is not in the first column fails the check.

| id | meaning | commercial use | modified copies | Earshot use |
|---|---|---|---|---|
| `CC0-1.0` | Creative Commons Zero, public domain dedication | yes | yes | train and eval |
| `PD` | public domain (for example US government works in MUSAN) | yes | yes | train and eval |
| `CC-BY-2.0` | Creative Commons Attribution 2.0 | yes, with attribution | yes | train and eval |
| `CC-BY-2.5` | Creative Commons Attribution 2.5 | yes, with attribution | yes | train and eval |
| `CC-BY-3.0` | Creative Commons Attribution 3.0 | yes, with attribution | yes | train and eval |
| `CC-BY-4.0` | Creative Commons Attribution 4.0 | yes, with attribution | yes | train and eval |
| `CC-BY` | Creative Commons Attribution, version not stated in the source | yes, with attribution | yes | train and eval |
| `CC-BY-SA-1.0` | Attribution-ShareAlike 1.0 (2 MUSAN hd-classical tracks) | yes, with attribution | yes, shared alike | train and eval |
| `CC-BY-SA-2.0` | Attribution-ShareAlike 2.0 | yes, with attribution | yes, shared alike | train and eval |
| `CC-BY-SA-2.5` | Attribution-ShareAlike 2.5 | yes, with attribution | yes, shared alike | train and eval |
| `CC-BY-SA-3.0` | Attribution-ShareAlike 3.0 (DEMAND) | yes, with attribution | yes, shared alike | train and eval |
| `CC-BY-SA-4.0` | Attribution-ShareAlike 4.0 (EdAcc) | yes, with attribution | yes, shared alike | train and eval |
| `CC-BY-SA` | Attribution-ShareAlike, version not stated | yes, with attribution | yes, shared alike | train and eval |
| `Apache-2.0` | Apache License 2.0 (OpenSLR 28 RIRs) | yes | yes, keep the license and NOTICE | train and eval |
| `MIT` | MIT License (LibriMix metadata, pyroomacoustics code) | yes | yes, keep the copyright notice | train and eval |
| `CDLA-Permissive-1.0` | Community Data License Agreement, Permissive 1.0 (DiPCo) | yes | yes, include the license text when sharing data | train and eval |
| `generated` | made by Earshot's own code (pyroomacoustics rooms), no third-party data | yes | yes | train and eval |
| `CC-BY-NC-4.0` | Attribution-NonCommercial 4.0 (WHAM! noise) | **no** | yes | **eval only** |
| `BUT-NTT-EVAL` | BUT/NTT software evaluation agreement (SpeakerBeam lists) - not in the pack | **no** | **no** | **eval only, and only if the user accepts the agreement** |

ShareAlike means that if we publish modified copies of those files (for example a rendered test set), they must carry
the same license. Using them to train or test a model and shipping the model is not a modified copy of the files
(this is the common reading; it is not legal advice).

## Speech (training pack)

| Source | License | URL | Attribution | Allowed use in Earshot |
|---|---|---|---|---|
| LibriTTS-R train (all three subsets) | `CC-BY-4.0` | https://www.openslr.org/141/ | LibriTTS-R, Y. Koizumi et al., Interspeech 2023, derived from LibriTTS (H. Zen et al., Interspeech 2019) and LibriSpeech (V. Panayotov et al., ICASSP 2015) | train - targets, interferers, babble, enrollment. Downloaded from OpenSLR, not the Hugging Face mirror whose card points at a CC BY-NC-ND paper license |
| VoxPopuli en train shards | `CC0-1.0` | https://huggingface.co/datasets/facebook/voxpopuli , https://github.com/facebookresearch/voxpopuli | VoxPopuli, C. Wang et al., ACL 2021; source audio from the European Parliament (see its legal notice, https://www.europarl.europa.eu/legal-notice/en/) | train |
| GLOBE V2 | `CC0-1.0` | https://huggingface.co/datasets/MushanW/GLOBE_V2 | GLOBE, W. Wang et al., Interspeech 2024, built from Mozilla Common Voice (CC0) | train |
| CSTR VCTK 0.92 | `CC-BY-4.0` | https://datashare.ed.ac.uk/handle/10283/3443 | CSTR VCTK Corpus 0.92, J. Yamagishi, C. Veaux, K. MacDonald, University of Edinburgh, 2019 | train; the held-out VCTK speakers are test only |
| Emilia-YODAS EN (optional) | `CC-BY-4.0`, gated | https://huggingface.co/datasets/amphion/Emilia-Dataset | Emilia, H. He et al., 2024/2025 | train, only after the user accepts the terms on Hugging Face; not in the pack today |
| MLS English | CC BY 4.0 | https://www.openslr.org/94/ | Pratap et al., Interspeech 2020 | not used (too many shards per speaker) |
| Mozilla Common Voice | CC0, gated through Mozilla Data Collective since 2025 | https://datacollective.mozillafoundation.org | Mozilla Common Voice | not used (GLOBE instead) |
| AMI headset | CC BY 4.0 | https://groups.inf.ed.ac.uk/ami/ | AMI Meeting Corpus | not used in v1 |
| DNS Challenge 5 clean speech | original terms of each source | https://github.com/microsoft/DNS-Challenge | Microsoft DNS Challenge | not used (repackaged LibriVox and VCTK) |
| Emilia (original) | CC BY-NC 4.0, gated | https://huggingface.co/datasets/amphion/Emilia-Dataset | Emilia, H. He et al. | not used (non-commercial) |
| EARS | CC BY-NC 4.0 | https://github.com/facebookresearch/ears_dataset | J. Richter et al., Interspeech 2024 | not used (non-commercial) |
| Expresso | CC BY-NC 4.0 | https://github.com/facebookresearch/textlesslib | T. A. Nguyen et al., Interspeech 2023 | not used (non-commercial, 4 speakers) |
| VoxCeleb | wording differs between mirrors, research-only on some | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ | Nagrani et al. | not used for TSE data (the speaker encoder was pretrained on it, see `fact-check.md`) |

## Noise and music

| Source | License | URL | Attribution | Allowed use in Earshot |
|---|---|---|---|---|
| FSD50K, CC0 and CC BY clips only | per clip `CC0-1.0`, `CC-BY-3.0`, `CC-BY-4.0`; dataset CC BY 4.0. CC BY-NC and Sampling+ clips are dropped | https://zenodo.org/records/4060432 ; 16 kHz mirror https://huggingface.co/datasets/quinnlue/FSD50K-16k | FSD50K, E. Fonseca et al., IEEE/ACM TASLP 2022; each CC BY clip also needs its Freesound uploader and title (kept in the manifest's `extra`) | train, val, test noise and music |
| DEMAND | `CC-BY-SA-3.0` (the bundled PDF; the Zenodo page says CC BY 4.0, we use the stricter one) | https://zenodo.org/records/1227121 | DEMAND, J. Thiemann, N. Ito, E. Vincent, 2013 | train on 8 scenes; SCAFE held out as test |
| MUSAN noise and music | CC BY 4.0 overall; per-file licenses from each folder's LICENSE, only files that allow commercial use and modification are kept (`CC0-1.0`, `PD`, `CC-BY-*`, `CC-BY-SA-*`) | https://www.openslr.org/17/ | MUSAN, D. Snyder, G. Chen, D. Povey, arXiv 1510.08484, 2015 | train noise and music |
| **WHAM! noise, tt split** | `CC-BY-NC-4.0` | http://wham.whisper.ai/ ; zip at https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip | WHAM!, G. Wichern et al., Interspeech 2019 | **evaluation only** (Libri2Mix noisy test, restaurant simulation test). Never train on it |
| DNS Challenge noise (optional) | AudioSet clips CC BY 4.0, Freesound clips CC0 | https://github.com/microsoft/DNS-Challenge | Dubey et al., ICASSP 2023 | not in v1 |
| ACE corpus (optional) | CC BY 4.0 on Zenodo, CC BY-ND 4.0 on Imperial's page (conflict, treat as ND) | https://zenodo.org/records/6257551 | J. Eaton et al., WASPAA 2015 | not in v1; test only if ever added |

## Room impulse responses

| Source | License | URL | Attribution | Allowed use in Earshot |
|---|---|---|---|---|
| Simulated rooms (pyroomacoustics image source model) | `generated` (our data); the tool is MIT | https://github.com/LCAV/pyroomacoustics | pyroomacoustics, R. Scheibler, E. Bezzam, I. Dokmanic, ICASSP 2018 | train, val, test banks from separate seeds |
| OpenSLR 28 RIRS_NOISES, real RIRs | `Apache-2.0` | https://www.openslr.org/28/ | T. Ko et al., ICASSP 2017 (RWCP, REVERB 2014 and Aachen AIR inside) | train |
| MIT IR Survey | `CC-BY-4.0` as stated by mirrors; the lab page states no license | https://mcdermottlab.mit.edu/Reverb/IR_Survey.html | J. Traer and J. H. McDermott, PNAS 2016 | test only (held-out real restaurants and bars) |
| OpenSLR 26 simulated RIRs (optional) | Apache 2.0 | https://www.openslr.org/26/ | T. Ko et al., ICASSP 2017 | not in v1 |
| BUT ReverbDB (optional) | CC BY 4.0 | https://speech.fit.vut.cz/software/but-speech-fit-reverb-database | I. Szoke et al., IEEE JSTSP 2019 | not in v1 |

## Evaluation pack (data/eval)

| Source | License | URL | Attribution | Allowed use in Earshot |
|---|---|---|---|---|
| LibriSpeech test-clean | `CC-BY-4.0` | https://www.openslr.org/12/ | LibriSpeech, V. Panayotov et al., ICASSP 2015 | eval (held out; its 40 speakers must not appear in LibriTTS-R train, `verify.py` checks) |
| LibriMix metadata (Libri2Mix test CSVs) | `MIT` (copy of the license in `data/eval/librimix/LICENSE`) | https://github.com/JorisCos/LibriMix | LibriMix, J. Cosentino et al., arXiv 2005.11262, 2020 | eval definitions (`data/eval/libri2mix_test.jsonl`) |
| Libri2Mix enrollment list "earshot-v1" | ours, derived from LibriSpeech names only | this repo, `dataprep/eval_libri2mix.py` | none needed | eval; numbers are close to, not identical with, papers that use the SpeakerBeam list |
| SpeakerBeam enrollment list (BUT/NTT) | `BUT-NTT-EVAL` - software evaluation agreement, internal testing only, no redistribution, Japanese law, indemnity | https://github.com/BUTSpeechFIT/speakerbeam (file `egs/libri2mix/data/wav8k/min/test/map_mixture2enrollment`) | Delcroix et al. | **not fetched**; accessing it accepts the agreement, which is the user's decision |
| **WHAM! noise tt** | `CC-BY-NC-4.0` | see Noise | WHAM!, G. Wichern et al., Interspeech 2019 | **evaluation only** |
| EdAcc test | `CC-BY-SA-4.0` | https://huggingface.co/datasets/edinburghcstr/edacc | EdAcc, R. Sanabria et al., ICASSP 2023 | eval (accents, conversational speech) |
| VoxPopuli en_accented test | `CC0-1.0` | https://huggingface.co/datasets/facebook/voxpopuli | VoxPopuli, C. Wang et al., ACL 2021 | eval (15 L2 accents; only speakers absent from VoxPopuli train and validation) |
| DiPCo (Dinner Party Corpus), partial | `CDLA-Permissive-1.0` (text in `data/eval/dipco/LICENSE.txt`) | https://zenodo.org/records/8122551 | DiPCo, M. Van Segbroeck et al., Interspeech 2020 (Amazon) | eval (real dinner-party recordings with music) |
| VOiCES | CC BY 4.0 | https://registry.opendata.aws/lab41-sri-voices/ | VOiCES, C. Richey et al., Interspeech 2018 (SRI International and Lab41) | **not fetched** - archive-only distribution, see `README.md` |
| Audio-Visual Lombard Grid (planned) | CC BY 4.0 | https://zenodo.org/records/3736465 | N. Alghamdi et al., JASA 2018 | planned Lombard test set (plan section 6.2); not in pack v1 |
| DNS 2023 personalized test sets (planned) | per the DNS-Challenge repo | https://github.com/microsoft/DNS-Challenge | Dubey et al., ICASSP 2023 | not fetched (size not checked) |
| REAL-M | not stated in what we read | https://arxiv.org/abs/2110.10812 | Subakan et al. | not used (license unknown) |

## Credits text for the app (training data only)

Speech from LibriTTS-R (Koizumi et al., CC BY 4.0), VCTK (Yamagishi, Veaux, MacDonald, CC BY 4.0), VoxPopuli
(Wang et al., CC0) and GLOBE (Wang et al., CC0, from Mozilla Common Voice). Noise from FSD50K (Fonseca et al.; CC BY
clips credited to their Freesound uploaders), DEMAND (Thiemann, Ito, Vincent, CC BY-SA 3.0) and MUSAN (Snyder, Chen,
Povey). Real room impulse responses from OpenSLR 28 (Ko et al., Apache 2.0).
