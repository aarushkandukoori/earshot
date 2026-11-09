# Earshot data pack (dataprep)

Code that downloads, converts and checks every dataset Earshot's target-speaker extraction model trains and is
tested on, following the approved plan in `docs/research/data-training.md` (pack v1, section 0). Everything lands
under `data/` (gitignored) as 16 kHz mono 16-bit FLAC, one file per utterance or clip, described by JSON Lines
manifests in `data/manifests/`. Nothing here needs an account or a gated dataset.

## Sizes

Plan targets (section 0) - speech about 5.0 GB, noise and music about 2.0 GB, RIRs about 1.0 GB, evaluation about
1.0 GB, about 9.3 GB in all. `data/manifests/SUMMARY.md` (written by `verify.py`) always has the measured numbers
per source and split.

Evaluation pack, measured 2026-11-07 (budget 1.2 GB under `data/eval`; LibriSpeech test-clean was already on disk):

| Set | Files | Speakers | Hours | GB | License |
|---|---|---|---|---|---|
| LibriSpeech test-clean (`data/LibriSpeech/test-clean`, not copied) | 2,620 | 40 | 5.40 | 0.367 | CC BY 4.0 |
| Libri2Mix test "min" definitions (`data/eval/libri2mix_test.jsonl`) | 3,000 mixtures | 40 | 4.19 in min mode | 0.003 | MIT definitions, CC BY 4.0 + CC BY-NC 4.0 audio |
| WHAM! noise tt, channel 0 (`data/eval/wham_tt`) | 2,127 | - | 6.81 | 0.448 | CC BY-NC 4.0, **eval only** |
| EdAcc test segments (`data/eval/edacc`) | 1,382 | 60 (24 accents) | 2.76 | 0.156 | CC BY-SA 4.0 |
| VoxPopuli en_accented test, unseen speakers (`data/eval/voxpopuli_accented`) | 476 | 81 (13 accents) | 1.66 | 0.114 | CC0 |
| DiPCo dinner-party recordings, partial (`data/eval/dipco`) | see SUMMARY.md | 32 in the transcripts | see SUMMARY.md | under 0.3 | CDLA-Permissive-1.0 |
| VOiCES | not fetched, see below | | | | CC BY 4.0 |

## Fetch

Everything, one step after another (they share about 11 MB/s of download):

```bash
bash dataprep/fetch_all.sh
# or overnight, surviving a closed terminal:
nohup bash dataprep/fetch_all.sh > /dev/null 2>&1 &
```

One part at a time, from the project root:

| Part | Command |
|---|---|
| speech | `.venv/bin/python -m dataprep.speech_fetch --source all` |
| noise and music | `.venv/bin/python -m dataprep.noise_fetch --source all` |
| RIR banks | `.venv/bin/python -m dataprep.rir_build --sim --real` |
| evaluation | `.venv/bin/python -m dataprep.eval_fetch` (or `--only librispeech wham libri2mix edacc voxpopuli dipco`) |
| check | `.venv/bin/python -m dataprep.verify` |

Rules every fetcher follows -
- Resumable - finished files are skipped (`common.flac_ok` plus the manifest), so after a crash, a reboot or a stop
  just run the same command again.
- Streaming - archives are read remotely (HTTP range requests into zips and parquet files, or one pass over a
  tar.gz) and only the kept files reach the disk. No multi-GB temporary files.
- Network errors are retried with exponential backoff.
- Disk floor - when free space drops below 12 GB a fetcher stops with a clear log line (eval_fetch logs
  `STOP disk floor` and exits with code 3). Free space, then re-run.
- Logs - `data/logs/<fetcher>.log`; `fetch_all.sh` also writes `data/logs/fetch_all.log` and
  `data/logs/fetch_all.<step>.out`.
- Stopping - `kill -INT <pid>` or `kill -TERM <pid>` stops eval_fetch cleanly even when it runs under nohup.

## Verify

```bash
.venv/bin/python -m dataprep.verify          # about 10 s
.venv/bin/python -m dataprep.verify --full   # decode every file
```

It reads every `data/manifests/*.jsonl` and the RIR banks in `data/rir/`, prints and writes
`data/manifests/SUMMARY.md` (files, speakers, hours and GB per source and split), and exits with code 1 on any
violation -
- a manifest line that does not parse or breaks the schema (missing field, sr not 16000, unknown split or use);
- a missing file, or a decoded file that is not 16 kHz mono or whose length differs from the manifest (every eval or
  test file plus a fixed 10 % sample of the rest, or all with `--full`);
- a speaker in two splits, or an eval/test speaker in any train or val manifest. Ids are compared source-aware, and
  LibriSpeech, LibriTTS and LibriTTS-R ids are treated as the same person (`librispeech:1089` = `librittsr:1089`),
  because they share LibriVox reader numbers;
- a license that is not listed in `LICENSES.md`, or a non-commercial or evaluation-only license on anything but a
  `"split": "test"`, `"use": ["eval"]` line;
- an eval definition file (`libri2mix_test.jsonl`, `dipco_segments.jsonl`) that points at a missing file;
- an RIR bank that does not load, holds NaN or Inf, or has a different number of rows than its `.jsonl`.

A manifest line that is cut off at the very end of a file (a fetcher still writing) is only a warning.

## Layout

```
data/
  LibriSpeech/test-clean/     LibriSpeech test-clean as shipped (16 kHz FLAC), used in place
  speech/                     training speech (speech_fetch)
  noise/                      noise and music (noise_fetch)
  rir/                        RIR banks, float16 .npy + one .jsonl line per row (rir_build, see data/rir/README.md)
  eval/                       evaluation pack (eval_fetch)
    librimix/                 LibriMix metadata CSVs and their MIT LICENSE
    libri2mix_test.jsonl      Libri2Mix test definitions
    wham_tt/                  WHAM! tt clips used by Libri2Mix test, plus WHAM!'s noise_meta_tt.csv
    edacc/<speaker>/          EdAcc segments; test_metadata.json caches the parquet metadata
    voxpopuli_accented/<accent>/<speaker>/
    dipco/                    far-field and close-talk FLACs, transcriptions/, README.md, LICENSE.txt
    dipco_segments.jsonl      one line per transcribed DiPCo utterance of the kept sessions
  manifests/                  *.jsonl manifests and SUMMARY.md
  logs/
```

## Manifest schema

The schema is defined in the docstring of `dataprep/common.py`; build lines with `common.entry()` and write them with
`common.ManifestWriter`. In short, one line per audio file -
- `path` - relative to `data/`;
- `source` - corpus short name; `speaker` - `"<source>:<id>"` (globally unique) or null for noise;
- `gender` (`F`, `M` or null), `accent`, `group` (chapter, session, conversation or recording);
- `duration` in seconds, `sr` always 16000;
- `split` - `train`, `val` (3 % of speakers, `common.split_for`) or `test`;
- `license` - one of the ids in `LICENSES.md`;
- `use` - any of target, interferer, babble, enrollment, noise, music, eval;
- `extra` - source-specific fields (transcript, original sample rate, Hugging Face row, WHAM! recording and so on).

Every evaluation line has `"split": "test"` and `"use": ["eval"]`, so a training loader that selects by use or split
never loads it.

## Evaluation pack, set by set

**LibriSpeech test-clean** - `eval_librispeech_test.jsonl`, the 2,620 files of `data/LibriSpeech/test-clean` with
transcripts in `extra.text`. Nothing converted.

**Libri2Mix test, 16 kHz, min** - `data/eval/libri2mix_test.jsonl` holds 3,000 definitions built from the official
LibriMix metadata (`JorisCos/LibriMix`, commit b898ca3, MIT): two LibriSpeech test-clean sources with their LibriMix
gains and offsets (always 0), the WHAM! clip with its gain, `length` (min of the two sources) and `max_length`, and
one enrollment utterance per source. Render with

```python
import json
from dataprep.common import DATA
from dataprep.eval_libri2mix import render

defs = [json.loads(line) for line in open(DATA / "eval/libri2mix_test.jsonl")]
mix, (s1, s2), noise = render(defs[0], DATA)                  # mix = s1 + s2 + noise ("mix_both")
mix_clean, (s1, s2), _ = render(defs[0], DATA, noisy=False)    # "mix_clean"
```

`render` follows LibriMix's `create_librimix_from_metadata.py` exactly (channel 0 of the WHAM! clip, its Hann
cross-fade extension when the clip is shorter than the longer source, then everything cut to the shorter source).
Our stored noise is 16-bit FLAC, LibriMix reads the original float WAV, so renders differ from the official files by
16-bit rounding only.

Enrollment - the literature uses BUT/NTT's SpeakerBeam mixture-to-enrollment list. That list ships with a click-through
software evaluation agreement (accessing the files accepts it; internal evaluation only, no redistribution,
indemnity, arbitration in Tokyo), which only the user can accept, so it is not fetched. Instead each source gets a
deterministic enrollment, rule `earshot-v1` - another LibriSpeech test-clean utterance of the same speaker, never one
of the mixture's own sources, at least 3 s long when the speaker has one, chosen by the smallest
`sha1("<mixture id>|<target utterance>|<candidate>")`. Results are comparable in spirit with published Libri2Mix TSE
numbers, not digit for digit. If the user accepts the BUT/NTT agreement, its list
(`egs/libri2mix/data/wav8k/min/test/map_mixture2enrollment` in `BUTSpeechFIT/speakerbeam`) can be converted into the
same `enrollment` field.

**WHAM! noise tt (non-commercial, evaluation only)** - the 2,127 distinct tt clips Libri2Mix test uses, read out of
the remote 18.2 GB `wham_noise.zip` with range requests (1.5 GB downloaded), channel 0, full length, 16 kHz.
`group` is WHAM!'s recording id ("File ID"), `extra` has the location, day and reverberation level from
`noise_meta_tt.csv`, and `extra.scale` records any gain change (none was needed; every clip peaked below full scale).
Also usable as held-out restaurant noise for the Earshot restaurant simulation (plan section 6.2), never for training.

**EdAcc test** - 1,382 segments of all 60 test speakers (30 Zoom conversations, 24 accent labels), 16 kHz from the
original 32 kHz. Instead of 1 or 2 whole shards (which would hold only 16 speakers) we read the parquet metadata of
all 10 test shards and fetch only the 30 row groups that give every speaker at least 20 segments (about 1.4 GB of
download, similar to 2 to 3 shards). Kept - segments of 1 to 30 s that are not marked `IGNORE_TIME_SEGMENT_IN_SCORING`,
up to 180 s per speaker, in corpus order. `group` is the conversation, so target and enrollment pairs can come from the
same speaker and interferers from other conversations. Transcripts, L1 and the speakers' own accent descriptions are in
`extra`.

**VoxPopuli en_accented test** - 476 segments, 81 speakers, 13 accents (en_cs, de, es, fi, fr, hr, hu, it, lt, nl, pl,
ro, sl). We measured that 238 of the 364 en_accented speakers also speak in VoxPopuli's English train shards, and 27
in its validation split (the speaker ids match across configs). Our training pack uses VoxPopuli train shards, so only
speakers absent from all 30 English train shards and the validation split are kept, each with at least 2 segments and
at most 300 s. That leaves about 1.7 h instead of the planned 2 h, and no Estonian (en_et) or Slovak (en_sk) speaker
qualifies. The unseen rows are spread over every row group, so the whole of both shards was read (about 4.9 GB of
download, nothing saved but the kept rows).

**DiPCo (partial)** - DiPCo is one 13.4 GB `DipCo.tgz`; a gzip stream cannot be seeked, so `eval_dipco.py` reads it from
the start and stops after 2.5 GB. The archive begins with the README, the CDLA license and all 10 transcription files,
then the eval-set WAVs in no fixed order, so the first part already holds whole far-field recordings of several
sessions. We keep one far-field channel per session (16 kHz, one microphone of a 7-mic device a few metres from the
table; music starts at a fixed time mark in each session) and every close-talk headset file we meet. Segment times,
speakers and words are in `data/eval/dipco_segments.jsonl` (`far_start`/`far_end` on the kept far-field channel,
`close_start`/`close_end` on the headset channel). Speakers are `dipco:P01` to `dipco:P32` and appear in no training set.

**VOiCES (not fetched)** - the public S3 bucket `lab41openaudiocorpus` holds only tar.gz archives (devkit 29.5 GB,
competition 20.9 GB, release 449 GB). We streamed the start of the devkit archive: the first 1.58 GB were room-response
recordings (tones, swoops, impulses) and 105 MB-per-microphone TV-distractor tracks, with no speech yet; the
competition archive starts with anonymised speaker-ID trial files. A speech subset of at most 300 MB would need most
of a 21 to 30 GB stream. VOiCES also replays LibriSpeech speakers, who are likely in LibriTTS-R train (not verified).
DiPCo covers the real-room case instead.

## Tests

```bash
.venv/bin/python -m pytest dataprep/tests -q
```

The eval and verify tests (`test_eval_*.py`, `test_verify.py`) run offline on tiny synthetic fixtures.

## Licenses

See `LICENSES.md` - license, URL, attribution and allowed use for every source in the plan.
