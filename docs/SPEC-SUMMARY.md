# Earshot SPEC summary (for Aarush, 2026-11-07)

Full spec - docs/SPEC.md (docs/spec/00-contract.md is the binding part). Choices and reasons - docs/DECISIONS.md.

## What it does
- **You wear AirPods Pro with ANC on. The iPhone mic listens, and Earshot plays only the person you picked.**
  Everyone else, the music and the clatter get pushed down. All of it runs on the phone, and no audio leaves it.
- **Pick someone** - tap Add person and point the phone at them for about 6 s while they talk. Later, face them and
  double-press the AirPods stem to switch (head tracking remembers where they sat).
- **Modes** - Focus (1 person), Table (up to 3 people), Everyone (all voices except yours, with noise removed).
- **Your own voice is always removed**, because hearing yourself 60 to 100 ms late is awful.

## The model (ESB-16-DF)
- Every 4 ms the model takes the newest 32 ms of sound and splits it into 16 frequency bands. Four LSTM blocks track
  each band over time and compare the bands, and a small filter rebuilds only the chosen voice.
- It is told who to keep from a voiceprint: WeSpeaker vectors plus up to 48 short voice tokens. Your own voiceprint is
  given as "never keep this".
- **2.58 M parameters, about 0.2 ms of one phone CPU core per 4 ms (estimate).** Hand-written C with Accelerate runs on
  the audio path. Core ML runs only the voiceprint encoder, and never on the audio thread.

## Latency (honest numbers)
- **Earshot adds about 11 ms.** Bluetooth adds the rest, and no app can remove that.
- **Estimated mouth to ear - about 60 ms on Pro 2 and 100 ms on Pro 1.** These are estimates from other people's
  tests. The app measures and shows the real number at every connect. **It never claims under 20 ms over Bluetooth.**

## Training plan (this Mac, MPS)
| Run | Steps | Time | Must pass |
|---|---|---|---|
| Smoke | 1,500 | 0.5 h | memorize 16 mixtures (>= 20 dB) |
| Pilot | 6,000 | 1.9 h | validation still improving at the end (6k better than 4k and 2k), no NaN steps |
| **Main** | **120,000** | **about 38.5 h of compute** (longer while it yields to your other builds) | the targets below |

- The run checkpoints every 19 min and restarts itself after a crash, and the Mac needs to stay on AC power.
- **Changed 2026-11-08** - a side-by-side test tried two other training losses (one without the spectral term, one
  with a compressed spectral term). Both were about 0.2 dB ahead at step 4,000, but the check at step 6,000 came out
  even, so the training loss stays as first planned. A test run measured 1.26 s per step while the Mac was busy, and
  the main run was briefly cut to 108,000 steps. The main run's own first hour then measured 1.10 s per step, so it
  is back to the full 120,000 steps (about 38.5 h of compute, longer while it yields to your other builds). The model
  never changed.

## "Done" targets (targets, not results yet)
- **Restaurant simulation - median SI-SNR improvement >= 6 dB (goal 9).** Libri2Mix >= 10 dB.
- **Right person picked >= 90% at equal loudness.** Own voice and absent-target output pushed down >= 25 dB.
- **Phone - p99 compute <= 1 ms per 4 ms hop, 0 dropouts in a 60 min locked session, <= 5% battery per hour.**

## Credits and limits
- **The idea (look at someone, then hear only them) comes from UW's Look Once to Hear (CHI 2024).** Earshot uses none
  of its code.
- **WeSpeaker voiceprint weights are research-terms (VoxCeleb), so this is a personal build only.** No App Store,
  TestFlight or sharing.
- Signing uses your free Personal Team, so the app needs a re-install from the Mac every 7 days (about 2 min).

## Questions (defaults I use if you do not answer)
1. **Can the Mac stay on AC with no other GPU jobs for about 40 h?** Default - I start when the pilot passes and
   other sessions are quiet.
2. **Personal build only?** Default - yes, which keeps WeSpeaker.
3. **Is it OK for the phone to keep short voice clips of people you save?** Default - yes, on the phone only.
   Unsaved people expire after 12 h, and the app asks you to confirm the person agreed.
4. **Is a 7-day re-install OK, or do you want the paid Apple Developer Program ($99/yr)?** Default - free team.
5. **For the delay check, is it OK to hold the phone to an AirPod in your ear (or turn off ear detection for a
   moment)?** Default - yes, the app walks you through it.
6. **Do you have 3 to 5 small speakers, or friends who would be recorded at a table, for a real-room test?**
   Default - skip it and use the DiPCo dinner-party recordings instead.
