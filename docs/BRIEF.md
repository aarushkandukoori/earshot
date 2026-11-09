# Earshot - product brief and hard constraints

Shared context for every research, design and build step. Read this first.

## The product

An iPhone app that lets the user hear only one chosen person in a loud room (restaurant, party, bar, crowded hallway).
The user wears AirPods Pro (owns AirPods Pro 1st gen and AirPods Pro 2 USB-C) or wired USB-C earbuds.
The phone runs the AI. Nothing goes to a server.

"The person you're looking at" - the user picks whom to hear by looking at them. Available signals:
- AirPods head tracking (CMHeadphoneMotionManager gives head yaw/pitch/roll).
- Pointing the phone (its back mic / camera side) at the person for a few seconds to capture a voice sample (enrollment).
- AirPods stem press (remote commands) as a hands-free trigger.

The user wants it industry grade: something they can use every day, as close to perfect as possible. Time is not a constraint.

**User answer 2026-11-06 - they own only AirPods (no wired earbuds).** So AirPods Pro over Bluetooth is the PRIMARY
daily path and must be excellent: lowest possible processing latency on top of the Bluetooth delay, ANC to hide the
direct sound, the user's own voice suppressed (hearing yourself delayed is very disruptive), clear in-app guidance.
The wired path stays supported (it is the only way to meet the under-20 ms promise) but is secondary.
Disk cleanup is approved, so expect about 25 GB free.

## Hard requirements

- Latency - under 20 ms mouth-to-ear on the low-latency path (wired earbuds). More than that and the processed voice
  smears against the direct sound (echo / comb filtering). Over AirPods, Bluetooth adds latency that no app can
  remove. The design must measure it, show it, and handle it honestly (for example ANC to hide the direct sound,
  own-voice suppression so the user never hears their own voice delayed, which is very disruptive).
- Real-time safety - zero dropouts. No allocations, locks, Objective-C messaging or file I/O on the audio thread.
  Survives route changes (plug/unplug, AirPods connect), interruptions (calls, Siri) and runs with the screen locked.
- Quality - strong suppression of competing talkers and noise, a natural-sounding target voice, few artifacts,
  works on voices it never saw in training, works with a noisy enrollment sample, goes quiet when the target is silent.
- Safety - output limiter, sane maximum gain, gentle fade-in, no hearing damage, no medical claims.
- Battery and heat - sustainable for hours of use.

## Hardware and tools

- Phone - iPhone 16 Pro (A18 Pro, 8 GB RAM), iOS 26.1, connected by USB, Developer Mode on.
  Signing - free Apple Personal Team (profiles last 7 days).
- Training machine - MacBook Pro M4 (10-core CPU: 4 performance + 6 efficiency, 10-core GPU, 24 GB unified memory),
  macOS 26.5. PyTorch on MPS. No cloud GPUs. Training may run 24-48 hours.
- Xcode 26.0.1, ffmpeg at /opt/homebrew/bin/ffmpeg, uv at /opt/homebrew/bin/uv.
- Project root - the repo folder. Python venv - `.venv` in the repo root (Python 3.12).
- Disk - about 12 GB free right now, likely about 25 GB after a cleanup the user may approve.
  All datasets together should fit in about 10 GB. Never fill the disk below 10 GB free.
- Download bandwidth - about 11 MB/s (openslr.org main site works; us.openslr.org mirror does not).

## Preferences

- Licenses - prefer commercially usable data and models (CC BY, CC0, Apache-2.0, MIT, BSD). Flag anything
  non-commercial or gated (gated = needs the user to accept terms, which they must do themselves).
- Code - simple enough for a CS sophomore to read and explain. A few hundred clear lines beat a giant framework.
  Quality still comes first.
- Honesty - credit prior work (for example UW's Look Once to Hear). Report only measured numbers. Never claim
  under 20 ms over Bluetooth.
- Writing rules for any doc - no em dashes. Use "Label - explanation", never "Label:" headers in bullets.
