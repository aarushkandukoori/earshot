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
