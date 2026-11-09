# iOS 26 platform facts for Earshot (audio and app architecture)

Research date - 2026-11-06. Target - iPhone 16 Pro (A18 Pro), iOS 26.1, Xcode 26.0.1 (iOS 26.0 SDK), free Personal Team signing.
Scope - the eight platform questions in the research task. Model design and datasets are covered elsewhere.

How to read this document
- Confidence - (H) high = Apple documentation, Apple engineer statement or direct measurement. (M) medium = credible secondary
  source, single developer report, or inference from primary sources. (L) low = anecdote, marketing claim or unverified.
- References - every claim cites [Rn]; the list with URLs is at the end.
- "My analysis" marks arithmetic or design reasoning that is mine, not a source's.
- Calendar note - iOS 27 shipped in September 2026 and WWDC26 happened in June 2026. The phone runs iOS 26.1 and Xcode is 26.0.1,
  so iOS 26.2 and iOS 27 APIs are listed only as future options. Anything needing the 26.2 or 27 SDK needs an Xcode update.

---

## 0. Decision summary

1. **Wired path is the only path that can plausibly meet 20 ms.** Public crowd data shows iPhone 13 Pro at 9 ms and iPhone 14 Pro at
   12 to 17 ms round trip (built-in mic to built-in speaker, 64-frame buffers, 48 kHz, measurement mode). No public number exists for
   iPhone 15 or 16, or for the built-in-mic to USB-C-headphone path. Measuring this on the actual phone is the first engineering task,
   because whether 20 ms is achievable depends on it (see section 1.6).
2. **Audio session** - `.playAndRecord`, mode `.measurement`, options `[.allowBluetoothA2DP]` only. Never set `.allowBluetoothHFP`
   (old name `.allowBluetooth`), `.defaultToSpeaker` or `.mixWithOthers`. Prefer 48 kHz and a 64-frame (1.33 ms) I/O buffer, then read
   back what iOS granted.
3. **Bluetooth (AirPods Pro 1 and 2)** - built-in mic in plus AirPods out over A2DP is a supported configuration. Expect roughly
   120 to 220 ms extra latency, varying over time. No public API removes it. Show the number, require ANC, and suppress the user's own
   voice. HFP and the iOS 26 high-quality AirPods recording mode both make things worse for Earshot.
4. **Inference** - run the streaming model with **BNNS Graph** (Accelerate) on the CPU. Compile it single-threaded from an `.mlpackage`
   or build it with the iOS 26 `BNNSGraph.Builder`, preallocate everything, and execute it once per hop. Run it on one real-time worker
   thread joined to the RemoteIO audio workgroup, or directly in the I/O callback if the I/O buffer can equal the hop. Never call Core ML
   from the audio thread. Core ML is fine for the one-shot enrollment embedding.
5. **Head tracking** - `CMHeadphoneMotionManager` works with AirPods Pro 1 and 2 and needs only `NSMotionUsageDescription`. Treat it as
   a slow (about 25 Hz, undocumented), drifting, yaw-relative control signal. Three behaviors must be tested on the device: with the
   screen locked, in a `.playAndRecord` session, and while A2DP output is active.
6. **Stem press** - with A2DP-only routing, AirPods presses arrive as `MPRemoteCommandCenter` commands (play/pause, next, previous) when
   Earshot is the Now Playing app. Calls over HFP never reach apps.
7. **Background** - `UIBackgroundModes` `audio` keeps a running session alive with the screen locked. A recording session cannot be
   started or restarted from the background, so after a phone call Earshot must ask the user to reopen it.
8. **Signing** - a free Personal Team can use Background Modes and HealthKit (so audiogram read is possible). It cannot use Head Pose,
   Sustained Execution, Push or Time Sensitive Notifications. Profiles expire after 7 days, with a limit of 3 apps per device.

---

## 1. Lowest achievable round-trip latency on recent iPhones (wired)

### 1.1 What Apple documents

- **Buffer request is a hint** - `AVAudioSession.setPreferredIOBufferDuration(_:)` "requests a change"; check `ioBufferDuration` to see
  what took effect. Apple: the minimum "is at least 0.005 seconds (256 frames) but might be lower depending on the hardware in use";
  typical maximum 0.093 s [R1]. (H)
- **When to set** - set preferred values before activation; they take effect at activation; non-mixable sessions get priority, and
  sessions using `mixWithOthers` are "unlikely" to have preferred hardware settings honored [R4]. (H)
- **Latency properties** - `inputLatency`, `outputLatency` (seconds, read-only) [R2]. `AVAudioIONode.presentationLatency` maps to
  `kAudioDevicePropertyLatency` and `kAudioStreamPropertyLatency` [R93]. Apple gives no formula beyond these. (H)
- **Measurement mode** - "minimize the amount of system-supplied signal processing"; with several built-in mics "the session uses the
  primary microphone"; it "disables some dynamics processing ... resulting in a lower-output playback level" [R3]. (H)
- **Screen lock** - the old I/O-size increase on lock (1024 to 4096 frames) applies to output-only graphs; "if the graph has input
  running, the I/O size does not change" [R10]. A forum answer adds that an explicit preferred buffer duration is honored even for
  output-only, regardless of lock state [R11]. (H for input-running case)

### 1.2 Measured numbers (best available)

Superpowered crowd data [R6], retrieved 2026-11-06 (page header: 2,898,390 measurements on 40,681 models/builds). The iOS test app uses
RemoteIO, `.playAndRecord` + `.defaultToSpeaker`, mode `.measurement`, 48 kHz, preferred buffer 64/48000 s, and reads mic input and
writes output in the same render callback [R7]. It measures acoustic round trip from the speaker to the mic [R5]. (H for the method;
the numbers are crowd-submitted.)

| Device | iOS | Buffer granted | Round trip |
|---|---|---|---|
| iPhone 11 | 17.6.1 | 64 @ 48 kHz | 10 ms |
| iPhone 12 / 12 mini / 12 Pro Max | 15.6 to 18.0.1 | 64 @ 48 kHz | 12 to 14 ms |
| iPhone 13 / 13 Pro | 15.6 to 18.1 | 64 @ 48 kHz | 9 to 10 ms |
| iPhone 13 Pro Max | 16.2 to 18.1 | 64 @ 48 kHz | 10 ms |
| iPhone 14 / 14 Plus | 16.4 to 17.6 | 64 @ 48 kHz | 15 ms |
| iPhone 14 Pro | 16.2 to 18.0.1 | 64 @ 48 kHz | 12 to 17 ms |
| iPhone 14 Pro Max | 17.5.1, 18.0.1 | 64 @ 48 kHz | 16 to 17 ms |
| iPhone SE (3rd gen) | 17.3 to 17.6 | 64 @ 48 kHz | 8 to 9 ms |
| iPhone 15 / 16 (any) | - | - | **no rows exist** |

Other measured wired figures (all older devices) -
- **Lightning EarPods, hearing app** - 16 ms mic-to-ear (Jacoti, reported 2017) [R15]. (M)
- **iPhone 8 + wired EarPods, 14 hearing apps** - all but one under the 30 ms delay threshold; wired beat Bluetooth by at least 100 ms
  (e-CEO 2021) [R16]. (M)
- **iPhone 6s, iOS 11, 128-frame buffer, analog headset jack (iRig 2)** - 11.5 ms round trip; USB interfaces 9.6 to 15.9 ms [R13]. (M)
- **USB interfaces on iOS at 64 frames / 48 kHz** - 8.9 to 14.6 ms round trip (Overloud) [R14]. (M)

Takeaways (my analysis)
- iOS granted 64 frames on every recent iPhone in the table. Superpowered requested exactly 64, so this table does not show the
  minimum. Older rows show 32 and 16 frames, so smaller sizes have existed. **The iPhone 16 Pro minimum is unknown and must be probed.**
- Non-buffer latency (round trip minus 2 x 1.33 ms) is about 6.3 ms on iPhone 13 Pro and 9.3 to 14.3 ms on iPhone 14 Pro. Device
  generation matters more than buffer size.
- Every recent iPhone row runs at 48 kHz. Many rows at 16 kHz or 44.1 kHz on newer iOS versions show 81 to 785 ms, though some old
  devices (iPhone 5, 6) ran 44.1 kHz at 8 to 17 ms. A forum reply notes latency "might also be larger if the RemoteIO buffer sample rate
  is different from the hardware sample rate" [R12]. **Run the session at 48 kHz and resample inside the app.**
- USB-C EarPods and the USB-C to 3.5 mm adapter are digital USB audio devices with their own converters (M, not separately verified).
  No public measurement exists for their latency on iPhone 15 or 16. USB adds converter and isochronous buffering, which shows up as a
  few extra ms in the interface data above [R13][R14]. (M)

### 1.3 Mode .measurement vs .default

- **Superpowered** - "The default audio session mode adds around 30 ms of additional latency due some feedback reduction and other
  pre-processing" with the built-in mic [R5]. (M)
- **Reported values** - on an iPad Air 2 built-in mic, `inputLatency` read 0.1 ms in measurement mode vs 58 ms in default mode [R12]. (M)
- **Cost of measurement mode** - primary mic only, no system dynamics, lower output level [R3]. A DTS engineer also notes a standard
  `playAndRecord` session's maximum volume is "noticeably lower than the voip audio session" [R75]. Earshot therefore needs its own gain
  stage with a limiter. (H)

### 1.4 How to read inputLatency, outputLatency, ioBufferDuration

- **Formula (estimate)** - round trip about `inputLatency + outputLatency + 2 x ioBufferDuration`. This is the formula used by the
  Round Trip Latency Meter app [R13] and by developers in [R12][R25]. A forum reply from user theanalogkid (the page does not label the
  poster as Apple staff) lists the system terms as I/O buffer frame size + output safety offset + output stream
  latency + output device latency, plus input latency and input safety offset for a round trip. It also notes that render-callback
  timestamps include the buffer and safety offset, but not stream and device latency [R12]. (M)
- **When to read** - after `setActive(true)` and again after every `routeChangeNotification`. Values are per route. They exclude
  Earshot's own algorithmic latency (hop accumulation, model lookahead, resampler delay). (H for route dependence [R12])
- **Do not trust them blindly** - the reported values have been wrong in both directions.
  - AirPods - reported `outputLatency` 160.667 ms, while measured latency ran 215 to 220 ms when fresh and drifted to 155 to 160 ms after
    20 to 30 minutes [R25]. (M)
  - Voice processing I/O - reported total about 969 samples vs measured about 4050 samples on iPhone (iPad 1102 vs 4125;
    coreaudio-api list, 2020) [R17]. (M)
  - The tool maker's own caveat: safety buffers at interface or OS level push real latency above the estimate [R13]. (M)
- **Recommendation** - show a measured number. Run a built-in acoustic loopback calibration (section 9.8) per route, and display it as
  the latency figure, with the formula value as a fallback.

### 1.5 Known iOS traps that silently raise the buffer

- **Sound Recognition or Vocal Shortcuts on (iOS 18)** - `setPreferredIOBufferDuration` is ignored and still returns true. A 256-frame
  request came back as 1104 frames (23 ms), with mismatched input/output buffer sizes and glitches (FB15715421, no Apple reply, fix
  status unknown for iOS 26) [R8]. Hearing-impaired users often enable Sound Recognition, so **Earshot must detect a granted buffer
  larger than requested and tell the user why.** (M)
- **iOS 18 asynchronous apply** - JUCE developers saw `ioBufferDuration` update only after a delay on iOS 18 hardware, and some saw a
  single 96-frame option [R9]. Re-read after a short delay and also from the actual `inNumberFrames` in the callback. (M)

### 1.6 Latency budget for the 20 ms target (my analysis)

Let X be the phone's non-buffer hardware latency for built-in mic in and USB-C out (unknown; 6 to 14 ms if it resembles the speaker
path data above). B = I/O buffer, H = model hop, C = worst-case compute time, L = model lookahead or window overhang.

| Design | Formula | X=6.3, H=4 ms, L=0 | X=14, H=4 ms, L=0 |
|---|---|---|---|
| B=1.33 ms, worker thread per hop | X + 2B + (H - B) + ceil(C/B) x B + L | about 14 ms | about 22 ms |
| B=H=4 ms, inference inside callback | X + 2H + L | about 14 ms | about 22 ms |

- Both designs land at about the same latency. The worker-thread design wins on robustness (section 4.6).
- If the iPhone 16 Pro looks like the 13 Pro, a 4 ms hop with no lookahead fits in 20 ms. If it looks like the 14 Pro, it does not,
  and Earshot would need a 2 ms hop, a 32-frame buffer, or both.
- Every ms of model lookahead or window overhang comes straight out of this budget. The model design must be told the measured X.

---

## 2. AirPods Pro 1 and Pro 2 over Bluetooth

### 2.1 Measured or reported output latency

| Source | Device | Number | Notes |
|---|---|---|---|
| Developer forum 2021 [R25] | AirPods 2 + iPhone 12 mini | reported `outputLatency` 160.667 ms; measured 215 to 220 ms falling to 155 to 160 ms over 20 to 30 min | built-in mic in; options `.defaultToSpeaker`, `.mixWithOthers`, `.allowBluetoothA2DP`; 64-frame buffer granted (`ioBufferDuration` 1.333 ms), reported `inputLatency` 2.438 ms |
| Stephen Coyle 2022 [R26] | AirPods Pro 2 / Pro 1 + iPad Pro | 126 ms / 167 ms average (Pro 1 was 144 ms in an earlier run); built-in speakers 83 ms | tap-to-sound (keyboard clicks), includes about 20 ms Apple Pencil latency |
| Jacoti 2017 [R15] | AirPods (1st gen) + iPhone mic in a hearing app | 258 to 295 ms (192 ms with AirPods mic) | mic-to-ear |
| macOS system log in a forum post 2024 [R27] | AirPods on a Mac | A2DP "In ear-Latency" 151 ms normally, 60 ms in Low Latency Game mode | shows the system's own latency target |

Interpretation (my analysis, M) - plan for **about 120 to 220 ms extra mouth-to-ear latency over AirPods**, drifting during the first
tens of minutes. Pro 2 is likely at the low end and Pro 1 higher. Measure it per session (section 9.8). Never present it as low latency.

### 2.2 Built-in mic in + AirPods out (A2DP) in .playAndRecord

- **Allowed** - "Starting with iOS 10.0, apps using the playAndRecord category may also allow routing output to paired Bluetooth A2DP
  devices" with `.allowBluetoothA2DP` [R18]. (H)
- **The exact recipe** - a developer got output pushed to the speaker whenever they selected the built-in mic. An Apple WWDC lab fixed
  it: drop `.allowBluetooth` and use only `.allowBluetoothA2DP` [R31]. (M, Apple lab advice relayed)
- **Worked in practice** - [R25] ran exactly this route (built-in mic, AirPods A2DP out, 64-frame buffer). (M)
- **Counter-report** - on iOS 18.4.1 (iPhone 12 Pro Max), `currentRoute` said built-in mic but the audio sounded like the AirPods mic
  with voice isolation [R32]. The options included `.defaultToSpeaker` and the case is unresolved. Earshot should verify the real input
  with a tap test (tap the phone near its bottom mic and look for the transient). (L)
- **Priority rule** - if both A2DP and HFP options are set and the device supports both, "the system gives hands-free ports a higher
  priority" [R18]. That is one more reason never to set the HFP option. (H)

### 2.3 What HFP does (.allowBluetoothHFP, formerly .allowBluetooth)

- **Both directions move** - selecting an HFP input moves output to HFP, and selecting HFP output moves input. "Both audio input and
  output are routed to the Bluetooth HFP device" [R19]. Current Apple docs name the option `allowBluetoothHFP`; the older
  `allowBluetooth` page shows a deprecated availability range [R19]. (H for routing, M for naming)
- **Consequences for Earshot** - the input becomes the AirPods mics, not the phone array. Output becomes the hands-free voice link, not
  A2DP. Selecting the built-in mic while HFP is allowed pushed output to the speaker [R31]. (H/M)
- **Head tracking may stop** - developers report `CMHeadphoneMotionManager` updates stop when the AirPods mic is in use via
  `.allowBluetooth` [R63]. (M)
- **Stem presses are lost** - HFP buttons are call commands, and "the system doesn't actual have any API an app can use to receive HFP
  commands" (Apple DTS) [R67]. (H)
- **Transparency is suppressed** - an iOS 26.x developer reports native Transparency on the receiving AirPod stops during HFP mic
  capture. The same report calls HFP latency "nearly real time" (no number) [R24]. (L)
- **Switch stalls (iOS 27.0.1)** - switching A2DP to HFP while audio renders blocked `AVAudioSession` calls for 20 to 36 s (FB25324099)
  [R33]. (L)
- **Verdict** - do not use HFP.

### 2.4 iOS 26 high-quality AirPods recording

- **Name** - `AVAudioSession.CategoryOptions.bluetoothHighQualityRecording` (iOS 26.0) [R20]. For capture sessions,
  `AVCaptureSession.configuresApplicationAudioSessionForBluetoothHighQualityRecording = true` [R21]. (H)
- **Behavior** - full-bandwidth Bluetooth input "when the Bluetooth route supports it". It combines with `.allowBluetoothHFP` as the
  fallback. It is allowed only in mode `.default`, not supported in the EU, and checked with
  `currentRoute.inputs.first?.bluetoothMicrophoneExtension?.highQualityRecording.isSupported` / `.isEnabled` [R20]. WWDC describes a
  "media tuning" for content creators over "a more reliable bluetooth link" [R21]. (H)
- **Latency** - Apple's note: "This option may increase input latency when enabled and isn't recommended for real-time communication
  usage" [R20]. A developer reports "noticeably higher latency" than HFP [R24]. (H/L)
- **Supported AirPods** - Apple lists AirPods 4, AirPods 4 with ANC and AirPods Pro 2 (later Pro 3). **Not AirPods Pro 1** [R22]. (H)
- **Verdict** - irrelevant for the live path. It is an input mode that moves the mic to the AirPods.

### 2.5 iOS 26.2 additions (not usable on 26.1 with Xcode 26.0.1)

- `.farFieldInput` (iOS 26.2) - prefers far-field capture from a Bluetooth mic; requires `.allowBluetoothHFP`; checked through
  `bluetoothMicrophoneExtension.farFieldCapture` [R23]. (H that it exists; behavior undocumented)
- `Mode.dualRoute` with `.multiRoute` (iOS 26.2) - the built-in mic/speaker is always the primary route, plus a secondary
  `headsetMic`, `headphones`, `bluetoothLE` or `bluetoothHFP` device; requires `.allowBluetoothHFP`. Apple: "This API may not be used to
  enable recordings of others without their awareness" [R23]. (H)
- **Why it matters later** - dualRoute could give phone-mic plus AirPods-mic input at once (for example, for own-voice detection). But
  the secondary route is HFP, and one developer measured the secondary AirPods mic around -40 dB during speech [R24]. Research option
  only. (L)

### 2.6 Any lower-latency Bluetooth path for third-party apps?

- **Game Mode (experimental)** - Apple says Game Mode "doubles the Bluetooth sampling rate, which reduces ... audio latency for wireless
  accessories like ... AirPods". It turns on when a supported game enters full screen and turns off when the game quits [R28]. Apple
  says AirPods Pro 2 in Game Mode give "the best wireless audio latency Apple has ever delivered for mobile gaming" (no number) [R30].
  The opt-in key is `LSSupportsGameMode` (iOS 18.6+) [R29]; one developer pairs it with
  `LSApplicationCategoryType = public.app-category.games` [R33]. The 151 to 60 ms log in [R27] shows
  the size of the effect on macOS. It is unverified on iOS for a non-game, it needs the app in full-screen foreground (screen on), and it
  would be App Store misuse. It is acceptable for a personal-use experiment. (M for the effect, L for Earshot applicability)
- **Nothing else public** - no LE Audio, aptX or codec selection API for AirPods was found. AirPods' own Hearing Aid, Conversation Boost
  and Transparency processing runs on the buds and cannot host app DSP. (M)

### 2.7 Handling the Bluetooth delay honestly

- **ANC isolation** - RTINGS measures AirPods Pro 2 full-range isolation at about -24.5 dB (bass -20.8, mid -26.7, treble -26.5).
  Original AirPods Pro measure about -21.0 dB [R34]. (M) The direct sound is reduced, not removed.
- **No API sets the listening mode** - forum requests for an ANC/Transparency API are unanswered [R89]. The Shortcuts action
  "Set Noise Control Mode" exists (iOS 13.2+) and can be run from a personal automation [R90]. Earshot should guide the user to set
  that up. (M)
- **Conversation Awareness (Pro 2)** - "reduces your media volume and boosts voices in front" when the user speaks, and pauses spoken
  content after a few seconds [R88]. Its effect on Earshot is untested. It could duck or even pause Earshot. Recommend turning it off,
  and test it. (M fact, L effect)
- **Own voice** - Earshot cannot capture the AirPods mics while using the phone mic in a normal route (section 2.3), and the motion API
  is far too slow for speech detection (section 5). Own-voice suppression must therefore be computed from the phone-mic signal, for
  example from an own-voice enrollment. (My analysis)

---

## 3. Built-in mic selection on iPhone 16 Pro, and voice processing

### 3.1 APIs

- **Ports and sources** - `availableInputs` gives an `AVAudioSessionPortDescription` with `portType == .builtInMic`. Its `dataSources`
  are `AVAudioSessionDataSourceDescription` objects with `dataSourceName`, `location`, `orientation` (`.front`, `.back`, `.bottom`, `.top`
  ...), `supportedPolarPatterns`, `selectedPolarPattern`, `preferredPolarPattern`, and `setPreferredPolarPattern(_:)`. Select with
  `port.setPreferredDataSource(_:)` and `session.setPreferredInput(_:)` [R35][R38]. (H)
- **Polar patterns** - `.omnidirectional`, `.cardioid`, `.subcardioid`, `.stereo` (iOS 14) [R35]. Front/back cardioid is a beamformer
  built from two mics. Subcardioid is shaped so it does not null the user's own voice. Stereo uses all mics with modeling [R37]. (H)
- **Orientation** - `setPreferredInputOrientation(_:)` with `AVAudioSession.StereoOrientation` decides left/right for the stereo
  pattern only; mono patterns ignore it [R36][R37]. (H)
- **Order of calls** - set category and mode, activate, then select input, data source and pattern [R38][R4]. Changing a data source on
  the active route reconfigures it, so do it before starting I/O. (H)
- **Hardware** - iPhone 16 Pro has "four new studio-quality mics" [R43]. Apple does not publish which data sources or patterns each
  model exposes. Enumerate them at runtime. (H)
- **No raw per-capsule access** - AVAudioSession gives at most a processed 2-channel stereo pattern. AVCaptureSession can give 4-channel
  first-order ambisonics (section 3.4). No public raw multi-capsule API was found. (M)

### 3.2 Do they work in .playAndRecord + .measurement with A2DP output?

- **Documentation is silent.** Measurement mode "uses the primary microphone" [R3]. An Apple Support reply showed
  `selectedDataSource = Bottom` for `.playAndRecord` + `.measurement` [R39]. Whether `setPreferredDataSource(Back)` and cardioid or
  stereo are honored in measurement mode is undocumented, and no reliable report was found either way. (L)
- **A2DP output should not matter in principle** - the data source belongs to the built-in input port. But see the iOS 18.4.1
  counter-report in 2.2 [R32]. (L)
- **Required test (Phase 0)** - for each mode (`.measurement`, `.default`, `.videoRecording`) and each output (USB-C, A2DP), log
  `selectedDataSource`, `selectedPolarPattern`, `inputNumberOfChannels` and the round-trip latency. Verify with tap tests near each mic.

### 3.3 Voice processing (AUVoiceProcessingIO / setVoiceProcessingEnabled)

- **What it is** - `kAudioUnitSubType_VoiceProcessingIO` (AUVoiceIO), or `AVAudioInputNode.setVoiceProcessingEnabled(true)` (iOS 13).
  Provides echo cancellation, noise suppression and AGC. Other-audio ducking is configured with
  `AVAudioVoiceProcessingOtherAudioDuckingConfiguration` (iOS 17). With voice processing on, users control Standard / Voice Isolation /
  Wide Spectrum mic modes [R40][R42]. (H)
- **Latency** - Apple publishes no number. One measured example: about 4050 samples round trip on iPhone with a roughly 21 ms buffer,
  while the reported properties summed to about 1000 samples [R17]. (M)
- **Fit for Earshot - it hurts.** It is tuned for the near talker (the phone owner), so it suppresses exactly the far voice Earshot
  wants. It adds latency, forces its own processing chain, and its echo canceller targets speaker echo that earbuds do not produce.
  Keep it off. (M, my analysis on the tuning and fit, from [R40][R42])
- **Echo-cancelled input (iOS 18.2)** - `setPrefersEchoCancelledInput(_:)` is valid only with `.playAndRecord` + `.default` on
  "certain 2024 or later iPhone models", for the built-in mic and speaker route; switching to a headset can disable it [R41].
  Not needed with earbuds. (H)

### 3.4 Option for enrollment only - Spatial Audio (FOA) capture

- iOS 26 lets `AVCaptureSession` deliver 4-channel first-order ambisonics. Set `AVCaptureDeviceInput.multichannelAudioMode =
  .firstOrderAmbisonics`; then up to two `AVCaptureAudioDataOutput`s, one FOA (4 channels, HOA ACN SN3D) and one stereo, are available,
  selected with `spatialAudioChannelLayoutTag` [R21]. (H)
- **Use** - a directional, multi-channel enrollment clip when the user points the phone's back at the person. It is not for the live
  path, since buffer sizes and latency are undocumented [R21]. Set `automaticallyConfiguresApplicationAudioSession = false` so the
  capture session does not reconfigure Earshot's session. (My analysis)

---

## 4. Real-time inference on iOS 26

### 4.1 BNNS Graph (Accelerate) - the recommended engine

- **What and when** - whole-graph CPU inference in Accelerate (iOS 18). Apple: "We recommend BNNSGraph for audio and other
  latency-sensitive use cases because it gives you control over allocating memory and multithreading" [R45]. Claimed "on average at
  least 2x faster" than previous BNNS primitives [R44]. (H that Apple claims it)
- **From a model file (iOS 18)** - convert PyTorch to an ML Program `.mlpackage` with coremltools; Xcode compiles it to `.mlmodelc`.
  The steps, in order -
  `BNNSGraphCompileOptionsMakeDefault()`, `BNNSGraphCompileOptionsSetTargetSingleThread(opts, true)`,
  `BNNSGraphCompileFromFile(path, nil, opts)`, `BNNSGraphContextMake(graph)`,
  `BNNSGraphContextSetArgumentType(ctx, BNNSGraphArgumentTypePointer)`, `BNNSGraphContextGetWorkspaceSize`, page-aligned workspace,
  `BNNSGraphGetArgumentPosition` (the argument order can differ from the source model), then `BNNSGraphContextExecute(...)` [R44][R46].
  Swift wrapper - `BNNSGraph.Context(compileFromPath:functionName:options:)` [R47]. (H)
- **No-allocation contract** - `BNNSGraphContextExecute` does not allocate if the workspace is non-nil, all tensors are bounded, no new
  shape information is passed, and the caller provides all output storage. A context may be used on only one thread at a time [R46].
  (H) Note: Apple's own example builds a Swift array of arguments at call time; on the audio thread, preallocate that array. (My analysis)
- **Dynamic shapes** - `BNNSGraphContextSetDynamicShapes` / `SetBatchSize` must be set before execute and must not change during
  execution [R46]. Prefer static shapes; WWDC says they perform better [R45]. (H)
- **Streaming state** - `BNNSGraphContextMakeStreaming` (iOS 18.0) allocates ring buffers for Core ML state arguments and optimizes the
  `output_state = concat(input_state, x)` pattern used for dilated-conv caches. It needs a function attribute `BNNSOptions` with
  `{ 'StateMode': 'Streaming' }`. `BNNSGraphContextSetStreamingAdvanceCount` (iOS 18.4) covers dynamic advance [R46]. How to set that
  attribute from coremltools is not documented anywhere I found (L). The simple, reliable alternative is explicit state tensors as
  inputs and outputs, ping-ponged by Earshot.
- **Build in code (iOS 26)** - `BNNSGraph.makeContext { builder in ... }` with `BNNSGraph.Builder` and `Builder.Tensor`. Ops include
  `conv`, `convTranspose` (dilations, groups), `matmul`, `linear`, `lstm`, `bidirectionalLSTM`, `gru`, `rnn`, `layerNorm`, `rmsNorm`,
  `batchNorm`, `instanceNorm`, `channelNorm`, `softmax`, `logSoftmax`, `gelu`, `silu`, `sigmoid`, `tanh`, `prelu`, `relu6`, `pad`,
  `pooling`, `reshape`, `transpose`, `concatenate`, `gather`/`scatter` variants, slicing by subscript, reductions, `topK`, and `cast`.
  Arguments can be `.inputOutput` (in-place state) [R47]. No FFT/STFT op is listed, so do STFT with vDSP. (H)
- **Precision** - FP32, FP16, Int32 and Bool appear in examples. FP16 is "significantly faster" in Apple's demo [R45]. (H)
- **Debug** - NaN/Inf checks (`BNNSGraphContextEnableNanAndInfChecks`, `checkForNaNsAndInfinities`), not for shipping [R44][R47]. (H)
- **Sample code** - "Supporting real-time ML inference on the CPU" (bitcrusher Audio Unit extension), with the graph created on a
  background thread, execute on the audio thread, and a separate context for the UI [R48][R44]. (H)
- **Limits** - Apple publishes no list of which MIL ops compile from a file. Compile the real model early and use
  `BNNSGraphCompileOptionsSetMessageLogCallback` to see failures [R46]. (M)

### 4.2 Core ML (including MLState)

- **MLState (iOS 18)** - `model.newState()` (Apple's Swift sample; `make_state()` in coremltools Python), `prediction(from:using:)`; "Each stateful prediction that uses the same
  MLState must be serialized" [R49]. Converted with `ct.StateType`, `register_buffer`, MIL `read_state` / `coreml_update_state`,
  iOS18 target [R50]. (H)
- **Thread safety** - "the synchronous prediction API is not thread safe"; the async API is thread safe [R51]. (H)
- **Per-prediction latency for tiny models** - Apple publishes none. Secondary sources: an Apple Neural Engine dispatch costs about 0.1 to
  0.23 ms of fixed overhead [R52]. Small face models measured 0.43 to 0.71 ms per prediction on an iPhone 15 Pro ANE [R52]. (L to M)
- **Render thread - unsafe.** No Apple page says it in one sentence, but every piece points the same way. Predictions go through
  Objective-C feature providers and `MLMultiArray`s, they must be serialized, and ANE work crosses process boundaries. Apple's own
  recommendation for audio is BNNS Graph [R45]. (M)
- **Use Core ML for** - the one-shot speaker-embedding at enrollment, on a background queue. The resulting vector is handed to the
  real-time side through a preallocated buffer and an atomic flag. (My analysis)

### 4.3 Hand-written Accelerate

- `cblas_sgemm` / `cblas_sgemv` - call `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)` (iOS 18; stored per thread) on the audio
  worker so BLAS never spawns threads [R53]. (H)
- **vDSP** (`vDSP_DFT_zrop_CreateSetup` once, then `vDSP_DFT_Execute`; `vDSP_mmul`; FIR/biquad) and **vForce** (`vvtanhf`, `vvexpf`) -
  no allocation at execute time. (M, standard Accelerate usage)
- **Classic BNNS layers** are filed under "Deprecated symbols" (Classic BNNS API) in Apple's BNNS index [R46]. Do not build on them. (M)
- **Matrix hardware** - A18 Pro is reported as Armv9.2 with SME [R54]. On M4, one P-core reached about 2009 FP32 GFLOPS with SME vs about
  113 GFLOPS with Neon [R54]. A streaming model running at batch 1 is mostly matrix-vector work and will not see SME peak. (M)

### 4.4 Audio workgroups and a real-time worker thread

- The I/O thread is joined to the device workgroup automatically; only real-time threads can join (iOS 14+) [R56]. (H)
- **Get the workgroup** - `kAudioOutputUnitProperty_OSWorkgroup` on the RemoteIO unit (v2) or `AUAudioUnit.osWorkgroup` (v3, iOS 14)
  [R56]. With AVAudioEngine, use `engine.outputNode.audioUnit` [R93]. One developer got an invalid value (0x40) from RemoteIO, likely
  because the `ioDataSize` argument was uninitialized. Always pass `sizeof(os_workgroup_t)` and check the result [R58]. (H/M)
- **Join** - `os_workgroup_join(wg, &token)` once at thread start, `os_workgroup_leave` at exit. Handle `EALREADY` / `EINVAL` [R56]. (H)
- **Thread count** - `os_workgroup_max_parallel_threads` [R55]. Earshot needs 1 worker. (H)
- **Real-time policy** - `thread_policy_set(..., THREAD_TIME_CONSTRAINT_POLICY, ...)` with computation and constraint in Mach time
  units (TN2169) [R57]. For a 4 ms hop, use period 4 ms, computation about 2 ms and constraint about 4 ms. Do not spin and do not create
  many such threads [R57]. (H for the API, my analysis for the numbers)
- **Asynchronous work on its own clock** - `AudioWorkIntervalCreate` plus `os_workgroup_interval_start` / `_finish` per cycle [R56].
  Earshot's worker runs in step with I/O, so join the device workgroup instead. (H)
- **Signaling** - Apple's sample pattern is "wait on a semaphore" in the worker and "signal a semaphore" from the controlling thread
  [R56]. Use lock-free single-producer/single-consumer ring buffers for audio. (H for the pattern)

### 4.5 A18 Pro compute budget (my analysis)

- A18 Pro has 2 performance and 4 efficiency CPU cores and a 16-core Neural Engine [R43]. (H)
- 1 to 3 GMAC/s at a 4 to 8 ms hop is 4 to 24 MMAC per hop. Neon peak on an Apple P-core is about 56 GMAC/s FP32 (from the M4 figure of
  113 GFLOPS [R54]); FP16 doubles it. At a realistic 15 to 30 percent efficiency for small conv/GRU layers, 3 GMAC/s needs roughly 0.4
  to 1.6 ms per 4 ms hop on one P-core. That fits, but only measured p99.9 timings count.
- Budget rule - worst-case compute at most 50 percent of the hop. Downshift the model when `ProcessInfo.thermalState` reaches
  `.serious` [R91] or Low Power Mode is on.

### 4.6 Recommended inference path for Earshot

1. **Engine** - BNNS Graph on the CPU, FP16, static shapes, single-threaded compile, pointer arguments, preallocated workspace and
   outputs, explicit state tensors. Build the graph from the `.mlpackage`. Use `BNNSGraph.Builder` (iOS 26) for pre/post-processing
   or if file compile fails on an op.
2. **Scheduling (production)** - one real-time worker thread joined to the RemoteIO workgroup runs one inference per hop. The I/O
   callback only moves samples through ring buffers and signals the worker. **If a hop is late, the callback plays a concealment frame
   (for example, the last mask or a gentle fade toward attenuated passthrough) instead of glitching.** This also keeps working when iOS
   changes the I/O buffer size (route changes, Sound Recognition, Bluetooth).
3. **Scheduling (prototype)** - if the probe shows iOS grants an I/O buffer equal to the hop (for example, 192 frames = 4 ms), execute
   inside the RemoteIO callback. It is simplest and has the same latency (section 1.6), but an overrun becomes an audible dropout.
4. **Fallback engine** - hand-written C with Accelerate (`cblas_sgemv` + vDSP/vForce), with single-threaded BLAS, for any layer BNNS Graph
   will not compile.
5. **Not on the live path** - Core ML/ANE (dispatch jitter, not real-time safe), and GPU/Metal (not evaluated here; Apple's real-time
   guidance points to the CPU [R45]).
6. **iOS 27 note** - iOS 27 adds real-time-safe block variants, such as `AVAudioSinkNode initWithRealtimeSafeReceiverBlock:` [R92]. They
   are not available on iOS 26.

---

## 5. CMHeadphoneMotionManager

- **Availability** - iOS 14+ [R59]. Supported - "audio products that support spatial audio with dynamic head tracking, like AirPods
  Pro" [R60]. Apple's AirPods guide lists AirPods 3, AirPods 4, AirPods Pro (all generations) and AirPods Max for head-tracked Spatial
  Audio [R61]. **Both of the user's AirPods Pro (1st gen and 2) qualify.** (H)
- **Info.plist** - `NSMotionUsageDescription` is required: "If this key is absent, the system crashes your app" [R59]. Check
  `CMHeadphoneMotionManager.authorizationStatus()` [R59][R60]. (H)
- **Entitlement** - none. The Head Pose entitlement (`com.apple.developer.coremotion.head-pose`) covers only system head-tracked
  spatial mixing (`AVAudioEnvironmentNode.isListenerHeadTrackingEnabled`, `kAudioUnitProperty_SpatialMixerEnableHeadTracking`, PHASE)
  [R66], and it is not available to free accounts [R85]. (H)
- **API** - `isDeviceMotionAvailable`, `startDeviceMotionUpdates(to:withHandler:)` (push) or `deviceMotion` (pull),
  `CMHeadphoneMotionManagerDelegate` connect/disconnect, `startConnectionStatusUpdates()` [R59]. Each `CMDeviceMotion` gives attitude,
  rotation rate, user acceleration and `sensorLocation` (left or right bud). Data comes from one bud at a time, with handoff when a bud
  is removed. Ear-detection changes fire disconnect/connect [R60]. (H)
- **Update rate** - undocumented and not settable. Community reports say about 25 Hz on iOS, and one measurement found 50 Hz with AirPods
  Pro 2 on macOS 26 [R62]. Compute the rate from timestamps; do not hard-code it. (L)
- **Reference frame and drift** - undocumented. There is no reference-frame parameter (unlike `CMMotionManager`). WWDC shows using a
  saved starting attitude and computing relative change [R60]. Yaw has no magnetic reference, so expect slow drift (only qualitative
  reports [R62]). On macOS, yaw re-centered when Spatial Audio playback started [R65]. Treat yaw as relative: re-zero it when the user
  confirms a target, and on any jump. (L/M)
- **Background, screen locked** - undocumented. A 2021 forum question about it is unanswered [R64]. **Must test.** Fallback design: use
  head tracking only to pick or switch targets while the screen is on; the speaker embedding keeps working after lock. (L)
- **With A2DP output from .playAndRecord** - not confirmed by any source. It works with the iPhone mic and no Bluetooth input; it
  stopped when the AirPods mic (HFP) was selected; one reply says only play-only or record-only sessions worked, in a voice-chat context
  [R63]. **Must test** with `.playAndRecord` + `.allowBluetoothA2DP` + built-in mic. (L)

---

## 6. AirPods stem press to MPRemoteCommandCenter in a .playAndRecord app

- **Mechanism** - "MPRemoteCommandCenter handles A2DP commands but phone calls go over HFP ... The system doesn't actual have any API an
  app can use to receive HFP commands" (Kevin Elliott, Apple DTS). The same thread's developer found the headset button works in
  `.playAndRecord` when output is the speaker or `.allowBluetoothA2DP` is set [R67]. (H)
- **Requirements** - be the Now Playing app: activate a non-mixable category (`.playAndRecord` is non-mixable unless `.mixWithOthers`
  [R74]), support at least one remote command [R68], and actually be playing audio [R69]. Set `MPNowPlayingInfoCenter.default().nowPlayingInfo`
  (title, playback rate) [R68]. (H)
- **Gestures (AirPods Pro 2)** - press = play/pause, double press = next track, triple press = previous track; press and hold switches
  listening modes (configurable) [R71]. Map these to `togglePlayPauseCommand` (or `playCommand` / `pauseCommand`), `nextTrackCommand`
  and `previousTrackCommand`. Press and hold is not available to apps. (H)
- **Caveats**
  - Spatial Audio - on iOS, Spatial Audio for AURemoteIO/AudioQueue apps "is automatically enabled for apps that register for Now
    Playing", and the user controls it in Control Center [R70]. Its effect on Earshot's latency is unknown; ask the user to set Spatial
    Audio Off for Earshot, and test it. (H fact, L effect)
  - Route disconnect - since iOS 17 the system interrupts active Now Playing sessions when the route disconnects [R79]. Keep the default;
    it helps stop audio safely when AirPods drop.
  - Conversation Awareness may send unexpected volume changes or pauses (section 2.7).
  - Camera-remote stem clicks via `AVCaptureEventInteraction` work only with H2 AirPods (not Pro 1), only with an active
    `AVCaptureSession`, and not in the background [R72]. Not a substitute.
  - Press to Mute (`AVAudioApplication.inputMuteStateChangeNotification`, iOS 17) is for communication apps during calls [R70][R73]. It
    likely does not fire on an A2DP route; unverified. (L)

---

## 7. Background audio, interruptions, route changes, headphone safety

### 7.1 Background and screen lock

- `.playAndRecord` audio "continues with the Silent switch set to silent and with the screen locked"; add `audio` to
  `UIBackgroundModes` to keep running in the background [R74]. The app keeps running "as long as it is playing audio or video content or
  recording audio content"; if recording or playback stops, it is suspended [R76]. (H)
- **Starting from the background is impossible** - "The audio system will not allow any kind of recording session to activate from the
  background" (Apple DTS, Feb 2025) [R75]. Start in the foreground. After an interruption ends in the background, reactivation is
  expected to fail. Post a local notification ("Tap to resume Earshot"); regular local notifications need no special capability. (H)
- **Interruption priority** - "any incoming call will immediately interrupt your playAndRecord session"; a non-mixable session is also
  interrupted by other non-mixable apps such as Music [R75]. (H)
- **Privacy** - the recording indicator cannot be hidden in the background [R76]. (M)

### 7.2 Interruptions (calls, Siri, alarms)

- **On iOS 26** - observe `AVAudioSession.interruptionNotification`; read `AVAudioSessionInterruptionTypeKey` (`.began` / `.ended`) and
  `AVAudioSessionInterruptionOptionKey` (`.shouldResume`) [R77]. Reasons include `.default`, `.builtInMicMuted`, `.routeDisconnected`,
  `.sceneWasBackgrounded`, `.appWasSuspended` [R78]. (H)
- **Fewer interruptions** - `setPrefersNoInterruptionsFromSystemAlerts(true)` (iOS 14.5) keeps banner-style incoming calls from
  interrupting until accepted; it has no effect with full-screen call style [R79]. (H)
- **iOS 27 change** - lifecycle notifications (`didBecomeActiveNotification`, `didBecomeInactiveNotification`,
  `resumptionRecommendationNotification`) replace the legacy interruption notification, which is deprecated after 26 [R77]. Handle both
  when Earshot moves to the iOS 27 SDK. (H)
- Also handle `mediaServicesWereResetNotification` by rebuilding the session and all units. (M, standard practice)

### 7.3 Route changes

- Observe `routeChangeNotification`. On `.oldDeviceUnavailable` (headphones unplugged, AirPods out of range), **mute immediately**. The
  next route is the receiver or speaker, which with a live mic means feedback howl and loud output. (Safety design, my analysis)
- Allow output only when `currentRoute.outputs` is `.headphones`, `.usbAudio` or `.bluetoothA2DP`; never `.builtInSpeaker` or
  `.builtInReceiver`. Re-assert `setPreferredInput(builtInMic)` after every change, because USB-C EarPods add their own mic as an input
  candidate. Re-read the latencies and re-run calibration. (My analysis, based on [R19][R31])

### 7.4 Headphone safety features and reading output levels

- **Reduce Loud Audio** (Settings > Sounds & Haptics > Headphone Safety) - "iPhone analyzes your headphone audio and reduces any sound
  above the level you set"; the screen shows an 85 dB default [R80]. **Headphone Notifications** fire at 100 percent of the 7-day dose
  (example: 80 dB for 40 hours in 7 days) and lower the volume on the next connection [R80]. Measurements are "most accurate when using
  Apple or Beats headphones"; other headphones are "estimated based on the volume of your iPhone" [R80]. (M, support pages)
- **What apps can read**
  - `AVAudioSession.outputVolume` (0.0 to 1.0, KVO-observable). Only the user can change it, through `MPVolumeView` [R81]. (H)
  - HealthKit `headphoneAudioExposure` (dB(A) SPL, equivalent continuous level samples written by iPhone/Watch) and
    `headphoneAudioExposureEvent` (iOS 14.2, saved when a loud-audio notification fires) [R82]. These are after-the-fact, not real time.
    (H)
  - No public API was found for real-time output dB SPL or for the user's Reduce Loud Audio setting. (M)
- **Earshot's own safety layer** - an output limiter, a hard digital gain cap, 300 to 500 ms fade-in on start and after route changes,
  and mute on speaker routes. Because measurement mode lowers output level [R3], expect to need gain, which makes the limiter mandatory.

---

## 8. HealthKit audiogram and Personal Team signing limits

### 8.1 Audiogram read

- `HKAudiogramSample` (iOS 13), read via `HKObjectType.audiogramSampleType()`; `sensitivityPoints: [HKAudiogramSensitivityPoint]` with
  `frequency`, `leftEarSensitivity`, `rightEarSensitivity` (dB HL) [R83]. (H)
- iOS 18.1 adds `HKAudiogramSensitivityTest` (`sensitivity`, `type: HKAudiogramConductionType`, `masked`, `side`, `clampingRange`) via
  `HKAudiogramSensitivityPoint.tests` [R83]. (H)
- **Source of data** - the AirPods Pro 2 Hearing Test (iOS 18.1/18.2+) stores "the detailed audiogram ... in the Health app", and
  professional audiograms can be added [R84]. (H)
- **Needs** - the HealthKit capability (`com.apple.developer.healthkit`), `NSHealthShareUsageDescription`, and a read authorization
  request. Use the audiogram only for comfort EQ and loudness limits, with no medical claims (brief requirement).

### 8.2 Free Personal Team (Apple's capability table, parsed 2026-11-06 [R85])

| Capability | Free account | Relevance |
|---|---|---|
| Background modes | Yes | `audio` mode for screen-locked use |
| HealthKit | Yes | audiogram read |
| App groups, Data protection, Keychain sharing, Inter-App Audio, HomeKit, Maps, Wireless Accessory Configuration | Yes | not needed |
| Head Pose (`com.apple.developer.coremotion.head-pose`) | **No** | only for system spatial mixer head tracking; `CMHeadphoneMotionManager` does not need it |
| Spatial Audio Profile | **No** | personalized HRTF in `AVAudioEnvironmentNode`; not needed |
| Sustained Execution (`com.apple.developer.sustained-execution`) | **No** | would cap burst performance for steady timing [R87]; nice to have, unavailable |
| Push notifications, Time Sensitive Notifications, Siri, iCloud, Game Center | **No** | use plain local notifications |
| Extended Virtual Addressing, Increased Debugging Memory Limit | **No** | not needed for a small model |

- **Account limits** - up to 10 App IDs (expire after 7 days), 3 registered devices, **3 apps per device**, and provisioning profiles
  that "expire 7 days from issuance" [R86]. (H)
- `NSMotionUsageDescription`, `NSMicrophoneUsageDescription` and `LSSupportsGameMode` are plain Info.plist keys and need no entitlement.

---

## 9. Recommended iOS audio architecture (decision-ready)

### 9.1 Session configuration (wired and AirPods)

```swift
let s = AVAudioSession.sharedInstance()
try s.setCategory(.playAndRecord, mode: .measurement, options: [.allowBluetoothA2DP])  // no HFP, no speaker, not mixable
try s.setPreferredSampleRate(48_000)
try s.setPreferredIOBufferDuration(64.0 / 48_000.0)   // probe 32, 48, 64, 96, 128, 192 in Phase 0
try s.setPrefersNoInterruptionsFromSystemAlerts(true)
try s.setActive(true)
if let mic = s.availableInputs?.first(where: { $0.portType == .builtInMic }) { try s.setPreferredInput(mic) }
// Then choose a data source / polar pattern (only if the Phase 0 probe shows measurement mode honors it).
// Verify: s.ioBufferDuration, s.sampleRate, s.inputLatency, s.outputLatency, currentRoute input/output port types.
```

- If the probe shows measurement mode breaks A2DP or the back data source, use `.default` mode for that route and record the latency
  difference.
- While recording, system haptics and sounds are suppressed unless `setAllowHapticsAndSystemSoundsDuringRecording(true)` is set
  [R94]. Set it if the UI uses haptic confirmation (for example, "target locked").

### 9.2 Audio I/O

- **RemoteIO (`kAudioUnitSubType_RemoteIO`)** with input enabled on bus 1. One render callback in C pulls the mic with `AudioUnitRender`
  (bus 1) and writes output in the same callback, the same structure Superpowered's latency app uses [R7]. This gives a deterministic
  two-buffer path. AVAudioEngine with sink and source nodes is an acceptable alternative, but RemoteIO is easier to reason about for
  timing.
- **Real-time rules in the callback** - no allocation, locks, Objective-C or Swift class references, logging, or `AVAudioConverter`.
  Use a C resampler (48 kHz to the model rate and back) with a known group delay counted in the latency budget.

### 9.3 Threads

| Thread | Work | Rules |
|---|---|---|
| RemoteIO I/O thread (system) | move samples to/from SPSC ring buffers, apply output gain, limiter and fades, conceal late hops | constant time, never waits |
| Inference worker (1, real-time) | BNNS Graph execute once per hop | time-constraint policy, joined to the RemoteIO `os_workgroup`, preallocated everything |
| Motion queue | `CMHeadphoneMotionManager` handler, writes smoothed yaw to an atomic | low rate, not real time |
| Main / background | session setup, route and interruption handling, Core ML enrollment embedding, HealthKit, UI | may allocate |

### 9.4 Bluetooth policy

- Same session; label the mode "AirPods (high delay)". Show the measured delay. Require ANC (offer the Shortcuts automation), force
  own-voice suppression on, and recommend wired earbuds for conversation. Never claim under 20 ms over Bluetooth.
- Optional experiment - `LSSupportsGameMode` plus a games category in a personal build, foreground only, measuring whether A2DP latency
  drops [R27][R28][R29].

### 9.5 Remote commands and Now Playing

- Register `togglePlayPauseCommand` (pause/resume), `nextTrackCommand` (lock onto the person in the look direction) and
  `previousTrackCommand` (undo or previous target). Set `nowPlayingInfo`. Only valid with the A2DP-only session [R67][R68].

### 9.6 Lifecycle

- Start only in the foreground. Keep the session active while locked. On interruption, fade out and mute. On end with `.shouldResume`,
  try to reactivate. If that fails in the background, post a local notification [R75][R77].
- On route change, mute, revalidate the route, re-assert the built-in mic, re-read the latencies, recalibrate, then fade in.

### 9.7 Safety

- Limiter and gain cap in the I/O callback, fades on every start and route change, mute on speaker/receiver routes. Watch
  `outputVolume` [R81]. Optionally read HealthKit exposure for a weekly summary [R82].

### 9.8 Phase 0 probe app (build before any model work)

1. **Buffer probe** - request 16, 32, 48, 64, 96, 128 and 192 frames at 48 kHz in `.measurement` and `.default`. Log the granted
   `ioBufferDuration`, the actual `inNumberFrames`, the reported latencies, and glitch counts over 10 minutes with the screen on and
   locked. Repeat with Sound Recognition on [R8].
2. **Wired round trip** - acoustic loopback: hold a USB-C EarPod (or a sealed USB-C earbud) against the phone's bottom mic, play a click
   or chirp, cross-correlate the input. This measures exactly mic-in to USB-out (X + 2B). Repeat with the USB-C to 3.5 mm adapter and
   wired earbuds.
3. **AirPods round trip** - same method with an AirPod held at the mic, for Pro 1 and Pro 2, at 0, 10 and 30 minutes (drift [R25]).
4. **Mic matrix** - data source and polar pattern honored per mode and route; tap tests per mic (section 3.2). Confirm the built-in
   mic stays selected while USB-C EarPods (which bring their own mic) are plugged in, and after unplug/replug.
5. **Motion** - `CMHeadphoneMotionManager` rate, jitter and drift in `.playAndRecord` + A2DP, with the screen on and locked (section 5).
6. **Remote commands** - press, double and triple press delivery while Earshot runs; Conversation Awareness on and off; Spatial Audio
   Off / Fixed / Head Tracked latency difference.
7. **Inference timing** - BNNS Graph execute time (median, p99, p99.9, max) for candidate models on the worker thread, joined vs not
   joined to the workgroup, FP16 vs FP32, under thermal load for 1 hour.
8. **Background** - interruption by a phone call, Siri and an alarm while locked; confirm what happens on resume.

---

## 10. Open questions (cannot be answered without the device)

1. iPhone 16 Pro minimum I/O buffer at 48 kHz, and its true built-in-mic to USB-C round trip.
2. Whether `.measurement` honors back/front data sources, cardioid or stereo, and whether it coexists with A2DP output.
3. Whether `CMHeadphoneMotionManager` keeps delivering in `.playAndRecord` + A2DP, and with the screen locked.
4. Whether the iOS 18 Sound Recognition / Vocal Shortcuts buffer bug still exists in iOS 26.1.
5. Whether system Spatial Audio (auto-enabled for Now Playing AURemoteIO apps) adds latency, and whether Conversation Awareness pauses
   Earshot.
6. Whether Game Mode can be triggered by a non-game app on iOS, and how much it lowers AirPods latency.
7. BNNS Graph op coverage for the chosen model file, and real p99.9 execute time on A18 Pro.
8. Whether reactivation after a phone call ever succeeds in the background on iOS 26.1.

---

## References

- [R1] Apple, setPreferredIOBufferDuration(_:) - https://developer.apple.com/documentation/avfaudio/avaudiosession/setpreferrediobufferduration(_:)
- [R2] Apple, inputLatency / outputLatency - https://developer.apple.com/documentation/avfaudio/avaudiosession/inputlatency and https://developer.apple.com/documentation/avfaudio/avaudiosession/outputlatency
- [R3] Apple, AVAudioSession.Mode.measurement - https://developer.apple.com/documentation/avfaudio/avaudiosession/mode-swift.struct/measurement
- [R4] Apple, Audio Session Programming Guide, Configuring Device Hardware - https://developer.apple.com/library/archive/documentation/Audio/Conceptual/AudioSessionProgrammingGuide/OptimizingForDeviceHardware/OptimizingForDeviceHardware.html
- [R5] Superpowered, iOS and Android Audio Latency Test App - https://superpowered.com/latency
- [R6] Superpowered latency data table (retrieved 2026-11-06) - https://superpowered.com/latencydata/data.html
- [R7] SuperpoweredLatency source, iOS/ViewController.mm - https://github.com/superpoweredSDK/SuperpoweredLatency
- [R8] Apple Developer Forums 769245, buffer sizes with Sound Recognition / Vocal Shortcuts - https://developer.apple.com/forums/thread/769245
- [R9] JUCE forum, iOS 18 tryBufferSize - https://forum.juce.com/t/bug-fix-in-iosaudioiodevice-trybuffersize-for-ios-18/62818
- [R10] Apple QA1606, playback during screen lock - https://developer.apple.com/library/archive/qa/qa1606/_index.html
- [R11] Apple Developer Forums 12432, buffer size while screen off - https://developer.apple.com/forums/thread/12432
- [R12] Apple Developer Forums 25197, input/output latency - https://developer.apple.com/forums/thread/25197
- [R13] Onyx3 Round Trip Latency Meter and interface table - https://onyx3.com/LatencyMeter/ and https://onyx3.com/LatencyMeter/ios-audio-guitar-interfaces-latency.html
- [R14] Overloud, How to reduce the audio latency in iOS - https://www.overloud.com/node/325
- [R15] Hearing Review (Jacoti tests), Can Apple AirPods be used as hearing aids - https://hearingreview.com/hearing-products/hearing-aids/can-apple-airpods-used-hearing-aids
- [R16] Electroacoustic Evaluation of Smartphone-Based Hearing Aid Applications (2021) - https://pmc.ncbi.nlm.nih.gov/articles/PMC9149231/
- [R17] coreaudio-api list, total latency with VoiceProcessingIO (2020) - https://mail-archive.com/coreaudio-api@lists.apple.com/msg01682.html
- [R18] Apple, allowBluetoothA2DP - https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetootha2dp
- [R19] Apple, allowBluetoothHFP and allowBluetooth - https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetoothhfp and https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/allowbluetooth
- [R20] Apple, bluetoothHighQualityRecording - https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/bluetoothhighqualityrecording
- [R21] WWDC25 251, Enhance your app's audio recording capabilities - https://developer.apple.com/videos/play/wwdc2025/251/
- [R22] Apple Newsroom, AirPods studio-quality audio recording and camera remote (June 2025) - https://www.apple.com/newsroom/2025/06/airpods-now-more-versatile-with-studio-quality-audio-recording-and-camera-remote/
- [R23] Apple, farFieldInput, dualRoute, AVAudioSessionPortExtensionBluetoothMicrophone - https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/farfieldinput , https://developer.apple.com/documentation/avfaudio/avaudiosession/mode-swift.struct/dualroute , https://developer.apple.com/documentation/avfaudio/avaudiosessionportextensionbluetoothmicrophone
- [R24] Apple Developer Forums 842699, AirPods mic capture and Transparency (Aug 2026) - https://developer.apple.com/forums/thread/842699
- [R25] Apple Developer Forums 679274, AirPods actual latency vs prediction - https://developer.apple.com/forums/thread/679274
- [R26] Stephen Coyle, AirPods Pro 2 latency - https://stephencoyle.net/airpods-pro-2
- [R27] Apple Developer Forums 764070, AirPods audio glitches in Game Mode - https://developer.apple.com/forums/thread/764070
- [R28] Apple Support, Use Game Mode - https://support.apple.com/en-us/105118
- [R29] Apple, LSSupportsGameMode - https://developer.apple.com/documentation/bundleresources/information-property-list/lssupportsgamemode
- [R30] Apple Newsroom, AirPods introduce convenient ways to communicate and interact (June 2024) - https://www.apple.com/newsroom/2024/06/airpods-introduce-convenient-ways-to-communicate-and-interact/
- [R31] Apple Developer Forums 741513, built-in mic plus AirPods playback - https://developer.apple.com/forums/thread/741513
- [R32] Apple Developer Forums 784318, Bluetooth output plus built-in mic input - https://developer.apple.com/forums/thread/784318
- [R33] Apple Developer Forums 853278, A2DP to HFP stall on iOS 27.0.1 - https://developer.apple.com/forums/thread/853278
- [R34] RTINGS comparisons (AirPods Pro 2 vs AirPods 4; AirPods Pro vs Pro 2) - https://www.rtings.com/headphones/tools/compare/apple-airpods-pro-2nd-generation-apple-airpods-4/29490/74581 and https://rtings.com/headphones/tools/compare/apple-airpods-pro-apple-airpods-pro-2nd-generation/1625/29490
- [R35] Apple, AVAudioSessionDataSourceDescription and AVAudioSession.PolarPattern - https://developer.apple.com/documentation/avfaudio/avaudiosessiondatasourcedescription and https://developer.apple.com/documentation/avfaudio/avaudiosession/polarpattern
- [R36] Apple sample, Capturing stereo audio from built-in microphones - https://developer.apple.com/documentation/avfaudio/capturing-stereo-audio-from-built-in-microphones
- [R37] WWDC20 10226, Record stereo audio with AVAudioSession - https://developer.apple.com/videos/play/wwdc2020/10226/
- [R38] Apple QA1799, Microphone Selection - https://developer.apple.com/library/archive/qa/qa1799/_index.html
- [R39] Apple Developer Forums 695892, playback and recording without processing - https://developer.apple.com/forums/thread/695892
- [R40] WWDC23 10235, What's new in voice processing - https://developer.apple.com/videos/play/wwdc2023/10235/
- [R41] Apple, setPrefersEchoCancelledInput(_:) - https://developer.apple.com/documentation/avfaudio/avaudiosession/setprefersechocancelledinput(_:)
- [R42] Apple, AVCaptureDevice.MicrophoneMode and preferredMicrophoneMode - https://developer.apple.com/documentation/avfoundation/avcapturedevice/microphonemode
- [R43] Apple Newsroom, Apple debuts iPhone 16 Pro (Sept 2024) - https://www.apple.com/newsroom/2024/09/apple-debuts-iphone-16-pro-and-iphone-16-pro-max/
- [R44] WWDC24 10211, Support real-time ML inference on the CPU - https://developer.apple.com/videos/play/wwdc2024/10211/
- [R45] WWDC25 276, What's new in BNNS Graph - https://developer.apple.com/videos/play/wwdc2025/276/
- [R46] Apple, BNNS library index; BNNSGraphCompileFromFile; BNNSGraphContextExecute; BNNSGraphContextMakeStreaming; BNNSGraphContextSetStreamingAdvanceCount - https://developer.apple.com/documentation/accelerate/bnns-library , https://developer.apple.com/documentation/accelerate/bnnsgraphcompilefromfile(_:_:_:) , https://developer.apple.com/documentation/accelerate/bnnsgraphcontextexecute(_:_:_:_:_:_:) , https://developer.apple.com/documentation/accelerate/bnnsgraphcontextmakestreaming(_:_:_:_:) , https://developer.apple.com/documentation/accelerate/bnnsgraphcontextsetstreamingadvancecount(_:_:)
- [R47] Apple, BNNSGraph, BNNSGraph.Builder, Builder.Tensor, Builder.Intent, BNNSGraph.Context - https://developer.apple.com/documentation/accelerate/bnnsgraph , https://developer.apple.com/documentation/accelerate/bnnsgraph/builder , https://developer.apple.com/documentation/accelerate/bnnsgraph/builder/tensor , https://developer.apple.com/documentation/accelerate/bnnsgraph/builder/intent , https://developer.apple.com/documentation/accelerate/bnnsgraph/context
- [R48] Apple sample, Supporting real-time ML inference on the CPU - https://developer.apple.com/documentation/accelerate/supporting-real-time-ml-inference-on-the-cpu
- [R49] Apple, MLState - https://developer.apple.com/documentation/coreml/mlstate
- [R50] coremltools, Stateful Models - https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
- [R51] WWDC23 10049, Improve Core ML integration with async prediction - https://developer.apple.com/videos/play/wwdc2023/10049/
- [R52] Secondary Core ML/ANE timing - https://github.com/aosama/astronomical/issues/534 and FaceLiVTv2 (arXiv 2604.09127) https://arxiv.org/pdf/2604.09127
- [R53] Apple, BLASSetThreading(_:) - https://developer.apple.com/documentation/accelerate/blassetthreading(_:)
- [R54] Hello SME! (M4 SME vs Neon, arXiv 2409.18779) https://arxiv.org/abs/2409.18779 ; A18 Pro SME report https://beebom.com/a18-pro-geekbench-score-snapdragon-8-gen-4-score-leaks
- [R55] WWDC20 10224, Meet Audio Workgroups - https://developer.apple.com/videos/play/wwdc2020/10224/
- [R56] Apple, Workgroup Management; Understanding Audio Workgroups; Adding Parallel / Asynchronous Real-Time Threads; kAudioOutputUnitProperty_OSWorkgroup; AUAudioUnit.osWorkgroup - https://developer.apple.com/documentation/audiotoolbox/workgroup-management , https://developer.apple.com/documentation/audiotoolbox/understanding-audio-workgroups , https://developer.apple.com/documentation/audiotoolbox/adding-parallel-real-time-threads-to-audio-workgroups , https://developer.apple.com/documentation/audiotoolbox/adding-asynchronous-real-time-threads-to-audio-workgroups , https://developer.apple.com/documentation/audiotoolbox/kaudiooutputunitproperty_osworkgroup , https://developer.apple.com/documentation/audiotoolbox/auaudiounit/osworkgroup
- [R57] Apple TN2169, High Precision Timers in iOS / OS X - https://developer.apple.com/library/archive/technotes/tn2169/_index.html
- [R58] Apple Developer Forums 735944, retrieve audio workgroup in iOS - https://developer.apple.com/forums/thread/735944
- [R59] Apple, CMHeadphoneMotionManager - https://developer.apple.com/documentation/coremotion/cmheadphonemotionmanager
- [R60] WWDC23 10179, What's new in Core Motion - https://developer.apple.com/videos/play/wwdc2023/10179
- [R61] Apple Support, Control Spatial Audio and head tracking on AirPods - https://support.apple.com/guide/airpods/control-spatial-audio-and-head-tracking-dev00eb7e0a3/web
- [R62] Community rate reports - https://github.com/andreaspitzer/subvox/issues/3 and https://anandchowdhary.com/notes/2025/airpods-based-head-motion-racing
- [R63] Apple Developer Forums 681301, IMU data and audio simultaneously - https://developer.apple.com/forums/thread/681301
- [R64] Apple Developer Forums 682768, CMHeadphoneMotionManager in background - https://developer.apple.com/forums/thread/682768
- [R65] Apple Developer Forums 844940, yaw jumps when Spatial Audio becomes active - https://developer.apple.com/forums/thread/844940
- [R66] Apple, com.apple.developer.coremotion.head-pose - https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.coremotion.head-pose
- [R67] Apple Developer Forums 761549 (DTS reply), headset button in a call - https://developer.apple.com/forums/thread/761549
- [R68] WWDC19 501, Reaching the Big Screen with AirPlay 2 (Now Playing rules) - https://developer.apple.com/videos/play/wwdc2019/501/
- [R69] Apple, Handling external player events notifications - https://developer.apple.com/documentation/mediaplayer/handling-external-player-events-notifications
- [R70] WWDC23 10233, Enhance your app's audio experience with AirPods - https://developer.apple.com/videos/play/wwdc2023/10233
- [R71] Apple Support, Pause, skip, and adjust volume with AirPods - https://support.apple.com/en-us/102628
- [R72] WWDC25 253, Enhancing your camera experience with capture controls - https://developer.apple.com/videos/play/wwdc2025/253/
- [R73] Apple, AVAudioApplication.inputMuteStateChangeNotification - https://developer.apple.com/documentation/avfaudio/avaudioapplication/inputmutestatechangenotification
- [R74] Apple, AVAudioSession.Category.playAndRecord - https://developer.apple.com/documentation/avfaudio/avaudiosession/category-swift.struct/playandrecord
- [R75] Apple Developer Forums 774784 (DTS reply, Feb 2025), voice calls in the background - https://developer.apple.com/forums/thread/774784
- [R76] Apple Developer Forums 106415, microphone background service - https://developer.apple.com/forums/thread/106415
- [R77] Apple, Handling audio interruptions - https://developer.apple.com/documentation/avfaudio/handling-audio-interruptions
- [R78] Apple, AVAudioSession.InterruptionReason - https://developer.apple.com/documentation/avfaudio/avaudiosession/interruptionreason
- [R79] Apple, setPrefersNoInterruptionsFromSystemAlerts(_:) and setPrefersInterruptionOnRouteDisconnect(_:) - https://developer.apple.com/documentation/avfaudio/avaudiosession/setprefersnointerruptionsfromsystemalerts(_:) and https://developer.apple.com/documentation/avfaudio/avaudiosession/setprefersinterruptiononroutedisconnect(_:)
- [R80] Apple Support, Headphone notifications and headphone audio levels - https://support.apple.com/en-us/108968 and https://support.apple.com/guide/iphone/check-your-headphone-audio-levels-iph0596a9152/ios
- [R81] Apple, AVAudioSession.outputVolume - https://developer.apple.com/documentation/avfaudio/avaudiosession/outputvolume
- [R82] Apple, headphoneAudioExposure and headphoneAudioExposureEvent - https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/headphoneaudioexposure and https://developer.apple.com/documentation/healthkit/hkcategorytypeidentifier/headphoneaudioexposureevent
- [R83] Apple, HKAudiogramSample, HKAudiogramSensitivityPoint, HKAudiogramSensitivityTest, audiogramSampleType() - https://developer.apple.com/documentation/healthkit/hkaudiogramsample , https://developer.apple.com/documentation/healthkit/hkaudiogramsensitivitypoint , https://developer.apple.com/documentation/healthkit/hkaudiogramsensitivitytest , https://developer.apple.com/documentation/healthkit/hkobjecttype/audiogramsampletype()
- [R84] Apple Support, Take a Hearing Test with AirPods Pro 2 - https://support.apple.com/en-au/120991
- [R85] Apple, Supported capabilities (iOS) - https://developer.apple.com/help/account/reference/supported-capabilities-ios
- [R86] Apple, Compare memberships (Personal Team limits) - https://developer.apple.com/support/compare-memberships/
- [R87] Apple, com.apple.developer.sustained-execution - https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.sustained-execution
- [R88] Apple Support, Use Adaptive Audio with your AirPods (Conversation Awareness) - https://support.apple.com/en-us/104979
- [R89] Apple Developer Forums 769565, API to switch AirPods Pro 2 mode - https://developer.apple.com/forums/thread/769565
- [R90] iMore, Shortcuts and automations for AirPods Pro - https://www.imore.com/great-shortcuts-and-automations-airpods-pro
- [R91] Apple, ProcessInfo.ThermalState - https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.enum
- [R92] Apple, AVAudioSinkNode initWithRealtimeSafeReceiverBlock: (iOS 27) - https://developer.apple.com/documentation/avfaudio/avaudiosinknode/initwithrealtimesafereceiverblock:
- [R93] Apple, AVAudioIONode (audioUnit, presentationLatency) - https://developer.apple.com/documentation/avfaudio/avaudioionode
- [R94] Apple, AVAudioSession (topic list incl. allowHapticsAndSystemSoundsDuringRecording) - https://developer.apple.com/documentation/avfaudio/avaudiosession
