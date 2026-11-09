// Copy+Phase0.swift - the Phase 0 screen's strings (I-1, 30-ios 15.3). Part of `Copy` (30-ios
// 12.1: every user-facing string lives in Copy); a separate file only so Copy.swift stays readable.
// CopyLintTests lints this file's lines and `phase0Samples` the same way as Copy.swift.
//
// The Phase 0 screen is for the owner, so it names developer things (frames, dBFS) plainly.
// Each step text says exactly what to do with the phone and the AirPods.

import Foundation

extension Copy {
    // MARK: Screen
    static let phase0Title = "Phase 0"
    static let phase0Intro = "Guided measurements for the Phase 0 file. Every step can be skipped and picked up later; each finished item is saved at once."
    static func phase0File(_ name: String) -> String { "File: \(name)" }
    static func phase0StepsSettled(_ n: Int, of total: Int) -> String { "\(n) of \(total) steps done or skipped" }
    static let phase0NewFile = "Start a new Phase 0 file"
    static let phase0Notes = "Notes"
    static let phase0StubWarning = "This build uses the test engine. Compute and power rows only check the plumbing; run them again with the real engine."
    static let phase0NeedsRealEngine = "Needs the real engine for real numbers"
    static let phase0Items = "Items"
    static let phase0Output = "Output"
    static let phase0Run = "Run"
    static let phase0Skip = "Skip this step"
    static let phase0Finish = "Finish this step"
    static let phase0Reopen = "Reopen this step"
    static func phase0Elapsed(_ elapsed: Int, _ total: Int) -> String {
        total > 0 ? "\(elapsed) of \(total) s" : "\(elapsed) s - tap Done when finished"
    }

    static func phase0Status(_ s: Phase0StepStatus) -> String {
        switch s {
        case .todo: return "To do"
        case .inProgress: return "In progress"
        case .done: return "Done"
        case .skipped: return "Skipped"
        }
    }

    static func phase0RouteName(_ label: String) -> String {
        switch label {
        case "airpods_pro1": return airPodsPro1
        case "airpods_pro2": return airPodsPro2
        case "no_headphones": return "No headphones"
        case "wired": return "Wired headphones"
        default: return "AirPods (which ones is not known yet)"
        }
    }

    static let phase0Quiet = "Quiet room (about 45 dB)"
    static let phase0Loud = "Loud room (about 70 dB, babble playing)"
    static let phase0MethodA = "Method A - AirPods in"
    static let phase0MethodB = "Method B - ear detection off"
    static func phase0Combo(_ combo: String) -> String {
        let p = Phase0Runner.comboParts(combo)
        return "Conversation Awareness \(p.ca), Spatial Audio \(p.spatial)"
    }

    // MARK: Steps (title and what the owner does)
    static func phase0StepTitle(_ id: Phase0StepID) -> String {
        switch id {
        case .buffers: return "Buffer sizes"
        case .tapTest: return "Microphone tap test"
        case .latency: return "Delay at 0, 10 and 30 min"
        case .probeLevel: return "Probe level"
        case .motion: return "Head motion"
        case .stem: return "Stem presses"
        case .locked: return "Route changes while locked"
        case .wespeaker: return "Voice check speed"
        case .alarm: return "Alarm listener while locked"
        case .micLevel: return "Speech level at 1 m"
        case .compute: return "Engine compute"
        case .power: return "Battery and heat"
        }
    }

    static func phase0StepWhat(_ id: Phase0StepID) -> String {
        switch id {
        case .buffers:
            return "Earshot restarts its audio 6 times, asking for 32, 48, 64, 96, 128 and 192 frames, and counts dropouts for 1 minute each. Do it once with AirPods Pro 2, once with AirPods Pro 1 and once with no headphones (sound stays off). Keep the app open. Then run 10 minutes at 64 frames with the screen locked."
        case .tapTest:
            return "Put the AirPods in. When asked, tap the bottom edge of the iPhone 3 times, then tap the right AirPod 3 times. It passes when the iPhone tap is at least 20 dB louder, which shows the iPhone microphone is the one listening."
        case .latency:
            return "Tap \"AirPods just connected\" right after putting them in. Measure right away, after 10 minutes and after 30 minutes, with method A and with method B, for both AirPods. Method A: \(measureHowTo) Method B: \(measureHowToB)"
        case .probeLevel:
            return "Five probes per box: each method, a quiet room and a loud room (play babble from another device at about 70 dB), both AirPods. The lowest level that works at least 4 times in 5 in both rooms becomes the probe level. The louder levels need their own builds; check them at half volume for comfort."
        case .motion:
            return "Wear the AirPods. Run 10 minutes with the screen on, moving your head now and then, and 10 minutes with the screen locked."
        case .stem:
            return "Set Conversation Awareness and Spatial Audio in the AirPods settings to match the box, start the run, lock the screen, then press the stem once, wait 3 seconds, press twice, wait 3 seconds, press three times. Unlock when the time is up. Earshot only counts the presses during this step."
        case .locked:
            return "Start, then lock the screen. After 30 seconds Earshot swaps its engine once by itself. Then take the AirPods out and put them back, and open and close the case, 10 times in all. Unlock and tap Done. The step records whether sound came back each time without a tap."
        case .wespeaker:
            return "Times the voice check on a 1.5 s and a 10 s clip. Run it once with the app open and once with the screen locked (you get 20 seconds to lock it)."
        case .alarm:
            return "Start, lock the screen within 20 seconds and wait 1 minute. Earshot checks that the alarm listener keeps classifying sound while locked."
        case .micLevel:
            return "Ask someone to talk normally about 1 m from the iPhone for 15 seconds. Only the parts with speech are measured."
        case .compute:
            return "Earshot fills 1 or 3 slots with the test voices and records the time of every model step. Lock the screen and leave the iPhone alone (60 or 30 minutes). For the 1,104-frame run, turn on Sound Recognition (Settings, Accessibility) first. Earshot stops listening when the run ends."
        case .power:
            return "Unplug the charger, fully charge first if you can. Earshot fills 1 or 3 slots and reads the battery every minute. Lock the screen and leave the iPhone in a pocket or on the table (2 hours or 1 hour). Earshot stops listening when the run ends."
        }
    }

    // MARK: Buttons
    static let phase0RunBuffers = "Measure the 6 sizes on this output"
    static let phase0RunBufferLocked = "10 min at 64 frames, screen locked"
    static let phase0RunTap = "Run the tap test"
    static let phase0Connected = "AirPods just connected"
    static func phase0MinutesSinceConnect(_ m: Int) -> String { "Connected \(m) min ago" }
    static let phase0NotConnectedYet = "Not marked connected yet"
    static let phase0RunProbe = "Measure once"
    static let phase0RunCell = "Run 5 probes"
    static let phase0RunMotion = "Run 10 min"
    static let phase0RunStem = "Run 75 s for this setting"
    static let phase0RunStemPaused = "Single press while paused at rate 0 (45 s)"
    static let phase0RunLocked = "Start the locked run"
    static let phase0RunOpen = "Run with the app open"
    static let phase0RunLockedScreen = "Run with the screen locked"
    static let phase0RunAlarm = "Run 1 min locked"
    static let phase0RunMic = "Run 15 s"
    static func phase0RunCompute(_ item: String) -> String {
        let p = item.split(separator: "/")
        guard p.count == 3 else { return item }
        return "\(p[0]) slot\(p[0] == "1" ? "" : "s"), \(p[1]) min, \(p[2]) frames"
    }
    static func phase0RunPower(_ item: String) -> String {
        let p = item.split(separator: "/")
        guard p.count == 2 else { return item }
        return "\(p[0]) slot\(p[0] == "1" ? "" : "s"), \(p[1]) min"
    }
    static func phase0ProbeBuild(_ level: Int) -> String {
        "This build plays the probe at \(level) dBFS. Boxes for other levels need a build with ES_PROBE_DBFS changed; rows from every build add up in the same file."
    }
    static func phase0Pick(_ level: Int?) -> String {
        level.map { "Probe level so far: \($0) dBFS" } ?? "No level qualifies yet"
    }
}

extension Copy {
    // MARK: While a run is going
    static let phase0KeepOpen = "Keep Earshot open and the iPhone still"
    static let phase0LockNow = "Lock the screen now and leave the iPhone alone"
    static let phase0LockIn20 = "Lock the screen within 20 seconds"
    static func phase0LockAndLeave(_ minutes: Int) -> String { "Lock the screen and leave the iPhone alone for \(minutes) min" }
    static let phase0Measuring = "Measuring"
    static func phase0BufferProgress(frames: Int, index: Int, of total: Int) -> String {
        "Asking for \(frames) frames (\(index) of \(total)), keep the app open"
    }
    static let phase0TapPhone = "Tap the bottom edge of the iPhone 3 times now"
    static let phase0TapAirPod = "Now tap the right AirPod 3 times"
    static let phase0ProbeQuiet = "Probe playing - keep still and quiet"
    static func phase0ProbeTry(_ n: Int, of total: Int) -> String { "Probe \(n) of \(total) - keep still" }
    static let phase0MotionHowTo = "Wear the AirPods and move your head now and then"
    static let phase0StemHowTo = "Lock the screen, then press once, twice and three times, 3 seconds apart"
    static let phase0StemPausedHowTo = "Earshot is paused. Lock the screen and press the stem once"
    static let phase0LockedHowTo = "Lock the screen. AirPods out and in, case open and closed, 10 times. Then unlock and tap Done"
    static let phase0AlarmRunning = "Listening for 1 minute - keep the screen locked"
    static let phase0MicHowTo = "Someone talks normally about 1 m from the iPhone"

    // MARK: Results
    static let phase0Saved = "Saved"
    static let phase0BuffersDone = "All sizes on this output saved"
    static let phase0ScreenWasOn = "Saved, but the screen was not locked for most of the run, so the item stays open"
    static func phase0TapResult(phone: Double, airpod: Double, pass: Bool) -> String {
        String(format: "Phone tap %.1f dBFS, AirPod tap %.1f dBFS - %@", phone, airpod, pass ? "pass" : "fail")
    }
    static func phase0ProbeResult(ok: Bool, loopMS: Double?, minute: Int) -> String {
        if ok, let loopMS { return String(format: "Loop %.1f ms at minute %d", loopMS, minute) }
        return "The probe did not get a clear result (saved as a failed try)"
    }
    static func phase0CellResult(ok: Int, of total: Int, pick: Int?) -> String { "\(ok) of \(total) probes worked. \(phase0Pick(pick))" }
    static func phase0MotionResult(rateHz: Double, gaps: Int, screen: String) -> String {
        String(format: "%.1f updates per second, %d gaps over 200 ms (screen %@)", rateHz, gaps, screen)
    }
    static func phase0StemResult(single: Bool, double: Bool, triple: Bool, screen: String) -> String {
        func yn(_ b: Bool) -> String { b ? "yes" : "no" }
        return "Single \(yn(single)), double \(yn(double)), triple \(yn(triple)) (screen \(screen))"
    }
    static func phase0StemPausedResult(_ arrived: Bool) -> String {
        arrived ? "A single press arrived while paused at rate 0" : "No single press arrived while paused at rate 0"
    }
    static func phase0LockedResult(changes: Int, resumed: Int, tap: Int) -> String {
        "\(changes) route changes, sound came back by itself \(resumed) times, needed a tap \(tap) times"
    }
    static func phase0AlarmResult(windows: Int, expected: Int, screen: String) -> String {
        "\(windows) of about \(expected) sound windows classified (screen \(screen))"
    }
    static func phase0MicResult(rms: Double, peak: Double) -> String {
        String(format: "Speech %.1f dBFS RMS, peak %.1f dBFS", rms, peak)
    }
    static func phase0ComputeResult(p99: Int, p999: Int, underruns: Int, allocs: UInt64, mallocArmed: Bool) -> String {
        let a = mallocArmed ? "\(allocs) allocations on the audio threads" : "allocations not counted in this build"
        return "p99 \(p99) us, p99.9 \(p999) us, \(underruns) dropouts, \(a)"
    }
    static func phase0PowerResult(pctPerHour: Double, thermal: String) -> String {
        String(format: "%.1f%% battery per hour, warmest %@", pctPerHour, thermal)
    }
    static func phase0WeSpeakerResult(ok: Bool, ms10: Double?, units: String) -> String {
        guard ok, let ms10 else { return "The voice check failed on this run" }
        return String(format: "10 s clip in %.0f ms on %@", ms10, units)
    }
    static func phase0NoteSaving(_ item: String) -> String {
        "Compute run \(item) hit power saving (thermal or Low Power), so fewer slots ran for part of it. "
    }

    // MARK: Problems
    static let phase0NotReady = "Open Phase 0 from Diagnostics once Earshot is ready"
    static let phase0NotListening = "Earshot could not start listening"
    static func phase0WrongRoute(_ actual: String) -> String { "The iPhone is playing to: \(actual). Switch the output or pick it above." }
    static let phase0NeedsAirPods = "Connect AirPods first"
    static let phase0ProbeNotRun = "The probe could not start. Start listening with the AirPods connected, then try again."
    static let phase0Unplug = "Unplug the charger, then run again"
    static func phase0NeedsSoundRecognition(_ frames: Int) -> String {
        "Callbacks are \(frames) frames. Turn on Sound Recognition (Settings, Accessibility), then run again."
    }
    static let phase0NoWeSpeaker = "The voice check model is not in this build"
    static let phase0NotEnoughSpeech = "Not enough speech heard (2 seconds needed). Try again."
    static let phase0AlarmOff = "Turn on \"Let alarm sounds through\" in Settings first"
    static let phase0Busy = "Another measurement is running"
    static let phase0SaveFailed = "Could not save the Phase 0 file"
    static let phase0Cancelled = "Cancelled - nothing from that run was saved"
    static let phase0RunFailed = "The measurement stopped"

    // MARK: Malloc counter (Diagnostics)
    static let mallocCounterTitle = "Audio thread allocations (debug)"

    /// Every Phase 0 string with sample values, for CopyLintTests.
    static var phase0Samples: [String] {
        var all = [phase0Title, phase0Intro, phase0File("phase0-2026-11-08.json"), phase0StepsSettled(3, of: 12),
                   phase0NewFile, phase0Notes, phase0StubWarning, phase0NeedsRealEngine, phase0Items, phase0Output,
                   phase0Run, phase0Skip, phase0Finish, phase0Reopen, phase0Elapsed(5, 60), phase0Elapsed(5, 0),
                   phase0Quiet, phase0Loud, phase0MethodA, phase0MethodB, phase0Combo("ca_on/spatial_off"),
                   phase0RunBuffers, phase0RunBufferLocked, phase0RunTap, phase0Connected, phase0MinutesSinceConnect(10),
                   phase0NotConnectedYet, phase0RunProbe, phase0RunCell, phase0RunMotion, phase0RunStem,
                   phase0RunStemPaused, phase0RunLocked, phase0RunOpen, phase0RunLockedScreen, phase0RunAlarm,
                   phase0RunMic, phase0RunCompute("3/30/1104"), phase0RunCompute("1/60/64"), phase0RunPower("1/120"),
                   phase0ProbeBuild(-30), phase0Pick(-24), phase0Pick(nil),
                   phase0KeepOpen, phase0LockNow, phase0LockIn20, phase0LockAndLeave(60), phase0Measuring,
                   phase0BufferProgress(frames: 32, index: 1, of: 6), phase0TapPhone, phase0TapAirPod, phase0ProbeQuiet,
                   phase0ProbeTry(2, of: 5), phase0MotionHowTo, phase0StemHowTo, phase0StemPausedHowTo, phase0LockedHowTo,
                   phase0AlarmRunning, phase0MicHowTo, phase0Saved, phase0BuffersDone, phase0ScreenWasOn,
                   phase0TapResult(phone: -12, airpod: -40, pass: true), phase0ProbeResult(ok: true, loopMS: 48.2, minute: 10),
                   phase0ProbeResult(ok: false, loopMS: nil, minute: 0), phase0CellResult(ok: 4, of: 5, pick: -30),
                   phase0MotionResult(rateHz: 25, gaps: 1, screen: "locked"),
                   phase0StemResult(single: true, double: true, triple: false, screen: "locked"),
                   phase0StemPausedResult(true), phase0StemPausedResult(false),
                   phase0LockedResult(changes: 10, resumed: 9, tap: 1), phase0AlarmResult(windows: 39, expected: 39, screen: "locked"),
                   phase0MicResult(rms: -32, peak: -12),
                   phase0ComputeResult(p99: 700, p999: 1500, underruns: 0, allocs: 0, mallocArmed: true),
                   phase0ComputeResult(p99: 0, p999: 0, underruns: 0, allocs: 0, mallocArmed: false),
                   phase0PowerResult(pctPerHour: 4.5, thermal: "fair"), phase0WeSpeakerResult(ok: true, ms10: 80, units: "cpuOnly"),
                   phase0WeSpeakerResult(ok: false, ms10: nil, units: "cpuOnly"), phase0NoteSaving("3/60/64"),
                   phase0NotReady, phase0NotListening, phase0WrongRoute("No headphones"), phase0NeedsAirPods, phase0ProbeNotRun, phase0Unplug,
                   phase0NeedsSoundRecognition(64), phase0NoWeSpeaker, phase0NotEnoughSpeech, phase0AlarmOff, phase0Busy,
                   phase0SaveFailed, phase0Cancelled, phase0RunFailed, mallocCounterTitle]
        for id in Phase0StepID.allCases { all += [phase0StepTitle(id), phase0StepWhat(id)] }
        for s in [Phase0StepStatus.todo, .inProgress, .done, .skipped] { all.append(phase0Status(s)) }
        for r in ["airpods_pro1", "airpods_pro2", "no_headphones", "wired", "airpods"] { all.append(phase0RouteName(r)) }
        return all
    }
}
