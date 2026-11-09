// Phase0Runner+Runs.swift - the audio-side Phase 0 runs (30-ios 15.3 bullets): buffers, tap test,
// latency and probe level, motion, stem presses, locked route changes, alarm classifier, mic level.
// Main actor only. Each function starts one run through launchRun; the shared plumbing (ticks,
// screen label, record, checkRoute) is in Phase0Runner.swift and the math in Phase0Math.swift.

import AVFoundation
import UIKit

extension Phase0Runner {
    // MARK: - Buffers (request 32 to 192 frames, 60 s each; then 10 min at 64 frames locked)

    /// Every buffer size not yet measured for `route`, one after the other (about 1 min each).
    func runBuffers(route: String) {
        let sizes = Phase0Plan.bufferSizes.filter {
            !progress.isFinished(.buffers, Phase0Plan.bufferKey(route: route, frames: $0))
        }
        launchRun(.buffers, title: Copy.phase0StepTitle(.buffers), prompt: Copy.phase0KeepOpen,
                  seconds: Double(sizes.count * 60)) { r in
            defer { r.audio?.stop() } // the next Start asks for 64 frames again (putBack)
            for (i, frames) in sizes.enumerated() {
                r.prompt(Copy.phase0BufferProgress(frames: frames, index: i + 1, of: sizes.count))
                let row = try await r.measureBuffer(route: route, frames: frames, seconds: 60)
                r.record(.buffers, item: Phase0Plan.bufferKey(route: route, frames: frames)) { $0.buffers.append(row) }
            }
            r.message = Copy.phase0BuffersDone
        }
    }

    /// 10 min at 64 frames with the screen locked.
    func runBufferLocked(route: String) {
        launchRun(.buffers, title: Copy.phase0StepTitle(.buffers), prompt: Copy.phase0LockNow, seconds: 600) { r in
            let row = try await r.measureBuffer(route: route, frames: 64, seconds: 600)
            let item = row.screen == "locked" ? Phase0Plan.bufferLockedItem : nil
            r.record(.buffers, item: item) { $0.buffers.append(row) }
            r.message = item == nil ? Copy.phase0ScreenWasOn : Copy.phase0Saved
        }
    }

    /// Session off, set the request, on; then `seconds` of underrun counting (screen state from this
    /// part only). The request goes back to 64 frames when the run ends (putBack).
    private func measureBuffer(route: String, frames: Int, seconds: Int) async throws -> Phase0BufferRow {
        guard let audio else { throw Phase0Error.notAttached }
        if audio.state != .idle { audio.stop() }
        SessionConfig.requestedFrames = frames
        try await ensureListening()
        try checkRoute(route)
        activeSeconds = 0
        inactiveSeconds = 0
        var total = CounterTotal(start: audio.stats.underruns)
        var cumulative: [Int] = []
        try await ticks(seconds) { _ in
            total.add(audio.stats.underruns)
            cumulative.append(total.total)
        }
        try checkRoute(route) // the AirPods are still the output
        let rt = audio.route
        return Phase0BufferRow(route: route, mode: Phase0Math.modeName(SessionConfig.mode.rawValue),
                               requested: frames, granted: rt.ioFrames, callback: audio.stats.lastFrames,
                               rate: Int(rt.sampleRate), inLatMS: Phase0Math.ms(rt.inputLatency), outLatMS: Phase0Math.ms(rt.outputLatency),
                               underruns60s: Phase0Math.worstWindow(cumulative: cumulative), seconds: seconds,
                               screen: screenLabel, underruns: total.total)
    }

    // MARK: - Tap test (phone bottom edge, then the right AirPod; 20 dB rule)

    func runTapTest(route: String) {
        launchRun(.tapTest, title: Copy.phase0StepTitle(.tapTest), prompt: Copy.phase0TapPhone, seconds: 12) { r in
            guard let audio = r.audio else { throw Phase0Error.notAttached }
            try await r.ensureListening()
            guard audio.route.isBluetooth else { throw Phase0Error.needsAirPods }
            try r.checkRoute(route)
            try await Task.sleep(nanoseconds: 1_000_000_000) // let the meter settle after a start
            let phone = try await r.loudestPeak(seconds: 5)
            r.prompt(Copy.phase0TapAirPod)
            let airpod = try await r.loudestPeak(seconds: 5)
            let pass = Phase0Math.tapPasses(phonePeakDB: phone, airpodPeakDB: airpod)
            let row = Phase0TapRow(route: route, phoneTapPeakDB: phone, airpodTapPeakDB: airpod,
                                   dataSource: audio.route.inputDataSource, pass: pass)
            r.record(.tapTest, item: route) { $0.tapTest.append(row) }
            r.message = Copy.phase0TapResult(phone: phone, airpod: airpod, pass: pass)
        }
    }

    /// The loudest input-meter peak (dBFS) over `seconds`, read every 50 ms (the meter itself
    /// covers every sample of the 16 kHz history, 10 times a second).
    private func loudestPeak(seconds: Double) async throws -> Double {
        var loudest: Float = -100
        let steps = Int(seconds * 20)
        for _ in 0..<steps {
            try await Task.sleep(nanoseconds: 50_000_000)
            loudest = max(loudest, audio?.meter.peakDB ?? -100)
            addElapsed(0.05)
        }
        return Double(loudest)
    }

    // MARK: - Latency (0, 10, 30 min after connect) and the probe level (5 tries per cell)

    /// One probe with the owner's method and room; finishes a 0 / 10 / 30 min item when it worked.
    func runLatency(route: String, method: String, ambient: Int) {
        launchRun(.latency, title: Copy.phase0StepTitle(.latency), prompt: Copy.phase0ProbeQuiet, seconds: 9) { r in
            let row = try await r.probeOnce(route: route, method: method, ambient: ambient)
            let slot = Phase0Math.minuteSlot(row.minute)
            let item = (row.ok ?? false) ? slot.map { Phase0Plan.latencyKey(route: route, method: method, slot: $0) } : nil
            r.record(.latency, item: item) { $0.latency.append(row) }
            r.message = Copy.phase0ProbeResult(ok: row.ok ?? false, loopMS: row.loopMS, minute: row.minute)
        }
    }

    /// Five probes in a row at this build's level, for the success rate of one cell.
    func runProbeCell(route: String, ambient: Int, method: String) {
        let tries = Phase0Plan.probesPerCell
        launchRun(.probeLevel, title: Copy.phase0StepTitle(.probeLevel), prompt: Copy.phase0ProbeQuiet,
                  seconds: Double(tries * 10)) { r in
            var ok = 0
            for i in 0..<tries {
                r.prompt(Copy.phase0ProbeTry(i + 1, of: tries))
                let row = try await r.probeOnce(route: route, method: method, ambient: ambient)
                if row.ok ?? false { ok += 1 }
                let done = r.cellTries(route: route, ambient: ambient, method: method) + 1 >= tries
                let key = Phase0Plan.probeKey(route: route, ambient: ambient, level: Self.probeDBFS, method: method)
                r.record(.probeLevel, item: done ? key : nil) { $0.latency.append(row) }
                try await Task.sleep(nanoseconds: 1_000_000_000)
            }
            r.message = Copy.phase0CellResult(ok: ok, of: tries, pick: Phase0Math.probePick(r.report.latency))
        }
    }

    /// Rows already in the file for this cell (this build's level).
    func cellTries(route: String, ambient: Int, method: String) -> Int {
        report.latency.filter {
            $0.route == route && $0.ambientDB == ambient && $0.method == method && $0.probeDBFS == Self.probeDBFS
        }.count
    }

    /// Runs AudioController.measureLatency once and turns its raw outcome into a row.
    private func probeOnce(route: String, method: String, ambient: Int) async throws -> Phase0LatencyRow {
        let audio = try await ensureListening()
        try checkRoute(route)
        if progress.connectedAt[route] == nil { markConnected(route: route) }
        audio.phase0.lastProbe = nil
        do { try await audio.measureLatency(method: ProbeMethod(rawValue: method) ?? .inEar) }
        catch { audio.note("phase0 probe: \(error)") } // a failed probe is a row too
        try Task.checkCancellation()
        // No outcome: the probe never started (not running, wrong route). Abandoned because the audio
        // changed: not a fair try either. Neither may count against the probe level.
        guard let o = audio.phase0.lastProbe else { throw Phase0Error.probeNotRun }
        if o.why == "audio changed" { throw Phase0Error.engineChanged }
        return Phase0LatencyRow(route: route, method: method, minute: minutesSinceConnect(route), ambientDB: ambient,
                                probeDBFS: Self.probeDBFS, loopMS: o.loopMS, spreadMS: o.spreadMS, psrDB: o.psrDB,
                                okSweeps: o.okSweeps, reportedMS: audio.latency.reportedMS, ok: o.ok)
    }

    // MARK: - Motion (10 min screen on, 10 min locked, inside the running session)

    func runMotion(route: String) {
        launchRun(.motion, title: Copy.phase0StepTitle(.motion), prompt: Copy.phase0MotionHowTo, seconds: 600) { r in
            try await r.ensureListening()
            try r.checkRoute(route)
            var gaps = MotionGaps()
            r.selector?.head.onSample = { gaps.add($0) }
            try await r.ticks(600)
            r.selector?.head.onSample = nil
            let label = r.screenLabel
            let row = Phase0MotionRow(route: route, screen: label, minutes: 10, rateHz: Phase0Math.round1(gaps.rateHz),
                                      gapsOver200ms: gaps.gapsOver200ms,
                                      maxGapMS: Phase0Math.round1(gaps.maxGapMS(end: HeadTracker.now)))
            r.record(.motion, item: label == "mixed" ? nil : label) { $0.motion.append(row) }
            r.message = Copy.phase0MotionResult(rateHz: row.rateHz, gaps: row.gapsOver200ms, screen: label)
        }
    }

    // MARK: - Stem presses (screen locked; counted only, never acted on)

    /// One minute in which the owner presses once, twice and three times with the screen locked.
    func runStem(route: String, combo: String) {
        launchRun(.stem, title: Copy.phase0StepTitle(.stem), prompt: Copy.phase0StemHowTo, seconds: 75) { r in
            try await r.ensureListening()
            try r.checkRoute(route)
            guard let remote = r.selector?.remote else { throw Phase0Error.notAttached }
            remote.countOnly = true
            let before = remote.counts
            try await r.ticks(75)
            let after = remote.counts
            func got(_ p: StemPress) -> Bool { (after[p] ?? 0) > (before[p] ?? 0) }
            let parts = Self.comboParts(combo)
            let row = Phase0StemRow(route: route, convAwareness: parts.ca, spatial: parts.spatial, screen: r.screenLabel,
                                    single: got(.single), double: got(.double), triple: got(.triple),
                                    singleWhilePausedRate0: nil)
            r.record(.stem, item: row.screen == "locked" ? combo : nil) { $0.stem.append(row) }
            r.message = Copy.phase0StemResult(single: got(.single), double: got(.double), triple: got(.triple),
                                              screen: row.screen)
        }
    }

    /// Earshot paused with Now Playing at rate 0; does a single press still arrive (30-ios 11)?
    func runStemPaused(route: String, combo: String) {
        launchRun(.stem, title: Copy.phase0StepTitle(.stem), prompt: Copy.phase0StemPausedHowTo, seconds: 45) { r in
            let audio = try await r.ensureListening()
            try r.checkRoute(route)
            guard let selector = r.selector else { throw Phase0Error.notAttached }
            selector.remote.countOnly = true
            selector.remote.pausedRateOverride = 0
            audio.pause()
            selector.refreshNowPlaying()
            let before = selector.remote.counts[.single] ?? 0
            try await r.ticks(45)
            let arrived = (selector.remote.counts[.single] ?? 0) > before
            selector.remote.pausedRateOverride = nil
            audio.resume()
            selector.refreshNowPlaying()
            let parts = Self.comboParts(combo)
            let row = Phase0StemRow(route: route, convAwareness: parts.ca, spatial: parts.spatial, screen: r.screenLabel,
                                    single: nil, double: nil, triple: nil, singleWhilePausedRate0: arrived)
            r.record(.stem, item: row.screen == "locked" ? Phase0Plan.stemPausedItem : nil) { $0.stem.append(row) }
            r.message = Copy.phase0StemPausedResult(arrived)
        }
    }

    /// "ca_on/spatial_off" -> ("on", "off").
    nonisolated static func comboParts(_ combo: String) -> (ca: String, spatial: String) {
        (combo.hasPrefix("ca_on") ? "on" : "off", combo.hasSuffix("spatial_on") ? "on" : "off")
    }

    // MARK: - Locked: 10 route changes and 1 forced swap, screen locked

    func runLocked() {
        launchRun(.locked, title: Copy.phase0StepTitle(.locked), prompt: Copy.phase0LockedHowTo, seconds: 0) { r in
            let audio = try await r.ensureListening()
            let base = audio.phase0.routeChanges
            let baseSwaps = audio.phase0.swaps
            var tracker = ResumeTracker()
            var forced = 0
            try await r.ticks(15 * 60, untilDone: true) { k in
                if k == 30 { audio.forceSwap(); forced += 1 } // 30 s in: time to lock the screen first
                let changes = audio.phase0.routeChanges - base + audio.phase0.swaps - baseSwaps
                let running = audio.state == .running && !audio.routeCheckPending
                tracker.sample(now: Double(k), changes: changes, routeAllowed: audio.route.allowed,
                               running: running, userPaused: audio.userPaused)
            }
            let row = Phase0LockedRow(routeChanges: audio.phase0.routeChanges - base, forcedSwaps: forced,
                                      resumedWithoutTap: tracker.resumedWithoutTap, neededTap: tracker.neededTap)
            r.record(.locked, item: "routes_and_swap") { $0.locked.append(row) }
            r.message = Copy.phase0LockedResult(changes: row.routeChanges, resumed: row.resumedWithoutTap,
                                                tap: row.neededTap)
        }
    }

    // MARK: - Alarm classifier, 60 s with the screen locked

    func runAlarm() {
        launchRun(.alarm, title: Copy.phase0StepTitle(.alarm), prompt: Copy.phase0LockIn20, seconds: 80) { r in
            let audio = try await r.ensureListening()
            guard audio.alarmEnabled else { throw Phase0Error.alarmOff }
            try await r.ticks(20) // time to lock the screen
            r.activeSeconds = 0
            r.inactiveSeconds = 0
            r.prompt(Copy.phase0AlarmRunning)
            var last = UInt64(audio.alarmListener.windows)
            var windows = 0
            try await r.ticks(60) { _ in
                let now = UInt64(audio.alarmListener.windows)
                windows += MinuteTelemetry.delta(now: now, last: last) // a listener restart starts at 0
                last = now
            }
            let expected = Phase0Math.expectedAlarmWindows(seconds: 60, overlap: PowerPolicy.alarmOverlap(audio.powerLevel))
            let row = Phase0AlarmRow(screen: r.screenLabel, seconds: 60, windows: windows,
                                     ok: Phase0Math.alarmOK(windows: windows, expected: expected))
            r.record(.alarm, item: row.screen == "locked" ? "locked" : nil) { $0.alarm.append(row) }
            r.message = Copy.phase0AlarmResult(windows: windows, expected: expected, screen: row.screen)
        }
    }

    // MARK: - Mic level: someone talking at about 1 m, VAD speech frames only

    func runMicLevel() {
        launchRun(.micLevel, title: Copy.phase0StepTitle(.micLevel), prompt: Copy.phase0MicHowTo, seconds: 15) { r in
            let audio = try await r.ensureListening()
            guard let engine = audio.engine, let model = r.launch?.model else { throw Phase0Error.notListening }
            var tap = engine.makeInputTap()
            var pcm: [Float] = []
            try await r.ticks(15) { _ in
                guard audio.engine === engine else { throw Phase0Error.engineChanged }
                pcm += tap.read(maxSamples: 32_000)
            }
            let mask = model.vad(pcm)
            guard let level = Phase0Math.micLevel(pcm: pcm, mask: mask), level.speechS >= 2 else {
                throw Phase0Error.notEnoughSpeech
            }
            let row = Phase0MicLevelRow(distanceM: 1.0, speechRmsDBFS: Phase0Math.round1(level.rmsDB),
                                        speechPeakDBFS: Phase0Math.round1(level.peakDB), speechS: Phase0Math.round1(level.speechS))
            r.record(.micLevel, item: "1m") { $0.micLevel.append(row) }
            r.message = Copy.phase0MicResult(rms: row.speechRmsDBFS, peak: row.speechPeakDBFS)
        }
    }
}
