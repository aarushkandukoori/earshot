// AudioController+Measure.swift - the latency probe flow and the measure prompts (30-ios 7), the
// 10 Hz stats and input meter poll, the one-second buffer-size check (30-ios 3.1) and the forced
// swap button. Main actor only.
// I-1: every probe also leaves its raw numbers in `phase0.lastProbe` for the Phase 0 runner.

import AVFoundation

/// Peak and RMS of the newest mic audio, for the input meter.
struct InputMeter: Equatable {
    var peakDB: Float = -100
    var rmsDB: Float = -100
}

extension AudioController {
    // MARK: - Latency measurement (30-ios 7)

    /// The screens call this: method A (AirPods in) or method B (ear detection off).
    func measureLatency(method: ProbeMethod) async throws {
        probeMethod = method.rawValue
        try await measureLatency()
    }

    /// Plays the probe sweeps and stores a valid result for this output port.
    func measureLatency() async throws {
        guard state == .running, !routeCheckPending, let e = engine else { throw AppError.notRunning }
        let r = route
        // True while the probe can still be trusted: same engine (a swap gives the render callback
        // a new one, so the old probe would never finish) and no route change since it began.
        func probeStillValid() -> Bool {
            engine === e && !routeCheckPending && route.primaryOutputUID == r.primaryOutputUID
        }
        try e.startLatencyProbe()
        isMeasuring = true
        lastProbeMessage = Copy.measuring
        phase0.lastProbe = nil // I-1: filled below with this probe's raw numbers, success or not
        // The probe replaces the output, and the engine scales it by the mute gain. In setup
        // mode (no own voice yet) we unmute only while the sweeps play (0 to 4.0 s) and mute
        // again before the capture ends at 4.5 s, so the user never hears themselves.
        let wasUnmuted = isUnmuted
        if !wasUnmuted { e.setMuted(false) }
        defer {
            isMeasuring = false
            if !wasUnmuted { e.setMuted(true); isUnmuted = false }
            refreshMute()
        }
        try await Task.sleep(nanoseconds: 4_100_000_000)
        if !wasUnmuted { e.setMuted(true) }
        let deadline = Date().addingTimeInterval(3.9) // 8 s in total
        while Date() < deadline {
            guard probeStillValid() else {
                phase0.lastProbe = ProbeOutcome(ok: false, raw: nil, method: probeMethod, why: "audio changed")
                lastProbeMessage = Copy.probeAudioChanged
                note("probe abandoned: the engine or the route changed")
                throw AppError.probeRejected(Copy.probeAudioChanged)
            }
            guard let result = e.latencyProbeResult() else {
                try await Task.sleep(nanoseconds: 100_000_000) // poll every 100 ms
                continue
            }
            switch result {
            case .failure(let err):
                phase0.lastProbe = ProbeOutcome(ok: false, raw: Self.rawProbe(e), method: probeMethod, why: err.message)
                lastProbeMessage = Copy.probeWeak
                offerMethodB = true // PROBE_WEAK: offer method B (30-ios 7)
                note("probe failed: \(err.message)")
                recordProbe("\(probeMethod) failed: \(err.message)")
                throw AppError.probeRejected(Copy.probeWeak) // the engine's text goes to Diagnostics only
            case .success(let p):
                if let why = LatencyMeter.validate(p, isBluetooth: r.isBluetooth) {
                    phase0.lastProbe = ProbeOutcome(ok: false, raw: Self.rawProbe(e), method: probeMethod, why: "\(why)")
                    lastProbeMessage = Copy.probeWeak
                    offerMethodB = true
                    note("probe rejected: \(why)")
                    recordProbe("\(probeMethod) rejected: \(why)")
                    throw AppError.probeRejected(Copy.probeWeak)
                }
                let m = LatencyMeasurement(loopMS: Double(p.loopMS), spreadMS: Double(p.spreadMS),
                                           psrDB: Double(p.psrDB), method: probeMethod, date: Date())
                latencyStore.save(m, forPort: r.primaryOutputUID)
                phase0.lastProbe = ProbeOutcome(ok: true, raw: Self.rawProbe(e), method: probeMethod, why: "")
                lastProbeMessage = String(format: "loop %.1f ms, spread %.1f ms, psr %.1f dB, %d sweeps",
                                          p.loopMS, p.spreadMS, p.psrDB, p.okSweeps)
                note("probe ok (\(probeMethod)): \(lastProbeMessage)")
                recordProbe("\(probeMethod) ok: \(lastProbeMessage)")
                offerMethodB = false
                latencyPrompts.dismiss()
                latencyPrompt = nil
                updateLatencyDisplay()
                return
            }
        }
        lastProbeMessage = Copy.probeWeak
        offerMethodB = true
        phase0.lastProbe = ProbeOutcome(ok: false, raw: nil, method: probeMethod, why: "timeout")
        throw AppError.probeTimeout
    }

    /// The engine's raw probe numbers, also for a failed probe (whose Result carries only text).
    /// es_engine_probe_result caches its answer, so asking again returns the same numbers.
    static func rawProbe(_ e: EarshotEngine) -> es_probe_result? {
        var r = es_probe_result()
        let status = es_engine_probe_result(e.cEngine, &r)
        return status == ES_OK || status == ES_ERR_PROBE_WEAK ? r : nil
    }

    // MARK: - Measure prompts (30-ios 7 "When")

    /// Every settled route goes through here (start and route changes): a connect offers the
    /// measurement, a bad route forgets the connection.
    func routeForPrompts(_ r: RouteInfo, now: Date = Date()) {
        latencyPrompts.routeChanged(r, now: now)
        if latencyPrompt != latencyPrompts.pending { latencyPrompt = latencyPrompts.pending }
    }

    /// "Not now" on the banner.
    func dismissLatencyPrompt() {
        latencyPrompts.dismiss()
        latencyPrompt = nil
    }

    /// Runs once a second (the age text); assigns only on a change so the screens do not redraw.
    func updateLatencyDisplay() {
        let d = LatencyMeter.display(route: route, earshotMS: engineLatencyMS,
                                     measurement: latencyStore.last(forPort: route.primaryOutputUID), now: Date())
        if latency != d { latency = d }
    }

    /// Plays 1 to 3 beeps (heard even while muted).
    func cue(beeps: Int) {
        guard let e = engine else { return }
        e.cue(beeps: beeps)
        onEngineEvent?(.cue(beeps))
    }

    // MARK: - 10 Hz stats and meter

    func startPolling() {
        pollTask?.cancel()
        pollTask = Task { @MainActor [weak self] in
            var tick = 0
            while !Task.isCancelled {
                self?.poll()
                tick += 1
                if tick % 10 == 0 { self?.secondTick() } // age text and re-check banners, once per second
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    /// Once a second: the delay chip's age text and the 10 / 30 min re-check banners.
    func secondTick() {
        safetySecond() // I-4: power recovery timer, compute timings, the per-minute log line
        updateLatencyDisplay()
        latencyPrompts.tick(now: Date())
        if latencyPrompt != latencyPrompts.pending { latencyPrompt = latencyPrompts.pending }
    }

    func poll() {
        guard let e = engine else { return }
        stats = e.stats()
        safetyPoll(stats) // I-4: feedback banner, alarm timer, session log minute
        onPoll?(stats) // Selector: talking dots, direction learning, Now Playing, locked check
        dropStaleQuickAddRAM() // the swap's RAM copy lives only until the new engine has 8 s
        guard var tap = meterTap else { return }
        let samples = tap.read(maxSamples: 16_000)
        meterTap = tap
        if samples.isEmpty { return }
        var peak: Float = 0
        var sum: Float = 0
        for x in samples {
            peak = max(peak, abs(x))
            sum += x * x
        }
        let rms = (sum / Float(samples.count)).squareRoot()
        meter = InputMeter(peakDB: max(-100, 20 * log10(peak + 1e-10)),
                           rmsDB: max(-100, 20 * log10(rms + 1e-10)))
    }

    /// Developer button: proves the swap path works (a Phase 0 row needs 1 forced swap).
    func forceSwap() {
        Task { @MainActor in await self.swapEngine(reason: .forced) }
    }

    /// One second after start, compare the requested io_frames with the real callback size,
    /// and swap once with the real value if they differ (30-ios 3.1).
    func scheduleBufferCheck() {
        bufferCheck?.cancel()
        bufferCheck = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            if Task.isCancelled { return }
            guard let e = self.engine else { return }
            let real = e.stats().lastFrames
            if real > 128 { self.notice = Copy.largeBuffer(real) }
            if real >= 16 && real <= 4096 && real != self.engineFrames {
                self.note("callback is \(real) frames, engine built for \(self.engineFrames)")
                await self.swapEngine(reason: .bufferSize, frames: real)
            }
        }
    }

    // MARK: - Diagnostics log

    func note(_ text: String) {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        notes.append("\(f.string(from: Date())) \(text)")
        if notes.count > 60 { notes.removeFirst(notes.count - 60) }
    }
}
