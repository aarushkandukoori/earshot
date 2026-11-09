// AudioController+Safety.swift - the I-4 parts of AudioController: alarm passthrough (30-ios 12.2),
// the feedback banner (12.1), the per-minute session log and the session compute percentiles (14,
// contract 5.2). Thermal and Low Power live in +System (they react to iOS notifications).
//
// Main actor only (rule 2 of 30-ios 0: only the main actor writes to the engine). The pure rules
// are elsewhere and unit-tested: AlarmDetector / AlarmPassthrough (Safety/AlarmListener.swift),
// MinuteTelemetry / ComputeHistogram / SessionLogRecord (Telemetry/SessionLog.swift).
//
// Alarm passthrough, step by step:
//   1. AlarmListener (off the main actor) hears an alarm-like sound in 2 windows in a row
//   2. alarmHeard(): bypass on, output at -6 dB, 3 beeps, banner; while paused it also plays
//   3. alarmTick() at 10 Hz: 5 s after the last trigger, the previous params come back
// Every params push during those 5 s goes through safetyApplied, so a slider change, a swap or a
// pause cannot switch the passthrough off early.
//
// drainTimings() has exactly one reader, safetySecond() (draining empties the engine's ring). Code
// that needs per-hop compute times reads `sessionCompute` or `minute.histogram`, or (the I-1 Phase 0
// compute run) receives each second's timings through `phase0.onTimings`, instead of draining twice.
// I-1 also refreshes the debug malloc counter here once a second (Telemetry/MallocCounter.swift).

import AVFoundation
import UIKit

extension AudioController {
    // MARK: - Start and stop (start step 5, teardown, media reset)

    /// The end of start step 5 (30-ios 6): battery readings on, a clean session log minute, the
    /// current thermal / Low Power state applied, the alarm listener, old logs purged.
    func startSafety() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        minute = MinuteTelemetry()
        minuteStart = Date()
        sessionCompute = ComputeHistogram()
        computeSummary = "-"
        _ = engine?.drainTimings() // the first minute starts with this engine's newest hops
        handlePowerChange()
        // Already hot or in Low Power Mode before Start: no level change happened above, so show
        // the banner here, and a critical phone pauses at once (30-ios 6).
        switch powerLevel {
        case .critical:
            pause()
            notice = Copy.tooHot
        case .saving:
            if notice == nil { notice = Copy.savingPower }
        case .normal:
            break
        }
        restartAlarmListener()
        sessionLog.purgeOld()
        mallocCounter.start() // I-1: debug builds install the malloc counter (no-op in Release)
        note(String(format: "Earshot share %.2f ms at %ld frames (es_engine_latency_ms)", engineLatencyMS, engineFrames))
    }

    /// Teardown and media reset: the last part-minute goes to the session log (its `t` is the stop
    /// time, so the last 0 to 59 s of battery and underruns are not lost), the listener stops (its
    /// tap would keep the old engine alive) and any passthrough ends without touching the engine
    /// that is going away.
    func stopSafety() {
        if engine != nil && minute.polls > 0 { writeMinute(now: Date()) }
        alarmListener.stop()
        mallocCounter.disarm() // I-1: the render and worker threads are going away
        if alarm.cancel() == .ended, notice == Copy.alarmPassthroughBanner { notice = nil }
    }

    /// Starts the alarm listener on the current engine, or stops it when the setting is off.
    /// Also called after every engine swap (taps belong to one engine) and power change (overlap).
    func restartAlarmListener() {
        guard alarmEnabled, let e = engine else {
            alarmListener.stop()
            return
        }
        alarmListener.onTrigger = { [weak self] in self?.alarmHeard() }
        alarmListener.onNote = { [weak self] text in self?.note(text) }
        alarmListener.start(engine: e, level: powerLevel)
    }

    /// Settings "Let alarm sounds through" (default on). Off ends a running passthrough at once.
    func setAlarmEnabled(_ on: Bool) {
        guard on != alarmEnabled else { return }
        alarmEnabled = on
        if on {
            if engine != nil { restartAlarmListener() }
        } else {
            alarmListener.stop()
            if alarm.cancel() == .ended { alarmChanged(on: false) }
        }
    }

    // MARK: - Alarm passthrough (30-ios 12.2)

    /// The listener heard an alarm-like sound. The first trigger starts the 5 s; later ones extend it.
    func alarmHeard(now: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        guard alarmEnabled, engine != nil else { return }
        if alarm.trigger(now: now) == .started { alarmChanged(on: true) }
    }

    /// 10 Hz (from the stats poll): ends the passthrough 5 s after the last trigger.
    func alarmTick(now: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        if alarm.tick(now: now) == .ended { alarmChanged(on: false) }
    }

    /// Sends the params again with or without the passthrough, then the mute rule decides (a user
    /// pause plays during the passthrough, shouldPlay(alarm:)).
    private func alarmChanged(on: Bool) {
        reapplyParamsForSafety()
        refreshMute()
        if on {
            cue(beeps: Cue.alarm)
            notice = Copy.alarmPassthroughBanner
            note("alarm passthrough on for 5 s")
        } else {
            if notice == Copy.alarmPassthroughBanner { notice = nil }
            note("alarm passthrough off, params restored")
        }
    }

    /// The quiet pause params while paused (resume() restores the real ones), else `params`.
    func reapplyParamsForSafety() {
        guard let e = engine else { return }
        if userPaused || state == .paused {
            var quiet = params
            quiet.slotGain = [0, 0, 0]
            quiet.roomDB = -60
            e.setParams(safetyApplied(quiet))
            onEngineEvent?(.gains(quiet.slotGain))
        } else {
            sendParams(params) // applies safetyApplied
        }
    }

    /// The params the engine really gets: during a passthrough, bypass on at -6 dB (contract 5.3).
    func safetyApplied(_ p: EngineParams) -> EngineParams {
        alarm.isActive ? AlarmPassthrough.applied(to: p) : p
    }

    // MARK: - 10 Hz and once a second

    /// Every stats poll: the session log minute, the alarm timer, and "Feedback detected - output
    /// lowered" while the engine's runaway guard is active (12.1).
    func safetyPoll(_ s: EngineStats) {
        minute.addPoll(s, engine: engine.map { ObjectIdentifier($0) }) // a swap restarts the counters
        alarmTick()
        if s.howl {
            if notice != Copy.feedback { notice = Copy.feedback }
        } else if notice == Copy.feedback {
            notice = nil
        }
    }

    /// Once a second: the power recovery timer, this second's per-hop timings (minute and session
    /// histograms), the Diagnostics summary, and one log line per 60 s.
    func safetySecond(now: Date = Date()) {
        handlePowerChange()
        guard let e = engine else { return }
        let timings = e.drainTimings()
        sessionCompute.add(timings)
        phase0.onTimings?(timings) // I-1: the Phase 0 compute run builds its own histogram from these
        mallocCounter.refresh(renderThreadID: host.map { eh_render_thread_id($0) } ?? 0,
                              engine: ObjectIdentifier(e), expectWorker: !(model?.isStub ?? true))
        minute.addSecond(timings: timings, thermal: ProcessInfo.processInfo.thermalState,
                         lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled)
        let h = sessionCompute
        computeSummary = h.total == 0 ? "-" :
            "p50 \(h.percentile(0.5)) / p99 \(h.percentile(0.99)) / p99.9 \(h.percentile(0.999)) / max \(h.maxUS) us, \(h.total) hops"
        alarmListener.refreshStatus()
        if now.timeIntervalSince(minuteStart) >= 60 { writeMinute(now: now) }
    }

    /// Appends this minute's line to the session log and starts the next minute.
    func writeMinute(now: Date) {
        sessionLog.append(makeRecord(now: now))
        minute.nextMinute()
        minuteStart = now
    }

    /// The contract 5.2 record for the minute that ends `now` (meanings in SessionLog.swift).
    func makeRecord(now: Date) -> SessionLogRecord {
        let table = telemetryTable?() ?? (mode: TableMode.everyone.rawValue, instances: 1)
        let level = UIDevice.current.batteryLevel
        return SessionLogRecord(
            t: now, route: route.outputs.first?.type.rawValue ?? "none", ioRate: Int(engineRate),
            ioFrames: engineFrames, slack16: stats.slack16, mode: table.mode, nInstances: table.instances,
            computeUsP50: minute.histogram.percentile(0.5), computeUsP99: minute.histogram.percentile(0.99),
            computeUsMax: minute.histogram.maxUS, underruns: minute.underruns, nanResets: minute.nanResets,
            howl: minute.howl, thermal: MinuteTelemetry.thermalText(minute.worstThermal),
            battery: level < 0 ? nil : Int((level * 100).rounded()), lowPower: minute.lowPower,
            outRmsDB: minute.outRmsMean, limiterGrDB: minute.limiterMax, pTargetMean: minute.pTargetMeans,
            pSelfMean: minute.pSelfMean, latencyMeasuredMS: latency.measuredMS, latencyReportedMS: latency.reportedMS,
            appVersion: Self.appVersion, modelHash: model?.modelHash ?? "")
    }

    /// "0.1 (1)": marketing version and build number from Info.plist.
    nonisolated static var appVersion: String {
        let info = Bundle.main.infoDictionary ?? [:]
        let short = info["CFBundleShortVersionString"] as? String ?? "?"
        let build = info["CFBundleVersion"] as? String ?? "?"
        return "\(short) (\(build))"
    }

    // MARK: - Probe history (Diagnostics)

    /// One line per probe of this run, newest last, at most 20.
    func recordProbe(_ text: String) {
        let f = DateFormatter()
        f.dateFormat = "HH:mm"
        probeHistory.append("\(f.string(from: Date())) \(text)")
        if probeHistory.count > 20 { probeHistory.removeFirst(probeHistory.count - 20) }
    }
}
