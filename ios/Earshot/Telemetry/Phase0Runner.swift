// Phase0Runner.swift - the guided Phase 0 runs (30-ios 15.3, contract 11 I-1): which step the
// owner is on, the measurement file, and the plumbing every run shares. The run bodies live in
// Phase0Runner+Runs.swift (audio steps) and Phase0Runner+Engine.swift (compute, power, WeSpeaker).
//
// Main actor only. Created once by EarshotApp (so a 60 min run keeps going when the owner leaves
// the screen or locks the phone; the audio background mode keeps the app running), attached to
// the audio controller, the Selector and the launch state when the Phase 0 screen first opens.
//
// One run at a time. A run:
//   1. begins its step (in_progress) and shows what the owner has to do (`run.prompt`)
//   2. measures, once a second, with the screen state sampled from the scene phase
//   3. appends its row to the report, saves phase0-<date>.json, finishes its item, saves progress
// Cancel stops the task; nothing half-measured is ever written. Every run ends by putting back
// what it changed (requested buffer size, stem press handling, Now Playing rate, hooks).

import SwiftUI
import UIKit

/// Problems a run can report (the text comes from Copy; raw details go to the Diagnostics log).
enum Phase0Error: Error, LocalizedError {
    case notAttached, notListening, wrongRoute(String), needsAirPods, plugCharger, needsLargeBuffer(Int)
    case noWeSpeaker, notEnoughSpeech, alarmOff, busy, engineChanged, probeNotRun

    var errorDescription: String? {
        switch self {
        case .notAttached: return Copy.phase0NotReady
        case .notListening: return Copy.phase0NotListening
        case .wrongRoute(let actual): return Copy.phase0WrongRoute(actual)
        case .needsAirPods: return Copy.phase0NeedsAirPods
        case .plugCharger: return Copy.phase0Unplug
        case .needsLargeBuffer(let frames): return Copy.phase0NeedsSoundRecognition(frames)
        case .noWeSpeaker: return Copy.phase0NoWeSpeaker
        case .notEnoughSpeech: return Copy.phase0NotEnoughSpeech
        case .alarmOff: return Copy.phase0AlarmOff
        case .busy: return Copy.phase0Busy
        case .engineChanged: return Copy.probeAudioChanged
        case .probeNotRun: return Copy.phase0ProbeNotRun
        }
    }
}

/// What is measuring right now (shown at the top of the Phase 0 screen).
struct Phase0ActiveRun: Equatable {
    var step: Phase0StepID
    var title: String
    var prompt: String         // what the owner has to do now, in plain words
    var seconds: Double        // planned length, 0 = open-ended (ends with Done)
    var elapsed: Double = 0
}

@MainActor
final class Phase0Runner: ObservableObject {
    @Published private(set) var progress: Phase0Progress
    @Published private(set) var report: Phase0Report
    @Published private(set) var run: Phase0ActiveRun? = nil
    /// The newest result or problem, one line.
    @Published var message = ""

    /// The probe level compiled into this build (earshot.h ES_PROBE_DBFS). The engine has no run-time
    /// level, so -24 and -18 dBFS need builds with that define changed (rows from every build add up).
    static let probeDBFS = Int(ES_PROBE_DBFS)

    weak var audio: AudioController?
    weak var selector: Selector?
    weak var launch: LaunchState?
    let store: Phase0Store
    let progressStore: Phase0ProgressStore
    private var task: Task<Void, Never>? = nil
    /// Set by `endOpenRun()` (the Done button of an open-ended run).
    var openRunDone = false
    /// Seconds of the current run the app was active / in the background (screen on / locked).
    var activeSeconds = 0
    var inactiveSeconds = 0

    init(store: Phase0Store = .standard(), progressStore: Phase0ProgressStore = .standard(), now: Date = Date()) {
        self.store = store
        self.progressStore = progressStore
        let p = progressStore.load() ?? Phase0Progress(reportDate: Phase0Store.dateString(now))
        progress = p
        report = store.load(date: p.reportDate) ?? Self.emptyReport(date: p.reportDate)
    }

    /// Called when the Phase 0 screen opens; fills the file header from the running app.
    func attach(audio: AudioController, selector: Selector, launch: LaunchState) {
        self.audio = audio
        self.selector = selector
        self.launch = launch
        fillHeader()
    }

    // MARK: - The file and the steps

    static func emptyReport(date: String) -> Phase0Report {
        Phase0Report(date: date, device: deviceModel(), ios: UIDevice.current.systemVersion,
                     appVersion: AudioController.appVersion, engine: "unknown", modelHash: "")
    }

    /// Hardware model, for example "iPhone17,1" (the simulator reports its Mac's model, "arm64").
    static func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }

    /// Header fields that are known once the launch sequence ran.
    private func fillHeader() {
        guard let r = launch?.report else { return }
        var changed = false
        if report.engine != r.engineKind { report.engine = r.engineKind; changed = true }
        if report.modelHash != r.modelHash { report.modelHash = r.modelHash; changed = true }
        if changed { saveReport() }
    }

    /// "Start a new Phase 0 file": today's date, every step back to "to do". The old file stays.
    func newFile(now: Date = Date()) {
        guard run == nil else { return }
        progress = Phase0Progress(reportDate: Phase0Store.dateString(now))
        report = store.load(date: progress.reportDate) ?? Self.emptyReport(date: progress.reportDate)
        fillHeader()
        saveProgress()
        saveReport()
        message = ""
    }

    func setNotes(_ text: String) {
        report.notes = text
        saveReport()
    }

    func skip(_ id: Phase0StepID) { guard run?.step != id else { return }; progress.skip(id); saveProgress() }
    func markDone(_ id: Phase0StepID) { guard run?.step != id else { return }; progress.markDone(id); saveProgress() }
    func reopen(_ id: Phase0StepID) { progress.reopen(id); saveProgress() }
    func begin(_ id: Phase0StepID) { if progress.status(id) != .inProgress { progress.begin(id); saveProgress() } }

    /// "AirPods just connected": the latency minutes count from now for this route.
    func markConnected(route: String, now: Date = Date()) {
        progress.connectedAt[route] = now
        saveProgress()
    }

    /// Whole minutes since `route` was marked connected (0 when never marked).
    func minutesSinceConnect(_ route: String, now: Date = Date()) -> Int {
        guard let t = progress.connectedAt[route] else { return 0 }
        return max(0, Int(now.timeIntervalSince(t) / 60))
    }

    var fileURL: URL { store.url(date: report.date) }

    func saveReport() {
        do { try store.save(report) }
        catch { message = Copy.phase0SaveFailed; audio?.note("phase0: save failed: \(error)") }
    }

    func saveProgress() {
        do { try progressStore.save(progress) }
        catch { audio?.note("phase0: progress save failed: \(error)") }
    }

    /// Appends a row (through `add`), saves the file, and finishes `item` when given.
    func record(_ step: Phase0StepID, item: String?, _ add: (inout Phase0Report) -> Void) {
        add(&report)
        saveReport()
        if let item { progress.finish(step, item: item) }
        saveProgress()
    }

    // MARK: - Running

    /// Starts one run. `body` measures and records; errors become the message line.
    func launchRun(_ step: Phase0StepID, title: String, prompt: String, seconds: Double,
                   _ body: @escaping @MainActor (Phase0Runner) async throws -> Void) {
        guard run == nil else { message = Copy.phase0Busy; return }
        progress.begin(step)
        saveProgress()
        fillHeader()
        message = ""
        openRunDone = false
        activeSeconds = 0
        inactiveSeconds = 0
        run = Phase0ActiveRun(step: step, title: title, prompt: prompt, seconds: seconds)
        task = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try await body(self)
            } catch is CancellationError {
                self.message = Copy.phase0Cancelled
            } catch {
                self.message = (error as? Phase0Error)?.errorDescription
                    ?? AppError.userText(error, fallback: Copy.phase0RunFailed)
                self.audio?.note("phase0 \(step.rawValue): \(error)")
            }
            self.putBack()
            self.run = nil
            self.task = nil
        }
    }

    /// Cancel: stops the run; its item stays open and nothing of it is written.
    func cancel() { task?.cancel() }

    /// Done on an open-ended run (the locked step): it records what it saw so far.
    func endOpenRun() { openRunDone = true }

    /// Changes the instruction line during a run.
    func prompt(_ text: String) { run?.prompt = text }

    /// Waits `seconds`, one tick a second: samples the screen state, updates the progress bar and
    /// calls `each(k)`. Throws CancellationError on Cancel. With `untilDone`, the Done button
    /// (endOpenRun) ends it early.
    func ticks(_ seconds: Int, untilDone: Bool = false,
               _ each: @MainActor (Int) throws -> Void = { _ in }) async throws {
        for k in 0..<seconds {
            try await Task.sleep(nanoseconds: 1_000_000_000)
            try Task.checkCancellation()
            if audio?.isForeground ?? true { activeSeconds += 1 } else { inactiveSeconds += 1 }
            run?.elapsed = Double(k + 1)
            try each(k)
            if untilDone && openRunDone { return }
        }
    }

    /// Moves the progress bar on by `seconds` (for steps that wait in smaller pieces).
    func addElapsed(_ seconds: Double) { run?.elapsed += seconds }

    /// "on", "locked" or "mixed" for the run so far.
    var screenLabel: String { Phase0Math.screenLabel(activeSeconds: activeSeconds, inactiveSeconds: inactiveSeconds) }

    /// Starts listening when Earshot is stopped (the measurement needs a running engine).
    @discardableResult
    func ensureListening() async throws -> AudioController {
        guard let audio else { throw Phase0Error.notAttached }
        switch audio.state {
        case .idle, .failed: try await audio.start()
        default: break
        }
        guard audio.engine != nil else { throw Phase0Error.notListening }
        return audio
    }

    /// The file's name for the live route; AirPods with no stored model take the owner's choice
    /// (and keep it for this port). Throws when the iPhone plays somewhere else than `wanted`.
    func checkRoute(_ wanted: String) throws {
        guard let audio else { throw Phase0Error.notAttached }
        let uid = audio.route.primaryOutputUID
        let models = AirPodsModelStore()
        if audio.route.isBluetooth, models.model(forPort: uid) == nil {
            if wanted == "airpods_pro1" { models.set(.pro1, forPort: uid) }
            if wanted == "airpods_pro2" { models.set(.pro2, forPort: uid) }
        }
        let live = Phase0Math.routeLabel(audio.route, model: models.model(forPort: uid)) ?? "airpods"
        if live != wanted { throw Phase0Error.wrongRoute(Copy.phase0RouteName(live)) }
    }

    /// Everything a run may have changed goes back to normal use.
    private func putBack() {
        SessionConfig.requestedFrames = SessionConfig.preferredFrames
        selector?.remote.countOnly = false
        selector?.remote.pausedRateOverride = nil
        selector?.head.onSample = nil
        audio?.phase0.onTimings = nil
    }

    /// Running totals of an engine counter (underruns) across engine swaps: a smaller value means
    /// a new engine whose counter started at 0 (MinuteTelemetry.delta).
    struct CounterTotal {
        private var last: UInt64
        private(set) var total = 0
        init(start: UInt64) { last = start }
        mutating func add(_ now: UInt64) {
            total += MinuteTelemetry.delta(now: now, last: last)
            last = now
        }
    }
}
