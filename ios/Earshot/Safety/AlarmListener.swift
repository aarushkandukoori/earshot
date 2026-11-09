// AlarmListener.swift - alarm passthrough (30-ios 12.2, contract 5.3 "Alarm passthrough", D46).
// A helper, never a safety device: SoundAnalysis can miss alarms, and the Settings footer says so.
//
// What it does: reads the engine's 16 kHz mic history with its own InputTap every 250 ms on a
// serial utility queue, feeds SNAudioStreamAnalyzer with SNClassifySoundRequest(.version1) (3 s
// windows, 50% overlap; no overlap, so one window every 3 s, while saving power) and, when an
// alarm-like label reaches confidence 0.6 in 2 windows in a row, tells the main actor. The main
// actor (AudioController+Safety.swift) then lets the room through for 5 s: bypass on, output at
// -6 dB, 3 beeps, a banner, and afterwards the previous params come back.
//
// Parts (top to bottom):
//   AlarmLabels        - the labels that count (contract 5.3 list, siren family spelled out)
//   AlarmDetector      - pure: confidence >= 0.6 in 2 consecutive windows (unit-tested)
//   AlarmPassthrough   - pure: the 5 s on / extend / off timer (unit-tested)
//   AlarmClassifier    - protocol for "audio in, window scores out", so tests can fake it
//   SoundAnalysisClassifier - the real one (SoundAnalysis, on device, no network)
//   AlarmWorker        - owns the queue, the tap, the classifier and the detector (not main actor)
//   AlarmListener      - main-actor front door used by AudioController
//
// Threads: the tap, the classifier and the detector are touched only on the worker's serial queue.
// SoundAnalysis may call back on its own thread, so window results hop back onto that queue. A
// trigger hops to the main actor. Nothing here runs on the audio thread or the engine's worker.

import AVFoundation
import Foundation
import SoundAnalysis

/// The labels of the built-in classifier that count as alarm-like (contract 5.3; the siren family
/// is listed in 30-ios 12.2 and was checked against the built-in label list, product.md B.5).
enum AlarmLabels {
    static let all: Set<String> = [
        "smoke_detector",
        "siren", "police_siren", "ambulance_siren", "fire_engine_siren", "civil_defense_siren", "emergency_vehicle",
        "car_horn", "screaming", "glass_breaking", "baby_crying",
    ]
}

/// Confidence >= 0.6 for any alarm label, in 2 windows in a row, triggers (30-ios 12.2).
struct AlarmDetector: Equatable {
    static let threshold = 0.6
    static let windowsNeeded = 2

    /// Alarm-like windows in a row so far.
    private(set) var streak = 0

    /// True when this window holds an alarm label at or above the threshold.
    static func isAlarmWindow(_ scores: [String: Double]) -> Bool {
        scores.contains { AlarmLabels.all.contains($0.key) && $0.value >= threshold }
    }

    /// Feeds one classified window; returns true when it fires. While the alarm goes on it keeps
    /// firing every window, which only extends the passthrough (no new beeps).
    mutating func window(_ scores: [String: Double]) -> Bool {
        streak = Self.isAlarmWindow(scores) ? streak + 1 : 0
        return streak >= Self.windowsNeeded
    }

    mutating func reset() { streak = 0 }
}

/// The 5 s passthrough timer (main actor). A trigger while active moves the end 5 s further.
struct AlarmPassthrough: Equatable {
    static let holdSeconds: TimeInterval = 5
    /// Output level while the room comes through (contract 5.3 and 30-ios 12.2 "out_gain_db -6").
    /// Exactly -6 dB, whatever the user's trim, so a quiet trim cannot hide the alarm; the engine's
    /// limiter still caps peaks at -3 dBFS.
    static let outGainDB: Float = -6

    enum Change: Equatable { case started, extended, ended, none }

    /// When the passthrough ends; nil = not active.
    private(set) var activeUntil: TimeInterval? = nil
    var isActive: Bool { activeUntil != nil }

    /// An alarm was heard at `now`.
    mutating func trigger(now: TimeInterval) -> Change {
        let wasActive = isActive
        activeUntil = now + Self.holdSeconds
        return wasActive ? .extended : .started
    }

    /// Called often (10 Hz); returns .ended once 5 s passed since the last trigger.
    mutating func tick(now: TimeInterval) -> Change {
        guard let end = activeUntil, now >= end else { return .none }
        activeUntil = nil
        return .ended
    }

    /// Stop at once (listening stopped, or the user turned the setting off).
    mutating func cancel() -> Change {
        guard isActive else { return .none }
        activeUntil = nil
        return .ended
    }

    /// The params while the room comes through: the delayed microphone (bypass), at -6 dB.
    static func applied(to p: EngineParams) -> EngineParams {
        var out = p
        out.bypass = true
        out.outGainDB = outGainDB
        return out
    }
}

/// Audio in, window scores out. `feed` and the callback happen on the worker's queue.
protocol AlarmClassifier: AnyObject {
    /// New 16 kHz mono samples, in order.
    func feed(_ samples: [Float])
    /// Called once per classified window with label -> confidence (alarm labels only are enough).
    var onWindow: (([String: Double]) -> Void)? { get set }
}

/// The real classifier: SoundAnalysis's built-in sound classifier (version 1), on device.
final class SoundAnalysisClassifier: NSObject, AlarmClassifier, SNResultsObserving {
    var onWindow: (([String: Double]) -> Void)?
    /// The window length SoundAnalysis really uses (it rounds an unsupported request, see
    /// SNClassifySoundRequest.windowDuration); shown in Diagnostics.
    let windowSeconds: Double

    private let analyzer: SNAudioStreamAnalyzer
    private let format: AVAudioFormat
    private var position: AVAudioFramePosition = 0

    /// 3 s windows; `overlap` 0.5 normally, 0 while saving power (PowerPolicy.alarmOverlap).
    init(overlap: Double) throws {
        guard let f = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1,
                                    interleaved: false) else { throw AppError.notReady }
        format = f
        let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
        request.windowDuration = CMTime(seconds: 3, preferredTimescale: 16_000)
        request.overlapFactor = overlap
        windowSeconds = request.windowDuration.seconds
        analyzer = SNAudioStreamAnalyzer(format: f)
        super.init()
        try analyzer.add(request, withObserver: self)
    }

    func feed(_ samples: [Float]) {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in channel.update(from: src.baseAddress!, count: samples.count) }
        analyzer.analyze(buffer, atAudioFramePosition: position)
        position += AVAudioFramePosition(samples.count)
    }

    // SNResultsObserving: one result per window. Only the alarm labels are passed on.
    func request(_ request: SNRequest, didProduce result: SNResult) {
        guard let r = result as? SNClassificationResult else { return }
        var scores: [String: Double] = [:]
        for c in r.classifications where AlarmLabels.all.contains(c.identifier) { scores[c.identifier] = c.confidence }
        onWindow?(scores)
    }

    func request(_ request: SNRequest, didFailWithError error: Error) {}
    func requestDidComplete(_ request: SNRequest) {}
}

/// The off-main part: queue, tap, classifier, detector. Every stored property below is used only
/// on `queue` (that is the whole thread-safety rule of this class).
final class AlarmWorker: @unchecked Sendable {
    /// How often the tap is read (30-ios 12.2).
    static let readInterval: TimeInterval = 0.25

    private let queue = DispatchQueue(label: "earshot.alarm", qos: .utility)
    private var timer: DispatchSourceTimer?
    private var tap: InputTap?
    private var classifier: AlarmClassifier?
    private var detector = AlarmDetector()
    private var windows = 0
    private var lastText = "-"
    /// Called on `queue` when the detector fires.
    var onTrigger: (@Sendable () -> Void)?
    /// Called on `queue` with a line for the Diagnostics log (classifier failures).
    var onNote: (@Sendable (String) -> Void)?

    /// Starts (or restarts) on `engine` with a classifier from `make`, which runs on the queue
    /// (loading the sound classifier takes a moment and must not block the main actor).
    func start(engine: EarshotEngine, make: @escaping @Sendable () throws -> AlarmClassifier) {
        queue.async {
            self.tap = engine.makeInputTap() // only audio from now on
            self.detector.reset()
            do {
                let c = try make()
                c.onWindow = { [weak self] scores in self?.queue.async { self?.receive(scores) } }
                self.classifier = c
            } catch {
                self.classifier = nil
                self.onNote?("alarm listener: classifier unavailable: \(error)")
            }
            if self.timer == nil {
                let t = DispatchSource.makeTimerSource(queue: self.queue)
                t.schedule(deadline: .now() + Self.readInterval, repeating: Self.readInterval)
                t.setEventHandler { [weak self] in self?.readTap() }
                t.resume()
                self.timer = t
            }
        }
    }

    /// Stops reading and drops the tap (which also lets the engine it belongs to be freed).
    func stop() {
        queue.async {
            self.timer?.cancel()
            self.timer = nil
            self.tap = nil
            self.classifier = nil
            self.detector.reset()
        }
    }

    /// One classified window (on the queue). Internal so tests can call it through a fake.
    func receive(_ scores: [String: Double]) {
        windows += 1
        if let top = scores.max(by: { $0.value < $1.value }) {
            lastText = String(format: "%@ %.2f", top.key, top.value)
        } else {
            lastText = "no alarm label"
        }
        if detector.window(scores) { onTrigger?() }
    }

    /// Windows classified so far and the last top alarm label (for Diagnostics), read on the queue.
    func status(_ reply: @escaping @Sendable (Int, String) -> Void) {
        queue.async { reply(self.windows, self.lastText) }
    }

    private func readTap() {
        guard var t = tap, let c = classifier else { return }
        let samples = t.read(maxSamples: 16_000) // 250 ms is 4,000 samples; more after a slow wake
        tap = t
        if !samples.isEmpty { c.feed(samples) }
    }
}

/// The main-actor front door. AudioController starts it after start step 5, restarts it on every
/// engine swap (taps belong to one engine) and on power-saving changes (the window overlap), and
/// stops it on Stop. `onTrigger` runs on the main actor.
@MainActor
final class AlarmListener {
    var onTrigger: (@MainActor () -> Void)?
    var onNote: (@MainActor (String) -> Void)?
    private(set) var running = false
    /// For Diagnostics: windows classified and the last top alarm label (refreshed by refreshStatus).
    private(set) var windows = 0
    private(set) var lastWindow = "-"

    private let worker = AlarmWorker()
    private let makeClassifier: @Sendable (Double) throws -> AlarmClassifier

    /// Tests pass a fake classifier factory; the app uses SoundAnalysis. The argument is the overlap.
    init(makeClassifier: @escaping @Sendable (Double) throws -> AlarmClassifier = { try SoundAnalysisClassifier(overlap: $0) }) {
        self.makeClassifier = makeClassifier
        worker.onTrigger = { [weak self] in Task { @MainActor in self?.onTrigger?() } }
        worker.onNote = { [weak self] text in Task { @MainActor in self?.onNote?(text) } }
    }

    /// Starts listening on `engine` (also used after an engine swap and a power change).
    func start(engine: EarshotEngine, level: PowerLevel) {
        let overlap = PowerPolicy.alarmOverlap(level)
        let make = makeClassifier
        worker.start(engine: engine) { try make(overlap) }
        running = true
    }

    func stop() {
        guard running else { return }
        worker.stop()
        running = false
    }

    /// Asks the worker for its counters; the values arrive a moment later.
    func refreshStatus() {
        worker.status { [weak self] n, text in
            Task { @MainActor in
                self?.windows = n
                self?.lastWindow = text
            }
        }
    }

    /// Test seam: hands one window to the worker as if the classifier had produced it.
    func injectWindowForTests(_ scores: [String: Double]) {
        worker.injectForTests(scores)
    }
}

extension AlarmWorker {
    /// Test seam: runs receive(_:) on the queue, like a classifier callback would.
    func injectForTests(_ scores: [String: Double]) {
        queue.async { self.receive(scores) }
    }
}
