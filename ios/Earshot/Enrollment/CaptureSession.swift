// CaptureSession.swift - records an enrollment from the engine's 16 kHz mic history (30-ios 8.1).
//
// How it works: an InputTap starts at the engine's current write position (so nothing from before
// the press). Every 250 ms it reads the new samples, runs es_vad on everything so far and shows
// speech seconds (kept frames x 0.032). Listening keeps running meanwhile; nothing about the audio
// session changes.
//
// Stop rules (contract 5.3):
//   target    stop at 6 s of speech or 15 s total; "Done" is allowed from 4 s of speech
//   own voice stop at 8 s of speech or 20 s total
// If the tap reports dropped samples, or the engine is swapped (route change), the capture fails
// with "Audio was interrupted - try again". Cues: 1 beep at start, 3 on failure (the enrollment
// screen plays 2 on success after the checks).

import Foundation

/// The stop limits for one kind of capture.
struct CaptureLimits: Equatable {
    let maxSamples: Int     // total length cap at 16 kHz
    let stopSpeech: Double  // seconds of VAD speech that end the capture
    let doneFrom: Double    // seconds of speech after which "Done" may end it early

    static let target = CaptureLimits(maxSamples: 15 * 16_000, stopSpeech: 6, doneFrom: 4)
    static let own = CaptureLimits(maxSamples: 20 * 16_000, stopSpeech: 8, doneFrom: 8)
}

/// The capture logic without timers or UI, so tests can drive it with a stub engine.
struct CaptureCore {
    enum Outcome: Equatable { case recording, enoughSpeech, timeUp, interrupted }

    let limits: CaptureLimits
    private(set) var samples: [Float] = []
    private(set) var speechSeconds = 0.0
    private(set) var outcome: Outcome = .recording

    init(limits: CaptureLimits) {
        self.limits = limits
        samples.reserveCapacity(limits.maxSamples) // preallocated once (15 or 20 s)
    }

    var elapsedSeconds: Double { Double(samples.count) / 16_000 }
    var canFinishEarly: Bool { outcome == .recording && speechSeconds >= limits.doneFrom }

    /// One 250 ms step: reads the tap, appends what fits, updates speech seconds, applies the stop
    /// rules. `sameEngine` is false once the engine the tap belongs to was swapped out.
    mutating func step(tap: inout InputTap, sameEngine: Bool, vad: ([Float]) -> [UInt8]) -> Outcome {
        guard outcome == .recording else { return outcome }
        guard sameEngine else { // a route change swapped the engine: this tap sees no new audio
            outcome = .interrupted
            return outcome
        }
        let droppedBefore = tap.droppedSamples
        let room = limits.maxSamples - samples.count
        // At most 3 s per step (a 250 ms timer normally finds 4,000 samples); anything left waits for
        // the next step, and a reader that falls more than 15.36 s behind shows up as dropped samples.
        let new = tap.read(maxSamples: max(1, min(room, 48_000)))
        if tap.droppedSamples > droppedBefore { // the reader fell behind: part of the capture is missing
            outcome = .interrupted
            return outcome
        }
        samples.append(contentsOf: new.prefix(room))
        speechSeconds = ClipTools.roundSeconds(ClipTools.speechSeconds(vad(samples)))
        if speechSeconds >= limits.stopSpeech { outcome = .enoughSpeech }
        else if samples.count >= limits.maxSamples { outcome = .timeUp }
        return outcome
    }

    /// The person's direction: the circular mean (degrees) of the yaw samples taken during speech
    /// frames, if head motion was live for at least half of the samples; else nil (30-ios 8.1).
    /// `yaw` holds (capture sample index, yaw or nil when motion was not live) every 250 ms.
    /// The result is in (-180, 180], the same range as CMAttitude.yaw.
    static func direction(yaw: [(sample: Int, degrees: Double?)], mask: [UInt8]) -> Double? {
        let live = yaw.filter { $0.degrees != nil }
        guard !yaw.isEmpty, live.count * 2 >= yaw.count else { return nil }
        var sx = 0.0, sy = 0.0, used = 0
        for y in live {
            let f = (y.sample + ClipTools.pad) / ClipTools.frameHop // the token frame holding this sample
            guard f < mask.count, mask[f] != 0, let d = y.degrees else { continue }
            sx += cos(d * .pi / 180)
            sy += sin(d * .pi / 180)
            used += 1
        }
        // No speech sample, or samples that cancel out (90 and 270 degrees): the mean direction is
        // undefined. Floating point leaves about 1e-16 behind instead of exactly 0, so compare the
        // mean vector length with a small tolerance rather than with 0.
        guard used > 0, (sx * sx + sy * sy).squareRoot() / Double(used) > 1e-6 else { return nil }
        return atan2(sy, sx) * 180 / .pi
    }
}

@MainActor
final class CaptureSession: ObservableObject {
    enum Phase: Equatable { case idle, recording, finished, failed(String) }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var speechSeconds = 0.0
    @Published private(set) var elapsedSeconds = 0.0
    @Published private(set) var canFinish = false

    /// Head yaw in degrees while motion is live (HeadTracker.yawForCapture, set by the screens and
    /// the Selector in I-3); nil = no direction.
    var yawProvider: (@MainActor () -> Double?)?
    /// Called once when the capture ends: `.finished`, `.failed` or `.idle` (cancelled). The
    /// screens watch `phase` instead; the Selector's hands-free capture uses this.
    var onEnd: (@MainActor (Phase) -> Void)?

    /// The capture so far (internal only so CaptureTests can check the buffer is never copied).
    private(set) var core: CaptureCore?
    private var tap: InputTap?
    private var loop: Task<Void, Never>?
    private var yaw: [(sample: Int, degrees: Double?)] = []
    private weak var audio: AudioController?
    private var model: EarshotModel?

    /// The captured audio once `phase` is `.finished`.
    private(set) var captured: [Float] = []
    /// The direction estimate once `phase` is `.finished` (nil without head tracking).
    private(set) var yawDegrees: Double? = nil

    /// Starts listening first if needed (setup mode stays muted), then records from a fresh tap.
    func start(audio: AudioController, model: EarshotModel, role: VoiceRole) async throws {
        cancel()
        if audio.engine == nil { try await audio.start() }
        guard let engine = audio.engine else { throw AppError.notRunning }
        self.audio = audio
        self.model = model
        tap = engine.makeInputTap()
        core = CaptureCore(limits: role == .own ? .own : .target)
        captured = []
        yawDegrees = nil
        yaw = []
        speechSeconds = 0
        elapsedSeconds = 0
        canFinish = false
        phase = .recording
        audio.cue(beeps: Cue.captureStarted)
        loop = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 250_000_000)
                guard let self, self.phase == .recording else { return }
                self.tick()
            }
        }
    }

    /// One 250 ms step on the main actor (es_vad on 15 s of audio takes well under 1 ms of C).
    /// Internal (not private) only so CaptureTests can call it without waiting for the timer.
    func tick() {
        guard core != nil, var tap, let model else { return }
        let sameEngine = audio?.engine === tap.engine
        // Step `self.core` in place. Copying it out first (`guard var core`) would leave two
        // references to the samples array, so the append inside step() would copy the whole
        // buffer (up to 1.28 MB) every tick and lose the preallocated capacity.
        let outcome = core!.step(tap: &tap, sameEngine: sameEngine, vad: model.vad)
        self.tap = tap
        guard let core else { return }
        yaw.append((core.samples.count, yawProvider?()))
        speechSeconds = core.speechSeconds
        elapsedSeconds = core.elapsedSeconds
        canFinish = core.canFinishEarly
        switch outcome {
        case .recording: break
        case .enoughSpeech, .timeUp: complete()
        case .interrupted: fail(Copy.captureInterrupted)
        }
    }

    /// "Done" (allowed once `canFinish` is true).
    func finishNow() {
        guard phase == .recording, canFinish else { return }
        complete()
    }

    func cancel() {
        loop?.cancel()
        loop = nil
        tap = nil
        if phase == .recording {
            phase = .idle
            onEnd?(.idle)
        }
    }

    private func complete() {
        guard let core, let model else { return }
        loop?.cancel()
        captured = core.samples
        yawDegrees = CaptureCore.direction(yaw: yaw, mask: model.vad(core.samples))
        tap = nil
        phase = .finished
        onEnd?(.finished)
    }

    private func fail(_ message: String) {
        loop?.cancel()
        tap = nil
        captured = []
        phase = .failed(message)
        audio?.cue(beeps: Cue.failure)
        onEnd?(phase)
    }
}
