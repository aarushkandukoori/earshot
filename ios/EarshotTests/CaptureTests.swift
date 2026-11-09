// CaptureTests.swift - CaptureCore's stop rules, dropped-sample and engine-swap failures, and the
// direction estimate (30-ios 8.1), driven by the linked engine's es_engine_render (the stub, or the
// real engine with random_v1 since lead integration).

import XCTest
@testable import Earshot

final class CaptureTests: XCTestCase {
    private var model: EarshotModel!
    private var engine: EarshotEngine!

    override func setUpWithError() throws {
        model = try TestSupport.model()
        engine = try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 480, instances: 3, slack16: 43)
    }

    /// Plays 16 kHz audio into the engine as 48 kHz (each sample 3 times; the stub averages 3:1, the
    /// real engine's FIR decimator low-passes it, which keeps this 220 Hz speech-like signal).
    private func render(_ x16: [Float]) {
        let x48 = x16.flatMap { [$0, $0, $0] }
        var out = [Float](repeating: 0, count: 480)
        var i = 0
        while i + 480 <= x48.count {
            x48.withUnsafeBufferPointer { p in
                out.withUnsafeMutableBufferPointer { o in
                    es_engine_render(engine.cEngine, p.baseAddress! + i, o.baseAddress, 480)
                }
            }
            i += 480
        }
    }

    /// Renders `x` in 250 ms steps and runs one capture step after each, like the 250 ms timer.
    private func run(_ core: inout CaptureCore, _ tap: inout InputTap, _ x: [Float]) -> CaptureCore.Outcome {
        var outcome = CaptureCore.Outcome.recording
        for start in stride(from: 0, to: x.count, by: 4_000) {
            render(Array(x[start..<min(x.count, start + 4_000)]))
            outcome = core.step(tap: &tap, sameEngine: true, vad: model.vad)
            if outcome != .recording { break }
        }
        return outcome
    }

    func testTargetStopsAtSixSecondsOfSpeech() {
        render(TestSupport.speechLike(seconds: 2, from: 0, to: 2)) // old audio the capture must not see
        var tap = engine.makeInputTap()
        var core = CaptureCore(limits: .target)
        let outcome = run(&core, &tap, TestSupport.speechLike(seconds: 15, from: 0.5, to: 15))
        XCTAssertEqual(outcome, .enoughSpeech)
        XCTAssertGreaterThanOrEqual(core.speechSeconds, 6)
        XCTAssertLessThan(core.elapsedSeconds, 15)
        // The real decimator's first output still carries filter memory of the -80 dBFS hum that ended
        // the old audio (measured -4.2e-6), so it gets the hum's own size as tolerance.
        XCTAssertEqual(core.samples.first ?? 1, 0.0001 * Float(sin(0.0)), accuracy: TestSupport.engineIsStub ? 1e-6 : 1e-4,
                       "the capture starts with the new audio, not the 2 s rendered before the tap")
    }

    func testDoneIsAllowedFromFourSeconds() {
        var tap = engine.makeInputTap()
        var core = CaptureCore(limits: .target)
        XCTAssertFalse(core.canFinishEarly)
        let x = TestSupport.speechLike(seconds: 15, from: 0.5, to: 15)
        var sawDoneBeforeStop = false
        for start in stride(from: 0, to: x.count, by: 4_000) {
            render(Array(x[start..<(start + 4_000)]))
            if core.step(tap: &tap, sameEngine: true, vad: model.vad) != .recording { break }
            if core.canFinishEarly {
                sawDoneBeforeStop = true
                XCTAssertGreaterThanOrEqual(core.speechSeconds, 4)
            }
        }
        XCTAssertTrue(sawDoneBeforeStop)
    }

    func testFifteenSecondCapWithoutSpeech() {
        var tap = engine.makeInputTap()
        var core = CaptureCore(limits: .target)
        let hum = [Float](repeating: 0.001, count: 17 * 16_000) // constant level: es_vad finds no speech
        XCTAssertEqual(run(&core, &tap, hum), .timeUp)
        XCTAssertEqual(core.samples.count, 15 * 16_000, "never more than the preallocated 15 s")
        XCTAssertEqual(core.speechSeconds, 0)
    }

    func testOwnVoiceLimits() {
        XCTAssertEqual(CaptureLimits.own.stopSpeech, 8)
        XCTAssertEqual(CaptureLimits.own.maxSamples, 20 * 16_000)
        var tap = engine.makeInputTap()
        var core = CaptureCore(limits: .own)
        XCTAssertEqual(run(&core, &tap, TestSupport.speechLike(seconds: 20, from: 0.5, to: 20)), .enoughSpeech)
        XCTAssertGreaterThanOrEqual(core.speechSeconds, 8)
    }

    func testDroppedSamplesFailTheCapture() {
        var tap = engine.makeInputTap()
        var core = CaptureCore(limits: .target)
        render(TestSupport.speechLike(seconds: 1, from: 0, to: 1))
        XCTAssertEqual(core.step(tap: &tap, sameEngine: true, vad: model.vad), .recording)
        render([Float](repeating: 0.01, count: 16 * 16_000)) // the reader stalls for 16 s (> 15.36 s held)
        XCTAssertEqual(core.step(tap: &tap, sameEngine: true, vad: model.vad), .interrupted)
        XCTAssertGreaterThan(tap.droppedSamples, 0)
        XCTAssertEqual(core.step(tap: &tap, sameEngine: true, vad: model.vad), .interrupted, "stays failed")
    }

    func testEngineSwapFailsTheCapture() {
        var tap = engine.makeInputTap()
        var core = CaptureCore(limits: .target)
        render(TestSupport.speechLike(seconds: 1, from: 0, to: 1))
        XCTAssertEqual(core.step(tap: &tap, sameEngine: false, vad: model.vad), .interrupted)
    }

    func testDirectionIsTheCircularMeanDuringSpeech() {
        var mask = [UInt8](repeating: 0, count: 100)
        for f in 20..<80 { mask[f] = 1 }
        // Samples inside speech frames at 350 and 10 degrees: the circular mean is 0, not 180.
        let yaw: [(sample: Int, degrees: Double?)] = [(512 * 30, 350), (512 * 40, 10), (512 * 90, 120)]
        XCTAssertEqual(CaptureCore.direction(yaw: yaw, mask: mask) ?? 99, 0, accuracy: 1e-9,
                       "the 120 degree sample is outside speech and ignored")
        // Motion live for less than half of the capture: no direction.
        let mostlyDead: [(sample: Int, degrees: Double?)] = [(512 * 30, 10), (512 * 31, nil), (512 * 32, nil)]
        XCTAssertNil(CaptureCore.direction(yaw: mostlyDead, mask: mask))
        XCTAssertNil(CaptureCore.direction(yaw: [], mask: mask))
        // Opposite yaws cancel out: the mean is undefined, not a confident 180 (review finding, I-2).
        let opposite: [(sample: Int, degrees: Double?)] = [(512 * 30, 90), (512 * 40, 270)]
        XCTAssertNil(CaptureCore.direction(yaw: opposite, mask: mask))
        // Live motion but no sample inside a speech frame: no direction either.
        XCTAssertNil(CaptureCore.direction(yaw: [(512 * 90, 10), (512 * 95, 12)], mask: mask))
    }

    @MainActor
    func testSessionTickKeepsThePreallocatedBuffer() async throws {
        // CaptureSession.tick must step its CaptureCore in place: a copy of the core would make the
        // append inside step() duplicate the whole buffer on every 250 ms tick (review finding, I-2).
        let model = try TestSupport.model()
        let engine = try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 480, instances: 3, slack16: 43)
        let audio = AudioController()
        audio.attach(model: model)
        audio.engine = engine // stub engine, no RemoteIO: the test renders audio itself
        let session = CaptureSession()
        try await session.start(audio: audio, model: model, role: .own) // 20 s preallocated
        let base = session.core?.samples.withUnsafeBufferPointer { $0.baseAddress }
        var out = [Float](repeating: 0, count: 480)
        let hum = [Float](repeating: 0.001, count: 480) // no speech, so the capture keeps recording
        for _ in 0..<12 { // 12 ticks of 250 ms (3 s)
            for _ in 0..<25 { // 250 ms at 48 kHz
                hum.withUnsafeBufferPointer { i in
                    out.withUnsafeMutableBufferPointer { o in es_engine_render(engine.cEngine, i.baseAddress, o.baseAddress, 480) }
                }
            }
            session.tick()
            XCTAssertEqual(session.core?.samples.withUnsafeBufferPointer { $0.baseAddress }, base,
                           "the samples buffer never moves")
        }
        XCTAssertEqual(session.core?.samples.count, 12 * 4_000)
        XCTAssertGreaterThanOrEqual(session.core?.samples.capacity ?? 0, CaptureLimits.own.maxSamples)
        XCTAssertEqual(session.phase, .recording)
        session.cancel()
        audio.engine = nil
    }
}
