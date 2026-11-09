// StubEngineTests.swift - the contract 5 wrappers on top of the linked engine (I-0). Written for the
// stub; since lead integration they run on the real engine too (TestSupport.model() is random_v1
// then), except testStubModelIdentity, which only the stub can pass (`make test-stub` runs it).

import XCTest
@testable import Earshot

final class StubEngineTests: XCTestCase {
    private func makeModel() throws -> EarshotModel { try TestSupport.model() }

    /// 3 s of a 300 Hz tone in the middle of near silence: clear speech-like energy for the VAD.
    private func toneClip(seconds: Double = 6) -> [Float] {
        let n = Int(seconds * 16_000)
        var x = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Float(i) / 16_000
            let loud = i > n / 4 && i < 3 * n / 4
            x[i] = loud ? 0.3 * sin(2 * .pi * 300 * t) : 0.0001 * sin(2 * .pi * 50 * t)
        }
        return x
    }

    func testStubModelIdentity() throws {
        try TestSupport.requireStub("the stub's model hash is 64 zeros and es_selftest accepts an empty test vector")
        let m = try makeModel()
        XCTAssertTrue(m.isStub)
        XCTAssertEqual(m.modelHash.count, 64)
        XCTAssertEqual(try m.selfTest(testVector: Data()), 0)
    }

    func testVadFrameCountAndSpeech() throws {
        let m = try makeModel()
        let clip = toneClip()
        let mask = m.vad(clip)
        XCTAssertEqual(mask.count, Int(es_enroll_num_frames(Int32(clip.count))))
        XCTAssertGreaterThan(mask.filter { $0 == 1 }.count, 50, "the tone is speech to the energy VAD")
        XCTAssertEqual(mask.first, 0, "leading silence is not speech")
        XCTAssertEqual(m.vad([Float](repeating: 0.1, count: 500)), [], "under 576 samples gives no frames")
    }

    func testVoiceprintAndConditioning() throws {
        let m = try makeModel()
        let vp = try Voiceprint(model: m, pcm16k: toneClip(), mask: nil, role: .target, weSpeaker: nil)
        XCTAssertGreaterThanOrEqual(vp.tokenCount, 4)
        XCTAssertLessThanOrEqual(vp.tokenCount, 48)
        XCTAssertFalse(vp.hasWeSpeaker)
        let norm = vp.g.reduce(0) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1, accuracy: 1e-4, "g is unit length")
        XCTAssertNoThrow(try Conditioning(model: m, mode: .person, target: vp, own: nil, useWeSpeaker: false))
        XCTAssertNoThrow(try Conditioning(model: m, mode: .everyone, target: nil, own: nil, useWeSpeaker: false))
        XCTAssertThrowsError(try Conditioning(model: m, mode: .person, target: nil, own: nil, useWeSpeaker: false))
    }

    func testShortEnrollmentFails() throws {
        let m = try makeModel()
        let silence = [Float](repeating: 0, count: 16_000)
        XCTAssertThrowsError(try Voiceprint(model: m, pcm16k: silence, mask: nil, role: .target, weSpeaker: nil)) {
            XCTAssertEqual(($0 as? EarshotError)?.status, -6, "ES_ERR_ENROLL_SHORT")
        }
        XCTAssertThrowsError(try Voiceprint(model: m, pcm16k: toneClip(), mask: [1, 1], role: .target,
                                            weSpeaker: nil), "mask length must match the frame count")
    }

    func testEngineRejectsBadConfig() throws {
        let m = try makeModel()
        XCTAssertThrowsError(try EarshotEngine(model: m, ioRate: 16_000, ioFrames: 64, instances: 3, slack16: 43)) {
            XCTAssertEqual(($0 as? EarshotError)?.status, -9, "ES_ERR_RATE")
        }
        XCTAssertThrowsError(try EarshotEngine(model: m, ioRate: 48_000, ioFrames: 64, instances: 3, slack16: 30))
        XCTAssertNoThrow(try EarshotEngine(model: m, ioRate: 44_100, ioFrames: 1104, instances: 1, slack16: 22))
    }

    func testDefaultParamsMatchC() {
        var c = es_params()
        es_params_default(&c)
        let s = EngineParams().cValue
        XCTAssertEqual(s.voice_db, c.voice_db)
        XCTAssertEqual(s.room_db, c.room_db)
        XCTAssertEqual(s.max_gain_db, c.max_gain_db)
        XCTAssertEqual(s.out_gain_db, c.out_gain_db)
        XCTAssertEqual(s.slot_gain.0, c.slot_gain.0)
        XCTAssertEqual(s.slot_gain.1, c.slot_gain.1)
        XCTAssertEqual(s.slot_gain.2, c.slot_gain.2)
        XCTAssertEqual(s.xfade_ms, c.xfade_ms)
        XCTAssertEqual(s.duck_own, c.duck_own)
        XCTAssertEqual(s.gate, c.gate)
        XCTAssertEqual(s.agc, c.agc)
        XCTAssertEqual(s.bypass, c.bypass)
    }

    @MainActor
    func testConditioningAndParamsOnEngine() async throws {
        let m = try makeModel()
        let e = try EarshotEngine(model: m, ioRate: 48_000, ioFrames: 64, instances: 3, slack16: 43)
        let everyone = try Conditioning(model: m, mode: .everyone, target: nil, own: nil, useWeSpeaker: false)
        try await e.setConditioning(slot: 0, everyone, resetState: true)
        try await e.clearConditioning(slot: 0)
        e.setParams(EngineParams())
        do {
            try await e.setConditioning(slot: 5, everyone, resetState: true)
            XCTFail("slot 5 does not exist")
        } catch {
            XCTAssertEqual((error as? EarshotError)?.status, -1, "ES_ERR_ARG")
        }
    }

    /// Starts muted, fades in after unmute, never exceeds the -3 dBFS ceiling.
    /// The real engine plays model output, so the test asks for bypass (the delayed input, contract
    /// 4.2 step 7) and runs the worker's hops itself after each callback; the stub ignores both.
    @MainActor
    func testMuteFadeAndCeiling() throws {
        let m = try makeModel()
        let e = try EarshotEngine(model: m, ioRate: 48_000, ioFrames: 480, instances: 1, slack16: 43)
        var p = EngineParams()
        p.bypass = true
        e.setParams(p)
        let input = [Float](repeating: 0.9, count: 480)
        var output = [Float](repeating: 0, count: 480)
        func renderOnce() {
            output = TestSupport.renderWithHops(e, input, frames: 480)
        }
        renderOnce()
        XCTAssertEqual(output.map { abs($0) }.max(), 0, "a new engine is muted")
        e.setMuted(false)
        renderOnce()
        XCTAssertLessThan(output[479], 0.05, "10 ms into a 400 ms fade-in is still quiet")
        var peak: Float = 0
        for _ in 0..<100 { // 1 s
            renderOnce()
            peak = max(peak, output.map { abs($0) }.max() ?? 0)
        }
        XCTAssertLessThanOrEqual(peak, 0.7079458 + 1e-6, "limiter ceiling -3 dBFS")
        XCTAssertGreaterThan(peak, 0.6, "fully faded in after 1 s")
        XCTAssertEqual(e.stats().lastFrames, 480)
    }

    /// The probe through a simulated loop (output delayed by 2,400 samples = 50 ms, -20 dB).
    func testLatencyProbeThroughSimulatedLoop() throws {
        let m = try makeModel()
        let e = try EarshotEngine(model: m, ioRate: 48_000, ioFrames: 64, instances: 1, slack16: 43)
        e.setMuted(false)
        let delay = 2_400
        var line = [Float](repeating: 0, count: 48_000 * 6)
        var t = 0
        var input = [Float](repeating: 0, count: 64)
        var output = [Float](repeating: 0, count: 64)
        try e.startLatencyProbe()
        XCTAssertNil(e.latencyProbeResult(), "nil while running")
        for _ in 0..<(48_000 * 5 / 64) {
            for i in 0..<64 { input[i] = t + i >= delay ? 0.1 * line[t + i - delay] : 0 }
            input.withUnsafeBufferPointer { inPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    es_engine_render(e.cEngine, inPtr.baseAddress, outPtr.baseAddress, 64)
                }
            }
            for i in 0..<64 { line[t + i] = output[i] }
            t += 64
        }
        guard case .success(let r)? = e.latencyProbeResult() else { return XCTFail("probe should succeed") }
        XCTAssertEqual(r.loopMS, 50, accuracy: 0.05)
        XCTAssertEqual(r.okSweeps, 5)
        XCTAssertLessThan(r.spreadMS, 0.1)
    }
}
