// AudioStartSmokeTests.swift - opt-in: really starts the audio session and RemoteIO with the
// linked engine in the simulator (start, callbacks flowing, one forced engine swap, stop). Written for
// the stub (hence the test name); since lead integration `make test` links the real engine, and then
// the test also checks that the real worker thread runs hops and makes no allocations, first in
// Everyone mode with 1 slot, then (I-4 review) in Person mode with all 3 slots computed.
// Skipped by default because it needs microphone permission. To run it:
//   xcrun simctl privacy booted grant microphone com.aarushkandukoori.earshot
//   TEST_RUNNER_EARSHOT_AUDIO_SMOKE=1 make test

import XCTest
@testable import Earshot

@MainActor
final class AudioStartSmokeTests: XCTestCase {
    func testStartSwapStopWithTheStub() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["EARSHOT_AUDIO_SMOKE"] == "1",
                          "set TEST_RUNNER_EARSHOT_AUDIO_SMOKE=1 to run")
        let audio = AudioController()
        audio.attach(model: try TestSupport.model())
        try await audio.start()
        try await Task.sleep(nanoseconds: 1_500_000_000)
        XCTAssertGreaterThan(audio.stats.lastFrames, 0, "render callbacks are flowing")
        if !TestSupport.engineIsStub {
            let s = audio.stats
            print("smoke real engine: hops \(s.hops), underruns \(s.underruns), frames \(s.lastFrames), "
                  + "compute_us_max \(s.computeUsMax), workgroup \(audio.hasWorkgroup)")
            XCTAssertGreaterThan(s.hops, 0, "the real engine's worker thread runs hops")
            try await checkNoAllocationsOnTheAudioThreads(audio, what: "Everyone, 1 instance")
            try await seatThreePeople(audio)
            try await Task.sleep(nanoseconds: 500_000_000) // the loads land and the slots warm up
            try await checkNoAllocationsOnTheAudioThreads(audio, what: "Person mode, 3 instances")
        }
        XCTAssertTrue(audio.state == .running || audio.state == .badRoute, "state is \(audio.state)")
        XCTAssertFalse(audio.isUnmuted, "no own voice yet, so setup mode stays muted")

        await audio.swapEngine(reason: .forced)
        XCTAssertTrue(audio.notes.contains { $0.contains("swap (forced) in") }, "notes: \(audio.notes)")
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertGreaterThan(audio.stats.lastFrames, 0, "the new engine renders")

        // A user pause survives an engine swap, and Resume leaves it.
        audio.pause()
        XCTAssertEqual(audio.state, .paused)
        await audio.swapEngine(reason: .forced)
        XCTAssertEqual(audio.state, .paused, "still paused after the swap")
        XCTAssertFalse(audio.isUnmuted)
        audio.resume()
        XCTAssertTrue(audio.state == .running || audio.state == .badRoute, "state is \(audio.state)")

        audio.stop()
        XCTAssertEqual(audio.state, .idle)
    }

    /// Lead integration, contract 4.1 on the simulator: with the debug malloc counter (Debug builds,
    /// Audio/MallocHook.c) armed on the RemoteIO render thread and the engine worker, 2 s of real
    /// engine audio must show 0 allocations and 0 frees on both. The counter arms itself once a
    /// second (safetySecond), so this waits until it reports `.armed` first.
    private func checkNoAllocationsOnTheAudioThreads(_ audio: AudioController, what: String) async throws {
        let counter = audio.mallocCounter
        guard counter.compiledIn else { return print("smoke malloc: hook not in this build (Release)") }
        for _ in 0..<40 where counter.status != .armed { try await Task.sleep(nanoseconds: 100_000_000) }
        XCTAssertEqual(counter.status, .armed, "render tid \(counter.renderTID), worker tid \(String(describing: counter.workerTID))")
        guard counter.status == .armed else { return }
        let before = counter.snapshot()
        let hops0 = audio.stats.hops
        try await Task.sleep(nanoseconds: 2_000_000_000)
        let d = counter.snapshot().minus(before)
        print("smoke malloc (2 s, real engine, \(what)): render \(d.renderAllocs) allocs / \(d.renderFrees) frees, worker "
              + "\(d.workerAllocs) / \(d.workerFrees), worker time constraint \(String(describing: counter.workerTimeConstraint)), "
              + "hops \(audio.stats.hops - hops0) in the window, compute_us_max \(audio.stats.computeUsMax)")
        XCTAssertEqual(d.total, 0, "no malloc or free on the render or worker thread (contract 4.1), \(what)")
    }

    /// I-4 review (ES_HOP_SGEMV evidence): the first check runs Everyone in slot 0 only. This puts a
    /// Person-mode conditioning (a 9.5 s speech-like target voiceprint, with an own voice as the
    /// negative) into all 3 slots at gain 1, so the worker computes 3 instances with their token
    /// attention while the counter stays armed on the same engine.
    private func seatThreePeople(_ audio: AudioController) async throws {
        let model = try XCTUnwrap(audio.model)
        let e = try XCTUnwrap(audio.engine)
        let clip = TestSupport.speechLike(seconds: 9.5, from: 0.2, to: 9.3)
        let mask = model.vad(clip)
        let target = try Voiceprint(model: model, pcm16k: clip, mask: mask, role: .target, weSpeaker: nil)
        let own = try Voiceprint(model: model, pcm16k: clip, mask: mask, role: .own, weSpeaker: nil)
        let c = try Conditioning(model: model, mode: .person, target: target, own: own, useWeSpeaker: false)
        for slot in 0..<3 { try await e.setConditioning(slot: slot, c, resetState: true) }
        var p = EngineParams()
        p.slotGain = [1, 1, 1]
        e.setParams(p)
        print("smoke: 3 person slots loaded, target speech \(target.speechSeconds) s")
    }
}
