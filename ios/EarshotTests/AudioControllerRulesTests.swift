// AudioControllerRulesTests.swift - the mute and pause rules of AudioController (30-ios 6 and 11),
// checked without real audio: the controller gets a stub engine and no RemoteIO host, and the
// tests call the same handlers iOS notifications would call.

import AVFoundation
import XCTest
@testable import Earshot

@MainActor
final class AudioControllerRulesTests: XCTestCase {
    private func route(_ type: AVAudioSession.Port) -> RouteInfo {
        RouteInfo(outputs: [Port(type: type, uid: "out-\(type.rawValue)", name: "out")], inputType: .builtInMic,
                  sampleRate: 48_000, ioBufferDuration: 64.0 / 48_000, inputLatency: 0.005, outputLatency: 0.04)
    }

    /// A controller "running" on AirPods with the pass-through test switch on, so it is unmuted.
    private func runningController() throws -> AudioController {
        let audio = AudioController()
        let model = try TestSupport.model() // the stub, or random_v1 with the real engine
        audio.attach(model: model)
        audio.engine = try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 64, instances: 3, slack16: 43)
        audio.route = route(.bluetoothA2DP)
        audio.state = .running
        audio.allowListenWithoutOwnVoice = true // its didSet calls refreshMute
        return audio
    }

    func testRouteChangeStaysMutedUntilTheCheckHasRun() throws {
        let audio = try runningController()
        XCTAssertTrue(audio.isUnmuted, "test setup: playing")

        audio.handleRouteChange(reasonRaw: 2, received: Date())
        audio.routeDebounce?.cancel() // the test plays the part of the 300 ms check itself
        XCTAssertFalse(audio.isUnmuted, "muted at once")
        XCTAssertTrue(audio.routeCheckPending)

        // state and route still describe the old route here; nothing may unmute on them.
        audio.refreshMute()   // what the end of a swap or of a probe calls
        audio.hasOwnVoice = true
        XCTAssertFalse(audio.isUnmuted, "no unmute while the new route is unknown")

        audio.routeCheckPending = false // what the check does once it has read the new route
        audio.refreshMute()
        XCTAssertTrue(audio.isUnmuted)
        audio.engine = nil
    }

    func testProbeIsRefusedWhileTheRouteIsChecked() async throws {
        let audio = try runningController()
        audio.handleRouteChange(reasonRaw: 2, received: Date())
        audio.routeDebounce?.cancel()
        do {
            try await audio.measureLatency()
            XCTFail("the probe must not start during a route check")
        } catch {
            XCTAssertFalse(audio.isMeasuring)
            XCTAssertFalse(audio.isUnmuted, "the probe must not unmute the engine")
        }
        audio.engine = nil
    }

    func testUserPauseWinsOverTheRoute() throws {
        let audio = try runningController()
        audio.pause()
        XCTAssertEqual(audio.state, .paused)
        XCTAssertTrue(audio.userPaused)
        // swapEngine, recreateIO and the route check all settle through this, so the pause survives them.
        XCTAssertEqual(audio.settledState(for: route(.bluetoothA2DP)), .paused)
        XCTAssertEqual(audio.settledState(for: route(.builtInSpeaker)), .paused)

        audio.resume()
        XCTAssertFalse(audio.userPaused)
        XCTAssertEqual(audio.state, .running)
        XCTAssertTrue(audio.isUnmuted, "resume fades back in")
        XCTAssertEqual(audio.settledState(for: route(.builtInSpeaker)), .badRoute)
        audio.engine = nil
    }

    func testPauseAlsoWorksOnABadRoute() throws {
        let audio = try runningController()
        audio.route = route(.builtInSpeaker)
        audio.state = .badRoute
        audio.refreshMute()
        XCTAssertFalse(audio.isUnmuted)

        audio.pause()
        XCTAssertEqual(audio.state, .paused, "the Pause button works on a bad route too")
        audio.route = route(.bluetoothA2DP) // AirPods come back while paused
        XCTAssertEqual(audio.settledState(for: audio.route), .paused, "still paused")

        audio.resume()
        XCTAssertEqual(audio.state, .running)
        XCTAssertTrue(audio.isUnmuted)
        audio.engine = nil
    }

    func testStopClearsThePendingCheckAndThePause() throws {
        let audio = try runningController()
        audio.handleRouteChange(reasonRaw: 2, received: Date())
        audio.pause()
        let before = audio.ioGeneration

        audio.stop()
        XCTAssertEqual(audio.state, .idle)
        XCTAssertNil(audio.engine)
        XCTAssertFalse(audio.routeCheckPending, "the next Start must be able to unmute")
        XCTAssertFalse(audio.userPaused, "the next Start begins unpaused")
        XCTAssertGreaterThan(audio.ioGeneration, before, "a swap or start waiting at an await gives up")
    }

    func testInterruptionMakesWaitingSwapsGiveUp() throws {
        let audio = try runningController()
        let before = audio.ioGeneration
        audio.handleInterruption(began: true, shouldResume: false, routeDisconnected: false)
        XCTAssertEqual(audio.state, .interrupted)
        XCTAssertTrue(audio.ioStoppedBySystem)
        XCTAssertFalse(audio.isUnmuted)
        XCTAssertGreaterThan(audio.ioGeneration, before)

        audio.stop() // Stop during the call
        XCTAssertEqual(audio.state, .idle)
        XCTAssertFalse(audio.ioStoppedBySystem, "the next session's swaps must not be blocked")
        XCTAssertFalse(audio.pendingRecreate)
    }

    // MARK: - Added with I-2 (review: pure mute rule, Stop during a swap, own voice)

    func testMuteRuleTable() {
        // The whole truth table, written by hand rather than copied from shouldPlay: of the 96
        // combinations exactly one plays, and it is running + allowed route + own voice OK + not
        // measuring + no route check pending. Every other combination mutes.
        struct Case: Equatable {
            let state: AudioState
            let allowed, own, measuring, pending: Bool
        }
        let expectedPlaying = [Case(state: .running, allowed: true, own: true, measuring: false, pending: false)]
        let states: [AudioState] = [.idle, .running, .paused, .badRoute, .interrupted, .failed("x")]
        var playing: [Case] = []
        var total = 0
        for state in states {
            for allowed in [false, true] {
                for own in [false, true] {
                    for measuring in [false, true] {
                        for pending in [false, true] {
                            total += 1
                            if AudioController.shouldPlay(state: state, routeAllowed: allowed, ownVoiceOK: own,
                                                          isMeasuring: measuring, routeCheckPending: pending) {
                                playing.append(Case(state: state, allowed: allowed, own: own, measuring: measuring,
                                                    pending: pending))
                            }
                        }
                    }
                }
            }
        }
        XCTAssertEqual(total, 96)
        XCTAssertEqual(playing, expectedPlaying, "only the one fully-allowed combination plays")
    }

    func testStopDuringASwapMakesTheSwapGiveUp() throws {
        // swapEngine remembers (generation, host, engine) before its await and checks them after it.
        let audio = try runningController()
        let old = try XCTUnwrap(audio.engine)
        let generation = audio.ioGeneration
        XCTAssertTrue(audio.swapStillValid(generation: generation, host: audio.host, old: old))

        audio.handleRouteChange(reasonRaw: 2, received: Date()) // a route change alone does not cancel it
        audio.routeDebounce?.cancel()
        XCTAssertTrue(audio.swapStillValid(generation: generation, host: audio.host, old: old))

        audio.stop() // Stop arrives while the swap waits at its await
        XCTAssertFalse(audio.swapStillValid(generation: generation, host: audio.host, old: old),
                       "the swap drops its new engine instead of touching a freed host")
    }

    func testInterruptionOrAnotherEngineAlsoInvalidatesASwap() throws {
        let audio = try runningController()
        let old = try XCTUnwrap(audio.engine)
        let generation = audio.ioGeneration
        audio.engine = try EarshotEngine(model: try TestSupport.model(), ioRate: 48_000, ioFrames: 64,
                                         instances: 3, slack16: 43)
        XCTAssertFalse(audio.swapStillValid(generation: generation, host: audio.host, old: old), "engine replaced")
        audio.engine = old
        audio.handleInterruption(began: true, shouldResume: false, routeDisconnected: false)
        XCTAssertFalse(audio.swapStillValid(generation: generation, host: audio.host, old: old), "a call began")
        audio.stop()
    }

    func testOwnVoiceLeavesSetupModeAndQuickAddHoldsMute() throws {
        let audio = try runningController()
        audio.allowListenWithoutOwnVoice = false
        XCTAssertFalse(audio.isUnmuted, "setup mode: muted until an own voice exists")
        XCTAssertEqual(audio.notice, Copy.setupMuted)

        let model = try TestSupport.model()
        let vp = try Voiceprint(model: model, pcm16k: TestSupport.speechLike(seconds: 10, from: 0.5, to: 9.5),
                                mask: nil, role: .own, weSpeaker: nil)
        audio.setOwnVoice(vp)
        XCTAssertTrue(audio.hasOwnVoice)
        XCTAssertTrue(audio.isUnmuted, "fades in once the own voice is set")
        XCTAssertNoThrow(try audio.everyoneConditioning(), "slot 0 = Everyone with the own voice")

        audio.holdMuted = true // quick add plays its 2 s check clip
        XCTAssertFalse(audio.isUnmuted)
        audio.holdMuted = false
        XCTAssertTrue(audio.isUnmuted)

        audio.setOwnVoice(nil) // the own voice was deleted
        XCTAssertFalse(audio.isUnmuted)
        audio.engine = nil
    }

    func testStopReportsItsTimeForTonightExpiry() throws {
        let audio = try runningController()
        var stoppedAt: Date? = nil
        audio.onStop = { stoppedAt = $0 }
        audio.stop()
        XCTAssertNotNil(stoppedAt)
        XCTAssertFalse(audio.isListening)
    }

    // MARK: - Quick add (review findings, I-2): the RAM copy lives only across an engine swap

    /// A stub engine that takes 480-frame renders (10 ms at 48 kHz).
    private func stubEngine(_ model: EarshotModel) throws -> EarshotEngine {
        try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 480, instances: 3, slack16: 43)
    }

    /// Plays `seconds` of a constant `level` into the engine (it keeps 16 kHz mic history).
    private func feed(_ e: EarshotEngine, seconds: Double, level: Float) {
        let input = [Float](repeating: level, count: 480)
        var out = [Float](repeating: 0, count: 480)
        for _ in 0..<Int(seconds * 100) {
            input.withUnsafeBufferPointer { i in
                out.withUnsafeMutableBufferPointer { o in es_engine_render(e.cEngine, i.baseAddress, o.baseAddress, 480) }
            }
        }
    }

    private func count(_ x: [Float], near level: Float) -> Int { x.filter { abs($0 - level) < 1e-6 }.count }

    /// `expected` samples at one level, of which the real engine's decimator spends up to
    /// decimatorSettle16k settling after the level step (0 with the stub).
    private func assertCount(_ got: Int, _ expected: Int, _ message: String, line: UInt = #line) {
        let range = (expected - TestSupport.decimatorSettle16k)...expected
        XCTAssertTrue(range.contains(got), "\(message): \(got) not in \(range)", line: line)
    }

    func testQuickAddJoinsTheRAMCopyOnlyToTheSwapsEngine() throws {
        let model = try TestSupport.model()
        let audio = AudioController()
        audio.attach(model: model)
        let a = try stubEngine(model)
        audio.engine = a
        feed(a, seconds: 10, level: 0.5)                // talk before the swap

        // What swapEngine does: copy the old engine's last 8 s, then hand it to the new engine.
        let ram = audio.quickAddHistory()
        XCTAssertEqual(count(ram, near: 0.5), 128_000)
        let b = try stubEngine(model)
        audio.engine = b
        audio.handOverQuickAddRAM(ram, to: b)
        feed(b, seconds: 1, level: 0.25)                // 1 s after the swap

        let joined = audio.quickAddHistory()
        XCTAssertEqual(joined.count, 128_000)
        XCTAssertEqual(count(joined, near: 0.5), 112_000, "the newest 7 s of the RAM copy in front")
        assertCount(count(joined, near: 0.25), 16_000, "then the new engine's 1 s")
        XCTAssertEqual(joined.last ?? 0, 0.25, accuracy: 1e-6)

        audio.dropStaleQuickAddRAM()
        XCTAssertFalse(audio.quickAddRAM.isEmpty, "still needed: the new engine has only 1 s")
        feed(b, seconds: 8, level: 0.25)
        audio.dropStaleQuickAddRAM()                    // the 10 Hz poll does this
        XCTAssertTrue(audio.quickAddRAM.isEmpty, "freed once the new engine holds 8 s of its own")
        XCTAssertEqual(count(audio.quickAddHistory(), near: 0.25), 128_000)
        audio.engine = nil
    }

    func testStopThenStartNeverReturnsAudioFromTheOldSession() throws {
        let model = try TestSupport.model()
        let audio = AudioController()
        audio.attach(model: model)
        let a = try stubEngine(model)
        audio.engine = a
        feed(a, seconds: 10, level: 0.5)
        let b = try stubEngine(model)
        let ram = audio.quickAddHistory()
        audio.engine = b
        audio.handOverQuickAddRAM(ram, to: b)            // a swap in session 1

        audio.stop()                                     // Stop (teardownIO)
        XCTAssertTrue(audio.quickAddRAM.isEmpty, "Stop clears the RAM copy")

        let c = try stubEngine(model)                    // Start hours later: a new engine
        audio.engine = c
        feed(c, seconds: 3, level: 0.125)
        let recent = QuickAdd.recentAudio(audio)
        XCTAssertEqual(recent.count, 48_000, "only the 3 s of the new session")
        assertCount(count(recent, near: 0.125), 48_000, "the new session's 3 s")
        XCTAssertEqual(count(recent, near: 0.5), 0, "nothing from session 1")
        XCTAssertLessThan(recent.count, ClipTools.minClip, "too little history: quick add asks to wait")
        audio.stop()
    }

    func testAnEngineWithoutTheHandOverNeverGetsTheRAMCopy() throws {
        // A RemoteIO rebuild (rate change, media reset) installs an engine without a swap hand-over.
        let model = try TestSupport.model()
        let audio = AudioController()
        audio.attach(model: model)
        let a = try stubEngine(model)
        audio.engine = a
        feed(a, seconds: 10, level: 0.5)
        let b = try stubEngine(model)
        audio.handOverQuickAddRAM(audio.quickAddHistory(), to: b)
        let other = try stubEngine(model)
        audio.engine = other
        feed(other, seconds: 2, level: 0.25)
        let recent = audio.quickAddHistory()
        XCTAssertEqual(recent.count, 32_000)
        XCTAssertEqual(count(recent, near: 0.5), 0)
        XCTAssertTrue(audio.quickAddRAM.isEmpty, "a copy that belongs to another engine is dropped")
        audio.engine = nil
    }

    func testQuickAddPlaybackNeedsAnAllowedRoute() async throws {
        XCTAssertTrue(QuickAddPlayer.mayPlay(routeAllowed: true, routeCheckPending: false))
        XCTAssertFalse(QuickAddPlayer.mayPlay(routeAllowed: false, routeCheckPending: false), "phone speaker")
        XCTAssertFalse(QuickAddPlayer.mayPlay(routeAllowed: true, routeCheckPending: true), "route being checked")

        let audio = try runningController()
        audio.route = route(.builtInSpeaker)
        audio.state = .badRoute
        let played = try await QuickAddPlayer().play([Float](repeating: 0.1, count: 32_000), audio: audio)
        XCTAssertFalse(played, "another person's voice never plays through the phone speaker")
        XCTAssertFalse(audio.holdMuted)
        audio.engine = nil
    }

    func testAppVersionComesFromTheProject() {
        // app_version in telemetry and the Phase 0 file is "0.1 (1)" (30-ios 15.3).
        let info = Bundle.main.infoDictionary ?? [:]
        XCTAssertEqual(info["CFBundleShortVersionString"] as? String, "0.1")
        XCTAssertEqual(info["CFBundleVersion"] as? String, "1")
    }
}
