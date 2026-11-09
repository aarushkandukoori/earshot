// SafetyIntegrationTests.swift - the I-4 rules wired into AudioController and Selector with the stub
// engine and no RemoteIO: the thermal downshift (fade, clear, refill cold), the critical pause,
// the alarm passthrough (params, beeps, banner, playing through a pause), the locked check banner,
// and the session log record. The tests call what iOS or the 10 Hz poll would call;
// AudioController.onEngineEvent records what reached the engine.

import AVFoundation
import XCTest
@testable import Earshot

@MainActor
final class SafetyIntegrationTests: XCTestCase {
    private var root: URL!
    private let clip = TestSupport.speechLike(seconds: 6, from: 0.5, to: 5.5)

    override func setUpWithError() throws { root = TestSupport.tempFolder() }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// A controller "running" on AirPods with the stub engine, a people store in a temp folder, a
    /// Selector attached to both, and the session log in the temp folder.
    private func rig() throws -> (AudioController, PeopleStore, Earshot.Selector, EarshotModel) {
        let model = try TestSupport.model()
        let audio = AudioController()
        audio.sessionLog = SessionLog(folder: root.appendingPathComponent("logs"))
        audio.attach(model: model)
        audio.engine = try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 64, instances: 3, slack16: 43)
        audio.route = RouteInfo(outputs: [Port(type: .bluetoothA2DP, uid: "pods", name: "AirPods")],
                                inputType: .builtInMic, sampleRate: 48_000, ioBufferDuration: 64.0 / 48_000,
                                inputLatency: 0.005, outputLatency: 0.04)
        audio.state = .running
        let people = PeopleStore(files: PersonFiles(root: root.appendingPathComponent("People")), prepareLogs: false)
        let selector = Earshot.Selector()
        selector.attach(audio: audio, people: people, model: model, enrollment: nil)
        return (audio, people, selector, model)
    }

    private func addPerson(_ name: String, _ people: PeopleStore, _ model: EarshotModel) throws -> Person {
        let mask = model.vad(clip)
        let p = Person(name: name, kind: .tonight, created: Date(), expires: Date().addingTimeInterval(3600),
                       yawDeg: nil, modelHash: model.modelHash, eWespk: nil, mask: mask, quality: Quality(speechS: 5))
        let vp = try Voiceprint(model: model, pcm16k: clip, mask: mask, role: .target, weSpeaker: nil)
        try people.add(p, clip: clip, voiceprint: vp)
        return p
    }

    // MARK: - Thermal downshift

    func testSavingPowerEmptiesTheOtherSlotsAndRecoveryRefillsThemCold() async throws {
        let (audio, people, selector, model) = try rig()
        let a = try addPerson("A", people, model)
        let b = try addPerson("B", people, model)
        selector.addToTable(a, freshYaw: nil) // slot 1
        selector.addToTable(b, freshYaw: nil) // slot 2, focused
        selector.setMode(.table)
        XCTAssertEqual(selector.table.mode, .table)
        await audio.condChainIdle()

        let log = SafetyEngineLog()
        audio.onEngineEvent = { log.events.append($0) }
        audio.handlePowerChange(thermal: .serious, lowPower: false, now: 0)
        XCTAssertEqual(audio.powerLevel, .saving)
        XCTAssertTrue(selector.powerSaving)
        XCTAssertEqual(selector.table.mode, .focus, "Table becomes Focus on the selected person")
        XCTAssertEqual(selector.table.selected, 2)
        XCTAssertEqual(audio.notice, Copy.savingPower)
        XCTAssertEqual(audio.params.slotGain, [0, 0, 1], "the selected person keeps playing")
        await audio.condChainIdle()
        let clear0 = try XCTUnwrap(log.events.firstIndex(of: .clear(slot: 0)), "\(log.events)")
        let clear1 = try XCTUnwrap(log.events.firstIndex(of: .clear(slot: 1)), "\(log.events)")
        let beforeClears = log.gains(log.events[..<min(clear0, clear1)])
        XCTAssertFalse(beforeClears.isEmpty)
        XCTAssertTrue(beforeClears.allSatisfy { $0[0] == 0 && $0[1] == 0 && $0[2] == 1 },
                      "slots 0 and 1 at gain 0 before the clears, slot 2 never interrupted: \(beforeClears)")
        XCTAssertEqual(selector.engineSlots().map { $0.slot }, [2], "only the selected slot is computed")
        XCTAssertEqual(selector.telemetryTable().instances, 1)

        // Table is refused while saving: nothing changes, 1 beep = still Focus.
        log.events.removeAll()
        selector.setMode(.table)
        XCTAssertEqual(selector.table.mode, .focus)
        XCTAssertEqual(log.cues, [Cue.mode(.focus)])

        // Recovery: 60 calm seconds, then the emptied slots come back cold.
        audio.handlePowerChange(thermal: .nominal, lowPower: false, now: 1)
        XCTAssertEqual(audio.powerLevel, .saving)
        audio.handlePowerChange(thermal: .nominal, lowPower: false, now: 61)
        XCTAssertEqual(audio.powerLevel, .normal)
        XCTAssertFalse(selector.powerSaving)
        XCTAssertNil(audio.notice)
        await audio.condChainIdle()
        XCTAssertTrue(log.events.contains(.load(slot: 0, reset: true)), "\(log.events)")
        XCTAssertTrue(log.events.contains(.load(slot: 1, reset: true)), "\(log.events)")
        XCTAssertFalse(log.events.contains(.load(slot: 2, reset: true)), "the playing slot is not reloaded")
        XCTAssertEqual(selector.engineSlots().map { $0.slot }, [0, 1, 2])
        audio.engine = nil
    }

    func testCriticalFadesToPauseWithANotice() throws {
        let (audio, _, _, _) = try rig()
        audio.allowListenWithoutOwnVoice = true
        XCTAssertTrue(audio.isUnmuted)
        audio.handlePowerChange(thermal: .critical, lowPower: false, now: 0)
        XCTAssertEqual(audio.powerLevel, .critical)
        XCTAssertEqual(audio.state, .paused)
        XCTAssertEqual(audio.notice, Copy.tooHot)
        audio.handlePowerChange(thermal: .fair, lowPower: false, now: 5)
        XCTAssertEqual(audio.powerLevel, .saving, "cooler, but still saving for 60 s")
        XCTAssertEqual(audio.state, .paused, "the user resumes, never automatic")
        XCTAssertEqual(audio.notice, Copy.tooHot, "the banner keeps saying why Earshot is paused")
        audio.engine = nil
    }

    /// A critical change during a call: iOS has stopped I/O, so pause() does nothing, but the pause
    /// is remembered and the swap after the call settles paused (I-4 review).
    func testCriticalDuringACallStaysPausedAfterTheCall() throws {
        let (audio, _, _, _) = try rig()
        audio.allowListenWithoutOwnVoice = true
        audio.handleInterruption(began: true, shouldResume: false, routeDisconnected: false)
        XCTAssertEqual(audio.state, .interrupted)
        audio.handlePowerChange(thermal: .critical, lowPower: false, now: 0)
        XCTAssertTrue(audio.userPaused, "remembered although pause() only acts while running")
        XCTAssertEqual(audio.settledState(for: audio.route), .paused, "the swap after the call stays paused")
        XCTAssertEqual(audio.notice, Copy.tooHot)
        XCTAssertTrue(audio.keepsHotNotice, "the end of the call keeps the banner")
        audio.resume()
        XCTAssertFalse(audio.userPaused, "only the user resumes")
        audio.engine = nil
    }

    func testAddingAPersonWhileSavingKeepsOneSlotComputed() async throws {
        let (audio, people, selector, model) = try rig()
        audio.handlePowerChange(thermal: .fair, lowPower: true, now: 0) // Low Power Mode
        XCTAssertTrue(selector.powerSaving)
        XCTAssertEqual(selector.engineSlots().map { $0.slot }, [0], "Everyone only")
        let a = try addPerson("A", people, model)
        selector.addToTable(a, freshYaw: nil)
        await audio.condChainIdle()
        XCTAssertEqual(selector.engineSlots().map { $0.slot }, [1], "the new person replaces Everyone in the engine")
        XCTAssertEqual(audio.params.slotGain, [0, 1, 0])
        audio.engine = nil
    }

    // MARK: - Alarm passthrough

    func testAlarmPassthroughWhileListening() throws {
        let (audio, _, _, _) = try rig()
        audio.allowListenWithoutOwnVoice = true
        let log = SafetyEngineLog()
        audio.onEngineEvent = { log.events.append($0) }
        audio.alarmHeard(now: 100)
        XCTAssertTrue(audio.alarm.isActive)
        XCTAssertEqual(log.cues, [Cue.alarm], "3 beeps")
        XCTAssertEqual(audio.notice, Copy.alarmPassthroughBanner)
        let sent = audio.safetyApplied(audio.params)
        XCTAssertTrue(sent.bypass)
        XCTAssertEqual(sent.outGainDB, -6)
        XCTAssertFalse(audio.params.bypass, "the real params are kept for the restore")

        audio.alarmHeard(now: 103) // still ringing: extends, no new beeps
        XCTAssertEqual(log.cues, [Cue.alarm])
        audio.alarmTick(now: 107.9)
        XCTAssertTrue(audio.alarm.isActive)
        audio.alarmTick(now: 108)
        XCTAssertFalse(audio.alarm.isActive)
        XCTAssertNil(audio.notice)
        XCTAssertFalse(audio.safetyApplied(audio.params).bypass)
        audio.engine = nil
    }

    func testAlarmPlaysThroughAPauseThenMutesAgain() async throws {
        let (audio, _, _, _) = try rig()
        audio.allowListenWithoutOwnVoice = true
        audio.pause()
        try await Task.sleep(nanoseconds: 250_000_000)
        XCTAssertFalse(audio.isUnmuted, "paused: muted after 150 ms")
        audio.alarmHeard(now: 10)
        XCTAssertTrue(audio.isUnmuted, "the room comes through while paused")
        XCTAssertEqual(audio.state, .paused)
        audio.alarmTick(now: 15)
        XCTAssertFalse(audio.isUnmuted, "muted again after 5 s")
        XCTAssertEqual(audio.state, .paused)
        audio.engine = nil
    }

    func testTurningTheSettingOffEndsThePassthrough() throws {
        let (audio, _, _, _) = try rig()
        audio.alarmHeard(now: 0)
        XCTAssertTrue(audio.alarm.isActive)
        var s = ListenSettings()
        s.alarmPassthrough = false
        audio.applySettings(s)
        XCTAssertFalse(audio.alarm.isActive)
        audio.alarmHeard(now: 1)
        XCTAssertFalse(audio.alarm.isActive, "ignored while the setting is off")
        audio.engine = nil
    }

    func testMuteRuleWithAnAlarm() {
        func play(_ state: AudioState, route: Bool = true, own: Bool = true, measuring: Bool = false,
                  pending: Bool = false, alarm: Bool) -> Bool {
            AudioController.shouldPlay(state: state, routeAllowed: route, ownVoiceOK: own, isMeasuring: measuring,
                                       routeCheckPending: pending, alarm: alarm)
        }
        XCTAssertTrue(play(.paused, alarm: true))
        XCTAssertFalse(play(.paused, alarm: false))
        XCTAssertTrue(play(.running, alarm: true))
        XCTAssertFalse(play(.paused, route: false, alarm: true), "never on a bad route")
        XCTAssertFalse(play(.paused, own: false, alarm: true), "setup mode stays muted")
        XCTAssertFalse(play(.paused, measuring: true, alarm: true))
        XCTAssertFalse(play(.paused, pending: true, alarm: true))
        XCTAssertFalse(play(.interrupted, alarm: true))
        XCTAssertFalse(play(.badRoute, alarm: true))
    }

    func testFeedbackBanner() throws {
        let (audio, _, _, _) = try rig()
        var s = EngineStats()
        s.howl = true
        audio.safetyPoll(s)
        XCTAssertEqual(audio.notice, Copy.feedback)
        s.howl = false
        audio.safetyPoll(s)
        XCTAssertNil(audio.notice)
        audio.engine = nil
    }

    // MARK: - Locked check

    func testLostForFiveSecondsShowsTheBannerAndALockClearsIt() throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model)
        selector.addToTable(maya, freshYaw: nil) // slot 1, focused
        selector.lockedCos = 0.3
        let job = LockJob(slot: 1, person: maya.id, g: [1, 0])
        for second in 1...4 {
            selector.finishLockedCheck(jobs: [job], cosines: [1: 0.1])
            XCTAssertFalse(selector.lostLockWarning, "second \(second)")
        }
        XCTAssertEqual(selector.lockStates[1], .lost)
        selector.finishLockedCheck(jobs: [job], cosines: [1: 0.1])
        XCTAssertTrue(selector.lostLockWarning, "lost for 5 s")
        selector.finishLockedCheck(jobs: [job], cosines: [1: 0.8])
        XCTAssertFalse(selector.lostLockWarning)
        XCTAssertEqual(selector.lockStates[1], .locked, "check mark on the card")
        audio.engine = nil
    }

    func testSwitchingAwayForgetsTheSlot() throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model)
        selector.addToTable(maya, freshYaw: nil)
        let job = LockJob(slot: 1, person: maya.id, g: [1, 0])
        for _ in 0..<5 { selector.finishLockedCheck(jobs: [job], cosines: [1: -1]) }
        XCTAssertTrue(selector.lostLockWarning)
        selector.setMode(.everyone) // Maya's slot goes to gain 0
        selector.runLockedCheck()
        XCTAssertFalse(selector.lostLockWarning)
        XCTAssertEqual(selector.lockStates, [.unknown, .unknown, .unknown])
        audio.engine = nil
    }

    func testSilenceIsNotChecked() throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model)
        selector.addToTable(maya, freshYaw: nil)
        var stats = EngineStats()
        stats.pTarget = [0, 0.2, 0] // Maya is quiet
        for i in 0..<30 { selector.tick(stats, now: 100 + Double(i) / 10) }
        XCTAssertFalse(selector.lockedCheckBusy, "no job: p_target mean under 0.5")
        XCTAssertEqual(selector.lockStates[1], .unknown)
        audio.engine = nil
    }

    // MARK: - Session log record

    func testMinuteRecordUsesTheTableAndTheRoute() async throws {
        let (audio, people, selector, model) = try rig()
        let a = try addPerson("A", people, model)
        selector.addToTable(a, freshYaw: nil)
        var stats = EngineStats()
        stats.underruns = 2
        stats.pTarget = [0, 0.8, 0]
        audio.safetyPoll(stats)
        let r = audio.makeRecord(now: Date(timeIntervalSince1970: 1_800_000_000))
        XCTAssertEqual(r.route, AVAudioSession.Port.bluetoothA2DP.rawValue, "port type, never the user-set name")
        XCTAssertEqual(r.mode, "focus")
        XCTAssertEqual(r.nInstances, 2, "Everyone warm in 0 and A in 1")
        XCTAssertEqual(r.underruns, 2)
        XCTAssertEqual(r.pTargetMean, [0, Double(Float(0.8)), 0])
        XCTAssertEqual(r.modelHash, model.modelHash)
        XCTAssertTrue(r.appVersion.contains("("), r.appVersion)

        audio.writeMinute(now: Date(timeIntervalSince1970: 1_800_000_000))
        let done = expectation(description: "written")
        audio.sessionLog.sync { done.fulfill() }
        await fulfillment(of: [done], timeout: 5)
        let files = try FileManager.default.contentsOfDirectory(atPath: audio.sessionLog.folder.path)
        XCTAssertEqual(files.count, 1)
        XCTAssertEqual(audio.minute.underruns, 0, "the next minute starts at 0")
        audio.engine = nil
    }

    /// Stop writes the last part-minute, so its underruns and battery are not lost (I-4 review).
    func testStopWritesTheLastPartMinute() async throws {
        let (audio, _, _, _) = try rig()
        var stats = EngineStats()
        stats.underruns = 3
        audio.safetyPoll(stats)
        audio.stopSafety()
        let done = expectation(description: "written")
        audio.sessionLog.sync { done.fulfill() }
        await fulfillment(of: [done], timeout: 5)
        let files = try FileManager.default.contentsOfDirectory(atPath: audio.sessionLog.folder.path)
        XCTAssertEqual(files.count, 1)
        let text = try String(contentsOf: audio.sessionLog.folder.appendingPathComponent(files[0]), encoding: .utf8)
        XCTAssertEqual(text.split(separator: "\n").count, 1, "one line")
        XCTAssertTrue(text.contains("\"underruns\":3"), text)
        audio.stopSafety() // nothing new since: no second line
        let again = expectation(description: "written again")
        audio.sessionLog.sync { again.fulfill() }
        await fulfillment(of: [again], timeout: 5)
        let text2 = try String(contentsOf: audio.sessionLog.folder.appendingPathComponent(files[0]), encoding: .utf8)
        XCTAssertEqual(text2.split(separator: "\n").count, 1)
        audio.engine = nil
    }
}

/// Collects AudioController.onEngineEvent in order (main actor, like the controller).
@MainActor
final class SafetyEngineLog {
    var events: [EngineEvent] = []
    var cues: [Int] { events.compactMap { if case .cue(let n) = $0 { return n } else { return nil } } }
    func gains(_ slice: ArraySlice<EngineEvent>) -> [[Float]] {
        slice.compactMap { if case .gains(let g) = $0 { return g } else { return nil } }
    }
}
