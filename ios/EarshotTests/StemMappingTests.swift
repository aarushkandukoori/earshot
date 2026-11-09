// StemMappingTests.swift - the stem press table of 30-ios 11 (contract 5.3, D50, DECISIONS M.6),
// the Now Playing title, the cue meanings, and the Selector wired to a stub engine: stem presses
// and their beeps, cycling, deleting a person on the table (fade to 0, 150 ms, clear), direction
// learning (never on a stale direction), the stale re-anchor, the fade that a param push must not
// cut short, a media reset, and the yaw_deg write at Stop. No RemoteIO and no AirPods: the tests
// call what iOS would call, and AudioController.onEngineEvent records what reached the engine.

import AVFoundation
import XCTest
@testable import Earshot

final class StemMappingTests: XCTestCase {
    func testPressForEachCommand() {
        XCTAssertEqual(StemMap.press(for: .togglePlayPause), .single)
        XCTAssertEqual(StemMap.press(for: .play), .single)
        XCTAssertEqual(StemMap.press(for: .pause), .single)
        XCTAssertEqual(StemMap.press(for: .nextTrack), .double)
        XCTAssertEqual(StemMap.press(for: .previousTrack), .triple)
    }

    /// Every command x paused x empty table (5 x 2 x 2 = 20 rows), written out by hand.
    func testTheWholeActionTable() {
        let expected: [(RemoteCommandKind, Bool, Bool, StemAction)] = [
            // command          paused  empty  action
            (.togglePlayPause,  false,  false, .pause),
            (.togglePlayPause,  false,  true,  .pause),
            (.togglePlayPause,  true,   false, .resume),
            (.togglePlayPause,  true,   true,  .resume),
            (.play,             false,  false, .nothing),   // play only resumes
            (.play,             false,  true,  .nothing),
            (.play,             true,   false, .resume),
            (.play,             true,   true,  .resume),
            (.pause,            false,  false, .pause),
            (.pause,            false,  true,  .pause),
            (.pause,            true,   false, .nothing),   // pause only pauses
            (.pause,            true,   true,  .nothing),
            (.nextTrack,        false,  false, .lookSelect),
            (.nextTrack,        false,  true,  .handsFreeCapture), // nobody at the table
            (.nextTrack,        true,   false, .lookSelect),
            (.nextTrack,        true,   true,  .handsFreeCapture),
            (.previousTrack,    false,  false, .toggleMode),
            (.previousTrack,    false,  true,  .toggleMode),
            (.previousTrack,    true,   false, .toggleMode),
            (.previousTrack,    true,   true,  .toggleMode),
        ]
        XCTAssertEqual(expected.count, RemoteCommandKind.allCases.count * 4)
        for (kind, paused, empty, action) in expected {
            XCTAssertEqual(StemMap.action(for: kind, isPaused: paused, tableEmpty: empty), action,
                           "\(kind) paused \(paused) empty \(empty)")
        }
    }

    func testNowPlayingTitles() {
        XCTAssertEqual(NowPlayingText.title(mode: .focus, focusName: "Maya", tablePeople: 2, paused: false),
                       "Earshot - Focus: Maya")
        XCTAssertEqual(NowPlayingText.title(mode: .table, focusName: "Maya", tablePeople: 2, paused: false),
                       "Earshot - Table: 2 people")
        XCTAssertEqual(NowPlayingText.title(mode: .everyone, focusName: nil, tablePeople: 0, paused: false),
                       "Earshot - Everyone")
        XCTAssertEqual(NowPlayingText.title(mode: .focus, focusName: "Maya", tablePeople: 1, paused: true),
                       "Earshot - Paused")
        XCTAssertEqual(NowPlayingText.status(mode: .table, focusName: nil, tablePeople: 1, paused: false),
                       "Table: 1 person")
    }

    func testCueMeanings() {
        XCTAssertEqual([0, 1, 2].map(Cue.slot), [1, 2, 3], "1 to 3 beeps name the slot")
        XCTAssertEqual([TableMode.focus, .table, .everyone].map(Cue.mode), [1, 2, 3])
        XCTAssertEqual([Cue.captureStarted, Cue.success, Cue.failure], [1, 2, 3])
    }
}

@MainActor
final class SelectorIntegrationTests: XCTestCase {
    private var root: URL!
    private let clip = TestSupport.speechLike(seconds: 6, from: 0.5, to: 5.5)

    override func setUpWithError() throws { root = TestSupport.tempFolder() }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// A controller "running" on AirPods with the stub engine (no RemoteIO), a people store in a
    /// temp folder, and a Selector attached to both.
    private func rig() throws -> (AudioController, PeopleStore, Earshot.Selector, EarshotModel) {
        let model = try TestSupport.model()
        let audio = AudioController()
        audio.attach(model: model)
        audio.engine = try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 64, instances: 3, slack16: 43)
        audio.route = RouteInfo(outputs: [Port(type: .bluetoothA2DP, uid: "pods", name: "AirPods")],
                                inputType: .builtInMic, sampleRate: 48_000, ioBufferDuration: 64.0 / 48_000,
                                inputLatency: 0.005, outputLatency: 0.04)
        audio.state = .running
        let people = PeopleStore(files: PersonFiles(root: root.appendingPathComponent("People")), prepareLogs: false)
        let selector = Earshot.Selector() // "Earshot." because Objective-C also has a type named Selector
        selector.attach(audio: audio, people: people, model: model, enrollment: nil)
        return (audio, people, selector, model)
    }

    private func addPerson(_ name: String, _ people: PeopleStore, _ model: EarshotModel,
                           yawDeg: Double? = nil) throws -> Person {
        let mask = model.vad(clip)
        let p = Person(name: name, kind: .tonight, created: Date(), expires: Date().addingTimeInterval(3600),
                       yawDeg: yawDeg, modelHash: model.modelHash, eWespk: nil, mask: mask, quality: Quality(speechS: 5))
        let vp = try Voiceprint(model: model, pcm16k: clip, mask: mask, role: .target, weSpeaker: nil)
        try people.add(p, clip: clip, voiceprint: vp)
        return p
    }

    private func failures(_ audio: AudioController) -> [String] { audio.notes.filter { $0.contains("failed") } }

    func testDeletingAPersonOnTheTableFadesWaitsThenClears() async throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model)
        selector.addToTable(maya, freshYaw: 10)
        XCTAssertEqual(selector.table.slot(of: maya.id), 1)
        XCTAssertEqual(audio.params.slotGain, [0, 1, 0], "Focus on Maya")
        await audio.condChainIdle()

        let log = EngineLog()
        audio.onEngineEvent = { log.events.append($0) }
        let t0 = Date()
        people.delete(maya) // PeopleStore.onPersonDeleted -> Selector.personDeleted
        XCTAssertNil(selector.table.slot(of: maya.id))
        XCTAssertEqual(selector.table.mode, .everyone, "nobody left: Everyone (still warm in slot 0)")
        XCTAssertEqual(audio.params.slotGain, [1, 0, 0])
        await audio.condChainIdle()
        XCTAssertGreaterThanOrEqual(Date().timeIntervalSince(t0), 0.14, "the clear waited for the 150 ms fade")
        XCTAssertEqual(failures(audio), [])
        // The order on the engine: slot 1 to gain 0 first, and never turned up again before the clear.
        let clearAt = try XCTUnwrap(log.events.firstIndex(of: .clear(slot: 1)), "no clear: \(log.events)")
        let before = gains(log.events[..<clearAt])
        XCTAssertFalse(before.isEmpty, "a gain push came before the clear")
        XCTAssertTrue(before.allSatisfy { $0[1] == 0 }, "slot 1 stayed at 0 until the clear: \(before)")
        XCTAssertTrue(audio.fadingSlots.isEmpty)
        XCTAssertNil(selector.sessionDirections[maya.id])
        audio.engine = nil
    }

    func testLookSelectCyclesWithoutHeadTracking() async throws {
        let (audio, people, selector, model) = try rig()
        let a = try addPerson("A", people, model)
        let b = try addPerson("B", people, model)
        selector.addToTable(a, freshYaw: nil) // slot 1
        selector.addToTable(b, freshYaw: nil) // slot 2, focused
        XCTAssertEqual(selector.table.selected, 2)
        selector.lookSelect() // no AirPods motion in the simulator: falls back to cycling
        XCTAssertEqual(selector.table.selected, 1, "next filled slot after 2 wraps to 1")
        selector.lookSelect()
        XCTAssertEqual(selector.table.selected, 2)
        XCTAssertEqual(selector.engineSlots().map { $0.slot }, [0, 1, 2], "Everyone warm in 0, A and B")
        await audio.condChainIdle()
        XCTAssertEqual(failures(audio), [])
        audio.engine = nil
    }

    func testStemPressesDriveTheController() async throws {
        let (audio, people, selector, model) = try rig()
        let log = EngineLog()
        audio.onEngineEvent = { log.events.append($0) }
        audio.allowListenWithoutOwnVoice = true
        selector.remote.received(.togglePlayPause)
        XCTAssertEqual(audio.state, .paused, "single press pauses")
        selector.remote.received(.pause)
        XCTAssertEqual(audio.state, .paused, "pause only pauses")
        selector.remote.received(.play)
        XCTAssertEqual(audio.state, .running, "play resumes")
        XCTAssertEqual(selector.remote.counts[.single], 3)

        // Double press with nobody at the table: hands-free capture; without enrollment it fails at once.
        selector.remote.received(.nextTrack)
        XCTAssertFalse(selector.handsFreeActive)
        XCTAssertEqual(log.cues, [Cue.failure], "3 beeps: the capture could not start")
        XCTAssertEqual(selector.remote.counts[.double], 1)

        let maya = try addPerson("Maya", people, model)
        selector.addToTable(maya, freshYaw: nil)
        XCTAssertEqual(selector.table.mode, .focus)
        selector.remote.received(.previousTrack) // triple: Focus <-> Everyone with 1 person
        XCTAssertEqual(selector.table.mode, .everyone)
        selector.remote.received(.previousTrack)
        XCTAssertEqual(selector.table.mode, .focus)
        XCTAssertEqual(selector.remote.counts[.triple], 2)
        XCTAssertEqual(log.cues, [Cue.failure, 3, 1], "then 3 beeps = Everyone, 1 beep = Focus")
        selector.remote.received(.nextTrack) // double press, no head tracking: cycle to Maya again
        XCTAssertEqual(log.cues.last, Cue.slot(1), "2 beeps name slot 1")
        await audio.condChainIdle()
        audio.engine = nil
    }

    func testDirectionLearningFromStatsAndHeadSamples() throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model)
        selector.addToTable(maya, freshYaw: 0) // slot 1, direction 0
        var stats = EngineStats()
        stats.pTarget = [0, 0.9, 0] // Maya talks
        for i in 0...12 {
            let now = 100 + Double(i) / 10
            selector.head.receive(t: now, yaw: 20) // the head points 20 degrees away, not moving
            selector.tick(stats, now: now)
        }
        XCTAssertEqual(selector.talking, [false, true, false])
        // One step after 1 s of talking: 0 + 0.1 * 20 = 2; the next step would need another second.
        XCTAssertEqual(selector.table.seats[1]?.dir ?? .nan, 2, accuracy: 1e-9)
        XCTAssertEqual(selector.sessionDirections[maya.id] ?? .nan, 2, accuracy: 1e-9)
        audio.engine = nil
    }

    func testStopWritesLearnedDirectionsToYawDeg() async throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model)
        selector.addToTable(maya, freshYaw: 33)
        selector.sessionStopped()
        XCTAssertEqual(people.person(maya.id)?.yawDeg, 33, "learned values go to yaw_deg at Stop")
        XCTAssertTrue(selector.table.directionsStale, "the next session may have another yaw frame")
        XCTAssertTrue(selector.sessionDirections.isEmpty)
        await people.flush()
        audio.engine = nil
    }

    func testOwnVoiceChangeRebuildsEveryFilledSlot() async throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model)
        selector.addToTable(maya, freshYaw: nil)
        let own = try Voiceprint(model: model, pcm16k: clip, mask: nil, role: .own, weSpeaker: nil)
        audio.setOwnVoice(own) // rebuilds Everyone (slot 0) and Maya (slot 1), warm
        await audio.condChainIdle()
        XCTAssertEqual(failures(audio), [])
        XCTAssertEqual(try audio.filledSlots().map { $0.slot }, [0, 1])
        audio.engine = nil
    }

    // MARK: - Review fixes: stale directions, the fade, media reset

    /// The dinner flow: Maya (yaw_deg 0) and Bob (yaw_deg 50) load stale; in tonight's frame they
    /// really sit at -60 and -10. Maya talks for 5 s while the user faces her. Learning must wait
    /// (one step would pull only Maya part of the way and spoil the offset), then the double press
    /// re-anchors both by one offset, and facing Bob focuses Bob.
    func testLearningWaitsForTheReanchorWhileDirectionsAreStale() throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model, yawDeg: 0)
        let bob = try addPerson("Bob", people, model, yawDeg: 50)
        selector.addToTable(maya, freshYaw: nil) // slot 1, stale
        selector.addToTable(bob, freshYaw: nil)  // slot 2, stale, focused
        XCTAssertTrue(selector.table.directionsStale)
        var stats = EngineStats()
        stats.pTarget = [0, 0.9, 0]
        var now = 200.0
        for _ in 0..<50 {
            now += 0.1
            selector.head.receive(t: now, yaw: -60)
            selector.tick(stats, now: now)
        }
        XCTAssertEqual(selector.table.seats[1]?.dir ?? .nan, 0, accuracy: 1e-9, "no learning on a stale direction")
        XCTAssertNil(selector.sessionDirections[maya.id], "a stale direction is not remembered as fresh")

        selector.lookSelect(now: now) // Maya is the one talker
        XCTAssertEqual(selector.table.selected, 1)
        XCTAssertFalse(selector.table.directionsStale)
        XCTAssertEqual(selector.table.seats[1]?.dir ?? .nan, -60, accuracy: 1e-9)
        XCTAssertEqual(selector.table.seats[2]?.dir ?? .nan, -10, accuracy: 1e-9, "Bob moved by the same offset")

        stats.pTarget = [0, 0, 0] // Maya stops; the user turns to Bob and presses again
        for _ in 0..<3 {
            now += 0.1
            selector.head.receive(t: now, yaw: -10)
            selector.tick(stats, now: now)
        }
        selector.lookSelect(now: now)
        XCTAssertEqual(selector.table.selected, 2, "facing Bob focuses Bob")
        audio.engine = nil
    }

    /// A saved person (stale) next to a person added just now (fresh): the fresh one talking does
    /// not make the stale one count as fresh, so its old-frame value never reaches yaw_deg.
    func testAFreshTalkerLeavesTheStalePersonStale() async throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model, yawDeg: 0)
        let dan = try addPerson("Dan", people, model)
        selector.addToTable(maya, freshYaw: nil) // slot 1, stale
        selector.addToTable(dan, freshYaw: 30)   // slot 2, fresh
        var stats = EngineStats()
        stats.pTarget = [0, 0, 0.9]              // Dan talks while the user faces him
        var now = 400.0
        for _ in 0..<12 {
            now += 0.1
            selector.head.receive(t: now, yaw: 30)
            selector.tick(stats, now: now)
        }
        selector.lookSelect(now: now)
        XCTAssertEqual(selector.table.selected, 2)
        XCTAssertTrue(selector.table.directionsStale, "Maya's offset is still unknown")
        XCTAssertEqual(selector.table.seats[1]?.dirStale, true)
        selector.sessionStopped()
        XCTAssertEqual(people.person(maya.id)?.yawDeg, 0, "Maya keeps her old yaw_deg")
        XCTAssertEqual(people.person(dan.id)?.yawDeg ?? .nan, 30, accuracy: 1e-9)
        await audio.condChainIdle()
        await people.flush()
        audio.engine = nil
    }

    /// A direction remembered this session is in the old frame after a yaw jump: re-seating that
    /// person loads it stale.
    func testARememberedDirectionTurnsStaleOnAFrameReset() async throws {
        let (audio, people, selector, model) = try rig()
        let maya = try addPerson("Maya", people, model)
        selector.addToTable(maya, freshYaw: 20)
        selector.removeFromTable(maya.id)
        XCTAssertEqual(selector.sessionDirections[maya.id], 20)
        selector.head.receive(t: 300.00, yaw: 0)
        selector.head.receive(t: 300.02, yaw: 90) // a yaw jump: the frame changed
        selector.addToTable(maya, freshYaw: nil)
        let s = try XCTUnwrap(selector.table.slot(of: maya.id))
        XCTAssertEqual(selector.table.seats[s]?.dir, 20)
        XCTAssertEqual(selector.table.seats[s]?.dirStale, true, "from before the jump: another frame")
        await audio.condChainIdle()
        audio.engine = nil
    }

    /// Three people seated, Focus on slot 2, then Everyone: it goes into slot 2 (cold). A voice
    /// slider push inside the 150 ms wait must not turn slot 2 up while it still plays the old
    /// person, and the cold load must happen at gain 0.
    func testAParamPushDuringTheFadeKeepsTheChangingSlotSilent() async throws {
        let (audio, people, selector, model) = try rig()
        for name in ["A", "B", "C"] { selector.addToTable(try addPerson(name, people, model), freshYaw: nil) }
        selector.focus(on: 2)
        await audio.condChainIdle()
        let log = EngineLog()
        audio.onEngineEvent = { log.events.append($0) }
        selector.setMode(.everyone) // all seats full: Everyone goes into the selected slot 2
        XCTAssertEqual(audio.params.slotGain, [0, 0, 1], "the final gains are stored at once")
        audio.pushParams()          // a settings change inside the wait
        await audio.condChainIdle()
        let loadAt = try XCTUnwrap(log.events.firstIndex(of: .load(slot: 2, reset: true)), "\(log.events)")
        let before = gains(log.events[..<loadAt])
        XCTAssertGreaterThanOrEqual(before.count, 2, "the settings push and the quiet step")
        XCTAssertTrue(before.allSatisfy { $0[2] == 0 }, "slot 2 silent until the load: \(before)")
        XCTAssertEqual(gains(log.events[loadAt...]).last, [0, 0, 1], "then Everyone fades in")
        XCTAssertTrue(audio.fadingSlots.isEmpty)
        XCTAssertEqual(failures(audio), [])
        audio.engine = nil
    }

    /// Resume inside the wait is the same case: the slot being loaded stays at 0 until its load.
    func testResumeDuringTheFadeKeepsTheChangingSlotSilent() async throws {
        let (audio, people, selector, model) = try rig()
        audio.allowListenWithoutOwnVoice = true
        selector.addToTable(try addPerson("A", people, model), freshYaw: nil) // slot 1
        await audio.condChainIdle()
        audio.pause()
        let log = EngineLog()
        audio.onEngineEvent = { log.events.append($0) }
        selector.addToTable(try addPerson("B", people, model), freshYaw: nil) // slot 2, cold, focused
        audio.resume()
        await audio.condChainIdle()
        let loadAt = try XCTUnwrap(log.events.firstIndex(of: .load(slot: 2, reset: true)), "\(log.events)")
        let before = gains(log.events[..<loadAt])
        XCTAssertFalse(before.isEmpty, "resume sent the params")
        XCTAssertTrue(before.allSatisfy { $0[2] == 0 }, "slot 2 silent until the load: \(before)")
        XCTAssertEqual(gains(log.events[loadAt...]).last, [0, 0, 1])
        XCTAssertEqual(audio.state, .running)
        audio.engine = nil
    }

    /// A media services reset in the background leaves no engine: the Now Playing card and head
    /// tracking go off, so stem presses do not reach a player that plays nothing.
    func testAMediaResetTurnsNowPlayingOff() throws {
        let (audio, _, selector, _) = try rig()
        selector.listeningStarted()
        XCTAssertTrue(selector.remote.active)
        audio.isForeground = false // background: no restart, "Tap to resume" instead
        audio.handleMediaServicesReset()
        XCTAssertEqual(audio.state, .idle)
        XCTAssertFalse(selector.remote.active, "no stale Now Playing card")
        XCTAssertFalse(selector.head.running)
    }

    /// The slot gains among `events`, in order.
    private func gains(_ events: ArraySlice<EngineEvent>) -> [[Float]] {
        events.compactMap { if case .gains(let g) = $0 { return g } else { return nil } }
    }
}

/// Collects AudioController.onEngineEvent in order (main actor, like the controller).
@MainActor
private final class EngineLog {
    var events: [EngineEvent] = []
    var cues: [Int] { events.compactMap { if case .cue(let n) = $0 { return n } else { return nil } } }
}
