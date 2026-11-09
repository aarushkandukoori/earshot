// TableStateTests.swift - the table's slot rules (30-ios 10.2, contract 5.3 "Start" and "Warm
// instances"), the fade-then-load order (contract 4.3, TableApply) and which conditioning each slot
// gets (30-ios 10.2 ModeConditioning). Pure: no engine except the stub for the build check.

import XCTest
@testable import Earshot

final class TableStateTests: XCTestCase {
    private let a = UUID(), b = UUID(), c = UUID(), d = UUID()

    private func load(_ slot: Int, _ content: SlotContent, reset: Bool = true) -> SlotLoad {
        SlotLoad(slot: slot, content: content, reset: reset)
    }

    // MARK: - Start and adding people

    func testStartsInEveryoneModeInSlot0() {
        let t = TableState()
        XCTAssertEqual(t.mode, .everyone)
        XCTAssertEqual(t.contents, [.everyone, .empty, .empty])
        XCTAssertEqual(t.gains, [1, 0, 0])
        XCTAssertEqual(t.personCount, 0)
        XCTAssertNil(t.current)
    }

    func testFirstPersonTakesAFreeSlotColdAndIsFocused() {
        var t = TableState()
        let loads = t.add(a, dir: 12, dirStale: false)
        XCTAssertEqual(loads, [load(1, .person(a))], "slot 0 still holds Everyone (warm), slot 1 is free")
        XCTAssertEqual(t.mode, .focus)
        XCTAssertEqual(t.selected, 1)
        XCTAssertEqual(t.gains, [0, 1, 0], "Focus: the selected slot 1, the rest 0 but still running")
        XCTAssertEqual(t.content(0), .everyone)
        XCTAssertFalse(t.directionsStale, "a direction from this session's capture is fresh")
    }

    func testThirdPersonReplacesTheEveryoneSlotAndAFourthTheLeastRecentlySelected() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false) // slot 1
        _ = t.add(b, dir: nil, dirStale: false) // slot 2
        XCTAssertEqual(t.add(c, dir: nil, dirStale: false), [load(0, .person(c))], "Everyone gives up slot 0")
        XCTAssertNil(t.everyoneSlot)
        XCTAssertEqual(t.personCount, 3)
        // a (slot 1) was selected longest ago: the 4th person replaces a, cold.
        XCTAssertEqual(t.add(d, dir: nil, dirStale: false), [load(1, .person(d))])
        XCTAssertNil(t.slot(of: a))
        XCTAssertEqual(t.slot(of: d), 1)
        XCTAssertEqual(t.personCount, 3, "the table holds at most 3 people")
    }

    func testFocusingSomeoneProtectsThemFromReplacement() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false) // slot 1
        _ = t.add(b, dir: nil, dirStale: false) // slot 2
        _ = t.add(c, dir: nil, dirStale: false) // slot 0
        XCTAssertEqual(t.focus(on: 1), [], "a is warm: no load, only gains change")
        _ = t.add(d, dir: nil, dirStale: false)
        XCTAssertNil(t.slot(of: b), "b (slot 2) is now the least recently selected")
        XCTAssertNotNil(t.slot(of: a))
    }

    func testReAddingASeatedPersonIsWarm() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false)
        _ = t.add(b, dir: nil, dirStale: false)
        XCTAssertEqual(t.add(a, dir: 30, dirStale: false), [], "already at the table: focus only")
        XCTAssertEqual(t.selected, t.slot(of: a))
        XCTAssertEqual(t.seats[t.slot(of: a)!]!.dir, 30, "a fresh capture direction replaces the old one")
    }

    func testAStoredDirectionLoadsStale() {
        var t = TableState()
        _ = t.add(a, dir: 40, dirStale: true)
        XCTAssertTrue(t.directionsStale, "yaw_deg from an earlier session is always stale (contract 5.1)")
        var u = TableState()
        _ = u.add(a, dir: nil, dirStale: true)
        XCTAssertFalse(u.directionsStale, "no direction, nothing stale")
        u.markDirectionsStale()
        XCTAssertFalse(u.directionsStale, "no directions to mark")
    }

    // MARK: - Fresh and stale directions are kept per seat (I-3 review fix)

    /// Maya loads stale at 0 (her true direction in the new frame is -60), Bob is added fresh at 30.
    /// Bob talking re-anchors only Bob: Maya's offset is unknown, so she stays stale. When Maya
    /// talks later, her offset moves her (and any other stale seat), never the fresh Bob.
    func testAFreshTalkerNeverClearsTheOtherSeatsStaleness() {
        var t = TableState()
        _ = t.add(a, dir: 0, dirStale: true)   // slot 1, from person.json
        _ = t.add(b, dir: 30, dirStale: false) // slot 2, a capture just now
        XCTAssertEqual(t.seats[1]?.dirStale, true)
        XCTAssertEqual(t.seats[2]?.dirStale, false)
        XCTAssertTrue(t.directionsStale)

        t.reanchor(slot: 2, yaw: 32) // Bob is the one talker
        XCTAssertEqual(t.seats[2]?.dir ?? .nan, 32, accuracy: 1e-9)
        XCTAssertEqual(t.seats[1]?.dir ?? .nan, 0, accuracy: 1e-9, "Maya is not moved by a fresh talker")
        XCTAssertTrue(t.directionsStale, "Maya is still in the old frame")

        t.reanchor(slot: 1, yaw: -60) // now Maya talks while the user faces her
        XCTAssertEqual(t.seats[1]?.dir ?? .nan, -60, accuracy: 1e-9)
        XCTAssertEqual(t.seats[2]?.dir ?? .nan, 32, accuracy: 1e-9, "a fresh direction keeps its value")
        XCTAssertFalse(t.directionsStale)
    }

    /// A stale talker moves every stale seat by one offset; a seat with no direction stays unknown.
    func testAStaleTalkerMovesOnlyTheStaleSeats() {
        var t = TableState()
        _ = t.add(a, dir: 0, dirStale: true)    // slot 1
        _ = t.add(b, dir: 50, dirStale: true)   // slot 2
        _ = t.add(c, dir: nil, dirStale: false) // slot 0, no direction yet
        t.reanchor(slot: 1, yaw: -60)
        XCTAssertEqual(t.seats[1]?.dir ?? .nan, -60, accuracy: 1e-9)
        XCTAssertEqual(t.seats[2]?.dir ?? .nan, -10, accuracy: 1e-9, "50 + (-60)")
        XCTAssertNil(t.seats[0]?.dir)
        XCTAssertFalse(t.directionsStale)
    }

    func testLearningAndFreshCapturesMakeASeatFresh() {
        var t = TableState()
        _ = t.add(a, dir: 10, dirStale: false)
        t.markDirectionsStale()
        XCTAssertEqual(t.seats[1]?.dirStale, true, "a yaw jump marks every known direction")
        t.setDirection(slot: 1, 15)
        XCTAssertEqual(t.seats[1]?.dirStale, false, "a learned direction comes from the current yaw")
        t.markDirectionsStale()
        _ = t.add(a, dir: 40, dirStale: false) // already seated, a new capture direction
        XCTAssertEqual(t.seats[1]?.dir, 40)
        XCTAssertEqual(t.seats[1]?.dirStale, false)
        _ = t.add(a, dir: 99, dirStale: true)  // a stale yaw_deg never replaces a seated direction
        XCTAssertEqual(t.seats[1]?.dir, 40)
    }

    // MARK: - Modes

    func testTableModeNeedsTwoAndSumsEveryPerson() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false)
        XCTAssertNil(t.setMode(.table), "Table needs at least 2 people")
        XCTAssertEqual(t.mode, .focus)
        _ = t.add(b, dir: nil, dirStale: false)
        XCTAssertEqual(t.setMode(.table), [])
        XCTAssertEqual(t.gains, [0, 1, 1], "every person slot 1; Everyone (slot 0) silent")
        // A new person joins the sum and Table stays Table.
        XCTAssertEqual(t.add(c, dir: nil, dirStale: false), [load(0, .person(c))])
        XCTAssertEqual(t.mode, .table)
        XCTAssertEqual(t.gains, [1, 1, 1])
    }

    func testEveryoneIsWarmWhileItKeepsAFreeSlot() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false)
        XCTAssertEqual(t.setMode(.everyone), [], "Everyone still runs in slot 0: warm, no reset")
        XCTAssertEqual(t.gains, [1, 0, 0], "only the Everyone slot plays")
        XCTAssertEqual(t.setMode(.focus), [])
        XCTAssertEqual(t.gains, [0, 1, 0])
    }

    func testEveryoneTakesTheSelectedPersonsSlotWhenAllAreFull() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false) // 1
        _ = t.add(b, dir: nil, dirStale: false) // 2
        _ = t.add(c, dir: nil, dirStale: false) // 0, focused
        XCTAssertEqual(t.setMode(.everyone), [load(0, .everyone)], "into the selected person's slot, reset")
        XCTAssertEqual(t.gains, [1, 0, 0])
        XCTAssertEqual(t.personCount, 3, "c stays at the table")
        XCTAssertEqual(t.runningPersonSlots, [1, 2], "but c's instance now runs Everyone")
        // Back to Focus on c: c returns cold.
        XCTAssertEqual(t.setMode(.focus), [load(0, .person(c))])
        XCTAssertNil(t.everyoneSlot)
        // Everyone again, then Table: the person Everyone sent out comes back cold.
        _ = t.setMode(.everyone)
        XCTAssertEqual(t.setMode(.table), [load(0, .person(c))])
        XCTAssertEqual(t.gains, [1, 1, 1])
    }

    func testFocusOnAnEmptySlotDoesNothing() {
        var t = TableState()
        XCTAssertEqual(t.focus(on: 1), [])
        XCTAssertEqual(t.focus(on: 7), [])
        XCTAssertEqual(t.mode, .everyone)
        XCTAssertNil(t.setMode(.focus), "Focus needs a person")
    }

    func testTriplePressTargets() {
        var t = TableState()
        XCTAssertNil(t.toggleTarget(), "nobody at the table and already Everyone")
        _ = t.add(a, dir: nil, dirStale: false)
        XCTAssertEqual(t.toggleTarget(), .everyone, "Focus with 1 person: Focus <-> Everyone")
        _ = t.setMode(.everyone)
        XCTAssertEqual(t.toggleTarget(), .focus)
        _ = t.add(b, dir: nil, dirStale: false) // focuses b
        XCTAssertEqual(t.toggleTarget(), .table, "Focus with 2 people: Focus <-> Table")
        _ = t.setMode(.table)
        XCTAssertEqual(t.toggleTarget(), .focus)
    }

    // MARK: - Leaving the table (deletion)

    func testDeletingTheFocusedPersonEmptiesTheSlotAndFocusesTheNextMostRecent() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false) // 1
        _ = t.add(b, dir: nil, dirStale: false) // 2, focused
        XCTAssertEqual(t.remove(b), [load(2, .empty, reset: false)])
        XCTAssertEqual(t.mode, .focus)
        XCTAssertEqual(t.selected, 1, "a was selected most recently after b")
        XCTAssertEqual(t.gains, [0, 1, 0])
        XCTAssertEqual(t.remove(UUID()), [], "not at the table: nothing to do")
    }

    func testDeletingTheLastPersonGoesBackToEveryone() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false) // slot 1, Everyone warm in 0
        XCTAssertEqual(t.remove(a), [load(1, .empty, reset: false)], "Everyone was still warm in slot 0")
        XCTAssertEqual(t.mode, .everyone)
        XCTAssertEqual(t.gains, [1, 0, 0])
    }

    func testClearThenEveryoneInTheSameSlotMergesIntoOneLoad() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false) // 1
        _ = t.add(b, dir: nil, dirStale: false) // 2
        _ = t.add(c, dir: nil, dirStale: false) // 0 (Everyone gone)
        _ = t.remove(a)
        _ = t.remove(b)
        // c is the last person, in slot 0; Everyone goes into the first free slot, which is slot 0.
        XCTAssertEqual(t.remove(c), [load(0, .everyone)], "the clear of slot 0 is replaced by the Everyone load")
        XCTAssertEqual(t.contents, [.everyone, .empty, .empty])
    }

    func testTableWithOnePersonLeftBecomesFocus() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false)
        _ = t.add(b, dir: nil, dirStale: false)
        _ = t.setMode(.table)
        _ = t.remove(a)
        XCTAssertEqual(t.mode, .focus)
        XCTAssertEqual(t.focusedPerson, b)
    }

    func testDeletingSomeoneEveryoneSentOutNeedsNoClear() {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false)
        _ = t.add(b, dir: nil, dirStale: false)
        _ = t.add(c, dir: nil, dirStale: false) // slot 0, focused
        _ = t.setMode(.everyone)                // Everyone into slot 0
        XCTAssertEqual(t.remove(c), [], "slot 0 runs Everyone, which stays")
        XCTAssertEqual(t.mode, .everyone)
        XCTAssertEqual(t.gains, [1, 0, 0])
    }

    func testMergedKeepsTheLastLoadPerSlot() {
        let m = TableState.merged([load(1, .empty, reset: false), load(2, .person(a)), load(1, .everyone)])
        XCTAssertEqual(m, [load(2, .person(a)), load(1, .everyone)])
    }

    // MARK: - Fade, then load or clear (contract 4.3, 30-ios 5)

    func testPureGainChangeIsOneStep() {
        XCTAssertEqual(TableApply.steps(newGains: [0, 1, 0], loads: []), [.gains([0, 1, 0])])
    }

    func testDeletionFadesTheSlotWaits150msThenClears() {
        let steps = TableApply.steps(newGains: [0, 1, 0], loads: [load(2, .empty, reset: false)])
        XCTAssertEqual(steps, [.gains([0, 1, 0]), .wait(ms: 150), .clear(slot: 2), .gains([0, 1, 0])])
    }

    func testColdLoadIntoAPlayingSlotIsSilencedFirst() {
        // Everyone sent c out of slot 0; Focus on c reloads slot 0 cold while slot 0 was playing.
        let steps = TableApply.steps(newGains: [1, 0, 0], loads: [load(0, .person(c))], crossfadeMS: 150)
        XCTAssertEqual(steps, [.gains([0, 0, 0]), .wait(ms: 150), .load(slot: 0, reset: true), .gains([1, 0, 0])])
    }
}

// MARK: - Mode conditioning choice (30-ios 10.2)

final class ModeConditioningTests: XCTestCase {
    private let id = UUID()

    func testPersonSlotUsesWeSpeakerOnlyWhenEveryVoiceprintHasIt() {
        func u(_ target: Bool, _ own: Bool?, _ selfTest: Bool) -> Bool? {
            ModeConditioning.choice(for: .person(id), targetHasWeSpeaker: target, ownHasWeSpeaker: own,
                                    selfTestPassed: selfTest)?.useWeSpeaker
        }
        XCTAssertEqual(u(true, true, true), true)
        XCTAssertEqual(u(false, true, true), false, "target lacks its vector")
        XCTAssertEqual(u(true, false, true), false, "own voice lacks its vector")
        XCTAssertEqual(u(true, nil, true), true, "no own voice yet: only the target is involved")
        XCTAssertEqual(u(true, true, false), false, "encoder self-test failed")
        let choice = ModeConditioning.choice(for: .person(id), targetHasWeSpeaker: true, ownHasWeSpeaker: true,
                                             selfTestPassed: true)
        XCTAssertEqual(choice?.mode, .person)
        XCTAssertEqual(choice?.target, id)
    }

    func testEveryoneSlotHasNoTargetAndChecksOnlyTheOwnVoice() {
        func choice(_ own: Bool?, _ selfTest: Bool) -> ModeConditioning.Choice? {
            ModeConditioning.choice(for: .everyone, targetHasWeSpeaker: false, ownHasWeSpeaker: own,
                                    selfTestPassed: selfTest)
        }
        XCTAssertEqual(choice(true, true), .init(mode: .everyone, target: nil, useWeSpeaker: true))
        XCTAssertEqual(choice(false, true)?.useWeSpeaker, false)
        XCTAssertEqual(choice(nil, true)?.useWeSpeaker, true, "setup mode: no voiceprint involved")
        XCTAssertEqual(choice(true, false)?.useWeSpeaker, false)
        XCTAssertNil(ModeConditioning.choice(for: .empty, targetHasWeSpeaker: true, ownHasWeSpeaker: true,
                                             selfTestPassed: true), "an empty slot gets no conditioning")
    }

    func testBuildWithTheStubEngine() throws {
        let model = try TestSupport.model()
        let clip = TestSupport.speechLike(seconds: 6, from: 0.5, to: 5.5)
        let vp = try Voiceprint(model: model, pcm16k: clip, mask: nil, role: .target, weSpeaker: TestSupport.basis(1))
        let own = try Voiceprint(model: model, pcm16k: clip, mask: nil, role: .own, weSpeaker: nil)
        let lookup: (UUID) -> Voiceprint? = { $0 == self.id ? vp : nil }
        XCTAssertNotNil(try ModeConditioning.build(.person(id), model: model, voiceprint: lookup, own: own,
                                                   selfTestPassed: true))
        XCTAssertNotNil(try ModeConditioning.build(.everyone, model: model, voiceprint: lookup, own: own,
                                                   selfTestPassed: true))
        XCTAssertNil(try ModeConditioning.build(.empty, model: model, voiceprint: lookup, own: own, selfTestPassed: true))
        let missing = UUID()
        XCTAssertThrowsError(try ModeConditioning.build(.person(missing), model: model, voiceprint: lookup, own: nil,
                                                        selfTestPassed: true)) { error in
            XCTAssertEqual(error as? ModeConditioning.Failure, .voiceprintNotReady(missing))
        }
    }
}
