// LockedCheckTests.swift - the locked check rule (30-ios 12.3, contract 5.3): lost for 5 s while
// talking shows the banner, a locked result resets, silent seconds pause the count, a new person
// starts fresh; the 3 s p_target mean; and the LockedCheck actor with fake output and fake
// embeddings (no engine, no model).

import XCTest
@testable import Earshot

final class LockedCheckTests: XCTestCase {
    private let maya = UUID()
    private let bob = UUID()

    func testFiveLostChecksWhileTalkingWarn() {
        var r = LockedRule()
        for i in 1...4 {
            r.record(slot: 1, person: maya, cosine: 0.1, lockedCos: 0.3)
            XCTAssertFalse(r.warning, "after \(i) s")
        }
        XCTAssertEqual(r.states[1], .lost)
        XCTAssertEqual(r.lostChecks[1], 4)
        r.record(slot: 1, person: maya, cosine: 0.1, lockedCos: 0.3)
        XCTAssertTrue(r.warning, "lost for 5 s")
    }

    func testALockedCheckResetsTheCount() {
        var r = LockedRule()
        for _ in 0..<4 { r.record(slot: 2, person: maya, cosine: -0.5, lockedCos: 0.3) }
        r.record(slot: 2, person: maya, cosine: 0.9, lockedCos: 0.3)
        XCTAssertEqual(r.states[2], .locked)
        XCTAssertEqual(r.lostChecks[2], 0)
        for _ in 0..<4 { r.record(slot: 2, person: maya, cosine: -0.5, lockedCos: 0.3) }
        XCTAssertFalse(r.warning, "the count started again after the lock")
    }

    func testSecondsWithoutACheckPauseTheCount() {
        var r = LockedRule()
        for _ in 0..<3 { r.record(slot: 1, person: maya, cosine: 0, lockedCos: 0.3) }
        for _ in 0..<10 { r.record(slot: 1, person: maya, cosine: nil, lockedCos: 0.3) } // Maya silent
        XCTAssertEqual(r.lostChecks[1], 3, "silence neither adds nor resets")
        XCTAssertEqual(r.states[1], .lost)
        r.record(slot: 1, person: maya, cosine: 0, lockedCos: 0.3)
        r.record(slot: 1, person: maya, cosine: 0, lockedCos: 0.3)
        XCTAssertTrue(r.warning, "3 + 2 lost seconds while talking")
    }

    func testANewPersonInTheSlotStartsFresh() {
        var r = LockedRule()
        for _ in 0..<5 { r.record(slot: 0, person: maya, cosine: 0, lockedCos: 0.3) }
        XCTAssertTrue(r.warning)
        r.claim(slot: 0, person: bob)
        XCTAssertFalse(r.warning)
        XCTAssertEqual(r.states[0], .unknown)
        XCTAssertEqual(r.owners[0], bob)
    }

    func testForgetAndSlotsAreIndependent() {
        var r = LockedRule()
        for _ in 0..<5 { r.record(slot: 1, person: maya, cosine: 0, lockedCos: 0.3) }
        r.record(slot: 2, person: bob, cosine: 0.9, lockedCos: 0.3)
        XCTAssertEqual(r.states, [.unknown, .lost, .locked])
        r.forget(slot: 1)
        XCTAssertEqual(r.states, [.unknown, .unknown, .locked])
        XCTAssertFalse(r.warning)
    }

    func testBoundaryAndNaN() {
        var r = LockedRule()
        r.record(slot: 0, person: maya, cosine: 0.3, lockedCos: 0.3)
        XCTAssertEqual(r.states[0], .locked, "at locked_cos counts as locked (below counts as lost)")
        r.record(slot: 0, person: maya, cosine: .nan, lockedCos: 0.3)
        XCTAssertEqual(r.states[0], .lost, "a NaN cosine is never a match")
        r.record(slot: 5, person: maya, cosine: 0.9, lockedCos: 0.3) // no such slot: ignored
        XCTAssertEqual(r.states.count, 3)
    }

    func testEligibility() {
        XCTAssertTrue(LockedRule.eligible(gain: 1, pTargetMean: 0.51))
        XCTAssertFalse(LockedRule.eligible(gain: 1, pTargetMean: 0.5), "above 0.5, not at it")
        XCTAssertFalse(LockedRule.eligible(gain: 0, pTargetMean: 0.9), "a silent slot is not checked")
    }

    func testCosine() {
        XCTAssertEqual(LockedRule.cosine([1, 0], [1, 0]), 1, accuracy: 1e-6)
        XCTAssertEqual(LockedRule.cosine([1, 0], [0, 1]), 0, accuracy: 1e-6)
        XCTAssertEqual(LockedRule.cosine([1, 0], [-2, 0]), -1, accuracy: 1e-6)
        XCTAssertEqual(LockedRule.cosine([3, 4], [6, 8]), 1, accuracy: 1e-6, "length does not matter")
        XCTAssertEqual(LockedRule.cosine([0, 0], [1, 0]), 0)
        XCTAssertEqual(LockedRule.cosine([1, 0, 0], [1, 0]), 0, "different sizes")
    }

    func testPTargetHistoryKeepsThreeSeconds() {
        var h = PTargetHistory()
        XCTAssertEqual(h.mean(slot: 1), 0)
        for _ in 0..<10 { h.add([0, 0, 0]) }
        for _ in 0..<30 { h.add([0, 0.8, 0.2]) }
        XCTAssertEqual(h.readings.count, 30)
        XCTAssertEqual(h.mean(slot: 1), 0.8, accuracy: 1e-6, "the first 10 readings are older than 3 s")
        XCTAssertEqual(h.mean(slot: 2), 0.2, accuracy: 1e-6)
        h.clear()
        XCTAssertEqual(h.mean(slot: 1), 0)
    }

    /// A new person in a slot starts the 3 s over: the previous person's readings no longer count
    /// and the slot is not settled until 30 new polls (I-4 review).
    func testNewPersonInASlotWaitsThreeSeconds() {
        var h = PTargetHistory()
        let a = UUID(), b = UUID()
        h.setOwner(slot: 1, a)
        for _ in 0..<30 { h.add([0, 0.9, 0]) }
        XCTAssertTrue(h.isSettled(slot: 1))
        XCTAssertEqual(h.mean(slot: 1), 0.9, accuracy: 1e-6)
        h.setOwner(slot: 1, a) // same person: nothing changes
        XCTAssertTrue(h.isSettled(slot: 1))
        h.setOwner(slot: 1, b)
        XCTAssertFalse(h.isSettled(slot: 1), "a new person is not checked at once")
        XCTAssertEqual(h.mean(slot: 1), 0, "the previous person's readings do not count")
        for _ in 0..<10 { h.add([0, 0.2, 0]) }
        XCTAssertEqual(h.mean(slot: 1), 0.2, accuracy: 1e-6)
        XCTAssertFalse(h.isSettled(slot: 1))
        for _ in 0..<20 { h.add([0, 0.2, 0]) }
        XCTAssertTrue(h.isSettled(slot: 1), "settled after 3 s")
        XCTAssertTrue(h.isSettled(slot: 0), "other slots keep counting")
    }

    /// The actor with fake output and fake embeddings: slot 1 sounds like its person (cosine 1),
    /// slot 2 like someone else (cosine 0), slot 0 has played under 3 s (skipped), and a failed
    /// embedding is skipped too.
    func testActorWithFakeEmbeddings() async {
        let personG: [Float] = [1, 0, 0]
        let read: @Sendable (Int, Int) -> [Float] = { slot, n in
            slot == 0 ? [Float](repeating: 0, count: 100) : [Float](repeating: Float(slot), count: n)
        }
        let embed: @Sendable ([Float]) throws -> [Float] = { pcm in
            switch pcm[0] {
            case 1: return [1, 0, 0]          // "Maya" as the person
            case 2: return [0, 1, 0]          // someone else
            default: throw EarshotError(status: -6, message: "no speech")
            }
        }
        let check = LockedCheck()
        let jobs = [LockJob(slot: 0, person: maya, g: personG), LockJob(slot: 1, person: maya, g: personG),
                    LockJob(slot: 2, person: bob, g: personG)]
        let out = await check.cosines(jobs: jobs, read: read, embed: embed)
        XCTAssertNil(out[0], "under 3 s of output")
        XCTAssertEqual(out[1] ?? -9, 1, accuracy: 1e-6)
        XCTAssertEqual(out[2] ?? -9, 0, accuracy: 1e-6)
        XCTAssertEqual(LockedCheck.samples, 48_000, "3 s at 16 kHz")

        let failing = await check.cosines(jobs: [LockJob(slot: 3, person: maya, g: personG)], read: read, embed: embed)
        XCTAssertTrue(failing.isEmpty, "an embedding error skips the slot")
    }
}
