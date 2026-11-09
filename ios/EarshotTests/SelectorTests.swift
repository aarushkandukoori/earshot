// SelectorTests.swift - the picking-by-looking math (30-ios 15.1 SelectorTests): wrap and circular
// mean, nearest within 35 degrees, the 10 degree hysteresis, no "current" in Table / Everyone, cycle
// order, the stale re-anchor, the learning step with its 1 s and 20 deg/s gates, wrap-around at the
// back of the circle, the jump detector (45 degrees / 50 ms), and HeadTracker's smoothing and rate.

import Combine
import XCTest
@testable import Earshot

final class SelectorTests: XCTestCase {
    typealias M = SelectorMath

    private func head(live: Bool = true, stale: Bool = false, yaw: Double?) -> HeadSnapshot {
        HeadSnapshot(live: live, stale: stale, smoothedYaw: yaw, rate: 0)
    }

    private func c(_ slot: Int, _ dir: Double?) -> M.Candidate { M.Candidate(slot: slot, dir: dir) }

    // MARK: - Angles

    func testWrapMapsToMinus180Up180() {
        XCTAssertEqual(M.wrap(0), 0)
        XCTAssertEqual(M.wrap(190), -170, accuracy: 1e-12)
        XCTAssertEqual(M.wrap(-190), 170, accuracy: 1e-12)
        XCTAssertEqual(M.wrap(180), -180, "180 itself maps to -180: the range is [-180, 180)")
        XCTAssertEqual(M.wrap(-180), -180)
        XCTAssertEqual(M.wrap(540), -180)
        XCTAssertEqual(M.wrap(359), -1, accuracy: 1e-12)
        XCTAssertEqual(M.wrap(-721), -1, accuracy: 1e-12)
        for a in stride(from: -1000.0, through: 1000.0, by: 7.3) {
            let w = M.wrap(a)
            XCTAssertTrue(w >= -180 && w < 180, "wrap(\(a)) = \(w)")
        }
        let tiny = M.wrap(-1e-15)
        XCTAssertTrue(tiny >= -180 && tiny < 180, "a tiny negative must not round to +180")
    }

    func testCircularMeanAcrossTheBack() {
        XCTAssertEqual(M.circularMean([350, 10])!, 0, accuracy: 1e-9, "a plain average would say 180")
        XCTAssertEqual(M.circularMean([170, -170])!.magnitude, 180, accuracy: 1e-9)
        XCTAssertNil(M.circularMean([90, 270]), "opposite angles cancel: no direction")
        XCTAssertNil(M.circularMean([]))
    }

    // MARK: - Look-select (30-ios 10.3)

    func testEmptyTableStartsHandsFreeCapture() {
        XCTAssertEqual(M.lookSelect(people: [], current: nil, head: head(yaw: 0), onlyTalker: nil), .handsFreeCapture)
    }

    func testNearestDirectionWithin35Degrees() {
        let people = [c(0, 0), c(1, 60), c(2, -90)]
        XCTAssertEqual(M.lookSelect(people: people, current: nil, head: head(yaw: 50), onlyTalker: nil), .focus(1))
        XCTAssertEqual(M.lookSelect(people: people, current: nil, head: head(yaw: -70), onlyTalker: nil), .focus(2))
        XCTAssertEqual(M.lookSelect(people: people, current: nil, head: head(yaw: 35), onlyTalker: nil), .focus(1),
                       "35 degrees is still inside")
        // 30 degrees from 0 and 30 from 60: the nearest is a tie, the first in slot order wins.
        XCTAssertEqual(M.lookSelect(people: [c(0, 0), c(1, 60)], current: nil, head: head(yaw: 30), onlyTalker: nil),
                       .focus(0))
        // Nobody within 35 degrees: cycle.
        XCTAssertEqual(M.lookSelect(people: [c(0, 0)], current: nil, head: head(yaw: 36), onlyTalker: nil), .cycle)
    }

    func testHysteresisCurrent20OffCandidate12CyclesCandidate9Switches() {
        // Focus on slot 0 whose direction is 20 degrees off; the candidate is 12 off: 12 + 10 > 20.
        XCTAssertEqual(M.lookSelect(people: [c(0, 20), c(1, -12)], current: 0, head: head(yaw: 0), onlyTalker: nil),
                       .cycle)
        // Candidate 9 off: 9 + 10 <= 20, it wins.
        XCTAssertEqual(M.lookSelect(people: [c(0, 20), c(1, -9)], current: 0, head: head(yaw: 0), onlyTalker: nil),
                       .focus(1))
        // Exactly 10 better also switches (dBest + 10 <= dCur).
        XCTAssertEqual(M.lookSelect(people: [c(0, 20), c(1, -10)], current: 0, head: head(yaw: 0), onlyTalker: nil),
                       .focus(1))
    }

    func testLookingAtTheCurrentPersonCycles() {
        // The pseudocode requires best.slot != current, so a double press while facing the current
        // person moves on to the next one.
        XCTAssertEqual(M.lookSelect(people: [c(0, 0), c(1, 90)], current: 0, head: head(yaw: 2), onlyTalker: nil),
                       .cycle)
    }

    func testTableAndEveryoneHaveNoCurrent() {
        // Same geometry as the cycling hysteresis case, but nobody is current: the candidate wins.
        XCTAssertEqual(M.lookSelect(people: [c(0, 20), c(1, -12)], current: nil, head: head(yaw: 0), onlyTalker: nil),
                       .focus(1))
        var t = TableState()
        let a = UUID(), b = UUID()
        _ = t.add(a, dir: 20, dirStale: false)
        _ = t.add(b, dir: -12, dirStale: false)
        XCTAssertEqual(t.current, t.slot(of: b), "Focus mode (the newest person is focused): current")
        _ = t.setMode(.table)
        XCTAssertNil(t.current, "Table: nobody is current")
        _ = t.setMode(.everyone)
        XCTAssertNil(t.current, "Everyone: nobody is current")
    }

    func testCurrentWithoutDirectionCountsAsInfinitelyFar() {
        XCTAssertEqual(M.lookSelect(people: [c(0, nil), c(1, 30)], current: 0, head: head(yaw: 0), onlyTalker: nil),
                       .focus(1))
    }

    func testNoHeadTrackingOrNoDirectionsCycles() {
        let people = [c(0, 0), c(1, 60)]
        XCTAssertEqual(M.lookSelect(people: people, current: nil, head: head(live: false, yaw: 60), onlyTalker: nil),
                       .cycle, "head tracking paused: fall back to cycling")
        XCTAssertEqual(M.lookSelect(people: people, current: nil, head: head(yaw: nil), onlyTalker: nil), .cycle)
        XCTAssertEqual(M.lookSelect(people: [c(0, nil), c(1, nil)], current: nil, head: head(yaw: 0), onlyTalker: nil),
                       .cycle)
    }

    func testWrapAroundAtTheBack() {
        // A person at 175 degrees, the head at -175: they are 10 degrees apart, not 350.
        XCTAssertEqual(M.lookSelect(people: [c(0, -60), c(1, 175)], current: nil, head: head(yaw: -175),
                                    onlyTalker: nil), .focus(1))
    }

    // MARK: - Stale re-anchor

    func testStaleDirectionsReanchorOnTheOneTalker() {
        let people = [c(0, 10), c(1, 50)]
        let staleHead = head(stale: true, yaw: 80)
        XCTAssertEqual(M.lookSelect(people: people, current: nil, head: staleHead, onlyTalker: 1),
                       .reanchor(slot: 1, yaw: 80))
        XCTAssertEqual(M.lookSelect(people: people, current: nil, head: staleHead, onlyTalker: nil), .cycle,
                       "nobody (or more than one person) talking: cycle")
        XCTAssertEqual(M.lookSelect(people: people, current: nil, head: head(live: false, stale: true, yaw: 80),
                                    onlyTalker: 1), .cycle, "no live head: cycle")
        XCTAssertEqual(M.lookSelect(people: people, current: nil, head: staleHead, onlyTalker: 2), .cycle,
                       "a talker who is not at the table is ignored")
    }

    func testReanchorShiftsEveryDirectionByOneOffset() {
        let moved = M.reanchor(dirs: [0: 10, 1: 50, 2: -170], talker: 1, yaw: 80)
        XCTAssertEqual(moved[1]!, 80, accuracy: 1e-9)
        XCTAssertEqual(moved[0]!, 40, accuracy: 1e-9)
        XCTAssertEqual(moved[2]!, -140, accuracy: 1e-9)
        // Across the back: offset wrap(-160 - 170) = +30.
        let back = M.reanchor(dirs: [0: 170, 1: -170], talker: 0, yaw: -160)
        XCTAssertEqual(back[0]!, -160, accuracy: 1e-9)
        XCTAssertEqual(back[1]!, -140, accuracy: 1e-9)
        // The talker had no direction: there is no offset, so only the talker is set.
        let unknown = M.reanchor(dirs: [1: 50], talker: 0, yaw: 20)
        XCTAssertEqual(unknown, [0: 20, 1: 50])
    }

    func testTableReanchorClearsStale() {
        var t = TableState()
        let a = UUID(), b = UUID()
        _ = t.add(a, dir: 10, dirStale: true) // from person.json: stale
        _ = t.add(b, dir: 50, dirStale: true)
        XCTAssertTrue(t.directionsStale)
        t.reanchor(slot: t.slot(of: b)!, yaw: 80)
        XCTAssertFalse(t.directionsStale)
        XCTAssertEqual(t.seats[t.slot(of: a)!]!.dir!, 40, accuracy: 1e-9)
        XCTAssertEqual(t.seats[t.slot(of: b)!]!.dir!, 80, accuracy: 1e-9)
    }

    // MARK: - Cycle

    func testCycleOrder() {
        XCTAssertEqual(M.cycleTarget(filled: [0, 1, 2], current: 0), 1)
        XCTAssertEqual(M.cycleTarget(filled: [0, 1, 2], current: 2), 0, "wraps to the first slot")
        XCTAssertEqual(M.cycleTarget(filled: [2, 0], current: nil), 0, "nobody current: first slot in order")
        XCTAssertEqual(M.cycleTarget(filled: [0, 2], current: 0), 2, "skips the empty slot")
        XCTAssertEqual(M.cycleTarget(filled: [1], current: 1), 1)
        XCTAssertNil(M.cycleTarget(filled: [], current: nil))
    }
}

// MARK: - Direction learning (30-ios 10.4)

final class DirectionLearningTests: XCTestCase {
    typealias M = SelectorMath

    func testLearningStepIsOneTenth() {
        XCTAssertEqual(M.learn(dir: nil, yaw: 33), 33, "unknown direction: dir = yaw")
        XCTAssertEqual(M.learn(dir: 0, yaw: 10), 1, accuracy: 1e-12)
        XCTAssertEqual(M.learn(dir: 20, yaw: 0), 18, accuracy: 1e-12)
    }

    func testLearningWrapsAroundTheBack() {
        // dir 170, yaw -170: the short way is +20, so dir moves to 172 (not toward 0).
        XCTAssertEqual(M.learn(dir: 170, yaw: -170), 172, accuracy: 1e-9)
        XCTAssertEqual(M.learn(dir: -178, yaw: 178), -178.4, accuracy: 1e-9)
        // 179 + 0.1 * 10 = 180, which is stored as -180 (always inside [-180, 180)).
        XCTAssertEqual(M.learn(dir: 179, yaw: -171), -180, accuracy: 1e-9)
    }

    /// p_target 0.8 on slot 0 every 100 ms from t = 0: a step at 1.0 s, none at 1.1 to 1.9, the
    /// next at 2.0 (the 1 s timer restarts after each step).
    func testOneSecondGateAndTimerRestart() {
        var t = TalkTracker()
        var due: [Int] = []
        for i in 0...25 {
            let now = Double(i) / 10
            t.update(pTarget: [0.8, 0, 0], personSlots: [0, 1, 2], now: now)
            if t.learnDue(slot: 0, headRate: 5, now: now) { due.append(i) }
        }
        XCTAssertEqual(due, [10, 20], "at most one update per second")
    }

    func testHeadRateGateAndTalkBreak() {
        var t = TalkTracker()
        for i in 0...10 { t.update(pTarget: [0.8, 0, 0], personSlots: [0], now: Double(i) / 10) }
        XCTAssertFalse(t.learnDue(slot: 0, headRate: 20, now: 1.0), "20 deg/s is not slower than 20")
        XCTAssertTrue(t.learnDue(slot: 0, headRate: 19.9, now: 1.0))
        // Talking stops at 1.5 s and starts again at 1.6 s: the next step needs 1 s from 1.6 s.
        t.update(pTarget: [0.8, 0, 0], personSlots: [0], now: 1.4)
        t.update(pTarget: [0.5, 0, 0], personSlots: [0], now: 1.5)
        t.update(pTarget: [0.8, 0, 0], personSlots: [0], now: 1.6)
        t.update(pTarget: [0.8, 0, 0], personSlots: [0], now: 2.5)
        XCTAssertFalse(t.learnDue(slot: 0, headRate: 5, now: 2.5))
        t.update(pTarget: [0.8, 0, 0], personSlots: [0], now: 2.6)
        XCTAssertTrue(t.learnDue(slot: 0, headRate: 5, now: 2.6))
    }

    func testP07IsNotTalkingAndOnlyPersonSlotsCount() {
        var t = TalkTracker()
        for i in 0...12 { t.update(pTarget: [0.7, 0.9, 0.9], personSlots: [0, 1], now: Double(i) / 10) }
        XCTAssertFalse(t.talked(0, for: 1, now: 1.2), "0.7 is not above 0.7")
        XCTAssertTrue(t.talked(1, for: 1, now: 1.2))
        XCTAssertFalse(t.talked(2, for: 1, now: 1.2), "slot 2 is not a person (Everyone or empty)")
        XCTAssertEqual(t.onlyPersonTalking(now: 1.2), 1)
    }

    func testOnlyPersonTalkingNeedsExactlyOne() {
        var t = TalkTracker()
        t.update(pTarget: [0.9, 0.9, 0], personSlots: [0, 1], now: 0)
        t.update(pTarget: [0.9, 0.9, 0], personSlots: [0, 1], now: 1.0)
        XCTAssertNil(t.onlyPersonTalking(now: 1.0), "two talkers: no re-anchor")
        t.update(pTarget: [0.9, 0.2, 0], personSlots: [0, 1], now: 1.1)
        XCTAssertEqual(t.onlyPersonTalking(now: 1.1), 0)
        var short = TalkTracker()
        short.update(pTarget: [0.9, 0, 0], personSlots: [0], now: 0.5)
        XCTAssertNil(short.onlyPersonTalking(now: 1.0), "talking for only 0.5 s")
    }
}

// MARK: - Head tracking math (30-ios 10.1)

final class HeadTrackingTests: XCTestCase {
    func testJumpDetector45DegreesIn50ms() {
        var y = YawTrack()
        XCTAssertFalse(y.add(t: 0, yaw: 0))
        XCTAssertTrue(y.add(t: 0.04, yaw: 50), "50 degrees in 40 ms")
        var a = YawTrack()
        a.add(t: 0, yaw: 0)
        XCTAssertFalse(a.add(t: 0.04, yaw: 44), "44 degrees is not a jump")
        var b = YawTrack()
        b.add(t: 0, yaw: 0)
        XCTAssertFalse(b.add(t: 0.06, yaw: 90), "60 ms apart is not under 50 ms")
        var c = YawTrack()
        c.add(t: 0, yaw: 170)
        XCTAssertFalse(c.add(t: 0.02, yaw: -150), "170 to -150 is 40 degrees the short way")
        var d = YawTrack()
        d.add(t: 0, yaw: 170)
        XCTAssertTrue(d.add(t: 0.02, yaw: -130), "170 to -130 is 60 degrees")
    }

    func testSmoothedYawIsACircularMeanOfTheLast200ms() {
        var y = YawTrack()
        y.add(t: 0.0, yaw: 90)  // older than 200 ms at now = 1.0: ignored
        y.add(t: 0.85, yaw: 350)
        y.add(t: 0.9, yaw: 10)
        y.add(t: 0.95, yaw: 350)
        y.add(t: 1.0, yaw: 10)
        XCTAssertEqual(y.smoothedYaw(now: 1.0)!, 0, accuracy: 1e-9)
        // Fewer than 2 samples in the window: the newest sample.
        XCTAssertEqual(y.smoothedYaw(now: 1.19)!, 10, "only one sample is that recent")
        XCTAssertNil(YawTrack().smoothedYaw(now: 0))
    }

    func testHeadRateLiveAndUnavailable() {
        var y = YawTrack()
        XCTAssertNil(y.headRate(), "no history yet")
        XCTAssertTrue(y.isUnavailable(now: 0))
        y.add(t: 1.0, yaw: 0)
        y.add(t: 1.2, yaw: 4)
        y.add(t: 1.25, yaw: 5)
        // y_now = 5 at 1.25; y_200ms_ago = the newest sample at least 200 ms older = 0 at 1.0.
        XCTAssertEqual(y.headRate()!, 25, accuracy: 1e-9)
        XCTAssertTrue(y.isLive(now: 1.7))
        XCTAssertFalse(y.isLive(now: 1.75), "0.5 s old is no longer live")
        XCTAssertFalse(y.isUnavailable(now: 3.2))
        XCTAssertTrue(y.isUnavailable(now: 3.25), "no sample for 2 s")
    }

    /// Samples reach the main actor in separate Tasks, which may run out of order: a sample older
    /// than the newest one is dropped, so `newest` and the head rate stay right.
    func testLateSamplesAreDropped() {
        var y = YawTrack()
        y.add(t: 10.00, yaw: 0)
        y.add(t: 10.25, yaw: 4)
        XCTAssertFalse(y.add(t: 10.10, yaw: 2), "a late sample is not a jump either")
        XCTAssertEqual(y.newest?.t, 10.25)
        XCTAssertEqual(y.samples.map { $0.t }, [10.00, 10.25])
        XCTAssertEqual(y.headRate() ?? .nan, 20, accuracy: 1e-9, "abs(4 - 0) / 0.2 s")
        y.add(t: 10.25, yaw: 9) // the same timestamp again
        XCTAssertEqual(y.samples.count, 2)
    }

    /// After a jump the 200 ms mean uses only the new frame (no yaw from neither frame).
    func testAJumpRestartsSmoothingInTheNewFrame() {
        var y = YawTrack()
        for i in 0..<5 { y.add(t: 20 + Double(i) * 0.04, yaw: 0) }
        XCTAssertTrue(y.add(t: 20.17, yaw: 90), "90 degrees in 10 ms")
        XCTAssertEqual(y.smoothedYaw(now: 20.21) ?? .nan, 90, accuracy: 1e-9)
        XCTAssertNil(y.headRate(), "the rate needs 200 ms of the new frame")
    }

    @MainActor
    func testRefreshPublishesOnlyChanges() {
        let tracker = HeadTracker()
        tracker.receive(t: 50.00, yaw: 0)
        tracker.receive(t: 50.04, yaw: 1)
        tracker.refresh(now: 50.05)
        var changes = 0
        let sub = tracker.objectWillChange.sink { _ in changes += 1 }
        tracker.refresh(now: 50.05)
        tracker.refresh(now: 50.06) // same flags, same rounded rate and age
        XCTAssertEqual(changes, 0, "nothing changed, so the screens are not told to redraw")
        tracker.refresh(now: 51.0)  // no longer live, the age moves on
        XCTAssertGreaterThan(changes, 0)
        sub.cancel()
    }

    @MainActor
    func testHeadTrackerReportsJumpsAsFrameResets() {
        let tracker = HeadTracker()
        var resets = 0
        tracker.onFrameReset = { resets += 1 }
        tracker.receive(t: 10.00, yaw: 0)
        tracker.receive(t: 10.04, yaw: 2)
        XCTAssertEqual(resets, 0)
        tracker.receive(t: 10.08, yaw: 80)
        XCTAssertEqual(resets, 1, "a 78 degree step in 40 ms marks directions stale")
        tracker.refresh(now: 10.1)
        XCTAssertTrue(tracker.live)
        XCTAssertNotNil(tracker.yawForCapture(now: 10.1))
        XCTAssertNil(tracker.yawForCapture(now: 11.0), "not live: no direction for a capture")
    }
}
