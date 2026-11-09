// Phase0ProgressTests.swift - the Phase 0 step state machine (Telemetry/Phase0Steps.swift): begin,
// finish items, skip, finish early, reopen, the next open step, the item plan, and resuming after
// the app was closed (through the store and through a second Phase0Runner on the same files).

import XCTest
@testable import Earshot

final class Phase0ProgressTests: XCTestCase {
    func testPlanItems() {
        XCTAssertEqual(Phase0Plan.items(.buffers).count, 3 * 6 + 1)
        XCTAssertEqual(Phase0Plan.items(.buffers).first, "airpods_pro1/32")
        XCTAssertEqual(Phase0Plan.items(.buffers).last, "locked/64")
        XCTAssertEqual(Phase0Plan.items(.tapTest), ["airpods_pro1", "airpods_pro2"])
        XCTAssertEqual(Phase0Plan.items(.latency).count, 2 * 2 * 3)
        XCTAssertTrue(Phase0Plan.items(.latency).contains("airpods_pro2/ear_detection_off/30"))
        XCTAssertEqual(Phase0Plan.items(.probeLevel).count, 2 * 2 * 3 * 2)
        XCTAssertTrue(Phase0Plan.items(.probeLevel).contains("airpods_pro1/70/-18/in_ear"))
        XCTAssertEqual(Phase0Plan.items(.stem).count, 5)
        XCTAssertEqual(Phase0Plan.items(.compute), ["1/60/64", "3/60/64", "3/30/1104"])
        XCTAssertEqual(Phase0Plan.items(.power), ["1/120", "3/60"])
        for id in Phase0StepID.allCases {
            let items = Phase0Plan.items(id)
            XCTAssertFalse(items.isEmpty, "\(id)")
            XCTAssertEqual(Set(items).count, items.count, "\(id) items are unique")
        }
        XCTAssertEqual(Phase0StepID.allCases.filter { $0.needsRealEngine }, [.compute, .power])
    }

    func testNewProgressIsAllToDo() {
        let p = Phase0Progress(reportDate: "2026-11-08")
        for id in Phase0StepID.allCases { XCTAssertEqual(p.status(id), .todo) }
        XCTAssertNil(p.current)
        XCTAssertEqual(p.nextOpen(after: nil), .buffers)
        XCTAssertEqual(p.settledCount, 0)
    }

    func testFinishingEveryItemMakesTheStepDone() {
        var p = Phase0Progress(reportDate: "2026-11-08")
        p.begin(.tapTest)
        XCTAssertEqual(p.status(.tapTest), .inProgress)
        XCTAssertEqual(p.current, .tapTest)
        p.finish(.tapTest, item: "airpods_pro2")
        XCTAssertEqual(p.status(.tapTest), .inProgress)
        XCTAssertEqual(p.remaining(.tapTest), ["airpods_pro1"])
        p.finish(.tapTest, item: "airpods_pro2") // twice changes nothing
        XCTAssertEqual(p.finishedCount(.tapTest), 1)
        p.finish(.tapTest, item: "airpods_pro1")
        XCTAssertEqual(p.status(.tapTest), .done)
        XCTAssertEqual(p.remaining(.tapTest), [])
    }

    /// Resume: the remaining items come in plan order and skip the finished ones.
    func testRemainingKeepsPlanOrder() {
        var p = Phase0Progress(reportDate: "2026-11-08")
        p.begin(.buffers)
        p.finish(.buffers, item: "airpods_pro1/64")
        p.finish(.buffers, item: "airpods_pro1/32")
        let left = p.remaining(.buffers)
        XCTAssertEqual(Array(left.prefix(4)), ["airpods_pro1/48", "airpods_pro1/96", "airpods_pro1/128", "airpods_pro1/192"])
        XCTAssertEqual(left.count, 17)
    }

    func testSkipKeepsFinishedItemsAndMovesOn() {
        var p = Phase0Progress(reportDate: "2026-11-08")
        p.begin(.buffers)
        p.finish(.buffers, item: "airpods_pro2/64")
        p.skip(.buffers)
        XCTAssertEqual(p.status(.buffers), .skipped)
        XCTAssertEqual(p.finishedCount(.buffers), 1, "skipping never drops measured items")
        XCTAssertEqual(p.current, .tapTest, "the screen moves on to the next open step")
        p.reopen(.buffers)
        XCTAssertEqual(p.status(.buffers), .inProgress)
        XCTAssertTrue(p.isFinished(.buffers, "airpods_pro2/64"))
        XCTAssertFalse(p.remaining(.buffers).contains("airpods_pro2/64"))
    }

    func testSkipAStepNeverBegun() {
        var p = Phase0Progress(reportDate: "2026-11-08")
        p.skip(.compute)
        XCTAssertEqual(p.status(.compute), .skipped)
        p.begin(.compute) // a run on a skipped step begins it again
        XCTAssertEqual(p.status(.compute), .inProgress)
    }

    func testFinishThisStepOnlyAfterBegin() {
        var p = Phase0Progress(reportDate: "2026-11-08")
        p.markDone(.motion)
        XCTAssertEqual(p.status(.motion), .todo, "a step never begun is skipped, not finished")
        p.begin(.motion)
        p.finish(.motion, item: "on")
        p.markDone(.motion)
        XCTAssertEqual(p.status(.motion), .done, "done with what is there")
        XCTAssertEqual(p.remaining(.motion), ["locked"])
        XCTAssertEqual(p.current, .stem, "the next open step after it")
    }

    func testReopenOnlyFromDoneOrSkipped() {
        var p = Phase0Progress(reportDate: "2026-11-08")
        p.reopen(.alarm)
        XCTAssertEqual(p.status(.alarm), .todo)
        p.begin(.alarm)
        p.finish(.alarm, item: "locked")
        XCTAssertEqual(p.status(.alarm), .done)
        p.reopen(.alarm)
        XCTAssertEqual(p.status(.alarm), .inProgress)
        p.finish(.alarm, item: "locked")
        XCTAssertEqual(p.status(.alarm), .done, "finishing an already finished item settles it again")
    }

    func testNextOpenIsNilWhenEveryStepIsSettled() {
        var p = Phase0Progress(reportDate: "2026-11-08")
        for id in Phase0StepID.allCases { p.skip(id) }
        XCTAssertNil(p.nextOpen(after: nil))
        XCTAssertEqual(p.settledCount, Phase0StepID.allCases.count)
        p.reopen(.locked)
        XCTAssertEqual(p.nextOpen(after: .power), .locked)
        XCTAssertEqual(p.nextOpen(after: .locked), .locked, "wraps around to itself")
    }

    func testProgressSurvivesAppRestart() throws {
        let store = Phase0ProgressStore(url: TestSupport.tempFolder().appendingPathComponent("phase0-progress.json"))
        XCTAssertNil(store.load())
        var p = Phase0Progress(reportDate: "2026-11-08")
        p.begin(.latency)
        p.finish(.latency, item: "airpods_pro2/in_ear/0")
        p.skip(.stem)
        p.connectedAt["airpods_pro2"] = Date(timeIntervalSince1970: 1_794_100_000)
        try store.save(p)
        let back = try XCTUnwrap(store.load())
        XCTAssertEqual(back, p)
        XCTAssertEqual(back.status(.latency), .inProgress)
        XCTAssertEqual(back.status(.stem), .skipped)
        XCTAssertEqual(back.remaining(.latency).count, 11)
    }

    /// The runner itself: a recorded row lands in the file at once, and a second runner made on the
    /// same files (the app opened again) resumes with the same report and progress.
    @MainActor
    func testRunnerResumesFromItsFiles() throws {
        let dir = TestSupport.tempFolder()
        let store = Phase0Store(folder: dir.appendingPathComponent("phase0"))
        let progressStore = Phase0ProgressStore(url: dir.appendingPathComponent("progress.json"))
        let day = Date(timeIntervalSince1970: 1_794_140_000) // 2026-11-08 midday UTC
        let first = Phase0Runner(store: store, progressStore: progressStore, now: day)
        let date = Phase0Store.dateString(day)
        XCTAssertEqual(first.report.date, date)
        first.record(.alarm, item: "locked") {
            $0.alarm.append(Phase0AlarmRow(screen: "locked", seconds: 60, windows: 39, ok: true))
        }
        first.skip(.buffers)
        first.markConnected(route: "airpods_pro2", now: day)
        XCTAssertEqual(store.load(date: date)?.alarm.count, 1, "saved at once")

        let second = Phase0Runner(store: store, progressStore: progressStore, now: day.addingTimeInterval(86_400 * 2))
        XCTAssertEqual(second.report.date, date, "keeps writing the file of the day it started")
        XCTAssertEqual(second.report.alarm.count, 1)
        XCTAssertEqual(second.progress.status(.alarm), .done)
        XCTAssertEqual(second.progress.status(.buffers), .skipped)
        XCTAssertEqual(second.minutesSinceConnect("airpods_pro2", now: day.addingTimeInterval(600)), 10)
        XCTAssertNil(second.run, "nothing is running after a restart; the open item is simply run again")

        second.newFile(now: day.addingTimeInterval(86_400 * 2))
        XCTAssertNotEqual(second.report.date, date)
        XCTAssertEqual(second.progress.status(.alarm), .todo)
        XCTAssertNotNil(store.load(date: date), "the old file stays")
    }
}
