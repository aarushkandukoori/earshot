// MallocHookTests.swift - the debug malloc counter (Audio/MallocHook.c, Telemetry/MallocCounter.swift;
// contract 11 Lead integration). `make test` builds Debug, so the hook is compiled in here and these
// tests prove on the simulator's libmalloc that: the self-test sees allocations, only the armed
// thread is counted, realloc counts as one allocation and one free, a named thread is found with
// its id, and MallocCounter arms and disarms its slots. The arming rules are tested as pure values.
// Whether the phone's libmalloc calls the hook the same way is shown by the self-test result that
// Diagnostics and every compute row carry (not checked here: no device).

import XCTest
@testable import Earshot

final class MallocHookTests: XCTestCase {
    private func requireHook() throws {
        guard mh_compiled_in() != 0 else { throw XCTSkip("Release build: the malloc counter is compiled out") }
        XCTAssertEqual(mh_install(), 1)
    }

    /// `n` malloc / free pairs. A while loop on purpose: at -Onone a Swift `for _ in 0..<n` loop
    /// itself makes one heap allocation per pass on this runtime (measured), which would be counted.
    private func churn(_ n: Int, size: Int) {
        var i = 0
        while i < n {
            free(malloc(size))
            i += 1
        }
    }

    private func counts(_ slot: Int32) -> (allocs: UInt64, frees: UInt64) {
        var a: UInt64 = 0, f: UInt64 = 0
        mh_counts(slot, &a, &f, nil)
        return (a, f)
    }

    func testHookIsCompiledIntoDebugTests() {
        XCTAssertEqual(mh_compiled_in(), 1, "make test builds Debug, where EARSHOT_MALLOC_HOOK is defined")
    }

    func testSelfTestSeesAllocations() throws {
        try requireHook()
        XCTAssertEqual(mh_installed(), 1)
        XCTAssertEqual(mh_self_test(), 1, "libmalloc calls malloc_logger on this runtime")
    }

    func testOnlyTheArmedThreadIsCounted() throws {
        try requireHook()
        let slot = Int32(MH_SLOT_CHECK)
        let me = mh_current_thread_id()
        XCTAssertNotEqual(me, 0)
        mh_arm(slot, me)
        let before = counts(slot)
        churn(5, size: 64)
        var q = calloc(4, 16)
        q = realloc(q, 4096) // one allocation and one free
        free(q)
        let armed = counts(slot)
        var size: UInt64 = 0
        mh_counts(slot, nil, nil, &size)
        // armed for another thread id before any assertion (XCTest itself allocates)
        mh_arm(slot, me &+ 1_000_000)
        XCTAssertEqual(armed.allocs - before.allocs, 5 + 1 + 1)
        XCTAssertEqual(armed.frees - before.frees, 5 + 1 + 1)
        XCTAssertEqual(size, 4096, "the newest counted allocation was the realloc to 4,096 bytes")

        // this thread's allocations no longer count
        churn(5, size: 32)
        XCTAssertEqual(counts(slot).allocs, armed.allocs)
        mh_arm(slot, 0)
        XCTAssertEqual(mh_armed(slot), 0)
        churn(5, size: 32)
        XCTAssertEqual(counts(slot).allocs, armed.allocs, "a disarmed slot counts nothing")
    }

    /// The worker lookup by name, on a thread that names itself like the engine worker does.
    func testFindsANamedThreadWithItsId() throws {
        try requireHook()
        final class Box: @unchecked Sendable { var id: UInt64 = 0 } // written once before `started`
        let started = DispatchSemaphore(value: 0)
        let finish = DispatchSemaphore(value: 0)
        let box = Box()
        let t = Thread {
            pthread_setname_np("earshot.test.worker")
            box.id = mh_current_thread_id()
            started.signal()
            finish.wait()
        }
        t.start()
        XCTAssertEqual(started.wait(timeout: .now() + 5), .success)
        var tid: UInt64 = 0
        var rt: Int32 = -1
        XCTAssertEqual(mh_find_thread_named("earshot.test.worker", &tid, &rt), 1)
        XCTAssertEqual(tid, box.id)
        XCTAssertEqual(rt, 0, "a plain thread has no time-constraint policy")
        XCTAssertEqual(mh_find_thread_named("no.such.thread", &tid, &rt), 0)
        finish.signal()
    }

    // MARK: Pure arming rules

    func testArmingStatus() {
        XCTAssertEqual(MallocArming.status(compiledIn: false, selfTestPassed: true, renderTID: 1, workerTID: 2,
                                           expectWorker: true), .notInBuild)
        XCTAssertEqual(MallocArming.status(compiledIn: true, selfTestPassed: false, renderTID: 1, workerTID: 2,
                                           expectWorker: true), .hookFailed, "a dead hook would report a false 0")
        XCTAssertEqual(MallocArming.status(compiledIn: true, selfTestPassed: true, renderTID: 0, workerTID: 2,
                                           expectWorker: true), .waiting, "no render callback yet")
        XCTAssertEqual(MallocArming.status(compiledIn: true, selfTestPassed: true, renderTID: 1, workerTID: nil,
                                           expectWorker: true), .waiting, "worker not found yet")
        XCTAssertEqual(MallocArming.status(compiledIn: true, selfTestPassed: true, renderTID: 1, workerTID: nil,
                                           expectWorker: false), .armed, "the stub has no worker thread")
        XCTAssertEqual(MallocArming.status(compiledIn: true, selfTestPassed: true, renderTID: 1, workerTID: 2,
                                           expectWorker: true), .armed)
    }

    func testArmingTargetsAndLookup() {
        let t = MallocArming.targets(renderTID: 11, workerTID: 22)
        XCTAssertEqual(t[Int(MH_SLOT_RENDER)], 11)
        XCTAssertEqual(t[Int(MH_SLOT_WORKER)], 22)
        XCTAssertNil(t[Int(MH_SLOT_CHECK)], "the check slot is never armed by the counter")
        XCTAssertEqual(MallocArming.targets(renderTID: 0, workerTID: nil)[Int(MH_SLOT_WORKER)], 0, "unknown = disarmed")
        XCTAssertTrue(MallocArming.needsWorkerLookup(engineChanged: true, workerTID: 22, expectWorker: true),
                      "a swap starts a new worker thread")
        XCTAssertTrue(MallocArming.needsWorkerLookup(engineChanged: false, workerTID: nil, expectWorker: true))
        XCTAssertFalse(MallocArming.needsWorkerLookup(engineChanged: false, workerTID: 22, expectWorker: true))
        XCTAssertFalse(MallocArming.needsWorkerLookup(engineChanged: true, workerTID: nil, expectWorker: false))
    }

    func testCountsPerRun() {
        let start = MallocCounts(renderAllocs: 3, renderFrees: 3, workerAllocs: 10, workerFrees: 9)
        let end = MallocCounts(renderAllocs: 3, renderFrees: 4, workerAllocs: 12, workerFrees: 9)
        XCTAssertEqual(end.minus(start), MallocCounts(renderAllocs: 0, renderFrees: 1, workerAllocs: 2, workerFrees: 0))
        XCTAssertEqual(end.minus(start).total, 3)
        let afterReset = MallocCounts(renderAllocs: 1, renderFrees: 0, workerAllocs: 0, workerFrees: 0)
        XCTAssertEqual(afterReset.minus(start).renderAllocs, 1, "a reset in between counts all of the new value")
    }

    // MARK: MallocCounter on the main actor

    @MainActor
    func testCounterArmsTheRenderSlotAndDisarms() throws {
        try requireHook()
        let counter = MallocCounter()
        counter.start()
        XCTAssertTrue(counter.selfTestPassed)
        let me = mh_current_thread_id()
        // pretend this test thread is the render thread (the stub has no worker to expect)
        counter.refresh(renderThreadID: me, engine: nil, expectWorker: false)
        XCTAssertEqual(counter.status, .armed)
        XCTAssertEqual(mh_armed(Int32(MH_SLOT_RENDER)), me)
        let before = counter.snapshot()
        churn(3, size: 48)
        let after = counter.snapshot()
        XCTAssertGreaterThanOrEqual(after.minus(before).renderAllocs, 3)
        XCTAssertTrue(counter.summary.hasPrefix("armed"))
        counter.disarm()
        XCTAssertEqual(mh_armed(Int32(MH_SLOT_RENDER)), 0)
        XCTAssertEqual(counter.status, .waiting)
        // expecting a worker that does not exist: the lookup fails and the status stays waiting
        counter.refresh(renderThreadID: me, engine: ObjectIdentifier(self), expectWorker: true)
        XCTAssertNil(counter.workerTID)
        XCTAssertEqual(counter.status, .waiting)
        counter.disarm()
    }
}
