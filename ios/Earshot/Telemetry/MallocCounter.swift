// MallocCounter.swift - the Swift side of the debug-only malloc counter (Audio/MallocHook.c).
//
// Owns: installing the hook, its self-test, arming it for the render thread and the engine
// worker, and the counts that Diagnostics and the Phase 0 compute run show (contract 11 "Lead
// integration": the 60 min locked compute run must log 0 allocations on those threads; 30-ios
// 15.3 "Compute"). Main actor only; AudioController calls refresh() once a second.
//
// Release builds compile the C hook out, so here `compiledIn` is false and the status reads
// "not in this build". A Phase 0 compute run that should prove 0 allocations needs a Debug
// build on the phone (make build CONFIG=Debug). The engine C code is -O2 in both configs.
//
// Why a self-test: a count of 0 only means something if the hook really sees allocations on
// this iPhone. start() makes one malloc on the main thread with the main thread armed and
// keeps the result; every report carries it.
//
// The pure rules (MallocArming, MallocCounts) are at the top so MallocHookTests can check them.

import Foundation

/// What the counter can say about the real-time threads right now.
enum MallocStatus: String, Codable, Equatable, Sendable {
    case notInBuild = "not_in_build"  // Release build: the hook was compiled out
    case hookFailed = "hook_failed"   // installed, but the self-test saw no allocation: counts mean nothing
    case waiting                      // hook works, a thread to arm is not known yet (no audio, no worker)
    case armed                        // hook works and every expected thread is armed
}

/// The arming rules (pure).
enum MallocArming {
    /// The status from what is known. The stub engine has no worker thread, so only the render
    /// thread is expected then.
    static func status(compiledIn: Bool, selfTestPassed: Bool, renderTID: UInt64, workerTID: UInt64?,
                       expectWorker: Bool) -> MallocStatus {
        if !compiledIn { return .notInBuild }
        if !selfTestPassed { return .hookFailed }
        if renderTID == 0 { return .waiting }
        if expectWorker && (workerTID ?? 0) == 0 { return .waiting }
        return .armed
    }

    /// The thread id each hook slot should hold (0 disarms). Only known ids are armed.
    static func targets(renderTID: UInt64, workerTID: UInt64?) -> [Int: UInt64] {
        [Int(MH_SLOT_RENDER): renderTID, Int(MH_SLOT_WORKER): workerTID ?? 0]
    }

    /// The worker must be looked up again when the engine changed (a swap starts a new worker)
    /// or it was not found yet.
    static func needsWorkerLookup(engineChanged: Bool, workerTID: UInt64?, expectWorker: Bool) -> Bool {
        expectWorker && (engineChanged || (workerTID ?? 0) == 0)
    }
}

/// Allocation and free events seen on the two real-time threads.
struct MallocCounts: Codable, Equatable, Sendable {
    var renderAllocs: UInt64 = 0
    var renderFrees: UInt64 = 0
    var workerAllocs: UInt64 = 0
    var workerFrees: UInt64 = 0

    enum CodingKeys: String, CodingKey {
        case renderAllocs = "render_allocs", renderFrees = "render_frees"
        case workerAllocs = "worker_allocs", workerFrees = "worker_frees"
    }

    var total: UInt64 { renderAllocs + renderFrees + workerAllocs + workerFrees }

    /// The events between `earlier` and self (for one run). Counters only grow; if one went down
    /// the counters were reset in between, and then all of the current value is new.
    func minus(_ earlier: MallocCounts) -> MallocCounts {
        func d(_ now: UInt64, _ then: UInt64) -> UInt64 { now >= then ? now - then : now }
        return MallocCounts(renderAllocs: d(renderAllocs, earlier.renderAllocs),
                            renderFrees: d(renderFrees, earlier.renderFrees),
                            workerAllocs: d(workerAllocs, earlier.workerAllocs),
                            workerFrees: d(workerFrees, earlier.workerFrees))
    }
}

@MainActor
final class MallocCounter: ObservableObject {
    @Published private(set) var status: MallocStatus
    @Published private(set) var counts = MallocCounts()
    @Published private(set) var renderTID: UInt64 = 0
    @Published private(set) var workerTID: UInt64? = nil
    /// True when the worker runs with the real-time time-constraint policy (nil = not found).
    @Published private(set) var workerTimeConstraint: Bool? = nil
    /// Size of the newest allocation seen on either thread (a hint for finding it).
    @Published private(set) var lastSize: UInt64 = 0
    private(set) var selfTestPassed = false
    private var lastEngine: ObjectIdentifier? = nil

    /// The name the engine gives its worker thread (engine/src/rt_thread.c, pthread_setname_np).
    static let workerThreadName = "earshot.worker"

    var compiledIn: Bool { mh_compiled_in() != 0 }

    init() {
        status = mh_compiled_in() != 0 ? .waiting : .notInBuild
    }

    /// Installs the hook once and runs its self-test (start step 5, Debug builds only).
    func start() {
        guard compiledIn else { status = .notInBuild; return }
        if mh_installed() == 0 { _ = mh_install() }
        if !selfTestPassed { selfTestPassed = mh_self_test() != 0 }
        refreshStatus()
    }

    /// Once a second while listening: arms the render thread (its id comes from the callback)
    /// and the engine worker (looked up by name after every engine change), then reads the counts.
    func refresh(renderThreadID: UInt64, engine: ObjectIdentifier?, expectWorker: Bool) {
        guard compiledIn, mh_installed() != 0 else { return }
        let changed = engine != lastEngine
        lastEngine = engine
        if changed { workerTID = nil; workerTimeConstraint = nil }
        if MallocArming.needsWorkerLookup(engineChanged: changed, workerTID: workerTID, expectWorker: expectWorker) {
            var tid: UInt64 = 0
            var rt: Int32 = 0
            if mh_find_thread_named(Self.workerThreadName, &tid, &rt) != 0 {
                workerTID = tid
                workerTimeConstraint = rt != 0
            }
        }
        if renderTID != renderThreadID { renderTID = renderThreadID }
        for (slot, tid) in MallocArming.targets(renderTID: renderThreadID, workerTID: workerTID) {
            if mh_armed(Int32(slot)) != tid { mh_arm(Int32(slot), tid) }
        }
        counts = Self.read()
        refreshStatus(expectWorker: expectWorker)
    }

    /// Audio stopped: the threads are gone, so both slots are disarmed (the counts are kept).
    func disarm() {
        guard compiledIn else { return }
        mh_arm(Int32(MH_SLOT_RENDER), 0)
        mh_arm(Int32(MH_SLOT_WORKER), 0)
        renderTID = 0
        workerTID = nil
        workerTimeConstraint = nil
        lastEngine = nil
        refreshStatus()
    }

    /// The current counts (for a run: read at the start and at the end, then `minus`).
    func snapshot() -> MallocCounts {
        counts = Self.read()
        return counts
    }

    private static func read() -> MallocCounts {
        var c = MallocCounts()
        mh_counts(Int32(MH_SLOT_RENDER), &c.renderAllocs, &c.renderFrees, nil)
        mh_counts(Int32(MH_SLOT_WORKER), &c.workerAllocs, &c.workerFrees, nil)
        return c
    }

    private func refreshStatus(expectWorker: Bool = true) {
        var r: UInt64 = 0, w: UInt64 = 0
        mh_counts(Int32(MH_SLOT_RENDER), nil, nil, &r)
        mh_counts(Int32(MH_SLOT_WORKER), nil, nil, &w)
        lastSize = max(r, w)
        status = MallocArming.status(compiledIn: compiledIn, selfTestPassed: selfTestPassed, renderTID: renderTID,
                                     workerTID: workerTID, expectWorker: expectWorker)
    }

    /// One line for Diagnostics, for example "armed - render 0 / 0, worker 0 / 0 (allocs / frees)".
    var summary: String {
        switch status {
        case .notInBuild: return "not in this build (Release)"
        case .hookFailed: return "hook not working on this device"
        case .waiting, .armed:
            let c = counts
            let worker = workerTID == nil ? "not found" : "\(c.workerAllocs) / \(c.workerFrees)"
            return "\(status.rawValue) - render \(c.renderAllocs) / \(c.renderFrees), worker \(worker) (allocs / frees)"
        }
    }
}
