// Phase0Steps.swift - the Phase 0 steps, the items each one is made of, and the step state
// machine that makes every step resumable and skippable (30-ios 15.3, contract 11 I-1).
//
// Pure values (no audio, no files), so Phase0ProgressTests checks every transition. The runner
// (Phase0Runner.swift, main actor) owns one Phase0Progress and saves it after every change to
// Application Support/phase0-progress.json (not Documents, so `make phase0` copies only the
// measurement file).
//
// A step is a list of items, for example "airpods_pro2/64" (buffer step, Pro 2, 64 frames).
// Each finished item has its row in the measurement file already, so a step can stop at any
// item and resume later, even after the app was closed. What a resume repeats is only the item
// that was running at that moment.
//
// The state machine (status per step):
//
//            begin                      last item finished, or "Finish this step"
//   todo ------------> in_progress ----------------------------------------------> done
//    |                  ^       |                                                    |
//    | skip             | begin | skip                                       reopen  |
//    v                  |       v                                                    |
//   skipped ------------+    skipped                         in_progress <-----------+
//
// Skipping keeps the items already finished; reopening a skipped or done step continues where
// it stopped. Steps marked (E-2) need the real engine but can still run with the stub (the file
// says "engine": "stub", so nobody mistakes those numbers).

import Foundation

enum Phase0StepID: String, Codable, CaseIterable, Identifiable, Sendable {
    case buffers, tapTest = "tap_test", latency, probeLevel = "probe_level", motion, stem, locked
    case wespeaker, alarm, micLevel = "mic_level", compute, power

    var id: String { rawValue }

    /// 30-ios 15.3 marks these (E-2): their numbers mean something only with the real engine.
    var needsRealEngine: Bool { self == .compute || self == .power }
}

enum Phase0StepStatus: String, Codable, Equatable, Sendable {
    case todo, inProgress = "in_progress", done, skipped
}

/// What each step is made of (30-ios 15.3 bullets), as item keys.
enum Phase0Plan {
    static let airPods = ["airpods_pro1", "airpods_pro2"]
    static let bufferRoutes = airPods + ["no_headphones"]
    static let bufferSizes = [32, 48, 64, 96, 128, 192]
    static let bufferLockedItem = "locked/64"
    static let methods = [ProbeMethod.inEar.rawValue, ProbeMethod.earDetectionOff.rawValue]
    static let minuteSlots = [0, 10, 30]
    static let ambients = [45, 70]            // dB: quiet room, played babble
    static let probeLevels = [-30, -24, -18]  // dBFS candidates for ES_PROBE_DBFS
    static let probesPerCell = 5              // 5 tries per cell gives the success rate in 20% steps
    static let stemCombos = ["ca_off/spatial_off", "ca_on/spatial_off", "ca_off/spatial_on", "ca_on/spatial_on"]
    static let stemPausedItem = "paused_rate0"
    static let computeItems = ["1/60/64", "3/60/64", "3/30/1104"]   // instances / minutes / frames
    static let powerItems = ["1/120", "3/60"]                        // instances / minutes

    static func bufferKey(route: String, frames: Int) -> String { "\(route)/\(frames)" }
    static func latencyKey(route: String, method: String, slot: Int) -> String { "\(route)/\(method)/\(slot)" }
    static func probeKey(route: String, ambient: Int, level: Int, method: String) -> String {
        "\(route)/\(ambient)/\(level)/\(method)"
    }

    /// Every item of a step, in the order the screen lists them.
    static func items(_ id: Phase0StepID) -> [String] {
        switch id {
        case .buffers:
            return bufferRoutes.flatMap { r in bufferSizes.map { bufferKey(route: r, frames: $0) } } + [bufferLockedItem]
        case .tapTest:
            return airPods
        case .latency:
            return airPods.flatMap { r in
                methods.flatMap { m in minuteSlots.map { latencyKey(route: r, method: m, slot: $0) } }
            }
        case .probeLevel:
            return airPods.flatMap { r in
                ambients.flatMap { a in
                    probeLevels.flatMap { l in methods.map { probeKey(route: r, ambient: a, level: l, method: $0) } }
                }
            }
        case .motion: return ["on", "locked"]
        case .stem: return stemCombos + [stemPausedItem]
        case .locked: return ["routes_and_swap"]
        case .wespeaker: return ["on", "locked"]
        case .alarm: return ["locked"]
        case .micLevel: return ["1m"]
        case .compute: return computeItems
        case .power: return powerItems
        }
    }
}

/// Where the owner is in Phase 0. Codable so it survives the app being closed.
struct Phase0Progress: Codable, Equatable {
    /// The measurement file this run writes (phase0-<reportDate>.json): the day it was started.
    var reportDate: String
    /// The step the screen shows first (the last one begun), nil before any.
    private(set) var current: Phase0StepID? = nil
    /// When each AirPods route was last connected, for the latency 0 / 10 / 30 min items.
    var connectedAt: [String: Date] = [:]
    private var statuses: [String: Phase0StepStatus] = [:]  // by Phase0StepID raw value
    private var finished: [String: [String]] = [:]          // finished items, in finishing order

    init(reportDate: String) { self.reportDate = reportDate }

    func status(_ id: Phase0StepID) -> Phase0StepStatus { statuses[id.rawValue] ?? .todo }

    func isFinished(_ id: Phase0StepID, _ item: String) -> Bool { finished[id.rawValue]?.contains(item) ?? false }

    /// The items still to do, in plan order.
    func remaining(_ id: Phase0StepID) -> [String] { Phase0Plan.items(id).filter { !isFinished(id, $0) } }

    func finishedCount(_ id: Phase0StepID) -> Int { finished[id.rawValue]?.count ?? 0 }

    /// Start or continue a step (todo, skipped or done -> in_progress).
    mutating func begin(_ id: Phase0StepID) {
        statuses[id.rawValue] = .inProgress
        current = id
    }

    /// One item is measured and its row saved. The last item makes the step done.
    mutating func finish(_ id: Phase0StepID, item: String) {
        var list = finished[id.rawValue] ?? []
        if !list.contains(item) { list.append(item) }
        finished[id.rawValue] = list
        current = id
        statuses[id.rawValue] = remaining(id).isEmpty ? .done : .inProgress
    }

    /// Skip the step for now. Finished items are kept; the screen moves on to the next open step.
    mutating func skip(_ id: Phase0StepID) {
        statuses[id.rawValue] = .skipped
        if current == id { current = nextOpen(after: id) }
    }

    /// "Finish this step": done with what is there (for example only AirPods Pro 2 are at hand).
    /// Only a step that was begun can be finished this way; a step never begun is skipped instead.
    mutating func markDone(_ id: Phase0StepID) {
        guard status(id) == .inProgress else { return }
        statuses[id.rawValue] = .done
        if current == id { current = nextOpen(after: id) }
    }

    /// Back to in_progress from done or skipped (more rows, or a redo).
    mutating func reopen(_ id: Phase0StepID) {
        guard status(id) == .done || status(id) == .skipped else { return }
        statuses[id.rawValue] = .inProgress
        current = id
    }

    /// The first step after `after` (wrapping around) that is todo or in_progress, nil when every
    /// step is done or skipped.
    func nextOpen(after: Phase0StepID?) -> Phase0StepID? {
        let all = Phase0StepID.allCases
        let start = after.flatMap { all.firstIndex(of: $0) }.map { $0 + 1 } ?? 0
        for k in 0..<all.count {
            let id = all[(start + k) % all.count]
            if status(id) == .todo || status(id) == .inProgress { return id }
        }
        return nil
    }

    /// Done or skipped steps, for the "8 of 12 steps" line.
    var settledCount: Int {
        Phase0StepID.allCases.filter { status($0) == .done || status($0) == .skipped }.count
    }
}

/// Application Support/phase0-progress.json (the runner's resume point; never copied off the phone).
struct Phase0ProgressStore {
    let url: URL

    static func standard() -> Phase0ProgressStore {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return Phase0ProgressStore(url: dir.appendingPathComponent("phase0-progress.json"))
    }

    func load() -> Phase0Progress? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Phase0Progress.self, from: data)
    }

    func save(_ p: Phase0Progress) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(p).write(to: url, options: .atomic)
    }
}
