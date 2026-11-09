// SessionLog.swift - the per-minute session log (contract 5.2, 30-ios 14, D47). On this iPhone
// only: no network, no audio, no names. `make logs` copies the files to ios/build/logs.
//
// Every 60 s while listening, and once more at Stop for the last part-minute, one JSON line is
// appended to Documents/logs/YYYY-MM-DD.jsonl with exactly the contract 5.2 keys:
//   t, route, io_rate, io_frames, slack16, mode, n_instances, compute_us_p50, compute_us_p99,
//   compute_us_max, underruns, nan_resets, howl, thermal, battery, low_power, out_rms_db,
//   limiter_gr_db, p_target_mean, p_self_mean, latency_measured_ms, latency_reported_ms,
//   app_version, model_hash
// A missing value is written as null (never left out), so every line has the same keys.
//
// What each value means (interpretations are marked):
//   t                    - end of the minute, ISO 8601 in UTC
//   route                - the first output's port type ("BluetoothA2DPOutput", ...), never its
//                          user-set name (interpretation: names can hold a person's name)
//   compute_us_*         - from that minute's drainTimings() (per-hop compute, microseconds)
//   underruns, nan_resets - per-minute deltas (an engine swap restarts the engine's counters; the
//                          delta still counts both engines, keyed by the engine object)
//   n_instances          - slots the engine really computes (interpretation: the engine is always
//                          built with 3; thermal downshift empties slots instead)
//   howl                 - true if the runaway guard was active at any 10 Hz poll in the minute
//   thermal              - the hottest thermalState seen in the minute (nominal/fair/serious/critical)
//   battery              - percent 0-100 at the end of the minute, null when unknown
//   low_power            - true if Low Power Mode was on at any second of the minute
//   out_rms_db           - mean of the 10 Hz out_rms_db readings; limiter_gr_db - the largest
//                          limiter gain reduction seen (interpretation: safety cares about the peak)
//   p_target_mean        - 3 numbers, mean per slot; p_self_mean - mean p_self
//   latency_measured_ms  - stored probe loop + Earshot share for this port, or null
//   latency_reported_ms  - the iOS-reported estimate (30-ios 7)
//
// Who calls what: AudioController (main actor, AudioController+Safety.swift) feeds MinuteTelemetry
// from its 10 Hz poll and once-a-second tick, and hands each finished record to SessionLog, which
// encodes and appends it on a serial utility queue. Logs older than 30 days are deleted.

import Foundation

/// One line of the session log. Keys are exactly the contract 5.2 names.
struct SessionLogRecord: Encodable, Equatable {
    var t: Date
    var route: String
    var ioRate: Int
    var ioFrames: Int
    var slack16: Int
    var mode: String
    var nInstances: Int
    var computeUsP50: Int
    var computeUsP99: Int
    var computeUsMax: Int
    var underruns: Int
    var nanResets: Int
    var howl: Bool
    var thermal: String
    var battery: Int?
    var lowPower: Bool
    var outRmsDB: Double
    var limiterGrDB: Double
    var pTargetMean: [Double]
    var pSelfMean: Double
    var latencyMeasuredMS: Int?
    var latencyReportedMS: Int
    var appVersion: String
    var modelHash: String

    enum CodingKeys: String, CodingKey, CaseIterable {
        case t, route, ioRate = "io_rate", ioFrames = "io_frames", slack16, mode, nInstances = "n_instances"
        case computeUsP50 = "compute_us_p50", computeUsP99 = "compute_us_p99", computeUsMax = "compute_us_max"
        case underruns, nanResets = "nan_resets", howl, thermal, battery, lowPower = "low_power"
        case outRmsDB = "out_rms_db", limiterGrDB = "limiter_gr_db", pTargetMean = "p_target_mean"
        case pSelfMean = "p_self_mean", latencyMeasuredMS = "latency_measured_ms"
        case latencyReportedMS = "latency_reported_ms", appVersion = "app_version", modelHash = "model_hash"
    }

    /// Written by hand so an optional becomes null instead of a missing key.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(SessionLog.timestamp(t), forKey: .t)
        try c.encode(route, forKey: .route)
        try c.encode(ioRate, forKey: .ioRate)
        try c.encode(ioFrames, forKey: .ioFrames)
        try c.encode(slack16, forKey: .slack16)
        try c.encode(mode, forKey: .mode)
        try c.encode(nInstances, forKey: .nInstances)
        try c.encode(computeUsP50, forKey: .computeUsP50)
        try c.encode(computeUsP99, forKey: .computeUsP99)
        try c.encode(computeUsMax, forKey: .computeUsMax)
        try c.encode(underruns, forKey: .underruns)
        try c.encode(nanResets, forKey: .nanResets)
        try c.encode(howl, forKey: .howl)
        try c.encode(thermal, forKey: .thermal)
        try c.encode(battery, forKey: .battery)              // Int? encodes null when nil
        try c.encode(lowPower, forKey: .lowPower)
        try c.encode(Self.finite(outRmsDB), forKey: .outRmsDB)
        try c.encode(Self.finite(limiterGrDB), forKey: .limiterGrDB)
        try c.encode(pTargetMean.map(Self.finite), forKey: .pTargetMean)
        try c.encode(Self.finite(pSelfMean), forKey: .pSelfMean)
        try c.encode(latencyMeasuredMS, forKey: .latencyMeasuredMS)
        try c.encode(latencyReportedMS, forKey: .latencyReportedMS)
        try c.encode(appVersion, forKey: .appVersion)
        try c.encode(modelHash, forKey: .modelHash)
    }

    /// JSON has no NaN or infinity; such a value is written as 0 rounded to 0.1 (never crash a log).
    static func finite(_ x: Double) -> Double { x.isFinite ? (x * 10).rounded() / 10 : 0 }
}

/// Per-hop compute times in 1 us bins up to 9,999 us, plus one bin for anything slower. 40 KB.
struct ComputeHistogram: Equatable {
    static let bins = 10_000
    private(set) var counts = [UInt32](repeating: 0, count: bins + 1)
    private(set) var total = 0
    private(set) var maxUS = 0

    mutating func add(_ us: [UInt16]) {
        for v in us {
            let x = Int(v)
            counts[min(x, Self.bins)] += 1
            maxUS = max(maxUS, x)
        }
        total += us.count
    }

    /// The smallest value with at least p of all hops at or below it (p in 0...1); 0 when empty.
    /// The overflow bin reports the true maximum.
    func percentile(_ p: Double) -> Int {
        guard total > 0 else { return 0 }
        let need = max(1, Int((p * Double(total)).rounded(.up)))
        var seen = 0
        for (value, n) in counts.enumerated() {
            seen += Int(n)
            if seen >= need { return value == Self.bins ? maxUS : value }
        }
        return maxUS
    }
}

/// Collects one minute of readings (main actor, plain struct so it is easy to test).
struct MinuteTelemetry {
    private(set) var histogram = ComputeHistogram()
    private(set) var underruns = 0
    private(set) var nanResets = 0
    private(set) var howl = false
    private(set) var worstThermal: ProcessInfo.ThermalState = .nominal
    private(set) var lowPower = false
    private(set) var polls = 0
    private var outRmsSum = 0.0
    private(set) var limiterMax = 0.0
    private var pTargetSum = [Double](repeating: 0, count: TableState.slotCount)
    private var pSelfSum = 0.0
    // Last engine counter values seen, to turn the engine's running totals into deltas, and the
    // engine they came from (a different engine object = a swap, whose counters start at 0).
    private var lastUnderruns: UInt64? = nil
    private var lastNaN: UInt64? = nil
    private var lastEngine: ObjectIdentifier? = nil

    /// The engine's running total `now` after `last`: a smaller value means a new engine (swap),
    /// whose counter started at 0, so all of `now` is new. No `last` yet (the session's first poll)
    /// also counts all of `now`, since every session starts with a new engine.
    static func delta(now: UInt64, last: UInt64?) -> Int {
        guard let last else { return Int(now) }
        return Int(now >= last ? now - last : now)
    }

    /// One 10 Hz stats poll. `engine` identifies the engine object the stats came from: when it
    /// changes, all of the new engine's counts are new, even when they already reach the old
    /// engine's last value (old 5, new 6 is 6 new underruns, not 1).
    mutating func addPoll(_ s: EngineStats, engine: ObjectIdentifier? = nil) {
        if engine != lastEngine {
            lastUnderruns = nil
            lastNaN = nil
            lastEngine = engine
        }
        underruns += Self.delta(now: s.underruns, last: lastUnderruns)
        nanResets += Self.delta(now: s.nanResets, last: lastNaN)
        lastUnderruns = s.underruns
        lastNaN = s.nanResets
        howl = howl || s.howl
        polls += 1
        outRmsSum += Double(s.outRmsDB)
        limiterMax = max(limiterMax, Double(s.limiterGrDB))
        for i in 0..<min(pTargetSum.count, s.pTarget.count) { pTargetSum[i] += Double(s.pTarget[i]) }
        pSelfSum += Double(s.pSelf)
    }

    /// Once a second: that second's per-hop timings and the device state.
    mutating func addSecond(timings: [UInt16], thermal: ProcessInfo.ThermalState, lowPower: Bool) {
        histogram.add(timings)
        if thermal.rawValue > worstThermal.rawValue { worstThermal = thermal }
        self.lowPower = self.lowPower || lowPower
    }

    /// Starts the next minute; the engine counters carry over so the next delta is right.
    mutating func nextMinute() {
        let u = lastUnderruns, n = lastNaN, e = lastEngine
        self = MinuteTelemetry()
        lastUnderruns = u
        lastNaN = n
        lastEngine = e
    }

    var outRmsMean: Double { polls > 0 ? outRmsSum / Double(polls) : -100 }
    var pTargetMeans: [Double] { pTargetSum.map { polls > 0 ? $0 / Double(polls) : 0 } }
    var pSelfMean: Double { polls > 0 ? pSelfSum / Double(polls) : 0 }

    static func thermalText(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

/// Appends records to Documents/logs/YYYY-MM-DD.jsonl on a serial utility queue and deletes logs
/// older than 30 days. Main actor calls append / purge; the file work happens on the queue.
final class SessionLog: @unchecked Sendable {
    static let keepDays = 30

    let folder: URL
    private let queue = DispatchQueue(label: "earshot.sessionlog", qos: .utility)

    init(folder: URL) { self.folder = folder }

    /// Documents/logs in the app container (what `make logs` copies).
    static func standard() -> SessionLog {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return SessionLog(folder: docs.appendingPathComponent("logs", isDirectory: true))
    }

    /// "2026-11-08T01:02:03Z" (UTC).
    static func timestamp(_ d: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f.string(from: d)
    }

    /// "2026-11-08.jsonl", in the phone's own calendar day (the owner reads the logs by local date).
    static func fileName(for d: Date, timeZone: TimeZone = .current) -> String {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = timeZone
        f.dateFormat = "yyyy-MM-dd"
        return f.string(from: d) + ".jsonl"
    }

    /// One JSON object on one line (keys sorted so lines are easy to compare by eye).
    static func encodeLine(_ r: SessionLogRecord) throws -> Data {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]
        var data = try e.encode(r)
        data.append(0x0A) // "\n"
        return data
    }

    /// Appends one record to its day's file. `done` (tests) runs on the queue afterwards.
    func append(_ r: SessionLogRecord, done: (@Sendable (Bool) -> Void)? = nil) {
        let line: Data
        do { line = try Self.encodeLine(r) } catch { done?(false); return }
        let url = folder.appendingPathComponent(Self.fileName(for: r.t))
        let dir = folder
        queue.async {
            var ok = true
            do {
                try Self.prepareFolder(dir)
                if FileManager.default.fileExists(atPath: url.path) {
                    let h = try FileHandle(forWritingTo: url)
                    try h.seekToEnd()
                    try h.write(contentsOf: line)
                    try h.close()
                } else {
                    try line.write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
                }
            } catch { ok = false }
            done?(ok)
        }
    }

    /// Creates the folder if needed and keeps it out of iCloud and computer backups, so the logs
    /// stay on this iPhone (`make logs` is the only way they leave it).
    static func prepareFolder(_ dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var folder = dir
        try folder.setResourceValues(values)
    }

    /// Deletes YYYY-MM-DD.jsonl files whose day is more than 30 days before `now`. Files with any
    /// other name are left alone. `done` (tests) gets the deleted names.
    func purgeOld(now: Date = Date(), done: (@Sendable ([String]) -> Void)? = nil) {
        let dir = folder
        queue.async {
            let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
            let deleted = Self.namesToDelete(names, now: now)
            for name in deleted { try? FileManager.default.removeItem(at: dir.appendingPathComponent(name)) }
            done?(deleted)
        }
    }

    /// Pure part of purgeOld: the log names older than keepDays.
    static func namesToDelete(_ names: [String], now: Date, timeZone: TimeZone = .current) -> [String] {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = timeZone
        f.dateFormat = "yyyy-MM-dd"
        let cutoff = now.addingTimeInterval(-Double(keepDays) * 86_400)
        return names.filter { name in
            guard name.hasSuffix(".jsonl"), let day = f.date(from: String(name.dropLast(6))) else { return false }
            return day.addingTimeInterval(86_400) < cutoff // the whole day is older than 30 days
        }.sorted()
    }

    /// Runs `body` on the queue after everything queued before it (tests wait on this).
    func sync(_ body: @escaping @Sendable () -> Void) { queue.async(execute: body) }
}
