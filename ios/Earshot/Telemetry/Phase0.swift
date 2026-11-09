// Phase0.swift - the Phase 0 measurement file (30-ios 15.3, contract 11 I-1): its schema, how
// it is written to Documents/phase0/phase0-YYYY-MM-DD.json, and the small hooks the runner reads
// from AudioController.
//
// Main actor for the store calls (the runner); the row types are plain values usable anywhere.
//
// The schema is exactly the 30-ios 15.3 JSON. Every documented key is always written; a value
// that was not measured (a failed probe has no loop_ms) is written as JSON null, never left out.
// A few rows carry additional keys the spec example does not show, marked "I-1 addition" below,
// because the row would be ambiguous without them (for example which callback size a compute
// run had). Additions are optional when reading, so the spec example itself decodes (tested).
// `make phase0` copies Documents/phase0 to docs/measurements.

import Foundation

// MARK: - Rows (30-ios 15.3)

struct Phase0BufferRow: Codable, Equatable {
    var route: String          // airpods_pro1, airpods_pro2, no_headphones, wired
    var mode: String           // the AVAudioSession mode, "measurement"
    var requested: Int         // frames asked for
    var granted: Int           // round(ioBufferDuration x rate) after activation
    var callback: Int          // frames of the real render callbacks (es_stats.last_frames)
    var rate: Int
    var inLatMS: Double
    var outLatMS: Double
    var underruns60s: Int      // worst 60 s window of the run (the whole run when it is 60 s)
    var seconds: Int?          // I-1 addition: run length
    var screen: String?        // I-1 addition: on, locked or mixed (measured from the scene phase)
    var underruns: Int?        // I-1 addition: underruns in the whole run

    enum CodingKeys: String, CodingKey {
        case route, mode, requested, granted, callback, rate, inLatMS = "in_lat_ms", outLatMS = "out_lat_ms"
        case underruns60s = "underruns_60s", seconds, screen, underruns
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(route, forKey: .route); try c.encode(mode, forKey: .mode)
        try c.encode(requested, forKey: .requested); try c.encode(granted, forKey: .granted)
        try c.encode(callback, forKey: .callback); try c.encode(rate, forKey: .rate)
        try c.encode(inLatMS, forKey: .inLatMS); try c.encode(outLatMS, forKey: .outLatMS)
        try c.encode(underruns60s, forKey: .underruns60s); try c.encodeOrNull(seconds, forKey: .seconds)
        try c.encodeOrNull(screen, forKey: .screen); try c.encodeOrNull(underruns, forKey: .underruns)
    }
}

struct Phase0TapRow: Codable, Equatable {
    var route: String
    var phoneTapPeakDB: Double
    var airpodTapPeakDB: Double
    var dataSource: String?    // the built-in mic data source iOS picked ("Bottom"), null if none
    var pass: Bool

    enum CodingKeys: String, CodingKey {
        case route, phoneTapPeakDB = "phone_tap_peak_db", airpodTapPeakDB = "airpod_tap_peak_db"
        case dataSource = "data_source", pass
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(route, forKey: .route); try c.encode(phoneTapPeakDB, forKey: .phoneTapPeakDB)
        try c.encode(airpodTapPeakDB, forKey: .airpodTapPeakDB); try c.encodeOrNull(dataSource, forKey: .dataSource)
        try c.encode(pass, forKey: .pass)
    }
}

/// One probe run. A failed probe is a row too (loop_ms null, ok false): the probe-level choice
/// needs the success rate, not only the successes.
struct Phase0LatencyRow: Codable, Equatable {
    var route: String
    var method: String         // in_ear (method A) or ear_detection_off (method B)
    var minute: Int            // whole minutes since this route was connected
    var ambientDB: Int         // about 45 (quiet room) or about 70 (played babble), from the owner
    var probeDBFS: Int         // ES_PROBE_DBFS of this build
    var loopMS: Double?
    var spreadMS: Double?
    var psrDB: Double?
    var okSweeps: Int
    var reportedMS: Int        // iOS-reported delay estimate (LatencyDisplay.reportedMS)
    var ok: Bool?              // I-1 addition: the probe gave a valid result

    enum CodingKeys: String, CodingKey {
        case route, method, minute, ambientDB = "ambient_db", probeDBFS = "probe_dbfs", loopMS = "loop_ms"
        case spreadMS = "spread_ms", psrDB = "psr_db", okSweeps = "ok_sweeps", reportedMS = "reported_ms", ok
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(route, forKey: .route); try c.encode(method, forKey: .method)
        try c.encode(minute, forKey: .minute); try c.encode(ambientDB, forKey: .ambientDB)
        try c.encode(probeDBFS, forKey: .probeDBFS); try c.encodeOrNull(loopMS, forKey: .loopMS)
        try c.encodeOrNull(spreadMS, forKey: .spreadMS); try c.encodeOrNull(psrDB, forKey: .psrDB)
        try c.encode(okSweeps, forKey: .okSweeps); try c.encode(reportedMS, forKey: .reportedMS)
        try c.encodeOrNull(ok, forKey: .ok)
    }
}

struct Phase0MotionRow: Codable, Equatable {
    var route: String
    var screen: String
    var minutes: Double
    var rateHz: Double
    var gapsOver200ms: Int
    var maxGapMS: Double

    enum CodingKeys: String, CodingKey {
        case route, screen, minutes, rateHz = "rate_hz", gapsOver200ms = "gaps_over_200ms", maxGapMS = "max_gap_ms"
    }
}

/// One stem session. A press-combination row has single / double / triple and a null
/// single_while_paused_rate0; the paused row is the other way round.
struct Phase0StemRow: Codable, Equatable {
    var route: String
    var convAwareness: String  // on or off (the owner's AirPods setting; iOS cannot read it)
    var spatial: String        // on or off
    var screen: String
    var single: Bool?
    var double: Bool?
    var triple: Bool?
    var singleWhilePausedRate0: Bool?

    enum CodingKeys: String, CodingKey {
        case route, convAwareness = "conv_awareness", spatial, screen, single, double, triple
        case singleWhilePausedRate0 = "single_while_paused_rate0"
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(route, forKey: .route); try c.encode(convAwareness, forKey: .convAwareness)
        try c.encode(spatial, forKey: .spatial); try c.encode(screen, forKey: .screen)
        try c.encodeOrNull(single, forKey: .single); try c.encodeOrNull(double, forKey: .double)
        try c.encodeOrNull(triple, forKey: .triple)
        try c.encodeOrNull(singleWhilePausedRate0, forKey: .singleWhilePausedRate0)
    }
}

/// One bin of a compute histogram, in microseconds: lo <= x < hi (hi null = no upper end).
struct Phase0HistBin: Codable, Equatable {
    var lo: Int
    var hi: Int?
    var n: Int

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(lo, forKey: .lo); try c.encodeOrNull(hi, forKey: .hi); try c.encode(n, forKey: .n)
    }
}

/// What the debug malloc counter saw during a compute run (contract 11 Lead integration).
struct Phase0MallocResult: Codable, Equatable {
    var status: String         // MallocStatus raw value; only "armed" makes the counts meaningful
    var counts: MallocCounts

    enum CodingKeys: String, CodingKey { case status, counts }
}

struct Phase0ComputeRow: Codable, Equatable {
    var instances: Int         // filled slots (1 or 3)
    var minutes: Double
    var screen: String
    var p50US: Int
    var p99US: Int
    var p999US: Int
    var maxUS: Int
    var underruns: Int
    var workgroup: Bool        // the worker was given RemoteIO's workgroup (eh_workgroup non-NULL)
    var frames: Int?           // I-1 addition: render callback size (64 normally, 1,104 with Sound Recognition)
    var histUS: [Phase0HistBin]?  // I-1 addition: per-hop compute histogram (Phase0Math.bins)
    var malloc: Phase0MallocResult?  // I-1 addition: debug malloc counter on the render and worker threads

    enum CodingKeys: String, CodingKey {
        case instances, minutes, screen, p50US = "p50_us", p99US = "p99_us", p999US = "p999_us", maxUS = "max_us"
        case underruns, workgroup, frames, histUS = "hist_us", malloc
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(instances, forKey: .instances); try c.encode(minutes, forKey: .minutes)
        try c.encode(screen, forKey: .screen); try c.encode(p50US, forKey: .p50US)
        try c.encode(p99US, forKey: .p99US); try c.encode(p999US, forKey: .p999US)
        try c.encode(maxUS, forKey: .maxUS); try c.encode(underruns, forKey: .underruns)
        try c.encode(workgroup, forKey: .workgroup); try c.encodeOrNull(frames, forKey: .frames)
        try c.encodeOrNull(histUS, forKey: .histUS); try c.encodeOrNull(malloc, forKey: .malloc)
    }
}

struct Phase0PowerRow: Codable, Equatable {
    var minutes: Double
    var instances: Int
    var pctPerHour: Double
    var maxThermal: String
    var batteryStart: Int?     // I-1 addition: battery percent at the start and the end
    var batteryEnd: Int?

    enum CodingKeys: String, CodingKey {
        case minutes, instances, pctPerHour = "pct_per_hour", maxThermal = "max_thermal"
        case batteryStart = "battery_start", batteryEnd = "battery_end"
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(minutes, forKey: .minutes); try c.encode(instances, forKey: .instances)
        try c.encode(pctPerHour, forKey: .pctPerHour); try c.encode(maxThermal, forKey: .maxThermal)
        try c.encodeOrNull(batteryStart, forKey: .batteryStart); try c.encodeOrNull(batteryEnd, forKey: .batteryEnd)
    }
}

struct Phase0LockedRow: Codable, Equatable {
    var routeChanges: Int
    var forcedSwaps: Int
    var resumedWithoutTap: Int
    var neededTap: Int

    enum CodingKeys: String, CodingKey {
        case routeChanges = "route_changes", forcedSwaps = "forced_swaps"
        case resumedWithoutTap = "resumed_without_tap", neededTap = "needed_tap"
    }
}

struct Phase0MicLevelRow: Codable, Equatable {
    var distanceM: Double
    var speechRmsDBFS: Double  // over VAD speech frames only
    var speechPeakDBFS: Double
    var speechS: Double?       // I-1 addition: seconds of speech the numbers come from

    enum CodingKeys: String, CodingKey {
        case distanceM = "distance_m", speechRmsDBFS = "speech_rms_dbfs", speechPeakDBFS = "speech_peak_dbfs"
        case speechS = "speech_s"
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(distanceM, forKey: .distanceM); try c.encode(speechRmsDBFS, forKey: .speechRmsDBFS)
        try c.encode(speechPeakDBFS, forKey: .speechPeakDBFS); try c.encodeOrNull(speechS, forKey: .speechS)
    }
}

struct Phase0WeSpeakerRow: Codable, Equatable {
    var units: String          // cpuAndGPU or cpuOnly (the units the timed calls used)
    var screen: String
    var selftestCos: Double?
    var ms1p5s: Double?
    var ms10s: Double?
    var firstCallMS: Double?
    var ok: Bool

    enum CodingKeys: String, CodingKey {
        case units, screen, selftestCos = "selftest_cos", ms1p5s = "ms_1p5s", ms10s = "ms_10s"
        case firstCallMS = "first_call_ms", ok
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(units, forKey: .units); try c.encode(screen, forKey: .screen)
        try c.encodeOrNull(selftestCos, forKey: .selftestCos); try c.encodeOrNull(ms1p5s, forKey: .ms1p5s)
        try c.encodeOrNull(ms10s, forKey: .ms10s); try c.encodeOrNull(firstCallMS, forKey: .firstCallMS)
        try c.encode(ok, forKey: .ok)
    }
}

struct Phase0AlarmRow: Codable, Equatable {
    var screen: String
    var seconds: Int
    var windows: Int           // sound classifier windows that came back
    var ok: Bool
}

// MARK: - The whole file

struct Phase0Report: Codable, Equatable {
    var date: String           // the day this Phase 0 file was started, YYYY-MM-DD
    var device: String         // hardware model, for example "iPhone17,1"
    var ios: String
    var appVersion: String     // "0.1 (1)"
    var engine: String         // stub or real
    var modelHash: String
    var buffers: [Phase0BufferRow] = []
    var tapTest: [Phase0TapRow] = []
    var latency: [Phase0LatencyRow] = []
    var motion: [Phase0MotionRow] = []
    var stem: [Phase0StemRow] = []
    var compute: [Phase0ComputeRow] = []
    var power: [Phase0PowerRow] = []
    var locked: [Phase0LockedRow] = []
    var micLevel: [Phase0MicLevelRow] = []
    var wespeaker: [Phase0WeSpeakerRow] = []
    var alarm: [Phase0AlarmRow] = []
    var notes: String = ""

    enum CodingKeys: String, CodingKey {
        case date, device, ios, appVersion = "app_version", engine, modelHash = "model_hash", buffers
        case tapTest = "tap_test", latency, motion, stem, compute, power, locked, micLevel = "mic_level"
        case wespeaker, alarm, notes
    }

    /// The documented top-level keys, in the 30-ios 15.3 order (the tests compare against this).
    static let topLevelKeys = ["date", "device", "ios", "app_version", "engine", "model_hash", "buffers",
                               "tap_test", "latency", "motion", "stem", "compute", "power", "locked",
                               "mic_level", "wespeaker", "alarm", "notes"]
}

extension KeyedEncodingContainer {
    /// Writes `value`, or JSON null when it is nil: the Phase 0 file always carries every key.
    mutating func encodeOrNull<T: Encodable>(_ value: T?, forKey key: Key) throws {
        if let value { try encode(value, forKey: key) } else { try encodeNil(forKey: key) }
    }
}

// MARK: - Store (Documents/phase0)

/// Reads and writes phase0-YYYY-MM-DD.json. Small file, written whole after every finished item
/// (atomically, so a crash never leaves half a file).
struct Phase0Store {
    let folder: URL

    /// Documents/phase0, the folder `make phase0` copies.
    static func standard() -> Phase0Store {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return Phase0Store(folder: docs.appendingPathComponent("phase0", isDirectory: true))
    }

    static func fileName(date: String) -> String { "phase0-\(date).json" }

    /// "2026-11-08" in the phone's time zone (en_US_POSIX, so the digits never change with the locale).
    static func dateString(_ d: Date, timeZone: TimeZone = .current) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = timeZone
        f.dateFormat = "yyyy-MM-dd"
        return f.string(from: d)
    }

    func url(date: String) -> URL { folder.appendingPathComponent(Self.fileName(date: date)) }

    func load(date: String) -> Phase0Report? {
        guard let data = try? Data(contentsOf: url(date: date)) else { return nil }
        return try? Self.decode(data)
    }

    /// The folder is kept out of iCloud backups like Documents/logs (SessionLog.prepareFolder):
    /// Phase 0 measurements stay on this iPhone until `make phase0` copies them.
    func save(_ r: Phase0Report) throws {
        try SessionLog.prepareFolder(folder)
        try Self.encode(r).write(to: url(date: r.date), options: .atomic)
    }

    /// Pretty printed with sorted keys, so two files diff cleanly.
    static func encode(_ r: Phase0Report) throws -> Data {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try e.encode(r)
    }

    static func decode(_ data: Data) throws -> Phase0Report {
        try JSONDecoder().decode(Phase0Report.self, from: data)
    }
}

// MARK: - What AudioController keeps for the runner

/// Plain counters and hooks on AudioController (`audio.phase0`). AudioController only adds to the
/// counters; the runner reads them at the start and the end of a run and subtracts.
struct Phase0Taps {
    var routeChanges = 0       // route change notifications while listening (handleRouteChange)
    var swaps = 0              // engine swaps that took over (any reason)
    var lastProbe: ProbeOutcome? = nil
    /// Each second's per-hop compute times, forwarded by safetySecond (the one drainTimings reader).
    var onTimings: (@MainActor ([UInt16]) -> Void)? = nil
}

/// One latency probe's raw result, success or not (AudioController.measureLatency fills it).
struct ProbeOutcome: Equatable {
    var ok: Bool
    var loopMS: Double?
    var spreadMS: Double?
    var psrDB: Double?
    var okSweeps: Int
    var method: String
    var why: String            // empty on success, else the engine's or validator's reason

    init(ok: Bool, raw: es_probe_result?, method: String, why: String) {
        self.ok = ok
        self.method = method
        self.why = why
        okSweeps = Int(raw?.n_ok ?? 0)
        // A weak or rejected probe still fills the struct, but its loop is not a measurement: keep
        // the loop only on success. The engine always fills spread_ms (contract 4.2), and a spread
        // over the route's limit is often why the app rejected the probe, so it is kept either way.
        // Non-finite numbers become nil (JSON has no NaN, and JSONEncoder would throw).
        func finite(_ x: Float?) -> Double? { x.flatMap { $0.isFinite ? Double($0) : nil } }
        loopMS = ok ? finite(raw?.loop_ms) : nil
        spreadMS = finite(raw?.spread_ms)
        psrDB = finite(raw?.psr_db)
    }
}
