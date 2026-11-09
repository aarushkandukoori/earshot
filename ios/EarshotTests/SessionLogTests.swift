// SessionLogTests.swift - the per-minute session log (contract 5.2, 30-ios 14): exactly the contract
// keys (nulls written, nothing extra), value formats, compute percentiles, per-minute counter deltas
// across an engine swap, means and worst values, file names, the 30-day purge and real appends.

import XCTest
@testable import Earshot

final class SessionLogTests: XCTestCase {
    /// Contract 5.2, in order, with compute_us_p50/p99/max written out.
    static let contractKeys = ["t", "route", "io_rate", "io_frames", "slack16", "mode", "n_instances",
                               "compute_us_p50", "compute_us_p99", "compute_us_max", "underruns", "nan_resets",
                               "howl", "thermal", "battery", "low_power", "out_rms_db", "limiter_gr_db",
                               "p_target_mean", "p_self_mean", "latency_measured_ms", "latency_reported_ms",
                               "app_version", "model_hash"]

    private func sample(battery: Int? = nil, measured: Int? = nil) -> SessionLogRecord {
        SessionLogRecord(t: Date(timeIntervalSince1970: 1_794_000_123), route: "BluetoothA2DPOutput", ioRate: 48_000,
                         ioFrames: 64, slack16: 43, mode: "focus", nInstances: 2, computeUsP50: 310,
                         computeUsP99: 520, computeUsMax: 900, underruns: 0, nanResets: 0, howl: false,
                         thermal: "fair", battery: battery, lowPower: false, outRmsDB: -31.26, limiterGrDB: 0,
                         pTargetMean: [0.1, 0.72, 0], pSelfMean: 0.05, latencyMeasuredMS: measured,
                         latencyReportedMS: 62, appVersion: "0.1 (1)", modelHash: String(repeating: "a", count: 64))
    }

    private func decode(_ line: Data) throws -> [String: Any] {
        try XCTUnwrap(try JSONSerialization.jsonObject(with: line) as? [String: Any])
    }

    func testEveryLineHasExactlyTheContractKeys() throws {
        XCTAssertEqual(SessionLogRecord.CodingKeys.allCases.map { $0.rawValue }, Self.contractKeys)
        for record in [sample(), sample(battery: 87, measured: 59)] {
            let line = try SessionLog.encodeLine(record)
            XCTAssertEqual(line.last, 0x0A, "one line, newline at the end")
            XCTAssertEqual(line.filter { $0 == 0x0A }.count, 1)
            let obj = try decode(line)
            XCTAssertEqual(Set(obj.keys), Set(Self.contractKeys), "no key missing, none extra")
        }
    }

    func testMissingValuesAreNullNotLeftOut() throws {
        let obj = try decode(try SessionLog.encodeLine(sample()))
        XCTAssertTrue(obj["battery"] is NSNull)
        XCTAssertTrue(obj["latency_measured_ms"] is NSNull)
        let full = try decode(try SessionLog.encodeLine(sample(battery: 87, measured: 59)))
        XCTAssertEqual(full["battery"] as? Int, 87)
        XCTAssertEqual(full["latency_measured_ms"] as? Int, 59)
    }

    func testValueFormats() throws {
        let obj = try decode(try SessionLog.encodeLine(sample()))
        XCTAssertEqual(obj["t"] as? String, "2026-11-06T21:22:03Z", "ISO 8601 in UTC")
        XCTAssertEqual(obj["p_target_mean"] as? [Double], [0.1, 0.7, 0], "3 numbers, one decimal")
        XCTAssertEqual(obj["out_rms_db"] as? Double, -31.3)
        XCTAssertEqual(obj["howl"] as? Bool, false)
        XCTAssertEqual(obj["model_hash"] as? String, String(repeating: "a", count: 64))
        var bad = sample()
        bad.outRmsDB = .nan
        bad.pSelfMean = .infinity
        let fixed = try decode(try SessionLog.encodeLine(bad)) // JSON has no NaN: never crash the log
        XCTAssertEqual(fixed["out_rms_db"] as? Double, 0)
        XCTAssertEqual(fixed["p_self_mean"] as? Double, 0)
    }

    func testNoAudioAndNoNames() throws {
        // The route is a port type and nothing in the record can carry a person's name or audio.
        let text = String(decoding: try SessionLog.encodeLine(sample()), as: UTF8.self)
        XCTAssertFalse(text.contains("AirPods Pro"), text)
        XCTAssertLessThan(text.utf8.count, 1_000, "a few hundred bytes per minute")
    }

    func testComputePercentiles() {
        var h = ComputeHistogram()
        XCTAssertEqual(h.percentile(0.99), 0, "empty")
        h.add((1...1000).map { UInt16($0) })
        XCTAssertEqual(h.percentile(0.5), 500)
        XCTAssertEqual(h.percentile(0.99), 990)
        XCTAssertEqual(h.percentile(0.999), 999)
        XCTAssertEqual(h.maxUS, 1000)
        h.add([20_000]) // slower than the 1 us bins: the overflow bin reports the true maximum
        XCTAssertEqual(h.percentile(1.0), 20_000)
        XCTAssertEqual(h.total, 1001)
    }

    func testCounterDeltasSurviveAnEngineSwap() {
        var m = MinuteTelemetry()
        for u in [UInt64(0), 2, 5] {
            var s = EngineStats()
            s.underruns = u
            m.addPoll(s)
        }
        XCTAssertEqual(m.underruns, 5)
        for u in [UInt64(1), 3] { // a new engine: its counter starts again at 0
            var s = EngineStats()
            s.underruns = u
            m.addPoll(s)
        }
        XCTAssertEqual(m.underruns, 8, "5 from the old engine + 3 from the new one")
        m.nextMinute()
        XCTAssertEqual(m.underruns, 0)
        var s = EngineStats()
        s.underruns = 4
        m.addPoll(s)
        XCTAssertEqual(m.underruns, 1, "the next minute counts only the new underrun")
    }

    /// With the engine identity, a new engine counts all of its own underruns, even when its count
    /// already reaches the old engine's last value (I-4 review: old 5, new 6 is 6, not 1).
    func testANewEngineCountsAllOfItsUnderruns() {
        let oldEngine = NSObject(), newEngine = NSObject()
        var m = MinuteTelemetry()
        var s = EngineStats()
        s.underruns = 5
        m.addPoll(s, engine: ObjectIdentifier(oldEngine))
        XCTAssertEqual(m.underruns, 5)
        s.underruns = 6
        m.addPoll(s, engine: ObjectIdentifier(newEngine))
        XCTAssertEqual(m.underruns, 11, "5 from the old engine + 6 from the new one")
        s.underruns = 7
        m.addPoll(s, engine: ObjectIdentifier(newEngine))
        XCTAssertEqual(m.underruns, 12)
        m.nextMinute()
        s.underruns = 8
        m.addPoll(s, engine: ObjectIdentifier(newEngine))
        XCTAssertEqual(m.underruns, 1, "the engine carries over into the next minute")
    }

    func testMeansAndWorstValues() {
        var m = MinuteTelemetry()
        var s = EngineStats()
        s.pTarget = [0, 1, 0]
        s.pSelf = 0.5
        s.outRmsDB = -30
        s.limiterGrDB = 2
        m.addPoll(s)
        s.pTarget = [0, 0, 0]
        s.pSelf = 0
        s.outRmsDB = -40
        s.limiterGrDB = 0.5
        s.howl = true
        m.addPoll(s)
        XCTAssertEqual(m.pTargetMeans, [0, 0.5, 0])
        XCTAssertEqual(m.pSelfMean, 0.25)
        XCTAssertEqual(m.outRmsMean, -35)
        XCTAssertEqual(m.limiterMax, 2, "the largest gain reduction")
        XCTAssertTrue(m.howl)
        m.addSecond(timings: [100, 200], thermal: .fair, lowPower: false)
        m.addSecond(timings: [300], thermal: .serious, lowPower: true)
        m.addSecond(timings: [], thermal: .nominal, lowPower: false)
        XCTAssertEqual(MinuteTelemetry.thermalText(m.worstThermal), "serious", "the hottest in the minute")
        XCTAssertTrue(m.lowPower)
        XCTAssertEqual(m.histogram.total, 3)
        XCTAssertEqual(m.histogram.maxUS, 300)
    }

    func testFileNamesAndThe30DayPurge() {
        let utc = TimeZone(identifier: "UTC")!
        let now = Date(timeIntervalSince1970: 1_794_052_800) // 2026-11-07 12:00 UTC
        XCTAssertEqual(SessionLog.fileName(for: now, timeZone: utc), "2026-11-07.jsonl")
        let names = ["2026-10-01.jsonl", "2026-10-07.jsonl", "2026-10-08.jsonl", "2026-11-07.jsonl", "notes.txt",
                     "bad.jsonl"]
        XCTAssertEqual(SessionLog.namesToDelete(names, now: now, timeZone: utc),
                       ["2026-10-01.jsonl", "2026-10-07.jsonl"], "whole days older than 30 days; other files kept")
    }

    func testAppendWritesOneLinePerRecord() throws {
        let folder = TestSupport.tempFolder().appendingPathComponent("logs")
        defer { try? FileManager.default.removeItem(at: folder.deletingLastPathComponent()) }
        let log = SessionLog(folder: folder)
        let written = expectation(description: "two appends")
        written.expectedFulfillmentCount = 2
        log.append(sample()) { ok in XCTAssertTrue(ok); written.fulfill() }
        log.append(sample(battery: 50)) { ok in XCTAssertTrue(ok); written.fulfill() }
        wait(for: [written], timeout: 5)
        let file = folder.appendingPathComponent(SessionLog.fileName(for: sample().t))
        let text = try String(contentsOf: file, encoding: .utf8)
        let lines = text.split(separator: "\n")
        XCTAssertEqual(lines.count, 2)
        for line in lines {
            let obj = try decode(Data(line.utf8))
            XCTAssertEqual(Set(obj.keys), Set(Self.contractKeys))
        }
        let backup = try folder.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup
        XCTAssertEqual(backup, true, "logs never leave the phone through a backup")
    }
}
