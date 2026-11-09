// Phase0SchemaTests.swift - the Phase 0 file (30-ios 15.3): the spec example decodes, a full
// report survives a write and a read, every documented key is always written (null when not
// measured), the only extra keys are the listed I-1 additions, and the store writes
// Documents/phase0/phase0-YYYY-MM-DD.json.

import XCTest
@testable import Earshot

final class Phase0SchemaTests: XCTestCase {
    /// The 30-ios 15.3 example, copied from the spec (placeholders like "on|locked" included).
    static let specExample = """
    {"date": "2026-11-..", "device": "iPhone17,1", "ios": "26.1", "app_version": "0.1 (1)", "engine": "stub|real",
     "model_hash": "..",
     "buffers":   [{"route": "airpods_pro2", "mode": "measurement", "requested": 32, "granted": 64, "callback": 64,
                    "rate": 48000, "in_lat_ms": 0, "out_lat_ms": 0, "underruns_60s": 0}],
     "tap_test":  [{"route": "airpods_pro2", "phone_tap_peak_db": 0, "airpod_tap_peak_db": 0, "data_source": "Bottom",
                    "pass": true}],
     "latency":   [{"route": "airpods_pro1", "method": "in_ear|ear_detection_off", "minute": 0, "ambient_db": 45,
                    "probe_dbfs": -30, "loop_ms": 0,
                    "spread_ms": 0, "psr_db": 0, "ok_sweeps": 5, "reported_ms": 0}],
     "motion":    [{"route": "airpods_pro2", "screen": "on|locked", "minutes": 10, "rate_hz": 0, "gaps_over_200ms": 0,
                    "max_gap_ms": 0}],
     "stem":      [{"route": "airpods_pro2", "conv_awareness": "on|off", "spatial": "off|on", "screen": "locked",
                    "single": true, "double": true, "triple": true, "single_while_paused_rate0": true}],
     "compute":   [{"instances": 1, "minutes": 60, "screen": "locked", "p50_us": 0, "p99_us": 0, "p999_us": 0,
                    "max_us": 0, "underruns": 0, "workgroup": true}],
     "power":     [{"minutes": 120, "instances": 1, "pct_per_hour": 0, "max_thermal": "nominal"}],
     "locked":    [{"route_changes": 10, "forced_swaps": 1, "resumed_without_tap": 0, "needed_tap": 0}],
     "mic_level": [{"distance_m": 1.0, "speech_rms_dbfs": 0, "speech_peak_dbfs": 0}],
     "wespeaker": [{"units": "cpuAndGPU|cpuOnly", "screen": "on|locked", "selftest_cos": 0, "ms_1p5s": 0, "ms_10s": 0,
                    "first_call_ms": 0, "ok": true}],
     "alarm":     [{"screen": "locked", "seconds": 60, "windows": 0, "ok": true}],
     "notes": ""}
    """

    /// The I-1 additions per section (the only keys allowed beyond the spec example's).
    static let additions: [String: Set<String>] = [
        "buffers": ["seconds", "screen", "underruns"], "latency": ["ok"],
        "compute": ["frames", "hist_us", "malloc"], "power": ["battery_start", "battery_end"],
        "mic_level": ["speech_s"],
    ]

    /// One row of every kind, with some values not measured (nil) and every addition filled.
    static func fullReport() -> Phase0Report {
        var r = Phase0Report(date: "2026-11-08", device: "iPhone17,1", ios: "26.1", appVersion: "0.1 (1)",
                             engine: "stub", modelHash: String(repeating: "0", count: 64))
        r.buffers = [Phase0BufferRow(route: "airpods_pro2", mode: "measurement", requested: 32, granted: 64,
                                     callback: 64, rate: 48_000, inLatMS: 2.5, outLatMS: 160.25, underruns60s: 1,
                                     seconds: 60, screen: "on", underruns: 1)]
        r.tapTest = [Phase0TapRow(route: "airpods_pro2", phoneTapPeakDB: -8.5, airpodTapPeakDB: -41, dataSource: nil,
                                  pass: true)]
        r.latency = [Phase0LatencyRow(route: "airpods_pro1", method: "in_ear", minute: 10, ambientDB: 70, probeDBFS: -30,
                                      loopMS: nil, spreadMS: nil, psrDB: 7.5, okSweeps: 2, reportedMS: 180, ok: false),
                     Phase0LatencyRow(route: "airpods_pro1", method: "ear_detection_off", minute: 0, ambientDB: 45,
                                      probeDBFS: -30, loopMS: 92.4, spreadMS: 1.5, psrDB: 18, okSweeps: 5,
                                      reportedMS: 180, ok: true)]
        r.motion = [Phase0MotionRow(route: "airpods_pro2", screen: "locked", minutes: 10, rateHz: 24.9, gapsOver200ms: 2,
                                    maxGapMS: 410)]
        r.stem = [Phase0StemRow(route: "airpods_pro2", convAwareness: "off", spatial: "off", screen: "locked",
                                single: true, double: true, triple: false, singleWhilePausedRate0: nil),
                  Phase0StemRow(route: "airpods_pro2", convAwareness: "off", spatial: "off", screen: "locked",
                                single: nil, double: nil, triple: nil, singleWhilePausedRate0: true)]
        var h = ComputeHistogram()
        h.add([120, 130, 900, 12_000])
        r.compute = [Phase0ComputeRow(instances: 3, minutes: 30, screen: "locked", p50US: 130, p99US: 900, p999US: 900,
                                      maxUS: 12_000, underruns: 0, workgroup: true, frames: 1104,
                                      histUS: Phase0Math.bins(h),
                                      malloc: Phase0MallocResult(status: "armed", counts: MallocCounts(renderAllocs: 0,
                                                                 renderFrees: 0, workerAllocs: 1, workerFrees: 1)))]
        r.power = [Phase0PowerRow(minutes: 120, instances: 1, pctPerHour: 4.5, maxThermal: "fair", batteryStart: 90,
                                  batteryEnd: 81)]
        r.locked = [Phase0LockedRow(routeChanges: 10, forcedSwaps: 1, resumedWithoutTap: 9, neededTap: 1)]
        r.micLevel = [Phase0MicLevelRow(distanceM: 1, speechRmsDBFS: -31.2, speechPeakDBFS: -12.4, speechS: 9.3)]
        r.wespeaker = [Phase0WeSpeakerRow(units: "cpuOnly", screen: "locked", selftestCos: 0.99995, ms1p5s: 21.5,
                                          ms10s: 80.1, firstCallMS: 140.2, ok: true),
                       Phase0WeSpeakerRow(units: "cpuOnly", screen: "locked", selftestCos: nil, ms1p5s: nil, ms10s: nil,
                                          firstCallMS: nil, ok: false)]
        r.alarm = [Phase0AlarmRow(screen: "locked", seconds: 60, windows: 39, ok: true)]
        r.notes = "test run"
        return r
    }

    private func json(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testSpecExampleDecodes() throws {
        let r = try Phase0Store.decode(Data(Self.specExample.utf8))
        XCTAssertEqual(r.date, "2026-11-..")
        XCTAssertEqual(r.appVersion, "0.1 (1)")
        XCTAssertEqual(r.buffers.first?.granted, 64)
        XCTAssertEqual(r.buffers.first?.underruns60s, 0)
        XCTAssertNil(r.buffers.first?.seconds, "additions are optional when reading")
        XCTAssertEqual(r.tapTest.first?.dataSource, "Bottom")
        XCTAssertEqual(r.latency.first?.probeDBFS, -30)
        XCTAssertEqual(r.latency.first?.okSweeps, 5)
        XCTAssertEqual(r.motion.first?.minutes, 10)
        XCTAssertEqual(r.stem.first?.singleWhilePausedRate0, true)
        XCTAssertEqual(r.compute.first?.workgroup, true)
        XCTAssertNil(r.compute.first?.histUS)
        XCTAssertEqual(r.power.first?.maxThermal, "nominal")
        XCTAssertEqual(r.locked.first?.forcedSwaps, 1)
        XCTAssertEqual(r.micLevel.first?.distanceM, 1.0)
        XCTAssertEqual(r.wespeaker.first?.ok, true)
        XCTAssertEqual(r.alarm.first?.seconds, 60)
    }

    func testFullReportRoundTrip() throws {
        let r = Self.fullReport()
        let back = try Phase0Store.decode(try Phase0Store.encode(r))
        XCTAssertEqual(back, r)
    }

    /// Top-level keys are exactly the spec's; every row carries every key of the spec example's row
    /// for its section, plus only the listed additions.
    func testKeysMatchTheSpecExample() throws {
        let spec = try json(Data(Self.specExample.utf8))
        let ours = try json(try Phase0Store.encode(Self.fullReport()))
        XCTAssertEqual(Set(ours.keys), Set(spec.keys))
        XCTAssertEqual(Set(ours.keys), Set(Phase0Report.topLevelKeys))
        for (section, value) in spec {
            guard let specRows = value as? [[String: Any]], let specRow = specRows.first else { continue }
            let documented = Set(specRow.keys)
            let rows = try XCTUnwrap(ours[section] as? [[String: Any]], section)
            XCTAssertFalse(rows.isEmpty, "the full report has a \(section) row")
            for row in rows {
                let keys = Set(row.keys)
                XCTAssertTrue(documented.isSubset(of: keys), "\(section) misses \(documented.subtracting(keys))")
                let extra = keys.subtracting(documented)
                XCTAssertTrue(extra.isSubset(of: Self.additions[section] ?? []), "\(section) has extra \(extra)")
            }
        }
    }

    /// A value that was not measured is written as null, never left out.
    func testUnmeasuredValuesAreNull() throws {
        let ours = try json(try Phase0Store.encode(Self.fullReport()))
        let latency = try XCTUnwrap(ours["latency"] as? [[String: Any]])
        XCTAssertTrue(latency[0]["loop_ms"] is NSNull)
        XCTAssertTrue(latency[0]["spread_ms"] is NSNull)
        XCTAssertEqual(latency[0]["ok"] as? Bool, false)
        let tap = try XCTUnwrap(ours["tap_test"] as? [[String: Any]])
        XCTAssertTrue(tap[0]["data_source"] is NSNull)
        let stem = try XCTUnwrap(ours["stem"] as? [[String: Any]])
        XCTAssertTrue(stem[0]["single_while_paused_rate0"] is NSNull)
        XCTAssertTrue(stem[1]["single"] is NSNull)
        let wespeaker = try XCTUnwrap(ours["wespeaker"] as? [[String: Any]])
        XCTAssertTrue(wespeaker[1]["ms_10s"] is NSNull)
        let compute = try XCTUnwrap(ours["compute"] as? [[String: Any]])
        let bins = try XCTUnwrap(compute[0]["hist_us"] as? [[String: Any]])
        XCTAssertTrue(bins.last?["hi"] is NSNull, "the open bin has no upper end")
        let malloc = try XCTUnwrap(compute[0]["malloc"] as? [String: Any])
        XCTAssertEqual(malloc["status"] as? String, "armed")
        XCTAssertEqual((malloc["counts"] as? [String: Any])?["worker_allocs"] as? Int, 1)
    }

    func testStoreWritesDatedFileAndReadsItBack() throws {
        let folder = TestSupport.tempFolder().appendingPathComponent("phase0")
        let store = Phase0Store(folder: folder)
        let r = Self.fullReport()
        try store.save(r)
        let url = folder.appendingPathComponent("phase0-2026-11-08.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertEqual(store.url(date: "2026-11-08"), url)
        XCTAssertEqual(store.load(date: "2026-11-08"), r)
        XCTAssertNil(store.load(date: "2026-11-09"))
        XCTAssertTrue(PersonFiles.isExcludedFromBackup(folder), "Documents/phase0 stays out of iCloud backups")
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.contains("\n"), "pretty printed")
        XCTAssertLessThan(try XCTUnwrap(text.range(of: "\"alarm\"")).lowerBound,
                          try XCTUnwrap(text.range(of: "\"buffers\"")).lowerBound, "keys sorted")
    }

    func testDateStringIsYearMonthDay() {
        let d = Date(timeIntervalSince1970: 1_794_096_000) // 2026-11-08 00:00 UTC
        XCTAssertEqual(Phase0Store.dateString(d, timeZone: TimeZone(identifier: "UTC")!), "2026-11-08")
        XCTAssertEqual(Phase0Store.fileName(date: "2026-11-08"), "phase0-2026-11-08.json")
    }

    /// JSON has no NaN: a probe that returned NaN is stored as null and the file still writes.
    func testProbeOutcomeDropsNonFiniteNumbers() throws {
        var raw = es_probe_result()
        raw.loop_ms = .nan
        raw.spread_ms = .infinity
        raw.psr_db = 12
        raw.n_ok = 4
        let o = ProbeOutcome(ok: true, raw: raw, method: "in_ear", why: "")
        XCTAssertNil(o.loopMS)
        XCTAssertNil(o.spreadMS)
        XCTAssertEqual(o.psrDB, 12)
        XCTAssertEqual(o.okSweeps, 4)
        let failed = ProbeOutcome(ok: false, raw: raw, method: "in_ear", why: "weak")
        XCTAssertNil(failed.loopMS, "a failed probe never carries a loop")
        raw.spread_ms = 12.5
        let rejected = ProbeOutcome(ok: false, raw: raw, method: "in_ear", why: "spread")
        XCTAssertNil(rejected.loopMS)
        XCTAssertEqual(rejected.spreadMS, 12.5, "the spread that got the probe rejected is kept")
        var r = Self.fullReport()
        r.latency[0].loopMS = o.loopMS
        XCTAssertNoThrow(try Phase0Store.encode(r))
    }
}
