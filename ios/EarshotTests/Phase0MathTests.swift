// Phase0MathTests.swift - the arithmetic behind the Phase 0 rows (Telemetry/Phase0Math.swift):
// histogram binning, worst 60 s of underruns, tap rule, motion gaps, battery drain, probe level
// pick, latency minute slots, locked resume tracking, speech level, alarm windows, labels.

import AVFoundation
import XCTest
@testable import Earshot

final class Phase0MathTests: XCTestCase {
    // MARK: Histogram binning

    func testBinEdges() {
        let e = Phase0Math.binEdges
        XCTAssertEqual(e.first, 0)
        XCTAssertEqual(e.count, 28)
        XCTAssertEqual(Array(e.prefix(3)), [0, 50, 100])
        XCTAssertEqual(e[20], 1000)
        XCTAssertEqual(e.last, 10_000)
        XCTAssertEqual(e, e.sorted(), "ascending")
    }

    func testBinIndexBoundaries() {
        XCTAssertEqual(Phase0Math.binIndex(0), 0)
        XCTAssertEqual(Phase0Math.binIndex(49), 0)
        XCTAssertEqual(Phase0Math.binIndex(50), 1, "lo is inside the bin, hi is not")
        XCTAssertEqual(Phase0Math.binIndex(999), 19)
        XCTAssertEqual(Phase0Math.binIndex(1000), 20)
        XCTAssertEqual(Phase0Math.binIndex(1249), 20)
        XCTAssertEqual(Phase0Math.binIndex(1250), 21)
        XCTAssertEqual(Phase0Math.binIndex(4000), 25)
        XCTAssertEqual(Phase0Math.binIndex(9999), 26)
        XCTAssertEqual(Phase0Math.binIndex(10_000), 27)
        XCTAssertEqual(Phase0Math.binIndex(65_535), 27)
    }

    func testBinsCountEveryHopAndChain() {
        var h = ComputeHistogram()
        h.add([0, 49, 50, 999, 1000, 4000, 9999, 10_000, 65_535])
        let bins = Phase0Math.bins(h)
        XCTAssertEqual(bins.count, 28, "every bin is listed, empty or not")
        XCTAssertEqual(bins.map(\.n).reduce(0, +), 9)
        XCTAssertEqual(bins[0].n, 2)
        XCTAssertEqual(bins[1].n, 1)
        XCTAssertEqual(bins[19].n, 1)
        XCTAssertEqual(bins[20].n, 1)
        XCTAssertEqual(bins[25].n, 1)
        XCTAssertEqual(bins[26].n, 1)
        XCTAssertEqual(bins[27].n, 2, "10,000 us and above share the open bin")
        XCTAssertNil(bins[27].hi)
        for i in 0..<(bins.count - 1) { XCTAssertEqual(bins[i].hi, bins[i + 1].lo, "bins touch") }
        XCTAssertEqual(Phase0Math.bins(ComputeHistogram()).map(\.n).reduce(0, +), 0)
    }

    /// The run histogram's percentiles (the compute row) come from the same 1 us histogram.
    func testRunPercentiles() {
        var h = ComputeHistogram()
        h.add([UInt16](repeating: 300, count: 998) + [1500, 2500])
        XCTAssertEqual(h.percentile(0.5), 300)
        XCTAssertEqual(h.percentile(0.99), 300)
        XCTAssertEqual(h.percentile(0.999), 1500)
        XCTAssertEqual(h.maxUS, 2500)
    }

    // MARK: Underruns, tap rule, battery

    func testWorstWindow() {
        XCTAssertEqual(Phase0Math.worstWindow(cumulative: []), 0)
        XCTAssertEqual(Phase0Math.worstWindow(cumulative: [0, 1, 1, 3]), 3, "a short run counts whole")
        XCTAssertEqual(Phase0Math.worstWindow(cumulative: [Int](repeating: 2, count: 60)), 2)
        // 2 in the first second, then 5 more between seconds 100 and 105, of a 600 s run
        var c: [Int] = []
        for k in 0..<600 { c.append(k < 100 ? 2 : (k < 105 ? 2 + (k - 99) : 7)) }
        XCTAssertEqual(Phase0Math.worstWindow(cumulative: c), 5)
        XCTAssertEqual(Phase0Math.worstWindow(cumulative: [3] + [Int](repeating: 3, count: 100)), 3,
                       "the first window counts what happened from the start")
    }

    func testTapRuleIsTwentyDecibels() {
        XCTAssertTrue(Phase0Math.tapPasses(phonePeakDB: -10, airpodPeakDB: -30))
        XCTAssertFalse(Phase0Math.tapPasses(phonePeakDB: -10, airpodPeakDB: -29.9))
        XCTAssertFalse(Phase0Math.tapPasses(phonePeakDB: -40, airpodPeakDB: -10))
    }

    func testBatteryPerHour() {
        XCTAssertEqual(Phase0Math.pctPerHour(startPct: 100, endPct: 95, minutes: 60), 5)
        XCTAssertEqual(Phase0Math.pctPerHour(startPct: 80, endPct: 70, minutes: 120), 5)
        XCTAssertNil(Phase0Math.pctPerHour(startPct: 80, endPct: 80, minutes: 0.5))
    }

    // MARK: Motion

    func testMotionGaps() {
        var g = MotionGaps()
        for k in 0..<26 { g.add(100 + Double(k) * 0.04) } // 25 Hz for 1 s
        g.add(101.5)                                       // a 0.5 s gap
        g.add(101.4)                                       // late sample: ignored
        g.add(101.54)
        XCTAssertEqual(g.count, 28)
        XCTAssertEqual(g.gapsOver200ms, 1)
        XCTAssertEqual(g.maxGapMS(end: nil), 500, accuracy: 0.01)
        XCTAssertEqual(g.maxGapMS(end: 103), 1460, accuracy: 0.01, "samples that stop early count as a gap")
        XCTAssertEqual(g.rateHz, 27 / 1.54, accuracy: 1e-9)
        XCTAssertEqual(MotionGaps().rateHz, 0)
    }

    // MARK: Probe level and latency slots

    private func probes(level: Int, ambient: Int, ok: Int, of n: Int) -> [Phase0LatencyRow] {
        (0..<n).map { i in
            Phase0LatencyRow(route: "airpods_pro2", method: i % 2 == 0 ? "in_ear" : "ear_detection_off", minute: 0,
                             ambientDB: ambient, probeDBFS: level, loopMS: i < ok ? 60 : nil, spreadMS: nil, psrDB: nil,
                             okSweeps: i < ok ? 5 : 1, reportedMS: 150, ok: i < ok)
        }
    }

    func testProbePickNeedsEightyPercentInBothRooms() {
        var rows = probes(level: -30, ambient: 45, ok: 5, of: 5) + probes(level: -30, ambient: 70, ok: 3, of: 5)
        XCTAssertNil(Phase0Math.probePick(rows), "-30 works in quiet only")
        rows += probes(level: -24, ambient: 45, ok: 5, of: 5) + probes(level: -24, ambient: 70, ok: 4, of: 5)
        XCTAssertEqual(Phase0Math.probePick(rows), -24)
        rows += probes(level: -18, ambient: 45, ok: 5, of: 5) + probes(level: -18, ambient: 70, ok: 5, of: 5)
        XCTAssertEqual(Phase0Math.probePick(rows), -24, "the lowest level that qualifies")
        let few = probes(level: -30, ambient: 45, ok: 4, of: 4) + probes(level: -30, ambient: 70, ok: 4, of: 4)
        XCTAssertNil(Phase0Math.probePick(few), "4 tries per room are not enough")
        let s = Phase0Math.probeSuccess(rows, level: -30, loud: true)
        XCTAssertEqual(s?.tries, 5)
        XCTAssertEqual(s?.rate ?? 0, 0.6, accuracy: 1e-9)
    }

    func testMinuteSlots() {
        XCTAssertEqual(Phase0Math.minuteSlot(0), 0)
        XCTAssertEqual(Phase0Math.minuteSlot(4), 0)
        XCTAssertNil(Phase0Math.minuteSlot(5))
        XCTAssertEqual(Phase0Math.minuteSlot(10), 10)
        XCTAssertEqual(Phase0Math.minuteSlot(19), 10)
        XCTAssertNil(Phase0Math.minuteSlot(25))
        XCTAssertEqual(Phase0Math.minuteSlot(30), 30)
        XCTAssertEqual(Phase0Math.minuteSlot(95), 30)
    }

    // MARK: Locked resume

    func testResumeTracker() {
        var t = ResumeTracker()
        t.sample(now: 0, changes: 0, routeAllowed: true, running: true, userPaused: false)
        // AirPods out (change 1), back in (change 2), audio running 2 s later: one event, resumed
        t.sample(now: 1, changes: 1, routeAllowed: false, running: false, userPaused: false)
        t.sample(now: 2, changes: 1, routeAllowed: false, running: false, userPaused: false)
        t.sample(now: 3, changes: 2, routeAllowed: true, running: false, userPaused: false)
        t.sample(now: 4, changes: 2, routeAllowed: true, running: false, userPaused: false)
        t.sample(now: 5, changes: 2, routeAllowed: true, running: true, userPaused: false)
        XCTAssertEqual(t.resumedWithoutTap, 1)
        XCTAssertEqual(t.neededTap, 0)
        // a swap that leaves audio stopped for 10 s on an allowed route: needed a tap
        t.sample(now: 10, changes: 3, routeAllowed: true, running: false, userPaused: false)
        t.sample(now: 15, changes: 3, routeAllowed: true, running: false, userPaused: false)
        XCTAssertEqual(t.neededTap, 0, "not yet")
        t.sample(now: 20, changes: 3, routeAllowed: true, running: false, userPaused: false)
        XCTAssertEqual(t.neededTap, 1)
        // AirPods out for a long time: no verdict while nothing is expected
        t.sample(now: 30, changes: 4, routeAllowed: false, running: false, userPaused: false)
        t.sample(now: 60, changes: 4, routeAllowed: false, running: false, userPaused: false)
        XCTAssertEqual(t.neededTap, 1)
        // the owner paused on purpose: the event is dropped
        t.sample(now: 61, changes: 4, routeAllowed: true, running: false, userPaused: true)
        t.sample(now: 90, changes: 4, routeAllowed: true, running: false, userPaused: false)
        XCTAssertEqual(t.resumedWithoutTap, 1)
        XCTAssertEqual(t.neededTap, 1)
    }

    // MARK: Speech level

    func testMicLevelUsesSpeechFramesOnly() throws {
        let n = 512 * 10
        var mask = [UInt8](repeating: 0, count: 10)
        for f in 3...5 { mask[f] = 1 }
        var pcm = [Float](repeating: 0.5, count: n) // loud non-speech everywhere
        for f in 3...5 { for i in ClipTools.frameSamples(f, n: n) { pcm[i] = (i % 2 == 0) ? 0.1 : -0.1 } }
        let level = try XCTUnwrap(Phase0Math.micLevel(pcm: pcm, mask: mask))
        XCTAssertEqual(level.rmsDB, -20, accuracy: 1e-4)
        XCTAssertEqual(level.peakDB, -20, accuracy: 1e-4)
        XCTAssertEqual(level.speechS, 3 * 0.032, accuracy: 1e-9)
        XCTAssertNil(Phase0Math.micLevel(pcm: pcm, mask: [UInt8](repeating: 0, count: 10)))
    }

    // MARK: Alarm windows

    func testAlarmWindows() {
        XCTAssertEqual(Phase0Math.expectedAlarmWindows(seconds: 60, overlap: 0.5), 39)
        XCTAssertEqual(Phase0Math.expectedAlarmWindows(seconds: 60, overlap: 0), 20)
        XCTAssertEqual(Phase0Math.expectedAlarmWindows(seconds: 2, overlap: 0.5), 0)
        XCTAssertFalse(Phase0Math.alarmOK(windows: 31, expected: 39))
        XCTAssertTrue(Phase0Math.alarmOK(windows: 32, expected: 39))
        XCTAssertFalse(Phase0Math.alarmOK(windows: 0, expected: 0))
    }

    // MARK: Labels and small helpers

    func testScreenLabel() {
        XCTAssertEqual(Phase0Math.screenLabel(activeSeconds: 0, inactiveSeconds: 0), "on")
        XCTAssertEqual(Phase0Math.screenLabel(activeSeconds: 10, inactiveSeconds: 90), "locked")
        XCTAssertEqual(Phase0Math.screenLabel(activeSeconds: 80, inactiveSeconds: 20), "on")
        XCTAssertEqual(Phase0Math.screenLabel(activeSeconds: 50, inactiveSeconds: 50), "mixed")
    }

    private func route(_ type: AVAudioSession.Port?) -> RouteInfo {
        RouteInfo(outputs: type.map { [Port(type: $0, uid: "uid-1", name: "x")] } ?? [], inputType: .builtInMic,
                  sampleRate: 48_000, ioBufferDuration: 64.0 / 48_000, inputLatency: 0, outputLatency: 0)
    }

    func testRouteLabel() {
        XCTAssertEqual(Phase0Math.routeLabel(route(.bluetoothA2DP), model: .pro1), "airpods_pro1")
        XCTAssertEqual(Phase0Math.routeLabel(route(.bluetoothA2DP), model: .pro2), "airpods_pro2")
        XCTAssertNil(Phase0Math.routeLabel(route(.bluetoothA2DP), model: nil), "unknown AirPods: ask the owner")
        XCTAssertEqual(Phase0Math.routeLabel(route(.builtInSpeaker), model: nil), "no_headphones")
        XCTAssertEqual(Phase0Math.routeLabel(route(.headphones), model: nil), "wired")
        XCTAssertEqual(Phase0Math.routeLabel(route(nil), model: nil), "no_headphones")
    }

    func testSmallHelpers() {
        XCTAssertEqual(Phase0Math.modeName("AVAudioSessionModeMeasurement"), "measurement")
        XCTAssertEqual(Phase0Math.modeName(AVAudioSession.Mode.measurement.rawValue), "measurement")
        XCTAssertEqual(Phase0Math.modeName("other"), "other")
        XCTAssertEqual(Phase0Math.ms(0.0025), 2.5)
        XCTAssertEqual(Phase0Math.round1(24.94), 24.9)
        XCTAssertTrue(Phase0Runner.comboParts("ca_on/spatial_off") == ("on", "off"))
        XCTAssertTrue(Phase0Runner.comboParts("ca_off/spatial_on") == ("off", "on"))
    }

    func testCounterTotalAcrossEngineSwap() {
        var c = Phase0Runner.CounterTotal(start: 5)
        c.add(7)
        XCTAssertEqual(c.total, 2)
        c.add(3) // a new engine counts from 0 again
        XCTAssertEqual(c.total, 5)
        c.add(3)
        XCTAssertEqual(c.total, 5)
    }

    func testNoiseClipIsQuietAndRepeatable() {
        let a = Phase0Runner.noiseClip(count: 16_000)
        XCTAssertEqual(a, Phase0Runner.noiseClip(count: 16_000))
        XCTAssertLessThanOrEqual(a.map { abs($0) }.max() ?? 1, 0.056)
    }
}
