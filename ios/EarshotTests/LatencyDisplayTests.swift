// LatencyDisplayTests.swift - the delay strings and probe checks of 30-ios 7.

import AVFoundation
import XCTest
@testable import Earshot

final class LatencyDisplayTests: XCTestCase {
    /// A Bluetooth route with plausible iOS-reported numbers.
    private func airPodsRoute() -> RouteInfo {
        RouteInfo(outputs: [Port(type: .bluetoothA2DP, uid: "AA:BB-tacl", name: "AirPods Pro")],
                  inputType: .builtInMic, sampleRate: 48_000, ioBufferDuration: 64.0 / 48_000.0,
                  inputLatency: 0.005, outputLatency: 0.040)
    }

    func testMeasuredStringAddsUp() {
        // The contract example: E 10.95 + loop 48.2 measured 3 minutes ago.
        let text = LatencyMeter.measuredText(earshotMS: 10.95, loopMS: 48.2, ageSeconds: 180)
        XCTAssertEqual(text, "Delay 59 ms (measured 3 min ago) - Earshot 11 + route 48")
    }

    func testAgeText() {
        XCTAssertEqual(LatencyMeter.ageText(seconds: 0), "just now")
        XCTAssertEqual(LatencyMeter.ageText(seconds: 59.9), "just now")
        XCTAssertEqual(LatencyMeter.ageText(seconds: 60), "1 min ago")
        XCTAssertEqual(LatencyMeter.ageText(seconds: 3_599), "59 min ago")
        XCTAssertEqual(LatencyMeter.ageText(seconds: 3_600), "1 h ago")
        XCTAssertEqual(LatencyMeter.ageText(seconds: 7_300), "2 h ago")
    }

    func testEstimateString() {
        // 1000 x (0.005 + 0.040 + 2 x 0.001333) + 10.95 = 58.6 -> 59
        let d = LatencyMeter.display(route: airPodsRoute(), earshotMS: 10.95, measurement: nil, now: Date())
        XCTAssertEqual(d.text, "about 59 ms (estimate)")
        XCTAssertNil(d.measuredMS)
        XCTAssertEqual(d.reportedMS, 59)
    }

    func testUnknownBeforeAnyRoute() {
        let d = LatencyMeter.display(route: .none, earshotMS: 0, measurement: nil, now: Date())
        XCTAssertEqual(d, .unknown)
    }

    func testBluetoothLoopUnder20msIsRejected() {
        let r = ProbeResult(loopMS: 12, spreadMS: 1, psrDB: 20, okSweeps: 5)
        XCTAssertEqual(LatencyMeter.validate(r, isBluetooth: true), .bluetoothTooShort(12))
        XCTAssertNil(LatencyMeter.validate(r, isBluetooth: false), "a 12 ms wired loop is fine")
    }

    func testSpreadLimits() {
        let wideBT = ProbeResult(loopMS: 48, spreadMS: 10.5, psrDB: 20, okSweeps: 5)
        let okBT = ProbeResult(loopMS: 48, spreadMS: 9.9, psrDB: 20, okSweeps: 5)
        XCTAssertEqual(LatencyMeter.validate(wideBT, isBluetooth: true), .spreadTooLarge(10.5))
        XCTAssertNil(LatencyMeter.validate(okBT, isBluetooth: true))
        let wideWired = ProbeResult(loopMS: 8, spreadMS: 2.5, psrDB: 20, okSweeps: 5)
        let okWired = ProbeResult(loopMS: 8, spreadMS: 1.5, psrDB: 20, okSweeps: 5)
        XCTAssertEqual(LatencyMeter.validate(wideWired, isBluetooth: false), .spreadTooLarge(2.5))
        XCTAssertNil(LatencyMeter.validate(okWired, isBluetooth: false))
    }

    func testLastValidResultIsKeptWithItsAge() throws {
        let suite = "earshot.tests.latency.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = LatencyStore(defaults: defaults)
        let route = airPodsRoute()
        let measuredAt = Date(timeIntervalSince1970: 1_000_000)
        store.save(LatencyMeasurement(loopMS: 48.2, spreadMS: 1, psrDB: 20, method: "in_ear", date: measuredAt),
                   forPort: route.primaryOutputUID)
        // A later probe fails: nothing is saved, so the old result is still shown with its age.
        let later = measuredAt.addingTimeInterval(600)
        let d = LatencyMeter.display(route: route, earshotMS: 10.95,
                                     measurement: store.last(forPort: route.primaryOutputUID), now: later)
        XCTAssertEqual(d.text, "Delay 59 ms (measured 10 min ago) - Earshot 11 + route 48")
        XCTAssertEqual(d.measuredMS, 59)
        XCTAssertNil(store.last(forPort: "another-port"))
    }
}

// MARK: - I-3: honesty guards, method B and the measure prompts (30-ios 7, D03, D49)

final class LatencyPromptTests: XCTestCase {
    private func route(_ type: AVAudioSession.Port, uid: String, input: Double = 0.005, output: Double = 0.040)
        -> RouteInfo {
        RouteInfo(outputs: [Port(type: type, uid: uid, name: "out")], inputType: .builtInMic, sampleRate: 48_000,
                  ioBufferDuration: 64.0 / 48_000.0, inputLatency: input, outputLatency: output)
    }

    private let t0 = Date(timeIntervalSince1970: 2_000_000)

    func testBluetoothNeverShowsATotalUnder20ms() {
        let pods = route(.bluetoothA2DP, uid: "pods", input: 0.001, output: 0.002)
        // A stored 12 ms loop (an old or edited value) is ignored on Bluetooth ...
        let bad = LatencyMeasurement(loopMS: 12, spreadMS: 1, psrDB: 20, method: "in_ear", date: t0)
        let d = LatencyMeter.display(route: pods, earshotMS: 5, measurement: bad, now: t0)
        XCTAssertNil(d.measuredMS)
        // ... and the estimate 1000 x (0.001 + 0.002 + 0.00267) + 5 = 10.7 -> 11 ms is not shown as a number.
        XCTAssertEqual(d.reportedMS, 11, "still logged as latency_reported_ms")
        XCTAssertEqual(d.text, Copy.delayUnknown)
        XCTAssertFalse(d.text.contains("ms"))
        // Wired: the same numbers are an honest estimate and are shown.
        let wired = route(.headphones, uid: "wired", input: 0.001, output: 0.002)
        XCTAssertEqual(LatencyMeter.display(route: wired, earshotMS: 5, measurement: nil, now: t0).text,
                       "about 11 ms (estimate)")
    }

    func testValidBluetoothMeasurementStillShows() {
        let pods = route(.bluetoothA2DP, uid: "pods")
        let ok = LatencyMeasurement(loopMS: 20.4, spreadMS: 1, psrDB: 20, method: "ear_detection_off", date: t0)
        let d = LatencyMeter.display(route: pods, earshotMS: 10.95, measurement: ok, now: t0.addingTimeInterval(30))
        XCTAssertEqual(d.text, "Delay 31 ms (measured just now) - Earshot 11 + route 20")
    }

    /// NaN passes every "<" check (comparisons with NaN are false), and JSONEncoder cannot store it,
    /// so impossible values are rejected first, on every route.
    func testImpossibleProbeValuesAreRejected() {
        let cases: [(Float, Float)] = [(.nan, 1), (40, .nan), (.infinity, 1), (0, 0), (-5, 0), (40, -1)]
        for (loop, spread) in cases {
            let r = ProbeResult(loopMS: loop, spreadMS: spread, psrDB: 20, okSweeps: 5)
            XCTAssertEqual(LatencyMeter.validate(r, isBluetooth: true), .impossible, "BT loop \(loop) spread \(spread)")
            XCTAssertEqual(LatencyMeter.validate(r, isBluetooth: false), .impossible, "wired loop \(loop) spread \(spread)")
        }
        XCTAssertNil(LatencyMeter.validate(ProbeResult(loopMS: 3, spreadMS: 0, psrDB: 20, okSweeps: 5),
                                           isBluetooth: false), "a short wired loop with no spread is fine")
    }

    func testAnImpossibleStoredMeasurementIsNotShown() {
        let wired = route(.headphones, uid: "wired")
        for loop in [-5.0, 0, .nan] {
            let m = LatencyMeasurement(loopMS: loop, spreadMS: 0, psrDB: 20, method: "in_ear", date: t0)
            let d = LatencyMeter.display(route: wired, earshotMS: 11, measurement: m, now: t0)
            XCTAssertNil(d.measuredMS, "loop \(loop)")
            XCTAssertTrue(d.text.hasPrefix("about "), "falls back to the estimate: \(d.text)")
        }
    }

    func testProbeMethodNames() {
        XCTAssertEqual(ProbeMethod.inEar.rawValue, "in_ear")
        XCTAssertEqual(ProbeMethod.earDetectionOff.rawValue, "ear_detection_off")
    }

    func testConnectPromptsOnceThenRechecksAt10And30Minutes() {
        var p = LatencyPrompts()
        p.routeChanged(route(.bluetoothA2DP, uid: "pods"), now: t0)
        XCTAssertEqual(p.pending, .measureAtConnect)
        p.dismiss()
        p.routeChanged(route(.bluetoothA2DP, uid: "pods"), now: t0.addingTimeInterval(5))
        XCTAssertNil(p.pending, "the same port again is not a new connect")
        p.tick(now: t0.addingTimeInterval(9 * 60 + 59))
        XCTAssertNil(p.pending)
        p.tick(now: t0.addingTimeInterval(10 * 60))
        XCTAssertEqual(p.pending, .recheck(minutes: 10))
        p.dismiss()
        p.tick(now: t0.addingTimeInterval(20 * 60))
        XCTAssertNil(p.pending, "each re-check is offered once")
        p.tick(now: t0.addingTimeInterval(30 * 60))
        XCTAssertEqual(p.pending, .recheck(minutes: 30))
        p.dismiss()
        p.tick(now: t0.addingTimeInterval(90 * 60))
        XCTAssertNil(p.pending, "nothing after 30 min")
    }

    func testReconnectAfterABadRoutePromptsAgain() {
        var p = LatencyPrompts()
        p.routeChanged(route(.bluetoothA2DP, uid: "pods"), now: t0)
        p.dismiss()
        p.routeChanged(route(.builtInSpeaker, uid: "speaker"), now: t0.addingTimeInterval(60))
        XCTAssertNil(p.pending)
        XCTAssertEqual(p.connectedPort, "", "a bad route is a disconnect")
        p.routeChanged(route(.bluetoothA2DP, uid: "pods"), now: t0.addingTimeInterval(120))
        XCTAssertEqual(p.pending, .measureAtConnect, "AirPods back in: a new connect")
        // When both re-checks are due at once (for example after a long background stretch), only
        // the later banner shows.
        p.dismiss()
        p.tick(now: t0.addingTimeInterval(120 + 31 * 60))
        XCTAssertEqual(p.pending, .recheck(minutes: 30))
    }

    func testWiredRoutesMeasureOncePerConnect() {
        var p = LatencyPrompts()
        p.routeChanged(route(.headphones, uid: "wired"), now: t0)
        XCTAssertEqual(p.pending, .measureAtConnect)
        p.dismiss()
        p.tick(now: t0.addingTimeInterval(45 * 60))
        XCTAssertNil(p.pending, "no re-check banners on wired routes")
    }
}
