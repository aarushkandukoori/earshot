// AlarmListenerTests.swift - alarm passthrough (30-ios 12.2): the label list, the 2-window 0.6 rule,
// the 5 s timer, the params during passthrough, the listener with a fake classifier (trigger reaches
// the main actor, overlap follows the power level), and the real SoundAnalysis classifier's labels
// and window length.

import SoundAnalysis
import XCTest
@testable import Earshot

/// A classifier that does nothing with audio; tests push windows through the listener.
final class FakeAlarmClassifier: AlarmClassifier {
    var onWindow: (([String: Double]) -> Void)?
    func feed(_ samples: [Float]) {}
}

/// Records the overlap each classifier was made with (thread-safe: the factory runs on the queue).
final class OverlapLog: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []
    func add(_ v: Double) { lock.lock(); values.append(v); lock.unlock() }
    var all: [Double] { lock.lock(); defer { lock.unlock() }; return values }
}

final class AlarmRulesTests: XCTestCase {
    func testTheLabelsAreTheContractList() {
        XCTAssertEqual(AlarmLabels.all, ["smoke_detector", "siren", "police_siren", "ambulance_siren",
                                         "fire_engine_siren", "civil_defense_siren", "emergency_vehicle",
                                         "car_horn", "screaming", "glass_breaking", "baby_crying"])
    }

    func testTwoAlarmWindowsInARowTrigger() {
        var d = AlarmDetector()
        XCTAssertFalse(d.window(["smoke_detector": 0.9]), "one window is not enough")
        XCTAssertTrue(d.window(["smoke_detector": 0.8]))
        XCTAssertTrue(d.window(["smoke_detector": 0.8]), "keeps firing while it goes on (extends only)")
        XCTAssertFalse(d.window([:]), "a quiet window resets")
        XCTAssertFalse(d.window(["car_horn": 0.7]))
    }

    func testThresholdIs0_6() {
        var d = AlarmDetector()
        XCTAssertFalse(d.window(["siren": 0.59]))
        XCTAssertFalse(d.window(["siren": 0.59]), "0.59 never counts")
        XCTAssertFalse(d.window(["siren": 0.6]))
        XCTAssertTrue(d.window(["siren": 0.6]), "0.6 counts")
    }

    func testOtherLabelsAreIgnored() {
        var d = AlarmDetector()
        XCTAssertFalse(d.window(["speech": 0.99, "music": 0.95]))
        XCTAssertFalse(d.window(["speech": 0.99, "dog_bark": 0.95]))
    }

    func testSirenFamilyLabelsCountTogether() {
        var d = AlarmDetector()
        XCTAssertFalse(d.window(["siren": 0.7]))
        XCTAssertTrue(d.window(["police_siren": 0.8]), "any alarm label in each window")
    }

    func testFiveSecondTimer() {
        var p = AlarmPassthrough()
        XCTAssertFalse(p.isActive)
        XCTAssertEqual(p.tick(now: 0), .none)
        XCTAssertEqual(p.trigger(now: 10), .started)
        XCTAssertEqual(p.trigger(now: 13), .extended)
        XCTAssertEqual(p.tick(now: 17.9), .none)
        XCTAssertEqual(p.tick(now: 18), .ended, "5 s after the last trigger")
        XCTAssertFalse(p.isActive)
        XCTAssertEqual(p.cancel(), .none)
        _ = p.trigger(now: 20)
        XCTAssertEqual(p.cancel(), .ended)
    }

    func testParamsDuringPassthroughUseMinusSixDB() {
        var base = EngineParams()
        base.slotGain = [0, 1, 0]
        let on = AlarmPassthrough.applied(to: base)
        XCTAssertTrue(on.bypass)
        XCTAssertEqual(on.outGainDB, -6)
        XCTAssertEqual(on.slotGain, base.slotGain)
        XCTAssertEqual(on.voiceDB, base.voiceDB)
        base.outGainDB = -30
        XCTAssertEqual(AlarmPassthrough.applied(to: base).outGainDB, -6, "contract 5.3: -6 dB even under a quiet trim")
    }

    /// The real built-in classifier knows every label and takes 3 s windows (skipped if the
    /// simulator cannot load it).
    func testSoundAnalysisKnowsEveryLabel() throws {
        guard let request = try? SNClassifySoundRequest(classifierIdentifier: .version1) else {
            throw XCTSkip("SoundAnalysis version 1 classifier not available on this host")
        }
        let known = Set(request.knownClassifications)
        for label in AlarmLabels.all { XCTAssertTrue(known.contains(label), "missing label \(label)") }
        let real = try SoundAnalysisClassifier(overlap: 0.5)
        XCTAssertEqual(real.windowSeconds, 3, accuracy: 0.01, "3 s windows")
        real.feed([Float](repeating: 0, count: 4_000)) // must not crash on a 250 ms chunk
    }
}

@MainActor
final class AlarmListenerTests: XCTestCase {
    func testATriggerReachesTheMainActor() async throws {
        let overlaps = OverlapLog()
        let listener = AlarmListener(makeClassifier: { overlap in
            overlaps.add(overlap)
            return FakeAlarmClassifier()
        })
        let fired = expectation(description: "trigger")
        listener.onTrigger = { fired.fulfill() }
        let model = try TestSupport.model()
        let engine = try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 64, instances: 3, slack16: 43)
        listener.start(engine: engine, level: .normal)
        XCTAssertTrue(listener.running)
        listener.injectWindowForTests(["glass_breaking": 0.9])
        listener.injectWindowForTests(["glass_breaking": 0.9])
        await fulfillment(of: [fired], timeout: 3)

        listener.start(engine: engine, level: .saving) // a power change restarts with no overlap
        listener.refreshStatus()
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertEqual(overlaps.all, [0.5, 0])
        XCTAssertGreaterThanOrEqual(listener.windows, 2)
        listener.stop()
        XCTAssertFalse(listener.running)
    }

    func testOneWindowDoesNotTrigger() async throws {
        let listener = AlarmListener(makeClassifier: { _ in FakeAlarmClassifier() })
        var count = 0
        listener.onTrigger = { count += 1 }
        let model = try TestSupport.model()
        let engine = try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 64, instances: 3, slack16: 43)
        listener.start(engine: engine, level: .normal)
        listener.injectWindowForTests(["screaming": 0.95])
        listener.injectWindowForTests(["speech": 0.95])
        listener.injectWindowForTests(["screaming": 0.95])
        try await Task.sleep(nanoseconds: 400_000_000)
        XCTAssertEqual(count, 0, "never two alarm windows in a row")
        listener.stop()
    }
}
