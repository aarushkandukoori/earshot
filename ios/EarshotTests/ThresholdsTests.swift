// ThresholdsTests.swift - thresholds.json decoding and every boundary of 30-ios 15.1.

import XCTest
@testable import Earshot

final class ThresholdsTests: XCTestCase {
    func testDecodesTheContractExample() throws {
        let t = try TestSupport.contractThresholds()
        XCTAssertEqual(t.version, 1)
        XCTAssertEqual(t.target.speechS, [4, 5])
        XCTAssertEqual(t.target.sbrDB, [0, 6])
        XCTAssertEqual(t.target.oneVoice, [0.45, 0.60])
        XCTAssertEqual(t.target.ownCos, [0.50, 0.60])
        XCTAssertEqual(t.target.dupCos, 0.75)
        XCTAssertEqual(t.target.clipPct, [0.1, 1.0])
        XCTAssertEqual(t.own.speechS, 8)
        XCTAssertEqual(t.own.snrDB, [15, 20])
        XCTAssertEqual(t.ownWindowCos, 0.60)
        XCTAssertEqual(t.lockedCos, 0.0)
    }

    func testTheBundledFileDecodes() throws {
        // Resources/Models/thresholds.json from WP-ML (random_v1), bundled by `make models`.
        guard let url = Bundle.main.url(forResource: "thresholds", withExtension: "json") else {
            throw XCTSkip("thresholds.json not bundled (run `make models`)")
        }
        let t = try Thresholds.decode(Data(contentsOf: url))
        XCTAssertEqual(t.version, 1)
        XCTAssertEqual(t.modelHash.count, 64)
        XCTAssertEqual(t.target.speechS, [4, 5], "speech_s, dup_cos and clip_pct stay fixed (contract 6)")
        XCTAssertEqual(t.target.dupCos, 0.75)
        XCTAssertEqual(t.target.clipPct, [0.1, 1.0])
    }

    func testRejectsAnotherVersionAndBadPairs() {
        XCTAssertThrowsError(try Thresholds.decode(TestSupport.thresholds(version: 2)))
        let reversed = TestSupport.thresholdsJSON.replacingOccurrences(of: "\"sbr_db\": [0, 6]", with: "\"sbr_db\": [6, 0]")
        XCTAssertThrowsError(try Thresholds.decode(Data(reversed.utf8)))
        let short = TestSupport.thresholdsJSON.replacingOccurrences(of: "\"speech_s\": [4, 5]", with: "\"speech_s\": [4]")
        XCTAssertThrowsError(try Thresholds.decode(Data(short.utf8)))
        XCTAssertThrowsError(try Thresholds.decode(Data(#"{"version": 1}"#.utf8)), "missing fields")
    }

    func testTargetBoundaries() throws {
        let t = try TestSupport.contractThresholds()
        // speech 5.0 pass, 4.0 warn, 3.99 fail
        XCTAssertEqual(t.speech(5.0), .pass)
        XCTAssertEqual(t.speech(4.0), .warn)
        XCTAssertEqual(t.speech(3.99), .fail)
        // sbr 6 pass, 0 warn, -0.01 fail
        XCTAssertEqual(t.sbr(6), .pass)
        XCTAssertEqual(t.sbr(0), .warn)
        XCTAssertEqual(t.sbr(-0.01), .fail)
        // one voice 0.60 pass, 0.45 warn, 0.4499 fail
        XCTAssertEqual(t.oneVoice(0.60), .pass)
        XCTAssertEqual(t.oneVoice(0.45), .warn)
        XCTAssertEqual(t.oneVoice(0.4499), .fail)
        // own_cos 0.4999 pass, 0.50 warn, 0.60 fail
        XCTAssertEqual(t.ownCos(0.4999), .pass)
        XCTAssertEqual(t.ownCos(0.50), .warn)
        XCTAssertEqual(t.ownCos(0.60), .fail)
        // clip 0.099 pass, 0.1 warn, 1.0 fail
        XCTAssertEqual(t.clip(0.099), .pass)
        XCTAssertEqual(t.clip(0.1), .warn)
        XCTAssertEqual(t.clip(1.0), .fail)
        // dup 0.75 asks, just under does not
        XCTAssertTrue(t.asksDuplicate(0.75))
        XCTAssertFalse(t.asksDuplicate(0.7499))
    }

    func testOwnVoiceBoundaries() throws {
        let t = try TestSupport.contractThresholds()
        XCTAssertEqual(t.ownSNR(20), .pass)
        XCTAssertEqual(t.ownSNR(15), .warn)
        XCTAssertEqual(t.ownSNR(14.99), .fail)
        XCTAssertEqual(t.ownSpeech(8), .pass)
        XCTAssertEqual(t.ownSpeech(7.99), .fail, "own voice has no warn band")
    }
}
