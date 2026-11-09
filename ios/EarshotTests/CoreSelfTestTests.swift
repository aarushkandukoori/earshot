// CoreSelfTestTests.swift - ContainerFile (contract 3.1) and the bundled self-test files (30-ios 15.1).

import XCTest
@testable import Earshot

final class CoreSelfTestTests: XCTestCase {
    func testBuiltContainerRoundTrip() throws {
        let data = TestSupport.container(meta: ["kind": "x", "tolerances": ["cosine_min": 0.5]],
                                         arrays: [("a", [1, 2, 3]), ("b", [-0.5])])
        let c = try ContainerFile(data: data, expectMagic: "EST1")
        XCTAssertEqual(try c.floats("a"), [1, 2, 3])
        XCTAssertEqual(try c.floats("b"), [-0.5])
        XCTAssertEqual(c.metaNumber("tolerances.cosine_min"), 0.5)
        XCTAssertEqual(c.metaString("kind"), "x")
        XCTAssertThrowsError(try c.floats("missing"))
        XCTAssertThrowsError(try c.uint8s("a"), "dtype is checked")
    }

    func testRejectsCorruptedHashMagicAndOffsets() {
        var data = TestSupport.container(meta: [:], arrays: [("a", [1, 2, 3])])
        data[data.count - 1] ^= 0x01 // one flipped bit in the arrays
        XCTAssertThrowsError(try ContainerFile(data: data)) { error in
            XCTAssertTrue("\(error)".contains("data_sha256"), "\(error)")
        }
        let good = TestSupport.container(meta: [:], arrays: [("a", [1])])
        XCTAssertThrowsError(try ContainerFile(data: good, expectMagic: "ESW1"), "wrong magic for the caller")
        var badMagic = good
        badMagic[0] = UInt8(ascii: "X")
        XCTAssertThrowsError(try ContainerFile(data: badMagic))
        XCTAssertThrowsError(try ContainerFile(data: TestSupport.container(meta: [:], arrays: [("a", [1])], misalign: true)),
                             "offsets must be 64-aligned")
        XCTAssertThrowsError(try ContainerFile(data: Data([1, 2, 3])), "shorter than the preamble")
    }

    func testBundledWeSpeakerSelfTestFile() throws {
        guard let url = Bundle.main.url(forResource: "wespk_selftest", withExtension: "est") else {
            throw XCTSkip("wespk_selftest.est not bundled (run `make models`)")
        }
        let c = try ContainerFile(data: Data(contentsOf: url), expectMagic: "EST1")
        XCTAssertEqual(c.metaString("kind"), "wespeaker-selftest")
        XCTAssertEqual(c.metaNumber("tolerances.cosine_min"), 0.999)
        XCTAssertEqual(try c.floats("clip").count, 80_000, "5 s at 16 kHz")
        let e = try c.floats("embedding")
        XCTAssertEqual(e.count, 256)
        XCTAssertEqual(e.reduce(0) { $0 + Double($1) * Double($1) }.squareRoot(), 1, accuracy: 1e-4, "unit length")
    }

    func testEngineSelfTestWithBundledVector() throws {
        // Meaningful only with ENGINE=real; the stub always passes with max error 0 (30-ios 15.1).
        guard let url = Bundle.main.url(forResource: "testvec_short", withExtension: "est"),
              let weights = Bundle.main.url(forResource: "earshot", withExtension: "esw") else {
            throw XCTSkip("model files not bundled (run `make models`)")
        }
        let model = try EarshotModel(weights: Data(contentsOf: weights))
        let maxErr = try model.selfTest(testVector: Data(contentsOf: url))
        XCTAssertLessThanOrEqual(maxErr, 2e-4)
        let c = try ContainerFile(data: Data(contentsOf: url), expectMagic: "EST1")
        XCTAssertEqual(c.metaString("weights_sha256")?.count, 64, "meta.weights_sha256 is the model_hash")
    }
}
