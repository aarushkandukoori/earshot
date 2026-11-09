// InputTapTests.swift - input taps on the linked engine (contract 5, 30-ios 15.1).
// The tests call es_engine_render directly, standing in for the audio callback. With the real engine
// the 16 kHz history comes from a FIR decimator, so a level step takes decimatorSettle16k samples to
// settle; sample counts and cursor positions are exact on both engines.

import XCTest
@testable import Earshot

final class InputTapTests: XCTestCase {
    private var model: EarshotModel!
    private var engine: EarshotEngine!

    override func setUpWithError() throws {
        model = try TestSupport.model() // the stub, or random_v1 with the real engine
        engine = try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 480, instances: 3, slack16: 43)
    }

    /// Feeds `seconds` of a constant value through the engine at 48 kHz, 480 frames per callback.
    private func render(seconds: Double, value: Float) {
        let n = 480
        let input = [Float](repeating: value, count: n)
        var output = [Float](repeating: 0, count: n)
        let callbacks = Int(seconds * 48_000) / n
        for _ in 0..<callbacks {
            input.withUnsafeBufferPointer { inPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    es_engine_render(engine.cEngine, inPtr.baseAddress, outPtr.baseAddress, Int32(n))
                }
            }
        }
    }

    func testTapReturnsOnlyAudioAfterItWasMade() {
        render(seconds: 10, value: 0.1)          // old audio the tap must never see
        var tap = engine.makeInputTap()
        render(seconds: 1, value: 0.5)           // new audio
        let got = tap.read(maxSamples: 100_000)
        XCTAssertEqual(got.count, 16_000, "1 s at 16 kHz")
        XCTAssertTrue(got.dropFirst(TestSupport.decimatorSettle16k).allSatisfy { abs($0 - 0.5) < 1e-6 },
                      "only audio written after the tap")
        XCTAssertEqual(tap.droppedSamples, 0)
        XCTAssertEqual(tap.read(maxSamples: 100).count, 0, "nothing new yet")
    }

    func testReadsContinueInOrder() {
        var tap = engine.makeInputTap()
        render(seconds: 0.5, value: 0.2)
        let first = tap.read(maxSamples: 3_000)
        let rest = tap.read(maxSamples: 100_000)
        XCTAssertEqual(first.count, 3_000)
        XCTAssertEqual(first.count + rest.count, 8_000)
    }

    func testSlowReaderReportsDroppedSamples() {
        var tap = engine.makeInputTap()
        render(seconds: 20, value: 0.3)          // more than the 15.36 s the history holds
        let got = tap.read(maxSamples: 400_000)
        XCTAssertEqual(got.count, 245_760, "the readable span")
        XCTAssertEqual(tap.droppedSamples, 20 * 16_000 - 245_760)
    }

    func testCursorFromTheFutureReadsNothing() {
        // A cursor ahead of the writer (for example one kept from another engine) must read
        // nothing and stay where it is (DECISIONS L, 2026-11-07), never jump 15 s back.
        render(seconds: 1, value: 0.5)
        let w = es_engine_input_position(engine.cEngine)
        var cursor = w + 50
        var buf = [Float](repeating: 0, count: 100)
        let got = buf.withUnsafeMutableBufferPointer { ptr in
            es_engine_read_input(engine.cEngine, &cursor, ptr.baseAddress, 100)
        }
        XCTAssertEqual(got, 0)
        XCTAssertEqual(cursor, w + 50, "the cursor is left unchanged")

        // Once the writer passes that cursor, reading resumes from it with nothing dropped.
        render(seconds: 0.1, value: 0.25)                  // 1,600 more samples
        let more = buf.withUnsafeMutableBufferPointer { ptr in
            es_engine_read_input(engine.cEngine, &cursor, ptr.baseAddress, 100)
        }
        XCTAssertEqual(more, 100)
        XCTAssertEqual(cursor, w + 150)
    }

    func testTapStartsAtTheInputPosition() {
        // contract 5: makeInputTap starts at es_engine_input_position, so its cursor equals it.
        render(seconds: 3, value: 0.1)
        let position = es_engine_input_position(engine.cEngine)
        XCTAssertEqual(position, 48_000, "3 s at 16 kHz")
        let tap = engine.makeInputTap()
        XCTAssertEqual(tap.cursor, position)
        XCTAssertEqual(tap.droppedSamples, 0)
    }

    func testRecentInputReturnsNewestSamples() {
        render(seconds: 2, value: 0.1)
        render(seconds: 1, value: 0.4)
        let recent = engine.recentInput(samples: 16_000)
        XCTAssertEqual(recent.count, 16_000)
        XCTAssertTrue(recent.dropFirst(TestSupport.decimatorSettle16k).allSatisfy { abs($0 - 0.4) < 1e-6 })
        XCTAssertEqual(engine.recentInput(samples: 1_000_000).count, 48_000, "capped by what exists")
    }
}
