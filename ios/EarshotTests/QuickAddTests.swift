// QuickAddTests.swift - "Add who just spoke" helpers (30-ios 8.5): the RAM copy across a swap,
// the 2 s playback window and the playback level.

import XCTest
@testable import Earshot

final class QuickAddTests: XCTestCase {
    func testCombineUsesFreshHistoryOnceItIsLongEnough() {
        let ram = [Float](repeating: 1, count: 128_000)
        let fresh = [Float](repeating: 2, count: 128_000)
        XCTAssertEqual(QuickAdd.combine(ram: ram, fresh: fresh), fresh)
        // Right after a swap: 1 s of new audio, so the newest 7 s of the RAM copy go in front.
        let short = [Float](repeating: 2, count: 16_000)
        let mixed = QuickAdd.combine(ram: ram, fresh: short)
        XCTAssertEqual(mixed.count, 128_000)
        XCTAssertEqual(mixed.first, 1)
        XCTAssertEqual(mixed.last, 2)
        XCTAssertEqual(mixed.filter { $0 == 2 }.count, 16_000)
        XCTAssertEqual(QuickAdd.combine(ram: [], fresh: short), short, "no swap: fresh history only")
    }

    func testRecentAudioComesFromTheEngineHistory() throws {
        let model = try TestSupport.model()
        let engine = try EarshotEngine(model: model, ioRate: 48_000, ioFrames: 480, instances: 3, slack16: 43)
        var out = [Float](repeating: 0, count: 480)
        let input = [Float](repeating: 0.25, count: 480)
        for _ in 0..<(10 * 100) { // 10 s at 48 kHz
            input.withUnsafeBufferPointer { i in
                out.withUnsafeMutableBufferPointer { o in es_engine_render(engine.cEngine, i.baseAddress, o.baseAddress, 480) }
            }
        }
        let recent = engine.recentInput(samples: QuickAdd.samples)
        XCTAssertEqual(recent.count, 128_000, "the last 8 s")
        XCTAssertTrue(recent.allSatisfy { abs($0 - 0.25) < 1e-6 })
    }

    func testPlaybackRangeIsTheTwoSecondsWithMostSpeech() {
        var mask = [UInt8](repeating: 0, count: 250) // 8 s clip
        for f in 150..<200 { mask[f] = 1 }
        let r = QuickAdd.playbackRange(mask: mask, clipLength: 128_000)
        XCTAssertEqual(r.count, 32_000)
        let mids = (150..<200).map(ClipTools.frameMid)
        XCTAssertGreaterThanOrEqual(mids.filter { r.contains($0) }.count, 50 - 1, "nearly every speech frame inside")
        XCTAssertEqual(QuickAdd.playbackRange(mask: mask, clipLength: 20_000), 0..<20_000)
    }

    func testPlaybackLevelIsMinus26RMSWithPeaksAtMostMinus6() {
        let tone = (0..<32_000).map { Float(0.01 * sin(Double($0) * 0.1)) }
        let a = QuickAdd.normalizeForPlayback(tone)
        let rms = (a.reduce(0) { $0 + $1 * $1 } / Float(a.count)).squareRoot()
        XCTAssertEqual(20 * log10(rms), -26, accuracy: 0.01)

        var spiky = tone
        spiky[100] = 0.5 // one big peak: the peak limit wins over the RMS target
        let b = QuickAdd.normalizeForPlayback(spiky)
        let peak = b.reduce(0) { max($0, abs($1)) }
        XCTAssertEqual(20 * log10(peak), -6, accuracy: 0.01)
        XCTAssertEqual(QuickAdd.normalizeForPlayback([0, 0]), [0, 0], "silence stays silence")
    }
}
