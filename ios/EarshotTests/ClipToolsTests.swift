// ClipToolsTests.swift - the contract 6 check formulas (30-ios 8.2, 15.1).
//
// The main test reads WP-ML's quality_vectors.json (ml/exports/random_v1, copied into the test
// bundle by `make models`) and checks every stored value for all 5 captures and both roles. es_vad
// comes from the linked engine (the stub implements the real contract 2.4 rule bit for bit).

import XCTest
@testable import Earshot

final class ClipToolsTests: XCTestCase {
    // MARK: - quality_vectors.json

    private func loadVectors() throws -> [[String: Any]] {
        guard let url = Bundle(for: ClipToolsTests.self).url(forResource: "quality_vectors", withExtension: "json") else {
            XCTFail("quality_vectors.json is not in the test bundle: run `make models` at the repo root, then `make test`")
            return []
        }
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        XCTAssertEqual(root?["kind"] as? String, "earshot-quality-vectors")
        return root?["vectors"] as? [[String: Any]] ?? []
    }

    private func pcm(_ v: [String: Any]) -> [Float] {
        guard let text = v["pcm_f32le_base64"] as? String, let data = Data(base64Encoded: text) else { return [] }
        var out = [Float](repeating: 0, count: data.count / 4) // little-endian floats, as on iPhone and Mac
        _ = out.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return out
    }

    private func pair(_ any: Any?) -> [Int]? { (any as? [NSNumber])?.map { $0.intValue } }

    func testEveryValueOfQualityVectors() throws {
        let vectors = try loadVectors()
        XCTAssertEqual(vectors.count, 5)
        let model = try TestSupport.model()
        for v in vectors {
            let name = v["name"] as? String ?? "?"
            let x = pcm(v)
            let e = v["expected"] as? [String: Any] ?? [:]
            XCTAssertEqual(x.count, e["n"] as? Int, name)

            // Full-capture values (steps 1, 2, 6, 7).
            let mask0 = model.vad(x)
            XCTAssertEqual(ClipTools.keptFrames(mask0).count, e["mask0_frames"] as? Int, "\(name) mask0_frames")
            XCTAssertEqual(ClipTools.roundSeconds(ClipTools.speechSeconds(mask0)),
                           (e["speech_s_full"] as? Double) ?? -1, accuracy: 1e-9, "\(name) speech_s_full")
            XCTAssertEqual(ClipTools.sbrDB(x, mask0: mask0) ?? .nan, (e["sbr_db"] as? Double) ?? 0, accuracy: 1e-6,
                           "\(name) sbr_db")
            XCTAssertEqual(ClipTools.sbrSplit(x, mask0: mask0).fallback, e["sbr_fallback"] as? Bool, "\(name) fallback")
            XCTAssertEqual(ClipTools.clipPct(x), (e["clip_pct"] as? Double) ?? -1, accuracy: 1e-9, "\(name) clip_pct")
            let span = ClipTools.frameSpan(mask0: mask0, n: x.count)
            XCTAssertEqual(span.map { [$0.start, $0.end] }, pair(e["span"]), "\(name) span")

            // Per role (steps 3, 4, 5, 9, 10).
            for (role, key) in [(VoiceRole.target, "target"), (VoiceRole.own, "own")] {
                let r = e[key] as? [String: Any] ?? [:]
                let label = "\(name) \(key)"
                let t = ClipTools.trim(n: x.count, mask0: mask0, role: role)
                XCTAssertEqual(t.region.map { [$0.start, $0.end] }, pair(r["region"]), "\(label) region")
                XCTAssertEqual(t.windowStart, r["window_start"] as? Int, "\(label) window_start")
                XCTAssertEqual(t.clipStart, r["clip_start"] as? Int, "\(label) clip_start")
                XCTAssertEqual(t.clipLen, r["clip_len"] as? Int, "\(label) clip_len")
                XCTAssertEqual(t.failure?.rawValue, r["fail"] as? String, "\(label) fail")
                guard t.failure == nil else { continue }

                let clip = ClipTools.clip(x, t)
                let wins = ClipTools.windows(clipLen: clip.count)
                let expectedWins = (r["windows"] as? [[NSNumber]])?.map { $0.map { $0.intValue } }
                XCTAssertEqual(wins.map { [$0.start, $0.length] }, expectedWins, "\(label) windows")
                let mask = model.vad(clip)
                XCTAssertEqual(mask.count, r["clip_frames"] as? Int, "\(label) clip_frames")
                let inside = wins.map { w in ClipTools.framesInside(windowStart: w.start, frameCount: mask.count) }
                let expectedInside = (r["frames_inside"] as? [Any])?.map { ($0 as? [NSNumber])?.map { $0.intValue } }
                XCTAssertEqual(inside.map { $0.map { [$0.first, $0.last] } }, expectedInside, "\(label) frames_inside")
                let kept = ClipTools.keptFrames(mask).count
                XCTAssertEqual(kept, r["speech_frames"] as? Int, "\(label) speech_frames")
                XCTAssertEqual(ClipTools.roundSeconds(ClipTools.speechSeconds(mask)), (r["speech_s"] as? Double) ?? -1,
                               accuracy: 1e-9, "\(label) speech_s")
                XCTAssertEqual(kept / 3, r["n_avail"] as? Int, "\(label) n_avail")

                // speech_s is what es_voiceprint_speech_seconds reports after es_enroll with this mask.
                if kept / 3 >= 4 {
                    let vp = try Voiceprint(model: model, pcm16k: clip, mask: mask, role: role, weSpeaker: nil)
                    XCTAssertEqual(Double(vp.speechSeconds), Double(kept) * 0.032, accuracy: 1e-4, "\(label) es_enroll")
                }
            }
        }
    }

    // MARK: - The formulas one by one

    func testFrameMapAndMidpoints() {
        XCTAssertEqual(ClipTools.frameMid(0), -192)
        XCTAssertEqual(ClipTools.frameMid(1), 320)
        XCTAssertEqual(ClipTools.frameSamples(0, n: 100_000), 0..<64, "frame 0 is mostly padding")
        XCTAssertEqual(ClipTools.frameSamples(1, n: 100_000), 64..<576)
        XCTAssertEqual(ClipTools.frameSamples(2, n: 600), 576..<600, "cut at the end of the capture")
    }

    func testSpanFromFirstToLastSpeechFrame() {
        let mask: [UInt8] = [0, 0, 1, 1, 0, 1, 0, 0]
        let span = ClipTools.frameSpan(mask0: mask, n: 4_000)
        XCTAssertEqual(span?.start, 512 * 2 - 448)
        XCTAssertEqual(span?.end, 512 * 5 + 64)
        XCTAssertNil(ClipTools.frameSpan(mask0: [0, 0, 0], n: 2_000))
    }

    func testBestWindowTakesMostMidpointsAndEarliestOnTies() {
        // 400 frames (12.8 s); speech in frames 300..399 only, so the best 10 s window ends late.
        var mask = [UInt8](repeating: 0, count: 400)
        for f in 300..<400 { mask[f] = 1 }
        let region = (0, 400 * 512)
        let start = ClipTools.bestWindow(mask: mask, regionStart: region.0, regionEnd: region.1)
        // Frames 300..399 have midpoints 153,408 .. 204,096. A window [s, s + 160,000) holds all 100
        // when s > 44,096; the earliest 512-step start after that is 87 x 512 = 44,544.
        XCTAssertEqual(start, 44_544)
        // Everything equal: the earliest start wins.
        let flat = [UInt8](repeating: 1, count: 400)
        XCTAssertEqual(ClipTools.bestWindow(mask: [UInt8](repeating: 0, count: 400), regionStart: 1024, regionEnd: 400 * 512), 1024)
        // All speech over 12.8 s: the 88 candidate starts (0 .. 87 x 512) all hold 312 midpoints, so
        // this is a real tie between many windows, and the earliest (0) must win.
        XCTAssertEqual(ClipTools.bestWindow(mask: flat, regionStart: 0, regionEnd: 400 * 512), 0)
    }

    func testRoundingAndShortClips() {
        XCTAssertEqual(ClipTools.roundLength(127_999), 120_000)
        XCTAssertEqual(ClipTools.roundLength(64_000), 64_000)
        var mask = [UInt8](repeating: 0, count: 200)
        for f in 10..<130 { mask[f] = 1 }      // span of about 3.8 s: too short
        let t = ClipTools.trim(n: 200 * 512, mask0: mask, role: .target)
        XCTAssertEqual(t.failure, .tooShort)
        XCTAssertEqual(ClipTools.trim(n: 5_000, mask0: [0, 0, 0, 0, 0, 0, 0, 0, 0], role: .target).failure, .noSpeech)
    }

    func testWindowsAndInsideRule() {
        XCTAssertTrue(ClipTools.windows(clipLen: 23_999).isEmpty)
        XCTAssertEqual(ClipTools.windows(clipLen: 24_000).count, 1)
        XCTAssertEqual(ClipTools.windows(clipLen: 64_000).count, 6) // w = 0 .. 5
        XCTAssertEqual(ClipTools.windows(clipLen: 160_000).count, 18)
        let r = ClipTools.framesInside(windowStart: 8_000, frameCount: 300)
        // midpoint 512 f - 192 in [8000, 32000): f = 16 (8,000) .. 62 (31,552)
        XCTAssertEqual(r?.first, 16)
        XCTAssertEqual(r?.last, 62)
    }

    func testOwnWindowMaskZeroesOnlyCloseWindows() {
        let mask = [UInt8](repeating: 1, count: 125) // a 4 s clip
        let wins = ClipTools.windows(clipLen: 64_000)
        var cos = [Double](repeating: 0.1, count: wins.count)
        cos[0] = 0.61 // window 0 sounds like the user (> 0.60)
        cos[1] = 0.60 // exactly the threshold: kept (strictly greater removes)
        let out = ClipTools.ownWindowMask(mask, windows: wins, windowCosOwn: cos, threshold: 0.60)
        let inside0 = ClipTools.framesInside(windowStart: 0, frameCount: 125)!
        for f in inside0.first...inside0.last { XCTAssertEqual(out[f], 0) }
        XCTAssertEqual(out[inside0.last + 1], 1)
        XCTAssertEqual(out[0], 1, "frame 0's midpoint is before every window")
    }

    func testSbrClipPctAndMedian() {
        let x = TestSupport.speechLike(seconds: 6, from: 1, to: 5)
        let mask0 = (try? TestSupport.model().vad(x)) ?? []
        XCTAssertGreaterThan(ClipTools.sbrDB(x, mask0: mask0) ?? 0, 40, "tone over a -80 dBFS hum")
        XCTAssertNil(ClipTools.sbrDB(x, mask0: [UInt8](repeating: 0, count: mask0.count)), "no speech frame")
        var loud = [Float](repeating: 0.1, count: 1_000)
        for i in 0..<10 { loud[i] = 0.98 }
        loud[10] = -0.99
        XCTAssertEqual(ClipTools.clipPct(loud), 1.1, accuracy: 1e-12)
        XCTAssertEqual(ClipTools.median([3, 1, 2]), 2)
        XCTAssertEqual(ClipTools.median([4, 1, 2, 3]), 2.5)
        XCTAssertNil(ClipTools.median([]))
        XCTAssertEqual(ClipTools.cosine([1, 0], [0, 1]), 0)
        XCTAssertEqual(ClipTools.cosine([2, 0], [3, 0]), 1)
        XCTAssertEqual(ClipTools.roundSeconds(Double(Float(125) * Float(0.032))), 4.0)
    }

    // MARK: - enroll.wav

    func testWavRoundTripWithinOneStep() throws {
        let x = (0..<16_000).map { i in Float(sin(Double(i) * 0.37)) * 0.9 } + [1.5, -1.5, 0]
        let data = ClipTools.encodeWAV(x)
        XCTAssertEqual(data.count, 44 + x.count * 2, "44-byte header + 16-bit samples")
        XCTAssertEqual(String(decoding: data[0..<4], as: UTF8.self), "RIFF")
        let y = try ClipTools.decodeWAV(data)
        XCTAssertEqual(y.count, x.count)
        for i in 0..<16_000 { XCTAssertEqual(y[i], x[i], accuracy: 1.0 / 32_767) }
        XCTAssertEqual(y[16_000], 1, "clamped to 1 then 32767")
        XCTAssertEqual(y[16_001], -1)
    }

    func testWavIsCappedAtTenSecondsAndRejectsOtherFormats() throws {
        let long = [Float](repeating: 0.1, count: 200_000)
        XCTAssertEqual(try ClipTools.decodeWAV(ClipTools.encodeWAV(long)).count, 160_000)
        var other = ClipTools.encodeWAV([0.1, 0.2])
        other[24] = 0x44; other[25] = 0xAC; other[26] = 0; other[27] = 0 // 44,100 Hz
        XCTAssertThrowsError(try ClipTools.decodeWAV(other))
        XCTAssertThrowsError(try ClipTools.decodeWAV(Data("not a wav".utf8)))
    }
}
