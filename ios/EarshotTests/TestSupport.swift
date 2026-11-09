// TestSupport.swift - small helpers shared by the I-2 tests: synthetic speech-like audio, the
// contract thresholds.json example, a container builder, fake WeSpeaker models and temp folders.

import CryptoKit
import Foundation
import XCTest
@testable import Earshot

enum TestSupport {
    /// The thresholds.json example of contract 6, written out in full.
    static let thresholdsJSON = """
    {"version": 1, "model_hash": "\(String(repeating: "0", count: 64))",
     "target": {"speech_s": [4, 5], "sbr_db": [0, 6], "one_voice": [0.45, 0.60], "own_cos": [0.50, 0.60],
                "dup_cos": 0.75, "clip_pct": [0.1, 1.0]},
     "own": {"speech_s": 8, "snr_db": [15, 20]}, "own_window_cos": 0.60, "locked_cos": 0.0}
    """

    static func thresholds(version: Int = 1, hash: String = String(repeating: "0", count: 64)) -> Data {
        Data(thresholdsJSON.replacingOccurrences(of: "\"version\": 1", with: "\"version\": \(version)")
            .replacingOccurrences(of: String(repeating: "0", count: 64), with: hash).utf8)
    }

    static func contractThresholds() throws -> Thresholds { try Thresholds.decode(thresholds()) }

    /// Speech-like audio for the energy VAD: inside [from, to) seconds a 220 Hz tone that is on for
    /// 300 ms and off for 200 ms (real speech has pauses, and es_vad needs quiet frames to set its
    /// floor); everywhere else a -80 dBFS hum. Deterministic.
    static func speechLike(seconds: Double, from: Double, to: Double, amp: Float = 0.3) -> [Float] {
        let n = Int(seconds * 16_000)
        return (0..<n).map { i in
            let t = Double(i) / 16_000
            let syllable = t >= from && t < to && (t - from).truncatingRemainder(dividingBy: 0.5) < 0.3
            return syllable ? amp * Float(sin(2 * .pi * 220 * t)) : 0.0001 * Float(sin(2 * .pi * 50 * t))
        }
    }

    /// The 256-dim unit vector with a 1 at index i (two different i are orthogonal: cosine 0).
    static func basis(_ i: Int) -> [Float] {
        var v = [Float](repeating: 0, count: 256)
        v[i] = 1
        return v
    }

    /// Little-endian float bytes.
    static func bytes(_ x: [Float]) -> [UInt8] {
        x.flatMap { f -> [UInt8] in withUnsafeBytes(of: f.bitPattern.littleEndian) { Array($0) } }
    }

    /// Builds a contract 3.1 container (EST1 by default) with the given f32 arrays and meta.
    static func container(magic: String = "EST1", meta: [String: Any], arrays: [(String, [Float])],
                          misalign: Bool = false) -> Data {
        var body: [UInt8] = []
        var entries: [[String: Any]] = []
        for (name, values) in arrays {
            while body.count % 64 != 0 { body.append(0) }
            let offset = body.count + (misalign ? 4 : 0)
            if misalign { body += [0, 0, 0, 0] }
            let b = bytes(values)
            entries.append(["name": name, "dtype": "f32", "shape": [values.count], "offset": offset, "nbytes": b.count])
            body += b
        }
        let sha = Data(body).sha256Hex
        let header: [String: Any] = ["format": "earshot-testvec", "format_version": 1, "meta": meta,
                                     "arrays": entries, "data_bytes": body.count, "data_sha256": sha]
        let json = try! JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var out = Data(magic.utf8)
        for v in [UInt32(1), UInt32(json.count), UInt32(0)] { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        out.append(json)
        while out.count % 64 != 0 { out.append(0) }
        out.append(contentsOf: body)
        return out
    }

    /// A self-test container like wespk_selftest.est: a 5 s clip, its reference embedding, cosine_min.
    static func selfTestFile(reference: [Float] = basis(0), cosineMin: Double = 0.999) -> Data {
        let clip = speechLike(seconds: 5, from: 0.5, to: 4.5)
        return container(meta: ["kind": "wespeaker-selftest", "tolerances": ["cosine_min": cosineMin]],
                         arrays: [("clip", clip), ("embedding", reference)])
    }

    /// A fresh empty folder under the test host's temporary directory.
    static func tempFolder() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("earshot-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: - Which engine is linked (lead integration)
    //
    // `make test` links the real engine (engine/src, ENGINE ?= real); `make test-stub` links
    // Stubs/earshot_stub.c. Most tests only need "a model" and run on both. A test that checks a
    // behavior only the stub has (plain pass-through, 3:1 averaging, a zero model hash, any bytes
    // accepted as weights) calls requireStub(_:) and is reported as skipped with that reason under the
    // real engine, so `make test` never hides it; `make test-stub` still runs it.

    /// True when the stub engine is linked: only the stub accepts an empty weights buffer.
    static let engineIsStub: Bool = (try? EarshotModel(weights: Data()))?.isStub ?? false

    /// A bundled model file (`make models` copies ml/exports/random_v1 into the app bundle, and the
    /// test host is the app, so Bundle.main holds it). nil when it is not bundled.
    static func bundled(_ name: String, _ ext: String) -> Data? {
        guard let url = Bundle.main.url(forResource: name, withExtension: ext) else { return nil }
        return try? Data(contentsOf: url)
    }

    /// random_v1 loaded once (10 MB; EarshotModel is immutable, so every test may share it).
    private static let sharedRealModel: Result<EarshotModel, Error> = Result {
        guard let weights = bundled("earshot", "esw") else {
            throw XCTSkip("earshot.esw not bundled (run `make models` from the repo root)")
        }
        return try EarshotModel(weights: weights)
    }

    /// The model every engine-agnostic test uses: the stub (empty buffer) or the bundled random_v1
    /// weights with the real engine.
    static func model() throws -> EarshotModel {
        if engineIsStub { return try EarshotModel(weights: Data()) }
        return try sharedRealModel.get()
    }

    /// The model hash of `model()`: 64 zeros with the stub, the .esw data_sha256 with the real engine.
    static func linkedModelHash() throws -> String { try model().modelHash }

    /// 16 kHz history samples after a level step (or after a new engine starts from silence) before the
    /// history holds the new level: 0 for the stub's plain 3:1 average; the real decimator is a FIR of
    /// 177 taps at 48 kHz (resampler_tables.c), which spans 59 samples at 16 kHz.
    static var decimatorSettle16k: Int { engineIsStub ? 0 : 59 }

    /// Skips the calling test unless the stub engine is linked. `why` names the stub-only behavior.
    static func requireStub(_ why: String) throws {
        if !engineIsStub { throw XCTSkip("stub engine only (run `make test-stub`): \(why)") }
    }

    /// The stub model, for a test that checks stub-only behavior (skipped with the real engine).
    static func stubModel(_ why: String) throws -> EarshotModel {
        try requireStub(why)
        return try EarshotModel(weights: Data())
    }

    /// Plays `input` through `engine` the way the app does, but on the test thread: one render per
    /// `frames`-sized callback, then es_engine_run_hops as the worker would (the real engine needs it
    /// to produce output; with the stub it returns 0 and changes nothing). Returns the output.
    @discardableResult
    static func renderWithHops(_ engine: EarshotEngine, _ input: [Float], frames: Int) -> [Float] {
        var out = [Float](repeating: 0, count: input.count)
        var start = 0
        input.withUnsafeBufferPointer { inBuf in
            out.withUnsafeMutableBufferPointer { outBuf in
                while start < input.count {
                    let n = min(frames, input.count - start)
                    es_engine_render(engine.cEngine, inBuf.baseAddress! + start, outBuf.baseAddress! + start, Int32(n))
                    _ = es_engine_run_hops(engine.cEngine)
                    start += n
                }
            }
        }
        return out
    }
}

extension Data {
    /// Lowercase hex SHA-256, the digest ContainerFile checks against data_sha256.
    var sha256Hex: String { SHA256.hash(data: self).map { String(format: "%02x", $0) }.joined() }
}

/// A WeSpeaker stand-in for SpeakerEncoder: returns whatever `output` says and counts calls.
final class FakeEmbeddingModel: EmbeddingModel {
    var calls = 0
    let output: ([Float]) throws -> [Float]
    init(_ output: @escaping ([Float]) throws -> [Float]) { self.output = output }
    func rawEmbedding(_ wav: [Float]) throws -> [Float] {
        calls += 1
        return try output(wav)
    }
}

/// A WeSpeaker stand-in for EnrollmentService and PeopleStore.
actor FakeEncoder: SpeakerEmbedding {
    private let f: @Sendable (Int, [Float]) -> [Float] // (call number from 0, audio) -> vector
    private(set) var calls = 0
    private(set) var appActiveSeen: [Bool] = []
    init(_ f: @escaping @Sendable ([Float]) -> [Float]) { self.f = { _, pcm in f(pcm) } }
    init(byCall f: @escaping @Sendable (Int, [Float]) -> [Float]) { self.f = f }
    func embed(_ pcm16k: [Float], appActive: Bool) async throws -> [Float] {
        let n = calls
        calls += 1
        appActiveSeen.append(appActive)
        return f(n, pcm16k)
    }
}
