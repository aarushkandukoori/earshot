// SpeakerEncoder.swift - WeSpeaker ResNet34 on Core ML (contract 6, 30-ios 8.4).
//
// What it does: turns 1 to 10 s of 16 kHz audio into a 256-number "voice vector" (unit length).
// Two vectors from the same person point the same way (cosine near 1).
//
// Rules (contract 6):
// - The fp32 package is loaded twice: `.cpuAndGPU` for the foreground and `.cpuOnly` for when the
//   app is not active (background apps cannot submit GPU work; hands-free capture runs locked).
//   Never `.all`: the Neural Engine broke the mixed-precision package (cosine 0.57).
// - Input `wav` float32 [1, N], N = 16,000 ... 160,000, raw samples in [-1, 1]. Output `embedding` [1, 256].
// - Launch self-test on `wespk_selftest.est`: cosine >= meta.tolerances.cosine_min passes; else reload
//   `.cpuOnly` and retry; else the app runs without WeSpeaker (use_wespk = 0).
// - It is an actor, so calls run one at a time off the main thread; never called from the audio
//   callback, the worker, or the live path.

import CoreML
import Foundation

/// Anything that maps raw audio to a raw (not yet normalized) embedding. The real one wraps an
/// MLModel; tests pass a fake, so the self-test logic is checked without Core ML.
protocol EmbeddingModel {
    func rawEmbedding(_ wav: [Float]) throws -> [Float]
}

/// What EnrollmentService and PeopleStore need from the encoder (a fake in tests).
protocol SpeakerEmbedding: Sendable {
    /// `appActive` is read on the main actor right before the call (UIApplication state).
    func embed(_ pcm16k: [Float], appActive: Bool) async throws -> [Float]
}

enum EncoderError: Error, Equatable {
    case badLength(Int)     // outside 16,000 ... 160,000 samples
    case badOutput(String)  // wrong size, a NaN or infinity, or a zero vector
}

/// The real Core ML model, loaded for one compute-unit setting.
final class CoreMLEmbeddingModel: EmbeddingModel {
    let model: MLModel

    init(url: URL, units: MLComputeUnits) throws {
        let config = MLModelConfiguration()
        config.computeUnits = units
        model = try MLModel(contentsOf: url, configuration: config)
    }

    func rawEmbedding(_ wav: [Float]) throws -> [Float] {
        let input = try MLMultiArray(shape: [1, NSNumber(value: wav.count)], dataType: .float32)
        let dst = input.dataPointer.bindMemory(to: Float.self, capacity: wav.count)
        wav.withUnsafeBufferPointer { src in dst.update(from: src.baseAddress!, count: wav.count) }
        let features = try MLDictionaryFeatureProvider(dictionary: [SpeakerEncoder.inputName: input])
        let result = try model.prediction(from: features)
        guard let out = result.featureValue(for: SpeakerEncoder.outputName)?.multiArrayValue else {
            throw EncoderError.badOutput("no \(SpeakerEncoder.outputName) output")
        }
        return (0..<out.count).map { out[$0].floatValue }
    }
}

actor SpeakerEncoder: SpeakerEmbedding {
    static let inputName = "wav"
    static let outputName = "embedding"
    static let samplesRange = 16_000...160_000
    static let dim = Int(ES_WESPK_DIM) // 256

    private var gpu: EmbeddingModel?          // .cpuAndGPU instance (nil if it failed to load)
    private var cpu: EmbeddingModel           // .cpuOnly instance
    private let reloadCPU: () throws -> EmbeddingModel
    private let selfTestData: () -> Data?

    /// Updated from the scene phase; used by the contract `embed(_:)` that takes no state.
    private(set) var appActive = true
    /// False after the GPU instance failed the self-test: every call then uses `.cpuOnly`.
    private(set) var gpuTrusted = true
    /// Shown in Diagnostics (30-ios 14): the self-test cosine and which compute units passed.
    private(set) var selfTestCosine: Double?
    private(set) var selfTestUnits = "-"

    /// Loads the compiled model (`wespeaker_resnet34_fp32.mlmodelc`) twice (contract 5 signature).
    init(modelURL: URL) throws {
        let cpuModel = try CoreMLEmbeddingModel(url: modelURL, units: .cpuOnly)
        self.cpu = cpuModel
        self.gpu = try? CoreMLEmbeddingModel(url: modelURL, units: .cpuAndGPU)
        self.reloadCPU = { try CoreMLEmbeddingModel(url: modelURL, units: .cpuOnly) }
        self.selfTestData = {
            guard let url = Bundle.main.url(forResource: "wespk_selftest", withExtension: "est") else { return nil }
            return try? Data(contentsOf: url)
        }
    }

    /// For tests: any models and any self-test file.
    init(gpu: EmbeddingModel?, cpu: EmbeddingModel, reloadCPU: @escaping () throws -> EmbeddingModel,
         selfTestData: @escaping () -> Data?) {
        self.gpu = gpu
        self.cpu = cpu
        self.reloadCPU = reloadCPU
        self.selfTestData = selfTestData
    }

    /// The bundled compiled model, or nil if `make models` did not run.
    static var bundledModelURL: URL? {
        Bundle.main.url(forResource: "wespeaker_resnet34_fp32", withExtension: "mlmodelc")
    }

    func setAppActive(_ active: Bool) { appActive = active }

    /// Contract 5: 256 floats, unit length. Uses the stored app state.
    func embed(_ pcm16k: [Float]) throws -> [Float] {
        try embed(pcm16k, appActive: appActive)
    }

    /// `.cpuAndGPU` only while the app is active and the GPU passed the self-test; else `.cpuOnly`.
    func embed(_ pcm16k: [Float], appActive: Bool) throws -> [Float] {
        let model: EmbeddingModel = (appActive && gpuTrusted) ? (gpu ?? cpu) : cpu
        return try Self.run(model, pcm16k)
    }

    /// Checks the length, runs the model, checks the output and L2-normalizes it.
    static func run(_ model: EmbeddingModel, _ pcm16k: [Float]) throws -> [Float] {
        guard samplesRange.contains(pcm16k.count) else { throw EncoderError.badLength(pcm16k.count) }
        let raw = try model.rawEmbedding(pcm16k)
        guard raw.count == dim else { throw EncoderError.badOutput("\(raw.count) values, expected \(dim)") }
        guard raw.allSatisfy({ $0.isFinite }) else { throw EncoderError.badOutput("non-finite value") }
        let norm = raw.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        guard norm > 1e-12 else { throw EncoderError.badOutput("zero vector") }
        return raw.map { Float(Double($0) / norm) }
    }

    /// Launch self-test (30-ios 8.4, 9.4 step 4). Never throws: false means "run without WeSpeaker".
    func selfTest() -> Bool {
        selfTestCosine = nil
        selfTestUnits = "-"
        guard let data = selfTestData(),
              let file = try? ContainerFile(data: data, expectMagic: "EST1"),
              let clip = try? file.floats("clip"), let reference = try? file.floats("embedding"),
              let cosineMin = file.metaNumber("tolerances.cosine_min") else {
            return false // missing or damaged self-test file
        }
        // 1. The foreground instance (.cpuAndGPU).
        if let gpu, let e = try? Self.run(gpu, clip) {
            let c = ClipTools.cosine(e, reference)
            selfTestCosine = c
            if c >= cosineMin {
                gpuTrusted = true
                selfTestUnits = "cpuAndGPU"
                return true
            }
        }
        // 2. Reload .cpuOnly and retry; if that passes, never use the GPU instance again.
        if let fresh = try? reloadCPU(), let e = try? Self.run(fresh, clip) {
            let c = ClipTools.cosine(e, reference)
            selfTestCosine = c
            if c >= cosineMin {
                cpu = fresh
                gpuTrusted = false
                selfTestUnits = "cpuOnly"
                return true
            }
        }
        // 3. Both failed: use_wespk = 0 for every conditioning.
        return false
    }
}
