// SpeakerEncoderTests.swift - SpeakerEncoder's self-test and embed rules with fake models (30-ios 8.4),
// plus one run of the real bundled Core ML package when `make models` has copied it.

import XCTest
@testable import Earshot

final class SpeakerEncoderTests: XCTestCase {
    private let reference = TestSupport.basis(0)

    /// A fake model that always returns `v` (scaled, to check the L2 normalization too).
    private func fixed(_ v: [Float]) -> FakeEmbeddingModel { FakeEmbeddingModel { _ in v.map { $0 * 3 } } }

    private func encoder(gpu: FakeEmbeddingModel?, cpu: FakeEmbeddingModel, reload: FakeEmbeddingModel? = nil,
                         file: Data? = TestSupport.selfTestFile()) -> SpeakerEncoder {
        SpeakerEncoder(gpu: gpu, cpu: cpu, reloadCPU: {
            guard let reload else { throw EncoderError.badOutput("no reload") }
            return reload
        }, selfTestData: { file })
    }

    func testSelfTestPassesOnTheForegroundInstance() async {
        let gpu = fixed(reference), cpu = fixed(TestSupport.basis(1))
        let enc = encoder(gpu: gpu, cpu: cpu, reload: cpu)
        let passed = await enc.selfTest()
        XCTAssertTrue(passed)
        let units = await enc.selfTestUnits
        XCTAssertEqual(units, "cpuAndGPU")
        let cosine = await enc.selfTestCosine
        XCTAssertEqual(cosine ?? 0, 1, accuracy: 1e-9)
        let trusted = await enc.gpuTrusted
        XCTAssertTrue(trusted)
        XCTAssertEqual(cpu.calls, 0, "no CPU retry needed")
    }

    func testSelfTestFallsBackToCPUOnlyAndStaysThere() async throws {
        let gpu = fixed(TestSupport.basis(1))   // wrong vector: cosine 0
        let fresh = fixed(reference)            // the reloaded .cpuOnly instance is right
        let enc = encoder(gpu: gpu, cpu: fixed(TestSupport.basis(2)), reload: fresh)
        let passed = await enc.selfTest()
        XCTAssertTrue(passed)
        let units = await enc.selfTestUnits
        XCTAssertEqual(units, "cpuOnly")
        // From now on even the foreground uses the CPU instance.
        let before = gpu.calls
        _ = try await enc.embed([Float](repeating: 0.1, count: 16_000), appActive: true)
        XCTAssertEqual(gpu.calls, before, "the GPU instance is never used again")
        XCTAssertEqual(fresh.calls, 2, "self-test plus this call")
    }

    func testSelfTestFailsWhenBothInstancesAreWrong() async {
        let enc = encoder(gpu: FakeEmbeddingModel { _ in throw EncoderError.badOutput("gpu down") },
                          cpu: fixed(reference),
                          reload: FakeEmbeddingModel { _ in [Float](repeating: .nan, count: 256) })
        let passed = await enc.selfTest()
        XCTAssertFalse(passed, "use_wespk = 0")
    }

    func testSelfTestUsesCosineMinFromTheFile() async {
        // cos(0.9985) style vector: mostly basis(0) with a little of basis(1).
        var near = reference
        near[1] = 0.055 // cosine = 1 / sqrt(1 + 0.055^2) = 0.99849
        let strict = encoder(gpu: fixed(near), cpu: fixed(near), reload: fixed(near))
        let strictPassed = await strict.selfTest()
        XCTAssertFalse(strictPassed, "0.9985 is below cosine_min 0.999")
        let loose = encoder(gpu: fixed(near), cpu: fixed(near), reload: fixed(near),
                            file: TestSupport.selfTestFile(cosineMin: 0.998))
        let loosePassed = await loose.selfTest()
        XCTAssertTrue(loosePassed)
    }

    func testSelfTestFailsWithoutAValidFile() async {
        let missing = encoder(gpu: fixed(reference), cpu: fixed(reference), reload: fixed(reference), file: nil)
        let missingPassed = await missing.selfTest()
        XCTAssertFalse(missingPassed)
        var corrupted = TestSupport.selfTestFile()
        corrupted[corrupted.count - 10] ^= 0xFF
        let bad = encoder(gpu: fixed(reference), cpu: fixed(reference), reload: fixed(reference), file: corrupted)
        let badPassed = await bad.selfTest()
        XCTAssertFalse(badPassed, "data_sha256 mismatch")
    }

    func testEmbedChecksLengthOutputAndNormalizes() async throws {
        let enc = encoder(gpu: fixed(reference), cpu: fixed(reference))
        do { _ = try await enc.embed([Float](repeating: 0, count: 15_999), appActive: true); XCTFail("too short") }
        catch { XCTAssertEqual(error as? EncoderError, .badLength(15_999)) }
        do { _ = try await enc.embed([Float](repeating: 0, count: 160_001), appActive: true); XCTFail("too long") }
        catch { XCTAssertEqual(error as? EncoderError, .badLength(160_001)) }
        let e = try await enc.embed([Float](repeating: 0.1, count: 160_000), appActive: true)
        XCTAssertEqual(e.count, 256)
        XCTAssertEqual(e[0], 1, accuracy: 1e-6, "3 x basis(0) normalized back to unit length")

        let nan = encoder(gpu: FakeEmbeddingModel { _ in [Float](repeating: .nan, count: 256) }, cpu: fixed(reference))
        do { _ = try await nan.embed([Float](repeating: 0, count: 16_000), appActive: true); XCTFail("NaN must throw") }
        catch { XCTAssertNotNil(error as? EncoderError) }
        let short = encoder(gpu: FakeEmbeddingModel { _ in [1, 2, 3] }, cpu: fixed(reference))
        do { _ = try await short.embed([Float](repeating: 0, count: 16_000), appActive: true); XCTFail("wrong size") }
        catch { XCTAssertNotNil(error as? EncoderError) }
    }

    func testBackgroundUsesCPUOnly() async throws {
        let gpu = fixed(reference), cpu = fixed(reference)
        let enc = encoder(gpu: gpu, cpu: cpu)
        _ = try await enc.embed([Float](repeating: 0.1, count: 16_000), appActive: false)
        XCTAssertEqual(gpu.calls, 0)
        XCTAssertEqual(cpu.calls, 1)
        _ = try await enc.embed([Float](repeating: 0.1, count: 16_000), appActive: true)
        XCTAssertEqual(gpu.calls, 1)
        await enc.setAppActive(false) // what the scene phase does; the contract embed(_:) reads it
        _ = try await enc.embed([Float](repeating: 0.1, count: 16_000))
        XCTAssertEqual(cpu.calls, 2)
    }

    /// The real fp32 package on the simulator's CPU (the phone runs the same test at launch).
    func testRealBundledPackagePassesItsSelfTest() async throws {
        guard let url = SpeakerEncoder.bundledModelURL else {
            throw XCTSkip("wespeaker_resnet34_fp32.mlmodelc not bundled (run `make models`)")
        }
        let enc = try SpeakerEncoder(modelURL: url)
        let t0 = Date()
        let passed = await enc.selfTest()
        let cosine = await enc.selfTestCosine
        let units = await enc.selfTestUnits
        print("WeSpeaker self-test on the simulator: passed \(passed), cosine \(cosine ?? -1), units \(units), "
              + String(format: "%.2f s", Date().timeIntervalSince(t0)))
        XCTAssertTrue(passed)
        XCTAssertGreaterThanOrEqual(cosine ?? 0, 0.999)
    }
}
