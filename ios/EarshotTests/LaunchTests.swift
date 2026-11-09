// LaunchTests.swift - every failure branch of the launch sequence (30-ios 9.4): missing weights, a
// self-test failure, missing or bad thresholds, a thresholds hash mismatch, an encoder self-test
// failure, a person without `mask`, and a person whose files fail to load.
// Runs with either engine (lead integration): goodFiles() serves bytes the linked engine accepts (any
// bytes for the stub, the bundled random_v1 weights and test vector for the real engine), and
// testRealEngineLaunchPassesTheSelfTest is the 9.4 check that es_selftest passes on random_v1.

import XCTest
@testable import Earshot

final class LaunchTests: XCTestCase {
    private let zeros = String(repeating: "0", count: 64)

    /// A loader that serves only the given files.
    private func loader(_ files: [String: Data]) -> Launch.ResourceLoader {
        { name, ext in files["\(name).\(ext)"] }
    }

    /// Steps 1 to 3 pass with these files: weights and a test vector the linked engine accepts, plus
    /// a thresholds.json made for its model (pass `thresholds` to replace it, nil to leave it out).
    private func goodFiles(thresholds: Data?? = .none) throws -> [String: Data] {
        var files: [String: Data]
        if TestSupport.engineIsStub {
            files = ["earshot.esw": Data([1, 2, 3]), "testvec_short.est": Data([4, 5])] // the stub takes any bytes
        } else {
            guard let w = TestSupport.bundled("earshot", "esw"), let tv = TestSupport.bundled("testvec_short", "est") else {
                throw XCTSkip("model files not bundled (run `make models`)")
            }
            files = ["earshot.esw": w, "testvec_short.est": tv]
        }
        switch thresholds {
        case .none: files["thresholds.json"] = TestSupport.thresholds(hash: try TestSupport.linkedModelHash())
        case .some(.none): break
        case .some(.some(let data)): files["thresholds.json"] = data
        }
        return files
    }

    // MARK: - Steps 1 to 3

    func testStubRunsWithoutModelFiles() throws {
        try TestSupport.requireStub("only the stub runs without earshot.esw (the real engine blocks: testMissingWeightsBlocks)")
        let r = Launch.run(load: loader([:]))
        XCTAssertNil(r.blockingError)
        XCTAssertEqual(r.engineKind, "stub")
        XCTAssertEqual(r.modelHash, zeros)
        XCTAssertTrue(r.selfTestPassed)
        XCTAssertTrue(r.startEnabled)
        XCTAssertFalse(r.useWeSpeaker)
        XCTAssertNil(r.thresholds, "no file: enrollment stays off, never hard-coded values")
    }

    func testMissingWeightsBlocks() {
        let r = Launch.run(load: loader([:]), allowStubWithoutFiles: false)
        XCTAssertEqual(r.blockingError, Copy.modelMissing)
        XCTAssertFalse(r.startEnabled)
        XCTAssertNil(r.model)
    }

    func testSelfTestFailureDisablesStart() throws {
        let r = Launch.run(load: loader(try goodFiles()), selfTest: { _, _ in
            throw EarshotError(status: -11, message: "self-test failed")
        })
        XCTAssertNil(r.blockingError, "a failed self-test is not a blocking screen")
        XCTAssertFalse(r.selfTestPassed)
        XCTAssertFalse(r.startEnabled, "contract 3.3: no audio if the self-test fails")
        XCTAssertTrue(r.selfTestMessage.contains(Copy.selfTestFailed))
    }

    func testMissingThresholdsBlocks() throws {
        // Weights and test vector present, thresholds.json missing, and no stub fallback: blocking.
        let r = Launch.run(load: loader(try goodFiles(thresholds: .some(nil))), allowStubWithoutFiles: false)
        XCTAssertEqual(r.blockingError, Copy.thresholdsMissing)
        XCTAssertFalse(r.startEnabled)
    }

    func testMissingTestVectorDisablesStart() throws {
        var files = try goodFiles()
        files["testvec_short.est"] = nil
        let r = Launch.run(load: loader(files), allowStubWithoutFiles: false)
        XCTAssertNil(r.blockingError)
        XCTAssertFalse(r.selfTestPassed)
        XCTAssertFalse(r.startEnabled)
    }

    func testThresholdsWrongVersionBlocks() throws {
        let r = Launch.run(load: loader(try goodFiles(thresholds: TestSupport.thresholds(version: 2))))
        XCTAssertEqual(r.blockingError, Copy.thresholdsMissing)
        XCTAssertFalse(r.startEnabled)
    }

    func testThresholdsUnreadableBlocks() throws {
        let r = Launch.run(load: loader(try goodFiles(thresholds: Data("not json".utf8))))
        XCTAssertEqual(r.blockingError, Copy.thresholdsMissing)
        let partial = Launch.run(load: loader(try goodFiles(thresholds: Data(#"{"version": 1, "model_hash": "x"}"#.utf8))))
        XCTAssertEqual(partial.blockingError, Copy.thresholdsMissing, "every field is needed by enrollment")
    }

    func testThresholdsHashMismatchWarns() throws {
        let r = Launch.run(load: loader(try goodFiles(thresholds: TestSupport.thresholds(hash: String(repeating: "a", count: 64)))))
        XCTAssertNil(r.blockingError)
        XCTAssertEqual(r.thresholdsWarning, Copy.thresholdsOtherModel)
        XCTAssertEqual(r.thresholdsVersion, 1)
        XCTAssertNotNil(r.thresholds, "the file is still used")
        XCTAssertTrue(r.startEnabled, "a hash mismatch is only a Diagnostics warning")
    }

    func testThresholdsMatchingHashHasNoWarning() throws {
        let r = Launch.run(load: loader(try goodFiles())) // thresholds made for the linked model's hash
        XCTAssertNil(r.thresholdsWarning)
        XCTAssertEqual(r.thresholds?.ownWindowCos, 0.60)
    }

    func testBundledFilesAreUsedWhenPresent() throws {
        // The "files present" path end to end (the stub accepts any bytes; the real engine runs es_selftest).
        let r = Launch.run(load: loader(try goodFiles()), allowStubWithoutFiles: false)
        XCTAssertNil(r.blockingError)
        XCTAssertTrue(r.startEnabled)
        if TestSupport.engineIsStub {
            XCTAssertEqual(r.selfTestMaxError, 0)
        } else {
            XCTAssertEqual(r.engineKind, "real")
            XCTAssertLessThanOrEqual(r.selfTestMaxError ?? .infinity, 2e-4)
        }
    }

    /// The 9.4 sequence on the real bundle with the real engine: step 1 loads random_v1, step 2's
    /// es_selftest (with the meta.weights_sha256 check) passes, step 3's thresholds.json was made for
    /// exactly this model (no warning), and Start is enabled.
    func testRealEngineLaunchPassesTheSelfTest() throws {
        guard !TestSupport.engineIsStub else {
            throw XCTSkip("real engine only: the stub's es_selftest always passes with max error 0 (make test links the real engine)")
        }
        guard TestSupport.bundled("earshot", "esw") != nil else { throw XCTSkip("model files not bundled (run `make models`)") }
        let r = Launch.run() // the app's own bundle loader and self-test, as at launch
        XCTAssertNil(r.blockingError, r.notes.joined(separator: "; "))
        XCTAssertEqual(r.engineKind, "real")
        XCTAssertTrue(r.selfTestPassed, r.selfTestMessage)
        XCTAssertLessThanOrEqual(r.selfTestMaxError ?? .infinity, 2e-4, r.selfTestMessage)
        XCTAssertEqual(r.modelHash.count, 64)
        XCTAssertNotEqual(r.modelHash, zeros)
        XCTAssertEqual(r.thresholds?.modelHash, r.modelHash, "random_v1 thresholds.json names these weights")
        XCTAssertNil(r.thresholdsWarning)
        XCTAssertTrue(r.startEnabled)
    }

    func testRealBundleWithTheStub() throws {
        // After `make models` the app bundles random_v1; its thresholds were made for those weights,
        // so with the stub's zero hash the warning shows and the file is still used.
        guard Bundle.main.url(forResource: "thresholds", withExtension: "json") != nil else {
            throw XCTSkip("model files not bundled (run `make models`)")
        }
        let r = Launch.run()
        XCTAssertNil(r.blockingError)
        XCTAssertTrue(r.startEnabled)
        XCTAssertNotNil(r.thresholds)
        if r.engineKind == "stub" { XCTAssertEqual(r.thresholdsWarning, Copy.thresholdsOtherModel) }
    }

    // MARK: - Step 4: WeSpeaker self-test (never blocks)

    private func fakeEncoder(passes: Bool) -> SpeakerEncoder {
        let out = passes ? TestSupport.basis(0) : TestSupport.basis(1)
        let model = FakeEmbeddingModel { _ in out }
        return SpeakerEncoder(gpu: model, cpu: model, reloadCPU: { model }, selfTestData: { TestSupport.selfTestFile() })
    }

    func testEncoderSelfTestPassTurnsWeSpeakerOn() async throws {
        let (r, encoder) = await Launch.checkEncoder(Launch.run(load: loader(try goodFiles()))) { self.fakeEncoder(passes: true) }
        XCTAssertTrue(r.useWeSpeaker)
        XCTAssertNotNil(encoder)
        XCTAssertEqual(r.encoderMessage, Copy.voiceCheckReady)
        XCTAssertEqual(r.encoderUnits, "cpuAndGPU")
        XCTAssertEqual(r.encoderCosine ?? 0, 1, accuracy: 1e-9)
    }

    func testEncoderSelfTestFailureRunsWithoutWeSpeaker() async throws {
        let (r, encoder) = await Launch.checkEncoder(Launch.run(load: loader(try goodFiles()))) { self.fakeEncoder(passes: false) }
        XCTAssertFalse(r.useWeSpeaker, "use_wespk = 0 for every conditioning")
        XCTAssertNil(encoder)
        XCTAssertEqual(r.encoderMessage, Copy.voiceCheckUnavailable)
        XCTAssertTrue(r.startEnabled, "step 4 never blocks")
    }

    func testMissingOrBrokenEncoderPackageRunsWithoutWeSpeaker() async throws {
        let (missing, e1) = await Launch.checkEncoder(Launch.run(load: loader(try goodFiles()))) { nil }
        XCTAssertFalse(missing.useWeSpeaker)
        XCTAssertNil(e1)
        XCTAssertEqual(missing.encoderMessage, Copy.voiceCheckUnavailable)
        let (broken, e2) = await Launch.checkEncoder(Launch.run(load: loader(try goodFiles()))) {
            throw EncoderError.badOutput("load failed")
        }
        XCTAssertFalse(broken.useWeSpeaker)
        XCTAssertNil(e2)
        XCTAssertTrue(broken.startEnabled)
    }

    func testBlockedLaunchSkipsTheEncoder() async {
        var made = false
        let blocked = Launch.run(load: loader([:]), allowStubWithoutFiles: false)
        let (r, encoder) = await Launch.checkEncoder(blocked) { made = true; return self.fakeEncoder(passes: true) }
        XCTAssertFalse(made)
        XCTAssertNil(encoder)
        XCTAssertEqual(r.blockingError, Copy.modelMissing)
    }

    // MARK: - Step 5: people

    @MainActor
    func testPersonWithoutMaskGetsItRecomputedAndSaved() async throws {
        let root = TestSupport.tempFolder().appendingPathComponent("People")
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        let model = try TestSupport.model()
        let clip = TestSupport.speechLike(seconds: 6, from: 0.5, to: 5.5)
        let files = PersonFiles(root: root)
        let own = Person(name: "You", kind: .own, created: Date(), modelHash: zeros, eWespk: TestSupport.basis(0),
                         mask: model.vad(clip), quality: Quality(speechS: 8.5))
        let old = Person(name: "Maya", kind: .saved, created: Date(), modelHash: zeros, eWespk: TestSupport.basis(1),
                         mask: nil, quality: Quality(speechS: 5.5)) // an old file without `mask`
        try files.write(own, wav: ClipTools.encodeWAV(clip))
        try files.write(old, wav: ClipTools.encodeWAV(clip))

        // WeSpeaker works: only window 1 sounds like the user, so 8.3 step 6 zeroes the frames inside it.
        let encoder = FakeEncoder(byCall: { call, _ in call == 1 ? TestSupport.basis(0) : TestSupport.basis(2) })
        let s = PeopleStore(files: files, prepareLogs: false)
        await s.loadAndRebuild(model: model, encoder: encoder, ownWindowCos: 0.6)
        await s.flush()
        let calls = await encoder.calls
        XCTAssertEqual(calls, ClipTools.windows(clipLen: clip.count).count, "one call per window of the maskless file")
        let expected = ClipTools.ownWindowMask(model.vad(clip), windows: ClipTools.windows(clipLen: clip.count),
                                               windowCosOwn: ClipTools.windows(clipLen: clip.count).indices.map { $0 == 1 ? 1 : 0 },
                                               threshold: 0.6)
        XCTAssertEqual(s.person(old.id)?.mask, expected)
        XCTAssertNotEqual(expected, model.vad(clip), "window 1's frames really were removed")
        XCTAssertEqual(s.status[old.id], .ready)
        let onDisk = files.loadAll().compactMap(\.person).first { $0.id == old.id }
        XCTAssertEqual(onDisk?.mask, expected, "saved, so the next launch needs no WeSpeaker call")

        // Without WeSpeaker the recomputed mask is plain es_vad(clip).
        let root2 = TestSupport.tempFolder().appendingPathComponent("People")
        defer { try? FileManager.default.removeItem(at: root2.deletingLastPathComponent()) }
        try PersonFiles(root: root2).write(old, wav: ClipTools.encodeWAV(clip))
        let s2 = PeopleStore(files: PersonFiles(root: root2), prepareLogs: false)
        await s2.loadAndRebuild(model: model, encoder: nil, ownWindowCos: 0.6)
        XCTAssertEqual(s2.person(old.id)?.mask, model.vad(clip))
    }

    @MainActor
    func testPersonWhoseFilesFailIsNeedsReAdding() async throws {
        let root = TestSupport.tempFolder().appendingPathComponent("People")
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        let model = try TestSupport.model()
        let clip = TestSupport.speechLike(seconds: 6, from: 0.5, to: 5.5)
        let files = PersonFiles(root: root)
        let good = Person(name: "Maya", kind: .saved, created: Date(), modelHash: zeros, eWespk: nil,
                          mask: model.vad(clip), quality: Quality(speechS: 5.5))
        try files.write(good, wav: ClipTools.encodeWAV(clip))
        let brokenID = UUID()
        try FileManager.default.createDirectory(at: files.folder(brokenID), withIntermediateDirectories: true)
        try Data("{ not json".utf8).write(to: files.folder(brokenID).appendingPathComponent("person.json"))
        let silent = Person(name: "Quiet", kind: .saved, created: Date(), modelHash: zeros, eWespk: nil,
                            mask: nil, quality: Quality(speechS: 0))
        try files.write(silent, wav: ClipTools.encodeWAV([Float](repeating: 0, count: 64_000))) // no speech at all

        let s = PeopleStore(files: files, prepareLogs: false)
        await s.loadAndRebuild(model: model, encoder: nil, ownWindowCos: nil)
        XCTAssertEqual(s.brokenFolders, [brokenID.uuidString], "unreadable files: shown and skipped")
        XCTAssertEqual(s.status[good.id], .ready, "the others still load")
        XCTAssertEqual(s.status[silent.id], .needsReAdding, "es_enroll failed (ES_ERR_ENROLL_SHORT)")
    }
}
