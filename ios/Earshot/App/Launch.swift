// Launch.swift - the launch sequence that runs before Start is enabled (30-ios 9.4).
//
// Runs once per launch off the main actor. Each failure has one fixed outcome:
//   1. load earshot.esw             missing or damaged -> blocking error screen, no audio ever
//   2. es_selftest(testvec_short)   fails -> Start stays disabled (contract 3.3)
//   3. thresholds.json              missing, unreadable or version != 1 -> blocking;
//                                   made for another model -> Diagnostics warning, file still used
//   4. SpeakerEncoder.selfTest()    .cpuAndGPU, then .cpuOnly, else use_wespk = 0 with a notice; never blocks
//   5. purge and rebuild people     (PeopleStore.loadAndRebuild) a person whose files fail to load is
//                                   "needs re-adding"; a file without `mask` gets it recomputed
//   6. Start is enabled after steps 1 to 4; a person becomes selectable once rebuilt.
//
// The stub engine (until E-2) accepts an empty weights buffer and reports a model hash of 64 zeros,
// so with the stub the sequence also runs without bundled files (`make models` not run yet).

import Foundation

/// What the launch sequence found. Shown in Diagnostics.
struct LaunchReport: @unchecked Sendable {
    var model: EarshotModel?
    var blockingError: String?            // non-nil: error screen, audio never starts
    var engineKind = "unknown"            // "stub" or "real" (Phase 0 records this)
    var modelHash = ""
    var selfTestPassed = false
    var selfTestMaxError: Float?
    var selfTestMessage = ""
    var thresholds: Thresholds?           // nil only with the stub and no bundled file
    var thresholdsVersion: Int?
    var thresholdsWarning: String?
    var useWeSpeaker = false              // set by the encoder self-test (step 4)
    var encoderMessage = Copy.voiceCheckPending
    var encoderCosine: Double?
    var encoderUnits = "-"
    var profileExpiry: Date?              // embedded provisioning profile, read once (30-ios 2.3)
    var notes: [String] = []

    /// Start is enabled after steps 1 to 4 (step 4 never blocks).
    var startEnabled: Bool { blockingError == nil && model != nil && selfTestPassed }
}

enum Launch {
    /// Returns a bundled resource by name and extension, or nil when it is missing.
    typealias ResourceLoader = (_ name: String, _ ext: String) -> Data?

    /// Reads from the app bundle (Resources/Models is copied flat into the bundle).
    static let bundleLoader: ResourceLoader = { name, ext in
        guard let url = Bundle.main.url(forResource: name, withExtension: ext) else { return nil }
        return try? Data(contentsOf: url)
    }

    /// Steps 1 to 3. The parameters exist so LaunchTests can force every branch.
    static func run(load: ResourceLoader = bundleLoader,
                    allowStubWithoutFiles: Bool = true,
                    selfTest: (EarshotModel, Data) throws -> Float = { try $0.selfTest(testVector: $1) })
        -> LaunchReport {
        var report = LaunchReport()

        // Step 1 - weights.
        let model: EarshotModel
        if let weights = load("earshot", "esw") {
            do {
                model = try EarshotModel(weights: weights)
            } catch {
                report.blockingError = Copy.modelMissing
                report.notes.append("es_model_load: \(error)")
                return report
            }
        } else if allowStubWithoutFiles, let stub = try? EarshotModel(weights: Data()), stub.isStub {
            model = stub
            report.notes.append("earshot.esw not bundled - running the stub engine")
        } else {
            report.blockingError = Copy.modelMissing
            report.notes.append("earshot.esw not found in the app bundle")
            return report
        }
        report.model = model
        report.modelHash = model.modelHash
        report.engineKind = model.isStub ? "stub" : "real"
        // Only the stub may run without the other files; tests turn this off to drive the real-engine branches.
        let stubFallback = allowStubWithoutFiles && model.isStub

        // Step 2 - engine self-test (no audio if it fails).
        let testVector = load("testvec_short", "est") ?? (stubFallback ? Data() : nil)
        if let testVector {
            do {
                let maxErr = try selfTest(model, testVector)
                report.selfTestPassed = true
                report.selfTestMaxError = maxErr
                report.selfTestMessage = "passed, max error \(maxErr)"
            } catch {
                report.selfTestPassed = false
                report.selfTestMessage = "\(Copy.selfTestFailed) (\(error))"
            }
        } else {
            report.selfTestPassed = false
            report.selfTestMessage = "\(Copy.selfTestFailed) (testvec_short.est not found)"
        }

        // Step 3 - thresholds.json (decoded in full: enrollment reads every value from it).
        if let data = load("thresholds", "json") {
            guard let thresholds = try? Thresholds.decode(data) else {
                report.blockingError = Copy.thresholdsMissing
                report.notes.append("thresholds.json unreadable or version is not 1")
                return report
            }
            report.thresholds = thresholds
            report.thresholdsVersion = thresholds.version
            if thresholds.modelHash != model.modelHash {
                report.thresholdsWarning = Copy.thresholdsOtherModel
                report.notes.append("thresholds.json model_hash \(thresholds.modelHash.prefix(12)) differs from the model")
            }
        } else if stubFallback {
            report.notes.append("thresholds.json not bundled (fine for the stub; enrollment stays off)")
        } else {
            report.blockingError = Copy.thresholdsMissing
            report.notes.append("thresholds.json not found in the app bundle")
            return report
        }
        return report
    }

    /// Step 4 - the WeSpeaker self-test. Never blocks: on any failure (no package, a load error,
    /// both compute units below cosine_min) the app runs with use_wespk = 0 and shows the notice.
    /// `makeEncoder` returns nil when the package is not bundled.
    static func checkEncoder(_ report: LaunchReport,
                             makeEncoder: () throws -> SpeakerEncoder?) async -> (LaunchReport, SpeakerEncoder?) {
        var r = report
        guard r.blockingError == nil, r.model != nil else { return (r, nil) }
        let encoder: SpeakerEncoder?
        do {
            encoder = try makeEncoder()
        } catch {
            r.notes.append("WeSpeaker package failed to load: \(error)")
            encoder = nil
        }
        guard let encoder else {
            r.useWeSpeaker = false
            r.encoderMessage = Copy.voiceCheckUnavailable
            r.notes.append("WeSpeaker not available - enrolling without the voice check")
            return (r, nil)
        }
        let passed = await encoder.selfTest()
        r.encoderCosine = await encoder.selfTestCosine
        r.encoderUnits = await encoder.selfTestUnits
        r.useWeSpeaker = passed
        r.encoderMessage = passed ? Copy.voiceCheckReady : Copy.voiceCheckUnavailable
        r.notes.append("WeSpeaker self-test \(passed ? "passed" : "failed"), cosine "
                       + (r.encoderCosine.map { String(format: "%.5f", $0) } ?? "-") + ", units \(r.encoderUnits)")
        return (r, passed ? encoder : nil)
    }

    /// The real encoder from the bundled compiled package, or nil when `make models` did not run.
    static func bundledEncoder() throws -> SpeakerEncoder? {
        guard let url = SpeakerEncoder.bundledModelURL else { return nil }
        return try SpeakerEncoder(modelURL: url)
    }
}
