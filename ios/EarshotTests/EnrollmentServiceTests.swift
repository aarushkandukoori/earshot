// EnrollmentServiceTests.swift - the 6 checks, the verdict rule, the own-voice path and the
// no-WeSpeaker path (30-ios 8.3), with the stub engine and a fake WeSpeaker.

import XCTest
@testable import Earshot

final class EnrollmentServiceTests: XCTestCase {
    private var model: EarshotModel!
    private var thresholds: Thresholds!

    override func setUpWithError() throws {
        model = try TestSupport.model()
        thresholds = try TestSupport.contractThresholds()
    }

    /// clip (>= 4 s) -> `clipVector`; 1.5 s windows -> `windowVector`.
    private func fake(clip clipVector: [Float] = TestSupport.basis(0),
                      window windowVector: [Float] = TestSupport.basis(0)) -> FakeEncoder {
        FakeEncoder { pcm in pcm.count == ClipTools.winLen ? windowVector : clipVector }
    }

    private func service(_ encoder: SpeakerEmbedding?, active: Bool = true) -> EnrollmentService {
        EnrollmentService(model: model, encoder: encoder, thresholds: thresholds, appIsActive: { active })
    }

    private func person(_ name: String, kind: PersonKind, e: [Float]?) -> Person {
        Person(name: name, kind: kind, created: Date(), modelHash: model.modelHash, eWespk: e, mask: nil,
               quality: Quality(speechS: 6))
    }

    private func verdicts(_ r: EnrollmentResult) -> [CheckResult.Kind: Verdict] {
        Dictionary(uniqueKeysWithValues: r.checks.map { ($0.kind, $0.verdict) })
    }

    /// 10 s capture with speech from 1 s to 9 s: a clean single talker.
    private let goodCapture = TestSupport.speechLike(seconds: 10, from: 1, to: 9)

    func testCleanTargetPasses() async throws {
        let own = person("You", kind: .own, e: TestSupport.basis(5))
        let r = try await service(fake()).enroll(pcm16k: goodCapture, role: .target, own: own, saved: [])
        XCTAssertEqual(r.verdict, .pass, "\(r.checks)")
        XCTAssertNil(r.message)
        XCTAssertEqual(r.checks.map(\.kind), [.background, .clipping, .oneVoice, .notYou, .speech, .distinct],
                       "the 6 checks in step order")
        XCTAssertNotNil(r.voiceprint)
        XCTAssertEqual(r.voiceprint?.hasWeSpeaker, true)
        XCTAssertEqual(r.clip.count % 8_000, 0, "rounded to 0.5 s")
        XCTAssertLessThanOrEqual(r.clip.count, 160_000)
        XCTAssertEqual(r.mask, model.vad(r.clip), "no window sounded like the user, so the mask is es_vad(clip)")
        XCTAssertEqual(r.quality.consistency ?? 0, 1, accuracy: 1e-9)
        XCTAssertEqual(r.quality.ownCos ?? 1, 0, accuracy: 1e-9)
        XCTAssertGreaterThanOrEqual(r.quality.speechS, 5)
        XCTAssertEqual(r.quality.snrDB, r.checks.first { $0.kind == .background }?.value)
    }

    func testShortSpeechWarnsAcceptedWillBeOK() async throws {
        // About 4.5 s of kept frames: inside the 4 to 5 s warn band.
        let x = TestSupport.speechLike(seconds: 9, from: 1, to: 6.6)
        let r = try await service(fake()).enroll(pcm16k: x, role: .target, own: nil, saved: [])
        let speech = r.checks.first { $0.kind == .speech }!
        XCTAssertEqual(speech.verdict, thresholds.speech(speech.value ?? 0))
        XCTAssertEqual(speech.verdict, .warn, "speech \(speech.value ?? -1) s")
        XCTAssertEqual(r.verdict, .warn)
        XCTAssertEqual(r.message, Copy.acceptedWillBeOK)
        XCTAssertNotNil(r.voiceprint, "a warn is still accepted")
    }

    func testOwnVoiceIsRejectedAndRemovedFromTheMask() async throws {
        let own = person("You", kind: .own, e: TestSupport.basis(0)) // same direction as the capture
        let r = try await service(fake()).enroll(pcm16k: goodCapture, role: .target, own: own, saved: [])
        XCTAssertEqual(verdicts(r)[.notYou], .fail)
        XCTAssertEqual(r.verdict, .fail)
        XCTAssertEqual(r.message, Copy.soundedLikeYou, "the first failing check in step order")
        XCTAssertNil(r.voiceprint)
        // Every window was over own_window_cos, so every frame inside one was zeroed; only frame 0
        // (midpoint -192, before every window) can be left.
        XCTAssertLessThanOrEqual(ClipTools.keptFrames(r.mask).count, 1)
    }

    func testMoreThanOneVoiceFails() async throws {
        let r = try await service(fake(window: TestSupport.basis(1))).enroll(pcm16k: goodCapture, role: .target,
                                                                           own: nil, saved: [])
        XCTAssertEqual(verdicts(r)[.oneVoice], .fail)
        XCTAssertEqual(r.message, Copy.moreThanOneVoice)
    }

    func testDuplicateAsksButDoesNotFail() async throws {
        let maya = person("Maya", kind: .tonight, e: TestSupport.basis(0))
        let other = person("Sam", kind: .saved, e: TestSupport.basis(7))
        let r = try await service(fake()).enroll(pcm16k: goodCapture, role: .target, own: nil, saved: [other, maya])
        XCTAssertEqual(r.duplicate?.name, "Maya")
        XCTAssertEqual(r.checks.first { $0.kind == .distinct }?.message, Copy.soundsLike("Maya"))
        XCTAssertEqual(r.verdict, .pass, "the duplicate question is a prompt, not a warn")
    }

    func testWithoutWeSpeakerThreeChecksAreNotChecked() async throws {
        let own = person("You", kind: .own, e: TestSupport.basis(0))
        let r = try await service(nil).enroll(pcm16k: goodCapture, role: .target, own: own, saved: [own])
        let v = verdicts(r)
        XCTAssertEqual(v[.oneVoice], .notChecked)
        XCTAssertEqual(v[.notYou], .notChecked)
        XCTAssertEqual(v[.distinct], .notChecked)
        XCTAssertEqual(r.verdict, .pass)
        XCTAssertNil(r.eWespk)
        XCTAssertEqual(r.voiceprint?.hasWeSpeaker, false)
        XCTAssertEqual(r.mask, model.vad(r.clip), "no own-window removal without WeSpeaker")
        XCTAssertNil(r.quality.consistency)
        XCTAssertNil(r.quality.ownCos)
    }

    func testTooShortNoisyAndClippedCaptures() async throws {
        let short = TestSupport.speechLike(seconds: 4, from: 0.5, to: 3.5)
        let rs = try await service(fake()).enroll(pcm16k: short, role: .target, own: nil, saved: [])
        XCTAssertEqual(rs.verdict, .fail)
        XCTAssertEqual(rs.trim.failure, .tooShort)
        XCTAssertEqual(rs.message, Copy.keepPhoneOnThem)
        XCTAssertEqual(rs.checks.count, 6)

        var clipped = goodCapture
        for i in stride(from: 20_000, to: 140_000, by: 50) { clipped[i] = 0.99 } // 1.5 % of samples
        let rc = try await service(fake()).enroll(pcm16k: clipped, role: .target, own: nil, saved: [])
        XCTAssertEqual(verdicts(rc)[.clipping], .fail)
        XCTAssertEqual(rc.message, Copy.tooLoud)

        // Loud noise under and between the syllables: speech barely above the background. At 0.17
        // es_vad still finds speech (quality.py: 168 kept frames, sbr 5.78 dB, a warn); at 0.2 it
        // finds none, which testSilentCaptureAsksForMoreSpeechNotLessNoise covers.
        let noisy = goodCapture.enumerated().map { i, v in v + 0.17 * Float(sin(Double(i) * 1.7)) }
        XCTAssertGreaterThan(ClipTools.keptFrames(model.vad(noisy)).count, 0, "test setup: speech frames exist")
        let rn = try await service(fake()).enroll(pcm16k: noisy, role: .target, own: nil, saved: [])
        let bg = rn.checks.first { $0.kind == .background }!
        let sbr = try XCTUnwrap(bg.value, "sbr is defined when speech frames exist")
        XCTAssertEqual(bg.verdict, thresholds.sbr(sbr))
        XCTAssertNotEqual(bg.verdict, .pass, "sbr \(sbr) dB")
    }

    func testSilentCaptureAsksForMoreSpeechNotLessNoise() async throws {
        // Nobody talked: es_vad finds no speech frame, so sbr is undefined (quality.py gives None).
        // The background row is "not checked" and the speech fail leads (review finding, I-2).
        let silent = [Float](repeating: 0.001, count: 10 * 16_000)
        XCTAssertNil(ClipTools.sbrDB(silent, mask0: model.vad(silent)), "test setup: no speech frame")
        let r = try await service(fake()).enroll(pcm16k: silent, role: .target, own: nil, saved: [])
        XCTAssertEqual(r.verdict, .fail)
        XCTAssertEqual(r.message, Copy.keepPhoneOnThem)
        XCTAssertEqual(verdicts(r)[.background], .notChecked)
        XCTAssertNil(r.quality.snrDB)

        let own = try await service(fake()).enroll(pcm16k: silent, role: .own, own: nil, saved: [])
        XCTAssertEqual(own.verdict, .fail)
        XCTAssertEqual(own.message, Copy.keepReading, "not 'Too noisy'")
        XCTAssertEqual(verdicts(own)[.background], .notChecked)
    }

    func testBackgroundUsesCPUOnlyFlag() async throws {
        let enc = fake()
        let r = try await service(enc, active: false).enroll(pcm16k: goodCapture, role: .target, own: nil, saved: [])
        let seen = await enc.appActiveSeen
        XCTAssertFalse(seen.isEmpty)
        XCTAssertTrue(seen.allSatisfy { !$0 }, "every embed call was told the app is not active")
        let calls = await enc.calls
        XCTAssertEqual(calls, 1 + ClipTools.windows(clipLen: r.clip.count).count, "clip + one call per 1.5 s window")
    }

    // MARK: - Own voice

    func testOwnVoicePassesWithTheBestTenSecondWindow() async throws {
        let x = TestSupport.speechLike(seconds: 20, from: 2, to: 17)
        let enc = fake()
        let r = try await service(enc).enroll(pcm16k: x, role: .own, own: nil, saved: [])
        XCTAssertEqual(r.verdict, .pass, "\(r.checks)")
        XCTAssertEqual(r.checks.map(\.kind), [.background, .speech], "own voice runs steps 1, 3 and 8 only")
        XCTAssertEqual(r.clip.count, 160_000, "the best 10 s window")
        XCTAssertNotNil(r.trim.windowStart)
        XCTAssertEqual(r.mask, model.vad(r.clip))
        let speechRow = try XCTUnwrap(r.checks.first { $0.kind == .speech })
        XCTAssertGreaterThanOrEqual(speechRow.value ?? 0, 8, "the 8 s rule counts the full capture")
        // person.json speech_s is the voiceprint's own value (contract 6 step 10), at most the 10 s clip.
        let vp = try XCTUnwrap(r.voiceprint)
        XCTAssertEqual(r.quality.speechS, ClipTools.roundSeconds(Double(vp.speechSeconds)))
        XCTAssertLessThanOrEqual(r.quality.speechS, 10)
        XCTAssertNotNil(r.eWespk)
        let calls = await enc.calls
        XCTAssertEqual(calls, 1, "one embedding of the clip, no windows")
    }

    func testOwnVoiceNeedsEightSecondsOfSpeech() async throws {
        let x = TestSupport.speechLike(seconds: 12, from: 1, to: 8)
        let r = try await service(fake()).enroll(pcm16k: x, role: .own, own: nil, saved: [])
        XCTAssertEqual(r.verdict, .fail)
        XCTAssertEqual(r.message, Copy.keepReading)
        XCTAssertNil(r.voiceprint)
    }
}
