// OnboardingTests.swift - onboarding rules (30-ios 13): steps 2 and 5 cannot be skipped, Continue
// needs the permissions answered and an own voice, the order of the 8 steps, the stored "done" flag,
// the microphone tap check (20 dB, 30-ios 3.2) and the "Which AirPods" answer per port (3.4).

import XCTest
@testable import Earshot

final class OnboardingTests: XCTestCase {
    func testEightStepsInTheSpecOrder() {
        XCTAssertEqual(OnboardingFlow.total, 8)
        XCTAssertEqual(OnboardingStep.allCases, [.whatItIs, .permissions, .whichAirPods, .checklist, .yourVoice,
                                                 .measureDelay, .firstListen, .controls])
        XCTAssertEqual(OnboardingStep.allCases.map { $0.rawValue }, Array(1...8))
    }

    func testOnlyPermissionsAndYourVoiceCannotBeSkipped() {
        let fixed = OnboardingStep.allCases.filter { !OnboardingFlow.canSkip($0) }
        XCTAssertEqual(fixed, [.permissions, .yourVoice])
    }

    func testContinueGates() {
        XCTAssertFalse(OnboardingFlow.canContinue(.permissions, permissionsAnswered: false, hasOwnVoice: true))
        XCTAssertTrue(OnboardingFlow.canContinue(.permissions, permissionsAnswered: true, hasOwnVoice: false))
        XCTAssertFalse(OnboardingFlow.canContinue(.yourVoice, permissionsAnswered: true, hasOwnVoice: false))
        XCTAssertTrue(OnboardingFlow.canContinue(.yourVoice, permissionsAnswered: true, hasOwnVoice: true))
        for s in OnboardingStep.allCases where s != .permissions && s != .yourVoice {
            XCTAssertTrue(OnboardingFlow.canContinue(s, permissionsAnswered: false, hasOwnVoice: false), "\(s)")
        }
    }

    /// Walks the whole flow: nothing past step 5 is reachable without an own voice.
    func testListeningStepsNeedTheOwnVoice() {
        var f = OnboardingFlow()
        XCTAssertTrue(f.next(permissionsAnswered: false, hasOwnVoice: false))       // 1 -> 2
        XCTAssertEqual(f.step, .permissions)
        XCTAssertFalse(f.skip(), "step 2 cannot be skipped")
        XCTAssertFalse(f.next(permissionsAnswered: false, hasOwnVoice: false), "unanswered permissions")
        XCTAssertTrue(f.next(permissionsAnswered: true, hasOwnVoice: false))        // 2 -> 3
        XCTAssertTrue(f.skip())                                                     // 3 -> 4
        XCTAssertTrue(f.skip())                                                     // 4 -> 5
        XCTAssertEqual(f.step, .yourVoice)
        XCTAssertFalse(f.skip(), "step 5 cannot be skipped")
        XCTAssertFalse(f.next(permissionsAnswered: true, hasOwnVoice: false), "no own voice yet")
        XCTAssertEqual(f.step, .yourVoice)
        XCTAssertTrue(f.next(permissionsAnswered: true, hasOwnVoice: true))         // 5 -> 6
        XCTAssertTrue(f.skip())                                                     // 6 -> 7
        XCTAssertTrue(f.skip())                                                     // 7 -> 8
        XCTAssertEqual(f.step, .controls)
        XCTAssertFalse(f.finished)
        XCTAssertTrue(f.next(permissionsAnswered: true, hasOwnVoice: true))         // finish
        XCTAssertTrue(f.finished)
        XCTAssertFalse(f.next(permissionsAnswered: true, hasOwnVoice: true), "nothing after finishing")
    }

    func testBackStopsAtTheFirstStep() {
        var f = OnboardingFlow()
        f.back()
        XCTAssertEqual(f.step, .whatItIs)
        f.next(permissionsAnswered: true, hasOwnVoice: true)
        f.next(permissionsAnswered: true, hasOwnVoice: true)
        XCTAssertEqual(f.step, .whichAirPods)
        f.back()
        XCTAssertEqual(f.step, .permissions)
    }

    @MainActor
    func testDoneIsRememberedAcrossLaunches() throws {
        let suite = "earshot.tests.onboarding.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let first = OnboardingStore(defaults: defaults)
        XCTAssertFalse(first.done, "first launch shows onboarding")
        first.finish()
        XCTAssertTrue(first.done)
        XCTAssertTrue(OnboardingStore(defaults: defaults).done, "the next launch skips it")
    }

    func testPermissionsAnswered() {
        XCTAssertTrue(PermissionAnswer.allAnswered([.allowed, .notAllowed, .allowed]), "a no is an answer")
        XCTAssertFalse(PermissionAnswer.allAnswered([.allowed, .notAsked, .allowed]))
    }

    func testTapCheckNeeds20dB() {
        var t = TapTest()
        XCTAssertFalse(t.passed, "no readings")
        t.add(InputMeter(peakDB: -100, rmsDB: -100))
        XCTAssertNil(t.quietDB, "the -100 dB floor (no audio yet) is ignored")
        t.add(InputMeter(peakDB: -50, rmsDB: -60))
        t.add(InputMeter(peakDB: -45, rmsDB: -55))
        XCTAssertFalse(t.passed, "15 dB above the quietest RMS")
        t.add(InputMeter(peakDB: -40.5, rmsDB: -50))
        XCTAssertFalse(t.passed)
        t.add(InputMeter(peakDB: -40, rmsDB: -48))
        XCTAssertEqual(t.riseDB, 20, accuracy: 1e-6)
        XCTAssertTrue(t.passed, "exactly 20 dB passes")
    }

    func testWhichAirPodsAnswerIsKeptPerPort() throws {
        let suite = "earshot.tests.airpods.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = AirPodsModelStore(defaults: defaults)
        XCTAssertNil(store.model(forPort: "pods-a"))
        store.set(.pro1, forPort: "pods-a")
        store.set(.pro2, forPort: "pods-b")
        store.set(.pro2, forPort: "") // no port: ignored
        XCTAssertEqual(AirPodsModelStore(defaults: defaults).model(forPort: "pods-a"), .pro1)
        XCTAssertEqual(store.model(forPort: "pods-b"), .pro2)
        XCTAssertNil(store.model(forPort: ""))
    }
}
