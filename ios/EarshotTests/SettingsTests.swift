// SettingsTests.swift - the Settings values and ranges (30-ios 13, 12.1: UI ranges equal the
// es_params ranges), how they fill EngineParams, their storage, and two I-3 fixes of I-2 leftovers:
// quick-add playback stops when the route goes away, and screens never show raw system errors.

import AVFoundation
import XCTest
@testable import Earshot

final class SettingsTests: XCTestCase {
    func testDefaultsAreEsParamsDefault() {
        let s = ListenSettings()
        let p = s.applied(to: EngineParams())
        let d = EngineParams()
        XCTAssertEqual(p.voiceDB, d.voiceDB)
        XCTAssertEqual(p.roomDB, -60, "AirPods default: room sound Off")
        XCTAssertEqual(p.maxGainDB, d.maxGainDB)
        XCTAssertEqual(p.outGainDB, d.outGainDB)
        XCTAssertEqual(p.duckOwnVoice, true)
        XCTAssertEqual(p.gate, true)
        XCTAssertEqual(p.agc, true)
        XCTAssertTrue(s.alarmPassthrough, "alarm passthrough defaults on")
    }

    func testRangesAreClamped() {
        var s = ListenSettings()
        s.voiceDB = 9
        s.maxGainDB = 30
        s.outGainDB = 3
        s.roomDB = -20
        let c = s.clamped()
        XCTAssertEqual(c.voiceDB, 6)
        XCTAssertEqual(c.maxGainDB, 18)
        XCTAssertEqual(c.outGainDB, 0)
        XCTAssertEqual(c.roomDB, -24, "room snaps to Off / -24 / -15 / -8")
        s.voiceDB = -40
        s.maxGainDB = -1
        s.outGainDB = -100
        s.roomDB = -70
        let low = s.clamped()
        XCTAssertEqual(low.voiceDB, -10)
        XCTAssertEqual(low.maxGainDB, 0)
        XCTAssertEqual(low.outGainDB, -40)
        XCTAssertEqual(low.roomDB, -60)
        XCTAssertLessThanOrEqual(ListenSettings.voiceRange.upperBound + ListenSettings.maxGainRange.upperBound, 24,
                                 "voice + max gain never passes the engine's 24 dB")
    }

    func testAppliedKeepsSlotGainsAndBypass() {
        var p = EngineParams()
        p.slotGain = [0, 1, 1]
        p.bypass = true
        var s = ListenSettings()
        s.voiceDB = -4
        s.roomDB = -15
        s.gate = false
        let out = s.applied(to: p)
        XCTAssertEqual(out.slotGain, [0, 1, 1])
        XCTAssertTrue(out.bypass)
        XCTAssertEqual(out.voiceDB, -4)
        XCTAssertEqual(out.roomDB, -15)
        XCTAssertFalse(out.gate)
    }

    @MainActor
    func testStoreKeepsValuesAcrossLaunches() throws {
        let suite = "earshot.tests.settings.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = SettingsStore(defaults: defaults)
        XCTAssertEqual(store.value, ListenSettings(), "first launch: defaults")
        var seen: [ListenSettings] = []
        store.onChange = { seen.append($0) }
        store.value.voiceDB = 12 // out of range: stored as +6
        XCTAssertEqual(store.value.voiceDB, 6)
        XCTAssertEqual(seen.last?.voiceDB, 6)
        store.value.duckOwnVoice = false
        let again = SettingsStore(defaults: defaults)
        XCTAssertEqual(again.value.voiceDB, 6)
        XCTAssertFalse(again.value.duckOwnVoice)
    }

    @MainActor
    func testControllerKeepsSettingsAndBypassInParams() throws {
        let audio = AudioController()
        var s = ListenSettings()
        s.roomDB = -8
        s.voiceDB = 3
        audio.applySettings(s)
        XCTAssertEqual(audio.params.roomDB, -8)
        XCTAssertEqual(audio.params.voiceDB, 3)
        audio.setBypass(true)
        XCTAssertTrue(audio.params.bypass)
        audio.applySettings(ListenSettings())
        XCTAssertTrue(audio.params.bypass, "settings never touch hold-to-compare")
        audio.setBypass(false)
        XCTAssertFalse(audio.params.bypass)
    }

    // MARK: - I-2 leftovers fixed in I-3

    func testQuickAddPlaybackStopsWhenTheRouteGoesAway() {
        XCTAssertTrue(QuickAddPlayer.shouldStop(reason: .oldDeviceUnavailable, newRouteAllowed: true),
                      "AirPods taken out or disconnected")
        XCTAssertTrue(QuickAddPlayer.shouldStop(reason: .newDeviceAvailable, newRouteAllowed: false),
                      "the clip must never move to a speaker")
        XCTAssertTrue(QuickAddPlayer.shouldStop(reason: .override, newRouteAllowed: false))
        XCTAssertFalse(QuickAddPlayer.shouldStop(reason: .categoryChange, newRouteAllowed: true),
                       "Earshot re-selecting the built-in mic lets the clip finish")
        XCTAssertFalse(QuickAddPlayer.shouldStop(reason: .newDeviceAvailable, newRouteAllowed: true))
    }

    func testScreensShowOnlyCopyTextForErrors() {
        XCTAssertEqual(AppError.userText(AppError.notForeground, fallback: Copy.startFailed), Copy.openToStart)
        XCTAssertEqual(AppError.userText(AppError.audioUnit("eh_start", -50), fallback: Copy.startFailed),
                       Copy.audioSetupFailed("eh_start", -50))
        let system = NSError(domain: NSOSStatusErrorDomain, code: 561_017_449,
                             userInfo: [NSLocalizedDescriptionKey: "The operation couldn't be completed."])
        XCTAssertEqual(AppError.userText(system, fallback: Copy.startFailed), Copy.startFailed)
        let engine = EarshotError(status: -2, message: "ES_ERR_FORMAT: bad header")
        XCTAssertEqual(AppError.userText(engine, fallback: Copy.captureStartFailed), Copy.captureStartFailed)
    }
}
