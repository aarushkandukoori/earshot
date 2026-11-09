// EarshotApp.swift - the @main SwiftUI app (30-ios 1).
//
// Creates the long-lived objects once (audio controller, people store, selector, settings, launch
// state), runs the launch sequence off the main actor, wires people, the table and the settings to
// the audio side, and forwards scene phase changes (listening may only start in the foreground;
// tonight people are purged on foreground). Main actor.
// Lead integration: LaunchState also writes one line with the launch result to the unified log
// (subsystem com.aarushkandukoori.earshot, category launch), so a simulator smoke run or Console on a
// Mac can check the real engine's self-test without opening Diagnostics. Engine facts only: no audio,
// no names. Never called from the audio thread.

import os
import SwiftUI
import UIKit

@main
struct EarshotApp: App {
    @StateObject private var audio = AudioController()
    @StateObject private var launch = LaunchState()
    @StateObject private var people = PeopleStore()
    @StateObject private var selector = Selector()
    @StateObject private var settings = SettingsStore()
    @StateObject private var phase0 = Phase0Runner() // I-1: lives here so a long run survives leaving its screen
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(audio)
                .environmentObject(launch)
                .environmentObject(people)
                .environmentObject(selector)
                .environmentObject(settings)
                .environmentObject(phase0)
                .task { await launch.runOnce(audio: audio, people: people, selector: selector, settings: settings) }
        }
        .onChange(of: scenePhase) { _, phase in
            let active = phase == .active
            audio.sceneDidChange(active: active)
            launch.sceneDidChange(active: active)
            if active { people.purgeExpired() } // 30-ios 9.2: on every foreground
        }
    }
}

/// Holds the launch report and the services built from it, for the screens.
@MainActor
final class LaunchState: ObservableObject {
    @Published private(set) var report: LaunchReport? = nil // nil while steps 1 to 4 run
    private(set) var encoder: SpeakerEncoder?               // nil without WeSpeaker
    private(set) var enrollment: EnrollmentService?         // nil without thresholds.json
    private var started = false

    var model: EarshotModel? { report?.model }

    func runOnce(audio: AudioController, people: PeopleStore, selector: Selector, settings: SettingsStore) async {
        if started { return }
        started = true
        // Settings reach the engine params at once and after every change (30-ios 13).
        audio.applySettings(settings.value)
        settings.onChange = { [weak audio] s in audio?.applySettings(s) }
        // Steps 1 to 3 at utility priority, off the main actor (30-ios 9.4).
        let firstSteps = await Task.detached(priority: .utility) { () -> LaunchReport in
            var r = Launch.run()
            r.profileExpiry = ProvisioningProfile.expiryDate() // read once per launch (30-ios 2.3)
            return r
        }.value
        // Step 4: loading Core ML and the self-test also run off the main actor (the encoder is an actor).
        let (result, encoder) = await Task.detached(priority: .utility) {
            await Launch.checkEncoder(firstSteps, makeEncoder: Launch.bundledEncoder)
        }.value
        self.encoder = encoder
        report = result
        for line in result.notes { audio.note("launch: \(line)") }
        Self.log(result)
        guard result.startEnabled, let model = result.model else { return }

        audio.attach(model: model)
        audio.setUseWeSpeaker(result.useWeSpeaker)
        if let thresholds = result.thresholds {
            enrollment = EnrollmentService(model: model, encoder: encoder, thresholds: thresholds)
        }
        // People reach the engine through these hooks; the Selector owns the table (I-3).
        people.onOwnVoiceChanged = { [weak audio] vp in audio?.setOwnVoice(vp) }
        selector.attach(audio: audio, people: people, model: model, enrollment: enrollment)
        selector.lockedCos = Float(result.thresholds?.lockedCos ?? 0) // I-4 locked check (30-ios 12.3)
        audio.onStop = { [weak people, weak selector] time in
            people?.sessionStopped(at: time)   // tonight people expire 12 h after this
            selector?.sessionStopped()         // learned directions to yaw_deg
        }
        people.startTimers(isListening: { [weak audio] in audio?.isListening ?? false })
        // Step 5 runs after Start is enabled; each person becomes selectable once rebuilt.
        let active = UIApplication.shared.applicationState == .active
        await people.loadAndRebuild(model: model, encoder: encoder, ownWindowCos: result.thresholds?.ownWindowCos,
                                    appActive: active)
        if let error = people.lastError { audio.note("people: \(error)") }
    }

    private static let logger = Logger(subsystem: "com.aarushkandukoori.earshot", category: "launch")

    /// One unified-log line per launch (30-ios 9.4 result). Every value is public: they describe the
    /// bundled model and engine, nothing about the user.
    private static func log(_ r: LaunchReport) {
        let maxError = r.selfTestMaxError.map { String(format: "%.3g", $0) } ?? "-"
        let text = "engine \(r.engineKind), model \(r.modelHash.prefix(12)), self-test "
            + (r.selfTestPassed ? "passed, max error \(maxError)" : "FAILED: \(r.selfTestMessage)")
            + ", thresholds v\(r.thresholdsVersion.map(String.init) ?? "-")"
            + (r.thresholdsWarning == nil ? "" : " (other model)")
            + ", WeSpeaker \(r.useWeSpeaker ? "on" : "off") units \(r.encoderUnits)"
            + ", start \(r.startEnabled ? "enabled" : "disabled")"
            + (r.blockingError.map { ", blocking: \($0)" } ?? "")
        logger.notice("launch: \(text, privacy: .public)")
    }

    /// The encoder uses .cpuOnly whenever the app is not active (contract 6).
    func sceneDidChange(active: Bool) {
        guard let encoder else { return }
        Task { await encoder.setAppActive(active) }
    }
}
