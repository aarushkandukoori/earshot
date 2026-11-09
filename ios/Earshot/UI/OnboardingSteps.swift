// OnboardingSteps.swift - onboarding steps 5 to 7 (30-ios 13): your voice, measure the delay, first
// listen; plus the small Start button they share. Steps 1 to 4 and the rules are in
// OnboardingView.swift, step 8 (ControlsCard) is in HelpView.swift.
//
// Main actor (SwiftUI). Every string comes from Copy.

import SwiftUI

/// Starts listening from inside onboarding (the microphone check, own voice and delay steps need
/// the engine running; Earshot stays muted until the own voice exists, 30-ios 6 "Setup mode").
struct StartListeningButton: View {
    @EnvironmentObject private var audio: AudioController
    let title: String
    @Binding var errorText: String?

    var body: some View {
        Button(title) {
            Task {
                do { try await audio.start(); errorText = nil }
                catch { errorText = AppError.userText(error, fallback: Copy.startFailed) }
            }
        }
        .buttonStyle(.borderedProminent)
        .disabled(!audio.canStart)
    }
}

/// Step 5: the own voice, mandatory before listening. OwnVoiceView opens as a sheet, so its
/// "Not now" closes only that sheet, never onboarding.
struct YourVoiceStep: View {
    @EnvironmentObject private var people: PeopleStore
    @EnvironmentObject private var launch: LaunchState
    @State private var recording = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(Copy.yourVoiceWhy).font(.title3)
            if people.own != nil {
                Label(Copy.ownVoiceSaved, systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                Button(Copy.reRecordMyVoice) { recording = true }.buttonStyle(.bordered)
            } else {
                Text(Copy.ownVoiceRequired).foregroundStyle(.secondary)
                Button(Copy.recordMyVoice) { recording = true }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(launch.enrollment == nil)
            }
        }
        .sheet(isPresented: $recording) { NavigationStack { OwnVoiceView() } }
    }
}

/// Step 6: measure the delay (30-ios 7), the result, and one sentence on why it matters.
struct MeasureStep: View {
    @EnvironmentObject private var audio: AudioController
    @State private var errorText: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(Copy.measureWhy)
            if audio.state == .running && audio.route.allowed {
                MeasureDelayButton()
                Text(audio.latency.text).font(.headline)
                Text(Copy.earshotShare(Int(audio.engineLatencyMS.rounded()))).font(.footnote)
            } else if audio.isListening {
                Text(Copy.connectAirPodsFirst)
            } else {
                StartListeningButton(title: Copy.start, errorText: $errorText)
            }
            if let errorText { Text(errorText).font(.footnote).foregroundStyle(.red) }
        }
    }
}

/// Step 7: 10 s in Everyone mode with hold-to-compare (bypass).
struct FirstListenStep: View {
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var selector: Selector
    @State private var secondsLeft: Int? = nil
    @State private var played = false
    @State private var errorText: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(Copy.firstListenHowTo)
            if !audio.isListening {
                StartListeningButton(title: Copy.start, errorText: $errorText)
            } else if let left = secondsLeft {
                Text(Copy.secondsLeft(left)).font(.title2.monospacedDigit())
                HoldToCompare()
            } else {
                Button(Copy.playTenSeconds) { play() }
                    .buttonStyle(.borderedProminent)
                    .disabled(audio.state != .running && audio.state != .paused)
                if played { Text(Copy.firstListenDone) }
            }
            if let errorText { Text(errorText).font(.footnote).foregroundStyle(.red) }
        }
    }

    private func play() {
        if selector.table.mode != .everyone { selector.setMode(.everyone) }
        if audio.state == .paused { audio.resume() }
        secondsLeft = 10
        Task { @MainActor in
            while let left = secondsLeft, left > 0 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                secondsLeft = left - 1
            }
            secondsLeft = nil
            played = true
        }
    }
}
