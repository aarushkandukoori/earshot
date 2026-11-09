// SettingsView.swift - the Settings screen and the listening settings it edits (30-ios 13
// "Settings", 12.1 UI ranges; contract 4 es_params ranges).
//
// Main actor (SwiftUI). Three parts:
//   ListenSettings - plain struct: the values, their ranges (equal to the es_params ranges) and how
//                    they fill EngineParams; unit-tested
//   SettingsStore  - keeps ListenSettings in UserDefaults (a few bytes, no audio) and tells
//                    AudioController about every change (applySettings)
//   SettingsView   - voice level, room sound, mute while I talk, activity gate, voice leveling, alarm
//                    passthrough, Advanced (max gain, trim), re-record my voice, measure delay, About

import SwiftUI

/// The user's listening settings. Defaults are es_params_default (AirPods: room sound Off).
struct ListenSettings: Codable, Equatable {
    var voiceDB: Float = 0          // added to the AGC target, -10 ... +6 dB
    var roomDB: Float = -60         // room mix-back: -60 = Off, or -24 / -15 / -8 dB
    var duckOwnVoice = true         // "mute while I talk": -15 dB while p_self > 0.5
    var gate = true                 // activity gate
    var agc = true                  // slow voice leveling
    var alarmPassthrough = true     // read by the alarm listener (I-4); default on (DECISIONS M.10)
    var maxGainDB: Float = 18       // AGC ceiling, 0 ... 18 dB (Advanced)
    var outGainDB: Float = 0        // final trim, -40 ... 0 dB (Advanced)

    static let voiceRange: ClosedRange<Float> = -10...6
    static let roomChoices: [Float] = [-60, -24, -15, -8]
    static let maxGainRange: ClosedRange<Float> = 0...18
    static let trimRange: ClosedRange<Float> = -40...0

    /// Every value inside its es_params range; room snaps to the nearest of its four choices.
    func clamped() -> ListenSettings {
        var s = self
        s.voiceDB = min(max(voiceDB, Self.voiceRange.lowerBound), Self.voiceRange.upperBound)
        s.maxGainDB = min(max(maxGainDB, Self.maxGainRange.lowerBound), Self.maxGainRange.upperBound)
        s.outGainDB = min(max(outGainDB, Self.trimRange.lowerBound), Self.trimRange.upperBound)
        s.roomDB = Self.roomChoices.min { abs($0 - roomDB) < abs($1 - roomDB) } ?? -60
        return s
    }

    /// The engine params with these settings filled in. Slot gains, crossfade and bypass are not
    /// settings and are kept. voice + max gain is at most 6 + 18 = 24 dB, the engine's limit.
    func applied(to p: EngineParams) -> EngineParams {
        let s = clamped()
        var out = p
        out.voiceDB = s.voiceDB
        out.roomDB = s.roomDB
        out.duckOwnVoice = s.duckOwnVoice
        out.gate = s.gate
        out.agc = s.agc
        out.maxGainDB = s.maxGainDB
        out.outGainDB = s.outGainDB
        return out
    }
}

/// Keeps the settings across launches and pushes each change to the engine.
@MainActor
final class SettingsStore: ObservableObject {
    @Published var value: ListenSettings {
        didSet {
            let fixed = value.clamped()
            if fixed != value { value = fixed; return } // the second didSet saves the fixed value
            save()
            onChange?(value)
        }
    }

    /// AudioController.applySettings (set by EarshotApp).
    var onChange: (@MainActor (ListenSettings) -> Void)?

    private let defaults: UserDefaults
    private static let key = "earshot.settings.v1"

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.key),
           let s = try? JSONDecoder().decode(ListenSettings.self, from: data) {
            value = s.clamped()
        } else {
            value = ListenSettings() // first launch, or a stored value that no longer decodes
        }
    }

    private func save() {
        if let data = try? JSONEncoder().encode(value) { defaults.set(data, forKey: Self.key) }
    }
}

struct SettingsView: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var launch: LaunchState

    var body: some View {
        Form {
            Section {
                VoiceLevelSlider()
                RoomSoundPicker()
                Toggle(Copy.muteWhileITalk, isOn: $settings.value.duckOwnVoice)
                Toggle(Copy.activityGate, isOn: $settings.value.gate)
                Toggle(Copy.voiceLeveling, isOn: $settings.value.agc)
            }
            Section {
                Toggle(Copy.alarmPassthrough, isOn: $settings.value.alarmPassthrough)
            } footer: {
                Text(Copy.alarmPassthroughNote)
            }
            Section(Copy.advanced) {
                LabeledContent(Copy.maxGain, value: Copy.decibels(settings.value.maxGainDB))
                Slider(value: $settings.value.maxGainDB, in: ListenSettings.maxGainRange, step: 1)
                    .accessibilityLabel(Copy.maxGain)
                LabeledContent(Copy.outputTrim, value: Copy.decibels(settings.value.outGainDB))
                Slider(value: $settings.value.outGainDB, in: ListenSettings.trimRange, step: 1)
                    .accessibilityLabel(Copy.outputTrim)
            }
            Section {
                NavigationLink(Copy.reRecordMyVoice) { OwnVoiceView() }
                    .disabled(launch.enrollment == nil)
                MeasureDelayButton()
            }
            Section(Copy.about) {
                Text(Copy.disclaimer)
                Text(Copy.privacySummary)
                Text(Copy.creditsLookOnce)
                Text(Copy.creditsWeSpeaker)
                Text(Copy.licenses)
            }
            .font(.callout)
        }
        .navigationTitle(Copy.settings)
    }
}

/// Voice level -10 ... +6 dB (also on the Listen screen).
struct VoiceLevelSlider: View {
    @EnvironmentObject private var settings: SettingsStore

    var body: some View {
        VStack(alignment: .leading) {
            LabeledContent(Copy.voiceLevel, value: Copy.decibels(settings.value.voiceDB))
            Slider(value: $settings.value.voiceDB, in: ListenSettings.voiceRange, step: 1)
                .accessibilityLabel(Copy.voiceLevel)
        }
    }
}

/// Room sound Off / -24 / -15 / -8 dB (AirPods default Off; also on the Listen screen).
struct RoomSoundPicker: View {
    @EnvironmentObject private var settings: SettingsStore

    var body: some View {
        Picker(Copy.roomSound, selection: $settings.value.roomDB) {
            ForEach(ListenSettings.roomChoices, id: \.self) { db in
                Text(db <= -60 ? Copy.roomOff : Copy.roomLevel(db)).tag(db)
            }
        }
    }
}
