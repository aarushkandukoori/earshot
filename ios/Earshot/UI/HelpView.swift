// HelpView.swift - Help (30-ios 13): the stem table, the beeps, the AirPods checklist, what the delay
// means, what Earshot is and is not, and the credits. ControlsCard is also onboarding step 8.
//
// Main actor (SwiftUI). Every string comes from Copy. Nothing here changes any setting.

import SwiftUI

/// Stem presses and beep meanings (onboarding step 8 and Help).
struct ControlsCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(Copy.stemTitle).font(.headline)
            Text(Copy.stemSingle)
            Text(Copy.stemDouble)
            Text(Copy.stemTriple)
            Text(Copy.beepsTitle).font(.headline).padding(.top, 6)
            Text(Copy.beepSlot)
            Text(Copy.beepMode)
            Text(Copy.beepCapture)
            Text(Copy.beepAlarm)
        }
    }
}

struct HelpView: View {
    @EnvironmentObject private var audio: AudioController

    var body: some View {
        List {
            Section(Copy.controlsTitle) { ControlsCard().padding(.vertical, 4) }
            Section(Copy.delayMeaningTitle) {
                Text(Copy.delayMeaning)
                if audio.isListening {
                    Text(audio.latency.text).font(.headline)
                    Text(Copy.earshotShare(Int(audio.engineLatencyMS.rounded()))).font(.footnote)
                }
            }
            Section(Copy.checklistTitle) {
                Text(Copy.checklistIntro).font(.footnote)
                ForEach(Copy.checklist.indices, id: \.self) { i in
                    DisclosureGroup(Copy.checklist[i].title) {
                        Text(Copy.checklist[i].how).font(.callout)
                    }
                }
            }
            Section(Copy.whatItIsTitle) {
                Text(Copy.whatItIs)
                Text(Copy.whatItIsNot)
                Text(Copy.ownVoiceAlwaysDown)
            }
            Section(Copy.creditsTitle) {
                Text(Copy.creditsLookOnce)
                Text(Copy.creditsWeSpeaker)
                Text(Copy.privacySummary)
            }
            .font(.callout)
        }
        .navigationTitle(Copy.help)
    }
}
