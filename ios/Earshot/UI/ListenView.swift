// ListenView.swift - the listening part of the main screen (30-ios 13 "Listen"; 10.3 card taps;
// 7 delay chip and prompts; 12.1 hold-to-compare).
//
// Main actor (SwiftUI). Pieces:
//   ListenPanel         - who is playing, table chips with talking dots, Look, the mode buttons,
//                         voice level, room sound, hold-to-compare, head-tracking banner
//   LatencyPromptBanner - "measure the delay?" at connect and the 10 / 30 min re-check banners
//   MeasureDelayButton  - the probe flow: how-to (method A), result, and method B after a failure
// Every string comes from Copy. I-4: the locked check mark on the chips, the "Might be following
// someone else" banner, Table off while saving power, and "Which AirPods are these?" (30-ios 3.4).

import SwiftUI

struct ListenPanel: View {
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var selector: Selector

    var body: some View {
        VStack(spacing: 14) {
            Text(selector.statusText)
                .font(.title3.weight(.semibold))
            WhichAirPodsBanner()
            if selector.table.personCount > 0 {
                TableChips()
                if selector.lostLockWarning {
                    Text(Copy.mightBeFollowingSomeoneElse)
                        .font(.callout)
                        .multilineTextAlignment(.center)
                        .padding(8)
                        .background(Color.orange.opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
                }
                HeadPausedBanner(head: selector.head)
            }
            Button(Copy.look) { selector.lookSelect() }
                .buttonStyle(.bordered)
                .disabled(selector.table.personCount == 0 || selector.handsFreeActive)
            ModeButtons()
            VoiceLevelSlider()
            RoomSoundPicker()
            HoldToCompare()
        }
    }
}

/// Up to 3 table people; tap = focus (same as look-select's focus), long press = remove.
struct TableChips: View {
    @EnvironmentObject private var selector: Selector

    var body: some View {
        HStack(spacing: 8) {
            ForEach(selector.table.filledPersonSlots, id: \.slot) { seat in
                let name = selector.name(seat.personID)
                let playing = selector.table.gains[seat.slot] > 0
                let talking = selector.talking[seat.slot]
                Button { selector.focus(on: seat.slot) } label: {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(talking ? Color.green : Color.secondary.opacity(0.3))
                            .frame(width: 10, height: 10)
                            .accessibilityHidden(true)
                        Text(name).lineLimit(1)
                        if selector.lockStates[seat.slot] == .locked { // 30-ios 12.3: check mark while locked
                            Image(systemName: "checkmark")
                                .font(.caption.weight(.bold))
                                .accessibilityLabel(Copy.followingRightVoice)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .frame(minHeight: 44)
                }
                .buttonStyle(.plain)
                .background(playing ? Color.accentColor.opacity(0.25) : Color.secondary.opacity(0.1),
                            in: Capsule())
                .accessibilityLabel(talking ? Copy.talkingLabel(name) : name)
                .accessibilityAddTraits(playing ? .isSelected : [])
                .contextMenu {
                    Button(Copy.removeFromTable, role: .destructive) { selector.removeFromTable(seat.personID) }
                }
            }
        }
    }
}

/// "Head tracking paused" while no head sample arrived for 2 s (look-select then cycles).
struct HeadPausedBanner: View {
    @ObservedObject var head: HeadTracker

    var body: some View {
        if head.running && head.unavailable {
            Text(Copy.headTrackingPaused)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }
}

/// Focus / Table / Everyone. Focus needs a person, Table needs 2 (30-ios 10.2).
struct ModeButtons: View {
    @EnvironmentObject private var selector: Selector

    var body: some View {
        VStack(spacing: 4) {
            HStack {
                button(.focus, Copy.modeFocus, enabled: selector.table.personCount >= 1)
                button(.table, Copy.modeTable, enabled: selector.table.personCount >= 2 && !selector.powerSaving)
                button(.everyone, Copy.modeEveryone, enabled: true)
            }
            if selector.powerSaving && selector.table.personCount >= 2 {
                Text(Copy.tableOffWhileSaving).font(.footnote).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private func button(_ m: TableMode, _ title: String, enabled: Bool) -> some View {
        if selector.table.mode == m {
            Button(title) { selector.setMode(m) }.buttonStyle(.borderedProminent)
                .accessibilityAddTraits(.isSelected)
        } else {
            Button(title) { selector.setMode(m) }.buttonStyle(.bordered).disabled(!enabled)
        }
    }
}

/// Plays the delayed room (bypass) while held, so the user can compare (30-ios 13).
struct HoldToCompare: View {
    @EnvironmentObject private var audio: AudioController

    var body: some View {
        Text(Copy.holdToCompare)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(Color.secondary.opacity(0.15), in: Capsule())
            .onLongPressGesture(minimumDuration: 3600, maximumDistance: 80, perform: {},
                                onPressingChanged: { pressing in audio.setBypass(pressing) })
            .onDisappear { audio.setBypass(false) }
            .accessibilityAddTraits(.isButton)
            .accessibilityHint(Copy.holdToCompareHint)
    }
}

/// The measure prompt at connect and the re-check banners (offered, never required).
struct LatencyPromptBanner: View {
    @EnvironmentObject private var audio: AudioController

    var body: some View {
        if let prompt = audio.latencyPrompt, audio.state == .running {
            VStack(spacing: 8) {
                Text(prompt == .measureAtConnect ? Copy.measurePrompt : Copy.recheckDelay)
                    .font(.callout)
                    .multilineTextAlignment(.center)
                HStack {
                    MeasureDelayButton()
                    Button(Copy.notNow) { audio.dismissLatencyPrompt() }.buttonStyle(.bordered)
                }
            }
            .padding(10)
            .background(Color.blue.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

/// Measure the delay: shows the how-to first (it needs the user's hands), runs the probe, and
/// after a failed probe offers method B (ear detection off). `label` nil = "Measure delay"; the
/// delay chip passes its own text so tapping the chip measures again (30-ios 7).
struct MeasureDelayButton: View {
    @EnvironmentObject private var audio: AudioController
    var label: String? = nil
    @State private var confirm: ProbeMethod? = nil
    @State private var errorText: String? = nil

    var body: some View {
        VStack(spacing: 6) {
            Button(label ?? Copy.measureDelay) { confirm = .inEar }
                .buttonStyle(.bordered)
                .disabled(audio.state != .running || audio.isMeasuring)
                .accessibilityHint(Copy.tapToMeasure)
            if audio.isMeasuring { Text(Copy.measuring).font(.footnote) }
            if let errorText { Text(errorText).font(.footnote).foregroundStyle(.red) }
            if audio.offerMethodB && !audio.isMeasuring {
                Button(Copy.tryMethodB) { confirm = .earDetectionOff }
                    .font(.footnote)
                    .disabled(audio.state != .running)
            }
        }
        .alert(Copy.measureDelay, isPresented: Binding(get: { confirm != nil }, set: { if !$0 { confirm = nil } }),
               presenting: confirm) { method in
            Button(Copy.measureDelay) { run(method) }
            Button(Copy.cancel, role: .cancel) {}
        } message: { method in
            Text(method == .inEar ? Copy.measureHowTo : Copy.measureHowToB)
        }
    }

    private func run(_ method: ProbeMethod) {
        errorText = nil
        Task {
            do { try await audio.measureLatency(method: method) }
            catch { errorText = AppError.userText(error, fallback: Copy.probeWeak) } // raw text is in Diagnostics
        }
    }
}

/// "Which AirPods are these?" the first time a Bluetooth output port appears (30-ios 3.4), until
/// answered or dismissed for this screen.
struct WhichAirPodsBanner: View {
    @EnvironmentObject private var audio: AudioController
    @State private var dismissed = false

    var body: some View {
        let uid = audio.route.primaryOutputUID
        if audio.route.isBluetooth && !dismissed && AirPodsModelStore().model(forPort: uid) == nil {
            VStack(spacing: 8) {
                WhichAirPodsPicker()
                Button(Copy.notNow) { dismissed = true }.font(.footnote)
            }
            .padding(10)
            .background(Color.blue.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
        }
    }
}
