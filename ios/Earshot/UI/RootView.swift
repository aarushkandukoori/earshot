// RootView.swift - the main screen (30-ios 13 "Listen"): launch status, banners, the Listen panel
// while listening (I-3: who is playing, table chips, Look, modes, voice and room, hold-to-compare,
// ListenView.swift), the delay chip (tap to measure), Start / Pause / Stop, the people entries
// (your voice, Add person, Add who just spoke, People) and links to Settings, Help and Diagnostics.
// I-4: on first launch the onboarding (OnboardingView.swift) covers this screen until it is finished.
// Main actor (SwiftUI).

import SwiftUI

struct RootView: View {
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var launch: LaunchState
    @StateObject private var onboarding = OnboardingStore()

    /// Onboarding shows once the launch sequence allows listening and enrollment (otherwise the
    /// home screen shows why not), until the user finishes it.
    private var showOnboarding: Bool {
        guard let r = launch.report, r.blockingError == nil, r.startEnabled, launch.enrollment != nil else { return false }
        return !onboarding.done
    }

    var body: some View {
        NavigationStack {
            Group {
                if let report = launch.report {
                    if let error = report.blockingError {
                        BlockingErrorView(message: error)
                    } else {
                        HomeView(report: report)
                    }
                } else {
                    ProgressView(Copy.gettingReady)
                }
            }
            .navigationTitle(Copy.appName)
        }
        .environmentObject(onboarding)
        .fullScreenCover(isPresented: Binding(get: { showOnboarding }, set: { _ in })) {
            OnboardingView(store: onboarding)
        }
    }
}

/// Shown when launch step 1 or 3 failed: no audio ever starts.
struct BlockingErrorView: View {
    let message: String

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .accessibilityHidden(true)
            Text(message)
                .font(.title3)
                .multilineTextAlignment(.center)
        }
        .padding()
    }
}

struct HomeView: View {
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var launch: LaunchState
    @EnvironmentObject private var people: PeopleStore
    @EnvironmentObject private var onboarding: OnboardingStore
    let report: LaunchReport
    @State private var errorText: String? = nil
    @State private var showOwnVoice = false
    @State private var askedForOwnVoice = false

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                Text(stateText)
                    .font(.title2)
                    .accessibilityLabel(Copy.status(stateText))

                if let notice = audio.notice {
                    Text(notice)
                        .font(.callout)
                        .multilineTextAlignment(.center)
                        .padding(10)
                        .background(Color.yellow.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
                }

                LatencyPromptBanner()

                if audio.isListening && report.startEnabled {
                    ListenPanel()
                }

                // The delay chip: the honest delay text; tapping it measures again (30-ios 7).
                MeasureDelayButton(label: audio.latency.text)
                    .font(.footnote)

                primaryButton

                if let errorText {
                    Text(errorText).font(.footnote).foregroundStyle(.red).multilineTextAlignment(.center)
                }
                if !report.selfTestPassed {
                    Text(Copy.selfTestFailed).font(.footnote).foregroundStyle(.red)
                }
                if !report.useWeSpeaker && report.startEnabled {
                    Text(Copy.voiceCheckUnavailable).font(.footnote).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                peopleButtons

                HStack(spacing: 24) {
                    NavigationLink(Copy.settings) { SettingsView() }
                    NavigationLink(Copy.help) { HelpView() }
                    NavigationLink(Copy.diagnostics) { DiagnosticsView(report: report) }
                }
                .padding(.top, 8)

                Text(Copy.disclaimer + " " + Copy.privacy)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding()
        }
        .sheet(isPresented: $showOwnVoice) { NavigationStack { OwnVoiceView() } }
        .onChange(of: people.loaded) { _, loaded in
            // The own voice comes first (Earshot stays muted without it). During onboarding, step 5
            // asks for it instead, so this prompt only runs after onboarding is done.
            if loaded && people.own == nil && launch.enrollment != nil && !askedForOwnVoice && onboarding.done {
                askedForOwnVoice = true
                showOwnVoice = true
            }
        }
    }

    /// I-2 entries; enrollment needs thresholds.json (launch step 3).
    @ViewBuilder private var peopleButtons: some View {
        let canEnroll = report.startEnabled && launch.enrollment != nil
        if people.loaded && people.own == nil {
            Button(Copy.recordMyVoice) { showOwnVoice = true }
                .buttonStyle(.borderedProminent)
                .disabled(!canEnroll)
        }
        HStack {
            NavigationLink(Copy.addPerson) { AddPersonView(quick: false) }.disabled(!canEnroll)
            // Quick add plays the other person's voice back, so it needs AirPods (or another allowed
            // output): never on a bad route, where the clip would come out of the phone's speaker.
            NavigationLink(Copy.addWhoJustSpoke) { AddPersonView(quick: true) }
                .disabled(!canEnroll || !audio.isListening || audio.state == .badRoute || !audio.route.allowed)
        }
        NavigationLink(Copy.people) { PeopleView() }
    }

    private var stateText: String {
        switch audio.state {
        case .idle: return Copy.notListening
        case .running: return audio.isUnmuted ? Copy.listening : Copy.muted
        case .paused: return Copy.paused
        case .badRoute: return Copy.connectAirPods
        case .interrupted: return Copy.interrupted
        case .failed(let why): return why
        }
    }

    @ViewBuilder private var primaryButton: some View {
        switch audio.state {
        case .idle, .failed:
            Button(Copy.start) {
                Task {
                    do { try await audio.start(); errorText = nil }
                    catch { errorText = AppError.userText(error, fallback: Copy.startFailed) } // raw text: Diagnostics
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!audio.canStart || !report.startEnabled)
        case .paused, .interrupted:
            HStack {
                Button(Copy.resume) { audio.resume() }.buttonStyle(.borderedProminent)
                Button(Copy.stop) { audio.stop() }.buttonStyle(.bordered)
            }
            .controlSize(.large)
        default:
            HStack {
                Button(Copy.pause) { audio.pause() }.buttonStyle(.bordered)
                Button(Copy.stop) { audio.stop() }.buttonStyle(.borderedProminent)
            }
            .controlSize(.large)
        }
    }
}
