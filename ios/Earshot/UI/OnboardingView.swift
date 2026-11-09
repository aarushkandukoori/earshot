// OnboardingView.swift - first-launch onboarding (30-ios 13 "Onboarding", D55), about 4 minutes,
// 8 steps, every step skippable except 2 (permissions) and 5 (your voice).
//   1 What it is and is not, credits     5 Your voice (mandatory before listening)
//   2 Permissions                        6 Measure the delay
//   3 Which AirPods, microphone tap check 7 First listen (10 s Everyone, hold to compare)
//   4 AirPods checklist with "how" sheets 8 Controls card (stem presses and beeps, also in Help)
//
// Main actor (SwiftUI). The rules are plain types at the top of this file so OnboardingTests can
// check them without a screen: OnboardingFlow (order and gating), OnboardingStore (the "done" flag
// in UserDefaults), PermissionAnswer, TapTest (the 20 dB rule of 30-ios 3.2), AirPodsModelStore
// (30-ios 3.4, the answer to "Which AirPods are these?" per port UID). Every string is in Copy.

import AVFoundation
import CoreMotion
import SwiftUI
import UserNotifications

// MARK: - Rules (pure, unit-tested)

enum OnboardingStep: Int, CaseIterable, Sendable {
    case whatItIs = 1, permissions, whichAirPods, checklist, yourVoice, measureDelay, firstListen, controls
}

/// Where the user is in onboarding and what lets them move on.
struct OnboardingFlow: Equatable {
    static let total = OnboardingStep.allCases.count // 8

    private(set) var step: OnboardingStep = .whatItIs
    private(set) var finished = false

    /// Steps 2 and 5 cannot be skipped (30-ios 13).
    static func canSkip(_ s: OnboardingStep) -> Bool { s != .permissions && s != .yourVoice }

    /// Continue needs: step 2 - every permission answered (allowed or not); step 5 - an own voice
    /// (Earshot never plays the user's own voice back, so listening needs it). Others - nothing.
    static func canContinue(_ s: OnboardingStep, permissionsAnswered: Bool, hasOwnVoice: Bool) -> Bool {
        switch s {
        case .permissions: return permissionsAnswered
        case .yourVoice: return hasOwnVoice
        default: return true
        }
    }

    /// Continue: moves on (or finishes after step 8) when allowed; returns whether it moved.
    @discardableResult
    mutating func next(permissionsAnswered: Bool, hasOwnVoice: Bool) -> Bool {
        guard !finished, Self.canContinue(step, permissionsAnswered: permissionsAnswered, hasOwnVoice: hasOwnVoice)
        else { return false }
        advance()
        return true
    }

    /// Skip: moves on without doing the step, only where skipping is allowed.
    @discardableResult
    mutating func skip() -> Bool {
        guard !finished, Self.canSkip(step) else { return false }
        advance()
        return true
    }

    mutating func back() {
        guard !finished, let prev = OnboardingStep(rawValue: step.rawValue - 1) else { return }
        step = prev
    }

    private mutating func advance() {
        if let n = OnboardingStep(rawValue: step.rawValue + 1) { step = n } else { finished = true }
    }
}

/// The "onboarding finished" flag (a few bytes in UserDefaults, no audio).
@MainActor
final class OnboardingStore: ObservableObject {
    static let key = "earshot.onboarding.v1.done"
    @Published private(set) var done: Bool
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        done = defaults.bool(forKey: Self.key)
    }

    func finish() {
        done = true
        defaults.set(true, forKey: Self.key)
    }
}

/// One permission's state as onboarding shows it.
enum PermissionAnswer: Equatable, Sendable {
    case notAsked, allowed, notAllowed

    /// Step 2 is answered once no permission is still waiting for its first request.
    static func allAnswered(_ answers: [PermissionAnswer]) -> Bool { !answers.contains(.notAsked) }
}

/// "Tap the bottom edge of the phone" must move the input meter by at least 20 dB (30-ios 3.2):
/// the loudest peak must be 20 dB above the quietest RMS reading of the check. If the route had
/// really moved the input to the AirPods, a tap on the phone would barely move the meter.
struct TapTest: Equatable {
    static let neededRiseDB: Float = 20

    private(set) var quietDB: Float? = nil
    private(set) var loudestPeakDB: Float = -100

    /// One 10 Hz meter reading. Readings at the -100 dB floor (no audio yet) are ignored.
    mutating func add(_ m: InputMeter) {
        guard m.rmsDB > -99 else { return }
        quietDB = min(quietDB ?? m.rmsDB, m.rmsDB)
        loudestPeakDB = max(loudestPeakDB, m.peakDB)
    }

    var riseDB: Float { quietDB.map { loudestPeakDB - $0 } ?? 0 }
    var passed: Bool { quietDB != nil && riseDB >= Self.neededRiseDB }
}

/// "Which AirPods are these?" (30-ios 3.4): iOS gives no model name, so the answer is stored per
/// output port UID. It picks the planning delay and the "Pro 2 is better" tip (contract 10.2).
enum AirPodsModel: String, Codable, CaseIterable, Sendable { case pro1, pro2 }

final class AirPodsModelStore {
    private let defaults: UserDefaults
    init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    private func key(_ uid: String) -> String { "airpods.model.\(uid)" }

    func model(forPort uid: String) -> AirPodsModel? {
        guard !uid.isEmpty, let raw = defaults.string(forKey: key(uid)) else { return nil }
        return AirPodsModel(rawValue: raw)
    }

    func set(_ m: AirPodsModel, forPort uid: String) {
        guard !uid.isEmpty else { return }
        defaults.set(m.rawValue, forKey: key(uid))
    }
}

// MARK: - Permissions (main actor)

/// Reads and requests microphone, motion and notification permission for step 2.
@MainActor
final class PermissionsModel: ObservableObject {
    @Published private(set) var mic: PermissionAnswer = .notAsked
    @Published private(set) var motion: PermissionAnswer = .notAsked
    @Published private(set) var notifications: PermissionAnswer = .notAsked
    private var motionManager: CMHeadphoneMotionManager?

    var allAnswered: Bool { PermissionAnswer.allAnswered([mic, motion, notifications]) }

    func refresh() {
        switch AVAudioApplication.shared.recordPermission {
        case .granted: mic = .allowed
        case .denied: mic = .notAllowed
        default: mic = .notAsked
        }
        switch CMHeadphoneMotionManager.authorizationStatus() {
        case .authorized: motion = .allowed
        case .denied, .restricted: motion = .notAllowed
        default: break // keep .notAsked, or the answer recorded by askMotion
        }
        UNUserNotificationCenter.current().getNotificationSettings { s in
            let status = s.authorizationStatus
            Task { @MainActor in
                switch status {
                case .notDetermined: break
                case .denied: self.notifications = .notAllowed
                default: self.notifications = .allowed
                }
            }
        }
    }

    func askMic() {
        AVAudioApplication.requestRecordPermission { granted in
            Task { @MainActor in self.mic = granted ? .allowed : .notAllowed }
        }
    }

    /// Head tracking asks for Motion & Fitness when updates start. Without AirPods iOS may not ask
    /// at all, so after 10 s the request counts as answered (the user can allow it later in Settings).
    func askMotion() {
        let m = CMHeadphoneMotionManager()
        motionManager = m
        m.startDeviceMotionUpdates(to: OperationQueue()) { _, _ in }
        Task { @MainActor in
            for _ in 0..<20 {
                try? await Task.sleep(nanoseconds: 500_000_000)
                if CMHeadphoneMotionManager.authorizationStatus() != .notDetermined { break }
            }
            m.stopDeviceMotionUpdates()
            self.motionManager = nil
            self.motion = CMHeadphoneMotionManager.authorizationStatus() == .authorized ? .allowed : .notAllowed
        }
    }

    func askNotifications() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, _ in
            Task { @MainActor in self.notifications = granted ? .allowed : .notAllowed }
        }
    }
}

// MARK: - The screen

struct OnboardingView: View {
    @EnvironmentObject private var people: PeopleStore
    @ObservedObject var store: OnboardingStore
    @StateObject private var permissions = PermissionsModel()
    @State private var flow = OnboardingFlow()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(Copy.stepOf(flow.step.rawValue, OnboardingFlow.total))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    stepContent
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
            }
            .safeAreaInset(edge: .bottom) { buttons.padding().background(.bar) }
            .navigationTitle(title(flow.step))
            .navigationBarTitleDisplayMode(.inline)
        }
        .interactiveDismissDisabled()
        .onAppear { permissions.refresh() }
    }

    @ViewBuilder private var stepContent: some View {
        switch flow.step {
        case .whatItIs: WhatItIsStep()
        case .permissions: PermissionsStep(permissions: permissions)
        case .whichAirPods: WhichAirPodsStep()
        case .checklist: ChecklistStep()
        case .yourVoice: YourVoiceStep()
        case .measureDelay: MeasureStep()
        case .firstListen: FirstListenStep()
        case .controls: ControlsCard()
        }
    }

    private var buttons: some View {
        let canGo = OnboardingFlow.canContinue(flow.step, permissionsAnswered: permissions.allAnswered,
                                               hasOwnVoice: people.own != nil)
        let last = flow.step == .controls
        return HStack {
            if flow.step != .whatItIs {
                Button(Copy.back) { flow.back() }.buttonStyle(.bordered)
            }
            Spacer()
            if OnboardingFlow.canSkip(flow.step) && !last {
                Button(Copy.skip) { flow.skip() }.buttonStyle(.bordered)
            }
            Button(last ? Copy.finishSetup : Copy.next) {
                flow.next(permissionsAnswered: permissions.allAnswered, hasOwnVoice: people.own != nil)
                if flow.finished { store.finish() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canGo)
        }
        .controlSize(.large)
    }

    private func title(_ s: OnboardingStep) -> String {
        switch s {
        case .whatItIs: return Copy.whatItIsTitle
        case .permissions: return Copy.permissionsTitle
        case .whichAirPods: return Copy.whichAirPodsTitle
        case .checklist: return Copy.checklistTitle
        case .yourVoice: return Copy.yourVoice
        case .measureDelay: return Copy.measureTitle
        case .firstListen: return Copy.firstListenTitle
        case .controls: return Copy.controlsTitle
        }
    }
}

// MARK: - Steps

/// Step 1: what Earshot is and is not, and the credits (Look Once to Hear, WeSpeaker terms).
struct WhatItIsStep: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(Copy.whatItIs).font(.title3)
            Text(Copy.whatItIsNotTitle).font(.headline)
            Text(Copy.whatItIsNot)
            Text(Copy.ownVoiceAlwaysDown)
            Text(Copy.creditsTitle).font(.headline)
            Text(Copy.creditsLookOnce).font(.callout)
            Text(Copy.creditsWeSpeaker).font(.callout)
        }
    }
}

/// Step 2: microphone, motion, notifications. Not skippable: each request must be answered.
struct PermissionsStep: View {
    @ObservedObject var permissions: PermissionsModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(Copy.permissionsWhy)
            row(Copy.microphone, permissions.mic) { permissions.askMic() }
            row(Copy.motion, permissions.motion) { permissions.askMotion() }
            row(Copy.notifications, permissions.notifications) { permissions.askNotifications() }
            if permissions.mic == .notAllowed { Text(Copy.micNeeded).foregroundStyle(.red) }
            if !permissions.allAnswered { Text(Copy.answerEachPermission).font(.footnote).foregroundStyle(.secondary) }
        }
    }

    private func row(_ name: String, _ answer: PermissionAnswer, ask: @escaping () -> Void) -> some View {
        HStack {
            Text(name).font(.headline)
            Spacer()
            switch answer {
            case .notAsked: Button(Copy.allow, action: ask).buttonStyle(.borderedProminent)
            case .allowed: Text(Copy.allowed).foregroundStyle(.green)
            case .notAllowed: Text(Copy.notAllowed).font(.footnote).multilineTextAlignment(.trailing)
            }
        }
        .frame(minHeight: 44)
    }
}

/// Step 3: which AirPods (stored per port UID) and the one-line tap check (30-ios 3.2, 3.4).
struct WhichAirPodsStep: View {
    @EnvironmentObject private var audio: AudioController
    @State private var tap = TapTest()
    @State private var errorText: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            WhichAirPodsPicker()
            if audio.isListening {
                Text(Copy.tapTestHowTo)
                Text(String(format: "%.0f dB", tap.riseDB)).font(.title2.monospacedDigit())
                    .accessibilityLabel(Copy.tapTestHowTo)
                if tap.passed {
                    Text(Copy.tapTestPass).foregroundStyle(.green)
                } else if tap.quietDB != nil && tap.riseDB > 0 {
                    Text(Copy.tapTestFail).font(.footnote).foregroundStyle(.secondary)
                }
                Button(Copy.tapTestRetry) { tap = TapTest() }.buttonStyle(.bordered)
            } else {
                StartListeningButton(title: Copy.startMicrophone, errorText: $errorText)
            }
            if let errorText { Text(errorText).font(.footnote).foregroundStyle(.red) }
        }
        .onChange(of: audio.meter) { _, m in tap.add(m) }
    }
}

/// "Which AirPods are these?" with the answer kept per port UID; the Pro 2 tip for Pro 1.
struct WhichAirPodsPicker: View {
    @EnvironmentObject private var audio: AudioController
    @State private var chosen: AirPodsModel? = nil
    private let store = AirPodsModelStore()

    var body: some View {
        let uid = audio.route.primaryOutputUID
        VStack(alignment: .leading, spacing: 10) {
            if audio.route.isBluetooth {
                Text(Copy.whichAirPodsQuestion).font(.headline)
                HStack {
                    choice(.pro1, Copy.airPodsPro1, uid: uid)
                    choice(.pro2, Copy.airPodsPro2, uid: uid)
                }
                if (chosen ?? store.model(forPort: uid)) == .pro1 { Text(Copy.pro2Tip).font(.footnote) }
            } else {
                Text(Copy.connectAirPodsFirst)
            }
        }
        .onAppear { chosen = store.model(forPort: uid) }
        .onChange(of: uid) { _, new in chosen = store.model(forPort: new) }
    }

    @ViewBuilder private func choice(_ m: AirPodsModel, _ title: String, uid: String) -> some View {
        let selected = (chosen ?? store.model(forPort: uid)) == m
        Button(title) {
            store.set(m, forPort: uid)
            chosen = m
        }
        .buttonStyle(.bordered)
        .tint(selected ? .accentColor : .secondary)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// Step 4: the seven AirPods settings, each with a "how" sheet. The app cannot read these
/// settings, so the check marks are only the user's own notes for this screen.
struct ChecklistStep: View {
    @State private var ticked = Set<Int>()
    @State private var how: HowItem? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(Copy.checklistIntro).font(.callout)
            ForEach(Copy.checklist.indices, id: \.self) { i in
                HStack {
                    Button {
                        if ticked.contains(i) { ticked.remove(i) } else { ticked.insert(i) }
                    } label: {
                        Label(Copy.checklist[i].title, systemImage: ticked.contains(i) ? "checkmark.circle.fill" : "circle")
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(ticked.contains(i) ? .isSelected : [])
                    Spacer()
                    Button(Copy.how) { how = HowItem(id: i) }.buttonStyle(.bordered)
                }
                .frame(minHeight: 44)
            }
        }
        .sheet(item: $how) { item in
            NavigationStack {
                ScrollView { Text(Copy.checklist[item.id].how).padding() }
                    .navigationTitle(Copy.checklist[item.id].title)
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar { Button(Copy.done) { how = nil } }
            }
            .presentationDetents([.medium])
        }
    }
}

struct HowItem: Identifiable { let id: Int }
