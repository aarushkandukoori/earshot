// OwnVoiceView.swift - "Your voice", onboarding step 5 and Settings > Re-record (30-ios 8.1, 13).
//
// The user reads passage.txt in a quiet place with the phone held normally. The capture stops at
// 8 s of speech or 20 s. Then EnrollmentService runs the own-voice checks (speech on the full
// capture, snr_db, the best 10 s window as the clip). A pass or warn saves the own voice at once;
// PeopleStore then hands the voiceprint to AudioController, which leaves setup mode (unmutes when
// allowed) and rebuilds every slot with the new negative.

import SwiftUI

struct OwnVoiceView: View {
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var launch: LaunchState
    @EnvironmentObject private var people: PeopleStore
    @Environment(\.dismiss) private var dismiss
    @StateObject private var capture = CaptureSession()
    @State private var result: EnrollmentResult? = nil
    @State private var checking = false
    @State private var errorText: String? = nil

    private let passage: String = {
        guard let url = Bundle.main.url(forResource: "passage", withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return "" }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(Copy.yourVoiceWhy).font(.title3)
                Text(Copy.readPassageHowTo).font(.callout).foregroundStyle(.secondary)
                Text(passage)
                    .font(.body)
                    .padding()
                    .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))

                content
                    .frame(maxWidth: .infinity)

                if let errorText { Text(errorText).font(.footnote).foregroundStyle(.red) }
            }
            .padding()
        }
        .navigationTitle(Copy.yourVoice)
        .onChange(of: capture.phase) { _, phase in
            if phase == .finished { Task { await check() } }
        }
        .onDisappear { capture.cancel() }
    }

    @ViewBuilder private var content: some View {
        if capture.phase == .recording {
            VStack(spacing: 12) {
                SpeechRing(seconds: capture.speechSeconds, target: CaptureLimits.own.stopSpeech)
                Button(Copy.cancel) { capture.cancel() }.buttonStyle(.bordered)
            }
        } else if checking {
            ProgressView(Copy.checking)
        } else if let result {
            VStack(spacing: 12) {
                CheckCard(result: result)
                if result.verdict == .fail {
                    Button(Copy.tryAgain) { start() }.buttonStyle(.borderedProminent)
                } else {
                    Text(Copy.ownVoiceSaved).font(.headline)
                    Button(Copy.done) { dismiss() }.buttonStyle(.borderedProminent)
                }
            }
        } else {
            VStack(spacing: 12) {
                if case .failed(let message) = capture.phase { Text(message).foregroundStyle(.red) }
                Button(people.own == nil ? Copy.recordMyVoice : Copy.reRecordMyVoice) { start() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(launch.enrollment == nil)
                if people.own == nil {
                    Button(Copy.notNow) { dismiss() } // Earshot stays muted until a voice is recorded
                }
            }
        }
    }

    private func start() {
        result = nil
        errorText = nil
        guard let model = launch.model else { errorText = Copy.notReady; return }
        Task {
            do { try await capture.start(audio: audio, model: model, role: .own) }
            catch {
                audio.note("own voice capture: \(error)") // the raw error goes to Diagnostics only
                errorText = AppError.userText(error, fallback: Copy.captureStartFailed)
            }
        }
    }

    private func check() async {
        guard let enrollment = launch.enrollment, let model = launch.model else { errorText = Copy.notReady; return }
        checking = true
        defer { checking = false }
        do {
            let r = try await enrollment.enroll(pcm16k: capture.captured, role: .own, own: nil, saved: [])
            result = r
            guard r.verdict != .fail, let vp = r.voiceprint else { audio.cue(beeps: 3); return }
            let me = Person.enrolled(r, name: Copy.you, kind: .own, modelHash: model.modelHash, yawDeg: nil)
            try people.add(me, clip: r.clip, voiceprint: vp) // replaces any older own voice
            audio.cue(beeps: 2)
        } catch {
            audio.note("own voice check: \(error)") // the raw error goes to Diagnostics only
            errorText = Copy.checkFailed
            audio.cue(beeps: 3)
        }
    }
}
