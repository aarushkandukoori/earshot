// AddPersonView.swift - "Add person" and "Add who just spoke" (30-ios 8.1, 8.3, 8.5, 13).
//
// Add person: hold the phone toward them; the capture stops at 6 s of speech or 15 s ("Done" from
// 4 s); the 6 checks run; a pass or warn asks for a name and Tonight / Save (Save asks for consent).
// Add who just spoke: the last 8 s of the engine's history go through the same checks, then the
// 2 s with the most speech play back (engine muted); "Is this them?" Yes saves a tonight person,
// No throws it all away.
// I-3: a saved person joins the table at once (Selector.addToTable; outside Table mode they are
// focused), with the head direction measured during the capture (HeadTracker yaw).

import SwiftUI

struct AddPersonView: View {
    let quick: Bool // true = "Add who just spoke"

    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var launch: LaunchState
    @EnvironmentObject private var people: PeopleStore
    @EnvironmentObject private var selector: Selector
    @Environment(\.dismiss) private var dismiss
    @StateObject private var capture = CaptureSession()
    @State private var player = QuickAddPlayer()
    @State private var result: EnrollmentResult? = nil
    @State private var checking = false
    @State private var playedBack = false
    @State private var name = ""
    @State private var askConsent = false
    @State private var errorText: String? = nil

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                content
                if let errorText { Text(errorText).font(.footnote).foregroundStyle(.red) }
            }
            .padding()
        }
        .navigationTitle(quick ? Copy.addWhoJustSpoke : Copy.addPerson)
        .task {
            let head = selector.head
            capture.yawProvider = { head.yawForCapture() } // the direction "when you add someone"
            if result == nil && !checking { begin() }
        }
        .onChange(of: capture.phase) { _, phase in
            if phase == .finished { Task { await check(capture.captured, yaw: capture.yawDegrees) } }
        }
        .onDisappear { capture.cancel() }
        .alert(Copy.consentQuestion(displayName), isPresented: $askConsent) {
            Button(Copy.theyAgreed) { save(kind: .saved, consent: true) }
            Button(Copy.tonightOnly) { save(kind: .tonight, consent: false) }
        }
    }

    private var displayName: String {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? Copy.nextPersonName(taken: people.people.map { $0.name }) : trimmed
    }

    @ViewBuilder private var content: some View {
        if capture.phase == .recording {
            Text(Copy.holdTowardThem).font(.title3).multilineTextAlignment(.center)
            SpeechRing(seconds: capture.speechSeconds, target: CaptureLimits.target.stopSpeech)
            Text(String(format: "%.1f dBFS", audio.meter.rmsDB)).font(.caption.monospacedDigit())
            HStack {
                Button(Copy.cancel) { capture.cancel(); dismiss() }.buttonStyle(.bordered)
                Button(Copy.done) { capture.finishNow() }.buttonStyle(.borderedProminent).disabled(!capture.canFinish)
            }
        } else if checking {
            ProgressView(Copy.checking)
        } else if let result {
            CheckCard(result: result)
            if result.verdict == .fail {
                Button(Copy.tryAgain) { begin() }.buttonStyle(.borderedProminent)
            } else if quick && !playedBack {
                ProgressView()
            } else {
                accepted(result)
            }
        } else if case .failed(let message) = capture.phase {
            Text(message).foregroundStyle(.red)
            Button(Copy.tryAgain) { begin() }.buttonStyle(.borderedProminent)
        }
    }

    /// Name, then the save choices (or the duplicate question).
    @ViewBuilder private func accepted(_ r: EnrollmentResult) -> some View {
        if quick { Text(Copy.isThisThem).font(.headline) }
        TextField(Copy.name, text: $name, prompt: Text(Copy.nextPersonName(taken: people.people.map { $0.name })))
            .textFieldStyle(.roundedBorder)
        if let dup = r.duplicate {
            Text(Copy.soundsLike(dup.name)).font(.callout)
            HStack {
                Button(Copy.update) { update(dup) }.buttonStyle(.borderedProminent)
                Button(Copy.keepBoth) { save(kind: .tonight, consent: false) }.buttonStyle(.bordered)
            }
        } else if quick {
            HStack {
                Button(Copy.yes) { save(kind: .tonight, consent: false) }.buttonStyle(.borderedProminent)
                Button(Copy.no) { dismiss() }.buttonStyle(.bordered) // discards everything
            }
        } else {
            HStack {
                Button(Copy.tonightOnly) { save(kind: .tonight, consent: false) }.buttonStyle(.borderedProminent)
                Button(Copy.saveForLater) { askConsent = true }.buttonStyle(.bordered)
            }
        }
    }

    // MARK: - Actions

    private func begin() {
        result = nil
        errorText = nil
        playedBack = false
        guard let model = launch.model, launch.enrollment != nil else { errorText = Copy.notReady; return }
        if quick {
            let recent = QuickAdd.recentAudio(audio)
            guard recent.count >= ClipTools.minClip else { errorText = Copy.notEnoughHistory; return }
            Task { await check(recent, yaw: nil) }
        } else {
            Task {
                do { try await capture.start(audio: audio, model: model, role: .target) }
                catch {
                    audio.note("capture start: \(error)") // the raw error goes to Diagnostics only
                    errorText = AppError.userText(error, fallback: Copy.captureStartFailed)
                }
            }
        }
    }

    private func check(_ pcm: [Float], yaw: Double?) async {
        guard let enrollment = launch.enrollment else { return }
        checking = true
        do {
            let r = try await enrollment.enroll(pcm16k: pcm, role: .target, own: people.own, saved: people.people)
            checking = false
            result = r
            audio.cue(beeps: r.verdict == .fail ? Cue.failure : Cue.success) // 2 success, 3 failure
            if quick && r.verdict != .fail {
                let range = QuickAdd.playbackRange(mask: r.mask, clipLength: r.clip.count)
                do {
                    // false = the route is not AirPods (or another allowed output): nothing played.
                    let played = try await player.play(Array(r.clip[range]), audio: audio)
                    if !played { errorText = Copy.playbackNeedsAirPods }
                } catch {
                    audio.note("quick add playback: \(error)")
                }
                playedBack = true
            }
        } catch {
            checking = false
            audio.note("enrollment check: \(error)") // the raw error goes to Diagnostics only
            errorText = Copy.checkFailed
        }
    }

    private func save(kind: PersonKind, consent: Bool) {
        guard let r = result, let vp = r.voiceprint, let model = launch.model else { return }
        let yaw = quick ? nil : capture.yawDegrees
        let p = Person.enrolled(r, name: displayName, kind: kind, modelHash: model.modelHash, yawDeg: yaw)
        do {
            try people.add(p, clip: r.clip, voiceprint: vp, consent: consent)
            selector.addToTable(p, freshYaw: yaw)
            dismiss()
        } catch {
            audio.note("save person: \(error)")
            errorText = Copy.saveFailed
        }
    }

    /// "Update Maya": Maya keeps her id, name, kind and created date; voice data is replaced.
    private func update(_ existing: Person) {
        guard let r = result, let vp = r.voiceprint, let model = launch.model else { return }
        var p = Person.enrolled(r, name: existing.name, kind: existing.kind, modelHash: model.modelHash,
                                yawDeg: existing.yawDeg)
        p.id = existing.id
        p.created = existing.created
        p.expires = existing.kind == .tonight ? p.expires : nil
        do {
            try people.add(p, clip: r.clip, voiceprint: vp, consent: existing.kind == .saved) // consent given before
            selector.voiceUpdated(existing.id) // on the table: the new voiceprint, warm
            dismiss()
        } catch {
            audio.note("update person: \(error)")
            errorText = Copy.saveFailed
        }
    }
}
