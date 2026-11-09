// EnrollmentService.swift - turns a capture into a checked voiceprint (contract 6, 30-ios 8.3).
//
// Target, in order (thresholds from thresholds.json):
//   1 sbr_db on the full capture             2 clip_pct on the full capture
//   3 trim (ClipTools), e = embed(clip), e_w = embed(each 1.5 s window)
//   4 one_voice = median cos(e_w, e)         5 own_cos = cos(e, own e)
//   6 mask = es_vad(clip) minus frames inside windows that sound like the user
//   7 Voiceprint (es_enroll); ES_ERR_ENROLL_SHORT is a fail
//   8 speech_s = voiceprint.speechSeconds    9 dup = max cos(e, e of saved and tonight people)
// Own voice runs steps 1, 3 and 8 only (role .own, mask = es_vad(clip)); its speech must reach
// own.speech_s (8 s) on the full capture, and its snr_db (= the sbr formula) has its own limits.
// Without WeSpeaker (launch self-test failed) steps 4, 5 and 9 are "not checked" and step 6 keeps
// es_vad(clip).
//
// Verdict: any fail is a fail (the first failing message in step order is shown); else any warn is
// "Accepted, will be OK"; else pass. This actor never touches the engine or the audio thread.

import Foundation
import UIKit

extension EarshotError {
    /// True for ES_ERR_ENROLL_SHORT (fewer than 4 tokens of speech): an enrollment fail, not a crash.
    var isEnrollShort: Bool { status == Int32(truncatingIfNeeded: ES_ERR_ENROLL_SHORT.rawValue) }
}

/// One row of the result card.
struct CheckResult: Equatable, Sendable {
    enum Kind: String, Sendable { case speech, background, oneVoice, notYou, distinct, clipping }
    let kind: Kind
    let value: Double?   // nil when not checked
    let verdict: Verdict
    let message: String? // shown when this check fails
}

struct EnrollmentResult: @unchecked Sendable {
    var verdict: Verdict = .fail
    var message: String? = nil           // first failing message, or "Accepted, will be OK"
    var checks: [CheckResult] = []
    var voiceprint: Voiceprint? = nil    // nil after a fail
    var clip: [Float] = []               // what goes to enroll.wav (at most 10 s)
    var mask: [UInt8] = []               // the mask given to es_enroll (stored in person.json)
    var eWespk: [Float]? = nil
    var quality = Quality(speechS: 0)
    var duplicate: Person? = nil         // set when dup >= dup_cos: ask "Sounds like X - update X?"
    var trim = ClipTools.Trim()
}

actor EnrollmentService {
    private let model: EarshotModel
    private let encoder: SpeakerEmbedding? // nil = run without WeSpeaker
    private let thresholds: Thresholds
    private let appIsActive: @Sendable () async -> Bool

    init(model: EarshotModel, encoder: SpeakerEmbedding?, thresholds: Thresholds,
         appIsActive: @escaping @Sendable () async -> Bool = {
             await MainActor.run { UIApplication.shared.applicationState == .active }
         }) {
        self.model = model
        self.encoder = encoder
        self.thresholds = thresholds
        self.appIsActive = appIsActive
    }

    /// Contract 5 signature. `own` is the user's own voice (targets only); `saved` are the saved and
    /// tonight people compared for duplicates.
    func enroll(pcm16k: [Float], role: VoiceRole, own: Person?, saved: [Person]) async throws -> EnrollmentResult {
        role == .own ? try await enrollOwn(pcm16k) : try await enrollTarget(pcm16k, own: own, saved: saved)
    }

    // MARK: - Target

    private func enrollTarget(_ x: [Float], own: Person?, saved: [Person]) async throws -> EnrollmentResult {
        var r = EnrollmentResult()
        let t = thresholds
        let mask0 = model.vad(x)

        // Steps 1 and 2 on the full capture. With no speech frame at all sbr is undefined (contract 6,
        // quality.py returns None): the row is "not checked", so the speech fail ("Keep the phone on
        // them") leads instead of a wrong "Too noisy".
        let background = Self.backgroundCheck(ClipTools.sbrDB(x, mask0: mask0), verdict: t.sbr, message: Copy.tooNoisy)
        let clipPct = ClipTools.clipPct(x)
        let clipping = CheckResult(kind: .clipping, value: clipPct, verdict: t.clip(clipPct), message: Copy.tooLoud)

        // Step 3: trim. A failed trim means too little speech.
        r.trim = ClipTools.trim(n: x.count, mask0: mask0, role: .target)
        r.clip = ClipTools.clip(x, r.trim)
        let speechFromCapture = ClipTools.roundSeconds(ClipTools.speechSeconds(mask0))
        guard r.trim.failure == nil else {
            let speech = CheckResult(kind: .speech, value: speechFromCapture, verdict: .fail, message: Copy.keepPhoneOnThem)
            r.checks = [background, clipping, notChecked(.oneVoice), notChecked(.notYou), speech, notChecked(.distinct)]
            r.quality = Quality(speechS: speechFromCapture, snrDB: background.value)
            return finish(r)
        }
        let wins = ClipTools.windows(clipLen: r.clip.count)
        var mask = model.vad(r.clip)

        // Steps 3 to 6 with WeSpeaker.
        var oneVoice: CheckResult = notChecked(.oneVoice)
        var notYou: CheckResult = notChecked(.notYou)
        var distinct: CheckResult = notChecked(.distinct)
        if let encoder {
            let active = await appIsActive()
            let e = try await encoder.embed(r.clip, appActive: active)
            var eW: [[Float]] = []
            for w in wins { eW.append(try await encoder.embed(Array(r.clip[w.start..<(w.start + w.length)]), appActive: active)) }
            r.eWespk = e
            if let median = ClipTools.median(eW.map { ClipTools.cosine($0, e) }) {
                oneVoice = CheckResult(kind: .oneVoice, value: median, verdict: t.oneVoice(median),
                                       message: Copy.moreThanOneVoice)
            }
            if let ownE = own?.eWespk {
                let c = ClipTools.cosine(e, ownE)
                notYou = CheckResult(kind: .notYou, value: c, verdict: t.ownCos(c), message: Copy.soundedLikeYou)
                mask = ClipTools.ownWindowMask(mask, windows: wins, windowCosOwn: eW.map { ClipTools.cosine($0, ownE) },
                                               threshold: t.ownWindowCos)
            }
            // Step 9: closest saved or tonight person (asks, never fails). Nobody to compare = distinct.
            distinct = CheckResult(kind: .distinct, value: nil, verdict: .pass, message: nil)
            var best: (person: Person, cosine: Double)? = nil
            for p in saved where p.kind != .own {
                guard let pe = p.eWespk else { continue }
                let c = ClipTools.cosine(e, pe)
                if c > (best?.cosine ?? -.infinity) { best = (p, c) }
            }
            if let best {
                let asks = t.asksDuplicate(best.cosine)
                distinct = CheckResult(kind: .distinct, value: best.cosine, verdict: asks ? .warn : .pass,
                                       message: asks ? Copy.soundsLike(best.person.name) : nil)
                if asks { r.duplicate = best.person }
            }
        }
        r.mask = mask

        // Steps 7 and 8: the voiceprint itself (es_enroll), then the speech check.
        var speech: CheckResult
        do {
            let vp = try Voiceprint(model: model, pcm16k: r.clip, mask: mask, role: .target, weSpeaker: r.eWespk)
            // es_cond_build once now, so a voiceprint the engine cannot use fails here, not at listen time.
            _ = try Conditioning(model: model, mode: .person, target: vp, own: nil, useWeSpeaker: r.eWespk != nil)
            let s = ClipTools.roundSeconds(Double(vp.speechSeconds))
            speech = CheckResult(kind: .speech, value: s, verdict: t.speech(s), message: Copy.keepPhoneOnThem)
            r.voiceprint = vp
            r.quality.speechS = s
        } catch let error as EarshotError where error.isEnrollShort {
            let s = ClipTools.roundSeconds(ClipTools.speechSeconds(mask))
            speech = CheckResult(kind: .speech, value: s, verdict: .fail, message: Copy.keepPhoneOnThem)
            r.quality.speechS = s
        }
        r.quality.snrDB = background.value
        r.quality.consistency = oneVoice.value
        r.quality.ownCos = notYou.value
        // Card order = the step order of 30-ios 8.3, which is also the order of the fail messages.
        r.checks = [background, clipping, oneVoice, notYou, speech, distinct]
        return finish(r)
    }

    // MARK: - Own voice (steps 1, 3 and 8)

    private func enrollOwn(_ x: [Float]) async throws -> EnrollmentResult {
        var r = EnrollmentResult()
        let t = thresholds
        let mask0 = model.vad(x)
        let background = Self.backgroundCheck(ClipTools.sbrDB(x, mask0: mask0), verdict: t.ownSNR,
                                              message: Copy.tooNoisyOwn)
        let speechFull = ClipTools.roundSeconds(ClipTools.speechSeconds(mask0)) // the 8 s rule counts the full capture
        var speech = CheckResult(kind: .speech, value: speechFull, verdict: t.ownSpeech(speechFull),
                                 message: Copy.keepReading)
        // person.json speech_s is the voiceprint's value (contract 6 step 10), set below once es_enroll
        // has run; the card row above keeps the full-capture value the 8 s rule checks.
        r.quality = Quality(speechS: 0, snrDB: background.value)
        r.trim = ClipTools.trim(n: x.count, mask0: mask0, role: .own)
        r.clip = ClipTools.clip(x, r.trim)
        if r.trim.failure == nil {
            r.mask = model.vad(r.clip)
            if let encoder {
                r.eWespk = try await encoder.embed(r.clip, appActive: await appIsActive())
            }
            do {
                let vp = try Voiceprint(model: model, pcm16k: r.clip, mask: r.mask, role: .own, weSpeaker: r.eWespk)
                _ = try Conditioning(model: model, mode: .everyone, target: nil, own: vp, useWeSpeaker: r.eWespk != nil)
                r.voiceprint = vp
                r.quality.speechS = ClipTools.roundSeconds(Double(vp.speechSeconds))
            } catch let error as EarshotError where error.isEnrollShort {
                speech = CheckResult(kind: .speech, value: speechFull, verdict: .fail, message: Copy.keepReading)
            }
        } else {
            speech = CheckResult(kind: .speech, value: speechFull, verdict: .fail, message: Copy.keepReading)
        }
        r.checks = [background, speech]
        return finish(r)
    }

    // MARK: - Verdict

    private func notChecked(_ kind: CheckResult.Kind) -> CheckResult {
        CheckResult(kind: kind, value: nil, verdict: .notChecked, message: nil)
    }

    /// Step 1's card row: sbr_db (snr_db for the own voice) judged by `verdict`, or "not checked"
    /// when `sbr` is nil because the capture has no speech frame (the speech row fails then).
    private static func backgroundCheck(_ sbr: Double?, verdict: (Double) -> Verdict, message: String) -> CheckResult {
        guard let sbr else { return CheckResult(kind: .background, value: nil, verdict: .notChecked, message: nil) }
        return CheckResult(kind: .background, value: sbr, verdict: verdict(sbr), message: message)
    }

    /// Any fail is a fail (first failing message); else any warn is "Accepted, will be OK"; else pass.
    /// The duplicate question is a separate prompt, not a warn of the enrollment itself.
    private func finish(_ result: EnrollmentResult) -> EnrollmentResult {
        var r = result
        if let firstFail = r.checks.first(where: { $0.verdict == .fail }) {
            r.verdict = .fail
            r.message = firstFail.message
            r.voiceprint = nil
        } else if r.checks.contains(where: { $0.verdict == .warn && $0.kind != .distinct }) {
            r.verdict = .warn
            r.message = Copy.acceptedWillBeOK
        } else {
            r.verdict = .pass
            r.message = nil
        }
        if r.voiceprint == nil && r.verdict != .fail { // es_enroll failed without a failing check row
            r.verdict = .fail
            r.message = Copy.keepPhoneOnThem
        }
        return r
    }
}
