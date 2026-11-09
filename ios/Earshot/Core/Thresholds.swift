// Thresholds.swift - decodes `thresholds.json` and turns a measured value into pass / warn / fail
// (contract 6, 30-ios 8.3). The numbers always come from the file, never from code.
//
// The file (contract 6):
//   {"version": 1, "model_hash": "...",
//    "target": {"speech_s": [4, 5], "sbr_db": [0, 6], "one_voice": [0.45, 0.60],
//               "own_cos": [0.50, 0.60], "dup_cos": 0.75, "clip_pct": [0.1, 1.0]},
//    "own": {"speech_s": 8, "snr_db": [15, 20]}, "own_window_cos": 0.60, "locked_cos": 0.0}
//
// Each pair is [lo, hi]. Two kinds of check:
//   higher is better (speech_s, sbr_db, one_voice, own snr_db): pass x >= hi, warn lo <= x < hi, fail x < lo
//   lower is better  (own_cos, clip_pct):                       pass x < lo,  warn lo <= x < hi, fail x >= hi
// So every boundary value lands on the kinder side except the fail edge of a lower-is-better check,
// which the contract table writes as ">= 0.60 fail" and ">= 1 % fail".

import Foundation

/// The outcome of one check. `notChecked` is grey on the result card (no WeSpeaker, or no speech frame for sbr).
enum Verdict: String, Equatable, Sendable {
    case pass, warn, fail, notChecked
}

struct Thresholds: Decodable, Equatable, Sendable {
    struct Target: Decodable, Equatable, Sendable {
        let speechS: [Double]
        let sbrDB: [Double]
        let oneVoice: [Double]
        let ownCos: [Double]
        let dupCos: Double
        let clipPct: [Double]

        enum CodingKeys: String, CodingKey {
            case speechS = "speech_s", sbrDB = "sbr_db", oneVoice = "one_voice", ownCos = "own_cos"
            case dupCos = "dup_cos", clipPct = "clip_pct"
        }
    }

    struct Own: Decodable, Equatable, Sendable {
        let speechS: Double
        let snrDB: [Double]

        enum CodingKeys: String, CodingKey { case speechS = "speech_s", snrDB = "snr_db" }
    }

    let version: Int
    let modelHash: String
    let target: Target
    let own: Own
    let ownWindowCos: Double
    let lockedCos: Double

    enum CodingKeys: String, CodingKey {
        case version, modelHash = "model_hash", target, own, ownWindowCos = "own_window_cos", lockedCos = "locked_cos"
    }

    /// Decodes and checks the file: version 1, and every pair has two numbers with lo <= hi.
    static func decode(_ data: Data) throws -> Thresholds {
        let t = try JSONDecoder().decode(Thresholds.self, from: data)
        guard t.version == 1 else { throw ContainerError("thresholds.json version \(t.version), expected 1") }
        let pairs = [t.target.speechS, t.target.sbrDB, t.target.oneVoice, t.target.ownCos, t.target.clipPct, t.own.snrDB]
        for p in pairs where p.count != 2 || p[0] > p[1] {
            throw ContainerError("thresholds.json has a bad [lo, hi] pair \(p)")
        }
        return t
    }

    // MARK: - The two rules

    /// pass x >= hi, warn lo <= x < hi, fail x < lo.
    static func higherIsBetter(_ x: Double, _ pair: [Double]) -> Verdict {
        if x >= pair[1] { return .pass }
        if x >= pair[0] { return .warn }
        return .fail
    }

    /// pass x < lo, warn lo <= x < hi, fail x >= hi.
    static func lowerIsBetter(_ x: Double, _ pair: [Double]) -> Verdict {
        if x < pair[0] { return .pass }
        if x < pair[1] { return .warn }
        return .fail
    }

    // MARK: - Target checks (30-ios 8.3 table)

    func speech(_ seconds: Double) -> Verdict { Self.higherIsBetter(seconds, target.speechS) }
    func sbr(_ db: Double) -> Verdict { Self.higherIsBetter(db, target.sbrDB) }
    func oneVoice(_ cosine: Double) -> Verdict { Self.higherIsBetter(cosine, target.oneVoice) }
    func ownCos(_ cosine: Double) -> Verdict { Self.lowerIsBetter(cosine, target.ownCos) }
    func clip(_ percent: Double) -> Verdict { Self.lowerIsBetter(percent, target.clipPct) }
    /// True when the new voice is close enough to a known person to ask "Sounds like X - update X?".
    func asksDuplicate(_ cosine: Double) -> Bool { cosine >= target.dupCos }

    // MARK: - Own-voice checks

    /// Own voice needs `own.speech_s` seconds of speech on the full capture; there is no warn band.
    func ownSpeech(_ seconds: Double) -> Verdict { seconds >= own.speechS ? .pass : .fail }
    func ownSNR(_ db: Double) -> Verdict { Self.higherIsBetter(db, own.snrDB) }
}
