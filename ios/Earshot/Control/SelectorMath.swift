// SelectorMath.swift - the pure math behind picking by looking (30-ios 10.3, 10.4; contract 5.3;
// D50, D51). No audio, no Core Motion, no UI: SelectorTests checks every rule here.
//
// Owns: angle wrap and circular mean, the look-select decision, the cycle order, the direction
// learning step, the stale re-anchor, and TalkTracker (who has been talking for 1 s).
// Called from: Selector (main actor) and HeadTracker / TableState. Any thread is fine, it is pure.
//
// Directions and yaw are degrees in the AirPods yaw frame. Only differences are ever used.

import Foundation

enum SelectorMath {
    static let selectRadius = 35.0     // nearest stored direction must be within 35 degrees
    static let hysteresis = 10.0       // and beat the current person's direction by 10 degrees
    static let learnRate = 0.1         // dir += 0.1 * wrap(yaw - dir)
    static let talkThreshold: Float = 0.7  // p_target above this counts as "talking"
    static let talkHold = 1.0          // seconds of talking before a learning step / re-anchor
    static let maxHeadRate = 20.0      // degrees per second: no learning while the head turns faster

    /// Maps any angle to [-180, 180). wrap(190) = -170, wrap(180) = -180, wrap(-190) = 170.
    static func wrap(_ a: Double) -> Double {
        var x = (a + 180).truncatingRemainder(dividingBy: 360)
        if x < 0 { x += 360 }
        if x >= 360 { x -= 360 } // -1e-15 + 360 rounds to 360.0
        return x - 180
    }

    /// atan2(mean sin, mean cos) in degrees, or nil when the angles cancel out (for example 90 and
    /// 270) or the list is empty. A naive average of 350 and 10 would give 180; this gives 0.
    static func circularMean(_ degrees: [Double]) -> Double? {
        guard !degrees.isEmpty else { return nil }
        var sx = 0.0, sy = 0.0
        for d in degrees {
            sx += cos(d * .pi / 180)
            sy += sin(d * .pi / 180)
        }
        let length = (sx * sx + sy * sy).squareRoot() / Double(degrees.count)
        guard length > 1e-6 else { return nil }
        return atan2(sy, sx) * 180 / .pi
    }

    /// One table person for look-select: the slot and the learned direction (nil = unknown).
    struct Candidate: Equatable {
        let slot: Int
        let dir: Double?
    }

    /// What a double press (or the Look button) should do.
    enum LookDecision: Equatable {
        case handsFreeCapture                 // nobody at the table
        case focus(Int)                       // the person you face
        case reanchor(slot: Int, yaw: Double) // directions stale: the one talker is where you look
        case cycle                            // no usable direction: next person in slot order
    }

    /// The decision of 30-ios 10.3, written as a pure function.
    /// - `current`: the focused slot in Focus mode, nil in Table or Everyone mode (nobody is current).
    /// - `onlyTalker`: the one table person talking for the last 1 s, or nil (TalkTracker).
    static func lookSelect(people: [Candidate], current: Int?, head: HeadSnapshot, onlyTalker: Int?) -> LookDecision {
        guard !people.isEmpty else { return .handsFreeCapture }
        if head.live, !head.stale, let y = head.smoothedYaw {
            // The nearest stored direction ...
            let withDir = people.compactMap { p in p.dir.map { (slot: p.slot, d: abs(wrap(y - $0))) } }
            if let best = withDir.min(by: { $0.d < $1.d }) {
                // ... compared with the current person's own distance (infinity when there is none).
                let curDir = current.flatMap { c in people.first { $0.slot == c }?.dir }
                let dCur = curDir.map { abs(wrap(y - $0)) } ?? .infinity
                // within 35 degrees, not the current person, and at least 10 degrees closer (contract 5.3)
                if best.d <= selectRadius, best.slot != current, best.d + hysteresis <= dCur {
                    return .focus(best.slot)
                }
            }
        } else if head.live, head.stale, let y = head.smoothedYaw, let talker = onlyTalker,
                  people.contains(where: { $0.slot == talker }) {
            return .reanchor(slot: talker, yaw: y)
        }
        return .cycle
    }

    /// The next filled slot after `current` in slot order (wrapping around); the first filled slot
    /// when nobody is current. nil when the table is empty.
    static func cycleTarget(filled: [Int], current: Int?) -> Int? {
        let sorted = filled.sorted()
        guard let first = sorted.first else { return nil }
        guard let c = current else { return first }
        return sorted.first { $0 > c } ?? first
    }

    /// One learning step: dir += 0.1 * wrap(yaw - dir), or dir = yaw when unknown. The result is
    /// wrapped to [-180, 180) so a direction never drifts past +-180 (wrap-around at the back).
    static func learn(dir: Double?, yaw: Double) -> Double {
        guard let d = dir else { return wrap(yaw) }
        return wrap(d + learnRate * wrap(yaw - d))
    }

    /// Stale re-anchor: the talker's direction becomes `yaw` and every other known direction shifts
    /// by the same offset, so the angles between people are kept. When the talker had no direction
    /// there is no offset to apply, so only the talker's direction is set.
    static func reanchor(dirs: [Int: Double], talker: Int, yaw: Double) -> [Int: Double] {
        guard let old = dirs[talker] else {
            var out = dirs
            out[talker] = wrap(yaw)
            return out
        }
        let offset = wrap(yaw - old)
        return dirs.mapValues { wrap($0 + offset) }
    }
}

/// Who has been talking, from the 10 Hz `p_target` stats (30-ios 10.3 onlyPersonTalking, 10.4).
/// Two clocks per slot: `aboveSince` (p_target > 0.7 without a break since then) and `lastLearn`
/// (the last learning step), so learning restarts its 1 s timer while the talking clock keeps going.
struct TalkTracker: Equatable {
    private(set) var aboveSince: [TimeInterval?] = Array(repeating: nil, count: TableState.slotCount)
    private(set) var lastLearn: [TimeInterval?] = Array(repeating: nil, count: TableState.slotCount)

    /// One stats tick. Only `personSlots` count (a slot running Everyone is not a person).
    mutating func update(pTarget: [Float], personSlots: Set<Int>, now: TimeInterval) {
        for s in 0..<aboveSince.count {
            let p = s < pTarget.count ? pTarget[s] : 0
            if personSlots.contains(s) && p > SelectorMath.talkThreshold {
                if aboveSince[s] == nil { aboveSince[s] = now }
            } else {
                aboveSince[s] = nil
            }
        }
    }

    /// True when slot `s` has talked for at least `seconds` up to `now`.
    func talked(_ s: Int, for seconds: Double, now: TimeInterval) -> Bool {
        guard let since = aboveSince[s] else { return false }
        return now - since >= seconds - 1e-9
    }

    /// Exactly one table person whose p_target stayed above 0.7 for the last `last` seconds, else nil.
    func onlyPersonTalking(last: Double = SelectorMath.talkHold, now: TimeInterval) -> Int? {
        let talkers = (0..<aboveSince.count).filter { talked($0, for: last, now: now) }
        return talkers.count == 1 ? talkers[0] : nil
    }

    /// The learning gate of 30-ios 10.4: talking for 1 s since the later of "started talking" and
    /// "last step", and the head turning slower than 20 deg/s. When it returns true the step is
    /// recorded, which restarts that person's 1 s timer (at most one update per second).
    mutating func learnDue(slot s: Int, headRate: Double, now: TimeInterval) -> Bool {
        guard let since = aboveSince[s], headRate < SelectorMath.maxHeadRate else { return false }
        let timerStart = max(since, lastLearn[s] ?? -.infinity)
        guard now - timerStart >= SelectorMath.talkHold - 1e-9 else { return false }
        lastLearn[s] = now
        return true
    }
}
