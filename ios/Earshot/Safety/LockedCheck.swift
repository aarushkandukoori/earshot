// LockedCheck.swift - "is Earshot still following the right person?" (30-ios 12.3, contract 5.3
// "Locked check", D52).
//
// Every 1 s, for each slot that runs a person, is audible (gain > 0) and whose mean p_target over
// the last 3 s is above 0.5: take the last 3 s of that slot's model output (recentOutput, 48,000
// samples), compute es_embed_g on it (EarshotModel.embedG, 128 floats, unit length) and compare it
// with the person's own g by cosine. Below `locked_cos` (thresholds.json) counts as lost. Lost for
// 5 s while p_target > 0.5 shows "Might be following someone else - double-press or re-add".
// There is no automatic gain change in v1; the card shows a check mark while locked.
//
// Three parts:
//   LockedRule      - pure decision rule per slot (unit-tested with fake cosines)
//   PTargetHistory  - pure ring of the last 3 s of p_target polls (10 Hz), for the eligibility mean
//   LockedCheck     - actor that reads the output and runs es_embed_g off the main actor
// Who calls what: Selector (main actor, Safety/Selector+Safety.swift) feeds PTargetHistory from the
// 10 Hz stats poll, asks the actor for cosines once a second and applies LockedRule to the answer.

import Foundation

/// What the card shows for one slot.
enum LockState: Equatable, Sendable {
    case unknown   // not checked yet (or not talking since the slot changed)
    case locked    // the last check matched the person: check mark
    case lost      // the last check did not match
}

/// The decision rule, one entry per engine slot.
struct LockedRule: Equatable {
    /// Lost for 5 s = 5 lost checks at one per second while the person talks.
    static let checksToWarn = 5
    /// The p_target mean that counts as "this person is talking" (contract 5.3).
    static let talkingP: Float = 0.5

    private(set) var states: [LockState] = Array(repeating: .unknown, count: TableState.slotCount)
    /// Lost checks in a row per slot. A check that is not run (person silent, slot muted) neither
    /// adds nor resets; a locked result resets it to 0.
    private(set) var lostChecks: [Int] = Array(repeating: 0, count: TableState.slotCount)
    /// The person each slot's results belong to; a different person in the slot starts fresh.
    private(set) var owners: [UUID?] = Array(repeating: nil, count: TableState.slotCount)

    /// True when some slot has been lost for 5 s of talking: show the banner.
    var warning: Bool { lostChecks.contains { $0 >= Self.checksToWarn } }

    /// Only an audible person slot whose 3 s p_target mean is above 0.5 is checked.
    static func eligible(gain: Float, pTargetMean: Float) -> Bool {
        gain > 0 && pTargetMean > talkingP
    }

    /// The slot now plays `person`: if its results belong to someone else, forget them first.
    mutating func claim(slot: Int, person: UUID) {
        guard states.indices.contains(slot), owners[slot] != person else { return }
        forget(slot: slot)
        owners[slot] = person
    }

    /// One check result for `slot` playing `person`: the cosine, or nil when the slot was not
    /// checked this second (the count then pauses).
    mutating func record(slot: Int, person: UUID, cosine: Float?, lockedCos: Float) {
        claim(slot: slot, person: person)
        guard states.indices.contains(slot), let c = cosine else { return }
        if c.isFinite && c >= lockedCos {
            states[slot] = .locked
            lostChecks[slot] = 0
        } else {
            states[slot] = .lost          // a NaN cosine is never a match
            lostChecks[slot] += 1
        }
    }

    /// The slot stopped playing a person (switched away, emptied, another person, paused):
    /// forget its results, so a new person never inherits an old warning.
    mutating func forget(slot: Int) {
        guard states.indices.contains(slot) else { return }
        states[slot] = .unknown
        lostChecks[slot] = 0
        owners[slot] = nil
    }

    /// Cosine of two vectors (dot product over both lengths); 0 when either has no length or the
    /// sizes differ. g vectors are unit length already, but the division keeps it honest.
    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        guard na > 0, nb > 0 else { return 0 }
        return dot / (na.squareRoot() * nb.squareRoot())
    }
}

/// The last 3 s of p_target per slot, from the 10 Hz stats poll (30 readings).
/// It also remembers which person each slot held and how many polls ago that started: when a slot
/// changes person, its old readings and its last 3 s of output still belong to the previous person,
/// so the slot is checked again only after 30 new polls (3 s) with the new person.
struct PTargetHistory: Equatable {
    static let capacity = 30   // 3 s at 10 Hz

    private(set) var readings: [[Float]] = []
    /// The person each slot held at the last setOwner call (nil = no person).
    private(set) var owners: [UUID?] = Array(repeating: nil, count: TableState.slotCount)
    /// Polls since that slot's person last changed, capped at `capacity`.
    private(set) var freshPolls: [Int] = Array(repeating: 0, count: TableState.slotCount)

    mutating func add(_ pTarget: [Float]) {
        readings.append(pTarget)
        if readings.count > Self.capacity { readings.removeFirst(readings.count - Self.capacity) }
        for s in freshPolls.indices { freshPolls[s] = min(freshPolls[s] + 1, Self.capacity) }
    }

    mutating func clear() {
        readings.removeAll()
        owners = Array(repeating: nil, count: TableState.slotCount)
        freshPolls = Array(repeating: 0, count: TableState.slotCount)
    }

    /// The slot now holds `person` (nil = empty or not a person). A change starts its 3 s over.
    mutating func setOwner(slot: Int, _ person: UUID?) {
        guard owners.indices.contains(slot), owners[slot] != person else { return }
        owners[slot] = person
        freshPolls[slot] = 0
    }

    /// True once the slot has held the same person for 3 s of polls (its readings and its last
    /// 3 s of output then hold no one else).
    func isSettled(slot: Int) -> Bool {
        freshPolls.indices.contains(slot) && freshPolls[slot] >= Self.capacity
    }

    /// Mean p_target of `slot` over the readings since its person last changed; 0 with none.
    func mean(slot: Int) -> Float {
        let n = freshPolls.indices.contains(slot) ? freshPolls[slot] : readings.count
        let values = readings.suffix(n).compactMap { slot < $0.count ? $0[slot] : nil }
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Float(values.count)
    }
}

/// One slot to check: which slot, who should be playing in it, and that person's g (128 floats).
struct LockJob: Sendable {
    let slot: Int
    let person: UUID
    let g: [Float]
}

/// Runs the slow part (reading 3 s of output and es_embed_g, a few ms of C, estimate) off the main
/// actor. One check at a time, since it is an actor.
actor LockedCheck {
    /// 3 s of 16 kHz output (contract 5.3: "the last 3 s of an instance's output").
    static let samples = 48_000

    /// The cosine for every job whose output is long enough and whose embedding worked.
    /// `read` returns a slot's newest output samples, `embed` turns audio into a g vector; the app
    /// passes EarshotEngine.recentOutput and EarshotModel.embedG, the tests pass fakes.
    func cosines(jobs: [LockJob],
                 read: @Sendable (Int, Int) -> [Float],
                 embed: @Sendable ([Float]) throws -> [Float]) -> [Int: Float] {
        var out: [Int: Float] = [:]
        for job in jobs {
            let audio = read(job.slot, Self.samples)
            guard audio.count >= Self.samples else { continue } // the slot has not played 3 s yet
            guard let g = try? embed(audio) else { continue }    // es_embed_g found no speech: skip
            out[job.slot] = LockedRule.cosine(g, job.g)
        }
        return out
    }
}
