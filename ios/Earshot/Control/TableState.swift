// TableState.swift - who sits in which engine slot, the listening mode and the slot gains
// (30-ios 10.2; contract 5.3 "Start", "Warm instances"; D07, D52). Also ModeConditioning: which
// conditioning each slot gets (30-ios 10.2 last paragraph).
//
// Owner: Selector (main actor). TableState is a plain struct with no audio and no engine calls, so
// TableStateTests can check every rule. Each change returns the slot loads the engine needs; the
// Selector turns them into Conditioning objects and AudioController+Table applies them in order
// (fade to 0, wait 150 ms, load or clear, new gains).
//
// Words used below:
//   slot     - one of the engine's 3 model instances (0, 1, 2), all kept running ("warm")
//   seat     - a table person sitting in a slot (the table holds up to 3 people)
//   Everyone - the "all voices except mine" conditioning; it lives in one slot while it is used
//   warm     - the slot already runs that conditioning, so a switch is only a 150 ms crossfade
//   cold     - the slot gets new conditioning with reset (its model state starts from zero)

import Foundation

/// Focus (one person), Table (every person at the table, summed), Everyone (all voices but mine).
enum TableMode: String, Codable, CaseIterable, Sendable {
    case focus, table, everyone
}

/// What one engine slot is conditioned for.
enum SlotContent: Equatable, Sendable {
    case empty
    case person(UUID)
    case everyone
}

/// One conditioning change for one slot. `content == .empty` means clearConditioning (4.3).
struct SlotLoad: Equatable, Sendable {
    let slot: Int
    let content: SlotContent
    let reset: Bool // true = cold switch (contract 4.3); a clear ignores it
}

/// A table person in a slot.
struct TableSeat: Equatable, Sendable {
    let personID: UUID
    var dir: Double?        // learned head direction (degrees, HeadTracker frame); nil = unknown
    var lastSelected: Int   // TableState clock at the last focus, for "least recently selected"
    /// True when `dir` may belong to another head-tracking frame: person.json yaw_deg from an
    /// earlier session, or a direction from before a yaw jump / AirPods reconnect. Kept per seat
    /// so a fresh direction (a capture just now) never gets mixed up with stale ones. Always false
    /// while `dir` is nil (nothing to be stale).
    var dirStale = false
}

struct TableState: Equatable, Sendable {
    static let slotCount = 3

    private(set) var seats: [TableSeat?] = Array(repeating: nil, count: slotCount)
    /// The slot holding the Everyone conditioning, or nil. With nobody added yet the app starts in
    /// Everyone mode in slot 0 (contract 5.3), which is also what AudioController did before I-3.
    private(set) var everyoneSlot: Int? = 0
    private(set) var mode: TableMode = .everyone
    /// The focused person's slot in Focus mode; remembered while in Table or Everyone mode.
    private(set) var selected: Int? = nil
    private var clock = 0 // counts focus events

    // MARK: - Reading the table

    /// What the engine runs in `slot`. A seated person whose slot holds Everyone is "sent out":
    /// they stay at the table and come back cold.
    func content(_ slot: Int) -> SlotContent {
        if everyoneSlot == slot { return .everyone }
        if let seat = seats[slot] { return .person(seat.personID) }
        return .empty
    }

    var contents: [SlotContent] { (0..<Self.slotCount).map { content($0) } }

    var personCount: Int { seats.compactMap { $0 }.count }

    /// Everyone sitting at the table, in slot order (30-ios 10.3 `filledPersonSlots`).
    var filledPersonSlots: [(slot: Int, personID: UUID, dir: Double?)] {
        seats.indices.compactMap { i in seats[i].map { (slot: i, personID: $0.personID, dir: $0.dir) } }
    }

    /// Slots whose engine instance really runs a person right now (not sent out by Everyone).
    var runningPersonSlots: Set<Int> {
        Set((0..<Self.slotCount).filter { if case .person = content($0) { return true } else { return false } })
    }

    func slot(of id: UUID) -> Int? { seats.firstIndex { $0?.personID == id } }

    /// True when at least one seated person's direction may belong to another head-tracking frame
    /// (TableSeat.dirStale). Look-select then re-anchors on the one talker instead of trusting the
    /// directions (30-ios 10.3), and Diagnostics shows "stale".
    var directionsStale: Bool { seats.contains { $0?.dirStale == true } }

    /// Look-select compares against the current person only in Focus mode (30-ios 10.3).
    var current: Int? { mode == .focus ? selected : nil }

    /// The focused person's id in Focus mode.
    var focusedPerson: UUID? { current.flatMap { seats[$0]?.personID } }

    /// Slot gains for the mode (30-ios 10.2): Focus - the selected slot 1, others 0 (still running);
    /// Table - every person slot 1, summed; Everyone - only the Everyone slot 1.
    var gains: [Float] {
        var g = [Float](repeating: 0, count: Self.slotCount)
        switch mode {
        case .focus:
            if let s = selected { g[s] = 1 }
        case .table:
            for s in runningPersonSlots { g[s] = 1 }
        case .everyone:
            if let s = everyoneSlot { g[s] = 1 }
        }
        return g
    }

    /// The least recently selected seated slot (a 4th person replaces it, 30-ios 10.2).
    func leastRecentlySelected() -> Int? {
        seats.indices.filter { seats[$0] != nil }.min { seats[$0]!.lastSelected < seats[$1]!.lastSelected }
    }

    func mostRecentlySelected() -> Int? {
        seats.indices.filter { seats[$0] != nil }.max { seats[$0]!.lastSelected < seats[$1]!.lastSelected }
    }

    // MARK: - Changes (each returns the slot loads, already merged per slot)

    /// Puts a person at the table (Add person, quick add, hands-free capture, People "add to table").
    /// Already seated - nothing to load. Otherwise the slot is: the first one with nobody and no
    /// Everyone, else one that holds only Everyone, else the least recently selected person's slot
    /// (a 4th person is a cold switch). Then Table mode stays Table (the new person joins the sum);
    /// any other mode focuses the new person, since the user just added them to hear them.
    /// `dirStale` - the direction is person.json's yaw_deg from an earlier session (contract 5.1).
    mutating func add(_ id: UUID, dir: Double?, dirStale: Bool) -> [SlotLoad] {
        var loads: [SlotLoad] = []
        let slot: Int
        if let s = self.slot(of: id) {
            slot = s
            if let d = dir, !dirStale {                  // a fresh capture direction replaces the old one
                seats[s]?.dir = SelectorMath.wrap(d)
                seats[s]?.dirStale = false
            }
        } else {
            if let s = (0..<Self.slotCount).first(where: { seats[$0] == nil && everyoneSlot != $0 }) {
                slot = s
            } else if let s = (0..<Self.slotCount).first(where: { seats[$0] == nil }) {
                slot = s             // this slot holds only Everyone, which the new person replaces
            } else {
                slot = leastRecentlySelected() ?? 0 // all 3 seats full: replace the least recently selected
            }
            if everyoneSlot == slot { everyoneSlot = nil }
            clock += 1
            seats[slot] = TableSeat(personID: id, dir: dir.map(SelectorMath.wrap), lastSelected: clock,
                                    dirStale: dir != nil && dirStale)
            loads.append(SlotLoad(slot: slot, content: .person(id), reset: true))
        }
        if mode != .table { loads += focus(on: slot) }
        return Self.merged(loads)
    }

    /// Focus on the person in `slot` (look-select, cycle, card tap, triple press). Warm, unless the
    /// slot held Everyone and so sent that person out: then they come back cold (contract 5.3).
    mutating func focus(on slot: Int) -> [SlotLoad] {
        guard slot >= 0, slot < Self.slotCount, let seat = seats[slot] else { return [] }
        var loads: [SlotLoad] = []
        if everyoneSlot == slot {
            everyoneSlot = nil
            loads.append(SlotLoad(slot: slot, content: .person(seat.personID), reset: true))
        }
        clock += 1
        seats[slot]?.lastSelected = clock
        selected = slot
        mode = .focus
        return loads
    }

    /// Switches mode. Returns nil (and changes nothing) when the mode is impossible: Focus needs a
    /// person, Table needs at least 2.
    mutating func setMode(_ m: TableMode) -> [SlotLoad]? {
        switch m {
        case .focus:
            let keep = selected.flatMap { seats[$0] != nil ? $0 : nil }
            guard let s = keep ?? mostRecentlySelected() else { return nil }
            return focus(on: s)
        case .table:
            guard personCount >= 2 else { return nil }
            var loads: [SlotLoad] = []
            if let e = everyoneSlot, let seat = seats[e] { // Everyone sent this person out: back, cold
                everyoneSlot = nil
                loads.append(SlotLoad(slot: e, content: .person(seat.personID), reset: true))
            }
            mode = .table
            return loads
        case .everyone:
            var loads: [SlotLoad] = []
            if everyoneSlot == nil { // already loaded = warm; otherwise a free slot, else the selected person's
                let free = (0..<Self.slotCount).first { seats[$0] == nil }
                let sel = selected.flatMap { seats[$0] != nil ? $0 : nil }
                let s = free ?? sel ?? leastRecentlySelected() ?? 0
                everyoneSlot = s
                loads.append(SlotLoad(slot: s, content: .everyone, reset: true))
            }
            mode = .everyone
            return loads
        }
    }

    /// The triple-press target: Focus <-> Table with >= 2 people, else Focus <-> Everyone
    /// (contract 5.3). nil when nothing can change (nobody at the table, already Everyone).
    func toggleTarget() -> TableMode? {
        if mode == .focus { return personCount >= 2 ? .table : .everyone }
        return personCount >= 1 ? .focus : nil
    }

    /// A person leaves (deleted, or removed from the table). Their slot is emptied (unless it holds
    /// Everyone), then the mode follows: nobody left -> Everyone; Table with 1 person -> Focus on
    /// them; Focus on the person who left -> Focus on the most recently selected person left.
    mutating func remove(_ id: UUID) -> [SlotLoad] {
        guard let s = slot(of: id) else { return [] }
        seats[s] = nil
        var loads: [SlotLoad] = []
        if everyoneSlot != s { loads.append(SlotLoad(slot: s, content: .empty, reset: false)) }
        if selected == s { selected = nil }
        if personCount == 0 {
            loads += setMode(.everyone) ?? []
        } else if (mode == .table && personCount < 2) || (mode == .focus && selected == nil) {
            loads += setMode(.focus) ?? []
        }
        return Self.merged(loads)
    }

    // MARK: - Directions

    /// Direction learning sets one person's direction (30-ios 10.4). A learned direction comes from
    /// the current yaw, so it is fresh. (The Selector never learns on a stale seat: one step would
    /// pull a single old-frame direction halfway into the new frame and spoil the re-anchor offset.)
    mutating func setDirection(slot: Int, _ dir: Double?) {
        guard slot >= 0, slot < Self.slotCount, seats[slot] != nil else { return }
        seats[slot]?.dir = dir.map(SelectorMath.wrap)
        seats[slot]?.dirStale = false
    }

    /// Stale re-anchor (30-ios 10.3) for the one talker, who is where the user looks (`yaw`).
    /// - Talker stale: its old direction and every other stale direction share the old frame, so
    ///   one offset (yaw - old) moves them all into the current frame, and they become fresh.
    ///   Fresh directions are already in the current frame and do not move.
    /// - Talker fresh or without a direction: only the talker's direction becomes `yaw`. The stale
    ///   ones have no known offset, so they stay stale until a stale talker re-anchors them.
    mutating func reanchor(slot: Int, yaw: Double) {
        guard slot >= 0, slot < Self.slotCount, let talker = seats[slot] else { return }
        if talker.dirStale {
            var staleDirs: [Int: Double] = [:]
            for i in seats.indices { if let s = seats[i], s.dirStale, let d = s.dir { staleDirs[i] = d } }
            for (i, d) in SelectorMath.reanchor(dirs: staleDirs, talker: slot, yaw: yaw) {
                seats[i]?.dir = d
                seats[i]?.dirStale = false
            }
        } else {
            seats[slot]?.dir = SelectorMath.wrap(yaw)
            seats[slot]?.dirStale = false
        }
    }

    /// A yaw jump, an AirPods connect / disconnect, or a new session (30-ios 10.1): every known
    /// direction may now be in another frame.
    mutating func markDirectionsStale() {
        for i in seats.indices where seats[i]?.dir != nil { seats[i]?.dirStale = true }
    }

    /// Keeps only the last load per slot, in order: a clear followed by a load of the same slot
    /// (the last person leaves and Everyone moves into that slot) becomes just the load.
    static func merged(_ loads: [SlotLoad]) -> [SlotLoad] {
        var out: [SlotLoad] = []
        for l in loads {
            out.removeAll { $0.slot == l.slot }
            out.append(l)
        }
        return out
    }
}

// MARK: - Which conditioning a slot gets (30-ios 10.2)

/// person   - Conditioning(mode: .person,   target: vp,  own: ownVP, useWeSpeaker: u)
/// Everyone - Conditioning(mode: .everyone, target: nil, own: ownVP, useWeSpeaker: u)
/// where u = the encoder self-test passed AND every voiceprint involved has WeSpeaker vectors
/// (contract 2.3: with one vector missing the engine would fall back to "both null" anyway).
enum ModeConditioning {
    /// The choice without the C objects, so tests can check it without an engine.
    struct Choice: Equatable {
        let mode: ListenMode
        let target: UUID?          // nil for Everyone
        let useWeSpeaker: Bool
    }

    enum Failure: Error, Equatable { case voiceprintNotReady(UUID) }

    static func useWeSpeaker(selfTestPassed: Bool, involvedHaveWeSpeaker: [Bool]) -> Bool {
        selfTestPassed && involvedHaveWeSpeaker.allSatisfy { $0 }
    }

    /// nil for an empty slot. `targetHasWeSpeaker` is only read for a person slot; `ownHasWeSpeaker`
    /// is nil when no own voice is enrolled (setup mode: nothing to check for it).
    static func choice(for content: SlotContent, targetHasWeSpeaker: Bool, ownHasWeSpeaker: Bool?,
                       selfTestPassed: Bool) -> Choice? {
        let own = ownHasWeSpeaker.map { [$0] } ?? []
        switch content {
        case .empty:
            return nil
        case .person(let id):
            return Choice(mode: .person, target: id,
                          useWeSpeaker: useWeSpeaker(selfTestPassed: selfTestPassed,
                                                     involvedHaveWeSpeaker: [targetHasWeSpeaker] + own))
        case .everyone:
            return Choice(mode: .everyone, target: nil,
                          useWeSpeaker: useWeSpeaker(selfTestPassed: selfTestPassed, involvedHaveWeSpeaker: own))
        }
    }

    /// Builds the Conditioning for one slot (es_cond_build, never on the audio thread or the worker).
    /// It runs on the main actor: measured 22 us p50 for a person and 17 us for Everyone on the M4
    /// Mac with random_v1 (I-3 fix, 2026-11-07); the phone is not measured yet.
    /// nil for an empty slot; throws when a person's voiceprint is not built.
    static func build(_ content: SlotContent, model: EarshotModel, voiceprint: (UUID) -> Voiceprint?,
                      own: Voiceprint?, selfTestPassed: Bool) throws -> Conditioning? {
        var target: Voiceprint? = nil
        if case .person(let id) = content {
            guard let vp = voiceprint(id) else { throw Failure.voiceprintNotReady(id) }
            target = vp
        }
        guard let c = choice(for: content, targetHasWeSpeaker: target?.hasWeSpeaker ?? false,
                             ownHasWeSpeaker: own?.hasWeSpeaker, selfTestPassed: selfTestPassed) else { return nil }
        return try Conditioning(model: model, mode: c.mode, target: target, own: own, useWeSpeaker: c.useWeSpeaker)
    }
}
