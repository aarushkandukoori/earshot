// PowerPolicy.swift - the thermal and Low Power downshift (contract 5.3 "Thermal", 30-ios 6
// "Thermal and Low Power", D53). Pure logic, no audio and no engine calls, so PowerPolicyTests can
// check every row of the table below.
//
// Who calls it: AudioController (main actor) feeds it on every thermal or Low Power notification and
// once a second while listening (the 60 s recovery timer needs a clock tick, iOS sends nothing when
// 60 calm seconds have passed). Selector (main actor) uses PowerSaving to decide which slots the
// engine keeps while saving power.
//
// The downshift table (contract 5.3, 30-ios 6, 12.2):
//
//   thermalState         Low Power   level      engine slots computed           alarm classifier
//   nominal / fair       off         normal     every table slot                3 s windows, 50% overlap
//   nominal / fair       on          saving     only the selected slot          3 s windows, every 3 s
//   serious              any         saving     only the selected slot          3 s windows, every 3 s
//   critical             any         critical   only the selected slot, paused  3 s windows, every 3 s
//
// Recovery: from saving back to normal only after thermalState is .fair or better AND Low Power is
// off for 60 s in a row; any non-calm moment restarts the 60 s. Critical always drops to saving
// first (never straight to normal), and Earshot stays paused until the user resumes.
// "No engine rebuild": the engine keeps its 3 instances; saving empties slots with clearConditioning.

import Foundation

/// How hard Earshot is saving power right now.
enum PowerLevel: String, Equatable, Sendable {
    case normal     // every table slot is computed
    case saving     // only the selected slot is computed (banner "Saving power ...")
    case critical   // saving, and the session is paused ("iPhone is hot - Earshot paused")
}

struct PowerPolicy: Equatable {
    /// Calm seconds needed before the other slots are refilled (contract 5.3).
    static let recoverySeconds: TimeInterval = 60

    private(set) var level: PowerLevel = .normal
    /// When the current calm stretch began (only tracked while not normal); nil = not calm now.
    private(set) var calmSince: TimeInterval? = nil

    /// Calm = thermalState .nominal or .fair, and Low Power Mode off.
    static func isCalm(thermal: ProcessInfo.ThermalState, lowPower: Bool) -> Bool {
        (thermal == .nominal || thermal == .fair) && !lowPower
    }

    /// Feeds one reading (a notification or the once-a-second tick) and returns the new level.
    /// `now` is any steadily increasing clock in seconds (the app passes systemUptime).
    @discardableResult
    mutating func update(thermal: ProcessInfo.ThermalState, lowPower: Bool, now: TimeInterval) -> PowerLevel {
        if thermal == .critical {
            level = .critical
            calmSince = nil
            return level
        }
        let calm = Self.isCalm(thermal: thermal, lowPower: lowPower)
        switch level {
        case .normal:
            if !calm {
                level = .saving
                calmSince = nil
            }
        case .saving, .critical:
            level = .saving // critical always passes through saving
            if !calm {
                calmSince = nil // the 60 s start again from the next calm reading
            } else {
                let since = calmSince ?? now
                calmSince = since
                if now - since >= Self.recoverySeconds {
                    level = .normal
                    calmSince = nil
                }
            }
        }
        return level
    }

    // MARK: - What each level means (the right-hand columns of the table above)

    /// True when only the selected slot may be computed.
    static func keepsOnlySelected(_ level: PowerLevel) -> Bool { level != .normal }

    /// The alarm classifier's window overlap (30-ios 12.2): 0.5 normally, 0 while saving power, so
    /// a new 3 s window is classified every 1.5 s normally and every 3 s while saving.
    static func alarmOverlap(_ level: PowerLevel) -> Double { level == .normal ? 0.5 : 0 }

    /// Seconds between two classified windows (3 s windows times one minus the overlap).
    static func alarmHopSeconds(_ level: PowerLevel) -> Double { 3.0 * (1 - alarmOverlap(level)) }
}

// MARK: - Which slots the engine keeps while saving power

/// Turns the table into what the engine should hold, with or without power saving, and plans the
/// loads and clears that get the engine there. Pure; Selector applies the result.
enum PowerSaving {
    /// The one slot kept while saving: Focus - the selected person; Everyone - the Everyone slot;
    /// Table - the selected (else the most recently selected) person, because saving leaves Table
    /// mode for Focus on that person (Selector.setPowerSaving).
    static func keptSlot(_ t: TableState) -> Int? {
        switch t.mode {
        case .focus: return t.selected
        case .everyone: return t.everyoneSlot
        case .table: return t.selected.flatMap { t.seats[$0] != nil ? $0 : nil } ?? t.mostRecentlySelected()
        }
    }

    /// What each engine slot should run: the table's contents, or while saving only the kept slot
    /// (every other slot empty, so the worker skips it).
    static func engineContents(_ t: TableState, saving: Bool) -> [SlotContent] {
        guard saving else { return t.contents }
        let keep = keptSlot(t)
        return (0..<TableState.slotCount).map { $0 == keep ? t.content($0) : .empty }
    }

    /// The table's slot gains with every slot that is not computed held at 0.
    static func engineGains(_ t: TableState, saving: Bool) -> [Float] {
        let contents = engineContents(t, saving: saving)
        return t.gains.enumerated().map { contents[$0.offset] == .empty ? 0 : $0.element }
    }

    /// The loads and clears that take the engine from `loaded` (what it holds now) to `want`.
    /// TableState's own loads come first and keep their reset flags (a warm reload stays warm);
    /// a load for a slot that `want` leaves empty is dropped (or becomes a clear if the slot still
    /// holds something). Then every other slot whose content differs is added in slot order: a slot
    /// that should be empty is cleared, and a slot that gets new content is loaded cold (reset),
    /// which is how recovery "refills them cold" (contract 5.3).
    /// Without power saving, `want` is the table's contents and the result equals `tableLoads`
    /// (PowerPolicyTests checks this over many table changes).
    static func reconcile(tableLoads: [SlotLoad], loaded: [SlotContent], want: [SlotContent]) -> [SlotLoad] {
        var out: [SlotLoad] = []
        var done = Set<Int>()
        for l in TableState.merged(tableLoads) where l.slot >= 0 && l.slot < want.count {
            done.insert(l.slot)
            if want[l.slot] == .empty {
                if loaded[l.slot] != .empty { out.append(SlotLoad(slot: l.slot, content: .empty, reset: false)) }
            } else {
                out.append(SlotLoad(slot: l.slot, content: want[l.slot], reset: l.reset))
            }
        }
        for slot in want.indices where !done.contains(slot) && loaded[slot] != want[slot] {
            out.append(SlotLoad(slot: slot, content: want[slot], reset: want[slot] != .empty))
        }
        return out
    }
}
