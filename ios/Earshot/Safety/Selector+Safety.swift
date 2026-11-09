// Selector+Safety.swift - the Selector's I-4 parts: the thermal / Low Power downshift of the table
// (contract 5.3 "Thermal", 30-ios 6), the 1 s locked check (30-ios 12.3) and the table fields of
// the session log (contract 5.2 `mode`, `n_instances`).
//
// Main actor only, like the rest of Selector. The rules themselves are pure and tested on their own:
// PowerPolicy / PowerSaving (Safety/PowerPolicy.swift) and LockedRule / PTargetHistory
// (Safety/LockedCheck.swift). This file only connects them to the table and the engine.

import Foundation

extension Selector {
    // MARK: - Thermal / Low Power downshift
    // setPowerSaving(_:) is in Selector.swift (it changes `table`, whose setter is private there).

    /// The session log's `mode` and `n_instances` (the slots the engine really computes).
    func telemetryTable() -> (mode: String, instances: Int) {
        let computed = PowerSaving.engineContents(table, saving: powerSaving).filter { $0 != .empty }.count
        return (mode: table.mode.rawValue, instances: computed)
    }

    // MARK: - Locked check (once a second from tick)

    /// For each audible person slot whose 3 s p_target mean is above 0.5, asks the LockedCheck actor
    /// for the cosine between es_embed_g of the slot's last 3 s and the person's g, then applies
    /// LockedRule. Slots that do not play a person (or play at gain 0) are forgotten. A slot whose
    /// person changed is checked again only after 3 s with the new person (PTargetHistory), so the
    /// previous person's p_target and output never count. One check at a time: a slow check is
    /// skipped, never queued.
    func runLockedCheck() {
        guard !lockedCheckBusy, let audio = audioController, let engine = audio.engine,
              let model = loadedModel, let people = peopleStore else { return }
        let playing = audio.state == .running
        let gains = PowerSaving.engineGains(table, saving: powerSaving)
        var jobs: [LockJob] = []
        for slot in 0..<TableState.slotCount {
            // Who the engine slot really holds, whatever its gain (a warm slot at gain 0 still runs).
            if case .person(let id) = loaded[slot] { pTargetHistory.setOwner(slot: slot, id) }
            else { pTargetHistory.setOwner(slot: slot, nil) }
            guard case .person(let id) = table.content(slot), gains[slot] > 0, loaded[slot] == .person(id) else {
                lockRule.forget(slot: slot)
                continue
            }
            lockRule.claim(slot: slot, person: id)
            guard playing, pTargetHistory.isSettled(slot: slot),
                  LockedRule.eligible(gain: gains[slot], pTargetMean: pTargetHistory.mean(slot: slot)),
                  let g = people.voiceprints[id]?.g else { continue } // not talking: the lost count pauses
            jobs.append(LockJob(slot: slot, person: id, g: g))
        }
        publishLockState()
        guard !jobs.isEmpty else { return }
        lockedCheckBusy = true
        let read: @Sendable (Int, Int) -> [Float] = { slot, n in engine.recentOutput(slot: slot, samples: n) }
        let embed: @Sendable ([Float]) throws -> [Float] = { pcm in try model.embedG(pcm) }
        let check = lockedCheck
        Task { @MainActor [weak self] in
            let cosines = await check.cosines(jobs: jobs, read: read, embed: embed)
            self?.finishLockedCheck(jobs: jobs, cosines: cosines)
        }
    }

    /// Applies one finished check. A slot that changed person during the await is not recorded.
    func finishLockedCheck(jobs: [LockJob], cosines: [Int: Float]) {
        lockedCheckBusy = false
        for job in jobs {
            guard case .person(let id) = table.content(job.slot), id == job.person else { continue }
            lockRule.record(slot: job.slot, person: id, cosine: cosines[job.slot], lockedCos: lockedCos)
        }
        publishLockState()
    }

    /// Copies the rule's results into the published values only when they changed.
    func publishLockState() {
        if lockStates != lockRule.states { lockStates = lockRule.states }
        if lostLockWarning != lockRule.warning { lostLockWarning = lockRule.warning }
    }

    /// Listening stopped: no results carry over to the next session.
    func resetLockedCheck() {
        lockRule = LockedRule()
        pTargetHistory.clear()
        publishLockState()
    }
}
