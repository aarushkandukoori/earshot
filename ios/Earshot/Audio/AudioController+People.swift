// AudioController+People.swift - how the own voice reaches the engine (30-ios 8.3, 8.5, 9.2).
// Main actor only.
//
// Only one thing changes the engine here: the user's own voice. It is the negative in every
// slot's conditioning (contract 2.3), so after a new own voice every filled slot (the table's
// people and Everyone, AudioController+Table filledSlots) is rebuilt with resetState false
// (30-ios 8.3 "Own voice"). The quick-add history ("Add who just spoke") with its swap-only RAM
// copy also lives here.

import Foundation

extension AudioController {
    /// Installs (or removes, with nil) the own voiceprint. Unmuting depends on it (setup mode).
    func setOwnVoice(_ voiceprint: Voiceprint?) {
        ownVoiceprint = voiceprint
        hasOwnVoice = voiceprint != nil // its didSet runs refreshMute
        rebuildRunningSlots()
    }

    /// Sets the launch WeSpeaker result (30-ios 9.4 step 4) and rebuilds the running slots with it.
    func setUseWeSpeaker(_ use: Bool) {
        guard use != useWeSpeaker else { return }
        useWeSpeaker = use
        rebuildRunningSlots()
    }

    /// Rebuilds every filled slot without resetting the streams (warm).
    /// Goes through applyConditioning, so it is chained after any change still in flight.
    func rebuildRunningSlots() {
        guard engine != nil else { return } // not listening: configure() uses the new values at start
        do {
            for s in try filledSlots() { applyConditioning(slot: s.slot, s.conditioning, reset: false) }
        } catch {
            note("own voice: conditioning failed: \(error)")
        }
    }

    // MARK: - Quick-add history (30-ios 8.5, contract 5.3)

    /// The last `want` samples for "Add who just spoke": the running engine's own history, with the
    /// RAM copy of the last swap in front only while that engine is still the one the swap installed
    /// and has less than `want` samples of its own. Anything else drops the RAM copy for good.
    func quickAddHistory(samples want: Int = QuickAdd.samples) -> [Float] {
        guard let e = engine else { return [] }
        let fresh = e.recentInput(samples: want)
        if quickAddRAMEngine !== e || fresh.count >= want { clearQuickAddRAM() }
        return QuickAdd.combine(ram: quickAddRAM, fresh: fresh, want: want)
    }

    /// Frees the RAM copy (the conversation audio it holds is gone after this).
    func clearQuickAddRAM() {
        quickAddRAM = []
        quickAddRAMEngine = nil
    }

    /// The swap's quick-add step: `ram` (made from the old engine) now belongs to the new engine.
    func handOverQuickAddRAM(_ ram: [Float], to new: EarshotEngine) {
        quickAddRAM = ram
        quickAddRAMEngine = new
    }

    /// Called by the 10 Hz poll: frees the RAM copy as soon as quick add can no longer use it
    /// (another engine runs, or the swap's engine already holds 8 s of its own history).
    func dropStaleQuickAddRAM() {
        guard !quickAddRAM.isEmpty else { return }
        if let e = engine, quickAddRAMEngine === e,
           es_engine_input_position(e.cEngine) < UInt64(QuickAdd.samples) { return } // still needed
        clearQuickAddRAM()
    }
}
