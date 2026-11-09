// AudioController+Table.swift - how the table, the settings and hold-to-compare reach the engine,
// plus start step 3 (configure every filled slot) and the conditioning chain helpers
// (30-ios 5 "Conditioning" and "Emptying a slot", 10.2, 12.1, 13 Settings; contract 4.3, 5.3).
//
// Main actor only (rule 2 of 30-ios 0: only the main actor writes to the engine).
//
// The one rule that matters here (contract 4.3): a slot whose conditioning changes is first faded
// to gain 0 and given xfade_ms (150 ms) before it is loaded or emptied, so nothing ever jumps in
// level. The steps are planned by the pure `TableApply.steps` (unit-tested) and run in order on
// the same task chain as every other conditioning call, so two quick stem presses never interleave.
//
// Params while paused: pause() sends "all slot gains 0, room off" to the engine and keeps the real
// values in `params`. Anything that changes params while paused only updates `params`; resume()
// sends them (30-ios 11 "Resume ... restores gains and room").
//
// Fading slots: applyTable stores the final gains in `params` at once, but a slot waiting for its
// load or clear must stay silent until that step ran. `fadingSlots` counts those waits, and every
// params push (sendParams, configure) holds such a slot at 0. Without this, a voice slider change
// or a resume inside the 150 ms wait would turn the slot up while it still plays the old person,
// and the cold load would then happen at full gain.

import Foundation

/// One step of a table change, as planned by TableApply.steps.
enum TableStep: Equatable {
    case gains([Float])            // es_engine_set_params with these slot gains
    case wait(ms: Int)             // let the slot-gain crossfade finish
    case load(slot: Int, reset: Bool)
    case clear(slot: Int)          // clearConditioning: the worker skips the slot from then on
}

/// What AudioController sent to the engine, for tests (AudioController.onEngineEvent).
enum EngineEvent: Equatable {
    case gains([Float])            // slot gains as really sent (fading slots already at 0)
    case load(slot: Int, reset: Bool)
    case clear(slot: Int)
    case cue(Int)                  // beeps
}

enum TableApply {
    /// The order of 30-ios 5: every slot whose conditioning changes goes to gain 0 at once (the
    /// other slots go straight to their new gains), wait the crossfade, load or clear, then set the
    /// final gains (the loaded slots fade in over the crossfade). Pure gain changes are one step.
    static func steps(newGains: [Float], loads: [SlotLoad], crossfadeMS: Int = 150) -> [TableStep] {
        guard !loads.isEmpty else { return [.gains(newGains)] }
        var quiet = newGains
        for l in loads where l.slot >= 0 && l.slot < quiet.count { quiet[l.slot] = 0 }
        var steps: [TableStep] = [.gains(quiet), .wait(ms: crossfadeMS)]
        for l in loads {
            steps.append(l.content == .empty ? .clear(slot: l.slot) : .load(slot: l.slot, reset: l.reset))
        }
        steps.append(.gains(newGains))
        return steps
    }
}

/// One slot's conditioning for configure() at start and at every engine swap.
struct SlotConditioning {
    let slot: Int
    let conditioning: Conditioning
}

extension AudioController {
    // MARK: - The task chain

    /// Runs `work` after every conditioning change queued before it (30-ios 5 condChain).
    func runOnCondChain(_ work: @escaping @MainActor () async -> Void) {
        let previous = condChain
        condChain = Task { @MainActor in
            await previous?.value
            await work()
        }
    }

    /// Waits until every queued conditioning change has run (tests, and code that must see the end).
    func condChainIdle() async {
        await condChain?.value
    }

    // MARK: - Table changes

    /// Applies one table change: `loads` with their Conditioning (nil = clear) and the final gains.
    /// The final gains are stored in `params` at once, so a start or engine swap that happens in
    /// the middle configures the new engine with the new table.
    func applyTable(_ loads: [(SlotLoad, Conditioning?)], gains: [Float]) {
        let crossfade = Int(params.crossfadeMS.rounded())
        params.slotGain = gains
        guard engine != nil else { return } // not listening: configure() reads the table at start
        let plain = loads.map { $0.0 }
        let byslot = Dictionary(loads.map { ($0.0.slot, $0.1) }, uniquingKeysWith: { _, last in last })
        let steps = TableApply.steps(newGains: gains, loads: plain, crossfadeMS: crossfade)
        for l in plain { fadingSlots[l.slot, default: 0] += 1 } // held at 0 until its step below ran
        runOnCondChain { [weak self] in
            guard let self else { return }
            for step in steps {
                switch step {
                case .gains(let g):
                    self.pushSlotGains(g)
                case .wait(let ms):
                    try? await Task.sleep(nanoseconds: UInt64(ms) * 1_000_000)
                case .load(let slot, let reset):
                    if let c = byslot[slot] ?? nil {
                        do {
                            try await self.engine?.setConditioning(slot: slot, c, resetState: reset)
                            self.onEngineEvent?(.load(slot: slot, reset: reset))
                        } catch { self.note("set_cond slot \(slot) failed: \(error)") } // after 200 ms of BUSY
                    }
                    self.finishedFading(slot)
                case .clear(let slot):
                    do {
                        try await self.engine?.clearConditioning(slot: slot)
                        self.onEngineEvent?(.clear(slot: slot))
                    } catch { self.note("clear slot \(slot) failed: \(error)") }
                    self.finishedFading(slot)
                }
            }
        }
    }

    /// One queued load or clear of `slot` has run: that slot may be heard again (unless another
    /// queued change still waits for it).
    func finishedFading(_ slot: Int) {
        let left = (fadingSlots[slot] ?? 0) - 1
        fadingSlots[slot] = left > 0 ? left : nil
    }

    /// `gains` with every slot that still waits for its load or clear held at 0.
    func audibleGains(_ gains: [Float]) -> [Float] {
        var g = gains
        for slot in fadingSlots.keys where slot >= 0 && slot < g.count { g[slot] = 0 }
        return g
    }

    /// The one place that sends params to the running engine (pause and configure aside): the
    /// fading slots are held at 0 first, and during an alarm passthrough bypass is on at no more
    /// than -6 dB (I-4, safetyApplied in AudioController+Safety.swift).
    func sendParams(_ p: EngineParams) {
        guard let e = engine else { return }
        var held = p
        held.slotGain = audibleGains(p.slotGain)
        e.setParams(safetyApplied(held))
        onEngineEvent?(.gains(held.slotGain))
    }

    /// Sends these slot gains (with the other params) unless the user paused (then the quiet pause
    /// params stay in the engine and resume() sends `params`).
    func pushSlotGains(_ gains: [Float]) {
        var p = params
        p.slotGain = gains
        if state != .paused { sendParams(p) }
    }

    /// Sends `params` unless paused (see the file header).
    func pushParams() {
        if state != .paused { sendParams(params) }
    }

    // MARK: - Settings and hold-to-compare (30-ios 12.1, 13)

    /// Voice level, room sound, mute while I talk, gate, leveling, max gain and trim. Slot gains
    /// and bypass are not settings and are kept.
    func applySettings(_ s: ListenSettings) {
        params = s.applied(to: params)
        pushParams()
        setAlarmEnabled(s.alarmPassthrough) // I-4: "Let alarm sounds through" (default on)
    }

    /// Hold-to-compare: plays the delayed microphone instead of the model while held (bypass).
    func setBypass(_ on: Bool) {
        guard params.bypass != on else { return }
        params.bypass = on
        pushParams()
    }

    // MARK: - Every filled slot (configure at start and swap, own-voice rebuild)

    /// Step 3: params, then conditioning for every filled slot with resetState true (slots with
    /// nobody stay empty; a new engine starts with every slot empty, contract 4.3). A user pause
    /// survives: the new engine gets the quiet pause params, and resume() sends the real ones.
    func configure(_ e: EarshotEngine) async throws {
        if userPaused {
            var quiet = params
            quiet.slotGain = [0, 0, 0]
            quiet.roomDB = -60
            e.setParams(safetyApplied(quiet))
        } else {
            var p = params
            p.slotGain = audibleGains(params.slotGain) // a slot still waiting for its load stays at 0
            e.setParams(safetyApplied(p))              // an alarm passthrough survives a swap
        }
        for s in try filledSlots() {
            try await e.setConditioning(slot: s.slot, s.conditioning, resetState: true)
        }
    }

    /// Everyone mode with the own voice (if any); u follows 30-ios 10.2 (ModeConditioning).
    func everyoneConditioning() throws -> Conditioning {
        guard let model else { throw AppError.notReady }
        guard let c = try ModeConditioning.build(.everyone, model: model, voiceprint: { _ in nil },
                                                 own: ownVoiceprint, selfTestPassed: useWeSpeaker) else {
            throw AppError.notReady
        }
        return c
    }

    /// Chains every conditioning change through one task, so two quick changes never interleave (30-ios 5).
    func applyConditioning(slot: Int, _ c: Conditioning, reset: Bool) {
        runOnCondChain { [weak self] in
            do { try await self?.engine?.setConditioning(slot: slot, c, resetState: reset) }
            catch { self?.note("set_cond slot \(slot) failed: \(error)") }
        }
    }


    /// The conditioning for every filled slot: from the Selector once it is attached, else Everyone
    /// in slot 0 (the I-0 / I-2 behaviour, still used by tests that run without a Selector).
    func filledSlots() throws -> [SlotConditioning] {
        if let provider = slotProvider { return provider() }
        return [SlotConditioning(slot: 0, conditioning: try everyoneConditioning())]
    }
}
