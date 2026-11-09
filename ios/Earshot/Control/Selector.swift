// Selector.swift - the contract Selector: look-select, cycle, setMode, direction learning, the table
// and its slot conditioning, stem presses, Now Playing and the cues (contract 5, 5.3; 30-ios 10,
// 11, the cue parts of 12; D50, D51, D52).
//
// Main actor only. It owns the TableState (who sits in which engine slot) and turns every table
// change into conditioning for AudioController (AudioController+Table fades, loads, clears).
// The math lives in SelectorMath.swift and TableState.swift, so the tests need no audio.
//
// Wiring (attach): AudioController.slotProvider / onPoll / onAudioStarted / onAudioStopped,
// PeopleStore.onPersonDeleted, HeadTracker.onFrameReset, RemoteCommands.onCommand. Stop also calls
// sessionStopped() (EarshotApp), which writes learned directions to person.json yaw_deg.
//
// Name note: Objective-C also has a type called `Selector`. Inside the app our class wins; code in
// another module (the tests) writes `Earshot.Selector`.

import Foundation

/// What the beeps mean (30-ios 10.3, 11, 12.2; contract 5.3). Every beep is 60 ms at 880 Hz,
/// -18 dBFS, made by the engine (es_engine_cue) and heard even while muted.
enum Cue {
    static let captureStarted = 1   // hands-free capture: recording
    static let success = 2          // a person was added
    static let failure = 3          // capture or checks failed
    static let alarm = 3            // alarm passthrough (I-4)

    /// Look-select and card taps: 1 to 3 beeps name the slot (slot 0 = 1 beep).
    static func slot(_ s: Int) -> Int { min(3, max(1, s + 1)) }

    /// Triple press: 1 / 2 / 3 beeps = Focus / Table / Everyone.
    static func mode(_ m: TableMode) -> Int {
        switch m {
        case .focus: return 1
        case .table: return 2
        case .everyone: return 3
        }
    }
}

@MainActor
final class Selector: ObservableObject {
    @Published private(set) var table = TableState()
    /// The talking dot per slot (p_target > 0.5 on a slot running a person, 30-ios 13).
    @Published private(set) var talking: [Bool] = [false, false, false]
    /// True while a stem double press is recording someone hands-free.
    @Published private(set) var handsFreeActive = false
    /// I-4: true while the thermal / Low Power downshift keeps only the selected slot computed
    /// (Safety/Selector+Safety.swift, Safety/PowerPolicy.swift).
    @Published var powerSaving = false
    /// I-4 locked check: per slot "locked" (check mark on the card), "lost" or not known yet, and
    /// the banner flag "Might be following someone else" (Safety/LockedCheck.swift).
    @Published var lockStates: [LockState] = [.unknown, .unknown, .unknown]
    @Published var lostLockWarning = false

    let head: HeadTracker
    let remote: RemoteCommands

    private var talk = TalkTracker()
    /// Directions learned or captured in this session, by person (written to yaw_deg at Stop).
    private(set) var sessionDirections: [UUID: Double] = [:]
    /// People whose sessionDirections value is from before a yaw jump or AirPods reconnect (another
    /// frame): re-seating them loads that direction as stale, not fresh.
    private(set) var staleSessionDirections: Set<UUID> = []
    private weak var audio: AudioController?
    private weak var people: PeopleStore?
    private var model: EarshotModel?
    private var enrollment: EnrollmentService?
    private let capture = CaptureSession()
    private var tickCount = 0
    /// What the engine was last told to hold per slot (after power-saving masking). apply() plans
    /// loads and clears from it with PowerSaving.reconcile; engineSlots() resets it.
    var loaded: [SlotContent] = TableState().contents
    // Locked check state (Safety/Selector+Safety.swift). `lockedCos` comes from thresholds.json.
    var lockedCos: Float = 0
    var lockRule = LockedRule()
    var pTargetHistory = PTargetHistory()
    let lockedCheck = LockedCheck()
    var lockedCheckBusy = false

    /// Tests may pass their own tracker and command center; the app uses new ones.
    init(head: HeadTracker? = nil, remote: RemoteCommands? = nil) {
        self.head = head ?? HeadTracker()
        self.remote = remote ?? RemoteCommands()
    }

    /// Connects the selector once the launch sequence has a model (EarshotApp).
    func attach(audio: AudioController, people: PeopleStore, model: EarshotModel, enrollment: EnrollmentService?) {
        self.audio = audio
        self.people = people
        self.model = model
        self.enrollment = enrollment
        audio.slotProvider = { [weak self] in self?.engineSlots() ?? [] }
        audio.onPoll = { [weak self] stats in self?.tick(stats) }
        audio.onAudioStarted = { [weak self] in self?.listeningStarted() }
        audio.onAudioStopped = { [weak self] in self?.listeningStopped() }
        people.onPersonDeleted = { [weak self] p in self?.personDeleted(p) }
        head.onFrameReset = { [weak self] in self?.frameReset() }
        remote.onCommand = { [weak self] kind in self?.handle(kind) }
        capture.yawProvider = { [weak head] in head?.yawForCapture() }
        capture.onEnd = { [weak self] phase in self?.handsFreeEnded(phase) }
        // I-4 (Safety/Selector+Safety.swift): the thermal downshift and the session log's table fields.
        audio.onPowerSavingChanged = { [weak self] on in self?.setPowerSaving(on) }
        audio.telemetryTable = { [weak self] in self?.telemetryTable() ?? (mode: "everyone", instances: 0) }
        if audio.powerLevel != .normal { setPowerSaving(true) } // already hot or in Low Power Mode
    }

    /// A person's display name, or "" when unknown.
    func name(_ id: UUID) -> String { people?.person(id)?.name ?? "" }

    /// "Focus: Maya", "Table: 2 people", "Everyone" or "Paused" (Listen header and Now Playing).
    var statusText: String {
        let paused = audio.map { $0.state == .paused || $0.state == .interrupted } ?? false
        return NowPlayingText.status(mode: table.mode, focusName: table.focusedPerson.map(name),
                                     tablePeople: table.personCount, paused: paused)
    }

    // MARK: - Contract API: lookSelect, cycle, setMode (30-ios 10.3)

    /// Stem double press or the Look button.
    func lookSelect(now: TimeInterval = HeadTracker.now) {
        let seated = table.filledPersonSlots
        guard !seated.isEmpty else { return startHandsFreeCapture() }
        let snap = head.snapshot(now: now, directionsStale: table.directionsStale)
        let decision = SelectorMath.lookSelect(people: seated.map { .init(slot: $0.slot, dir: $0.dir) },
                                               current: table.current, head: snap,
                                               onlyTalker: talk.onlyPersonTalking(now: now))
        switch decision {
        case .handsFreeCapture: startHandsFreeCapture()
        case .focus(let s): focus(on: s)
        case .reanchor(let s, let yaw):
            table.reanchor(slot: s, yaw: yaw) // every direction shifts by the same offset
            rememberDirections()
            focus(on: s)
        case .cycle: cycle()
        }
    }

    /// The next filled slot after the current one in slot order, then Focus on it.
    func cycle() {
        let filled = table.filledPersonSlots.map { $0.slot }
        guard let next = SelectorMath.cycleTarget(filled: filled, current: table.current) else { return }
        focus(on: next)
    }

    /// Focus / Table / Everyone from the mode switch or the triple press; 1 / 2 / 3 beeps.
    /// An impossible mode (Table with fewer than 2 people, Focus with nobody) changes nothing and
    /// beeps the current mode, so a hands-free user hears where they still are.
    func setMode(_ m: TableMode) {
        // While saving power only one slot is computed, so Table (a sum of people) is refused too.
        if powerSaving && m == .table { return cue(Cue.mode(table.mode)) }
        guard let loads = table.setMode(m) else { return cue(Cue.mode(table.mode)) }
        apply(loads)
        cue(Cue.mode(m))
    }

    /// Focus on one person: warm, slot gains crossfade, Now Playing, slot + 1 beeps. Also what
    /// tapping a person's card does.
    func focus(on slot: Int, beep: Bool = true) {
        guard table.seats.indices.contains(slot), table.seats[slot] != nil else { return }
        apply(table.focus(on: slot))
        if beep { cue(Cue.slot(slot)) }
    }

    /// Triple press: Focus <-> Table with >= 2 people, else Focus <-> Everyone.
    func toggleMode() {
        guard let target = table.toggleTarget() else { return cue(Cue.mode(table.mode)) }
        setMode(target)
    }

    // MARK: - People at the table

    /// Puts a person at the table and (outside Table mode) focuses them, with no beep (the
    /// enrollment flows play their own). `freshYaw` is a direction measured now (the capture's
    /// yaw); without it a direction learned this session is used, else person.json yaw_deg, which
    /// always loads as stale (contract 5.1).
    func addToTable(_ person: Person, freshYaw: Double?) {
        guard person.kind != .own, people?.voiceprints[person.id] != nil else { return }
        var dir: Double? = nil
        var stale = false
        if let y = freshYaw {
            dir = y
            sessionDirections[person.id] = SelectorMath.wrap(y)
            staleSessionDirections.remove(person.id)
        } else if let y = sessionDirections[person.id] {
            dir = y
            stale = staleSessionDirections.contains(person.id) // learned before a frame reset
        } else if let y = person.yawDeg {
            dir = y
            stale = true
        }
        apply(table.add(person.id, dir: dir, dirStale: stale))
    }

    /// People "remove from table": the slot fades to 0 for 150 ms, then is emptied.
    func removeFromTable(_ id: UUID) {
        apply(table.remove(id))
    }

    /// A person's voice was re-recorded ("Update Maya"): their slot gets the new conditioning, warm.
    func voiceUpdated(_ id: UUID) {
        guard let s = table.slot(of: id), table.content(s) == .person(id) else { return }
        apply([SlotLoad(slot: s, content: .person(id), reset: false)])
    }

    /// PeopleStore.onPersonDeleted: a person on the table fades to 0, then clearConditioning
    /// (contract 5.3 "Warm instances").
    func personDeleted(_ p: Person) {
        sessionDirections[p.id] = nil
        staleSessionDirections.remove(p.id)
        removeFromTable(p.id)
    }

    /// I-4, AudioController.onPowerSavingChanged (contract 5.3 "Thermal"). On: every slot but the
    /// selected one fades to gain 0, waits 150 ms and is emptied with clearConditioning (no engine
    /// rebuild); Table mode becomes Focus on the selected person, since Table is a sum of people.
    /// Off (60 calm seconds): the emptied slots are refilled cold. Both go through apply(), which
    /// plans the loads with PowerSaving.reconcile (Safety/PowerPolicy.swift). Here and not in
    /// Selector+Safety.swift because it changes `table`, whose setter is private to this file.
    func setPowerSaving(_ on: Bool) {
        guard on != powerSaving else { return }
        powerSaving = on
        var loads: [SlotLoad] = []
        if on, table.mode == .table, let s = PowerSaving.keptSlot(table) {
            loads = table.focus(on: s) // no beep: the user did not ask for this switch
        }
        apply(loads)
        let kept = PowerSaving.keptSlot(table).map { "slot \($0)" } ?? "no slot"
        audio?.note(on ? "power saving on: only \(kept) computed" : "power saving off: slots refilled cold")
    }

    /// HeadTracker.onFrameReset (a yaw jump or an AirPods connect / disconnect): every direction
    /// known so far, on the table or remembered for this session, may be in the old frame.
    func frameReset() {
        table.markDirectionsStale()
        staleSessionDirections = Set(sessionDirections.keys)
    }

    // MARK: - Engine slots

    /// Every filled slot's conditioning (configure at start and swap, own-voice rebuild). A slot
    /// whose person has no voiceprint yet is left empty (fail quiet). While saving power only the
    /// kept slot is filled (PowerSaving.engineContents).
    func engineSlots() -> [SlotConditioning] {
        var out: [SlotConditioning] = []
        let contents = PowerSaving.engineContents(table, saving: powerSaving)
        loaded = contents
        for (slot, content) in contents.enumerated() {
            do {
                if let c = try conditioning(for: content) { out.append(SlotConditioning(slot: slot, conditioning: c)) }
            } catch {
                audio?.note("table: slot \(slot) not loaded: \(error)")
            }
        }
        return out
    }

    private func conditioning(for content: SlotContent) throws -> Conditioning? {
        guard let audio, let model else { throw AppError.notReady }
        return try ModeConditioning.build(content, model: model, voiceprint: { [weak people] in people?.voiceprints[$0] },
                                          own: audio.ownVoiceprint, selfTestPassed: audio.useWeSpeaker)
    }

    /// Builds the conditioning for each load and hands the change to AudioController. A load whose
    /// conditioning cannot be built becomes a clear, so that slot is silent rather than wrong.
    /// I-4: the loads go through PowerSaving.reconcile first. Without power saving that returns the
    /// table's loads unchanged; while saving it empties every slot but the kept one, and on
    /// recovery it refills the emptied slots cold. The gains hold uncomputed slots at 0.
    func apply(_ loads: [SlotLoad]) {
        guard let audio else { return }
        let want = PowerSaving.engineContents(table, saving: powerSaving)
        let plan = PowerSaving.reconcile(tableLoads: loads, loaded: loaded, want: want)
        loaded = want
        var built: [(SlotLoad, Conditioning?)] = []
        for l in plan {
            do {
                built.append((l, try conditioning(for: l.content)))
            } catch {
                audio.note("table: slot \(l.slot) emptied: \(error)")
                built.append((SlotLoad(slot: l.slot, content: .empty, reset: false), nil))
            }
        }
        audio.applyTable(built, gains: PowerSaving.engineGains(table, saving: powerSaving))
        refreshNowPlaying()
    }

    func cue(_ beeps: Int) { audio?.cue(beeps: beeps) }

    /// The audio controller this selector drives (for Selector+Safety.swift).
    var audioController: AudioController? { audio }
    /// The people store (for the locked check's voiceprints).
    var peopleStore: PeopleStore? { people }
    /// The loaded model (for the locked check's es_embed_g).
    var loadedModel: EarshotModel? { model }
}

// MARK: - 10 Hz tick, stem presses, hands-free capture, Now Playing, session start / stop

extension Selector {
    /// Called by AudioController's 10 Hz stats poll: head flags, talking dots, direction learning
    /// (30-ios 10.4) and, once a second, the Now Playing card.
    func tick(_ stats: EngineStats, now: TimeInterval = HeadTracker.now) {
        head.refresh(now: now)
        let personSlots = table.runningPersonSlots
        talk.update(pTarget: stats.pTarget, personSlots: personSlots, now: now)
        let dots = (0..<TableState.slotCount).map { s in
            personSlots.contains(s) && s < stats.pTarget.count && stats.pTarget[s] > 0.5
        }
        if dots != talking { talking = dots }
        learnDirections(now: now)
        pTargetHistory.add(stats.pTarget)                    // I-4: the locked check's 3 s means
        tickCount += 1
        if tickCount % 10 == 0 {
            refreshNowPlaying()
            runLockedCheck()                                 // I-4: once a second (30-ios 12.3)
        }
    }

    /// For each table person talking (p_target > 0.7) for 1 s while the head turns slower than
    /// 20 deg/s: dir += 0.1 * wrap(yaw - dir), at most once a second per person.
    /// A seat whose direction is stale is skipped: a step would move that one old-frame direction
    /// part of the way into the current frame, and the re-anchor (which computes one offset from
    /// the talker and applies it to the other stale people) would then move everyone else wrong.
    func learnDirections(now: TimeInterval) {
        let snap = head.snapshot(now: now, directionsStale: table.directionsStale)
        guard snap.live, let yaw = snap.smoothedYaw, let rate = snap.rate else { return }
        for s in table.runningPersonSlots.sorted() where talk.learnDue(slot: s, headRate: rate, now: now) {
            guard let seat = table.seats[s], !seat.dirStale else { continue } // wait for the re-anchor
            table.setDirection(slot: s, SelectorMath.learn(dir: seat.dir, yaw: yaw))
        }
        rememberDirections()
    }

    /// Copies the fresh table directions into `sessionDirections`. Stale ones belong to another
    /// frame and are not copied (person.json keeps its older value for them).
    func rememberDirections() {
        for seat in table.seats.compactMap({ $0 }) where !seat.dirStale {
            guard let d = seat.dir else { continue }
            sessionDirections[seat.personID] = d
            staleSessionDirections.remove(seat.personID)
        }
    }

    /// One AirPods stem press (or Lock Screen button), mapped by StemMap (30-ios 11).
    func handle(_ kind: RemoteCommandKind) {
        guard let audio else { return }
        let paused = audio.state == .paused || audio.state == .interrupted
        switch StemMap.action(for: kind, isPaused: paused, tableEmpty: table.personCount == 0) {
        case .pause: audio.pause()
        case .resume: audio.resume()
        case .lookSelect: lookSelect()
        case .handsFreeCapture: startHandsFreeCapture()
        case .toggleMode: toggleMode()
        case .nothing: break
        }
        refreshNowPlaying()
    }

    // MARK: Hands-free capture (double press with nobody at the table, DECISIONS M.6 default)

    /// Records a target from the engine's mic history like "Add person": 1 beep at start (from
    /// CaptureSession), 2 when the person was added, 3 on failure.
    func startHandsFreeCapture() {
        guard !handsFreeActive else { return } // a second double press while recording is ignored
        guard let audio, audio.isListening, let model, enrollment != nil else { return cue(Cue.failure) }
        handsFreeActive = true
        Task { @MainActor in
            do {
                try await self.capture.start(audio: audio, model: model, role: .target)
            } catch {
                self.handsFreeActive = false
                audio.note("hands-free capture: \(error)")
                self.cue(Cue.failure)
            }
        }
    }

    /// CaptureSession.onEnd for the hands-free capture.
    func handsFreeEnded(_ phase: CaptureSession.Phase) {
        guard handsFreeActive else { return }
        guard phase == .finished else { // failed (3 beeps already) or cancelled
            handsFreeActive = false
            return
        }
        let pcm = capture.captured
        let yaw = capture.yawDegrees
        Task { @MainActor in await self.finishHandsFree(pcm, yaw: yaw) }
    }

    /// Runs the 6 checks; a pass or warn saves a tonight person "Person N" and seats them (no
    /// screen is open to ask about a duplicate, so it is kept as a new person, like "Keep both").
    private func finishHandsFree(_ pcm: [Float], yaw: Double?) async {
        defer { handsFreeActive = false }
        guard let enrollment, let people, let model, let audio else { return }
        do {
            let r = try await enrollment.enroll(pcm16k: pcm, role: .target, own: people.own, saved: people.people)
            guard r.verdict != .fail, let vp = r.voiceprint else { return cue(Cue.failure) }
            let name = Copy.nextPersonName(taken: people.people.map { $0.name }) // never a duplicate "Person N"
            let p = Person.enrolled(r, name: name, kind: .tonight, modelHash: model.modelHash, yawDeg: yaw)
            try people.add(p, clip: r.clip, voiceprint: vp)
            addToTable(p, freshYaw: yaw)
            cue(Cue.success)
        } catch {
            audio.note("hands-free add: \(error)")
            cue(Cue.failure)
        }
    }

    // MARK: Now Playing and the session

    /// Title "Earshot - Focus: Maya", artist = the delay chip text (30-ios 11).
    func refreshNowPlaying() {
        guard let audio, remote.active else { return }
        let paused = audio.state == .paused || audio.state == .interrupted
        let title = NowPlayingText.title(mode: table.mode, focusName: table.focusedPerson.map(name),
                                         tablePeople: table.personCount, paused: paused)
        remote.update(title: title, artist: audio.latency.text, paused: paused)
    }

    /// AudioController.onAudioStarted: Now Playing and head tracking (30-ios 6 step 5).
    func listeningStarted() {
        remote.start()
        head.start()
        refreshNowPlaying()
    }

    /// AudioController.onAudioStopped: Now Playing and head tracking off, no hands-free capture.
    func listeningStopped() {
        capture.cancel()
        handsFreeActive = false
        remote.stop()
        head.stop()
        talk = TalkTracker()
        if talking.contains(true) { talking = [false, false, false] }
        resetLockedCheck() // I-4: the next session starts with no lock results
    }

    /// Stop: learned directions are written to person.json yaw_deg (30-ios 10.4), then forgotten as
    /// "fresh", since the next session may have another yaw frame.
    func sessionStopped() {
        rememberDirections()
        if let people {
            for (id, dir) in sessionDirections {
                guard var p = people.person(id), p.kind != .own, p.yawDeg != dir else { continue }
                p.yawDeg = dir
                try? people.save(p)
            }
        }
        sessionDirections = [:]
        staleSessionDirections = []
        table.markDirectionsStale()
    }
}
