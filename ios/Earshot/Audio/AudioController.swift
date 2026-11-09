// AudioController.swift - start, stop, the one engine swap path, routes, interruptions
// (30-ios 6, contract 5 and 5.3).
//
// Who owns what:
//   AVAudioSession  - configured by SessionConfig, owned here
//   eh_host (C)     - the RemoteIO unit and its render callback (AudioHost.c)
//   EarshotEngine   - the C engine the callback renders through
//
// Three rules (30-ios 0): Swift never runs on the audio thread; only the main actor
// writes to the engine; any doubt about the route or the engine mutes.
//
// The class is split over five files so each reads in one sitting:
//   AudioController.swift          - state, start / stop, mute rules, pause, the engine swap
//   AudioController+System.swift   - route changes, interruptions, media reset, thermal
//   AudioController+Measure.swift  - latency probe and prompts, 10 Hz stats and meter, buffer check
//   AudioController+People.swift   - own voice, WeSpeaker switch, quick-add RAM copy (I-2)
//   AudioController+Table.swift    - table changes (fade, load / clear, gains), settings, bypass (I-3)
//   AudioController+Safety.swift   - alarm passthrough, feedback banner, session log, compute stats (I-4)
// I-1 adds `phase0` (counters the Phase 0 runner reads) and `mallocCounter` (debug builds only).
//
// I-0 scope: everything below runs with the stub engine. I-2 adds the own voice
// (`ownVoiceprint`, AudioController+People.swift) and the stop hook for "tonight" expiry.
// I-3: the Selector owns the table (Control/) and hands every filled slot to configure() through
// `slotProvider`; the 10 Hz poll feeds it through `onPoll`; `onAudioStarted` starts Now Playing and
// head tracking. Without a Selector (tests) slot 0 holds Everyone, as before.
// I-4: start step 5 also starts the alarm listener and the session log minute (startSafety); the
// locked check runs in the Selector from the 10 Hz poll; thermal / Low Power is in +System.

import AVFoundation

enum AudioState: Equatable {
    case idle               // no session, no I/O
    case running            // I/O running, output allowed
    case paused             // I/O running, output muted (user pause, or waiting for a tap)
    case badRoute           // I/O running, muted, output is not headphones
    case interrupted        // a call or another app took the audio; iOS stopped I/O
    case failed(String)
}

enum SwapReason: String { case route, bufferSize = "buffer size", interruption, forced }
enum RecreateReason: String { case rate, mediaReset = "media services reset" }

@MainActor
final class AudioController: ObservableObject {
    // MARK: Published state (the screens read these)
    @Published var route: RouteInfo = .none
    @Published var latency: LatencyDisplay = .unknown
    @Published var state: AudioState = .idle
    @Published var stats = EngineStats()
    @Published var meter = InputMeter()
    @Published private(set) var hasWorkgroup = false
    @Published var isUnmuted = false
    @Published var isMeasuring = false
    @Published var notice: String? = nil     // one banner line
    @Published var notes: [String] = []                   // Diagnostics log, newest last (see note())
    @Published private(set) var engineFrames = 0          // io_frames the current engine was built with
    @Published var lastProbeMessage = ""

    /// Diagnostics test switch: allow unmuting before an own voice exists (I-0 pass-through test).
    @Published var allowListenWithoutOwnVoice = false { didSet { refreshMute() } }

    /// Set by PeopleStore once an own voice is enrolled (I-2). Until then Earshot stays muted (setup mode).
    var hasOwnVoice = false { didSet { refreshMute() } }

    /// The user's own voiceprint, conditioned as the negative into every slot (contract 2.3). Set through
    /// setOwnVoice(_:) in AudioController+People.swift, which also rebuilds the running slots.
    var ownVoiceprint: Voiceprint?
    /// Result of the launch WeSpeaker self-test (30-ios 9.4 step 4); used for every conditioning.
    var useWeSpeaker = false
    /// Called by stop() with the stop time, so PeopleStore can set "tonight" expiry to stop + 12 h (30-ios 9.2).
    var onStop: ((Date) -> Void)?
    /// True while quick add plays its 2 s check clip: the engine stays muted (30-ios 8.5).
    var holdMuted = false { didSet { refreshMute() } }

    // MARK: I-3 hooks (set by Selector.attach)
    /// Every filled slot's conditioning, for configure() at start and at every engine swap.
    var slotProvider: (@MainActor () -> [SlotConditioning])?
    /// Called by the 10 Hz stats poll (direction learning, talking dots, Now Playing refresh).
    var onPoll: (@MainActor (EngineStats) -> Void)?
    /// Called once audio runs after start or a rebuild (Now Playing, head tracking; 30-ios 6 step 5).
    var onAudioStarted: (@MainActor () -> Void)?
    /// Called when listening stops (Stop, after onStop) or the engine is gone without a swap (a
    /// media services reset, a failed rebuild): Now Playing off, head tracking off. Start calls
    /// onAudioStarted again.
    var onAudioStopped: (@MainActor () -> Void)?
    /// Test seam: every slot-gain push, load, clear and cue sent to the engine, in order. nil in the app.
    var onEngineEvent: (@MainActor (EngineEvent) -> Void)?

    // MARK: Latency prompts (30-ios 7 "When")
    /// The measure prompt at connect or a re-check banner at 10 / 30 min; nil when nothing is offered.
    @Published var latencyPrompt: LatencyPrompts.Prompt? = nil
    /// True after a failed probe: the screens offer method B (ear detection off).
    @Published var offerMethodB = false
    var latencyPrompts = LatencyPrompts()

    /// Set from the SwiftUI scene phase. Start only works in the foreground.
    var isForeground = true

    // MARK: Internal state (internal, not private, only so the +System and +Measure files can use it)
    private(set) var model: EarshotModel?     // set once by attach(model:)
    var engine: EarshotEngine?
    var host: OpaquePointer?                  // eh_host*
    var engineRate: Double = 0        // I/O rate the current RemoteIO and engine were built for
    var engineSignature = ""          // RouteInfo.signature the current engine was built for
    var params = EngineParams()               // the real params; see AudioController+Table (pause)
    var condChain: Task<Void, Never>? = nil   // every conditioning change runs on this chain (30-ios 5)
    /// Slots with a load or clear still waiting on condChain, counted per slot: they are held at gain
    /// 0 whatever else sends params meanwhile, until their step ran (AudioController+Table, contract 4.3).
    var fadingSlots: [Int: Int] = [:]
    var meterTap: InputTap?
    var pollTask: Task<Void, Never>?
    var routeDebounce: Task<Void, Never>?
    var bufferCheck: Task<Void, Never>?
    private var isSwapping = false
    private var swapAgain = false
    var pendingRecreate = false       // a rate change or reset happened in the background
    var ioStoppedBySystem = false     // an interruption began: iOS stopped RemoteIO, eh_start needed
    var userPaused = false            // the user (or the hot-phone rule) paused; only resume() or stop() clears it
    var routeCheckPending = false     // a route changed and the 300 ms check has not read the new route yet
    /// Goes up by one whenever the running I/O goes away or is stopped by iOS (teardown, media
    /// reset, interruption). A swap or start that was waiting at an `await` compares it with the
    /// value it saw before, and gives up if it changed, so it never touches a freed host.
    var ioGeneration = 0
    var observers: [NSObjectProtocol] = []
    var askedForNotifications = false
    let latencyStore = LatencyStore()
    var probeMethod = "in_ear"                // method A by default (30-ios 7)

    // MARK: I-4 safety and telemetry (AudioController+Safety.swift, +System for thermal)
    /// Thermal / Low Power downshift (Safety/PowerPolicy.swift); `powerLevel` is what screens show.
    var power = PowerPolicy()
    @Published var powerLevel: PowerLevel = .normal
    /// Selector.setPowerSaving: only the selected slot is computed while saving power.
    var onPowerSavingChanged: (@MainActor (Bool) -> Void)?
    /// Alarm passthrough (Safety/AlarmListener.swift): the listener, the 5 s timer, the Settings switch.
    let alarmListener = AlarmListener()
    var alarm = AlarmPassthrough()
    var alarmEnabled = true
    /// Per-minute session log (Telemetry/SessionLog.swift); tests point it at a temp folder.
    var sessionLog = SessionLog.standard()
    var minute = MinuteTelemetry()
    var minuteStart = Date()
    var sessionCompute = ComputeHistogram()   // the whole session's per-hop times, for Diagnostics
    @Published var computeSummary = "-"
    /// The session log's `mode` and `n_instances`, from the Selector.
    var telemetryTable: (@MainActor () -> (mode: String, instances: Int))?
    /// Diagnostics: this run's probe results, newest last.
    @Published var probeHistory: [String] = []

    // MARK: I-1 Phase 0 (Telemetry/Phase0Runner.swift reads these)
    /// Route change and swap counters, the newest probe's raw numbers, and a copy of each second's
    /// per-hop timings (drainTimings keeps exactly one reader, safetySecond, which forwards them).
    var phase0 = Phase0Taps()
    /// Debug builds only: counts malloc / free on the render and worker threads (contract 11).
    let mallocCounter = MallocCounter()

    /// RAM copy of the last 8 s of mic history, made at every engine swap for "Add who just
    /// spoke" (30-ios 8.5). Never written to disk. Contract 5.3 keeps it "only across an engine
    /// swap", so it belongs to the engine that swap installed (`quickAddRAMEngine`). The rules are
    /// in AudioController+People.swift (quickAddHistory); only those functions change these two.
    var quickAddRAM: [Float] = []
    /// The engine the RAM copy was made for (weak: the copy never keeps an engine alive).
    weak var quickAddRAMEngine: EarshotEngine?

    init() {
        observeSystemNotifications()
    }

    /// Called once the launch sequence has loaded the model and its self-test passed.
    func attach(model: EarshotModel) {
        self.model = model
    }

    var isReady: Bool { model != nil }

    /// Start needs the model, the foreground, and no running session.
    var canStart: Bool {
        guard isReady, isForeground else { return false }
        switch state {
        case .idle, .failed: return true
        default: return false
        }
    }

    var engineLatencyMS: Double { Double(engine?.latencyMS ?? 0) }

    // MARK: - start / stop (30-ios 6)

    func start() async throws {
        guard canStart else { throw isReady ? AppError.notForeground : AppError.notReady }
        requestNotificationPermissionOnce()
        let r: RouteInfo
        do {
            r = try SessionConfig.activateSession() // step 1
        } catch {
            note("start: session failed: \(error)") // raw text to Diagnostics only
            state = .failed(AppError.userText(error, fallback: Copy.startFailed))
            throw error
        }
        route = r
        note("session: \(Int(r.sampleRate)) Hz, \(r.ioFrames) frames, out \(r.outputDescription)")
        do {
            try await startIO(route: r, frames: r.ioFrames)
        } catch AppError.stopped {
            return // Stop or a media reset came while starting; they already cleaned up
        } catch {
            teardownIO()
            SessionConfig.deactivate()
            note("start failed: \(error)")
            state = .failed(AppError.userText(error, fallback: Copy.startFailed))
            throw error
        }
        if AVAudioSession.sharedInstance().outputVolume > 0.75 { notice = Copy.volumeHigh }
    }

    func stop() {
        teardownIO()
        SessionConfig.deactivate()
        userPaused = false
        state = .idle
        notice = nil
        latencyPrompts = LatencyPrompts() // the next Start counts as a connect (measure prompt again)
        latencyPrompt = nil
        onStop?(Date()) // PeopleStore: every "tonight" person expires 12 h after this
        onAudioStopped?()
    }

    /// True while an engine exists (I/O running, paused or interrupted), used for the expiry refresh.
    /// A paused or interrupted session counts on purpose: only Stop (or the app being killed) ends a
    /// session, and contract 5.1 says tonight people expire 12 h after the session ends.
    var isListening: Bool { engine != nil }

    /// Steps 2 to 5 of start: engine, conditioning, RemoteIO, worker, unmute.
    private func startIO(route r: RouteInfo, frames: Int) async throws {
        guard r.rateSupported else { throw AppError.unsupportedRate(r.sampleRate) }
        let generation = ioGeneration
        let e = try makeEngine(route: r, frames: frames)          // step 2, starts muted
        try await configure(e)                                    // step 3
        // Other code may have run during the await. If Stop or a media reset happened, do not
        // start audio again behind the user's back.
        guard ioGeneration == generation, host == nil else { throw AppError.stopped }
        var h: OpaquePointer? = nil                               // step 4
        let st = eh_create(UnsafeMutableRawPointer(e.cEngine), r.sampleRate, Int32(frames), &h)
        guard st == noErr, let newHost = h else { throw AppError.audioUnit("eh_create", st) }
        do {
            try e.startWorker(workgroup: eh_workgroup(newHost))
        } catch {
            eh_destroy(newHost)
            throw error
        }
        let startStatus = eh_start(newHost)
        if startStatus != noErr {
            e.stopWorker()
            eh_destroy(newHost)
            throw AppError.audioUnit("eh_start", startStatus)
        }
        host = newHost
        engine = e
        engineRate = r.sampleRate
        engineSignature = r.signature
        engineFrames = frames
        hasWorkgroup = eh_workgroup(newHost) != nil
        meterTap = e.makeInputTap()
        // I-4: read thermal / Low Power before the unmute below, so a phone that is already too
        // hot starts paused and no fade-in leaks out before startSafety pauses (30-ios 6).
        handlePowerChange()
        if powerLevel == .critical { userPaused = true }
        state = settledState(for: r)
        startPolling()
        scheduleBufferCheck()
        refreshMute()                                             // step 5, unmute last
        updateLatencyDisplay()
        routeForPrompts(r)                                        // a connect offers the delay measurement
        onAudioStarted?()                                         // I-3: Now Playing, head tracking
        startSafety()                                             // I-4: alarm listener, power, session log
    }

    /// Mute, wait for the 5 ms ramp, stop RemoteIO and the worker, free everything.
    private func teardownIO() {
        ioGeneration += 1     // any swap or start waiting at an await now gives up
        stopSafety()          // I-4: alarm listener off (its tap holds the engine), passthrough reset
        stopPolling()
        routeDebounce?.cancel()
        routeCheckPending = false // the cancelled check would have cleared it
        bufferCheck?.cancel()
        if let e = engine { e.setMuted(true) }
        if engine != nil { waitForMuteRamp() }
        if let h = host { eh_stop(h) }
        engine?.stopWorker()
        if let h = host { eh_destroy(h) }
        host = nil
        engine = nil // EarshotEngine.deinit frees the C engine
        meterTap = nil
        isUnmuted = false
        hasWorkgroup = false
        ioStoppedBySystem = false // that RemoteIO is gone; a Stop during a call must not block later swaps
        pendingRecreate = false   // and opening the app after Stop must not rebuild audio
        clearQuickAddRAM()        // the old conversation must never reach the next session's quick add
    }

    /// Blocks the main thread 10 ms so the 5 ms mute ramp finishes. Not `await Task.sleep` on purpose:
    /// an await lets Stop or a reset run mid-swap, and a cancelled task cuts it short. (Swift warns
    /// about Thread.sleep written directly in an async function, hence this small sync helper.)
    private func waitForMuteRamp() {
        Thread.sleep(forTimeInterval: 0.010)
    }

    /// Step 2: a new muted engine for this route (instances 3, slack 43 on Bluetooth, 22 wired).
    private func makeEngine(route r: RouteInfo, frames: Int) throws -> EarshotEngine {
        guard let model else { throw AppError.notReady }
        return try EarshotEngine(model: model, ioRate: Int(r.sampleRate), ioFrames: frames,
                                 instances: 3, slack16: r.slack16)
    }

    // Step 3 (configure), everyoneConditioning and applyConditioning live in
    // AudioController+Table.swift with the other conditioning code.

    // MARK: - Mute rules

    /// Unmuted only while running on an allowed route with an own voice (or the test switch),
    /// and never while a route change is still being checked (`state` and `route` are old then).
    /// Every unmute fades in over 400 ms inside the engine.
    func refreshMute() {
        guard let e = engine else { return }
        let ownVoiceOK = hasOwnVoice || allowListenWithoutOwnVoice
        let shouldPlay = Self.shouldPlay(state: state, routeAllowed: route.allowed, ownVoiceOK: ownVoiceOK,
                                         isMeasuring: isMeasuring || holdMuted, routeCheckPending: routeCheckPending,
                                         alarm: alarm.isActive)
        if shouldPlay != isUnmuted {
            e.setMuted(!shouldPlay)
            isUnmuted = shouldPlay
        }
        if state == .badRoute { notice = Copy.connectAirPods }
        else if state == .running && !ownVoiceOK { notice = Copy.setupMuted }
        else if notice == Copy.connectAirPods || notice == Copy.setupMuted { notice = nil }
    }

    /// The mute rule as a pure function (unit-tested for every combination): output plays only
    /// while running, on an allowed route, with an own voice (or the test switch), not during a
    /// latency probe and not while a route change is still being checked.
    /// I-4: during an alarm passthrough a user pause also plays, for those 5 s (30-ios 12.2); every
    /// other condition (allowed route, own voice, no probe, no pending route check) still applies.
    nonisolated static func shouldPlay(state: AudioState, routeAllowed: Bool, ownVoiceOK: Bool,
                                       isMeasuring: Bool, routeCheckPending: Bool, alarm: Bool = false) -> Bool {
        (state == .running || (alarm && state == .paused))
            && routeAllowed && ownVoiceOK && !isMeasuring && !routeCheckPending
    }

    /// Mutes at once (RT-safe, 5 ms ramp). Used first on every route change and interruption.
    func muteNow() {
        engine?.setMuted(true)
        isUnmuted = false
    }

    /// The state to settle in once audio runs on route `r`: a user pause wins, then the route decides.
    func settledState(for r: RouteInfo) -> AudioState {
        if userPaused { return .paused }
        return r.allowed ? .running : .badRoute
    }

    // MARK: - Pause / resume (pause never stops I/O, 30-ios 11)

    func pause() {
        guard state == .running || state == .badRoute, let e = engine else { return }
        var quiet = params
        quiet.slotGain = [0, 0, 0]
        quiet.roomDB = -60
        e.setParams(safetyApplied(quiet)) // gains and room fade out over the 150 ms crossfade
        onEngineEvent?(.gains(quiet.slotGain))
        userPaused = true  // remembered across engine swaps, route changes and calls
        state = .paused
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 150_000_000)
            // An alarm passthrough keeps playing through a pause until its 5 s end (12.2).
            if self.state == .paused && !self.alarm.isActive { self.muteNow() }
        }
    }

    func resume() {
        userPaused = false
        if ioStoppedBySystem || pendingRecreate { // after a call: restart I/O first
            resumeAfterInterruption(fromUser: true)
            return
        }
        guard state == .paused else { return }    // a user pause: I/O kept running
        sendParams(params)                        // restore gains and room (a fading slot stays at 0)
        state = settledState(for: route)
        refreshMute()                             // 400 ms fade-in
    }

    // MARK: - The one engine swap path (30-ios 6, D41)

    /// Replaces the running engine while RemoteIO keeps running (works with the screen locked).
    func swapEngine(reason: SwapReason, frames overrideFrames: Int? = nil) async {
        if isSwapping { swapAgain = true; return } // one swap at a time; run once more afterwards
        // During a call iOS has stopped RemoteIO; the swap after the call (interruption) does the work.
        guard let old = engine, let h = host, !ioStoppedBySystem else { return }
        isSwapping = true
        defer { isSwapping = false }
        let generation = ioGeneration
        let t0 = Date()
        let r = RouteInfo(session: AVAudioSession.sharedInstance())
        route = r
        if r.sampleRate != engineRate {
            isSwapping = false
            await recreateIO(reason: .rate)
            return
        }
        let frames = overrideFrames ?? r.ioFrames
        do {
            let new = try makeEngine(route: r, frames: frames)    // muted, same io_rate
            try await configure(new)
            // The await above lets other code run. If Stop, a media reset or a call came
            // meanwhile, `h` may be freed: drop the new engine (never started) and give up.
            guard swapStillValid(generation: generation, host: h, old: old) else {
                swapAgain = false
                note("swap (\(reason.rawValue)) dropped: audio stopped meanwhile")
                return
            }
            try new.startWorker(workgroup: eh_workgroup(h))
            // RAM only, for quick add: the old engine's last 8 s (with the previous swap's copy in
            // front if that swap was under 8 s ago). Kept below only if the new engine takes over.
            let ram = quickAddHistory()
            old.setMuted(true)
            isUnmuted = false
            waitForMuteRamp() // blocking, not an await: nothing may run between the check and eh_set_engine
            let st = eh_set_engine(h, UnsafeMutableRawPointer(new.cEngine))
            if st != noErr {                                       // callback never took it: old stays
                new.stopWorker()
                note("swap (\(reason.rawValue)) failed: no acknowledge (\(st))")
                if isForeground { await recreateIO(reason: .rate) } else { state = .paused }
                return
            }
            old.stopWorker()
            engine = new                                           // the old engine is freed here
            phase0.swaps += 1                                      // I-1: Phase 0 "locked" row
            handOverQuickAddRAM(ram, to: new)                      // valid only with this new engine
            engineSignature = r.signature
            engineFrames = frames
            meterTap = new.makeInputTap()                          // taps belong to one engine
            restartAlarmListener()                                 // I-4: the alarm tap too
            state = settledState(for: route)                       // a user pause survives the swap
            refreshMute()                                          // 400 ms fade-in (unless paused)
            updateLatencyDisplay()
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            note("swap (\(reason.rawValue)) in \(ms) ms, \(frames) frames")
        } catch {
            if ioGeneration != generation { return } // audio was stopped meanwhile; nothing to fix
            note("swap (\(reason.rawValue)) failed: \(error)")
            // The old engine keeps rendering. After a call it is still muted, so leave a state
            // that the Resume button can leave; otherwise follow the route (stay muted either way).
            if state == .interrupted { state = .paused }
            else if !r.allowed && state != .paused { state = .badRoute }
        }
        if swapAgain {
            swapAgain = false
            isSwapping = false
            await swapEngine(reason: .route)
        }
    }

    /// True while a swap that started at `generation` may still use host `h` and engine `old`:
    /// no Stop, media reset or call came while it waited, and nobody replaced the engine.
    func swapStillValid(generation: Int, host h: OpaquePointer?, old: EarshotEngine) -> Bool {
        ioGeneration == generation && host == h && engine === old
    }

    /// RemoteIO and engine rebuilt: only for a hardware rate change or a media services reset.
    func recreateIO(reason: RecreateReason) async {
        muteNow()
        if !isForeground { // a recording session cannot start from the background
            state = .paused
            pendingRecreate = true
            postResumeNotification(Copy.tapToResume)
            note("recreate (\(reason.rawValue)) postponed until the app is opened")
            return
        }
        pendingRecreate = false
        teardownIO() // mutes, waits 10 ms for the ramp (no await), then frees RemoteIO and the engine
        do {
            try SessionConfig.selectBuiltInMic(AVAudioSession.sharedInstance())
            let r = RouteInfo(session: AVAudioSession.sharedInstance())
            route = r
            try await startIO(route: r, frames: r.ioFrames) // keeps a user pause (settledState)
            note("recreate (\(reason.rawValue)) done")
        } catch AppError.stopped {
            note("recreate (\(reason.rawValue)) dropped: audio was stopped meanwhile")
        } catch {
            state = .failed(AppError.userText(error, fallback: Copy.startFailed))
            note("recreate (\(reason.rawValue)) failed: \(error)")
            onAudioStopped?() // no engine any more: Now Playing and head tracking off
        }
    }

    // Scene phase (sceneDidChange) lives in AudioController+System.swift and the
    // Diagnostics log (note) in AudioController+Measure.swift.
}
