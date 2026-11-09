// EarshotCore.swift - thin Swift wrappers around the C engine (contract 5).
//
// Every public name and signature here is copied from contract section 5. Each
// class owns exactly one C object (`es_model`, `es_voiceprint`, `es_cond`,
// `es_engine`) and frees it in `deinit`. Nothing in this file ever runs on the
// audio thread: the render callback is C (Audio/AudioHost.c).
//
// Threading (contract 4.1):
// - EarshotModel, Voiceprint, Conditioning: any thread that is not real time.
// - EarshotEngine.setConditioning / clearConditioning / setParams: main actor only.
// - setMuted, cue, stats, latencyMS: safe from anywhere.

import Foundation

public struct EarshotError: Error { public let status: Int32; public let message: String }
public enum ListenMode: Int32 { case person = 0, everyone = 1 }
public enum VoiceRole: Int32 { case target = 0, own = 1 }

extension EarshotError {
    /// Builds an error from a C status, with the engine's own text for it.
    init(_ status: es_status, _ detail: String = "") {
        let text = String(cString: es_status_str(status))
        self.init(status: Int32(truncatingIfNeeded: status.rawValue),
                  message: detail.isEmpty ? text : "\(text): \(detail)")
    }
}

/// Throws when a C call did not return ES_OK.
func checkStatus(_ status: es_status, _ detail: String = "") throws {
    if status != ES_OK { throw EarshotError(status, detail) }
}

/// Calls `body` with a pointer to the array's storage, or with nil when the array is nil.
/// (C functions take NULL for "not given", for example the mask of es_enroll.)
func withOptionalPointer<T, R>(_ array: [T]?, _ body: (UnsafePointer<T>?) -> R) -> R {
    guard let array else { return body(nil) }
    return array.withUnsafeBufferPointer { body($0.baseAddress) }
}

// MARK: - Model

/// The loaded weights (`es_model`). Immutable after init, so it may be shared between threads.
public final class EarshotModel: @unchecked Sendable {
    let cModel: OpaquePointer // es_model*

    public init(weights: Data) throws { // es_model_load
        var out: OpaquePointer? = nil
        let errLen = 256
        var err = [CChar](repeating: 0, count: errLen)
        let status = weights.withUnsafeBytes { raw in
            es_model_load(raw.baseAddress, raw.count, &out, &err, errLen)
        }
        guard status == ES_OK, let model = out else {
            throw EarshotError(status, String(cString: err))
        }
        cModel = model
    }

    deinit { es_model_free(cModel) }

    public var configHash: String { String(cString: es_model_config_hash(cModel)) }

    /// es_model_data_hash = model_hash (contract 3.1), 64 lowercase hex characters.
    public var modelHash: String { String(cString: es_model_data_hash(cModel)) }

    /// es_selftest on `testvec_short.est`; returns the largest output error.
    public func selfTest(testVector: Data) throws -> Float {
        var maxErr: Float = 0
        let status = testVector.withUnsafeBytes { raw in
            es_selftest(cModel, raw.baseAddress, raw.count, &maxErr)
        }
        try checkStatus(status, "max error \(maxErr)")
        return maxErr
    }

    /// es_vad: one 0/1 entry per token frame (es_enroll_num_frames(n) entries).
    public func vad(_ pcm16k: [Float]) -> [UInt8] {
        let frames = Int(es_enroll_num_frames(Int32(pcm16k.count)))
        if frames == 0 { return [] }
        var mask = [UInt8](repeating: 0, count: frames)
        pcm16k.withUnsafeBufferPointer { pcm in
            mask.withUnsafeMutableBufferPointer { m in
                es_vad(pcm.baseAddress, Int32(pcm.count), m.baseAddress)
            }
        }
        return mask
    }

    /// es_embed_g: the 128-float unit vector used by the "locked" check.
    public func embedG(_ pcm16k: [Float]) throws -> [Float] {
        var g = [Float](repeating: 0, count: Int(ES_G_DIM))
        let status = pcm16k.withUnsafeBufferPointer { pcm in
            g.withUnsafeMutableBufferPointer { out in
                es_embed_g(cModel, pcm.baseAddress, Int32(pcm.count), out.baseAddress)
            }
        }
        try checkStatus(status)
        return g
    }
}

extension EarshotModel {
    /// True when the stub engine is linked (its model hash is 64 zeros; a real
    /// SHA-256 never is). Shown in Diagnostics and recorded by Phase 0 as "engine".
    var isStub: Bool { modelHash == String(repeating: "0", count: 64) }
}

// MARK: - Voiceprint

/// One enrolled voice (`es_voiceprint`). Immutable after init.
public final class Voiceprint: @unchecked Sendable {
    let cVoiceprint: OpaquePointer // es_voiceprint*
    let model: EarshotModel        // keeps the weights alive as long as this voiceprint

    public init(model: EarshotModel, pcm16k: [Float], mask: [UInt8]?, role: VoiceRole, weSpeaker: [Float]?) throws {
        if let mask, mask.count != Int(es_enroll_num_frames(Int32(pcm16k.count))) {
            throw EarshotError(ES_ERR_ARG, "mask has \(mask.count) entries")
        }
        if let weSpeaker, weSpeaker.count != Int(ES_WESPK_DIM) {
            throw EarshotError(ES_ERR_ARG, "WeSpeaker vector has \(weSpeaker.count) values")
        }
        guard let vp = es_voiceprint_new() else { throw EarshotError(ES_ERR_NOMEM) }
        let cRole = role == .target ? ES_ROLE_TARGET : ES_ROLE_OWN
        let status: es_status = pcm16k.withUnsafeBufferPointer { pcm in
            withOptionalPointer(mask) { maskPtr in
                withOptionalPointer(weSpeaker) { ePtr in
                    es_enroll(model.cModel, pcm.baseAddress, Int32(pcm.count), maskPtr, cRole, ePtr, vp)
                }
            }
        }
        if status != ES_OK {
            es_voiceprint_free(vp)
            throw EarshotError(status)
        }
        self.cVoiceprint = vp
        self.model = model
    }

    deinit { es_voiceprint_free(cVoiceprint) }

    public var tokenCount: Int { Int(es_voiceprint_num_tokens(cVoiceprint)) }

    /// 128 floats, unit length.
    public var g: [Float] {
        var out = [Float](repeating: 0, count: Int(ES_G_DIM))
        out.withUnsafeMutableBufferPointer { es_voiceprint_get_g(cVoiceprint, $0.baseAddress) }
        return out
    }

    public var speechSeconds: Float { es_voiceprint_speech_seconds(cVoiceprint) }
    public var hasWeSpeaker: Bool { es_voiceprint_has_wespk(cVoiceprint) != 0 }
}

// MARK: - Conditioning

/// Who to keep, for one engine slot (`es_cond`). The engine copies it, so it may be dropped after use.
public final class Conditioning: @unchecked Sendable {
    let cCond: OpaquePointer // es_cond*
    let model: EarshotModel

    public init(model: EarshotModel, mode: ListenMode, target: Voiceprint?, own: Voiceprint?, useWeSpeaker: Bool) throws {
        guard let c = es_cond_new() else { throw EarshotError(ES_ERR_NOMEM) }
        let cMode = mode == .person ? ES_MODE_PERSON : ES_MODE_EVERYONE
        let status = es_cond_build(model.cModel, cMode, target?.cVoiceprint, own?.cVoiceprint,
                                   useWeSpeaker ? 1 : 0, c)
        if status != ES_OK {
            es_cond_free(c)
            throw EarshotError(status)
        }
        self.cCond = c
        self.model = model
    }

    deinit { es_cond_free(cCond) }
}

// MARK: - Params, stats, probe result

/// Mirrors es_params; the defaults equal es_params_default.
public struct EngineParams {
    public var voiceDB: Float = 0, roomDB: Float = -60, maxGainDB: Float = 18, outGainDB: Float = 0
    public var slotGain: [Float] = [1, 0, 0], crossfadeMS: Float = 150
    public var duckOwnVoice = true, gate = true, agc = true, bypass = false

    public init() {}

    /// The C struct handed to es_engine_set_params.
    var cValue: es_params {
        func gain(_ i: Int) -> Float { i < slotGain.count ? slotGain[i] : 0 }
        var p = es_params()
        p.voice_db = voiceDB
        p.room_db = roomDB
        p.max_gain_db = maxGainDB
        p.out_gain_db = outGainDB
        p.slot_gain = (gain(0), gain(1), gain(2))
        p.xfade_ms = crossfadeMS
        p.duck_own = duckOwnVoice ? 1 : 0
        p.gate = gate ? 1 : 0
        p.agc = agc ? 1 : 0
        p.bypass = bypass ? 1 : 0
        return p
    }
}

/// Mirrors es_stats (a snapshot; safe to read at 10 Hz on the main actor).
public struct EngineStats {
    public var hops: UInt64 = 0, underruns: UInt64 = 0, nanResets: UInt64 = 0, probeRuns: UInt64 = 0
    public var slack16: Int = 0, lastFrames: Int = 0, howl = false
    public var pTarget: [Float] = [0, 0, 0], pSelf: Float = 0
    public var inRmsDB: Float = -100, outRmsDB: Float = -100, agcGainDB: Float = 0, limiterGrDB: Float = 0
    public var computeUsLast: Float = 0, computeUsMax: Float = 0

    public init() {}

    init(_ s: es_stats) {
        hops = s.hops
        underruns = s.underruns
        nanResets = s.nan_resets
        probeRuns = s.probe_runs
        slack16 = Int(s.slack16)
        lastFrames = Int(s.last_frames)
        howl = s.howl != 0
        pTarget = [s.p_target.0, s.p_target.1, s.p_target.2]
        pSelf = s.p_self
        inRmsDB = s.in_rms_db
        outRmsDB = s.out_rms_db
        agcGainDB = s.agc_gain_db
        limiterGrDB = s.limiter_gr_db
        computeUsLast = s.compute_us_last
        computeUsMax = s.compute_us_max
    }
}

public struct ProbeResult { public let loopMS: Float, spreadMS: Float, psrDB: Float, okSweeps: Int }

// MARK: - Engine

/// The running engine (`es_engine`): rings, resampling, worker, instances, safety.
public final class EarshotEngine: @unchecked Sendable {
    public let model: EarshotModel
    private let engine: OpaquePointer // es_engine*

    public init(model: EarshotModel, ioRate: Int, ioFrames: Int, instances: Int, slack16: Int) throws {
        var cfg = es_engine_config(io_rate: Int32(ioRate), io_frames: Int32(ioFrames), max_frames: 4096,
                                   n_instances: Int32(instances), slack16: Int32(slack16))
        var out: OpaquePointer? = nil
        let status = es_engine_new(model.cModel, &cfg, &out)
        guard status == ES_OK, let e = out else {
            throw EarshotError(status, "rate \(ioRate), frames \(ioFrames)")
        }
        self.model = model
        self.engine = e
    }

    deinit { es_engine_free(engine) } // the real engine stops a running worker first

    /// Passed to AudioHost (eh_create / eh_set_engine) as the engine the callback renders through.
    public var cEngine: OpaquePointer { engine }

    /// `workgroup` is the value of eh_workgroup(), may be nil.
    public func startWorker(workgroup: UnsafeMutableRawPointer?) throws {
        try checkStatus(es_engine_start_worker(engine, workgroup))
    }

    public func stopWorker() { es_engine_stop_worker(engine) }

    /// Copies `c` into the slot. ES_ERR_BUSY (the last change not yet picked up) is
    /// retried every 4 ms for up to 200 ms (contract 4.3).
    @MainActor public func setConditioning(slot: Int, _ c: Conditioning, resetState: Bool) async throws {
        try await retryWhileBusy { es_engine_set_cond(engine, Int32(slot), c.cCond, resetState ? 1 : 0) }
    }

    /// Empties the slot (es_engine_set_cond with NULL), same BUSY retry.
    @MainActor public func clearConditioning(slot: Int) async throws {
        try await retryWhileBusy { es_engine_set_cond(engine, Int32(slot), nil, 0) }
    }

    /// The first try plus 50 retries 4 ms apart = 200 ms.
    @MainActor private func retryWhileBusy(_ call: () -> es_status) async throws {
        for attempt in 0...50 {
            let status = call()
            if status == ES_OK { return }
            if status != ES_ERR_BUSY { throw EarshotError(status) }
            if attempt < 50 { try await Task.sleep(nanoseconds: 4_000_000) }
        }
        throw EarshotError(ES_ERR_BUSY, "still busy after 200 ms")
    }

    @MainActor public func setParams(_ p: EngineParams) {
        var c = p.cValue
        es_engine_set_params(engine, &c)
    }

    /// RT-safe. Mute ramps down in 5 ms; every unmute fades in over 400 ms.
    public func setMuted(_ muted: Bool) { es_engine_set_mute(engine, muted ? 1 : 0) }

    /// 1 to 3 short beeps (heard even while muted).
    public func cue(beeps: Int) { es_engine_cue(engine, Int32(beeps)) }

    public func stats() -> EngineStats {
        var s = es_stats()
        es_engine_get_stats(engine, &s)
        return EngineStats(s)
    }

    /// Earshot's delay over a plain pass-through, in ms (es_engine_latency_ms).
    public var latencyMS: Float { es_engine_latency_ms(engine) }

    /// A tap that returns only mic audio written after this call.
    public func makeInputTap() -> InputTap {
        InputTap(engine: self, cursor: es_engine_input_position(engine))
    }

    /// The newest `samples` (at most 245,760 = 15.36 s) of 16 kHz mic history, oldest first.
    public func recentInput(samples: Int) -> [Float] {
        let n = max(0, min(samples, 245_760))
        if n == 0 { return [] }
        let written = es_engine_input_position(engine)
        var cursor: UInt64 = written > UInt64(n) ? written - UInt64(n) : 0
        var out = [Float](repeating: 0, count: n)
        let got = out.withUnsafeMutableBufferPointer { buf in
            es_engine_read_input(engine, &cursor, buf.baseAddress, Int32(n))
        }
        return Array(out.prefix(Int(max(0, got))))
    }

    /// The newest `samples` (at most 48,000) of one slot's raw model output.
    public func recentOutput(slot: Int, samples: Int) -> [Float] {
        let n = max(0, min(samples, 48_000))
        if n == 0 { return [] }
        var out = [Float](repeating: 0, count: n)
        let got = out.withUnsafeMutableBufferPointer { buf in
            es_engine_read_output(engine, Int32(slot), buf.baseAddress, Int32(n))
        }
        return Array(out.prefix(Int(max(0, got))))
    }

    /// Every per-hop compute time (microseconds) since the last call.
    public func drainTimings() -> [UInt16] {
        var all: [UInt16] = []
        var chunk = [UInt16](repeating: 0, count: 1024)
        while true {
            let got = chunk.withUnsafeMutableBufferPointer { buf in
                es_engine_read_timings(engine, buf.baseAddress, Int32(buf.count))
            }
            if got <= 0 { break }
            all.append(contentsOf: chunk.prefix(Int(got)))
            if got < Int32(chunk.count) { break }
        }
        return all
    }

    public func startLatencyProbe() throws { try checkStatus(es_engine_probe_start(engine)) }

    /// nil while the probe is still running (or was never started).
    public func latencyProbeResult() -> Result<ProbeResult, EarshotError>? {
        var r = es_probe_result()
        let status = es_engine_probe_result(engine, &r)
        if status == ES_ERR_STATE { return nil }
        if status != ES_OK { return .failure(EarshotError(status, "\(r.n_ok) good sweeps")) }
        return .success(ProbeResult(loopMS: r.loop_ms, spreadMS: r.spread_ms, psrDB: r.psr_db, okSweeps: Int(r.n_ok)))
    }
}

// MARK: - Input tap

/// Reads the engine's 16 kHz mic history from its own cursor (contract 5).
/// A tap belongs to one engine; after an engine swap every tap is recreated.
public struct InputTap {
    let engine: EarshotEngine // keeps the engine alive while the tap exists
    var cursor: UInt64        // total history samples this tap has consumed

    /// Samples skipped because the reader fell more than 15.36 s behind.
    public private(set) var droppedSamples: Int = 0

    init(engine: EarshotEngine, cursor: UInt64) {
        self.engine = engine
        self.cursor = cursor
    }

    /// New samples since the last read, in order (at most `maxSamples`).
    public mutating func read(maxSamples: Int) -> [Float] {
        if maxSamples <= 0 { return [] }
        let e = engine.cEngine
        let oldCursor = cursor
        var newCursor = cursor
        var out = [Float](repeating: 0, count: maxSamples)
        let got = out.withUnsafeMutableBufferPointer { buf in
            es_engine_read_input(e, &newCursor, buf.baseAddress, Int32(maxSamples))
        }
        let returned = Int(max(0, got))
        // skipped = new cursor - old cursor - returned (contract 5)
        let skipped = Int(newCursor) - Int(oldCursor) - returned
        if skipped > 0 { droppedSamples += skipped }
        cursor = newCursor
        return Array(out.prefix(returned))
    }
}
