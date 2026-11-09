// Phase0Runner+Engine.swift - the engine-side Phase 0 runs (30-ios 15.3): compute histogram with 1
// and 3 instances (and 1,104-frame callbacks), battery and thermal, and WeSpeaker with the screen on
// and locked. Main actor only; es_enroll for the test voiceprints runs off the main actor.
//
// Compute and power fill the engine slots with voiceprints built from the test-vector clips
// `enr_t` / `enr_o` of testvec_short.est (30-ios 15.3 "Compute"), through the same applyTable path
// as the table (fade, wait 150 ms, load). The Selector's own table is not told about it, so these
// runs end by stopping Earshot: the next Start configures the engine from the Selector again.
//
// Compute rows are marked (E-2): with the stub engine the worker does nothing and the histogram
// stays empty, so the row only proves the plumbing. The debug malloc counter (MallocCounter.swift)
// is read at the start and the end of the run; its status says whether the counts mean anything.

import Foundation
import UIKit

extension Phase0Runner {
    // MARK: - Compute: 1 instance 60 min, 3 instances 60 min, 3 instances 30 min at 1,104 frames

    /// `item` is "instances/minutes/frames" from Phase0Plan.computeItems.
    func runCompute(item: String) {
        let parts = item.split(separator: "/").compactMap { Int($0) }
        guard parts.count == 3 else { return }
        let (instances, minutes, frames) = (parts[0], parts[1], parts[2])
        launchRun(.compute, title: Copy.phase0StepTitle(.compute), prompt: Copy.phase0LockAndLeave(minutes),
                  seconds: Double(minutes * 60)) { r in
            let audio = try await r.ensureListening()
            if frames > 1000 && audio.stats.lastFrames < 1000 { throw Phase0Error.needsLargeBuffer(audio.stats.lastFrames) }
            let savedGains = audio.params.slotGain
            defer { r.stopAndRestore(gains: savedGains) }
            try await r.fillSlots(instances)
            var hist = ComputeHistogram()
            audio.phase0.onTimings = { hist.add($0) }
            let malloc0 = audio.mallocCounter.snapshot()
            var underruns = CounterTotal(start: audio.stats.underruns)
            var hitSaving = false
            r.activeSeconds = 0
            r.inactiveSeconds = 0
            try await r.ticks(minutes * 60) { _ in
                underruns.add(audio.stats.underruns)
                if audio.powerLevel != .normal { hitSaving = true }
            }
            audio.phase0.onTimings = nil
            let counts = audio.mallocCounter.snapshot().minus(malloc0)
            let row = Phase0ComputeRow(
                instances: instances, minutes: Double(minutes), screen: r.screenLabel,
                p50US: hist.percentile(0.5), p99US: hist.percentile(0.99), p999US: hist.percentile(0.999),
                maxUS: hist.maxUS, underruns: underruns.total, workgroup: audio.hasWorkgroup,
                frames: audio.stats.lastFrames, histUS: Phase0Math.bins(hist),
                malloc: Phase0MallocResult(status: audio.mallocCounter.status.rawValue, counts: counts))
            r.record(.compute, item: row.screen == "locked" && !hitSaving ? item : nil) { rep in
                rep.compute.append(row)
                if hitSaving { rep.notes += Copy.phase0NoteSaving(item) }
            }
            r.message = Copy.phase0ComputeResult(p99: row.p99US, p999: row.p999US, underruns: row.underruns,
                                                 allocs: counts.total, mallocArmed: audio.mallocCounter.status == .armed)
        }
    }

    // MARK: - Power: 2 h with 1 instance, 60 min with 3 warm instances, screen off, unplugged

    /// `item` is "instances/minutes" from Phase0Plan.powerItems.
    func runPower(item: String) {
        let parts = item.split(separator: "/").compactMap { Int($0) }
        guard parts.count == 2 else { return }
        let (instances, minutes) = (parts[0], parts[1])
        launchRun(.power, title: Copy.phase0StepTitle(.power), prompt: Copy.phase0LockAndLeave(minutes),
                  seconds: Double(minutes * 60)) { r in
            UIDevice.current.isBatteryMonitoringEnabled = true
            guard UIDevice.current.batteryState == .unplugged else { throw Phase0Error.plugCharger }
            let audio = try await r.ensureListening()
            let savedGains = audio.params.slotGain
            defer { r.stopAndRestore(gains: savedGains) }
            try await r.fillSlots(instances)
            let start = Self.batteryPercent()
            var worst = ProcessInfo.processInfo.thermalState
            r.activeSeconds = 0
            r.inactiveSeconds = 0
            try await r.ticks(minutes * 60) { _ in
                if UIDevice.current.batteryState != .unplugged { throw Phase0Error.plugCharger }
                let now = ProcessInfo.processInfo.thermalState
                if now.rawValue > worst.rawValue { worst = now } // nominal < fair < serious < critical
            }
            let end = Self.batteryPercent()
            let pct = start.flatMap { s in end.flatMap { Phase0Math.pctPerHour(startPct: s, endPct: $0, minutes: Double(minutes)) } }
            let row = Phase0PowerRow(minutes: Double(minutes), instances: instances, pctPerHour: Phase0Math.round1(pct ?? 0),
                                     maxThermal: MinuteTelemetry.thermalText(worst), batteryStart: start, batteryEnd: end)
            r.record(.power, item: r.screenLabel == "locked" && pct != nil ? item : nil) { $0.power.append(row) }
            r.message = Copy.phase0PowerResult(pctPerHour: row.pctPerHour, thermal: row.maxThermal)
        }
    }

    static func batteryPercent() -> Int? {
        let level = UIDevice.current.batteryLevel
        return level < 0 ? nil : Int((level * 100).rounded())
    }

    // MARK: - Shared by compute and power

    /// Loads `instances` slots (1 or 3) with the test-vector person, gains [1, 0, 0] (Focus on slot 0,
    /// the others warm), and waits until the loads ran. Without the test vector (a build without
    /// `make models`), every slot gets the Everyone conditioning instead.
    func fillSlots(_ instances: Int) async throws {
        guard let audio, let model = launch?.model else { throw Phase0Error.notListening }
        let useWeSpeaker = audio.useWeSpeaker
        let cond: Conditioning = try await Task.detached(priority: .userInitiated) {
            try Self.testConditioning(model: model, useWeSpeaker: useWeSpeaker)
        }.value ?? audio.everyoneConditioning()
        var loads: [(SlotLoad, Conditioning?)] = []
        for slot in 0..<TableState.slotCount {
            if slot < instances { loads.append((SlotLoad(slot: slot, content: .everyone, reset: true), cond)) }
            else { loads.append((SlotLoad(slot: slot, content: .empty, reset: true), nil)) }
        }
        audio.applyTable(loads, gains: [1, 0, 0])
        await audio.condChainIdle()
    }

    /// PERSON(target from enr_t, own from enr_o) like the test vector's `person` case (contract 3.3),
    /// or nil when testvec_short.est is not in the bundle. Any thread (es_enroll is not real time).
    nonisolated static func testConditioning(model: EarshotModel, useWeSpeaker: Bool) throws -> Conditioning? {
        guard let url = Bundle.main.url(forResource: "testvec_short", withExtension: "est"),
              let data = try? Data(contentsOf: url) else { return nil }
        let file = try ContainerFile(data: data, expectMagic: "EST1")
        let eT = useWeSpeaker ? try? file.floats("e_t") : nil
        let eO = useWeSpeaker ? try? file.floats("e_o") : nil
        let target = try Voiceprint(model: model, pcm16k: try file.floats("enr_t"), mask: try file.uint8s("mask_t"),
                                    role: .target, weSpeaker: eT)
        let own = try Voiceprint(model: model, pcm16k: try file.floats("enr_o"), mask: try file.uint8s("mask_o"),
                                 role: .own, weSpeaker: eO)
        return try Conditioning(model: model, mode: .person, target: target, own: own,
                                useWeSpeaker: useWeSpeaker && eT != nil && eO != nil)
    }

    /// Ends a compute or power run: Stop (the Selector configures the next Start) and the table's
    /// gains back as they were before the run.
    func stopAndRestore(gains: [Float]) {
        guard let audio else { return }
        audio.stop()
        audio.params.slotGain = gains
        audio.note("phase0: run ended, Earshot stopped (press Start to listen again)")
    }

    // MARK: - WeSpeaker: embed 1.5 s and 10 s clips, screen on or locked

    /// `locked` gives the owner 20 s to lock the screen first; the clip is the newest 10 s of mic
    /// history when listening, else a fixed noise clip (only the timing matters here).
    func runWeSpeaker(locked: Bool) {
        let wait = locked ? 20 : 0
        launchRun(.wespeaker, title: Copy.phase0StepTitle(.wespeaker), prompt: locked ? Copy.phase0LockIn20 : Copy.phase0KeepOpen,
                  seconds: Double(wait + 10)) { r in
            guard let url = SpeakerEncoder.bundledModelURL else { throw Phase0Error.noWeSpeaker }
            try await r.ticks(wait)
            r.activeSeconds = 0
            r.inactiveSeconds = 0
            r.prompt(Copy.phase0Measuring)
            let active = r.audio?.isForeground ?? true
            var clip = r.audio?.engine?.recentInput(samples: 160_000) ?? []
            if clip.count < 160_000 { clip = Self.noiseClip(count: 160_000) }
            let short = Array(clip.prefix(24_000))
            let encoder = try await Task.detached(priority: .userInitiated) { try SpeakerEncoder(modelURL: url) }.value
            var ok = true
            func timed(_ pcm: [Float]) async -> Double? {
                let t0 = Date()
                do { _ = try await encoder.embed(pcm, appActive: active) } catch { ok = false; return nil }
                return Date().timeIntervalSince(t0) * 1000
            }
            let first = await timed(short)
            let ms15 = await timed(short)
            let ms10 = await timed(clip)
            let selfTestOK = await encoder.selfTest()
            let cosine = await encoder.selfTestCosine
            let trusted = await encoder.gpuTrusted
            if active { r.activeSeconds += 1 } else { r.inactiveSeconds += 1 } // the calls above, at least
            let row = Phase0WeSpeakerRow(units: active && trusted ? "cpuAndGPU" : "cpuOnly", screen: r.screenLabel,
                                         selftestCos: cosine, ms1p5s: ms15.map(Phase0Math.round1),
                                         ms10s: ms10.map(Phase0Math.round1), firstCallMS: first.map(Phase0Math.round1),
                                         ok: ok && selfTestOK)
            let item = row.screen == "mixed" ? nil : row.screen
            r.record(.wespeaker, item: item) { $0.wespeaker.append(row) }
            r.message = Copy.phase0WeSpeakerResult(ok: row.ok, ms10: row.ms10s, units: row.units)
        }
    }

    /// A deterministic noise clip at about -30 dBFS (a small linear congruential generator).
    nonisolated static func noiseClip(count: Int) -> [Float] {
        var state: UInt32 = 12_345
        return (0..<count).map { _ in
            state = state &* 1_664_525 &+ 1_013_904_223
            return (Float(state >> 8) / Float(1 << 24) - 0.5) * 0.11
        }
    }
}
