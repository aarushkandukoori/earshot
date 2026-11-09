// DiagnosticsView.swift - the developer screen (30-ios 14, I-0 subset).
//
// Shows what the app and iOS actually did: route and port UIDs, rates, requested /
// granted / callback frames, reported latencies, slack, workgroup, compute times,
// underruns, self-test, model hash, thermal, battery, profile expiry and the input
// meter. Row labels here are developer terms; user-facing sentences come from Copy.
// I-3 adds the Control section: mode, table slots, directions, head motion rate and newest-sample
// age, stem press counters. I-4 adds session compute p50 / p99 / p99.9 from drainTimings, the probe
// history, and the Safety section (power level, alarm listener, locked check, session log).
// I-1 adds the Phase 0 runner link (UI/Phase0View.swift) and the debug malloc counter rows.

import SwiftUI
import UIKit

struct DiagnosticsView: View {
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var selector: Selector
    let report: LaunchReport
    @State private var measureError: String? = nil

    var body: some View {
        List {
            Section("Engine") {
                row("engine", report.engineKind)
                row("model_hash", report.modelHash.isEmpty ? "-" : String(report.modelHash.prefix(16)) + "...")
                row("es_selftest", report.selfTestMessage)
                row("thresholds", report.thresholdsVersion.map { "version \($0)" } ?? "not loaded")
                if let warning = report.thresholdsWarning { row("warning", warning) }
                row("WeSpeaker", report.encoderMessage)
                row("WeSpeaker cosine", report.encoderCosine.map { String(format: "%.5f", $0) } ?? "-")
                row("WeSpeaker units", report.encoderUnits)
                row("state", "\(audio.state)")
                row("output", audio.isUnmuted ? "playing" : "muted")
                row("workgroup", audio.hasWorkgroup ? "yes" : "no")
            }

            Section("Route") {
                row("outputs", audio.route.outputDescription)
                ForEach(audio.route.outputs, id: \.uid) { port in row("uid", port.uid) }
                row("input", audio.route.inputType?.rawValue ?? "none")
                row("data source", audio.route.inputDataSource ?? "-")
                row("allowed", audio.route.allowed ? "yes" : "no")
                row("rate", "\(Int(audio.route.sampleRate)) Hz")
                row("frames requested", "\(SessionConfig.requestedFrames)")
                row("frames granted", "\(audio.route.ioFrames)")
                row("frames callback", "\(audio.stats.lastFrames)")
                row("engine built for", "\(audio.engineFrames)")
                row("input latency", ms(audio.route.inputLatency))
                row("output latency", ms(audio.route.outputLatency))
                row("io buffer", ms(audio.route.ioBufferDuration))
                row("slack16", "\(audio.stats.slack16)")
            }

            Section("Engine stats") {
                row("hops", "\(audio.stats.hops)")
                row("compute last / max", String(format: "%.0f / %.0f us", audio.stats.computeUsLast,
                                                 audio.stats.computeUsMax))
                row("compute session", audio.computeSummary)
                row("underruns", "\(audio.stats.underruns)")
                row("NaN resets", "\(audio.stats.nanResets)")
                row("howl", audio.stats.howl ? "yes" : "no")
                row("in / out RMS", String(format: "%.1f / %.1f dBFS", audio.stats.inRmsDB, audio.stats.outRmsDB))
                row("mic peak / RMS", String(format: "%.1f / %.1f dBFS", audio.meter.peakDB, audio.meter.rmsDB))
            }

            Section("Delay") {
                Text(audio.latency.text)
                row("measured", audio.latency.measuredMS.map { "\($0) ms" } ?? "-")
                row("reported", "\(audio.latency.reportedMS) ms")
                row("Earshot share", String(format: "%.2f ms", audio.engineLatencyMS))
                row("probe runs", "\(audio.stats.probeRuns)")
                ForEach(Array(audio.probeHistory.enumerated().reversed()), id: \.offset) { _, line in
                    Text(line).font(.caption.monospaced())
                }
                if !audio.lastProbeMessage.isEmpty { Text(audio.lastProbeMessage).font(.footnote) }
                Button(Copy.measureDelay) {
                    Task {
                        do { try await audio.measureLatency(); measureError = nil }
                        catch { measureError = AppError.userText(error, fallback: Copy.probeWeak) }
                    }
                }
                .disabled(audio.state != .running || audio.isMeasuring)
                Text(Copy.measureHowTo).font(.footnote).foregroundStyle(.secondary)
                if let measureError { Text(measureError).font(.footnote).foregroundStyle(.red) }
            }

            ControlDiagnostics(head: selector.head, remote: selector.remote)

            Section("Safety") {
                row("power level", audio.powerLevel.rawValue)
                row("alarm listener", audio.alarmEnabled ? (audio.alarmListener.running ? "running" : "off") : "disabled")
                row("alarm windows", "\(audio.alarmListener.windows), last \(audio.alarmListener.lastWindow)")
                row("alarm passthrough", audio.alarm.isActive ? "on" : "off")
                row("locked check", selector.lockStates.enumerated()
                    .map { "\($0.offset): \(String(describing: $0.element))" }.joined(separator: ", "))
                row("lost warning", selector.lostLockWarning ? "yes" : "no")
                row("session log", audio.sessionLog.folder.lastPathComponent + "/" + SessionLog.fileName(for: Date()))
            }

            Section("Device") {
                row("thermal", thermalText(ProcessInfo.processInfo.thermalState))
                row("low power", ProcessInfo.processInfo.isLowPowerModeEnabled ? "on" : "off")
                row("battery", batteryText())
                row("profile", ProvisioningProfile.expiryText(report.profileExpiry)) // read once at launch
            }

            Section(Copy.phase0Title) {
                NavigationLink(Copy.phase0Title) { Phase0View() }
            }

            MallocDiagnostics(counter: audio.mallocCounter)

            Section("Tests") {
                Toggle(Copy.listenTest, isOn: $audio.allowListenWithoutOwnVoice)
                Text(Copy.listenTestNote).font(.footnote).foregroundStyle(.secondary)
                Button(Copy.testBeep) { audio.cue(beeps: 1) }.disabled(audio.state == .idle)
                Button(Copy.forceSwap) { audio.forceSwap() }.disabled(audio.state == .idle)
            }

            Section("Log") {
                ForEach(Array(audio.notes.enumerated().reversed()), id: \.offset) { _, line in
                    Text(line).font(.caption.monospaced())
                }
            }
        }
        .navigationTitle(Copy.diagnostics)
        .onAppear { UIDevice.current.isBatteryMonitoringEnabled = true }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).multilineTextAlignment(.trailing).textSelection(.enabled)
        }
        .font(.callout)
    }

    private func ms(_ seconds: Double) -> String { String(format: "%.2f ms", seconds * 1000) }

    private func thermalText(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private func batteryText() -> String {
        let level = UIDevice.current.batteryLevel
        return level < 0 ? "unknown" : "\(Int((level * 100).rounded()))%"
    }
}

/// The table, head tracking and stem presses (30-ios 14 "motion rate and newest-sample age, stem
/// press counters"). HeadTracker and RemoteCommands are observed directly so their rows update.
struct ControlDiagnostics: View {
    @EnvironmentObject private var selector: Selector
    @ObservedObject var head: HeadTracker
    @ObservedObject var remote: RemoteCommands

    var body: some View {
        Section("Control") {
            row("mode", selector.table.mode.rawValue)
            ForEach(0..<TableState.slotCount, id: \.self) { s in
                row("slot \(s)", slotText(s))
            }
            row("directions", selector.table.directionsStale ? "stale" : "fresh")
            row("head tracking", head.running ? (head.live ? "live" : (head.unavailable ? "paused" : "late")) : "off")
            row("motion rate", String(format: "%.1f Hz", head.updateRateHz))
            row("newest sample", head.newestAge.map { String(format: "%.1f s ago", $0) } ?? "-")
            row("stem single / double / triple", StemPress.allCases.map { "\(remote.counts[$0] ?? 0)" }
                .joined(separator: " / "))
            row("Now Playing", remote.active ? "on" : "off")
        }
    }

    /// "Maya, gain 1, dir 12" or "Everyone, gain 0" or "empty".
    private func slotText(_ s: Int) -> String {
        let gain = String(format: "gain %.0f", selector.table.gains[s])
        switch selector.table.content(s) {
        case .empty: return "empty"
        case .everyone: return "Everyone, \(gain)"
        case .person(let id):
            let dir = selector.table.seats[s]?.dir.map { String(format: ", dir %.0f", $0) } ?? ""
            return "\(selector.name(id)), \(gain)\(dir)"
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).multilineTextAlignment(.trailing)
        }
        .font(.callout)
    }
}

/// The debug malloc counter (contract 11 Lead integration): allocations and frees seen on the
/// render and worker threads while listening. Release builds say "not in this build".
struct MallocDiagnostics: View {
    @ObservedObject var counter: MallocCounter

    var body: some View {
        Section(Copy.mallocCounterTitle) {
            row("status", counter.summary)
            row("render thread", counter.renderTID == 0 ? "-" : "\(counter.renderTID)")
            row("worker thread", counter.workerTID.map { "\($0)" } ?? "-")
            row("worker time-constraint", counter.workerTimeConstraint.map { $0 ? "yes" : "no" } ?? "-")
            row("newest size", counter.lastSize == 0 ? "-" : "\(counter.lastSize) bytes")
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).multilineTextAlignment(.trailing)
        }
        .font(.callout)
    }
}

/// Reads the expiry date of the embedded provisioning profile (free team profiles last 7 days,
/// 30-ios 2.3). The file is a signed container with a plain XML plist inside. The launch
/// sequence reads it once (LaunchReport.profileExpiry); views only format that date.
enum ProvisioningProfile {
    static func expiryDate() -> Date? {
        guard let url = Bundle.main.url(forResource: "embedded", withExtension: "mobileprovision"),
              let data = try? Data(contentsOf: url) else { return nil }
        let start = Data("<?xml".utf8)
        let end = Data("</plist>".utf8)
        guard let a = data.range(of: start), let b = data.range(of: end, in: a.lowerBound..<data.endIndex) else {
            return nil
        }
        let plistData = data.subdata(in: a.lowerBound..<b.upperBound)
        guard let plist = try? PropertyListSerialization.propertyList(from: plistData, format: nil),
              let dict = plist as? [String: Any] else { return nil }
        return dict["ExpirationDate"] as? Date
    }

    /// "Profile expires in N days" (whole days, rounded down), or the no-profile text.
    static func expiryText(_ expiry: Date?, now: Date = Date()) -> String {
        guard let expiry else { return Copy.noProfile }
        let days = max(0, Int(expiry.timeIntervalSince(now) / 86_400))
        return Copy.profileExpires(days: days)
    }
}
