// LatencyMeter.swift - delay strings, probe checks, stored measurements and when to offer a
// measurement (30-ios 7; contract 5.3 "Latency display"; D03, D49).
// Plain logic with no audio: any thread may call it; AudioController (main actor) runs the probe.
//
// Mouth-to-ear delay = loop_ms (measured route round trip, from the probe) + Earshot's
// own share (es_engine_latency_ms). The chip always shows both parts so they add up:
//   "Delay 59 ms (measured 3 min ago) - Earshot 11 + route 48"
// Without a measurement it shows the iOS-reported estimate:
//   "about 62 ms (estimate)"
// Honesty rules (D03): a Bluetooth loop under 20 ms is a method error and is never stored or shown,
// and a Bluetooth estimate under 20 ms is not shown as a number either ("Delay not known yet").
// Everything here is plain logic so LatencyDisplayTests can check it without audio.

import Foundation

/// One valid probe result for one output port (a few bytes, no audio).
struct LatencyMeasurement: Codable, Equatable {
    var loopMS: Double
    var spreadMS: Double
    var psrDB: Double
    var method: String // "in_ear" (method A) or "ear_detection_off" (method B)
    var date: Date
}

/// What the delay chip, Diagnostics and Now Playing show.
struct LatencyDisplay: Equatable {
    var text: String
    var measuredMS: Int? // E + R when a measurement exists for this route, else nil
    var reportedMS: Int  // the iOS-reported estimate (logged as latency_reported_ms)

    static let unknown = LatencyDisplay(text: Copy.delayUnknown, measuredMS: nil, reportedMS: 0)
}

/// Method A (default) keeps the AirPods in; method B is offered after a failed probe (30-ios 7).
enum ProbeMethod: String {
    case inEar = "in_ear"
    case earDetectionOff = "ear_detection_off"
}

enum LatencyMeter {
    /// Earshot never claims a Bluetooth delay under this (D03).
    static let bluetoothFloorMS = 20.0

    /// "just now" under 60 s, then "N min ago", then "N h ago".
    static func ageText(seconds: TimeInterval) -> String {
        if seconds < 60 { return Copy.justNow }
        let minutes = Int(seconds / 60)
        if minutes < 60 { return Copy.minutesAgo(minutes) }
        return Copy.hoursAgo(minutes / 60)
    }

    /// E = round(latencyMS), R = round(loopMS), total = E + R, so the parts always add up.
    static func measuredText(earshotMS: Double, loopMS: Double, ageSeconds: TimeInterval) -> String {
        let e = Int(earshotMS.rounded())
        let r = Int(loopMS.rounded())
        return Copy.delayMeasured(total: e + r, age: ageText(seconds: ageSeconds), earshot: e, route: r)
    }

    /// N = round(1000 x (inputLatency + outputLatency + 2 x ioBufferDuration) + latencyMS).
    static func estimateMS(inputLatency: Double, outputLatency: Double, ioBufferDuration: Double,
                           earshotMS: Double) -> Int {
        Int((1000 * (inputLatency + outputLatency + 2 * ioBufferDuration) + earshotMS).rounded())
    }

    /// Picks the measured string when this route has a stored result, else the estimate.
    static func display(route: RouteInfo, earshotMS: Double, measurement: LatencyMeasurement?,
                        now: Date) -> LatencyDisplay {
        if route.sampleRate == 0 { return .unknown }
        let reported = estimateMS(inputLatency: route.inputLatency, outputLatency: route.outputLatency,
                                  ioBufferDuration: route.ioBufferDuration, earshotMS: earshotMS)
        // A Bluetooth loop under 20 ms can only be a method error, and NaN or a loop <= 0 cannot
        // exist (validate() never stores either; this guards an old or hand-edited value too).
        let trusted = measurement.flatMap { m -> LatencyMeasurement? in
            let bad = !plausible(loopMS: m.loopMS, spreadMS: m.spreadMS)
                || (route.isBluetooth && m.loopMS < bluetoothFloorMS)
            return bad ? nil : m
        }
        if let m = trusted {
            let e = Int(earshotMS.rounded())
            let r = Int(m.loopMS.rounded())
            let text = measuredText(earshotMS: earshotMS, loopMS: m.loopMS,
                                    ageSeconds: now.timeIntervalSince(m.date))
            return LatencyDisplay(text: text, measuredMS: e + r, reportedMS: reported)
        }
        // iOS-reported numbers can be wrong (D03); never show a Bluetooth total under 20 ms.
        if route.isBluetooth && Double(reported) < bluetoothFloorMS {
            return LatencyDisplay(text: Copy.delayUnknown, measuredMS: nil, reportedMS: reported)
        }
        return LatencyDisplay(text: Copy.delayEstimate(reported), measuredMS: nil, reportedMS: reported)
    }

    /// Why a successful probe result is still not trusted.
    enum Rejection: Error, Equatable {
        case impossible                // NaN, infinite, a loop of 0 ms or less, or a negative spread
        case bluetoothTooShort(Double) // a Bluetooth loop under 20 ms is a method error (sound leaked)
        case spreadTooLarge(Double)    // sweeps disagree: over 10 ms on Bluetooth, over 2 ms wired
    }

    /// A loop that can exist physically: a finite number above 0 ms, with a finite spread of 0 or
    /// more. (NaN slips through every "<" check, since any comparison with NaN is false, and
    /// JSONEncoder cannot store it, so it must be caught here.)
    static func plausible(loopMS: Double, spreadMS: Double) -> Bool {
        loopMS.isFinite && loopMS > 0 && spreadMS.isFinite && spreadMS >= 0
    }

    /// The route checks the engine cannot make, because it is never told the route (contract 4.2).
    static func validate(_ r: ProbeResult, isBluetooth: Bool) -> Rejection? {
        guard plausible(loopMS: Double(r.loopMS), spreadMS: Double(r.spreadMS)) else { return .impossible }
        if isBluetooth && Double(r.loopMS) < bluetoothFloorMS { return .bluetoothTooShort(Double(r.loopMS)) }
        let maxSpread: Float = isBluetooth ? 10 : 2
        if r.spreadMS > maxSpread { return .spreadTooLarge(Double(r.spreadMS)) }
        return nil
    }
}

/// The last valid measurement per output port UID, kept in UserDefaults. A failed probe
/// never overwrites it, so the old result stays on screen with its age.
final class LatencyStore {
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    private func key(_ uid: String) -> String { "latency.\(uid)" }

    func last(forPort uid: String) -> LatencyMeasurement? {
        guard !uid.isEmpty, let data = defaults.data(forKey: key(uid)) else { return nil }
        return try? JSONDecoder().decode(LatencyMeasurement.self, from: data)
    }

    func save(_ m: LatencyMeasurement, forPort uid: String) {
        guard !uid.isEmpty, let data = try? JSONEncoder().encode(m) else { return }
        defaults.set(data, forKey: key(uid))
    }
}

/// When to offer a delay measurement (30-ios 7 "When"): a prompt at every connect of an allowed
/// output (never automatic, since it needs the user's hands), then on Bluetooth a banner at 10 and
/// at 30 minutes after that connect ("Re-check the delay - it can drift"), offered and never
/// required. Wired routes measure once per connect (no re-check banners).
struct LatencyPrompts: Equatable {
    enum Prompt: Equatable {
        case measureAtConnect
        case recheck(minutes: Int)
    }

    static let recheckMinutes = [10, 30]

    private(set) var connectedPort = ""      // "" = no allowed output right now
    private(set) var connectedAt: Date? = nil
    private(set) var isBluetooth = false
    private(set) var pending: Prompt? = nil  // what the Listen screen offers now
    private var offered: Set<Int> = []       // re-check minutes already offered for this connect

    /// A route was read (at start, and after each route change settled). An allowed output whose
    /// port UID differs from the connected one is a connect; a bad route is a disconnect, so the
    /// same AirPods coming back (case closed and opened) is a new connect too.
    mutating func routeChanged(_ r: RouteInfo, now: Date) {
        guard r.allowed, !r.primaryOutputUID.isEmpty else {
            connectedPort = ""
            connectedAt = nil
            pending = nil
            return
        }
        guard r.primaryOutputUID != connectedPort else { return }
        connectedPort = r.primaryOutputUID
        connectedAt = now
        isBluetooth = r.isBluetooth
        offered = []
        pending = .measureAtConnect
    }

    /// Called once a second: offers the next re-check once its time has come (Bluetooth only).
    mutating func tick(now: Date) {
        guard isBluetooth, let t0 = connectedAt else { return }
        let minutes = now.timeIntervalSince(t0) / 60
        for m in Self.recheckMinutes where minutes >= Double(m) && !offered.contains(m) {
            offered.insert(m) // when both are due at once, only the later banner shows
            pending = .recheck(minutes: m)
        }
    }

    /// "Not now", or a measurement that worked: the banner goes away (later re-checks still come).
    mutating func dismiss() { pending = nil }
}
