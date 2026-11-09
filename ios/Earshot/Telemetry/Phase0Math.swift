// Phase0Math.swift - the arithmetic behind each Phase 0 row (30-ios 15.3), as pure functions so
// Phase0MathTests can check them without a phone. Any thread (no state).
//
//   Phase0Math.bins          compute histogram, fixed microsecond bins for the file
//   Phase0Math.worstWindow   worst 60 s of underruns from per-second running totals
//   Phase0Math.tapPasses     tap test: phone tap peak at least 20 dB above the AirPod tap peak
//   MotionGaps               head-motion rate, gaps over 200 ms, longest gap
//   Phase0Math.pctPerHour    battery drain
//   Phase0Math.probePick     the probe level for ES_PROBE_DBFS
//   ResumeTracker            locked route changes: did audio come back without a tap?
//   Phase0Math.micLevel      speech level over VAD speech frames only
//   Phase0Math.expectedAlarmWindows / alarmOK
//   Phase0Math.screenLabel   on / locked / mixed from the seconds the app was active
//   Phase0Math.routeLabel    airpods_pro1 / airpods_pro2 / no_headphones / wired
//   Phase0Math.minuteSlot    which 0 / 10 / 30 min latency item a probe counts for

import Foundation

enum Phase0Math {
    // MARK: Compute histogram

    /// Bin edges in microseconds: 50 us wide up to 1 ms (where the 1-instance target lives), then
    /// wider to 10 ms, then one open bin. A hop is 4 ms, so anything past 4,000 us is a deadline risk.
    static let binEdges: [Int] = Array(stride(from: 0, through: 1000, by: 50)) + [1250, 1500, 2000, 3000, 4000, 5000, 10_000]

    /// Counts per bin (lo <= x < hi, the last bin has no hi). Every bin is listed, empty or not,
    /// so files from different runs line up. Uses the 1 us ComputeHistogram of SessionLog.swift;
    /// values past its last 1 us bin (10,000 us) land in the open bin.
    static func bins(_ h: ComputeHistogram) -> [Phase0HistBin] {
        var out = binEdges.indices.map { i in
            Phase0HistBin(lo: binEdges[i], hi: i + 1 < binEdges.count ? binEdges[i + 1] : nil, n: 0)
        }
        for (value, n) in h.counts.enumerated() where n > 0 {
            out[binIndex(value)].n += Int(n)
        }
        return out
    }

    /// The bin a value in microseconds belongs to.
    static func binIndex(_ us: Int) -> Int {
        var i = 0
        while i + 1 < binEdges.count && us >= binEdges[i + 1] { i += 1 }
        return i
    }

    // MARK: Underruns

    /// The most underruns in any `window` consecutive seconds, from running totals sampled once a
    /// second (cumulative[k] = underruns from the start of the run up to and including second k).
    /// A run that fits in one window gives its whole count.
    static func worstWindow(cumulative: [Int], window: Int = 60) -> Int {
        guard let last = cumulative.last else { return 0 }
        if cumulative.count <= window { return last }
        var worst = cumulative[window - 1] // seconds 0 ..< window (the totals start at 0)
        for end in window..<cumulative.count {
            worst = max(worst, cumulative[end] - cumulative[end - window])
        }
        return worst
    }

    // MARK: Tap test (30-ios 15.3 "Tap test")

    static let tapNeededDB = 20.0

    /// Pass when the phone tap peak is at least 20 dB above the AirPod tap peak: the iPhone mic is
    /// the one listening, not an AirPods mic.
    static func tapPasses(phonePeakDB: Double, airpodPeakDB: Double) -> Bool {
        phonePeakDB - airpodPeakDB >= tapNeededDB
    }

    // MARK: Battery

    /// Percent per hour from the first and the last battery reading of a run (nil when too short).
    static func pctPerHour(startPct: Int, endPct: Int, minutes: Double) -> Double? {
        guard minutes >= 1 else { return nil }
        return Double(startPct - endPct) * 60 / minutes
    }

    // MARK: Probe level (30-ios 15.3 "Latency (addition)")

    /// Fewest tries a room needs at a level before its success rate counts.
    static let minTriesPerRoom = 5
    /// Rooms are split at 58 dB: "about 45 dB" is quiet, "about 70 dB" is loud.
    static let loudRoomDB = 58

    /// Success rate (0...1) and tries for one level in the quiet or the loud room, over both
    /// methods and both AirPods. nil when there are no tries.
    static func probeSuccess(_ rows: [Phase0LatencyRow], level: Int, loud: Bool) -> (rate: Double, tries: Int)? {
        let tries = rows.filter { $0.probeDBFS == level && ($0.ambientDB >= loudRoomDB) == loud }
        guard !tries.isEmpty else { return nil }
        let ok = tries.filter { $0.ok ?? ($0.loopMS != nil) }.count
        return (Double(ok) / Double(tries.count), tries.count)
    }

    /// The lowest of -30, -24, -18 dBFS with at least 80% success in both rooms (each with at
    /// least 5 tries); nil while no level qualifies. The pick becomes a DECISIONS row.
    static func probePick(_ rows: [Phase0LatencyRow], levels: [Int] = Phase0Plan.probeLevels) -> Int? {
        for level in levels.sorted() {
            guard let quiet = probeSuccess(rows, level: level, loud: false),
                  let loud = probeSuccess(rows, level: level, loud: true),
                  quiet.tries >= minTriesPerRoom, loud.tries >= minTriesPerRoom else { continue }
            if quiet.rate >= 0.8 && loud.rate >= 0.8 { return level }
        }
        return nil
    }

    // MARK: Mic level (30-ios 15.3 "Mic level")

    /// RMS and peak in dBFS over the speech frames of `mask` (es_vad, one entry per 512-sample
    /// token frame, ClipTools.frameSamples), and the speech seconds. nil without speech frames.
    static func micLevel(pcm: [Float], mask: [UInt8]) -> (rmsDB: Double, peakDB: Double, speechS: Double)? {
        var sum = 0.0
        var peak: Float = 0
        var count = 0
        for f in ClipTools.keptFrames(mask) {
            for i in ClipTools.frameSamples(f, n: pcm.count) {
                let x = pcm[i]
                sum += Double(x) * Double(x)
                peak = max(peak, abs(x))
                count += 1
            }
        }
        guard count > 0 else { return nil }
        let rms = (sum / Double(count)).squareRoot()
        return (20 * log10(max(rms, 1e-10)), 20 * log10(max(Double(peak), 1e-10)), ClipTools.speechSeconds(mask))
    }

    // MARK: Alarm classifier (30-ios 15.3 "WeSpeaker and alarm, locked")

    /// Windows a run of `seconds` should produce: 3 s windows starting every 3 x (1 - overlap) s.
    static func expectedAlarmWindows(seconds: Double, windowS: Double = 3, overlap: Double) -> Int {
        let hop = windowS * (1 - overlap)
        guard seconds >= windowS, hop > 0 else { return 0 }
        return Int(((seconds - windowS) / hop).rounded(.down)) + 1
    }

    /// OK when at least 80% of the expected windows came back (the classifier kept running).
    static func alarmOK(windows: Int, expected: Int) -> Bool {
        expected > 0 && Double(windows) >= 0.8 * Double(expected)
    }

    // MARK: Latency minute slots

    /// The 0 / 10 / 30 min item a probe taken `minute` minutes after connect counts for: under 5
    /// is "0", 10 to 19 is "10", 30 or more is "30"; other minutes count for none.
    static func minuteSlot(_ minute: Int) -> Int? {
        if minute < 5 { return 0 }
        if minute >= 10 && minute < 20 { return 10 }
        if minute >= 30 { return 30 }
        return nil
    }

    // MARK: Small formatting helpers for the file

    /// Seconds to milliseconds, two decimals.
    static func ms(_ seconds: Double) -> Double { (seconds * 100_000).rounded() / 100 }

    /// One decimal, so the file stays readable.
    static func round1(_ x: Double) -> Double { (x * 10).rounded() / 10 }

    /// "AVAudioSessionModeMeasurement" -> "measurement" (the file's `mode`).
    static func modeName(_ raw: String) -> String {
        let prefix = "AVAudioSessionMode"
        guard raw.hasPrefix(prefix), raw.count > prefix.count else { return raw }
        let rest = raw.dropFirst(prefix.count)
        return rest.prefix(1).lowercased() + rest.dropFirst()
    }

    // MARK: Screen and route labels

    /// "locked" when the app was in the background at least 80% of the run (locking the phone
    /// sends the app to the background), "on" when active at least 80%, else "mixed".
    static func screenLabel(activeSeconds: Int, inactiveSeconds: Int) -> String {
        let total = activeSeconds + inactiveSeconds
        guard total > 0 else { return "on" }
        if Double(inactiveSeconds) >= 0.8 * Double(total) { return "locked" }
        if Double(activeSeconds) >= 0.8 * Double(total) { return "on" }
        return "mixed"
    }

    /// The route name of the file for the current output. AirPods need the owner's answer to
    /// "Which AirPods are these?" (iOS gives no model name); nil means "ask".
    static func routeLabel(_ r: RouteInfo, model: AirPodsModel?) -> String? {
        guard let out = r.outputs.first else { return "no_headphones" }
        switch out.type {
        case .bluetoothA2DP:
            guard let model else { return nil }
            return model == .pro1 ? "airpods_pro1" : "airpods_pro2"
        case .headphones, .usbAudio: return "wired"
        default: return "no_headphones" // speaker, receiver, HFP, AirPlay: Earshot stays muted there
        }
    }
}

/// Head-motion samples of one run: rate, gaps over 200 ms, longest gap (sensor timestamps).
struct MotionGaps: Equatable {
    static let gapLimit = 0.2 // seconds

    private(set) var count = 0
    private(set) var first: TimeInterval? = nil
    private(set) var last: TimeInterval? = nil
    private(set) var gapsOver200ms = 0
    private(set) var maxGap: TimeInterval = 0

    mutating func add(_ t: TimeInterval) {
        if let last {
            guard t > last else { return } // late or repeated sample
            let gap = t - last
            if gap > Self.gapLimit { gapsOver200ms += 1 }
            maxGap = max(maxGap, gap)
        } else {
            first = t
        }
        last = t
        count += 1
    }

    /// Samples per second between the first and the last sample.
    var rateHz: Double {
        guard let first, let last, last > first, count >= 2 else { return 0 }
        return Double(count - 1) / (last - first)
    }

    /// The longest gap in ms, also counting the time from the last sample to `end` (a run where the
    /// samples stopped early must not look clean).
    func maxGapMS(end: TimeInterval?) -> Double {
        var g = maxGap
        if let end, let last, end > last { g = max(g, end - last) }
        return g * 1000
    }
}

/// The locked step (30-ios 15.3 "Locked"): after each route change or engine swap, did audio come
/// back on its own? Quick changes in a row (AirPods out, then in) are one event, so the two
/// counts add up to the recoveries seen, while the row's route_changes counts every notification. An event is open until the route is allowed and audio runs again (resumed
/// without a tap); if the route is allowed but audio is still not running `deadline` seconds after
/// the last change, it needed a tap. While the route is not allowed (AirPods out) nothing is
/// expected yet, so the clock waits.
struct ResumeTracker: Equatable {
    static let deadline: TimeInterval = 10

    private(set) var resumedWithoutTap = 0
    private(set) var neededTap = 0
    private var openSince: TimeInterval? = nil  // time of the newest change while an event is open
    private var lastChanges = 0

    /// Called once a second. `changes` = route changes + swaps so far (running total), `running` =
    /// audio plays (state .running), `routeAllowed` = output is headphones, `userPaused` = the owner
    /// paused on purpose (then not resuming is correct, and the event is dropped).
    mutating func sample(now: TimeInterval, changes: Int, routeAllowed: Bool, running: Bool, userPaused: Bool) {
        if changes > lastChanges {
            lastChanges = changes
            openSince = now // a new change (re)opens the event; several quick ones count as one
            return
        }
        guard let since = openSince else { return }
        if userPaused { openSince = nil; return }
        if !routeAllowed { openSince = now; return } // AirPods out: wait for them to come back
        if running {
            resumedWithoutTap += 1
            openSince = nil
        } else if now - since >= Self.deadline {
            neededTap += 1
            openSince = nil
        }
    }
}
