// HeadTracker.swift - AirPods head tracking: smoothed yaw, head rate, live / unavailable, and the
// "frame reset" events that make stored directions stale (30-ios 10.1, contract 5.3, D51).
//
// Owns: the CMHeadphoneMotionManager and the recent yaw samples.
// Threads: Core Motion calls back on a private serial OperationQueue; every handler copies the two
// numbers it needs (timestamp, yaw) and hops to the main actor. Everything else is main actor.
// The hops may arrive out of order, so YawTrack.add drops any sample older than the newest one.
// Spec: 30-ios 10.1 (smoothing 200 ms, rate, live 0.5 s, unavailable 2 s, jump 45 deg / 50 ms).
//
// Two parts so the math can be tested without AirPods:
//   YawTrack     - pure struct: keeps samples, answers smoothed yaw, head rate, live, jump
//   HeadTracker  - the main-actor class around Core Motion that feeds a YawTrack
//
// Only differences of yaw are ever used, so the sign and the zero of AirPods yaw do not matter.

import CoreMotion
import Foundation

/// What look-select and direction learning need from the head at one moment.
struct HeadSnapshot: Equatable {
    var live: Bool             // newest sample under 0.5 s old
    var stale: Bool            // stored directions may belong to another yaw frame (TableState)
    var smoothedYaw: Double?   // circular mean of the last 200 ms, degrees
    var rate: Double?          // degrees per second over the last 200 ms, nil while unknown
}

/// The yaw samples of the last couple of seconds and the formulas of 30-ios 10.1. Pure.
struct YawTrack {
    struct Sample: Equatable { let t: TimeInterval; let yaw: Double }

    static let smoothWindow = 0.2      // seconds (contract 5.3 "yaw smoothed 200 ms")
    static let liveAge = 0.5           // seconds: newer than this counts as live
    static let unavailableAge = 2.0    // seconds without a sample: head tracking paused
    static let jumpDegrees = 45.0      // a raw step larger than this ...
    static let jumpSeconds = 0.05      // ... within less than this marks directions stale
    static let keep = 2.5              // seconds of history kept (enough for every rule above)

    private(set) var samples: [Sample] = []

    /// Adds one raw sample. Returns true when it is a jump: more than 45 degrees away from the
    /// previous sample, less than 50 ms after it (the AirPods frame was probably reset).
    /// - A sample that is not newer than the newest one is dropped. Each sample reaches the main
    ///   actor in its own Task, and Swift does not promise those run in order; every rule below
    ///   assumes `samples` is sorted by time (newest last).
    /// - On a jump the older samples are forgotten: they belong to the old frame, and mixing them
    ///   into the 200 ms mean would give a yaw from neither frame.
    @discardableResult
    mutating func add(t: TimeInterval, yaw: Double) -> Bool {
        if let last = samples.last, t <= last.t { return false } // late or repeated: drop it
        var jump = false
        if let last = samples.last, t - last.t < Self.jumpSeconds,
           abs(SelectorMath.wrap(yaw - last.yaw)) > Self.jumpDegrees {
            jump = true
            samples.removeAll() // smoothing and head rate restart in the new frame
        }
        samples.append(Sample(t: t, yaw: yaw))
        // Forget samples older than `keep` seconds (a few dozen at about 25 Hz).
        if let first = samples.first, t - first.t > Self.keep {
            samples.removeAll { t - $0.t > Self.keep }
        }
        return jump
    }

    /// Forgets every sample (AirPods disconnected or tracking stopped).
    mutating func clear() { samples.removeAll() }

    var newest: Sample? { samples.last }

    func isLive(now: TimeInterval) -> Bool {
        guard let n = newest else { return false }
        return now - n.t < Self.liveAge
    }

    func isUnavailable(now: TimeInterval) -> Bool {
        guard let n = newest else { return true }
        return now - n.t >= Self.unavailableAge
    }

    /// Circular mean atan2(mean sin, mean cos) of the samples in the last 200 ms before `now`;
    /// the newest sample alone if fewer than 2 are that recent; nil with no samples at all.
    func smoothedYaw(now: TimeInterval) -> Double? {
        guard let n = newest else { return nil }
        let recent = samples.filter { now - $0.t <= Self.smoothWindow }.map { $0.yaw }
        if recent.count < 2 { return n.yaw }
        return SelectorMath.circularMean(recent) ?? n.yaw
    }

    /// abs(wrap(y_now - y_200ms_ago)) / 0.2 s, where y_now is the newest sample and y_200ms_ago the
    /// newest sample at least 200 ms older than it. nil until 200 ms of history exist.
    func headRate() -> Double? {
        guard let n = newest,
              let old = samples.last(where: { n.t - $0.t >= Self.smoothWindow }) else { return nil }
        return abs(SelectorMath.wrap(n.yaw - old.yaw)) / Self.smoothWindow
    }

    /// Samples per second over the last second, for Diagnostics (about 25 Hz is a community
    /// report, undocumented).
    func updateRate(now: TimeInterval) -> Double {
        let lastSecond = samples.filter { now - $0.t <= 1.0 }
        guard lastSecond.count >= 2, let a = lastSecond.first, let b = lastSecond.last, b.t > a.t else { return 0 }
        return Double(lastSecond.count - 1) / (b.t - a.t)
    }
}

/// The main-actor wrapper around CMHeadphoneMotionManager (no entitlement needed; Info.plist has
/// NSMotionUsageDescription, without which the app would crash).
@MainActor
final class HeadTracker: NSObject, ObservableObject, CMHeadphoneMotionManagerDelegate {
    /// Newest sample under 0.5 s old (refreshed by `refresh(now:)` at 10 Hz).
    @Published private(set) var live = false
    /// No sample for 2 s: the Listen screen shows "Head tracking paused" and look-select cycles.
    @Published private(set) var unavailable = true
    /// Samples per second and the age of the newest sample, for Diagnostics.
    @Published private(set) var updateRateHz = 0.0
    @Published private(set) var newestAge: Double? = nil
    @Published private(set) var running = false

    /// Called on a yaw jump and on every AirPods connect or disconnect: the reference frame may
    /// have changed, so the Selector marks the stored directions stale (30-ios 10.1).
    var onFrameReset: (@MainActor () -> Void)?
    /// Phase 0 motion step (I-1): every raw sample's sensor timestamp, for the rate and gap count.
    var onSample: (@MainActor (TimeInterval) -> Void)?

    private(set) var track = YawTrack()
    private let manager = CMHeadphoneMotionManager()
    private let queue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 1 // serial, as 30-ios 10.1 asks
        q.name = "earshot.headmotion"
        return q
    }()

    override init() {
        super.init()
        manager.delegate = self
    }

    /// Seconds since boot, the same clock as CMDeviceMotion.timestamp.
    nonisolated static var now: TimeInterval { ProcessInfo.processInfo.systemUptime }

    /// Starts motion updates (called when listening starts). Safe to call twice.
    func start() {
        guard !running, manager.isDeviceMotionAvailable else { return }
        running = true
        manager.startDeviceMotionUpdates(to: queue) { [weak self] motion, _ in
            guard let motion else { return }
            let t = motion.timestamp
            let yaw = motion.attitude.yaw * 180 / .pi // radians to degrees, range (-180, 180]
            Task { @MainActor in self?.receive(t: t, yaw: yaw) }
        }
    }

    /// Stops motion updates (called when listening stops) and forgets the samples.
    func stop() {
        guard running else { return }
        manager.stopDeviceMotionUpdates()
        running = false
        track.clear()
        refresh(now: Self.now)
    }

    /// One raw sample on the main actor. Internal (not private) so tests can feed samples.
    func receive(t: TimeInterval, yaw: Double) {
        onSample?(t)
        if track.add(t: t, yaw: yaw) { onFrameReset?() }
    }

    /// Updates the published flags (called with the 10 Hz stats poll). Each value is assigned only
    /// when it changed, so the screens watching this tracker do not redraw 10 times a second for
    /// nothing. The rate and age are rounded to what Diagnostics shows (0.1 Hz, 0.1 s).
    func refresh(now: TimeInterval) {
        let isLive = track.isLive(now: now)
        let isUnavailable = track.isUnavailable(now: now)
        let rate = (track.updateRate(now: now) * 10).rounded() / 10
        let age = track.newest.map { ((now - $0.t) * 10).rounded() / 10 }
        if live != isLive { live = isLive }
        if unavailable != isUnavailable { unavailable = isUnavailable }
        if updateRateHz != rate { updateRateHz = rate }
        if newestAge != age { newestAge = age }
    }

    /// Smoothed yaw, rate and live flag right now; `stale` comes from the table's directions.
    func snapshot(now: TimeInterval, directionsStale: Bool) -> HeadSnapshot {
        HeadSnapshot(live: track.isLive(now: now), stale: directionsStale,
                     smoothedYaw: track.smoothedYaw(now: now), rate: track.headRate())
    }

    /// The yaw CaptureSession records every 250 ms while someone is enrolled: the smoothed yaw while
    /// motion is live, else nil (then the capture stores no direction, 30-ios 8.1).
    func yawForCapture(now: TimeInterval = HeadTracker.now) -> Double? {
        track.isLive(now: now) ? track.smoothedYaw(now: now) : nil
    }

    // MARK: - CMHeadphoneMotionManagerDelegate (called off the main thread)

    nonisolated func headphoneMotionManagerDidConnect(_ manager: CMHeadphoneMotionManager) {
        Task { @MainActor in self.frameReset() }
    }

    nonisolated func headphoneMotionManagerDidDisconnect(_ manager: CMHeadphoneMotionManager) {
        Task { @MainActor in self.frameReset() }
    }

    /// A connect or disconnect: the old samples belong to the old frame.
    private func frameReset() {
        track.clear()
        refresh(now: Self.now)
        onFrameReset?()
    }
}
