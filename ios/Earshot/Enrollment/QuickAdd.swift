// QuickAdd.swift - "Add who just spoke" (30-ios 8.5).
//
// 1. Take the last 8 s of mic audio from the engine's 16 kHz history (it already holds 15.36 s).
//    Right after an engine swap the new engine has little history, so the RAM copy the swap made
//    (AudioController.quickAddRAM, never written to disk) is put in front of the new audio.
// 2. Run the normal target checks (EnrollmentService).
// 3. Play back the 2 s with the most speech, engine muted, at -26 dBFS RMS with peaks at most
//    -6 dBFS, from an in-memory WAV, then unmute. "Is this them?" Yes saves a tonight person,
//    No throws everything away.

import AVFoundation
import Foundation

enum QuickAdd {
    static let samples = 128_000          // 8 s at 16 kHz
    static let playbackSamples = 32_000   // 2 s
    static let targetRMS = Float(pow(10.0, -26.0 / 20.0))  // -26 dBFS
    static let peakLimit = Float(pow(10.0, -6.0 / 20.0))   // -6 dBFS

    /// The newest `want` samples of RAM copy + fresh history (fresh alone once it is long enough).
    static func combine(ram: [Float], fresh: [Float], want: Int = samples) -> [Float] {
        if fresh.count >= want || ram.isEmpty { return Array(fresh.suffix(want)) }
        return Array((ram + fresh).suffix(want))
    }

    /// The last 8 s, from the running engine (plus the swap's RAM copy only while it still belongs
    /// to that engine, AudioController.quickAddHistory).
    @MainActor static func recentAudio(_ audio: AudioController) -> [Float] {
        audio.quickAddHistory(samples: samples)
    }

    /// The 2 s of the clip with the most speech frames (same search as the 10 s window).
    static func playbackRange(mask: [UInt8], clipLength: Int) -> Range<Int> {
        guard clipLength > playbackSamples else { return 0..<clipLength }
        let start = ClipTools.bestWindow(mask: mask, regionStart: 0, regionEnd: clipLength, length: playbackSamples)
        return start..<(start + playbackSamples)
    }

    /// Scales to -26 dBFS RMS, then lowers it further if a peak would pass -6 dBFS.
    static func normalizeForPlayback(_ x: [Float]) -> [Float] {
        guard !x.isEmpty else { return x }
        let rms = (x.reduce(0) { $0 + $1 * $1 } / Float(x.count)).squareRoot()
        let peak = x.reduce(0) { max($0, abs($1)) }
        guard rms > 1e-9, peak > 0 else { return x }
        var gain = targetRMS / rms
        if peak * gain > peakLimit { gain = peakLimit / peak }
        return x.map { $0 * gain }
    }
}

/// Plays the 2 s check clip with the engine muted, then lets the mute rule unmute again.
/// If the route goes away during the clip (AirPods taken out or disconnected), the clip stops at
/// once, so someone else's voice never continues from the phone (I-3 fix of the I-2 route race).
@MainActor
final class QuickAddPlayer: NSObject, AVAudioPlayerDelegate {
    private var player: AVAudioPlayer?
    private var done: CheckedContinuation<Void, Never>?
    private var stoppedByRoute = false

    /// True when the clip may play: AVAudioPlayer skips Earshot's own route rule, so the clip
    /// (someone else's voice) plays only on an allowed route (A2DP, headphones, USB; contract 5.3)
    /// that is not being re-checked after a route change. Never through the phone's speaker.
    nonisolated static func mayPlay(routeAllowed: Bool, routeCheckPending: Bool) -> Bool {
        routeAllowed && !routeCheckPending
    }

    /// True when a route change during the clip must stop it: the old output went away (reason
    /// oldDeviceUnavailable) or the route now is not an allowed output. Other changes (for example
    /// Earshot re-selecting the built-in mic) let it finish.
    nonisolated static func shouldStop(reason: AVAudioSession.RouteChangeReason, newRouteAllowed: Bool) -> Bool {
        reason == .oldDeviceUnavailable || !newRouteAllowed
    }

    /// Plays the clip and returns true, or returns false (nothing or only part played) when the
    /// route does not allow it or went away during the clip.
    @discardableResult
    func play(_ clip: [Float], audio: AudioController) async throws -> Bool {
        guard Self.mayPlay(routeAllowed: audio.route.allowed, routeCheckPending: audio.routeCheckPending) else {
            return false
        }
        let wav = ClipTools.encodeWAV(QuickAdd.normalizeForPlayback(clip))
        let p = try AVAudioPlayer(data: wav) // in memory: nothing is written to disk
        p.delegate = self
        player = p
        stoppedByRoute = false
        audio.holdMuted = true // engine output fades out (5 ms); the probe flag works the same way
        // Route changes arrive off the main thread: copy the reason, hop to the main actor.
        let observer = NotificationCenter.default.addObserver(forName: AVAudioSession.routeChangeNotification,
                                                              object: nil, queue: nil) { [weak self] n in
            let raw = n.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt ?? 0
            let reason = AVAudioSession.RouteChangeReason(rawValue: raw) ?? .unknown
            Task { @MainActor in self?.routeChanged(reason) }
        }
        defer {
            NotificationCenter.default.removeObserver(observer)
            audio.holdMuted = false // refreshMute fades back in if listening is allowed
            player = nil
        }
        try await Task.sleep(nanoseconds: 20_000_000) // let the 5 ms mute ramp finish first
        // The route may have changed during those 20 ms.
        guard !stoppedByRoute,
              Self.mayPlay(routeAllowed: audio.route.allowed, routeCheckPending: audio.routeCheckPending) else {
            return false
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            done = cont
            if !p.play() { finish() }
        }
        return !stoppedByRoute
    }

    /// A route change while the clip is loaded or playing: stop at once if the output went away.
    private func routeChanged(_ reason: AVAudioSession.RouteChangeReason) {
        guard let p = player else { return }
        let allowed = RouteInfo(session: AVAudioSession.sharedInstance()).allowed
        guard Self.shouldStop(reason: reason, newRouteAllowed: allowed) else { return }
        stoppedByRoute = true
        p.stop() // stop() does not call the delegate, so finish here
        finish()
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.finish() }
    }

    nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        Task { @MainActor in self.finish() }
    }

    private func finish() {
        done?.resume()
        done = nil
    }
}
