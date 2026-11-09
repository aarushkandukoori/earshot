// SessionConfig.swift - AVAudioSession setup and read-back (30-ios 3.1, contract 5.3).
//
// One rule matters most: the category options are ONLY .allowBluetoothA2DP. Never
// HFP (.allowBluetoothHFP), never .defaultToSpeaker, never .mixWithOthers, never
// .bluetoothHighQualityRecording. HFP would move the mic to the AirPods and win over
// A2DP (fact-check.md 5), and Earshot listens through the iPhone's own mic.

import AVFoundation

/// Errors the audio side can report to the screen.
enum AppError: Error, LocalizedError {
    case noBuiltInMic
    case unsupportedRate(Double)
    case audioUnit(String, OSStatus)
    case notReady
    case notForeground
    case notRunning
    case probeRejected(String)
    case probeTimeout
    case stopped   // Stop or a media reset arrived while start or recreate was waiting

    var errorDescription: String? {
        switch self {
        case .noBuiltInMic: return Copy.noBuiltInMic
        case .unsupportedRate(let rate): return Copy.rateUnsupported(rate)
        case .audioUnit(let step, let status): return Copy.audioSetupFailed(step, status)
        case .notReady: return Copy.notReady
        case .notForeground: return Copy.openToStart
        case .notRunning: return Copy.startFirst
        case .probeRejected(let why): return why
        case .probeTimeout: return Copy.probeWeak
        case .stopped: return Copy.notListening
        }
    }

    /// The text a screen may show for any error: an AppError's own text (always from Copy), else
    /// `fallback`. System and engine error texts are never shown; callers send them to
    /// Diagnostics (AudioController.note) instead.
    static func userText(_ error: Error, fallback: String) -> String {
        (error as? AppError)?.errorDescription ?? fallback
    }
}

@MainActor
enum SessionConfig {
    /// Contract mode. If Phase 0 shows it breaks A2DP output, this one constant changes (30-ios 3.1).
    static let mode: AVAudioSession.Mode = .measurement
    static let preferredSampleRate = 48_000.0
    static let preferredFrames = 64 // 1.33 ms at 48 kHz, only a hint; iOS decides
    /// What the next activation asks for. Always preferredFrames, except while the Phase 0 buffer
    /// step (I-1, 30-ios 15.3 "Buffers") requests 32 to 192 frames; it puts 64 back afterwards.
    static var requestedFrames = preferredFrames

    /// Sets up and activates the session, picks the built-in mic, and reads the route back.
    static func activateSession() throws -> RouteInfo {
        let s = AVAudioSession.sharedInstance()
        try s.setCategory(.playAndRecord, mode: mode, options: [.allowBluetoothA2DP]) // never HFP, speaker, mixable
        try s.setPreferredSampleRate(preferredSampleRate)
        try s.setPreferredIOBufferDuration(Double(requestedFrames) / preferredSampleRate)
        try s.setPrefersNoInterruptionsFromSystemAlerts(true)
        try s.setAllowHapticsAndSystemSoundsDuringRecording(true)
        try s.setActive(true)
        try selectBuiltInMic(s) // after activation, and after every route change
        return RouteInfo(session: s)
    }

    /// Makes the iPhone's built-in mic the input. Skips the call when it is already preferred,
    /// because setPreferredInput itself fires a route change notification.
    static func selectBuiltInMic(_ s: AVAudioSession) throws {
        guard let mic = s.availableInputs?.first(where: { $0.portType == .builtInMic }) else {
            throw AppError.noBuiltInMic
        }
        if s.preferredInput?.uid == mic.uid, s.currentRoute.inputs.first?.portType == .builtInMic { return }
        try s.setPreferredInput(mic) // no data source or polar pattern in v1
    }

    /// Ends the session and lets other apps (music) resume.
    static func deactivate() {
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}
