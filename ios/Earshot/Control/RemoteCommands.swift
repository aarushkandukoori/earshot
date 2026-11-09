// RemoteCommands.swift - AirPods stem presses and the Now Playing card (30-ios 11; contract 5.3
// "Stem single / double / triple"; D50; DECISIONS M.6 default mapping).
//
// Owns: the MPRemoteCommandCenter targets and MPNowPlayingInfoCenter's info while listening.
// Threads: the command handlers may arrive on any thread; each one hops to the main actor and
// hands the command to `onCommand` (the Selector). Everything else is main actor.
//
// With the A2DP-only session, AirPods presses arrive as AVRCP commands, but only while Earshot is
// the Now Playing app: a non-mixable .playAndRecord session, audio running, at least one handler,
// and nowPlayingInfo set. So start() runs once audio runs and stop() when listening stops.
//
//   press   command(s)                                Earshot action
//   single  togglePlayPause, play, pause              pause / resume (play only resumes, pause only pauses)
//   double  nextTrack                                 look-select; with nobody at the table, hands-free capture
//   triple  previousTrack                             Focus <-> Table if >= 2 people, else Focus <-> Everyone
//
// The mapping is the pure `StemMap` (StemMappingTests checks the whole table).

import Foundation
import MediaPlayer

/// The AVRCP commands Earshot listens to.
enum RemoteCommandKind: String, CaseIterable, Sendable {
    case togglePlayPause, play, pause, nextTrack, previousTrack
}

/// The physical press behind a command (counted for Diagnostics, 30-ios 14).
enum StemPress: String, CaseIterable, Sendable {
    case single, double, triple
}

/// What Earshot does for one command.
enum StemAction: Equatable, Sendable {
    case pause, resume, lookSelect, handsFreeCapture, toggleMode, nothing
}

enum StemMap {
    static func press(for kind: RemoteCommandKind) -> StemPress {
        switch kind {
        case .togglePlayPause, .play, .pause: return .single
        case .nextTrack: return .double
        case .previousTrack: return .triple
        }
    }

    /// `isPaused` - the user paused (or a call did); `tableEmpty` - nobody is at the table.
    static func action(for kind: RemoteCommandKind, isPaused: Bool, tableEmpty: Bool) -> StemAction {
        switch kind {
        case .togglePlayPause: return isPaused ? .resume : .pause
        case .play: return isPaused ? .resume : .nothing     // play only resumes
        case .pause: return isPaused ? .nothing : .pause     // pause only pauses (also an AirPod taken out)
        case .nextTrack: return tableEmpty ? .handsFreeCapture : .lookSelect
        case .previousTrack: return .toggleMode
        }
    }
}

/// The Now Playing title (also the "who is playing" line on Listen without the "Earshot - " part).
enum NowPlayingText {
    /// "Focus: Maya", "Table: 2 people", "Everyone", "Paused".
    static func status(mode: TableMode, focusName: String?, tablePeople: Int, paused: Bool) -> String {
        if paused { return Copy.modePaused }
        switch mode {
        case .focus: return focusName.map(Copy.playingFocus) ?? Copy.modeFocus
        case .table: return Copy.playingTable(tablePeople)
        case .everyone: return Copy.modeEveryone
        }
    }

    /// "Earshot - Focus: Maya" (30-ios 11).
    static func title(mode: TableMode, focusName: String?, tablePeople: Int, paused: Bool) -> String {
        Copy.nowPlayingTitle(status(mode: mode, focusName: focusName, tablePeople: tablePeople, paused: paused))
    }
}

@MainActor
final class RemoteCommands: ObservableObject {
    /// Presses received per kind since launch (Diagnostics "stem press counters").
    @Published private(set) var counts: [StemPress: Int] = [:]
    @Published private(set) var active = false

    /// The Selector's handler; called on the main actor.
    var onCommand: (@MainActor (RemoteCommandKind) -> Void)?

    /// Phase 0 stem step (I-1): presses are only counted, never acted on, so a test double press
    /// cannot start a hands-free capture and a single press cannot resume.
    var countOnly = false
    /// Phase 0 "single press while paused at rate 0" (30-ios 15.3 Stem): when set, the paused
    /// Now Playing card uses this rate instead of `pausedRate`. nil in normal use.
    var pausedRateOverride: Double? = nil {
        didSet { lastInfo = nil } // the next update() rewrites the card with the new rate
    }

    /// Playback rate while paused. 30-ios 11 keeps rate 1 unless Phase 0 shows a single press
    /// still arrives with rate 0, so this stays 1 until a Phase 0 row says otherwise.
    static let pausedRate = 1.0

    private var targets: [(command: MPRemoteCommand, token: Any)] = []
    private var lastInfo: (title: String, artist: String, paused: Bool)? = nil

    /// Registers the five handlers and disables every other command (called when audio runs).
    func start() {
        guard targets.isEmpty else { return }
        let c = MPRemoteCommandCenter.shared()
        add(c.togglePlayPauseCommand, .togglePlayPause)
        add(c.playCommand, .play)
        add(c.pauseCommand, .pause)
        add(c.nextTrackCommand, .nextTrack)
        add(c.previousTrackCommand, .previousTrack)
        // All other commands (skip intervals, seek, position, rate, like, bookmark, ...) are off.
        let off: [MPRemoteCommand] = [c.stopCommand, c.skipForwardCommand, c.skipBackwardCommand,
                                      c.seekForwardCommand, c.seekBackwardCommand, c.changePlaybackPositionCommand,
                                      c.changePlaybackRateCommand, c.changeRepeatModeCommand,
                                      c.changeShuffleModeCommand, c.likeCommand, c.dislikeCommand,
                                      c.bookmarkCommand, c.enableLanguageOptionCommand,
                                      c.disableLanguageOptionCommand, c.ratingCommand]
        for cmd in off { cmd.isEnabled = false }
        active = true
    }

    /// Removes the handlers and the Now Playing card (called when listening stops).
    func stop() {
        for t in targets { t.command.removeTarget(t.token) }
        targets = []
        MPNowPlayingInfoCenter.default().nowPlayingInfo = nil
        lastInfo = nil
        active = false
    }

    /// Sets the Now Playing card: title "Earshot - Focus: Maya", artist = the delay chip text, a
    /// live stream at rate 1. Skips the call when nothing changed (it runs once a second).
    func update(title: String, artist: String, paused: Bool) {
        guard active else { return }
        if let l = lastInfo, l.title == title, l.artist == artist, l.paused == paused { return }
        lastInfo = (title, artist, paused)
        MPNowPlayingInfoCenter.default().nowPlayingInfo = [
            MPMediaItemPropertyTitle: title,
            MPMediaItemPropertyArtist: artist,
            MPNowPlayingInfoPropertyIsLiveStream: true,
            MPNowPlayingInfoPropertyPlaybackRate: paused ? (pausedRateOverride ?? Self.pausedRate) : 1.0,
        ]
    }

    /// One command arrived (main actor). Internal so tests can simulate a press.
    func received(_ kind: RemoteCommandKind) {
        counts[StemMap.press(for: kind), default: 0] += 1
        if countOnly { return } // Phase 0 stem step: counted only
        onCommand?(kind)
    }

    private func add(_ command: MPRemoteCommand, _ kind: RemoteCommandKind) {
        command.isEnabled = true
        let token = command.addTarget { [weak self] _ in
            Task { @MainActor in self?.received(kind) }
            return .success
        }
        targets.append((command, token))
    }
}
