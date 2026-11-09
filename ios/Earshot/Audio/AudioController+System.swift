// AudioController+System.swift - how AudioController reacts to iOS events (30-ios 6):
// route changes, interruptions (calls, Siri, alarms), media services resets, thermal and
// Low Power changes, and the local "Tap to resume Earshot" notification.
//
// iOS posts these notifications on its own threads, so every observer copies the values it
// needs and hops to the main actor before touching any state.

import AVFoundation
import UIKit
import UserNotifications

extension AudioController {
    // MARK: - Route changes

    func handleRouteChange(reasonRaw: UInt?, received: Date) {
        guard engine != nil else { // not listening: just refresh what the screen shows
            route = RouteInfo(session: AVAudioSession.sharedInstance())
            return
        }
        muteNow() // 1. mute at once
        routeCheckPending = true // and stay muted until the check below has read the new route
        phase0.routeChanges += 1 // I-1: Phase 0 "locked" row counts route changes while listening
        let hopMS = Int(Date().timeIntervalSince(received) * 1000)
        note("route change (reason \(reasonRaw ?? 0)), muted after \(hopMS) ms")
        routeDebounce?.cancel()
        routeDebounce = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 300_000_000) // 2. debounce 300 ms
            if Task.isCancelled { return }
            try? SessionConfig.selectBuiltInMic(AVAudioSession.sharedInstance())
            let r = RouteInfo(session: AVAudioSession.sharedInstance()) // 3. new route
            self.route = r
            self.routeCheckPending = false // the new route is known; refreshMute may unmute again
            guard self.engine != nil else { return }
            // While iOS has I/O stopped (a call) or a rebuild waits for the app to open, the
            // resume path re-reads the route itself, so there is nothing to do here.
            if self.state == .interrupted || self.ioStoppedBySystem || self.pendingRecreate { return }
            // A user pause is kept by settledState, swapEngine and recreateIO, so the engine
            // still follows the route while paused and Resume finds it ready (30-ios 6 step 3).
            if !r.allowed {
                self.state = self.settledState(for: r) // badRoute; I/O keeps running while locked
                self.refreshMute()
            } else if !r.rateSupported || r.sampleRate != self.engineRate {
                await self.recreateIO(reason: .rate)
            } else if r.signature != self.engineSignature {
                await self.swapEngine(reason: .route)
            } else {
                self.state = self.settledState(for: r)
                self.refreshMute()
            }
            self.routeForPrompts(r) // a new AirPods connect offers the delay measurement (30-ios 7)
        }
    }

    // MARK: - Interruptions (calls, Siri, alarms, other apps)

    func handleInterruption(began: Bool, shouldResume: Bool, routeDisconnected: Bool, received: Date = Date()) {
        guard engine != nil else { return }
        if began {
            muteNow()
            // 30-ios 15.2: the time from the notification to the mute call (target under 10 ms).
            let hopMS = Int(Date().timeIntervalSince(received) * 1000)
            state = .interrupted // iOS has already stopped I/O
            ioStoppedBySystem = true
            ioGeneration += 1    // a swap waiting at an await gives up; the swap after the call redoes it
            notice = Copy.interrupted
            note("interruption began, muted after \(hopMS) ms")
            return
        }
        note("interruption ended, shouldResume \(shouldResume)")
        if shouldResume && !routeDisconnected {
            resumeAfterInterruption(fromUser: false)
        } else {
            state = .paused
            if isForeground { notice = Copy.interrupted } else { postResumeNotification(Copy.tapToResume) }
        }
    }

    /// setActive(true), restart RemoteIO, then swap the engine (the route may have changed).
    func resumeAfterInterruption(fromUser: Bool) {
        if pendingRecreate {
            Task { @MainActor in await self.recreateIO(reason: .rate) }
            return
        }
        guard let h = host else { return }
        do {
            try AVAudioSession.sharedInstance().setActive(true)
            let st = eh_start(h)
            if st != noErr { throw AppError.audioUnit("eh_start", st) }
            ioStoppedBySystem = false
            if !keepsHotNotice { notice = nil }
            Task { @MainActor in await self.swapEngine(reason: .interruption) }
        } catch {
            state = .paused
            note("resume failed: \(error)")
            if !isForeground && !fromUser { postResumeNotification(Copy.pausedByCall) }
        }
    }

    // MARK: - Media services reset

    func handleMediaServicesReset() {
        note("media services were reset")
        ioGeneration += 1 // a start or swap waiting at an await gives up
        guard engine != nil || host != nil else { return } // not listening: nothing to rebuild
        // Drop the host without further audio calls (its unit is dead); keep the model.
        stopPolling()
        routeDebounce?.cancel()
        routeCheckPending = false // start() reads the route afresh
        bufferCheck?.cancel()
        stopSafety() // I-4: the alarm tap belongs to the dead engine
        engine?.stopWorker()
        engine = nil
        host = nil
        meterTap = nil
        isUnmuted = false
        ioStoppedBySystem = false
        pendingRecreate = false
        clearQuickAddRAM() // the restart below is a new session for quick add
        state = .idle
        onAudioStopped?() // no engine: Now Playing card and head tracking off (start turns them on again)
        if isForeground {
            Task { @MainActor in try? await self.start() }
        } else {
            postResumeNotification(Copy.tapToResume)
        }
    }

    // MARK: - Thermal and Low Power (30-ios 6, contract 5.3; the rule is Safety/PowerPolicy.swift)

    /// Runs on every thermal / Low Power notification and once a second while listening (the 60 s
    /// recovery needs a clock). Acts only when the level changes:
    /// - saving (serious or Low Power): Selector empties every slot but the selected one (fade to 0,
    ///   150 ms, clearConditioning; no engine rebuild), the alarm classifier runs every 3 s, banner
    /// - critical: also fade to pause, "iPhone is hot - Earshot paused" (a notification when locked)
    /// - normal (fair or better and Low Power off for 60 s): Selector refills the slots cold
    /// The arguments default to the live values; tests pass their own.
    func handlePowerChange(thermal: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState,
                           lowPower: Bool = ProcessInfo.processInfo.isLowPowerModeEnabled,
                           now: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        let before = power.level
        let level = power.update(thermal: thermal, lowPower: lowPower, now: now)
        guard level != before else { return }
        powerLevel = level
        note("power \(before.rawValue) -> \(level.rawValue) (thermal \(MinuteTelemetry.thermalText(thermal)), "
             + "low power \(lowPower))")
        let saving = PowerPolicy.keepsOnlySelected(level)
        if saving != PowerPolicy.keepsOnlySelected(before) {
            onPowerSavingChanged?(saving)                // Selector: empty or refill the other slots
            if alarmListener.running { restartAlarmListener() } // window overlap 0.5 <-> 0
        }
        switch level {
        case .critical:
            pause()
            // pause() acts only while running; during a call (or while paused by a call) the pause
            // is remembered here, so audio does not come back after the call while still hot.
            if engine != nil { userPaused = true }
            notice = Copy.tooHot
            if !isForeground { postResumeNotification(Copy.tooHot) }
        case .saving:
            if !keepsHotNotice { notice = Copy.savingPower }
        case .normal:
            if notice == Copy.savingPower || notice == Copy.tooHot { notice = nil }
        }
    }

    /// True while Earshot is still paused by the hot-phone rule: the banner keeps saying why
    /// (cooling to "serious" or the end of a call does not replace it) until the user resumes.
    var keepsHotNotice: Bool { userPaused && notice == Copy.tooHot }

    // MARK: - Notifications ("Tap to resume Earshot")

    func requestNotificationPermissionOnce() {
        if askedForNotifications { return }
        askedForNotifications = true
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
    }

    func postResumeNotification(_ text: String) {
        let content = UNMutableNotificationContent()
        content.title = Copy.appName
        content.body = text
        let request = UNNotificationRequest(identifier: "earshot.resume", content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }

    // MARK: - System notifications (they arrive off the main thread; hop to the main actor)

    func observeSystemNotifications() {
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil,
                                        queue: nil) { [weak self] n in
            let reason = n.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt
            let received = Date()
            Task { @MainActor in self?.handleRouteChange(reasonRaw: reason, received: received) }
        })
        observers.append(nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil,
                                        queue: nil) { [weak self] n in
            let typeRaw = n.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt ?? 0
            let optionsRaw = n.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0
            let reasonRaw = n.userInfo?[AVAudioSessionInterruptionReasonKey] as? UInt ?? 0
            let began = AVAudioSession.InterruptionType(rawValue: typeRaw) == .began
            let shouldResume = AVAudioSession.InterruptionOptions(rawValue: optionsRaw).contains(.shouldResume)
            let disconnected = AVAudioSession.InterruptionReason(rawValue: reasonRaw) == .routeDisconnected
            let received = Date()
            Task { @MainActor in
                self?.handleInterruption(began: began, shouldResume: shouldResume, routeDisconnected: disconnected,
                                         received: received)
            }
        })
        observers.append(nc.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: nil,
                                        queue: nil) { [weak self] _ in
            Task { @MainActor in self?.handleMediaServicesReset() }
        })
        observers.append(nc.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil,
                                        queue: nil) { [weak self] _ in
            Task { @MainActor in self?.handlePowerChange() }
        })
        observers.append(nc.addObserver(forName: Notification.Name.NSProcessInfoPowerStateDidChange, object: nil,
                                        queue: nil) { [weak self] _ in
            Task { @MainActor in self?.handlePowerChange() }
        })
    }

    // MARK: - Scene phase

    func sceneDidChange(active: Bool) {
        isForeground = active
        if active && pendingRecreate {
            Task { @MainActor in await self.recreateIO(reason: .rate) }
        }
    }
}
