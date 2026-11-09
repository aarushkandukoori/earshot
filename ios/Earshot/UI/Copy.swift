// Copy.swift - every user-facing string lives here (30-ios 12.1).
//
// CopyLintTests fails on health claims, speed promises and em dashes,
// so a new string must be added here (and to `allSamples`) rather than typed into a view.

import Foundation

enum Copy {
    // MARK: What it is
    static let appName = "Earshot"
    static let disclaimer = "Earshot is a listening tool, not a hearing aid."
    static let privacy = "Audio never leaves this iPhone."

    // MARK: Launch
    static let gettingReady = "Getting ready"
    static let modelMissing = "Earshot's model is missing or damaged - reinstall"
    static let thresholdsMissing = "Earshot's check settings are missing or damaged - reinstall"
    static let thresholdsOtherModel = "Thresholds were calibrated for another model"
    static let selfTestFailed = "The engine self-test failed, so audio stays off"
    static let stubEngine = "Test engine: plays the microphone straight through"
    static let voiceCheckReady = "Voice check ready"
    static let voiceCheckUnavailable = "Voice check unavailable - Earshot still works, but picks people less reliably"
    static let voiceCheckPending = "Voice check not run yet"

    /// "Profile expires in 5 days" (30-ios 2.3, read once at launch).
    static func profileExpires(days: Int) -> String { "Profile expires in \(days) days" }
    static let noProfile = "No profile (simulator or unsigned build)"

    // MARK: Listening
    static let start = "Start"
    static let stop = "Stop"
    static let resume = "Resume"
    static let pause = "Pause"
    static let notListening = "Not listening"
    static let listening = "Listening"
    static let muted = "Muted"
    static let paused = "Paused"
    static func status(_ text: String) -> String { "Status \(text)" }
    static let notReady = "Earshot is not ready yet"
    static let openToStart = "Open Earshot to start listening"
    static let startFirst = "Start listening first"
    static let connectAirPods = "Connect AirPods to listen"
    static let setupMuted = "Muted until your own voice is recorded"
    static let noBuiltInMic = "The iPhone microphone is not available"
    static let tapToResume = "Tap to resume Earshot"
    static let pausedByCall = "Earshot paused by a call. Tap to resume."
    static let interrupted = "Paused by another app or a call"
    static let savingPower = "Saving power - only the selected person is ready"
    static let tooHot = "iPhone is hot - Earshot paused"
    static let volumeHigh = "Start with the volume at about half"
    static let feedback = "Feedback detected - output lowered"

    static func rateUnsupported(_ rate: Double) -> String {
        "This audio route runs at \(Int(rate)) Hz, which Earshot does not support"
    }

    static func largeBuffer(_ frames: Int) -> String {
        "iOS gave Earshot a large audio buffer (\(frames) frames). Turn off Sound Recognition and Vocal Shortcuts for the lowest delay."
    }

    /// An AudioUnit call failed: `step` is the call name, `status` the OSStatus.
    static func audioSetupFailed(_ step: String, _ status: Int32) -> String {
        "Audio setup failed at \(step) (error \(status))"
    }

    // MARK: Delay (30-ios 7)
    static let delayUnknown = "Delay not known yet"
    static let measureDelay = "Measure delay"
    static let measuring = "Measuring - keep the phone against your right AirPod"
    static let measureHowTo = "Keep your AirPods in. Hold the bottom edge of the phone against the outside of your right AirPod for 5 seconds. Volume about half."
    static let probeWeak = "Couldn't hear the test sound. Turn the volume up a little and try again"
    static let probeAudioChanged = "The audio changed during the test. Try again"

    /// "Delay 59 ms (measured 3 min ago) - Earshot 11 + route 48"
    static func delayMeasured(total: Int, age: String, earshot: Int, route: Int) -> String {
        "Delay \(total) ms (measured \(age)) - Earshot \(earshot) + route \(route)"
    }

    /// "about 62 ms (estimate)"
    static func delayEstimate(_ ms: Int) -> String { "about \(ms) ms (estimate)" }

    static let justNow = "just now"
    static func minutesAgo(_ n: Int) -> String { "\(n) min ago" }
    static func hoursAgo(_ n: Int) -> String { "\(n) h ago" }

    // MARK: Enrollment (30-ios 8)
    static let addPerson = "Add person"
    static let addWhoJustSpoke = "Add who just spoke"
    static let holdTowardThem = "Hold the phone toward them while they talk"
    static let captureInterrupted = "Audio was interrupted - try again"
    static let done = "Done"
    static let cancel = "Cancel"
    static let tryAgain = "Try again"
    static let checking = "Checking"
    static func speechSeconds(_ s: Double) -> String { String(format: "%.1f s of speech", s) }
    static let keepPhoneOnThem = "Keep the phone on them a few more seconds"
    static let tooNoisy = "Too noisy - move the phone closer"
    static let moreThanOneVoice = "I heard more than one voice - try when only they are talking"
    static let soundedLikeYou = "That sounded like your voice"
    static let tooLoud = "Too loud - hold the phone a bit farther away"
    static let accepted = "Accepted"
    static let acceptedWillBeOK = "Accepted, will be OK"
    static let notChecked = "not checked"
    static func soundsLike(_ name: String) -> String { "Sounds like \(name) - update \(name)?" }
    static let update = "Update"
    static let keepBoth = "Keep both"
    static let notEnoughHistory = "Not enough recent audio yet - try again in a few seconds"
    static let isThisThem = "Is this them?"
    static let playbackNeedsAirPods = "Connect AirPods to hear the clip"
    static let checkFailed = "Couldn't check this voice - try again"
    static let saveFailed = "Couldn't save this person - try again"
    static let yes = "Yes"
    static let no = "No"

    // Check names on the result card (the 6 checks of contract 6)
    static let checkSpeech = "Speech"
    static let checkBackground = "Background noise"
    static let checkOneVoice = "One voice"
    static let checkNotYou = "Not your voice"
    static let checkDistinct = "Not someone already saved"
    static let checkClipping = "Not too loud"

    // MARK: Own voice (30-ios 8.1, onboarding step 5)
    static let yourVoice = "Your voice"
    static let yourVoiceWhy = "So Earshot never plays your own voice back to you."
    static let readPassageHowTo = "Read this out loud in a quiet place, with the phone held normally."
    static let recordMyVoice = "Record my voice"
    static let reRecordMyVoice = "Re-record my voice"
    static let notNow = "Not now"
    static let keepReading = "Keep reading a little longer"
    static let tooNoisyOwn = "Too noisy - try again somewhere quieter"
    static let ownVoiceSaved = "Your voice is saved"

    // MARK: People (30-ios 9)
    static let people = "People"
    static let you = "You"
    static let tonight = "Tonight"
    static let saved = "Saved"
    static let name = "Name"
    static func personN(_ n: Int) -> String { "Person \(n)" }
    /// The default name for a new person: "Person N" with the smallest N that no current person
    /// uses, so after a deletion a new person never gets a name someone else already has.
    static func nextPersonName(taken names: [String]) -> String {
        let used = Set(names)
        var n = 1
        while used.contains(personN(n)) { n += 1 }
        return personN(n)
    }
    static let delete = "Delete"
    static let rename = "Rename"
    static let deleteEveryone = "Delete everyone"
    static let deleteEveryoneConfirm = "Delete every saved and tonight person from this iPhone?"
    static let deleteMyVoice = "Delete my voice"
    static let deleteMyVoiceConfirm = "Delete your voice from this iPhone? Earshot stays muted until you record it again."
    static let gettingReadyPerson = "getting ready"
    static let needsReAdding = "needs re-adding"
    static let tonightOnly = "Tonight only"
    static let saveForLater = "Save"
    static let theyAgreed = "They agreed - save"
    static func consentQuestion(_ name: String) -> String {
        "Did \(name) agree to have their voice saved on this phone? Voiceprints count as biometric data."
    }
    static let noOneYet = "No one yet"

    // MARK: Errors shown instead of raw system text (I-3)
    static let startFailed = "Couldn't start audio - try again"
    static let captureStartFailed = "Couldn't start recording - try again"

    // MARK: Listen: modes, table, look (30-ios 10, 11, 13)
    static let modeFocus = "Focus"
    static let modeTable = "Table"
    static let modeEveryone = "Everyone"
    static let modePaused = "Paused"
    static func playingFocus(_ name: String) -> String { "Focus: \(name)" }
    static func playingTable(_ n: Int) -> String { n == 1 ? "Table: 1 person" : "Table: \(n) people" }
    /// Now Playing title, "Earshot - Focus: Maya" (30-ios 11).
    static func nowPlayingTitle(_ status: String) -> String { "Earshot - \(status)" }
    static let look = "Look"
    static let headTrackingPaused = "Head tracking paused"
    static let addToTable = "Add to table"
    static let removeFromTable = "Remove from table"
    static func talkingLabel(_ name: String) -> String { "\(name), talking" }
    static let voiceLevel = "Voice level"
    static func decibels(_ db: Float) -> String { String(format: "%+.0f dB", db) }
    static let roomSound = "Room sound"
    static let roomOff = "Off"
    static func roomLevel(_ db: Float) -> String { "\(Int(db)) dB" }
    static let holdToCompare = "Hold to compare"
    static let holdToCompareHint = "Plays the room without Earshot while you hold"

    // MARK: Delay prompts (30-ios 7 "When")
    static let measurePrompt = "New headphones connected - measure the delay?"
    static let recheckDelay = "Re-check the delay - it can drift"
    static let measureHowToB = "Turn off Automatic Ear Detection (Settings, Bluetooth, your AirPods), take out the right AirPod and hold its speaker to the bottom of the phone."
    static let tryMethodB = "Try with ear detection off"
    static let tapToMeasure = "Tap to measure the delay"

    // MARK: Settings (30-ios 13)
    static let settings = "Settings"
    static let muteWhileITalk = "Mute while I talk"
    static let activityGate = "Lower the sound between their words"
    static let voiceLeveling = "Voice leveling"
    static let alarmPassthrough = "Let alarm sounds through"
    static let alarmPassthroughNote = "A helper, not a safety device. Earshot may miss alarms."
    static let advanced = "Advanced"
    static let maxGain = "Maximum boost"
    static let outputTrim = "Output trim"
    static let about = "About"
    static let creditsLookOnce = "The pick-by-looking idea comes from UW's Look Once to Hear (Veluri et al., CHI 2024). No code from it is used."
    static let creditsWeSpeaker = "The voice check uses WeSpeaker ResNet34 trained on VoxCeleb. Its weights are granted for research use, so this build is for personal use only."
    static let licenses = "Licenses - WeSpeaker code under the Apache License 2.0; its VoxCeleb weights for research use only. Everything else is Earshot's own code on Apple frameworks."
    static let privacySummary = "Everything runs on this iPhone and nothing is uploaded. Each person you add is kept on this phone as a short voice clip and a voiceprint. Tonight people are deleted 12 hours after you stop listening."

    // MARK: Diagnostics test controls
    static let diagnostics = "Diagnostics"
    static let listenTest = "Listen without own voice (test)"
    static let listenTestNote = "For the pass-through test only: you will hear yourself, slightly late."
    static let testBeep = "Play one beep"
    static let forceSwap = "Force engine swap"

    // MARK: Safety (30-ios 12, I-4)
    static let alarmPassthroughBanner = "Alarm-like sound - letting the room through"
    static let mightBeFollowingSomeoneElse = "Might be following someone else - double-press or re-add"
    static let followingRightVoice = "Following the right voice"
    static let tableOffWhileSaving = "Table is off while saving power"

    // MARK: Onboarding (30-ios 13, I-4)
    static func stepOf(_ n: Int, _ total: Int) -> String { "Step \(n) of \(total)" }
    static let next = "Continue"
    static let skip = "Skip"
    static let back = "Back"
    static let finishSetup = "Start using Earshot"
    static let help = "Help"

    // Step 1 - what it is and is not, and credits
    static let whatItIsTitle = "What Earshot is"
    static let whatItIs = "Earshot brings one person forward and pushes the rest down. It is a listening tool, not a hearing aid. Over AirPods their voice reaches you late, about 0.06 s on AirPods Pro 2 and 0.1 s on Pro 1 (estimate until Earshot measures it), so keep Noise Cancellation on."
    static let whatItIsNotTitle = "What it is not"
    static let whatItIsNot = "Nothing is uploaded: everything runs on this iPhone, and the only audio Earshot keeps is the short voice clip of each person you add. It is not a safety device and can miss alarms. Over Bluetooth the voice always arrives a little late: Earshot itself adds about 11 ms, and Earshot offers to measure the full delay from their mouth to your ear on your AirPods each time you connect."
    static let ownVoiceAlwaysDown = "Your own voice is always turned down, because hearing yourself late is very distracting."
    static let creditsTitle = "Credits"

    // Step 2 - permissions
    static let permissionsTitle = "Permissions"
    static let permissionsWhy = "Earshot needs the microphone to hear the room. Motion lets you pick a person by facing them. Notifications let Earshot ask you to resume after a call."
    static let microphone = "Microphone"
    static let motion = "Motion"
    static let notifications = "Notifications"
    static let allow = "Allow"
    static let allowed = "Allowed"
    static let notAllowed = "Not allowed - you can change this in Settings"
    static let micNeeded = "Earshot cannot work without the microphone. Turn it on in Settings, Privacy & Security, Microphone."
    static let answerEachPermission = "Answer each request to continue."

    // Step 3 - which AirPods and the microphone tap check (30-ios 3.2, 3.4)
    static let whichAirPodsTitle = "Which AirPods"
    static let whichAirPodsQuestion = "Which AirPods are these?"
    static let airPodsPro1 = "AirPods Pro (1st generation)"
    static let airPodsPro2 = "AirPods Pro 2"
    static let connectAirPodsFirst = "Put your AirPods in and connect them to this iPhone first."
    static let pro2Tip = "AirPods Pro 2 block more of the room and add less delay, so they are better for long conversations."
    static let startMicrophone = "Start the microphone"
    static let tapTestHowTo = "With your AirPods in, tap the bottom edge of the phone a few times."
    static let tapTestPass = "The iPhone microphone is listening"
    static let tapTestFail = "That did not reach the iPhone microphone. Tap the bottom edge, near the speaker holes, and try again."
    static let tapTestRetry = "Check again"

    // Step 4 - AirPods checklist, each with a "how" sheet (D55)
    static let checklistTitle = "AirPods settings"
    static let checklistIntro = "These settings help Earshot. Some also change how your AirPods behave in other apps."
    static let how = "How"
    static let checklist: [(title: String, how: String)] = [
        ("Noise Cancellation on",
         "Open Control Center, touch and hold the volume slider, tap Noise Control and choose Noise Cancellation. Optional: in the Shortcuts app, add an automation \"When AirPods connect, Set Noise Control Mode to Noise Cancellation\"."),
        ("Conversation Awareness off",
         "With your AirPods connected, open Settings, tap your AirPods near the top and turn off Conversation Awareness. It turns Earshot down whenever you speak."),
        ("Personalized Volume off",
         "Settings, your AirPods, turn off Personalized Volume, so the volume does not change on its own."),
        ("Spatial Audio off for Earshot",
         "While Earshot is playing, open Control Center, touch and hold the volume slider, tap Spatial Audio and choose Off. iPhone remembers this for Earshot."),
        ("Connect to This iPhone set to When Last Connected",
         "Settings, your AirPods, Connect to This iPhone, choose When Last Connected to This iPhone, so a Mac or iPad nearby does not take the AirPods away."),
        ("Sound Recognition off",
         "Settings, Accessibility, Sound Recognition, turn it off. While it is on, iOS gives Earshot a much larger audio buffer, which adds delay."),
        ("Reduce Loud Audio on",
         "Settings, Sounds & Haptics, Headphone Safety, turn on Reduce Loud Audio and set it to 85 dB."),
    ]

    // Step 5 - your voice (yourVoice and yourVoiceWhy above)
    static let ownVoiceRequired = "Earshot stays silent until your voice is recorded."

    // Step 6 - measure the delay
    static let measureTitle = "Measure the delay"
    static let measureWhy = "Why it matters - the voice reaches you a little after the real sound, so Noise Cancellation has to hide the real one. Knowing the delay on your AirPods tells you what to expect."
    static func earshotShare(_ ms: Int) -> String {
        "Earshot itself adds \(ms) ms; the rest is the iPhone audio path, Bluetooth and the AirPods."
    }

    // Step 7 - first listen
    static let firstListenTitle = "First listen"
    static let firstListenHowTo = "Earshot plays everyone except you for 10 seconds. Hold the button to hear the room without Earshot and compare."
    static let playTenSeconds = "Play 10 seconds"
    static func secondsLeft(_ n: Int) -> String { "\(n) s left" }
    static let firstListenDone = "That was Everyone mode. Add a person to hear only them."

    // Step 8 - controls (also in Help)
    static let controlsTitle = "Controls"
    static let stemTitle = "AirPods stem"
    static let stemSingle = "Press once - pause or resume"
    static let stemDouble = "Press twice - switch to the person you are facing. With nobody added yet, it records the person in front of you."
    static let stemTriple = "Press three times - Focus and Table, or Focus and Everyone"
    static let beepsTitle = "Beeps"
    static let beepSlot = "1, 2 or 3 beeps after a switch - the person in seat 1, 2 or 3"
    static let beepMode = "1, 2 or 3 beeps after a triple press - Focus, Table or Everyone"
    static let beepCapture = "When recording someone - 1 beep to start, 2 when they are added, 3 if it did not work"
    static let beepAlarm = "3 beeps and the room comes through - an alarm-like sound was heard"

    // Help (30-ios 13)
    static let delayMeaningTitle = "What the delay means"
    static let delayMeaning = "Sound takes a moment to reach you through Earshot. Earshot itself adds about 11 ms; Bluetooth and the AirPods add more, about 0.06 s in total on AirPods Pro 2 and 0.1 s on Pro 1 (estimate). Earshot offers to measure the full delay on your AirPods each time you connect and shows the last result with its age. That is why Noise Cancellation stays on and your own voice is always turned down."

    /// Every string above, with sample values for the templates, for CopyLintTests.
    static var allSamples: [String] {
        [appName, disclaimer, privacy, gettingReady, modelMissing, thresholdsMissing, thresholdsOtherModel,
         selfTestFailed, stubEngine, voiceCheckReady, voiceCheckUnavailable, voiceCheckPending,
         profileExpires(days: 5), noProfile, start, stop, resume, pause, notListening, listening, muted, paused,
         status(listening), notReady, openToStart, startFirst, connectAirPods, setupMuted, noBuiltInMic,
         tapToResume, pausedByCall, interrupted, savingPower, tooHot, volumeHigh, feedback,
         rateUnsupported(96_000), largeBuffer(1104), audioSetupFailed("eh_start", -50), delayUnknown,
         measureDelay, measuring, measureHowTo, probeWeak, probeAudioChanged,
         delayMeasured(total: 59, age: minutesAgo(3), earshot: 11, route: 48), delayEstimate(62), justNow,
         minutesAgo(3), hoursAgo(2),
         addPerson, addWhoJustSpoke, holdTowardThem, captureInterrupted, done, cancel, tryAgain, checking,
         speechSeconds(4.5), keepPhoneOnThem, tooNoisy, moreThanOneVoice, soundedLikeYou, tooLoud, accepted,
         acceptedWillBeOK, notChecked, soundsLike("Maya"), update, keepBoth, notEnoughHistory, isThisThem, yes, no,
         playbackNeedsAirPods, checkFailed, saveFailed,
         checkSpeech, checkBackground, checkOneVoice, checkNotYou, checkDistinct, checkClipping,
         yourVoice, yourVoiceWhy, readPassageHowTo, recordMyVoice, reRecordMyVoice, notNow, keepReading,
         tooNoisyOwn, ownVoiceSaved,
         people, you, tonight, saved, name, personN(3), delete, rename, deleteEveryone, deleteEveryoneConfirm,
         deleteMyVoice, deleteMyVoiceConfirm,
         gettingReadyPerson, needsReAdding, tonightOnly, saveForLater, theyAgreed, consentQuestion("Maya"),
         noOneYet, diagnostics, listenTest, listenTestNote, testBeep, forceSwap,
         startFailed, captureStartFailed, modeFocus, modeTable, modeEveryone, modePaused, playingFocus("Maya"),
         playingTable(1), playingTable(2), nowPlayingTitle(playingFocus("Maya")), look, headTrackingPaused,
         addToTable, removeFromTable, talkingLabel("Maya"), voiceLevel, decibels(-4), roomSound, roomOff,
         roomLevel(-15), holdToCompare, holdToCompareHint, measurePrompt, recheckDelay, measureHowToB, tryMethodB,
         tapToMeasure, settings, muteWhileITalk, activityGate, voiceLeveling, alarmPassthrough,
         alarmPassthroughNote, advanced, maxGain, outputTrim, about, creditsLookOnce, creditsWeSpeaker, licenses,
         privacySummary] + i4Samples + phase0Samples // I-1: Phase 0 strings live in Copy+Phase0.swift
    }

    /// The I-4 strings (safety, onboarding, Help), every template filled with sample values.
    static var i4Samples: [String] {
        [alarmPassthroughBanner, mightBeFollowingSomeoneElse, followingRightVoice, tableOffWhileSaving,
         stepOf(3, 8), next, skip, back, finishSetup, help,
         whatItIsTitle, whatItIs, whatItIsNotTitle, whatItIsNot, ownVoiceAlwaysDown, creditsTitle,
         permissionsTitle, permissionsWhy, microphone, motion, notifications, allow, allowed, notAllowed, micNeeded,
         answerEachPermission,
         whichAirPodsTitle, whichAirPodsQuestion, airPodsPro1, airPodsPro2, connectAirPodsFirst, pro2Tip,
         startMicrophone, tapTestHowTo, tapTestPass, tapTestFail, tapTestRetry,
         checklistTitle, checklistIntro, how, ownVoiceRequired,
         measureTitle, measureWhy, earshotShare(11),
         firstListenTitle, firstListenHowTo, playTenSeconds, secondsLeft(7), firstListenDone,
         controlsTitle, stemTitle, stemSingle, stemDouble, stemTriple, beepsTitle, beepSlot, beepMode, beepCapture,
         beepAlarm, delayMeaningTitle, delayMeaning]
            + checklist.flatMap { [$0.title, $0.how] }
    }
}
