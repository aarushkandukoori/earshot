// CopyLintTests.swift - banned words and em dashes in user-facing strings (30-ios 12.1).

import XCTest
@testable import Earshot

final class CopyLintTests: XCTestCase {
    /// Regular expressions that must never match (case-insensitive). Word starts are anchored,
    /// so "treatment" and "cures" are caught but "secure" is not.
    private let banned = [#"\btreat"#, #"\bcure"#, "hearing loss", "medical", "real[ -]time", "under 20 ms",
                          "removes all", "hearing aid"]
    /// The one allowed use of "hearing aid".
    private let allowedDisclaimer = "a listening tool, not a hearing aid"

    private func problems(in text: String) -> [String] {
        var found: [String] = []
        if text.contains("\u{2014}") { found.append("em dash") }
        let cleaned = text.replacingOccurrences(of: allowedDisclaimer, with: "", options: .caseInsensitive)
        for pattern in banned where cleaned.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil {
            found.append(pattern)
        }
        return found
    }

    func testEveryCopyStringIsClean() {
        XCTAssertFalse(Copy.allSamples.isEmpty)
        for s in Copy.allSamples {
            XCTAssertEqual(problems(in: s), [], "bad wording in: \(s)")
        }
    }

    /// The default "Person N" takes the smallest free number, so a deletion never leads to two
    /// people with the same name (hands-free adds have no screen to rename on).
    func testDefaultPersonNamesNeverRepeat() {
        XCTAssertEqual(Copy.nextPersonName(taken: []), "Person 1")
        XCTAssertEqual(Copy.nextPersonName(taken: ["Person 1", "Person 2"]), "Person 3")
        XCTAssertEqual(Copy.nextPersonName(taken: ["Person 2"]), "Person 1", "Person 1 was deleted")
        XCTAssertEqual(Copy.nextPersonName(taken: ["Maya", "Person 1", "Person 3"]), "Person 2")
    }

    func testLinterCatchesBadWords() {
        XCTAssertFalse(problems(in: "Earshot works in real time").isEmpty)
        XCTAssertFalse(problems(in: "A treatment for noise").isEmpty)
        XCTAssertFalse(problems(in: "Like a hearing aid").isEmpty)
        XCTAssertFalse(problems(in: "Delay \u{2014} 59 ms").isEmpty)
        XCTAssertEqual(problems(in: "It is a listening tool, not a hearing aid."), [])
        XCTAssertEqual(problems(in: "Your voiceprints stay secure"), [])
    }

    /// Also scans the string lines of Copy.swift itself, so a string missing from `allSamples`
    /// is still checked. The simulator can read the source tree; a device run skips this part.
    /// I-1: Copy+Phase0.swift (the Phase 0 screen's part of Copy) is scanned the same way.
    func testCopySourceFile() throws {
        for name in ["Copy.swift", "Copy+Phase0.swift"] {
            let source = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Earshot/UI/\(name)")
            guard let text = try? String(contentsOf: source, encoding: .utf8) else {
                throw XCTSkip("\(name) source not readable from this test host")
            }
            XCTAssertFalse(text.contains("\u{2014}"), "\(name) contains an em dash")
            for line in text.split(separator: "\n") {
                let code = Self.stripLineComment(String(line))
                guard code.contains("\"") else { continue }
                XCTAssertEqual(problems(in: code), [], "bad wording in \(name): \(line)")
            }
        }
    }

    /// Removes a trailing `// comment` from one line of Swift, but only when the `//` is outside a
    /// string literal, so `"http://x"` survives and `"Real time" // note` is still linted.
    static func stripLineComment(_ line: String) -> String {
        var inString = false
        var escaped = false
        var previous: Character? = nil
        var index = line.startIndex
        while index < line.endIndex {
            let c = line[index]
            if inString {
                if escaped { escaped = false }
                else if c == "\\" { escaped = true }
                else if c == "\"" { inString = false }
            } else if c == "\"" {
                inString = true
            } else if c == "/" && previous == "/" {
                return String(line[..<line.index(before: index)]) // cut before the first slash
            }
            previous = inString ? nil : c
            index = line.index(after: index)
        }
        return line
    }

    /// passage.txt is user-facing too (the own-voice reading text).
    func testPassageIsClean() throws {
        guard let url = Bundle.main.url(forResource: "passage", withExtension: "txt") else {
            XCTFail("passage.txt missing from the app bundle")
            return
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertEqual(problems(in: text), [])
        XCTAssertGreaterThan(text.split(separator: " ").count, 60, "enough to read for 20 s")
    }

    // MARK: - I-4 strings (safety, onboarding, Help)

    func testI4StringsAreInTheLintedList() {
        let all = Set(Copy.allSamples)
        for s in [Copy.whatItIs, Copy.whatItIsNot, Copy.delayMeaning, Copy.alarmPassthroughBanner,
                  Copy.mightBeFollowingSomeoneElse, Copy.pro2Tip, Copy.earshotShare(11)] {
            XCTAssertTrue(all.contains(s), "not in allSamples: \(s)")
        }
        for item in Copy.checklist {
            XCTAssertTrue(all.contains(item.title) && all.contains(item.how), item.title)
        }
    }

    /// Honest delay wording: any number of milliseconds under 20 may only describe Earshot's own
    /// share ("Earshot itself adds ..."), never a total over Bluetooth (D03).
    func testNoStringClaimsASmallTotalDelay() throws {
        let ms = try NSRegularExpression(pattern: #"(\d+(?:\.\d+)?)\s?ms\b"#)
        for s in Copy.allSamples {
            for m in ms.matches(in: s, range: NSRange(s.startIndex..., in: s)) {
                guard let r = Range(m.range(at: 1), in: s), let value = Double(s[r]), value < 20 else { continue }
                XCTAssertTrue(s.contains("Earshot itself adds"), "a sub-20 ms number that is not Earshot's share: \(s)")
            }
        }
    }

    func testOnboardingSaysWhatItIsAndIsNot() {
        XCTAssertTrue(Copy.whatItIs.contains("a listening tool, not a hearing aid"))
        XCTAssertTrue(Copy.whatItIs.contains("about 0.06 s on AirPods Pro 2 and 0.1 s on Pro 1"))
        XCTAssertTrue(Copy.whatItIs.contains("estimate until Earshot measures it"))
        XCTAssertTrue(Copy.whatItIs.contains("keep Noise Cancellation on"))
        XCTAssertTrue(Copy.whatItIsNot.contains("not a safety device"))
        XCTAssertTrue(Copy.whatItIsNot.contains("Earshot itself adds about 11 ms"))
        XCTAssertTrue(Copy.whatItIsNot.contains("offers to measure the full delay"))
        XCTAssertFalse(Copy.whatItIsNot.contains("is measured on your AirPods"), "the measurement is an offer (DECISIONS 2026-11-08)")
        XCTAssertTrue(Copy.delayMeaning.contains("offers to measure the full delay"))
        XCTAssertTrue(Copy.alarmPassthroughNote.contains("not a safety device"))
    }

    func testCredits() {
        XCTAssertTrue(Copy.creditsLookOnce.contains("Look Once to Hear"))
        XCTAssertTrue(Copy.creditsLookOnce.contains("CHI 2024"))
        XCTAssertTrue(Copy.creditsLookOnce.contains("No code"))
        XCTAssertTrue(Copy.creditsWeSpeaker.contains("WeSpeaker"))
        XCTAssertTrue(Copy.creditsWeSpeaker.contains("research"))
        XCTAssertTrue(Copy.creditsWeSpeaker.contains("personal use only"))
    }

    /// The seven AirPods settings of 30-ios 13 step 4, in order, each with its own "how" text.
    func testAirPodsChecklist() {
        let expected = ["Noise Cancellation on", "Conversation Awareness off", "Personalized Volume off",
                        "Spatial Audio off for Earshot", "When Last Connected", "Sound Recognition off",
                        "Reduce Loud Audio on"]
        XCTAssertEqual(Copy.checklist.count, expected.count)
        for (item, want) in zip(Copy.checklist, expected) {
            XCTAssertTrue(item.title.contains(want), "\(item.title) should say \(want)")
            XCTAssertGreaterThan(item.how.count, 40, "a real how-to for \(item.title)")
        }
        XCTAssertTrue(Copy.checklist[6].how.contains("85 dB"), "Reduce Loud Audio at 85 dB (30-ios 12.1)")
    }

    func testProfileExpiryWording() {
        let now = Date()
        XCTAssertEqual(ProvisioningProfile.expiryText(now.addingTimeInterval(5.5 * 86_400), now: now),
                       "Profile expires in 5 days")
        XCTAssertEqual(ProvisioningProfile.expiryText(nil, now: now), Copy.noProfile)
        XCTAssertEqual(ProvisioningProfile.expiryText(now.addingTimeInterval(-86_400), now: now),
                       "Profile expires in 0 days")
    }

    func testCommentStripperKeepsStringsAndDropsComments() {
        XCTAssertEqual(Self.stripLineComment(#"let a = "Real time" // note"#), #"let a = "Real time" "#)
        XCTAssertEqual(Self.stripLineComment(#"let u = "http://x""#), #"let u = "http://x""#)
        XCTAssertEqual(Self.stripLineComment("// only a comment"), "")
        XCTAssertEqual(Self.stripLineComment(#"let q = "say \"hi\" // still text""#),
                       #"let q = "say \"hi\" // still text""#)
        // A string followed by a comment is linted now (the old test skipped such lines).
        XCTAssertFalse(problems(in: Self.stripLineComment(#"static let x = "works in real time" // c"#)).isEmpty)
    }
}
