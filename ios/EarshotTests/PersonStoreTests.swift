// PersonStoreTests.swift - person.json, enroll.wav, kinds, expiry, consent, privacy flags and the
// launch rebuild (contract 5.1, 30-ios 9, 15.1). Every test uses its own temporary People folder.

import XCTest
@testable import Earshot

@MainActor
final class PersonStoreTests: XCTestCase {
    private var root: URL!
    private var model: EarshotModel!
    private let clip = TestSupport.speechLike(seconds: 6, from: 0.5, to: 5.5) // 96,000 samples

    override func setUpWithError() throws {
        root = TestSupport.tempFolder().appendingPathComponent("People")
        model = try TestSupport.model()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root.deletingLastPathComponent())
    }

    private func store() -> PeopleStore { PeopleStore(files: PersonFiles(root: root), prepareLogs: false) }

    private func makePerson(_ name: String = "Maya", kind: PersonKind = .tonight, now: Date = Date(),
                            mask: [UInt8]? = nil) -> Person {
        Person(name: name, kind: kind, created: now,
               expires: kind == .tonight ? now.addingTimeInterval(Person.tonightLifetime) : nil,
               yawDeg: kind == .own ? nil : 12.5, modelHash: model.modelHash, eWespk: TestSupport.basis(3),
               mask: mask ?? model.vad(clip), quality: Quality(speechS: 5.12, snrDB: 31.5, consistency: 0.8, ownCos: 0.1))
    }

    private func voiceprint(_ p: Person) throws -> Voiceprint {
        try Voiceprint(model: model, pcm16k: clip, mask: p.mask, role: p.kind == .own ? .own : .target, weSpeaker: p.eWespk)
    }

    // MARK: - person.json

    func testPersonJSONHasExactlyTheContractKeys() throws {
        var p = makePerson()
        p.yawDeg = nil
        p.eWespk = nil
        p.expires = nil
        p.quality = Quality(speechS: 4.5)
        let json = try JSONSerialization.jsonObject(with: Person.jsonEncoder().encode(p)) as! [String: Any]
        XCTAssertEqual(Set(json.keys), ["id", "name", "kind", "created", "updated", "expires", "yaw_deg", "model_hash",
                                        "e_wespk", "mask", "quality"])
        XCTAssertTrue(json["expires"] is NSNull, "missing values are written as null, never left out")
        XCTAssertTrue(json["yaw_deg"] is NSNull)
        XCTAssertTrue(json["e_wespk"] is NSNull)
        let quality = json["quality"] as! [String: Any]
        XCTAssertEqual(Set(quality.keys), ["speech_s", "snr_db", "consistency", "own_cos"])
        XCTAssertTrue(quality["own_cos"] is NSNull)
        XCTAssertEqual(json["kind"] as? String, "tonight")
        XCTAssertEqual(json["mask"] as? String, Data(p.mask!).base64EncodedString(), "mask = base64 of the uint8 mask")
        XCTAssertNotNil(ISO8601DateFormatter().date(from: json["created"] as! String), "ISO 8601 dates")
    }

    func testRoundTripWithMaskAndExpiry() throws {
        let p = makePerson(now: Date(timeIntervalSince1970: 1_800_000_000.75))
        let back = try Person.jsonDecoder().decode(Person.self, from: Person.jsonEncoder().encode(p))
        XCTAssertEqual(back, p)
        XCTAssertEqual(back.mask, model.vad(clip))
        XCTAssertEqual(back.expires, Date(timeIntervalSince1970: 1_800_000_000 + 12 * 3600), "whole seconds, + 12 h")
        XCTAssertEqual(back.eWespk, TestSupport.basis(3))
    }

    func testFileWithoutMaskLoadsWithNilMask() throws {
        var json = try JSONSerialization.jsonObject(with: Person.jsonEncoder().encode(makePerson())) as! [String: Any]
        json.removeValue(forKey: "mask")
        let back = try Person.jsonDecoder().decode(Person.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(back.mask)
    }

    // MARK: - Files, consent, privacy

    func testAddWritesBothFilesWithBackupExclusion() async throws {
        let s = store()
        let p = makePerson()
        try s.add(p, clip: clip, voiceprint: try voiceprint(p))
        await s.flush()
        let folder = root.appendingPathComponent(p.id.uuidString)
        XCTAssertTrue(FileManager.default.fileExists(atPath: folder.appendingPathComponent("person.json").path))
        let wav = try ClipTools.decodeWAV(Data(contentsOf: folder.appendingPathComponent("enroll.wav")))
        XCTAssertEqual(wav.count, clip.count)
        XCTAssertTrue(PersonFiles.isExcludedFromBackup(root), "People/ never reaches an iCloud backup")
        let loaded = PersonFiles(root: root).loadAll()
        XCTAssertEqual(loaded.count, 1)
        XCTAssertEqual(loaded.first?.person, p)
        XCTAssertEqual(s.status[p.id], .ready)

        let logs = TestSupport.tempFolder().appendingPathComponent("logs")
        try PersonFiles(root: root).prepareRoot(logs: logs)
        XCTAssertTrue(PersonFiles.isExcludedFromBackup(logs), "Documents/logs/ is excluded too")
    }

    func testSavedNeedsConsent() throws {
        let s = store()
        let saved = makePerson("Sam", kind: .saved)
        XCTAssertThrowsError(try s.add(saved, clip: clip, voiceprint: nil)) {
            XCTAssertEqual($0 as? PeopleError, .consentRequired)
        }
        XCTAssertTrue(s.people.isEmpty)
        XCTAssertNoThrow(try s.add(saved, clip: clip, voiceprint: nil, consent: true))

        let tonight = makePerson("Maya")
        try s.add(tonight, clip: clip, voiceprint: nil)
        var promoted = tonight
        promoted.kind = .saved
        XCTAssertThrowsError(try s.save(promoted), "save(_:) cannot skip the consent prompt")
        XCTAssertThrowsError(try s.makeSaved(tonight, consent: false))
        try s.makeSaved(tonight, consent: true)
        XCTAssertEqual(s.person(tonight.id)?.kind, .saved)
        XCTAssertNil(s.person(tonight.id)?.expires, "saved people never expire")
    }

    func testRenameTrimsAndIsWrittenToDisk() async throws {
        let s = store()
        let maya = makePerson("Maya")
        try s.add(maya, clip: clip, voiceprint: nil)
        try s.rename(maya.id, to: "  Maya R  ")
        XCTAssertEqual(s.person(maya.id)?.name, "Maya R")
        try s.rename(maya.id, to: "   ") // an empty name keeps the old one
        XCTAssertEqual(s.person(maya.id)?.name, "Maya R")
        XCTAssertEqual(s.person(maya.id)?.kind, .tonight, "rename never changes the kind")
        XCTAssertThrowsError(try s.rename(UUID(), to: "Nobody"))
        await s.flush()
        let onDisk = PersonFiles(root: root).loadAll().compactMap(\.person)
        XCTAssertEqual(onDisk.first { $0.id == maya.id }?.name, "Maya R")
    }

    func testOwnVoiceIsReplacedAndReported() async throws {
        let s = store()
        var reported: [Bool] = []
        s.onOwnVoiceChanged = { reported.append($0 != nil) }
        let a = makePerson("You", kind: .own)
        let b = makePerson("You", kind: .own)
        try s.add(a, clip: clip, voiceprint: try voiceprint(a))
        try s.add(b, clip: clip, voiceprint: try voiceprint(b))
        await s.flush()
        XCTAssertEqual(s.own?.id, b.id)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(a.id.uuidString).path),
                       "only one own voice is kept")
        s.delete(b)
        XCTAssertNil(s.own)
        XCTAssertEqual(reported, [true, true, false])
    }

    func testDeleteRemovesTheFolderAtOnce() async throws {
        let s = store()
        var deleted: [String] = []
        s.onPersonDeleted = { deleted.append($0.name) }
        let p = makePerson()
        try s.add(p, clip: clip, voiceprint: try voiceprint(p))
        await s.flush()
        s.delete(p)
        await s.flush()
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(p.id.uuidString).path))
        XCTAssertNil(s.voiceprints[p.id])
        XCTAssertEqual(deleted, ["Maya"])
    }

    // MARK: - Expiry (30-ios 9.2)

    func testTonightExpiryRefreshStopAndPurge() async throws {
        let s = store()
        let t0 = Date(timeIntervalSince1970: 1_800_000_000)
        let maya = makePerson("Maya", now: t0)
        let sam = makePerson("Sam", kind: .saved, now: t0)
        try s.add(maya, clip: clip, voiceprint: nil)
        try s.add(sam, clip: clip, voiceprint: nil, consent: true)
        XCTAssertEqual(s.person(maya.id)?.expires, t0.addingTimeInterval(12 * 3600), "created: now + 12 h")

        s.refreshTonightExpiry(now: t0.addingTimeInterval(3600))           // a listening minute
        XCTAssertEqual(s.person(maya.id)?.expires, t0.addingTimeInterval(13 * 3600))
        s.sessionStopped(at: t0.addingTimeInterval(2 * 3600))              // Stop
        let stopExpiry = t0.addingTimeInterval(14 * 3600)
        XCTAssertEqual(s.person(maya.id)?.expires, stopExpiry)
        XCTAssertNil(s.person(sam.id)?.expires)

        // The new expiry was written to disk, so a killed app still purges on the next launch.
        await s.flush()
        let written = PersonFiles(root: root).loadAll().compactMap(\.person)
        XCTAssertEqual(written.first { $0.id == maya.id }?.expires, stopExpiry, "stop + 12 h read back from person.json")
        XCTAssertNil(written.first { $0.id == sam.id }?.expires, "saved people never get an expiry")

        s.purgeExpired(now: stopExpiry.addingTimeInterval(-1))
        XCTAssertNotNil(s.person(maya.id), "not yet")
        s.purgeExpired(now: stopExpiry)
        XCTAssertNil(s.person(maya.id), "expired 12 h after the session ended")
        XCTAssertNotNil(s.person(sam.id), "saved people stay")
        await s.flush()
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(maya.id.uuidString).path))

        // The purge removed Maya's folder; only Sam is left on disk.
        let onDisk = PersonFiles(root: root).loadAll().compactMap(\.person)
        XCTAssertEqual(onDisk.map(\.name), ["Sam"])
    }

    // MARK: - Launch rebuild (30-ios 9.1)

    func testLaunchRebuildUsesTheStoredMaskAndNoWeSpeaker() async throws {
        let writer = store()
        let own = makePerson("You", kind: .own)
        let maya = makePerson("Maya")
        try writer.add(own, clip: clip, voiceprint: nil)
        try writer.add(maya, clip: clip, voiceprint: nil)
        await writer.flush()

        let s = store()
        var ownReady = false
        s.onOwnVoiceChanged = { ownReady = $0 != nil }
        let encoder = FakeEncoder { _ in TestSupport.basis(0) }
        await s.loadAndRebuild(model: model, encoder: encoder, ownWindowCos: 0.6)
        let calls = await encoder.calls
        XCTAssertEqual(calls, 0, "a stored mask means no WeSpeaker call")
        XCTAssertEqual(s.status[maya.id], .ready)
        XCTAssertEqual(s.status[own.id], .ready)
        XCTAssertNotNil(s.voiceprints[maya.id])
        XCTAssertTrue(ownReady, "the own voiceprint reaches AudioController")
        XCTAssertEqual(s.person(maya.id)?.mask, maya.mask, "mask unchanged")
        XCTAssertTrue(s.loaded)
    }
}
