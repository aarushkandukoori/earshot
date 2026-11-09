// PersonFiles.swift - the files behind PeopleStore (contract 5.1, 30-ios 9.1 and 9.3).
//
//   Application Support/People/<uuid>/person.json   who it is (Person.swift)
//   Application Support/People/<uuid>/enroll.wav    16 kHz mono 16-bit, at most 10 s
//
// Privacy rules:
// - Files are written `.atomic` with `.completeFileProtectionUntilFirstUserAuthentication`, so a
//   locked phone (after its first unlock) can still read them for hands-free work.
// - The People/ and Documents/logs/ folders are excluded from backup, so voiceprints never reach iCloud.
// - enroll.wav is the only audio Earshot ever writes to disk.
//
// Every method here is synchronous plain file work. PeopleStore runs them on its serial utility
// queue, never on the main thread and never on the audio thread.

import Foundation

/// One folder as found on disk at launch.
struct LoadedPerson: Sendable {
    let folderName: String
    let person: Person?   // nil when person.json is missing or unreadable
    let clip: [Float]?    // nil when enroll.wav is missing or unreadable
    let error: String?
}

struct PersonFiles: Sendable {
    let root: URL // .../People

    static var defaultRoot: URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return support.appendingPathComponent("People", isDirectory: true)
    }

    static var logsFolder: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("logs", isDirectory: true)
    }

    private static let protection: [FileAttributeKey: Any] = [
        .protectionKey: FileProtectionType.completeUntilFirstUserAuthentication
    ]

    func folder(_ id: UUID) -> URL { root.appendingPathComponent(id.uuidString, isDirectory: true) }

    /// Creates People/ (and Documents/logs/) and excludes both from backup (30-ios 9.1).
    func prepareRoot(logs: URL? = PersonFiles.logsFolder) throws {
        for dir in [root] + (logs.map { [$0] } ?? []) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true,
                                                    attributes: Self.protection)
            try Self.excludeFromBackup(dir)
        }
    }

    /// Writes person.json, and enroll.wav when `wav` is given.
    func write(_ person: Person, wav: Data?) throws {
        let dir = folder(person.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: Self.protection)
        let options: Data.WritingOptions = [.atomic, .completeFileProtectionUntilFirstUserAuthentication]
        if let wav { try wav.write(to: dir.appendingPathComponent("enroll.wav"), options: options) }
        let json = try Person.jsonEncoder().encode(person)
        try json.write(to: dir.appendingPathComponent("person.json"), options: options)
    }

    /// Removes one person's folder at once (30-ios 9.2 "Delete removes the folder at once").
    func delete(_ id: UUID) throws {
        let dir = folder(id)
        if FileManager.default.fileExists(atPath: dir.path) { try FileManager.default.removeItem(at: dir) }
    }

    /// Reads every folder under People/. A folder that fails to load is still returned (with `error`)
    /// so the screen can show it as "needs re-adding".
    func loadAll() -> [LoadedPerson] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        return names.sorted().compactMap { name in
            guard UUID(uuidString: name) != nil else { return nil } // skip anything that is not ours
            let dir = root.appendingPathComponent(name, isDirectory: true)
            var person: Person? = nil
            var clip: [Float]? = nil
            var problems: [String] = []
            do {
                let data = try Data(contentsOf: dir.appendingPathComponent("person.json"))
                person = try Person.jsonDecoder().decode(Person.self, from: data)
            } catch { problems.append("person.json: \(error)") }
            do {
                clip = try ClipTools.decodeWAV(Data(contentsOf: dir.appendingPathComponent("enroll.wav")))
            } catch { problems.append("enroll.wav: \(error)") }
            return LoadedPerson(folderName: name, person: person, clip: clip,
                                error: problems.isEmpty ? nil : problems.joined(separator: "; "))
        }
    }

    // MARK: - Backup exclusion

    static func excludeFromBackup(_ url: URL) throws {
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var u = url
        try u.setResourceValues(values)
    }

    static func isExcludedFromBackup(_ url: URL) -> Bool {
        let fresh = URL(fileURLWithPath: url.path) // a new URL, so no cached resource values
        return (try? fresh.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup) == true
    }
}
