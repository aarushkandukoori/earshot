// PeopleStore.swift - everyone Earshot knows, in memory and on disk (contract 5, 5.1; 30-ios 9).
//
// Main actor (the screens read it). File work goes to one serial utility queue, in order, so
// a save and a delete of the same person can never swap places and the main thread never waits
// on the disk. Changes show on screen at once; `flush()` waits until the disk has caught up.
//
// Kinds (30-ios 9.2):
//   tonight - the default; expires 12 h after the last listening session (refreshed every minute
//             while listening, set to stop time + 12 h by Stop); purged at launch, on every
//             foreground and every 10 minutes
//   saved   - needs the person's consent; never expires
//   own     - the user; exactly one; never expires; replaced by re-recording

import Foundation

enum PersonStatus: Equatable, Sendable {
    case gettingReady   // launch rebuild not done yet (the card says "getting ready")
    case ready          // voiceprint built, selectable
    case needsReAdding  // files failed to load or the rebuild failed
}

enum PeopleError: Error, Equatable {
    case consentRequired // a "saved" person needs the consent prompt first (30-ios 9.2)
    case notFound
}

@MainActor
final class PeopleStore: ObservableObject {
    @Published private(set) var people: [Person] = []         // tonight and saved, oldest first
    @Published private(set) var own: Person? = nil
    @Published private(set) var status: [UUID: PersonStatus] = [:]
    @Published private(set) var brokenFolders: [String] = []  // unreadable folders ("needs re-adding")
    @Published private(set) var lastError: String? = nil
    @Published private(set) var loaded = false                 // the launch load has run
    private(set) var voiceprints: [UUID: Voiceprint] = [:]

    /// Called with the new own voiceprint (or nil when it is deleted), for AudioController.setOwnVoice.
    var onOwnVoiceChanged: ((Voiceprint?) -> Void)?
    /// Called after any delete, for the table (I-3 fades the slot to 0 and empties it).
    var onPersonDeleted: ((Person) -> Void)?

    let files: PersonFiles
    private let queue = DispatchQueue(label: "earshot.people.files", qos: .utility)
    private var timers: Task<Void, Never>?

    init(files: PersonFiles = PersonFiles(root: PersonFiles.defaultRoot), prepareLogs: Bool = true) {
        self.files = files
        let f = files
        onQueue { try f.prepareRoot(logs: prepareLogs ? PersonFiles.logsFolder : nil) }
    }

    /// Own voice first, then everyone else (what enrollment compares against).
    var everyone: [Person] { (own.map { [$0] } ?? []) + people }

    func person(_ id: UUID) -> Person? { everyone.first { $0.id == id } }

    // MARK: - The file queue

    /// Runs file work on the serial utility queue; an error is shown in Diagnostics, never thrown.
    private func onQueue(_ work: @escaping @Sendable () throws -> Void) {
        queue.async { [weak self] in
            do { try work() } catch {
                let text = "\(error)"
                Task { @MainActor in self?.lastError = text }
            }
        }
    }

    /// Runs file work on the queue and returns its result to the caller.
    private func onQueueResult<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { cont in
            queue.async { cont.resume(with: Result { try work() }) }
        }
    }

    /// Waits until every file write queued so far has finished.
    func flush() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in queue.async { cont.resume() } }
    }

    // MARK: - Add, save, delete (contract 5: save(_:), delete(_:), purgeExpired())

    /// Adds a newly enrolled person with their clip (written as enroll.wav). A "saved" person needs
    /// `consent`; a new own voice replaces the old one.
    func add(_ person: Person, clip: [Float], voiceprint: Voiceprint?, consent: Bool = false) throws {
        if person.kind == .saved && !consent { throw PeopleError.consentRequired }
        if person.kind == .own {
            if let old = own, old.id != person.id { removeFolder(old.id) }
            own = person
            onOwnVoiceChanged?(voiceprint)
        } else {
            people.removeAll { $0.id == person.id }
            people.append(person)
            people.sort { $0.created < $1.created }
        }
        voiceprints[person.id] = voiceprint
        status[person.id] = voiceprint == nil ? .gettingReady : .ready
        let f = files
        let wav = ClipTools.encodeWAV(clip)
        onQueue { try f.write(person, wav: wav) }
    }

    /// Saves changed details of a known person (name, expiry, direction). Turning someone into a
    /// "saved" person must go through makeSaved(_:consent:), so this refuses that change.
    func save(_ person: Person) throws {
        guard let old = self.person(person.id) else { throw PeopleError.notFound }
        if person.kind == .saved && old.kind != .saved { throw PeopleError.consentRequired }
        var p = person
        p.updated = Date().wholeSeconds
        if p.kind == .own { own = p } else if let i = people.firstIndex(where: { $0.id == p.id }) { people[i] = p }
        let f = files
        let snapshot = p // a constant copy for the file queue
        onQueue { try f.write(snapshot, wav: nil) }
    }

    /// Rename in People (30-ios 13). Spaces around the name are trimmed; an empty name keeps the
    /// old one. Works on the stored copy, so an expiry refreshed meanwhile is not overwritten.
    func rename(_ id: UUID, to newName: String) throws {
        guard var p = person(id) else { throw PeopleError.notFound }
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != p.name else { return }
        p.name = trimmed
        try save(p)
    }

    /// "She agreed - save": a tonight person becomes saved (no expiry) only with consent.
    func makeSaved(_ person: Person, consent: Bool) throws {
        guard consent else { throw PeopleError.consentRequired }
        guard var p = self.person(person.id), p.kind != .own else { throw PeopleError.notFound }
        p.kind = .saved
        p.expires = nil
        p.updated = Date().wholeSeconds
        if let i = people.firstIndex(where: { $0.id == p.id }) { people[i] = p }
        let f = files
        let snapshot = p // a constant copy for the file queue
        onQueue { try f.write(snapshot, wav: nil) }
    }

    /// Deletes the person and their folder at once.
    func delete(_ person: Person) {
        if own?.id == person.id {
            own = nil
            onOwnVoiceChanged?(nil)
        }
        people.removeAll { $0.id == person.id }
        removeFolder(person.id)
        onPersonDeleted?(person)
    }

    /// "Delete everyone" in People: every tonight and saved person (the own voice stays; it has its
    /// own Delete on the "You" row and is re-recorded from People or Settings).
    func deleteEveryone() {
        for p in people { delete(p) }
    }

    private func removeFolder(_ id: UUID) {
        voiceprints[id] = nil
        status[id] = nil
        let f = files
        onQueue { try f.delete(id) }
    }

    // MARK: - Expiry (30-ios 9.2)

    /// Removes every tonight person whose `expires` has passed.
    func purgeExpired(now: Date = Date()) {
        for p in people where p.isExpired(at: now) { delete(p) }
    }

    /// While listening, once a minute: every tonight person now expires 12 h from now.
    func refreshTonightExpiry(now: Date = Date()) {
        setTonightExpiry(now.addingTimeInterval(Person.tonightLifetime))
    }

    /// AudioController.stop() hook: tonight people expire 12 h after the session ended.
    func sessionStopped(at time: Date) {
        setTonightExpiry(time.addingTimeInterval(Person.tonightLifetime))
    }

    private func setTonightExpiry(_ expires: Date) {
        let f = files
        for i in people.indices where people[i].kind == .tonight {
            people[i].expires = expires.wholeSeconds
            let p = people[i]
            onQueue { try f.write(p, wav: nil) }
        }
    }

    /// The minute refresh while listening and the 10-minute purge. Call once at launch.
    func startTimers(isListening: @escaping @MainActor () -> Bool) {
        timers?.cancel()
        timers = Task { @MainActor [weak self] in
            var minutes = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 60_000_000_000)
                guard let self else { return }
                minutes += 1
                if isListening() { self.refreshTonightExpiry() }
                if minutes % 10 == 0 { self.purgeExpired() }
            }
        }
    }
}

// MARK: - Launch: load, purge, rebuild one at a time (30-ios 9.1, 9.4 step 5)

extension PeopleStore {
    /// What one rebuild produced: the voiceprint, and a recomputed mask when the file had none.
    struct Rebuilt: @unchecked Sendable {
        let voiceprint: Voiceprint
        let newMask: [UInt8]?
    }

    /// Reads every folder, purges expired tonight people, then rebuilds each voiceprint from enroll.wav
    /// and the stored mask, own voice first, one person at a time, off the main thread. No WeSpeaker
    /// call happens unless a file has no mask (then 8.3 step 6 runs with the stored own vector).
    /// `encoder` is nil when the WeSpeaker self-test failed; `appActive` is the app state at launch.
    func loadAndRebuild(model: EarshotModel, encoder: SpeakerEmbedding?, ownWindowCos: Double?,
                        appActive: Bool = true, now: Date = Date()) async {
        let f = files
        let found = (try? await onQueueResult { f.loadAll() }) ?? []
        var clips: [UUID: [Float]] = [:]
        var broken: [String] = []
        var loadedOwn: Person? = nil
        var others: [Person] = []
        for item in found {
            guard let p = item.person, let clip = item.clip, p.id.uuidString == item.folderName else {
                broken.append(item.folderName) // shown as "needs re-adding" and skipped
                continue
            }
            clips[p.id] = clip
            if p.kind != .own {
                others.append(p)
            } else if let o = loadedOwn, o.updated >= p.updated {
                onQueue { try f.delete(p.id) }  // a second own folder: keep only the newest
            } else {
                if let o = loadedOwn { onQueue { try f.delete(o.id) } }
                loadedOwn = p
            }
        }
        own = loadedOwn
        people = others.sorted { $0.created < $1.created }
        brokenFolders = broken
        for p in everyone { status[p.id] = .gettingReady }
        loaded = true
        purgeExpired(now: now)

        for p in everyone { // own first: its vector is needed to recompute a missing target mask
            guard let clip = clips[p.id] else { continue }
            do {
                let r = try await Self.rebuild(p, clip: clip, model: model, encoder: encoder, ownE: own?.eWespk,
                                               ownWindowCos: ownWindowCos, appActive: appActive)
                guard var current = person(p.id) else { continue } // deleted while it was rebuilding
                voiceprints[p.id] = r.voiceprint
                status[p.id] = .ready
                if let mask = r.newMask {
                    current.mask = mask
                    try? save(current) // written once, so the next launch needs no recompute
                }
                if p.kind == .own { onOwnVoiceChanged?(r.voiceprint) }
            } catch {
                status[p.id] = .needsReAdding
                lastError = "\(p.name): \(error)"
            }
        }
    }

    /// Builds one voiceprint. Runs off the main actor (a nonisolated async function), so es_enroll
    /// and any WeSpeaker work never block the screen.
    nonisolated static func rebuild(_ p: Person, clip: [Float], model: EarshotModel, encoder: SpeakerEmbedding?,
                                    ownE: [Float]?, ownWindowCos: Double?, appActive: Bool) async throws -> Rebuilt {
        let frames = Int(es_enroll_num_frames(Int32(clip.count)))
        if let mask = p.mask, mask.count == frames {
            let vp = try Voiceprint(model: model, pcm16k: clip, mask: mask, role: p.kind == .own ? .own : .target,
                                    weSpeaker: p.eWespk)
            return Rebuilt(voiceprint: vp, newMask: nil)
        }
        // No usable mask: 30-ios 8.3 step 6 again (es_vad(clip), minus own-voice windows when possible).
        var mask = model.vad(clip)
        if p.kind != .own, let encoder, let ownE, let threshold = ownWindowCos {
            let wins = ClipTools.windows(clipLen: clip.count)
            var cosOwn: [Double] = []
            for w in wins {
                let e = try await encoder.embed(Array(clip[w.start..<(w.start + w.length)]), appActive: appActive)
                cosOwn.append(ClipTools.cosine(e, ownE))
            }
            mask = ClipTools.ownWindowMask(mask, windows: wins, windowCosOwn: cosOwn, threshold: threshold)
        }
        let vp = try Voiceprint(model: model, pcm16k: clip, mask: mask, role: p.kind == .own ? .own : .target,
                                weSpeaker: p.eWespk)
        return Rebuilt(voiceprint: vp, newMask: mask)
    }
}
