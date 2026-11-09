// Person.swift - one enrolled person as stored in person.json (contract 5.1, 30-ios 9.1).
//
// person.json keys, exactly (explicit CodingKeys, no automatic snake case):
//   {id, name, kind ("tonight" | "saved" | "own"), created, updated, expires (ISO 8601 or null),
//    yaw_deg (or null), model_hash, e_wespk [256] (or null), mask (base64 of the uint8 token-frame mask
//    passed to es_enroll), quality {speech_s, snr_db, consistency, own_cos}}
// Every key is always written; a missing value is written as null, never left out.
// The voiceprint itself is never stored: it is rebuilt from enroll.wav and `mask` at launch.

import Foundation

enum PersonKind: String, Codable, Sendable {
    case tonight, saved, own
}

struct Person: Codable, Identifiable, Equatable, Sendable {
    var id: UUID
    var name: String
    var kind: PersonKind
    var created: Date
    var updated: Date
    var expires: Date?        // tonight people only: 12 h after the session ends (30-ios 9.2)
    var yawDeg: Double?       // a hint from the last session; always loads as stale (contract 5.1)
    var modelHash: String     // model_hash of the weights in use when the person was saved
    var eWespk: [Float]?      // WeSpeaker vector, nil when the voice check was unavailable
    var mask: [UInt8]?        // nil only in an old file; the launch rebuild recomputes it
    var quality: Quality

    /// "Tonight" people live 12 hours after the last listening session.
    static let tonightLifetime: TimeInterval = 12 * 60 * 60

    enum CodingKeys: String, CodingKey {
        case id, name, kind, created, updated, expires
        case yawDeg = "yaw_deg", modelHash = "model_hash", eWespk = "e_wespk", mask, quality
    }

    init(id: UUID = UUID(), name: String, kind: PersonKind, created: Date, updated: Date? = nil, expires: Date? = nil,
         yawDeg: Double? = nil, modelHash: String, eWespk: [Float]?, mask: [UInt8]?, quality: Quality) {
        self.id = id
        self.name = name
        self.kind = kind
        self.created = created.wholeSeconds
        self.updated = (updated ?? created).wholeSeconds
        self.expires = expires?.wholeSeconds
        self.yawDeg = yawDeg
        self.modelHash = modelHash
        self.eWespk = eWespk
        self.mask = mask
        self.quality = quality
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        kind = try c.decode(PersonKind.self, forKey: .kind)
        created = try c.decode(Date.self, forKey: .created)
        updated = try c.decode(Date.self, forKey: .updated)
        expires = try c.decodeIfPresent(Date.self, forKey: .expires)
        yawDeg = try c.decodeIfPresent(Double.self, forKey: .yawDeg)
        modelHash = try c.decode(String.self, forKey: .modelHash)
        eWespk = try c.decodeIfPresent([Float].self, forKey: .eWespk)
        // A missing, null or unreadable mask becomes nil, so the launch rebuild recomputes it.
        if let text = try? c.decodeIfPresent(String.self, forKey: .mask), let data = Data(base64Encoded: text) {
            mask = [UInt8](data)
        } else {
            mask = nil
        }
        quality = try c.decode(Quality.self, forKey: .quality)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(kind, forKey: .kind)
        try c.encode(created, forKey: .created)
        try c.encode(updated, forKey: .updated)
        try c.encode(expires, forKey: .expires)       // null when nil (Optional encodes as null)
        try c.encode(yawDeg, forKey: .yawDeg)
        try c.encode(modelHash, forKey: .modelHash)
        try c.encode(eWespk, forKey: .eWespk)
        try c.encode(mask.map { Data($0).base64EncodedString() }, forKey: .mask)
        try c.encode(quality, forKey: .quality)
    }

    /// ISO 8601 dates, sorted keys (stable files that diff well).
    static func jsonEncoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }

    static func jsonDecoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }

    /// A new person from a passed (or warned) enrollment: the mask given to es_enroll, the WeSpeaker
    /// vector and the quality block. Tonight people expire 12 h from now (30-ios 9.2).
    static func enrolled(_ r: EnrollmentResult, name: String, kind: PersonKind, modelHash: String,
                         yawDeg: Double?, now: Date = Date()) -> Person {
        Person(name: name, kind: kind, created: now,
               expires: kind == .tonight ? now.addingTimeInterval(Person.tonightLifetime) : nil,
               yawDeg: yawDeg, modelHash: modelHash, eWespk: r.eWespk, mask: r.mask, quality: r.quality)
    }

    /// True for a tonight person whose time is up.
    func isExpired(at now: Date) -> Bool {
        kind == .tonight && (expires.map { $0 <= now } ?? false)
    }
}

/// The `quality` block of person.json: speech_s, snr_db (= sbr_db for targets), consistency
/// (= one_voice) and own_cos. The last three are null when not checked.
struct Quality: Codable, Equatable, Sendable {
    var speechS: Double
    var snrDB: Double?
    var consistency: Double?
    var ownCos: Double?

    enum CodingKeys: String, CodingKey {
        case speechS = "speech_s", snrDB = "snr_db", consistency, ownCos = "own_cos"
    }

    // Write every key, null included, so person.json always has exactly the contract keys.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(speechS, forKey: .speechS)
        try c.encode(snrDB, forKey: .snrDB)
        try c.encode(consistency, forKey: .consistency)
        try c.encode(ownCos, forKey: .ownCos)
    }
}

extension Date {
    /// The date without fractional seconds, because ISO 8601 in person.json stores whole seconds.
    var wholeSeconds: Date { Date(timeIntervalSince1970: timeIntervalSince1970.rounded(.down)) }
}
