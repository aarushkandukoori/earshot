// ContainerFile.swift - reads the `.esw` / `.est` container of contract 3.1 on the Swift side.
//
// The C engine has its own loader. Swift needs one only for test data it uses itself, mainly
// `wespk_selftest.est` (the WeSpeaker launch self-test, 30-ios 8.4).
//
// Layout (all little-endian):
//   offset 0   4 bytes   magic "ESW1" or "EST1"
//   offset 4   uint32    format_version = 1
//   offset 8   uint32    header_len (bytes of UTF-8 JSON that follow)
//   offset 12  uint32    reserved = 0
//   offset 16  JSON header
//   data_start = round_up(16 + header_len, 64): the arrays, each at a 64-aligned offset
// The header's data_sha256 must equal the SHA-256 of everything from data_start to the end.

import CryptoKit
import Foundation

struct ContainerError: Error, CustomStringConvertible {
    let description: String
    init(_ text: String) { description = text }
}

struct ContainerFile {
    /// One entry of the header's `arrays` list.
    struct ArrayInfo {
        let name: String
        let dtype: String  // "f32", "u8" or "i32"
        let shape: [Int]
        let offset: Int    // relative to data_start
        let nbytes: Int
    }

    let magic: String
    let header: [String: Any]
    let arrays: [String: ArrayInfo]
    private let data: Data // everything from data_start to the end of the file

    /// Parses and verifies a container. `expectMagic` ("EST1" or "ESW1") is checked when given.
    init(data: Data, expectMagic: String? = nil) throws {
        let bytes = [UInt8](data) // a plain array, so indexes start at 0 whatever `data` is
        guard bytes.count >= 16 else { throw ContainerError("file shorter than the 16-byte preamble") }
        func u32(_ at: Int) -> Int {
            Int(bytes[at]) | Int(bytes[at + 1]) << 8 | Int(bytes[at + 2]) << 16 | Int(bytes[at + 3]) << 24
        }
        magic = String(decoding: bytes[0..<4], as: UTF8.self)
        guard magic == "ESW1" || magic == "EST1" else { throw ContainerError("magic") }
        if let expectMagic, magic != expectMagic { throw ContainerError("magic is \(magic), not \(expectMagic)") }
        guard u32(4) == 1 else { throw ContainerError("format_version") }
        let headerLen = u32(8)
        guard headerLen > 0, 16 + headerLen <= bytes.count else { throw ContainerError("header_len") }

        let json = try JSONSerialization.jsonObject(with: Data(bytes[16..<(16 + headerLen)]))
        guard let header = json as? [String: Any] else { throw ContainerError("header is not a JSON object") }
        self.header = header

        let start = (16 + headerLen + 63) / 64 * 64
        guard start <= bytes.count else { throw ContainerError("data_start beyond the end") }
        let body = Data(bytes[start...])
        guard let dataBytes = header["data_bytes"] as? Int, dataBytes == body.count else {
            throw ContainerError("data_bytes")
        }
        let digest = SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined()
        guard let expected = header["data_sha256"] as? String, expected == digest else {
            throw ContainerError("data_sha256 mismatch")
        }

        var arrays: [String: ArrayInfo] = [:]
        for entry in header["arrays"] as? [[String: Any]] ?? [] {
            guard let name = entry["name"] as? String, let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int], let offset = entry["offset"] as? Int,
                  let nbytes = entry["nbytes"] as? Int else { throw ContainerError("bad arrays entry") }
            let itemSize = dtype == "u8" ? 1 : 4
            guard ["f32", "u8", "i32"].contains(dtype) else { throw ContainerError("\(name): dtype \(dtype)") }
            guard nbytes == shape.reduce(1, *) * itemSize else { throw ContainerError("\(name): nbytes") }
            guard offset >= 0, offset % 64 == 0, offset + nbytes <= body.count else {
                throw ContainerError("\(name): offset")
            }
            arrays[name] = ArrayInfo(name: name, dtype: dtype, shape: shape, offset: offset, nbytes: nbytes)
        }
        self.arrays = arrays
        self.data = body
    }

    /// The f32 array `name` as Swift floats.
    func floats(_ name: String) throws -> [Float] {
        let raw = try rawBytes(name, dtype: "f32")
        return stride(from: 0, to: raw.count, by: 4).map { i in
            let bits = UInt32(raw[i]) | UInt32(raw[i + 1]) << 8 | UInt32(raw[i + 2]) << 16 | UInt32(raw[i + 3]) << 24
            return Float(bitPattern: bits)
        }
    }

    /// The u8 array `name`.
    func uint8s(_ name: String) throws -> [UInt8] { try rawBytes(name, dtype: "u8") }

    /// A number inside `meta` by a dotted path, for example "tolerances.cosine_min".
    func metaNumber(_ path: String) -> Double? {
        var node: Any? = header["meta"]
        for key in path.split(separator: ".") { node = (node as? [String: Any])?[String(key)] }
        return (node as? NSNumber)?.doubleValue
    }

    /// A string inside `meta` by a dotted path, for example "kind".
    func metaString(_ path: String) -> String? {
        var node: Any? = header["meta"]
        for key in path.split(separator: ".") { node = (node as? [String: Any])?[String(key)] }
        return node as? String
    }

    private func rawBytes(_ name: String, dtype: String) throws -> [UInt8] {
        guard let info = arrays[name] else { throw ContainerError("no array named \(name)") }
        guard info.dtype == dtype else { throw ContainerError("\(name) is \(info.dtype), not \(dtype)") }
        let from = data.startIndex + info.offset
        return [UInt8](data[from..<(from + info.nbytes)])
    }
}
