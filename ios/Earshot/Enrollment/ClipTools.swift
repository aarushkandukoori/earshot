// ClipTools.swift - the contract 6 enrollment check formulas as pure functions (30-ios 8.2),
// plus the WAV format of enroll.wav (30-ios 9.1).
//
// Every function here copies ml/earshot/quality.py step for step, and ClipToolsTests checks
// them against WP-ML's quality_vectors.json. "mask0" is es_vad of the FULL capture (one 0/1
// entry per token frame); "mask" is es_vad of the trimmed clip. x is 16 kHz audio in [-1, 1].
//
// Token frame f covers raw samples [512f - 448, 512f + 64) and its midpoint is 512f - 192
// (contract 2.4: hop 512 after 448 zeros of padding). Frame 0 therefore starts before the clip.
//
// Steps (contract 6 "Check formulas"):
//   2. span      first to last frame with mask0 = 1: [max(0, 512 f1 - 448), min(n, 512 f2 + 64))
//   3. 10 s cap  region longer than 10 s: the 160,000-sample window starting at region_start + 512 i
//                with the most mask0 midpoints, earliest on ties (DECISIONS L, 2026-11-07). The region
//                is the span for a target and the whole capture for own voice.
//   4. rounding  length rounded down to a multiple of 8,000 (0.5 s); under 64,000 (4 s) fails
//   5. windows   WeSpeaker windows [8000 w, 8000 w + 24000), w = 0 .. floor((len - 24000) / 8000)
//   6. sbr_db    10 log10((P_speech + 1e-12) / (P_other + 1e-12)) on the full capture
//   7. clip_pct  100 x count(|x| >= 0.98) / n on the full capture
//   9. mask      targets: es_vad(clip), then frames inside a window too close to the own voice set to 0

import Foundation

enum ClipTools {
    static let frameHop = 512        // token frame hop and length
    static let pad = 448             // left padding of the token framing
    static let clipMax = 160_000     // 10 s
    static let roundStep = 8_000     // 0.5 s
    static let minClip = 64_000      // 4 s
    static let winLen = 24_000       // 1.5 s WeSpeaker windows
    static let winHop = 8_000        // every 0.5 s
    static let clipLevel: Float = 0.98
    static let frameSeconds = 0.032  // speech seconds per kept frame
    static let powerFloor = 1e-12    // added to both sbr means

    // MARK: - Step 1: frame map

    /// Raw sample range [a, b) of token frame f in a capture of n samples (padding removed).
    static func frameSamples(_ f: Int, n: Int) -> Range<Int> {
        let a = max(0, frameHop * f - pad)
        let b = min(n, frameHop * f + 64)
        return a..<max(a, b)
    }

    /// Midpoint of token frame f in raw samples (negative for f = 0).
    static func frameMid(_ f: Int) -> Int { frameHop * f - 192 }

    /// Indexes of the frames whose mask entry is not 0.
    static func keptFrames(_ mask: [UInt8]) -> [Int] { mask.indices.filter { mask[$0] != 0 } }

    // MARK: - Steps 2 to 4: trimming

    /// Step 2: (start, end) samples from the first to the last frame with mask0 = 1, or nil if none.
    static func frameSpan(mask0: [UInt8], n: Int) -> (start: Int, end: Int)? {
        guard let f1 = mask0.firstIndex(where: { $0 != 0 }), let f2 = mask0.lastIndex(where: { $0 != 0 }) else {
            return nil
        }
        return (max(0, frameHop * f1 - pad), min(n, frameHop * f2 + 64))
    }

    /// Step 3: start of the `length`-sample window inside [regionStart, regionEnd) holding the most
    /// mask frame midpoints. Candidates start at regionStart + 512 i; a tie keeps the earliest.
    /// Only meaningful when the region is longer than `length` (else it returns regionStart).
    static func bestWindow(mask: [UInt8], regionStart: Int, regionEnd: Int, length: Int = clipMax) -> Int {
        let mids = keptFrames(mask).map(frameMid)
        var bestStart = regionStart
        var bestCount = -1
        var start = regionStart
        while start + length <= regionEnd {
            var count = 0
            for m in mids where m >= start && m < start + length { count += 1 }
            if count > bestCount { // strictly more, so the earliest start keeps a tie
                bestStart = start
                bestCount = count
            }
            start += frameHop
        }
        return bestStart
    }

    /// Step 4: round a length down to a multiple of 8,000 samples (0.5 s).
    static func roundLength(_ length: Int) -> Int { length - length % roundStep }

    enum TrimFailure: String, Equatable { case noSpeech = "no speech", tooShort = "too short" }

    /// What steps 1 to 4 decided for one capture (the same fields as quality_vectors.json).
    struct Trim: Equatable {
        var span: (start: Int, end: Int)?
        var region: (start: Int, end: Int)?
        var windowStart: Int?  // set only when the region was longer than 10 s
        var clipStart = 0
        var clipLen = 0
        var failure: TrimFailure?

        static func == (a: Trim, b: Trim) -> Bool {
            a.span?.start == b.span?.start && a.span?.end == b.span?.end && a.region?.start == b.region?.start
                && a.region?.end == b.region?.end && a.windowStart == b.windowStart && a.clipStart == b.clipStart
                && a.clipLen == b.clipLen && a.failure == b.failure
        }
    }

    /// Steps 1 to 4 for a capture of n samples whose es_vad is mask0.
    static func trim(n: Int, mask0: [UInt8], role: VoiceRole) -> Trim {
        var out = Trim()
        guard let span = frameSpan(mask0: mask0, n: n) else {
            out.failure = .noSpeech
            return out
        }
        out.span = span
        let region = role == .target ? span : (0, n)
        out.region = region
        var start = region.0
        var end = region.1
        if end - start > clipMax {
            start = bestWindow(mask: mask0, regionStart: region.0, regionEnd: region.1)
            out.windowStart = start
            end = start + clipMax
        }
        out.clipStart = start
        out.clipLen = roundLength(end - start)
        if out.clipLen < minClip { out.failure = .tooShort }
        return out
    }

    /// The clip itself: x[clipStart ..< clipStart + clipLen] (empty after a failure).
    static func clip(_ x: [Float], _ t: Trim) -> [Float] {
        guard t.failure == nil, t.clipStart + t.clipLen <= x.count else { return [] }
        return Array(x[t.clipStart..<(t.clipStart + t.clipLen)])
    }

    // MARK: - Step 5: WeSpeaker windows

    /// (start, length) of every 1.5 s window of a clip; empty if the clip is shorter than 1.5 s.
    static func windows(clipLen: Int) -> [(start: Int, length: Int)] {
        guard clipLen >= winLen else { return [] }
        return (0...((clipLen - winLen) / winHop)).map { (winHop * $0, winLen) }
    }

    /// First and last clip frame (inclusive) whose midpoint lies in [windowStart, windowStart + length),
    /// or nil if none does.
    static func framesInside(windowStart: Int, frameCount: Int, length: Int = winLen) -> (first: Int, last: Int)? {
        let inside = (0..<frameCount).filter { frameMid($0) >= windowStart && frameMid($0) < windowStart + length }
        guard let first = inside.first, let last = inside.last else { return nil }
        return (first, last)
    }
}

// MARK: - Steps 6 to 10: levels, mask, speech seconds, cosines

extension ClipTools {
    /// es_vad's frame energies in dB (steps 1 and 2 of contract 2.4): E_f = 10 log10(sum / 512 + 1e-10)
    /// over the 512 padded samples of frame f, summed in Double in index order like the C code.
    static func frameEnergiesDB(_ x: [Float]) -> [Double] {
        let frames = x.count < 576 ? 0 : (x.count - 64) / frameHop + 1 // es_enroll_num_frames
        return (0..<frames).map { f in
            var sum = 0.0
            for i in 0..<frameHop {
                let j = frameHop * f + i - pad
                if j >= 0 { let v = Double(x[j]); sum += v * v }
            }
            return 10 * log10(sum / 512 + 1e-10)
        }
    }

    /// Which frames are "speech" and which are "other" for sbr. Normally speech = mask0 frames and
    /// other = the rest. If every frame is speech, other = the max(1, F / 10) quietest frames
    /// (stable sort: ties go to the lower index) and speech = all the rest (`fallback` true).
    static func sbrSplit(_ x: [Float], mask0: [UInt8]) -> (speech: [Int], other: [Int], fallback: Bool) {
        let speech = mask0.indices.filter { mask0[$0] != 0 }
        let other = mask0.indices.filter { mask0[$0] == 0 }
        if !other.isEmpty || mask0.isEmpty { return (speech, other, false) }
        let energies = frameEnergiesDB(x)
        let nLow = max(1, energies.count / 10)
        let order = energies.indices.sorted { (energies[$0], $0) < (energies[$1], $1) } // stable
        let lowest = Array(order.prefix(nLow)).sorted()
        let lowSet = Set(lowest)
        return (energies.indices.filter { !lowSet.contains($0) }, lowest, true)
    }

    /// Mean of x^2 over the raw samples of the given frames (padding zeros are not samples).
    static func framesPower(_ x: [Float], frames: [Int]) -> Double {
        var total = 0.0
        var count = 0
        for f in frames {
            let r = frameSamples(f, n: x.count)
            for i in r { let v = Double(x[i]); total += v * v }
            count += r.count
        }
        return count > 0 ? total / Double(count) : 0
    }

    /// Step 6: speech-to-background ratio in dB on the full capture; nil when no frame is speech.
    static func sbrDB(_ x: [Float], mask0: [UInt8]) -> Double? {
        let split = sbrSplit(x, mask0: mask0)
        if split.speech.isEmpty { return nil }
        let pSpeech = framesPower(x, frames: split.speech)
        let pOther = framesPower(x, frames: split.other)
        return 10 * log10((pSpeech + powerFloor) / (pOther + powerFloor))
    }

    /// Step 7: percentage of samples with |x| >= 0.98 (compared in Float).
    static func clipPct(_ x: [Float]) -> Double {
        if x.isEmpty { return 0 }
        let count = x.reduce(0) { $0 + (abs($1) >= clipLevel ? 1 : 0) }
        return 100 * Double(count) / Double(x.count)
    }

    /// Step 9 (targets): a copy of `mask` with every frame inside window w set to 0 when
    /// windowCosOwn[w] = cos(e_w, own e) is above `threshold` (own_window_cos).
    static func ownWindowMask(_ mask: [UInt8], windows: [(start: Int, length: Int)], windowCosOwn: [Double],
                              threshold: Double) -> [UInt8] {
        var out = mask
        for (w, c) in zip(windows, windowCosOwn) where c > threshold {
            for f in out.indices where frameMid(f) >= w.start && frameMid(f) < w.start + w.length {
                out[f] = 0
            }
        }
        return out
    }

    /// Step 10: kept frames x 0.032 (what es_voiceprint_speech_seconds returns).
    static func speechSeconds(_ mask: [UInt8]) -> Double { Double(keptFrames(mask).count) * frameSeconds }

    /// Rounds a speech time to whole milliseconds. kept x 0.032 is exact to 3 decimals, but the C
    /// engine returns it as a Float (4.0 may come back as 3.9999998), so checks round it first.
    static func roundSeconds(_ s: Double) -> Double { (s * 1000).rounded() / 1000 }

    /// Cosine similarity in Double; 0 when either vector is all zeros or the lengths differ.
    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in a.indices {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        return na > 0 && nb > 0 ? dot / (na.squareRoot() * nb.squareRoot()) : 0
    }

    /// Median like numpy: the middle value, or the mean of the two middle values. nil when empty.
    static func median(_ values: [Double]) -> Double? {
        if values.isEmpty { return nil }
        let s = values.sorted()
        let m = s.count / 2
        return s.count % 2 == 1 ? s[m] : (s[m - 1] + s[m]) / 2
    }
}

// MARK: - enroll.wav (16 kHz mono 16-bit, 44-byte header, at most 10 s)

extension ClipTools {
    /// Encodes a clip as a 44-byte-header WAV: sample = Int16(clamp(x, -1, 1) * 32767) (30-ios 9.1).
    static func encodeWAV(_ pcm: [Float], sampleRate: Int = 16_000) -> Data {
        let samples = pcm.prefix(clipMax) // never more than 10 s on disk
        var d = Data()
        func put32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { d.append(contentsOf: $0) } }
        func put16(_ v: Int) { withUnsafeBytes(of: UInt16(v).littleEndian) { d.append(contentsOf: $0) } }
        let dataBytes = samples.count * 2
        d.append(contentsOf: Array("RIFF".utf8)); put32(36 + dataBytes)
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); put32(16)
        put16(1); put16(1)                                   // PCM, mono
        put32(sampleRate); put32(sampleRate * 2)             // sample rate, byte rate
        put16(2); put16(16)                                  // block align, bits per sample
        d.append(contentsOf: Array("data".utf8)); put32(dataBytes)
        for x in samples {
            let v = Int16(min(1, max(-1, x)) * 32767)
            withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
        }
        return d
    }

    /// Decodes a WAV written by encodeWAV (16-bit PCM mono 16 kHz) back to floats (x = v / 32767).
    /// Walks the chunks, so a file with extra chunks still reads. Throws on any other format.
    static func decodeWAV(_ data: Data) throws -> [Float] {
        let b = [UInt8](data)
        func u32(_ i: Int) -> Int { Int(b[i]) | Int(b[i + 1]) << 8 | Int(b[i + 2]) << 16 | Int(b[i + 3]) << 24 }
        func u16(_ i: Int) -> Int { Int(b[i]) | Int(b[i + 1]) << 8 }
        guard b.count >= 12, String(decoding: b[0..<4], as: UTF8.self) == "RIFF",
              String(decoding: b[8..<12], as: UTF8.self) == "WAVE" else { throw ContainerError("not a WAV file") }
        var i = 12
        var formatOK = false
        while i + 8 <= b.count {
            let id = String(decoding: b[i..<(i + 4)], as: UTF8.self)
            let size = u32(i + 4)
            let body = i + 8
            guard body + size <= b.count else { throw ContainerError("WAV chunk \(id) runs past the end") }
            if id == "fmt " {
                guard size >= 16, u16(body) == 1, u16(body + 2) == 1, u32(body + 4) == 16_000, u16(body + 14) == 16
                else { throw ContainerError("enroll.wav must be 16 kHz mono 16-bit PCM") }
                formatOK = true
            } else if id == "data" {
                guard formatOK else { throw ContainerError("WAV data before fmt") }
                return stride(from: body, to: body + size - 1, by: 2).map { j in
                    Float(Int16(bitPattern: UInt16(b[j]) | UInt16(b[j + 1]) << 8)) / 32767
                }
            }
            i = body + size + (size & 1) // chunks are padded to an even size
        }
        throw ContainerError("WAV has no data chunk")
    }
}
