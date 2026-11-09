// RouteRulesTests.swift - allowed outputs, slack and route signatures (30-ios 3.4).

import AVFoundation
import XCTest
@testable import Earshot

final class RouteRulesTests: XCTestCase {
    private func route(_ outputs: [AVAudioSession.Port], input: AVAudioSession.Port? = .builtInMic,
                       rate: Double = 48_000, buffer: Double = 64.0 / 48_000.0, uid: String = "uid") -> RouteInfo {
        let ports = outputs.enumerated().map { Port(type: $1, uid: "\(uid)-\($0)", name: "port \($0)") }
        return RouteInfo(outputs: ports, inputType: input, sampleRate: rate, ioBufferDuration: buffer,
                         inputLatency: 0.005, outputLatency: 0.01)
    }

    func testAllowedOutputs() {
        XCTAssertTrue(route([.bluetoothA2DP]).allowed)
        XCTAssertTrue(route([.headphones]).allowed)
        XCTAssertTrue(route([.usbAudio]).allowed)
    }

    func testForbiddenOutputs() {
        for port: AVAudioSession.Port in [.builtInSpeaker, .builtInReceiver, .bluetoothHFP, .airPlay, .carAudio,
                                          .HDMI, .bluetoothLE] {
            XCTAssertFalse(route([port]).allowed, "\(port.rawValue) must mute")
        }
        XCTAssertFalse(route([]).allowed, "no output at all must mute")
        XCTAssertFalse(route([.bluetoothA2DP, .builtInSpeaker]).allowed, "every output must be allowed")
    }

    func testInputMustBeBuiltInMic() {
        XCTAssertFalse(route([.bluetoothA2DP], input: .bluetoothHFP).allowed)
        XCTAssertFalse(route([.headphones], input: .headsetMic).allowed)
        XCTAssertFalse(route([.bluetoothA2DP], input: nil).allowed)
    }

    func testSlack() {
        XCTAssertEqual(route([.bluetoothA2DP]).slack16, 43)
        XCTAssertTrue(route([.bluetoothA2DP]).isBluetooth)
        XCTAssertEqual(route([.headphones]).slack16, 22)
        XCTAssertEqual(route([.usbAudio]).slack16, 22)
    }

    func testIOFrames() {
        XCTAssertEqual(route([.headphones]).ioFrames, 64)
        XCTAssertEqual(route([.headphones], buffer: 1104.0 / 48_000.0).ioFrames, 1104)
        XCTAssertEqual(route([.headphones], buffer: 0.0001).ioFrames, 16, "clamped to 16")
        XCTAssertEqual(route([.headphones], buffer: 1.0).ioFrames, 4096, "clamped to 4096")
    }

    func testSignatureChanges() {
        let base = route([.bluetoothA2DP])
        XCTAssertEqual(base.signature, route([.bluetoothA2DP]).signature, "same route, same signature")
        XCTAssertNotEqual(base.signature, route([.bluetoothA2DP], uid: "other").signature, "new port uid")
        XCTAssertNotEqual(base.signature, route([.bluetoothA2DP], rate: 44_100).signature, "new rate")
        XCTAssertNotEqual(base.signature, route([.bluetoothA2DP], buffer: 128.0 / 48_000.0).signature,
                          "new buffer size")
        XCTAssertNotEqual(base.signature, route([.bluetoothA2DP], input: .headsetMic).signature, "new input")
    }

    func testRates() {
        XCTAssertTrue(route([.headphones], rate: 48_000).rateSupported)
        XCTAssertTrue(route([.headphones], rate: 44_100).rateSupported)
        XCTAssertFalse(route([.headphones], rate: 16_000).rateSupported)
        XCTAssertFalse(route([.headphones], rate: 96_000).rateSupported)
    }
}
