// RouteInfo.swift - a snapshot of the audio route and the "allowed output" rule (30-ios 3.4).
//
// Earshot plays only into headphones: AirPods over A2DP, wired headphones or USB audio.
// Anything else (phone speaker, receiver, HFP headset, AirPlay, car) mutes at once
// (contract 5.3). The input must be the iPhone's built-in mic (30-ios 3.2).

import AVFoundation

struct Port: Equatable { let type: AVAudioSession.Port; let uid: String; let name: String }

struct RouteInfo: Equatable {
    var outputs: [Port]
    var inputType: AVAudioSession.Port?          // must be .builtInMic
    var sampleRate: Double, ioBufferDuration: Double, ioFrames: Int
    var inputLatency: Double, outputLatency: Double   // seconds, as reported
    var allowed: Bool        // every output is .bluetoothA2DP, .headphones or .usbAudio, and input is .builtInMic
    var isBluetooth: Bool    // output .bluetoothA2DP -> slack16 43, else 22 (contract 4.2)
    var signature: String    // output uids + input type + sampleRate + ioFrames; a change triggers the engine swap (6)
    var inputDataSource: String?                 // logged, never set (30-ios 3.1)

    /// The only output port types Earshot will play into (contract 5.3).
    static let allowedOutputs: Set<AVAudioSession.Port> = [.bluetoothA2DP, .headphones, .usbAudio]

    /// Builds a route from plain values (used by `init(session:)` and by the unit tests).
    init(outputs: [Port], inputType: AVAudioSession.Port?, sampleRate: Double, ioBufferDuration: Double,
         inputLatency: Double, outputLatency: Double, inputDataSource: String? = nil) {
        self.outputs = outputs
        self.inputType = inputType
        self.sampleRate = sampleRate
        self.ioBufferDuration = ioBufferDuration
        self.ioFrames = RouteInfo.frames(ioBufferDuration: ioBufferDuration, sampleRate: sampleRate)
        self.inputLatency = inputLatency
        self.outputLatency = outputLatency
        self.inputDataSource = inputDataSource

        // Allowed: at least one output, every output an allowed type, and the built-in mic as input.
        var allOutputsOK = !outputs.isEmpty
        for port in outputs where !RouteInfo.allowedOutputs.contains(port.type) {
            allOutputsOK = false
        }
        self.allowed = allOutputsOK && inputType == .builtInMic
        self.isBluetooth = outputs.contains { $0.type == .bluetoothA2DP }

        let uids = outputs.map { $0.uid }.joined(separator: ",")
        self.signature = "\(uids)|\(inputType?.rawValue ?? "none")|\(Int(sampleRate))|\(ioFrames)"
    }

    /// Reads everything back from the live session (after activation and after every route change).
    init(session s: AVAudioSession) {
        let route = s.currentRoute
        let outs = route.outputs.map { Port(type: $0.portType, uid: $0.uid, name: $0.portName) }
        let input = route.inputs.first
        self.init(outputs: outs, inputType: input?.portType, sampleRate: s.sampleRate,
                  ioBufferDuration: s.ioBufferDuration, inputLatency: s.inputLatency,
                  outputLatency: s.outputLatency, inputDataSource: input?.selectedDataSource?.dataSourceName)
    }

    /// Nothing known yet (before the first activation).
    static let none = RouteInfo(outputs: [], inputType: nil, sampleRate: 0, ioBufferDuration: 0,
                                inputLatency: 0, outputLatency: 0)

    /// io_frames = round(ioBufferDuration x sampleRate), clamped to 16...4096 (30-ios 3.1).
    static func frames(ioBufferDuration: Double, sampleRate: Double) -> Int {
        let raw = Int((ioBufferDuration * sampleRate).rounded())
        return min(max(raw, 16), 4096)
    }

    /// Extra deadline slack for the engine: 43 on Bluetooth, 22 wired (contract 4.2).
    var slack16: Int { isBluetooth ? 43 : 22 }

    /// The engine accepts only 48 kHz and 44.1 kHz (es_engine_config.io_rate).
    var rateSupported: Bool { sampleRate == 48_000 || sampleRate == 44_100 }

    /// The output port UID that latency measurements are stored under.
    var primaryOutputUID: String { outputs.first?.uid ?? "" }

    /// Short text for the screen, for example "AirPods Pro (BluetoothA2DPOutput)".
    var outputDescription: String {
        if outputs.isEmpty { return "none" }
        return outputs.map { "\($0.name) (\($0.type.rawValue))" }.joined(separator: ", ")
    }
}
