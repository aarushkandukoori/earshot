// Phase0View.swift - Diagnostics > Phase 0, the guided runner screen (30-ios 14 "Phase 0 runner",
// 15.3; contract 11 I-1). Main actor (SwiftUI).
//
// The list shows every step with its status; a step's page says plainly what the owner has to do,
// lets them pick the output (and method, room or AirPods setting where it matters), runs one item,
// and offers Skip / Finish / Reopen. The running measurement shows at the top of both screens with
// its instruction line, a progress bar and Cancel (or Done for the open-ended locked run).
// All logic is in Phase0Runner; every sentence comes from Copy (Copy+Phase0.swift).

import SwiftUI

struct Phase0View: View {
    @EnvironmentObject private var runner: Phase0Runner
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var selector: Selector
    @EnvironmentObject private var launch: LaunchState
    @State private var notes = ""

    var body: some View {
        List {
            Phase0RunSection()
            Section {
                Text(Copy.phase0Intro).font(.footnote)
                Text(Copy.phase0File(runner.fileURL.lastPathComponent)).font(.footnote.monospaced())
                Text(Copy.phase0StepsSettled(runner.progress.settledCount, of: Phase0StepID.allCases.count))
                    .font(.footnote)
                if runner.report.engine == "stub" {
                    Text(Copy.phase0StubWarning).font(.footnote).foregroundStyle(.orange)
                }
            }
            Section {
                ForEach(Phase0StepID.allCases) { id in
                    NavigationLink { Phase0StepView(id: id) } label: { stepRow(id) }
                }
            }
            Section(Copy.phase0Notes) {
                TextField(Copy.phase0Notes, text: $notes, axis: .vertical)
                    .onSubmit { runner.setNotes(notes) }
                Button(Copy.done) { runner.setNotes(notes) }
            }
            Section {
                Button(Copy.phase0NewFile, role: .destructive) { runner.newFile() }.disabled(runner.run != nil)
            }
        }
        .navigationTitle(Copy.phase0Title)
        .onAppear {
            runner.attach(audio: audio, selector: selector, launch: launch)
            notes = runner.report.notes
        }
        .onChange(of: runner.report.notes) { _, text in notes = text } // a run may add a note
    }

    private func stepRow(_ id: Phase0StepID) -> some View {
        HStack {
            VStack(alignment: .leading) {
                Text(Copy.phase0StepTitle(id))
                Text("\(runner.progress.finishedCount(id)) / \(Phase0Plan.items(id).count)")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Text(Copy.phase0Status(runner.progress.status(id))).font(.caption)
                .foregroundStyle(runner.progress.current == id ? .primary : .secondary)
        }
    }
}

/// The running measurement (shown on the list and on every step page) and the newest message.
struct Phase0RunSection: View {
    @EnvironmentObject private var runner: Phase0Runner

    var body: some View {
        if let run = runner.run {
            Section(run.title) {
                Text(run.prompt).font(.headline)
                if run.seconds > 0 {
                    ProgressView(value: min(run.elapsed, run.seconds), total: run.seconds)
                }
                Text(Copy.phase0Elapsed(Int(run.elapsed), Int(run.seconds))).font(.caption.monospacedDigit())
                HStack {
                    Button(Copy.cancel, role: .cancel) { runner.cancel() }
                    if run.seconds == 0 {
                        Spacer()
                        Button(Copy.done) { runner.endOpenRun() }.buttonStyle(.borderedProminent)
                    }
                }
            }
        }
        if !runner.message.isEmpty {
            Section { Text(runner.message).font(.callout) }
        }
    }
}

/// One step: what to do, the choices it needs, its run buttons, its items and Skip / Finish / Reopen.
struct Phase0StepView: View {
    let id: Phase0StepID
    @EnvironmentObject private var runner: Phase0Runner
    @EnvironmentObject private var audio: AudioController
    @State private var route = "airpods_pro2"
    @State private var method = ProbeMethod.inEar.rawValue
    @State private var ambient = 45
    @State private var combo = Phase0Plan.stemCombos[0]

    private var busy: Bool { runner.run != nil }
    private var status: Phase0StepStatus { runner.progress.status(id) }

    var body: some View {
        List {
            Phase0RunSection()
            Section {
                Text(Copy.phase0StepWhat(id))
                HStack { Text(Copy.phase0Status(status)).font(.caption); Spacer() }
                if id.needsRealEngine && runner.report.engine != "real" {
                    Text(Copy.phase0NeedsRealEngine).font(.footnote).foregroundStyle(.orange)
                }
            }
            Section { controls }.disabled(busy)
            Section(Copy.phase0Items) {
                ForEach(Phase0Plan.items(id), id: \.self) { item in
                    HStack {
                        Text(item).font(.caption.monospaced())
                        Spacer()
                        Image(systemName: runner.progress.isFinished(id, item) ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(runner.progress.isFinished(id, item) ? .green : .secondary)
                    }
                }
            }
            Section {
                if status == .done || status == .skipped {
                    Button(Copy.phase0Reopen) { runner.reopen(id) }
                } else {
                    if status == .inProgress { Button(Copy.phase0Finish) { runner.markDone(id) } }
                    Button(Copy.phase0Skip) { runner.skip(id) }
                }
            }
            .disabled(runner.run?.step == id)
        }
        .navigationTitle(Copy.phase0StepTitle(id))
        .onAppear { route = defaultRoute() }
    }

    /// The live output's file name when this step offers it, else the last choice.
    private func defaultRoute() -> String {
        let offered = id == .buffers ? Phase0Plan.bufferRoutes : Phase0Plan.airPods
        let uid = audio.route.primaryOutputUID
        guard let live = Phase0Math.routeLabel(audio.route, model: AirPodsModelStore().model(forPort: uid)),
              offered.contains(live) else { return route }
        return live
    }

    // MARK: Choices and run buttons per step

    @ViewBuilder private var controls: some View {
        switch id {
        case .buffers:
            routePicker(Phase0Plan.bufferRoutes)
            Button(Copy.phase0RunBuffers) { runner.runBuffers(route: route) }
            Button(Copy.phase0RunBufferLocked) { runner.runBufferLocked(route: route) }
        case .tapTest:
            routePicker(Phase0Plan.airPods)
            Button(Copy.phase0RunTap) { runner.runTapTest(route: route) }
        case .latency:
            routePicker(Phase0Plan.airPods)
            methodPicker
            ambientPicker
            connectRow
            Button(Copy.phase0RunProbe) { runner.runLatency(route: route, method: method, ambient: ambient) }
        case .probeLevel:
            Text(Copy.phase0ProbeBuild(Phase0Runner.probeDBFS)).font(.footnote)
            Text(Copy.phase0Pick(Phase0Math.probePick(runner.report.latency))).font(.footnote)
            routePicker(Phase0Plan.airPods)
            methodPicker
            ambientPicker
            Button(Copy.phase0RunCell) { runner.runProbeCell(route: route, ambient: ambient, method: method) }
        case .motion:
            routePicker(Phase0Plan.airPods)
            Button(Copy.phase0RunMotion) { runner.runMotion(route: route) }
        case .stem:
            routePicker(Phase0Plan.airPods)
            Picker(Copy.phase0StepTitle(.stem), selection: $combo) {
                ForEach(Phase0Plan.stemCombos, id: \.self) { Text(Copy.phase0Combo($0)).tag($0) }
            }
            Button(Copy.phase0RunStem) { runner.runStem(route: route, combo: combo) }
            Button(Copy.phase0RunStemPaused) { runner.runStemPaused(route: route, combo: combo) }
        case .locked:
            Button(Copy.phase0RunLocked) { runner.runLocked() }
        case .wespeaker:
            Button(Copy.phase0RunOpen) { runner.runWeSpeaker(locked: false) }
            Button(Copy.phase0RunLockedScreen) { runner.runWeSpeaker(locked: true) }
        case .alarm:
            Button(Copy.phase0RunAlarm) { runner.runAlarm() }
        case .micLevel:
            Button(Copy.phase0RunMic) { runner.runMicLevel() }
        case .compute:
            ForEach(Phase0Plan.computeItems, id: \.self) { item in
                Button(Copy.phase0RunCompute(item)) { runner.runCompute(item: item) }
            }
        case .power:
            ForEach(Phase0Plan.powerItems, id: \.self) { item in
                Button(Copy.phase0RunPower(item)) { runner.runPower(item: item) }
            }
        }
    }

    private func routePicker(_ routes: [String]) -> some View {
        Picker(Copy.phase0Output, selection: $route) {
            ForEach(routes, id: \.self) { Text(Copy.phase0RouteName($0)).tag($0) }
        }
    }

    private var methodPicker: some View {
        Picker(Copy.measureDelay, selection: $method) {
            Text(Copy.phase0MethodA).tag(ProbeMethod.inEar.rawValue)
            Text(Copy.phase0MethodB).tag(ProbeMethod.earDetectionOff.rawValue)
        }
    }

    private var ambientPicker: some View {
        Picker(Copy.phase0Quiet, selection: $ambient) {
            Text(Copy.phase0Quiet).tag(45)
            Text(Copy.phase0Loud).tag(70)
        }
    }

    private var connectRow: some View {
        HStack {
            Text(runner.progress.connectedAt[route] == nil ? Copy.phase0NotConnectedYet
                 : Copy.phase0MinutesSinceConnect(runner.minutesSinceConnect(route)))
                .font(.footnote)
            Spacer()
            Button(Copy.phase0Connected) { runner.markConnected(route: route) }.buttonStyle(.bordered)
        }
    }
}
