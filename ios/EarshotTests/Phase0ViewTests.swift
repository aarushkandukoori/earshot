// Phase0ViewTests.swift - a smoke test for the Phase 0 screens (UI/Phase0View.swift): the list and
// every step page are put in a window with the stub-backed objects, laid out, and given a moment
// to appear. Proves the views build and attach the runner; it does not tap anything.

import SwiftUI
import XCTest
@testable import Earshot

@MainActor
final class Phase0ViewTests: XCTestCase {
    private func host<V: View>(_ view: V, runner: Phase0Runner) throws -> UIWindow {
        let root = NavigationStack { view }
            .environmentObject(runner)
            .environmentObject(AudioController())
            .environmentObject(Earshot.Selector())
            .environmentObject(LaunchState())
        guard let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first else {
            throw XCTSkip("the test host has no window scene")
        }
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 393, height: 852)
        window.rootViewController = UIHostingController(rootView: root)
        window.makeKeyAndVisible()
        window.layoutIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.2)) // let onAppear run
        return window
    }

    private func tempRunner() -> Phase0Runner {
        let dir = TestSupport.tempFolder()
        return Phase0Runner(store: Phase0Store(folder: dir.appendingPathComponent("phase0")),
                            progressStore: Phase0ProgressStore(url: dir.appendingPathComponent("progress.json")))
    }

    func testListScreenAppearsAndAttachesTheRunner() throws {
        let runner = tempRunner()
        let window = try host(Phase0View(), runner: runner)
        XCTAssertNotNil(runner.audio, "onAppear attached the runner")
        XCTAssertNotNil(window.rootViewController?.view)
        window.isHidden = true
    }

    func testEveryStepPageBuilds() throws {
        let runner = tempRunner()
        for id in Phase0StepID.allCases {
            let window = try host(Phase0StepView(id: id), runner: runner)
            XCTAssertNotNil(window.rootViewController?.view, "\(id)")
            window.isHidden = true
        }
        XCTAssertNil(runner.run, "opening a page never starts a run")
    }
}
