// PowerPolicyTests.swift - the thermal and Low Power downshift (contract 5.3 "Thermal", 30-ios 6):
// the level table, the 60 s recovery, what each level means, which slots the engine keeps
// (PowerSaving), and the whole path through Selector and AudioController with the stub engine.

import XCTest
@testable import Earshot

final class PowerPolicyTests: XCTestCase {
    /// Every row of the downshift table, starting from normal.
    func testDownshiftTableFromNormal() {
        let rows: [(ProcessInfo.ThermalState, Bool, PowerLevel)] = [
            // thermal    low power  level
            (.nominal,    false,     .normal),
            (.fair,       false,     .normal),
            (.nominal,    true,      .saving),
            (.fair,       true,      .saving),
            (.serious,    false,     .saving),
            (.serious,    true,      .saving),
            (.critical,   false,     .critical),
            (.critical,   true,      .critical),
        ]
        for (thermal, lowPower, want) in rows {
            var p = PowerPolicy()
            XCTAssertEqual(p.update(thermal: thermal, lowPower: lowPower, now: 0), want, "\(thermal.rawValue) \(lowPower)")
        }
    }

    func testWhatEachLevelMeans() {
        XCTAssertFalse(PowerPolicy.keepsOnlySelected(.normal))
        XCTAssertTrue(PowerPolicy.keepsOnlySelected(.saving))
        XCTAssertTrue(PowerPolicy.keepsOnlySelected(.critical))
        XCTAssertEqual(PowerPolicy.alarmOverlap(.normal), 0.5, "3 s windows, 50% overlap")
        XCTAssertEqual(PowerPolicy.alarmOverlap(.saving), 0, "every 3 s while saving power")
        XCTAssertEqual(PowerPolicy.alarmHopSeconds(.normal), 1.5)
        XCTAssertEqual(PowerPolicy.alarmHopSeconds(.saving), 3)
        XCTAssertEqual(PowerPolicy.alarmHopSeconds(.critical), 3)
    }

    func testRecoveryNeeds60CalmSeconds() {
        var p = PowerPolicy()
        XCTAssertEqual(p.update(thermal: .serious, lowPower: false, now: 0), .saving)
        XCTAssertEqual(p.update(thermal: .fair, lowPower: false, now: 10), .saving, "calm from 10 s")
        XCTAssertEqual(p.update(thermal: .fair, lowPower: false, now: 69.9), .saving)
        XCTAssertEqual(p.update(thermal: .nominal, lowPower: false, now: 70), .normal, "60 calm seconds")
        XCTAssertNil(p.calmSince)
    }

    func testAnyHotOrLowPowerMomentRestartsTheMinute() {
        var p = PowerPolicy()
        p.update(thermal: .serious, lowPower: false, now: 0)
        p.update(thermal: .fair, lowPower: false, now: 10)
        XCTAssertEqual(p.update(thermal: .fair, lowPower: true, now: 40), .saving, "Low Power came back")
        XCTAssertNil(p.calmSince)
        XCTAssertEqual(p.update(thermal: .fair, lowPower: false, now: 50), .saving)
        XCTAssertEqual(p.update(thermal: .fair, lowPower: false, now: 109), .saving, "only 59 s since 50")
        XCTAssertEqual(p.update(thermal: .fair, lowPower: false, now: 110), .normal)
    }

    func testCriticalPassesThroughSaving() {
        var p = PowerPolicy()
        XCTAssertEqual(p.update(thermal: .critical, lowPower: false, now: 0), .critical)
        XCTAssertEqual(p.update(thermal: .nominal, lowPower: false, now: 1), .saving, "never straight to normal")
        XCTAssertEqual(p.update(thermal: .nominal, lowPower: false, now: 60.9), .saving)
        XCTAssertEqual(p.update(thermal: .nominal, lowPower: false, now: 61), .normal)
    }

    // MARK: - Which slots the engine keeps

    private let a = UUID(), b = UUID(), c = UUID()

    /// Everyone in slot 0, A in 1, B in 2, Focus on B.
    private func twoPeople() -> TableState {
        var t = TableState()
        _ = t.add(a, dir: nil, dirStale: false)
        _ = t.add(b, dir: nil, dirStale: false)
        return t
    }

    func testKeptSlotPerMode() {
        var t = TableState()
        XCTAssertEqual(PowerSaving.keptSlot(t), 0, "Everyone with nobody added: the Everyone slot")
        t = twoPeople()
        XCTAssertEqual(t.mode, .focus)
        XCTAssertEqual(PowerSaving.keptSlot(t), 2, "Focus: the selected person")
        _ = t.setMode(.table)
        XCTAssertEqual(PowerSaving.keptSlot(t), 2, "Table: the selected person")
        _ = t.setMode(.everyone)
        XCTAssertEqual(PowerSaving.keptSlot(t), 0, "Everyone: the Everyone slot (still warm in 0)")
    }

    func testEngineContentsAndGainsWhileSaving() {
        let t = twoPeople()
        XCTAssertEqual(PowerSaving.engineContents(t, saving: false), [.everyone, .person(a), .person(b)])
        XCTAssertEqual(PowerSaving.engineContents(t, saving: true), [.empty, .empty, .person(b)])
        XCTAssertEqual(PowerSaving.engineGains(t, saving: true), [0, 0, 1])
        var table = t
        _ = table.setMode(.table)
        XCTAssertEqual(PowerSaving.engineGains(table, saving: false), [0, 1, 1])
        XCTAssertEqual(PowerSaving.engineGains(table, saving: true), [0, 0, 1], "uncomputed slots held at 0")
    }

    func testEnteringSavingClearsEveryOtherSlot() {
        let plan = PowerSaving.reconcile(tableLoads: [], loaded: [.everyone, .person(a), .person(b)],
                                         want: [.empty, .empty, .person(b)])
        XCTAssertEqual(plan, [SlotLoad(slot: 0, content: .empty, reset: false),
                              SlotLoad(slot: 1, content: .empty, reset: false)])
    }

    func testFocusingAnotherPersonWhileSavingLoadsThemCold() {
        var t = twoPeople()
        let loads = t.focus(on: 1) // warm in the table's view: no load
        XCTAssertEqual(loads, [])
        let want = PowerSaving.engineContents(t, saving: true)
        XCTAssertEqual(want, [.empty, .person(a), .empty])
        let plan = PowerSaving.reconcile(tableLoads: loads, loaded: [.empty, .empty, .person(b)], want: want)
        XCTAssertEqual(plan, [SlotLoad(slot: 1, content: .person(a), reset: true),
                              SlotLoad(slot: 2, content: .empty, reset: false)])
    }

    func testRecoveryRefillsTheEmptiedSlotsCold() {
        let t = twoPeople()
        let plan = PowerSaving.reconcile(tableLoads: [], loaded: [.empty, .empty, .person(b)],
                                         want: PowerSaving.engineContents(t, saving: false))
        XCTAssertEqual(plan, [SlotLoad(slot: 0, content: .everyone, reset: true),
                              SlotLoad(slot: 1, content: .person(a), reset: true)])
    }

    func testAWarmReloadStaysWarm() {
        let plan = PowerSaving.reconcile(tableLoads: [SlotLoad(slot: 2, content: .person(b), reset: false)],
                                         loaded: [.empty, .empty, .person(b)], want: [.empty, .empty, .person(b)])
        XCTAssertEqual(plan, [SlotLoad(slot: 2, content: .person(b), reset: false)], "re-recorded voice: warm")
    }

    /// Without power saving the reconcile step must change nothing: over 400 random table changes
    /// its plan equals TableState's own (merged) loads exactly.
    func testWithoutSavingTheTableLoadsPassUnchanged() {
        var rng = SplitMix(seed: 7)
        let ids = (0..<5).map { _ in UUID() }
        var t = TableState()
        for step in 0..<400 {
            let before = t.contents
            let loads: [SlotLoad]
            switch rng.next() % 4 {
            case 0: loads = t.add(ids[Int(rng.next() % 5)], dir: nil, dirStale: false)
            case 1: loads = t.focus(on: Int(rng.next() % 3))
            case 2: loads = t.setMode([TableMode.focus, .table, .everyone][Int(rng.next() % 3)]) ?? []
            default: loads = t.remove(ids[Int(rng.next() % 5)])
            }
            let plan = PowerSaving.reconcile(tableLoads: loads, loaded: before, want: t.contents)
            XCTAssertEqual(plan, TableState.merged(loads), "step \(step)")
        }
    }
}

/// A tiny deterministic random generator for the table walk.
struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
