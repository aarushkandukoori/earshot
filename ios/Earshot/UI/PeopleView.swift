// PeopleView.swift - You, Tonight, Saved (30-ios 9, 13 "People"). Main actor (SwiftUI).
// I-3: swipe right to add a person to the table or remove them from it (Selector).
//
// Delete removes a person's folder at once; "Delete everyone" removes every tonight and saved
// person. Rename changes only the name. Turning a tonight person into a saved one asks for
// consent first. The own voice can be
// deleted too (it is biometric data, 30-ios 9.2); that asks first, because Earshot is muted
// again until a new own voice is recorded.

import SwiftUI

struct PeopleView: View {
    @EnvironmentObject private var people: PeopleStore
    @EnvironmentObject private var selector: Selector
    @State private var confirmDeleteAll = false
    @State private var confirmDeleteOwn = false
    @State private var consentFor: Person? = nil
    @State private var renaming: Person? = nil
    @State private var newName = ""

    var body: some View {
        List {
            Section(Copy.you) {
                if let own = people.own {
                    row(own)
                        .swipeActions {
                            Button(Copy.delete, role: .destructive) { confirmDeleteOwn = true }
                        }
                }
                NavigationLink(people.own == nil ? Copy.recordMyVoice : Copy.reRecordMyVoice) { OwnVoiceView() }
            }
            section(Copy.tonight, people.people.filter { $0.kind == .tonight })
            section(Copy.saved, people.people.filter { $0.kind == .saved })
            if !people.brokenFolders.isEmpty {
                Section(Copy.needsReAdding) {
                    ForEach(people.brokenFolders, id: \.self) { folder in
                        Text(String(folder.prefix(8))).font(.callout.monospaced())
                    }
                }
            }
            if !people.people.isEmpty {
                Section {
                    Button(Copy.deleteEveryone, role: .destructive) { confirmDeleteAll = true }
                }
            }
        }
        .navigationTitle(Copy.people)
        .confirmationDialog(Copy.deleteEveryoneConfirm, isPresented: $confirmDeleteAll, titleVisibility: .visible) {
            Button(Copy.deleteEveryone, role: .destructive) { people.deleteEveryone() }
        }
        .confirmationDialog(Copy.deleteMyVoiceConfirm, isPresented: $confirmDeleteOwn, titleVisibility: .visible) {
            // PeopleStore.delete tells AudioController the own voice is gone (setup mode, muted).
            Button(Copy.deleteMyVoice, role: .destructive) { if let own = people.own { people.delete(own) } }
        }
        .alert(Copy.consentQuestion(consentFor?.name ?? ""), isPresented: Binding(
            get: { consentFor != nil }, set: { if !$0 { consentFor = nil } })) {
            Button(Copy.theyAgreed) {
                if let p = consentFor { try? people.makeSaved(p, consent: true) }
                consentFor = nil
            }
            Button(Copy.tonightOnly, role: .cancel) { consentFor = nil }
        }
        .alert(Copy.rename, isPresented: Binding(
            get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField(Copy.name, text: $newName)
            Button(Copy.done) {
                if let p = renaming { try? people.rename(p.id, to: newName) }
                renaming = nil
            }
            Button(Copy.cancel, role: .cancel) { renaming = nil }
        }
    }

    @ViewBuilder private func section(_ title: String, _ list: [Person]) -> some View {
        Section(title) {
            if list.isEmpty { Text(Copy.noOneYet).foregroundStyle(.secondary) }
            ForEach(list) { p in
                row(p)
                    .swipeActions {
                        Button(Copy.delete, role: .destructive) { people.delete(p) }
                        Button(Copy.rename) { newName = p.name; renaming = p }
                        if p.kind == .tonight { Button(Copy.saveForLater) { consentFor = p } }
                    }
                    .swipeActions(edge: .leading) { tableAction(p) }
            }
        }
    }

    /// "Add to table" for a ready person not at the table, "Remove from table" for one who is.
    @ViewBuilder private func tableAction(_ p: Person) -> some View {
        if selector.table.slot(of: p.id) != nil {
            Button(Copy.removeFromTable) { selector.removeFromTable(p.id) }
        } else if people.status[p.id] == .ready {
            Button(Copy.addToTable) { selector.addToTable(p, freshYaw: nil) }.tint(.accentColor)
        }
    }

    private func row(_ p: Person) -> some View {
        HStack {
            Text(p.name)
            Spacer()
            switch people.status[p.id] ?? .gettingReady {
            case .gettingReady: Text(Copy.gettingReadyPerson).foregroundStyle(.secondary)
            case .needsReAdding: Text(Copy.needsReAdding).foregroundStyle(.red)
            case .ready: Text(Copy.speechSeconds(p.quality.speechS)).foregroundStyle(.secondary)
            }
        }
        .font(.callout)
    }
}
