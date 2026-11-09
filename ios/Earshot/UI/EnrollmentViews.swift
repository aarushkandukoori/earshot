// EnrollmentViews.swift - pieces shared by the enrollment screens (30-ios 13 "Add person"):
// the speech ring with seconds, and the result card with each check in green / yellow / red / grey.

import SwiftUI

/// The speech progress ring: speech seconds against the stop target.
struct SpeechRing: View {
    let seconds: Double
    let target: Double

    var body: some View {
        ZStack {
            Circle().stroke(Color.secondary.opacity(0.2), lineWidth: 10)
            Circle()
                .trim(from: 0, to: min(1, seconds / max(target, 0.1)))
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 10, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Text(Copy.speechSeconds(seconds)).font(.callout.monospacedDigit())
        }
        .frame(width: 140, height: 140)
        .accessibilityElement(children: .combine)
    }
}

/// Each check of an enrollment result, one row per check, colored by verdict.
struct CheckCard: View {
    let result: EnrollmentResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(result.checks.enumerated()), id: \.offset) { _, check in
                HStack {
                    Circle().fill(color(check.verdict)).frame(width: 12, height: 12).accessibilityHidden(true)
                    Text(name(check.kind))
                    Spacer()
                    Text(valueText(check)).foregroundStyle(.secondary).font(.callout.monospacedDigit())
                }
                .accessibilityElement(children: .combine)
            }
            if let message = result.message {
                Text(message)
                    .font(.callout)
                    .foregroundStyle(result.verdict == .fail ? Color.red : Color.orange)
                    .padding(.top, 4)
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }

    private func color(_ v: Verdict) -> Color {
        switch v {
        case .pass: return .green
        case .warn: return .yellow
        case .fail: return .red
        case .notChecked: return .gray
        }
    }

    private func name(_ kind: CheckResult.Kind) -> String {
        switch kind {
        case .speech: return Copy.checkSpeech
        case .background: return Copy.checkBackground
        case .oneVoice: return Copy.checkOneVoice
        case .notYou: return Copy.checkNotYou
        case .distinct: return Copy.checkDistinct
        case .clipping: return Copy.checkClipping
        }
    }

    private func valueText(_ c: CheckResult) -> String {
        guard let v = c.value else { return c.verdict == .notChecked ? Copy.notChecked : "-" }
        switch c.kind {
        case .speech: return String(format: "%.1f s", v)
        case .background: return String(format: "%.0f dB", v)
        case .clipping: return String(format: "%.2f %%", v)
        case .oneVoice, .notYou, .distinct: return String(format: "%.2f", v)
        }
    }
}
