//
//  BunnyRegistrationView.swift
//  BcpdSwiftDemoApp
//
//  BCPD Point Cloud Registration Demo
//

import SwiftUI
import SceneKit
import BCPD

struct BunnyRegistrationView: View {
    @StateObject private var viewModel = BunnyRegistrationViewModel()

    private let parameterColumns = [
        GridItem(.flexible(minimum: 180), spacing: 12),
        GridItem(.flexible(minimum: 180), spacing: 12),
        GridItem(.flexible(minimum: 180), spacing: 12)
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("BCPD Point Cloud Registration Demo")
                    .font(.title)

                HStack(spacing: 20) {
                    Button("Load Bunny") {
                        Task {
                            await viewModel.loadBunny()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isLoading)

                    Button("Apply Random Transform") {
                        viewModel.applyRandomTransform()
                    }
                    .disabled(!viewModel.isReady)

                    Button("Run BCPD") {
                        Task {
                            await viewModel.runBCPD()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!viewModel.canRunBCPD)

                    Button("Reset") {
                        viewModel.reset()
                    }
                }
                .padding(.vertical, 4)

                parameterPanel

                GroupBox("Status") {
                    if viewModel.isLoading {
                        ProgressView(viewModel.statusMessage)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        Text(viewModel.statusMessage)
                            .foregroundColor(viewModel.hasError ? .red : .secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Overlay View")
                            .font(.headline)
                        Spacer()
                        HStack(spacing: 12) {
                            LegendBadge(label: "Ground Truth", color: .blue)
                            LegendBadge(label: "Source", color: .orange)
                            LegendBadge(label: "Registered", color: .green)
                            LegendBadge(label: "Gizmo", color: .red)
                        }
                    }

                    SceneView(
                        scene: viewModel.comparisonScene,
                        options: [.allowsCameraControl, .autoenablesDefaultLighting]
                    )
                    .border(Color.secondary.opacity(0.35), width: 1)
                    .frame(height: 520)
                }

                if let result = viewModel.registrationResult {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Registration Results")
                            .font(.headline)

                        HStack {
                            VStack(alignment: .leading) {
                                Text("Iterations: \(result.iterations)")
                                Text("Residual: \(String(format: "%.6f", result.residual))")
                            }
                            Spacer()
                            VStack(alignment: .leading) {
                                Text("Matched Points: \(String(format: "%.1f", result.matchedPointsCount))")
                                Text("Scale: \(String(format: "%.4f", result.scale))")
                            }
                        }
                        .font(.caption)
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                    }
                }
            }
            .padding(20)
        }
        .frame(minWidth: 1200, minHeight: 800)
    }

    private var parameterPanel: some View {
        GroupBox("Registration Parameters") {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(viewModel.parameterPresetName)
                            .font(.headline)
                        Text("Current default is tuned for rigid or nearly rigid registration.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Reset to Rigid Defaults") {
                        viewModel.resetParametersToRigidDefaults()
                    }
                    .disabled(viewModel.isLoading)
                }

                Toggle("Detailed Mode", isOn: $viewModel.isDetailedMode)

                Text(viewModel.parameterSummaryText)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if viewModel.isDetailedMode {
                    Divider()

                    LazyVGrid(columns: parameterColumns, alignment: .leading, spacing: 12) {
                        ParameterEditorField(
                            title: "omega",
                            value: $viewModel.parameterSettings.omega,
                            formatter: DemoNumberFormatters.decimal
                        )
                        ParameterEditorField(
                            title: "lambda",
                            value: $viewModel.parameterSettings.lambda,
                            formatter: DemoNumberFormatters.scientific
                        )
                        ParameterEditorField(
                            title: "kappa",
                            value: $viewModel.parameterSettings.kappa,
                            formatter: DemoNumberFormatters.scientific
                        )
                        ParameterEditorField(
                            title: "gamma",
                            value: $viewModel.parameterSettings.gamma,
                            formatter: DemoNumberFormatters.decimal
                        )
                        ParameterEditorField(
                            title: "beta",
                            value: $viewModel.parameterSettings.beta,
                            formatter: DemoNumberFormatters.decimal
                        )
                        ParameterEditorField(
                            title: "tolerance",
                            value: $viewModel.parameterSettings.tolerance,
                            formatter: DemoNumberFormatters.scientific
                        )
                        IntegerParameterEditorField(
                            title: "maxIterations",
                            value: $viewModel.parameterSettings.maxIterations
                        )
                    }

                    HStack(spacing: 20) {
                        Picker("Kernel", selection: $viewModel.parameterSettings.kernel) {
                            ForEach(DemoKernelOption.allCases) { option in
                                Text(option.rawValue).tag(option)
                            }
                        }
                        .pickerStyle(.menu)

                        Toggle("Use Debias", isOn: $viewModel.parameterSettings.useDebias)
                        Toggle("Verbose", isOn: $viewModel.parameterSettings.verbose)
                    }
                } else {
                    LazyVGrid(columns: parameterColumns, alignment: .leading, spacing: 12) {
                        ParameterSummaryValue(title: "omega", value: formatDecimal(viewModel.parameterSettings.omega))
                        ParameterSummaryValue(title: "lambda", value: formatScientific(viewModel.parameterSettings.lambda))
                        ParameterSummaryValue(title: "kappa", value: formatScientific(viewModel.parameterSettings.kappa))
                        ParameterSummaryValue(title: "gamma", value: formatDecimal(viewModel.parameterSettings.gamma))
                        ParameterSummaryValue(title: "beta", value: formatDecimal(viewModel.parameterSettings.beta))
                        ParameterSummaryValue(title: "maxIterations", value: "\(viewModel.parameterSettings.maxIterations)")
                        ParameterSummaryValue(title: "tolerance", value: formatScientific(viewModel.parameterSettings.tolerance))
                        ParameterSummaryValue(title: "kernel", value: viewModel.parameterSettings.kernel.rawValue)
                        ParameterSummaryValue(title: "debias", value: viewModel.parameterSettings.useDebias ? "On" : "Off")
                    }
                }
            }
        }
    }

    private func formatDecimal(_ value: Double) -> String {
        String(format: "%.4g", value)
    }

    private func formatScientific(_ value: Double) -> String {
        String(format: "%.2e", value)
    }
}

private enum DemoNumberFormatters {
    static let decimal: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.maximumFractionDigits = 6
        formatter.minimumFractionDigits = 0
        formatter.generatesDecimalNumbers = false
        return formatter
    }()

    static let scientific: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .scientific
        formatter.exponentSymbol = "e"
        formatter.maximumFractionDigits = 6
        formatter.minimumFractionDigits = 0
        formatter.generatesDecimalNumbers = false
        return formatter
    }()

    static let integer: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .none
        formatter.maximumFractionDigits = 0
        formatter.minimum = 1
        return formatter
    }()
}

private struct ParameterSummaryValue: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.body.monospacedDigit())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color.gray.opacity(0.08))
        .cornerRadius(8)
    }
}

private struct LegendBadge: View {
    let label: String
    let color: Color

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.gray.opacity(0.08))
        .cornerRadius(999)
    }
}

private struct ParameterEditorField: View {
    let title: String
    @Binding var value: Double
    let formatter: NumberFormatter

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(title, value: $value, formatter: formatter)
                .textFieldStyle(.roundedBorder)
                .font(.body.monospacedDigit())
        }
    }
}

private struct IntegerParameterEditorField: View {
    let title: String
    @Binding var value: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(title, value: $value, formatter: DemoNumberFormatters.integer)
                .textFieldStyle(.roundedBorder)
                .font(.body.monospacedDigit())
        }
    }
}

#Preview {
    BunnyRegistrationView()
}
