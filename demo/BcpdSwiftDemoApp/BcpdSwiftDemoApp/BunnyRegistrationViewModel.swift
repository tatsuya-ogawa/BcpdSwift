//
//  BunnyRegistrationViewModel.swift
//  BcpdSwiftDemoApp
//
//  ViewModel for BCPD Demo
//

import SwiftUI
import SceneKit
import simd
import Combine
import BCPD

enum DemoKernelOption: String, CaseIterable, Identifiable {
    case inverseMultiquadric = "Inverse Multiquadric"
    case gaussian = "Gaussian"
    case rationalQuadratic = "Rational Quadratic"
    case laplace = "Laplace"

    var id: String { rawValue }

    var kernelType: KernelType {
        switch self {
        case .inverseMultiquadric:
            return .inverseMultiquadric
        case .gaussian:
            return .gaussian
        case .rationalQuadratic:
            return .rationalQuadratic
        case .laplace:
            return .laplace
        }
    }
}

struct DemoBCPDSettings: Equatable {
    var omega: Double
    var lambda: Double
    var kappa: Double
    var gamma: Double
    var beta: Double
    var maxIterations: Int
    var tolerance: Double
    var kernel: DemoKernelOption
    var useDebias: Bool
    var verbose: Bool

    static let rigidDefault = DemoBCPDSettings(
        omega: 0.1,
        lambda: 1e10,
        kappa: 1e80,
        gamma: 0.5,
        beta: 1.0,
        maxIterations: 100,
        tolerance: 1e-4,
        kernel: .inverseMultiquadric,
        useDebias: false,
        verbose: false
    )

    var parameters: BCPDParameters {
        var parameters = BCPDParameters()
        parameters.omega = omega
        parameters.lambda = lambda
        parameters.kappa = kappa
        parameters.gamma = gamma
        parameters.beta = beta
        parameters.maxIterations = maxIterations
        parameters.tolerance = tolerance
        parameters.kernelType = kernel.kernelType
        parameters.useDebias = useDebias
        parameters.verbose = verbose
        return parameters
    }
}

@MainActor
class BunnyRegistrationViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var statusMessage = "Click 'Load Bunny' to start"
    @Published var isLoading = false
    @Published var hasError = false

    @Published var comparisonScene = SCNScene()

    @Published var registrationResult: BCPDResult?
    @Published var isDetailedMode = false
    @Published var parameterSettings = DemoBCPDSettings.rigidDefault

    // MARK: - Private Properties

    private var originalPoints: [Point3D] = []
    private var transformedPoints: [Point3D] = []
    private var groundTruthPoints: [Point3D] = []

    private var transformScale: Double = 1.0
    private var transformRotation = simd_double3x3(1)
    private var transformTranslation = simd_double3(0, 0, 0)

    // Bunny URL
    private let bunnyURL = URL(string: "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj")!

    // MARK: - Computed Properties

    var isReady: Bool {
        !originalPoints.isEmpty && !isLoading
    }

    var canRunBCPD: Bool {
        !transformedPoints.isEmpty && !isLoading
    }

    var isUsingRigidDefaults: Bool {
        parameterSettings == .rigidDefault
    }

    var parameterPresetName: String {
        isUsingRigidDefaults ? "Rigid Default" : "Custom"
    }

    var parameterSummaryText: String {
        """
        Preset: \(parameterPresetName)
        Kernel: \(parameterSettings.kernel.rawValue)
        omega \(formatDecimal(parameterSettings.omega)), lambda \(formatScientific(parameterSettings.lambda)), gamma \(formatDecimal(parameterSettings.gamma)), beta \(formatDecimal(parameterSettings.beta))
        kappa \(formatScientific(parameterSettings.kappa)), maxIter \(parameterSettings.maxIterations), tol \(formatScientific(parameterSettings.tolerance))
        """
    }

    // MARK: - Methods

    func loadBunny() async {
        isLoading = true
        hasError = false
        statusMessage = "Loading bunny model..."

        do {
            // Download and load OBJ file (with caching)
            let vertices = try await downloadAndLoadOBJ(from: bunnyURL)

            // Downsample to manageable size for demo
            let sampledVertices = downsample(vertices, to: min(vertices.count, 500))

            originalPoints = sampledVertices
            groundTruthPoints = sampledVertices

            updateComparisonScene()

            statusMessage = "Loaded \(originalPoints.count) points. Click 'Apply Random Transform'"
            isLoading = false

        } catch {
            statusMessage = "Error loading bunny: \(error.localizedDescription)"
            hasError = true
            isLoading = false
        }
    }

    func applyRandomTransform() {
        guard !originalPoints.isEmpty else { return }

        // Generate random transformation
        let (scale, rotation, translation) = randomTransform()

        transformScale = scale
        transformRotation = rotation
        transformTranslation = translation

        // Apply transformation
        transformedPoints = applyTransform(
            to: originalPoints,
            scale: scale,
            rotation: rotation,
            translation: translation
        )

        // Add small noise
        transformedPoints = addNoise(to: transformedPoints, stdDev: 0.01)

        updateComparisonScene()

        statusMessage = """
        Applied random transform:
        Scale: \(String(format: "%.3f", scale)),
        Rotation: (\(String(format: "%.2f", rotation[0][0])), ...),
        Translation: (\(String(format: "%.2f", translation.x)), \(String(format: "%.2f", translation.y)), \(String(format: "%.2f", translation.z)))
        Click 'Run BCPD' to register
        """
    }

    func runBCPD() async {
        guard !transformedPoints.isEmpty else { return }

        isLoading = true
        let parameters = parameterSettings.parameters
        statusMessage = "Running BCPD registration with \(parameterPresetName.lowercased()) parameters..."

        do {
            // Convert Point3D arrays to PointCloud
            let sourceCloud = PointCloud(points3D: groundTruthPoints)
            let targetCloud = PointCloud(points3D: transformedPoints)

            // Run BCPD registration
            let bcpd = BCPD()
            let result = try await bcpd.register(
                source: sourceCloud,
                target: targetCloud,
                parameters: parameters
            )

            // Convert registered points back to Point3D array
            let registeredPoints = result.deformedSource.toPoints3D()

            // Store result
            registrationResult = result

            updateComparisonScene(registeredPoints: registeredPoints)

            statusMessage = """
            Registration completed!
            Iterations: \(result.iterations)
            Residual: \(String(format: "%.6f", result.residual))
            """

        } catch {
            statusMessage = "Registration failed: \(error.localizedDescription)"
            hasError = true
        }

        isLoading = false
    }

    func reset() {
        originalPoints = []
        transformedPoints = []
        groundTruthPoints = []
        registrationResult = nil
        isDetailedMode = false
        parameterSettings = .rigidDefault

        comparisonScene = SCNScene()

        statusMessage = "Click 'Load Bunny' to start"
        hasError = false
    }

    func resetParametersToRigidDefaults() {
        parameterSettings = .rigidDefault
    }

    // MARK: - Helper Methods

    private func downloadAndLoadOBJ(from url: URL) async throws -> [Point3D] {
        // Check cache
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let cacheURL = cacheDir.appendingPathComponent("bunny.obj")

        if FileManager.default.fileExists(atPath: cacheURL.path) {
            print("Loading from cache")
            let content = try String(contentsOf: cacheURL, encoding: .utf8)
            return parseOBJ(content)
        }

        // Download
        print("Downloading bunny model...")
        let (data, _) = try await URLSession.shared.data(from: url)

        // Save to cache
        try data.write(to: cacheURL)

        guard let content = String(data: data, encoding: .utf8) else {
            throw NSError(domain: "OBJLoader", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid encoding"])
        }

        return parseOBJ(content)
    }

    private func parseOBJ(_ content: String) -> [Point3D] {
        var vertices: [Point3D] = []

        for line in content.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("v ") else { continue }

            let components = trimmed.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            guard components.count >= 4,
                  let x = Double(components[1]),
                  let y = Double(components[2]),
                  let z = Double(components[3]) else { continue }

            vertices.append(Point3D(x: x, y: y, z: z))
        }

        return vertices
    }

    private func downsample(_ points: [Point3D], to count: Int) -> [Point3D] {
        guard count < points.count else { return points }
        return Array(points.shuffled().prefix(count))
    }

    private func randomTransform() -> (Double, simd_double3x3, simd_double3) {
        let scale = Double.random(in: 0.9...1.1)
        let rx = Double.random(in: 0...(Double.pi / 4))
        let ry = Double.random(in: 0...(Double.pi / 4))
        let rz = Double.random(in: 0...(Double.pi / 4))

        let rotation = rotationMatrix(rx: rx, ry: ry, rz: rz)
        let translation = simd_double3(
            Double.random(in: -0.3...0.3),
            Double.random(in: -0.3...0.3),
            Double.random(in: -0.3...0.3)
        )

        return (scale, rotation, translation)
    }

    private func rotationMatrix(rx: Double, ry: Double, rz: Double) -> simd_double3x3 {
        let Rx = simd_double3x3(
            simd_double3(1, 0, 0),
            simd_double3(0, cos(rx), -sin(rx)),
            simd_double3(0, sin(rx), cos(rx))
        )

        let Ry = simd_double3x3(
            simd_double3(cos(ry), 0, sin(ry)),
            simd_double3(0, 1, 0),
            simd_double3(-sin(ry), 0, cos(ry))
        )

        let Rz = simd_double3x3(
            simd_double3(cos(rz), -sin(rz), 0),
            simd_double3(sin(rz), cos(rz), 0),
            simd_double3(0, 0, 1)
        )

        return Rz * Ry * Rx
    }

    private func applyTransform(
        to points: [Point3D],
        scale: Double,
        rotation: simd_double3x3,
        translation: simd_double3
    ) -> [Point3D] {
        points.map { point in
            let p = simd_double3(point.x, point.y, point.z)
            let transformed = scale * (rotation * p) + translation
            return Point3D(x: transformed.x, y: transformed.y, z: transformed.z)
        }
    }

    private func addNoise(to points: [Point3D], stdDev: Double) -> [Point3D] {
        points.map { point in
            Point3D(
                x: point.x + Double.random(in: -stdDev...stdDev),
                y: point.y + Double.random(in: -stdDev...stdDev),
                z: point.z + Double.random(in: -stdDev...stdDev)
            )
        }
    }

    // MARK: - Scene Updates

    private func updateComparisonScene(registeredPoints: [Point3D]? = nil) {
        let resolvedRegisteredPoints: [Point3D]
        if let registeredPoints {
            resolvedRegisteredPoints = registeredPoints
        } else if let registrationResult {
            resolvedRegisteredPoints = registrationResult.deformedSource.toPoints3D()
        } else {
            resolvedRegisteredPoints = []
        }

        comparisonScene = createComparisonScene(
            groundTruth: groundTruthPoints,
            transformed: transformedPoints,
            registered: resolvedRegisteredPoints
        )
    }

    private func createComparisonScene(
        groundTruth: [Point3D],
        transformed: [Point3D],
        registered: [Point3D]
    ) -> SCNScene {
        let scene = SCNScene()

        let allPoints = groundTruth + transformed + registered
        let bounds = computeBounds(for: allPoints)
        let diagonal = max(bounds.max - bounds.min, simd_double3(repeating: 0.001))
        let maxExtent = max(diagonal.x, max(diagonal.y, diagonal.z))
        let pointRadius = max(0.003, maxExtent * 0.008)
        let axisLength = max(0.2, maxExtent * 0.35)

        scene.rootNode.addChildNode(
            makePointCloudNode(points: groundTruth, color: .systemBlue, radius: pointRadius, opacity: 0.55)
        )
        scene.rootNode.addChildNode(
            makePointCloudNode(points: transformed, color: .systemOrange, radius: pointRadius, opacity: 0.9)
        )
        scene.rootNode.addChildNode(
            makePointCloudNode(points: registered, color: .systemGreen, radius: pointRadius, opacity: 0.85)
        )

        scene.rootNode.addChildNode(makeAxisGizmo(length: axisLength, radius: pointRadius * 0.35))

        let center = (bounds.min + bounds.max) * 0.5
        let cameraDistance = max(maxExtent * 2.4, 1.5)

        let cameraNode = SCNNode()
        let camera = SCNCamera()
        camera.fieldOfView = 45
        camera.automaticallyAdjustsZRange = true
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(
            x: Float(center.x + maxExtent * 0.5),
            y: Float(center.y + maxExtent * 0.4),
            z: Float(center.z + cameraDistance)
        )
        cameraNode.look(at: SCNVector3(Float(center.x), Float(center.y), Float(center.z)))
        scene.rootNode.addChildNode(cameraNode)

        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light?.type = .omni
        lightNode.light?.intensity = 1200
        lightNode.position = SCNVector3(
            x: Float(center.x + cameraDistance * 0.3),
            y: Float(center.y + cameraDistance * 0.8),
            z: Float(center.z + cameraDistance * 0.7)
        )
        scene.rootNode.addChildNode(lightNode)

        let ambientLightNode = SCNNode()
        ambientLightNode.light = SCNLight()
        ambientLightNode.light?.type = .ambient
        ambientLightNode.light?.color = UIColor(white: 0.25, alpha: 1.0)
        scene.rootNode.addChildNode(ambientLightNode)

        return scene
    }

    private func makePointCloudNode(
        points: [Point3D],
        color: UIColor,
        radius: Double,
        opacity: CGFloat
    ) -> SCNNode {
        let container = SCNNode()

        for point in points {
            let sphere = SCNSphere(radius: radius)
            sphere.firstMaterial?.diffuse.contents = color
            sphere.firstMaterial?.emission.contents = color.withAlphaComponent(0.35)
            sphere.firstMaterial?.lightingModel = .constant
            sphere.firstMaterial?.transparency = opacity

            let node = SCNNode(geometry: sphere)
            node.position = SCNVector3(Float(point.x), Float(point.y), Float(point.z))
            container.addChildNode(node)
        }

        return container
    }

    private func makeAxisGizmo(length: Double, radius: Double) -> SCNNode {
        let gizmoNode = SCNNode()
        gizmoNode.addChildNode(makeAxisNode(color: .systemRed, axis: .x, length: length, radius: radius))
        gizmoNode.addChildNode(makeAxisNode(color: .systemGreen, axis: .y, length: length, radius: radius))
        gizmoNode.addChildNode(makeAxisNode(color: .systemBlue, axis: .z, length: length, radius: radius))
        return gizmoNode
    }

    private enum GizmoAxis {
        case x
        case y
        case z
    }

    private func makeAxisNode(
        color: UIColor,
        axis: GizmoAxis,
        length: Double,
        radius: Double
    ) -> SCNNode {
        let axisNode = SCNNode()

        let shaft = SCNCylinder(radius: radius, height: length * 0.82)
        shaft.firstMaterial?.diffuse.contents = color
        shaft.firstMaterial?.emission.contents = color.withAlphaComponent(0.25)
        shaft.firstMaterial?.lightingModel = .constant
        let shaftNode = SCNNode(geometry: shaft)

        let tip = SCNCone(topRadius: 0, bottomRadius: radius * 2.5, height: length * 0.18)
        tip.firstMaterial?.diffuse.contents = color
        tip.firstMaterial?.emission.contents = color.withAlphaComponent(0.25)
        tip.firstMaterial?.lightingModel = .constant
        let tipNode = SCNNode(geometry: tip)

        switch axis {
        case .x:
            axisNode.eulerAngles = SCNVector3(0, 0, -Float.pi / 2)
            shaftNode.position = SCNVector3(Float(length * 0.41), 0, 0)
            tipNode.position = SCNVector3(Float(length * 0.91), 0, 0)
        case .y:
            shaftNode.position = SCNVector3(0, Float(length * 0.41), 0)
            tipNode.position = SCNVector3(0, Float(length * 0.91), 0)
        case .z:
            axisNode.eulerAngles = SCNVector3(Float.pi / 2, 0, 0)
            shaftNode.position = SCNVector3(0, 0, Float(length * 0.41))
            tipNode.position = SCNVector3(0, 0, Float(length * 0.91))
        }

        axisNode.addChildNode(shaftNode)
        axisNode.addChildNode(tipNode)
        return axisNode
    }

    private func computeBounds(for points: [Point3D]) -> (min: simd_double3, max: simd_double3) {
        guard let first = points.first else {
            return (simd_double3(-0.5, -0.5, -0.5), simd_double3(0.5, 0.5, 0.5))
        }

        var minValue = simd_double3(first.x, first.y, first.z)
        var maxValue = minValue

        for point in points.dropFirst() {
            let vector = simd_double3(point.x, point.y, point.z)
            minValue = simd.min(minValue, vector)
            maxValue = simd.max(maxValue, vector)
        }

        return (minValue, maxValue)
    }

    private func formatDecimal(_ value: Double) -> String {
        String(format: "%.4g", value)
    }

    private func formatScientific(_ value: Double) -> String {
        String(format: "%.2e", value)
    }
}

// MARK: - PointCloud Extensions

extension PointCloud {
    func toPoints3D() -> [Point3D] {
        var points: [Point3D] = []
        points.reserveCapacity(count)

        for i in 0..<count {
            let offset = i * dimension
            let x = data[offset]
            let y = data[offset + 1]
            let z = data[offset + 2]
            points.append(Point3D(x: x, y: y, z: z))
        }

        return points
    }
}
