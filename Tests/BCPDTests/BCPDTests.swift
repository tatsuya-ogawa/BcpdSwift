import XCTest
import Foundation
@testable import BCPD

final class BCPDTests: XCTestCase {

    // MARK: - Point Cloud Tests

    func testPointCloudCreation() {
        let points = [
            Point([1.0, 2.0, 3.0]),
            Point([4.0, 5.0, 6.0]),
            Point([7.0, 8.0, 9.0])
        ]
        let cloud = PointCloud(points: points)
        XCTAssertEqual(cloud.dimension, 3)
        XCTAssertEqual(cloud.count, 3)
        XCTAssertEqual(cloud[0, 0], 1.0)
        XCTAssertEqual(cloud[1, 1], 5.0)
        XCTAssertEqual(cloud[2, 2], 9.0)
    }

    // MARK: - Kernel Function Tests

    func testGaussianKernel() {
        let kernel = GaussianKernel()
        let value = kernel.compute(x: [0, 0, 0], y: [1, 0, 0], beta: 1.0)
        XCTAssertEqual(value, exp(-0.5), accuracy: 1e-10)
    }

    func testInverseMultiquadricKernel() {
        let kernel = InverseMultiquadricKernel()
        let value = kernel.compute(x: [0, 0, 0], y: [1, 0, 0], beta: 1.0)
        XCTAssertEqual(value, 1.0 / sqrt(2.0), accuracy: 1e-10)
    }

    // MARK: - Linear Algebra Tests

    func testMatrixMultiplication() {
        // Column-major: A = [[1,3,5],[2,4,6]] (2x3)
        let A = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        // Column-major: B = [[7,9,11],[8,10,12]] (3x2)
        let B = Matrix(rows: 3, cols: 2, data: [7, 8, 9, 10, 11, 12])
        let C = matrixMultiply(A, B)
        XCTAssertEqual(C.rows, 2)
        XCTAssertEqual(C.cols, 2)
        // C[0,0] = 1*7 + 3*9 + 5*11 = 7+27+55 = 89... hmm, let's just verify the operation works
        // A[0,0]=1, A[0,1]=3, A[0,2]=5; A[1,0]=2, A[1,1]=4, A[1,2]=6
        // B[0,0]=7, B[1,0]=8, B[2,0]=9; B[0,1]=10, B[1,1]=11, B[2,1]=12
        // C[0,0] = 1*7 + 3*8 + 5*9 = 7+24+45 = 76
        // C[1,0] = 2*7 + 4*8 + 6*9 = 14+32+54 = 100
        XCTAssertEqual(C[0, 0], 76.0, accuracy: 1e-10)
        XCTAssertEqual(C[1, 0], 100.0, accuracy: 1e-10)
    }

    func testSolvePositiveDefinite() {
        let A = Matrix(rows: 2, cols: 2, data: [4, 12, 12, 37])
        let b = [8.0, 24.0]
        let x = solvePositiveDefinite(A, b)
        XCTAssertNotNil(x)
        if let x = x {
            let result = matrixVectorMultiply(A, x)
            for i in 0..<b.count {
                XCTAssertEqual(result[i], b[i], accuracy: 1e-6)
            }
        }
    }

    func testDigamma() {
        let val1 = digamma(1.0)
        let val2 = digamma(2.0)
        XCTAssertEqual(val1, -0.5772156649015329, accuracy: 1e-6)
        XCTAssertEqual(val2, 0.4227843350984671, accuracy: 1e-6)
    }

    // MARK: - SVD Tests

    func testSVDDecomposition() {
        // Test with a known matrix
        let A = Matrix(rows: 2, cols: 2, data: [3.0, 1.0, 1.0, 3.0])
        guard let svd = svdDecomposition(A) else {
            XCTFail("SVD failed")
            return
        }
        // Reconstruct: A = U * diag(S) * Vt
        var sigmaMatrix = Matrix(rows: 2, cols: 2)
        sigmaMatrix[0, 0] = svd.S[0]
        sigmaMatrix[1, 1] = svd.S[1]
        let US = matrixMultiply(svd.U, sigmaMatrix)
        let reconstructed = matrixMultiply(US, svd.Vt)
        for i in 0..<2 {
            for j in 0..<2 {
                XCTAssertEqual(reconstructed[i, j], A[i, j], accuracy: 1e-10)
            }
        }
    }

    // MARK: - Simple Registration Test

    func testSimpleTranslation() throws {
        // Source: unit square
        let source = PointCloud(points: [
            Point([0, 0, 0]),
            Point([1, 0, 0]),
            Point([0, 1, 0]),
            Point([1, 1, 0]),
            Point([0.5, 0.5, 0])
        ])

        // Target: same but translated by (0.1, 0.1, 0)
        let target = PointCloud(points: [
            Point([0.1, 0.1, 0]),
            Point([1.1, 0.1, 0]),
            Point([0.1, 1.1, 0]),
            Point([1.1, 1.1, 0]),
            Point([0.6, 0.6, 0])
        ])

        let bcpd = BCPD()
        let result = try bcpd.register(
            source: source,
            target: target,
            w: 0.0,
            maxIter: 50,
            tol: 1e-6,
            lambda: 1e10,
            kappa: 1e20,
            gamma: 1.0,
            beta: 1.0,
            kernelType: .inverseMultiquadric,
            useDebias: false,
            verbose: false
        )

        // Check that scale is close to 1
        XCTAssertEqual(result.scale, 1.0, accuracy: 0.1)

        // Check RMSE of result against target
        let rmse = computeRMSE(result: result.deformedSource, target: target)
        print("Simple translation test RMSE: \(rmse)")
        XCTAssertLessThan(rmse, 0.05, "RMSE should be small for simple translation")
    }

    // MARK: - Rotation + Translation Test

    func testRotationAndTranslation() throws {
        // Create source points
        var sourcePoints = [Point]()
        for i in 0..<10 {
            let angle = Double(i) * 2.0 * .pi / 10.0
            sourcePoints.append(Point([cos(angle), sin(angle), 0]))
        }
        let source = PointCloud(points: sourcePoints)

        // Apply a small rotation (15 degrees around Z) and translation
        let theta = 15.0 * .pi / 180.0
        let cosT = cos(theta)
        let sinT = sin(theta)
        let tx = 0.05, ty = 0.03

        var targetPoints = [Point]()
        for p in sourcePoints {
            let nx = cosT * p[0] - sinT * p[1] + tx
            let ny = sinT * p[0] + cosT * p[1] + ty
            targetPoints.append(Point([nx, ny, 0]))
        }
        let target = PointCloud(points: targetPoints)

        let bcpd = BCPD()
        let result = try bcpd.register(
            source: source,
            target: target,
            w: 0.0,
            maxIter: 100,
            tol: 1e-6,
            lambda: 1e10,
            kappa: 1e20,
            gamma: 1.0,
            beta: 1.0,
            kernelType: .inverseMultiquadric,
            useDebias: false,
            verbose: false
        )

        let rmse = computeRMSE(result: result.deformedSource, target: target)
        print("Rotation+Translation test RMSE: \(rmse)")
        XCTAssertLessThan(rmse, 0.01, "RMSE should be very small for rigid transform")
        XCTAssertEqual(result.scale, 1.0, accuracy: 0.05)
    }

    // MARK: - Nonrigid Registration Test

    func testNonrigidRegistration() throws {
        // Create source: grid of 3D points
        var sourcePoints = [Point]()
        for i in 0..<5 {
            for j in 0..<5 {
                sourcePoints.append(Point([Double(i) * 0.25, Double(j) * 0.25, 0]))
            }
        }
        let source = PointCloud(points: sourcePoints)

        // Create target: slightly deformed version
        var targetPoints = [Point]()
        for p in sourcePoints {
            let dx = 0.02 * sin(p[0] * .pi)
            let dy = 0.02 * cos(p[1] * .pi)
            targetPoints.append(Point([p[0] + dx + 0.05, p[1] + dy + 0.03, 0]))
        }
        let target = PointCloud(points: targetPoints)

        let bcpd = BCPD()
        let result = try bcpd.register(
            source: source,
            target: target,
            w: 0.0,
            maxIter: 100,
            tol: 1e-6,
            lambda: 2.0,
            kappa: 1e20,
            gamma: 1.0,
            beta: 1.0,
            kernelType: .inverseMultiquadric,
            useDebias: false,
            verbose: false
        )

        let rmse = computeRMSE(result: result.deformedSource, target: target)
        print("Nonrigid test RMSE: \(rmse)")
        XCTAssertLessThan(rmse, 0.05, "Nonrigid registration should achieve reasonable RMSE")
    }

    // MARK: - Verification Test (matches bcpd_lab verification)
    // Uses bunny-like test data to verify BCPD matches C implementation

    func testBunnyDataRigidParams() throws {
        // Load test data from bundled zip
        let data = try getTestData()
        let xPath = data.xPath
        let yPath = data.yPath

        let target = try loadPointCloud(path: xPath)
        let source = try loadPointCloud(path: yPath)

        print("Target points: \(target.count), Source points: \(source.count)")

        let rmseBefore = computeRMSE_bruteForce(source: source, target: target)
        print("RMSE Before: \(String(format: "%.6f", rmseBefore))")

        // Test with C-equivalent parameters (nearly rigid, high lambda)
        // This is the key verification case matching Test 2 from verify_fix.py
        let bcpd = BCPD()
        let result = try bcpd.register(
            source: source,
            target: target,
            w: 0.1,
            maxIter: 100,
            tol: 1e-5,  // relaxed for early termination
            lambda: 1e10,
            kappa: 1e80,
            gamma: 0.5,
            beta: 1.0,
            kernelType: .inverseMultiquadric,
            useDebias: false,
            verbose: true
        )

        let rmseAfter = computeRMSE_bruteForce(source: result.deformedSource, target: target)
        print("RMSE After (C params): \(String(format: "%.6f", rmseAfter))")
        print("Scale: \(String(format: "%.6f", result.scale))")
        print("Iterations: \(result.iterations)")

        // The C implementation achieves RMSE ~0.003603
        // The corrected probreg achieves RMSE ~0.003600
        // We require RMSE < 0.005 (within ~40% of C result)
        XCTAssertLessThan(rmseAfter, 0.005,
            "RMSE should be close to C implementation result (~0.0036)")

        // RMSE should improve from before registration
        XCTAssertLessThan(rmseAfter, rmseBefore,
            "Registration should improve RMSE")
    }

    func testBunnyDataDefaultParams() throws {
        let data = try getTestData()
        let xPath = data.xPath
        let yPath = data.yPath

        let target = try loadPointCloud(path: xPath)
        let source = try loadPointCloud(path: yPath)

        let rmseBefore = computeRMSE_bruteForce(source: source, target: target)

        // Test 1: Default nonrigid params (lmd=2.0, IMQ)
        let bcpd = BCPD()
        let result = try bcpd.register(
            source: source,
            target: target,
            w: 0.1,
            maxIter: 100,
            tol: 1e-4,  // relaxed for early termination
            lambda: 2.0,
            kappa: 1e20,
            gamma: 1.0,
            beta: 1.0,
            kernelType: .inverseMultiquadric,
            useDebias: false,
            verbose: true
        )

        let rmseAfter = computeRMSE_bruteForce(source: result.deformedSource, target: target)
        print("RMSE Before: \(String(format: "%.6f", rmseBefore))")
        print("RMSE After (lmd=2.0, IMQ): \(String(format: "%.6f", rmseAfter))")
        print("Scale: \(String(format: "%.6f", result.scale))")

        // probreg achieves ~0.007585 with these params
        XCTAssertLessThan(rmseAfter, 0.01,
            "RMSE with default params should be reasonable")
        XCTAssertLessThan(rmseAfter, rmseBefore,
            "Registration should improve RMSE")
    }

    // MARK: - Helpers

    func computeRMSE(result: PointCloud, target: PointCloud) -> Double {
        let tree = KDTree(points: target)
        var total: Double = 0
        for m in 0..<result.count {
            var point = [Double](repeating: 0, count: result.dimension)
            for d in 0..<result.dimension { point[d] = result[m, d] }
            let (_, dist) = tree.nearestNeighbor(to: point)
            total += dist
        }
        return total / Double(result.count)
    }

    func computeRMSE_bruteForce(source: PointCloud, target: PointCloud) -> Double {
        var total: Double = 0
        for m in 0..<source.count {
            var minDist = Double.infinity
            for n in 0..<target.count {
                var distSq: Double = 0
                for d in 0..<source.dimension {
                    let diff = source[m, d] - target[n, d]
                    distSq += diff * diff
                }
                minDist = min(minDist, sqrt(distSq))
            }
            total += minDist
        }
        return total / Double(source.count)
    }

    func getTestData() throws -> (xPath: String, yPath: String) {
        guard let url = Bundle.module.url(forResource: "bunny_data", withExtension: "zip") else {
            throw NSError(domain: "BCPDTests", code: 1, userInfo: [NSLocalizedDescriptionKey: "bunny_data.zip not found"])
        }
        
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("BCPDTests")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-o", url.path, "-d", tempDir.path]
        try process.run()
        process.waitUntilExit()
        
        return (
            xPath: tempDir.appendingPathComponent("X.txt").path,
            yPath: tempDir.appendingPathComponent("Y.txt").path
        )
    }

    func loadPointCloud(path: String) throws -> PointCloud {
        let content = try String(contentsOfFile: path, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        var points = [Point]()
        for line in lines {
            let values = line.trimmingCharacters(in: .whitespaces)
                .components(separatedBy: .whitespaces)
                .filter { !$0.isEmpty }
                .compactMap { Double($0) }
            if !values.isEmpty {
                points.append(Point(values))
            }
        }
        return PointCloud(points: points)
    }
}
