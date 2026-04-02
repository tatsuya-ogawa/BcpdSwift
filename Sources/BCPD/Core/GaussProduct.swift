import Foundation
import Accelerate

// MARK: - Gauss Product Flags

struct GaussProductFlags: OptionSet {
    let rawValue: Int

    static let local = GaussProductFlags(rawValue: 1 << 1)  // Use KD-tree local search
    static let reuse = GaussProductFlags(rawValue: 1 << 2)  // Reuse previous computation
    static let trans = GaussProductFlags(rawValue: 1 << 3)  // Transpose operation
    static let build = GaussProductFlags(rawValue: 1 << 4)  // Build KD-tree
}

// MARK: - Gauss Product Computation

/// Compute Gaussian-weighted product for BCPD
/// This is a core operation in the E-step of BCPD
///
/// Computes: w[m] = Σ_n q[n] * exp(-||y_m - x_n||²/(2h²))
///           U[m,d] = Σ_n q[n] * exp(-||y_m - x_n||²/(2h²)) * x[n,d]
class GaussProduct {

    /// Compute Gaussian product
    /// - Parameters:
    ///   - Y: Source points (D × M)
    ///   - X: Target points (D × N)
    ///   - q: Weight vector (length N or M depending on transpose)
    ///   - D: Dimension
    ///   - M: Number of source points
    ///   - N: Number of target points
    ///   - h: Gaussian bandwidth
    ///   - flags: Computation flags
    /// - Returns: (w, U) where w is weighted sum and U is weighted position
    static func compute(
        Y: PointCloud,
        X: PointCloud,
        q: [Double],
        h: Double,
        flags: GaussProductFlags = []
    ) -> (w: [Double], U: PointCloud?) {

        let D = Y.dimension
        let M = Y.count
        let N = X.count

        precondition(X.dimension == D, "Point clouds must have same dimension")

        if flags.contains(.trans) {
            // Transpose mode: compute q = P^T * 1
            precondition(q.count == M, "q length must equal M in transpose mode")
            return computeTranspose(Y: Y, X: X, q: q, h: h)
        } else {
            // Normal mode: compute w = P * q and U = P * X
            precondition(q.count == N, "q length must equal N in normal mode")
            return computeNormal(Y: Y, X: X, q: q, h: h)
        }
    }

    /// Normal mode: w = P * q, U = P * X
    private static func computeNormal(
        Y: PointCloud,
        X: PointCloud,
        q: [Double],
        h: Double
    ) -> (w: [Double], U: PointCloud) {

        let D = Y.dimension
        let M = Y.count
        let N = X.count

        var w = [Double](repeating: 0, count: M)
        var U = PointCloud(dimension: D, count: M)

        let kernel = GaussianKernel()
        let h2 = 2.0 * h * h

        Y.data.withUnsafeBufferPointer { yBuffer in
            X.data.withUnsafeBufferPointer { xBuffer in
                let yPtr = yBuffer.baseAddress!
                let xPtr = xBuffer.baseAddress!

                // For each source point m
                for m in 0..<M {
                    let ym = yPtr.advanced(by: D * m)
                    var wm: Double = 0
                    var um = [Double](repeating: 0, count: D)

                    // Sum over all target points n
                    for n in 0..<N {
                        let xn = xPtr.advanced(by: D * n)

                        // Compute Gaussian weight
                        let distSq = distanceL2Squared(ym, xn, dimension: D)
                        let gaussWeight = exp(-distSq / h2) * q[n]

                        wm += gaussWeight

                        // Accumulate weighted position
                        for d in 0..<D {
                            um[d] += gaussWeight * xn[d]
                        }
                    }

                    w[m] = wm

                    // Store weighted position
                    for d in 0..<D {
                        U[m, d] = um[d]
                    }
                }
            }
        }

        return (w, U)
    }

    /// Transpose mode: q = P^T * b
    private static func computeTranspose(
        Y: PointCloud,
        X: PointCloud,
        q: [Double],  // Here q is the weight for Y (length M)
        h: Double
    ) -> (w: [Double], U: PointCloud?) {

        let D = Y.dimension
        let M = Y.count
        let N = X.count

        var w = [Double](repeating: 0, count: N)

        let h2 = 2.0 * h * h

        Y.data.withUnsafeBufferPointer { yBuffer in
            X.data.withUnsafeBufferPointer { xBuffer in
                let yPtr = yBuffer.baseAddress!
                let xPtr = xBuffer.baseAddress!

                // For each target point n
                for n in 0..<N {
                    let xn = xPtr.advanced(by: D * n)
                    var wn: Double = 0

                    // Sum over all source points m
                    for m in 0..<M {
                        let ym = yPtr.advanced(by: D * m)

                        // Compute Gaussian weight
                        let distSq = distanceL2Squared(xn, ym, dimension: D)
                        let gaussWeight = exp(-distSq / h2) * q[m]

                        wn += gaussWeight
                    }

                    w[n] = wn
                }
            }
        }

        return (w, nil)
    }

    /// Parallel version for large point clouds
    static func computeParallel(
        Y: PointCloud,
        X: PointCloud,
        q: [Double],
        h: Double,
        flags: GaussProductFlags = []
    ) -> (w: [Double], U: PointCloud?) {

        let D = Y.dimension
        let M = Y.count
        let N = X.count

        if flags.contains(.trans) {
            return computeTransposeParallel(Y: Y, X: X, q: q, h: h)
        } else {
            return computeNormalParallel(Y: Y, X: X, q: q, h: h)
        }
    }

    private static func computeNormalParallel(
        Y: PointCloud,
        X: PointCloud,
        q: [Double],
        h: Double
    ) -> (w: [Double], U: PointCloud) {

        let D = Y.dimension
        let M = Y.count
        let N = X.count

        var w = [Double](repeating: 0, count: M)
        var U = PointCloud(dimension: D, count: M)

        let h2 = 2.0 * h * h

        Y.data.withUnsafeBufferPointer { yBuffer in
            X.data.withUnsafeBufferPointer { xBuffer in
                let yPtr = yBuffer.baseAddress!
                let xPtr = xBuffer.baseAddress!

                DispatchQueue.concurrentPerform(iterations: M) { m in
                    let ym = yPtr.advanced(by: D * m)
                    var wm: Double = 0
                    var um = [Double](repeating: 0, count: D)

                    for n in 0..<N {
                        let xn = xPtr.advanced(by: D * n)
                        let distSq = distanceL2Squared(ym, xn, dimension: D)
                        let gaussWeight = exp(-distSq / h2) * q[n]

                        wm += gaussWeight
                        for d in 0..<D {
                            um[d] += gaussWeight * xn[d]
                        }
                    }

                    w[m] = wm
                    for d in 0..<D {
                        U[m, d] = um[d]
                    }
                }
            }
        }

        return (w, U)
    }

    private static func computeTransposeParallel(
        Y: PointCloud,
        X: PointCloud,
        q: [Double],
        h: Double
    ) -> (w: [Double], U: PointCloud?) {

        let D = Y.dimension
        let M = Y.count
        let N = X.count

        var w = [Double](repeating: 0, count: N)

        let h2 = 2.0 * h * h

        Y.data.withUnsafeBufferPointer { yBuffer in
            X.data.withUnsafeBufferPointer { xBuffer in
                let yPtr = yBuffer.baseAddress!
                let xPtr = xBuffer.baseAddress!

                DispatchQueue.concurrentPerform(iterations: N) { n in
                    let xn = xPtr.advanced(by: D * n)
                    var wn: Double = 0

                    for m in 0..<M {
                        let ym = yPtr.advanced(by: D * m)
                        let distSq = distanceL2Squared(xn, ym, dimension: D)
                        let gaussWeight = exp(-distSq / h2) * q[m]

                        wn += gaussWeight
                    }

                    w[n] = wn
                }
            }
        }

        return (w, nil)
    }
}

// MARK: - Nystrom Approximation for Gauss Product

/// Nystrom method for low-rank approximation of Gaussian kernel matrix
class NystromGaussProduct {

    /// Compute Gauss product using Nystrom approximation
    /// - Parameters:
    ///   - Y: Source points
    ///   - X: Target points
    ///   - q: Weight vector
    ///   - h: Gaussian bandwidth
    ///   - rank: Number of samples for Nystrom approximation
    /// - Returns: Approximated (w, U)
    static func compute(
        Y: PointCloud,
        X: PointCloud,
        q: [Double],
        h: Double,
        rank: Int
    ) -> (w: [Double], U: PointCloud?) {

        let M = Y.count
        let N = X.count
        let P = min(rank, min(M, N))

        // Sample random indices
        let sampleIndices = randomSample(N, k: P)

        // Create sampled point cloud
        var sampledX = PointCloud(dimension: X.dimension, count: P)
        for (i, idx) in sampleIndices.enumerated() {
            for d in 0..<X.dimension {
                sampledX[i, d] = X[idx, d]
            }
        }

        // Compute K_YS: Gaussian kernel between Y and sampled X
        let kernel = GaussianKernel()
        var KYS = Matrix(rows: M, cols: P)

        for m in 0..<M {
            for p in 0..<P {
                let ym = Y.data[(Y.dimension * m)..<(Y.dimension * (m + 1))]
                let xp = sampledX.data[(sampledX.dimension * p)..<(sampledX.dimension * (p + 1))]
                KYS[m, p] = kernel.compute(x: Array(ym), y: Array(xp), beta: h)
            }
        }

        // Compute K_SS: Kernel matrix for sampled points
        var KSS = Matrix(rows: P, cols: P)
        for i in 0..<P {
            for j in i..<P {
                let xi = sampledX.data[(sampledX.dimension * i)..<(sampledX.dimension * (i + 1))]
                let xj = sampledX.data[(sampledX.dimension * j)..<(sampledX.dimension * (j + 1))]
                let value = kernel.compute(x: Array(xi), y: Array(xj), beta: h)
                KSS[i, j] = value
                KSS[j, i] = value
            }
        }

        // Nystrom approximation: K ≈ K_YS * K_SS^(-1) * K_SY
        // w ≈ K_YS * K_SS^(-1) * q_sampled

        var qSampled = [Double](repeating: 0, count: P)
        for (i, idx) in sampleIndices.enumerated() {
            qSampled[i] = q[idx]
        }

        guard let alpha = solvePositiveDefinite(KSS, qSampled) else {
            // Fallback to direct computation
            return GaussProduct.compute(Y: Y, X: X, q: q, h: h)
        }

        let w = matrixVectorMultiply(KYS, alpha)

        // Compute U if needed
        var U = PointCloud(dimension: X.dimension, count: M)
        for m in 0..<M {
            for d in 0..<X.dimension {
                var sum: Double = 0
                for (i, idx) in sampleIndices.enumerated() {
                    sum += alpha[i] * KYS[m, i] * X[idx, d]
                }
                U[m, d] = sum
            }
        }

        return (w, U)
    }
}

// MARK: - KD-Tree Accelerated Gauss Product

extension GaussProduct {

    /// Compute Gauss product using KD-tree acceleration (normal mode)
    static func computeNormalWithKDTree(
        Y: PointCloud,
        X: PointCloud,
        q: [Double],
        h: Double,
        kdTree: KDTree,
        radius: Double
    ) -> (w: [Double], U: PointCloud) {

        let D = Y.dimension
        let M = Y.count

        var w = [Double](repeating: 0, count: M)
        var U = PointCloud(dimension: D, count: M)

        let h2 = 2.0 * h * h
        let searchRadius = radius * h  // Adaptive radius based on sigma

        // For each source point
        for m in 0..<M {
            var ym = [Double](repeating: 0, count: D)
            for d in 0..<D {
                ym[d] = Y[m, d]
            }

            // Find neighbors within radius using KD-tree
            let neighbors = kdTree.neighborsWithinRadius(to: ym, radius: searchRadius)

            var wm: Double = 0
            var um = [Double](repeating: 0, count: D)

            // Compute weighted sum over neighbors only
            for n in neighbors {
                var distSq: Double = 0
                for d in 0..<D {
                    let diff = ym[d] - X[n, d]
                    distSq += diff * diff
                }

                let gaussWeight = exp(-distSq / h2) * q[n]
                wm += gaussWeight

                for d in 0..<D {
                    um[d] += gaussWeight * X[n, d]
                }
            }

            w[m] = wm
            for d in 0..<D {
                U[m, d] = um[d]
            }
        }

        return (w, U)
    }

    /// Compute Gauss product using KD-tree acceleration (transpose mode)
    static func computeTransposeWithKDTree(
        Y: PointCloud,
        X: PointCloud,
        q: [Double],
        h: Double,
        kdTree: KDTree,
        radius: Double
    ) -> (w: [Double], U: PointCloud?) {

        let D = Y.dimension
        let N = X.count

        var w = [Double](repeating: 0, count: N)

        let h2 = 2.0 * h * h
        let searchRadius = radius * h

        // For each target point
        for n in 0..<N {
            var xn = [Double](repeating: 0, count: D)
            for d in 0..<D {
                xn[d] = X[n, d]
            }

            // Find neighbors in Y within radius
            // Note: We'd need a KD-tree built on Y for this
            // For now, fall back to direct computation
            var wn: Double = 0

            for m in 0..<Y.count {
                var distSq: Double = 0
                for d in 0..<D {
                    let diff = xn[d] - Y[m, d]
                    distSq += diff * diff
                }

                if sqrt(distSq) <= searchRadius {
                    let gaussWeight = exp(-distSq / h2) * q[m]
                    wn += gaussWeight
                }
            }

            w[n] = wn
        }

        return (w, nil)
    }
}
