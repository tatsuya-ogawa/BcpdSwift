import Foundation
import Accelerate

// MARK: - Kernel Function Protocol

/// Protocol for kernel functions used in BCPD
public protocol KernelFunction {
    /// Compute kernel value between two points
    /// - Parameters:
    ///   - x: First point coordinates
    ///   - y: Second point coordinates
    ///   - beta: Kernel bandwidth parameter
    /// - Returns: Kernel value
    func compute(x: [Double], y: [Double], beta: Double) -> Double

    /// Compute kernel value between two points (using data pointers for efficiency)
    /// - Parameters:
    ///   - xPtr: Pointer to first point
    ///   - yPtr: Pointer to second point
    ///   - dimension: Number of dimensions
    ///   - beta: Kernel bandwidth parameter
    /// - Returns: Kernel value
    func compute(xPtr: UnsafePointer<Double>, yPtr: UnsafePointer<Double>, dimension: Int, beta: Double) -> Double
}

// MARK: - Distance Functions

/// Compute L1 distance (Manhattan distance)
@inlinable
func distanceL1(_ x: UnsafePointer<Double>, _ y: UnsafePointer<Double>, dimension: Int) -> Double {
    var sum: Double = 0.0
    for i in 0..<dimension {
        sum += abs(x[i] - y[i])
    }
    return sum
}

/// Compute squared L2 distance (Euclidean distance squared)
@inlinable
func distanceL2Squared(_ x: UnsafePointer<Double>, _ y: UnsafePointer<Double>, dimension: Int) -> Double {
    var sum: Double = 0.0
    for i in 0..<dimension {
        let diff = x[i] - y[i]
        sum += diff * diff
    }
    return sum
}

/// Compute squared L2 distance using vDSP (optimized for larger dimensions)
@inlinable
func distanceL2SquaredVDSP(_ x: UnsafePointer<Double>, _ y: UnsafePointer<Double>, dimension: Int) -> Double {
    var diff = [Double](repeating: 0, count: dimension)
    var result: Double = 0.0

    // diff = x - y
    vDSP_vsubD(y, 1, x, 1, &diff, 1, vDSP_Length(dimension))

    // result = sum(diff^2)
    vDSP_svesqD(diff, 1, &result, vDSP_Length(dimension))

    return result
}

// MARK: - Kernel Implementations

/// Gaussian kernel: exp(-||x-y||²/(2β²))
public struct GaussianKernel: KernelFunction {
    public init() {}

    public func compute(x: [Double], y: [Double], beta: Double) -> Double {
        x.withUnsafeBufferPointer { xBuf in
            y.withUnsafeBufferPointer { yBuf in
                compute(xPtr: xBuf.baseAddress!, yPtr: yBuf.baseAddress!, dimension: x.count, beta: beta)
            }
        }
    }

    @inlinable
    public func compute(xPtr: UnsafePointer<Double>, yPtr: UnsafePointer<Double>, dimension: Int, beta: Double) -> Double {
        let distSq = dimension > 10
            ? distanceL2SquaredVDSP(xPtr, yPtr, dimension: dimension)
            : distanceL2Squared(xPtr, yPtr, dimension: dimension)
        return Foundation.exp(-distSq / (2.0 * beta * beta))
    }
}

/// Laplace kernel: exp(-|x-y|/β)
public struct LaplaceKernel: KernelFunction {
    public init() {}

    public func compute(x: [Double], y: [Double], beta: Double) -> Double {
        x.withUnsafeBufferPointer { xBuf in
            y.withUnsafeBufferPointer { yBuf in
                compute(xPtr: xBuf.baseAddress!, yPtr: yBuf.baseAddress!, dimension: x.count, beta: beta)
            }
        }
    }

    @inlinable
    public func compute(xPtr: UnsafePointer<Double>, yPtr: UnsafePointer<Double>, dimension: Int, beta: Double) -> Double {
        let dist = distanceL1(xPtr, yPtr, dimension: dimension)
        return Foundation.exp(-dist / beta)
    }
}

/// Inverse multiquadric kernel: (||x-y||²+β²)^(-1/2)
public struct InverseMultiquadricKernel: KernelFunction {
    public init() {}

    public func compute(x: [Double], y: [Double], beta: Double) -> Double {
        x.withUnsafeBufferPointer { xBuf in
            y.withUnsafeBufferPointer { yBuf in
                compute(xPtr: xBuf.baseAddress!, yPtr: yBuf.baseAddress!, dimension: x.count, beta: beta)
            }
        }
    }

    @inlinable
    public func compute(xPtr: UnsafePointer<Double>, yPtr: UnsafePointer<Double>, dimension: Int, beta: Double) -> Double {
        let distSq = dimension > 10
            ? distanceL2SquaredVDSP(xPtr, yPtr, dimension: dimension)
            : distanceL2Squared(xPtr, yPtr, dimension: dimension)
        return 1.0 / Foundation.sqrt(beta * beta + distSq)
    }
}

/// Rational quadratic kernel: 1 - ||x-y||²/(||x-y||²+β²)
public struct RationalQuadraticKernel: KernelFunction {
    public init() {}

    public func compute(x: [Double], y: [Double], beta: Double) -> Double {
        x.withUnsafeBufferPointer { xBuf in
            y.withUnsafeBufferPointer { yBuf in
                compute(xPtr: xBuf.baseAddress!, yPtr: yBuf.baseAddress!, dimension: x.count, beta: beta)
            }
        }
    }

    @inlinable
    public func compute(xPtr: UnsafePointer<Double>, yPtr: UnsafePointer<Double>, dimension: Int, beta: Double) -> Double {
        let distSq = dimension > 10
            ? distanceL2SquaredVDSP(xPtr, yPtr, dimension: dimension)
            : distanceL2Squared(xPtr, yPtr, dimension: dimension)
        return 1.0 - distSq / (distSq + beta * beta)
    }
}

// MARK: - Kernel Factory

/// Factory for creating kernel functions
public struct KernelFactory {
    public static func create(type: KernelType) -> KernelFunction {
        switch type {
        case .gaussian:
            return GaussianKernel()
        case .laplace:
            return LaplaceKernel()
        case .inverseMultiquadric:
            return InverseMultiquadricKernel()
        case .rationalQuadratic:
            return RationalQuadraticKernel()
        case .geodesic:
            fatalError("Geodesic kernel not yet implemented")
        }
    }
}

// MARK: - Gram Matrix Computation

/// Compute Gram matrix G[i,j] = kernel(y_i, y_j)
public func computeGramMatrix(
    points: PointCloud,
    kernel: KernelFunction,
    beta: Double
) -> [Double] {
    let M = points.count
    let D = points.dimension
    var G = [Double](repeating: 0, count: M * M)

    points.data.withUnsafeBufferPointer { buf in
        let basePtr = buf.baseAddress!

        // Compute upper triangle (including diagonal)
        for i in 0..<M {
            let yi = basePtr.advanced(by: D * i)
            for j in i..<M {
                let yj = basePtr.advanced(by: D * j)
                let value = kernel.compute(xPtr: yi, yPtr: yj, dimension: D, beta: beta)
                G[i + M * j] = value
                G[j + M * i] = value  // Symmetric
            }
        }
    }

    return G
}

/// Compute Gram matrix with parallel execution
public func computeGramMatrixParallel(
    points: PointCloud,
    kernel: KernelFunction,
    beta: Double
) -> [Double] {
    let M = points.count
    let D = points.dimension
    var G = [Double](repeating: 0, count: M * M)

    points.data.withUnsafeBufferPointer { buf in
        let basePtr = buf.baseAddress!

        DispatchQueue.concurrentPerform(iterations: M) { i in
            let yi = basePtr.advanced(by: D * i)
            for j in i..<M {
                let yj = basePtr.advanced(by: D * j)
                let value = kernel.compute(xPtr: yi, yPtr: yj, dimension: D, beta: beta)
                G[i + M * j] = value
                if i != j {
                    G[j + M * i] = value  // Symmetric
                }
            }
        }
    }

    return G
}
