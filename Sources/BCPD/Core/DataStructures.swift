import Foundation
import simd

// MARK: - Point and PointCloud

/// 3D point representation
public struct Point3D: Equatable {
    public var x: Double
    public var y: Double
    public var z: Double

    public init(x: Double, y: Double, z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }

    public init(_ vector: SIMD3<Double>) {
        self.x = vector.x
        self.y = vector.y
        self.z = vector.z
    }

    public var simd: SIMD3<Double> {
        SIMD3<Double>(x, y, z)
    }

    public subscript(dimension: Int) -> Double {
        get {
            switch dimension {
            case 0: return x
            case 1: return y
            case 2: return z
            default: fatalError("Invalid dimension index: \(dimension)")
            }
        }
        set {
            switch dimension {
            case 0: x = newValue
            case 1: y = newValue
            case 2: z = newValue
            default: fatalError("Invalid dimension index: \(dimension)")
            }
        }
    }
}

/// N-dimensional point representation
public struct Point: Equatable {
    public var coordinates: [Double]

    public init(_ coordinates: [Double]) {
        self.coordinates = coordinates
    }

    public var dimension: Int {
        coordinates.count
    }

    public subscript(index: Int) -> Double {
        get { coordinates[index] }
        set { coordinates[index] = newValue }
    }
}

/// Point cloud representation
public struct PointCloud {
    /// Points stored as flat array in interleaved format: [x0,y0,z0, x1,y1,z1, ...]
    public var data: [Double]
    public let dimension: Int
    public let count: Int

    public init(dimension: Int, count: Int) {
        self.dimension = dimension
        self.count = count
        self.data = Array(repeating: 0.0, count: dimension * count)
    }

    public init(points: [Point]) {
        guard let first = points.first else {
            self.dimension = 3
            self.count = 0
            self.data = []
            return
        }
        self.dimension = first.dimension
        self.count = points.count
        self.data = Array(repeating: 0.0, count: dimension * count)
        for (i, point) in points.enumerated() {
            for d in 0..<dimension {
                self[i, d] = point[d]
            }
        }
    }

    public init(points3D: [Point3D]) {
        self.dimension = 3
        self.count = points3D.count
        self.data = Array(repeating: 0.0, count: 3 * count)
        for (i, point) in points3D.enumerated() {
            self[i, 0] = point.x
            self[i, 1] = point.y
            self[i, 2] = point.z
        }
    }

    /// Access point coordinate: pointCloud[pointIndex, dimensionIndex]
    public subscript(pointIndex: Int, dimension: Int) -> Double {
        get {
            data[dimension + self.dimension * pointIndex]
        }
        set {
            data[dimension + self.dimension * pointIndex] = newValue
        }
    }

    public func point(at index: Int) -> Point {
        var coords = [Double](repeating: 0.0, count: dimension)
        for d in 0..<dimension {
            coords[d] = self[index, d]
        }
        return Point(coords)
    }

    public var points: [Point] {
        (0..<count).map { point(at: $0) }
    }
}

// MARK: - Kernel Function Type

public enum KernelType {
    case gaussian
    case inverseMultiquadric
    case rationalQuadratic
    case laplace
    case geodesic(tau: Double)
}

// MARK: - Transformation Type

public enum TransformationType {
    case rigid
    case similarity
    case affine
    case nonrigid
    case similarityNonrigid
    case affineNonrigid
}

// MARK: - Registration Parameters

/// Parameters for BCPD registration
public struct BCPDParameters {
    /// Deformation control parameter (lambda)
    public var lambda: Double = 2.0
    /// Smoothing range parameter (beta)
    public var beta: Double = 1.0
    /// Outlier probability (omega/w)
    public var omega: Double = 0.0
    /// Regularization parameter (gamma)
    public var gamma: Double = 1.0
    /// Dirichlet prior parameter (kappa)
    public var kappa: Double = 1.0e20
    /// Maximum number of iterations
    public var maxIterations: Int = 50
    /// Convergence tolerance
    public var tolerance: Double = 0.001
    /// Kernel function type
    public var kernelType: KernelType = .inverseMultiquadric
    /// Whether to use debiasing
    public var useDebias: Bool = false
    /// Whether to print progress information
    public var verbose: Bool = false

    public init() {}
}
