import Foundation

// MARK: - Special Functions

/// Digamma function (ψ): derivative of log Gamma function
/// Uses recurrence ψ(x+1) = ψ(x) + 1/x to shift x to large value,
/// then applies asymptotic expansion. Matches scipy.special.psi.
public func digamma(_ x: Double) -> Double {
    var val = x
    var result: Double = 0.0

    // Use recurrence to shift x >= 6
    while val < 6.0 {
        result -= 1.0 / val
        val += 1.0
    }

    // Asymptotic expansion for large x
    let invX = 1.0 / val
    let invX2 = invX * invX

    result += log(val) - 0.5 * invX
    result -= invX2 * (1.0/12.0
                     - invX2 * (1.0/120.0
                     - invX2 * (1.0/252.0
                     - invX2 * (1.0/240.0
                     - invX2 * (1.0/132.0)))))

    return result
}

// MARK: - Statistics

/// Compute mean of array
public func mean(_ data: [Double]) -> Double {
    guard !data.isEmpty else { return 0.0 }
    return data.reduce(0.0, +) / Double(data.count)
}

/// Compute variance of array
public func variance(_ data: [Double]) -> Double {
    guard data.count > 1 else { return 0.0 }
    let m = mean(data)
    let squaredDiffs = data.map { pow($0 - m, 2) }
    return squaredDiffs.reduce(0.0, +) / Double(data.count - 1)
}

/// Compute standard deviation
public func standardDeviation(_ data: [Double]) -> Double {
    sqrt(variance(data))
}

// MARK: - Point Cloud Utilities

/// Compute centroid of point cloud
public func centroid(_ points: PointCloud) -> [Double] {
    var center = [Double](repeating: 0, count: points.dimension)

    for i in 0..<points.count {
        for d in 0..<points.dimension {
            center[d] += points[i, d]
        }
    }

    let count = Double(points.count)
    for d in 0..<points.dimension {
        center[d] /= count
    }

    return center
}

/// Compute bounding box volume (for normalization)
public func volume(_ points: PointCloud) -> Double {
    guard points.count > 0 else { return 1.0 }

    var minBounds = [Double](repeating: .infinity, count: points.dimension)
    var maxBounds = [Double](repeating: -.infinity, count: points.dimension)

    for i in 0..<points.count {
        for d in 0..<points.dimension {
            let value = points[i, d]
            minBounds[d] = min(minBounds[d], value)
            maxBounds[d] = max(maxBounds[d], value)
        }
    }

    var vol = 1.0
    for d in 0..<points.dimension {
        vol *= (maxBounds[d] - minBounds[d])
    }

    return vol > 0 ? vol : 1.0
}

/// Normalize point cloud to unit scale
public func normalize(_ points: PointCloud) -> (normalized: PointCloud, scale: Double, translation: [Double]) {
    let center = centroid(points)
    var normalized = points

    // Translate to origin
    for i in 0..<points.count {
        for d in 0..<points.dimension {
            normalized[i, d] = points[i, d] - center[d]
        }
    }

    // Compute scale (max distance from origin)
    var maxDist: Double = 0
    for i in 0..<normalized.count {
        var distSq: Double = 0
        for d in 0..<normalized.dimension {
            distSq += pow(normalized[i, d], 2)
        }
        maxDist = max(maxDist, sqrt(distSq))
    }

    let scale = maxDist > 0 ? maxDist : 1.0

    // Scale to unit sphere
    for i in 0..<normalized.count {
        for d in 0..<normalized.dimension {
            normalized[i, d] /= scale
        }
    }

    return (normalized, scale, center)
}

// MARK: - Normalization Options

public enum NormalizationMode {
    case separate  // Normalize X and Y separately (default)
    case useTarget // Normalize using target's scale and location
    case useSource // Normalize using source's scale and location
    case none      // No normalization
}

/// Normalize source and target point clouds
public func normalizePointClouds(
    source: PointCloud,
    target: PointCloud,
    mode: NormalizationMode = .separate
) -> (source: PointCloud, target: PointCloud, sourceScale: Double, sourceTranslation: [Double], targetScale: Double, targetTranslation: [Double]) {

    switch mode {
    case .separate:
        let (normSource, scaleS, transS) = normalize(source)
        let (normTarget, scaleT, transT) = normalize(target)
        return (normSource, normTarget, scaleS, transS, scaleT, transT)

    case .useTarget:
        let center = centroid(target)
        let (_, scale, _) = normalize(target)

        var normSource = source
        var normTarget = target

        // Apply target's transformation to both
        for i in 0..<source.count {
            for d in 0..<source.dimension {
                normSource[i, d] = (source[i, d] - center[d]) / scale
            }
        }

        for i in 0..<target.count {
            for d in 0..<target.dimension {
                normTarget[i, d] = (target[i, d] - center[d]) / scale
            }
        }

        return (normSource, normTarget, scale, center, scale, center)

    case .useSource:
        let center = centroid(source)
        let (_, scale, _) = normalize(source)

        var normSource = source
        var normTarget = target

        // Apply source's transformation to both
        for i in 0..<source.count {
            for d in 0..<source.dimension {
                normSource[i, d] = (source[i, d] - center[d]) / scale
            }
        }

        for i in 0..<target.count {
            for d in 0..<target.dimension {
                normTarget[i, d] = (target[i, d] - center[d]) / scale
            }
        }

        return (normSource, normTarget, scale, center, scale, center)

    case .none:
        let zeroTrans = [Double](repeating: 0, count: source.dimension)
        return (source, target, 1.0, zeroTrans, 1.0, zeroTrans)
    }
}

// MARK: - Random Utilities

/// Generate random permutation of integers 0..<n
public func randomPermutation(_ n: Int) -> [Int] {
    var indices = Array(0..<n)
    for i in stride(from: n - 1, through: 1, by: -1) {
        let j = Int.random(in: 0...i)
        indices.swapAt(i, j)
    }
    return indices
}

/// Random sampling without replacement
public func randomSample(_ n: Int, k: Int) -> [Int] {
    guard k <= n else { return Array(0..<n) }
    let perm = randomPermutation(n)
    return Array(perm.prefix(k))
}

// MARK: - Product Function

/// Compute product of array elements
public func product(_ array: [Double]) -> Double {
    array.reduce(1.0, *)
}
