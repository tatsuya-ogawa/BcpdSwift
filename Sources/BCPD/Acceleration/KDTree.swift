import Foundation

// MARK: - KD Tree

/// KD-Tree for efficient nearest neighbor search
/// Based on the original C implementation from bcpd/base/kdtree.c
public class KDTree {

    /// Maximum tree depth
    private static let maxTreeDepth = 32

    /// Tree structure: [depth(N), left(N), right(N), root(1)]
    /// - depth[i]: depth of node i
    /// - left[i]: left child of node i (or -1)
    /// - right[i]: right child of node i (or -1)
    /// - root: index of root node (stored at index 3*N)
    private var tree: [Int]

    /// Points data
    private let points: PointCloud

    /// Number of points
    private let N: Int

    /// Dimension
    private let D: Int

    // MARK: - Initialization

    /// Build KD-tree from point cloud
    public init(points: PointCloud) {
        self.points = points
        self.N = points.count
        self.D = points.dimension
        self.tree = Array(repeating: -1, count: 3 * N + 1)

        build()
    }

    // MARK: - Tree Building

    private func build() {
        var indices = Array(0..<N)
        var workInt = Array(repeating: 0, count: 6 * N)
        var workDouble = Array(repeating: 0.0, count: 2 * N)

        buildRecursive(
            tree: &tree,
            indices: &indices,
            workInt: &workInt,
            workDouble: &workDouble,
            depth: 0
        )
    }

    private func buildRecursive(
        tree: inout [Int],
        indices: inout [Int],
        workInt: inout [Int],
        workDouble: inout [Double],
        depth: Int
    ) {
        let size = indices.count
        guard size > 0 else { return }

        let axis = depth % D
        let medianIndex = size / 2

        // Sort indices by current axis
        indices.sort { i1, i2 in
            points[i1, axis] < points[i2, axis]
        }

        let medianPointIndex = indices[medianIndex]

        // Store node information
        tree[medianPointIndex] = depth  // Store depth

        // Set root if this is the first call
        if depth == 0 {
            tree[3 * N] = medianPointIndex
        }

        // Recursively build left and right subtrees
        if medianIndex > 0 {
            var leftIndices = Array(indices[..<medianIndex])
            buildRecursive(
                tree: &tree,
                indices: &leftIndices,
                workInt: &workInt,
                workDouble: &workDouble,
                depth: depth + 1
            )

            // Store left child
            if !leftIndices.isEmpty {
                let leftChild = leftIndices[leftIndices.count / 2]
                tree[medianPointIndex + N] = leftChild
            }
        }

        if medianIndex + 1 < size {
            var rightIndices = Array(indices[(medianIndex + 1)...])
            buildRecursive(
                tree: &tree,
                indices: &rightIndices,
                workInt: &workInt,
                workDouble: &workDouble,
                depth: depth + 1
            )

            // Store right child
            if !rightIndices.isEmpty {
                let rightChild = rightIndices[rightIndices.count / 2]
                tree[medianPointIndex + 2 * N] = rightChild
            }
        }
    }

    // MARK: - Nearest Neighbor Search

    /// Find nearest neighbor
    /// - Parameter point: Query point
    /// - Returns: (index, distance) of nearest neighbor
    public func nearestNeighbor(to point: [Double]) -> (index: Int, distance: Double) {
        precondition(point.count == D, "Point dimension must match tree dimension")

        // First, get approximate nearest neighbor
        var (bestIndex, bestDistance) = approximateNearestNeighbor(to: point)

        // Refine with exact search
        var stack = [Int]()
        let root = tree[3 * N]
        stack.append(root)

        while !stack.isEmpty {
            let nodeIndex = stack.removeLast()
            let depth = tree[nodeIndex]
            let axis = depth % D

            // Check distance to current node
            let distance = euclideanDistance(point, nodeIndex: nodeIndex)
            if distance < bestDistance {
                bestDistance = distance
                bestIndex = nodeIndex
            }

            // Determine which subtree to search
            let splitValue = points[nodeIndex, axis]
            let queryValue = point[axis]
            let diff = queryValue - splitValue

            let leftChild = tree[nodeIndex + N]
            let rightChild = tree[nodeIndex + 2 * N]

            if diff > 0 {
                // Query point is on right side
                if rightChild >= 0 {
                    stack.append(rightChild)
                }
                // Check if we need to search left side too
                if leftChild >= 0 && abs(diff) <= bestDistance {
                    stack.append(leftChild)
                }
            } else {
                // Query point is on left side
                if leftChild >= 0 {
                    stack.append(leftChild)
                }
                // Check if we need to search right side too
                if rightChild >= 0 && abs(diff) <= bestDistance {
                    stack.append(rightChild)
                }
            }
        }

        return (bestIndex, bestDistance)
    }

    /// Approximate nearest neighbor (fast)
    private func approximateNearestNeighbor(to point: [Double]) -> (index: Int, distance: Double) {
        var currentNode = tree[3 * N]  // Start at root
        var bestDistance = Double.infinity
        var bestIndex = currentNode

        while true {
            let depth = tree[currentNode]
            let axis = depth % D

            let distance = euclideanDistance(point, nodeIndex: currentNode)
            if distance < bestDistance {
                bestDistance = distance
                bestIndex = currentNode
            }

            let splitValue = points[currentNode, axis]
            let queryValue = point[axis]

            let leftChild = tree[currentNode + N]
            let rightChild = tree[currentNode + 2 * N]

            if queryValue > splitValue && rightChild >= 0 {
                currentNode = rightChild
            } else if queryValue <= splitValue && leftChild >= 0 {
                currentNode = leftChild
            } else {
                break
            }
        }

        return (bestIndex, bestDistance)
    }

    // MARK: - K Nearest Neighbors Search

    /// Find k nearest neighbors
    /// - Parameters:
    ///   - point: Query point
    ///   - k: Number of neighbors
    ///   - maxDistance: Maximum search distance (optional)
    ///   - excludeIndex: Index to exclude (e.g., the query point itself if it's in the tree)
    /// - Returns: Array of (index, distance) pairs
    public func kNearestNeighbors(
        to point: [Double],
        k: Int,
        maxDistance: Double = .infinity,
        excludeIndex: Int = -1
    ) -> [(index: Int, distance: Double)] {
        precondition(point.count == D, "Point dimension must match tree dimension")
        precondition(k > 0, "k must be positive")

        var neighbors: [(index: Int, distance: Double)] = []
        var maxDist = maxDistance

        var stack = [Int]()
        let root = tree[3 * N]
        stack.append(root)

        while !stack.isEmpty {
            let nodeIndex = stack.removeLast()
            let depth = tree[nodeIndex]
            let axis = depth % D

            // Skip excluded index
            if nodeIndex == excludeIndex {
                continue
            }

            // Check distance to current node
            let distance = euclideanDistance(point, nodeIndex: nodeIndex)

            if distance < maxDist {
                if neighbors.count < k {
                    neighbors.append((nodeIndex, distance))

                    // Update maxDist when we have k neighbors
                    if neighbors.count == k {
                        neighbors.sort { $0.distance > $1.distance }
                        maxDist = neighbors[0].distance
                    }
                } else {
                    // Replace furthest neighbor
                    neighbors[0] = (nodeIndex, distance)
                    neighbors.sort { $0.distance > $1.distance }
                    maxDist = neighbors[0].distance
                }
            }

            // Determine which subtrees to search
            let splitValue = points[nodeIndex, axis]
            let queryValue = point[axis]
            let diff = queryValue - splitValue

            let leftChild = tree[nodeIndex + N]
            let rightChild = tree[nodeIndex + 2 * N]

            if diff > 0 {
                // Search right first
                if leftChild >= 0 && abs(diff) <= maxDist {
                    stack.append(leftChild)
                }
                if rightChild >= 0 {
                    stack.append(rightChild)
                }
            } else {
                // Search left first
                if rightChild >= 0 && abs(diff) <= maxDist {
                    stack.append(rightChild)
                }
                if leftChild >= 0 {
                    stack.append(leftChild)
                }
            }
        }

        // Sort by distance (ascending)
        neighbors.sort { $0.distance < $1.distance }

        return neighbors
    }

    // MARK: - Range Search

    /// Find all neighbors within a radius
    /// - Parameters:
    ///   - point: Query point
    ///   - radius: Search radius
    /// - Returns: Array of indices within radius
    public func neighborsWithinRadius(
        to point: [Double],
        radius: Double
    ) -> [Int] {
        precondition(point.count == D, "Point dimension must match tree dimension")

        var results = [Int]()
        var stack = [Int]()
        let root = tree[3 * N]
        stack.append(root)

        while !stack.isEmpty {
            let nodeIndex = stack.removeLast()
            let depth = tree[nodeIndex]
            let axis = depth % D

            // Check if current node is within radius
            let distance = euclideanDistance(point, nodeIndex: nodeIndex)
            if distance <= radius {
                results.append(nodeIndex)
            }

            // Determine which subtrees to search
            let splitValue = points[nodeIndex, axis]
            let queryValue = point[axis]
            let diff = queryValue - splitValue

            let leftChild = tree[nodeIndex + N]
            let rightChild = tree[nodeIndex + 2 * N]

            if diff > 0 {
                if rightChild >= 0 {
                    stack.append(rightChild)
                }
                if leftChild >= 0 && abs(diff) <= radius {
                    stack.append(leftChild)
                }
            } else {
                if leftChild >= 0 {
                    stack.append(leftChild)
                }
                if rightChild >= 0 && abs(diff) <= radius {
                    stack.append(rightChild)
                }
            }
        }

        return results
    }

    // MARK: - Helper Methods

    /// Compute Euclidean distance between query point and a tree node
    private func euclideanDistance(_ point: [Double], nodeIndex: Int) -> Double {
        var sum: Double = 0
        for d in 0..<D {
            let diff = point[d] - points[nodeIndex, d]
            sum += diff * diff
        }
        return sqrt(sum)
    }
}

// MARK: - KDTree for Point3D

extension KDTree {
    /// Convenience initializer for Point3D arrays
    public convenience init(points3D: [Point3D]) {
        let cloud = PointCloud(points3D: points3D)
        self.init(points: cloud)
    }

    /// Find nearest neighbor to a Point3D
    public func nearestNeighbor(to point: Point3D) -> (index: Int, distance: Double) {
        nearestNeighbor(to: [point.x, point.y, point.z])
    }

    /// Find k nearest neighbors to a Point3D
    public func kNearestNeighbors(
        to point: Point3D,
        k: Int,
        maxDistance: Double = .infinity,
        excludeIndex: Int = -1
    ) -> [(index: Int, distance: Double)] {
        kNearestNeighbors(
            to: [point.x, point.y, point.z],
            k: k,
            maxDistance: maxDistance,
            excludeIndex: excludeIndex
        )
    }
}
