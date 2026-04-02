# BCPD Swift

iOS/Swift implementation of Bayesian Coherent Point Drift (BCPD) algorithm.

## Overview

BCPD is a point cloud registration algorithm for:
- 3D model reconstruction
- Shape analysis
- AR applications with point cloud alignment

This is a Swift port of the [original C implementation](https://github.com/ohirose/bcpd).

## Features

- ✅ **Core BCPD algorithm**: Variational Bayes optimization
- ✅ **Multiple kernel functions**: Gaussian, Laplace, Inverse multiquadric, Rational quadratic
- ✅ **Transformation types**: Rigid, Similarity, Affine, Nonrigid
- 🚧 **Acceleration**: Nystrom method, KD-tree search, Downsampling
- 🚧 **iOS Integration**: ARKit support, SceneKit/Metal visualization

## Project Structure
```
BcpdSwift/
├── Sources/BCPD/
│   ├── Core/              # Core data structures and algorithm
│   ├── Kernels/           # Kernel functions
│   ├── Acceleration/      # KD-tree, Nystrom, downsampling
│   └── Utilities/         # Helper functions
├── Tests/BCPDTests/       # Unit tests
└── Examples/              # Example applications
```

## Requirements

- iOS 15.0+ / macOS 12.0+
- Xcode 14.0+
- Swift 5.9+

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/tatsuya-ogawa/BcpdSwift.git", from: "0.1.0")
]
```

## Usage

```swift
import BCPD

// Create source and target point clouds
let source = PointCloud(points3D: sourcePoints)
let target = PointCloud(points3D: targetPoints)

// Configure parameters
var params = BCPDParameters.accelerated
params.lambda = 1.0
params.beta = 2.0
params.omega = 0.1

// Run registration
let bcpd = BCPD()
let result = try await bcpd.register(source: source, target: target, parameters: params)

// Use results
print("Residual: \(result.residual)")
print("Matched points: \(result.matchedPointsCount)")
```

## Implementation Progress

- [x] **Phase 1: Foundation structures** ✅
  - [x] Data structures (Point, PointCloud, BCPDParameters, BCPDResult)
  - [x] Kernel functions (Gaussian, Laplace, Inverse multiquadric, Rational quadratic)
  - [x] Linear algebra utilities (Accelerate/BLAS/LAPACK integration)
  - [x] Math utilities (statistics, normalization, digamma)

- [x] **Phase 2: Core algorithm** ✅
  - [x] BCPD main algorithm (Variational Bayes optimization)
  - [x] E-step: Correspondence update with Gaussian product
  - [x] M-step: Transformation (scale, rotation, translation) and deformation update
  - [x] Convergence checking
  - [x] Parallel computation support

- [ ] **Phase 3: Acceleration** 🚧
  - [x] Nystrom approximation (basic)
  - [ ] KD-tree search
  - [ ] Downsampling (voxel, ball, random)
  - [x] SIMD optimization (vDSP)

- [ ] **Phase 4: iOS integration**
  - [ ] ARKit point cloud capture
  - [ ] SceneKit/Metal visualization
  - [ ] Async operations with progress
  - [ ] Demo application

### Current Status
- ✅ Core BCPD algorithm implemented and building successfully
- ✅ All unit tests passing (verified against original C implementation results)
- ✅ Main features (kernels, normalization, linear algebra) verified
- 🚧 Working on acceleration features and numerical precision improvements

## References

- [GBCPD/GBCPD++] O. Hirose, "[Geodesic-Based Bayesian Coherent Point Drift](https://ieeexplore.ieee.org/document/9918058)," IEEE TPAMI, Oct 2022.
- [BCPD++] O. Hirose, "[Acceleration of non-rigid point set registration with downsampling and Gaussian process regression](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9290402)," IEEE TPAMI, Dec 2020.
- [BCPD] O. Hirose, "[A Bayesian formulation of coherent point drift](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8985307)," IEEE TPAMI, Feb 2020.

## License

MIT License (same as original implementation)
