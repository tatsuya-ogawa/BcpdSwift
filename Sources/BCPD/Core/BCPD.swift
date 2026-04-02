import Foundation
import Accelerate

private let defaultParallelMinimumCount = 128

@inline(__always)
private func matrixElementIndex(row: Int, col: Int, rows: Int) -> Int {
    row + rows * col
}

@inline(__always)
private func pointElementIndex(point: Int, dimension: Int, stride: Int) -> Int {
    dimension + stride * point
}

private func parallelFor(
    count: Int,
    minimumCount: Int = defaultParallelMinimumCount,
    _ body: @escaping (Int) -> Void
) {
    let workerCount = ProcessInfo.processInfo.activeProcessorCount
    guard workerCount > 1, count >= minimumCount else {
        for index in 0..<count {
            body(index)
        }
        return
    }

    DispatchQueue.concurrentPerform(iterations: count, execute: body)
}

private func parallelReduce(
    count: Int,
    minimumCount: Int = 1024,
    _ body: @escaping (Int) -> Double
) -> Double {
    let workerCount = min(ProcessInfo.processInfo.activeProcessorCount, max(count, 1))
    guard workerCount > 1, count >= minimumCount else {
        var sum: Double = 0
        for index in 0..<count {
            sum += body(index)
        }
        return sum
    }

    var partialSums = [Double](repeating: 0, count: workerCount)
    DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
        let start = count * worker / workerCount
        let end = count * (worker + 1) / workerCount
        var localSum: Double = 0
        for index in start..<end {
            localSum += body(index)
        }
        partialSums[worker] = localSum
    }

    return partialSums.reduce(0, +)
}

private func parallelVectorReduce(
    count: Int,
    dimension: Int,
    minimumCount: Int = 512,
    _ body: @escaping (Int, inout [Double]) -> Void
) -> [Double] {
    let workerCount = min(ProcessInfo.processInfo.activeProcessorCount, max(count, 1))
    guard workerCount > 1, count >= minimumCount else {
        var result = [Double](repeating: 0, count: dimension)
        for index in 0..<count {
            body(index, &result)
        }
        return result
    }

    var partialVectors = Array(
        repeating: [Double](repeating: 0, count: dimension),
        count: workerCount
    )
    DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
        let start = count * worker / workerCount
        let end = count * (worker + 1) / workerCount
        var localVector = [Double](repeating: 0, count: dimension)
        for index in start..<end {
            body(index, &localVector)
        }
        partialVectors[worker] = localVector
    }

    var result = [Double](repeating: 0, count: dimension)
    for partial in partialVectors {
        for d in 0..<dimension {
            result[d] += partial[d]
        }
    }
    return result
}

// MARK: - BCPD Main Algorithm

/// Bayesian Coherent Point Drift implementation
/// Matches the corrected probreg Python implementation and the original C implementation (ohirose/bcpd)
public class BCPD {

    public init() {}

    // MARK: - Public Registration API

    /// Register source point cloud to target point cloud using the CombinedBCPD algorithm.
    /// Exactly follows the corrected probreg `bcpd.py` `CombinedBCPD` class,
    /// which was verified to match the original C implementation (ohirose/bcpd).
    public func register(
        source: PointCloud,
        target: PointCloud,
        w: Double = 0.0,
        maxIter: Int = 50,
        tol: Double = 0.001,
        lambda: Double = 2.0,
        kappa: Double = 1.0e20,
        gamma: Double = 1.0,
        beta: Double = 1.0,
        kernelType: KernelType = .inverseMultiquadric,
        useDebias: Bool = false,
        verbose: Bool = false
    ) throws -> BCPDResult {

        guard source.dimension == target.dimension else {
            throw BCPDError.dimensionMismatch
        }

        let D = source.dimension
        let M = source.count

        // --- Initialize (matches CombinedBCPD._initialize) ---

        // Compute kernel Gram matrix G and its inverse
        let kernel = KernelFactory.create(type: kernelType)
        let gmatData = computeGramMatrixParallel(points: source, kernel: kernel, beta: beta)
        let gmat = Matrix(rows: M, cols: M, data: gmatData)

        guard let gmatInv = invertSymmetricPositiveDefinite(gmat) else {
            throw BCPDError.computationFailed("Failed to invert Gram matrix")
        }

        // Initial sigma2 = gamma^2 * squared_kernel_sum(source, target)
        var sigma2 = (gamma * gamma) * squaredKernelSum(source: source, target: target)

        // Initial transformation: R=I, t=0, s=1
        var rot = Matrix.identity(size: D)
        var trans = [Double](repeating: 0, count: D)
        var scale: Double = 1.0

        var sigmaMat = Matrix.identity(size: M)
        var alpha = [Double](repeating: 1.0 / Double(M), count: M)
        var vHat = Matrix(rows: M, cols: D)  // zero displacement

        // Build KD-tree for RMSE computation
        let targetTree = KDTree(points: target)

        var rmse: Double? = nil
        var finalIter = maxIter

        var nP: Double = 0.0

        for iteration in 0..<maxIter {
            // Transform source: t_source = s * (source + v) @ R^T + t
            let tSource = transformPointCloud(
                source: source,
                displacement: vHat,
                rotation: rot,
                scale: scale,
                translation: trans
            )

            // --- E-step ---
            let estep = expectationStep(
                tSource: tSource,
                target: target,
                scale: scale,
                alpha: alpha,
                sigmaMat: sigmaMat,
                sigma2: sigma2,
                w: w,
                useDebias: useDebias
            )
            nP = estep.nP

            // --- M-step ---
            let mstep = try maximizationStep(
                source: source,
                target: target,
                rigidScale: scale,
                rigidRot: rot,
                rigidTrans: trans,
                estepRes: estep,
                gmatInv: gmatInv,
                lambda: lambda,
                kappa: kappa,
                sigma2: sigma2,
                useDebias: useDebias
            )

            rot = mstep.rotation
            trans = mstep.translation
            scale = mstep.scale
            vHat = mstep.vHat
            sigmaMat = mstep.sigmaMat
            alpha = mstep.alpha
            sigma2 = mstep.sigma2

            // Compute RMSE for convergence check
            let tmpRmse = computeRMSE(source: tSource, targetTree: targetTree)

            if verbose {
                print(String(format: "Iteration %3d: sigma2=%.8f, RMSE=%.6f, Np=%.2f",
                            iteration + 1, sigma2, tmpRmse, nP))
            }

            if let prevRmse = rmse, abs(prevRmse - tmpRmse) < tol {
                finalIter = iteration + 1
                break
            }
            rmse = tmpRmse
        }

        // Compute final transformed source
        let deformedSource = transformPointCloud(
            source: source,
            displacement: vHat,
            rotation: rot,
            scale: scale,
            translation: trans
        )

        return BCPDResult(
            deformedSource: deformedSource,
            displacementVectors: vHat,
            scale: scale,
            rotation: rot.data,
            translation: trans,
            sigma2: sigma2,
            iterations: finalIter,
            residual: rmse ?? 0.0,
            matchedPointsCount: nP
        )
    }

    /// Register source point cloud to target point cloud using parameters.
    public func register(
        source: PointCloud,
        target: PointCloud,
        parameters: BCPDParameters
    ) throws -> BCPDResult {
        return try register(
            source: source,
            target: target,
            w: parameters.omega,
            maxIter: parameters.maxIterations,
            tol: parameters.tolerance,
            lambda: parameters.lambda,
            kappa: parameters.kappa,
            gamma: parameters.gamma,
            beta: parameters.beta,
            kernelType: parameters.kernelType,
            useDebias: parameters.useDebias,
            verbose: parameters.verbose
        )
    }

    /// Asynchronous registration.
    public func register(
        source: PointCloud,
        target: PointCloud,
        parameters: BCPDParameters
    ) async throws -> BCPDResult {
        return try await Task.detached(priority: .userInitiated) {
            return try self.register(source: source, target: target, parameters: parameters)
        }.value
    }

    // MARK: - E-step

    /// Expectation step - computes correspondence probabilities.
    /// Exactly matches probreg `expectation_step` (corrected version).
    func expectationStep(
        tSource: PointCloud,
        target: PointCloud,
        scale: Double,
        alpha: [Double],
        sigmaMat: Matrix,
        sigma2: Double,
        w: Double,
        useDebias: Bool
    ) -> EstepResult {
        let D = tSource.dimension
        let M = tSource.count
        let N = target.count

        let norm = pow(2.0 * .pi * sigma2, Double(D) * 0.5)

        // pmat[m * N + n] = N(x_n | y_m, sigma2 * I)
        var pmat = [Double](repeating: 0, count: M * N)
        let baseCoeff = 1.0 - w
        let sigmaScale = -(scale * scale) / (2.0 * sigma2) * Double(D)

        tSource.data.withUnsafeBufferPointer { tSourceBuffer in
            target.data.withUnsafeBufferPointer { targetBuffer in
                sigmaMat.data.withUnsafeBufferPointer { sigmaBuffer in
                    pmat.withUnsafeMutableBufferPointer { pmatBuffer in
                        let tSourceBase = tSourceBuffer.baseAddress!
                        let targetBase = targetBuffer.baseAddress!
                        let sigmaBase = sigmaBuffer.baseAddress!
                        let pmatBase = pmatBuffer.baseAddress!

                        parallelFor(count: M) { m in
                            let sourceOffset = D * m
                            let rowOffset = m * N
                            let debias = useDebias
                                ? exp(sigmaScale * sigmaBase[matrixElementIndex(row: m, col: m, rows: M)])
                                : 1.0
                            let rowCoeff = baseCoeff * alpha[m] * debias / norm

                            for n in 0..<N {
                                let targetOffset = D * n
                                var distSq: Double = 0
                                for d in 0..<D {
                                    let diff = targetBase[targetOffset + d] - tSourceBase[sourceOffset + d]
                                    distSq += diff * diff
                                }
                                pmatBase[rowOffset + n] = exp(-distSq / (2.0 * sigma2)) * rowCoeff
                            }
                        }
                    }
                }
            }
        }

        // Outlier term: w / volume (bounding box of target)
        var bbMin = [Double](repeating: .infinity, count: D)
        var bbMax = [Double](repeating: -.infinity, count: D)
        for n in 0..<N {
            for d in 0..<D {
                let v = target[n, d]
                bbMin[d] = min(bbMin[d], v)
                bbMax[d] = max(bbMax[d], v)
            }
        }
        var vol: Double = 1.0
        for d in 0..<D {
            vol *= max(bbMax[d] - bbMin[d], 1e-10)
        }
        let outlierTerm = w / vol

        // den[n] = outlier_term + sum_m pmat[m][n]
        var den = [Double](repeating: outlierTerm, count: N)
        pmat.withUnsafeBufferPointer { pmatBuffer in
            den.withUnsafeMutableBufferPointer { denBuffer in
                let pmatBase = pmatBuffer.baseAddress!
                let denBase = denBuffer.baseAddress!

                parallelFor(count: N) { n in
                    var value = outlierTerm
                    for m in 0..<M {
                        value += pmatBase[m * N + n]
                    }
                    denBase[n] = value == 0 ? Double.leastNonzeroMagnitude : value
                }
            }
        }

        // Normalize row-wise and accumulate nu / px together.
        var nu = [Double](repeating: 0, count: M)
        var px = Matrix(rows: M, cols: D)
        pmat.withUnsafeMutableBufferPointer { pmatBuffer in
            den.withUnsafeBufferPointer { denBuffer in
                target.data.withUnsafeBufferPointer { targetBuffer in
                    nu.withUnsafeMutableBufferPointer { nuBuffer in
                        px.data.withUnsafeMutableBufferPointer { pxBuffer in
                            let pmatBase = pmatBuffer.baseAddress!
                            let denBase = denBuffer.baseAddress!
                            let targetBase = targetBuffer.baseAddress!
                            let nuBase = nuBuffer.baseAddress!
                            let pxBase = pxBuffer.baseAddress!

                            parallelFor(count: M) { m in
                                let rowOffset = m * N
                                var rowSum: Double = 0
                                var weightedTarget = [Double](repeating: 0, count: D)

                                for n in 0..<N {
                                    let normalized = pmatBase[rowOffset + n] / denBase[n]
                                    pmatBase[rowOffset + n] = normalized
                                    rowSum += normalized

                                    let targetOffset = D * n
                                    for d in 0..<D {
                                        weightedTarget[d] += normalized * targetBase[targetOffset + d]
                                    }
                                }

                                nuBase[m] = max(rowSum, 1e-20)
                                for d in 0..<D {
                                    pxBase[matrixElementIndex(row: m, col: d, rows: M)] = weightedTarget[d]
                                }
                            }
                        }
                    }
                }
            }
        }

        // nu_d[n] = sum_m pmat[m][n]
        var nuD = [Double](repeating: 0, count: N)
        pmat.withUnsafeBufferPointer { pmatBuffer in
            nuD.withUnsafeMutableBufferPointer { nuDBuffer in
                let pmatBase = pmatBuffer.baseAddress!
                let nuDBase = nuDBuffer.baseAddress!

                parallelFor(count: N) { n in
                    var value: Double = 0
                    for m in 0..<M {
                        value += pmatBase[m * N + n]
                    }
                    nuDBase[n] = value
                }
            }
        }

        let nP = nu.reduce(0, +)

        // x_hat[m, d] = px[m, d] / nu[m]
        var xHat = Matrix(rows: M, cols: D)
        nu.withUnsafeBufferPointer { nuBuffer in
            px.data.withUnsafeBufferPointer { pxBuffer in
                xHat.data.withUnsafeMutableBufferPointer { xHatBuffer in
                    let nuBase = nuBuffer.baseAddress!
                    let pxBase = pxBuffer.baseAddress!
                    let xHatBase = xHatBuffer.baseAddress!

                    parallelFor(count: M) { m in
                        let invNu = 1.0 / nuBase[m]
                        for d in 0..<D {
                            let index = matrixElementIndex(row: m, col: d, rows: M)
                            xHatBase[index] = pxBase[index] * invNu
                        }
                    }
                }
            }
        }

        return EstepResult(nuD: nuD, nu: nu, nP: nP, px: px, xHat: xHat)
    }

    // MARK: - M-step

    /// Maximization step - updates transformation parameters.
    /// Exactly matches probreg `CombinedBCPD._maximization_step` (corrected version).
    func maximizationStep(
        source: PointCloud,
        target: PointCloud,
        rigidScale: Double,
        rigidRot: Matrix,
        rigidTrans: [Double],
        estepRes: EstepResult,
        gmatInv: Matrix,
        lambda: Double,
        kappa: Double,
        sigma2: Double,
        useDebias: Bool
    ) throws -> MstepResult {
        let D = source.dimension
        let M = source.count
        let nu = estepRes.nu
        let nP = estepRes.nP
        let px = estepRes.px
        let xHat = estepRes.xHat
        let nuD = estepRes.nuD

        let s2s2 = (rigidScale * rigidScale) / sigma2

        // sigma_mat_inv = lambda * G^{-1} + s2s2 * diag(nu)
        var sigmaMatInv = Matrix(rows: M, cols: M)
        gmatInv.data.withUnsafeBufferPointer { gmatInvBuffer in
            sigmaMatInv.data.withUnsafeMutableBufferPointer { sigmaMatInvBuffer in
                let gmatInvBase = gmatInvBuffer.baseAddress!
                let sigmaMatInvBase = sigmaMatInvBuffer.baseAddress!

                parallelFor(count: M) { i in
                    for j in 0..<M {
                        sigmaMatInvBase[matrixElementIndex(row: i, col: j, rows: M)] =
                            lambda * gmatInvBase[matrixElementIndex(row: i, col: j, rows: M)]
                    }
                    sigmaMatInvBase[matrixElementIndex(row: i, col: i, rows: M)] += s2s2 * nu[i]
                }
            }
        }

        guard let sigmaMat = invertSymmetricPositiveDefinite(sigmaMatInv) else {
            throw BCPDError.computationFailed("Failed to invert sigma_mat_inv")
        }

        // residual = R^T @ (x_hat - t) / s - source
        var residual = Matrix(rows: M, cols: D)
        xHat.data.withUnsafeBufferPointer { xHatBuffer in
            rigidRot.data.withUnsafeBufferPointer { rigidRotBuffer in
                source.data.withUnsafeBufferPointer { sourceBuffer in
                    residual.data.withUnsafeMutableBufferPointer { residualBuffer in
                        let xHatBase = xHatBuffer.baseAddress!
                        let rigidRotBase = rigidRotBuffer.baseAddress!
                        let sourceBase = sourceBuffer.baseAddress!
                        let residualBase = residualBuffer.baseAddress!

                        parallelFor(count: M) { m in
                            let sourceOffset = D * m
                            for d in 0..<D {
                                var value: Double = 0
                                for i in 0..<D {
                                    value += rigidRotBase[matrixElementIndex(row: i, col: d, rows: D)] *
                                        (xHatBase[matrixElementIndex(row: m, col: i, rows: M)] - rigidTrans[i])
                                }
                                residualBase[matrixElementIndex(row: m, col: d, rows: M)] =
                                    value / rigidScale - sourceBase[sourceOffset + d]
                            }
                        }
                    }
                }
            }
        }

        // v_hat[m, d] = s2s2 * sum_j sigma_mat[m, j] * nu[j] * residual[j, d]
        var vHat = Matrix(rows: M, cols: D)
        sigmaMat.data.withUnsafeBufferPointer { sigmaMatBuffer in
            residual.data.withUnsafeBufferPointer { residualBuffer in
                vHat.data.withUnsafeMutableBufferPointer { vHatBuffer in
                    let sigmaMatBase = sigmaMatBuffer.baseAddress!
                    let residualBase = residualBuffer.baseAddress!
                    let vHatBase = vHatBuffer.baseAddress!

                    parallelFor(count: M) { m in
                        var rowValues = [Double](repeating: 0, count: D)
                        for j in 0..<M {
                            let coeff = sigmaMatBase[matrixElementIndex(row: m, col: j, rows: M)] * nu[j]
                            if coeff == 0 { continue }

                            for d in 0..<D {
                                rowValues[d] += coeff * residualBase[matrixElementIndex(row: j, col: d, rows: M)]
                            }
                        }

                        for d in 0..<D {
                            vHatBase[matrixElementIndex(row: m, col: d, rows: M)] = s2s2 * rowValues[d]
                        }
                    }
                }
            }
        }

        // u_hat = source + v_hat
        var uHat = Matrix(rows: M, cols: D)
        source.data.withUnsafeBufferPointer { sourceBuffer in
            vHat.data.withUnsafeBufferPointer { vHatBuffer in
                uHat.data.withUnsafeMutableBufferPointer { uHatBuffer in
                    let sourceBase = sourceBuffer.baseAddress!
                    let vHatBase = vHatBuffer.baseAddress!
                    let uHatBase = uHatBuffer.baseAddress!

                    parallelFor(count: M) { m in
                        let sourceOffset = D * m
                        for d in 0..<D {
                            uHatBase[matrixElementIndex(row: m, col: d, rows: M)] =
                                sourceBase[sourceOffset + d] + vHatBase[matrixElementIndex(row: m, col: d, rows: M)]
                        }
                    }
                }
            }
        }

        // alpha = exp(psi(kappa + nu) - psi(kappa * M + n_p))
        var newAlpha = [Double](repeating: 0, count: M)
        let psiDenom = digamma(kappa * Double(M) + nP)
        newAlpha.withUnsafeMutableBufferPointer { newAlphaBuffer in
            let newAlphaBase = newAlphaBuffer.baseAddress!
            parallelFor(count: M) { m in
                newAlphaBase[m] = exp(digamma(kappa + nu[m]) - psiDenom)
            }
        }

        // Weighted means
        var xM = parallelVectorReduce(count: M, dimension: D) { m, partial in
            let weight = nu[m]
            for d in 0..<D {
                partial[d] += weight * xHat[m, d]
            }
        }
        for d in 0..<D {
            xM[d] /= nP
        }

        let sigma2M = parallelReduce(count: M) { m in
            nu[m] * sigmaMat[m, m]
        } / nP

        var uM = parallelVectorReduce(count: M, dimension: D) { m, partial in
            let weight = nu[m]
            for d in 0..<D {
                partial[d] += weight * uHat[m, d]
            }
        }
        for d in 0..<D {
            uM[d] /= nP
        }

        // u_hm = u_hat - u_m
        var uHm = Matrix(rows: M, cols: D)
        uHat.data.withUnsafeBufferPointer { uHatBuffer in
            uHm.data.withUnsafeMutableBufferPointer { uHmBuffer in
                let uHatBase = uHatBuffer.baseAddress!
                let uHmBase = uHmBuffer.baseAddress!

                parallelFor(count: M) { m in
                    for d in 0..<D {
                        let index = matrixElementIndex(row: m, col: d, rows: M)
                        uHmBase[index] = uHatBase[index] - uM[d]
                    }
                }
            }
        }

        let centeredXHat = centerColumns(xHat, means: xM)
        let weightedUHm = scaleRows(uHm, weights: nu)

        // S_xu = (1/n_p) * (x_hat - x_m)^T @ (diag(nu) * u_hm)
        let sXU = matrixMultiply(
            centeredXHat,
            weightedUHm,
            transposeA: true,
            alpha: 1.0 / nP
        )

        // S_uu = (1/n_p) * u_hm^T @ (diag(nu) * u_hm)
        var sUU = matrixMultiply(
            uHm,
            weightedUHm,
            transposeA: true,
            alpha: 1.0 / nP
        )

        if useDebias {
            for i in 0..<D {
                sUU[i, i] += sigma2M
            }
        }

        // SVD of S_xu
        guard let svdRes = svdDecomposition(sXU) else {
            throw BCPDError.computationFailed("SVD failed")
        }
        var phi = svdRes.U
        let psih = svdRes.Vt

        // Ensure proper rotation: c[-1] = det(phi @ psih)
        let phiPsih = matrixMultiply(phi, psih)
        guard let detVal = determinant(phiPsih) else {
            throw BCPDError.computationFailed("Determinant failed")
        }
        if detVal < 0 {
            for i in 0..<D {
                phi[i, D - 1] *= -1.0
            }
        }

        let newRot = matrixMultiply(phi, psih)

        // Frobenius inner product: sum(R * S_xu)
        let trRsxu = frobeniusInnerProduct(newRot, sXU)

        // scale = tr_rsxu / trace(S_uu)
        var traceSuu: Double = 0
        for i in 0..<D { traceSuu += sUU[i, i] }
        let newScale = trRsxu / traceSuu

        // t = x_m - scale * R @ u_m
        let newTrans = vectorAdd(xM, scalarMultiply(-newScale, matrixVectorMultiply(newRot, uM)))

        // y_hat = scale * u_hat @ R^T + t  (using updated R, s, t)
        let yHat = addVectorToEachRow(
            matrixMultiply(uHat, newRot, transposeB: true, alpha: newScale),
            vector: newTrans
        )

        // sigma2 = (s1 - 2*s2 + s3) / (n_p * D)
        let s1 = parallelReduce(count: target.count) { n in
            var normSq: Double = 0
            for d in 0..<D {
                let value = target[n, d]
                normSq += value * value
            }
            return nuD[n] * normSq
        }

        let s2Val = parallelReduce(count: M) { m in
            var value: Double = 0
            for d in 0..<D {
                value += px[m, d] * yHat[m, d]
            }
            return value
        }

        let s3 = parallelReduce(count: M) { m in
            var normSq: Double = 0
            for d in 0..<D {
                let value = yHat[m, d]
                normSq += value * value
            }
            return nu[m] * normSq
        }

        var newSigma2 = (s1 - 2.0 * s2Val + s3) / (nP * Double(D))
        if useDebias { newSigma2 += rigidScale * rigidScale * sigma2M }
        newSigma2 = max(abs(newSigma2), Double.ulpOfOne)

        return MstepResult(
            rotation: newRot,
            translation: newTrans,
            scale: newScale,
            vHat: vHat,
            sigmaMat: sigmaMat,
            alpha: newAlpha,
            sigma2: newSigma2
        )
    }

    // MARK: - Helper Functions

    private func transformPointCloud(
        source: PointCloud,
        displacement: Matrix,
        rotation: Matrix,
        scale: Double,
        translation: [Double]
    ) -> PointCloud {
        let D = source.dimension
        let M = source.count
        var transformed = PointCloud(dimension: D, count: M)

        source.data.withUnsafeBufferPointer { sourceBuffer in
            displacement.data.withUnsafeBufferPointer { displacementBuffer in
                rotation.data.withUnsafeBufferPointer { rotationBuffer in
                    transformed.data.withUnsafeMutableBufferPointer { transformedBuffer in
                        let sourceBase = sourceBuffer.baseAddress!
                        let displacementBase = displacementBuffer.baseAddress!
                        let rotationBase = rotationBuffer.baseAddress!
                        let transformedBase = transformedBuffer.baseAddress!

                        parallelFor(count: M) { m in
                            let pointOffset = D * m
                            for d in 0..<D {
                                var value: Double = 0
                                for i in 0..<D {
                                    value += rotationBase[matrixElementIndex(row: d, col: i, rows: D)] *
                                        (sourceBase[pointOffset + i] +
                                         displacementBase[matrixElementIndex(row: m, col: i, rows: M)])
                                }
                                transformedBase[pointElementIndex(point: m, dimension: d, stride: D)] =
                                    scale * value + translation[d]
                            }
                        }
                    }
                }
            }
        }

        return transformed
    }

    /// Compute squared kernel sum (matches probreg squared_kernel_sum)
    func squaredKernelSum(source: PointCloud, target: PointCloud) -> Double {
        let D = source.dimension
        let M = source.count
        let N = target.count

        let sumYsq = parallelReduce(count: M, minimumCount: 256) { m in
            var pointNormSq: Double = 0
            for d in 0..<D {
                let value = source[m, d]
                pointNormSq += value * value
            }
            return pointNormSq
        }
        let sumXsq = parallelReduce(count: N, minimumCount: 256) { n in
            var pointNormSq: Double = 0
            for d in 0..<D {
                let value = target[n, d]
                pointNormSq += value * value
            }
            return pointNormSq
        }
        let sumY = parallelVectorReduce(count: M, dimension: D, minimumCount: 256) { m, partial in
            for d in 0..<D {
                partial[d] += source[m, d]
            }
        }
        let sumX = parallelVectorReduce(count: N, dimension: D, minimumCount: 256) { n, partial in
            for d in 0..<D {
                partial[d] += target[n, d]
            }
        }
        let crossTerm = zip(sumY, sumX).reduce(0.0) { partial, values in
            partial + values.0 * values.1
        }

        let total = Double(N) * sumYsq + Double(M) * sumXsq - 2.0 * crossTerm
        return total / Double(M * D * N)
    }

    /// Compute mean nearest-neighbor distance (RMSE matching probreg compute_rmse)
    func computeRMSE(source: PointCloud, targetTree: KDTree) -> Double {
        let totalDist = parallelReduce(count: source.count, minimumCount: 64) { m in
            var point = [Double](repeating: 0, count: source.dimension)
            for d in 0..<source.dimension { point[d] = source[m, d] }
            let (_, dist) = targetTree.nearestNeighbor(to: point)
            return dist
        }
        return totalDist / Double(source.count)
    }

    /// Invert a symmetric positive definite matrix (with fallback to general inverse)
    func invertSymmetricPositiveDefinite(_ A: Matrix) -> Matrix? {
        let identity = Matrix.identity(size: A.rows)
        if let result = solvePositiveDefiniteMultiple(A, identity) {
            return result
        }
        // Fallback to general matrix inverse (matches probreg's np.linalg.inv)
        return invertMatrix(A)
    }
}

// MARK: - SVD Decomposition

public struct SVDResult {
    public var U: Matrix
    public var S: [Double]
    public var Vt: Matrix
}

public func svdDecomposition(_ A: Matrix) -> SVDResult? {
    let m = A.rows
    let n = A.cols
    let minMN = min(m, n)

    var matrix = A
    var s = [Double](repeating: 0, count: minMN)
    var u = [Double](repeating: 0, count: m * m)
    var vt = [Double](repeating: 0, count: n * n)
    var info: Int32 = 0
    var lwork: Int32 = -1
    var workQuery = [Double](repeating: 0, count: 1)
    var iwork = [Int32](repeating: 0, count: 8 * minMN)

    var jobz = Int8(UInt8(ascii: "A"))
    var mI = Int32(m)
    var nI = Int32(n)
    var lda = Int32(m)
    var ldu = Int32(m)
    var ldvt = Int32(n)

    dgesdd_(&jobz, &mI, &nI, &matrix.data, &lda, &s, &u, &ldu, &vt, &ldvt, &workQuery, &lwork, &iwork, &info)
    guard info == 0 else { return nil }

    lwork = Int32(workQuery[0])
    var work = [Double](repeating: 0, count: Int(lwork))

    dgesdd_(&jobz, &mI, &nI, &matrix.data, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &iwork, &info)
    guard info == 0 else { return nil }

    return SVDResult(
        U: Matrix(rows: m, cols: m, data: u),
        S: s,
        Vt: Matrix(rows: n, cols: n, data: vt)
    )
}

// MARK: - Internal Result Types

struct EstepResult {
    let nuD: [Double]
    let nu: [Double]
    let nP: Double
    let px: Matrix
    let xHat: Matrix
}

struct MstepResult {
    let rotation: Matrix
    let translation: [Double]
    let scale: Double
    let vHat: Matrix
    let sigmaMat: Matrix
    let alpha: [Double]
    let sigma2: Double
}

// MARK: - BCPD Result

public struct BCPDResult {
    public var deformedSource: PointCloud
    public var displacementVectors: Matrix
    public var scale: Double
    public var rotation: [Double]
    public var translation: [Double]
    public var sigma2: Double
    public var iterations: Int
    /// Convergence value (RMSE)
    public var residual: Double
    /// Matched points count (nP)
    public var matchedPointsCount: Double
}

// MARK: - Errors

public enum BCPDError: Error, LocalizedError {
    case dimensionMismatch
    case computationFailed(String)
    case invalidParameters(String)

    public var errorDescription: String? {
        switch self {
        case .dimensionMismatch: return "Source and target dimensions do not match"
        case .computationFailed(let msg): return "Computation failed: \(msg)"
        case .invalidParameters(let msg): return "Invalid parameters: \(msg)"
        }
    }
}
