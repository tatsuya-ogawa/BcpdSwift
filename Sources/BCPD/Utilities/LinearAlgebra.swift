import Foundation
import Accelerate

// MARK: - Matrix Operations

/// Matrix utilities using BLAS/LAPACK from Accelerate framework
public struct Matrix {
    /// Matrix data in column-major order (LAPACK convention)
    public var data: [Double]
    public let rows: Int
    public let cols: Int

    public init(rows: Int, cols: Int, repeating value: Double = 0.0) {
        self.rows = rows
        self.cols = cols
        self.data = Array(repeating: value, count: rows * cols)
    }

    public init(rows: Int, cols: Int, data: [Double]) {
        precondition(data.count == rows * cols, "Data count must equal rows * cols")
        self.rows = rows
        self.cols = cols
        self.data = data
    }

    /// Create identity matrix
    public static func identity(size: Int) -> Matrix {
        var matrix = Matrix(rows: size, cols: size)
        for i in 0..<size {
            matrix[i, i] = 1.0
        }
        return matrix
    }

    /// Access element at (row, col)
    public subscript(row: Int, col: Int) -> Double {
        get { data[row + rows * col] }
        set { data[row + rows * col] = newValue }
    }

    /// Transpose matrix
    public func transposed() -> Matrix {
        var result = Matrix(rows: cols, cols: rows)
        vDSP_mtransD(data, 1, &result.data, 1, vDSP_Length(cols), vDSP_Length(rows))
        return result
    }
}

// MARK: - BLAS Operations

/// Matrix multiplication: C = alpha * A * B + beta * C
/// - Parameters:
///   - A: Left matrix (M x K)
///   - B: Right matrix (K x N)
///   - alpha: Scalar multiplier for A*B
///   - beta: Scalar multiplier for C
/// - Returns: Result matrix (M x N)
public func matrixMultiply(
    _ A: Matrix,
    _ B: Matrix,
    transposeA: Bool = false,
    transposeB: Bool = false,
    alpha: Double = 1.0,
    beta: Double = 0.0
) -> Matrix {
    let aRows = transposeA ? A.cols : A.rows
    let aCols = transposeA ? A.rows : A.cols
    let bRows = transposeB ? B.cols : B.rows
    let bCols = transposeB ? B.rows : B.cols

    precondition(aCols == bRows, "Matrix dimensions incompatible for multiplication")

    let M = Int32(aRows)
    let N = Int32(bCols)
    let K = Int32(aCols)

    var C = Matrix(rows: aRows, cols: bCols)

    var mutableA = A
    var mutableB = B
    let transA = transposeA ? CblasTrans : CblasNoTrans
    let transB = transposeB ? CblasTrans : CblasNoTrans

    // dgemm: C := alpha*A*B + beta*C
    cblas_dgemm(
        CblasColMajor,
        transA, transB,
        M, N, K,
        alpha,
        &mutableA.data, Int32(A.rows),
        &mutableB.data, Int32(B.rows),
        beta,
        &C.data, M
    )

    return C
}

/// Matrix-vector multiplication: y = alpha * A * x + beta * y
public func matrixVectorMultiply(
    _ A: Matrix,
    _ x: [Double],
    alpha: Double = 1.0,
    beta: Double = 0.0
) -> [Double] {
    precondition(A.cols == x.count, "Matrix columns must match vector length")

    var y = [Double](repeating: 0, count: A.rows)
    var mutableA = A
    var mutableX = x

    cblas_dgemv(
        CblasColMajor,
        CblasNoTrans,
        Int32(A.rows), Int32(A.cols),
        alpha,
        &mutableA.data, Int32(A.rows),
        &mutableX, 1,
        beta,
        &y, 1
    )

    return y
}

// MARK: - LAPACK Operations

/// Cholesky decomposition: A = L*L^T (for positive definite matrix)
/// - Parameter A: Positive definite symmetric matrix
/// - Returns: Lower triangular matrix L, or nil if decomposition failed
public func choleskyDecomposition(_ A: Matrix) -> Matrix? {
    precondition(A.rows == A.cols, "Matrix must be square")

    var result = A
    let n = Int32(A.rows)
    var nCopy = n
    var info: Int32 = 0

    // dpotrf: Cholesky factorization
    result.data.withUnsafeMutableBufferPointer { buffer in
        var uplo = Int8(bitPattern: UInt8(ascii: "U"))
        var lda = n
        dpotrf_(&uplo, &nCopy, buffer.baseAddress!, &lda, &info)
    }

    return info == 0 ? result : nil
}

/// Solve linear system Ax = b using Cholesky decomposition
/// Assumes A is positive definite and symmetric
public func solvePositiveDefinite(_ A: Matrix, _ b: [Double]) -> [Double]? {
    precondition(A.rows == A.cols, "Matrix must be square")
    precondition(A.rows == b.count, "Dimensions incompatible")

    var factored = A
    var solution = b
    let n = Int32(A.rows)
    var nCopy = n
    var nrhs = Int32(1)
    var info: Int32 = 0

    // dposv: Solve Ax=b for positive definite A
    factored.data.withUnsafeMutableBufferPointer { aBuffer in
        solution.withUnsafeMutableBufferPointer { bBuffer in
            var uplo = Int8(bitPattern: UInt8(ascii: "U"))
            var lda = n
            var ldb = n
            dposv_(&uplo, &nCopy, &nrhs, aBuffer.baseAddress!, &lda, bBuffer.baseAddress!, &ldb, &info)
        }
    }

    return info == 0 ? solution : nil
}

/// Solve multiple right-hand sides: AX = B
public func solvePositiveDefiniteMultiple(_ A: Matrix, _ B: Matrix) -> Matrix? {
    precondition(A.rows == A.cols, "Matrix must be square")
    precondition(A.rows == B.rows, "Dimensions incompatible")

    var factored = A
    var solution = B
    let n = Int32(A.rows)
    var nCopy = n
    var nrhs = Int32(B.cols)
    var info: Int32 = 0

    factored.data.withUnsafeMutableBufferPointer { aBuffer in
        solution.data.withUnsafeMutableBufferPointer { bBuffer in
            var uplo = Int8(bitPattern: UInt8(ascii: "U"))
            var lda = n
            var ldb = n
            dposv_(&uplo, &nCopy, &nrhs, aBuffer.baseAddress!, &lda, bBuffer.baseAddress!, &ldb, &info)
        }
    }

    return info == 0 ? solution : nil
}

/// Eigenvalue decomposition for symmetric matrix
/// - Parameter A: Symmetric matrix
/// - Returns: Tuple of (eigenvalues, eigenvectors), or nil if failed
public func eigenDecomposition(_ A: Matrix) -> (values: [Double], vectors: Matrix)? {
    precondition(A.rows == A.cols, "Matrix must be square")

    var matrix = A
    let n = Int32(A.rows)
    var nCopy = n
    var eigenvalues = [Double](repeating: 0, count: A.rows)
    var info: Int32 = 0
    var lwork = Int32(-1)
    var workQuery = [Double](repeating: 0, count: 1)

    // Query optimal workspace size
    matrix.data.withUnsafeMutableBufferPointer { buffer in
        var jobz = Int8(bitPattern: UInt8(ascii: "V"))  // Compute eigenvalues and eigenvectors
        var uplo = Int8(bitPattern: UInt8(ascii: "U"))
        var lda = n
        dsyev_(&jobz, &uplo, &nCopy, buffer.baseAddress!, &lda, &eigenvalues, &workQuery, &lwork, &info)
    }

    guard info == 0 else { return nil }

    lwork = Int32(workQuery[0])
    var work = [Double](repeating: 0, count: Int(lwork))

    // Compute eigenvalues and eigenvectors
    nCopy = n
    matrix.data.withUnsafeMutableBufferPointer { buffer in
        var jobz = Int8(bitPattern: UInt8(ascii: "V"))
        var uplo = Int8(bitPattern: UInt8(ascii: "U"))
        var lda = n
        dsyev_(&jobz, &uplo, &nCopy, buffer.baseAddress!, &lda, &eigenvalues, &work, &lwork, &info)
    }

    return info == 0 ? (eigenvalues, matrix) : nil
}

/// LU decomposition with partial pivoting
/// - Parameter A: Square matrix
/// - Returns: Tuple of (LU factorization, pivot indices), or nil if singular
public func luDecomposition(_ A: Matrix) -> (lu: Matrix, ipiv: [Int32])? {
    precondition(A.rows == A.cols, "Matrix must be square")

    var lu = A
    let m = Int32(A.rows)
    let n = Int32(A.cols)
    var mCopy = m
    var nCopy = n
    var ipiv = [Int32](repeating: 0, count: A.rows)
    var info: Int32 = 0

    lu.data.withUnsafeMutableBufferPointer { buffer in
        ipiv.withUnsafeMutableBufferPointer { ipivBuffer in
            var lda = m
            dgetrf_(&mCopy, &nCopy, buffer.baseAddress!, &lda, ipivBuffer.baseAddress!, &info)
        }
    }

    return info == 0 ? (lu, ipiv) : nil
}

/// Compute determinant using LU decomposition
public func determinant(_ A: Matrix) -> Double? {
    guard let (lu, ipiv) = luDecomposition(A) else {
        return nil
    }

    var det: Double = 1.0
    for i in 0..<A.rows {
        det *= lu[i, i]
        // Account for row swaps
        if ipiv[i] != Int32(i + 1) {
            det *= -1.0
        }
    }

    return det
}

/// Invert a general square matrix using LU decomposition (LAPACK dgetrf + dgetri)
public func invertMatrix(_ A: Matrix) -> Matrix? {
    precondition(A.rows == A.cols, "Matrix must be square")

    let n = A.rows
    var result = A
    var ipiv = [Int32](repeating: 0, count: n)
    var info: Int32 = 0
    var nI = Int32(n)
    var nI2 = Int32(n)
    var lda = Int32(n)

    // LU factorization
    dgetrf_(&nI, &nI2, &result.data, &lda, &ipiv, &info)
    guard info == 0 else { return nil }

    // Query optimal workspace
    var lwork: Int32 = -1
    var workQuery = [Double](repeating: 0, count: 1)
    nI = Int32(n)
    lda = Int32(n)

    dgetri_(&nI, &result.data, &lda, &ipiv, &workQuery, &lwork, &info)
    guard info == 0 else { return nil }

    lwork = Int32(workQuery[0])
    var work = [Double](repeating: 0, count: Int(lwork))
    nI = Int32(n)
    lda = Int32(n)

    // Compute inverse
    dgetri_(&nI, &result.data, &lda, &ipiv, &work, &lwork, &info)

    return info == 0 ? result : nil
}

// MARK: - Vector Operations

/// Compute dot product of two vectors
public func dotProduct(_ x: [Double], _ y: [Double]) -> Double {
    precondition(x.count == y.count, "Vectors must have same length")
    var result: Double = 0.0
    vDSP_dotprD(x, 1, y, 1, &result, vDSP_Length(x.count))
    return result
}

/// Compute L2 norm of vector
public func norm(_ x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_svesqD(x, 1, &result, vDSP_Length(x.count))
    return sqrt(result)
}

/// Vector addition: z = x + y
public func vectorAdd(_ x: [Double], _ y: [Double]) -> [Double] {
    precondition(x.count == y.count, "Vectors must have same length")
    var result = [Double](repeating: 0, count: x.count)
    vDSP_vaddD(x, 1, y, 1, &result, 1, vDSP_Length(x.count))
    return result
}

/// Vector subtraction: z = x - y
public func vectorSubtract(_ x: [Double], _ y: [Double]) -> [Double] {
    precondition(x.count == y.count, "Vectors must have same length")
    var result = [Double](repeating: 0, count: x.count)
    vDSP_vsubD(y, 1, x, 1, &result, 1, vDSP_Length(x.count))
    return result
}

/// Scalar multiplication: y = alpha * x
public func scalarMultiply(_ alpha: Double, _ x: [Double]) -> [Double] {
    var result = [Double](repeating: 0, count: x.count)
    var mutableAlpha = alpha
    vDSP_vsmulD(x, 1, &mutableAlpha, &result, 1, vDSP_Length(x.count))
    return result
}

/// Subtract column means from a matrix.
public func centerColumns(_ A: Matrix, means: [Double]) -> Matrix {
    precondition(A.cols == means.count, "Means count must match matrix columns")

    var result = A
    result.data.withUnsafeMutableBufferPointer { buffer in
        let base = buffer.baseAddress!
        for col in 0..<A.cols {
            var offset = -means[col]
            let column = base.advanced(by: col * A.rows)
            vDSP_vsaddD(column, 1, &offset, column, 1, vDSP_Length(A.rows))
        }
    }

    return result
}

/// Multiply each matrix row by the corresponding weight.
public func scaleRows(_ A: Matrix, weights: [Double]) -> Matrix {
    precondition(A.rows == weights.count, "Weights count must match matrix rows")

    var result = Matrix(rows: A.rows, cols: A.cols)
    A.data.withUnsafeBufferPointer { aBuffer in
        result.data.withUnsafeMutableBufferPointer { resultBuffer in
            weights.withUnsafeBufferPointer { weightsBuffer in
                let aBase = aBuffer.baseAddress!
                let resultBase = resultBuffer.baseAddress!
                let weightsBase = weightsBuffer.baseAddress!

                for col in 0..<A.cols {
                    let inputColumn = aBase.advanced(by: col * A.rows)
                    let outputColumn = resultBase.advanced(by: col * A.rows)
                    vDSP_vmulD(inputColumn, 1, weightsBase, 1, outputColumn, 1, vDSP_Length(A.rows))
                }
            }
        }
    }

    return result
}

/// Add a column-wise offset vector to every matrix row.
public func addVectorToEachRow(_ A: Matrix, vector: [Double]) -> Matrix {
    precondition(A.cols == vector.count, "Vector count must match matrix columns")

    var result = A
    result.data.withUnsafeMutableBufferPointer { buffer in
        let base = buffer.baseAddress!
        for col in 0..<A.cols {
            var offset = vector[col]
            let column = base.advanced(by: col * A.rows)
            vDSP_vsaddD(column, 1, &offset, column, 1, vDSP_Length(A.rows))
        }
    }

    return result
}

/// Frobenius inner product: sum(A .* B)
public func frobeniusInnerProduct(_ A: Matrix, _ B: Matrix) -> Double {
    precondition(A.rows == B.rows && A.cols == B.cols, "Matrix dimensions must match")
    return dotProduct(A.data, B.data)
}
