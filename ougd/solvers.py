import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence, ArpackError

def spectrum(mv, n, k, end, tol, maxiter, seed=0, dtype=np.float32):
    """ARPACK implicitly restarted Lanczos; never substitutes LM for LA/SA."""
    if not 0 < k < n:
        raise ValueError('Lanczos requires 0 < k < dimension')
    calls = 0
    def counted(v):
        nonlocal calls
        calls += 1
        return mv(v)
    op = LinearOperator((n, n), matvec=counted, dtype=dtype)
    converged = True
    failure = None
    try:
        values, vectors = eigsh(op, k=k, which=end, tol=tol, maxiter=maxiter,
            ncv=min(n, max(2*k+1, 20)), v0=np.random.default_rng(seed).normal(size=n).astype(dtype))
    except ArpackNoConvergence as e:
        values, vectors, converged = e.eigenvalues, e.eigenvectors, False
    except ArpackError as e:
        values, vectors, converged = np.empty(0), np.empty((n,0)), False
        failure = str(e)
    order = np.argsort(values)
    if end == 'LA': order = order[::-1]
    values, vectors = values[order], vectors[:, order]
    applied = np.column_stack([mv(v) for v in vectors.T]) if len(values) else np.empty((n, 0))
    residual = np.linalg.norm(applied-vectors*values, axis=0)
    relative = residual / (np.linalg.norm(applied, axis=0)+np.abs(values)+1e-30)
    return dict(values=values, vectors=vectors, residual=residual, relative_residual=relative, failure=failure,
                converged=converged and len(values) == k and bool(np.all(relative <= tol)),
                orthogonality_error=float(np.linalg.norm(vectors.T@vectors-np.eye(len(values)))),
                gaps=np.abs(np.diff(values)), solver_matvecs=calls)

def cross_svd(forward, transpose, ni, nj, k, tol, maxiter, seed, dtype):
    if not 0 < k <= min(ni, nj):
        raise ValueError('SVD k must not exceed the smaller block dimension')
    def dilation(v):
        return np.concatenate((forward(v[ni:]), transpose(v[:ni])))
    s = spectrum(dilation, ni+nj, k, 'LA', tol, maxiter, seed, dtype)
    u, v = s['vectors'][:ni].copy(), s['vectors'][ni:].copy()
    u /= np.maximum(np.linalg.norm(u, axis=0), 1e-30)
    v /= np.maximum(np.linalg.norm(v, axis=0), 1e-30)
    fv = np.column_stack([forward(x) for x in v.T]) if v.shape[1] else np.empty_like(u)
    tv = np.column_stack([transpose(x) for x in u.T]) if u.shape[1] else np.empty_like(v)
    sigma = np.einsum('ij,ij->j', u, fv)
    signs = np.where(sigma < 0, -1, 1)
    u *= signs
    tv *= signs
    sigma = np.abs(sigma)
    left = np.linalg.norm(fv-u*sigma, axis=0)
    right = np.linalg.norm(tv-v*sigma, axis=0)
    rel_left = left/(np.linalg.norm(fv, axis=0)+sigma+1e-30)
    rel_right = right/(np.linalg.norm(tv, axis=0)+sigma+1e-30)
    order = np.argsort(sigma)[::-1]
    return dict(values=sigma[order], left=u[:, order], right=v[:, order],
        left_residual=left[order], right_residual=right[order],
        left_relative_residual=rel_left[order], right_relative_residual=rel_right[order],
        converged=s['converged'] and bool(np.all(rel_left <= tol)) and bool(np.all(rel_right <= tol)),
        solver_matvecs=s['solver_matvecs'],failure=s.get('failure'))

def overlap(a, b):
    if a.shape[1] == 0 or b.shape[1] == 0:
        return dict(overlap=None, angles=None)
    # QR is needed for sliced joint vectors; caller uses full/self vectors here.
    a, _ = np.linalg.qr(a)
    b, _ = np.linalg.qr(b)
    cosines = np.clip(np.linalg.svd(a.T@b, compute_uv=False), 0, 1)
    return dict(overlap=float(np.sum(cosines**2)/min(a.shape[1], b.shape[1])),
                angles=np.arccos(cosines), dimensions=[a.shape[1], b.shape[1]])


def zero_spectrum(n, k, dtype):
    """Analytically known zero operator; every direction is an eigenvector."""
    if not 0 < k < n: raise ValueError('require 0 < k < dimension')
    vectors=np.zeros((n,k),dtype=dtype)
    vectors[np.arange(k),np.arange(k)]=1
    return dict(values=np.zeros(k,dtype=dtype),vectors=vectors,residual=np.zeros(k),
        relative_residual=np.zeros(k),converged=True,orthogonality_error=0.,gaps=np.zeros(max(0,k-1)),
        solver_matvecs=0,structural_zero=True,fully_degenerate=True,
        verification='anchor affine in this parameter block; residual self Hessian is identically zero',
        direction_note='canonical arbitrary basis; no identifiable sharp subspace')
