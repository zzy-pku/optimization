"""Positive-spectrum and block-subspace statistics for the seven-block experiment."""
import math
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence, ArpackError


def positive_spectrum(mv, n, requested, seed, solver_tol=1e-6, residual_tol=1e-4,
                      maxiter=500, ncv=None, progress=None, progress_every=50):
    """LA only. Preserve all returned eigenpairs, with numerical sign and convergence flags."""
    if not 0 < requested < n:
        raise ValueError('require 0 < requested < operator dimension')
    calls = 0
    def counted(x):
        nonlocal calls
        calls += 1
        if progress and calls % progress_every == 0:
            progress(calls)
        return np.asarray(mv(x), dtype=np.float64)
    op = LinearOperator((n, n), matvec=counted, dtype=np.float64)
    ncv = min(n, max(2 * requested + 1, 40) if ncv is None else ncv)
    if ncv <= requested:
        raise ValueError('ncv must exceed requested eigenpairs')
    rng = np.random.default_rng(seed)
    try:
        values, vectors = eigsh(op, k=requested, which='LA', tol=solver_tol,
                                maxiter=maxiter, ncv=ncv, v0=rng.normal(size=n))
        solver_converged, failure = True, None
    except ArpackNoConvergence as exc:
        values, vectors = exc.eigenvalues, exc.eigenvectors
        solver_converged, failure = False, str(exc)
    except ArpackError as exc:
        values, vectors = np.empty(0), np.empty((n, 0))
        solver_converged, failure = False, str(exc)
    order = np.argsort(values)[::-1]
    values, vectors = np.asarray(values)[order], np.asarray(vectors)[:, order]
    if len(values):
        vectors /= np.linalg.norm(vectors, axis=0, keepdims=True)
        applied = np.column_stack([counted(v) for v in vectors.T])
        absolute = np.linalg.norm(applied - vectors * values, axis=0)
        relative = absolute / (np.linalg.norm(applied, axis=0) + np.abs(values) + 1e-30)
    else:
        absolute = relative = np.empty(0)
    passed = relative <= residual_tol
    positive = (values > absolute) & passed
    # A partial ARPACK result does not certify the requested ordered leading space.
    valid_order = bool(solver_converged and len(values) == requested)
    return dict(values=values, vectors=vectors, absolute_residual=absolute,
                relative_residual=relative, residual_passed=passed,
                numerically_positive=positive, solver_converged=solver_converged,
                ordered_leading_space_certified=valid_order, failure=failure,
                requested=requested, returned=len(values), solver_calls=calls,
                solver_tol=solver_tol, acceptance_tol=residual_tol, ncv=ncv,
                orthogonality_error=float(np.linalg.norm(vectors.T @ vectors - np.eye(len(values)))))


def positive_count(spec):
    """Use only the leading consecutive positive, accepted and ordered pairs."""
    if not spec['ordered_leading_space_certified']:
        return 0
    count = 0
    for valid in spec['numerically_positive']:
        if not valid:
            break
        count += 1
    return count


def boundary(spec, rank):
    vals, res = spec['values'], spec['absolute_residual']
    if not spec['ordered_leading_space_certified'] or rank > positive_count(spec):
        return {'status': 'unavailable', 'rank': rank}
    if rank >= len(vals):
        return {'status': 'not_computed', 'rank': rank}
    if not spec['residual_passed'][rank]:
        return {'status': 'next_mode_unconverged', 'rank': rank,
                'next_value': float(vals[rank])}
    gap = float(vals[rank - 1] - vals[rank])
    threshold = float(10 * (res[rank - 1] + res[rank]))
    return {'status': 'cluster_boundary_ambiguous' if gap <= threshold else 'resolved',
            'rank': rank, 'gap': gap, 'residual_gap_threshold': threshold,
            'next_value': float(vals[rank]),
            'note': 'ambiguous cutoff is preliminary; extend solve and report the entire cluster'}


def block_coefficients(bases, target, slices):
    """Return P_i^T Q_i, using zero rows for absent blocks."""
    return {key: np.asarray(basis, dtype=np.float64).T @ target[slices[key]]
            for key, basis in bases.items()}


def geometry(coefficients, target_dim, basis_dim, target_gram=None):
    if target_dim <= 0:
        raise ValueError('target space must be nonempty')
    c = np.vstack(list(coefficients.values())) if coefficients else np.empty((0, target_dim))
    mode_capture = np.sum(c*c, axis=0).tolist()
    if target_gram is not None:
        # Orthonormalize the saved target basis without changing its subspace.
        factor = np.linalg.cholesky(target_gram)
        c = np.linalg.solve(factor, c.T).T
    if basis_dim:
        singular = np.clip(np.linalg.svd(c, compute_uv=False), 0, 1)
        energy = float(np.sum(c * c))
        angles = np.degrees(np.arccos(singular)).tolist()
        overlap = energy / min(basis_dim, target_dim)
        reverse = energy / basis_dim
    else:
        singular, angles, energy, overlap, reverse = np.empty(0), [], 0., None, None
    return dict(target_dim=target_dim, candidate_dim=basis_dim,
                H_to_candidate_containment=energy / target_dim,
                candidate_to_H_containment=reverse, overlap=overlap,
                principal_angles_deg=angles,
                angle_range_deg=[min(angles), max(angles)] if angles else None,
                unavoidable_unrepresented_target_dimensions=max(0, target_dim - basis_dim),
                capture_each_mode=mode_capture)


def _haar_top_rows(n, rows, cols, rng):
    """First rows of a Haar n x cols Stiefel matrix, sampled without an n x cols matrix."""
    if not (0 <= rows <= n and 0 <= cols <= n):
        raise ValueError('invalid Stiefel dimensions')
    if not cols:
        return np.empty((rows, 0))
    g1 = rng.normal(size=(rows, cols))
    df = n - rows
    if df >= cols:
        # Bartlett factor for G2^T G2, where G2 has df independent Gaussian rows.
        lower = np.zeros((cols, cols))
        for i in range(cols):
            lower[i, i] = math.sqrt(rng.chisquare(df - i))
            lower[i, :i] = rng.normal(size=i)
        gram = g1.T @ g1 + lower @ lower.T
    else:
        g2 = rng.normal(size=(df, cols))
        gram = g1.T @ g1 + g2.T @ g2
    factor = np.linalg.cholesky(gram)
    return np.linalg.solve(factor, g1.T).T


def block_random_baseline(target, slices, dimensions, repeats, seed):
    """Exact blockwise Haar null via a small Bartlett/Wishart Gram draw."""
    if repeats < 1:
        raise ValueError('random baseline repetitions must be positive')
    rng = np.random.default_rng(seed)
    r = target.shape[1]
    # QR of the target's rows supplies its Gram, including rank-deficient blocks.
    factors = {}
    expected = 0.
    for key, m in dimensions.items():
        qi = np.asarray(target[slices[key]], dtype=np.float64)
        n = qi.shape[0]
        if m > n:
            raise ValueError('candidate dimension exceeds block dimension')
        _, upper = np.linalg.qr(qi, mode='reduced')
        factors[key] = (n, upper)
        expected += m / n * float(np.sum(qi * qi)) / r
    samples = []
    for _ in range(repeats):
        energy = 0.
        for key, m in dimensions.items():
            n, upper = factors[key]
            top = _haar_top_rows(n, upper.shape[0], m, rng)
            energy += float(np.sum((upper.T @ top) ** 2))
        samples.append(energy / r)
    return dict(expected=expected, mean=float(np.mean(samples)),
                percentile_5=float(np.quantile(samples, .05)),
                percentile_95=float(np.quantile(samples, .95)),
                repetitions=repeats, seed=seed, samples=samples,
                method='independent blockwise Haar; QR and Bartlett/Wishart projection')
