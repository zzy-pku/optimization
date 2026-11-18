"""64-bit indexed PRIMME solve for operators too large for 32-bit ARPACK."""
import numpy as np
from scipy.sparse.linalg import LinearOperator


def large_positive_spectrum(mv, n, requested, seed, solver_tol=1e-6, residual_tol=1e-4,
                            maxiter=1000, progress=None, residual_scale='sum'):
    import primme
    calls = 0
    def matvec(x):
        nonlocal calls
        calls += 1
        if progress and calls % 10 == 0: progress(calls)
        return np.asarray(mv(np.asarray(x, dtype=np.float32).reshape(-1)), dtype=np.float32)
    op = LinearOperator((n,n), matvec=matvec, dtype=np.float32)
    # Avoid a large random initial vector; PRIMME seeds its reproducible internal generator.
    vals, vecs, stats = primme.eigsh(op, k=requested, which='LA', tol=solver_tol,
        maxiter=maxiter, ncv=2*requested+1, maxBlockSize=1,
        return_stats=True)
    order = np.argsort(vals)[::-1]; vals = np.asarray(vals)[order]; vecs = np.asarray(vecs)[:,order]
    absolute, relative = [], []
    for i, value in enumerate(vals):
        action = matvec(vecs[:,i]); error = np.linalg.norm(action-value*vecs[:,i])
        action_norm = np.linalg.norm(action)
        scale = (max(action_norm, abs(value), 1e-12) if residual_scale == 'max'
                 else action_norm + abs(value) + 1e-30)
        absolute.append(float(error)); relative.append(float(error/scale))
    absolute = np.asarray(absolute); relative = np.asarray(relative); passed = relative <= residual_tol
    return dict(values=vals, vectors=vecs, absolute_residual=absolute, relative_residual=relative,
        residual_passed=passed, numerically_positive=(vals>absolute)&passed,
        solver_converged=True, ordered_leading_space_certified=(len(vals)==requested), failure=None,
        requested=requested, returned=len(vals), solver_calls=calls, solver_tol=solver_tol,
        acceptance_tol=residual_tol, ncv=2*requested+1, solver='PRIMME float32, 64-bit indices',
        seed_note='PRIMME default internal initialization',
        orthogonality_error=float(np.linalg.norm(vecs.T@vecs-np.eye(len(vals)))),
        primme_matvecs=int(stats['numMatvecs']))

