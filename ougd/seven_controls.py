"""Optional, explicitly budgeted controls for the seven-block experiment."""
import numpy as np
from .positive_subspaces import positive_spectrum, positive_count, geometry, boundary


class SignedPermutationCross:
    """D + R(H-D)R^T; R is blockwise signed permutation, never a dense rotation."""
    def __init__(self, backend, seed):
        self.backend = backend
        self.keys = tuple(backend.params)
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.permutations = {k: rng.permutation(backend.sizes[k]) for k in self.keys}
        self.signs = {k: rng.choice([-1., 1.], backend.sizes[k]) for k in self.keys}
        self.calls = 0

    def rotate(self, vector, transpose=False):
        out = np.empty(self.backend.n, dtype=np.float64)
        for key in self.keys:
            sl = self.backend.slices[key]; x = np.asarray(vector[sl]); perm = self.permutations[key]; signs = self.signs[key]
            if transpose:
                y = np.empty_like(x, dtype=np.float64); y[perm] = signs * x
                out[sl] = y
            else:
                out[sl] = signs * x[perm]
        return out

    def diagonal(self, vector):
        out = np.zeros(self.backend.n, dtype=np.float64)
        for key in self.keys:
            sl = self.backend.slices[key]
            out[sl] = self.backend.block(key, key, np.asarray(vector[sl]))
        return out

    def __call__(self, vector):
        self.calls += 1
        vector = np.asarray(vector, dtype=np.float64)
        rotated = self.rotate(vector, transpose=True)
        h = self.backend.request(rotated, 1)
        return self.diagonal(vector) + self.rotate(h - self.diagonal(rotated))


def adaptive_diagonal_top(backend, original_specs, target, candidate_dim, seed,
                          solver_tol, residual_tol, maxiter, ncv=None, progress=None,
                          on_refined=None):
    """Top-d of D_H: refine only blocks whose unreturned edge might change ordering.

    Certification relies on an ordered converged LA solve plus residual margins;
    ARPACK itself is numerical, not a formal exact-arithmetic proof.
    """
    keys = tuple(backend.params)
    specs = dict(original_specs)
    refinements = []
    if candidate_dim < 1:
        return {'status':'no_positive_candidate_directions'}
    while True:
        for key in keys:
            s = specs[key]
            if not s['ordered_leading_space_certified'] or not np.all(s['residual_passed']):
                return dict(status='uncertified_self_spectrum', block=key, refinements=refinements)
        candidates = sorted([(float(specs[k]['values'][j]), k, j)
                             for k in keys for j in range(positive_count(specs[k]))], reverse=True)
        cutoff = None
        if len(candidates) >= candidate_dim:
            val, key, j = candidates[candidate_dim-1]
            cutoff = val-float(specs[key]['absolute_residual'][j])
        needs = []
        for key in keys:
            s = specs[key]; last = float(s['values'][-1]); err = float(s['absolute_residual'][-1])
            # The next unseen value is no larger than the last returned value,
            # up to the estimated error on that edge eigenpair.
            upper = last + err
            if upper > 0 and (cutoff is None or upper >= cutoff):
                needs.append(key)
        if not needs:
            if len(candidates) < candidate_dim:
                return dict(status='insufficient_positive_directions', requested_dimension=candidate_dim,
                            available_positive=len(candidates), refinements=refinements)
            break
        key = needs[0]; old = specs[key]; n = backend.sizes[key]
        requested = min(n-1, candidate_dim+1, max(old['requested']+1, 2*old['requested']))
        if requested <= old['requested']:
            return dict(status='unresolved_unreturned_boundary', block=key,
                        requested_dimension=candidate_dim, refinements=refinements)
        if progress: progress('diagonal_refinement', block=key, requested=requested)
        safe_ncv = None if ncv is None or ncv <= requested else ncv
        new = positive_spectrum(lambda v, key=key: backend.block(key,key,v), n, requested,
                                seed+len(refinements), solver_tol, residual_tol, maxiter, safe_ncv,
                                (lambda calls, key=key: progress('diagonal_refinement_progress', block=key, calls=calls)) if progress else None)
        refinements.append(dict(block=key, previous=old['requested'], requested=requested,
                                converged=new['ordered_leading_space_certified'] and bool(np.all(new['residual_passed']))))
        specs[key] = new
        if on_refined: on_refined(key, new)
    selected = candidates[:candidate_dim]
    coefficients = np.vstack([specs[key]['vectors'][:, j] @ target[backend.slices[key]]
                              for _, key, j in selected])
    result = geometry({'D_H': coefficients}, target.shape[1], candidate_dim, target.T @ target)
    edge = None
    if len(candidates) > candidate_dim:
        left, right = candidates[candidate_dim-1], candidates[candidate_dim]
        gap = left[0]-right[0]
        threshold = 10*(specs[left[1]]['absolute_residual'][left[2]] + specs[right[1]]['absolute_residual'][right[2]])
        edge = dict(gap=float(gap), residual_gap_threshold=float(threshold),
                    status='cluster_boundary_ambiguous' if gap <= threshold else 'resolved')
    return dict(status='complete' if edge is None or edge['status']=='resolved' else 'preliminary_boundary_ambiguous',
                requested_dimension=candidate_dim, geometry=result, boundary=edge,
                sources=[dict(block=k,index=j+1,value=value) for value,k,j in selected],
                refinements=refinements,
                note='Numerically certified using ordered LA returns and residual margins; not an exact-arithmetic eigenvalue bound.')


def rotated_cross_control(backend, bases4, bases7, target_spec, k, repeats, seed,
                          solver_tol, residual_tol, maxiter, ncv=None, progress=None):
    if repeats < 1:
        return {'status':'skipped_by_request', 'replicates':[]}
    keys = tuple(backend.params); results = []
    for i in range(repeats):
        control = SignedPermutationCross(backend, seed+i)
        before = backend.calls
        if progress: progress('cross_randomization_start', replicate=i+1, repetitions=repeats)
        spec = positive_spectrum(control, backend.n, k+1, seed+100+i, solver_tol,
                                 residual_tol, maxiter, ncv,
                                 (lambda calls, i=i: progress('cross_randomization_progress', replicate=i+1, calls=calls)) if progress else None,
                                 progress_every=5)
        item = dict(replicate=i+1, seed=seed+i, matvecs=control.calls,
                    underlying_H_requests=backend.calls-before,
                    ordered_leading_space_certified=spec['ordered_leading_space_certified'],
                    returned=len(spec['values']), values=spec['values'].tolist())
        if positive_count(spec) >= k:
            target = spec['vectors'][:, :k]
            for label, bases in [('S4',bases4),('S7',bases7)]:
                coefficients={key:basis.T @ target[backend.slices[key]] for key,basis in bases.items()}
                item[label] = geometry(coefficients, k, sum(b.shape[1] for b in bases.values()), target.T@target)
            item['target_boundary']=boundary(spec,k)
            item['status']='complete' if item['target_boundary']['status']=='resolved' else 'preliminary_boundary_ambiguous'
        else:
            item['status']='uncertified_or_insufficient_positive_target'
        results.append(item)
    return dict(status='complete' if all(x['status']=='complete' for x in results) else 'followup_required',
                distribution='independent signed permutations within parameter blocks; not Haar',
                artificial_operator='D_H + R (H-D_H) R^T',
                baseline_H_values=target_spec['values'][:k].tolist(), replicates=results,
                note='The rotated matrix is a synthetic control, not the Hessian of the trained model; its spectrum can change.')
