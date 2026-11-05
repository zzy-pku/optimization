"""Full-matrix H/A/R block Frobenius statistics with paired probes."""
from itertools import combinations_with_replacement
import numpy as np


def _stats(values, rng, bootstrap):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(np.mean(values))
    stderr = float(np.std(values, ddof=1) / np.sqrt(n))
    picks = rng.integers(0, n, size=(bootstrap, n))
    lo, hi = np.quantile(np.mean(values[picks], axis=1), [.025, .975])
    return dict(mean=mean, stderr=stderr, ci95=[float(lo), float(hi)],
                relative_ci_halfwidth=float((hi-lo)/(2*abs(mean))) if mean > 1e-30 else None)


def frobenius_statistics(backend, probes=64, seed=23, bootstrap=1000, progress=None):
    if probes < 2 or bootstrap < 20:
        raise ValueError('at least two probes and 20 bootstrap draws required')
    keys = tuple(backend.params)
    rng = np.random.default_rng(seed)
    samples = {i+j: {k: [] for k in ('H2', 'A2', 'R2', 'AR', 'R2_raw_difference')}
               for i, j in combinations_with_replacement(keys, 2)}
    affine = set(getattr(getattr(backend, 'model', None), 'anchor_affine_blocks', ()))
    for j_index, j in enumerate(keys):
        for index in range(probes):
            full = np.zeros(backend.n, dtype=np.float64)
            full[backend.slices[j]] = rng.choice([-1., 1.], backend.sizes[j])
            h = np.asarray(backend.request(full, 1), dtype=np.float64)
            a = np.asarray(backend.request(full, 2), dtype=np.float64)
            residual = h - a
            for i in keys[:j_index + 1]:
                h_i, a_i, r_i = h[backend.slices[i]], a[backend.slices[i]], residual[backend.slices[i]]
                row = samples[i+j]
                row['R2_raw_difference'].append(float(r_i @ r_i))
                if i == j and i in affine:
                    # The block-output anchor is affine in this block: R_ii is exactly zero.
                    # Retain raw H-A roundoff separately for validation.
                    r_i = np.zeros_like(r_i)
                row['H2'].append(float(h_i @ h_i))
                row['A2'].append(float(a_i @ a_i))
                row['R2'].append(float(r_i @ r_i))
                row['AR'].append(float(a_i @ r_i))
        if progress:
            progress('frobenius_block', block=j, probes=probes)
    boot_rng = np.random.default_rng(seed + 401)
    results = {}
    for pair, sample in samples.items():
        i, j = pair
        row = {}
        for label in ('H', 'A', 'R'):
            stats = _stats(sample[label+'2'], boot_rng, bootstrap)
            lo, hi = stats['ci95']
            row[label] = dict(norm2=stats['mean'], norm2_stderr=stats['stderr'],
                              norm2_ci95=stats['ci95'],
                              norm2_relative_ci_halfwidth=stats['relative_ci_halfwidth'],
                              norm=float(np.sqrt(max(stats['mean'], 0))),
                              norm_ci95=[float(np.sqrt(max(lo, 0))), float(np.sqrt(max(hi, 0)))],
                              element_rms=float(np.sqrt(max(stats['mean'], 0) /
                                                        (backend.sizes[i] * backend.sizes[j]))),
                              sqrt_estimator_biased=True)
        inner = _stats(sample['AR'], boot_rng, bootstrap)
        denom = np.sqrt(row['A']['norm2'] * row['R']['norm2'])
        row['inner_AR'] = inner
        row['cosine_AR'] = float(inner['mean'] / denom) if denom > 1e-30 else None
        # Resample paired probe indices for the nonlinear cosine confidence interval.
        picks = boot_rng.integers(0, probes, size=(bootstrap, probes))
        a = np.mean(np.asarray(sample['A2'])[picks], axis=1)
        r = np.mean(np.asarray(sample['R2'])[picks], axis=1)
        ar = np.mean(np.asarray(sample['AR'])[picks], axis=1)
        valid = (a * r) > 1e-60
        row['cosine_AR_ci95'] = [float(x) for x in np.quantile(ar[valid] / np.sqrt(a[valid]*r[valid]), [.025, .975])] if valid.any() else None
        row['paired_frobenius_identity_error'] = float(row['H']['norm2'] - row['A']['norm2'] - row['R']['norm2'] - 2*inner['mean'])
        row['R_self_analytic_zero'] = bool(i == j and i in affine)
        if row['R_self_analytic_zero']:
            row['R_raw_difference_norm2_mean'] = float(np.mean(sample['R2_raw_difference']))
            row['identity_note'] = 'R self is analytically zero; H/A floating-point roundoff can leave a small identity discrepancy.'
        row['probes'] = probes
        row['samples'] = sample
        results[pair] = row
    scopes = {}
    for scope, selected in [('seven', keys), ('four', tuple(k for k in keys if k in 'OUGD'))]:
        scopes[scope] = {}
        # Probe samples for different target blocks share each source-block draw.
        # Resample source blocks independently, but retain that within-source pairing.
        picks_by_source = {j: boot_rng.integers(0, probes, size=(bootstrap, probes)) for j in selected}
        for label in ('H', 'A', 'R'):
            diagonal = sum(results[k+k][label]['norm2'] for k in selected)
            cross = 2 * sum(results[i+j][label]['norm2'] for i, j in combinations_with_replacement(selected, 2) if i != j)
            total = diagonal + cross
            total_draws = np.zeros(bootstrap); cross_draws = np.zeros(bootstrap)
            for i, j in combinations_with_replacement(selected, 2):
                means = np.mean(np.asarray(samples[i+j][label+'2'])[picks_by_source[j]], axis=1)
                total_draws += means * (1 if i == j else 2)
                if i != j: cross_draws += 2 * means
            valid = total_draws > 1e-30
            scopes[scope][label] = dict(norm2=total, norm=float(np.sqrt(max(total, 0))),
                norm2_ci95=[float(x) for x in np.quantile(total_draws, [.025, .975])],
                cross_norm2=cross, cross_fraction=cross/total if total else None,
                cross_fraction_ci95=[float(x) for x in np.quantile(cross_draws[valid]/total_draws[valid], [.025, .975])] if valid.any() else None,
                ci_method='paired within source block; independent across source blocks')
    return dict(block_order=list(keys), probes=probes, seed=seed, bootstrap_draws=bootstrap,
                scope='full matrix; all spectral signs', estimator='shared Rademacher probes',
                blocks=results, scopes=scopes,
                ci_note='probe bootstrap intervals; not a guarantee under adaptive repeated looks')
