"""Paired operator measurements and Hessian subspace/heatmap statistics."""
import numpy as np
import torch

from .large_spectrum import large_positive_spectrum
from .positive_subspaces import boundary, positive_count, positive_spectrum
from .seven_measure import write


PAIRS = (('M', 'G'), ('U', 'G'), ('U', 'M'))


def norm2(x):
    return float(torch.linalg.vector_norm(x)) ** 2


def directional_differences(actions, vectors, moments=None):
    rows = {f'{q}|{p}': [] for q, p in PAIRS}
    for vector in vectors:
        applied = {key: action(vector) for key, action in actions.items()}
        squares = {key: norm2(value) for key, value in applied.items()}
        if moments is not None:
            for key, value in applied.items():
                moments[key].append([float(torch.dot(vector, value)), squares[key]])
        for q, p in PAIRS:
            rows[f'{q}|{p}'].append([norm2(applied[q] - applied[p]), squares[p]])
    return rows


def ratio_summary(rows, bootstrap_seed=None):
    samples = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
    n, d = samples.sum(axis=0)
    ratio = n / d if d > 0 else None
    report = dict(squared_relative_error=ratio, relative_error=np.sqrt(ratio) if ratio is not None else None,
                  samples=samples, sample_columns=['numerator_norm2', 'denominator_norm2'])
    if bootstrap_seed is not None:
        rng = np.random.default_rng(bootstrap_seed)
        n, d = samples[rng.integers(len(samples), size=(2000, len(samples)))].sum(axis=1).T
        report['paired_bootstrap_ci95'] = np.quantile(n[d > 0] / d[d > 0], [.025, .975]) if (d > 0).any() else None
    return report


def rademacher(n, device, generator):
    return torch.randint(0, 2, (n,), dtype=torch.int8, device=device,
                         generator=generator).float().mul_(2).sub_(1)


def probe_differences(actions, n, device, count, seed, moments=None):
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = directional_differences(actions, (rademacher(n, device, generator) for _ in range(count)), moments)
    return {pair: ratio_summary(samples, seed+1) for pair, samples in rows.items()}


def momentum_differences(actions, momentum, native, scheme):
    outputs = {key: action(momentum) for key, action in actions.items()} if momentum is not None else {}
    metrics = {}
    for q, p in PAIRS:
        if outputs:
            n, d, qp = norm2(outputs[q]), norm2(outputs[p]), norm2(outputs[q] - outputs[p])
            cosine = float(torch.dot(outputs[q], outputs[p])) / np.sqrt(d*n) if d and n else None
            metrics[f'{q}|{p}'] = dict(ratio_summary([[qp, d]]), cosine=cosine,
                                        amplitude_ratio=np.sqrt(n/d) if d else None)
        else:
            metrics[f'{q}|{p}'] = dict(ratio_summary([]), cosine=None, amplitude_ratio=None)
    native_report = None if native is None else dict(
        ratio_summary([[norm2(outputs[scheme] - native), norm2(native)]]),
        native_norm=norm2(native) ** 0.5, calibrated_norm=norm2(outputs[scheme]) ** 0.5)
    return metrics, native_report


def eigenpairs(apply, n, k, device, args, progress=None):
    """Solve the algebraic upper edge, with at least one cutoff neighbor where possible."""
    requested = k + 1

    def mv(vector):
        x = torch.tensor(vector, device=device, dtype=torch.float32)
        return apply(x).detach().cpu().numpy()

    if n <= requested + 1:
        columns = np.eye(n, dtype=np.float32)
        matrix = np.column_stack([mv(x) for x in columns]).astype(np.float64)
        vals, vecs = np.linalg.eigh((matrix + matrix.T) * 0.5)
        vals, vecs = vals[::-1], vecs[:, ::-1]
        actions = matrix @ vecs
        absolute = np.linalg.norm(actions - vecs*vals, axis=0)
        residual = absolute / np.maximum(1e-12, np.maximum(np.linalg.norm(actions, axis=0), np.abs(vals)))
        passed = residual <= args.residual_tol
        return dict(values=vals, vectors=vecs, absolute_residual=absolute, relative_residual=residual,
                    residual_passed=passed, numerically_positive=(vals > absolute) & passed,
                    ordered_leading_space_certified=True, solver='dense small-scope check')
    solver = (large_positive_spectrum if n * max(2*requested+1, 40) > 2**31-1 else positive_spectrum)
    return solver(mv, n, requested, args.probe_seed, args.solver_tol, args.residual_tol,
                  args.maxiter, progress=progress, residual_scale='max')


def saved_spectrum(path, factory, k):
    vectors_path, metadata_path = path.with_suffix('.npy'), path.with_suffix('.json')
    if metadata_path.exists() and vectors_path.exists():
        import json
        info = json.loads(metadata_path.read_text(encoding='utf-8'))
    else:
        spec = factory()
        rank = min(k, positive_count(spec))
        info = {key: value for key, value in spec.items() if key != 'vectors'}
        info.update(used_positive_rank=rank,
                    boundary=boundary(spec, rank) if rank else {'status': 'unavailable'})
        temporary = path.with_suffix('.tmp.npy')
        np.save(temporary, spec['vectors'].astype(np.float32, copy=False))
        temporary.replace(vectors_path)
        write(metadata_path, info)
    return np.load(vectors_path, mmap_mode='r'), info


def coverage(partition, top, values, local, ranks):
    if len(values) == 0:
        return dict(status='unavailable', reason='no accepted global positive modes')
    blocks = {}
    mode_coverage = np.zeros(len(values))
    participation = np.zeros(len(values))
    for label, (sliced, vectors) in local.items():
        x = top[sliced, :len(values)]
        p = np.einsum('ij,ij->j', x, x, dtype=np.float64)
        a = np.sum((vectors[:, :ranks[label]].T @ x)**2, axis=0, dtype=np.float64)
        c = np.divide(a, p, out=np.full_like(a, np.nan), where=p > 0)
        blocks[label] = dict(rank=ranks[label], p=p, r=np.sqrt(p), c=c, a=a)
        mode_coverage += a
        participation += p
    if np.max(np.abs(participation - 1)) > 1e-4 or np.max(mode_coverage) > 1+1e-4:
        raise RuntimeError('Hessian participation/coverage failed the partition sum check')
    return dict(rank=sum(ranks.values()), p_sum=participation, mode_coverage=mode_coverage,
                f_1=np.mean(mode_coverage), f_lambda=np.average(mode_coverage, weights=values),
                f_lambda2=np.average(mode_coverage, weights=np.square(values)), blocks=blocks,
                dimension=partition.n)


def matched_ranks(first, second, first_values, second_values):
    """Remove lowest block eigenmodes if one scheme has fewer accepted directions."""
    target = min(sum(first.values()), sum(second.values()))
    result = []
    for ranks, eigenvalues in ((first, first_values), (second, second_values)):
        selected = sorted(((eigenvalues[label][i], label, i) for label, rank in ranks.items()
                           for i in range(rank)), key=lambda item: (-item[0], item[1], item[2]))[:target]
        kept = dict.fromkeys(ranks, 0)
        for _, label, _ in selected:
            kept[label] += 1
        result.append(kept)
    return result


def hessian_heatmap(apply, partition, scheme, probes, seed, device, progress=None):
    blocks = partition.blocks[scheme]
    slices = [partition.interval(entries) for entries in blocks.values()]
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = np.empty((len(blocks), len(blocks), probes))
    for j, source in enumerate(slices):
        for sample in range(probes):
            z = torch.zeros(partition.n, device=device)
            z[source] = rademacher(source.stop-source.start, device, generator)
            hz = apply(z)
            # Copy once per HVP to avoid synchronizing every individual CUDA norm.
            norms = torch.stack([torch.linalg.vector_norm(hz[part]) for part in slices])
            samples[:, j, sample] = norms.cpu().numpy().astype(np.float64)**2
        if progress:
            progress(column=list(blocks)[j])
    square = samples.mean(axis=-1)
    total = square.sum()
    sizes = np.array([part.stop-part.start for part in slices], dtype=np.float64)
    return dict(blocks=list(blocks), norm=np.sqrt(square),
                norm2_stderr=samples.std(axis=-1, ddof=1)/np.sqrt(probes),
                energy_fraction=square/total if total else None,
                off_fraction=1-np.trace(square)/total if total else None,
                element_rms=np.sqrt(square/np.outer(sizes, sizes)),
                norm2_samples=samples, probes=probes, probe_seed=seed)


def factor_heatmaps(partition, z, operators):
    result = {}
    for scheme in ('M', 'U'):
        groups = partition.blocks[scheme]
        rows = [z[entries[0].row_start:entries[-1].row_stop] for entries in groups.values()]
        right = list(operators[scheme].right_statistics.values())
        size = len(rows)
        left_norm, right_diff = np.zeros((size, size)), np.zeros((size, size))
        for i in range(size):
            for j in range(i+1):
                left_norm[i, j] = left_norm[j, i] = float(torch.linalg.vector_norm(rows[i] @ rows[j].T))
                right_diff[i, j] = right_diff[j, i] = float(torch.linalg.vector_norm(right[i] - right[j]))
        global_right = operators['G'].right_statistics['global']
        result[scheme] = dict(blocks=list(groups), global_left_norm=left_norm,
                              right_difference_norm=right_diff,
                              right_norm=[float(torch.linalg.vector_norm(x)) for x in right],
                              right_sum_relative_error=float(torch.linalg.vector_norm(
                                  torch.stack(right).sum(dim=0) - global_right) /
                                  torch.linalg.vector_norm(global_right).clamp_min(1e-30)))
    return result
