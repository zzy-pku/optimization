"""Single-process seven-block positive-spectrum and full-Frobenius experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from .components import Decomposition
from .data import sha256
from .model import ModelConfig, Transformer, SEVEN_BLOCKS
from .positive_subspaces import (positive_spectrum, positive_count, boundary,
                                 block_coefficients, geometry, block_random_baseline)
from .seven_frobenius import frobenius_statistics
from .seven_controls import adaptive_diagonal_top, rotated_cross_control

FOUR = ('O', 'U', 'G', 'D')


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def batch_hash(batch):
    return hashlib.sha256(b''.join(batch[k].contiguous().cpu().numpy().tobytes()
                                   for k in ('input_ids', 'labels'))).hexdigest()


def git_provenance():
    try:
        root = Path(__file__).resolve().parent.parent
        def call(*args):
            return subprocess.check_output(['git', *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL, timeout=5).strip()
        return dict(commit=call('rev-parse', 'HEAD'), dirty=bool(call('status', '--porcelain')))
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, dirty=None)


def _expand_subset(vectors, subset, backend):
    out = np.zeros((backend.n, vectors.shape[1]), dtype=np.float64)
    pos = 0
    for key in subset:
        n = backend.sizes[key]
        out[backend.slices[key]] = vectors[pos:pos+n]
        pos += n
    return out


def _prepare_bases(specs, keys, m):
    bases, counts = {}, {}
    for key in keys:
        count = min(m, positive_count(specs[key]))
        counts[key] = count
        if count:
            bases[key] = np.linalg.qr(specs[key]['vectors'][:, :count], mode='reduced')[0]
    return bases, counts


def _boundary_summary(specs, keys, m):
    return {key: boundary(specs[key], m) for key in keys}


def _spectrum_metadata(spec, cutoffs):
    metadata = {key: value for key, value in spec.items() if key != 'vectors'}
    metadata['positive_count_certified'] = positive_count(spec)
    metadata['boundaries'] = {str(rank): boundary(spec, rank) for rank in cutoffs}
    metadata['nonpositive_note'] = 'Nonpositive or uncertain LA values are recorded for sign/boundary diagnosis, never included as positive directions.'
    return metadata


def verify_backend(backend, seed, tolerance):
    rng = np.random.default_rng(seed)
    u, v = (rng.normal(size=backend.n) for _ in range(2))
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    h_u, h_v = backend.request(u, 1), backend.request(v, 1)
    h_sum = backend.request(u+v, 1)
    a_v, r_v = backend.request(v, 2), backend.request(v, 4)
    scale = np.linalg.norm(h_v) + np.linalg.norm(a_v) + np.linalg.norm(r_v) + 1e-30
    decomposition = float(np.linalg.norm(h_v-a_v-r_v) / scale)
    symmetry = float(abs(u @ h_v - v @ h_u) /
                     (np.linalg.norm(h_u)+np.linalg.norm(h_v)+1e-30))
    linearity = float(np.linalg.norm(h_sum-h_u-h_v) /
                      (np.linalg.norm(h_sum)+np.linalg.norm(h_u)+np.linalg.norm(h_v)+1e-30))
    # Replay as individual sequences to independently exercise token-weighted aggregation.
    singles = [{key: value[i:i+1] for key, value in batch.items()}
               for batch in backend.batches for i in range(len(batch['labels']))
               if (batch['labels'][i:i+1] != -100).any()]
    alternate = Decomposition(backend.model, singles, backend.total_tokens, 0, blocks=SEVEN_BLOCKS)
    alt_h = alternate.request(v, 1)
    aggregation = float(np.linalg.norm(h_v-alt_h)/(np.linalg.norm(h_v)+np.linalg.norm(alt_h)+1e-30))
    grad, alt_grad = backend.gradient(), alternate.gradient()
    gradient_aggregation = float(np.linalg.norm(grad-alt_grad)/(np.linalg.norm(grad)+np.linalg.norm(alt_grad)+1e-30))
    values = dict(H_equals_A_plus_direct_R=decomposition, H_symmetry=symmetry,
                  H_linearity=linearity, H_token_aggregation=aggregation,
                  gradient_token_aggregation=gradient_aggregation,
                  independent_aggregation_HVP_requests=alternate.calls, tolerance=tolerance,
                  passed=max(decomposition, symmetry, linearity, aggregation, gradient_aggregation) <= tolerance)
    if not values['passed']:
        raise RuntimeError(f'operator verification failed: {values}')
    return values


def projection(backend, bases, progress):
    order = list(bases)
    offsets = {}; pos = 0
    for key in order:
        offsets[key] = slice(pos, pos + bases[key].shape[1]); pos += bases[key].shape[1]
    if not pos:
        return np.empty((0, 0)), np.empty((backend.n, 0)), offsets
    dtype = np.float64 if backend.dtype == torch.float64 else np.float32
    hp = np.empty((backend.n, pos), dtype=dtype)
    for key in order:
        for j in range(bases[key].shape[1]):
            v = np.zeros(backend.n)
            v[backend.slices[key]] = bases[key][:, j]
            hp[:, offsets[key].start+j] = backend.request(v, 1)
        progress('projection', block=key, completed=offsets[key].stop, total=pos)
    matrix = np.vstack([bases[key].T @ hp[backend.slices[key], :].astype(np.float64) for key in order])
    asym = np.linalg.norm(matrix - matrix.T) / (np.linalg.norm(matrix) + 1e-30)
    if asym > 1e-4:
        raise RuntimeError(f'projected Hessian symmetry failed: {asym}')
    return (matrix+matrix.T)/2, hp, offsets


def ritz_analysis(K, hp, bases, offsets, backend, basis_keys, target_keys, target_spec, target_r,
                  residual_tolerance=1e-4):
    indices = np.concatenate([np.arange(offsets[k].start, offsets[k].stop) for k in basis_keys if k in offsets]) if any(k in offsets for k in basis_keys) else np.array([], dtype=int)
    if not len(indices):
        return {'status': 'no_positive_candidate_directions'}
    local = K[np.ix_(indices, indices)]
    vals, vecs = np.linalg.eigh(local)
    order = np.argsort(vals)[::-1]; vals, vecs = vals[order], vecs[:, order]
    nr = min(target_r, len(vals))
    theta, y = vals[:nr], vecs[:, :nr]
    rel = []
    local_offsets = {}; pos = 0
    for key in basis_keys:
        if key in bases:
            local_offsets[key] = slice(pos, pos+bases[key].shape[1]); pos += bases[key].shape[1]
    # The candidate can be S4 even when the target/residual scope is the full H7.
    for j in range(nr):
        norm_res2 = norm_action2 = 0.
        for key in target_keys:
            vector = (bases[key] @ y[local_offsets[key], j] if key in local_offsets
                      else np.zeros(backend.sizes[key]))
            action = hp[backend.slices[key]][:, indices] @ y[:, j]
            norm_res2 += float(np.sum((action-theta[j]*vector)**2))
            norm_action2 += float(np.sum(action**2))
        rel.append(float(np.sqrt(norm_res2)/(np.sqrt(norm_action2)+abs(theta[j])+1e-30)))
    eig = target_spec['values'][:nr]
    valid = (theta > 0) & (eig > 0)
    ratios = [float(theta[i]/eig[i]) if valid[i] else None for i in range(nr)]
    complete = nr == target_r and all(x is not None for x in ratios)
    return dict(status='complete' if complete else 'insufficient_positive_ritz_values',
                candidate_dim=len(indices), target_rank=target_r,
                Ritz_values=[float(value) if value > 0 else None for value in theta],
                target_eigenvalues=eig.tolist(), rankwise_ratio=ratios,
                spectrum_mass_ratio=float(np.sum(theta)/np.sum(eig)) if complete else None,
                spectrum_relative_L2_error=float(np.linalg.norm(theta-eig)/np.linalg.norm(eig)) if complete else None,
                full_scope_relative_residual=rel,
                residual_tolerance=residual_tolerance,
                all_ritz_formally_converged=bool(complete and all(x <= residual_tolerance for x in rel)),
                note='Ritz residual is in H_4 or H_7 according to selected scope; value agreement is not eigenvector convergence.')


def run_analysis(backend, args, output, progress):
    targets = {}; specs = {}
    for key in SEVEN_BLOCKS:
        n = backend.sizes[key]
        extra = 35 if args.equal_dimension_control and key in FOUR else args.k
        requested = max(args.k, extra) + 1
        progress('spectrum', scope=key, requested=requested)
        specs[key] = positive_spectrum(lambda v, key=key: backend.block(key, key, v), n,
                                        requested, args.seed, args.solver_tol, args.residual_tol,
                                        args.maxiter, args.ncv,
                                        lambda calls, key=key: progress('spectrum_progress', scope=key, calls=calls))
    for label, keys in [('H4', FOUR), ('H7', SEVEN_BLOCKS)]:
        n = sum(backend.sizes[k] for k in keys)
        progress('spectrum', scope=label, requested=args.k+1)
        targets[label] = positive_spectrum(lambda v, keys=keys: backend.joint(keys, v), n,
                                            args.k+1, args.seed+1, args.solver_tol,
                                            args.residual_tol, args.maxiter, args.ncv,
                                            lambda calls, label=label: progress('spectrum_progress', scope=label, calls=calls))
    allspec = {**specs, **targets}
    for key, spec in allspec.items():
        count = positive_count(spec)
        np.savez_compressed(output / f'spectrum_{key}.npz',
                            positive_vectors=spec['vectors'][:, :count].astype(np.float32),
                            values=spec['values'], absolute_residual=spec['absolute_residual'],
                            relative_residual=spec['relative_residual'])
    write(output/'spectra.json', {key: _spectrum_metadata(spec, [10, 20]) for key, spec in allspec.items()})
    targets_full = {name: _expand_subset(spec['vectors'][:, :positive_count(spec)],
                    FOUR if name == 'H4' else SEVEN_BLOCKS, backend) for name, spec in targets.items()}
    summaries = []; primary_bases = None
    for m in sorted(set(x for x in [1, 2, 5, 10, args.k] if x <= args.k)):
        bases7, counts7 = _prepare_bases(specs, SEVEN_BLOCKS, m)
        bases4 = {key: bases7[key] for key in FOUR if key in bases7}
        if m == args.k: primary_bases = bases7
        for label, target_name, bases, keys in [('S4_H4', 'H4', bases4, FOUR),
                                                 ('S4_H7', 'H7', bases4, FOUR),
                                                 ('S7_H7', 'H7', bases7, SEVEN_BLOCKS)]:
            for rank in sorted(set(x for x in [10, args.k] if x <= args.k)):
                if rank > positive_count(targets[target_name]):
                    summaries.append(dict(comparison=label, target_rank=rank, self_rank=m,
                                          status='insufficient_certified_positive_target',
                                          candidate_block_dimensions={k: counts7[k] for k in keys}))
                    continue
                q = targets_full[target_name][:, :rank]
                coefficients = block_coefficients(bases, q, backend.slices)
                dim = sum(b.shape[1] for b in bases.values())
                result = geometry(coefficients, rank, dim, q.T @ q)
                block_capture = {key: (np.sum(coefficients[key]**2, axis=0).tolist()
                                       if key in coefficients else [0.]*rank) for key in keys}
                participation = {key: float(np.sum(q[backend.slices[key]]**2)/rank) for key in keys}
                dims = {key: counts7[key] for key in keys}
                null = block_random_baseline(q, backend.slices, dims, args.random_repeats,
                                             args.seed + 1000 + m*31 + rank)
                warnings = _boundary_summary(specs, keys, m)
                target_boundary = boundary(targets[target_name], rank)
                stable = all(x['status']=='resolved' for x in warnings.values()) and target_boundary['status']=='resolved'
                status = 'complete' if stable and all(x == m for x in dims.values()) else 'preliminary_boundary_or_insufficient_positive'
                item = dict(comparison=label, target_rank=rank, self_rank=m, status=status,
                            candidate_block_dimensions=dims, geometry=result,
                            block_capture_each_mode=block_capture, block_participation=participation,
                            random_block_baseline=null, self_boundaries=warnings,
                            target_boundary=target_boundary)
                if label == 'S4_H7':
                    ceiling = sum(participation.values())
                    item['four_coordinate_participation_ceiling'] = ceiling
                    item['conditional_four_capture'] = (result['H_to_candidate_containment']/ceiling if ceiling > 1e-12 else None)
                summaries.append(item)
    write(output/'coverage.json', summaries)
    ritz = {}; optional_controls = {}
    if primary_bases:
        K, hp, offsets = projection(backend, primary_bases, progress)
        np.savez_compressed(output/'projected_H.npz', K=K, offsets=np.array([[offsets[k].start, offsets[k].stop] if k in offsets else [-1, -1] for k in SEVEN_BLOCKS]))
        if args.save_hp: np.save(output/'projected_H_actions.npy', hp)
        for name, target, basis_keys, target_keys in [('H4', 'H4', FOUR, FOUR),
                                                       ('S4_H7', 'H7', FOUR, SEVEN_BLOCKS),
                                                       ('H7', 'H7', SEVEN_BLOCKS, SEVEN_BLOCKS)]:
            if positive_count(targets[target]) >= args.k:
                ritz[name] = ritz_analysis(K, hp, primary_bases, offsets, backend,
                                           basis_keys, target_keys, targets[target], args.k,
                                           args.residual_tol)
            else:
                ritz[name] = {'status':'insufficient_certified_positive_target'}
        if args.cross_randomizations:
            if positive_count(targets['H7']) >= args.k:
                four_bases = {key: primary_bases[key] for key in FOUR if key in primary_bases}
                optional_controls['cross_randomization'] = rotated_cross_control(
                    backend, four_bases, primary_bases, targets['H7'], args.k,
                    args.cross_randomizations, args.seed+2000, args.solver_tol,
                    args.residual_tol, args.maxiter, args.ncv, progress)
            else:
                optional_controls['cross_randomization'] = {'status':'insufficient_certified_positive_target'}
            write(output/'cross_randomization.json', optional_controls['cross_randomization'])
    else:
        ritz = {'status': 'no_certified_positive_candidate_space'}
        if args.cross_randomizations:
            optional_controls['cross_randomization'] = {'status': 'no_certified_positive_candidate_space'}
            write(output/'cross_randomization.json', optional_controls['cross_randomization'])
    write(output/'ritz.json', ritz)
    # Equal-dimensional D_H-top-r against H_r, using only certified positive pairs.
    d_h = {}
    for label, keys in [('H4', FOUR), ('H7', SEVEN_BLOCKS)]:
        if any(not specs[key]['ordered_leading_space_certified'] for key in keys):
            d_h[label] = {'status': 'uncertified_self_spectrum'}; continue
        candidates = sorted([(float(specs[key]['values'][j]), key, j)
                             for key in keys for j in range(positive_count(specs[key]))], reverse=True)
        if len(candidates) < args.k or positive_count(targets[label]) < args.k:
            d_h[label] = {'status': 'insufficient_certified_positive_directions'}; continue
        target = targets_full[label][:, :args.k]
        coeff = np.vstack([specs[key]['vectors'][:, j] @ target[backend.slices[key]]
                           for _, key, j in candidates[:args.k]])
        g = geometry({'D_H': coeff}, args.k, args.k, target.T @ target)
        next_gap = None
        if len(candidates) > args.k:
            before, after = candidates[args.k-1], candidates[args.k]
            gap = before[0]-after[0]
            threshold = 10*(specs[before[1]]['absolute_residual'][before[2]] +
                            specs[after[1]]['absolute_residual'][after[2]])
            next_gap = dict(gap=float(gap), residual_gap_threshold=float(threshold),
                            status='cluster_boundary_ambiguous' if gap <= threshold else 'resolved')
        target_edge = boundary(targets[label], args.k)
        stable = (target_edge['status'] == 'resolved' and
                  all(np.all(specs[key]['residual_passed'][:args.k+1]) for key in keys) and
                  (next_gap is None or next_gap['status'] == 'resolved'))
        d_h[label] = dict(status='complete' if stable else 'preliminary_boundary_ambiguous', geometry=g,
                          D_H_boundary=next_gap, target_boundary=target_edge,
                          sources=[dict(block=key, index=j+1, value=value) for value,key,j in candidates[:args.k]],
                          note='Positive D_H-top-k from certified ordered self spectra; no SA solve.')
    write(output/'diagonal_baseline.json', d_h)
    if args.diagonal_equal_dimension_control:
        if primary_bases and positive_count(targets['H7']) >= args.k:
            def save_refinement(key, spec):
                count = positive_count(spec)
                np.savez_compressed(output/f'diagonal_refined_{key}.npz',
                    positive_vectors=spec['vectors'][:,:count].astype(np.float32),
                    values=spec['values'], absolute_residual=spec['absolute_residual'],
                    relative_residual=spec['relative_residual'])
            target = targets_full['H7'][:, :args.k]
            dim = sum(b.shape[1] for b in primary_bases.values())
            control = adaptive_diagonal_top(backend, specs, target, dim, args.seed+4000,
                args.solver_tol, args.residual_tol, args.maxiter, args.ncv, progress, save_refinement)
            control['target_boundary'] = boundary(targets['H7'], args.k)
            if control['status'] == 'complete' and control['target_boundary']['status'] != 'resolved':
                control['status'] = 'preliminary_boundary_ambiguous'
            optional_controls['diagonal_equal_dimension'] = control
        else:
            optional_controls['diagonal_equal_dimension'] = {'status':'insufficient_certified_positive_target'}
        write(output/'diagonal_equal_dimension_control.json', optional_controls['diagonal_equal_dimension'])
    if args.equal_dimension_control:
        bases35, dims35 = _prepare_bases(specs, FOUR, 35)
        matching = {'requested_four_per_block':35, 'actual_dimensions':dims35,
                    'status':'not_available'}
        if (all(dims35[k] == 35 for k in FOUR) and
                all(positive_count(specs[k]) >= 20 for k in SEVEN_BLOCKS) and
                positive_count(targets['H7']) >= args.k):
            target = targets_full['H7'][:, :args.k]
            coords = block_coefficients(bases35, target, backend.slices)
            edges = _boundary_summary(specs, FOUR, 35)
            stable = all(x['status'] == 'resolved' for x in edges.values()) and boundary(targets['H7'], args.k)['status'] == 'resolved'
            matching.update(status='complete' if stable else 'preliminary_boundary_ambiguous',
                            geometry=geometry(coords, args.k, 140, target.T@target), self_boundaries=edges,
                            note='Compare against S7(m=20) only when all seven blocks have 20 positive directions; inspect boundary status.')
        write(output/'equal_dimension_control.json', matching)
    del primary_bases
    return dict(spectra_certified={k:bool(v['ordered_leading_space_certified'] and np.all(v['residual_passed'])) for k,v in allspec.items()},
                comparisons=len(summaries),
                preliminary_coverage_rows=sum(x['status'] != 'complete' for x in summaries),
                optional_controls=optional_controls,
                ritz=ritz)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, help='new directory; existing results are never overwritten')
    p.add_argument('--scope', choices=['TB', 'FP'], default='TB')
    p.add_argument('--probe-batch', help='FP torch file with input_ids and labels')
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--expected-tokens', type=int, default=8192,
                   help='expected valid tokens; 0 disables this check for tiny tests')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--solver-tol', type=float, default=1e-6)
    p.add_argument('--residual-tol', type=float, default=1e-4)
    p.add_argument('--operator-tol', type=float, default=1e-4)
    p.add_argument('--maxiter', type=int, default=500)
    p.add_argument('--ncv', type=int)
    p.add_argument('--probes', type=int, default=64)
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    p.add_argument('--random-repeats', type=int, default=100)
    p.add_argument('--seed', type=int, default=23)
    p.add_argument('--precision', choices=['fp32', 'fp64'], default='fp32')
    p.add_argument('--equal-dimension-control', action='store_true',
                   help='also compute four-block top35 per block to match seven-block top20 dimension')
    p.add_argument('--diagonal-equal-dimension-control', action='store_true',
                   help='adaptively refine self spectra to obtain D_7 with the same dimension as S7')
    p.add_argument('--cross-randomizations', type=int, default=0,
                   help='optional synthetic blockwise signed-permutation controls; expensive')
    p.add_argument('--save-hp', action='store_true', help='save full H acting on the positive candidate basis')
    p.add_argument('--skip-frobenius', action='store_true', help='explicitly mark the F-norm experiment skipped')
    args = p.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('seven_measure currently supports one process only; do not use multi-rank torchrun')
    if args.microbatch < 1 or args.k < 1 or args.expected_tokens < 0 or args.cross_randomizations < 0:
        raise ValueError('invalid microbatch, k, expected token count or randomization count')
    if args.equal_dimension_control and args.k != 20:
        raise ValueError('the equal-dimension control requires --k 20')
    if args.scope == 'FP' and not args.probe_batch:
        raise ValueError('FP requires --probe-batch')
    if args.scope == 'TB' and args.probe_batch:
        raise ValueError('--probe-batch applies only to FP')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def progress(phase, **extra):
        event = dict(phase=phase, elapsed_seconds=time.perf_counter()-start, **extra)
        write(out/'status.json', event)
        print(json.dumps(event), flush=True)
    try:
        progress('initializing')
        checkpoint_hash = sha256(args.checkpoint)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if not state.get('pre_update_state'):
            raise ValueError('a PRE-update checkpoint is required')
        config = ModelConfig(**state['config'])
        if config.layers != 1 or state.get('layer') != 0:
            raise ValueError('this experiment requires a single Transformer layer (layer 0)')
        raw_batch = (torch.load(args.probe_batch, map_location='cpu', weights_only=True)
                     if args.scope == 'FP' else state['batch'])
        batch = {key: raw_batch[key].detach().cpu().contiguous() for key in ('input_ids', 'labels')}
        if batch['input_ids'].ndim != 2 or batch['labels'].shape != batch['input_ids'].shape:
            raise ValueError('input_ids and labels must have the same two-dimensional shape')
        count = int((batch['labels'] != -100).sum())
        if count <= 0 or (args.expected_tokens and count != args.expected_tokens):
            raise ValueError(f'valid token count {count} does not match expected {args.expected_tokens}')
        if args.scope == 'TB' and state.get('valid_tokens') != count:
            raise ValueError('checkpoint valid_tokens does not match its stored train batch')
        model = Transformer(config)
        model.load_state_dict(state['model'])
        model.to(device=device, dtype=torch.float64 if args.precision=='fp64' else torch.float32)
        batches = [{key: value[i:i+args.microbatch].to(device) for key, value in batch.items()}
                   for i in range(0, len(batch['labels']), args.microbatch)]
        batches = [x for x in batches if (x['labels'] != -100).any()]
        backend = Decomposition(model, batches, count, 0, blocks=SEVEN_BLOCKS)
        names = {id(param): name for name,param in model.named_parameters()}
        meta = dict(scope=args.scope, checkpoint_sha256=checkpoint_hash,
                    batch_sha256=batch_hash(batch), probe_file_sha256=sha256(args.probe_batch) if args.probe_batch else None,
                    pre_update_state=True, step=state['step'], layer=0,
                    model=state['config'], total_model_parameters=sum(x.numel() for x in model.parameters()),
                    parameter_order=list(SEVEN_BLOCKS), parameters={k:dict(model_name=names[id(v)], shape=list(v.shape), size=v.numel()) for k,v in backend.params.items()},
                    selected_parameter_count=backend.n, valid_tokens=count, precision=args.precision,
                    torch_version=str(torch.__version__), numpy_version=np.__version__,
                    device=str(device), git=git_provenance(), arguments=vars(args),
                    CE_definition='token-weighted mean; no weight decay, clipping or optimizer terms',
                    F_definition='full H/A/R matrices, including every spectral sign; no positive-truncation F',
                    anchor='block_output', training_gradient_precision=state.get('gradient_precision'),
                    data_sha256=state.get('data_hash'), training_arguments=state.get('args'))
        progress('operator_verification')
        checks = verify_backend(backend, args.seed+700, args.operator_tol)
        if args.scope == 'TB':
            gradient = backend.gradient()
            saved = state.get('raw_gradients', {})
            present = [k for k in SEVEN_BLOCKS if k in saved]
            if present:
                diff = np.concatenate([gradient[backend.slices[k]] for k in present])
                ref = np.concatenate([saved[k].detach().cpu().numpy().ravel() for k in present])
                meta['training_gradient_comparison'] = dict(blocks=present,
                    relative_error=float(np.linalg.norm(diff-ref)/(np.linalg.norm(ref)+1e-30)),
                    note='Training precision may be BF16; this is not an equal-precision identity check.')
        write(out/'operator_checks.json', checks)
        write(out/'metadata.json', meta)
        progress('positive_spectra_start')
        summary = run_analysis(backend, args, out, progress)
        flagged_blocks = []
        if args.skip_frobenius:
            write(out/'frobenius.json', {'status':'skipped_by_request'})
        else:
            progress('frobenius_start')
            f = frobenius_statistics(backend, args.probes, args.seed+900,
                                     args.bootstrap_samples, progress)
            f['H_blocks_needing_independent_more_probes'] = [name for name, row in f['blocks'].items()
                if row['H']['norm2_relative_ci_halfwidth'] is not None and
                   row['H']['norm2_relative_ci_halfwidth'] > 0.10]
            f['precision_note'] = ('A CI wider than 10% is flagged; rerun with independent '
                                   '128/256 probes and a new seed rather than silently pooling adaptive looks.')
            flagged_blocks = f['H_blocks_needing_independent_more_probes']
            write(out/'frobenius.json', f)
        meta['elapsed_seconds'] = time.perf_counter()-start
        meta['operator_requests'] = backend.calls
        meta['operator_request_note'] = 'Main backend H/A/R requests; independent aggregation HVP requests are listed in operator_checks.json.'
        meta['checkpoint_unchanged'] = sha256(args.checkpoint) == checkpoint_hash
        if not meta['checkpoint_unchanged']:
            raise RuntimeError('input checkpoint changed while the measurement was running')
        if device.type == 'cuda': meta['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
        write(out/'metadata.json', meta)
        failed = [name for name, passed in summary['spectra_certified'].items() if not passed]
        progress('completed' if not failed else 'completed_with_uncertified_spectra',
                 failed_spectra=failed, frobenius_skipped=args.skip_frobenius,
                 preliminary_coverage_rows=summary['preliminary_coverage_rows'],
                 H_blocks_needing_independent_more_probes=flagged_blocks,
                 optional_control_status={name:value['status'] for name,value in summary['optional_controls'].items()},
                 scientific_followup_required=bool(failed or args.skip_frobenius or flagged_blocks or
                    summary['preliminary_coverage_rows'] or any(value['status']!='complete' for value in summary['optional_controls'].values())),
                 operator_requests=backend.calls)
        if failed:
            raise RuntimeError('spectra did not converge: ' + ', '.join(failed))
    except Exception as exc:
        traceback.print_exc()
        progress('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
