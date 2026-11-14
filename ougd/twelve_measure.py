"""Matrix-free multi-layer positive spectrum, coverage and unnormalized block F norms."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import sha256
from .hessian import Hessian
from .model import ModelConfig, SEVEN_BLOCKS, Transformer
from .positive_subspaces import positive_spectrum, positive_count, boundary
from .seven_measure import batch_hash, jsonable, git_provenance, write


def groups(model):
    return {f'L{layer+1:02d}.{name}': p for layer in range(model.config.layers)
            for name, p in model.selected_seven(layer).items()}


def scopes(keys, layer, pair):
    per_layer = {f'L{i:02d}': [k for k in keys if k.startswith(f'L{i:02d}.')]
                 for i in sorted({int(k[1:3]) for k in keys})}
    return {'global': list(keys), **per_layer,
            f'pair_L{pair[0]:02d}_L{pair[1]:02d}': per_layer[f'L{pair[0]:02d}'] + per_layer[f'L{pair[1]:02d}'],
            **{key: [key] for key in keys}}


def offsets(keys, params):
    result, pos = {}, 0
    for key in keys:
        n = params[key].numel()
        result[key] = slice(pos, pos + n)
        pos += n
    return result


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def block_metrics(target, target_keys, candidates, params, k):
    """Candidates are orthonormal bases for disjoint subsets of target coordinates."""
    sl = offsets(target_keys, params)
    rows = []
    for name, keys, basis in candidates:
        local = np.concatenate([target[sl[key], :k] for key in keys], axis=0)
        energy = np.sum(local * local, axis=0, dtype=np.float64)
        captured = np.sum((basis.T @ local) ** 2, axis=0, dtype=np.float64)
        e, c = float(np.mean(energy)), float(np.mean(captured))
        rows.append(dict(block=name, participation=e, capture_contribution=c,
                         conditional_capture=c/e if e > 1e-12 else None,
                         participation_per_mode=energy.tolist(), capture_per_mode=captured.tolist(),
                         basis_dimension=basis.shape[1]))
    return dict(blocks=rows, containment=sum(x['capture_contribution'] for x in rows),
                participation_sum=sum(x['participation'] for x in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--probe-batch', required=True, help='fixed held-out batch .pt with input_ids and labels')
    p.add_argument('--output', required=True)
    p.add_argument('--phase', choices=['spectra', 'frobenius', 'coverage', 'all'], default='all')
    p.add_argument('--spectrum-scopes', choices=['all','global','local'], default='all')
    p.add_argument('--single-layer', type=int, default=6, help='1-based')
    p.add_argument('--pair', type=int, nargs=2, default=[6, 7], help='two 1-based layers')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--probes', type=int, default=32)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--solver-tol', type=float, default=1e-6)
    p.add_argument('--residual-tol', type=float, default=1e-4)
    p.add_argument('--maxiter', type=int, default=1000)
    p.add_argument('--seed', type=int, default=23)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.k < 1 or args.probes < 2 or args.microbatch < 1: p.error('invalid k, probes or microbatch')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Transformer(ModelConfig(**state['config']))
    if not (1 <= args.single_layer <= model.config.layers and len(set(args.pair)) == 2
            and all(1 <= x <= model.config.layers for x in args.pair)):
        p.error('invalid layer selection')
    model.load_state_dict(state['model']); model.to(args.device).eval()
    raw = torch.load(args.probe_batch, map_location='cpu', weights_only=True)
    batch = {key: raw[key].detach().cpu().contiguous() for key in ('input_ids', 'labels')}
    if batch['input_ids'].ndim != 2 or batch['labels'].shape != batch['input_ids'].shape:
        raise ValueError('invalid batch shape')
    count = int((batch['labels'] != -100).sum())
    if not count: raise ValueError('empty probe batch')
    batches = [{key: value[i:i+args.microbatch].to(args.device) for key, value in batch.items()}
               for i in range(0, len(batch['labels']), args.microbatch)]
    batches = [b for b in batches if (b['labels'] != -100).any()]
    params = groups(model); keys = list(params); subsets = scopes(keys, args.single_layer, args.pair)
    fingerprint = dict(checkpoint_sha256=sha256(args.checkpoint), batch_sha256=batch_hash(batch),
                       k=args.k, probes=args.probes, seed=args.seed, microbatch=args.microbatch,
                       single_layer=args.single_layer, pair=args.pair, solver_tol=args.solver_tol,
                       residual_tol=args.residual_tol, maxiter=args.maxiter, device=args.device)
    meta_path = out/'metadata.json'
    if meta_path.exists() and json.loads(meta_path.read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
        raise ValueError('output directory belongs to different inputs or settings')
    write(meta_path, dict(fingerprint=fingerprint, model=state['config'], checkpoint_step=state['step'],
        checkpoint_pre_update=bool(state.get('pre_update_state')), valid_tokens=count,
        parameters={k: {'shape': list(v.shape), 'numel': v.numel()} for k,v in params.items()},
        ordering=keys, git=git_provenance(), torch_version=str(torch.__version__),
        definition='Hessian of mean token CE w.r.t. the 7 matrix weights in each selected layer; all other parameters fixed; no optimizer, decay or truncation in F norms'))
    def backend(selected):
        return Hessian(model, batches, count, 0, parameters={key:params[key] for key in selected})
    def log(phase, **kw):
        event = dict(phase=phase, time=time.time(), **kw); write(out/'status.json', event); print(json.dumps(event), flush=True)
    checks_path = out/'operator_checks.json'
    if checks_path.exists() and not json.loads(checks_path.read_text(encoding='utf-8'))['passed']:
        raise RuntimeError('previous HVP operator checks failed; use a new output directory to retry')
    if not checks_path.exists():
        log('operator_checks')
        op = backend(keys); rng = np.random.default_rng(args.seed + 901)
        u = rng.standard_normal(op.n).astype(np.float32); u /= np.linalg.norm(u)
        v = rng.standard_normal(op.n).astype(np.float32); v /= np.linalg.norm(v)
        hu, hv, huv = op(u), op(v), op(u+v)
        singles = [{key: value[i:i+1] for key, value in b.items()}
                   for b in batches for i in range(len(b['labels']))
                   if (b['labels'][i:i+1] != -100).any()]
        alt = Hessian(model, singles, count, 0, parameters=params)(v)
        symmetry = float(abs(np.dot(u, hv)-np.dot(v, hu))/(np.linalg.norm(hu)+np.linalg.norm(hv)+1e-30))
        linearity = float(np.linalg.norm(huv-hu-hv)/(np.linalg.norm(huv)+np.linalg.norm(hu)+np.linalg.norm(hv)+1e-30))
        aggregation = float(np.linalg.norm(alt-hv)/(np.linalg.norm(alt)+np.linalg.norm(hv)+1e-30))
        write(checks_path, dict(symmetry_error=symmetry, linearity_error=linearity,
                                token_aggregation_error=aggregation, tolerance=1e-4,
                                passed=max(symmetry, linearity, aggregation) <= 1e-4))
        if max(symmetry, linearity, aggregation) > 1e-4: raise RuntimeError('HVP operator checks failed')
    if args.phase in ('all', 'spectra'):
        folder = out/'spectra'; folder.mkdir(exist_ok=True)
        for index, (name, selected) in enumerate(subsets.items()):
            if args.spectrum_scopes == 'global' and name != 'global': continue
            if args.spectrum_scopes == 'local' and name == 'global': continue
            path = folder/f'{name}.npz'; metadata = folder/f'{name}.json'
            if path.exists() and metadata.exists(): continue
            op = backend(selected); requested = args.k + 1
            log('spectrum', scope=name, index=index+1, total=len(subsets), dimension=op.n)
            if op.n * max(2*requested+1, 40) > 2**31-1:
                from .large_spectrum import large_positive_spectrum
                solver = large_positive_spectrum
            else:
                solver = positive_spectrum
            spec = solver(op, op.n, requested, args.seed, args.solver_tol,
                          args.residual_tol, args.maxiter,
                          progress=lambda calls: log('spectrum_progress', scope=name, calls=calls))
            save_npz(path, values=spec['values'], vectors=spec['vectors'].astype(np.float32),
                     absolute_residual=spec['absolute_residual'], relative_residual=spec['relative_residual'])
            info = {k:v for k,v in spec.items() if k != 'vectors'}
            info.update(positive_count=positive_count(spec), boundary=boundary(spec, args.k), keys=selected)
            write(metadata, info)
    if args.phase in ('all', 'frobenius'):
        folder = out/'frobenius'; folder.mkdir(exist_ok=True)
        op = backend(keys); rng = np.random.default_rng(args.seed + 10000)
        for j, key in enumerate(keys):
            path = folder/f'{key}.npz'
            # Recreate each column's independent random stream when resuming.
            rng = np.random.default_rng(args.seed + 10000 + j)
            if path.exists(): continue
            log('frobenius', column=key, index=j+1, total=len(keys))
            samples = np.empty((args.probes, len(keys)))
            for r in range(args.probes):
                v = np.zeros(op.n, dtype=np.float32)
                v[op.slices[key]] = rng.choice(np.array([-1., 1.], dtype=np.float32), op.sizes[key])
                hv = op(v)
                samples[r] = [np.dot(hv[op.slices[i]].astype(np.float64), hv[op.slices[i]].astype(np.float64)) for i in keys]
            save_npz(path, samples=samples)
        directional = np.column_stack([np.load(folder/f'{k}.npz')['samples'].mean(axis=0) for k in keys])
        variances = np.column_stack([np.load(folder/f'{k}.npz')['samples'].var(axis=0, ddof=1)/args.probes for k in keys])
        squared = (directional + directional.T)/2
        se = np.sqrt(variances + variances.T)/2
        np.fill_diagonal(se, np.sqrt(np.diag(variances)))
        write(out/'frobenius.json', dict(keys=keys, norm=np.sqrt(squared), norm_squared=squared,
              norm_squared_standard_error=se, directional_norm_squared=directional,
              probes_per_column=args.probes, definition='absolute unnormalized ||H_ij||_F; square root of symmetrized unbiased squared-norm estimate',
              uncertainty='standard error is for squared norms; off-diagonal uses independent column streams; no adaptive confidence claim'))
    if args.phase in ('all', 'coverage'):
        folder = out/'spectra'
        def load(name):
            info = json.loads((folder/f'{name}.json').read_text(encoding='utf-8'))
            data = np.load(folder/f'{name}.npz')
            if info['positive_count'] < args.k:
                raise ValueError(f'{name} lacks {args.k} certified positive leading directions')
            return data['vectors'][:, :args.k].astype(np.float64), info
        results = {}
        for target_name in [f'L{args.single_layer:02d}', f'pair_L{args.pair[0]:02d}_L{args.pair[1]:02d}', 'global']:
            log('coverage', scope=target_name)
            target, ti = load(target_name); selected = subsets[target_name]
            block_candidates = [(key, [key], load(key)[0]) for key in selected]
            item = {'matrices': block_metrics(target, selected, block_candidates, params, args.k),
                    'target_boundary': ti['boundary'], 'candidate_boundaries': {key: load(key)[1]['boundary'] for key in selected}}
            if target_name == 'global':
                layer_names = [f'L{i:02d}' for i in range(1, model.config.layers+1)]
                layer_candidates = [(name, subsets[name], load(name)[0]) for name in layer_names]
                item['layers'] = block_metrics(target, selected, layer_candidates, params, args.k)
                item['layer_candidate_boundaries'] = {name: load(name)[1]['boundary'] for name in layer_names}
            results[target_name] = item
        write(out/'coverage.json', dict(k=args.k, results=results,
              definition='participation=mean ||q_i||²; capture contribution=mean ||B_i^T q_i||²; conditional capture=contribution/participation; containment=sum contributions; B_i is positive self top-k'))
    log('completed', requested_phase=args.phase)


if __name__ == '__main__':
    main()
