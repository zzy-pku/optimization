"""M1/M2 driver. Rank 0 owns every stochastic probe and Lanczos iteration."""
import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import distributed as dd
from .hessian import Hessian
from .model import Transformer, ModelConfig, BLOCKS
from .solvers import spectrum, cross_svd, overlap, zero_spectrum


def jsonable(x):
    if isinstance(x, dict): return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def write(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False))

def norm_statistics(h, probes, seed):
    if probes < 2: raise ValueError('at least two probes needed for stderr')
    rng = np.random.default_rng(seed)
    samples, traces = {}, {}
    for j in BLOCKS:
        samples.update({i+j: [] for i in BLOCKS if BLOCKS.index(i) <= BLOCKS.index(j)})
        traces[j] = []
        for _ in range(probes):
            v = rng.choice([-1., 1.], h.sizes[j])
            full = np.zeros(h.n, dtype=np.float64 if h.dtype == torch.float64 else np.float32)
            full[h.slices[j]] = v
            out = h(full)
            traces[j].append(float(v@out[h.slices[j]]))
            for i in BLOCKS[:BLOCKS.index(j)+1]:
                samples[i+j].append(float(np.sum(out[h.slices[i]].astype(np.float64)**2)))
    result = {}
    for ij, vals in samples.items():
        i, j = ij
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1)/np.sqrt(probes))
        result[ij] = dict(norm2=mean, norm2_stderr=se, norm=np.sqrt(mean),
            norm_delta_stderr=se/(2*np.sqrt(mean)) if mean else None,
            element_rms=np.sqrt(mean/(h.sizes[i]*h.sizes[j])), probes=probes,
            sqrt_estimator_biased=True)
        if i == j:
            result[ij].update(trace=np.mean(traces[i]), trace_stderr=np.std(traces[i], ddof=1)/np.sqrt(probes))
    for i, j in combinations(BLOCKS, 2):
        denom = np.sqrt(result[i+i]['norm']*result[j+j]['norm'])
        result[i+j]['coupling'] = result[i+j]['norm']/denom if denom > 1e-20 else None
    cross = 2*sum(result[i+j]['norm2'] for i,j in combinations(BLOCKS,2))
    total = cross + sum(result[i+i]['norm2'] for i in BLOCKS)
    return dict(blocks=result, cross_energy=cross/total if total else None)

def projections(h, output, rank, repeats, seed):
    """Row-major Kronecker bases; conclusions apply only inside projected coordinates."""
    rows = []
    first = {}
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        bases, dims = {}, {}
        for k, w in h.params.items():
            a, b = w.shape
            ra, rb = min(rank, a), min(rank, b)
            u = np.linalg.qr(rng.normal(size=(a, ra)))[0]
            v = np.linalg.qr(rng.normal(size=(b, rb)))[0]
            bases[k] = np.einsum('ia,jb->ijab', u, v).reshape(a*b, ra*rb)
            dims[k] = (ra, rb)
            np.savez(output / f'basis_{repeat}_{k}.npz', row=u, column=v)
        for j in BLOCKS:
            applied = []
            for v in bases[j].T:
                full = np.zeros(h.n, dtype=v.dtype)
                full[h.slices[j]] = v
                applied.append(h(full))
            hv = np.column_stack(applied)
            for i in BLOCKS[:BLOCKS.index(j)+1]:
                projected = bases[i].T @ hv[h.slices[i]]
                ai, bi = dims[i]; aj, bj = dims[j]
                rearranged = projected.reshape(ai,bi,aj,bj).transpose(0,2,1,3).reshape(ai*aj,bi*bj)
                sv = np.linalg.svd(rearranged, compute_uv=False)
                energy = float(np.sum(sv**2))
                stats = dict(block=i+j, repeat=repeat, measurement='projected',
                    kron_score=float(sv[0]**2/energy) if energy else None,
                    rank_r_relative_errors=[float(np.sqrt(max(0.,1-np.sum(sv[:r]**2)/energy))) if energy else None for r in range(1,len(sv)+1)])
                rows.append(stats)
                np.savez(output / f'projection_{repeat}_{i+j}.npz', matrix=projected)
        for k in BLOCKS:
            if repeat:
                rows.append(dict(block=k, repeat=repeat, basis_stability=overlap(first[k], bases[k]),
                                 note='independent random projected subspaces; scores are basis dependent'))
            else: first[k] = bases[k]
    return rows

def contributions(h, subset, spec, gradient, update):
    records = []
    for eig, q in zip(spec['values'], spec['vectors'].T):
        chunks, pos = {}, 0
        for i in subset:
            chunks[i] = q[pos:pos+h.sizes[i]]
            pos += h.sizes[i]
        terms = {}
        participation, enrichment, align = {}, {}, {}
        for i in subset:
            qi = chunks[i]
            participation[i] = float(qi@qi)
            enrichment[i] = participation[i]/(h.sizes[i]/len(q))
            align[i] = {}
            for name, direction in [('gradient', gradient), ('update', update)]:
                d = direction[h.slices[i]]
                denom = np.linalg.norm(d)*np.linalg.norm(qi)
                align[i][name] = float(abs(qi@d)/denom) if denom else None
            terms[i+i] = float(qi@h.block(i,i,qi))
        for i,j in combinations(subset,2):
            terms[i+j] = float(2*chunks[i]@h.block(i,j,chunks[j]))
        records.append(dict(eigenvalue=eig, participation=participation, enrichment=enrichment,
            signed_terms=terms, sum_error=float(sum(terms.values())-eig), block_direction_alignment=align))
    return records

def run(args, state, h, gradient, output, norms=None):
    start = time.perf_counter()
    operator = getattr(h, 'operator', 'H')
    dtype = np.float64 if h.dtype == torch.float64 else np.float32
    norms = norms if norms is not None else norm_statistics(h, args.probes, args.seed)
    write(output/'norms_and_coupling.json', norms)
    directions = {}
    for name, direction in [('gradient', gradient), ('update', np.concatenate([state['updates'][k].flatten().numpy() for k in BLOCKS]))]:
        unit = {}
        for i in BLOCKS:
            d = direction[h.slices[i]]
            unit[i] = d/np.linalg.norm(d) if np.linalg.norm(d) else None
        directions[name] = {}
        for j in BLOCKS:
            if unit[j] is None: continue
            full = np.zeros(h.n, dtype=direction.dtype)
            full[h.slices[j]] = unit[j]
            out = h(full)
            for i in BLOCKS[:BLOCKS.index(j)+1]:
                directions[name][i+j] = float(unit[i]@out[h.slices[i]]) if unit[i] is not None else None
    write(output/'directional_curvature.json', directions)
    if args.norms_only:
        write(output/'coverage.json', [dict(metric=operator+'_norms', status='pilot', scope=args.scope),
            dict(metric=operator+'_spectra_and_structure',status='pending')])
        return dict(seconds=time.perf_counter()-start,matvecs=h.calls)
    update = np.concatenate([state['updates'][k].flatten().numpy() for k in BLOCKS])
    coverage, spectra, summary = [dict(metric=operator+'_norms',status='pilot' if args.probes < 16 else 'complete')], {}, {}
    subsets = [tuple(s) for s in args.sets.split(',')] if args.sets else [s for n in range(1,5) for s in combinations(BLOCKS,n)]
    def save_spec(name, spec):
        arrays = {k: v for k,v in spec.items() if isinstance(v, np.ndarray)}
        np.savez(output/'vectors'/f'{name}.npz', **arrays)
        write(output/f'{name}.json', {k:v for k,v in spec.items() if k not in ('vectors','left','right')})
        coverage.append(dict(metric=name, status='complete' if spec['converged'] else 'failed_convergence'))
    for subset in subsets:
        if not subset or any(i not in BLOCKS for i in subset) or len(set(subset)) != len(subset) or tuple(i for i in BLOCKS if i in subset) != subset:
            raise ValueError('invalid parameter set')
        key = ''.join(subset)
        n = sum(h.sizes[i] for i in subset)
        summary[key] = {}
        for end in ('LA','SA'):
            spec = zero_spectrum(n,args.k,dtype) if getattr(h,'is_zero',lambda *_:False)(subset) else spectrum(lambda v: h.joint(subset,v), n, args.k, end,args.tol,args.maxiter,args.seed,dtype)
            spectra[key,end] = spec
            save_spec(f'{operator}_{key}_{end}',spec)
            summary[key][end] = contributions(h,subset,spec,gradient,update)
            if len(subset) > 1:
                diag = zero_spectrum(n,args.k,dtype) if getattr(h,'is_zero',lambda *_:False)(subset,True) else spectrum(lambda v: h.joint(subset,v,self_only=True),n,args.k,end,args.tol,args.maxiter,args.seed,dtype)
                save_spec(f'self_only_{key}_{end}',diag)
                summary[key][end+'_self_only_overlap'] = dict(overlap=None,angles=None,reason='structurally zero, fully degenerate spectrum') if spec.get('structural_zero') or diag.get('structural_zero') else overlap(spec['vectors'],diag['vectors'])
                summary[key][end+'_self_only_participation'] = participation(h, subset, diag['vectors'])
    cross_rows = []
    for i,j in combinations(BLOCKS,2):
        s = cross_svd(lambda v:h.block(i,j,v),lambda v:h.block(j,i,v),h.sizes[i],h.sizes[j],args.k,args.tol,args.maxiter,args.seed,dtype)
        save_spec(f'{operator}_{i+j}_SVD',s)
        row = dict(block=i+j, captured_frobenius_energy=float(np.sum(s['values']**2)/norms['blocks'][i+j]['norm2']) if norms['blocks'][i+j]['norm2'] else None,
                   energy_denominator='stochastic Frobenius estimate; ratio may exceed 1')
        for side,k in [('left',i),('right',j)]:
            for end in ('LA','SA'):
                if (k,end) in spectra:
                    row[side+'_'+end] = overlap(s[side],spectra[k,end]['vectors'])
        cross_rows.append(row)
    if not args.no_ablation:
        for i,j in combinations(BLOCKS,2):
            for end in ('LA','SA'):
                s = spectrum(lambda v:h.joint(BLOCKS,v,drop=(i,j)),h.n,args.k,end,args.tol,args.maxiter,args.seed,dtype)
                save_spec(f'drop_{i+j}_OUGD_{end}',s)
                if ('OUGD',end) in spectra:
                    summary[f'drop_{i+j}_{end}_overlap'] = overlap(s['vectors'],spectra['OUGD',end]['vectors'])
                summary[f'drop_{i+j}_{end}_participation'] = participation(h, BLOCKS, s['vectors'])
    interlacing = []
    for subset in subsets:
        key = ''.join(subset)
        if len(subset) < 2 or any((i,'LA') not in spectra for i in subset): continue
        if not all(spectra[key,e]['converged'] and all(spectra[i,e]['converged'] for i in subset) for e in ('LA','SA')):
            interlacing.append(dict(parameter_set=key,status='unverified_unconverged'))
            continue
        la = spectra[key,'LA']['values'][0]; sa = spectra[key,'SA']['values'][0]
        interlacing.append(dict(parameter_set=key,
            LA_margin=float(la-max(spectra[i,'LA']['values'][0] for i in subset)),
            SA_margin=float(min(spectra[i,'SA']['values'][0] for i in subset)-sa)))
    if args.no_ablation:
        coverage.append(dict(metric='pair_cross_ablation',status='skipped_budget'))
    if not args.projection_rank:
        coverage.append(dict(metric='projected_structure',status='skipped_budget'))
    if args.projection_rank:
        write(output/'projected_structure.json',projections(h,output,args.projection_rank,args.basis_repeats,args.seed))
        coverage.append(dict(metric='projected_structure',status='complete',measurement='projected'))
    for n in range(1,5):
        for s in combinations(BLOCKS,n):
            if s not in subsets: coverage.append(dict(metric=operator+'_'+''.join(s),status='pending'))
    write(output/'contributions.json',summary)
    write(output/'cross_subspaces.json',cross_rows)
    write(output/'interlacing.json',interlacing)
    write(output/'coverage.json',coverage)
    return dict(seconds=time.perf_counter()-start,matvecs=h.calls,
                peak_memory_bytes=torch.cuda.max_memory_allocated() if h.device.type == 'cuda' else None)

def participation(h, subset, vectors):
    result, pos = {}, 0
    for i in subset:
        result[i] = np.sum(vectors[pos:pos+h.sizes[i]]**2, axis=0)
        pos += h.sizes[i]
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--module', choices=['M1','M2','M1+M2'], default='M1+M2')
    p.add_argument('--check-probes',type=int,default=2,help='independent decomposition checks for M2')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scope', choices=['TB','ST','FP'],default='TB')
    p.add_argument('--probe-batch', help='FP-only torch file with input_ids and labels')
    p.add_argument('--token', default='0:0',help='ST sequence:prediction-position in saved global batch')
    p.add_argument('--microbatch',type=int,default=1)
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--probes',type=int,default=16)
    p.add_argument('--tol',type=float,default=1e-4)
    p.add_argument('--maxiter',type=int,default=120)
    p.add_argument('--seed',type=int,default=23)
    p.add_argument('--sets',help='comma-separated sets; omitted means all 15')
    p.add_argument('--norms-only', action='store_true',help='T1 noisy norm proxy, no spectral/structure claims')
    p.add_argument('--no-ablation', action='store_true')
    p.add_argument('--projection-rank',type=int,default=4)
    p.add_argument('--basis-repeats',type=int,default=2)
    p.add_argument('--precision',choices=['fp32','fp64'],default='fp32')
    args = p.parse_args()
    device,rank,world = dd.init()
    torch.set_num_threads(1)
    if args.microbatch < 1: raise ValueError('microbatch must be positive')
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if not state.get('pre_update_state'): raise ValueError('requires PRE-update measurement checkpoint')
    model = Transformer(ModelConfig(**state['config']))
    model.load_state_dict(state['model'])
    model.to(device=device,dtype=torch.float64 if args.precision=='fp64' else torch.float32)
    batch = state['batch']
    if args.scope == 'FP':
        if not args.probe_batch: raise ValueError('FP requires --probe-batch')
        batch = torch.load(args.probe_batch,map_location='cpu',weights_only=True)
    elif args.scope == 'ST':
        seq,pos = map(int,args.token.split(':'))
        if not 0 <= seq < len(batch['labels']) or not 0 <= pos < batch['labels'].shape[1]:
            raise ValueError('invalid ST token')
        batch = {k:v[seq:seq+1].clone() for k,v in batch.items()}
        target = batch['labels'][0,pos].item()
        if target == -100: raise ValueError('ST target is masked')
        batch['labels'].fill_(-100)
        batch['labels'][0,pos] = target
    count = int((batch['labels'] != -100).sum())
    indices = list(range(rank,len(batch['input_ids']),world))
    batches = [{k:v[indices[i:i+args.microbatch]].to(device) for k,v in batch.items()} for i in range(0,len(indices),args.microbatch)]
    # All-ignored local slices contribute exactly zero and need no graph.
    batches = [b for b in batches if (b['labels'] != -100).any()]
    if args.module in ('M2','M1+M2'):
        from .components import Decomposition
        h = Decomposition(model,batches,count,state['layer'])
    else:
        h = Hessian(model,batches,count,state['layer'])
    gradient = h.gradient()
    if rank:
        h.serve()
    else:
        output = Path(args.output)
        output.mkdir(parents=True,exist_ok=True)
        if any(output.iterdir()):
            h.stop()
            raise ValueError('output must be empty to prevent mixing operators/checkpoints')
        (output/'vectors').mkdir()
        spec = dict(checkpoint_sha256=None, step=state['step'],pre_update_state=True,layer=state['layer'],
                    run_id=state['args']['output'],
                    parameter_version='pre_step_'+str(state['step']),
                    scope=args.scope,token=args.token if args.scope=='ST' else None,
                    operator='H',anchor='parameter',measurement='fixed_probe_proxy' if args.scope=='FP' else 'exact/full_batch',
                    valid_tokens=count,model=state['config'],precision=args.precision,backend='explicit_math_attention',
                    parameter_order=list(BLOCKS),parameters={k:dict(shape=list(v.shape),size=v.numel()) for k,v in h.params.items()},
                    torch=torch.__version__,world_size=world,args=vars(args),stage='user_selected')
        from .data import sha256
        spec['checkpoint_sha256'] = sha256(args.checkpoint)
        spec['batch_sha256'] = hashlib.sha256(b''.join(batch[k].numpy().tobytes() for k in ('input_ids','labels'))).hexdigest()
        if args.scope == 'TB':
            raw = np.concatenate([state['raw_gradients'][k].flatten().numpy() for k in BLOCKS])
            spec['gradient_replay_relative_error'] = float(np.linalg.norm(gradient-raw)/(np.linalg.norm(raw)+1e-30))
            spec['gradient_replay_tolerance'] = 0.05 if state['gradient_precision']=='bf16' else 1e-5
            spec['gradient_replay_passed'] = spec['gradient_replay_relative_error'] <= spec['gradient_replay_tolerance']
            spec['training_gradient_precision'] = state['gradient_precision']
            spec['gradient_comparison_note'] = 'FP32/64 curvature versus saved raw training gradient; BF16 differences are reported'
        try:
            if args.module in ('M2','M1+M2'):
                from .m2 import run_m2
                spec['anchor'] = 'block_output'
                spec['module'] = args.module
                spec['modules'] = ['M1','M2'] if args.module=='M1+M2' else ['M2']
                spec['shared_H'] = True
                spec['anchor_path'] = f'blocks.{state["layer"]}.output_after_ffn_residual'
                stats = run_m2(args,state,h,gradient,output,spec)
            else:
                stats = run(args,state,h,gradient,output)
            spec.update(stats)
            spec['gpu_seconds'] = stats['seconds']*world if device.type == 'cuda' else 0.
        finally:
            write(output/'operator_meta.json',spec)
            h.stop()
    dd.close()

if __name__=='__main__':
    main()
