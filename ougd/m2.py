"""M2: paired strength statistics, independent spectra, fixed-H attribution, leakage."""
import copy
from itertools import combinations
import json
import time
import numpy as np
import torch
from .model import BLOCKS
from .measure import run, write
from .solvers import overlap


def standard_error(values):
    return float(np.std(values,ddof=1)/np.sqrt(len(values)))


def shared_statistics(backend, probes, seed):
    if probes < 2: raise ValueError('at least two shared probes required')
    ops = {k:backend.view(k) for k in ('H','A','R')}
    samples = {i+j:{k:[] for k in ('H','A','R','AR')} for j in BLOCKS for i in BLOCKS[:BLOCKS.index(j)+1]}
    traces = {k:{i:[] for i in BLOCKS} for k in ops}
    rng = np.random.default_rng(seed)
    for j in BLOCKS:
        for _ in range(probes):
            v = np.zeros(backend.n)
            v[backend.slices[j]] = rng.choice([-1.,1.],backend.sizes[j])
            hv, av = ops['H'](v), ops['A'](v)
            outputs = {'H':hv,'A':av,'R':hv-av}
            if j in backend.model.anchor_affine_blocks:
                outputs['R'][backend.slices[j]]=0.
            for k,out in outputs.items():
                traces[k][j].append(float(v[backend.slices[j]]@out[backend.slices[j]]))
            for i in BLOCKS[:BLOCKS.index(j)+1]:
                sub = {k:out[backend.slices[i]].astype(np.float64) for k,out in outputs.items()}
                for k,out in sub.items(): samples[i+j][k].append(float(out@out))
                samples[i+j]['AR'].append(float(sub['A']@sub['R']))
    norms = {k:dict(blocks={}) for k in ops}
    paired = {}
    for block,values in samples.items():
        i,j = block
        means = {k:float(np.mean(v)) for k,v in values.items()}
        for k in ops:
            mean = means[k]; se = standard_error(values[k])
            norm = np.sqrt(mean)
            norms[k]['blocks'][block] = dict(norm2=mean,norm2_stderr=se,norm=norm,
                norm_delta_stderr=se/(2*norm) if norm else None,
                element_rms=norm/np.sqrt(backend.sizes[i]*backend.sizes[j]),
                probes=probes,sqrt_estimator_biased=True)
            if i==j:
                norms[k]['blocks'][block].update(trace=float(np.mean(traces[k][i])),trace_stderr=standard_error(traces[k][i]))
        denominator = np.sqrt(means['A']*means['R'])
        paired[block] = dict(inner_AR=means['AR'],inner_AR_stderr=standard_error(values['AR']),
            cosine_AR=means['AR']/denominator if denominator>1e-30 else None,
            ratio_A_H=np.sqrt(means['A']/means['H']) if means['H']>1e-30 else None,
            ratio_R_H=np.sqrt(means['R']/means['H']) if means['H']>1e-30 else None,
            frobenius_identity_error=means['H']-means['A']-means['R']-2*means['AR'],
            identity_note='paired H/A probes; R=H-A; independent R_direct check stored separately')
    for k in ops:
        for i,j in combinations(BLOCKS,2):
            d=np.sqrt(norms[k]['blocks'][i+i]['norm']*norms[k]['blocks'][j+j]['norm'])
            norms[k]['blocks'][i+j]['coupling']=norms[k]['blocks'][i+j]['norm']/d if d>1e-20 else None
        cross=2*sum(norms[k]['blocks'][i+j]['norm2'] for i,j in combinations(BLOCKS,2))
        total=cross+sum(norms[k]['blocks'][i+i]['norm2'] for i in BLOCKS)
        norms[k]['cross_energy']=cross/total if total else None
    return norms,paired


def verify_decomposition(backend, probes, seed, tol):
    if probes < 1: raise ValueError('at least one independent check probe is required')
    rng=np.random.default_rng(seed)
    ops={k:backend.view(k) for k in ('H','A','R','R_direct')}
    rows=[]
    for probe in range(probes):
        v=rng.normal(size=backend.n); v/=np.linalg.norm(v)
        w=rng.normal(size=backend.n); w/=np.linalg.norm(w)
        out={k:op(v) for k,op in ops.items()}
        error=np.linalg.norm(out['H']-out['A']-out['R_direct'])
        scale=np.linalg.norm(out['H'])+np.linalg.norm(out['A'])+np.linalg.norm(out['R_direct'])
        row=dict(probe=probe,H_equals_A_plus_direct_R_relative_error=error/(scale+1e-30),
                 R_vs_direct_R_relative_error=np.linalg.norm(out['R']-out['R_direct'])/(np.linalg.norm(out['R'])+np.linalg.norm(out['R_direct'])+1e-30),
                 per_block={})
        for i in BLOCKS:
            sl=backend.slices[i]
            row['per_block'][i]=float(np.linalg.norm(out['H'][sl]-out['A'][sl]-out['R_direct'][sl])/(scale+1e-30))
        for k in ('H','A','R'):
            applied_w=ops[k](w)
            left=float(w@out[k]); right=float(v@applied_w)
            row[k+'_symmetry_relative_error']=abs(left-right)/(np.linalg.norm(out[k])+np.linalg.norm(applied_w)+1e-30)
        row['passed']=all(float(value)<=tol for key,value in row.items() if key.endswith('_error'))
        rows.append(row)
    return dict(tolerance=tol,probes=rows,passed=all(x['passed'] for x in rows))


def load_spec(root, operator, key, end):
    name=f'{operator}_{key}_{end}'
    meta_path=root/operator/f'{name}.json'
    if not meta_path.exists(): return None
    meta=json.loads(meta_path.read_text())
    with np.load(root/operator/'vectors'/f'{name}.npz') as arrays:
        return {**meta,**{k:arrays[k] for k in arrays.files}}


def signed_terms(op, subset, q):
    chunks={}; pos=0
    for i in subset:
        chunks[i]=q[pos:pos+op.sizes[i]]; pos+=op.sizes[i]
    terms={}
    for j in subset:
        v=np.zeros(op.n,dtype=q.dtype); v[op.slices[j]]=chunks[j]
        out=op(v)
        for i in subset[:subset.index(j)+1]:
            terms[i+j]=float(chunks[i]@out[op.slices[i]])*(1 if i==j else 2)
    return terms


def leakage(apply, basis, source=None):
    source=basis if source is None else source
    if not basis.shape[1] or not source.shape[1]: return dict(absolute=None,relative=None)
    y=np.column_stack([apply(v) for v in source.T])
    q=np.linalg.qr(basis)[0]
    outside=y-q@(q.T@y)
    absolute=float(np.linalg.norm(outside))
    denom=float(np.linalg.norm(y))
    return dict(absolute=absolute,relative=absolute/denom if denom>1e-30 else None)


def compare(root, backend, subsets):
    ops={k:backend.view(k) for k in ('H','A','R')}
    attribution=[]; spaces=[]; cross_attribution=[]
    for subset in subsets:
        key=''.join(subset)
        for end in ('LA','SA'):
            specs={k:load_spec(root,k,key,end) for k in ops}
            if any(s is None for s in specs.values()): continue
            confirmed=all(s['converged'] for s in specs.values())
            row=dict(parameter_set=key,eig_end=end,source_spectra_converged=confirmed)
            for a,b in combinations(ops,2):
                row[a+'_'+b]=dict(overlap=None,angles=None,reason='structurally zero spectrum; fully degenerate') if specs[a].get('structural_zero') or specs[b].get('structural_zero') else overlap(specs[a]['vectors'],specs[b]['vectors'])
            for base,other in [('A','R'),('R','A')]:
                row[other+'_to_'+base]=dict(absolute=None,relative=None,reason='base spectrum fully degenerate') if specs[base].get('structural_zero') else leakage(lambda v:ops[other].joint(subset,v),specs[base]['vectors'])
                row[other+'_to_'+base]['source_spectral_gaps']=specs[base]['gaps']
            spaces.append(row)
            h=specs['H']
            for index,(eig,q) in enumerate(zip(h['values'],h['vectors'].T)):
                terms={k:signed_terms(ops[k],subset,q) for k in ('A','R')}
                rayleigh={k:float(sum(v.values())) for k,v in terms.items()}
                scale=sum(abs(v) for v in rayleigh.values())+abs(eig)
                attribution.append(dict(parameter_set=key,eig_end=end,index=index,
                    source_operator='H',source_vector=f'H/vectors/H_{key}_{end}.npz',
                    source_converged=h['converged'],eigenvalue=eig,contributions=rayleigh,
                    signed_terms=terms,sum_error=float(sum(rayleigh.values())-eig),
                    fractions={k:v/eig for k,v in rayleigh.items()} if abs(eig)>1e-8*max(scale,1e-30) else None))
    for i,j in combinations(BLOCKS,2):
        specs={k:load_spec(root,k,i+j,'SVD') for k in ops}
        if any(s is None for s in specs.values()): continue
        h=specs['H']
        for index,(sigma,u,v) in enumerate(zip(h['values'],h['left'].T,h['right'].T)):
            values={k:float(u@ops[k].block(i,j,v)) for k in ('A','R')}
            cross_attribution.append(dict(block=i+j,index=index,source_operator='H',
                source_converged=h['converged'],singular_value=sigma,contributions=values,
                sum_error=float(sum(values.values())-sigma)))
        row=dict(block=i+j,source_spectra_converged=all(s['converged'] for s in specs.values()))
        for side in ('left','right'):
            for a,b in combinations(ops,2): row[side+'_'+a+'_'+b]=overlap(specs[a][side],specs[b][side])
        for base,other in [('A','R'),('R','A')]:
            row[other+'_to_'+base+'_left']=leakage(lambda v:ops[other].block(i,j,v),specs[base]['left'],specs[base]['right'])
            row[other+'_to_'+base+'_right']=leakage(lambda u:ops[other].block(j,i,u),specs[base]['right'],specs[base]['left'])
        spaces.append(row)
    return attribution,cross_attribution,spaces


def run_m2(args,state,backend,gradient,output,metadata):
    start=time.perf_counter()
    check_tol=1e-9 if args.precision=='fp64' else 1e-4
    checks=verify_decomposition(backend,args.check_probes,args.seed+10000,check_tol)
    write(output/'decomposition_checks.json',checks)
    if not checks['passed']:
        write(output/'coverage.json',[dict(metric='M2_decomposition',status='failed_numerical_check')])
        raise RuntimeError('Independent decomposition check failed; see decomposition_checks.json')
    norms,paired=shared_statistics(backend,args.probes,args.seed)
    write(output/'component_alignment.json',paired)
    coverage=[dict(metric='M2_decomposition',status='complete')]
    for name in ('H','A','R'):
        childargs=component_options(args,name)
        folder=output/name; folder.mkdir(); (folder/'vectors').mkdir()
        view=backend.view(name)
        component_meta={**metadata,'operator':name,'anchor':'block_output',
                        'modules':metadata.get('modules',['M2']) if name=='H' else ['M2'],
                        'args':vars(childargs),
                        'definition':{'H':'original CE Hessian','A':'J^T B J; possibly indefinite',
                                      'R':'H-A; checked independently against Hessian(<stopgrad(q),z>)'}[name]}
        try:
            stats=run(childargs,state,view,gradient,folder,norms[name])
            component_meta.update(stats)
        finally:
            write(folder/'operator_meta.json',component_meta)
        coverage.extend({**row,'operator':name,'modules':component_meta['modules']} for row in json.loads((folder/'coverage.json').read_text()))
    if not args.norms_only:
        subsets=[tuple(s) for s in args.sets.split(',')] if args.sets else [s for n in range(1,5) for s in combinations(BLOCKS,n)]
        attr,cross,spaces=compare(output,backend,subsets)
        write(output/'H_direction_contributions.json',attr)
        write(output/'H_cross_contributions.json',cross)
        write(output/'component_subspaces.json',spaces)
        for row in attr:
            passed=abs(row['sum_error']) <= args.tol*(abs(row['eigenvalue'])+sum(abs(v) for v in row['contributions'].values())+1e-30)
            coverage.append(dict(metric='M2_H_direction_attribution',parameter_set=row['parameter_set'],
                eig_end=row['eig_end'],index=row['index'],status='complete' if row['source_converged'] and passed else 'failed_convergence'))
        for row in cross:
            passed=abs(row['sum_error']) <= args.tol*(abs(row['singular_value'])+sum(abs(v) for v in row['contributions'].values())+1e-30)
            coverage.append(dict(metric='M2_H_cross_attribution',block=row['block'],index=row['index'],
                                 status='complete' if row['source_converged'] and passed else 'failed_convergence'))
        for row in spaces:
            coverage.append(dict(metric='M2_subspaces_and_leakage',parameter_set=row.get('parameter_set'),block=row.get('block'),
                eig_end=row.get('eig_end'),status='complete' if row['source_spectra_converged'] else 'failed_convergence'))
    else:
        coverage.append(dict(metric='M2_attribution_and_subspaces',status='pending'))
    for row in coverage:
        row.setdefault('modules',['M2'])
    write(output/'coverage.json',coverage)
    return dict(seconds=time.perf_counter()-start,matvecs=backend.calls,
                peak_memory_bytes=torch.cuda.max_memory_allocated() if backend.device.type=='cuda' else None)


def component_options(args, operator):
    """Run H once for both modules; only H receives the M1 cross ablations."""
    options=copy.copy(args)
    options.no_ablation = not (args.module=='M1+M2' and operator=='H' and not args.no_ablation)
    return options
