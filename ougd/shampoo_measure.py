"""Measure one post-update Llama checkpoint under common calibrated Shampoo states."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import TokenData, sha256
from .hessian import Hessian
from .model import ModelConfig, Transformer
from .positive_subspaces import positive_count
from .seven_measure import batch_hash, git_provenance, write
from .shampoo import Partition
from .shampoo_calibration import (CalibratedOperator, bias_corrected_momentum, calibration_history,
                                  native_direction)
from .shampoo_metrics import (PAIRS, coverage, directional_differences, eigenpairs, factor_heatmaps,
                              hessian_heatmap, matched_ranks, momentum_differences,
                              probe_differences, ratio_summary, saved_spectrum)


STAGES = ('operator', 'factor_heatmaps', 'top', 'coverage', 'hessian_heatmaps', 'effective', 'effective_spectrum')


def draw_heatmap(path, values, rows, columns, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(columns)*0.18), max(5, len(rows)*0.18)))
    image = ax.imshow(values, aspect='auto', interpolation='nearest', cmap='viridis')
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(rows)), rows, fontsize=6)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def analysis_spectrum(action, size, k, device, args, path, progress):
    return saved_spectrum(path, lambda: eigenpairs(action, size, k, device, args,
                                                   lambda calls: progress(calls=calls)), k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--fp1', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hessian-cache', help='reuse raw Hessian eigensolves across W/ridge variants')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--calibration-microbatch', type=int, help='defaults to the training microbatch size')
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--ridge', type=float)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--probes', type=int, default=32)
    parser.add_argument('--probe-seed', type=int, default=23)
    parser.add_argument('--solver-tol', type=float, default=1e-6)
    parser.add_argument('--residual-tol', type=float, default=1e-4)
    parser.add_argument('--maxiter', type=int, default=1000)
    parser.add_argument('--spectral-modes', type=int, default=3,
                        help='number of eigenvalues measured at each effective-Hessian edge')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if min(args.k, args.microbatch, args.window, args.spectral_modes, args.maxiter) < 1 or args.probes < 2:
        parser.error('invalid ranks, iteration limits, probe count or batch settings')
    if not np.isfinite([args.solver_tol, args.residual_tol]).all() or min(args.solver_tol, args.residual_tol) <= 0:
        parser.error('invalid spectral tolerances')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    training = state['args']
    calibration_microbatch = (training['microbatch'] if args.calibration_microbatch is None
                              else args.calibration_microbatch)
    if calibration_microbatch < 1:
        parser.error('calibration microbatch size must be positive')
    provenance = state['provenance']
    config = ModelConfig(**state['config'])
    model = Transformer(config)
    model.load_state_dict(state['model'])
    model.to(device).eval()
    partition = Partition(model)
    if partition.hash != provenance['partition_hash']:
        raise ValueError('checkpoint partition hash differs from the model layout')
    if args.ridge is None:
        args.ridge = training['ridge']
    if not np.isfinite(args.ridge) or args.ridge <= 0:
        parser.error('ridge must be finite and positive')
    fp1_hash = sha256(args.fp1)
    if fp1_hash != provenance['fp1_hash']:
        raise ValueError('FP1 file does not match the checkpoint')
    if args.window > len(provenance['calibration']):
        raise ValueError('requested window exceeds the registered calibration sequence')
    records = provenance['calibration'][:args.window]
    raw = torch.load(args.fp1, map_location='cpu', weights_only=True)
    batch = {key: raw[key].contiguous() for key in ('input_ids', 'labels')}
    tokens = int((batch['labels'] != -100).sum())
    if (batch['labels'].ndim != 2 or batch['input_ids'].shape != batch['labels'].shape
            or tokens != training['batch_tokens']):
        raise ValueError('FP1 must match the configured global prediction-token count')
    batches = [{key: value[i:i+args.microbatch].to(device) for key, value in batch.items()}
               for i in range(0, len(batch['labels']), args.microbatch)]
    batches = [b for b in batches if (b['labels'] != -100).any()]
    scope = 'llama120m_omega' if partition.n == 99_090_432 else 'development_model_omega'
    stage_status = {stage: {'status': 'pending' if stage in args.stages else 'not_requested'} for stage in STAGES}
    identity = dict(checkpoint_sha256=sha256(args.checkpoint), fp1_hash=fp1_hash,
                    fp1_token_hash=batch_hash(batch), partition_hash=partition.hash, k=args.k,
                    solver_tol=args.solver_tol, residual_tol=args.residual_tol, maxiter=args.maxiter,
                    probe_seed=args.probe_seed, microbatch=args.microbatch, device=str(device),
                    torch=str(torch.__version__))
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    hcache = Path(args.hessian_cache or out/'hessian_cache') / cache_key
    hcache.mkdir(parents=True, exist_ok=True)
    software = dict(torch=str(torch.__version__), cuda=torch.version.cuda, git=git_provenance())
    report = dict(identity=identity, origin=state['origin'], completed_steps=state['completed_steps'],
                  scope=scope, dimension=partition.n, state_protocol='cal', W=args.window,
                  ridge=args.ridge, probe_seed=args.probe_seed, probes=args.probes,
                  spectral_modes=args.spectral_modes, calibration_microbatch=calibration_microbatch,
                  calibration_precision=training['precision'],
                  root_target_tolerance=1e-5, calibration=records, software=software,
                  analysis_world_size=1, hessian_precision='float32, TF32 disabled',
                  root_backend='torch symmetric FP64 eigendecomposition and global thin Gram',
                  gpu=(torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
                  root_last_refresh_step=state['optimizer']['primary']['root_last_refresh_step'],
                  native_root_accuracy=state['optimizer']['primary']['root_accuracy'],
                  pairs={f'{q}|{p}': {} for q, p in PAIRS}, stages=stage_status)
    report['native_root_age'] = state['completed_steps'] - report['root_last_refresh_step']
    report_path = out/'metrics.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        keys = ('identity', 'W', 'ridge', 'probes', 'calibration', 'spectral_modes',
                'calibration_microbatch')
        if any(previous[key] != report[key] for key in keys):
            raise ValueError('output directory holds a different measurement configuration')
        report = previous
        requested = set(args.stages)
        stage_status = report['stages']
        for stage in STAGES:
            if stage in requested:
                stage_status[stage] = {'status': 'pending'}

    current = None

    def progress(**event):
        print(json.dumps(dict(stage=current, **event)), flush=True)

    def begin(stage):
        nonlocal current
        current = stage
        stage_status[stage]['status'] = 'running'
        write(report_path, report)
        progress(status='running')
        return time.perf_counter()

    def done(start, status='completed'):
        stage_status[current].update(status=status, seconds=time.perf_counter()-start)
        if device.type == 'cuda':
            stage_status[current]['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        write(report_path, report)
        progress(status=status)

    def store(label, measurements):
        for pair, value in measurements.items():
            report['pairs'][pair][label] = value

    operators = {}
    needs_factors = set(args.stages).intersection(('operator', 'factor_heatmaps', 'top', 'effective', 'effective_spectrum'))
    try:
        if needs_factors:
            stage_status['calibration'] = {'status': 'pending'}
            start = begin('calibration')
            if sha256(args.data) != state['data_hash']:
                raise ValueError('training data do not match the checkpoint')
            data = TokenData(args.data, config.context, config.vocab_size)
            z = calibration_history(model, partition, data, records, training['beta2'],
                                    calibration_microbatch, training['precision'], device)
            half = bool(set(args.stages).intersection(('effective', 'effective_spectrum')))
            for scheme in ('G', 'M', 'U'):
                progress(scheme=scheme)
                operators[scheme] = CalibratedOperator(partition, scheme, z, args.ridge, half)
            report['root_accuracy'] = {key: op.accuracy for key, op in operators.items()}
            action_errors = [error for op in operators.values() for group in op.accuracy.values()
                             for power in ('0.25', '0.125') for error in group.get(power, {}).values()
                             if error is not None]
            action_errors.append(operators['G'].accuracy['global']['left_composition_error'])
            report['root_accuracy_met'] = max(action_errors) <= report['root_target_tolerance']
            done(start)
            actions = {key: op.apply for key, op in operators.items()}
            if 'factor_heatmaps' in args.stages:
                start = begin('factor_heatmaps')
                factors = factor_heatmaps(partition, z, operators)
                write(out/'factor_heatmaps.json', factors)
                if not args.no_plots:
                    for scheme, heatmap in factors.items():
                        labels = heatmap['blocks']
                        for key in ('global_left_norm', 'right_difference_norm'):
                            draw_heatmap(out/f'{scheme}_{key}.png', heatmap[key], labels, labels, key.replace('_', ' '))
                done(start)
            del z
        if 'operator' in args.stages:
            start = begin('operator')
            m = bias_corrected_momentum(state, partition, device)
            native = native_direction(state, partition, m) if m is not None else None
            momentum, report['native_vs_calibrated'] = momentum_differences(actions, m, native,
                                                                           training['shampoo_partition'])
            store('E_m', momentum)
            del m, native
            store('E_R_I', probe_differences(actions, partition.n, device, args.probes, args.probe_seed))
            done(start)
        del state

        def backend(params):
            return Hessian(model, batches, tokens, 0, parameters=params)

        hessian = backend(partition.parameters)
        report['hvp_configuration'] = dict(tokens=tokens, microbatch=args.microbatch,
                                           operator='true token-mean CE Hessian on Omega')
        if set(args.stages).intersection(('top', 'coverage')):
            start = begin('top' if 'top' in args.stages else 'coverage')
            vectors, info = analysis_spectrum(hessian.local, partition.n, args.k, device, args,
                                              hcache/'global', progress)
            rank = info['used_positive_rank']
            values = np.asarray(info['values'][:rank])
            report['hessian_top'] = info
            if 'top' in args.stages:
                rows = directional_differences(actions, (torch.tensor(vectors[:, i], device=device)
                                                          for i in range(rank)))
                store('E_R_V', {pair: dict(ratio_summary(samples), eigenvalues=values)
                                for pair, samples in rows.items()})
            top_status = ('completed' if rank else 'unavailable') if info['ordered_leading_space_certified'] else 'incomplete'
            if 'coverage' in args.stages and current == 'top':
                done(start, top_status)
                start = begin('coverage')
            if 'coverage' in args.stages:
                result = {}
                local, hist, budget, spectra = {}, {}, {}, {}
                for scheme in ('M', 'U'):
                    local[scheme], hist[scheme], budget[scheme], spectra[scheme] = {}, {}, {}, {}
                    for label, entries in partition.blocks[scheme].items():
                        local_hessian = backend({e.name: partition.parameters[e.name] for e in entries})
                        basis, spec = analysis_spectrum(local_hessian.local, local_hessian.n, args.k,
                                                       device, args, hcache/f'{scheme}_{label.replace(".", "_")}',
                                                       lambda **event: progress(block=label, **event))
                        local[scheme][label] = (partition.interval(entries), basis)
                        spectra[scheme][label] = spec
                        hist[scheme][label] = spec['used_positive_rank']
                        cap = args.k if scheme == 'U' else (9 if label[-1] in 'QKVO' else 8)
                        budget[scheme][label] = min(cap, hist[scheme][label])
                    result[scheme] = dict(historical=coverage(partition, vectors, values, local[scheme], hist[scheme]),
                                          budget_720=coverage(partition, vectors, values, local[scheme], budget[scheme]),
                                          spectra=spectra[scheme])
                matched = matched_ranks(budget['M'], budget['U'],
                                        *({key: item['values'] for key, item in spectra[scheme].items()}
                                          for scheme in ('M', 'U')))
                for scheme, ranks in zip(('M', 'U'), matched):
                    result[scheme]['matched_actual_rank'] = coverage(partition, vectors, values, local[scheme], ranks)
                    if not args.no_plots and rank:
                        for metric in ('p', 'a'):
                            blocks = result[scheme]['historical']['blocks']
                            draw_heatmap(out/f'{scheme}_coverage_{metric}.png',
                                         [row[metric] for row in blocks.values()], list(blocks),
                                         list(range(1, rank+1)), f'{scheme} coverage {metric}')
                write(out/'coverage.json', result)
                hessian = backend(partition.parameters)  # Principal-block solvers changed requires_grad.
                if any(not spec['ordered_leading_space_certified']
                       for scheme in spectra.values() for spec in scheme.values()):
                    top_status = 'incomplete'
            done(start, top_status)

        if 'hessian_heatmaps' in args.stages:
            start = begin('hessian_heatmaps')
            for scheme in ('M', 'U'):
                data = hessian_heatmap(hessian.local, partition, scheme, args.probes, args.probe_seed,
                                       device, progress)
                write(out/f'{scheme}_hessian_heatmap.json', data)
                if not args.no_plots:
                    for key in ('norm', 'energy_fraction', 'element_rms'):
                        if data[key] is not None:
                            draw_heatmap(out/f'{scheme}_hessian_{key}.png', data[key],
                                         data['blocks'], data['blocks'], key.replace('_', ' '))
            done(start)

        def effective(op, x):
            return op.apply(hessian.local(op.apply(x, 0.125)), 0.125)

        effective_actions = {key: (lambda x, op=op: effective(op, x)) for key, op in operators.items()}
        if 'effective' in args.stages:
            start = begin('effective')
            moments = {key: [] for key in effective_actions}
            store('E_S', probe_differences(effective_actions, partition.n, device, args.probes,
                                           args.probe_seed, moments))
            report['effective_spectral_moments'] = {}
            for key, rows in moments.items():
                samples = np.asarray(rows) / partition.n
                mean, second = samples.mean(axis=0)
                report['effective_spectral_moments'][key] = dict(
                    mean=mean, mean_square=second, standard_error=samples.std(axis=0, ddof=1)/np.sqrt(args.probes),
                    sample_columns=['trace_over_dimension', 'frobenius_square_over_dimension'],
                    samples=samples)
            done(start)
        if 'effective_spectrum' in args.stages:
            start = begin('effective_spectrum')
            summary = {}
            for scheme, action in effective_actions.items():
                edges = {}
                for label, sign in (('upper', 1), ('lower', -1)):
                    spec = eigenpairs(lambda x: sign * action(x), partition.n, args.spectral_modes,
                                      device, args, lambda calls: progress(scheme=scheme, edge=label, calls=calls))
                    edge = {k: v for k, v in spec.items() if k != 'vectors'}
                    edge['values'] = sign * edge['values']
                    edges[label] = edge
                upper, lower = edges['upper'], edges['lower']
                hi = float(upper['values'][0]) if positive_count(upper) else None
                lo = float(lower['values'][0]) if positive_count(lower) else None
                certified = all(edge['ordered_leading_space_certified'] and len(edge['values'])
                                and edge['residual_passed'][0] for edge in (upper, lower))
                summary[scheme] = dict(largest_positive=hi, most_negative=lo,
                                       spectral_norm=max(abs(upper['values'][0]), abs(lower['values'][0]))
                                       if certified else None, edges=edges)
            write(out/'effective_spectrum.json', summary)
            done(start, 'completed' if all(item['spectral_norm'] is not None for item in summary.values())
                 else 'incomplete')
    except Exception as exc:
        if current and stage_status[current]['status'] == 'running':
            stage_status[current].update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        write(report_path, report)
        raise
    if any(stage_status[stage]['status'] == 'incomplete' for stage in args.stages):
        raise SystemExit('One or more stages did not certify the requested eigensolve. See metrics.json.')


if __name__ == '__main__':
    main()
