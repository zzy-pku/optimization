"""Prepare the shared initialization, select a paired pilot LR, or summarize measurements."""
import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from .model import ModelConfig, Transformer
from .seven_measure import write
from .train import atomic_save


PEAKS = (1e-4, 2e-4, 4e-4)
STEPS = (0, 300, 3000, 6000, 9000, 12208)


def pilot_verdict(log, warmup=300, steps=400):
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if [row['completed_steps'] for row in rows] != list(range(1, steps+1)) or not rows:
        return ['incomplete pilot']
    loss, updates, roots = (np.array([r[key] for r in rows])
                            for key in ('train_ce', 'update_parameter_ratio', 'root_accuracy'))
    reasons = []
    if not np.isfinite([loss, updates, roots]).all():
        reasons.append('nonfinite loss, update ratio or root action error')
    if updates.max() > 0.1:
        reasons.append('update/parameter norm exceeds 0.1')
    if roots.max() > 1e-5:
        reasons.append('root action error exceeds 1e-5')
    baseline = median(loss[max(0, warmup-20):warmup])
    if any(median(loss[start:start+20]) > 2*baseline for start in range(warmup, steps-19)):
        reasons.append('post-warmup 20-step median loss exceeds twice the warmup-end median')
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--config', required=True)
    init.add_argument('--output', type=Path, required=True)
    init.add_argument('--seed', type=int, default=17)
    pilot = sub.add_parser('select-lr')
    pilot.add_argument('--root', type=Path, required=True)
    summary = sub.add_parser('summarize')
    summary.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'init':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed)
        model = Transformer(ModelConfig(**json.loads(Path(args.config).read_text())))
        atomic_save(model.state_dict(), args.output)
    elif args.command == 'select-lr':
        candidates = {}
        stable = []
        for peak in PEAKS:
            candidate = {}
            for scheme in ('M', 'U'):
                folder = args.root/f'pilot-{scheme}-{peak:.4f}'
                reasons = pilot_verdict(folder/'t0.jsonl')
                if not (folder/'checkpoint_000400.pt').exists():
                    reasons.append('missing final pilot checkpoint')
                candidate[scheme] = dict(stable=not reasons, reasons=reasons)
            candidates[f'{peak:.4f}'] = candidate
            if all(item['stable'] for item in candidate.values()):
                stable.append(peak)
        chosen = max(stable, default=None)
        write(args.root/'pilot_selection.json', dict(candidates=candidates, shared_peak_lr=chosen,
              criteria=dict(warmup_steps=300, steps=400, root_tolerance=1e-5, max_update_ratio=0.1,
                            loss_median_window=20, max_loss_median_ratio=2.0)))
        if chosen is None:
            raise SystemExit('No shared stable peak LR. See pilot_selection.json.')
        print(f'{chosen:.4f}')
    else:
        records = []
        for scheme in ('M', 'U'):
            for step in STEPS:
                variants = ['main'] + (['w16', 'ridge_1e-7', 'ridge_1e-5'] if step in (3000, 12208) else [])
                for variant in variants:
                    path = args.root/f'train-{scheme}'/'analysis'/f'{step:06d}'/variant/'metrics.json'
                    if path.exists():
                        report = json.loads(path.read_text(encoding='utf-8'))
                        records.append(dict(path=str(path.relative_to(args.root)), origin=report['origin'],
                                            completed_steps=report['completed_steps'], variant=variant,
                                            W=report['W'], ridge=report['ridge'], scope=report['scope'],
                                            state_protocol=report['state_protocol'], probe_seed=report['probe_seed'],
                                            root_accuracy_met=report.get('root_accuracy_met'),
                                            metrics={pair: {label: {key: value for key, value in metric.items()
                                                                    if key not in ('samples', 'sample_columns')}
                                                            for label, metric in row.items()}
                                                     for pair, row in report['pairs'].items()},
                                            stages=report['stages']))
                    else:
                        records.append(dict(origin=f'train-{scheme}', completed_steps=step, variant=variant,
                                            status='missing'))
        write(args.root/'comparison.json', records)


if __name__ == '__main__':
    main()
