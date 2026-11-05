"""Create disjoint, reproducible fixed probe batches from packed validation rows."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import torch
from .data import TokenData, sha256


def batch_hash(batch):
    return hashlib.sha256(b''.join(batch[k].contiguous().numpy().tobytes()
                                   for k in ('input_ids', 'labels'))).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--vocab', type=int, default=16000)
    parser.add_argument('--tokens', type=int, default=8192)
    parser.add_argument('--cohorts', type=int, default=3)
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    if args.cohorts < 1 or args.tokens <= 0 or args.tokens % args.context:
        raise ValueError('tokens must be a positive multiple of context; cohorts must be positive')
    data = TokenData(args.data, args.context, args.vocab)
    rows = args.tokens // args.context
    if rows * args.cohorts > len(data.tokens):
        raise ValueError('not enough disjoint packed rows for all fixed probes')
    indices = np.random.default_rng(args.seed).permutation(len(data.tokens))[:rows * args.cohorts]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(data_sha256=sha256(args.data), seed=args.seed, context=args.context,
                vocab=args.vocab, tokens_per_cohort=args.tokens, cohorts=[],
                disjointness='Packed row indices are disjoint; document-level separation requires document IDs and is not established here.')
    for cohort in range(args.cohorts):
        selected = indices[cohort*rows:(cohort+1)*rows].tolist()
        batch = data.batch(selected, 'cpu')
        assert int((batch['labels'] != -100).sum()) == args.tokens
        name = f'FP{cohort+1}.pt'
        torch.save(batch, out / name)
        meta['cohorts'].append(dict(name=name, row_indices=selected,
                                    batch_sha256=batch_hash(batch)))
    (out/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
