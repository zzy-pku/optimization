"""Token-weighted DDP training with replayable PRE-update measurement checkpoints."""
import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from . import distributed as dd
from .data import TokenData, sha256, make_batches
from .model import ModelConfig, Transformer, ce_sum


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    return copy.deepcopy(value)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--validation-data')
    p.add_argument('--validation-every', type=int, default=100)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=15259)
    p.add_argument('--batch-tokens', type=int, default=131072)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--layer', type=int, default=5, help='zero-based measurement layer')
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--min-lr', type=float, default=4e-5)
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--measure-steps', default='0,10,50,150,300,1000,3000,7500,12000')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--save-every', type=int, default=250, help='post-update recovery checkpoint interval')
    p.add_argument('--resume')
    args = p.parse_args()
    device, rank, world = dd.init()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    c = ModelConfig(**json.loads(Path(args.config).read_text()))
    if args.steps < 1 or args.microbatch < 1 or args.save_every < 1 or not 0 <= args.layer < c.layers:
        raise ValueError('invalid steps, microbatch or layer')
    sequences, remainder = divmod(args.batch_tokens, c.context)
    if remainder or sequences < world or sequences % (world * args.microbatch):
        raise ValueError('batch-tokens must be divisible by context * world-size * microbatch')
    data = TokenData(args.data, c.context, c.vocab_size)
    data_hash = sha256(args.data)
    model = Transformer(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.1)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['data_hash'] != data_hash or state['config'] != asdict(c):
            raise ValueError('resume dataset/model mismatch')
        if state['args']['batch_tokens'] != args.batch_tokens:
            raise ValueError('resume must preserve global batch size')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        torch.set_rng_state(state['rng'])
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(state['cuda_rng'], device)
    wrapped = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else model
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = model.selected(args.layer)
    measure_steps = {int(x) for x in args.measure_steps.split(',') if x}
    def cpu_state():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def checkpoint(step):
        return dict(config=asdict(c), model=cpu_state(), optimizer=optimizer.state_dict(),
                    step=step, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None, args=vars(args), data_hash=data_hash)
    if rank == 0:
        (output / 'run.json').write_text(json.dumps(dict(args=vars(args), model=asdict(c),
            parameters=sum(p.numel() for p in model.parameters()), world_size=world,
            data_hash=data_hash, torch=torch.__version__), indent=2))
    for step in range(start, args.steps):
        begin = time.perf_counter()
        indices = [(step * sequences + j) % len(data.tokens) for j in range(sequences)]
        local = indices[rank::world]
        batches = make_batches(data, local, args.microbatch, device)
        count = dd.sum_(torch.tensor(sum(int((b['labels'] != -100).sum()) for b in batches), device=device))
        if count.item() == 0:
            raise ValueError('empty effective batch')
        lr = args.lr * (step + 1) / args.warmup if step < args.warmup else args.min_lr + .5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * (step-args.warmup) / max(1, args.steps-args.warmup-1)))
        for g in optimizer.param_groups:
            g['lr'] = lr
        measured = step in measure_steps or step == args.steps-1
        before = {k: v.detach().clone() for k, v in selected.items()}
        pre = checkpoint(step) if measured and rank == 0 else None
        if pre is not None:
            pre['optimizer'] = cpu_tree(pre['optimizer'])
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for n, batch in enumerate(batches):
            sync = wrapped.no_sync() if world > 1 and n < len(batches)-1 else nullcontext()
            with sync:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                    loss = ce_sum(wrapped, batch)
                loss_sum += loss.detach().float()
                (loss * (world / count)).backward()
        raw = {k: v.grad.detach().cpu().clone() for k, v in selected.items()}
        gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        updates = {k: v.detach() - before[k] for k, v in selected.items()}
        if measured and rank == 0:
            pre.update(pre_update_state=True, layer=args.layer,
                       batch={k: v.cpu() for k, v in data.batch(indices, 'cpu').items()},
                       batch_indices=indices, data_cursor=step * sequences,
                       raw_gradients=raw, updates={k: v.cpu() for k, v in updates.items()},
                       gradient_precision=args.precision, valid_tokens=int(count))
            torch.save(pre, output / f'pre_step_{step:06d}.pt')
        train_loss = dd.sum_(loss_sum) / count
        metrics = dict(step=step, train_ce=float(train_loss), lr=lr, valid_tokens=int(count),
            total_tokens=(step+1)*args.batch_tokens, grad_norm=float(gradnorm), clipped=bool(gradnorm > 1),
            tokens_per_second=int(count) / (time.perf_counter()-begin),
            blocks={k: dict(parameter_norm=float(v.detach().norm()), gradient_norm=float(raw[k].norm()),
                            update_norm=float(updates[k].norm())) for k, v in selected.items()})
        if args.validation_data and (step % args.validation_every == 0 or step == args.steps-1):
            val = TokenData(args.validation_data, c.context, c.vocab_size)
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            model.eval()
            with torch.no_grad():
                for pos in range(rank, len(val.tokens), world):
                    b = val.batch([pos], device)
                    totals[0] += ce_sum(model, b)
                    totals[1] += (b['labels'] != -100).sum()
            dd.sum_(totals)
            metrics['validation_ce'] = float(totals[0]/totals[1])
            model.train()
        if rank == 0:
            with (output / 't0.jsonl').open('a') as f:
                f.write(json.dumps(metrics) + '\n')
            print(json.dumps(metrics), flush=True)
            if (step+1) % args.save_every == 0 or step == args.steps-1:
                torch.save(checkpoint(step+1), output / 'latest.pt.tmp')
                (output / 'latest.pt.tmp').replace(output / 'latest.pt')
    dd.close()

if __name__ == '__main__':
    main()
