"""Token-weighted DDP training with replayable PRE-update measurement checkpoints."""
import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from . import distributed as dd
from .data import TokenData, sha256, make_batches
from .model import ModelConfig, Transformer, ce_sum
from .shampoo import Partition, Shampoo


def optimizer_settings(args):
    settings = {'optimizer': args.optimizer, 'weight_decay': args.weight_decay}
    if args.optimizer in ('adamw', 'adam', 'soap', 'muon', 'shampoo'):
        settings.update(beta1=args.beta1, beta2=args.beta2, eps=args.eps)
    if args.optimizer == 'sgd':
        settings.update(momentum=args.momentum, nesterov=args.nesterov)
    if args.optimizer in ('muon', 'soap'):
        settings.update(aux_lr=args.aux_lr, lr=args.lr)
    if args.optimizer == 'muon':
        settings.update(muon_momentum=args.muon_momentum)
    if args.optimizer == 'soap':
        settings.update(soap_precondition_frequency=args.soap_precondition_frequency,
                        soap_max_precond_dim=args.soap_max_precond_dim)
    if args.optimizer == 'shampoo':
        settings.update(shampoo_partition=args.shampoo_partition, ridge=args.ridge,
                        root_refresh=args.root_refresh)
    return settings


class PairedOptimizer:
    """One stateful optimizer for block matrices and AdamW for remaining parameters."""
    def __init__(self, primary, auxiliary):
        self.primary, self.auxiliary = primary, auxiliary
        self.param_groups = primary.param_groups + auxiliary.param_groups

    def zero_grad(self, **kwargs):
        self.primary.zero_grad(**kwargs); self.auxiliary.zero_grad(**kwargs)

    def step(self):
        self.primary.step(); self.auxiliary.step()

    def state_dict(self):
        return {'primary': self.primary.state_dict(), 'auxiliary': self.auxiliary.state_dict()}

    def load_state_dict(self, state):
        self.primary.load_state_dict(state['primary'])
        self.auxiliary.load_state_dict(state['auxiliary'])
        self.param_groups = self.primary.param_groups + self.auxiliary.param_groups


def make_optimizer(model, args):
    parameters = model.parameters()
    if args.optimizer == 'shampoo':
        partition = Partition(model)
        other = [p for name, p in model.named_parameters() if name not in partition.parameters]
        primary = Shampoo(partition, args.shampoo_partition, args.lr, (args.beta1, args.beta2),
                          args.ridge, args.weight_decay, args.root_refresh)
        auxiliary = torch.optim.AdamW(other, lr=args.lr, betas=(args.beta1, args.beta2),
                                      eps=args.eps, weight_decay=args.weight_decay)
        return PairedOptimizer(primary, auxiliary)
    if args.optimizer in ('muon', 'soap'):
        named = list(model.named_parameters())
        matrices = [p for name, p in named if name.startswith('blocks.') and p.ndim == 2]
        other = [p for name, p in named if not (name.startswith('blocks.') and p.ndim == 2)]
        if not matrices or not other:
            raise ValueError('hybrid optimizer requires block matrices and auxiliary parameters')
        if args.optimizer == 'muon':
            from .vendor.muon import SingleDeviceMuon
            primary = SingleDeviceMuon(matrices, lr=args.lr, momentum=args.muon_momentum,
                                       weight_decay=args.weight_decay)
        else:
            from .vendor.soap import SOAP
            primary = SOAP(matrices, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
                           weight_decay=args.weight_decay,
                           precondition_frequency=args.soap_precondition_frequency,
                           max_precond_dim=args.soap_max_precond_dim)
        auxiliary = torch.optim.AdamW(other, lr=args.aux_lr, betas=(args.beta1, args.beta2),
                                      eps=args.eps, weight_decay=args.weight_decay)
        for group in auxiliary.param_groups: group['lr_scale'] = args.aux_lr / args.lr
        return PairedOptimizer(primary, auxiliary)
    if args.optimizer == 'sgd':
        return torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                               nesterov=args.nesterov, weight_decay=args.weight_decay)
    cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    return cls(parameters, lr=args.lr, betas=(args.beta1, args.beta2),
               eps=args.eps, weight_decay=args.weight_decay)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
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
    p.add_argument('--measure-blocks', choices=['four', 'seven'], default='four',
                   help='which block gradients and updates to save in pre-step checkpoints')
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--optimizer', choices=['adamw', 'adam', 'sgd', 'muon', 'soap', 'shampoo'], default='adamw')
    p.add_argument('--shampoo-partition', choices=['M', 'U'])
    p.add_argument('--ridge', type=float, default=1e-6)
    p.add_argument('--root-refresh', type=int, default=10)
    p.add_argument('--initial-state', help='shared initial model state_dict for Shampoo training')
    input_source = p.add_mutually_exclusive_group()
    input_source.add_argument('--tokenizer', help='tokenizer file whose SHA256 is recorded')
    input_source.add_argument('--pretokenized-provenance',
                              help='JSON with data_sha256 and source for pretokenized data without a tokenizer file')
    p.add_argument('--fp1', help='fixed Hessian validation batch whose SHA256 is recorded')
    p.add_argument('--calibration-seed', type=int, default=19)
    p.add_argument('--calibration-windows', type=int, default=16)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--beta1', type=float, default=.9)
    p.add_argument('--beta2', type=float, default=.95)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--momentum', type=float, default=.9)
    p.add_argument('--nesterov', action='store_true')
    p.add_argument('--aux-lr', type=float, default=4e-4, help='Muon/SOAP AdamW auxiliary peak LR')
    p.add_argument('--muon-momentum', type=float, default=.95)
    p.add_argument('--soap-precondition-frequency', type=int, default=10)
    p.add_argument('--soap-max-precond-dim', type=int, default=1024)
    p.add_argument('--min-lr', type=float, default=4e-5)
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--measure-steps', default='0,10,50,150,300,1000,3000,7500,12000')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--save-every', type=int, default=250, help='post-update recovery checkpoint interval')
    p.add_argument('--resume')
    args = p.parse_args()
    shampoo = args.optimizer == 'shampoo'
    if shampoo and (not all((args.shampoo_partition, args.initial_state,
                             args.tokenizer or args.pretokenized_provenance, args.fp1))
                    or not math.isfinite(args.ridge) or args.ridge <= 0 or args.root_refresh < 1
                    or args.calibration_windows < 1):
        p.error('Shampoo requires partition, initial state, data provenance, FP1 and valid calibration settings')
    if args.nesterov and (args.optimizer != 'sgd' or args.momentum <= 0):
        p.error('--nesterov requires SGD and positive momentum')
    if (args.weight_decay < 0 or args.eps <= 0 or not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1)
            or args.momentum < 0 or not 0 <= args.muon_momentum < 1 or args.lr <= 0 or args.aux_lr <= 0
            or args.soap_precondition_frequency < 1 or args.soap_max_precond_dim < 1):
        p.error('invalid optimizer hyperparameters')
    if args.precision == 'fp32' or shampoo:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
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
    if shampoo:
        model.load_state_dict(torch.load(args.initial_state, map_location='cpu', weights_only=True))
    optimizer = make_optimizer(model, args)
    if shampoo:
        from .seven_measure import batch_hash
        pretokenized = None
        if args.pretokenized_provenance:
            source = json.loads(Path(args.pretokenized_provenance).read_text(encoding='utf-8'))
            if source.get('data_sha256') != data_hash or not source.get('source'):
                raise ValueError('pretokenized provenance must match the training hash and name its source')
            pretokenized = dict(manifest_hash=sha256(args.pretokenized_provenance), source=source['source'])
        generator = np.random.default_rng(args.calibration_seed)
        count = sequences * args.calibration_windows
        if count > len(data.tokens):
            raise ValueError('not enough distinct training rows for calibration')
        chosen = generator.choice(len(data.tokens), count, replace=False).reshape(-1, sequences)
        calibration = [dict(indices=indices.tolist(), token_hash=batch_hash(data.batch(indices, 'cpu')))
                       for indices in chosen]
        provenance = dict(initialization_hash=sha256(args.initial_state),
                          tokenizer_hash=sha256(args.tokenizer) if args.tokenizer else None,
                          pretokenized=pretokenized,
                          fp1_hash=sha256(args.fp1), partition=optimizer.primary.partition.manifest,
                          partition_hash=optimizer.primary.partition.hash, calibration=calibration,
                          rmsnorm_epsilon=1e-6, roots='torch.linalg.eigh float64; factors and cache float32',
                          hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                          software=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
                          world_size=world)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['data_hash'] != data_hash or state['config'] != asdict(c):
            raise ValueError('resume dataset/model mismatch')
        if state['args']['batch_tokens'] != args.batch_tokens:
            raise ValueError('resume must preserve global batch size')
        old = state['args']
        defaults = dict(optimizer='adamw', weight_decay=.1, beta1=.9, beta2=.95,
                        eps=1e-8, momentum=.9, nesterov=False)
        current_settings = optimizer_settings(args)
        if any(old.get(key, defaults.get(key)) != value for key, value in current_settings.items()):
            raise ValueError('resume must preserve optimizer type and hyperparameters')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        if shampoo:
            for key in ('steps', 'warmup', 'lr', 'min_lr', 'seed', 'precision', 'microbatch'):
                if old[key] != getattr(args, key):
                    raise ValueError(f'resume must preserve {key}')
            resume_keys = ('initialization_hash', 'tokenizer_hash', 'pretokenized', 'fp1_hash', 'partition_hash',
                           'calibration', 'roots', 'world_size')
            if (any(state['provenance'].get(key) != provenance[key] for key in resume_keys)
                    or state['data_cursor'] != start * sequences):
                raise ValueError('resume Shampoo provenance or data cursor mismatch')
            rng = state['rank_rng'][rank]
            np.random.set_state(rng['numpy'])
            random.setstate(rng['python'])
        else:
            rng = state
        torch.set_rng_state(rng['rng'])
        if device.type == 'cuda' and rng.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(rng['cuda_rng'], device)
        del state
    wrapped = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else model
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = model.selected_seven(args.layer) if args.measure_blocks == 'seven' else model.selected(args.layer)
    if shampoo:
        names = {id(p): name for name, p in optimizer.primary.partition.parameters.items()}
        selected = {names[id(p)]: p for p in selected.values()}
    measure_steps = {int(x) for x in args.measure_steps.split(',') if x}
    def cpu_state():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def checkpoint(step):
        return dict(config=asdict(c), model=cpu_state(), optimizer=optimizer.state_dict(),
                    step=step, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None, args=vars(args), data_hash=data_hash)
    def shampoo_checkpoint(step):
        local_rng = dict(rng=torch.get_rng_state(), numpy=np.random.get_state(),
                         python=random.getstate(),
                         cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        if world > 1:
            rng_by_rank = [None] * world
            torch.distributed.all_gather_object(rng_by_rank, local_rng)
        else:
            rng_by_rank = [local_rng]
        if rank == 0:
            state = checkpoint(step)
            state.update(optimizer=cpu_tree(state['optimizer']), rank_rng=rng_by_rank,
                         data_cursor=step * sequences, completed_steps=step,
                         origin=f'train-{args.shampoo_partition}', provenance=provenance,
                         scheduler=dict(completed_steps=step, warmup=args.warmup, steps=args.steps,
                                        peak_lr=args.lr, min_lr=args.min_lr))
            if step in measure_steps or step == args.steps:
                destination = output / f'checkpoint_{step:06d}.pt'
                atomic_save(state, destination)
                if step and (step % args.save_every == 0 or step == args.steps):
                    latest = output / 'latest.pt'
                    link = output / 'latest.link'
                    link.unlink(missing_ok=True)
                    try:
                        os.link(destination, link)
                        link.replace(latest)
                    except OSError:
                        atomic_save(state, latest)
            elif step % args.save_every == 0:
                atomic_save(state, output / 'latest.pt')
    if rank == 0:
        (output / 'run.json').write_text(json.dumps(dict(args=vars(args), model=asdict(c),
            parameters=sum(p.numel() for p in model.parameters()), world_size=world,
            data_hash=data_hash, torch=torch.__version__,
            optimizer_parameter_groups=({'primary': [n for n,p in model.named_parameters() if n.startswith('blocks.') and p.ndim == 2],
              'adamw_auxiliary': [n for n,p in model.named_parameters() if not (n.startswith('blocks.') and p.ndim == 2)]}
              if args.optimizer in ('muon', 'soap', 'shampoo') else {'all': [n for n,p in model.named_parameters()]}),
            **({'shampoo': provenance} if shampoo else {})), indent=2))
    if shampoo and start == 0:
        shampoo_checkpoint(0)
    if shampoo and args.resume and rank == 0 and (output / 't0.jsonl').exists():
        log = output / 't0.jsonl'
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        suffix = log.with_suffix('.tmp')
        suffix.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['completed_steps'] <= start))
        suffix.replace(log)
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
            g['lr'] = lr * g.get('lr_scale', 1.)
        measured = not shampoo and (step in measure_steps or step == args.steps-1)
        observed = (optimizer.primary.partition.parameters if shampoo else selected)
        before = {k: v.detach().clone() for k, v in observed.items()}
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
        gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=shampoo)
        if shampoo and device.type == 'cuda':
            torch.cuda.synchronize(device)
        optim_start = time.perf_counter()
        optimizer.step()
        if shampoo and device.type == 'cuda':
            torch.cuda.synchronize(device)
        optim_seconds = time.perf_counter()-optim_start
        updates = {k: v.detach() - before[k] for k, v in selected.items()}
        if measured and rank == 0:
            pre.update(pre_update_state=True, layer=args.layer,
                       batch={k: v.cpu() for k, v in data.batch(indices, 'cpu').items()},
                       batch_indices=indices, data_cursor=step * sequences,
                       raw_gradients=raw, updates={k: v.cpu() for k, v in updates.items()},
                       parameter_order=list(selected),
                       gradient_precision=args.precision, valid_tokens=int(count))
            torch.save(pre, output / f'pre_step_{step:06d}.pt')
        train_loss = dd.sum_(loss_sum) / count
        metrics = dict(step=step, train_ce=float(train_loss), lr=lr, valid_tokens=int(count),
            total_tokens=(step+1)*args.batch_tokens, grad_norm=float(gradnorm), clipped=bool(gradnorm > 1),
            tokens_per_second=int(count) / (time.perf_counter()-begin),
            blocks={k: dict(parameter_norm=float(v.detach().norm()), gradient_norm=float(raw[k].norm()),
                            update_norm=float(updates[k].norm())) for k, v in selected.items()})
        if shampoo:
            update2 = torch.stack([(v.detach()-before[k]).norm().square() for k, v in observed.items()]).sum()
            param2 = torch.stack([v.norm().square() for v in before.values()]).sum()
            metrics.update(completed_steps=step+1, update_parameter_ratio=float((update2/param2).sqrt()),
                           root_last_refresh_step=optimizer.primary.root_last_refresh_step,
                           root_accuracy=max(max(x) for x in optimizer.primary.root_accuracy.values()),
                           optimizer_seconds=optim_seconds,
                           cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                                      if device.type == 'cuda' else None))
            if not all(math.isfinite(metrics[k]) for k in ('train_ce', 'update_parameter_ratio', 'root_accuracy')):
                raise FloatingPointError('nonfinite Shampoo loss, update or root error')
            del before
        if args.optimizer in ('muon', 'soap'):
            metrics['aux_lr'] = optimizer.auxiliary.param_groups[0]['lr']
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
            if not shampoo and ((step+1) % args.save_every == 0 or step == args.steps-1):
                torch.save(checkpoint(step+1), output / 'latest.pt.tmp')
                (output / 'latest.pt.tmp').replace(output / 'latest.pt')
        if shampoo and ((step+1) in measure_steps or (step+1) % args.save_every == 0 or step+1 == args.steps):
            shampoo_checkpoint(step+1)
    dd.close()


def atomic_save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)

if __name__ == '__main__':
    main()
