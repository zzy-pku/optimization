"""Common-gradient calibration and matrix-free Shampoo actions."""
import torch

from .data import make_batches
from .model import ce_sum
from .seven_measure import batch_hash
from .shampoo import DensePowers, LowRankPowers, relative_norm


def calibration_history(model, partition, data, records, beta2, microbatch, precision, device):
    """Store each weighted gradient directly as a column block of the global Z."""
    width = partition.columns
    members = partition.blocks['G']['global']
    window = len(records)
    z = torch.empty((members[-1].row_stop, width * window), device=device)
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        for index, record in enumerate(records):
            indices = record['indices']
            batch = data.batch(indices, 'cpu')
            if batch_hash(batch) != record['token_hash']:
                raise ValueError('calibration batch does not match the checkpoint token hash')
            tokens = int((batch['labels'] != -100).sum())
            if not tokens:
                raise ValueError('empty calibration batch')
            model.zero_grad(set_to_none=True)
            for micro in make_batches(data, indices, microbatch, device):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                    loss = ce_sum(model, micro) / tokens
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            weight = (1 - beta2) * beta2 ** (window - 1 - index) / (1 - beta2 ** window)
            for entry in members:
                grad = partition.parameters[entry.name].grad
                z[entry.row_start:entry.row_stop, index*width:(index+1)*width].copy_(
                    (grad.T if entry.transpose else grad) * weight ** 0.5)
    model.zero_grad(set_to_none=True)
    return z


class CalibratedOperator:
    def __init__(self, partition, scheme, z, ridge, square_root):
        self.partition = partition
        self.groups = partition.blocks[scheme]
        self.roots = {}
        self.right_statistics = {}
        self.accuracy = {}
        self.global_left = None
        for label, entries in self.groups.items():
            rows = z[entries[0].row_start:entries[-1].row_stop]
            right_inputs = rows.reshape(-1, partition.columns)
            right = right_inputs.T @ right_inputs
            self.right_statistics[label] = right
            b = DensePowers(right, ridge)
            if scheme == 'G':
                self.global_left = LowRankPowers(z, ridge)
                a = None
                generator = torch.Generator(device=z.device).manual_seed(41)
                test = torch.randn((z.shape[0], 4), device=z.device, generator=generator, dtype=z.dtype)
                half = self.global_left(self.global_left(test, 0.125), 0.125)
                full = self.global_left(test, 0.25)
                self.accuracy[label] = dict(
                    left_composition_error=relative_norm(half - full, full),
                    left_basis_orthogonality_error=self.global_left.orthogonality_error,
                    left_rank=(0 if self.global_left.vectors is None else self.global_left.vectors.shape[1]),
                    discarded_gram_eigenvalues=self.global_left.discarded_count,
                    discarded_max_gram_eigenvalue=self.global_left.discarded_max,
                    discarded_relative_gain_bound=self.global_left.discarded_max / (4 * ridge))
            else:
                a = DensePowers(rows @ rows.T, ridge)
                self.accuracy[label] = {}
            for power in ((0.25, 0.125) if square_root else (0.25,)):
                left = a.matrix(power) if a is not None else None
                right_root = b.matrix(power)
                self.roots[label, power] = left, right_root
                self.accuracy[label][str(power)] = dict(
                    left=a.action_error(left, power) if a is not None else None,
                    right=b.action_error(right_root, power))

    def apply(self, vector, power=0.25):
        if vector.ndim == 2:
            return torch.stack([self.apply(x, power) for x in vector.T], dim=1)
        if vector.shape != (self.partition.n,):
            raise ValueError('expected a vector in original parameter coordinates')
        output = torch.empty_like(vector)
        for label, entries in self.groups.items():
            left, right = self.roots[label, power]
            x = self.partition.pack_vector(entries, vector)
            y = self.global_left(x, power) if left is None else left @ x
            y = y @ right
            for entry, (_, member) in zip(entries, self.partition.unpack(entries, y)):
                output[entry.start:entry.stop].copy_(member.reshape(-1))
        return output


def bias_corrected_momentum(checkpoint, partition, device):
    state = checkpoint['optimizer']['primary']
    t = checkpoint['completed_steps']
    if state['step'] != t:
        raise ValueError('native momentum bias correction counter does not match checkpoint')
    if t == 0:
        return None
    factor = 1 - checkpoint['args']['beta1'] ** t
    return torch.cat([state['momentum'][name].flatten() for name in partition.parameters]).to(device).div_(factor)


def native_direction(checkpoint, partition, momentum):
    state = checkpoint['optimizer']['primary']
    result = torch.empty_like(momentum)
    for label, entries in partition.blocks[state['scheme']].items():
        left, right = (root.to(momentum.device) for root in state['roots'][label])
        y = left @ partition.pack_vector(entries, momentum) @ right
        for entry, (_, member) in zip(entries, partition.unpack(entries, y)):
            result[entry.start:entry.stop].copy_(member.reshape(-1))
    return result
