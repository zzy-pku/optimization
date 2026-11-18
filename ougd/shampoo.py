"""EMA-Shampoo in the fixed G/M/U Transformer matrix layout."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


@dataclass(frozen=True)
class Matrix:
    name: str
    shape: tuple
    start: int
    stop: int
    row_start: int
    row_stop: int
    transpose: bool


class Partition:
    """Original row-major coordinates and the matrix view used for Shampoo."""

    def __init__(self, model):
        self.parameters = {}
        self.blocks = {name: {} for name in ('G', 'M', 'U')}
        position = row = 0
        for layer in range(model.config.layers):
            scope = f'L{layer + 1:02d}'
            matrices = {}
            for key, parameter in model.selected_seven(layer).items():
                path = f'attn.{key.lower() if key in "QKV" else key}' if key in 'QKVO' else key
                name = f'blocks.{layer}.{path}.weight'
                rows, columns = parameter.T.shape if key == 'D' else parameter.shape
                if columns != model.config.dim:
                    raise ValueError(f'{name} cannot be packed into {model.config.dim} columns')
                entry = Matrix(name, tuple(parameter.shape), position,
                               position + parameter.numel(), row, row + rows, key == 'D')
                matrices[key] = entry
                self.parameters[name] = parameter
                self.blocks['M'][f'{scope}.{key}'] = (entry,)
                position, row = entry.stop, entry.row_stop
            for group, members in {'QK': 'QK', 'VO': 'VO', 'FFN': 'UGD'}.items():
                self.blocks['U'][f'{scope}.{group}'] = tuple(matrices[key] for key in members)
        self.blocks['G']['global'] = tuple(block[0] for block in self.blocks['M'].values())
        self.n = position
        self.columns = model.config.dim
        self.manifest = {}
        for scheme, groups in self.blocks.items():
            self.manifest[scheme] = {}
            for label, entries in groups.items():
                start = entries[0].row_start
                self.manifest[scheme][label] = {
                    'shape': [entries[-1].row_stop - start, self.columns],
                    'members': [dict(asdict(e), packed_rows=[e.row_start-start, e.row_stop-start])
                                for e in entries],
                }
        # JSON normalization gives the same manifest in tensor checkpoints and JSON reports.
        encoded = json.dumps(self.manifest, sort_keys=True, separators=(',', ':'))
        self.manifest = json.loads(encoded)
        self.hash = hashlib.sha256(encoded.encode()).hexdigest()

    @staticmethod
    def interval(entries):
        return slice(entries[0].start, entries[-1].stop)

    @staticmethod
    def pack(entries, tensors):
        return torch.cat([tensors[e.name].T if e.transpose else tensors[e.name]
                          for e in entries], dim=0)

    @staticmethod
    def unpack(entries, packed):
        start = entries[0].row_start
        for entry in entries:
            part = packed[entry.row_start-start:entry.row_stop-start]
            yield entry.name, part.T if entry.transpose else part

    def pack_vector(self, entries, vector):
        return self.pack(entries, {e.name: vector[e.start:e.stop].reshape(e.shape) for e in entries})


class DensePowers:
    """Symmetric powers of an unregularized PSD factor plus an absolute ridge."""

    def __init__(self, factor, ridge, correction=1.0):
        matrix = factor.to(torch.float64) / correction
        self.values, self.vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
        self.values = regularize(self.values, ridge, factor.dtype)
        self.dtype = factor.dtype

    def matrix(self, power, dtype=None):
        values = self.values.pow(-power)
        result = (self.vectors * values) @ self.vectors.T
        return ((result + result.T) * 0.5).to(dtype or self.dtype)

    def action_error(self, root, power):
        generator = torch.Generator(device=root.device).manual_seed(41)
        test = torch.randn((root.shape[0], 4), device=root.device, dtype=root.dtype,
                           generator=generator)
        projected = self.vectors.T @ test.double()
        reference = self.vectors @ (self.values.pow(-power)[:, None] * projected)
        return relative_norm((root @ test).double() - reference, reference)


def regularize(values, ridge, source_dtype):
    tolerance = 32 * torch.finfo(source_dtype).eps * float(values.abs().max())
    minimum = float(values.min())
    if not math.isfinite(minimum) or minimum < -tolerance:
        raise ValueError(f'non-PSD statistic: minimum {minimum:.4g}, tolerance {tolerance:.4g}')
    return values.clamp_min(0).add_(ridge)


def relative_norm(error, reference):
    return float(error.norm() / reference.norm().clamp_min(1e-30))


class LowRankPowers:
    """Powers of ridge * I + Z Z^T via a float64 thin Gram decomposition."""

    def __init__(self, z, ridge):
        self.ridge = ridge
        self.vectors = None
        z = z.double()
        gram = z.T @ z
        values, vectors = torch.linalg.eigh((gram + gram.T) * 0.5)
        regularize(values, ridge, torch.float64)
        # Resolve Gram eigenvalues above an absolute effect threshold on the inverse powers.
        # Below this threshold their relative change from the ridge-only gain is < 1e-8.
        cutoff = ridge * 4e-8
        keep = values > cutoff
        self.discarded_max = float(values[~keep].clamp_min(0).max()) if (~keep).any() else 0.
        self.discarded_count = int((~keep).sum())
        if keep.any():
            basis = z @ (vectors[:, keep] / values[keep].sqrt())
            check = basis.T @ basis
            check.diagonal().sub_(1)
            self.orthogonality_error = float(check.abs().sum(dim=1).max())
            if self.orthogonality_error > 1e-5:
                raise RuntimeError(f'global Gram basis lost orthogonality: {self.orthogonality_error:.3g}')
            self.vectors = basis
        else:
            self.orthogonality_error = 0.
        self.values = values[keep] + ridge

    def __call__(self, matrix, power):
        base_gain = self.ridge ** (-power)
        if self.vectors is None:
            return base_gain * matrix
        # Decompose into range and complement before scaling, including the ridge-only gain.
        # Float64 projection avoids loss of precision when these gains are very different.
        basis = self.vectors
        projection = basis.T @ matrix.double()
        result = base_gain * matrix.double()
        result.addmm_(basis, (self.values.pow(-power) - base_gain)[:, None] * projection)
        return result.to(matrix.dtype)


class Shampoo:
    """Training optimizer with persistent EMA state and refresh-time root caches."""

    def __init__(self, partition, scheme, lr, betas, ridge, weight_decay, refresh=10):
        if scheme not in ('M', 'U') or ridge <= 0 or refresh < 1:
            raise ValueError('invalid training Shampoo configuration')
        self.partition = partition
        self.scheme = scheme
        self.groups = partition.blocks[scheme]
        self.param_groups = [{'params': list(partition.parameters.values()), 'lr': lr}]
        self.beta1, self.beta2 = betas
        self.ridge, self.weight_decay, self.refresh = ridge, weight_decay, refresh
        self.steps = self.root_last_refresh_step = 0
        self.root_accuracy = {}
        self.momentum = {}
        self.factors = {}
        self.roots = {}

    def state_dict(self):
        return dict(scheme=self.scheme, partition_hash=self.partition.hash,
                    step=self.steps, root_last_refresh_step=self.root_last_refresh_step,
                    momentum=self.momentum, factors=self.factors, roots=self.roots,
                    root_accuracy=self.root_accuracy)

    def load_state_dict(self, state):
        if (state['partition_hash'], state['scheme']) != (self.partition.hash, self.scheme):
            raise ValueError('checkpoint Shampoo layout mismatch')
        device = self.param_groups[0]['params'][0].device
        self.steps = state['step']
        self.root_last_refresh_step = state['root_last_refresh_step']
        self.root_accuracy = state['root_accuracy']
        self.momentum = {k: x.to(device) for k, x in state['momentum'].items()}
        self.factors = {k: tuple(x.to(device) for x in pair) for k, pair in state['factors'].items()}
        self.roots = {k: tuple(x.to(device) for x in pair) for k, pair in state['roots'].items()}

    def zero_grad(self, set_to_none=True):
        for parameter in self.param_groups[0]['params']:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self):
        parameters = self.partition.parameters
        gradients = {k: p.grad for k, p in parameters.items()}
        if any(g is None for g in gradients.values()):
            raise ValueError('Shampoo requires gradients for all matrices in its layout')
        t = self.steps + 1
        lr = self.param_groups[0]['lr']
        refresh = t == 1 or t % self.refresh == 0
        for key, gradient in gradients.items():
            if key not in self.momentum:
                self.momentum[key] = torch.zeros_like(gradient)
            self.momentum[key].lerp_(gradient, 1 - self.beta1)
        # Update and refresh every group before changing any model parameter.
        for label, members in self.groups.items():
            x = self.partition.pack(members, gradients)
            if label not in self.factors:
                self.factors[label] = tuple(x.new_zeros((n, n)) for n in x.shape)
            left, right = self.factors[label]
            left.mul_(self.beta2).addmm_(x, x.T, alpha=1 - self.beta2)
            right.mul_(self.beta2).addmm_(x.T, x, alpha=1 - self.beta2)
            if refresh:
                pair = [DensePowers(f, self.ridge, 1 - self.beta2 ** t) for f in (left, right)]
                roots = tuple(f.matrix(0.25) for f in pair)
                self.root_accuracy[label] = [f.action_error(r, 0.25) for f, r in zip(pair, roots)]
                self.roots[label] = roots
                del pair
        for label, members in self.groups.items():
            left, right = self.roots[label]
            m = self.partition.pack(members, self.momentum) / (1 - self.beta1 ** t)
            for key, direction in self.partition.unpack(members, left @ m @ right):
                parameters[key].mul_(1 - lr * self.weight_decay).add_(direction, alpha=-lr)
        self.steps = t
        if refresh:
            self.root_last_refresh_step = t
