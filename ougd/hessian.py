"""Fixed CE Hessian with exact token weighting, streamed microbatches, and rank-0 RPC."""
import numpy as np
import torch
import torch.distributed as dist
from .distributed import sum_
from .model import BLOCKS, ce_sum

class Hessian:
    def __init__(self, model, batches, total_tokens, layer):
        self.model, self.batches, self.total_tokens = model, batches, total_tokens
        self.params = model.selected(layer)
        chosen = {id(p) for p in self.params.values()}
        for p in model.parameters():
            p.requires_grad_(id(p) in chosen)
        self.sizes = {k: p.numel() for k, p in self.params.items()}
        self.slices = {}
        pos = 0
        for k, n in self.sizes.items():
            self.slices[k] = slice(pos, pos+n)
            pos += n
        self.n = pos
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.calls = 0
        self.model.eval()
        if total_tokens <= 0:
            raise ValueError('no valid prediction tokens')

    def local(self, vector):
        result = torch.zeros_like(vector)
        pp = tuple(self.params.values())
        for batch in self.batches:
            loss = ce_sum(self.model, batch) / self.total_tokens
            g = torch.autograd.grad(loss, pp, create_graph=True)
            dot = sum((a * vector[self.slices[k]].reshape_as(a)).sum() for k, a in zip(BLOCKS, g))
            h = torch.autograd.grad(dot, pp)
            result += torch.cat([a.detach().flatten() for a in h])
        return sum_(result)

    def gradient(self):
        pp = tuple(self.params.values())
        result = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        for batch in self.batches:
            g = torch.autograd.grad(ce_sum(self.model, batch) / self.total_tokens, pp)
            result += torch.cat([v.detach().flatten() for v in g])
        return sum_(result).cpu().numpy()

    def local_component(self, vector, code):
        if code != 1:
            raise ValueError("unknown operator")
        return self.local(vector)

    def __call__(self, vector):
        return self.request(vector, 1)

    def request(self, vector, code):
        """Only rank 0 calls this; other ranks serve()."""
        v = torch.as_tensor(np.array(vector, copy=True), device=self.device, dtype=self.dtype)
        if dist.is_initialized():
            dist.broadcast(torch.tensor(code, device=self.device), 0)
            dist.broadcast(v, 0)
        self.calls += 1
        return self.local_component(v, code).cpu().numpy()

    def serve(self):
        while True:
            command = torch.zeros((), dtype=torch.long, device=self.device)
            dist.broadcast(command, 0)
            if command.item() == 0:
                return
            v = torch.empty(self.n, device=self.device, dtype=self.dtype)
            dist.broadcast(v, 0)
            self.local_component(v, command.item())

    def stop(self):
        if dist.is_initialized():
            dist.broadcast(torch.tensor(0, device=self.device), 0)

    def block(self, i, j, v):
        full = np.zeros(self.n, dtype=np.float64 if self.dtype == torch.float64 else np.float32)
        full[self.slices[j]] = v
        return self(full)[self.slices[i]]

    def joint(self, subset, v, self_only=False, drop=None):
        full = np.zeros(self.n, dtype=v.dtype)
        offset = 0
        for k in subset:
            n = self.sizes[k]
            full[self.slices[k]] = v[offset:offset+n]
            offset += n
        if self_only:
            return np.concatenate([self.block(k, k, full[self.slices[k]]) for k in subset])
        result = self(full)
        if drop:
            i, j = drop
            result[self.slices[i]] -= self.block(i, j, full[self.slices[j]])
            result[self.slices[j]] -= self.block(j, i, full[self.slices[i]])
        return np.concatenate([result[self.slices[k]] for k in subset])
