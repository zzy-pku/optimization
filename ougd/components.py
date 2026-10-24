"""M2 exact common-anchor decomposition. All operations are matrix-free."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp
from .distributed import sum_
from .hessian import Hessian
from .model import BLOCKS

CODES = {'H': 1, 'A': 2, 'R': 3, 'R_direct': 4}

class Decomposition(Hessian):
    def __init__(self, model, batches, total_tokens, layer):
        super().__init__(model, batches, total_tokens, layer)
        self.layer = layer
        names = {id(p): name for name, p in model.named_parameters()}
        self.names = [names[id(p)] for p in self.params.values()]

    def anchor(self, batch, parameters):
        replacements = dict(zip(self.names, parameters))
        return functional_call(self.model, replacements, (batch['input_ids'],),
                               {'anchor_layer': self.layer})

    def tail_loss(self, z, batch):
        logits = self.model.from_anchor(z, self.layer)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch['labels'].flatten(),
                               reduction='sum', ignore_index=-100) / self.total_tokens

    def activation_product(self, vector, direct_residual=False):
        result = torch.zeros_like(vector)
        pp = tuple(self.params.values())
        directions = tuple(vector[self.slices[k]].reshape_as(p) for k,p in self.params.items())
        for batch in self.batches:
            if direct_residual:
                # q is frozen: differentiate only the anchor's own nonlinearity.
                z = self.model(batch['input_ids'], anchor_layer=self.layer)
                leaf = z.detach().requires_grad_(True)
                q = torch.autograd.grad(self.tail_loss(leaf,batch), leaf)[0].detach()
                psi = (q*z).sum()
                g = torch.autograd.grad(psi, pp, create_graph=True)
                dot = sum((a*v).sum() for a,v in zip(g,directions))
                out = torch.autograd.grad(dot, pp)
            else:
                # Forward AD for Jv, reverse AD for B(Jv), then VJP for J^T.
                # Jv and B(Jv) must be detached before the next derivative.
                with torch.no_grad():
                    _, tangent = jvp(lambda *weights:self.anchor(batch,weights), pp, directions)
                tangent = tangent.detach()
                z = self.model(batch['input_ids'], anchor_layer=self.layer)
                leaf = z.detach().requires_grad_(True)
                q = torch.autograd.grad(self.tail_loss(leaf,batch),leaf,create_graph=True)[0]
                b = torch.autograd.grad((q*tangent).sum(),leaf)[0].detach()
                out = torch.autograd.grad(z, pp, grad_outputs=b)
            result += torch.cat([x.detach().flatten() for x in out])
        return sum_(result)

    def local_component(self, vector, code):
        if code == CODES['H']:
            return self.local(vector)
        if code == CODES['A']:
            return self.activation_product(vector)
        if code == CODES['R']:
            return self.local(vector) - self.activation_product(vector)
        if code == CODES['R_direct']:
            return self.activation_product(vector,direct_residual=True)
        raise ValueError('unknown component request')

    def view(self, operator):
        return ComponentView(self, operator)

class ComponentView(Hessian):
    """Reuse block/joint interfaces while routing RPC to a specified component."""
    def __init__(self, backend, operator):
        if operator not in CODES: raise ValueError(operator)
        self.__dict__.update(backend.__dict__)
        self.backend, self.operator, self.calls = backend, operator, 0

    def is_zero(self, subset, self_only=False):
        return self.operator == 'R' and all(k in self.model.anchor_affine_blocks for k in subset) and (self_only or len(subset)==1)

    def block(self, i, j, v):
        if i == j and self.is_zero((i,)):
            return np.zeros_like(v)
        return super().block(i,j,v)

    def joint(self, subset, v, self_only=False, drop=None):
        if self.is_zero(subset,self_only):
            return np.zeros_like(v)
        return super().joint(subset,v,self_only,drop)

    def __call__(self, vector):
        self.calls += 1
        result = self.backend.request(vector, CODES[self.operator])
        if self.operator == 'R':
            # Enforce analytically zero self blocks rather than report H-A roundoff.
            for name in self.model.anchor_affine_blocks:
                sl = self.slices[name]
                if not np.any(vector[:sl.start]) and not np.any(vector[sl.stop:]):
                    result[sl] = 0.
        return result
