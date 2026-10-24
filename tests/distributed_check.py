"""Run with torchrun --standalone --nproc-per-node=2 -m tests.distributed_check."""
import numpy as np
import torch
from ougd import distributed as dd
from ougd.hessian import Hessian
from ougd.model import Transformer, ModelConfig

def main():
    device,rank,world = dd.init()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = Transformer(ModelConfig(vocab_size=9,dim=4,hidden_dim=6,layers=2,heads=1,kv_heads=1,context=3)).double().to(device)
    batch = dict(input_ids=torch.tensor([[0,1,2],[3,4,5]],device=device),
                 labels=torch.tensor([[1,2,3],[4,5,-100]],device=device))
    local = [{k:v[rank::world] for k,v in batch.items()}] if rank < 2 else []
    h = Hessian(model,local,5,0)
    g = h.gradient()
    if rank:
        h.serve()
    else:
        try:
            v = np.random.default_rng(2).normal(size=h.n)
            observed = h(v)
            # local() invokes a collective: independent serial truth uses autograd directly.
            from ougd.model import ce_sum
            pp = tuple(h.params.values())
            grad = torch.autograd.grad(ce_sum(model,batch)/5,pp,create_graph=True)
            direction = torch.tensor(v,device=device)
            dot = sum((a*direction[h.slices[k]].reshape_as(a)).sum() for k,a in zip(h.params,grad))
            truth = torch.cat([x.flatten() for x in torch.autograd.grad(dot,pp)]).detach().cpu().numpy()
            np.testing.assert_allclose(observed,truth,atol=1e-11)
            np.testing.assert_allclose(g,torch.cat([x.flatten() for x in grad]).detach().cpu().numpy(),atol=1e-11)
            print('PASS: distributed uneven-token HVP/gradient equals serial; rank-0 RPC completed')
        finally: h.stop()
    dd.close()

if __name__=='__main__': main()
