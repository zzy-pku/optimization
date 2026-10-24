"""torchrun --standalone --nproc-per-node=2 -m tests.distributed_components"""
from unittest.mock import patch
import numpy as np
import torch
from ougd import distributed as dd
from ougd.components import Decomposition, CODES
from tests.test_numerics import fixture


def main():
    device,rank,world=dd.init()
    for scope in ('TB','ST'):
        model,batch=fixture()
        model.to(device)
        batch={k:v.to(device) for k,v in batch.items()}
        if scope=='ST':
            batch={k:v[:1].clone() for k,v in batch.items()}
            batch['labels'].fill_(-100)
            batch['labels'][0,1]=2
        count=int((batch['labels']!=-100).sum())
        local=[{k:v[rank::world] for k,v in batch.items()}] if rank<len(batch['labels']) else []
        local=[b for b in local if b['labels'].numel()]
        backend=Decomposition(model,local,count,0)
        if rank:
            backend.serve()
        else:
            try:
                vector=np.random.default_rng(42).normal(size=backend.n)
                for name,code in CODES.items():
                    observed=backend.view(name)(vector)
                    serial=Decomposition(model,[batch],count,0)
                    # Workers wait for the next RPC while rank 0 computes a serial oracle.
                    with patch('ougd.components.sum_',lambda x:x),patch('ougd.hessian.sum_',lambda x:x):
                        truth=serial.local_component(torch.tensor(vector,device=device),code).cpu().numpy()
                    np.testing.assert_allclose(observed,truth,atol=1e-11)
                print(f'PASS: {scope} distributed H/A/R/R_direct equals serial',flush=True)
            finally: backend.stop()
    dd.close()

if __name__=='__main__': main()
