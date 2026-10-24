import unittest
import numpy as np
import torch
from ougd.components import Decomposition
from tests.test_numerics import fixture

class Components(unittest.TestCase):
    def test_dense_anchor_decomposition_and_paths(self):
        model,batch = fixture()
        d = Decomposition(model,[batch],5,0)
        z = model(batch['input_ids'],anchor_layer=0)
        torch.testing.assert_close(model.from_anchor(z,0),model(batch['input_ids']),atol=0,rtol=0)
        # Explicit dense J and B only inside the tiny validation test.
        pp = tuple(d.params.values())
        coords = np.concatenate([np.arange(d.slices[k].start,d.slices[k].start+2) for k in d.params])
        full = torch.cat([p.flatten() for p in pp]).detach()
        indices = torch.tensor(coords)
        def anchor(window):
            vec = full.scatter(0,indices,window)
            weights = tuple(vec[d.slices[k]].reshape_as(p) for k,p in d.params.items())
            return d.anchor(batch,weights)
        window = full[indices]
        jac = torch.autograd.functional.jacobian(anchor,window).flatten(0,2)
        b = torch.autograd.functional.hessian(lambda x:d.tail_loss(x.reshape_as(z),batch),z.detach().flatten())
        truth_a = jac.T@b@jac
        q = torch.autograd.grad(d.tail_loss(z,batch),z)[0].detach()
        truth_r = torch.autograd.functional.hessian(lambda x:(q*anchor(x)).sum(),window)
        observed = {'H':[], 'A':[], 'R':[], 'R_direct':[]}
        for idx in coords:
            v = np.zeros(d.n)
            v[idx] = 1
            for name in observed:
                observed[name].append(d.view(name)(v)[coords])
        mats = {k:np.column_stack(v) for k,v in observed.items()}
        np.testing.assert_allclose(mats['A'],truth_a.detach().numpy(),atol=1e-11,rtol=1e-8)
        np.testing.assert_allclose(mats['R_direct'],truth_r.numpy(),atol=1e-11,rtol=1e-8)
        np.testing.assert_allclose(mats['H'],mats['A']+mats['R_direct'],atol=1e-11)
        np.testing.assert_allclose(mats['R'],mats['R_direct'],atol=1e-11)
        for m in mats.values(): np.testing.assert_allclose(m,m.T,atol=1e-11)
        self.assertGreater(np.linalg.norm(mats['A']),0)
        self.assertGreater(np.linalg.norm(mats['R']),0)

    def test_affine_anchor_residual_self_blocks_are_zero(self):
        model,batch=fixture()
        d=Decomposition(model,[batch],5,0)
        rng=np.random.default_rng(10)
        for name in model.anchor_affine_blocks:
            v=np.zeros(d.n)
            v[d.slices[name]]=rng.normal(size=d.sizes[name])
            direct=d.view('R_direct')(v)[d.slices[name]]
            np.testing.assert_allclose(direct,0,atol=1e-12)
            np.testing.assert_array_equal(d.view('R').joint((name,),v[d.slices[name]]),0)

    def test_unequal_microbatches_and_single_token_aggregation(self):
        model,b = fixture()
        full = Decomposition(model,[b],5,0)
        v = np.random.default_rng(41).normal(size=full.n)
        pieces = [{k:t[i:i+1] for k,t in b.items()} for i in range(2)]
        split = Decomposition(model,pieces,5,0)
        for component in ('A','R','R_direct'):
            expected = full.view(component)(v)
            np.testing.assert_allclose(split.view(component)(v),expected,atol=1e-11)
            total = np.zeros(full.n)
            for seq,pos in ((0,0),(0,1),(0,2),(1,0),(1,1)):
                token = {k:t[seq:seq+1].clone() for k,t in b.items()}
                token['labels'].fill_(-100)
                token['labels'][0,pos] = b['labels'][seq,pos]
                total += Decomposition(model,[token],1,0).view(component)(v)/5
            np.testing.assert_allclose(total,expected,atol=1e-11)

if __name__=='__main__': unittest.main()
