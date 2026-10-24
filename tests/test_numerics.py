import itertools
import unittest
import numpy as np
import torch
from torch.func import functional_call
from ougd.model import Transformer, ModelConfig, BLOCKS, ce_sum
from ougd.hessian import Hessian
from ougd.solvers import spectrum, cross_svd

def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = Transformer(ModelConfig(vocab_size=9,dim=4,hidden_dim=6,layers=2,heads=1,kv_heads=1,context=3)).double()
    batch = dict(input_ids=torch.tensor([[0,1,2],[3,4,5]]),labels=torch.tensor([[1,2,3],[4,5,-100]]))
    return model, batch

class Numerics(unittest.TestCase):
    def test_dense_hvp_and_fifteen_sets(self):
        model,b = fixture()
        h = Hessian(model,[b],5,0)
        selected = model.selected(0)
        names = {id(p):name for name,p in model.named_parameters()}
        # Independent dense autograd truth in a small coordinate window, never production.
        windows = [torch.arange(h.slices[k].start,h.slices[k].start+2) for k in BLOCKS]
        coords = torch.cat(windows)
        original = torch.cat([v.flatten() for v in selected.values()]).detach()
        def loss(window):
            v = original.scatter(0,coords,window)
            state = {names[id(p)]:v[h.slices[k]].reshape_as(p) for k,p in selected.items()}
            logits = functional_call(model,state,(b['input_ids'],))
            return torch.nn.functional.cross_entropy(logits.reshape(-1,9),b['labels'].flatten(),ignore_index=-100)
        dense = torch.autograd.functional.hessian(loss,original[coords])
        actual = []
        for coord in coords:
            v = np.zeros(h.n)
            v[coord] = 1
            actual.append(h(v)[coords])
        np.testing.assert_allclose(np.column_stack(actual),dense.numpy(),atol=1e-11,rtol=1e-8)
        np.testing.assert_allclose(dense,dense.T,atol=1e-11)
        for n in range(1,5):
            for s in itertools.combinations(range(4),n):
                idx = np.concatenate([np.arange(2*i,2*i+2) for i in s])
                a = dense.numpy()[np.ix_(idx,idx)]
                truth = np.linalg.eigvalsh(a)
                for end,target in [('LA',truth[-1]),('SA',truth[0])]:
                    result = spectrum(lambda v:a@v,len(a),1,end,1e-8,200,dtype=np.float64)
                    self.assertTrue(result['converged'])
                    self.assertAlmostEqual(result['values'][0],target,places=9)
                q = np.linalg.eigh(a)[1][:,-1]
                terms = sum(q[2*i:2*i+2]@a[2*i:2*i+2,2*i:2*i+2]@q[2*i:2*i+2] for i in range(n))
                terms += sum(2*q[2*i:2*i+2]@a[2*i:2*i+2,2*j:2*j+2]@q[2*j:2*j+2] for i,j in itertools.combinations(range(n),2))
                self.assertAlmostEqual(terms,truth[-1],places=10)
        for i,j in itertools.combinations(range(4),2):
            a = dense.numpy()[2*i:2*i+2,2*j:2*j+2]
            r = cross_svd(lambda v:a@v,lambda v:a.T@v,2,2,1,1e-8,200,4,np.float64)
            self.assertTrue(r['converged'])
            self.assertAlmostEqual(r['values'][0],np.linalg.svd(a,compute_uv=False)[0],places=9)

    def test_token_aggregation_and_unequal_microbatches(self):
        model,b = fixture()
        full = Hessian(model,[b],5,0)
        rng = np.random.default_rng(4)
        v = rng.normal(size=full.n)
        expected = full(v)
        micros = [{k:t[i:i+1] for k,t in b.items()} for i in range(2)]
        np.testing.assert_allclose(Hessian(model,micros,5,0)(v),expected,atol=1e-11)
        total = np.zeros(full.n)
        for seq in range(2):
            for pos in range(3):
                if b['labels'][seq,pos] == -100: continue
                token = {k:t[seq:seq+1].clone() for k,t in b.items()}
                token['labels'].fill_(-100)
                token['labels'][0,pos] = b['labels'][seq,pos]
                total += Hessian(model,[token],1,0)(v)/5
        np.testing.assert_allclose(total,expected,atol=1e-11)
        w = rng.normal(size=full.n)
        self.assertAlmostEqual(float(w@full(v)),float(v@full(w)),places=10)

    def test_dilation_signed_spectrum(self):
        a = np.array([[3.,1.,0.],[0.,-2.,1.]])
        r = cross_svd(lambda v:a@v,lambda v:a.T@v,2,3,2,1e-9,200,7,np.float64)
        np.testing.assert_allclose(r['values'],np.linalg.svd(a,compute_uv=False),atol=1e-10)
        self.assertTrue(r['converged'])

if __name__=='__main__': unittest.main()
