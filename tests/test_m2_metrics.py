import unittest
from types import SimpleNamespace
import numpy as np
import torch
from ougd.components import Decomposition
from ougd.m2 import shared_statistics, leakage
from ougd.model import BLOCKS
from ougd.solvers import spectrum
from tests.test_numerics import fixture

class Metrics(unittest.TestCase):
    def test_shared_probes_preserve_cancellation_identity(self):
        a = np.kron(np.arange(1,17,dtype=float).reshape(4,4),np.eye(2))
        a = a+a.T
        r = -.9*a+.2*np.eye(8)
        matrices = dict(A=a,R=r,H=a+r)
        backend = SimpleNamespace(model=SimpleNamespace(anchor_affine_blocks=()),n=8,sizes=dict.fromkeys(BLOCKS,2),
            slices={k:slice(2*i,2*i+2) for i,k in enumerate(BLOCKS)})
        backend.view = lambda name:lambda v:matrices[name]@v
        norms,paired=shared_statistics(backend,8,11)
        for name,m in matrices.items():
            for block,stats in norms[name]['blocks'].items():
                sl1,sl2=(backend.slices[k] for k in block)
                self.assertAlmostEqual(stats['norm2'],np.linalg.norm(m[sl1,sl2])**2,places=10)
        for row in paired.values():
            self.assertAlmostEqual(row['frobenius_identity_error'],0,places=9)
            self.assertGreater(row['ratio_A_H'],1)
            self.assertLess(row['cosine_AR'],0)

    def test_bilateral_leakage_has_expected_scale(self):
        q=np.array([[1.],[0.]])
        r=np.array([[2.,3.],[3.,0.]])
        out=leakage(lambda v:r@v,q)
        self.assertAlmostEqual(out['absolute'],3)
        self.assertAlmostEqual(out['relative'],3/np.sqrt(13))

    def test_last_block_and_matrix_free_extreme_eigenvalues(self):
        model,b=fixture()
        d=Decomposition(model,[b],5,1)
        z=model(b['input_ids'],anchor_layer=1)
        torch.testing.assert_close(model.from_anchor(z,1),model(b['input_ids']),atol=0,rtol=0)
        coords=np.concatenate([np.arange(d.slices[k].start,d.slices[k].start+2) for k in BLOCKS])
        for name in ('A','R'):
            op=d.view(name)
            def projected(v):
                full=np.zeros(d.n); full[coords]=v
                return op(full)[coords]
            dense=np.column_stack([projected(v) for v in np.eye(8)])
            truth=np.linalg.eigvalsh(dense)
            for end,expected in [('LA',truth[-1]),('SA',truth[0])]:
                spec=spectrum(projected,8,1,end,1e-8,200,dtype=np.float64)
                self.assertTrue(spec['converged'])
                self.assertAlmostEqual(spec['values'][0],expected,places=10)

if __name__=='__main__': unittest.main()
