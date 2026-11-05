import unittest
import numpy as np
from ougd.model import SEVEN_BLOCKS
from ougd.positive_subspaces import positive_spectrum
from ougd.seven_controls import SignedPermutationCross, adaptive_diagonal_top, rotated_cross_control


class DenseBackend:
    def __init__(self, matrix, block_size):
        self.params = {key:None for key in SEVEN_BLOCKS}
        self.sizes = {key:block_size for key in SEVEN_BLOCKS}
        self.slices = {key:slice(i*block_size,(i+1)*block_size) for i,key in enumerate(SEVEN_BLOCKS)}
        self.n = len(SEVEN_BLOCKS)*block_size
        self.matrix = np.asarray(matrix); self.calls = 0
    def request(self, v, code):
        assert code == 1
        self.calls += 1
        return self.matrix @ v
    def block(self, i, j, v):
        self.calls += 1
        return self.matrix[self.slices[i],self.slices[j]] @ v


class OptionalControls(unittest.TestCase):
    def test_signed_permutation_preserves_self_and_cross_norms(self):
        rng=np.random.default_rng(3); a=rng.normal(size=(21,21)); h=(a+a.T)/2
        backend=DenseBackend(h,3); control=SignedPermutationCross(backend,42)
        eye=np.eye(backend.n)
        rotation=np.column_stack([control.rotate(v) for v in eye.T])
        np.testing.assert_allclose(rotation.T@rotation,eye,atol=1e-12)
        diag=np.zeros_like(h)
        for k in SEVEN_BLOCKS: diag[backend.slices[k],backend.slices[k]]=h[backend.slices[k],backend.slices[k]]
        expected=diag+rotation@(h-diag)@rotation.T
        actual=np.column_stack([control(v) for v in eye.T])
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        np.testing.assert_allclose(actual,actual.T,atol=1e-12)
        for i in SEVEN_BLOCKS:
            np.testing.assert_allclose(actual[backend.slices[i],backend.slices[i]],h[backend.slices[i],backend.slices[i]])
            for j in SEVEN_BLOCKS:
                np.testing.assert_allclose(np.linalg.norm(actual[backend.slices[i],backend.slices[j]],'fro'),
                                           np.linalg.norm(h[backend.slices[i],backend.slices[j]],'fro'),atol=1e-12)
        bases7={k:np.ones((3,1))/np.sqrt(3) for k in SEVEN_BLOCKS}
        bases4={k:bases7[k] for k in 'OUGD'}
        out=rotated_cross_control(backend,bases4,bases7,{'values':np.array([5.])},1,1,7,1e-9,1e-8,500)
        self.assertEqual(len(out['replicates']),1)
        self.assertGreater(out['replicates'][0]['underlying_H_requests'],out['replicates'][0]['matvecs'])

    def test_diagonal_top_refines_only_needed_blocks(self):
        blocksize=25; diagonal=[]
        for index in range(7):
            diagonal.extend((20.-.01*np.arange(blocksize)) if index==0 else (1.-.001*np.arange(blocksize)))
        backend=DenseBackend(np.diag(diagonal),blocksize)
        specs={k:positive_spectrum(lambda v,k=k:backend.block(k,k,v),blocksize,3,10+i,1e-10,1e-8)
               for i,k in enumerate(SEVEN_BLOCKS)}
        target=np.eye(backend.n)[:,:2]
        result=adaptive_diagonal_top(backend,specs,target,7,11,1e-10,1e-8,500)
        self.assertEqual(result['status'],'complete')
        self.assertEqual({row['block'] for row in result['refinements']},{'Q'})
        self.assertEqual({row['block'] for row in result['sources']},{'Q'})
        self.assertAlmostEqual(result['geometry']['H_to_candidate_containment'],1.,places=8)


if __name__=='__main__': unittest.main()
