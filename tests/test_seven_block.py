import unittest
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from ougd.model import Transformer, ModelConfig, SEVEN_BLOCKS, ce_sum
from ougd.components import Decomposition
from ougd.positive_subspaces import positive_spectrum, positive_count, geometry, block_random_baseline
from ougd.seven_frobenius import frobenius_statistics
from ougd.seven_measure import ritz_analysis, FOUR


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = Transformer(ModelConfig(vocab_size=9, dim=4, hidden_dim=6,
                                    layers=1, heads=1, kv_heads=1, context=3)).double()
    batch = dict(input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
                 labels=torch.tensor([[1, 2, 3], [4, 5, -100]]))
    return model, batch


class SevenBlockNumerics(unittest.TestCase):
    def test_exact_hvp_and_common_anchor_decomposition(self):
        model, batch = fixture()
        backend = Decomposition(model, [batch], 5, 0, blocks=SEVEN_BLOCKS)
        self.assertEqual(tuple(backend.params), SEVEN_BLOCKS)
        self.assertNotIn(id(model.embedding.weight), {id(p) for p in backend.params.values()})
        names = {id(p): name for name, p in model.named_parameters()}
        coords = torch.tensor([backend.slices[k].start+j for k in SEVEN_BLOCKS for j in range(2)])
        original = torch.cat([v.flatten() for v in backend.params.values()]).detach()
        def loss(window):
            full = original.scatter(0, coords, window)
            state = {names[id(p)]: full[backend.slices[k]].reshape_as(p) for k, p in backend.params.items()}
            logits = functional_call(model, state, (batch['input_ids'],))
            return F.cross_entropy(logits.reshape(-1, 9), batch['labels'].flatten(), ignore_index=-100)
        dense = torch.autograd.functional.hessian(loss, original[coords]).detach().numpy()
        columns = []
        for coord in coords:
            v = np.zeros(backend.n); v[coord] = 1
            columns.append(backend.request(v, 1)[coords])
        np.testing.assert_allclose(np.column_stack(columns), dense, rtol=1e-8, atol=1e-11)
        rng = np.random.default_rng(3)
        v = rng.normal(size=backend.n); v /= np.linalg.norm(v)
        h = backend.request(v, 1); a = backend.request(v, 2); r = backend.request(v, 4)
        np.testing.assert_allclose(h, a+r, rtol=1e-7, atol=1e-10)
        # Unequal token-weighted microbatches must reproduce the whole batch.
        micros = [{k: t[i:i+1] for k, t in batch.items()} for i in range(2)]
        streamed = Decomposition(model, micros, 5, 0, blocks=SEVEN_BLOCKS)
        np.testing.assert_allclose(streamed.request(v, 1), h, rtol=1e-8, atol=1e-11)

    def test_spectrum_and_nested_coverage(self):
        diag = np.array([4., 3., 2., 1., -.5, -1.])
        spec = positive_spectrum(lambda x: diag*x, len(diag), 4, 1, 1e-10, 1e-8)
        self.assertEqual(positive_count(spec), 4)
        np.testing.assert_allclose(spec['values'], [4., 3., 2., 1.], atol=1e-10)
        short = positive_spectrum(lambda x: np.array([2., -1., -2., -3.])*x, 4, 2, 4, 1e-10, 1e-8)
        self.assertEqual(positive_count(short), 1)
        q = np.eye(5)[:, :2]
        g1 = geometry({'x': np.array([[1., 0.]])}, 2, 1)
        g2 = geometry({'x': np.eye(2)}, 2, 2)
        self.assertAlmostEqual(g1['H_to_candidate_containment'], .5)
        self.assertEqual(g1['unavoidable_unrepresented_target_dimensions'], 1)
        self.assertAlmostEqual(g2['H_to_candidate_containment'], 1.)
        self.assertEqual(len(g2['principal_angles_deg']), 2)

    def test_haar_baseline_expectation_and_seed(self):
        rng = np.random.default_rng(5)
        q = np.linalg.qr(rng.normal(size=(30, 3)), mode='reduced')[0]
        slices = {'a': slice(0, 9), 'b': slice(9, 30)}
        first = block_random_baseline(q, slices, {'a': 2, 'b': 3}, 1500, 4)
        second = block_random_baseline(q, slices, {'a': 2, 'b': 3}, 1500, 4)
        self.assertEqual(first['samples'], second['samples'])
        self.assertLess(abs(first['mean'] - first['expected']), .015)

    def test_paired_frobenius_matches_exact_block_norms(self):
        keys = SEVEN_BLOCKS
        class DenseBackend:
            def __init__(self):
                self.params = {k: None for k in keys}
                self.sizes = {k: 2 for k in keys}
                self.slices = {k: slice(2*i, 2*i+2) for i, k in enumerate(keys)}
                self.n = 2*len(keys)
                coeff = np.arange(1., 50.).reshape(7, 7); coeff = (coeff+coeff.T)/2
                self.H = np.kron(coeff, np.eye(2))
                self.A = np.kron(coeff*.3, np.eye(2))
            def request(self, v, code):
                return (self.H if code == 1 else self.A) @ v
        backend = DenseBackend()
        result = frobenius_statistics(backend, probes=4, bootstrap=30, seed=2)
        self.assertEqual(len(result['blocks']), 28)
        for pair, row in result['blocks'].items():
            mat = backend.H[backend.slices[pair[0]], backend.slices[pair[1]]]
            self.assertAlmostEqual(row['H']['norm2'], float(np.sum(mat*mat)), places=9)
            self.assertAlmostEqual(row['paired_frobenius_identity_error'], 0., places=9)
        self.assertAlmostEqual(result['scopes']['seven']['H']['norm2'], float(np.sum(backend.H**2)), places=8)

    def test_four_candidate_residual_uses_actual_target_scope(self):
        class Backend:
            sizes = {key: 1 for key in SEVEN_BLOCKS}
            slices = {key: slice(i,i+1) for i,key in enumerate(SEVEN_BLOCKS)}
        backend=Backend(); K=np.eye(7)
        K[0,3]=K[3,0]=2.  # Q/O coupling vanishes when restricted to H4.
        bases={key:np.ones((1,1)) for key in SEVEN_BLOCKS}
        offsets={key:slice(i,i+1) for i,key in enumerate(SEVEN_BLOCKS)}
        target={'values':np.array([1.])}
        within=ritz_analysis(K,K,bases,offsets,backend,FOUR,FOUR,target,1)
        global_scope=ritz_analysis(K,K,bases,offsets,backend,FOUR,SEVEN_BLOCKS,target,1)
        # The first eigenvector of I4 may be chosen as any basis direction; test all by
        # putting coupling into every four-block column.
        K[0,3:]=2.;K[3:,0]=2.
        global_scope=ritz_analysis(K,K,bases,offsets,backend,FOUR,SEVEN_BLOCKS,target,1)
        self.assertAlmostEqual(within['full_scope_relative_residual'][0],0.,places=12)
        self.assertGreater(global_scope['full_scope_relative_residual'][0],.1)


if __name__ == '__main__':
    unittest.main()
