import numpy as np
import torch
from ougd.hessian import Hessian
from ougd.model import ModelConfig, Transformer
from ougd.twelve_measure import groups, block_metrics, save_npz


def test_selected_hessian_is_principal_submatrix():
    torch.manual_seed(1)
    torch.set_num_threads(1)
    model = Transformer(ModelConfig(vocab_size=8, dim=4, hidden_dim=6, layers=2,
                                    heads=2, kv_heads=1, context=3)).double()
    b = {'input_ids': torch.tensor([[1, 2, 3]]), 'labels': torch.tensor([[2, 3, 4]])}
    params = groups(model)
    keys = ['L01.O', 'L02.V']
    full = Hessian(model, [b], 3, 0, parameters=params)
    rng = np.random.default_rng(4)
    v = rng.normal(size=sum(params[k].numel() for k in keys))
    embedded = np.zeros(full.n)
    pos = 0
    for key in keys:
        n = params[key].numel(); embedded[full.slices[key]] = v[pos:pos+n]; pos += n
    y = full(embedded)
    expected = np.concatenate([y[full.slices[key]] for key in keys])
    sub = Hessian(model, [b], 3, 0, parameters={k:params[k] for k in keys})
    np.testing.assert_allclose(sub(v), expected, atol=1e-11, rtol=1e-10)


def test_coverage_breakdown():
    params = {'a': torch.empty(2), 'b': torch.empty(2)}
    q = np.array([[.5], [.5], [.5], [.5]])
    bases = [('a', ['a'], np.array([[1.], [0.]])),
             ('b', ['b'], np.array([[1.], [1.]])/np.sqrt(2))]
    result = block_metrics(q, ['a', 'b'], bases, params, 1)
    np.testing.assert_allclose([result['blocks'][0]['participation'],
                                result['blocks'][0]['conditional_capture'],
                                result['blocks'][1]['conditional_capture'],
                                result['containment']], [.5, .5, 1., .75])


def test_atomic_spectrum_storage(tmp_path):
    path = tmp_path/'spectrum.npz'
    save_npz(path, values=np.array([2., 1.]))
    np.testing.assert_array_equal(np.load(path)['values'], [2., 1.])
    assert not list(tmp_path.glob('*.tmp.npz'))
