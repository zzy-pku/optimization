from dataclasses import asdict
import json
import sys

import numpy as np
import pytest
import torch

from ougd import shampoo_measure, train
from ougd.model import ModelConfig, Transformer
from ougd.shampoo import DensePowers, LowRankPowers, Partition, Shampoo
from ougd.shampoo_calibration import CalibratedOperator
from ougd.shampoo_metrics import directional_differences, ratio_summary


def small_model():
    return Transformer(ModelConfig(vocab_size=11, dim=4, hidden_dim=5, layers=1,
                                   heads=1, kv_heads=1, context=3, tie_embeddings=False))


def assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_packing_and_operator_identities():
    torch.manual_seed(2)
    part = Partition(small_model())
    x, y = (torch.randn(part.n, dtype=torch.float64) for _ in range(2))
    for groups in part.blocks.values():
        inner = 0.
        for entries in groups.values():
            a, b = (part.pack_vector(entries, value) for value in (x, y))
            inner += (a * b).sum()
            restored = torch.cat([t.reshape(-1) for _, t in part.unpack(entries, a)])
            assert_close(restored, x[part.interval(entries)])
        assert_close(inner, x @ y)

    d = part.blocks['M']['L01.D']
    grad = part.pack_vector(d, x)
    ridge = 0.3
    left = DensePowers(grad @ grad.T, ridge).matrix(.25)
    right = DensePowers(grad.T @ grad, ridge).matrix(.25)
    direct = left @ grad @ right
    assert_close(direct.T.contiguous().flatten(), torch.kron(right.contiguous(), left.contiguous()) @
                 grad.T.contiguous().flatten())
    assert_close(dict(part.unpack(d, direct))[d[0].name], right @ grad.T @ left)

    rows = part.blocks['G']['global'][-1].row_stop
    z = torch.randn(rows, part.columns * 2, dtype=x.dtype)
    operators = {scheme: CalibratedOperator(part, scheme, z, ridge, True) for scheme in ('G', 'M', 'U')}
    for op in operators.values():
        assert_close(op.apply(op.apply(x, .125), .125), op.apply(x))
        assert_close(x @ op.apply(y), y @ op.apply(x))
        assert x @ op.apply(x) > 0
    for scheme in ('M', 'U'):
        assert_close(sum(operators[scheme].right_statistics.values()), operators['G'].right_statistics['global'])

    lr = LowRankPowers(z, ridge)
    complete = torch.linalg.qr(z, mode='complete').Q
    for power in (.125, .25):
        dense = DensePowers(z @ z.T, ridge).matrix(power)
        assert_close(lr(complete, power), dense @ complete)
        assert_close(lr(complete[:, z.shape[1]:], power), ridge**(-power) * complete[:, z.shape[1]:])

    # Orthonormal bases and gains of the explicit operators verify the modal error formula.
    eye = torch.eye(part.n, dtype=x.dtype)
    g, m = (operators[key].apply(eye) for key in ('G', 'M'))
    g_vals, g_vecs = torch.linalg.eigh(g)
    m_vals, m_vecs = torch.linalg.eigh(m)
    modal = ((g_vals[:, None] - m_vals) ** 2 * (g_vecs.T @ m_vecs).square()).sum()
    assert_close(modal, (g - m).square().sum())

    h = torch.diag(torch.linspace(-1., 1., part.n, dtype=x.dtype))
    half = operators['U'].apply(eye, .125)
    s = half @ h @ half
    assert_close(s @ x, operators['U'].apply(h @ operators['U'].apply(x, .125), .125))
    assert_close(s, s.T)
    assert int((torch.linalg.eigvalsh(s) < 0).sum()) == int((h.diagonal() < 0).sum())

    diagonal = x.square().add(ridge).rsqrt()
    for ties in ({key: operators['G'].apply for key in operators},
                 {key: lambda v: diagonal * v for key in operators}):
        for samples in directional_differences(ties, [x, y]).values():
            assert ratio_summary(samples)['squared_relative_error'] == 0


@pytest.mark.parametrize('scheme', ['M', 'U'])
def test_training_cache_sequence(scheme):
    model = small_model()
    part = Partition(model)
    opt = Shampoo(part, scheme, 0.002, (0.9, 0.95), 1e-6, 0.1)
    for step in range(1, 11):
        for p in part.parameters.values():
            p.grad = torch.full_like(p, 0.1 if step == 1 else -0.2)
        opt.step()
        assert opt.root_last_refresh_step == (1 if step < 10 else 10)
        label, members = next(iter(opt.groups.items()))
        if step == 1:
            packed = part.pack(members, {k: p.grad for k, p in part.parameters.items()})
            torch.testing.assert_close(opt.factors[label][0], 0.05 * (packed @ packed.T))
    restored = Shampoo(part, scheme, 0.002, (0.9, 0.95), 1e-6, 0.1)
    restored.load_state_dict(train.cpu_tree(opt.state_dict()))
    assert restored.steps == restored.root_last_refresh_step == 10
    for key, pair in restored.roots.items():
        for a, b in zip(pair, opt.roots[key]):
            assert torch.equal(a, b)


def test_training_resume_and_checkpoint_analysis(tmp_path, monkeypatch):
    def invoke(module, *args):
        monkeypatch.setattr(sys, 'argv', [module.__name__, *map(str, args)])
        module.main()

    rng = np.random.default_rng(12)
    rows = rng.integers(11, size=(10, 4))
    config, data, fp1 = (tmp_path / name for name in ('config.json', 'train.npy', 'FP1.pt'))
    tokenizer, init = tmp_path/'tokenizer', tmp_path/'init.pt'
    tokenizer.write_bytes(b'test tokenizer')
    model = small_model()
    config.write_text(json.dumps(asdict(model.config)))
    torch.save(model.state_dict(), init)
    np.save(data, rows)
    torch.save(dict(input_ids=torch.tensor(rows[-2:, :-1]), labels=torch.tensor(rows[-2:, 1:])), fp1)
    run = tmp_path/'run_M'
    options = ('--config', config, '--data', data, '--output', run, '--optimizer', 'shampoo',
               '--shampoo-partition', 'M', '--initial-state', init, '--fp1', fp1, '--tokenizer',
               tokenizer, '--precision', 'fp32', '--steps', 2, '--batch-tokens', 6,
               '--measure-steps', '0,1,2', '--layer', 0, '--calibration-windows', 2)
    invoke(train, *options)
    first = torch.load(run/'checkpoint_000002.pt', weights_only=False)
    invoke(train, *options, '--resume', run/'checkpoint_000001.pt')
    second = torch.load(run/'checkpoint_000002.pt', weights_only=False)
    for key, value in first['model'].items():
        assert torch.equal(value, second['model'][key])
    assert len((run/'t0.jsonl').read_text().splitlines()) == 2
    output = tmp_path/'analysis'
    invoke(shampoo_measure, '--checkpoint', run/'checkpoint_000002.pt', '--data', data,
           '--fp1', fp1, '--output', output, '--window', 2, '--k', 2, '--probes', 2,
           '--spectral-modes', 1, '--no-plots')
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['completed_steps'] == 2
    assert metrics['root_last_refresh_step'] == 1
    assert set(metrics['pairs']['U|M']) == {'E_R_I', 'E_m', 'E_R_V', 'E_S'}
    assert all(row['status'] == 'completed' for row in metrics['stages'].values())
    coverage = json.loads((output/'coverage.json').read_text())
    np.testing.assert_allclose(coverage['U']['historical']['p_sum'], 1, atol=1e-5)
    assert coverage['U']['matched_actual_rank']['rank'] == coverage['M']['matched_actual_rank']['rank']
    assert (output/'effective_spectrum.json').exists()
