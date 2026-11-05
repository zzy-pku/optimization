import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch


class OptimizerTraining(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
    def test_hybrid_optimizers_on_cuda(self):
        from ougd.model import ModelConfig, Transformer, ce_sum
        from ougd.train import make_optimizer
        for name in ('muon', 'soap'):
            with self.subTest(optimizer=name):
                torch.manual_seed(17)
                model = Transformer(ModelConfig(vocab_size=9, dim=4, hidden_dim=6,
                    layers=1, heads=1, kv_heads=1, context=3)).cuda()
                args = SimpleNamespace(optimizer=name, lr=.003, aux_lr=.0004, beta1=.9,
                    beta2=.95, eps=1e-8, weight_decay=.1, muon_momentum=.95,
                    soap_precondition_frequency=2, soap_max_precond_dim=16)
                opt = make_optimizer(model, args)
                before = model.blocks[0].U.weight.detach().clone()
                batch = {'input_ids': torch.tensor([[0, 1, 2]], device='cuda'),
                         'labels': torch.tensor([[1, 2, 3]], device='cuda')}
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    ce_sum(model, batch).backward()
                    opt.step()
                self.assertTrue(torch.isfinite(model.blocks[0].U.weight).all().item())
                self.assertFalse(torch.equal(before, model.blocks[0].U.weight))

    def test_optimizer_training_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'model.json').write_text(json.dumps(dict(vocab_size=9, dim=4, hidden_dim=6,
                layers=1, heads=1, kv_heads=1, context=3)))
            np.save(root/'data.npy', np.array([[0,1,2,3], [3,4,5,6], [2,3,4,5]]))
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1'}
            base = [sys.executable, '-m', 'ougd.train', '--config', str(root/'model.json'),
                    '--data', str(root/'data.npy'), '--batch-tokens', '6', '--layer', '0',
                    '--precision', 'fp32', '--measure-blocks', 'seven', '--warmup', '10',
                    '--measure-steps', '0', '--save-every', '1']
            for name in ['adamw', 'adam', 'sgd', 'muon', 'soap']:
                with self.subTest(optimizer=name):
                    def run(output, steps, extra=(), check=True):
                        return subprocess.run(base + ['--optimizer', name, '--output', str(root/output),
                            '--steps', str(steps)] + list(extra), env=env, text=True,
                            capture_output=True, check=check)
                    run(name+'_full', 3)
                    run(name+'_split', 1)
                    latest = root/(name+'_split')/'latest.pt'
                    run(name+'_split', 3, ['--resume', str(latest)])
                    full = torch.load(root/(name+'_full')/'latest.pt', map_location='cpu', weights_only=False)
                    split = torch.load(latest, map_location='cpu', weights_only=False)
                    self.assertEqual(split['args']['optimizer'], name)
                    for key in full['model']:
                        torch.testing.assert_close(full['model'][key], split['model'][key], rtol=0, atol=0)
                    # Resuming with a different optimizer is rejected before loading state.
                    mismatch = subprocess.run(base + ['--optimizer', 'adam' if name != 'adam' else 'sgd',
                        '--output', str(root/'mismatch'), '--steps', '3', '--resume', str(latest)],
                        env=env, text=True, capture_output=True)
                    self.assertNotEqual(mismatch.returncode, 0)
                    self.assertIn('resume must preserve optimizer', mismatch.stderr)


if __name__ == '__main__':
    unittest.main()
