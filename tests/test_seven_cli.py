import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class SevenBlockPipeline(unittest.TestCase):
    def test_tiny_train_and_measure_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(vocab_size=9, dim=4, hidden_dim=6, layers=1,
                          heads=1, kv_heads=1, context=3)
            (root/'model.json').write_text(json.dumps(config))
            np.save(root/'train.npy', np.array([[0,1,2,3], [3,4,5,6], [2,3,4,5]]))
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1'}
            train = [sys.executable, '-m', 'ougd.train', '--config', str(root/'model.json'),
                     '--data', str(root/'train.npy'), '--output', str(root/'run'),
                     '--steps', '1', '--batch-tokens', '6', '--microbatch', '1', '--layer', '0',
                     '--precision', 'fp32', '--measure-blocks', 'seven', '--measure-steps', '0']
            subprocess.run(train, check=True, env=env, capture_output=True, text=True)
            measure = [sys.executable, '-m', 'ougd.seven_measure', '--checkpoint',
                       str(root/'run/pre_step_000000.pt'), '--output', str(root/'measure'),
                       '--expected-tokens', '6', '--microbatch', '1', '--precision', 'fp64',
                       '--k', '1', '--probes', '2', '--bootstrap-samples', '20',
                       '--random-repeats', '3', '--solver-tol', '1e-8', '--maxiter', '500']
            process = subprocess.run(measure, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + '\n' + process.stderr)
            out = root/'measure'
            meta = json.loads((out/'metadata.json').read_text())
            self.assertEqual(meta['parameter_order'], list('QKVOUGD'))
            self.assertTrue(meta['checkpoint_unchanged'])
            norms=json.loads((out/'frobenius.json').read_text())['blocks']
            self.assertEqual(len(norms), 28)
            for key in ['UU','DD']:
                self.assertTrue(norms[key]['R_self_analytic_zero'])
                self.assertEqual(norms[key]['R']['norm2'], 0.)
                self.assertIsNone(norms[key]['cosine_AR'])
            records = json.loads((out/'coverage.json').read_text())
            self.assertEqual({r['comparison'] for r in records}, {'S4_H4','S4_H7','S7_H7'})
            four = next(r for r in records if r['comparison']=='S4_H7')
            seven = next(r for r in records if r['comparison']=='S7_H7')
            if four.get('geometry') and seven.get('geometry'):
                self.assertLessEqual(four['geometry']['H_to_candidate_containment'],
                                     seven['geometry']['H_to_candidate_containment'] + 1e-10)
            self.assertEqual(json.loads((out/'status.json').read_text())['phase'], 'completed')
            second = subprocess.run(measure, env=env, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)  # Existing results are not overwritten.
            probe = [sys.executable, '-m', 'ougd.make_fixed_probes', '--data', str(root/'train.npy'),
                     '--output', str(root/'probes'), '--context', '3', '--vocab', '9',
                     '--tokens', '3', '--cohorts', '3', '--seed', '10']
            subprocess.run(probe, check=True, env=env, capture_output=True, text=True)
            probe_meta = json.loads((root/'probes/metadata.json').read_text())
            indices = [c['row_indices'][0] for c in probe_meta['cohorts']]
            self.assertEqual(len(set(indices)), 3)
            fp = [sys.executable, '-m', 'ougd.seven_measure', '--checkpoint',
                  str(root/'run/pre_step_000000.pt'), '--output', str(root/'fp_measure'),
                  '--scope', 'FP', '--probe-batch', str(root/'probes/FP1.pt'),
                  '--expected-tokens', '3', '--precision', 'fp64', '--k', '1',
                  '--skip-frobenius', '--random-repeats', '3', '--solver-tol', '1e-8']
            result = subprocess.run(fp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
            fp_meta = json.loads((root/'fp_measure/metadata.json').read_text())
            self.assertEqual(fp_meta['batch_sha256'], probe_meta['cohorts'][0]['batch_sha256'])
            self.assertEqual(json.loads((root/'fp_measure/frobenius.json').read_text())['status'], 'skipped_by_request')


if __name__ == '__main__':
    unittest.main()
