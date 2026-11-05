import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib plot extra unavailable')
class SevenReport(unittest.TestCase):
    def test_report_preserves_separate_condition_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'measurement'; source.mkdir()
            def save(name, obj): (source/name).write_text(json.dumps(obj), encoding='utf-8')
            save('metadata.json', dict(step=3, scope='FP', valid_tokens=8,
                 checkpoint_sha256='checkpoint', batch_sha256='batch', training_arguments={'seed':17}))
            save('status.json', {'phase':'completed'})
            geom = {'candidate_dim':4, 'H_to_candidate_containment':.5, 'overlap':.5,
                    'angle_range_deg':[0.,90.], 'capture_each_mode':[.7,.3],
                    'principal_angles_deg':[0.,90.]}
            record = dict(comparison='S4_H7', target_rank=2, self_rank=1,
                          status='preliminary_boundary_or_insufficient_positive', geometry=geom,
                          random_block_baseline={'expected':.1})
            save('coverage.json', [record])
            save('ritz.json', {'H7':{'Ritz_values':[1.,None], 'target_eigenvalues':[2.,1.]}})
            block = {'norm':1.,'norm2':1.,'norm2_stderr':.1,'norm2_ci95':[.8,1.2],'element_rms':.2}
            save('frobenius.json', {'block_order':['Q'], 'blocks':{'QQ':{'H':block,'A':block,'R':block}}})
            cmd=[sys.executable,'-m','ougd.seven_report',str(source),'--output',str(root/'report')]
            process=subprocess.run(cmd,capture_output=True,text=True,env={**os.environ,'MPLBACKEND':'Agg',
                                                                          'MPLCONFIGDIR':str(root/'mplconfig')})
            self.assertEqual(process.returncode,0,process.stdout+'\n'+process.stderr)
            report=(root/'report/report.md').read_text(encoding='utf-8')
            self.assertIn('preliminary or unavailable',report)
            self.assertTrue((root/'report/condition_01/coverage.png').exists())
            self.assertTrue((root/'report/condition_01/frobenius.png').exists())
            self.assertTrue((root/'report/coverage_summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
