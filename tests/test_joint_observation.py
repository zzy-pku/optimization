import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from ougd.m2 import component_options, run_m2
from ougd.model import BLOCKS
from ougd.components import Decomposition
from tests.test_numerics import fixture

class JointObservation(unittest.TestCase):
    def test_joint_mode_enables_only_H_ablation_and_honors_budget(self):
        for mode in ('M2','M1+M2'):
            for skip in (True,False):
                args=SimpleNamespace(module=mode,no_ablation=skip)
                for op in ('H','A','R'):
                    actual=component_options(args,op)
                    self.assertEqual(actual.no_ablation, mode=='M2' or op!='H' or skip)
                    self.assertEqual(args.no_ablation,skip)

    def test_joint_run_shares_H_statistics_and_one_H_observation(self):
        model,batch=fixture()
        backend=Decomposition(model,[batch],5,0)
        args=SimpleNamespace(module='M1+M2',no_ablation=False,precision='fp64',
            check_probes=1,seed=23,probes=4,norms_only=True)
        calls=[]
        def observed(options,state,view,gradient,folder,norms):
            calls.append((view.operator,options.no_ablation,norms))
            from ougd.measure import write
            write(folder/'coverage.json',[dict(metric='norms',status='pilot')])
            return dict(matvecs=0)
        with tempfile.TemporaryDirectory() as temp,patch('ougd.m2.run',side_effect=observed):
            out=Path(temp)
            run_m2(args,{},backend,np.zeros(backend.n),out,dict(modules=['M1','M2']))
            import json
            for op in ('H','A','R'):
                meta=json.loads((out/op/'operator_meta.json').read_text())
                self.assertEqual(meta['modules'],['M1','M2'] if op=='H' else ['M2'])
        self.assertEqual([c[0] for c in calls],['H','A','R'])
        self.assertEqual([c[1] for c in calls],[False,True,True])
        self.assertEqual(set(calls[0][2]['blocks']),{i+j for j in BLOCKS for i in BLOCKS[:BLOCKS.index(j)+1]})

if __name__=='__main__': unittest.main()
