import unittest
import numpy as np
from ougd.solvers import spectrum, zero_spectrum

class DegenerateSpectra(unittest.TestCase):
    def test_analytic_zero_spectrum_is_explicitly_degenerate(self):
        s=zero_spectrum(8,3,np.float64)
        self.assertTrue(s['structural_zero'])
        self.assertTrue(s['fully_degenerate'])
        self.assertTrue(s['converged'])
        np.testing.assert_array_equal(s['values'],0)
        np.testing.assert_array_equal(s['vectors'].T@s['vectors'],np.eye(3))

    def test_zero_operator_does_not_abort_the_measurement(self):
        s=spectrum(lambda v:np.zeros_like(v),8,1,'LA',1e-8,10,dtype=np.float64)
        if s['converged']:
            np.testing.assert_array_equal(s['values'],0)
        else:
            self.assertIsNotNone(s['failure'])
            self.assertEqual(s['vectors'].shape,(8,0))

if __name__=='__main__': unittest.main()
