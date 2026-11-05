import copy
import unittest
import numpy as np
import torch
from ougd.components import Decomposition
from ougd.model import SEVEN_BLOCKS
from tests.test_seven_block import fixture


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
class SevenBlockCUDA(unittest.TestCase):
    def test_fp32_gpu_hessian_and_anchor_match_cpu(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, batch = fixture()
        cpu_model = model.float()
        gpu_model = copy.deepcopy(cpu_model).cuda()
        cpu = Decomposition(cpu_model, [batch], 5, 0, blocks=SEVEN_BLOCKS)
        gpu_batch = {k: v.cuda() for k, v in batch.items()}
        gpu = Decomposition(gpu_model, [gpu_batch], 5, 0, blocks=SEVEN_BLOCKS)
        v = np.random.default_rng(5).normal(size=cpu.n); v /= np.linalg.norm(v)
        h_cpu, h_gpu = cpu.request(v, 1), gpu.request(v, 1)
        a_gpu, r_gpu = gpu.request(v, 2), gpu.request(v, 4)
        self.assertLess(np.linalg.norm(h_cpu-h_gpu)/(np.linalg.norm(h_cpu)+np.linalg.norm(h_gpu)+1e-30), 1e-4)
        self.assertLess(np.linalg.norm(h_gpu-a_gpu-r_gpu)/(np.linalg.norm(h_gpu)+np.linalg.norm(a_gpu)+np.linalg.norm(r_gpu)+1e-30), 1e-4)


if __name__ == '__main__':
    unittest.main()
