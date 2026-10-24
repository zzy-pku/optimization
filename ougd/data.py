import hashlib
from pathlib import Path
import numpy as np
import torch

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

class TokenData:
    """Memory-mapped packed token rows: [N, context+1]. -100 labels mask padding."""
    def __init__(self, path, context, vocab):
        self.path = str(Path(path).resolve())
        self.tokens = np.load(path, mmap_mode='r', allow_pickle=False)
        if self.tokens.ndim != 2 or self.tokens.shape[1] != context + 1 or not len(self.tokens):
            raise ValueError('expected nonempty [N, context+1] .npy tokens')
        if not np.issubdtype(self.tokens.dtype, np.integer):
            raise ValueError('token IDs must be integers')
        if self.tokens.min() < 0 or self.tokens.max() >= vocab:
            raise ValueError('token out of vocabulary')

    def batch(self, indices, device):
        rows = torch.tensor(np.asarray(self.tokens[indices]), dtype=torch.long, device=device)
        return {'input_ids': rows[:, :-1].contiguous(), 'labels': rows[:, 1:].contiguous()}

def make_batches(data, indices, micro, device):
    return [data.batch(indices[i:i+micro], device) for i in range(0, len(indices), micro)]
