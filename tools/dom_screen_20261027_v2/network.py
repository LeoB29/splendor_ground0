"""Pinned browser policy: double accumulation, float32 hidden layers, ReLU."""
from pathlib import Path
import hashlib
import numpy as np
ASSETS = Path(__file__).resolve().parents[2] / 'external/dom_splendor_426fa50'
WEIGHT = ASSETS / 'play/weights.bin'
WEIGHT_SHA256 = 'e80e468a7e99ead87b43277c8c17219be669e55fac54604abbd55d68b1586191'
DIMENSIONS = (267, 512, 512, 77)
WEIGHT_BYTES = 1757492

class FrozenPolicy:
    def __init__(self):
        raw = WEIGHT.read_bytes()
        if len(raw) != WEIGHT_BYTES or hashlib.sha256(raw).hexdigest() != WEIGHT_SHA256:
            raise ValueError('Dom exported policy bytes changed')
        flat = np.frombuffer(raw, dtype='<f4')
        self.weights, self.biases, offset = [], [], 0
        for n, m in zip(DIMENSIONS, DIMENSIONS[1:]):
            w = flat[offset:offset+n*m].reshape(m, n).astype(np.float64); offset += n*m
            b = flat[offset:offset+m].astype(np.float64); offset += m
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite Dom weight')
            w.flags.writeable = b.flags.writeable = False
            self.weights.append(w); self.biases.append(b)
        if offset != len(flat):
            raise ValueError('Trailing Dom weights')

    def predict(self, features):
        x = np.asarray(features)
        if x.shape != (267,) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('Expected267 finite float32 features')
        for index, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = w @ x.astype(np.float64) + b
            if index < 2:
                x = np.maximum(x, 0.).astype(np.float32)
        if x.shape != (77,) or x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError('Invalid Dom policy output')
        return x
