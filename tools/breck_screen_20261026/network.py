"""Pinned historical single-head policy, source-exact float32 forward arithmetic."""
from pathlib import Path
import hashlib
import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / 'external/breck_webapp_6ec66ef'
WEIGHT = ASSETS / 'Splendor/RL/trained_agents/inference_model.npz'
WEIGHT_SHA256 = '5044383cadd98aa6c753b9d0a2bd8557b1ad017c21b6d8ba762027f63277438a'
DIMENSIONS = (251, 512, 512, 256, 141)

class FrozenPolicy:
    def __init__(self):
        if hashlib.sha256(WEIGHT.read_bytes()).hexdigest() != WEIGHT_SHA256:
            raise ValueError('Breck policy weight bytes changed')
        self.weights, self.biases = [], []
        with np.load(WEIGHT, allow_pickle=False) as data:
            if set(data.files) != {f'{prefix}_{suffix}' for prefix in ('dense1', 'dense2', 'dense3', 'q') for suffix in ('W', 'b')}:
                raise ValueError('Unexpected Breck exported policy structure')
            for i, (n, m) in enumerate(zip(DIMENSIONS, DIMENSIONS[1:]), 1):
                prefix = f'dense{i}' if i < 4 else 'q'
                w, b = data[f'{prefix}_W'], data[f'{prefix}_b']
                if w.shape != (n, m) or b.shape != (m,):
                    raise ValueError('Unexpected Breck layer dimensions')
                w, b = w.astype(np.float32), b.astype(np.float32)
                if not np.isfinite(w).all() or not np.isfinite(b).all():
                    raise ValueError('Nonfinite policy weight')
                self.weights.append(w)
                self.biases.append(b)

    def predict(self, features):
        if np.asarray(features).shape != (251,):
            raise ValueError('Expected exactly 251 feature values')
        x = np.asarray(features).astype(np.float32)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = x @ w + b
            x = np.where(x > 0, x, 0.3 * x)
        result = x @ self.weights[-1] + self.biases[-1]
        if result.dtype != np.float32 or result.shape != (141,) or not np.isfinite(result).all():
            raise ValueError('Invalid policy output')
        return result
