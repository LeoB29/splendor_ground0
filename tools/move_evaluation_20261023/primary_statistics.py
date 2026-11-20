"""Primary literal replicate/root bootstrap; independently checked by audit.py."""
import hashlib


def calculate(rows, boundary=lambda: None):
    import numpy as np
    if len(rows) != 18:
        raise ValueError('Expected18 fixed roots')
    values = np.asarray([row['differences_halfpoints'] for row in rows], dtype=np.int64)
    if values.shape != (18, 32):
        raise ValueError('Expected32 paired worlds each')
    rng = np.random.Generator(np.random.PCG64(2026102301))
    sums = np.zeros(50000, dtype=np.int64)
    for replicate in range(50000):
        if replicate % 100 == 0:
            boundary()
        numerator = 0
        for root_index in range(18):
            indices = rng.integers(0, 32, size=32, dtype=np.int64)
            numerator += int(values[root_index, indices].sum(dtype=np.int64))
        sums[replicate] = numerator
    return {'mean_gain': int(values.sum())/1152,
            'by_fit_mean_gain': {str(fit): sum(sum(r['differences_halfpoints']) for r in rows if r['fit'] == fit)/384
                                 for fit in (161001, 161002, 161003)},
            'conditional_ci95': np.percentile(sums.astype(np.float64)/1152, [2.5, 97.5], method='linear').tolist(),
            'bootstrap_integer_totals_sha256': hashlib.sha256(sums.astype('<i8').tobytes()).hexdigest()}
