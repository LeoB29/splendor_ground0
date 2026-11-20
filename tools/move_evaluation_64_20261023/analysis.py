"""Strict A64/B64 primary analysis and authenticated independent comparison.

No model is imported or loaded here. The original independent analyzer remains
unchanged; only this study's sealed A64 control replaces its historical A128 row.
"""
from collections import Counter
import hashlib

from . import common as c

FITS = (161001, 161002, 161003)
FIELDS = {'id', 'fit', 'control_choice', 'alternative_choice',
          'differences_halfpoints', 'a_scores_halfpoints', 'b_scores_halfpoints'}


def validate_rows(rows, candidate_lock):
    roots = candidate_lock['roots']
    c.require(type(rows) is list and type(roots) is list and len(rows) == len(roots) == 18,
              'Analysis requires all eighteen ordered roots')
    c.require(len({r['id'] for r in roots}) == 18 and
              Counter(r['fit'] for r in roots) == Counter({f: 6 for f in FITS}),
              'Analysis root identity/fit coverage')
    changed = Counter({f: 0 for f in FITS})
    for row, root in zip(rows, roots, strict=True):
        c.require(type(row) is dict and set(row) == FIELDS and row['id'] == root['id'] and
                  type(row['fit']) is int and row['fit'] == root['fit'], 'Ordered analysis schema')
        support = len(root['candidates']['actions'])
        for key in ('control_choice', 'alternative_choice'):
            c.require(type(row[key]) is int and 0 <= row[key] < support, 'Sealed choice index')
        c.require(type(root['control_choice']) is int and row['control_choice'] == root['control_choice'],
                  'Analysis control must be sealed A64')
        ds = row['differences_halfpoints']
        c.require(type(ds) is list and len(ds) == 32 and
                  all(type(v) is int and -2 <= v <= 2 for v in ds), 'Exact integer paired differences')
        aa, bb = row['a_scores_halfpoints'], row['b_scores_halfpoints']
        if row['control_choice'] == row['alternative_choice']:
            c.require(aa is None and bb is None and ds == [0]*32,
                      'Unchanged action requires structural zeros without imputed arm scores')
        else:
            changed[row['fit']] += 1
            c.require(all(type(values) is list and len(values) == 32 and
                          all(type(v) is int and v in (0, 1, 2) for v in values) for values in (aa, bb)),
                      'Every changed root needs 32 natural terminal score pairs')
            c.require(ds == [b-a for a, b in zip(aa, bb, strict=True)], 'Paired arithmetic differs')
    c.require(sum(changed.values()) >= 6 and all(changed[f] >= 1 for f in FITS),
              'Insufficient intervention breadth')


def calculate(rows, candidate_lock, boundary=lambda: None):
    """Literal replicate/root draws, independent of the chunked reference code."""
    import numpy as np
    validate_rows(rows, candidate_lock)
    values = np.asarray([r['differences_halfpoints'] for r in rows], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(2026102301))
    totals = np.empty(50000, dtype=np.int64)
    for replicate in range(50000):
        if replicate % 100 == 0:
            boundary()
        total = 0
        for root in range(18):
            indices = rng.integers(0, 32, size=32, dtype=np.int64)
            total += int(values[root, indices].sum(dtype=np.int64))
        totals[replicate] = total
    boundary()
    return {'mean_gain': int(values.sum(dtype=np.int64))/1152,
            'by_fit_mean_gain': {str(f): sum(sum(r['differences_halfpoints']) for r in rows if r['fit'] == f)/384
                                 for f in FITS},
            'conditional_ci95': np.percentile(totals.astype(np.float64)/1152, [2.5, 97.5], method='linear').tolist(),
            'bootstrap_integer_totals_sha256': hashlib.sha256(totals.astype('<i8').tobytes(order='C')).hexdigest()}


def calculate_and_verify(rows, candidate_lock, boundary=lambda: None):
    from tools.move_evaluation_20261023.audit import statistics
    from tools.matched_strength_20261020.statistics import runtime_contract
    expected = c.read(c.OUT/'statistics_runtime.json')
    c.require(runtime_contract() == expected, 'Statistics runtime changed before calculation')
    primary = calculate(rows, candidate_lock, boundary)
    independent = statistics(rows, candidate_lock, checks_complete=True, boundary=boundary)
    c.require(all(primary[key] == independent[key] for key in primary),
              'Independent statistics/bootstrap stream differs')
    c.require(runtime_contract() == expected, 'Statistics runtime changed during calculation')
    return {'primary': primary, 'independent': independent, 'comparison': 'exact_match',
            'runtime': c.artifact(c.OUT/'statistics_runtime.json'),
            'comparison_identity': 'A64_vs_B64_fixed_development_panel',
            'reduced_world_quality_effect_estimated': False}
