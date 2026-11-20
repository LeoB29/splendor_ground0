"""Prospective paired statistics, called only after the complete benchmark is verified.

The score functions read no files and load no game engine or model. Runtime
provenance reads only the numerical runtime files that preparation pins. The
analyzer supplies scores derived from verified natural endings or forfeits.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

FITS = (161001, 161002, 161003)
BOOTSTRAP_SEED = 202610206701
BOOTSTRAP_REPLICATES = 50000
BOOTSTRAP_CHUNK = 1000
CONFIDENCE_LEVEL = .95
GAME_FIELDS = {'game','pair','fit','local_seat','local_score_halfpoints','termination','forfeiting_player'}


def _runtime_paths():
    """Relevant loaded implementations, not an environment-wide inventory."""
    modules = [
        ('pcg64', np.random.PCG64.__module__),
        ('generator', np.random.Generator.__module__),
        ('bit_generator', 'numpy.random.bit_generator'),
        ('mtrand', 'numpy.random.mtrand'),
        ('random_common', 'numpy.random._common'),
        ('array_core', 'numpy._core._multiarray_umath' if int(np.__version__.split('.')[0]) >= 2
                       else 'numpy.core._multiarray_umath'),
    ]
    quantile = inspect.getsourcefile(inspect.unwrap(np.quantile))
    require(quantile is not None, 'Quantile implementation source is required')
    paths = [('python_executable', sys.executable), ('numpy_package', np.__file__),
             ('quantile_implementation', quantile)]
    paths.extend((role, importlib.import_module(module).__file__) for role, module in modules)
    if sys.platform == 'win32':
        # python.exe is a small launcher on Windows; pin its actual interpreter too.
        paths.append(('python_shared_library', str(Path(sys.base_prefix) /
                      f'python{sys.version_info.major}{sys.version_info.minor}.dll')))
    return paths


def runtime_contract():
    """Fresh, JSON-only runtime hashes for exact preparation/analysis comparison.

    The frozen statistics_runtime.json binds these external absolute paths and
    hashes transitively; they are not workspace-relative artifact descriptors.
    No result or production game is read, and hashes are deliberately uncached.
    """
    files = []
    for role, source in _runtime_paths():
        require(source is not None, 'Numerical runtime file is required')
        path = Path(source).resolve(strict=True)
        before = path.stat()
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                'Numerical runtime file changed while hashing')
        files.append({'role': role, 'path': str(path), 'bytes': after.st_size, 'sha256': digest})
    require(len({item['role'] for item in files}) == len(files), 'Unique numerical runtime roles')
    return {'schema': 'matched_strength_statistics_runtime_v1',
            'python': {'version': sys.version, 'implementation': sys.implementation.name,
                       'executable': str(Path(sys.executable).resolve(strict=True))},
            'numpy_version': np.__version__, 'files': files}


def require(value, label):
    if not value: raise ValueError(label)


def validated_games(games, pairs_per_fit):
    require(type(pairs_per_fit) is int and 1 <= pairs_per_fit <= 100, 'Fixed positive pairs per fit')
    require(type(games) is list and len(games) == 6*pairs_per_fit, 'Complete fixed game sample required')
    ordered=[]
    for row in games:
        require(type(row) is dict and set(row) == GAME_FIELDS, 'Exact verified-game score schema')
        require(all(type(row[key]) is int for key in ('game','pair','fit','local_seat','local_score_halfpoints')),
                'Exact integer game, pair, fit, seat and score identities')
        game,pair,fit,seat,score=(row[k] for k in ('game','pair','fit','local_seat','local_score_halfpoints'))
        require(0 <= game < 6*pairs_per_fit and pair == game//2 and fit == FITS[pair%3] and
                seat == 1-((pair+game%2)%2), 'Frozen paired schedule identity')
        require(score in (0,1,2), 'Game score is loss0, draw1, win2 halfpoints')
        require(row['termination'] in ('natural','forfeit'), 'Only resolved natural or forfeit endings')
        if row['termination'] == 'natural':
            require(row['forfeiting_player'] is None, 'Natural ending has no forfeiting player')
        else:
            player=row['forfeiting_player']
            require(player in ('local','opponent') and score == (0 if player=='local' else 2),
                    'Genuine forfeit is a loss for the forfeiting player')
        ordered.append(dict(row))
    ordered.sort(key=lambda row:row['game'])
    require([row['game'] for row in ordered] == list(range(6*pairs_per_fit)), 'Unique complete fixed game IDs')
    for pair in range(3*pairs_per_fit):
        first,second=ordered[2*pair:2*pair+2]
        require(first['pair'] == second['pair'] == pair and first['fit'] == second['fit'] and
                {first['local_seat'],second['local_seat']} == {0,1}, 'Complete swapped-seat pair')
    return ordered


def pair_numerators(ordered):
    """A pair's score is its sum of game halfpoints divided by four."""
    return [ordered[i]['local_score_halfpoints']+ordered[i+1]['local_score_halfpoints']
            for i in range(0,len(ordered),2)]


def bootstrap_totals(games, *, pairs_per_fit, bootstrap_seed=BOOTSTRAP_SEED, boundary=lambda:None):
    ordered=validated_games(games,pairs_per_fit)
    require(type(bootstrap_seed) is int and 0 <= bootstrap_seed < 2**64, 'Fixed unsigned64 bootstrap seed')
    pair_values=pair_numerators(ordered)
    strata=[np.asarray(pair_values[index::3],dtype=np.int64) for index in range(3)]
    require(all(len(values)==pairs_per_fit for values in strata), 'Equal fixed fit allocation')
    random=np.random.Generator(np.random.PCG64(bootstrap_seed))
    totals=np.zeros(BOOTSTRAP_REPLICATES,dtype=np.int64)
    for start in range(0,BOOTSTRAP_REPLICATES,BOOTSTRAP_CHUNK):
        boundary()
        count=min(BOOTSTRAP_CHUNK,BOOTSTRAP_REPLICATES-start)
        for values in strata:
            indices=random.integers(0,pairs_per_fit,size=(count,pairs_per_fit))
            totals[start:start+count] += values[indices].sum(axis=1)
    boundary()
    return totals


def score_summary(rows):
    counts=Counter(row['local_score_halfpoints'] for row in rows)
    forfeits=Counter(row['forfeiting_player'] for row in rows if row['termination']=='forfeit')
    return {'games':len(rows),'local_wins':counts[2],'draws':counts[1],'local_losses':counts[0],
        'local_score_halfpoints_total':sum(row['local_score_halfpoints'] for row in rows),
        'mean_local_game_score':sum(row['local_score_halfpoints'] for row in rows)/(2*len(rows)),
        'forfeits':{'local':forfeits['local'],'opponent':forfeits['opponent']}}


def statistics(games, *, pairs_per_fit, bootstrap_seed=BOOTSTRAP_SEED, boundary=lambda:None):
    """One complete fixed-sample primary estimate; no interim or optional-stopping mode."""
    boundary();ordered=validated_games(games,pairs_per_fit)
    pairs=pair_numerators(ordered)
    totals=bootstrap_totals(ordered,pairs_per_fit=pairs_per_fit,bootstrap_seed=bootstrap_seed,boundary=boundary)
    denominator=4*3*pairs_per_fit
    integer_total=sum(pairs);mean=integer_total/denominator
    interval=[float(q)/denominator for q in np.quantile(totals,[.025,.975],method='linear')]
    conclusion=('local_above_half' if interval[0]>.5 else
                'local_below_half' if interval[1]<.5 else 'interval_includes_half')
    canonical=json.dumps(ordered,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    histogram=Counter(pairs)
    by_fit={}
    for fit in FITS:
        subset=[row for row in ordered if row['fit']==fit]
        by_fit[str(fit)]={**score_summary(subset),'pairs':pairs_per_fit}
    by_seat={str(seat):score_summary([row for row in ordered if row['local_seat']==seat]) for seat in (0,1)}
    boundary()
    return {'schema':'matched_strength_paired_statistics_v1','status':'complete_fixed_sample',
        'games':len(ordered),'pairs':len(pairs),'fits':list(FITS),'pairs_per_fit':pairs_per_fit,
        'primary':{'estimand':'equal_fit_mean_of_swapped_seat_pair_mean_game_scores',
            'pair_score_denominator':4,'integer_pair_numerator_total':integer_total,
            'mean_local_game_score':mean,'mean_local_game_score_percentage':100*mean,
            'mean_minus_half':mean-.5,'confidence_level':CONFIDENCE_LEVEL,
            'ci95':interval,'ci95_percentage':[100*q for q in interval],
            'comparison_to_half':conclusion,'single_primary_comparison':True},
        'overall':score_summary(ordered),'by_fit':by_fit,'by_local_seat':by_seat,
        'pair_score_histogram':{str(numerator):histogram[numerator] for numerator in range(5)},
        'pair_score_histogram_encoding':'integer_pair_numerator_divided_by_four',
        'bootstrap':{'replicates':BOOTSTRAP_REPLICATES,'seed':bootstrap_seed,'generator':'numpy.PCG64',
            'numpy_version':np.__version__,'chunk':BOOTSTRAP_CHUNK,
            'resampling_unit':'complete_swapped_seat_pair','strata':'fit','fit_order':list(FITS),
            'pairs_drawn_per_fit_per_replicate':pairs_per_fit,'fit_weights':[1/3,1/3,1/3],
            'draw_order':'chunk_then_fit_then_replicate_then_pair',
            'percentile_probabilities':[.025,.975],'percentile_method':'linear_on_integer_totals_then_divide',
            'integer_total_denominator':denominator,'totals_shape':[BOOTSTRAP_REPLICATES],
            'totals_encoding':'little_endian_int64_C_order',
            'totals_sha256':hashlib.sha256(np.asarray(totals,dtype='<i8',order='C').tobytes(order='C')).hexdigest()},
        'verified_score_rows_sha256':hashlib.sha256(canonical).hexdigest(),
        'scope':{'fixed_complete_sample_only':True,'genuine_forfeit_losses_included':True,
            'operational_and_qualification_games_excluded_by_analyzer':True,
            'game_verification_precedes_statistics_in_analyzer':True,
            'per_fit_and_per_seat_summaries_descriptive_without_joint_superiority_claim':True,
            'bootstrap_nominal_coverage_approximate':True,'empirical_interval_can_be_degenerate':True,
            'no_additional_games_authorized_by_result':True}}
