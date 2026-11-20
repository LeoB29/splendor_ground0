"""Fresh, fixed paired workload and strict identities; no simulation."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from tools.splendorust_search256_recovery_20261030_02 import schedule, analysis


def test_exact_workload_and_counterbalanced_pairs():
    ops, main = schedule.make_schedule('operational'), schedule.make_schedule('main')
    assert len(ops) == 6 and len(main) == 400
    assert schedule.validate_schedule(ops + main) == ops + main
    assert schedule.CANDIDATES == ('research',)
    assert analysis.REPLICATES == 50000 and analysis.SEED == 20261029256
    for stage, rows, pairs in [('operational', ops, 3), ('main', main, 200)]:
        assert schedule.validate_stage_schedule(rows, stage) == rows
        assert len({row['seed_decimal'] for row in rows}) == pairs
        for pair in range(pairs):
            a, b = rows[2*pair:2*pair+2]
            assert a['seed_decimal'] == b['seed_decimal']
            assert (a['challenger_seat'], b['challenger_seat']) == (pair % 2, 1-pair % 2)
            assert (a['game_index'], b['game_index']) == (2*pair, 2*pair+1)
        assert all(row['challenger_quota'] == 256 and row['opponent_quota'] == 128 and
                   row['challenger_cpuct'] == row['opponent_cpuct'] == .4 for row in rows)
    assert not {row['seed_decimal'] for row in ops} & {row['seed_decimal'] for row in main}


def test_fresh_from_both_exploration_reserved_branches():
    root = Path(__file__).resolve().parents[1]
    fresh = {row['seed_decimal'] for row in schedule.make_schedule()}
    for attempt in ('01', '02'):
        old = json.loads((root / f'outputs/splendorust_exploration128_20261029_{attempt}/schedule.json').read_text())
        prior = {row['seed_decimal'] for stage in ('operational', 'development') for row in old[stage]}
        prior |= {row['seed_decimal'] for branch in old['confirmation'].values() for row in branch}
        assert len(prior) == 253 and not fresh & prior
    assert len(fresh) == 203 and str(analysis.SEED) not in fresh


@pytest.mark.parametrize('field,value', [('candidate','low'),('challenger_quota',128),
    ('opponent_quota',64),('challenger_cpuct',.2),('opponent_cpuct',.8),('game_index',True),
    ('pair_index',True),('challenger_seat',False),('seed_decimal','1'),('stage','development')])
def test_wrong_or_cross_attempt_identity_rejected(field,value):
    row = deepcopy(schedule.make_schedule('main')[0]);row[field] = value
    with pytest.raises(ValueError):schedule.validate_spec(row)


def test_no_alternate_branch_or_partial_reordered_schedule():
    for stage in ('development','confirmation','unknown'):
        with pytest.raises(ValueError):schedule.make_schedule(stage)
    with pytest.raises(ValueError):schedule.make_schedule('main','low')
    rows = schedule.make_schedule()
    for bad in (rows[:-1],rows[::-1],rows+rows[:1]):
        with pytest.raises(ValueError):schedule.validate_schedule(bad)
