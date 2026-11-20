"""Prospective paired identities, fresh seed domain and exact runtime modes."""
from copy import deepcopy
import hashlib
import json
import pytest

from tools.splendorust_graph64_strength_20261025 import schedule
from tools.splendorust_search_scale_opt_20261024 import schedule as previous


def test_full_schedule_fresh_seeds_swapped_seats_and_exact_player_modes():
    rows = schedule.make_schedule()
    assert schedule.DOMAIN == 'splendorust_graph64_vs_eager32_1s_20261025_v1'
    assert len(rows) == 206 and len({row['seed_decimal'] for row in rows}) == 103
    assert not {row['seed_decimal'] for row in rows} & {row['seed_decimal'] for row in previous.make_schedule()}
    assert schedule.validate_schedule(rows) == rows
    for stage, pairs in (('operational', 3), ('main', 100)):
        selected = schedule.stage_schedule(stage)
        assert len(selected) == 2*pairs
        for pair in range(pairs):
            a, b = selected[2*pair:2*pair+2]
            encoded = json.dumps([schedule.DOMAIN, stage, pair], separators=(',', ':')).encode()
            seed = str(int.from_bytes(hashlib.sha256(encoded).digest(), 'big'))
            assert a['seed_decimal'] == b['seed_decimal'] == seed
            assert (a['challenger_seat'], b['challenger_seat']) == (pair % 2, 1-pair % 2)
            assert all(row['challenger_mode'] == 'candidate' and row['challenger_quota'] == 64 and
                       row['opponent_mode'] == 'baseline' and row['opponent_quota'] == 32 for row in (a, b))
            assert schedule.validate_spec(a) == a and schedule.validate_spec(b) == b


@pytest.mark.parametrize('stage,pair', [('main', True), ('main', -1), ('main', 100),
                                      ('operational', 3), ('other', 0), ('main', 1.)])
def test_invalid_stage_pair_rejected(stage, pair):
    with pytest.raises(ValueError): schedule.expected_seed(stage, pair)


@pytest.mark.parametrize('field,value', [('challenger_mode', 'baseline'), ('opponent_mode', 'candidate'),
    ('challenger_quota', 32), ('opponent_quota', 64), ('seed_decimal', '1'), ('game_index', True),
    ('challenger_seat', True)])
def test_game_identity_cannot_switch_graph_to32_or_change_fixed_fields(field, value):
    spec = schedule.stage_schedule('main')[0]
    spec[field] = value
    with pytest.raises(ValueError): schedule.validate_spec(spec)


@pytest.mark.parametrize('change', ['missing', 'duplicated', 'reordered', 'extra_field'])
def test_entire_schedule_is_complete_ordered_and_immutable(change):
    rows = deepcopy(schedule.make_schedule())
    if change == 'missing': rows.pop()
    elif change == 'duplicated': rows[-1] = deepcopy(rows[-2])
    elif change == 'reordered': rows[0], rows[1] = rows[1], rows[0]
    else: rows[0]['new_switch'] = True
    with pytest.raises(ValueError): schedule.validate_schedule(rows)
