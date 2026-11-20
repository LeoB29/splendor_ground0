"""Fresh phase-specific schedule; operational games never enter assessment."""
from . import common as c
import hashlib
import json

DOMAIN = 'splendorust_pool6_live_1s_20261027_v1'
PAIRS = 100
OPERATIONAL_PAIRS = 3
TOTAL_PAIRS = OPERATIONAL_PAIRS if c.PHASE=='qualification' else PAIRS
TOTAL_GAMES = 2 * TOTAL_PAIRS
CHALLENGER_QUOTA = 64
OPPONENT_QUOTA = 64
CHALLENGER_MODE = 'research'
OPPONENT_MODE = 'frozen'


def require(value, message):
    if not value:
        raise ValueError('ISOLATED_RELIABILITY_SCHEDULE: ' + message)


def expected_seed(stage, pair_index):
    require(stage in ('operational', 'main') and type(pair_index) is int and
            0 <= pair_index < (OPERATIONAL_PAIRS if stage == 'operational' else PAIRS),
            'stage/pair identity')
    encoded = json.dumps([DOMAIN, stage, pair_index], separators=(',', ':'), ensure_ascii=True)
    return str(int.from_bytes(hashlib.sha256(encoded.encode('utf-8')).digest(), 'big'))


def stage_schedule(stage):
    require(stage in ('operational', 'main'), 'stage identity')
    rows = []
    for pair in range(OPERATIONAL_PAIRS if stage == 'operational' else PAIRS):
        seed = expected_seed(stage, pair)
        for game in (0, 1):
            rows.append(dict(stage=stage, pair_index=pair, game_index=2*pair+game,
                             challenger_seat=(pair+game) % 2, seed_decimal=seed,
                             challenger_quota=CHALLENGER_QUOTA, opponent_quota=OPPONENT_QUOTA,
                             challenger_mode=CHALLENGER_MODE, opponent_mode=OPPONENT_MODE))
    return rows


def make_schedule():
    return stage_schedule('operational' if c.PHASE=='qualification' else 'main')


def validate_spec(value):
    """Validate every fixed field and return the canonical specification only."""
    require(type(value) is dict, 'game object')
    stage, pair, game = (value.get(key) for key in ('stage', 'pair_index', 'game_index'))
    seed = expected_seed(stage, pair)
    require(type(game) is int and game // 2 == pair and game in (2*pair, 2*pair+1),
            'stage-global game identity')
    expected = dict(stage=stage, pair_index=pair, game_index=game,
                    challenger_seat=(pair + game % 2) % 2, seed_decimal=seed,
                    challenger_quota=CHALLENGER_QUOTA, opponent_quota=OPPONENT_QUOTA,
                    challenger_mode=CHALLENGER_MODE, opponent_mode=OPPONENT_MODE)
    require(all(key in value and type(value[key]) is type(wanted) and value[key] == wanted
                for key, wanted in expected.items()), 'frozen schedule field differs')
    return expected


def validate_schedule(values):
    """Require the entire prospectively ordered phase-specific schedule."""
    require(type(values) is list and len(values) == TOTAL_GAMES, 'complete phase-specific game schedule required')
    expected = make_schedule()
    require([validate_spec(value) for value in values] == expected,
            'schedule order, duplication or omission')
    require(values == expected, 'unexpected schedule fields')
    return expected
