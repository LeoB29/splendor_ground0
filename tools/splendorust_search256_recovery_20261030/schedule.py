"""Prospective fresh paired schedule for one fixed256-versus128 assessment."""
import hashlib
import json

DOMAIN = 'splendorust_search256_strength_20261029_v1'
CANDIDATES = ('research',)
PAIRS = {'operational': 3, 'main': 200}
CPUCTS = {'research': .4}


def require(value, message):
    if not value:
        raise ValueError('SEARCH256_SCHEDULE: ' + message)


def expected_seed(stage, pair_index):
    require(stage in PAIRS and type(pair_index) is int and 0 <= pair_index < PAIRS[stage],
            'stage/pair')
    encoded = json.dumps([DOMAIN, stage, pair_index], separators=(',', ':'))
    return str(int.from_bytes(hashlib.sha256(encoded.encode()).digest(), 'big'))


def _spec(stage, pair, seat_index):
    return dict(stage=stage, candidate='research', pair_index=pair,
                game_index=2 * pair + seat_index, seed_decimal=expected_seed(stage, pair),
                challenger_seat=(pair + seat_index) % 2,
                challenger_quota=256, opponent_quota=128,
                challenger_mode='research', opponent_mode='frozen',
                challenger_cpuct=.4, opponent_cpuct=.4)


def make_schedule(stage=None, candidate=None):
    require(candidate in (None, 'research'), 'one preselected research candidate')
    if stage is None:
        return make_schedule('operational') + make_schedule('main')
    require(stage in PAIRS, 'stage')
    return [_spec(stage, pair, seat) for pair in range(PAIRS[stage]) for seat in (0, 1)]


stage_schedule = make_schedule


def validate_spec(value):
    """Return the canonical specification from an optionally enriched game row."""
    require(type(value) is dict, 'game object')
    stage, pair, index = (value.get(key) for key in ('stage', 'pair_index', 'game_index'))
    require(stage in PAIRS and type(pair) is int and type(index) is int,
            'exact game identity types')
    expected_seed(stage, pair)
    expected = next((row for row in (_spec(stage, pair, seat) for seat in (0, 1))
                     if row['game_index'] == index), None)
    require(expected is not None and all(key in value and type(value[key]) is type(item) and
                                        value[key] == item for key, item in expected.items()),
            'frozen game specification')
    return expected


def validate_stage_schedule(values, stage, candidate=None):
    expected = make_schedule(stage, candidate)
    require(values == expected and all(validate_spec(row) == row for row in values),
            'exact prospective phase order')
    return expected


def validate_schedule(values):
    expected = make_schedule()
    require(values == expected and all(validate_spec(row) == row for row in values),
            'exact complete operational/main schedule')
    return expected
