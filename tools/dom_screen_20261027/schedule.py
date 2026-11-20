"""Fresh paired admission workload for Dom PPO512 and unchanged32."""
import hashlib
import json

DOMAIN = 'dom_screen_1s_20261027_v1'
PAIRS = 30  # Main stage; the three operational pairs are excluded from60 screened games.
OPERATIONAL_PAIRS = 3
TOTAL_PAIRS = OPERATIONAL_PAIRS + PAIRS
TOTAL_GAMES = 2 * TOTAL_PAIRS
CHALLENGER_QUOTA = None
OPPONENT_QUOTA = 32
CHALLENGER_MODE = 'dom'
OPPONENT_MODE = 'baseline'


def require(value, message):
    if not value:
        raise ValueError('DOM_SCREEN_SCHEDULE: ' + message)


def expected_seed(stage, pair_index, seed_namespace):
    require(stage in ('operational', 'main') and type(pair_index) is int and
            0 <= pair_index < (OPERATIONAL_PAIRS if stage == 'operational' else PAIRS),
            'stage/pair identity')
    encoded = json.dumps([DOMAIN, seed_namespace, stage, pair_index], separators=(',', ':'), ensure_ascii=True)
    return str(int.from_bytes(hashlib.sha256(encoded.encode('utf-8')).digest(), 'big'))


def stage_schedule(stage, seed_namespace):
    require(stage in ('operational', 'main'), 'stage identity')
    rows = []
    for pair in range(OPERATIONAL_PAIRS if stage == 'operational' else PAIRS):
        seed = expected_seed(stage, pair, seed_namespace)
        for game in (0, 1):
            rows.append(dict(stage=stage, pair_index=pair, game_index=2*pair+game,
                             challenger_seat=(pair+game) % 2, seed_decimal=seed, seed_namespace=seed_namespace,
                             challenger_quota=CHALLENGER_QUOTA, opponent_quota=OPPONENT_QUOTA,
                             challenger_mode=CHALLENGER_MODE, opponent_mode=OPPONENT_MODE))
    return rows


def generate_schedule(seed_namespace):
    require(type(seed_namespace) is str and len(seed_namespace)==64 and all(ch in '0123456789abcdef' for ch in seed_namespace), 'fresh256bit seed namespace')
    return stage_schedule('operational',seed_namespace) + stage_schedule('main',seed_namespace)

def make_schedule():
    from . import common as c
    return generate_schedule(c.read(c.OUT/'freeze.json')['seed_namespace'])


def validate_spec(value):
    """Validate every fixed field and return the canonical specification only."""
    require(type(value) is dict, 'game object')
    stage, pair, game = (value.get(key) for key in ('stage', 'pair_index', 'game_index'))
    seed_namespace=value.get('seed_namespace')
    require(type(seed_namespace) is str and len(seed_namespace)==64 and all(ch in '0123456789abcdef' for ch in seed_namespace),'seed namespace')
    seed = expected_seed(stage, pair, seed_namespace)
    require(type(game) is int and game // 2 == pair and game in (2*pair, 2*pair+1),
            'stage-global game identity')
    expected = dict(stage=stage, pair_index=pair, game_index=game,
                    challenger_seat=(pair + game % 2) % 2, seed_decimal=seed, seed_namespace=seed_namespace,
                    challenger_quota=CHALLENGER_QUOTA, opponent_quota=OPPONENT_QUOTA,
                    challenger_mode=CHALLENGER_MODE, opponent_mode=OPPONENT_MODE)
    require(all(key in value and type(value[key]) is type(wanted) and value[key] == wanted
                for key, wanted in expected.items()), 'frozen schedule field differs')
    return expected


def validate_schedule(values):
    """Require the entire prospectively ordered six-plus60 schedule."""
    require(type(values) is list and len(values) == TOTAL_GAMES == 66, 'complete66-game schedule required')
    expected = generate_schedule(values[0]["seed_namespace"])
    require([validate_spec(value) for value in values] == expected,
            'schedule order, duplication or omission')
    require(values == expected, 'unexpected schedule fields')
    return expected
