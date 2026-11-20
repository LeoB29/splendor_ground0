"""Fixed repetition namespace, sharing the owning study's limits and inputs."""
import os
from pathlib import Path
from .. import common as parent

_rep = os.environ.get('INDEPENDENT_DATA_REPETITION')
parent.require(_rep in ('0', '1'), 'INDEPENDENT_DATA_REPETITION must be exactly 0 or 1')
REPETITION = int(_rep)
ROOT, STUDY_OUT = parent.ROOT, parent.OUT
SOURCE_OUT = parent.SOURCE_OUT
OUT = STUDY_OUT / 'strength' / f'r{REPETITION}'
PHASE = SCOPE = 'pilot'
VARIANTS = ('control', 'candidate')
FORECAST_POLICY = 'advisory_only'
ROUTE_QUOTAS = {'control': 128, 'candidate': 128}
OPERATIONAL_GAMES, MAIN_GAMES = 6, 200
MAX_GAMES = OPERATIONAL_GAMES + MAIN_GAMES
GAMES_PER_BLOCK = 10
MAX_PLIES = parent.MAX_PLIES
MAX_LIVE_DECISIONS = MAX_GAMES * MAX_PLIES
MAX_REPLAY_DECISIONS = 96
MAX_WORKER_LIFECYCLES = 406
MAX_SCIENTIFIC_QUALIFICATION = 67 * MAX_WORKER_LIFECYCLES
MAX_GRAPH_CASES = 11 * MAX_WORKER_LIFECYCLES
STAGE_MAP = {'qualification': 'strength', 'operational': 'strength',
             'main': 'strength', 'replay': 'strength_replay', 'final': 'final'}
STAGES = {name: parent.STAGES[target] for name, target in STAGE_MAP.items()}
for _name in ('GiB', 'MiB', 'JOB_SECONDS', 'COLLECTION_CUTOFF_SECONDS', 'INDEPENDENT_REVIEW_SECONDS',
              'INITIAL_AVAILABLE_RAM_BYTES', 'MINIMUM_AVAILABLE_RAM_BYTES', 'RAM_WAIT_SECONDS',
              'INITIAL_FREE_DISK_BYTES', 'MINIMUM_FREE_DISK_BYTES', 'MAX_EVIDENCE_BYTES',
              'PREFINAL_EVIDENCE_BYTES', 'FINAL_RESERVE_BYTES', 'MAX_RSS_BYTES', 'MAX_LOG_BYTES',
              'MAX_RESPONSE_BYTES', 'SECONDS_PER_MOVE', 'INTERNAL_SECONDS', 'MAX_REQUEST_BYTES'):
    globals()[_name] = getattr(parent, _name)
for _name in ('read', 'verify', 'artifact', 'file_hash', 'canonical', 'digest', 'require', 'now'):
    globals()[_name] = getattr(parent, _name)


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT), 'Fixed repetition output only')
    return parent.save(path, value)


def atomic_save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT), 'Fixed repetition output only')
    return parent.atomic_save(path, value)


EXECUTION_SCHEMA = 'splendorust_independent_data_execution_v1'

# The owning collector has a separate larger tree bound; strength retains two GiB.
MAX_RSS_BYTES = 2 * GiB
