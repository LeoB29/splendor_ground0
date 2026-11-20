"""Assessment namespace under the single owning pilot's clock and resources."""
from .. import common as parent
ROOT, OUT = parent.ROOT, parent.OUT/'assessment'
PHASE, SCOPE = parent.PHASE, parent.SCOPE
VARIANTS=('control','candidate')
FORECAST_POLICY='advisory_only'
ROUTE_QUOTAS={'control':128,'candidate':128}
OPERATIONAL_GAMES=6
MAIN_GAMES=400 if SCOPE=='pilot' else 0
MAX_GAMES=OPERATIONAL_GAMES+MAIN_GAMES
GAMES_PER_BLOCK=10
MAX_PLIES=parent.MAX_PLIES
MAX_LIVE_DECISIONS=MAX_GAMES*MAX_PLIES
MAX_REPLAY_DECISIONS=96 if SCOPE=='pilot' else 12
MAX_WORKER_LIFECYCLES=parent.MAX_ASSESSMENT_WORKER_LIFECYCLES
MAX_SCIENTIFIC_QUALIFICATION=67*MAX_WORKER_LIFECYCLES
MAX_GRAPH_CASES=11*MAX_WORKER_LIFECYCLES
STAGE_MAP={'qualification':'assessment_qualification','operational':'operational',
           'main':'assessment','replay':'reproduction','final':'final'}
STAGES={name:parent.STAGES[target] for name,target in STAGE_MAP.items()}
for _name in ('GiB','MiB','JOB_SECONDS','COLLECTION_CUTOFF_SECONDS','INDEPENDENT_REVIEW_SECONDS',
 'INITIAL_AVAILABLE_RAM_BYTES','MINIMUM_AVAILABLE_RAM_BYTES','RAM_WAIT_SECONDS','INITIAL_FREE_DISK_BYTES',
 'MINIMUM_FREE_DISK_BYTES','MAX_EVIDENCE_BYTES','PREFINAL_EVIDENCE_BYTES','FINAL_RESERVE_BYTES',
 'MAX_RSS_BYTES','MAX_LOG_BYTES','MAX_RESPONSE_BYTES','SECONDS_PER_MOVE','INTERNAL_SECONDS'):
    globals()[_name]=getattr(parent,_name)
for _name in ('read','verify','artifact','file_hash','canonical','digest','require','now'):
    globals()[_name]=getattr(parent,_name)

def save(path,value):
    path=__import__('pathlib').Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()),'Assessment output isolation')
    return parent.save(path,value)

def atomic_save(path,value):
    path=__import__('pathlib').Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()),'Assessment output isolation')
    return parent.atomic_save(path,value)

EXECUTION_SCHEMA='splendorust_selfplay_assessment_execution_v1'
