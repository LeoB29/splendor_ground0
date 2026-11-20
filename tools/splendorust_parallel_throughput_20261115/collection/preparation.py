"""Model-free plan derivation from the accepted collection4096 route."""
from copy import deepcopy
from . import common as c, collector, qualification, backend, storage


def build_plan(specifications,engineering_games=None):
    c.require(len(specifications)==32 and len({x['seed_decimal'] for x in specifications})==32,'Exactly32 fresh matched setups')
    c.require(all(x.get('purpose')=='excluded_parallel_throughput_engineering_only' for x in specifications),'Excluded engineering setups only')
    if engineering_games is None:
        from ..schedule import qualification_specs
        engineering_games=qualification_specs()
    prior_path=c.ROOT/'outputs/splendorust_collection4096_20261115_01/collection/plan.json'
    prior=c.read(prior_path)
    keys=('runtime','cases','source_guard','native_capture','backend_source_equivalence')
    result={key:deepcopy(prior[key]) for key in keys}
    result.update(schema='parallel_throughput_collection_v1',accepted_plan=c.artifact(prior_path),
        qualification=qualification.build_plan(),engineering_games=deepcopy(engineering_games),storage=storage.contract(),packing=collector.packing_contract(),
        source_equivalence=backend.source_equivalence(),capture_provenance=backend.capture_provenance(),
        internal_seconds=60.,outer_seconds=75.,execution_policy_id=c.EXECUTION_POLICY_ID,
        teacher=dict(simulations=512,worlds=3,cpuct=.4,depth=64,precision='float32'),
        games=dict(specifications=deepcopy(specifications),maximum_plies=2000,retries=0,replacements=0,training_updates=0),
        cells={key:dict(cell_id=key,route='cpu',concurrency=8,scheduler='original8',implementation='eager',
            specifications=deepcopy(specifications[i*16:(i+1)*16])) for i,key in enumerate(('X','Y'))},
        order=['control_X','candidate_X','candidate_Y','control_Y'],excluded_from_every_training_corpus=True,
        control_processes=1,control_clients=8,candidate_processes=4,candidate_clients=2)
    return result


def validate_plan(plan):
    c.require(plan==build_plan(plan['games']['specifications'],plan['engineering_games']),'Exact separately frozen accepted collection plan')
