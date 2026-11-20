"""Prospective fresh setup pairs; qualification games never enter assessment."""
from hashlib import sha256
from . import common as c
from tools.external_strength_screen_20261018 import rules
from tools.future_training_rules_20261021 import decode_training_snapshot

DOMAIN='splendorust_sixworld512_sequential_20261114_v1'
COUNTS={'operational':3,'main':1800}


def expected_seed(stage,pair_index):
    c.require(stage in COUNTS and type(pair_index) is int and 0<=pair_index<COUNTS[stage],
              'Frozen stage/pair identity')
    return str(int.from_bytes(sha256(c.canonical([DOMAIN,stage,pair_index]).encode()).digest(),'big'))


def expected_spec(stage,pair_index,leg):
    c.require(type(leg) is int and leg in (0,1),'Two swapped-seat legs')
    return dict(stage=stage,pair_index=pair_index,game_index=2*pair_index+leg,
                candidate_seat=(pair_index+leg)%2,seed_decimal=expected_seed(stage,pair_index),
                candidate_pool=6,control_pool=3,total_simulations_each=512)


def validate_spec(spec):
    c.require(type(spec) is dict and type(spec.get('game_index')) is int,'Explicit game specification')
    wanted=expected_spec(spec.get('stage'),spec.get('pair_index'),spec['game_index']%2)
    c.require(c.canonical(spec)==c.canonical(wanted),'Exact prospective game specification')
    return wanted


def stage_schedule(stage):
    c.require(stage in COUNTS,'Known schedule stage')
    return [expected_spec(stage,pair,leg) for pair in range(COUNTS[stage]) for leg in (0,1)]


def build_schedule():
    return {stage:stage_schedule(stage) for stage in COUNTS}


def validate_stage_schedule(specs,stage):
    c.require(type(specs) is list and specs==stage_schedule(stage),'Whole fixed schedule, no missing/extra/reordered games')
    for spec in specs:validate_spec(spec)
    return specs


def validate_block(specs):
    c.require(type(specs) is list and bool(specs),'Nonempty block')
    for spec in specs:validate_spec(spec)
    stage=specs[0]['stage']
    c.require(all(s['stage']==stage for s in specs),'One stage per block')
    if stage=='operational':
        validate_stage_schedule(specs,stage)
    else:
        start=specs[0]['game_index']
        c.require(start%400==0 and len(specs)==400 and
                  specs==stage_schedule('main')[start:start+400],
                  'Exactly one prospective 400-game block ending on complete pairs')
    return stage


def freeze_initials(folder,schedules):
    """Caller authenticates runtime sources once before this model-free helper."""
    snapshots={};seen=set()
    for stage in COUNTS:
        validate_stage_schedule(schedules[stage],stage)
        for pair in range(COUNTS[stage]):
            seed=expected_seed(stage,pair)
            c.require(seed not in seen,'Fresh unique setup seed across both stages');seen.add(seed)
            snapshot=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(seed)).initial_state())
            decode_training_snapshot(snapshot)
            item=c.save_gzip(folder/stage/f'p{pair:04d}.json.gz',snapshot)
            snapshots[f'{stage}:{pair}']=dict(artifact=item,snapshot_sha256=c.digest(snapshot),seed_decimal=seed)
    return dict(schema='sixworld512_initial_setup_inventory_v1',domain=DOMAIN,
                independent_setups=len(snapshots),snapshots=snapshots,
                identical_setup_within_each_swapped_pair=True)


def verify_seed_freshness(schedules,prior_schedule_artifacts):
    """Explicit prior inventory is frozen by the owner; no recursive corpus scan."""
    fresh={s['seed_decimal'] for stage in COUNTS for s in validate_stage_schedule(schedules[stage],stage)}
    prior=set()
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('seed_decimal','seed') and type(item) in (int,str):prior.add(str(item))
                elif isinstance(item,(list,dict)):visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    c.verify(prior_schedule_artifacts)
    for item in prior_schedule_artifacts:visit(c.read(c.resolve(item)))
    c.require(not fresh.intersection(prior),'Fresh setup seeds overlap prior frozen schedule')
    return dict(status='passed',domain=DOMAIN,fresh_seeds=len(fresh),prior_seeds=len(prior),
                prior_schedules=prior_schedule_artifacts,intersection=0)
