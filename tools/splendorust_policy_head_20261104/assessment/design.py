"""Prospective paired setup and outcome-blind reproduction design; no models."""
from itertools import product
import hashlib
import json
import subprocess
from . import common as c

DOMAIN='splendorust_policy_head_20261104_'+c.SCOPE+'_assessment_v1'
REPLAY_DOMAIN=DOMAIN+'_replay_v1'
BOOTSTRAP_SEED=20261104128
PRIOR_ASSESSMENT_SCHEDULE=c.ROOT/'outputs/splendorust_selfplay_pilot_resume2_20261103_01/assessment/schedule.json'
PAIRS={'operational':3,'main':c.MAIN_GAMES//2}
VARIANTS=('control','candidate')
PHASES=('0..19','20..39','40+')
PRIVATE=('neither','own','opponent','both')


def require(value,message):
    if not value:raise ValueError('SELFPLAY_ASSESSMENT_DESIGN: '+message)


def expected_seed(stage,pair_index):
    require(stage in PAIRS and type(pair_index) is int and 0<=pair_index<PAIRS[stage],'fixed stage/pair')
    encoded=json.dumps([DOMAIN,stage,pair_index],separators=(',',':'))
    return str(int.from_bytes(hashlib.sha256(encoded.encode()).digest(),'big'))


def _spec(stage,pair,order):
    return dict(stage=stage,candidate='candidate',pair_index=pair,game_index=2*pair+order,
        seed_decimal=expected_seed(stage,pair),challenger_seat=(pair+order)%2,
        challenger_quota=128,opponent_quota=128,challenger_mode='candidate',opponent_mode='control',
        challenger_variant='candidate',opponent_variant='control',challenger_cpuct=.4,opponent_cpuct=.4)


def make_schedule(stage=None):
    if stage is None:return make_schedule('operational')+make_schedule('main')
    require(stage in PAIRS,'fixed stage')
    return [_spec(stage,pair,order) for pair in range(PAIRS[stage]) for order in (0,1)]


def validate_spec(value):
    require(type(value) is dict,'game object')
    stage,pair,index=(value.get(k) for k in ('stage','pair_index','game_index'))
    require(stage in PAIRS and type(pair) is int and type(index) is int and index in (2*pair,2*pair+1),'game identity')
    expected=_spec(stage,pair,index%2)
    require(all(k in value and type(value[k]) is type(v) and value[k]==v for k,v in expected.items()),
        'exact frozen game specification')
    return expected


def validate_schedule(values,stage=None):
    require(type(values) is list and values==make_schedule(stage),'complete fixed prospective order')
    for value in values:require(validate_spec(value)==value,'no enriched schedule records')
    return values


def initial_snapshot(spec):
    from tools.external_strength_screen_20261018 import rules
    from tools.future_training_rules_20261021 import decode_training_snapshot
    validate_spec(spec)
    value=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
    decode_training_snapshot(value)
    require(not value['state']['terminal'] and value['state']['turn_index']==0,'fresh corrected initial state')
    return value


def replay_rule():
    return dict(schema='selfplay_assessment_replay_rule_v1',domain=REPLAY_DOMAIN,maximum=c.MAX_REPLAY_DECISIONS,
        main_primary_cells='variant x acting seat x absolute-ply phase x hidden-reserve category',main_primary_maximum=48 if c.SCOPE=='pilot' else 0,
        operational_cells='game_index x variant',operational_maximum=12,
        main_temporal_cells='pair-index third x variant x acting seat x absolute-ply phase',main_temporal_maximum=36 if c.SCOPE=='pilot' else 0,
        pair_index_thirds=[[0,66],[67,133],[134,199]] if c.SCOPE=='pilot' else [],phases=list(PHASES),private_categories=list(PRIVATE),
        per_cell='minimum SHA256 of [domain,stage,pair_index,game_index,ply,variant]',
        selection_order='main primary, operational, main temporal; fixed Cartesian cell order; deduplicate identities',
        duplicate_selection=False,selection_uses_outcomes=False,selection_uses_timing=False,
        failed_moves_excluded=True,all_available_cells_selected=True,
        within_variant_full_scientific_result_equality_required=True,cross_variant_output_equality_required=False)


def fixed(path,value):
    if path.exists():require(c.read(path)==value,'prepared artifact changed');return c.artifact(path)
    return c.save(path,value)


def prepare():
    schedule=make_schedule()
    initials=[dict(**spec,initial=initial_snapshot(spec)) for spec in schedule]
    for index in range(0,len(initials),2):
        require(initials[index]['initial']==initials[index+1]['initial'],'paired full deck/noble setup')
    return dict(schedule=fixed(c.OUT/'schedule.json',schedule),
        initial_states=fixed(c.OUT/'initial_states.json',initials),replay_rule=fixed(c.OUT/'replay_rule.json',replay_rule()))


def seed_inventory():
    """Finite registered-seed check, not a claim to scan arbitrary raw logs."""
    game_seeds={int(x['seed_decimal']) for x in make_schedule()}
    require(len(game_seeds)==sum(PAIRS.values()) and BOOTSTRAP_SEED not in game_seeds,'complete scoped new setups and distinct analysis seed')
    targets=game_seeds|({BOOTSTRAP_SEED} if c.SCOPE=='pilot' else set())
    prior=c.read(PRIOR_ASSESSMENT_SCHEDULE)
    require(len(prior)==406 and len({x['seed_decimal'] for x in prior})==203,
        'Complete previous fixed pilot assessment schedule')
    require(not game_seeds&{int(x['seed_decimal']) for x in prior},'Fresh policy-head study disjoint previous pilot')
    previous=c.artifact(PRIOR_ASSESSMENT_SCHEDULE)
    found=subprocess.run(['rg','--files','--no-ignore','outputs','-g','*manifest*.json','-g','*schedule*.json',
        '-g','*seed_registry*.json','-g','*reserved_panel*.json','-g','*freeze*.json'],
        cwd=c.ROOT,capture_output=True,text=True,check=True,timeout=120)
    checked=[];skipped=[];collisions=[]
    def visit(value,name):
        if type(value) is int or type(value) is str and value.isdecimal():
            if int(value) in targets:collisions.append(name)
        elif type(value) is list:
            for item in value:visit(item,name)
        elif type(value) is dict:
            for item in value.values():visit(item,name)
    for name in sorted(found.stdout.splitlines()):
        path=(c.ROOT/name).resolve()
        if path.is_relative_to(c.parent.OUT.resolve()):continue
        if path.stat().st_size>8*c.MiB:
            skipped.append(dict(path=path.relative_to(c.ROOT).as_posix(),bytes=path.stat().st_size));continue
        visit(c.read(path),name);checked.append(c.artifact(path))
    require(not collisions,'registered seed collision; do not replace a seed after the check')
    return dict(status='passed',domain=DOMAIN,new_setup_count=len(game_seeds),bootstrap_seed=BOOTSTRAP_SEED if c.SCOPE=='pilot' else None,
        checked=checked,skipped_oversized=skipped,collisions=collisions,prior_seed_reuse=False,
        explicitly_disjoint_previous_pilot=previous,
        scope='Registered manifests/schedules/seed registries/reserved panels/freezes up to8MiB; excludes arbitrary logs and disclosed oversized files',
        search_randomness='unchanged canonical observer SHA256 prefix seed; no game seed or hidden deck order sent to player')


def phase(ply):return PHASES[0] if ply<20 else PHASES[1] if ply<40 else PHASES[2]


def temporal_block(pair):return 0 if pair<67 else 1 if pair<134 else 2


def select_replay_sample(rows,maximum=None):
    """Single streaming pass; retain only compact minima and identity keys."""
    if maximum is None:maximum=c.MAX_REPLAY_DECISIONS
    require(type(maximum) is int and maximum==c.MAX_REPLAY_DECISIONS,'fixed scope-specific reproduction bound')
    primary={};ops={};temporal={};seen=set();eligible=0;failed=0
    for row in rows:
        spec=validate_spec(row);stage=spec['stage'];game=spec['game_index'];pair=spec['pair_index']
        ply=row.get('ply');variant=row.get('variant')
        require(type(ply) is int and 0<=ply<c.MAX_PLIES and variant in VARIANTS,'replay identity')
        identity=(stage,game,ply);require(identity not in seen,'duplicate played decision');seen.add(identity)
        require(type(row.get('completed')) is bool,'explicit completion')
        if not row['completed']:failed+=1;continue
        seat=row.get('seat');role='challenger' if seat==spec['challenger_seat'] else 'opponent'
        require(type(seat) is int and seat in (0,1) and row.get('player_role')==role and variant==spec[role+'_variant'] and
            type(row.get('quota')) is int and row['quota']==c.ROUTE_QUOTAS[variant] and row.get('arm')=='search' and
            row.get('route')==row.get('mode')==variant,'role/seat/quota replay binding')
        view=row['observer']['observation'];require(view['current_player']==seat,'acting observer seat')
        own=any(x['source']=='deck' for x in view['players'][seat]['reserves'])
        other=any(x['source']=='deck' for x in view['players'][1-seat]['reserves'])
        private='both' if own and other else 'own' if own else 'opponent' if other else 'neither'
        key=hashlib.sha256(json.dumps([REPLAY_DOMAIN,stage,pair,game,ply,variant],separators=(',',':')).encode()).hexdigest()
        item=dict(stage=stage,candidate='candidate',game_index=game,pair_index=pair,ply=ply,variant=variant,
            player_role=role,seat=seat,quota=c.ROUTE_QUOTAS[variant],route=variant,phase=phase(ply),
            private_category=private,selection_sha256=key,observer_sha256=c.digest(row['observer']))
        maps=[(ops,(game,variant))] if stage=='operational' else [
            (primary,(variant,seat,phase(ply),private)),(temporal,(temporal_block(pair),variant,seat,phase(ply)))]
        for mapping,cell in maps:
            if cell not in mapping or key<mapping[cell]['selection_sha256']:mapping[cell]=item
        eligible+=1
    groups=[('main_primary',primary,list(product(VARIANTS,(0,1),PHASES,PRIVATE)) if c.SCOPE=='pilot' else []),
        ('operational',ops,list(product(range(6),VARIANTS))),
        ('main_temporal',temporal,list(product(range(3),VARIANTS,(0,1),PHASES)) if c.SCOPE=='pilot' else [])]
    decisions=[];selected=set();coverage={}
    for name,mapping,cells in groups:
        for cell in cells:
            item=mapping.get(cell)
            if item is None:continue
            identity=(item['stage'],item['game_index'],item['ply'])
            if identity not in selected:decisions.append(item);selected.add(identity)
        coverage[name]=dict(available_cells=len(mapping),maximum_cells=len(cells),
            empty_cells=[list(cell) for cell in cells if cell not in mapping],
            cells=[dict(cell=list(cell),stage=mapping[cell]['stage'],game_index=mapping[cell]['game_index'],
                ply=mapping[cell]['ply']) for cell in cells if cell in mapping])
    require(len(decisions)<=maximum,'bounded coverage selection')
    return dict(schema='selfplay_assessment_replay_selection_v1',rule=replay_rule(),maximum_decisions=maximum,
        eligible_completed_decisions=eligible,excluded_failed_decisions=failed,decisions=decisions,coverage=coverage,
        selection_uses_outcomes=False,selection_uses_timing=False,failed_moves_excluded=True)
