"""Prospective paired setup and outcome-blind reproduction design; no models."""
from itertools import product
import hashlib
import json
import subprocess
from . import common as c

DOMAIN='splendorust_network_live_20261031_v1'
REPLAY_DOMAIN=DOMAIN+'_replay_v1'
VARIANTS=('control','candidate')
PHASES=('0..19','20..39','40+')
PRIVATE=('neither','own','opponent','both')


def require(value,message):
    if not value:raise ValueError('NETWORK_LIVE_DESIGN: '+message)


def expected_seed(pair_index):
    require(type(pair_index) is int and 0<=pair_index<3,'one of three fixed pairs')
    encoded=json.dumps([DOMAIN,'operational',pair_index],separators=(',',':'))
    return str(int.from_bytes(hashlib.sha256(encoded.encode()).digest(),'big'))


def make_schedule():
    return [dict(stage='operational',candidate='candidate',pair_index=pair,
        game_index=2*pair+order,seed_decimal=expected_seed(pair),challenger_seat=(pair+order)%2,
        challenger_quota=256,opponent_quota=256,challenger_mode='research',opponent_mode='research',
        challenger_variant='candidate',opponent_variant='control',challenger_cpuct=.4,opponent_cpuct=.4)
        for pair in range(3) for order in range(2)]


def validate_spec(value):
    require(type(value) is dict and type(value.get('game_index')) is int and 0<=value['game_index']<6,
        'game identity')
    expected=make_schedule()[value['game_index']]
    require(all(k in value and type(value[k]) is type(v) and value[k]==v for k,v in expected.items()),
        'exact frozen game specification')
    return expected


def validate_schedule(values):
    require(type(values) is list and values==make_schedule(),'complete six-game order')
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


def lifecycle_inventory():
    return {name:dict(route='research',execution_variant=name.split('_')[1],
        graph_request_id='request-00000001',numerical_request_id='request-00000002',
        entrypoint_request_id='request-00000003',ordinary_first_request_id='request-00000004',
        ordinary_requests=6*c.MAX_PLIES//2 if name.startswith('live_') else 48,startup_requests=3)
        for name in c.LIFECYCLES}


def replay_rule():
    return dict(schema='network_live_replay_rule_v1',domain=REPLAY_DOMAIN,maximum=48,
        primary_cells='game_index x variant x absolute-ply phase',primary_maximum=36,
        phases=list(PHASES),secondary_cells='variant x acting seat x hidden-reserve category',
        secondary_maximum=12,secondary_order='variant then seat then neither/own/opponent/both',
        per_cell='minimum SHA256 of [domain,game_index,ply,variant]',
        duplicate_selection=False,selection_uses_outcomes=False,selection_uses_timing=False,
        failed_moves_excluded=True,within_variant_full_scientific_result_equality_required=True,
        cross_variant_output_equality_required=False)


def fixed(path,value):
    if path.exists():require(c.read(path)==value,'prepared artifact changed');return c.artifact(path)
    return c.save(path,value)


def prepare():
    schedule=make_schedule()
    initials=[dict(**spec,initial=initial_snapshot(spec)) for spec in schedule]
    for pair in range(3):require(initials[2*pair]['initial']==initials[2*pair+1]['initial'],'paired full deck/noble setup')
    return dict(schedule=fixed(c.OUT/'schedule.json',schedule),
        initial_states=fixed(c.OUT/'initial_states.json',initials),
        replay_rule=fixed(c.OUT/'replay_rule.json',replay_rule()),
        lifecycles=fixed(c.OUT/'declared_worker_lifecycles.json',lifecycle_inventory()))


def seed_inventory():
    """Finite registered-seed check, never a claim to scan arbitrary raw logs."""
    targets={int(x['seed_decimal']) for x in make_schedule()};require(len(targets)==3,'three new setups')
    found=subprocess.run(['rg','--files','--no-ignore','outputs','-g','*manifest*.json','-g','*schedule*.json',
        '-g','*seed_registry*.json','-g','*reserved_panel*.json'],cwd=c.ROOT,capture_output=True,text=True,check=True,timeout=120)
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
        if path.is_relative_to(c.OUT.resolve()):continue
        if path.stat().st_size>8*c.MiB:
            skipped.append(dict(path=path.relative_to(c.ROOT).as_posix(),bytes=path.stat().st_size));continue
        visit(c.read(path),name);checked.append(c.artifact(path))
    require(not collisions,'registered seed collision; do not replace a seed after the check')
    return dict(status='passed',domain=DOMAIN,new_setup_count=3,checked=checked,skipped_oversized=skipped,
        collisions=collisions,prior_seed_reuse=False,
        scope='Registered manifests/schedules/seed registries/reserved panels up to8MiB; excludes arbitrary logs and disclosed oversized files',
        search_randomness='unchanged canonical observer SHA256 prefix seed; no game seed or hidden deck order sent to player')


def phase(ply):
    return PHASES[0] if ply<20 else PHASES[1] if ply<40 else PHASES[2]


def select_replay_sample(rows,maximum=48):
    require(type(maximum) is int and maximum==48,'fixed48 reproduction bound')
    primary={};secondary={};seen=set();eligible=0
    for row in rows:
        spec=validate_spec(row);game=spec['game_index'];ply=row.get('ply');variant=row.get('variant')
        require(type(ply) is int and 0<=ply<c.MAX_PLIES and variant in VARIANTS,'replay identity')
        identity=(game,ply);require(identity not in seen,'duplicate played decision');seen.add(identity)
        require(type(row.get('completed')) is bool,'explicit completion')
        if not row['completed']:continue
        seat=row.get('seat');role='challenger' if seat==spec['challenger_seat'] else 'opponent'
        require(type(seat) is int and seat in (0,1) and row.get('player_role')==role and
            variant==spec[role+'_variant'] and row.get('quota')==256 and row.get('arm')=='search' and
            row.get('route')==row.get('mode')=='research','role/seat/quota replay binding')
        view=row['observer']['observation'];require(view['current_player']==seat,'acting observer seat')
        own=any(x['source']=='deck' for x in view['players'][seat]['reserves'])
        other=any(x['source']=='deck' for x in view['players'][1-seat]['reserves'])
        private='both' if own and other else 'own' if own else 'opponent' if other else 'neither'
        key=hashlib.sha256(json.dumps([REPLAY_DOMAIN,game,ply,variant],separators=(',',':')).encode()).hexdigest()
        item=dict(game_index=game,pair_index=spec['pair_index'],ply=ply,variant=variant,player_role=role,seat=seat,
            phase=phase(ply),private_category=private,selection_sha256=key,observer_sha256=c.digest(row['observer']))
        for mapping,cell in ((primary,(game,variant,phase(ply))),(secondary,(variant,seat,private))):
            if cell not in mapping or key<mapping[cell]['selection_sha256']:mapping[cell]=item
        eligible+=1
    cells=list(product(range(6),VARIANTS,PHASES));choices=[primary[cell] for cell in cells if cell in primary]
    selected={(x['game_index'],x['ply']) for x in choices};extra=[]
    for cell in product(VARIANTS,(0,1),PRIVATE):
        item=secondary.get(cell)
        if item is not None and (item['game_index'],item['ply']) not in selected and len(extra)<12:
            extra.append(item);selected.add((item['game_index'],item['ply']))
    choices+=extra;require(len(choices)<=maximum,'bounded coverage selection')
    private_cells={(x['variant'],x['seat'],x['private_category']) for x in choices}
    return dict(schema='network_live_replay_selection_v1',rule=replay_rule(),maximum_decisions=maximum,
        eligible_completed_decisions=eligible,decisions=choices,primary_selected=len(choices)-len(extra),
        secondary_selected=len(extra),empty_primary_cells=[list(cell) for cell in cells if cell not in primary],
        unrepresented_private_cells=[list(cell) for cell in sorted(secondary) if cell not in private_cells],
        selection_uses_outcomes=False,selection_uses_timing=False,failed_moves_excluded=True)
