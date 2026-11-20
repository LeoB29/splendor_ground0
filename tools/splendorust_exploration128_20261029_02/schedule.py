"""Prospective common-setup development and disjoint confirmation branches."""
import hashlib
import json
DOMAIN='splendorust_exploration128_20261029_02_v1'
CANDIDATES=('low','high')
PAIRS={'operational':3,'development':50,'confirmation':200}
CPUCTS={'low':.2,'high':.8}
def require(x,message):
    if not x:raise ValueError('EXPLORATION_SCHEDULE: '+message)
def expected_seed(stage,pair_index):
    require(stage in PAIRS and type(pair_index) is int and 0<=pair_index<PAIRS[stage],'stage/pair')
    encoded=json.dumps([DOMAIN,stage,pair_index],separators=(',',':'))
    return str(int.from_bytes(hashlib.sha256(encoded.encode()).digest(),'big'))
def _spec(stage,candidate,pair,seat_index):
    ci=CANDIDATES.index(candidate)
    # Stage-global ids remain unique for the two common-setup development arms.
    index=2*pair+seat_index if stage=='confirmation' else 4*pair+2*ci+seat_index
    return dict(stage=stage,candidate=candidate,pair_index=pair,game_index=index,
      seed_decimal=expected_seed(stage,pair),challenger_seat=(pair+seat_index)%2,
      challenger_quota=128,opponent_quota=128,challenger_mode=candidate,opponent_mode='frozen',
      challenger_cpuct=CPUCTS[candidate],opponent_cpuct=.4)
def make_schedule(stage=None,candidate=None):
    if stage is None:
        require(candidate is None,'candidate requires stage')
        return make_schedule('operational')+make_schedule('development')
    require(stage in PAIRS and (candidate is None or candidate in CANDIDATES),'stage/candidate')
    require(stage!='confirmation' or candidate in CANDIDATES,'one selected confirmation branch')
    out=[]
    for pair in range(PAIRS[stage]):
        order=CANDIDATES if pair%2==0 else tuple(reversed(CANDIDATES))
        for arm in order:
            if candidate is not None and arm!=candidate:continue
            for seat in (0,1):out.append(_spec(stage,arm,pair,seat))
    return out
stage_schedule=make_schedule
def validate_spec(value):
    require(type(value) is dict,'game object')
    stage,arm,pair,index=(value.get(k) for k in ('stage','candidate','pair_index','game_index'))
    require(stage in PAIRS and arm in CANDIDATES and type(pair) is int and type(index) is int,'exact game identity types')
    expected_seed(stage,pair)
    candidates=[_spec(stage,arm,pair,i) for i in (0,1)]
    expected=next((s for s in candidates if s['game_index']==index),None)
    require(expected is not None and all(k in value and type(value[k]) is type(v) and value[k]==v for k,v in expected.items()),'frozen game specification')
    return expected
def validate_stage_schedule(values,stage,candidate=None):
    expected=make_schedule(stage,candidate)
    require(values==expected and all(validate_spec(v)==v for v in values),'exact prospective phase order')
    return expected
def validate_schedule(values,selected_candidate=None):
    expected=make_schedule()
    if selected_candidate is not None:expected+=make_schedule('confirmation',selected_candidate)
    require(values==expected and all(validate_spec(v)==v for v in values),'exact conditional combined schedule')
    return expected
