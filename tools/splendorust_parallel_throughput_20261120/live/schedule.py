"""Sixteen fresh identical setups; arms live in paths, never in model seeds."""
from hashlib import sha256
from . import common as c

DOMAIN='splendorust_parallel_throughput_20261120_v1'

def expected_spec(index):
    c.require(type(index) is int and 0<=index<16,'Sixteen engineering setups')
    seed=str(int.from_bytes(sha256(c.canonical([DOMAIN,'live',index]).encode()).digest(),'big'))
    return dict(stage='live',pair_index=index,game_index=index,candidate_seat=0,seed_decimal=seed,
                candidate_pool=3,control_pool=3,total_simulations_each=512)

def validate_spec(spec):
    c.require(type(spec) is dict and spec==expected_spec(spec.get('game_index')),'Exact prospective engineering setup')
    return spec

def validate_block(specs):
    c.require(type(specs) is list and bool(specs),'Nonempty fixed live work')
    for spec in specs:validate_spec(spec)
    c.require(len({s['game_index'] for s in specs})==len(specs),'No duplicated live setup')
    return 'live'

def phase_specs(phase):
    c.require(phase in (0,1,2,3),'Four frozen counterbalanced phases')
    offset=0 if phase<2 else 8
    return [expected_spec(i) for i in range(offset,offset+8)]

def phase_arm(phase):
    return ('A1','A4','A4','A1')[phase]
