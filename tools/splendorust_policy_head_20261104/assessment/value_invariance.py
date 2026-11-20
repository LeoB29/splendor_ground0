"""Exact original-value probes on existing actual graph/native qualification calls."""
import math
import struct
from . import common as c

PREFIX='outputs/splendorust_selfplay_pilot_resume2_20261103_01'
SOURCES={
    'acceptance':{'path':PREFIX+'/final_closeout.json','bytes':32791,
        'sha256':'33edbb9a8858d6a34cddc1893f6c69efeefead6c0e1eabdc3bb7d5c19a8b9db7'},
    'independent_review':{'path':PREFIX+'/independent_review.json','bytes':4661,
        'sha256':'87c0a7b3f83702d68b637e2be9ead49ba6aaadd3f72ba530536872dc58a74f86'},
    'graph':{'path':PREFIX+'/assessment/workers/operational_g000_control/graph_parity.json','bytes':745798,
        'sha256':'c63290141dfcb4dabb79715561485544cd57b7bac07c085b07a0878debb4f8f4'},
    'entrypoint':{'path':PREFIX+'/assessment/workers/operational_g000_control/entrypoint_clock.json','bytes':1615933,
        'sha256':'dec5c5527ec9604aeba9ee2d7ad3d615222ba2ae3d47542a882a847d599b9f20'},
}


def sources():
    c.verify(list(SOURCES.values()))
    c.require(c.read(c.ROOT/SOURCES['acceptance']['path'])['status']=='accepted_completed_verified'
        and c.read(c.ROOT/SOURCES['independent_review']['path'])['approved'] is True,
        'Accepted original-parent value reference')
    return {k:dict(v) for k,v in SOURCES.items()}


def frozen_sources(freeze):
    c.verify([freeze['qualification_plan']])
    declared=c.read(c.ROOT/freeze['qualification_plan']['path'])['value_invariance_sources']
    c.require(declared==sources(),'Frozen original-parent value probe sources')
    return declared


def value_bits(values):
    c.require(type(values) is list and len(values)==2
        and all(type(x) is float and math.isfinite(x) for x in values),'Two finite canonical value outputs')
    return struct.pack('<ff',*values)


def require_same_values(actual,expected,label):
    c.require(value_bits(actual)==value_bits(expected) and actual==expected,
        'Exact original canonical value outputs: '+label)


def compare_graph(saved,original):
    c.require(saved['fixtures']==original['fixtures'] and len(saved['fixtures'])==11
        and len(saved['evidence'])==len(original['evidence'])==11,'Identical complete graph probe inputs')
    samples=0
    for fixture,actual,prior in zip(saved['fixtures'],saved['evidence'],original['evidence'],strict=True):
        c.require(actual['fixture_id']==prior['fixture_id']==fixture['fixture_id']
            and actual['batch']==prior['batch']==fixture['batch'],'Graph probe identity')
        batch=fixture['batch']
        for key in ('eager_outputs','candidate_outputs'):
            c.require(len(actual[key])==len(prior[key])==batch and
                all(len(row)==83 for row in actual[key]+prior[key]),'Complete graph output83 batches')
            for left,right in zip(actual[key],prior[key],strict=True):
                require_same_values(left[81:],right[81:],fixture['fixture_id']+'/'+key)
        samples+=batch
    return dict(status='passed',fixtures=11,batch_rows=samples,paths=['eager','compiled'],
        original_value_float_and_bits_exact=True,policy_equality_required=False,new_model_calls=0)


def compare_native(suite,original):
    c.require(set(suite['fresh_numeric_parity'])==set(original['fresh_numeric_parity'])=={'reference','search'},
        'Both actual native/host qualification paths')
    checks=0
    for arm in ('reference','search'):
        actual,prior=(value['fresh_numeric_parity'][arm] for value in (suite,original))
        c.require(actual['status']==prior['status']=='passed' and actual['checks']==prior['checks']==23
            and len(actual['rows'])==len(prior['rows'])==23,'Complete23 same-input parity probes')
        for left,right in zip(actual['rows'],prior['rows'],strict=True):
            c.require({k:v for k,v in left.items() if k!='response'}==
                      {k:v for k,v in right.items() if k!='response'},'Native probe input identity')
            for key in ('host_value','native_value'):
                require_same_values(left['response'][key],right['response'][key],arm+'/'+key)
            checks+=1
    return dict(status='passed',same_input_probes=checks,paths=['host','native'],
        original_value_float_and_bits_exact=True,policy_equality_required=False,new_model_calls=0)


def graph_proof(folder,freeze,*,boundary=lambda:None):
    boundary();pins=frozen_sources(freeze)
    return dict(compare_graph(c.read(folder/'graph_parity.json'),c.read(c.ROOT/pins['graph']['path'])),
        original_reference=pins['graph'])


def native_proof(suite,freeze,*,boundary=lambda:None):
    boundary();pins=frozen_sources(freeze);prior=c.read(c.ROOT/pins['entrypoint']['path'])
    c.require(prior['accepted'] is True and prior['status']=='ok'
        and prior['result']['execution_route']=='control','Accepted original-parent actual native probes')
    return dict(compare_native(suite,prior['result']),original_reference=pins['entrypoint'])
