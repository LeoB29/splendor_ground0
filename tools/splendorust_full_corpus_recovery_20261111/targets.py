"""Authenticated actor-oriented search/outcome mixtures; no model execution."""
from copy import deepcopy
import math
import struct
from . import common as c
from tools.splendorust_independent_data_20261107.collection import data as source_data

def contract(arm='mixed'):
    c.require(arm in ('terminal','mixed'), 'Fixed value-target arm')
    if arm == 'terminal':
        return dict(id='official_terminal_only_v1', terminal_weight=1.0, root_q_weight=0.,
            denominator=1., root_q_units='signed_actor_win_credit', formula='z',
            canonical_actor_slot=0, opponent_target='negative_actor_target')
    return dict(id='terminal_root_q_mix0837_v1', terminal_weight=1.0, root_q_weight=.837,
        denominator=1.837, root_q_units='signed_actor_win_credit',
        formula='(z+0.837*(2*root_value-1))/1.837', canonical_actor_slot=0,
        opponent_target='negative_actor_target')

def f32(x):
    return struct.unpack('<f',struct.pack('<f',x))[0]

def mixture(terminal, root_credit):
    c.require(type(terminal) is list and len(terminal)==2 and all(type(x) in (int,float)
        and x in (-1.,0.,1.) for x in terminal) and sum(terminal)==0., 'Official zero-sum terminal values')
    c.require(type(root_credit) in (int,float) and math.isfinite(root_credit) and 0.<=root_credit<=1.,
        'Finite actor root win credit')
    q=2.*root_credit-1.
    value=(terminal[0]+.837*q)/1.837
    c.require(-1.<=value<=1., 'Convex signed value mixture')
    return [value,-value]

def search_label(example, row, descriptor, spec):
    """Authenticate a saved actor Q for training or explicit diagnostic use."""
    decoded=source_data.validate_row(row,spec,row['before'],example['ply'])
    c.require(decoded['eligible'] is True and c.digest(row)==example['row_sha256'], 'Exact accepted source row')
    for key in ('ply','actor','tokens','policy','mask','canonical_players','observer_sha256','row_sha256','root_input_sha256'):
        c.require(c.canonical(decoded[key])==c.canonical(example[key]), 'Unchanged scientific example '+key)
    capture=row['training_capture'];native=capture['native_capture']
    c.require(capture['actor_seat']==example['actor'] and type(example['actor']) is int
        and capture['canonical_actor_slot']==0 and example['canonical_players']==[example['actor'],1-example['actor']],
        'Absolute seat maps to canonical actor slot zero')
    q=native['root_value'];visits=native['visits67'];qs=native['q67']
    expected=math.fsum(n*v for n,v in zip(visits,qs) if v is not None)/512.
    c.require(sum(visits)==512 and abs(q-expected)<=2e-6, 'Search Q is visit-weighted actor win credit')
    return dict(game_index=example['game_index'],ply=example['ply'],split=example['split'],
        row_sha256=example['row_sha256'],root_input_sha256=example['root_input_sha256'],
        actor=example['actor'],root_q_signed=2.*q-1.,root_q_win_credit=q,source_move=deepcopy(descriptor))

def attach(example, row, descriptor, spec):
    """Runs the same corrected decoder, then adds separately labelled target data."""
    c.require(example['split'] in ('train','engineering'), 'No held-out target used for fitting')
    label=search_label(example,row,descriptor,spec)
    q=label['root_q_win_credit']
    result=deepcopy(example)
    result.update(terminal_value=deepcopy(example['value']),root_q_signed=2.*q-1.,
        root_q_win_credit=q,root_q_source=deepcopy(descriptor),value_target_contract=contract(),
        value=mixture(example['value'],q))
    validate(result)
    return result

def validate(example):
    arm=example.get('value_target_arm','mixed')
    c.require(example.get('value_target_contract')==contract(arm), 'Exact prospectively fixed value objective')
    c.require(example['root_q_signed']==2.*example['root_q_win_credit']-1., 'Signed search Q conversion')
    mixed=mixture(example['terminal_value'],example['root_q_win_credit'])
    c.require(example['value']==(mixed if arm=='mixed' else example['terminal_value']), 'Exact fixed target')
    c.require(type(example['root_q_source']) is dict and set(example['root_q_source'])=={'path','bytes','sha256'}, 'Authenticated root-Q source')
    return example

def source_digest(examples):
    import hashlib
    h=hashlib.sha256();h.update(b'[')
    for index,x in enumerate(examples):
        if index:h.update(b',')
        h.update(c.canonical(dict(game_index=x['game_index'],ply=x['ply'],row_sha256=x['row_sha256'],
            root_q_source=x['root_q_source'],root_q_win_credit=x['root_q_win_credit'],terminal_value=x['terminal_value'],
            value=x['value'])).encode())
    h.update(b']');return h.hexdigest()

def for_arm(example,arm):
    out=dict(example,value_target_arm=arm,value_target_contract=contract(arm))
    out['value']=mixture(out['terminal_value'],out['root_q_win_credit']) if arm=='mixed' else list(out['terminal_value'])
    return validate(out)

def prediction_losses(row):
    """Independent saved arithmetic for continuous *training* targets only."""
    mask,target,output,value=(row[k] for k in ('legal_mask81','policy_target81','output83','value_target2'))
    c.require(len(mask)==len(target)==81 and all(type(x) is bool for x in mask) and any(mask), 'Full legal mask')
    c.require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in target)
        and abs(math.fsum(target)-1.)<=1e-12 and all(ok or t==0 for ok,t in zip(mask,target)), 'Legal policy target')
    c.require(len(value)==2 and all(type(x) in (int,float) and math.isfinite(x) and -1<=x<=1 for x in value)
        and sum(value)==0., 'Bounded continuous zero-sum target')
    c.require(len(output)==83 and all(math.isfinite(x) for x in output), 'Finite saved outputs')
    maximum=max(x for x,ok in zip(output[:81],mask) if ok)
    norm=maximum+math.log(math.fsum(math.exp(x-maximum) for x,ok in zip(output[:81],mask) if ok))
    policy=max(0.,math.fsum(t*(norm-x) for t,x in zip(target,output[:81]) if t))
    # Torch receives a float32 target; reporting must use those actual bits.
    loss=math.fsum((x-f32(y))**2 for x,y in zip(output[81:],value))/2.
    return dict(policy=policy,value=loss,combined=policy+loss)
