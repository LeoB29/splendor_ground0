"""Strict cross-arm scientific identity, including signed float32 zero bits."""
from pathlib import Path
import struct
from . import common as c


def exact(left,right,path='record'):
    if type(left) is not type(right):
        raise ValueError('Identity type mismatch '+path)
    if isinstance(left,float):
        if struct.pack('<f',left)!=struct.pack('<f',right) or left!=right:
            raise ValueError('Exact float32 mismatch '+path)
    elif isinstance(left,dict):
        if left.keys()!=right.keys():
            raise ValueError('Identity field inventory mismatch '+path)
        for key in left:
            exact(left[key],right[key],path+'.'+key)
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):
            raise ValueError('Identity length mismatch '+path)
        for i,(a,b) in enumerate(zip(left,right)):
            exact(a,b,path+f'[{i}]')
    elif left!=right:
        raise ValueError('Identity mismatch '+path)


def scientific_move(row):
    # Explicit allowlist: metadata clocks/client IDs/physical placement differ.
    # Entire response and capture are retained, not just action or top move.
    fields=('game_id','spec','route','ply','actor','before','observer','result','action','trace','after','training_capture')
    return {key:row[key] for key in fields}


def target_projection(target,rows):
    projected=dict(target);projected['examples']=[]
    for example in target['examples']:
        row=rows[example['ply']]
        c.require(example['row_sha256']==c.digest(row),'Authentic own-arm target source digest')
        c.require(example['actor']==row['actor'] and example['game_id']==row['game_id'],'Target source actor/game')
        # Only this digest varies: it includes the measured clock and routing.
        # Every actual target field and its float32 bytes remains compared.
        projected['examples'].append({k:v for k,v in example.items() if k!='row_sha256'})
    return projected


def compare_cells(control,candidate,specifications,*,boundary=lambda:None):
    control,candidate=Path(control),Path(candidate);receipts=[]
    for spec in specifications:
        boundary();game=spec['game_id'];left=[];right=[]
        a=control/'games'/game;b=candidate/'games'/game
        adoc=c.read(a/'target_game.json');bdoc=c.read(b/'target_game.json')
        c.require(len(adoc['rows'])==len(bdoc['rows'])>0,'Exact complete decision count')
        for i,(aa,bb) in enumerate(zip(adoc['rows'],bdoc['rows'])):
            boundary();c.verify([aa,bb]);ar=c.read(c.resolve(aa));br=c.read(c.resolve(bb))
            exact(scientific_move(ar),scientific_move(br),f'{game}.move[{i}]');left.append(ar);right.append(br)
        exact({k:v for k,v in adoc.items() if k!='rows'},{k:v for k,v in bdoc.items() if k!='rows'},game+'.terminal_document')
        at=c.read(control/'targets'/(game+'.json'));bt=c.read(candidate/'targets'/(game+'.json'))
        exact(target_projection(at,left),target_projection(bt,right),game+'.every_target')
        receipts.append(dict(game_id=game,plies=len(left),targets=len(at['examples']),
            control=c.artifact(a/'target_game.json'),candidate=c.artifact(b/'target_game.json'),
            exact_actions_visits_values_all_targets=True))
    return dict(status='passed',games=len(receipts),receipts=receipts,numerical_tolerance=None,
        exact_float32_and_response_identity=True,
        excluded_target_fields=['row_sha256: separately authenticated source-record provenance only'],
        models=0,new_games=0)
