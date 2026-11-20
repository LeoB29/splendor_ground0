"""Outcome-blind complete-pass order and authenticated memory-mapped inputs."""
from copy import deepcopy
from collections.abc import Sequence
from . import common as c, targets
from tools.splendorust_independent_data_20261107 import sampling as original
from tools.splendorust_independent_data_20261107 import corpus as original_corpus
from tools.splendorust_policy_head_20261104 import training as accepted_training

STUDY=DOMAIN='splendorust_full_corpus_20261110_v1'
IDENTITY=deepcopy(original.IDENTITY)
SEEDS=original.SEEDS
BOOTSTRAP_SEED=2026111057022
ARMS=('terminal','mixed')
GAME_COUNTS=original.GAME_COUNTS
SCHEMA='full_corpus_selection_v1'
TRAIN_ROOTS=57022
TOTAL_ROOTS=64108
BATCH=32
STEPS=1782
_sha=original._sha
root_identity=original.root_identity
require,canonical,digest=c.require,c.canonical,c.digest
_VALIDATED={}

def fit_identity(repetition,breadth,scope='joint'):
    require(type(repetition) is int and repetition in (0,1) and breadth in ARMS and scope=='joint','Fixed full-corpus joint fit')
    return dict(fit_id=f'r{repetition}_{breadth}_joint',repetition=repetition,breadth=breadth,scope=scope)

def fits():
    return [fit_identity(r,a) for r in (0,1) for a in ARMS]

def build_selection(games,receipt):
    original_corpus.authenticate_selection_source(dict(games=games,source_receipt=receipt))
    roots=[x for g in games if g['spec']['split']=='train' for x in g['roots']]
    refs=[[x['game_index'],x['ply']] for x in roots]
    require(len(refs)==TRAIN_ROOTS and len(set(map(tuple,refs)))==TRAIN_ROOTS,'All 57022 unique accepted training roots')
    counts={s:sum(len(g['roots']) for g in games if g['spec']['split']==s) for s in GAME_COUNTS}
    require(counts==dict(train=57022,validation=3533,offline=3553),'Exact complete accepted corpus')
    train_ids=[g['spec']['game_index'] for g in games if g['spec']['split']=='train']
    cells=[]
    for rep in (0,1):
        order=sorted(range(TRAIN_ROOTS),key=lambda i:(digest([DOMAIN,'full_pass',SEEDS[rep],refs[i]]),i))
        batches=[order[i:i+BATCH] for i in range(0,TRAIN_ROOTS,BATCH)]
        require(len(batches)==STEPS and len(batches[-1])==30 and all(len(x)==32 for x in batches[:-1]),'One complete pass with last 30')
        for arm in ARMS:
            cells.append(dict(**fit_identity(rep,arm),cell_id=f'r{rep}_{arm}',game_ids=train_ids,root_ids=refs,
                permutations=[order],batches=batches,probe_batches=[],roots=TRAIN_ROOTS,optimizer_steps=STEPS,
                presentations=TRAIN_ROOTS,rng_seed=SEEDS[rep],training_weights='one_per_record',
                batch_reduction='mean_actual_batch_size',last_batch_size=30))
    panel=sorted(sorted(refs,key=lambda x:(digest([DOMAIN,'diagnostic_train_panel',x]),x))[:3072])
    return dict(schema=SCHEMA,study_id=STUDY,**IDENTITY,source_receipt=deepcopy(receipt),games=deepcopy(games),
        original_selection=c.artifact(c.SOURCE_SELECTION),source_acceptance=c.artifact(c.SOURCE_ACCEPTANCE),
        source_closeout=c.artifact(c.SOURCE_CLOSEOUT),cells=cells,game_counts=dict(GAME_COUNTS),root_counts=counts,
        heldout_game_ids=[g['spec']['game_index'] for g in games if g['spec']['split']!='train'],
        diagnostic_train_root_ids=panel,selection_uses_outcomes=False,selection_uses_targets=False,
        value_target_contracts={arm:targets.contract(arm) for arm in ARMS},historical_data_relabelled=False,
        heldout_reuse='previously_examined_diagnostic_only_fresh_live_primary')

def prepare_selection(*,boundary=lambda:None):
    require(c.read(c.SOURCE_ACCEPTANCE)['approved'] is True and c.read(c.SOURCE_CLOSEOUT)['status']=='completed_verified','Accepted completed corrected source study')
    old=c.read(c.SOURCE_SELECTION);original.validate_selection(old,boundary=boundary)
    return build_selection(old['games'],old['source_receipt'])

def validate_selection(selection,*,authenticate=True,boundary=lambda:None):
    key=digest(selection)
    if key not in _VALIDATED:
        expected=prepare_selection(boundary=boundary)
        require(canonical(selection)==canonical(expected),'All roots, both matched arm orders, and fixed recipes')
        _VALIDATED[key]=True
    if authenticate:
        c.verify([selection[k] for k in ('original_selection','source_acceptance','source_closeout')],boundary=boundary)
    return selection

class CachedExamples(Sequence):
    """Only requested rows become Python lists; full features stay memory mapped."""
    def __init__(self,cache,indices,arm):self.cache,self.indices,self.arm=cache,indices,arm
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        offset=self.indices[index];row=dict(self.cache['index'][offset])
        row['tokens']=self.cache['arrays']['tokens'][offset].reshape(-1).tolist()
        row['policy']=self.cache['arrays']['policy'][offset].tolist()
        row['mask']=self.cache['arrays']['mask'][offset].tolist()
        return targets.for_arm(row,self.arm)

def load_selected(selection,repetition,breadth,*,boundary=lambda:None):
    from . import corpus
    spec=fit_identity(repetition,breadth);validate_selection(selection,boundary=boundary)
    cache=corpus.load_cache(selection,boundary=boundary)
    cell=next(x for x in selection['cells'] if x['fit_id']==spec['fit_id'])
    indices=[i for i,x in enumerate(cache['index']) if x['split']=='train']
    require([[cache['index'][i]['game_index'],cache['index'][i]['ply']] for i in indices]==cell['root_ids'],'Complete cache row order matches selection')
    return dict(train_examples=CachedExamples(cache,indices,breadth),train_weights=[1.]*TRAIN_ROOTS,
        train_game_ids=cell['game_ids'],batches=cell['batches'],probe_batches=[],selection_cell=cell,
        source_receipt=selection['source_receipt'],value_target_sources_sha256=cache['manifest']['training_target_digests'][breadth],
        corpus_cache=c.artifact(c.CACHE_ACCEPTANCE),cache_validated=True)

def load_terminal_selected(selection,repetition=0,*,boundary=lambda:None):
    return load_selected(selection,repetition,'terminal',boundary=boundary)

def load_heldout(selection,commitment,*,boundary=lambda:None):
    from . import fit,corpus
    fit.validate_commitment(selection,commitment,boundary=boundary);validate_selection(selection,boundary=boundary)
    cache=corpus.load_cache(selection,boundary=boundary)
    return CachedExamples(cache,[i for i,x in enumerate(cache['index']) if x['split']!='train'],'terminal')

def load_diagnostic_train(selection,*,boundary=lambda:None):
    from . import corpus
    cache=corpus.load_cache(selection,boundary=boundary)
    lookup={(x['game_index'],x['ply']):i for i,x in enumerate(cache['index'])}
    return CachedExamples(cache,[lookup[tuple(x)] for x in selection['diagnostic_train_root_ids']],'terminal')

def collate_inputs(examples,weights):return accepted_training.collate_inputs(examples,weights)

def collate(examples,weights):
    import torch
    for row in examples:targets.validate(row)
    batch=collate_inputs(examples,weights)
    value=torch.tensor([x['value'] for x in examples],dtype=torch.float32)
    require(value.shape==(len(examples),2) and bool(torch.isfinite(value).all()) and
        bool((value.abs()<=1).all()) and bool((value.sum(dim=1)==0).all()),'Exact bounded actual target tensor')
    return {**batch,'value':value}
