"""Reuse exactly the two accepted broad selections and their training order."""
from copy import deepcopy
from . import common as c, targets
from tools.splendorust_independent_data_20261107 import sampling as original
from tools.splendorust_independent_data_20261107 import corpus as original_corpus
from tools.splendorust_independent_data_20261107 import fit as original_fit
from tools.splendorust_policy_head_20261104 import training as accepted_training

STUDY='splendorust_value_targets_20261110_v1'
DOMAIN=STUDY
IDENTITY=deepcopy(original.IDENTITY)
SEEDS=original.SEEDS
BOOTSTRAP_SEED=202611101283
ARMS=('broad',)
GAME_COUNTS=original.GAME_COUNTS
SCHEMA='value_targets_selection_v1'
_sha=original._sha
root_identity=original.root_identity
require,canonical,digest=c.require,c.canonical,c.digest

def fits():
    return [original.fit_identity(r,'broad','joint') for r in (0,1)]

def fit_identity(repetition,breadth,scope='joint'):
    c.require(type(breadth) is str and breadth=='broad', 'Only fixed broad fits')
    return original.fit_identity(repetition,breadth,scope)

def _build(old,controls):
    return dict(schema=SCHEMA,study_id=STUDY,**IDENTITY,
        original_selection=c.artifact(c.SOURCE_SELECTION),original_commitment=c.artifact(c.OLD_OUT/'all_fits_committed.json'),
        source_acceptance=c.artifact(c.SOURCE_ACCEPTANCE),source_closeout=c.artifact(c.SOURCE_CLOSEOUT),
        source_receipt=deepcopy(old['source_receipt']),games=deepcopy(old['games']),
        cells=[deepcopy(x) for x in old['cells'] if x['breadth']=='broad'],
        terminal_controls=deepcopy(controls),heldout_game_ids=deepcopy(old['heldout_game_ids']),
        game_counts=deepcopy(old['game_counts']),root_counts=deepcopy(old['root_counts']),
        value_target_contract=targets.contract(),selection_uses_outcomes=False,selection_uses_targets=False,
        historical_data_relabelled=False,heldout_reuse='previously_examined_diagnostic_only_fresh_live_primary')

def prepare_selection(*,boundary=lambda:None):
    acceptance=c.read(c.SOURCE_ACCEPTANCE);closeout=c.read(c.SOURCE_CLOSEOUT)
    c.require(acceptance['approved'] is True and closeout['status']=='completed_verified', 'Accepted completed source study')
    old=c.read(c.SOURCE_SELECTION);original.validate_selection(old,boundary=boundary)
    commitment=c.artifact(c.OLD_OUT/'all_fits_committed.json')
    verified=original_fit.validate_commitment(old,commitment,boundary=boundary)
    controls=[deepcopy(x) for x in verified['fits'] if x['spec']['breadth']=='broad']
    c.require([x['spec'] for x in controls]==fits(), 'Both fixed terminal control models')
    return _build(old,controls)

def validate_selection(selection,*,authenticate=True,boundary=lambda:None):
    c.require(type(selection) is dict and selection.get('schema')==SCHEMA, 'New objective-study selection')
    expected=prepare_selection(boundary=boundary)
    c.require(c.canonical(selection)==c.canonical(expected), 'Exact old roots/order and fixed new objective')
    return selection

def source_selection(selection,*,boundary=lambda:None):
    c.verify([selection['original_selection']],boundary=boundary)
    return c.read(c.resolve(selection['original_selection']))

def load_terminal_selected(selection,repetition,*,boundary=lambda:None):
    validate_selection(selection,boundary=boundary)
    return original.load_selected(source_selection(selection,boundary=boundary),repetition,'broad',boundary=boundary)

def load_selected(selection,repetition,breadth,*,boundary=lambda:None):
    from . import corpus
    fit_identity(repetition,breadth)
    loaded=load_terminal_selected(selection,repetition,boundary=boundary)
    by_game={}
    for row in loaded['train_examples']:by_game.setdefault(row['game_index'],[]).append(row)
    transformed=[]
    for game in selection['games']:
        if game['spec']['game_index'] not in by_game:continue
        boundary();transformed.extend(corpus.attach_game_targets(game,by_game[game['spec']['game_index']],boundary=boundary))
    transformed.sort(key=lambda x:(x['game_index'],x['ply']))
    c.require([[x['game_index'],x['ply']] for x in transformed]==loaded['selection_cell']['root_ids'], 'Exact same selected root order')
    loaded['train_examples']=transformed
    loaded['value_target_sources_sha256']=targets.source_digest(transformed)
    loaded['terminal_control_fit']=deepcopy(selection['terminal_controls'][repetition]['final_checkpoint'])
    return loaded

def load_heldout(selection,commitment,*,boundary=lambda:None):
    from . import fit
    fit.validate_commitment(selection,commitment,boundary=boundary)
    validate_selection(selection,boundary=boundary)
    examples=[]
    for game in selection['games']:
        if game['spec']['split']=='train':continue
        boundary();decoded=original_corpus.load_selected_game(game,selection['source_receipt'],boundary=boundary)
        c.require([root_identity(x) for x in decoded['examples']]==game['roots'],'Complete accepted held-out roots')
        examples.extend(decoded['examples'])
    examples.sort(key=lambda x:(x['game_index'],x['ply']))
    return examples

def collate_inputs(examples,weights):
    return accepted_training.collate_inputs(examples,weights)

def collate(examples,weights):
    import torch
    for row in examples:targets.validate(row)
    batch=collate_inputs(examples,weights)
    value=torch.tensor([x['value'] for x in examples],dtype=torch.float32)
    c.require(value.shape==(len(examples),2) and bool(torch.isfinite(value).all())
        and bool((value.abs()<=1).all()) and bool((value.sum(dim=1)==0).all()), 'Actual continuous actor/opponent target tensors')
    return {**batch,'value':value}
