"""Actual corrected collector corpus loader; no model or optimizer imports.

The accepted root decoder is reused exactly. This wrapper binds the new real
60/75-second clocks and streams individual move artifacts instead of inventing
the old training worker's response envelope.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from . import common as c
from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.splendorust_selfplay_pilot_resume2_20261103 import data as accepted
from tools.splendorust_gpu_throughput_20261105 import games as original

IDENTITY = dict(accepted.IDENTITY)
GAME_SCHEMA = 'gpu_collector_target_game_v1'
CORPUS_SCHEMA = 'gpu_collector_corpus_v1'
decode_row = accepted.decode_row
collate_arrays = accepted.collate_arrays
signed_outcomes = accepted.signed_outcomes


def read_artifact(item):
    c.require(type(item) is dict and set(item) == {'path','bytes','sha256'}, 'exact artifact declaration')
    path = c.resolve(item).resolve()
    c.require(path.is_relative_to(c.ROOT.resolve()) and path.relative_to(c.ROOT).as_posix() == item['path'],
              'canonical workspace artifact')
    c.require(type(item['bytes']) is int and 0 <= item['bytes'] <= 128*1024**2,'bounded exact artifact size')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    c.require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
              and len(raw)==before.st_size==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],
              'exact parsed artifact bytes authenticated and stable')
    return json.loads(raw)


def initial_snapshot(spec):
    if 'fixture_start' in spec:
        c.require(spec.get('purpose') == 'engineering_qualification_only'
                  and spec.get('game_id') == 'loader_noble_final_round', 'explicit isolated engineering fixture')
        from tools.splendorust_selfplay_pilot_resume2_20261103 import corpus
        expected=next(x['snapshot'] for x in corpus.build_fixtures() if x['id']=='noble_final_round')
        c.require(spec['fixture_start'] == expected,'fixed actual noble final-round fixture start')
        guard.decode_training_snapshot(expected)
        return deepcopy(expected)
    c.require(type(spec['seed_decimal']) is str and spec['seed_decimal'].isdigit()
              and type(spec['start_player']) is int and spec['start_player'] in (0,1), 'explicit setup identity')
    state = rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state()
    state.current_player = state.start_player = spec['start_player']
    value = rules.serialize_snapshot(state)
    guard.decode_training_snapshot(value)
    return value


def validate_row(row, spec, before, ply):
    c.require(row['spec'] == spec and row['game_id'] == spec['game_id'] and row['before'] == before
              and row['ply'] == ply, 'exact contiguous scheduled collector row')
    original.validate_clock(row['clock'])
    c.require(row['result']['action'] == row['action'] and row['result']['diagnostics'] == row['trace'],
              'original full scientific response binds capture source')
    from tools.splendorust_gpu_throughput_20261105 import scientific
    scientific.validate_result(row['result'],row['observer'],'search',expected_quota=512)
    scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',
                              before=before,after=row['after'],expected_quota=512)
    decoded = decode_row(row,expected_spec=spec,expected_ply=ply,expected_quota=512)
    c.require(type(row['actor']) is int and decoded['actor'] == row['actor'], 'absolute actor binding')
    return decoded


def build_manifest(game_artifacts, specs, *, route, cell_id):
    core = dict(schema=CORPUS_SCHEMA,cell_id=cell_id,route=route,teacher_simulations=512,
                specimens=deepcopy(specs),games=deepcopy(game_artifacts),purpose='collector_qualification_no_training')
    return dict(core,metadata=dict(schema='corrected_training_data_v1',**IDENTITY,data_sha256=c.digest(core)))


def validate_manifest(manifest, specs):
    c.require(set(manifest) == {'schema','cell_id','route','teacher_simulations','specimens','games','purpose','metadata'},
              'exact collector manifest schema')
    guard.require_data_contract(manifest['metadata'])
    core = {key:value for key,value in manifest.items() if key != 'metadata'}
    c.require(manifest['schema'] == CORPUS_SCHEMA and manifest['purpose'] == 'collector_qualification_no_training'
              and manifest['teacher_simulations'] == 512 and manifest['route'] in ('cpu','gpu')
              and manifest['metadata']['data_sha256'] == c.digest(core), 'collector corpus identity and digest')
    c.require(manifest['specimens'] == specs and bool(specs) and len(manifest['games']) == len(specs)
              and len({x['game_id'] for x in specs}) == len(specs)
              and len({x['seed_decimal'] for x in specs}) == len(specs)
              and len({x['path'] for x in manifest['games']}) == len(specs), 'complete unique scheduled game inventory')


def load_game(document, spec, *, boundary=lambda:None):
    accepted.verify_identity(document)
    c.require(document['schema'] == GAME_SCHEMA and document['spec'] == spec and document['reason'] == 'completed',
              'only complete scheduled official games may produce targets')
    before = initial_snapshot(spec)
    c.require(document['initial'] == before and 0 < len(document['rows']) <= 2000, 'seed and bounded full trajectory')
    examples=[]; exclusions=Counter(); previous_pass=False
    for ply,item in enumerate(document['rows']):
        boundary(); row=read_artifact(item)
        c.require(row['route']==document['route'],'same declared source route for every game row')
        decoded=validate_row(row,spec,before,ply)
        is_pass=row['action']['action_type']=='PASS'
        c.require(not(previous_pass and is_pass) and row['after'] != before, 'no deadlock or invented terminal labels')
        previous_pass=is_pass; before=row['after']
        if decoded['eligible']: examples.append(decoded)
        else: exclusions[decoded['reason']]+=1
    c.require(before == document['final'], 'exact terminal state')
    winner=guard.require_official_terminal_target(before,document['reason'])
    c.require(document['winner'] == winner and (winner is None or type(document['winner']) is int)
              and bool(examples), 'official winner and nonempty eligible roots')
    for example in examples:
        example.update(value=signed_outcomes(before,'completed',example['actor']),game_id=spec['game_id'],
                       game_index=spec['game_index'],route=document['route'])
    # Exercise the same actual minibatch conversion that future fitting would use.
    batches=[]
    for start in range(0,len(examples),32):
        selected=examples[start:start+32]; arrays=collate_arrays(selected,[1.]*len(selected))
        c.require(arrays['tokens'].reshape(len(selected),-1).tolist() == [x['tokens'] for x in selected],
                  'actual loader f32 input batch exact')
        batches.append(dict(start=start,count=len(selected),shape=list(arrays['tokens'].shape)))
    return dict(spec=spec,route=document['route'],plies=len(document['rows']),eligible_roots=len(examples),
                exclusions=dict(exclusions),winner=winner,examples=examples,batches=batches,
                initial_sha256=c.digest(document['initial']),final_sha256=c.digest(before),game_sha256=c.digest(document))


def load_corpus(manifest, *, expected_specs, boundary=lambda:None, sink=None):
    validate_manifest(manifest,expected_specs)
    games=[]; examples_count=0
    for spec,item in zip(expected_specs,manifest['games'],strict=True):
        boundary(); document=read_artifact(item)
        c.require(document['route'] == manifest['route'], 'one route per corpus')
        loaded=load_game(document,spec,boundary=boundary)
        examples=loaded.pop('examples'); examples_count+=len(examples)
        if sink is not None: sink(spec,examples)
        games.append(loaded)
    return dict(status='actual_collector_loader_passed',manifest_sha256=c.digest(manifest),games=games,
                game_count=len(games),eligible_roots=examples_count,training_updates=0,
                actual_training_loader_used=True,historical_training_loader_used=False,**IDENTITY)


def qualify_captured_rows(plan, rows, *, boundary=lambda:None):
    # The frozen routine calls the identical decode_row/collate functions exposed
    # above and used by load_game; this is actual-path qualification, not a mock.
    return accepted.qualify_captured_rows(plan,rows,boundary=boundary)


def qualify_manifest_rejections(manifest, specs):
    bad_cases=[]
    for key in ('ruleset_id','catalogue_sha256','turn_contract','origin','data_sha256'):
        bad=deepcopy(manifest);bad['metadata'].pop(key);bad_cases.append(('missing_'+key,bad))
    bad=deepcopy(manifest);bad['games'][0]['sha256']='0'*64
    bad['metadata']['data_sha256']=c.digest({k:v for k,v in bad.items() if k!='metadata'})
    bad_cases.append(('artifact_hash_mismatch',bad))
    checks=[]
    for name,bad in bad_cases:
        try:load_corpus(bad,expected_specs=specs)
        except (ValueError,KeyError,TypeError,AssertionError):checks.append(dict(case=name,rejected=True))
        else:raise ValueError('Actual collector loader accepted '+name)
    return checks
