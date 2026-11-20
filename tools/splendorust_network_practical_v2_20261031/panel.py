"""Fresh saved-input assessment, selected before any practical candidate output.

Reuses accepted corrected-rule provenance readers without editing their source.
The sample is a stratified fixed panel, not an independent training/test corpus
or a random whole-game latency/strength assessment.
"""
from collections import Counter
import hashlib
import math
from . import common as c
from tools.splendorust_search256_efficiency_20261031 import panel as old

SCHEMA='splendorust_network_practical_panel_v1'
SELECTION_DOMAIN='splendorust_network_practical_20261031_independent_saved_v1'
PROGRESS=old.PROGRESS
OLD_PANEL=c.PREVIOUS_OUT/'panel.json'
OLD_PANEL_SHA='5a41ce0a8863f42c1550ae849a148f5a1659bf114476ff2debabeaae4d9caada'
EXECUTION_OUT=c.ROOT/'outputs/splendorust_network_execution_20261031_01'
EXECUTION_FREEZE_SHA='ce9b995aafe8794f17317fe8cbdb57eef3f1a8fbf01bc504cc57ce24a0125b4f'
require=c.require


def selection_rule():
    return dict(schema='network_practical_selection_rule_v1',hash_domain=SELECTION_DOMAIN,
        population='all11066 accepted main256 decisions from400 games',
        hash_encoding='UTF-8 domain|purpose|decimal game_index|decimal ply',
        progress='min(2,floor(3*ply/final_game_plies)); early,middle,late',
        cell_order=[dict(progress=p,seat=s) for p in PROGRESS for s in (0,1)],
        systematic='two ascending SHA256 timing priorities per cell',
        tail='one largest saved complete-response arrival per cell after systematic selection',
        numerical_extra='three ascending SHA256 numerical priorities per cell after timing selection',
        exclusions='all prior18 panel games; any token hash from prior11 compatibility fixtures or six localization fixtures; all already selected games and batch1 token hashes',
        retained_tensor='first full batch1 callback in phase order; skip candidate if absent, nonfinite, duplicate or development hash',
        advance='advance in fixed ordered cell list after exclusion, with game_index/ply tie breaks',
        timing_roots=18,systematic_roots=12,tail_roots=6,numerical_extra_roots=18,
        numerical_batch1=36,composites=[dict(batch=2,indices=[0,1]),dict(batch=8,indices=list(range(8)))],
        outcome_used=False,new_models=0,new_games=0,
        full_saved_transition_and_result_validation=True,
        independence='new saved positions/games and tensors relative to development panels; same accepted400-game source corpus',
        population_latency_estimate=False)


def priority(row,purpose):
    return hashlib.sha256(f"{SELECTION_DOMAIN}|{purpose}|{row['game_index']}|{row['ply']}".encode()).hexdigest()


def first_tensor(root):
    for phase_index,phase in enumerate(root['trace']['phases']):
        for callback_index,item in enumerate(phase.get('inference',[])):
            if 'tokens' in item:
                tokens=item['tokens']
                require(item['batch']==1 and len(tokens)==31*48 and
                    all(type(x) in (int,float) and math.isfinite(x) for x in tokens),
                    'retained finite batch1 tensor')
                require(c.digest(tokens)==item['input_sha256'],'retained callback input digest')
                return dict(tokens=tokens,input_sha256=item['input_sha256'],
                    phase_index=phase_index,callback_index=callback_index,
                    expected_output83=item['output83'])
    return None


def panel_roots(panel):
    require(panel.get('schema')==SCHEMA and panel.get('selection')==selection_rule(),
            'exact independent panel selection/schema')
    roots=panel.get('roots')
    require(type(roots) is list and len(roots)==18,'all18 timing roots')
    require(len({r['game_index'] for r in roots})==18,'distinct timing source games')
    result={}
    for root in roots:
        name=root['root_id']
        require(name==root['panel_id'] and name not in result,'unique root aliases')
        require(root['progress']==old.progress(root['ply'],root['game_plies']) and
                type(root['seat']) is int and root['seat'] in (0,1),'root progress/seat')
        original=root['original_result256']
        require(original['simulations_quota']==256 and original['observer_sha256']==c.digest(root['observer']),
                'historical256 lawful observer binding')
        require(root['observer_seed']==int(c.digest(root['observer'])[:16],16) and
                root['trace']['observer_seed']==root['observer_seed'],'historical deterministic seed')
        require(root['action']==original['action'] and root['trace']==original['diagnostics'],
                'historical full action/trace binding')
        result[name]=root
    for progress in PROGRESS:
        for seat in (0,1):
            require(Counter(r['group'] for r in roots if r['progress']==progress and r['seat']==seat)==
                    Counter(systematic=2,tail=1),'two systematic plus one tail per cell')
    return result


def select_bound(census,binder,excluded_games,excluded_hashes):
    """Fixed outcome-blind advancing selector; binder authenticates saved inputs."""
    require(len({(r['game_index'],r['ply']) for r in census})==len(census),'unique census identities')
    for row in census:
        require(type(row['game_index']) is int and 0<=row['game_index']<400 and
                type(row['seat']) is int and row['seat'] in (0,1) and
                row['progress']==old.progress(row['ply'],row['game_plies']),
                'genuine saved census identity/progress')
        require(type(row['original_receipt_seconds']) in (int,float) and
                math.isfinite(row['original_receipt_seconds']) and
                0<row['original_receipt_seconds']<=10.,'accepted saved census response')
    used_games=set(excluded_games);used_hashes=set(excluded_hashes)
    roots=[];extras=[];skipped=[]
    for group,count in (('systematic',2),('tail',1),('numerical',3)):
        for phase in PROGRESS:
            for seat in (0,1):
                cell=[r for r in census if r['progress']==phase and r['seat']==seat]
                purpose='numerical' if group=='numerical' else 'timing'
                key=(lambda r:(-r['original_receipt_seconds'],r['game_index'],r['ply'])) if group=='tail' else (
                    lambda r:(priority(r,purpose),r['game_index'],r['ply']))
                admitted=0
                for row in sorted(cell,key=key):
                    if row['game_index'] in used_games:continue
                    root=binder(dict(row,group=group,selection_priority=priority(row,purpose)),len(roots)+len(extras))
                    token=first_tensor(root)
                    if token is None or token['input_sha256'] in used_hashes:
                        skipped.append(dict(game_index=row['game_index'],ply=row['ply'],
                            reason='no_full_tensor' if token is None else 'duplicate_or_development_tensor'))
                        continue
                    (extras if group=='numerical' else roots).append(root)
                    used_games.add(row['game_index']);used_hashes.add(token['input_sha256']);admitted+=1
                    if admitted==count:break
                require(admitted==count,'sufficient distinct eligible roots in fixed cell '+group+phase+str(seat))
    require(len(roots)==len(extras)==18,'18 timing and18 numerical-only roots')
    return roots,extras,skipped


def numerical_plan(roots,extras):
    require(len(roots)==len(extras)==18,'fixed36 retained tensors')
    fixtures=[];provenance=[]
    for index,root in enumerate(roots+extras):
        token=first_tensor(root);require(token is not None,'full retained tensor')
        name=f'fresh_saved_{index:02d}'
        fixtures.append(dict(fixture_id=name,batch=1,tokens=token['tokens']))
        provenance.append(dict(fixture_id=name,root_id=root['root_id'],game_index=root['game_index'],
            ply=root['ply'],group=root['group'],progress=root['progress'],seat=root['seat'],
            source_row=root['source_row'],source_clock=root['source_clock'],source_input=root['source_input'],
            input_sha256=token['input_sha256'],phase_index=token['phase_index'],callback_index=token['callback_index'],
            historical_output83=token['expected_output83']))
    require(len({c.digest(f['tokens']) for f in fixtures})==36,'36 distinct batch1 tensors')
    for batch in (2,8):
        fixtures.append(dict(fixture_id=f'fresh_saved_batch{batch}',batch=batch,
            tokens=[x for item in fixtures[:batch] for x in item['tokens']]))
    return dict(schema='network_practical_numerical_plan_v1',fixtures=fixtures,provenance=provenance,
        cases=38,batch1=36,composites=selection_rule()['composites'],
        identical_input_comparison=True,callback_path_alignment=False,
        numerical_policy=dict(policy_absolute=1e-5,policy_relative=1e-6,value_absolute=1e-6,
            relative_reference='eager_control',finite=True,dtype='float32'),new_models=0,new_games=0)


def build_panel(boundary=lambda:None):
    # Precommit is checked before inspecting any new saved position or tensor.
    rule_path=c.OUT/'selection_rule.json'
    require(rule_path.is_file() and c.read(rule_path)==selection_rule(),'selection rule saved before extraction')
    from tools.future_training_rules_20261021 import verify_runtime_sources
    source_guard=verify_runtime_sources()
    reader=old.Reader(boundary)
    freeze,schedule=old._source_context(reader)
    old_panel=reader.bound(OLD_PANEL,expected_sha=OLD_PANEL_SHA)
    development=reader.bound(EXECUTION_OUT/'freeze.json',expected_sha=EXECUTION_FREEZE_SHA)
    reader.add(development['profile_plan']);profile=reader.bound(c.ROOT/development['profile_plan']['path'])
    exclusion_fixtures=development['graph_parity_fixtures']+profile['fixtures']
    excluded_hashes=set()
    for item in exclusion_fixtures:
        for offset in range(item['batch']):
            excluded_hashes.add(c.digest(item['tokens'][offset*31*48:(offset+1)*31*48]))
    excluded_games={r['game_index'] for r in old_panel['roots']}
    census=old.compact_census(reader,schedule)
    binder=lambda row,index:old.bind_root(reader,row,schedule[row['game_index']],freeze,index)
    roots,extras,skipped=select_bound(census,binder,excluded_games,excluded_hashes)
    panel=dict(schema=SCHEMA,selection=selection_rule(),roots=roots,source_guard=source_guard,
        accepted_source_freeze_sha256=old.FREEZE_SHA,
        census=dict(decisions=len(census),games=400,scalar_rows_sha256=c.digest(census)),
        exclusion=dict(prior_panel_games=sorted(excluded_games),development_token_hashes=sorted(excluded_hashes)),
        numerical_extra_positions=[dict(game_index=r['game_index'],ply=r['ply'],root_id=r['root_id'],
            progress=r['progress'],seat=r['seat'],source_row=r['source_row']) for r in extras],
        deterministic_skips=skipped,inputs=sorted(reader.consumed.values(),key=lambda x:x['path']),
        new_models=0,new_games=0,source_evidence_modified=False)
    panel_roots(panel)
    return panel,numerical_plan(roots,extras)
