"""Model-free recovery fixtures: state chain, forfeits, path admission and exclusivity."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.live_game_operational_recovery_20261020 import common as c, runner as r, worker as w
from tools.live_game_operational_20261019 import common as original


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True),encoding='utf-8')


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'SOURCE',tmp_path/'source')
    monkeypatch.setattr(c,'OUTPUT',tmp_path/'recovery')
    # Artifact helper is frozen to the real workspace; fixtures supply a pure equivalent.
    def artifact(path):
        path=Path(path)
        return {'path':path.relative_to(tmp_path).as_posix(),'bytes':path.stat().st_size,'sha256':c.file_hash(path)}
    monkeypatch.setattr(c,'artifact',artifact)
    monkeypatch.setattr(c,'check_wrapper',lambda result,obs,legal,trace: result['decision']['action'] in legal)
    monkeypatch.setattr(r,'load_profile',lambda *args:{})
    monkeypatch.setattr(r,'check_opponent',lambda *args:{'synthetic_accepted':True})
    c._recovery_input_map.cache_clear();c._launch_epoch.cache_clear()
    yield tmp_path
    c._recovery_input_map.cache_clear();c._launch_epoch.cache_clear()


def prefix(workspace, *, moves=3, timeout=False):
    spec=c.make_schedule()[0]
    source=c.SOURCE;folder=source/'sealed/games'/f"{spec['game']:03d}"
    env=r.rules.CorrectedEnv(seed=int(spec['seed']));state=env.initial_state()
    initial=r.rules.serialize_snapshot(state)
    write(folder/'initial.json',{**spec,'initial':initial,'initial_sha256':c.digest(initial)})
    descriptors=[]
    for ply in range(moves+int(timeout)):
        before=r.rules.serialize_snapshot(state);actor=state.current_player
        player='local' if actor==spec['local_seat'] else 'opponent'
        observer=(c.make_observer_payload(before,c.operational_context(spec['game'],ply))
                  if player=='local' else r.rules.observation_payload(state))
        name=f"games/g{spec['game']:03d}/t{ply:04d}"
        row={'id':name,'player':player,'observer':observer,'correct':not(timeout and ply==moves)}
        receipt={'request_id':'request-00000001','status':'ok','accepted':True,'deadline_seconds':120}
        if timeout and ply==moves:
            receipt.update(status='timeout',accepted=False)
            row['status']='forfeit_timeout'
        else:
            action=env.legal_actions(state)[0];serialized=c.serialize_action(action)
            result={'decision':{'action':serialized}} if player=='local' else {'action':serialized}
            receipt['result']=result
            row.update(status='completed',result_sha256=c.digest(result))
            if player=='local':
                trace=source/'workers'/name/'continuations.jsonl.gz'
                trace.parent.mkdir(parents=True,exist_ok=True);trace.write_bytes(b'synthetic trace')
                row['trace']=c.artifact(trace)
            state=env.step(state,action)
            row['after_sha256']=c.digest(r.rules.serialize_snapshot(state))
        receipt_path=source/'workers'/name/'request-00000001.json';write(receipt_path,receipt)
        row.update(receipt=c.artifact(receipt_path),clock=r.without_result(receipt))
        write(source/'decision_attempts'/(name+'.json'),row)
        move={**row,'game':spec['game'],'pair':spec['pair'],'fit':spec['fit'],'ply':ply,'actor':actor,'before_sha256':c.digest(before)}
        path=folder/'moves'/f'{ply:04d}.json';write(path,move);descriptors.append(c.artifact(path))
    if timeout:
        write(folder/'result.json',{**spec,'initial_sha256':c.digest(initial),'moves':descriptors,'complete':True,
            'reason':'forfeit_timeout','winner':1-state.current_player,'failing_player':player,'final':r.rules.serialize_snapshot(state)})
    return spec,state


def test_exact_three_move_prefix_reconstructed_from_seed_and_saved_actions(workspace):
    spec,state=prefix(workspace)
    result=r.reconstruct_game(spec,c.SOURCE,complete=False,next_ply=3)
    assert result['snapshot']==r.rules.serialize_snapshot(state)
    assert result['state'].turn_index==3 and len(result['rows'])==3 and result['failure'] is None


def test_recovered_environment_remainder_matches_uninterrupted_state(workspace):
    spec,_=prefix(workspace)
    recovered=r.reconstruct_game(spec,c.SOURCE,complete=False,next_ply=3)
    uninterrupted=r.rules.CorrectedEnv(seed=int(spec['seed']));expected=uninterrupted.initial_state()
    for _ in range(3):expected=uninterrupted.step(expected,uninterrupted.legal_actions(expected)[0])
    actual=recovered['state']
    assert r.rules.serialize_snapshot(actual)==r.rules.serialize_snapshot(expected)
    for _ in range(5):
        action=uninterrupted.legal_actions(expected)[0]
        expected=uninterrupted.step(expected,action)
        actual=recovered['env'].step(actual,action)
        assert r.rules.serialize_snapshot(actual)==r.rules.serialize_snapshot(expected)


@pytest.mark.parametrize('damage',['initial','gap','actor','before','after','observer','attempt','receipt','count'])
def test_recovery_rejects_changed_prefix_or_input_chain(workspace,damage):
    spec,_=prefix(workspace);folder=c.SOURCE/'sealed/games/006'
    path=folder/'moves/0001.json';row=c.read(path)
    if damage=='initial':
        p=folder/'initial.json';value=c.read(p);value['initial_sha256']='0'*64;write(p,value)
    elif damage=='gap':path.rename(folder/'moves/0004.json')
    elif damage=='count':pass
    elif damage=='receipt':
        p=c.ROOT/row['receipt']['path'];value=c.read(p);value['accepted']=False;write(p,value)
    elif damage=='attempt':
        p=c.SOURCE/'decision_attempts/games/g006/t0001.json';value=c.read(p);value['correct']=False;write(p,value)
    else:
        if damage=='actor':row['actor']=1-row['actor']
        elif damage=='before':row['before_sha256']='0'*64
        elif damage=='after':row['after_sha256']='0'*64
        else:row['observer']['observation']['bank_tokens']['white']+=1
        write(path,row)
    with pytest.raises((ValueError,KeyError)):
        r.reconstruct_game(spec,c.SOURCE,complete=False,next_ply=4 if damage=='count' else 3)


def test_timeout_forfeit_remains_completed_game_without_replay(workspace):
    spec,state=prefix(workspace,moves=2,timeout=True)
    result=r.reconstruct_game(spec,c.SOURCE,complete=True)
    assert len(result['rows'])==3 and result['failure']==('forfeit_timeout',0,'local')
    assert result['snapshot']==r.rules.serialize_snapshot(state)
    with pytest.raises(ValueError):r.reconstruct_game(spec,c.SOURCE,complete=False,next_ply=3)


def failed_before_init(recovered):
    name=f'games/g{c.PARTIAL_GAME:03d}/t{c.NEXT_PLY:04d}'
    state=recovered['state'];spec=next(x for x in c.make_schedule() if x['game']==c.PARTIAL_GAME)
    player='local' if state.current_player==spec['local_seat'] else 'opponent'
    observer=(c.make_observer_payload(recovered['snapshot'],c.operational_context(c.PARTIAL_GAME,c.NEXT_PLY))
              if player=='local' else r.rules.observation_payload(state))
    path=c.SOURCE/'decision_attempts'/(name+'.json')
    write(path,{'id':name,'player':player,'observer':observer,'status':'incomplete','correct':False,'failure_category':'ResourceHold'})
    return path


@pytest.mark.parametrize('damage',[None,'clock','initialization','worker','observer','failure'])
def test_only_never_started_next_decision_can_resume(workspace,monkeypatch,damage):
    spec,_=prefix(workspace);monkeypatch.setattr(c,'PARTIAL_GAME',6);monkeypatch.setattr(c,'NEXT_PLY',3)
    recovered=r.reconstruct_game(spec,c.SOURCE,complete=False,next_ply=3);path=failed_before_init(recovered)
    if damage=='worker':(c.SOURCE/'workers/games/g006/t0003').mkdir(parents=True)
    elif damage:
        row=c.read(path)
        if damage=='observer':row['observer']['observation']['turn_index']+=1
        elif damage=='failure':row['failure_category']='WorkerError'
        else:row[damage]={}
        write(path,row)
    if damage:
        with pytest.raises(ValueError):r.check_unstarted_partial_attempt(recovered)
    else:assert r.check_unstarted_partial_attempt(recovered)==path


def test_resumed_game_calls_only_next_ply_and_preserves_copied_prefix(workspace,monkeypatch):
    spec,_=prefix(workspace);monkeypatch.setattr(c,'PARTIAL_GAME',6);monkeypatch.setattr(c,'NEXT_PLY',3)
    recovered=r.reconstruct_game(spec,c.SOURCE,complete=False,next_ply=3)
    folder=c.SOURCE/'sealed/games/006'
    copied=[r.copy_new(p,c.OUTPUT/p.relative_to(c.SOURCE)) for p in folder.rglob('*.json')]
    calls=[]
    def decision(name,player,before,observer,deadline,cpu,**kwargs):
        calls.append(name)
        return {'id':name,'player':player,'observer':observer,'status':'forfeit_timeout','correct':False,
                'clock':{'receipt_elapsed_seconds':None}}
    monkeypatch.setattr(r,'decision_attempt',decision)
    game=r.play_game(spec,10**12,2,recovered=recovered)
    assert calls==['games/g006/t0003'] and len(game['moves'])==4 and game['reason']=='forfeit_timeout'
    assert all(c.artifact(c.ROOT/item['destination']['path'])==item['destination'] for item in copied)


def test_exclusive_import_preserves_source_and_destination(workspace):
    source=c.SOURCE/'metadata.json';source.parent.mkdir();source.write_bytes(b'exact bytes\n')
    destination=c.OUTPUT/'metadata.json';item=r.copy_new(source,destination)
    assert item['source']['sha256']==item['destination']['sha256']
    with pytest.raises(FileExistsError):r.copy_new(source,destination)
    assert source.read_bytes()==destination.read_bytes()==b'exact bytes\n'


def freeze_trace(path):
    write(c.OUTPUT/'freeze.json',{'schema':'live_operational_recovery_freeze_v1','inputs':[c.artifact(path)]})
    c._recovery_input_map.cache_clear()


def test_old_trace_is_read_only_audit_input_and_source_authenticated(workspace):
    path=c.SOURCE/'workers/games/g006/t0000/continuations.jsonl.gz'
    path.parent.mkdir(parents=True);path.write_bytes(b'oldtrace');freeze_trace(path)
    assert w.audit_trace_path(path)==path.resolve()
    with pytest.raises(ValueError):w.trace_path(path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):w.audit_trace_path(path)


@pytest.mark.parametrize('relative',['outside/continuations.jsonl.gz','source/workers/validation/v000/continuations.jsonl.gz',
                                     'source/workers/games/g006/t0000/not-a-trace.gz','source/workers/games/../../../continuations.jsonl.gz'])
def test_audit_trace_rejects_unapproved_path(workspace,relative):
    with pytest.raises(ValueError):w.audit_trace_path(workspace/relative)


def test_source_symlink_escape_rejected_when_platform_allows(workspace):
    outside=workspace/'outside';outside.mkdir();(outside/'continuations.jsonl.gz').write_bytes(b'outside')
    link=c.SOURCE/'workers/games/link';link.parent.mkdir(parents=True)
    try:link.symlink_to(outside,target_is_directory=True)
    except (OSError,NotImplementedError):pytest.skip('Creating symlinks unavailable to this Windows user')
    with pytest.raises(ValueError):w.audit_trace_path(link/'continuations.jsonl.gz')


def test_worker_ast_changes_only_common_import_and_audit_path_dispatch():
    root=Path(__file__).resolve().parents[1]
    before=ast.parse((root/'tools/live_game_operational_20261019/worker.py').read_text())
    after=ast.parse((root/'tools/live_game_operational_recovery_20261020/worker.py').read_text())
    after.body=[n for n in after.body if not isinstance(n,ast.FunctionDef) or n.name!='audit_trace_path']
    for tree in (before,after):
        if isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant):tree.body.pop(0)
    for node in ast.walk(after):
        if isinstance(node,ast.ImportFrom) and node.module=='tools.live_game_operational_recovery_20261020':
            node.module='tools.live_game_operational_20261019'
        if isinstance(node,ast.Name) and node.id=='audit_trace_path':node.id='trace_path'
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
    assert c.make_schedule()==original.make_schedule() and c.LOCAL_PROFILE_ID==original.LOCAL_PROFILE_ID
    assert c.LIVE_NAMESPACE==original.LIVE_NAMESPACE and c.AUDIT_SEED==original.AUDIT_SEED
    assert c.OUTPUT!=original.OUTPUT and c.SOURCE==original.OUTPUT


def test_context_sampling_proof_and_decision_ast_are_unchanged():
    root=Path(__file__).resolve().parents[1]
    def functions(path):
        return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old=functions(root/'tools/live_game_operational_20261019/common.py')
    new=functions(root/'tools/live_game_operational_recovery_20261020/common.py')
    for name in ('local_seat','make_schedule','pilot_observer','pilot_context','operational_context',
                 'check_context','make_observer_payload','opponent_observer','sample_worlds','stress_priority',
                 'stress_cells','expected_stress_cells','check_wrapper','check_audit_proof'):
        assert ast.dump(new[name],include_attributes=False)==ast.dump(old[name],include_attributes=False)
    old=functions(root/'tools/live_game_operational_20261019/runner.py')['decision_attempt']
    new=functions(root/'tools/live_game_operational_recovery_20261020/runner.py')['decision_attempt']
    for node in ast.walk(new):
        if isinstance(node,ast.Constant) and node.value=='tools.live_game_operational_recovery_20261020.worker':
            node.value='tools.live_game_operational_20261019.worker'
    assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)


def test_epoch_clock_can_stop_recovery_before_monotonic_clock(workspace,monkeypatch):
    c.OUTPUT.mkdir();write(c.OUTPUT/'launch.json',{'maximum_seconds':c.MAX_SECONDS,'epoch_seconds':100})
    monkeypatch.setattr(c.time,'time',lambda:100+c.MAX_SECONDS+1)
    monkeypatch.setattr(c.time,'monotonic',lambda:1)
    monkeypatch.setattr(c,'directory_bytes',lambda base:0)
    monkeypatch.setattr(c,'memory',lambda:{'available_physical':3*1024**3,'available_page':5*1024**3})
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:SimpleNamespace(free=10*1024**3))
    with pytest.raises(c.ResourceHold,match='ceiling'):c.boundary(c.OUTPUT,10**12)


def test_wall_clock_forward_jump_preserves_six_hour_worker_reserve(workspace,monkeypatch):
    c.OUTPUT.mkdir();write(c.OUTPUT/'launch.json',{'maximum_seconds':c.MAX_SECONDS,'epoch_seconds':100})
    wall=[100+c.MAX_SECONDS-22000]
    monkeypatch.setattr(c.time,'time',lambda:wall[0])
    monkeypatch.setattr(c.time,'monotonic',lambda:100)
    monkeypatch.setattr(c,'directory_bytes',lambda base:0)
    monkeypatch.setattr(c,'memory',lambda:{'available_physical':3*1024**3,'available_page':5*1024**3})
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:SimpleNamespace(free=10*1024**3))
    options=r.worker_options(100000,2,reserve=c.POST_GAME_RESERVE,mode='local')
    assert options['absolute_deadline']==500
    assert options['resource_guard']()['remaining_seconds']==22000
    wall[0]+=500
    with pytest.raises(c.ResourceHold,match='admission reserve'):
        r.worker_options(100000,2,reserve=c.POST_GAME_RESERVE,mode='local')
    with pytest.raises(c.ResourceHold,match='protected reserve'):
        options['resource_guard']()


def test_audit_admission_keeps_full_request_and_final_reserve(workspace,monkeypatch):
    monkeypatch.setattr(c,'boundary',lambda *args,**kwargs:{'remaining_seconds':1259})
    with pytest.raises(c.ResourceHold,match='admission reserve'):
        r.worker_options(10**12,2,reserve=c.FINAL_RESERVE,mode='audit')


def test_worker_deadline_does_not_refund_resource_guard_latency(workspace,monkeypatch):
    clock=[100]
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    def slow_boundary(*args,**kwargs):
        clock[0]+=7
        return {'remaining_seconds':22000}
    monkeypatch.setattr(c,'boundary',slow_boundary)
    options=r.worker_options(100000,2,reserve=c.POST_GAME_RESERVE,mode='local')
    assert clock[0]==107
    assert options['absolute_deadline']==500
    options=r.worker_options(22000,2,reserve=c.POST_GAME_RESERVE,mode='local')
    assert options['absolute_deadline']==400
