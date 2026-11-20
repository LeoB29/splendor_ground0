"""Independent model-free checks of quota scope and preserved player routing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_inference64_20261025 import runner


def test_reused_graph_and_worker_still_match_the_joint_negative_freeze():
    root = runner.c.ROOT
    path = runner.c.JOINT / 'freeze.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        '2339f755ce171811d9474a53d53881e4d8e1476baa60f91517e778b903e7264a'
    frozen = {item['path']:item for item in json.loads(path.read_text())['inputs']}
    for name in ('graph.py', 'worker.py'):
        relative = 'tools/splendorust_inference_opt_20261025/' + name
        encoded = (root / relative).read_bytes()
        assert len(encoded) == frozen[relative]['bytes']
        assert hashlib.sha256(encoded).hexdigest() == frozen[relative]['sha256']


def test_graph32_is_rejected_before_worker_creation_or_move_admission():
    envelope = SimpleNamespace(admit=lambda:pytest.fail('Forbidden graph32 cannot reach admission'))
    with pytest.raises(ValueError, match='Only eager32/eager64'):
        runner.make_worker('forbidden', envelope, {}, mode='candidate', quota=32)
    child = SimpleNamespace(runtime_mode='candidate', challenger_quota=32)
    with pytest.raises(ValueError, match='Only eager32/eager64'):
        runner.decision(child, None, envelope)


def test_eager32_constructs_original_worker_without_graph_parity(monkeypatch, tmp_path):
    from tools.splendorust_search_scale_20261024 import adapters
    seen = []
    class Eager:
        def __init__(self, module, folder, **kwargs):
            seen.append((module, kwargs))
            self.output_dir, self.process = folder, SimpleNamespace(pid=123)
            self.initialization = {'result':{'scientific':'original'}}
            self._failure = None
        def _sample(self): return {'process_tree':[]}
        def close(self): pass
    envelope = SimpleNamespace(admit=lambda:None, register_worker=lambda folder:None,
        guard_worker_sample=lambda *args:None, timeout=lambda maximum:maximum,
        effective_deadline=lambda:999, own_worker=lambda child:None, guard=lambda:None)
    monkeypatch.setattr(runner.c, 'OUT', tmp_path)
    monkeypatch.setattr(runner.c, 'BaselineStudyWorker', Eager)
    monkeypatch.setattr(runner.c, 'CandidateStudyWorker', lambda *args,**kwargs:pytest.fail('Reference changed backend'))
    monkeypatch.setattr(adapters, 'validate_ready_metadata', lambda *args,**kwargs:None)
    monkeypatch.setattr(runner, 'graph_parity', lambda *args:pytest.fail('Eager reference cannot compile or shadow infer'))
    child = runner.make_worker('eager32', envelope,
        {'native_artifacts':['same-native'], 'topology':{'selected_logical_processor':2}},
        mode='baseline', quota=32)
    assert seen[0][0] == 'tools.splendorust_search_scale_20261024.worker'
    assert seen[0][1]['env']['STUDY_SEARCH_SIMULATIONS'] == '32'
    assert seen[0][1]['native_artifacts'] == ['same-native']
    assert child.runtime_mode == 'baseline' and child.challenger_quota == 32


def test_final_reproductions_preserve_six_graph64_and_six_eager32_players(monkeypatch, tmp_path):
    """Exercise final routing, not just schedule labels or source-text matching."""
    monkeypatch.setattr(runner.c, 'OUT', tmp_path)
    monkeypatch.setattr(runner.c, 'ROOT', tmp_path)
    saved = {}
    def write(path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        saved[path] = deepcopy(value)
    monkeypatch.setattr(runner.c, 'save', write)
    schedule = runner.operational_schedule()
    panel_summary = {'qualified':True}
    plan = {'fixed':'panel'}
    modes_by_folder = {}
    def initialize_folder(name, mode, quota):
        folder = tmp_path / 'workers' / name
        metadata = {'quota':quota, 'mode':mode}
        if mode == 'candidate': metadata['inference_backend'] = runner.BACKEND
        write(folder/'initialization.json', {'result':metadata})
        modes_by_folder[name] = (mode, quota)
        return folder
    for quota, mode in runner.ENTRYPOINTS:
        folder = initialize_folder(f'entrypoint_{quota}_{mode}', mode, quota)
        receipt = {'request_id':'r1', 'status':'ok', 'result':{
            'status':'entrypoint_counterexamples_passed', 'challenger_quota':quota,
            'counts':{'complete_decisions':52}}}
        write(folder/'r1.json', receipt)
        write(tmp_path/'entrypoint_checks'/f'{quota}_{mode}.json', receipt)
    for mode in ('baseline','candidate'):
        initialize_folder('qualification_64_' + mode, mode, 64)
    for index in range(144): write(tmp_path/'paired_timing'/f'{index:04d}.json', {'index':index})
    originals, all_moves = {}, []
    for spec in schedule:
        folder = runner.game_folder(spec)
        initial = {'seed':int(spec['seed_decimal'])}
        write(folder/'initial.json', dict(**spec, initial=initial))
        moves = []
        for ply, role in enumerate(('challenger','opponent')):
            mode, quota = spec[role+'_mode'], spec[role+'_quota']
            initialize_folder(f"operational_g{spec['game_index']}_{role}", mode, quota)
            before = {'game':spec['game_index'], 'role':role}
            row = dict(before=before, action={'backend':mode,'quota':quota},
                trace={'exact_mode':mode,'quota':quota}, game_role=role, mode=mode, quota=quota,
                stage='operational', game_index=spec['game_index'], pair_index=spec['pair_index'],
                ply=ply, completed=True, clock={'status':'ok','deadline_seconds':1.,'receipt_elapsed_seconds':.2})
            moves.append(row); originals[(spec['game_index'],role)] = row
            write(folder/'moves'/f'{ply:04d}.json', row)
        all_moves.extend(moves)
        write(folder/'result.json', dict(**spec, plies=2))
    operational_summary = runner.summarize_operational(all_moves,6)
    write(tmp_path/'operational_summary.json', operational_summary)
    monkeypatch.setattr(runner.rules, 'CorrectedEnv', lambda seed:SimpleNamespace(initial_state=lambda:{'seed':seed}))
    monkeypatch.setattr(runner.rules, 'serialize_snapshot', lambda value:value)
    monkeypatch.setattr(runner, 'decode_training_snapshot', lambda value:value)
    monkeypatch.setattr(runner, 'verify_saved_decision', lambda row:None)
    monkeypatch.setattr(runner.plan_rules, 'validate_panel', lambda *args,**kwargs:panel_summary)
    monkeypatch.setattr(runner, 'replay_game', lambda *args,**kwargs:{'status':'passed'})
    repeated = []
    class Child:
        def __init__(self,name,mode,quota):
            self.runtime_mode, self.challenger_quota = mode,quota
            initialize_folder(name,mode,quota)
        def __enter__(self): return self
        def __exit__(self,*args): pass
    def make(name,envelope,freeze,*,mode,quota):
        runner.require_player(mode,quota)
        repeated.append((mode,quota))
        return Child(name,mode,quota)
    monkeypatch.setattr(runner,'make_worker',make)
    def decision(child,state,envelope):
        original=originals[(state['game'],state['role'])]
        assert (child.runtime_mode,child.challenger_quota)==(original['mode'],original['quota'])
        return deepcopy(original)
    monkeypatch.setattr(runner,'decision',decision)
    monkeypatch.setattr(runner,'save_and_check',lambda row,path,envelope:write(path,row))
    monkeypatch.setattr(runner,'verify_worker_logs',lambda folder,**kwargs:{'status':'passed'})
    graph_checks=[]
    def verify_graph(folder,freeze,**kwargs):
        assert modes_by_folder[folder.name]==('candidate',64)
        graph_checks.append(folder.name)
        return {'weights_sha256':'same-preserved-weights'}
    monkeypatch.setattr(runner,'verify_graph_evidence',verify_graph)
    monkeypatch.setattr(runner,'verify_runtime',lambda *args:None)
    monkeypatch.setattr(runner,'predecessor_acceptance',lambda:{'same':'negative-joint'})
    envelope=SimpleNamespace(guard=lambda:None, effective_deadline=lambda:999, peak_combined_rss=123)
    runner.final_checks(envelope, {'baseline_acceptance':{'same':'negative-joint'}}, plan, schedule,
        {'qualification':panel_summary,'operational':operational_summary})
    assert repeated == [('candidate',64),('baseline',32)] * 6
    assert len(modes_by_folder)==29 and len(graph_checks)==14
    assert saved[tmp_path/'all_worker_log_integrity.json']['workers']==29
    assert saved[tmp_path/'all_graph_parity.json']['workers']==14
    assert saved[tmp_path/'repeated_decision_summary.json']['deadline_seconds']==1.
    assert saved[tmp_path/'verification_summary.json']['operational_reproductions']==12
