"""Model-free checks for sample, lifecycle, and cross-module interfaces."""
import ast
from collections import Counter
from pathlib import Path
import pytest
from . import common as c, panel, control

def source():
    return c.read(panel.SOURCE/'panel.json')

def frozen_panel():
    p=source();primary,deep=panel.selections(p)
    p.update(primary_root_ids=primary,deep_root_ids=deep,probe_actions67={r:[0,1] for r in primary},
        teacher_variability_study=c.STUDY,prior_bindings=[])
    return p

def test_balanced_whole_game_outcome_blind_selection():
    p=frozen_panel();ids=p['primary_root_ids'];m=p['host_metadata']
    assert len({m[r]['game_index'] for r in ids})==128
    assert Counter((m[r]['split'],m[r]['quantile_band']) for r in ids)=={
        (s,b):16 for s in ('validation','offline') for b in range(4)}
    assert Counter((m[r]['split'],m[r]['quantile_band']) for r in p['deep_root_ids'])=={
        (s,b):8 for s in ('validation','offline') for b in range(4)}
    for r in p['roots']: r['observer']['arbitrary_uninspected_test_marker']='not used by selection'
    assert panel.selections(p)==(ids,p['deep_root_ids'])

def test_matrix_has_exact_work_and_paired_conditions():
    p=frozen_panel();rows=panel.request_matrix(p)
    assert len(rows)==len({r['request_id'] for r in rows})==3072
    assert Counter(r['quota'] for r in rows)=={512:2048,2048:1024}
    assert Counter(r['pool'] for r in rows)=={3:1536,6:1536}
    assert sum(r['quota'] for r in rows)==3145728
    for r in rows:
        assert r['replicate']==r['world_seed_label']+'_'+r['encoding_seed_label']
        if r['mode']=='legacy':assert r['world_seed_label']==r['encoding_seed_label']

def test_repeats_cover_every_condition_at_each_budget():
    rows=panel.reproduction_matrix(frozen_panel())
    assert len(rows)==32 and len({r['root_id'] for r in rows})==16
    for q in (512,2048):
        assert {(r['pool'],r['mode'],r['world_seed_label'],r['encoding_seed_label']) for r in rows if r['quota']==q}==set(panel.CONDITIONS)
    assert all(r['request_id']=='repeat_'+r['original_request_id'] for r in rows)

def test_panel_rejects_selection_changes():
    p=frozen_panel();p['primary_root_ids'][0],p['primary_root_ids'][1]=p['primary_root_ids'][1],p['primary_root_ids'][0]
    with pytest.raises(ValueError):panel.validate_panel(p)

def test_exclusive_save_preserves_previous_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    p=tmp_path/'receipt.json';c.save(p,{'first':True})
    with pytest.raises(FileExistsError):c.save(p,{'second':True})
    assert c.read(p)=={'first':True}

def test_save_rejects_frozen_predecessor_or_parent(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path/'new')
    with pytest.raises(ValueError):c.save(tmp_path/'prior.json',{})

def test_no_forecast_or_available_ram_gate():
    d=c.design()
    assert not d['available_ram_gate'] and not d['forecast_gate']
    assert d['one_second_descriptive_only'] and d['margin700ms_descriptive_only']
    assert not d['automatic_retries'] and not d['automatic_followup']

def test_stage_caps_do_not_extend_global_or_model_clock(monkeypatch):
    monkeypatch.setattr(control.time,'monotonic',lambda:100.)
    monkeypatch.setattr(control.time,'time',lambda:1100.)
    parent=dict(monotonic_origin=0.,unix_origin=1000.,monotonic_deadline=200.,unix_deadline=1200.)
    result=control.clipped_limits(parent,999999,model=True)
    assert result['monotonic_deadline']==200 and result['unix_deadline']==1200

def test_async_absence_grace_waits_for_actual_identity(monkeypatch):
    ticks=[0.];calls=[0]
    monkeypatch.setattr(control.time,'monotonic',lambda:ticks[0])
    monkeypatch.setattr(control.time,'sleep',lambda seconds:ticks.__setitem__(0,ticks[0]+seconds))
    monkeypatch.setattr(c,'check_deadlines',lambda limits:None)
    def alive(i):calls[0]+=1;return calls[0]<4
    monkeypatch.setattr(c,'exact_alive',alive)
    r=control.observe_absence([{'pid':1,'created':1.}],{},seconds=1.)
    assert r['exact_owned_absence'] and r['samples']==4 and r['elapsed_seconds']>=.15

def test_async_grace_does_not_accept_live_process(monkeypatch):
    ticks=[0.]
    monkeypatch.setattr(control.time,'monotonic',lambda:ticks[0])
    monkeypatch.setattr(control.time,'sleep',lambda seconds:ticks.__setitem__(0,ticks[0]+seconds))
    monkeypatch.setattr(c,'check_deadlines',lambda limits:None)
    monkeypatch.setattr(c,'exact_alive',lambda i:True)
    assert not control.observe_absence([{}],{},seconds=.1)['exact_owned_absence']

def test_all_python_sources_compile_without_loading_models():
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))

def test_actual_pipeline_calls_runtime_guard_once_per_worker():
    for name in ('worker.py','service.py'):
        tree=ast.parse((Path(__file__).parent/name).read_text())
        assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_runtime_sources' for n in ast.walk(tree))==1

def test_ordinary_validation_kwargs_bind_real_checker_signature():
    import inspect
    from . import qualification
    signature=inspect.signature(qualification.validate_result)
    tree=ast.parse((Path(__file__).parent/'run_models.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='validate_result']
    assert len(calls)==1
    call=calls[0]
    signature.bind(*[object() for a in call.args],**{k.arg:object() for k in call.keywords})

def test_only_exact_mutable_cargo_bookkeeping_is_excluded():
    from . import owner,native_build
    receipt=c.read(native_build.BUILD/'build_receipt.json')
    before=c.read(c.resolve(receipt['source_inventory']))
    after=owner.immutable_native_inventory(receipt)
    removed=[x for x in before if x not in after]
    assert len(removed)==1 and removed[0]['path']==(native_build.BUILD/'cargo_home/.global-cache').relative_to(c.ROOT).as_posix()
    assert any(x['path'].endswith('/neural_search.rs') for x in after)
    assert any(x['path'].endswith('/Cargo.lock') for x in after)
