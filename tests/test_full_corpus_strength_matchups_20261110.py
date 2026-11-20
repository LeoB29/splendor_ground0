"""Cross-matchup counterexamples: both learned players, exact roles and multiplicity."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
os.environ.setdefault('INDEPENDENT_DATA_REPETITION','0')
os.environ.setdefault('FULL_CORPUS_MATCHUP','mixed_parent')
from tools.splendorust_full_corpus_20261110 import common as root, schedule
from tools.splendorust_full_corpus_20261110.strength import common as c, adapters, checkpoint as cp, design, analysis, transport, worker

@pytest.fixture
def matchup(monkeypatch):
    def select(name):
        monkeypatch.setattr(c,'MATCHUP',name)
        monkeypatch.setattr(c,'ROUTE_ARMS',schedule.MATCHUP_ARMS[name])
        monkeypatch.setattr(design,'BOOTSTRAP_SEED',schedule.BOOTSTRAP_SEEDS[name]['strength_primary'])
        monkeypatch.setattr(design,'INDIVIDUAL_SEEDS',tuple(schedule.BOOTSTRAP_SEEDS[name][f'strength_r{r}'] for r in (0,1)))
        rows=schedule.strength_schedule(name)
        monkeypatch.setattr(design,'_reserved',lambda:{(x['stage'],x['repetition'],x['local_pair_index'],x['order_in_pair']):(x,{}) for x in rows})
    return select


def test_canonical_route_map_and_all_global_bounds_agree():
    assert root.MATCHUPS == schedule.MATCHUPS
    assert root.MATCHUP_ARMS == schedule.MATCHUP_ARMS
    rows=schedule.strength_schedule()
    assert len(rows)==root.MAX_TOTAL_GAMES==1236
    assert len({r['seed_decimal'] for r in rows})==618
    assert set(x['candidate'] for x in rows)=={f'r{r}_{arm}_joint' for r in (0,1) for arm in ('mixed','terminal')}
    for name in schedule.MATCHUPS:
        group=schedule.strength_schedule(name)
        assert len(group)==412
        assert len({r['seed_decimal'] for r in group})==206
        assert len([x for x in group if x['stage']=='main'])==400
        assert len([x for x in group if x['stage']=='operational'])==12


@pytest.mark.parametrize('name',schedule.MATCHUPS)
@pytest.mark.parametrize('rep',(0,1))
def test_actual_fresh_import_binds_each_matchup_before_models(name,rep):
    env=os.environ.copy();env['FULL_CORPUS_MATCHUP']=name;env['INDEPENDENT_DATA_REPETITION']=str(rep)
    code="from tools.splendorust_full_corpus_20261110.strength import common as c,checkpoint,adapters,worker;import sys;assert 'torch' not in sys.modules;assert c.OUT==c.STUDY_OUT/'strength'/c.MATCHUP/f'r{c.REPETITION}';assert c.fit_id('candidate')==f'r{c.REPETITION}_{c.ROUTE_ARMS[\"candidate\"]}_joint'"
    result=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr.decode()


@pytest.mark.parametrize('bad',('', 'mixed_parent ', 'broad_parent','Mixed_parent','0'))
def test_invalid_matchup_namespace_rejected_before_models(bad):
    env=os.environ.copy();env['FULL_CORPUS_MATCHUP']=bad
    result=subprocess.run([sys.executable,'-B','-c','from tools.splendorust_full_corpus_20261110.strength import common'],env=env,capture_output=True,timeout=10)
    assert result.returncode!=0 and b'FULL_CORPUS_MATCHUP' in result.stderr


@pytest.mark.parametrize('name',schedule.MATCHUPS)
@pytest.mark.parametrize('route',('candidate','control'))
def test_live_loader_uses_exact_role_specific_checkpoint(monkeypatch,matchup,name,route):
    matchup(name);seen=[]
    model=SimpleNamespace(training=False)
    manifests={r:{'path':c.fit_id(r)+'/manifest.json'} for r in c.VARIANTS if not c.is_parent_route(r)}
    bindings={r:{'native_reference':{'path':c.fit_id(r)+'/state.bin'}} for r in c.VARIANTS}
    monkeypatch.setattr(cp,'load_parent',lambda:seen.append(('parent',None)) or model)
    monkeypatch.setattr(cp,'load_candidate',lambda manifest:(seen.append(('learned',manifest)) or model,{}))
    class Dummy:
        def __init__(self,profile,shared_from):self.model=shared_from.model
        def close(self):pass
    monkeypatch.setattr(adapters,'BaselineOpponent',Dummy)
    result=adapters.initialize_adapters(None,{'16':{},'128':{}},route=route,freeze={'trained_manifests':manifests,'checkpoint_bindings':bindings})
    assert seen==[('parent',None)] if c.is_parent_route(route) else seen==[('learned',manifests[route])]
    assert result['search'].model is result['reference'].model is model
    assert result['search'].native_reference_bin==c.ROOT/bindings[route]['native_reference']['path']


@pytest.mark.parametrize('route',('candidate','control'))
def test_direct_target_contrast_profiles_import_both_weights_without_parent_relabelling(monkeypatch,matchup,route):
    matchup('mixed_terminal')
    original=adapters.original_profile(128)
    descriptor={'path':f'unit/{route}/manifest.json','bytes':1,'sha256':'a'*64}
    manifest={'schema':cp.SCHEMA,'repetition':c.REPETITION,'route':route,'matchup':c.MATCHUP,
              'checkpoint':{'path':f'unit/{route}/state.bin','bytes':1,'sha256':'b'*64},
              'tensor_manifest':{'path':f'unit/{route}/tensors.json','bytes':1,'sha256':'c'*64}}
    read=c.read
    monkeypatch.setattr(c,'verify',lambda *args,**kwargs:None)
    monkeypatch.setattr(c,'read',lambda path:deepcopy(manifest) if Path(path)==c.ROOT/descriptor['path'] else read(path))
    profile=adapters.candidate_profile(128,descriptor,manifest['checkpoint'],route=route)
    changed={key for key in original if profile[key]!=original[key]}
    assert changed=={'profile_id','wrapper_id','checkpoint','artifacts','execution_policy'}
    assert profile['checkpoint']==manifest['checkpoint']['path']
    assert profile['execution_policy']['original_profile']['path'].endswith('splendorust_search128_strength_20261028_01/profiles/puct128.json')
    assert profile['iterations']==128 and profile['world_pool']==3 and profile['depth']==64 and profile['cpuct']==.4
    assert 'mixed_terminal' in profile['profile_id'] and route in profile['profile_id']
    other='control' if route=='candidate' else 'candidate'
    with pytest.raises(ValueError,match='learned profile route'):
        adapters.candidate_profile(128,descriptor,manifest['checkpoint'],route=other)


@pytest.mark.parametrize('name',schedule.MATCHUPS)
@pytest.mark.parametrize('route',('candidate','control'))
def test_exact_graph_keeps_historical_parity_only_for_actual_parent(monkeypatch,matchup,tmp_path,name,route):
    from tools.splendorust_boundary_timing_20261025 import runner
    matchup(name)
    fixtures=[{'fixture_id':str(i),'batch':1} for i in range(11)]
    row={'exact_float_equal':True,'float32_bits_equal':True,'eager_output_sha256':'new'}
    result={'rows':[row.copy() for _ in fixtures],'graph_identity_sha256':'graph'}
    evidence=[dict(fixture_id=f['fixture_id'],batch=1,eager_outputs=[2.],candidate_outputs=[2.],comparison=row.copy()) for f in fixtures]
    plan={'fixtures':fixtures,'provenance_by_fixture':{f['fixture_id']:{'historical_outputs':[1.],'historical_output_sha256':'old'} for f in fixtures}}
    saved={'schema':worker.graph.PARITY_SCHEMA,'summary':result,'fixtures':fixtures,'evidence':evidence}
    def read(path):
        if Path(path).name=='plan.json':return plan
        if Path(path).name=='graph_parity.json':return saved
        if Path(path).name=='player_identity.json':return {'route':route}
        raise AssertionError(path)
    monkeypatch.setattr(c,'read',read);monkeypatch.setattr(c,'verify',lambda *a,**k:None)
    monkeypatch.setattr(runner,'validate_graph_parity',lambda *a:None)
    monkeypatch.setattr(worker.graph,'compare_outputs',lambda *a:row.copy())
    if c.is_parent_route(route):
        with pytest.raises(ValueError,match='historical graph outputs'):worker.verify_graph_admission(tmp_path,{'graph_parity_plan':{'path':'plan.json'}},result)
    else:
        assert worker.verify_graph_admission(tmp_path,{'graph_parity_plan':{'path':'plan.json'}},result)['accepted'] is True
        row['float32_bits_equal']=False
        with pytest.raises(ValueError):worker.verify_graph_admission(tmp_path,{'graph_parity_plan':{'path':'plan.json'}},result)


def games(rep,wins):
    return [dict(s,complete=True,winner=s['challenger_seat'] if s['pair_index']<wins else 1-s['challenger_seat'],reason='natural',loser=None,administrative_forfeit=False) for s in design.make_schedule('main',rep)]


@pytest.mark.parametrize('name',schedule.MATCHUPS)
def test_primary_and_parent_interval_roles_have_independent_reproduction(monkeypatch,matchup,name):
    matchup(name);monkeypatch.setattr(analysis,'REPLICATES',1000)
    data={0:games(0,65),1:games(1,65)}
    value=analysis.analyze_pooled(data,True)
    assert value['reference_arm']==schedule.MATCHUP_ARMS[name]['control']
    assert value['checkpoint_names']==[c.fit_id('candidate',r) for r in (0,1)]
    assert value['reference_checkpoints']==[c.fit_id('control',r) for r in (0,1)]
    assert value['useful_gain_established']==(name=='mixed_terminal')
    assert value['familywise_parent_useful_gain']==(name!='mixed_terminal')
    assert not value['automatic_promotion'] and not value['individual_checkpoint_selection']
    for key in ('official','conservative'):
        assert value[key]['paired_ci975'][0]<=value[key]['paired_ci95'][0]
        assert value[key]['paired_ci975'][1]>=value[key]['paired_ci95'][1]
    assert analysis.verify_pooled(data,value)['verified_result_sha256']==c.digest(value)
    value['official']['paired_ci975'][0]+=.001
    with pytest.raises(ValueError):analysis.verify_pooled(data,value)


def test_97_5_requirement_is_stricter_than_descriptive_95(monkeypatch):
    import numpy as np
    score={'numerator':440,'denominator':800,'pairs':200,'mean_score':.55,'seat_numerators':{'0':220,'1':220},'seat_denominator':400}
    totals=np.array([360]*20+[408]*30+[440]*950,dtype=np.int64)
    value=analysis._main_score(score,totals)
    assert value['useful_gain'] is True
    assert value['familywise_parent_useful_gain'] is False
    assert value['paired_ci95'][0]>.5 and value['paired_ci975'][0]<.5


@pytest.mark.parametrize('name',schedule.MATCHUPS)
def test_worker_supervision_rejects_other_matchup(monkeypatch,matchup,name):
    matchup(name)
    start={'repetition':0,'matchup':name,'phase':'main_block','block_start':0,'owner':{'pid':1,'created':1.},'root_claim':{},'root_phase_clock':{}}
    segment={'repetition':0,'matchup':name,'mode':'main_block','block_start':0,'claim':{},'phase_clock':{},'freeze_sha256':'x'}
    freeze={'repetition':0,'matchup':name,'root_freeze':{'sha256':'x'}}
    transport.validate_segment_supervisor(start,segment,{'pid':1,'created':1.},freeze,'main_block',0)
    segment['matchup']=next(m for m in schedule.MATCHUPS if m!=name)
    with pytest.raises(ValueError,match='Exact matchup'):transport.validate_segment_supervisor(start,segment,{'pid':1,'created':1.},freeze,'main_block',0)
