"""Synthetic boundary/parity checks; no checkpoints, inference or search."""
import ast
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import pytest

from tools.matched_strength_20261020 import common as c, local, verify as v, worker
from tools.external_strength_screen_20261018 import rules

CONFIG = {'schema':'matched_strength_config_v1','pair_count':30,'execution_cap_seconds':259200,
          'post_game_reserve_seconds':64800,'evidence_cap_bytes':8*1024**3,
          'cumulative_memory_wait_seconds':21600,'bootstrap_seed':202610206701,
          'bootstrap_replicates':50000,'confidence_level':.95}
MATERIAL = {'schema':'matched_strength_private_seed_material_v1','private_nonce':'a'*64}


@pytest.fixture(autouse=True)
def configured(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'config', lambda: deepcopy(CONFIG))
    path = tmp_path/'outputs/matched_strength_20261020_01/config.json'
    path.parent.mkdir(parents=True); path.write_text(json.dumps(CONFIG))
    monkeypatch.setattr(v, 'ROOT', tmp_path)


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef)}


def test_unchanged_numerical_orchestration_and_full_assurance_ast():
    names = {'local.py':('initialize','state_from_observer_payload','make_observer_payload','choose'),
             'common.py':('pilot_observer','pilot_context','make_observer_payload','opponent_observer',
                          'sample_worlds','stress_priority','stress_cells','expected_stress_cells',
                          'check_wrapper','check_audit_proof'),
             'verify.py':('sample_worlds','_hash','check_outcome','check_trace_structure','audit_decision'),
             'worker.py':('trace_path',)}
    for filename, selected in names.items():
        old = functions(c.ROOT/'tools/live_game_operational_20261019'/filename)
        new = functions(c.ROOT/'tools/matched_strength_20261020'/filename)
        assert all(old[key] == new[key] for key in selected)
    assert c.LOCAL_PROFILE_ID == v.PROFILE_ID == 'four_candidate128_corrected_live_operational_20261019_v1'
    assert c.ADAPTER_ID == 'matched_strength_adapter_20261020_v1'
    for name in ('registered_model','score_actions','trajectory','proposal_for_root'):
        assert getattr(local,name) is getattr(local.qualified,name)


def test_worker_only_changes_import_module_paths():
    old=ast.parse((c.ROOT/'tools/live_game_operational_20261019/worker.py').read_text())
    new=ast.parse((c.ROOT/'tools/matched_strength_20261020/worker.py').read_text())
    for node in ast.walk(new):
        if isinstance(node,ast.ImportFrom) and node.module:
            node.module=node.module.replace('tools.matched_strength_20261020','tools.live_game_operational_20261019')
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)


def test_complete_balanced_pairs_and_private_referee_entropy():
    rows=c.make_schedule(MATERIAL)
    assert len(rows)==60 and sorted(x['game'] for x in rows)==list(range(60))
    assert [x['game'] for x in rows[:6]]==[0,2,4,1,3,5]
    assert len({x['seed'] for x in rows})==30
    alternate=c.make_schedule({**MATERIAL,'private_nonce':'b'*64})
    assert not ({x['seed'] for x in rows}&{x['seed'] for x in alternate})
    for pair in range(30):
        two=[x for x in rows if x['pair']==pair]
        assert {x['fit'] for x in two}=={c.FITS[pair%3]}
        assert {x['local_seat'] for x in two}=={0,1} and len({x['seed'] for x in two})==1
        contexts=[c.operational_context(x['game'],0) for x in two]
        assert contexts[0]['stream']!=contexts[1]['stream']
        for ctx in contexts:
            assert set(ctx)=={'mode','namespace','pair','game','turn','fit','stream','identity'}
            assert MATERIAL['private_nonce'] not in c.canonical(ctx)
            assert two[0]['seed'] not in c.canonical(ctx)
            assert v.check_context(ctx,{'current_player':c.local_seat(ctx['game']),'turn_index':0})==ctx
    assert [c.operational_context(g,0)['fit'] for g in c.VALIDATION_GAMES]==list(c.FITS)
    assert all(c.local_seat(g)==0 for g in c.VALIDATION_GAMES)


@pytest.mark.parametrize('game',[True,False,-1,60,0.,'0'])
def test_outside_schedule_rejected(game):
    with pytest.raises(ValueError): c.operational_context(game,0)


@pytest.mark.parametrize('damage',['fit','stream','pair','identity','namespace','private_nonce','referee_seed'])
def test_common_and_independent_context_reject_tampering(damage):
    ctx=c.operational_context(1,0)
    if damage=='fit':ctx[damage]=c.FITS[1]
    elif damage=='pair':ctx[damage]=True
    elif damage=='identity':ctx[damage]=['wrong',1,0]
    else:ctx[damage]='wrong'
    with pytest.raises(ValueError):c.check_context(ctx)
    with pytest.raises(ValueError):v.check_context(ctx)


@pytest.mark.parametrize('damage',[{'pair_count':31},{'pair_count':True},{'bootstrap_replicates':49999},
    {'confidence_level':.9},{'post_game_reserve_seconds':600},{'cumulative_memory_wait_seconds':1},
    {'private_nonce':'not_public'},{'execution_cap_seconds':3600}])
def test_prospective_config_rejects_unbalanced_or_relaxed_contract(damage):
    with pytest.raises(ValueError):c.validate_config({**CONFIG,**damage})


def test_observer_excludes_hidden_referee_material_and_deck_order():
    state=rules.CorrectedEnv(seed=123).initial_state()
    game=next(g for g in range(6) if c.local_seat(g)==state.current_player)
    first=c.make_observer_payload(rules.serialize_snapshot(state),c.operational_context(game,0))
    for deck in state.hidden_tier_decks.values():deck.reverse()
    second=c.make_observer_payload(rules.serialize_snapshot(state),c.operational_context(game,0))
    assert first==second
    local.state_from_observer_payload(first)


def test_worker_exact_trace_delivery_and_exclusive_use(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path)
    path=tmp_path/'continuations.jsonl.gz';monkeypatch.setenv('LIVE_TRACE_PATH',str(path))
    decision={'action':{'fixture':True},'record':{}}
    def choose(payload,*,boundary,trace_sink):
        boundary();trace_sink({'fixture':'trace'});return decision
    context={'mode':'local','local':SimpleNamespace(choose=choose),'used':False}
    result=worker.handle(context,{'observer':{'fixture':True}})
    assert result['decision']==decision and result['trace']['records']==1
    assert result['trace']['sha256']==c.file_hash(path)
    assert gzip.decompress(path.read_bytes())==b'{"fixture":"trace"}\n'
    with pytest.raises(ValueError,match='Exactly one'):worker.handle(context,{'observer':{}})


def test_worker_rejects_unapproved_path_and_mode_before_models(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path/'allowed')
    with pytest.raises(ValueError):worker.trace_path(tmp_path/'continuations.jsonl.gz')
    monkeypatch.setenv('LIVE_WORKER_MODE','opponent')
    monkeypatch.setattr(local,'initialize',lambda:pytest.fail('No model initialization'))
    with pytest.raises(ValueError):worker.initialize()
