"""Saved actual suite-to-play-to-closure evidence; no generated games/models."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import pytest
from tools import splendorust_search128_closure_check_20261029 as check


def data(name='main_g38_challenger'):
    c=check.c;folder=c.OUT/'workers'/name
    return dict(name=name,route=check.WORKERS[name],
        payloads={p.stem:c.read(p).get('operation') for p in (folder/'startup_inputs').glob('*.json')},
        summary=c.read(c.OUT/'recovery_qualification_summary.json'),
        entrypoint=c.read(c.OUT/'recovery_entrypoint_checks'/f'{check.WORKERS[name]}.json'),
        clock=c.read(folder/'request-00000002.json'),graph_clock=c.read(folder/'graph_parity_clock.json'),
        initialization=c.read(folder/'initialization.json'),identity=c.read(folder/'route_identity.json'))


@pytest.mark.parametrize('name',list(check.WORKERS))
def test_exact_retained_lifecycle(name):
    value=check.validate_lifecycle_data(**data(name))
    assert value['startup_requests']==2 and value['entrypoint_request_id']=='request-00000002'


@pytest.mark.parametrize('mutation',['unknown_folder','wrong_route','suite_before_graph','repeated_suite',
    'missing_graph','wrong_worker_binding','wrong_count','changed_clock','changed_ready','missing_other_arm'])
def test_lifecycle_rejects_unapproved_or_mutated_data(mutation):
    value=data()
    if mutation=='unknown_folder':value['name']='main_g39_challenger'
    elif mutation=='wrong_route':value['route']='frozen'
    elif mutation=='suite_before_graph':value['payloads']={'request-00000001':'entrypoint_checks','request-00000002':'graph_parity'}
    elif mutation=='repeated_suite':value['payloads']['request-00000003']='entrypoint_checks'
    elif mutation=='missing_graph':value['payloads'].pop('request-00000001')
    elif mutation=='wrong_worker_binding':value['summary']['receipts'][0]['worker_folder']='workers/some_other_worker'
    elif mutation=='wrong_count':value['summary']['scientific_decisions']=104
    elif mutation=='changed_clock':value['clock']['elapsed_seconds']+=.001
    elif mutation=='changed_ready':value['identity']['metadata']={}
    elif mutation=='missing_other_arm':value['summary']['receipts'].pop()
    with pytest.raises(ValueError):check.validate_lifecycle_data(**value)


def test_full_saved_both_suites_ordinary_closure_and_segmented_game():
    c=check.c;freeze=c.read(c.OUT/'freeze.json');boundary=check.Boundary(120)
    proofs=[]
    for name in check.WORKERS:
        folder=c.OUT/'workers'/name
        with pytest.raises(ValueError,match='Unexpected entrypoint request during play'):
            check.held.verify_raw_transport(folder,freeze,boundary)
        proof=check.verify_closed_worker(folder,freeze,boundary);proofs.append(proof)
        assert proof['status']=='passed' and proof['raw']['startup_requests']==2
        assert proof['raw']['requests']==9 and proof['raw']['failed_requests']==0
        assert proof['placement_requests']==9
    assert sum(p['raw']['requests'] for p in proofs)==18
    game=check.verify_saved_game(freeze,boundary)
    assert game['new_ordinary_moves']==14 and game['prior_accepted_moves']==39 and game['recovered_actions']==1
    assert game['proof']['transitions']==54 and game['proof']['failed_moves']==0
    assert game['original_attempt_accepted'] is False and game['model_calls']==0
    assert c.read(c.OUT/'completion.json')['status']=='incomplete'


def test_real_raw_packet_corruption_is_rejected(monkeypatch):
    original=check._packet_evidence;changed=[]
    def corrupt(raw):
        value=original(raw)
        if value['packet'].get('type')=='response' and not changed:
            value['packet']['request_id']='request-99999999';changed.append(True)
        return value
    monkeypatch.setattr(check,'_packet_evidence',corrupt)
    with pytest.raises(ValueError,match='Completed raw response differs'):
        check.verify_raw_transport(check.c.OUT/'workers/main_g38_challenger',check.c.read(check.c.OUT/'freeze.json'),check.Boundary(60))
    assert changed==[True]


def test_saved_placement_corruption_still_rejected():
    c=check.c;folder=c.OUT/'workers/main_g38_challenger';freeze=c.read(c.OUT/'freeze.json')
    admission=check.held.placement.verify_worker_placement(folder,freeze,'isolated')
    rid='request-00000003';parent=c.read(folder/'parent_profiles'/f'{rid}.json')
    proof=c.read(folder/'placement_requests'/f'{rid}.json');proof['placement_mode']='original'
    with pytest.raises(ValueError):
        check.held.placement.verify_saved_request(proof,freeze,'isolated',request_id=rid,
            player_pid=admission['player_pid'],supervisor_identity=admission['supervisor'],parent_profile=parent)


def test_only_declared_raw_verifier_statements_changed():
    old=inspect.getsource(check.held.verify_raw_transport)
    expected=old.replace("    paths=sorted(path for path in folder.glob", "    lifecycle=authenticated_lifecycle(folder,freeze)\n    paths=sorted(path for path in folder.glob",1)
    expected=expected.replace("c.require(folder.name.startswith('qualification_'),'Unexpected entrypoint request during play')",
        "c.require(rid==lifecycle['entrypoint_request_id'],'Exact authenticated retained-worker entrypoint request')")
    expected=expected.replace("            else:\n                plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])",
        "            else:\n                c.require(rid==lifecycle['graph_request_id'],'Exact graph request precedes actual suite')\n                plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])",1)
    expected=expected.replace("expected_startups=1+(1 if folder.name.startswith('qualification_') else 0)",
        "expected_startups=lifecycle['startup_requests']")
    assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(inspect.getsource(check.verify_raw_transport)))
    expected_closed=inspect.getsource(check.held.verify_closed_worker).replace('    from . import adapters\n','',1)
    assert ast.dump(ast.parse(expected_closed))==ast.dump(ast.parse(inspect.getsource(check.verify_closed_worker)))


def test_shared_original_deadline_and_no_live_child_ownership(monkeypatch):
    monkeypatch.setattr(check.time,'monotonic',lambda:check.c.ORIGINAL_MONOTONIC+86300)
    monkeypatch.setattr(check.time,'time',lambda:check.c.ORIGINAL_UNIX+86300)
    boundary=check.Boundary(900)
    assert boundary.end==check.c.ORIGINAL_MONOTONIC+86400 and boundary.wall_end==check.c.ORIGINAL_UNIX+86400
    with pytest.raises(ValueError):boundary.close_worker(check.c.OUT/'workers/main_g38_challenger',object())
    with pytest.raises(ValueError):boundary.close_worker(check.c.OUT/'workers/main_g39_challenger')
