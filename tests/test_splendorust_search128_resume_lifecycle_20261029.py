"""Saved-only lifecycle projection; no new decisions or frozen-file writes."""
from copy import deepcopy
from pathlib import Path
import ast
import inspect
import shutil
import tempfile
import time
import pytest
from tools.splendorust_search128_resume_20261029 import common as c,runner,transport
from tools import splendorust_search128_closure_check_20261029 as checker


def declarations():
    return {f'qualification_resume_{r}':dict(route=r,game_index=39,
        graph_request_id='request-00000001',entrypoint_request_id='request-00000002',
        ordinary_first_request_id='request-00000003') for r in ('research','frozen')}


class Boundary:
    def __init__(self):
        self.start=c.ORIGINAL_MONOTONIC;self.wall_start=c.ORIGINAL_UNIX
        self.end=time.monotonic()+120;self.waited=c.PRIOR_RAM_WAIT_SECONDS
        self.source_runtime_seconds=0.;self.placement_mode='isolated'
    def guard(self):assert time.monotonic()<self.end
    def remaining(self):return self.end-time.monotonic()
    def effective_deadline(self):return self.end
    def close_worker(self,folder,child=None):assert child is None


@pytest.fixture
def projection(monkeypatch):
    """Copy saved bytes into an explicitly test-only new lifecycle fixture."""
    base=c.OUT/'prelaunch';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test_lifecycle_',dir=base) as temporary:
        out=Path(temporary);monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
        freeze=c.read(c.FIRST_RECOVERY_OUT/'freeze.json');freeze['declared_worker_lifecycles']=declarations()
        summary=c.read(c.FIRST_RECOVERY_OUT/'recovery_qualification_summary.json')
        summary.update(first_recovery_scientific_decisions=119,prior_scientific_decisions=322,combined_scientific_decisions=441)
        shutil.copyfile(c.FIRST_RECOVERY_OUT/'launch.json',out/'launch.json')
        # Retain the authentic old READY/raw freeze identity; only the test's
        # prospective lifecycle declaration is added to the passed mapping.
        shutil.copyfile(c.FIRST_RECOVERY_OUT/'freeze.json',out/'freeze.json')
        for oldname,route in checker.WORKERS.items():
            folder=out/'workers'/f'qualification_resume_{route}'
            shutil.copytree(c.FIRST_RECOVERY_OUT/'workers'/oldname,folder)
            identity=c.read(folder/'route_identity.json');identity['module']=transport.OBSERVER_MODULE
            (folder/'route_identity.json').write_text(c.canonical(identity),encoding='utf-8')
            path=out/'recovery_entrypoint_checks'/f'{route}.json'
            c.save(path,c.read(c.FIRST_RECOVERY_OUT/'recovery_entrypoint_checks'/f'{route}.json'))
            item=next(x for x in summary['receipts'] if x['route']==route)
            item.update(worker_folder=folder.relative_to(c.ROOT).as_posix(),clock=c.artifact(path))
        c.save(out/'recovery_qualification_summary.json',summary)
        yield freeze,out,Boundary()


def test_actual_saved_suite_ordinary_and_full_closure_on_both_declared_trees(projection):
    freeze,out,boundary=projection
    proofs=[transport.verify_closed_worker(out/'workers'/name,freeze,boundary) for name in declarations()]
    assert sum(p['raw']['requests'] for p in proofs)==18
    assert all(p['raw']['startup_requests']==2 and p['placement_requests']==9 and p['graph']['fixtures']==11 for p in proofs)
    assert all(p['raw']['failed_requests']==0 and p['raw']['exact_bytes_checked'] for p in proofs)
    assert sum(len(list((out/'workers'/name/'inputs').glob('*.json'))) for name in declarations())==14
    assert c.read(c.FIRST_RECOVERY_OUT/'completion.json')['status']=='incomplete'


@pytest.mark.parametrize('mutation',['extra_declaration','wrong_route','wrong_game','wrong_entry_id','unknown_qualification','ordinary_game39'])
def test_reject_unfrozen_lifecycle(mutation):
    freeze={'declared_worker_lifecycles':declarations()};name='qualification_resume_research';route='research'
    if mutation=='extra_declaration':freeze['declared_worker_lifecycles']['qualification_other']=deepcopy(freeze['declared_worker_lifecycles'][name])
    elif mutation=='wrong_route':route='frozen'
    elif mutation=='wrong_game':freeze['declared_worker_lifecycles'][name]['game_index']=38
    elif mutation=='wrong_entry_id':freeze['declared_worker_lifecycles'][name]['entrypoint_request_id']='request-00000003'
    elif mutation=='unknown_qualification':name='qualification_research'
    else:name='main_g39_challenger'
    with pytest.raises(ValueError):transport.declared_lifecycle(c.OUT/'workers'/name,freeze,route)


@pytest.mark.parametrize('mutation',['repeated_suite','swapped_suite','missing_suite','wrong_summary','wrong_clock','wrong_worker'])
def test_saved_administrative_corruption_rejected(projection,mutation):
    freeze,out,boundary=projection;folder=out/'workers/qualification_resume_research'
    if mutation=='repeated_suite':c.save(folder/'startup_inputs/request-00000003.json',{'operation':'entrypoint_checks'})
    elif mutation=='swapped_suite':
        path=folder/'startup_inputs/request-00000001.json';path.write_text(c.canonical({'operation':'entrypoint_checks'}),encoding='utf-8')
    elif mutation=='missing_suite':(folder/'startup_inputs/request-00000002.json').unlink()
    elif mutation=='wrong_clock':
        path=folder/'request-00000002.json';value=c.read(path);value['elapsed_seconds']+=.001;path.write_text(c.canonical(value),encoding='utf-8')
    else:
        path=out/'recovery_qualification_summary.json';value=c.read(path)
        if mutation=='wrong_summary':value['combined_scientific_decisions']=322
        else:value['receipts'][0]['worker_folder']='wrong_worker'
        path.write_text(c.canonical(value),encoding='utf-8')
    with pytest.raises(ValueError):transport.verify_raw_transport(folder,freeze,boundary)


def test_normal_later_and_reproduction_trees_never_gain_an_actual_suite():
    freeze={'declared_worker_lifecycles':declarations()}
    for name,route in [('main_g40_challenger','research'),('main_g399_opponent','frozen'),('audit_47','research')]:
        life=transport.declared_lifecycle(c.OUT/'workers'/name,freeze,route)
        assert life['startup_requests']==1 and life['entrypoint_request_id'] is None
    for name,route in [('main_g400_challenger','research'),('audit_48','frozen'),('main_g40_opponent','research')]:
        with pytest.raises(ValueError):transport.declared_lifecycle(c.OUT/'workers'/name,freeze,route)


def test_raw_and_closed_proof_changes_are_only_the_declared_lifecycle():
    from tools.splendorust_search128_recovery_20261028 import transport as old
    expected=inspect.getsource(old.verify_raw_transport).replace(
        "    paths=sorted(path for path in folder.glob", "    lifecycle=authenticated_lifecycle(folder,freeze,route)\n    paths=sorted(path for path in folder.glob",1)
    expected=expected.replace("c.require(folder.name.startswith('qualification_'),'Unexpected entrypoint request during play')",
        "c.require(rid==lifecycle['entrypoint_request_id'],'Exact authenticated retained-worker entrypoint request')")
    expected=expected.replace("            else:\n                plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])",
        "            else:\n                c.require(rid==lifecycle['graph_request_id'],'Exact graph request precedes ordinary play')\n                plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])",1)
    expected=expected.replace("expected_startups=1+(1 if folder.name.startswith('qualification_') else 0)","expected_startups=lifecycle['startup_requests']")
    assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(inspect.getsource(transport.verify_raw_transport)))
    assert ast.dump(ast.parse(inspect.getsource(old.verify_closed_worker)))==ast.dump(ast.parse(inspect.getsource(transport.verify_closed_worker)))


def test_actual_prior_two_workers_and_full54_event_segment_are_preserved(monkeypatch):
    from tools.splendorust_search128_recovery_20261028 import runner as prior
    boundary=Boundary();first=c.read(c.FIRST_RECOVERY_OUT/'freeze.json')
    context=prior.recovery_context(first,boundary)
    context.update(first_recovery_freeze=first,closure_check=c.read(c.CLOSURE_OUT/'closure_check.json'))
    base=c.OUT/'prelaunch';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test_segments_',dir=base) as temporary:
        monkeypatch.setattr(c,'OUT',Path(temporary));monkeypatch.setattr(c,'WRITE_OBSERVER',None)
        c.save(c.OUT/'recovered_bridge.json',context['bridge'])
        proof=runner.verify_first_recovery(context,boundary)
        assert proof['actual_entrypoint_decisions']==119 and proof['first_recovery_accepted'] is False
        # A second final readback is idempotent and never overwrites the proof.
        pins=[c.artifact(p) for p in (c.OUT/'prior_saved_proofs').glob('*.json')]
        runner.verify_first_recovery(context,boundary);c.verify(pins)
        spec=context['schedule'][44]
        entry,game,rows=runner.game_manifest_entry(spec,context,first,boundary)
        assert spec['game_index']==38 and game['plies']==54 and len(rows)==53
        assert len(entry['events'])==54 and sum(e['kind']=='recovery_bridge' for e in entry['events'])==1
        assert entry['supplemental_proof_source']==c.artifact(c.CLOSURE_OUT/'closure_check.json')
        assert [e['source_freeze_sha256'] for e in entry['events'][:40]]==[c.OLD_FREEZE_SHA]*40
        assert [e['source_freeze_sha256'] for e in entry['events'][40:]]==[c.FIRST_RECOVERY_SHA]*14
        assert c.read(c.ROOT/entry['online_replay']['path'])['transitions']==54
        assert context['bridge']['source_row']['clock']['accepted'] is False

