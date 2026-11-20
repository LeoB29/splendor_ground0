"""Recovery entrypoint admission and unchanged scientific contract regressions."""
import ast
from pathlib import Path
import pytest
from . import common as c
from .collection import worker,audit,collector,qualification
from tools.splendorust_collection4096_20261115.collection import collector as old_collector


def test_new_output_does_not_alias_frozen_original():
    assert c.OUT!=c.ORIGINAL_OUT and not c.OUT.is_relative_to(c.ORIGINAL_OUT)
    assert c.PACKAGE=='tools.splendorust_collection4096_resume_20261118_v2'


def test_entire_production_schedule_and_fixed_seeds_preserved():
    assert collector.build_plan()==old_collector.build_plan()
    assert collector.cells()==old_collector.cells()
    assert len(collector.cells())==208


def test_actual_corrected_fixture_schedule_preserved():
    from tools.splendorust_collection4096_20261115.collection import qualification as previous
    assert qualification.build_plan()==previous.build_plan()


def test_original_qualification_gate_reused_without_relabel():
    gate=worker.verify_full_qualification_gate()
    assert gate['plan']==c.artifact(c.ORIGINAL_OUT/'freeze.json')
    assert len(gate['actual_exits'])==3
    assert all(c.resolve(x).is_relative_to(c.ORIGINAL_OUT) for x in gate['actual_exits'])


def test_recovery_qualification_calls_actual_selected_collector_not_helpers_only():
    tree=ast.parse(Path(worker.__file__).read_text())
    source=Path(worker.__file__).read_text()
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert 'run_qualification' in calls and 'run_continuation' in calls and 'run_cell' in calls
    assert 'original_engineering_games_reused=2' in source
    assert 'collector.run_engineering(' not in source


def test_fresh_qualifier_audit_retains_actual_entrypoint_capture_and_loader_checks():
    tree=ast.parse(Path(audit.__file__).read_text())
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit_recovery_qualification')
    calls={n.func.attr for n in ast.walk(func) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert {'audit_entrypoint','audit_decision','qualify_captured_rows','audit_full_loader_fixture',
            'runtime_closure','audit_initialization','complete_attribution'}<=calls


@pytest.mark.parametrize('mode',['fit','qualification','resume_game','skip_replays','audit_qualification'])
def test_cli_refuses_unapproved_worker_roles(mode,monkeypatch):
    monkeypatch.setattr('sys.argv',['worker','--mode',mode,'--plan-sha256','x','--output','x'])
    with pytest.raises(SystemExit) as caught:worker.main()
    assert caught.value.code==2


def test_no_overwriting_original_evidence(tmp_path):
    with pytest.raises(ValueError,match='New study output only'):
        c.save(c.ORIGINAL_OUT/'must_not_exist.json',{})
    assert not (c.ORIGINAL_OUT/'must_not_exist.json').exists()
