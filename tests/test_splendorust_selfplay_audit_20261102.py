"""Narrow saved-failure reproduction and no-overwrite verifier regressions."""
import ast
import inspect
import json
from pathlib import Path
from copy import deepcopy
import pytest

from tools.splendorust_selfplay_audit_20261102 import verifier as v


def test_actual_saved_failure_reproduces_without_changing_existing_receipt():
    failure=v.c.read(v.c.OUT/'failure.json')
    assert 'FileExistsError' in failure['error'] and 'capture_qualification.json' in failure['error']
    path=v.c.OUT/'capture_qualification.json';before=v.c.artifact(path);value=v.c.read(path)
    with pytest.raises(FileExistsError):
        v.original.save_or_verify(path,value,True)
    assert v.c.artifact(path)==before
    assert v.verify_existing(path,value)==before


def test_actual_existing_receipt_tampering_is_rejected_read_only():
    path=v.c.OUT/'capture_qualification.json';before=v.c.artifact(path);bad=deepcopy(v.c.read(path))
    bad['full_scientific_parity_pairs']+=1
    with pytest.raises(ValueError,match='Independent saved result differs'):
        v.verify_existing(path,bad)
    assert v.c.artifact(path)==before


def test_saved_verifier_preserves_every_original_check_except_publication():
    old=ast.parse(inspect.getsource(v.original.verify_saved)).body[0]
    new=ast.parse(inspect.getsource(v.verify_saved)).body[0]
    old.args.kwonlyargs=[];old.args.kw_defaults=[]
    changes=0
    for statement in old.body:
        if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call):
            call=statement.value
            if isinstance(call.func,ast.Name) and call.func.id=='verify_capture_fixtures':
                for keyword in call.keywords:
                    if keyword.arg=='writing':keyword.value=ast.Constant(False);changes+=1
    old.body=[node for node in old.body if not(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='save_or_verify')]
    assert changes==1
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)


def test_existing_missing_receipt_is_not_created(tmp_path):
    path=tmp_path/'absent.json'
    with pytest.raises(FileNotFoundError):v.verify_existing(path,{'expected':True})
    assert not path.exists()


def test_new_audit_receipts_are_exclusive_and_separate(monkeypatch,tmp_path):
    monkeypatch.setattr(v.c,'ROOT',tmp_path);monkeypatch.setattr(v.c,'OUT',tmp_path/'failed')
    monkeypatch.setattr(v,'AUDIT_OUT',tmp_path/'audit')
    result=v.publish_new('proof.json',{'passed':True})
    assert json.loads((tmp_path/'audit/proof.json').read_text())=={'passed':True}
    with pytest.raises(FileExistsError):v.publish_new('proof.json',{'passed':False})
    assert v.c.artifact(tmp_path/'audit/proof.json')==result
    assert not (tmp_path/'failed').exists()


@pytest.mark.parametrize('name',['../capture_qualification.json','x/proof.json','proof.txt'])
def test_new_receipt_cannot_escape_audit_or_use_old_path(monkeypatch,tmp_path,name):
    monkeypatch.setattr(v.c,'ROOT',tmp_path);monkeypatch.setattr(v.c,'OUT',tmp_path/'failed')
    monkeypatch.setattr(v,'AUDIT_OUT',tmp_path/'audit')
    with pytest.raises(ValueError):v.publish_new(name,{})
    assert not (tmp_path/'audit').exists()


def test_new_receipt_writer_rejects_old_evidence_destination(monkeypatch,tmp_path):
    monkeypatch.setattr(v.c,'ROOT',tmp_path);monkeypatch.setattr(v.c,'OUT',tmp_path/'failed')
    monkeypatch.setattr(v,'AUDIT_OUT',tmp_path/'failed/subdir')
    with pytest.raises(ValueError):v.publish_new('proof.json',{})
    assert not (tmp_path/'failed').exists()
