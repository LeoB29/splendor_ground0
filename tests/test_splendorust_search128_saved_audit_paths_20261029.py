"""Model-free regressions for the exact saved Windows path failure."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import inspect
import sys
import pytest

from tools.splendorust_search128_saved_audit_20261029 import path_runtime as p
from tools.splendorust_search128_strength_20261028.review_runtime import ReviewRuntime

ROOT = Path(__file__).resolve().parents[1]
FROZEN_OUT = ROOT/'outputs/splendorust_search128_resume_20261029_01'


@pytest.fixture
def reader():
    return p.AuthorizedRuntimeReview(ROOT, FROZEN_OUT, default_seconds=60.)


def test_exact_original_failure_is_reproduced_then_narrowly_dispatched(reader):
    failure = json.loads((FROZEN_OUT/'postrun/independent_review_failure.json').read_text())
    assert failure['status'] == 'failed_no_acceptance' and 'Canonical project-relative path' in failure['error']
    item = reader.absolute_runtime_descriptors[0]
    assert item['path'].endswith(r'torch\_C.cp312-win_amd64.pyd')
    old = ReviewRuntime(ROOT, FROZEN_OUT, default_seconds=60.)
    with pytest.raises(ValueError, match='Canonical project-relative path'):
        old.verify(item)
    assert reader.dispatch_descriptor(item) == Path(item['path'])
    assert json.loads((FROZEN_OUT/'postrun/independent_review_failure.json').read_text()) == failure


def test_all95_actual_absolute_descriptors_dispatch_without_loading_libraries(reader):
    before = set(sys.modules)
    items = reader.absolute_runtime_descriptors
    assert len(items) == 95
    for item in items:
        assert reader.dispatch_descriptor(item) == Path(item['path'])
    assert set(sys.modules) == before
    assert reader.closure_receipt['sha256'] == p.CLOSURE_SHA256
    assert reader.out == FROZEN_OUT.resolve() and reader.shared_deadlines is None


def test_small_actual_runtime_file_hash_and_cache_identity_are_verified(reader):
    item = min((x for x in reader.absolute_runtime_descriptors if x['bytes'] > 0), key=lambda x:x['bytes'])
    assert item['bytes'] < 1024
    reader.verify(item)
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
    before = reader.metrics['cached_verifications']; reader.verify(item)
    assert reader.metrics['cached_verifications'] == before+1
    assert reader.revalidate_identities()['status'] == 'passed'


def test_relative_project_descriptor_uses_unchanged_verifier(reader, monkeypatch):
    item = reader.pin(Path(__file__))
    calls=[]; original=ReviewRuntime.verify
    def spy(self, value):
        calls.append(deepcopy(value)); return original(self, value)
    monkeypatch.setattr(ReviewRuntime,'verify',spy)
    reader.verify(item)
    assert calls == [item] and reader.dispatch_descriptor(item) == ROOT/item['path']


@pytest.mark.parametrize('name',[
    '../x','tools/../x','tools/./x',r'tools\x','/etc/passwd',r'C:relative.txt',
    r'\\server\share\x',r'\\?\C:\Windows\x',r'\\.\C:\Windows\x',
    'C:/Windows/../x',r'C:\Windows\.\x',r'C:\Windows\\x',
    r'C:\Windows\x:stream',r'C:\Windows\x.',r'C:\Windows\x ',
    'C:/Windows/\x00x','//server/share/x',r'\Windows\x','', '.',
])
def test_malformed_or_noncanonical_paths_rejected(reader,name):
    item=dict(path=name,bytes=0,sha256='0'*64)
    with pytest.raises(ValueError): reader.verify(item)


@pytest.mark.parametrize('mutation',['path','forward_slash_alias','case_alias','bytes','sha','bool_bytes','bad_sha','extra'])
def test_tampered_absolute_authority_rejected_before_io(reader,monkeypatch,mutation):
    item=deepcopy(reader.absolute_runtime_descriptors[0])
    if mutation=='path':item['path']=r'C:\Windows\not_authorized.dll'
    elif mutation=='forward_slash_alias':item['path']=item['path'].replace('\\','/')
    elif mutation=='case_alias':item['path']=item['path'].swapcase()
    elif mutation=='bytes':item['bytes']+=1
    elif mutation=='sha':item['sha256']='0'*64
    elif mutation=='bool_bytes':item['bytes']=False
    elif mutation=='bad_sha':item['sha256']='z'*64
    else:item['extra']='not permitted'
    monkeypatch.setattr(reader,'pin',lambda *a,**k:pytest.fail('Invalid authority reached I/O'))
    with pytest.raises(ValueError):reader.verify(item)


def test_authenticated_allowlist_cannot_be_modified_through_export(reader):
    items=reader.absolute_runtime_descriptors;items[0]['sha256']='0'*64
    assert reader.absolute_runtime_descriptors[0]['sha256']!='0'*64
    with pytest.raises(TypeError):reader._authorized_runtime['new']=(0,'0'*64)


def test_wrong_closure_hash_fails_before_allowlist_construction(monkeypatch):
    original=ReviewRuntime.pin
    def corrupt(self,path,**kwargs):
        pin,raw=original(self,path,**kwargs);pin['sha256']='0'*64;return pin,raw
    monkeypatch.setattr(ReviewRuntime,'pin',corrupt)
    with pytest.raises(ValueError,match='authenticate before'):
        p.AuthorizedRuntimeReview(ROOT,FROZEN_OUT,default_seconds=60.)


def test_inherited_guards_cache_reparse_and_identity_methods_unchanged():
    for name in ('pin','_expect','physical','_stat','_parents','guard','verify_many','revalidate_identities','read_path','exact_tree','absent'):
        assert getattr(p.AuthorizedRuntimeReview,name) is getattr(ReviewRuntime,name)
    source=inspect.getsource(p)
    assert 'subprocess' not in source and 'import torch' not in source and 'exec(' not in source


def test_wrong_physical_bytes_still_fatal_after_authorization(reader,monkeypatch):
    item=reader.absolute_runtime_descriptors[0]
    monkeypatch.setattr(reader,'pin',lambda path:dict(path=str(path),bytes=item['bytes'],sha256='0'*64))
    with pytest.raises(ValueError,match='runtime bytes differ'):reader.verify(item)


def test_contradictory_expected_hash_and_api_read_failures_remain_fatal(reader,monkeypatch):
    item=reader.absolute_runtime_descriptors[0];path=reader.dispatch_descriptor(item)
    reader._expect(path,dict(item,sha256='0'*64))
    with pytest.raises(ValueError,match='Contradictory expected'):reader.verify(item)
    other=reader.absolute_runtime_descriptors[1]
    def failure(path):raise OSError('preserved read failure')
    monkeypatch.setattr(reader,'pin',failure)
    with pytest.raises(OSError,match='preserved read failure'):reader.verify(other)
