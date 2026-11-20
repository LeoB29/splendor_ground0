"""Real saved closed-tree regression; explicit test projection, no new model work."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import time
import pytest

from tools.splendorust_search128_repeat_20261029 import common as c, transport, independent_review

PRIOR = c.ROOT/'outputs/splendorust_search128_recovery_20261028_01'
PRIOR_FREEZE_SHA = '524daba942dd6478ccbfc2a9293f391b1502ffca94163228aef8d1d3dba718f4'
WORKERS = {'main_g38_challenger': 'research', 'main_g38_opponent': 'frozen'}


@pytest.fixture
def saved_repeat_projection(monkeypatch):
    """Keep raw science/clock/compression bytes; project only admission metadata.

    This does not claim the historical worker ran in the new phase. The test
    explicitly retains its historical phase and freeze for authentic READY
    checks, while using the new repeat lifecycle and the real saved-only Guard.
    """
    assert c.file_hash(PRIOR/'freeze.json') == PRIOR_FREEZE_SHA
    base=c.OUT/'prelaunch';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test_closed_projection_',dir=base) as temporary:
        out=Path(temporary)
        monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
        monkeypatch.setattr(c,'PHASE','strength')
        freeze=c.read(PRIOR/'freeze.json')
        freeze['declared_worker_lifecycles']=transport.lifecycle_inventory()
        for name in ('launch.json','freeze.json'):shutil.copyfile(PRIOR/name,out/name)
        prior_summary=c.read(PRIOR/'recovery_qualification_summary.json')
        entries=[];retained=[]
        for oldname,route in WORKERS.items():
            old=PRIOR/'workers'/oldname
            folder=out/'workers'/f'qualification_repeat_{route}'
            shutil.copytree(old,folder)
            # Raw READY remains untouched. Its separately saved admission
            # module and lifecycle path are explicitly test-only projections.
            identity=c.read(folder/'route_identity.json');identity['module']=transport.OBSERVER_MODULE
            (folder/'route_identity.json').write_text(c.canonical(identity),encoding='utf-8')
            receipt_path=out/'entrypoint_checks'/f'{route}.json'
            c.save(receipt_path,c.read(PRIOR/'recovery_entrypoint_checks'/f'{route}.json'))
            item=deepcopy(next(x for x in prior_summary['receipts'] if x['route']==route))
            item.update(worker_folder=folder.relative_to(c.ROOT).as_posix(),clock=c.artifact(receipt_path))
            entries.append(item)
            for path in old.rglob('*'):
                if path.is_file() and path.name!='route_identity.json':
                    counterpart=folder/path.relative_to(old)
                    assert counterpart.read_bytes()==path.read_bytes()
                    retained.append((path,c.file_hash(path)))
        c.save(out/'qualification_summary.json',dict(status='passed',scientific_decisions=119,
            parent_requests=2,graph_cases=22,same_workers_retained_for_repeats=True,receipts=entries))
        mono,wall=time.monotonic(),time.time()
        guard=independent_review.Guard(dict(monotonic_origin=mono,unix_origin=wall),
            dict(monotonic_deadline=mono+120,unix_deadline=wall+120))
        yield freeze,out,guard
        # No projection ever writes the original failed attempt.
        for path,expected in retained:assert c.file_hash(path)==expected


def test_actual119_suites_14_moves_and_full_closed_proofs_with_real_reader_guard(saved_repeat_projection):
    freeze,out,guard=saved_repeat_projection
    assert type(guard) is independent_review.Guard and guard.remaining()>0
    proofs=[transport.verify_closed_worker(out/'workers'/name,freeze,guard)
            for name in transport.lifecycle_inventory()]
    assert sum(p['raw']['requests'] for p in proofs)==18
    assert all(p['raw']['startup_requests']==2 and p['raw']['failed_requests']==0 for p in proofs)
    assert all(p['placement_requests']==9 and p['graph']['fixtures']==11 for p in proofs)
    assert all(p['raw']['exact_bytes_checked'] and p['logs']['streams']==3 for p in proofs)
    assert sum(len(list((out/'workers'/name/'inputs').glob('*.json')))
               for name in transport.lifecycle_inventory())==14
    assert c.read(PRIOR/'completion.json')['status']=='incomplete'


@pytest.mark.parametrize('mutation',['repeated_suite','wrong_summary','wrong_clock'])
def test_saved_closed_admission_corruption_is_rejected(saved_repeat_projection,mutation):
    freeze,out,guard=saved_repeat_projection
    folder=out/'workers/qualification_repeat_research'
    if mutation=='repeated_suite':
        c.save(folder/'startup_inputs/request-00000003.json',dict(operation='entrypoint_checks'))
    else:
        path=out/'qualification_summary.json' if mutation=='wrong_summary' else folder/'request-00000002.json'
        value=c.read(path)
        if mutation=='wrong_summary':value['scientific_decisions']=104
        else:value['elapsed_seconds']+=.001
        path.write_text(c.canonical(value),encoding='utf-8')
    with pytest.raises(ValueError):transport.verify_closed_worker(folder,freeze,guard)
