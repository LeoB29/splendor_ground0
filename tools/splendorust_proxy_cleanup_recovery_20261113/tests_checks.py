"""Model-free regression against the retained real premature-cleanup failure."""
import ast
from copy import deepcopy
from pathlib import Path
import pytest

from . import checks as x


def saved_bundle():
    return {k:x.c.read(x.c.resolve(v)) for k,v in x.PINS.items() if k!='freeze'}


def saved_plan():
    return dict(original_freeze=deepcopy(x.PINS['freeze']),
        original_limits=saved_bundle()['root_owner']['limits'],cleanup_exception=x.exception_contract())


def test_real_failure_is_preserved_and_later_original_cleanup_is_complete():
    x.c.verify(x.exception_inputs())
    bundle=saved_bundle()
    assert bundle['premature_exit']['status']=='failed'
    assert bundle['premature_exit']['exact_owned_absence'] is False
    proof=x.verify_cleanup_bundle(bundle,original_limits=bundle['root_owner']['limits'])
    assert proof['status']=='saved_later_outer_cleanup_verified'
    assert proof['cleanup_interval_microseconds']==233332
    assert len(proof['exact_owned_identities'])==8
    assert proof['new_decisions']==proof['replacements']==0
    assert proof['exact_reproduction_comparisons_still_required'] is True


def change(bundle,path,value):
    for key in path[:-1]:bundle=bundle[key]
    bundle[path[-1]]=value


@pytest.mark.parametrize('path,value',[
    (('premature_exit','exit_code'),1),
    (('premature_exit','exit_code'),False),
    (('premature_exit','error'),'real service failure'),
    (('premature_exit','status'),'passed'),
    (('premature_exit','exact_owned_absence'),True),
    (('service_exit','status'),'failed'),
    (('service_exit','error'),'actual native close failure'),
    (('service_exit','guarded_decisions'),31),
    (('outer_exit','exit_code'),1),
    (('outer_exit','exit_code'),False),
    (('outer_exit','status'),'failed'),
    (('outer_exit','exact_owned_absence'),False),
    (('outer_exit','actual_exit_observed'),False),
    (('outer_exit','error'),'cleanup error'),
    (('outer_exit','cleanup_errors'),['owned child remains']),
    (('outer_exit','role'),'diagnostic'),
    (('outer_exit','freeze_sha256'),'0'*64),
    (('outer_exit','result_artifact','sha256'),'0'*64),
    (('worker_result','status'),'failed'),
    (('worker_result','result','requests'),31),
    (('worker_result','result','service_actual_exit','sha256'),'0'*64),
    (('worker_result','result','native_owners','sha256'),'0'*64),
    (('worker_owner','windows_job_assigned'),False),
    (('worker_owner','role'),'diagnostic'),
    (('worker_owner','parent','created'),1),
    (('root_owner','freeze_sha256'),'0'*64),
    (('outer_exit','time'),'2026-11-13T16:22:23.400000+00:00'),
    (('outer_exit','elapsed_seconds'),7201),
])
def test_every_real_failure_mutation_remains_rejected(path,value):
    bundle=saved_bundle();limits=deepcopy(bundle['root_owner']['limits'])
    change(bundle,path,value)
    with pytest.raises(ValueError):x.verify_cleanup_bundle(bundle,original_limits=limits)


@pytest.mark.parametrize('mutation',['missing','wrong_birth','duplicate','wrong_worker'])
def test_outer_cleanup_requires_every_original_exact_identity(mutation):
    bundle=saved_bundle();rows=bundle['outer_exit']['observed_owned_identities']
    if mutation=='missing':rows.pop()
    elif mutation=='wrong_birth':rows[-1]['created']+=1
    elif mutation=='duplicate':rows.append(deepcopy(rows[-1]))
    else:
        # A complete service list without this original worker is insufficient.
        rows[:]=[r for r in rows if r!=bundle['worker_owner']['identity']]
    with pytest.raises(ValueError):x.verify_cleanup_bundle(bundle,original_limits=bundle['root_owner']['limits'])


def test_original_stage_deadline_is_not_extended_by_later_cleanup():
    bundle=saved_bundle();bundle['worker_owner']['limits']['unix_deadline']=1794586943.5
    with pytest.raises(ValueError):x.verify_cleanup_bundle(bundle,original_limits=bundle['root_owner']['limits'])


def test_current_absence_does_not_replace_original_outer_evidence(monkeypatch):
    monkeypatch.setattr(x.c,'exact_alive',lambda item:False)
    bundle=saved_bundle();bundle['outer_exit']['exact_owned_absence']=False
    with pytest.raises(ValueError):x.verify_cleanup_bundle(bundle,original_limits=bundle['root_owner']['limits'])


def test_contract_and_artifacts_are_defensive_copies():
    contract=x.exception_contract();contract['original_inputs']['premature_exit']['sha256']='0'*64
    inputs=x.exception_inputs();inputs[0]['sha256']='0'*64
    assert x.exception_contract()['original_inputs']['premature_exit']==x.PINS['premature_exit']
    assert all(row['sha256']!='0'*64 for row in x.exception_inputs())


def test_original_clock_check_is_mandatory_before_service_validation(monkeypatch):
    called=[]
    def expired(limits):
        called.append(deepcopy(limits));raise ValueError('expired original clock')
    monkeypatch.setattr(x.c,'check_deadlines',expired)
    plan=saved_plan()
    with pytest.raises(ValueError,match='expired original clock'):
        x._load_exception(plan,x.c.read(x.c.PLAN),lambda:None)
    assert called==[plan['original_limits']]


@pytest.mark.parametrize('field',['original_freeze','original_limits','cleanup_exception'])
def test_altered_recovery_contract_or_clocks_are_rejected(field):
    plan=saved_plan()
    if field=='original_freeze':plan[field]['sha256']='0'*64
    elif field=='original_limits':plan[field]['unix_deadline']+=3600
    else:plan[field]['replacements']=1
    with pytest.raises(ValueError):x._load_exception(plan,x.c.read(x.c.PLAN),lambda:None)


def test_other_service_roles_remain_strict_original(monkeypatch):
    calls=[]
    def strict(role,plan,boundary):
        calls.append(role);raise ValueError('original refusal')
    monkeypatch.setattr(x.original,'check_service',strict)
    for role in ('qualification','diagnostic'):
        with pytest.raises(ValueError,match='original refusal'):
            x.check_service(role,{},lambda:None,{})
    assert calls==['qualification','diagnostic']


def test_complete_original_service_scientific_checks_are_preserved():
    """An AST comparison restricts the deliberate source delta to final exit proof."""
    def body(path):
        tree=ast.parse(Path(path).read_text(encoding='utf8'))
        return next(n.body for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_service')
    old=body(x.original.__file__);new=body(x.__file__)
    # Ignore only imports and our new prefix; start at the original folder binding.
    def beginning(nodes):
        return next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='folder' for t in n.targets))
    old=old[beginning(old):];new=new[beginning(new):]
    class OriginalAlias(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='original' and node.attr=='passed_job':
                return ast.Name(id='passed_job',ctx=ast.Load())
            return self.generic_visit(node)
    def preserved(nodes,is_new):
        result=[]
        for node in nodes:
            text=ast.unparse(node)
            if isinstance(node,ast.Return):continue
            if not is_new and 'Observed graceful service exit0' in text:continue
            if is_new and ('Exact authenticated cleanup bundle consumed by original service checker' in text or
                           isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cleanup' for t in node.targets)):
                continue
            result.append(ast.dump(OriginalAlias().visit(node),include_attributes=False))
        return result
    assert preserved(old,False)==preserved(new,True)
