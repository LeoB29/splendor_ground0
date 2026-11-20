"""Freeze the authorized corrected attempt using the exact saved assessment."""
import argparse
from copy import deepcopy
from pathlib import Path
from . import common as c

TOLERANCES=dict(policy_atol=1e-5,policy_rtol=1e-6,value_atol=1e-6,value_rtol=0.0)
REUSED_NAMES=('selection_rule.json','panel.json','numerical_plan.json','evaluation_matrix.json','qualification_plan.json')


def _same_or_save(path,value):
    if path.exists():
        c.require(c.read(path)==value,'Never replace preparation artifact: '+str(path))
        return c.artifact(path)
    return c.save(path,value)


def _passed_tests(path):
    value=c.read(path)
    c.require(value['exit_code']==0 and value['models']==value['games']==0,
              'Passed actual model-free tests required')
    c.require(bool(value['inputs']),'Tests bind current source and tests')
    c.verify(value['inputs']+[value['log']]);return value


def _approved_review(path):
    value=c.read(path)
    c.require(value['approved'] is True and not value.get('blocking_findings',[]),
              'Independent implementation review must approve current code')
    c.require(bool(value['inputs']),'Review binds inspected sources')
    c.verify(value['inputs']);return value


def preserved_attempt():
    c.require(c.file_hash(c.FAILED_OUT/'freeze.json')==c.FAILED_FREEZE_SHA and
        c.file_hash(c.FAILED_CLOSEOUT)==c.FAILED_CLOSEOUT_SHA,'Exact failed practical attempt and independent closeout')
    closed=c.read(c.FAILED_CLOSEOUT)
    c.require(closed['original_experiment_status']=='incomplete' and
        closed['actual_original_owner_exit_code']==1 and closed['qualification_accepted'] is True and
        closed['ordinary_timing_measurements']==0 and closed['completed_timing_study'] is False and
        closed['no_active_experiment_process'] is True,'Prior qualification is not a completed timing study')
    c.verify([closed['freeze'],closed['prepared_handoff_helper'],closed['results_report']]+closed['reviews'])
    inventory=c.read(c.FAILED_INVENTORY)
    c.require(inventory['root']==c.FAILED_OUT.relative_to(c.ROOT).as_posix() and
        inventory['preserved_in_place'] is True and inventory['file_count']==155,
        'Exact preserved failed evidence inventory')
    expected={x['path'] for x in inventory['files']}
    actual={p.relative_to(c.ROOT).as_posix() for p in c.FAILED_OUT.rglob('*') if p.is_file()}
    c.require(len(expected)==len(inventory['files'])==155 and actual==expected,
        'No replaced, discarded or appended failed-attempt evidence')
    c.verify(inventory['files'])
    c.require(c.read(c.FAILED_OUT/'completion.json')['status']=='incomplete',
        'Never relabel the prior failed completion')
    old=c.read(c.FAILED_OUT/'freeze.json');c.verify(old['inputs'])
    c.require(c.file_hash(c.HANDOFF_HELPER)==c.HANDOFF_HELPER_SHA and
        c.file_hash(c.HANDOFF_REVIEW)==c.HANDOFF_REVIEW_SHA,'Exact separately tested handoff correction')
    _approved_review(c.HANDOFF_REVIEW)
    c.require(c.file_hash(c.CANDIDATE_SOURCE)==c.CANDIDATE_SOURCE_SHA,'Same sole neural execution candidate')
    return old,closed,inventory


def verify_reused_assessment(old):
    from . import panel,analysis
    for name in REUSED_NAMES:
        c.require(c.file_hash(c.OUT/name)==c.file_hash(c.FAILED_OUT/name),
            'Repeated assessment must retain exact saved input bytes: '+name)
    for field in ('panel','numerical_plan','evaluation_matrix','qualification_plan','selection_rule'):
        c.verify([old[field]])
        c.require(old[field]['sha256']==c.file_hash(c.OUT/(field+'.json')),
                  'Prior freeze input identity: '+field)
    selected=c.read(c.OUT/'panel.json');panel.panel_roots(selected)
    c.require(c.read(c.OUT/'selection_rule.json')==panel.selection_rule() and
        c.read(c.OUT/'evaluation_matrix.json')==analysis.evaluation_matrix(selected),
        'Same root selection and complete counterbalanced request matrix')
    c.require(old['numerical_tolerances']==TOLERANCES,'No result-informed tolerance change')
    c.verify(selected['inputs']);return selected


def prepare_reuse():
    old,closed,inventory=preserved_attempt();verify_reused_assessment(old)
    value=dict(schema='practical_repeat_assessment_provenance_v1',
        prior_freeze=c.artifact(c.FAILED_OUT/'freeze.json'),prior_closeout=c.artifact(c.FAILED_CLOSEOUT),
        prior_inventory=c.artifact(c.FAILED_INVENTORY),
        original_inputs=[c.artifact(c.FAILED_OUT/name) for name in REUSED_NAMES],
        repeated_inputs=[c.artifact(c.OUT/name) for name in REUSED_NAMES],
        numerical_fixtures_previously_evaluated=True,decision_timing_panel_previously_run=False,
        new_source_selection=False,new_extraction=False,new_models=0,new_games=0,
        original_failed_attempt_preserved=True,authorization='User: ok do it after corrected-attempt recommendation')
    return _same_or_save(c.OUT/'assessment_reuse.json',value)


def freeze():
    from . import control
    c.require(not(c.OUT/'freeze.json').exists(),'One exclusive corrected-attempt freeze')
    old,closed,inventory=preserved_attempt();verify_reused_assessment(old)
    reuse=prepare_reuse()
    tests=c.OUT/'prelaunch/model_free_tests.json';_passed_tests(tests)
    reviews=[c.OUT/'prelaunch'/name for name in ('independent_review_worker.json',
        'independent_review_controller.json','independent_review_design.json')]
    for path in reviews:_approved_review(path)
    value=deepcopy(old)
    value.update(control.contract(),time=c.now(),study_kind='corrected_practical_network_repeat_attempt',
        protocol=c.artifact(c.PROTOCOL),attempt_number=2,separate_user_authorized_attempt=True,
        previous_practical_attempt=c.artifact(c.FAILED_OUT/'freeze.json'),
        previous_practical_closeout=c.artifact(c.FAILED_CLOSEOUT),
        previous_failed_inventory=c.artifact(c.FAILED_INVENTORY),assessment_reuse=reuse,
        handoff_helper=c.artifact(c.HANDOFF_HELPER),handoff_correction_review=c.artifact(c.HANDOFF_REVIEW),
        numerical_assessment='repeat of unchanged prospectively selected saved fixtures',
        timing_assessment='first execution of unchanged preselected18-root panel',
        original_failed_attempts_preserved=True,model_free_tests=c.artifact(tests),
        implementation_reviews=[c.artifact(p) for p in reviews],new_games=0,training=False,
        reference_promoted=False,decision_disagreement_stops_timing=False)
    for field in ('panel','numerical_plan','evaluation_matrix','qualification_plan','selection_rule'):
        value[field]=c.artifact(c.OUT/(field+'.json'))
    pins={}
    def add(item):
        c.require(pins.setdefault(item['path'],item)==item,'Conflicting source pin')
    for item in old['inputs']+inventory['files']+closed['reviews']:add(item)
    paths=list(Path(__file__).parent.glob('*.py'))
    paths+=list((c.ROOT/'tests').glob('test_splendorust_network_practical_v2*_20261031.py'))
    paths+=[p for p in c.OUT.rglob('*') if p.is_file()]
    paths+=[c.PROTOCOL,c.FAILED_CLOSEOUT,c.FAILED_INVENTORY,c.HANDOFF_HELPER,c.HANDOFF_REVIEW,
            c.ROOT/'tests/test_splendorust_network_practical_handoff_20261031.py']
    for path in paths:add(c.artifact(path))
    value['inputs']=sorted(pins.values(),key=lambda x:x['path'])
    c.verify(value['inputs']);control.validate_freeze_contract(value)
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',value),
        source_inputs=len(pins),models=0,games=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare-reuse','freeze'])
    args=parser.parse_args();result={'prepare-reuse':prepare_reuse,'freeze':freeze}[args.action]()
    print(c.canonical(result),flush=True)


if __name__=='__main__':main()
