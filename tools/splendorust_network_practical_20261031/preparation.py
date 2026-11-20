"""Bounded saved-input preparation and one exclusive practical-study freeze."""
from pathlib import Path
import argparse
from . import common as c
from tools.splendorust_search256_efficiency_20261031.preparation import INHERITED


TOLERANCES = dict(policy_atol=1e-5, policy_rtol=1e-6,
                  value_atol=1e-6, value_rtol=0.0)


def accepted_predecessor():
    path = c.PREVIOUS_OUT / 'freeze.json'
    c.require(c.file_hash(path) == c.PREVIOUS_FREEZE_SHA, 'Exact accepted efficiency predecessor')
    c.require(c.file_hash(c.PREVIOUS_ACCEPTANCE) == c.PREVIOUS_ACCEPTANCE_SHA,
              'Exact independent predecessor acceptance')
    accepted = c.read(c.PREVIOUS_ACCEPTANCE)
    c.require(accepted['approved'] is True and accepted['all_exact_identities_absent'] is True and
              accepted['coverage']['status'] == 'verified_full_diagnostic',
              'Predecessor fully accepted and closed')
    old = c.read(path)
    c.verify([old['panel'], old['qualification_plan'], old['original256_suite_reference']])
    return old


def preserved_rejection():
    c.require(c.file_hash(c.REJECTED_CLOSEOUT) == c.REJECTED_CLOSEOUT_SHA,
              'Preserved verified exactness rejection')
    closed = c.read(c.REJECTED_CLOSEOUT)
    c.require(closed['candidate_accepted'] is False and closed['completed_timing_study'] is False and
              closed['original_experiment_status'] == 'incomplete', 'Never relabel failed exactness attempt')
    c.verify([closed['freeze'], closed['candidate_supplement']] + closed['reviews'])
    c.require(c.file_hash(c.CANDIDATE_SOURCE) == c.CANDIDATE_SOURCE_SHA, 'The same held implementation')
    return closed


def _same_or_save(path, value):
    if path.exists():
        c.require(c.read(path) == value, 'Never replace preparation artifact: ' + str(path))
        return c.artifact(path)
    return c.save(path, value)


def verify_extraction():
    """Authenticate the already completed, one saved-only extraction honestly."""
    from . import panel, analysis
    path=c.OUT/'prelaunch/panel_extraction_01.json'; proof=c.read(path)
    actual=c.read(c.OUT/'prelaunch/panel_extraction_actual_exit_01.json')
    c.require(proof['status']=='passed' and actual['actual_exit_code']==0 and
              actual['extraction_receipt']==c.artifact(path) and
              actual['models']==actual['games']==proof['models']==proof['games']==0,
              'Actual successful saved-only extraction exit')
    c.require(proof['new_timing_roots']==proof['numerical_only_roots']==18 and
              proof['distinct_source_games']==36 and proof['numerical_fixtures']==38 and
              proof['eligible_original_decisions']==11066 and
              0 < proof['elapsed_seconds'] < proof['bound_seconds']==600.,
              'Bounded complete saved assessment extraction')
    c.require(actual['saved_only'] is True and actual['status']=='observed_actual_exit' and
              actual['actual_elapsed_seconds']==proof['elapsed_seconds'] and
              actual['real_guard']==dict(cooperative_boundary_seconds=600,external_timeout=False,
                                        peak_rss_measured=False,rss_guard=False),
              'Honest extraction scope: cooperative600s, no retrospective RSS or external-guard claim')
    c.verify(proof['inputs']+[actual['invoked_script']])
    precommit=actual['selection_precommit']
    c.require(precommit['actual_exit_code']==0 and
              precommit['artifact']==c.artifact(c.OUT/'selection_rule.json'),
              'Selection rule saved before the only extraction')
    c.verify([precommit['artifact']])
    expected_sources=[c.artifact(c.OUT/'selection_rule.json'),
                      c.artifact(Path(panel.__file__)),c.artifact(Path(analysis.__file__))]
    c.require(proof['inputs']==expected_sources,'Exact held extraction source and selection rule')
    for key in ('panel','numerical_plan','evaluation_matrix'):
        c.require(proof['artifacts'][key]==c.artifact(c.OUT/(key+'.json')),
                  'Exact extracted '+key)
    selected=c.read(c.OUT/'panel.json');panel.panel_roots(selected)
    c.require(c.read(c.OUT/'selection_rule.json')==panel.selection_rule() and
              c.read(c.OUT/'evaluation_matrix.json')==analysis.evaluation_matrix(selected),
              'Unchanged selection and prospective evaluation order')
    c.verify(selected['inputs'])
    return selected


def prepare_plan():
    old = accepted_predecessor()
    previous = c.read(c.ROOT/old['qualification_plan']['path'])
    plan = dict(schema='network_practical_actual_qualification_v1',
        actual_suites=previous['actual_suites'], accepted_suite=previous['accepted_suite'],
        accepted_plan=previous['accepted_plan'], worker_variants=['control','candidate'],
        decisions_per_worker=67, graph_cases_per_worker=11, fresh_numerical_cases_per_worker=38,
        maximum_scientific_decisions=134, maximum_graph_cases=22, maximum_fresh_numerical_cases=76,
        corrected_actual_entrypoint_required=True, control_historical_exact=True,
        candidate_historical_equality_required=False, candidate_within_worker_checks_unchanged=True,
        no_retries=True)
    c.require(plan['accepted_suite'] == old['original256_suite_reference'], 'Accepted original256 suite')
    c.verify([plan['accepted_suite'], plan['accepted_plan']])
    return _same_or_save(c.OUT/'qualification_plan.json', plan)


def _passed_tests(path):
    value = c.read(path)
    c.require(value['exit_code'] == 0 and value['models'] == value['games'] == 0,
              'Passed actual model-free tests required')
    c.require(bool(value['inputs']), 'Tests bind current source and tests')
    c.verify(value['inputs']+[value['log']]); return value


def _approved_review(path):
    value = c.read(path)
    c.require(value['approved'] is True and not value.get('blocking_findings', []),
              'Independent implementation review must approve current code')
    c.require(bool(value['inputs']), 'Review binds inspected sources')
    c.verify(value['inputs']); return value


def freeze():
    from . import control
    c.require(not (c.OUT/'freeze.json').exists(), 'One exclusive practical freeze')
    old = accepted_predecessor(); rejected = preserved_rejection(); selected = verify_extraction()
    plan = prepare_plan()
    test_path = c.OUT/'prelaunch/model_free_tests.json'; _passed_tests(test_path)
    reviews = [c.OUT/'prelaunch'/name for name in ('independent_review_worker.json',
        'independent_review_controller.json', 'independent_review_design.json')]
    for path in reviews: _approved_review(path)
    value = {name: old[name] for name in INHERITED}
    value.update(control.contract(), time=c.now(), study_kind='practical_network_numerical_decision_speed_followup',
        protocol=c.artifact(c.PROTOCOL), accepted_predecessor=c.artifact(c.PREVIOUS_OUT/'freeze.json'),
        accepted_predecessor_receipt=c.artifact(c.PREVIOUS_ACCEPTANCE),
        preserved_exactness_rejection=c.artifact(c.REJECTED_CLOSEOUT),
        original256_suite_reference=old['original256_suite_reference'],
        panel=c.artifact(c.OUT/'panel.json'), numerical_plan=c.artifact(c.OUT/'numerical_plan.json'),
        evaluation_matrix=c.artifact(c.OUT/'evaluation_matrix.json'), qualification_plan=plan,
        selection_rule=c.artifact(c.OUT/'selection_rule.json'), numerical_tolerances=dict(TOLERANCES),
        candidate_module=c.artifact(c.CANDIDATE_SOURCE), candidate_import_module=c.CANDIDATE_MODULE,
        candidate_function='build_candidate', candidate_id=c.CANDIDATE_ID,
        model_free_tests=c.artifact(test_path), implementation_reviews=[c.artifact(p) for p in reviews],
        original_failed_attempts_preserved=True, new_games=0, training=False,
        decision_disagreement_stops_timing=False, reference_promoted=False)
    pins = {}
    def add(item):
        c.require(pins.setdefault(item['path'], item) == item, 'Conflicting source pin')
    for item in old['inputs']+selected['inputs']: add(item)
    for key in ('freeze','candidate_supplement'):
        add(rejected[key]); inherited=c.read(c.ROOT/rejected[key]['path'])
        for item in inherited['inputs']: add(item)
    for item in rejected['reviews']: add(item)
    for key in ('accepted_predecessor','accepted_predecessor_receipt','preserved_exactness_rejection'):
        add(value[key])
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((c.ROOT/'tests').glob('test_splendorust_network_practical*_20261031.py'))
    paths += [p for p in c.OUT.rglob('*') if p.is_file()]
    paths += [c.PROTOCOL,c.CANDIDATE_SOURCE]
    for path in paths: add(c.artifact(path))
    value['inputs'] = sorted(pins.values(),key=lambda x:x['path'])
    c.verify(value['inputs']); control.validate_freeze_contract(value)
    return dict(status='frozen_not_launched', freeze=c.save(c.OUT/'freeze.json',value),
                source_inputs=len(pins), models=0, games=0)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare-plan','freeze'])
    action=parser.parse_args().action
    result={'prepare-plan':prepare_plan,'freeze':freeze}[action]()
    print(c.canonical(result),flush=True)


if __name__=='__main__': main()
