"""Saved-only preparation and exclusive two-stage freezes; never loads a model."""
from pathlib import Path
import argparse
from . import common as c
from tools.splendorust_search256_efficiency_20261031.preparation import INHERITED


def accepted_predecessor():
    path = c.PREVIOUS_OUT / 'freeze.json'
    c.require(c.file_hash(path) == c.PREVIOUS_FREEZE_SHA,
              'Exact accepted efficiency predecessor')
    c.require(c.file_hash(c.PREVIOUS_ACCEPTANCE) == c.PREVIOUS_ACCEPTANCE_SHA,
              'Exact independent predecessor acceptance')
    acceptance = c.read(c.PREVIOUS_ACCEPTANCE)
    c.require(acceptance['approved'] is True and
              acceptance['all_exact_identities_absent'] is True and
              acceptance['coverage']['status'] == 'verified_full_diagnostic',
              'Predecessor is fully accepted and closed')
    old = c.read(path)
    c.require(old['panel']['sha256'] ==
              '5a41ce0a8863f42c1550ae849a148f5a1659bf114476ff2debabeaae4d9caada',
              'Exact accepted eighteen-position panel')
    c.verify([old['panel'], old['qualification_plan'], old['original256_suite_reference']])
    return old


def _same_or_save(path, value):
    if path.exists():
        c.require(c.read(path) == value, 'Never replace preparation artifact: ' + str(path))
        return c.artifact(path)
    return c.save(path, value)


def prepare_inputs():
    from . import analysis, profile_worker
    old = accepted_predecessor()
    panel = c.read(c.ROOT / old['panel']['path'])
    previous_plan = c.read(c.ROOT / old['qualification_plan']['path'])
    plan = dict(schema='network_execution_actual_qualification_v1',
        actual_suites=previous_plan['actual_suites'],
        accepted_suite=previous_plan['accepted_suite'],
        accepted_plan=previous_plan['accepted_plan'],
        worker_variants=['localization', 'control', 'candidate'],
        decisions_per_worker=67, graph_cases_per_worker=11,
        maximum_scientific_decisions=201, maximum_graph_cases=33,
        corrected_actual_entrypoint_required=True, no_retries=True)
    c.require(plan['accepted_suite'] == old['original256_suite_reference'],
              'Qualified suite remains bound to accepted original evidence')
    c.verify([plan['accepted_suite'], plan['accepted_plan']])
    return dict(
        panel=old['panel'],
        qualification_plan=_same_or_save(c.OUT / 'qualification_plan.json', plan),
        profile_plan=_same_or_save(c.OUT / 'profile_plan.json', profile_worker.profiling_plan(old)),
        evaluation_matrix=_same_or_save(c.OUT / 'evaluation_matrix.json',
                                        analysis.evaluation_matrix(panel)))


def _passed_tests(path):
    value = c.read(path)
    c.require(value['exit_code'] == 0 and value['models'] == value['games'] == 0,
              'Passed actual model-free tests required')
    c.require(bool(value['inputs']), 'Tests bind current source and test files')
    c.verify(value['inputs'] + [value['log']])
    return value


def _approved_review(path):
    value = c.read(path)
    c.require(value['approved'] is True and not value.get('blocking_findings', []),
              'Independent implementation review must approve current code')
    c.require(bool(value['inputs']), 'Review binds inspected sources')
    c.verify(value['inputs'])
    return value


def freeze():
    from . import control
    c.require(not (c.OUT / 'freeze.json').exists(), 'One exclusive initial freeze')
    old = accepted_predecessor()
    inputs = prepare_inputs()
    test_path = c.OUT / 'prelaunch/model_free_tests.json'
    _passed_tests(test_path)
    review_paths = [c.OUT / 'prelaunch' / name for name in (
        'independent_review_worker.json', 'independent_review_controller.json',
        'independent_review_design.json')]
    for path in review_paths:
        _approved_review(path)
    value = {name: old[name] for name in INHERITED}
    value.update(control.contract(), **inputs, time=c.now(),
        study_kind='one_exact_output_network_execution_attempt',
        protocol=c.artifact(c.PROTOCOL),
        accepted_predecessor=c.artifact(c.PREVIOUS_OUT / 'freeze.json'),
        accepted_predecessor_receipt=c.artifact(c.PREVIOUS_ACCEPTANCE),
        original256_suite_reference=old['original256_suite_reference'],
        original_failed_attempts_preserved=True,
        model_free_tests=c.artifact(test_path),
        implementation_reviews=[c.artifact(path) for path in review_paths],
        candidate_implementation_selected=False,
        candidate_supplement_required=True,
        raw_forward_candidate_sweeps=0)
    pins = {}
    def add(item):
        c.require(pins.setdefault(item['path'], item) == item, 'Conflicting source pin')
    for item in old['inputs']:
        add(item)
    for item in (value['accepted_predecessor'], value['accepted_predecessor_receipt']):
        add(item)
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((c.ROOT / 'tests').glob('test_splendorust_network_execution*_20261031.py'))
    paths += [p for p in c.OUT.rglob('*') if p.is_file()]
    paths += [c.PROTOCOL]
    for path in paths:
        add(c.artifact(path))
    value['inputs'] = sorted(pins.values(), key=lambda x: x['path'])
    c.verify(value['inputs'])
    control.validate_freeze_contract(value)
    return dict(status='frozen_not_launched',
        freeze=c.save(c.OUT / 'freeze.json', value), source_inputs=len(pins),
        new_models=0, new_games=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'freeze'])
    args = parser.parse_args()
    print(c.canonical(prepare_inputs() if args.action == 'prepare' else freeze()), flush=True)


if __name__ == '__main__':
    main()
