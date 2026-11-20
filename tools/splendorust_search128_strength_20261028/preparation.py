"""Finite preservation and a single immutable qualification-to-acceptance freeze."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
from . import common as c, design, schedule
from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS, nested_artifacts
from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT

PRIOR = c.ROOT / 'outputs/splendorust_pool6_strengthfirst_20261028_assessment_01'
PRIOR_SHA = '1e5a91af8d73f5969baa3bdf24d640b1cf00cf27bdb500f7aa22a2c052fc8e05'
RECOVERY = c.ROOT / 'outputs/splendorust_pool6_strengthfirst_20261028_review_recovery_01'
RECOVERY_REVIEW_SHA = 'f4883e5640ca39ba94638afc21fad2c705f6ce5f30a4cc80f075c9b2cdd4697a'
NATIVE_SHA = 'b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c'


def preserve_previous():
    """Keep original inputs and exact accepted receipts; do not recursively reopen unrelated games."""
    c.require(c.file_hash(PRIOR/'freeze.json') == PRIOR_SHA, 'Historical main freeze changed')
    c.require(c.file_hash(RECOVERY/'independent_completion_review.json') == RECOVERY_REVIEW_SHA,
              'Historical separate recovery acceptance changed')
    old = c.read(PRIOR/'freeze.json')
    review = c.read(RECOVERY/'independent_completion_review.json')
    c.require(review['accepted'] is True and review['freeze_sha256'] == PRIOR_SHA and
              review['original_reader_failure_preserved'] is True, 'Historical recovered acceptance identity')
    pins = {x['path']: x for x in old['inputs']}
    def add(path):
        item = c.artifact(path)
        c.require(item['path'] not in pins or pins[item['path']] == item, 'Historical pin conflict')
        pins[item['path']] = item
    for path in (PRIOR/'freeze.json', PRIOR/'completion.json', PRIOR/'analysis.json',
                 PRIOR/'postrun/independent_review_os_exit.json', PRIOR/'postrun/independent_review_timeout_diagnostic.json',
                 RECOVERY/'independent_completion_review.json', RECOVERY/'independent_review_os_exit.json',
                 RECOVERY/'independent_acceptance_os_exit.json', RECOVERY/'recovery_owner_completion.json',
                 RECOVERY/'root_owner_exit_observation.json'):
        add(path)
    statuses = [
        dict(path=(PRIOR/'completion.json').relative_to(c.ROOT).as_posix(), field='status', expected='completed_verified'),
        dict(path=(PRIOR/'postrun/independent_review_os_exit.json').relative_to(c.ROOT).as_posix(), field='exit_code', expected=1),
        dict(path=(RECOVERY/'independent_completion_review.json').relative_to(c.ROOT).as_posix(), field='accepted', expected=True),
        dict(path=(RECOVERY/'recovery_owner_completion.json').relative_to(c.ROOT).as_posix(), field='status', expected='passed')]
    absent = [(PRIOR/'postrun/independent_completion_review.json').relative_to(c.ROOT).as_posix()]
    for row in statuses:
        c.require(c.read(c.ROOT/row['path'])[row['field']] == row['expected'], 'Historical status changed')
    for name in absent:
        c.require(not(c.ROOT/name).exists(), 'Historical failed attempt cannot be relabelled')
    c.verify(list(pins.values()))
    manifest = dict(scope='finite_referenced_dependencies', artifacts=sorted(pins.values(), key=lambda x:x['path']),
        historical_statuses=statuses, required_absent=absent, earlier_failures_preserved=True,
        scope_note='Exact inherited dependency pins and accepted status anchors; no new audit of unrelated historical game trees')
    design.fixed(c.OUT/'prelaunch/prior_preservation.json', manifest)
    return old, manifest


def prepare_profiles():
    from . import adapters
    for q in (16,64,128):
        design.fixed(adapters.profile_path(q), adapters.profile(q))
    baseline = {str(q):c.artifact(adapters.profile_path(q)) for q in (16,64)}
    research = {str(q):c.artifact(adapters.profile_path(q)) for q in (16,128)}
    return research, baseline


def seed_inventory():
    reserved = {int(x['seed_decimal']) for x in c.read(design.PANEL)['positions']}
    operations = {int(x['seed_decimal']) for x in schedule.stage_schedule('operational')}
    main = {int(x['seed_decimal']) for x in schedule.stage_schedule('main')}
    targets = reserved | operations | main
    c.require(len(reserved)==18 and len(operations)==3 and len(main)==200 and len(targets)==221,
              'Reserved and game setups must be disjoint')
    found = subprocess.run(['rg','--files','--no-ignore','outputs','-g','*manifest*.json',
        '-g','*schedule*.json','-g','*seed_registry*.json','-g','*reserved_panel*.json'],
        cwd=c.ROOT,capture_output=True,text=True,check=True,timeout=120)
    checked, skipped, collisions = [], [], []
    def visit(value, name):
        if type(value) is int or isinstance(value,str) and value.isdecimal():
            if int(value) in targets: collisions.append(name)
        elif isinstance(value,list):
            for item in value: visit(item,name)
        elif isinstance(value,dict):
            for item in value.values(): visit(item,name)
    for name in sorted(found.stdout.splitlines()):
        path = (c.ROOT/name).resolve()
        if path.is_relative_to(c.OUT.resolve()): continue
        if path.stat().st_size > 8*c.MiB:
            skipped.append(c.artifact(path)); continue
        visit(c.read(path), path.relative_to(c.ROOT).as_posix())
        checked.append(c.artifact(path))
    c.require(not collisions, 'Fresh seed collision; no replacement')
    return dict(status='passed', checked=checked, skipped_oversized=skipped, collisions=collisions,
        direct_new_namespace_disjointness=dict(reserved=18,operational=3,assessment=200,passed=True),
        scope='Registered manifests/schedules/seed registries/reserved panels up to8MiB; not arbitrary raw logs')


def contract():
    return dict(phase='strength', job_seconds=c.JOB_SECONDS,
        collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS, stage_seconds=c.STAGES,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES, final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES, max_rss_bytes=c.MAX_RSS_BYTES,
        minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,
        startup_minimum_available_ram_bytes=c.STARTUP_MINIMUM_AVAILABLE_RAM_BYTES,
        initial_ram_stability_seconds=60, maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES, minimum_free_disk_bytes=5*c.GiB,
        seconds_per_move=10., internal_seconds=9., research_seconds_per_move=10., research_internal_seconds=9.,
        research_log_max_bytes=128*c.MiB, frozen_log_max_bytes=128*c.MiB,
        research_response_max_bytes=4*c.MiB, frozen_response_max_bytes=4*c.MiB,
        request_max_bytes=c.MiB, placement_mode='isolated', max_workers=c.MAX_WORKERS,
        max_graph_builds=c.MAX_GRAPH_BUILDS, qualification_decisions=203,
        qualification_ordinary_requests=84, operational_games=6, main_games=400,
        maximum_reproductions=48, automatic_retries=0, failed_decision_retries=0, training=False,
        deployment_target_seconds=1., timing_thresholds_do_not_gate_admission=True,
        late_completed_moves_accepted=True, one_nonresettable_clock=True,
        independent_review_seconds=3600, independent_review_maximum_seconds=3600, forecast_contingency_factor=1.5,
        main_hard_guard_failures='administrative_forfeit_loss_continue_fixed_schedule',
        scientific_infrastructure_resource_failures='stop_incomplete')


def freeze():
    from . import qualification, runner
    c.require(not(c.OUT/'freeze.json').exists(), 'Never replace execution freeze')
    design.prepare()
    research, baseline = prepare_profiles()
    source_guard = verify_runtime_sources()
    old, preservation = preserve_previous()
    c.require(shutil.disk_usage(c.ROOT).free >= c.INITIAL_FREE_DISK_BYTES, 'Initial disk reserve')
    test = c.read(c.OUT/'prelaunch/model_free_tests.json')
    c.require(test['passed'] is True and test['exit_code']==0, 'Model-free tests not passed')
    review = c.read(c.OUT/'prelaunch/implementation_review.json')
    c.require(review['approved'] is True, 'Independent implementation review not passed')
    c.verify(review['inputs'])
    qplan = qualification.build_plan()
    c.require(qplan['complete_decisions']==203 and len(qplan['parity_requests'])==84,
              'Frozen finite qualification inventory')
    qitem = design.fixed(c.OUT/'qualification_plan.json', qplan)
    design.fixed(c.OUT/'runtime_sources.json', source_guard)
    design.fixed(c.OUT/'seed_inventory.json', seed_inventory())
    pins = {}
    def add(item):
        c.require(c.artifact(c.ROOT/item['path'])==item, 'Changed frozen input: '+item['path'])
        c.require(item['path'] not in pins or pins[item['path']]==item, 'Conflicting frozen identity')
        pins[item['path']] = item
    for item in preservation['artifacts'] + review['inputs']: add(item)
    for folder in (c.ROOT/'tools', c.ROOT/'src'):
        for path in sorted(folder.rglob('*.py')):
            if '__pycache__' not in path.parts: add(c.artifact(path))
    for path in sorted((c.ROOT/'tests').glob('test_splendorust_search128*20261028*.py')): add(c.artifact(path))
    add(c.artifact(c.ROOT/'tests/test_worker128.py'))
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file(): add(c.artifact(path))
    for item in list(research.values())+list(baseline.values())+list(nested_artifacts(qplan)): add(item)
    for path in (c.PROTOCOL,c.PREPARED_PROTOCOL,c.ROOT/'docs/strategy/splendorust_roadmap_update_20261028.md',
                 c.ROOT/'docs/engineering/future_training_rules_20261021.md'): add(c.artifact(path))
    frozen = {key:deepcopy(old[key]) for key in RUNTIME_KEYS}
    frozen.update(contract())
    native = [c.artifact(c.ROOT/c.original.profile_for(q)['native_binary']) for q in (16,64)]
    c.require(all(x['sha256']==NATIVE_SHA for x in native), 'Original native bridge identity')
    build = c.artifact(c.ROOT/'outputs/external_opponent_search_20261018_02/native_build_v2/build_receipt_v2.json')
    for item in native+[build]: add(item)
    frozen.update(native_artifacts=native,research_native_artifacts=native,
        research_native_roles={item['path']:role for item,role in zip(native,('reference','search'))},
        research_native_sha256=NATIVE_SHA,build_receipt=build)
    frozen.update(schema='splendorust_search128_strength_execution_v1',time=c.now(),attempt=1,
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        protocol=c.artifact(c.PROTOCOL),study_plan=c.artifact(c.OUT/'study_plan.json'),
        schedule=c.artifact(c.OUT/'schedule.json'),qualification_plan=qitem,reserved_panel=c.artifact(design.PANEL),
        research_profiles=research,baseline_profiles=baseline,
        prior_preservation=c.artifact(c.OUT/'prelaunch/prior_preservation.json'),
        candidate_world_pool=3,opponent_world_pool=3,candidate_simulations=128,opponent_simulations=64,
        seed_label='A',external_candidate_testing='paused',larger_data_deferred=True,
        native_recompiled=False,native_core_unchanged=True,weights_unchanged=True,
        numerical_policy_change='candidate quota128 instead of64 only; baseline unchanged',
        execution_deadlines_changed_prospectively=True,original_attempts_unchanged=True,
        same_evidence_path_both_arms=True,inputs=sorted(pins.values(),key=lambda x:x['path']))
    runner.validate_freeze_contract(frozen)
    c.verify(frozen['inputs'])
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',frozen),inputs=len(pins))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','preserve','freeze'])
    args=parser.parse_args()
    if args.action=='prepare':
        result=design.prepare();research,baseline=prepare_profiles()
        print(c.canonical(dict(result,research_profiles=research,baseline_profiles=baseline)))
    elif args.action=='preserve':
        _,result=preserve_previous();print(c.canonical(dict(status='preserved',inputs=len(result['artifacts']))))
    else: print(c.canonical(freeze()))
