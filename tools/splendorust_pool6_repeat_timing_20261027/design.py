"""Fixed saved roots and counterbalanced repeats; no model execution."""
from copy import deepcopy
from . import common as c

PREVIOUS = c.QUAL_OUT
PREVIOUS_FREEZE_SHA = '247b143d1f5256048b336231fb3ba3c544f74ceb7580e7087a1e69b967d480a0'
PREVIOUS_REVIEW_SHA = '01405c2631723accafda75a642a64d481ead73c31fdd1307e9fde80295198feb'
PANEL = c.OUT / 'saved_panel.json'
MATRIX = c.OUT / 'request_matrix.json'
ROUNDS = 12
TARGETS = ('reserved_06', 'gold_returns', 'reserved_00')


def fixed(path, value):
    if path.exists(): c.require(c.read(path) == value, 'Prepared artifact changed: ' + str(path))
    else: c.save(path, value)
    return c.artifact(path)


def build_panel():
    from tools.future_training_rules_20261021 import decode_training_snapshot
    from tools.external_strength_screen_20261018 import audit as ledger
    reserved = c.read(PREVIOUS / 'reserved_panel.json')['positions']
    # First saved ID within each pre-existing turn stratum. No latency/choice filter.
    controls = [min(p['id'] for p in reserved if p['target_ply'] in (base, base+1)
                    and p['id'] not in TARGETS) for base in (12, 36, 60)]
    c.require(controls == ['reserved_01', 'reserved_07', 'reserved_12'], 'Fixed control identity')
    old = c.read(PREVIOUS / 'qualification_plan.json')['parity_requests']
    positions = []
    for index, name in enumerate([*TARGETS, *controls]):
        source = next(x for x in old if x['case_id'] == name)
        observer, before = source['observer'], source['before_snapshot']
        decode_training_snapshot(before)
        ledger.check_observer(before, observer)
        expectations = {}
        for route in ('frozen', 'research'):
            matches = [x for x in old if x['case_id'] == name and x['route'] == route]
            if matches:
                c.require(len(matches) == 1, 'Exactly one saved source per arm/root')
                path = PREVIOUS / 'qualification_rows' / f"{matches[0]['index']:04d}.json"
                row = c.read(path)
                c.require(row['completed'] and row['clock']['status'] == 'ok' and
                          row['observer'] == observer, 'Saved ordinary expectation incomplete')
                expectations[route] = dict(expected_record=c.artifact(path))
            else:
                c.require(name == 'gold_returns' and route == 'frozen', 'Only gold frozen uses actual suite')
                path = PREVIOUS / 'entrypoint_checks/frozen.json'
                receipt = c.read(path)
                fixture = next(x for x in receipt['result']['fixtures'] if x['id'] == name)
                c.require(receipt['status'] == 'ok' and fixture['observer_sha256'] == c.digest(observer),
                          'Frozen gold suite expectation differs')
                expectations[route] = dict(expected_entrypoint=c.artifact(path), expected_case=name)
        positions.append(dict(position_index=index, case_id=name,
            selection='observed_slow_target' if name in TARGETS else 'fixed_structural_control',
            observer=observer, observer_sha256=c.digest(observer), before_snapshot=before,
            before_sha256=c.digest(before), expectations=expectations))
    return dict(schema='pool6_repeat_saved_panel_v1', positions=positions,
        targets=list(TARGETS), controls=controls,
        selection_rule='Targets from prior three700ms crossings; controls first saved ID per12/36/60 ply stratum excluding target IDs; no timing filter',
        source_panel=c.artifact(PREVIOUS/'reserved_panel.json'),
        source_plan=c.artifact(PREVIOUS/'qualification_plan.json'),
        new_games=0, new_trajectories=0)


def build_matrix(panel=None):
    panel = build_panel() if panel is None else panel
    requests = []
    for repeat in range(ROUNDS):
        order = [(repeat % 6 + offset) % 6 for offset in range(6)]
        if repeat >= 6: order.reverse()
        for index in order:
            pos = panel['positions'][index]
            arms = ('frozen', 'research') if (repeat + index) % 2 == 0 else ('research', 'frozen')
            for route in arms:
                requests.append(dict(index=len(requests), sequence=len(requests), round=repeat,
                    position_index=index, route=route, kind='repeat_timing', case_id=pos['case_id'],
                    quota=64, world_pool=3 if route=='frozen' else 6, seed_label='A', telemetry=False,
                    observer=deepcopy(pos['observer']), observer_sha256=pos['observer_sha256'],
                    before_snapshot=deepcopy(pos['before_snapshot']), **pos['expectations'][route]))
    c.require(len(requests)==144, 'Fixed144 ordinary requests')
    return dict(schema='pool6_repeat_request_matrix_v1', requests=requests, rounds=ROUNDS,
        position_order='six cyclic rounds, then reversed six cyclic rounds',
        arm_order='frozen first iff (round + position_index) is even',
        no_warmup_dropping=True, no_retries=True, no_replacements=True)


def build_plan():
    return dict(schema='pool6_repeat_bounded_plan_v1', phase=c.PHASE,
        purpose='Characterize recurrence of selected slow saved positions, not strength or admission',
        ordinary_requests=144, rounds=12, positions=6, controls=3, selected_slow_targets=3,
        startup_complete_decisions=104, graph_cases=22, parent_requests=148,
        scientific_decisions=248, resident_workers=2, worker_replacements=0,
        candidate=dict(route='research', world_pool=6, simulations=64, seed_label='A'),
        control=dict(route='frozen', world_pool=3, simulations=64, seed_label='A'),
        seconds_per_move=1., internal_seconds=.9, margin_seconds=.7,
        margin_role='descriptive only; prior negative admission remains negative',
        failure_policy='Stop on first timeout, mismatch, unexpected rejection or resource/integrity failure; preserve verified prefix; no retry',
        player_changes=False, observer_changes=False, payload_changes=False,
        inference_timing_instrumentation_changes=False, new_games=0, promotion=False,
        training=False, larger_data_deferred=True, external_testing='paused',
        scope_limits=['Selected failure inputs and small controls do not estimate live-game tail risk',
                      'Repeat timing is not a controlled host-load intervention or causal proof',
                      'Both workers resident; prior admission ran them serially; matched within-study comparison primary'],
        resources=dict(job_seconds=c.JOB_SECONDS, collection_seconds=c.COLLECTION_CUTOFF_SECONDS,
            stage_seconds=c.STAGES, evidence_bytes=c.MAX_EVIDENCE_BYTES,
            final_reserve_bytes=c.FINAL_RESERVE_BYTES, max_worker_rss=2*c.GiB,
            initial_disk=c.INITIAL_FREE_DISK_BYTES, maintained_disk=5*c.GiB,
            minimum_available_ram=2*c.GiB, stable_initial_ram_seconds=60,
            cumulative_ram_wait_seconds=c.RAM_WAIT_SECONDS))


def prepare():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    fixed(c.OUT/'runtime_sources.json', verify_runtime_sources())
    panel=build_panel()
    fixed(PANEL,panel);fixed(MATRIX,build_matrix(panel));fixed(c.OUT/'study_plan.json',build_plan())
    return dict(status='prepared_model_free', positions=6, ordinary_requests=144, new_games=0)


if __name__=='__main__': print(c.canonical(prepare()))
