"""Separate lean core comparison; all additional host sampling is disabled."""
from copy import deepcopy
from . import common as c
from tools.splendorust_pool6_core_compare_20261027 import design as original

PRESERVED_ATTEMPT=c.ROOT/'outputs/splendorust_pool6_core_compare_20261027_01'
PRESERVED_FREEZE_SHA='e095d6e8e04100925968d95a39a0f4e49aa1781988cfef4f7fe617f0eaf1a43c'
PRESERVED_REVIEW_SHA='ec45ea0f2817fbea36eedfe86f5c8c9df3c80833e7bf557dc5d96e81d733d0a9'
PANEL=c.OUT/'saved_panel.json'
MATRIX=c.OUT/'request_matrix.json'
TARGETS=original.TARGETS
CONTROLS=original.CONTROLS
CORES=original.CORES
ROUTES=original.ROUTES
ROUNDS=original.ROUNDS
ARMS=original.ARMS


def fixed(path,value):
    if path.exists():c.require(c.read(path)==value,'Prepared artifact changed: '+str(path))
    else:c.save(path,value)
    return c.artifact(path)


def build_panel():
    c.require(c.file_hash(PRESERVED_ATTEMPT/'freeze.json')==PRESERVED_FREEZE_SHA,'Prior negative attempt identity')
    panel=original.build_panel()
    c.require(panel==c.read(PRESERVED_ATTEMPT/'saved_panel.json'),'Exact same four roots and authenticated historical targets')
    return panel


def build_matrix(panel=None):
    panel=build_panel() if panel is None else panel
    previous=original.build_matrix(panel)
    rows=[]
    for old in previous['requests']:
        if old['stage']!='diagnostic':continue
        spec=deepcopy(old)
        spec.update(index=len(rows),sequence=len(rows),core_pair_index=len(rows)//2,observer_enabled=False)
        rows.append(spec)
    c.require(len(rows)==192 and all(r['observer_enabled'] is False for r in rows),'Exactly192 OFF-only requests')
    return dict(schema='pool6_core_compare_request_matrix_v2',requests=rows,rounds=ROUNDS,
        position_order='Same four-root Williams square repeated3 as the prior planned full comparison',
        arm_order=previous['arm_order'],observer_order='OFF for every request; no extra host counter collection',
        no_warmup_dropping=True,no_retries=True,no_replacements=True)


def engineering_contract():
    return deepcopy(original.engineering_contract())


def host_observer_contract():
    return dict(schema='core_compare_extra_host_sampler_disabled_v1',enabled=False,
        requests=192,enabled_requests=0,disabled_requests=192,
        interval_seconds=.1,maximum_buffer_bytes=c.HOST_OBSERVER_MAX_BYTES,
        maximum_compute_targets=12,supervisor_logical_processors=list(range(4,20)),
        lifecycle='Original OFF handshake; parked lifecycle thread waits on finish event without periodic sampling',
        per_window_snapshots=0,per_window_context_reads=0,active_decision_disk_writes=False,
        placement_checks='Lifecycle thread verifies its own placement before and after each original decision clock',
        retained_measurements=['original complete-response monotonic clock','trusted boundary timings',
            'existing worker resource/ownership sampler','pre-request available-RAM admission'],
        unavailable_measurements=['additional per-process CPU/fault/I/O snapshots','additional host-wide CPU/commit samples',
            'additional context-switch aggregates'],
        disabled_receipts_required=True,accidental_enabled_request_fatal=True,
        source_counter_implementation_preserved=True,overhead_gate='not_applicable_extra_sampler_disabled',
        limitations=['One fixed process instance per player/core; core and instance effects cannot be separated',
            'CPU placement is not exclusive reservation against external programs',
            'No added host counters are available to diagnose scheduling or paging causes',
            'New prospective OFF-only study does not amend the prior negative overhead result'])


def build_plan():
    return dict(schema='pool6_core_compare_bounded_plan_v2',phase=c.PHASE,
        purpose='Compare complete-response headroom on fixed cores with extra host sampler disabled',
        positions=4,selected_slow_targets=2,controls=2,rounds=12,logical_cpus=[0,2],arms=list(ARMS),
        overhead_requests=0,diagnostic_requests=192,maximum_ordinary_requests=192,
        startup_complete_decisions=208,graph_cases=44,maximum_parent_requests=200,
        maximum_scientific_decisions=400,resident_workers=4,worker_replacements=0,
        candidate=dict(route='research',world_pool=6,simulations=64,seed_label='A'),
        control=dict(route='frozen',world_pool=3,simulations=64,seed_label='A'),
        seconds_per_move=1.,internal_seconds=.9,margin_seconds=.7,
        margin_role='Unchanged engineering criterion; margin crossing recorded, not a stop; prior negative admissions remain negative',
        engineering_contract=engineering_contract(),host_observer_contract=host_observer_contract(),
        failure_policy='Stop at first timeout, mismatch, unexpected rejection or resource/integrity failure; preserve the ordered prefix. No retry, replacement or selective continuation.',
        players_unchanged=True,payload_changes=False,trusted_boundary_observer_unchanged=True,
        extra_host_sampler_enabled=False,new_games=0,promotion=False,training=False,
        larger_data_deferred=True,external_testing='paused',
        resources=dict(job_seconds=c.JOB_SECONDS,collection_seconds=c.COLLECTION_CUTOFF_SECONDS,
            stage_seconds=c.STAGES,evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
            max_worker_rss=2*c.GiB,initial_disk=c.INITIAL_FREE_DISK_BYTES,maintained_disk=5*c.GiB,
            inherited_pre_request_ram_reserve=2*c.GiB,diagnostic_available_ram=c.DIAGNOSTIC_AVAILABLE_RAM_BYTES,
            available_ram_guard_scope='Before requests; no new active-request RAM stop or clock reset',
            stable_initial_and_recovery_seconds=60,cumulative_ram_wait_seconds=c.RAM_WAIT_SECONDS))


def prepare():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    fixed(c.OUT/'runtime_sources.json',verify_runtime_sources())
    panel=build_panel();matrix=build_matrix(panel)
    fixed(PANEL,panel);fixed(MATRIX,matrix);fixed(c.OUT/'study_plan.json',build_plan())
    return dict(status='prepared_model_free',positions=4,overhead_requests=0,diagnostic_requests=192,
        extra_host_sampler_enabled=False,new_games=0)


if __name__=='__main__':print(c.canonical(prepare()))
