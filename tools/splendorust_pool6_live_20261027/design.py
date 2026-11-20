"""Prospective live sampling comparison and model-free reserved positions."""
from . import common as c, schedule
from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules

PREVIOUS=c.ROOT/'outputs/splendorust_sampling_stability_20261027_01'
PREVIOUS_FREEZE_SHA='cc550ef7f48597840ed5e41166b317574ef78d8ad88d3ae11b6ef5baca5b1eaa'
PREVIOUS_REVIEW_SHA='7c1fbf2740a7acfd83bf98eb8ffe2a57266b80271f763362d092542bb9f0a952'
PANEL=c.QUAL_OUT/'reserved_panel.json'
RESERVED_DOMAIN='splendorust_pool6_live_reserved_20261027_v1'


def fixed(path,value):
    if path.exists(): c.require(c.read(path)==value,'Prepared artifact must not change: '+str(path))
    else:c.save(path,value)
    return c.artifact(path)


def build_reserved_panel():
    """No learned policy or outcome selection; all18 seeds/ply targets fixed."""
    source_guard=guard.verify_runtime_sources()
    positions=[]
    for base_ply in (12,36,60):
        for seat in (0,1):
            for replicate in range(3):
                seed=int(c.digest([RESERVED_DOMAIN,base_ply,seat,replicate]),16)
                env=rules.CorrectedEnv(seed=seed)
                state=env.initial_state()
                target=base_ply+seat
                c.require(state.current_player==0,'Reserved target assumes original startseat0')
                for ply in range(target):
                    c.require(not state.terminal,'Predetermined reserved trajectory ended early; no replacement')
                    before=rules.serialize_snapshot(state)
                    legal=env.legal_actions(state)
                    c.require(bool(legal),'Reserved trajectory lacks complete legal support')
                    index=int(c.digest([RESERVED_DOMAIN,str(seed),ply,'action']),16)%len(legal)
                    action=legal[index]
                    state=env.step(state,action)
                    guard.check_training_transition(before,rules.serialize_action(action),rules.serialize_snapshot(state))
                snapshot=rules.serialize_snapshot(state)
                guard.decode_training_snapshot(snapshot)
                c.require(not state.terminal and state.current_player==seat and state.turn_index==target,
                          'Reserved position target differs; no replacement')
                positions.append(dict(id=f'reserved_{len(positions):02d}',observer=rules.observation_payload(state),
                    before_snapshot=snapshot,seed_decimal=str(seed),target_ply=target,seat=seat))
    return dict(schema='pool6_live_reserved_panel_v1',ruleset_id=guard.RULESET_ID,
        catalogue_sha256=guard.CATALOGUE_SHA256,turn_contract=guard.TURN_CONTRACT,
        source_guard=source_guard,domain=RESERVED_DOMAIN,positions=positions,
        model_calls=0,training=False,action_selection='SHA256 fixed legal-index policy',
        no_result_selection=True,no_replacement=True)


def build_plan():
    return dict(schema='pool6_live_bounded_plan_v1',phase=c.PHASE,
        candidate=dict(route='research',world_pool=6,simulations=64,seed_label='A',telemetry=False),
        opponent=dict(route='frozen',world_pool=3,simulations=64,seed_label='A',
            profile_sha256='2b60b56419c938ecd5e96d5b9c9a5153f36b2796d4e4b35219b746dc7dfed6d9'),
        games=6 if c.PHASE=='qualification' else 200,
        pairs=3 if c.PHASE=='qualification' else 100,
        operational_games_excluded_from_primary=True,
        total_scientific_qualification_decisions=194 if c.PHASE=='qualification' else 0,
        qualification_ordinary_requests=90 if c.PHASE=='qualification' else 0,
        reproduction_maximum=48,workers_maximum=c.MAX_WORKERS,graph_workers_maximum=c.MAX_GRAPH_BUILDS,
        graph_cases_per_worker=11,seconds_per_move=1.,internal_seconds=.9,
        admission_engineering_margin_seconds=.7,assessment_margin_descriptive_only=True,
        timeouts='loss; stop admission, continue fixed assessment',
        infrastructure_scientific_resource_failures='stop incomplete',
        no_retry=True,no_replacement=True,no_sample_extension=True,
        primary=None if c.PHASE=='qualification' else dict(
            score='win1 draw0.5 loss0 including timeouts',resampling_unit='swapped-seat pair',
            paired_bootstrap_replicates=50000,paired_bootstrap_seed=2026102706,
            minimum_useful_score=.55,ci95_lower_strictly_above=.5,minimum_score_each_seat=.5,
            all_verification_required=True,automatic_promotion=False),
        coupled_intervention='Root-pool draws also advance shared RNG for later hidden input completions',
        larger_data_deferred=True,external_testing='paused',training=False,
        resources=dict(job_seconds=c.JOB_SECONDS,collection_seconds=c.COLLECTION_CUTOFF_SECONDS,
            stages=c.STAGES,evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
            initial_disk_bytes=c.INITIAL_FREE_DISK_BYTES,maintained_disk_bytes=5*c.GiB,
            available_ram_bytes=2*c.GiB,combined_worker_rss_bytes=2*c.GiB,
            initial_stable_ram_seconds=60,cumulative_ram_wait_seconds=c.RAM_WAIT_SECONDS))


def prepare():
    if c.PHASE=='qualification':fixed(PANEL,build_reserved_panel())
    fixed(c.OUT/'study_plan.json',build_plan())
    fixed(c.OUT/'schedule.json',schedule.make_schedule())
    return dict(status='prepared_model_free',phase=c.PHASE,games=schedule.TOTAL_GAMES)


if __name__=='__main__': print(c.canonical(prepare()))
