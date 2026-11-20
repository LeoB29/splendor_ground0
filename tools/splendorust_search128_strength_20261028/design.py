"""Prospective fixed128 comparison and corrected model-free reserved positions."""
from . import common as c, schedule
from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules

PANEL = c.OUT / 'reserved_panel.json'
RESERVED_DOMAIN = 'splendorust_search128_strength_reserved_20261028_v1'


def fixed(path, value):
    if path.exists():
        c.require(c.read(path) == value, 'Prepared artifact must not change: ' + str(path))
    else:
        c.save(path, value)
    return c.artifact(path)


def build_reserved_panel():
    source_guard = guard.verify_runtime_sources()
    positions = []
    for base_ply in (12, 36, 60):
        for seat in (0, 1):
            for replicate in range(3):
                seed = int(c.digest([RESERVED_DOMAIN, base_ply, seat, replicate]), 16)
                env = guard.make_environment(seed=seed)
                state = env.initial_state()
                target = base_ply + seat
                c.require(state.current_player == 0, 'Reserved startseat must be zero')
                for ply in range(target):
                    c.require(not state.terminal, 'Predetermined trajectory ended early; no replacement')
                    before = rules.serialize_snapshot(state)
                    legal = env.legal_actions(state)
                    c.require(bool(legal), 'Reserved trajectory has no complete legal support')
                    index = int(c.digest([RESERVED_DOMAIN, str(seed), ply, 'action']), 16) % len(legal)
                    action = legal[index]
                    state = env.step(state, action)
                    guard.check_training_transition(before, rules.serialize_action(action), rules.serialize_snapshot(state))
                snapshot = rules.serialize_snapshot(state)
                guard.decode_training_snapshot(snapshot)
                c.require(not state.terminal and state.current_player == seat and state.turn_index == target,
                          'Reserved position target differs; no replacement')
                positions.append(dict(id=f'reserved_{len(positions):02d}', observer=rules.observation_payload(state),
                                      before_snapshot=snapshot, seed_decimal=str(seed), target_ply=target, seat=seat))
    return dict(schema='search128_reserved_panel_v1', ruleset_id=guard.RULESET_ID,
                catalogue_sha256=guard.CATALOGUE_SHA256, turn_contract=guard.TURN_CONTRACT,
                source_guard=source_guard, domain=RESERVED_DOMAIN, positions=positions,
                model_calls=0, training=False, action_selection='SHA256 fixed legal-index policy',
                no_result_selection=True, no_replacement=True)


def build_plan():
    return dict(schema='search128_strength_bounded_plan_v1', phase='strength',
        candidate=dict(route='research', world_pool=3, simulations=128, seed_label='A'),
        opponent=dict(route='frozen', world_pool=3, simulations=64, seed_label='A',
            original_policy_profile_sha256='2b60b56419c938ecd5e96d5b9c9a5153f36b2796d4e4b35219b746dc7dfed6d9'),
        games=406, operational_games=6, main_games=400, main_pairs=200,
        operational_games_excluded_from_primary=True,
        total_scientific_qualification_decisions=203, qualification_ordinary_requests=84,
        qualification_bundled_decisions=dict(frozen=52, research=67),
        reproduction_maximum=48, workers_maximum=c.MAX_WORKERS, graph_workers_maximum=c.MAX_GRAPH_BUILDS,
        graph_cases_per_worker=11, seconds_per_move=10., internal_seconds=9.,
        deployment_target_seconds=1., descriptive_thresholds_seconds=[.7, .9, 1.],
        timing_thresholds_do_not_gate_admission=True, late_completed_moves_accepted=True,
        timeouts='9s/10s player administrative guard: stop qualification/operations; main loss and continue',
        infrastructure_scientific_resource_failures='stop incomplete',
        no_retry=True, no_replacement=True, no_sample_extension=True,
        primary=dict(score='win1 draw0.5 loss0 including player research-guard forfeits',
            resampling_unit='swapped-seat pair', paired_bootstrap_replicates=50000,
            paired_bootstrap_seed=20261028128, minimum_useful_score=.55,
            ci95_lower_strictly_above=.5, minimum_score_each_seat=.5,
            all_verification_required=True, automatic_promotion=False),
        estimand='Fixed128 versus64, both3worlds, under symmetric research abort guard; not one-second deployment',
        independent_initial_worlds=200, operational_worlds=3,
        preserved_original_one_second_players=True, new_execution_profiles_required=True,
        same_evidence_path_both_arms=True, increased_compute_not_equal_budget=True,
        larger_data_deferred=True, external_testing='paused', training=False,
        resources=dict(job_seconds=c.JOB_SECONDS, collection_seconds=c.COLLECTION_CUTOFF_SECONDS,
            stages=c.STAGES, evidence_bytes=c.MAX_EVIDENCE_BYTES, final_reserve_bytes=c.FINAL_RESERVE_BYTES,
            initial_disk_bytes=c.INITIAL_FREE_DISK_BYTES, maintained_disk_bytes=5*c.GiB,
            available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES, startup_available_ram_bytes=c.STARTUP_MINIMUM_AVAILABLE_RAM_BYTES,
            combined_worker_rss_bytes=c.MAX_RSS_BYTES, initial_stable_ram_seconds=60,
            cumulative_ram_wait_seconds=c.RAM_WAIT_SECONDS, independent_review_seconds=3600,
            forecast_contingency_factor=1.5, one_nonresettable_clock=True))


def prepare():
    fixed(PANEL, build_reserved_panel())
    fixed(c.OUT / 'study_plan.json', build_plan())
    fixed(c.OUT / 'schedule.json', schedule.make_schedule())
    return dict(status='prepared_model_free', phase=c.PHASE, games=406, main_games=400, model_calls=0)


if __name__ == '__main__':
    print(c.canonical(prepare()))
