"""Accepted fixed training sample; new games belong only to assessment."""
from . import common as c
from tools.splendorust_selfplay_pilot_resume2_20261103.design import schedule as accepted_schedule

DOMAIN = 'splendorust_selfplay_pilot_v2_20261102_v1'
FIT_SEED = 202611012

def schedule(scope=None):
    c.require(scope in (None, 'pilot'), 'Policy-head study reuses only the accepted pilot corpus')
    return accepted_schedule('pilot')

def validate_spec(spec):
    c.require(type(spec) is dict and type(spec.get('game_index')) is int and 0<=spec['game_index']<256, 'Saved source-game index')
    c.require(spec == schedule()[spec['game_index']], 'Exact accepted source specification')
    return spec

def fixed(path, value):
    if path.exists():
        c.require(c.read(path) == value, 'Prepared immutable document changed')
        return c.artifact(path)
    return c.save(path, value)

def source_lifecycle_inventory():
    return {}

def collection_blocks():
    return []

def remaining_collection_blocks():
    return []

def resource_plan():
    return dict(schema='policy_head_resource_plan_v1', source_games_reused=256,
        source_games_generated=0, training_games=192, training_roots=10660,
        original_reference_preserved=True, new_operational_games=6, new_main_games=400,
        job_seconds=c.JOB_SECONDS, model_seconds=c.COLLECTION_CUTOFF_SECONDS, stages=c.STAGES,
        available_ram_reserve_bytes=0, stable_ram_seconds=0, forecasts_advisory=True,
        resource_forecasts_may_stop=False, maximum_inference_rss_bytes=c.MAX_INFERENCE_RSS_BYTES,
        maximum_fitter_rss_bytes=c.MAX_FITTER_RSS_BYTES, minimum_actual_free_disk_bytes=c.MINIMUM_FREE_DISK_BYTES,
        maximum_new_evidence_bytes=c.MAX_EVIDENCE_BYTES, final_evidence_reserve_bytes=c.FINAL_RESERVE_BYTES,
        previous_measured_fit_seconds=693.5, previous_measured_400_games_seconds=27002.,
        previous_measured_independent_verification_seconds=7209.7,
        measured_times_are_descriptive_not_launch_gates=True)
