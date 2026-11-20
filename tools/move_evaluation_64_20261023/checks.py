"""Model-free result, exact-repeat and archived-prefix admission checks."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

from tools.external_strength_screen_20261018 import rules
from . import common as c

PROFILES = {'A': 'fixed4_world64_raw035_both_seats_20261023_v1',
            'B': 'fixed4_world64_raw035_own_greedy_entity_opponent_20261023_v1'}


def arm_profile(arm):
    c.require(type(arm) is str and arm in PROFILES, 'Explicit A64/B64 arm required')
    return PROFILES[arm]


def prefix_schedule(full):
    c.require(type(full) is dict and set(full) == {'belief_sha256', 'candidate_nonce', 'world_seeds'},
              'Full128 schedule schema')
    seeds = full['world_seeds']
    c.require(type(seeds) is list and len(seeds) == 128 and all(type(seed) is str for seed in seeds) and
              len(set(seeds)) == 128, 'Full128 unique schedule')
    return {**deepcopy(full), 'world_seeds': seeds[:64]}


def select_outcomes64(worlds, support):
    c.require(type(worlds) is list and len(worlds) == 64 and type(support) is int and 1 <= support <= 4,
              'Complete64 support')
    totals = [0] * support
    for world in worlds:
        c.require(len(world['outcomes']) == support, 'Complete forced candidate support')
        for index, outcome in enumerate(world['outcomes']):
            c.require(type(outcome['wdl']) in (int, float) and outcome['wdl'] in (-1, 0, 1), 'WDL domain')
            totals[index] += int(outcome['wdl'])
    return max(range(support), key=lambda index: totals[index]), [value/64 for value in totals]


def check_selection(result, payload, arm):
    profile = arm_profile(arm)
    c.require(type(result) is dict and set(result) == {
        'decision', 'trace', 'profile_id', 'observer_sha256', 'ruleset_id', 'catalogue_sha256'}, 'Selection result schema')
    c.require(result['profile_id'] == profile and result['observer_sha256'] == c.digest(payload['observer']) and
              result['ruleset_id'] == rules.RULESET_ID and result['catalogue_sha256'] == rules.CATALOGUE_SHA256,
              'Selection result identity')
    decision = result['decision']; record = decision['record']
    c.require(type(decision) is dict and set(decision) == {'action', 'record'} and
              record['profile'] == profile and record['ruleset_id'] == rules.RULESET_ID and
              record['catalogue_sha256'] == rules.CATALOGUE_SHA256 and
              record['planner'] == ('reference' if arm == 'A' else 'opponent_policy_substitution'), 'Decision identity')
    actions = payload['candidates']['actions']; support = len(actions)
    c.require(1 <= support <= 4 and c.digest(record['candidates']) == c.digest(payload['candidates']) and
              c.digest(record['schedule']) == c.digest(payload['selection_schedule']) and
              c.digest(record['schedule']) == c.digest(prefix_schedule(record['proposal']['schedule'])),
              'Candidate/schedule/prefix identity')
    worlds = record['worlds']
    c.require(type(worlds) is list and len(worlds) == 64, 'Exactly64 complete worlds')
    for seed, world in zip(record['schedule']['world_seeds'], worlds, strict=True):
        c.require(type(world) is dict and set(world) == {'seed', 'state_sha256', 'outcomes'} and
                  world['seed'] == seed and type(world['outcomes']) is list and len(world['outcomes']) == support,
                  'World identity/support')
    choice, means = select_outcomes64(worlds, support)
    c.require(type(record['choice']) is int and record['choice'] == choice and
              c.digest(record['means']) == c.digest(means) and
              c.digest(decision['action']) == c.digest(actions[choice]), 'Complete64 reduction/first maximum')
    trace = result['trace']
    c.require(type(trace) is dict and set(trace) == {'path', 'sha256', 'bytes', 'records'} and
              type(trace['records']) is int and trace['records'] == support*64, 'Complete64 trace count')
    c.verify([{k: trace[k] for k in ('path', 'sha256', 'bytes')}])
    return {'status': 'passed', 'arm': arm, 'worlds': 64, 'continuations': support*64,
            'result_sha256': c.digest(result), 'scientific_replay_still_required': True}


def compare_duplicate(first, result):
    left, right = deepcopy(first), deepcopy(result)
    for item in (left, right):
        c.verify([{k: item['trace'][k] for k in ('path', 'sha256', 'bytes')}])
        del item['trace']['path']
    c.require(c.digest(left) == c.digest(right), 'Repeated scientific output/gzip differs')
    return {'status': 'passed', 'scientific_output_and_gzip_exact': True,
            'excluded_field': 'trace.path', 'first_sha256': c.digest(first), 'repeat_sha256': c.digest(result)}


def check_archived_prefix(result, archived_result, arm, *, trace_path=None, archived_trace_path=None):
    """Caller authenticates archived artifact provenance before supplying it.

    Archived result may be its scientific wrapper or the extracted decision.
    B requires exact decompressed step rows as well as state/outcome identities.
    A's historical action-trace hash remains in every outcome comparison.
    """
    arm_profile(arm)
    new = result['decision']['record']
    old = archived_result['decision'] if 'decision' in archived_result else archived_result
    old = old['record']
    c.require(type(old['worlds']) is list and len(old['worlds']) == 128 and
              type(new['worlds']) is list and len(new['worlds']) == 64, 'Archived128/new64 lengths')
    for key in ('root_identity', 'privacy', 'candidates', 'proposal', 'scores'):
        c.require(c.digest(new[key]) == c.digest(old[key]), 'Archived prefix root field: '+key)
    c.require(c.digest(new['schedule']) == c.digest(prefix_schedule(old['schedule'])) and
              c.digest(new['worlds']) == c.digest(old['worlds'][:64]), 'Archived first64 world/outcome/hash equality')
    support = len(new['candidates']['actions'])
    compared = 0
    if arm == 'B':
        c.require(trace_path is not None and archived_trace_path is not None, 'B prefix requires full step-row comparison')
    if trace_path is not None or archived_trace_path is not None:
        c.require(trace_path is not None and archived_trace_path is not None, 'Both prefix trace paths required')
        with gzip.open(Path(trace_path), 'rt', encoding='utf-8') as current, \
             gzip.open(Path(archived_trace_path), 'rt', encoding='utf-8') as previous:
            for index in range(64*support):
                lines = [stream.readline(8*1024**2) for stream in (current, previous)]
                c.require(all(line and line.endswith('\n') for line in lines), 'Missing/oversized archived prefix trace')
                left, right = [json.loads(line) for line in lines]
                c.require(left['world_index'] == index//support and left['candidate_index'] == index%support and
                          c.digest(left) == c.digest(right), 'Archived prefix step identities differ')
                compared += 1
            c.require(not current.read(1), 'Extra64 continuation traces')
    return {'status': 'passed', 'arm': arm, 'worlds': 64, 'outcomes': 64*support,
            'full_step_rows_compared': compared, 'historical_choices_used_for_selection': False,
            'historical_choice': old['choice'], 'new_choice': new['choice'],
            'choice_changed_from128': old['choice'] != new['choice']}
