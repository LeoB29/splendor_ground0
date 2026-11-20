"""Measured prospective prices and deterministic saved-snapshot forecasts."""
import math
from . import common as c

FACTOR = 1.5


def nonwait_seconds(value):
    if 'nonwait_seconds' in value:
        seconds = value['nonwait_seconds']
    else:
        total, waited = value['seconds'], value.get('ram_wait_seconds', 0.)
        c.require(type(total) in (int, float) and type(waited) in (int, float)
                  and math.isfinite(total) and math.isfinite(waited) and 0 <= waited < total,
                  'Measured RAM wait is finite and inside the completed cost interval')
        seconds = total - waited
    c.require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
              'Finite positive nonwait elapsed time')
    return seconds


def _cost_rows(paths):
    rows = []
    for path in paths:
        value = c.read(path)
        seconds = nonwait_seconds(value)
        growth = value['accounted_growth_bytes']
        c.require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
                  and type(growth) is int and growth > 0, 'Finite measured nonwait cost and positive evidence growth')
        rows.append(dict(seconds=seconds, bytes=growth, artifact=c.artifact(path)))
    c.require(rows, 'Measured costs required before admission')
    return rows


def resource_plan():
    """Pilot uses the completed new preflight matched128 costs, never quota scaling."""
    teacher = _cost_rows([c.ENGINEERING_OUT / 'games' / f'g{i:03d}' / 'cost.json' for i in range(4)])
    starts = _cost_rows([c.ENGINEERING_OUT / 'startup_costs' / f'{x}.json'
                        for x in ('capture_off', 'capture_on', 'replay_on')])
    # These intervals use one supervisor monotonic domain throughout. They are
    # conservative cost observations only, never accepted training examples or
    # certification of the failed learned-request clock handshake.
    teacher += _cost_rows([c.FAILED_QUALIFICATION_OUT / 'games' / f'g{i:03d}' / 'cost.json'
                          for i in range(4)])
    starts += _cost_rows([c.FAILED_QUALIFICATION_OUT / 'startup_costs' / f'{x}.json'
                         for x in ('capture_off', 'capture_on', 'replay_on')])
    if c.SCOPE == 'pilot':
        base = c.QUALIFICATION_OUT / 'assessment'
        teacher += _cost_rows([c.QUALIFICATION_OUT / 'games' / f'g{i:03d}' / 'cost.json' for i in range(4)])
        starts += _cost_rows([c.QUALIFICATION_OUT / 'startup_costs' / f'{x}.json'
                             for x in ('capture_off', 'capture_on', 'replay_on')])
        provenance = 'Completed new-route preflight matched128 games, fixed before pilot collection'
    else:
        base = c.ACCEPTED_OUT
        provenance = 'Conservative accepted512-versus256 proxy for six engineering operational games only'
    assessment = _cost_rows([base / 'games' / 'operational' / f'g{i:03d}' / 'cost.json' for i in range(6)])
    assessment_starts = _cost_rows([base / 'startup_costs' / f'{name}.json'
                                    for name in ('operational_g000_control', 'operational_g000_candidate',
                                                 'replay_control', 'replay_candidate')])
    groups = dict(source_game=teacher, source_worker=starts, assessment_game=assessment,
                  assessment_worker=assessment_starts)
    prices = {name: dict(seconds=max(x['seconds'] for x in rows) * FACTOR,
                        bytes=math.ceil(max(x['bytes'] for x in rows) * FACTOR))
              for name, rows in groups.items()}
    return dict(schema='selfplay_resource_prices_v1', scope=c.SCOPE, factor=FACTOR,
                prices=prices, measured=groups,
                inputs=[x['artifact'] for rows in groups.values() for x in rows],
                assessment_time_provenance=provenance,
                source_time_provenance='Historical engineering and failed-preflight same-domain source/startup intervals; new accepted preflight additionally required for pilot. No learned-request mixed-clock intervals are priced.',
                failed_attempt_costs_are_qualification=False,
                failed_attempt_training_examples_reused=0,
                fit_seconds_reserve=120 if c.SCOPE == 'qualification' else c.STAGES['fit'],
                fit_bytes_reserve=c.MAX_FIT_EVIDENCE_BYTES,
                source_replay_seconds_each=11., assessment_replay_seconds_each=11.,
                replay_bytes_each=2 * c.MiB,
                bookkeeping_seconds=300 if c.SCOPE == 'qualification' else 1800,
                bookkeeping_bytes=128 * c.MiB if c.SCOPE == 'qualification' else 2 * c.GiB,
                assessment_normal_workers=4 if c.SCOPE == 'qualification' else 84,
                source_collection_bookkeeping_seconds=120 if c.SCOPE == 'qualification' else 600,
                final_seconds_reserve=c.FINAL_SECONDS, final_bytes_reserve=c.FINAL_RESERVE_BYTES,
                caveats=['Measured maxima times1.5 are forecasts, not worst-case tail guarantees.',
                         'Every remaining downstream stage and remaining cumulative RAM allowance is priced.',
                         'Forfeit replacement lifecycle projections are advisory; actual clock, storage and process limits remain enforced.',
                         'Forecasts never stop this continuation; never reduce the sample or extend the original clocks.'])


def source_forecast(freeze, snapshot, *, source_costs=None):
    """Pure replayable forecast; snapshot contains clocks/resources, never outcomes."""
    descriptor = freeze['resource_plan']
    c.verify([descriptor])
    plan = c.read(c.ROOT / descriptor['path'])
    c.require(plan['scope'] == c.SCOPE and plan['factor'] == FACTOR, 'Exact scope resource prices')
    required = {'monotonic_now', 'unix_now', 'original_monotonic_origin', 'original_unix_origin',
                'stage_monotonic_deadline', 'stage_unix_deadline', 'accounted_bytes',
                'cumulative_ram_wait_seconds', 'completed_games', 'qualified_source_workers'}
    c.require(set(snapshot) == required, 'Exact saved source-forecast snapshot fields')
    c.require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in snapshot.values()),
              'Finite nonnegative resource snapshot')
    completed, workers = snapshot['completed_games'], snapshot['qualified_source_workers']
    c.require(type(completed) is int and 0 <= completed <= c.MAX_SOURCE_GAMES
              and type(workers) is int and 0 <= workers <= c.MAX_SOURCE_WORKER_LIFECYCLES,
              'Exact completed source counts')
    prices = {k: dict(v) for k, v in plan['prices'].items()}
    observed = list(source_costs or [])
    for row in observed:
        sec = nonwait_seconds(row)
        growth = row['accounted_growth_bytes']
        c.require(type(sec) in (int, float) and math.isfinite(sec) and sec > 0
                  and type(growth) is int and growth > 0, 'Finite actual source game cost')
        prices['source_game']['seconds'] = max(prices['source_game']['seconds'], sec * FACTOR)
        prices['source_game']['bytes'] = max(prices['source_game']['bytes'], math.ceil(growth * FACTOR))
    c.require(len(observed) <= completed, 'No cost from an uncompleted source game')
    source_left = c.MAX_SOURCE_GAMES - completed
    source_workers_left = c.MAX_SOURCE_WORKER_LIFECYCLES - workers
    ram_left = max(0., c.RAM_WAIT_SECONDS - snapshot['cumulative_ram_wait_seconds'])
    source_seconds = source_left * prices['source_game']['seconds'] + source_workers_left * prices['source_worker']['seconds']
    source_seconds += c.MAX_SOURCE_REPLAY_DECISIONS * plan['source_replay_seconds_each']
    source_seconds += plan['source_collection_bookkeeping_seconds'] + ram_left
    assessment_seconds = c.MAX_ASSESSMENT_GAMES * prices['assessment_game']['seconds']
    assessment_seconds += plan['assessment_normal_workers'] * prices['assessment_worker']['seconds']
    assessment_seconds += c.MAX_ASSESSMENT_REPLAY_DECISIONS * plan['assessment_replay_seconds_each']
    downstream_seconds = plan['fit_seconds_reserve'] + assessment_seconds + plan['bookkeeping_seconds']
    model_seconds = source_seconds + downstream_seconds
    prefinal_bytes = snapshot['accounted_bytes'] + source_left * prices['source_game']['bytes']
    prefinal_bytes += source_workers_left * prices['source_worker']['bytes']
    prefinal_bytes += c.MAX_ASSESSMENT_GAMES * prices['assessment_game']['bytes']
    prefinal_bytes += plan['assessment_normal_workers'] * prices['assessment_worker']['bytes']
    prefinal_bytes += c.MAX_REPLAY_DECISIONS * plan['replay_bytes_each'] + plan['fit_bytes_reserve'] + plan['bookkeeping_bytes']
    remaining_model = min(snapshot['original_monotonic_origin'] + c.MODEL_CUTOFF_SECONDS - snapshot['monotonic_now'],
                          snapshot['original_unix_origin'] + c.MODEL_CUTOFF_SECONDS - snapshot['unix_now'])
    remaining_global = min(snapshot['original_monotonic_origin'] + c.JOB_SECONDS - snapshot['monotonic_now'],
                           snapshot['original_unix_origin'] + c.JOB_SECONDS - snapshot['unix_now'])
    remaining_collection = min(snapshot['stage_monotonic_deadline'] - snapshot['monotonic_now'],
                               snapshot['stage_unix_deadline'] - snapshot['unix_now'])
    main_seconds = (0. if c.SCOPE == 'qualification' else
                    400 * prices['assessment_game']['seconds'] + 80 * prices['assessment_worker']['seconds'])
    checks = dict(collection=source_seconds <= remaining_collection, model=model_seconds <= remaining_model,
                  total=model_seconds + plan['final_seconds_reserve'] <= remaining_global,
                  evidence=prefinal_bytes <= c.PREFINAL_EVIDENCE_BYTES,
                  assessment_stage=main_seconds <= c.STAGES['assessment'],
                  cumulative_ram_wait=snapshot['cumulative_ram_wait_seconds'] <= c.RAM_WAIT_SECONDS)
    return dict(schema='selfplay_saved_resource_forecast_v1', scope=c.SCOPE, snapshot=snapshot,
                resource_plan=descriptor, observed_source_costs=observed, effective_prices=prices,
                remaining_source_games=source_left, remaining_source_workers=source_workers_left,
                reserved_remaining_ram_seconds=ram_left, projected_collection_seconds=source_seconds,
                projected_downstream_model_seconds=downstream_seconds, projected_model_seconds=model_seconds,
                projected_main_assessment_seconds=main_seconds, projected_prefinal_bytes=prefinal_bytes,
                reserved_final_bytes=c.FINAL_RESERVE_BYTES, reserved_final_seconds=plan['final_seconds_reserve'],
                remaining_model_seconds=remaining_model, remaining_collection_seconds=remaining_collection,
                remaining_global_seconds=remaining_global, checks=checks, affordable=all(checks.values()),
                outcomes_used=False, changes_to_sample_or_deadlines=False)
