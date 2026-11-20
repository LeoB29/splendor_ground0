"""Additive saved-only completion; never publish into the failed qualification.

The original final owner failed while recreating an already passed capture
receipt. The inherited scientific checks remain identical. Existing receipts
are compared, and newly computed coverage is returned to the separate audit
owner. Neither the expired original final stage nor its incomplete status is
relabelled as a successful original attempt.
"""
from pathlib import Path
import json
import os

from tools.splendorust_selfplay_20261101 import qualification_control as original

c = original.c
data, design, transport = original.data, original.design, original.transport
load_inputs = original.load_inputs
game_folder = original.game_folder
validate_row = original.validate_row
verify_capture_fixtures = original.verify_capture_fixtures
lifecycle_inventory = original.lifecycle_inventory
qualification_evidence = original.qualification_evidence
select_replays = original.select_replays
verify_reproduction = original.verify_reproduction
verify_fit = original.verify_fit
capture_forecast = original.capture_forecast
verify_phase_transition = original.verify_phase_transition
verify_exchanges = original.verify_exchanges
verify_runtime = original.verify_runtime

AUDIT_OUT = c.ROOT / 'outputs/splendorust_selfplay_audit_20261102_02'
ORIGINAL_FREEZE_SHA = '59af3864e79509a22307386a8d73e80373e4934af10e54f558a7538b2de75736'


def verify_existing(path, expected):
    """Exact existing-receipt verification; missing files remain failures."""
    return original.save_or_verify(Path(path), expected, False)


def publish_new(name, value):
    """Exclusive new audit output only; no old output or overwrite route."""
    c.require(Path(name).name == name and name.endswith('.json'), 'Simple new audit receipt name')
    target = AUDIT_OUT / name
    c.require(AUDIT_OUT.resolve().is_relative_to(c.ROOT.resolve()) and
              not AUDIT_OUT.resolve().is_relative_to(c.OUT.resolve()), 'Separate audit destination')
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return c.artifact(target)


def verify_saved(envelope,freeze):
    schedule,plan=load_inputs(freeze,envelope);manifest=c.read(c.OUT/'corpus_manifest.json')
    c.require(manifest==data.build_manifest([c.artifact(game_folder(s)/'game.json') for s in schedule],schedule,qualification_only=True),'Exact fixed completed corpus')
    count=0
    for spec,initial in zip(schedule,design.initial_states(),strict=True):
        game=c.read(game_folder(spec)/'game.json');c.require(game['initial']==initial['initial'],'Frozen source setup')
        c.require([p.name for p in sorted((game_folder(spec)/'moves').glob('*.json'))]==[f'{i:06d}.json' for i in range(len(game['rows']))],'Exact contiguous saved game ledger')
        for row in game['rows']:
            envelope.guard();c.require(row==c.read(game_folder(spec)/'moves'/f"{row['ply']:06d}.json"),'Full game/row binding');validate_row(row,freeze,envelope);count+=1
    loaded=data.load_corpus(manifest,expected_specs=schedule,boundary=envelope.guard)
    verify_capture_fixtures(envelope,freeze,plan,writing=False)
    names=set(lifecycle_inventory());c.require({p.name for p in (c.OUT/'workers').iterdir() if p.is_dir()}==names,'Only three declared inference lifecycles')
    for name in sorted(names):
        qualification_evidence(name,envelope,freeze,plan,writing=False)
        proof=transport.verify_closed_worker(c.OUT/'workers'/name,freeze,envelope)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{name}.json'),'Independent full raw/clock/cleanup closure')
        ordinary_count=len(list((c.OUT/'workers'/name/'inputs').glob('*.json')))
        c.require(ordinary_count=={'capture_off':16,'capture_on':16+count,'replay_on':12}[name],
            'Every ordinary request belongs to a declared fixture, game, or reproduction')
    selection=select_replays(manifest['games'],envelope)
    c.require(selection==c.read(c.OUT/'replay_selection.json') and len(list((c.OUT/'replays').glob('*.json')))==12,'Outcome-blind exact twelve replay inventory')
    for index,item in enumerate(selection):
        observed=c.read(c.OUT/'replays'/f'{index:03d}.json');c.require(observed['selection']==item,'Replay source selection')
        verify_reproduction(c.read(c.ROOT/item['source']['path']),observed['row'],freeze,envelope)
    fit=verify_fit(envelope,freeze)
    gate=capture_forecast(envelope,freeze,writing=False);c.require(gate['passed'],'Passed whole-stage resource admission')
    verify_phase_transition(envelope)
    verify_exchanges(envelope,freeze)
    for stage in ('qualification','collection','fit'):
        start=c.read(c.OUT/'stages'/f'{stage}_start.json');end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        for key,origin in [('monotonic',envelope.start),('unix_time',envelope.wall_start)]:
            deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
            c.require(start[key]<=end[key]<start[deadline]==min(start[key]+c.STAGES[stage],origin+c.COLLECTION_CUTOFF_SECONDS),'Original stage/model clocks')
    result=dict(status='verified_corrected_selfplay_engineering_pipeline',completed_games=4,workers=3,
        graph_cases=33,scientific_qualification_decisions=201,capture_fixture_decisions=32,live_decisions=count,
        exact_reproductions=12,parent_requests=6+32+count+12,optimizer_steps=1,
        fit_direct_forward_invocations=53,learned_adapter_decisions=6,learned_adapter_simulations=128,
        corpus_game_counts=loaded['game_counts'],corpus_example_counts=loaded['example_counts'],fit=fit,
        qualification_only=True,pilot_data=False,strength_promotion=False,automatic_retries=0,game_replacements=0)
    verify_runtime(freeze,envelope);return result


def publish_verification(coverage, *, original_inventory, audit_freeze):
    c.require(c.file_hash(c.OUT/'freeze.json') == ORIGINAL_FREEZE_SHA, 'Exact original frozen qualification')
    c.require(c.read(c.OUT/'completion.json')['status'] == 'incomplete', 'Original incomplete attempt is preserved')
    return publish_new('recovered_verification.json', dict(
        status='saved_evidence_verification_recovered', original_attempt_status='incomplete',
        original_final_stage_completed=False, original_freeze=c.artifact(c.OUT/'freeze.json'),
        original_failure=c.artifact(c.OUT/'failure.json'), original_completion=c.artifact(c.OUT/'completion.json'),
        original_inventory=original_inventory, audit_freeze=audit_freeze, coverage=coverage,
        new_models=0, new_games=0, optimizer_updates=0, decision_retries=0,
        requires_independent_audit_acceptance_and_actual_owner_exit=True))
