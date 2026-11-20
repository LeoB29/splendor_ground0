"""Independent live replay for preselected256 candidate and unchanged128 reference.

Both players retain the original scientific arm ``search``. Matchup roles are
separate labels and never imply the old PUCT16 ``reference`` scientific arm.
No model, native process or environment transition is invoked by this module.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from itertools import product
import json
import math

from tools.external_strength_screen_20261018 import audit as ledger
from tools.external_strength_screen_20261018.common import digest
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.splendorust_search_scale_20261024 import audit as original
from .schedule import validate_spec, validate_schedule, make_schedule
from tools.splendorust_boundary_timing_20261025 import analyze as boundaries

ROLES = ("challenger", "opponent")
QUOTAS = {"challenger": 256, "opponent": 128}
PLACEMENT_MODE = "isolated"
AUDIT_DOMAIN = "splendorust_search256_strength_20261029_02_audit_v1"
from tools.splendorust_sampling_stability_20261027 import audit as saved_science
from . import common as c
from .scientific import check_clock, classify_failure
from .adapters import validate_result
from .performance import performance, boundary_row, validate_boundary_row, summarize_boundary_rows


validate_research_result = saved_science.validate_research_result


def decode_interrupted_gzip(path, descriptor, *, maximum=128*1024**2, boundary=lambda:None):
    """Read preserved available bytes without inventing a gzip trailer/closure."""
    import zlib
    from pathlib import Path
    path=Path(path);boundary();c.verify([descriptor])
    require(c.artifact(path)==descriptor and path.stat().st_size<=maximum,'Bounded exact interrupted stream')
    stream=zlib.decompressobj(31);raw=bytearray()
    with path.open('rb') as handle:
        while chunk:=handle.read(65536):
            boundary();raw.extend(stream.decompress(chunk,maximum-len(raw)+1))
            require(len(raw)<=maximum and not stream.unconsumed_tail,'Interrupted decoded stream bound')
            require(not stream.unused_data,'No extra gzip members or unknown trailing bytes')
    boundary();require(c.artifact(path)==descriptor,'Interrupted source changed during read')
    return bytes(raw),dict(source=descriptor,decoded_bytes=len(raw),decoded_sha256=hashlib.sha256(raw).hexdigest(),
        gzip_complete=stream.eof,original_stream_rewritten=False,normal_closure_proven=False)


def validate_prefix_chain(initial,rows,spec,*,boundary=lambda:None):
    """Independent30-transition seam proof; never accepts the unresolved move."""
    _spec_identity(spec);require(spec['stage']=='main' and spec['game_index']==235,'Only authorized interrupted game')
    require(type(rows) is list and len(rows)==30,'Exactly30 preserved accepted moves')
    current=deepcopy(initial);decode_training_snapshot(current)
    for ply,row in enumerate(rows):
        boundary();require(not current['state']['terminal'] and row['ply']==ply and row['before']==current,'Exact contiguous prefix')
        require(all(row[k]==spec[k] for k in ('stage','candidate','pair_index','game_index')),'Prefix schedule identity')
        seat=current['state']['current_player'];role='challenger' if seat==spec['challenger_seat'] else 'opponent'
        require(row['seat']==seat and row['player_role']==role and row['mode']==modes_for(spec['candidate'])[role] and
            row['quota']==QUOTAS[role] and row['arm']=='search','Prefix role/quota identity')
        require(row['completed'] is True and row['clock']['status']=='ok','All retained prefix moves were accepted')
        ledger.check_observer(current,row['observer']);current=ledger.transition(current,row['action'])
        require(row['after']==current,'Independent preserved prefix transition');decode_training_snapshot(current)
    require(not current['state']['terminal'] and current['state']['current_player']==spec['challenger_seat'],
        'Authorized unresolved ply30 belongs to challenger256')
    return current


def validate_segment_seam(rows,prefix,spec):
    require(spec['stage']=='main' and spec['game_index']==235 and len(prefix)==30 and len(rows)>30,
        'Exactly one authorized continuation with retained30moves')
    require(all(_same(a,b) for a,b in zip(rows[:30],prefix,strict=True)),'Preserved prefix cannot change or be discarded')
    first=rows[30]
    require(first['ply']==30 and first['before']==prefix[-1]['after'] and first['player_role']=='challenger' and
        first['route']==first['mode']=='research' and first['quota']==256 and first['clock']['request_id']=='request-00000003',
        'Exact disclosed fresh-clock retry seam after graph and suite')
    return dict(status='passed',retained_moves=30,original_last_ply=29,continuation_first_ply=30,
        authorized_fresh_move_clocks=1,unresolved_original_clock_rewritten=False)


def verify_interrupted_worker(folder,freeze,inventory,*,route,boundary=lambda:None):
    """Authenticate successful packets while retaining missing closure as a failure."""
    from pathlib import Path
    import psutil
    from tools.splendorust_search256_strength_20261029_02 import transport as oldtransport,adapters as oldadapters
    folder=Path(folder);expected_name='main_research_g235_'+('challenger' if route=='research' else 'opponent')
    require(route in ('research','frozen') and folder.name==expected_name,'Exact interrupted worker identity')
    require(not(folder/'cleanup.json').exists(),'Interrupted worker must not acquire normal cleanup')
    def pin(path):
        item=inventory[path.resolve().relative_to(c.ROOT).as_posix()];boundary();c.verify([item]);return item
    def read(path):pin(path);return c.read(path)
    init=read(folder/'initialization.json');oldadapters.verify_saved_ready(folder,freeze,route=route)
    require(read(folder/'route_identity.json')==dict(route=route,module=oldtransport.OBSERVER_MODULE,
        native_artifacts=freeze['native_artifacts_by_route'][route],native_roles=freeze['native_roles_by_route'][route],metadata=init['result']),
        'Interrupted exact module/native/READY identity')
    for identity in init['telemetry']['process_tree']:
        boundary()
        try:present=psutil.Process(identity['pid']).create_time()==identity['created']
        except psutil.NoSuchProcess:present=False
        require(not present,'Interrupted worker/helper identity remains active')
    ids=[f'request-{i:08d}' for i in range(1,17)]
    require([p.stem for p in sorted(folder.glob('request-*.json')) if oldtransport.REQUEST_ID.fullmatch(p.stem)]==ids,
        'Exactly16 accepted original request receipts')
    receipts={rid:read(folder/(rid+'.json')) for rid in ids}
    require(all(x['status']=='ok' and x['accepted'] is True for x in receipts.values()),'No rejected prefix receipt accepted')
    ordinary=[f'request-{i:08d}' for i in range(2,18 if route=='research' else 17)]
    require(sorted(p.stem for p in(folder/'inputs').glob('*.json'))==ordinary,'Exact accepted inputs plus one unresolved staged challenger input')
    require(sorted(p.stem for p in(folder/'startup_inputs').glob('*.json'))==ids[:1],'One original graph startup')
    require(sorted(p.stem for p in(folder/'parent_profiles').glob('*.json'))==ids and
        sorted(p.stem for p in(folder/'placement_requests').glob('*.json'))==ids and
        sorted(p.name for p in folder.glob('request-*_child_profile.json'))==[rid+'_child_profile.json' for rid in ids],
        'No missing accepted or invented unresolved sidecar')
    decoded={};streams=[]
    for name in ('stdout.log','events.jsonl'):
        path=folder/(name+'.gz');raw,proof=decode_interrupted_gzip(path,pin(path),boundary=boundary)
        require(not proof['gzip_complete'] and raw.endswith(b'\n'),'Preserved unclosed stream has only complete decoded records')
        decoded[name]=raw.splitlines(keepends=True);streams.append(dict(name=name,**proof))
    require(len(decoded['stdout.log'])==17 and len(decoded['events.jsonl'])==16,'READY and exact16 original responses/events')
    require(json.loads(decoded['stdout.log'][0])==dict(type='ready',metadata=init['result']),'Interrupted raw READY')
    previous=None
    for index,rid in enumerate(ids):
        boundary();receipt=receipts[rid];raw=decoded['stdout.log'][index+1]
        require(len(raw)<=4*c.MiB and raw.endswith(b'\n'),'Bounded complete preserved packet')
        packet=oldtransport._packet_evidence(raw)
        require(packet['packet']==dict(type='response',request_id=rid,result=receipt['result']),'Preserved packet/trusted result equality')
        require(json.loads(decoded['events.jsonl'][index])=={k:v for k,v in receipt.items() if k!='result'},'Preserved event/trusted receipt equality')
        payload=read(folder/('startup_inputs' if index==0 else 'inputs')/(rid+'.json'))
        parent=read(folder/'parent_profiles'/(rid+'.json'));child=read(folder/(rid+'_child_profile.json'))
        require(oldtransport._bind_parent_wire(parent,[packet],completed=True),'Trusted parent raw wire required')
        oldtransport._bind_child_wire(child,[packet],receipt)
        oldtransport.validate_profiles(parent,child,receipt,payload,route,worker_pid=init['identity']['pid'])
        oldtransport.verify_placement_request(folder,rid,freeze,parent)
        if previous is not None:require(previous<parent['request_start_ns'],'Nonoverlapping preserved request chronology')
        previous=parent['capture_snapshot_ns']
    stderr=folder/'stderr.log';require(stderr.exists() and pin(stderr)['bytes']==0,'Original empty unclosed stderr preserved')
    graph=oldtransport.verify_graph_evidence(folder,freeze,boundary=boundary)
    native=oldadapters.verify_native_configuration_receipts(folder,route,boundary=boundary)
    return dict(status='verified_accepted_prefix_with_preserved_interrupted_closure',worker=folder.name,route=route,
        requests=16,ordinary_requests=15,graph=graph,native_configuration=native,streams=streams,
        unresolved_staged_request='request-00000017' if route=='research' else None,
        unresolved_dispatch_known=False if route=='research' else None,normal_closure_proven=False,
        actual_process_exit_code=None,original_attempt_accepted=False,original_logs_or_receipts_rewritten=False)






def require(value, message):
    if not value:
        raise ValueError("SEARCH256_AUDIT: " + message)


def _same(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == \
        json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_isolated(row):
    require(row.get('placement_mode') == PLACEMENT_MODE, 'qualified isolated supervision required')
    require('placement_error' not in row and 'profile_capture_error' not in row,
            'placement/observer capture failure is an infrastructure error')


def _spec_identity(spec):
    validate_spec(spec)
    require(type(spec) is dict and spec.get("stage") in ("main", "operational"), "game stage")
    require(type(spec.get("challenger_seat")) is int and spec["challenger_seat"] in (0, 1), "challenger seat")
    for role, quota in QUOTAS.items():
        require(type(spec.get(role+"_quota")) is int and spec[role+"_quota"] == quota, "externally frozen quota")
    for key in ("pair_index", "game_index"):
        require(type(spec.get(key)) is int and spec[key] >= 0, "game schedule identity")


def modes_for(candidate):
    require(candidate == 'research', 'exact candidate route')
    return {'challenger': candidate, 'opponent': 'frozen'}


def audit_game(initial_snapshot, rows, ending, spec, *, freeze, boundary=lambda: None):
    """Replay every actual move with the dictionary ledger; never call env.step.

    The runner additionally binds the initial snapshot to the frozen seed and
    each row's clock/input files. This checker binds all subsequent transitions.
    """
    _spec_identity(spec)
    require(all(key in ending and _same(ending[key], value) for key, value in spec.items()), "ending schedule binding")
    require(ending.get("complete") is True and type(ending.get("plies")) is int and ending["plies"] == len(rows),
            "complete game move count")
    require(type(rows) is list and 0 < len(rows) <= 2000, "bounded nonempty move inventory")
    decode_training_snapshot(initial_snapshot)
    modes = modes_for(spec['candidate'])
    identities={r:profile_identity(freeze,c.ROUTE_QUOTAS[r],3,route=r,boundary=boundary)
                for r in ('frozen',spec['candidate'])}
    current = deepcopy(initial_snapshot)
    failure, transitions = None, 0
    latency = {role: [] for role in ROLES}
    for ply, row in enumerate(rows):
        boundary()
        require(failure is None and not current["state"]["terminal"], "moves after final ending")
        require(type(row.get("ply")) is int and row["ply"] == ply and _same(row.get("before"), current),
                "noncontiguous move evidence")
        require(all(key in row and _same(row[key], spec[key]) for key in
                    ("stage", "candidate", "pair_index", "game_index")),
                "move schedule binding")
        seat = current["state"]["current_player"]
        role = "challenger" if seat == spec["challenger_seat"] else "opponent"
        quota = QUOTAS[role]
        require(type(row.get("seat")) is int and row["seat"] == seat and row.get("player_role") == role and
                row.get("arm") == "search" and row.get("mode") == modes[role] and
                type(row.get("quota")) is int and row["quota"] == quota, "acting role/scientific arm/quota")
        ledger.check_observer(current, row["observer"])
        receipt = row["clock"]
        received = check_clock(receipt)
        require(receipt["deadline_seconds"] == c.SECONDS_PER_MOVE, "full research move allowance required; stage clamp is not a loss")
        validate_boundary_row(row)
        if receipt["status"] == "ok":
            require(row.get("completed") is True and type(row.get("after")) is dict,
                    "completed move requires a concrete successor snapshot")
            validate_result(receipt["result"], row["observer"], modes[role], identity=identities[modes[role]])
            require(_same(row["action"], receipt["result"]["action"]) and
                    _same(row["trace"], receipt["result"]["diagnostics"]), "move/response evidence coupling")
            successor = ledger.transition(current, row["action"])
            require(_same(successor, row["after"]), "independent dictionary transition differs")
            decode_training_snapshot(successor)
            current = successor
            latency[role].append(received)
            transitions += 1
        else:
            require(row.get("completed") is False and row.get("action") is None and row.get("trace") is None and
                    row.get("after") is None, "failed move must not invent a successor or played action")
            failure = {"seat": seat, "role": role, "status": classify_failure(receipt)}
    boundary()
    require(type(ending.get("final")) is dict and _same(ending["final"], current), "saved final state")
    if failure is None:
        require(ending.get("reason") == "natural" and current["state"]["terminal"] and
                _same(ending.get("winner"), current["state"]["winner"]) and ending.get("loser") is None and
                ending.get("failed_role") is None and ending.get("failed_seat") is None and
                not ending.get("administrative_forfeit",False) and ending.get("failure_policy") is None and ending.get("failure") is None,
                "official natural ending")
    else:
        require(ending.get("reason") == "forfeit" and ending.get("administrative_forfeit") is True and
                ending.get("failure_policy")=="research_hard_guard_9s_internal_10s_complete_response" and not current["state"]["terminal"] and
                type(ending.get("loser")) is int and ending["loser"] == failure["seat"] and
                type(ending.get("winner")) is int and ending["winner"] == 1-failure["seat"] and
                ending.get("failed_role") == failure["role"] and
                type(ending.get("failed_seat")) is int and ending["failed_seat"] == failure["seat"],
                "forfeit attribution")
        saved_failure = ending.get("failure")
        require(saved_failure == failure["status"] or
                type(saved_failure) is dict and saved_failure.get("status") == failure["status"],
                "forfeit classifier identity")
    return {"status": "passed", "transitions": transitions, "failed_moves": int(failure is not None),
            "final_sha256": digest(current), "latencies": latency,
            "transition_oracle": "frozen independent dictionary ledger; no env.step",
            "scientific_arms": {role: {"arm": "search", "quota": quota, "mode": modes[role]} for role, quota in QUOTAS.items()},
            "trust_boundary": "shared corrected catalogue/atomic bridge and pinned native numerical search; initial seed and raw clock binding checked by runner"}


def select_replay_sample(rows, maximum=48, *, stage, candidate, allow_operational=False):
    """Fixed-hash minimum per role/seat/phase/private-reserve cell, at most48."""
    require(type(maximum) is int and maximum == 48, "frozen48 replay-cell bound")
    require(type(allow_operational) is bool, "explicit stopped-prefix selector mode")
    require(stage in (('operational','main') if allow_operational else ('main',)), 'approved replay sample stage')
    modes = modes_for(candidate)
    selected, seen = {}, set()
    for row in rows:
        require(row.get('stage') == stage and row.get('candidate') == candidate,
                'replay caller must supply exactly one stage/candidate')
        _require_isolated(row)
        require(type(row.get("completed")) is bool, "explicit completed flag")
        if not row["completed"]:
            continue
        pair, game, ply, role, seat = (row[key] for key in ("pair_index", "game_index", "ply", "player_role", "seat"))
        from .schedule import PAIRS
        pair_limit = PAIRS[stage]
        game_base = 2*pair
        require(all(type(value) is int for value in (pair, game, ply, seat)) and
                0 <= pair < pair_limit and game in (game_base,game_base+1) and 0 <= ply < 2000 and seat in (0, 1) and
                role in ROLES and row.get("arm") == "search" and row.get("mode") == modes[role] and
                type(row.get("quota")) is int and row["quota"] == QUOTAS[role], "sample role/quota identity")
        identity = stage, candidate, pair, game, ply
        require(identity not in seen, "duplicate completed decision")
        seen.add(identity)
        view = row["observer"]["observation"]
        require(view["current_player"] == seat, "sample observation seat")
        private = [any(reserve["source"] == "deck" for reserve in view["players"][actor]["reserves"])
                   for actor in (seat, 1-seat)]
        category = "both" if all(private) else "own" if private[0] else "opponent" if private[1] else "neither"
        phase = "0..19" if ply < 20 else "20..39" if ply < 40 else "40+"
        cell = role, phase, seat, category
        encoded = json.dumps([AUDIT_DOMAIN, stage, candidate, pair, game, ply, role], separators=(",", ":"), ensure_ascii=True)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        item = {"stage": stage, "candidate": candidate, "pair_index": pair, "game_index": game, "ply": ply, "player_role": role,
                "arm": "search", "quota": QUOTAS[role], "mode": modes[role], "seat": seat,
                "placement_mode": PLACEMENT_MODE,
                "phase": phase, "private_category": category, "selection_sha256": key,
                "observer_sha256": digest(row["observer"])}
        if cell not in selected or key < selected[cell]["selection_sha256"]:
            selected[cell] = item
    cells = list(product(ROLES, ("0..19", "20..39", "40+"), (0, 1), ("neither", "own", "opponent", "both")))
    return {"schema": "splendorust_search256_replay_sample_v1", "domain": AUDIT_DOMAIN,
            "stage": stage, "candidate": candidate,
            "placement_mode": PLACEMENT_MODE,
            "maximum_decisions": maximum, "decisions": [selected[cell] for cell in cells if cell in selected],
            "empty_cells": [{"player_role": role, "phase": phase, "seat": seat, "private_category": private}
                            for role, phase, seat, private in cells if (role, phase, seat, private) not in selected],
            "selection_uses_outcomes": False, "selection_uses_timing": False,
            "scope": "all_completed_played_prefix_cells" if allow_operational else "fixed_stage_candidate_completed_cells",
            "failed_moves_excluded": True, "all_available_cells_selected": True}






summarise_performance = performance






def validate_stopped_prefix(schedule,game_results,stop=None):
    validate_schedule(schedule)
    require(type(game_results) is list and len(game_results)<=len(schedule),'Bounded scheduled prefix')
    for index,game in enumerate(game_results):
        require(all(_same(game.get(k),v) for k,v in schedule[index].items()),'Exact scheduled prefix')
        require(game.get('complete') is True and game.get('reason') in ('natural','forfeit') and
            type(game.get('plies')) is int and 0<game['plies']<=2000,'Official ending')
        if game['stage']=='operational':require(game['reason']!='forfeit' or index==len(game_results)-1,'Ops stop first player fault')
    complete=len(game_results)==len(schedule)
    if stop is not None:
        require(game_results and game_results[-1]['stage']=='operational' and game_results[-1]['reason']=='forfeit','Only ops fault stops collection')
        last=game_results[-1]
        require(all(stop.get(k)==last[k] for k in ('stage','candidate','pair_index','game_index')) and
            stop['ply']==last['plies']-1 and stop['seat']==last['loser'] and stop['player_role']==last['failed_role'],
            'Stop exact failed game/ply/role')
        require(classify_failure(stop['clock'])==stop['failure_classification'],'Preserved fault classification')
    return dict(status='full_workload_verified' if complete else 'verified_prefix',planned_games=len(schedule),
        verified_games=len(game_results),unplayed_games=len(schedule)-len(game_results),
        complete_planned_workload=complete,forfeits=sum(g['reason']=='forfeit' for g in game_results),
        continuation_after_forfeit='main_only',no_strength_claim=not complete)




def profile_identity(freeze, quota, world_pool=3, *, route=None, boundary=lambda:None):
    from . import adapters
    require(route in ('research','frozen'), 'explicit256/128 route required')
    require(quota==c.ROUTE_QUOTAS[route] and world_pool==3,'Fixed three-world numerical player')
    boundary()
    return adapters.profile_identity(freeze,route,quota=quota)
