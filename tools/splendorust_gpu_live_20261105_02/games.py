"""Eight matched-setup self-play games per backend, with full saved replay.

Only explicitly invoked runtime functions play games. Preparation and saved
audits are model-free. Outcomes describe this tiny throughput diagnostic only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import queue
import statistics
import threading
import time
import traceback

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import audit as ledger, rules
from tools.splendorust_search512_20261101 import scientific
from .qualification import ROOT, artifact, canonical, digest, ordinary, read, require, save, timed_call, validate_runtime

GAMES = 8
MAX_PLIES = 2000
PHASES = ((0,19), (20,39), (40,MAX_PLIES-1))
SEED_DOMAIN = "splendorust_gpu_live_20261105_matched_setup_throughput_v1"
PRIOR_SCHEDULES = (
    "outputs/splendorust_policy_head_20261104_01/assessment/schedule.json",
    "outputs/splendorust_selfplay_pilot_resume2_20261103_01/game_schedule.json",
    "outputs/splendorust_search512_20261101_01/schedule.json",
)


def seed_for(index):
    require(type(index) is int and 0 <= index < GAMES, "fixed eight setup indices")
    return str(int.from_bytes(hashlib.sha256(canonical([SEED_DOMAIN,index]).encode()).digest(), "big"))


def initial_snapshot(spec):
    require(spec == dict(game_id=f"g{spec['game_index']:03d}", game_index=spec["game_index"],
                         seed_decimal=seed_for(spec["game_index"]), start_player=spec["game_index"]%2,
                         quota=512, world_pool=3, cpuct=.4), "exact prospective game specification")
    state = rules.CorrectedEnv(seed=int(spec["seed_decimal"])).initial_state()
    state.start_player = state.current_player = spec["start_player"]
    result = rules.serialize_snapshot(state)
    guard.decode_training_snapshot(result)
    return result


def build_plan():
    specs = [dict(game_id=f"g{i:03d}",game_index=i,seed_decimal=seed_for(i),start_player=i%2,
                  quota=512,world_pool=3,cpuct=.4) for i in range(GAMES)]
    planned_seeds = {s["seed_decimal"] for s in specs}
    prior = []
    for relative in PRIOR_SCHEDULES:
        value = read(ROOT/relative)
        require(type(value) is list, "prior registered schedule format")
        old_seeds = {str(row["seed_decimal"]) for row in value}
        require(not planned_seeds & old_seeds, "fresh setup namespace disjoint registered predecessor schedules")
        prior.append(artifact(ROOT/relative))
    return dict(schema="gpu_live_eight_game_throughput_20261105_v1", specifications=specs,
                initial_states=[dict(game_id=s["game_id"],initial=initial_snapshot(s)) for s in specs],
                prior_schedules=prior, seed_domain=SEED_DOMAIN, route_order=["cpu","gpu"],
                games_per_route=GAMES, concurrency=GAMES, maximum_plies=MAX_PLIES,
                both_seats_same_backend=True, same_initial_states_across_routes=True,
                preserve_internal_seconds=9., preserve_outer_seconds=10.,
                one_second_is_descriptive=True, stop_wave_on_any_failure=True, replacements=0, retries=0,
                full_independent_transition_replay=True,
                model_replay_selection="first neural decision in each game in each fixed ply band 0..19,20..39,40+",
                model_replay_slot="(game_index+1)%8; other game clients inactive",
                model_replay_maximum_per_route=24, within_route_full_result_equality_required=True,
                throughput_includes="all concurrent gameplay, complete-response observations, transition checks and durable move/result writes",
                throughput_excludes="runtime/model/native startup, qualification, saved audits and selected model replays",
                strength_test=False, training_data=False)


def actual_after(before, action):
    state = guard.decode_training_snapshot(before)
    chosen = rules.deserialize_action(action)
    environment = rules.CorrectedEnv()
    require(chosen in environment.legal_actions(state), "complete corrected chosen action support")
    after = rules.serialize_snapshot(environment.step(state,chosen))
    require(after == ledger.transition(before,action), "actual transition matches independent dictionary ledger")
    guard.check_training_transition(before,action,after)
    return after


def validate_move(row, spec, route, expected_before, ply):
    require(row["game_id"] == spec["game_id"] and row["route"] == route and row["ply"] == ply and
            row["before"] == expected_before, "contiguous same-game route and snapshot")
    state = guard.decode_training_snapshot(row["before"])
    observer = rules.observation_payload(state)
    require(row["observer"] == observer and row["actor"] == state.current_player,
            "lawful public observer and acting seat")
    clock = row["clock"]
    require(clock["status"] == "ok" and clock["completed"] is True and
            clock["outer_seconds"] == 10. and clock["internal_seconds"] == 9. and
            0 <= clock["elapsed_seconds"] < 10. and clock["no_retry"] is True,
            "accepted complete response within unchanged guard")
    scientific.validate_result(row["result"],observer,"search",expected_quota=512)
    scientific.audit_decision(observer,row["result"]["action"],row["result"]["diagnostics"],"search",
                              before=row["before"],after=row["after"],expected_quota=512)
    require(row["after"] == actual_after(row["before"],row["result"]["action"]),
            "full independent transition reconstruction")
    return row["after"]


def replay_selection(rows):
    selected = []
    for lower,upper in PHASES:
        matching = [row for row in rows if lower <= row["ply"] <= upper and
                    row["result"]["diagnostics"]["inference_calls"] > 0]
        if matching:
            selected.append(dict(ply=matching[0]["ply"],band=[lower,upper]))
    return selected


def _play_one(plan, runtime, folder, spec, adapter, client_id, start_barrier, cancelled, boundary):
    initial = initial_snapshot(spec)
    save(folder/"spec.json",spec); save(folder/"initial.json",initial)
    start_barrier.wait(timeout=10.)
    started = time.perf_counter(); before = initial; rows = []; previous_pass = False
    for ply in range(MAX_PLIES):
        if cancelled.is_set():
            raise RuntimeError("Another scheduled game failed; no replacement or continuation")
        boundary()
        state = guard.decode_training_snapshot(before)
        if state.terminal:
            break
        observer = rules.observation_payload(state)
        request = dict(player="search",observer=observer)
        clock_path = folder/"clocks"/f"{ply:06d}.json"
        value, clock = timed_call(lambda started:ordinary(runtime,{"search":adapter},request,started=started,
                                                        client_ids={"search":client_id},
                                                        evidence_callback=lambda raw:save(folder/"raw_results"/f"{ply:06d}.json",raw)),
                                  abort=runtime.abort,on_receipt=lambda receipt:save(clock_path,receipt))
        after = actual_after(before,value["action"])
        row = dict(game_id=spec["game_id"],route=runtime.route,ply=ply,actor=state.current_player,
                   before=before,observer=observer,result=value,clock=clock,after=after)
        save(folder/"moves"/f"{ply:06d}.json",row)
        validate_move(row,spec,runtime.route,before,ply)
        rows.append(row)
        is_pass = value["action"]["action_type"] == "PASS"
        require(after != before and not (previous_pass and is_pass),
                "actual deadlock remains incomplete; no synthetic terminal target")
        previous_pass = is_pass; before = after
    final = guard.decode_training_snapshot(before)
    require(final.terminal and bool(rows), "every throughput game finishes naturally before fixed safety cap")
    winner = guard.require_official_terminal_target(before,"completed")
    summary = dict(status="naturally_completed",route=runtime.route,game_id=spec["game_id"],
                   specification=spec,plies=len(rows),initial_sha256=digest(initial),final=before,winner=winner,
                   gameplay_seconds=time.perf_counter()-started,
                   complete_response_seconds=[row["clock"]["elapsed_seconds"] for row in rows],
                   inference_calls=sum(row["result"]["diagnostics"]["inference_calls"] for row in rows),
                   simulations=sum(phase["simulations"] for row in rows for phase in row["result"]["diagnostics"]["phases"]),
                   model_replay_selection=replay_selection(rows),training_data=False)
    save(folder/"result.json",summary)
    return summary


def audit_games(plan, folder):
    """Saved-only verification of every actual game transition, with no models."""
    folder = Path(folder)
    wave = read(folder/"wave.json"); route = wave["route"]
    selected = plan["games"]
    require(selected == build_plan() and route in ("cpu","gpu"), "fixed game plan and route")
    games, moves = [], []
    for spec in selected["specifications"]:
        game_folder = folder/spec["game_id"]
        initial = initial_snapshot(spec)
        require(read(game_folder/"spec.json") == spec and read(game_folder/"initial.json") == initial,
                "prospective corrected initial state")
        ending = read(game_folder/"result.json")
        require(ending["status"] == "naturally_completed" and ending["specification"] == spec and
                ending["route"] == route and ending["game_id"] == spec["game_id"], "natural exact scheduled game")
        paths = sorted((game_folder/"moves").glob("*.json"))
        require([path.name for path in paths] == [f"{i:06d}.json" for i in range(ending["plies"])],
                "complete exact move inventory")
        rows = []
        before = initial
        for ply,path in enumerate(paths):
            row = read(path)
            require(row["clock"] == read(game_folder/"clocks"/f"{ply:06d}.json"), "independently retained response clock")
            raw=read(game_folder/"raw_results"/f"{ply:06d}.json")
            require(raw["result"]==row["result"] and raw["observer"]==row["observer"], "prevalidation raw complete game result")
            before = validate_move(row,spec,route,before,ply); rows.append(row)
        winner = guard.require_official_terminal_target(before,"completed")
        require(ending["final"] == before and ending["winner"] == winner and
                ending["initial_sha256"] == digest(initial) and ending["model_replay_selection"] == replay_selection(rows) and
                ending["complete_response_seconds"] == [row["clock"]["elapsed_seconds"] for row in rows],
                "saved final target, timing and blind replay inventory")
        require(ending["inference_calls"] == sum(row["result"]["diagnostics"]["inference_calls"] for row in rows) and
                ending["simulations"] == sum(phase["simulations"] for row in rows for phase in row["result"]["diagnostics"]["phases"]),
                "all actual game neural calls and simulations reconstructed")
        games.append(ending); moves.extend(rows)
    require(wave["game_ids"] == [x["game_id"] for x in selected["specifications"]] and
            wave["finished_games"] == len(games) == GAMES and wave["concurrency"] == GAMES and
            wave["total_plies"] == sum(x["plies"] for x in games), "all eight scheduled concurrent games")
    elapsed = wave["wall_seconds"]
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed > 0 and
            elapsed == wave["ended_perf_counter"]-wave["started_perf_counter"] and
            wave["games_per_hour"] == GAMES*3600/elapsed and
            wave["decisions_per_second"] == len(moves)/elapsed, "raw wall-clock throughput reconstruction")
    durations = [row["clock"]["elapsed_seconds"] for row in moves]
    require(wave["mean_response_seconds"] == statistics.fmean(durations) and
            wave["p50_response_seconds"] == statistics.median(durations) and
            wave["p95_response_seconds"] == percentile(durations,.95) and
            wave["maximum_response_seconds"] == max(durations) and
            wave["responses_over_one_second"] == sum(x>1. for x in durations), "complete-response descriptive summaries")
    require(wave["inference_calls"] == sum(x["inference_calls"] for x in games) and
            wave["simulations"] == sum(x["simulations"] for x in games) and
            wave["backend_counter_end"]["completed_requests"]-wave["backend_counter_start"]["completed_requests"] == wave["inference_calls"],
            "gameplay-only backend service callback accounting")
    return dict(status="all_saved_transitions_independently_replayed",route=route,games=GAMES,
                transitions=len(moves),wave_sha256=digest(wave),
                selected_model_replays=sum(len(x["model_replay_selection"]) for x in games),
                model_calls=0,training_data=False)


def replay_games(plan, runtime, folder, boundary=lambda:None):
    """Reproduce chosen fixed-ply decisions with different slots/companions."""
    folder = Path(folder); receipts = []
    for spec in plan["games"]["specifications"]:
        ending = read(folder/spec["game_id"]/"result.json")
        client_id = "replay_"+spec["game_id"]
        slot = (spec["game_index"]+1)%GAMES
        adapter = runtime.make_adapter(client_id,slot,512)
        try:
            for selected in ending["model_replay_selection"]:
                boundary(); source = read(folder/spec["game_id"]/"moves"/f"{selected['ply']:06d}.json")
                request = dict(player="search",observer=source["observer"])
                value, clock = timed_call(lambda started:ordinary(runtime,{"search":adapter},request,started=started,
                                                                client_ids={"search":client_id}),abort=runtime.abort,
                                          on_receipt=lambda receipt:save(folder/"model_replay_clocks"/f"{spec['game_id']}_{selected['ply']:06d}.json",receipt))
                receipt = dict(game_id=spec["game_id"],ply=selected["ply"],route=runtime.route,
                               original_slot=spec["game_index"],replay_slot=slot,other_game_clients_inactive=True,
                               result=value,source_result_sha256=digest(source["result"]),
                               exact_full_scientific_result=canonical(value) == canonical(source["result"]),clock=clock)
                save(folder/"model_replays"/f"{spec['game_id']}_{selected['ply']:06d}.json",receipt)
                require(receipt["exact_full_scientific_result"],
                        "within-route complete action/science/callbacks differ with changed slot/companions")
                receipts.append(receipt)
        finally:
            adapter.close()
    require(0 < len(receipts) <= 24, "finite prospective selected model replays")
    result = dict(status="exact_within_route_replays_passed",route=runtime.route,
                  count=len(receipts),receipts_sha256=digest(receipts),changed_slots=True,changed_companions=True)
    save(folder/"model_replay_summary.json",result)
    save(folder/"model_replay_saved_audit.json",verify_model_replays(plan,folder))
    return result


def verify_model_replays(plan,folder):
    folder=Path(folder); rows=[]; expected_paths=[]
    for spec in plan["games"]["specifications"]:
        ending=read(folder/spec["game_id"]/"result.json")
        for selected in ending["model_replay_selection"]:
            name=f"{spec['game_id']}_{selected['ply']:06d}.json"; expected_paths.append(name)
            row=read(folder/"model_replays"/name)
            source=read(folder/spec["game_id"]/"moves"/f"{selected['ply']:06d}.json")
            require(row["game_id"]==spec["game_id"] and row["ply"]==selected["ply"] and row["route"]==ending["route"] and
                    row["original_slot"]==spec["game_index"] and row["replay_slot"]==(spec["game_index"]+1)%8 and
                    row["other_game_clients_inactive"] is True and row["exact_full_scientific_result"] is True and
                    row["source_result_sha256"]==digest(source["result"]) and canonical(row["result"])==canonical(source["result"]),
                    "exact same-route replay under changed slot and companions")
            require(row["clock"]==read(folder/"model_replay_clocks"/name) and row["clock"]["status"]=="ok" and
                    row["clock"]["completed"] is True and 0<=row["clock"]["elapsed_seconds"]<10., "replay actual response clock")
            scientific.validate_result(row["result"],source["observer"],"search",expected_quota=512)
            rows.append(row)
    require([path.name for path in sorted((folder/"model_replays").glob("*.json"))]==sorted(expected_paths),
            "exact prospective replay inventory; no dropped or replaced replay")
    summary=read(folder/"model_replay_summary.json")
    require(summary["status"]=="exact_within_route_replays_passed" and summary["count"]==len(rows) and
            0<len(rows)<=24 and summary["receipts_sha256"]==digest(rows), "saved replay summary binds all exact outputs")
    return dict(status="saved_exact_replays_verified",route=summary["route"],count=len(rows),model_calls=0,
                changed_slots=True,changed_companions=True)


def run_games(plan, runtime, folder, boundary=lambda:None):
    validate_runtime(runtime)
    require(plan["games"] == build_plan(), "fixed eight-game plan")
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=False)
    specs = plan["games"]["specifications"]
    adapters = {}; threads = []; results = queue.Queue(); cancelled = threading.Event()
    started_values = {}
    def begin_wave():
        started_values.update(perf=time.perf_counter(),unix=time.time())
    barrier = threading.Barrier(GAMES+1,action=begin_wave)
    try:
        for spec in specs:
            adapters[spec["game_id"]] = runtime.make_adapter("game_"+spec["game_id"],spec["game_index"],512)
        def launch(spec):
            try:
                summary = _play_one(plan,runtime,folder/spec["game_id"],spec,adapters[spec["game_id"]],
                                    "game_"+spec["game_id"],barrier,cancelled,boundary)
                results.put((True,summary))
            except BaseException:
                cancelled.set()
                try:
                    save(folder/spec["game_id"]/"failure.json",dict(error=traceback.format_exc(),no_retry=True,
                                                                  game_id=spec["game_id"],route=runtime.route))
                finally:
                    results.put((False,spec["game_id"]))
        for spec in specs:
            thread = threading.Thread(target=launch,args=(spec,),daemon=True,name="gpu-live-"+spec["game_id"])
            thread.start(); threads.append(thread)
        with runtime.service.lock:
            counter_start = dict(batch_count=runtime.service.batch_count,completed_requests=runtime.service.completed_requests)
        barrier.wait(timeout=10.)
        started, started_wall = started_values["perf"],started_values["unix"]
        completed = []
        while len(completed) < GAMES:
            boundary()
            try:
                ok,value = results.get(timeout=.1)
            except queue.Empty:
                continue
            if not ok:
                runtime.abort()
                raise RuntimeError("Scheduled concurrent game failed: "+value)
            completed.append(value)
        ended = time.perf_counter(); elapsed = ended-started
        ordered = sorted(completed,key=lambda x:x["specification"]["game_index"])
        durations = [x for game in ordered for x in game["complete_response_seconds"]]
        with runtime.service.lock:
            counter_end = dict(batch_count=runtime.service.batch_count,completed_requests=runtime.service.completed_requests)
        wave = dict(status="eight_natural_games_completed",route=runtime.route,game_ids=[s["game_id"] for s in specs],
                    concurrency=GAMES,finished_games=GAMES,total_plies=sum(x["plies"] for x in ordered),
                    started_perf_counter=started,ended_perf_counter=ended,started_unix=started_wall,ended_unix=time.time(),
                    wall_seconds=elapsed,games_per_hour=GAMES*3600/elapsed,decisions_per_second=len(durations)/elapsed,
                    mean_response_seconds=statistics.fmean(durations),maximum_response_seconds=max(durations),
                    p50_response_seconds=statistics.median(durations),p95_response_seconds=percentile(durations,.95),
                    responses_over_one_second=sum(x>1. for x in durations),one_second_descriptive=True,
                    inference_calls=sum(x["inference_calls"] for x in ordered),simulations=sum(x["simulations"] for x in ordered),
                    backend_counter_start=counter_start,backend_counter_end=counter_end,
                    natural_results=[dict(game_id=x["game_id"],winner=x["winner"],plies=x["plies"]) for x in ordered],
                    strength_claim=False,training_data=False)
        save(folder/"wave.json",wave)
    except BaseException:
        cancelled.set(); runtime.abort()
        save(folder/"failure.json",dict(error=traceback.format_exc(),no_retry=True,route=runtime.route))
        raise
    finally:
        for adapter in adapters.values():
            adapter.close()
        for thread in threads:
            thread.join(timeout=.1)
    audit = audit_games(plan,folder); save(folder/"saved_transition_audit.json",audit)
    replay = replay_games(plan,runtime,folder,boundary)
    return dict(wave=wave,audit=audit,replay=replay)


def percentile(values,fraction):
    require(bool(values) and 0 <= fraction <= 1, "nonempty percentile sample")
    ordered = sorted(values); index = (len(ordered)-1)*fraction
    low,high = math.floor(index),math.ceil(index)
    return ordered[low]+(ordered[high]-ordered[low])*(index-low)


def compare_game_rates(cpu, gpu):
    left,right = cpu["wave"],gpu["wave"]
    require(left["route"] == "cpu" and right["route"] == "gpu" and left["game_ids"] == right["game_ids"] and
            left["finished_games"] == right["finished_games"] == GAMES and left["concurrency"] == right["concurrency"] == GAMES,
            "same eight initial setups and concurrency")
    require(left["ended_unix"] <= right["started_unix"], "CPU/GPU gameplay waves never overlap")
    return dict(status="bounded_same_setup_throughput_complete",games_each=GAMES,concurrency=GAMES,
                cpu_wall_seconds=left["wall_seconds"],gpu_wall_seconds=right["wall_seconds"],
                cpu_over_gpu_wall_ratio=left["wall_seconds"]/right["wall_seconds"],
                cpu_games_per_hour=left["games_per_hour"],gpu_games_per_hour=right["games_per_hour"],
                cpu_decisions_per_second=left["decisions_per_second"],gpu_decisions_per_second=right["decisions_per_second"],
                routes_may_have_different_trajectories=True,game_pair_results_are_not_strength_comparisons=True,
                startup_and_replay_excluded=True,training_data=False,larger_collection_authorized=False)
