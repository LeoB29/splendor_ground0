"""Eight fresh matched games in fixed waves, with durable evidence and replays.

Imports, planning and saved verification are model-free. Model calls occur only
in explicitly invoked run_games/replay_games through the qualified new handler.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import queue
import statistics
import threading
import time
import traceback

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import audit as ledger, rules
from . import common as c

GAMES = 8
MAX_PLIES = 2000
GAMEPLAY_SECONDS = 2*3600
PHASES = ((0,19), (20,39), (40,MAX_PLIES-1))
SEED_DOMAIN = "splendorust_gpu_throughput_20261105_fresh_matched_games_v1"
PRIOR_SCHEDULES = (
    "outputs/splendorust_policy_head_20261104_01/assessment/schedule.json",
    "outputs/splendorust_selfplay_pilot_resume2_20261103_01/game_schedule.json",
    "outputs/splendorust_search512_20261101_01/schedule.json",
)
PRIOR_GAME_PLAN = "outputs/splendorust_gpu_live_20261105_03/plan.json"


def seed_for(index):
    c.require(type(index) is int and 0 <= index < GAMES, "fixed eight fresh game indices")
    return str(int.from_bytes(hashlib.sha256(c.canonical([SEED_DOMAIN,index]).encode()).digest(), "big"))


def initial_snapshot(spec):
    c.require(spec == dict(game_id=f"g{spec['game_index']:03d}",game_index=spec["game_index"],
        seed_decimal=seed_for(spec["game_index"]),start_player=spec["game_index"]%2,
        quota=512,world_pool=3,cpuct=.4), "exact fresh prospective game specification")
    state = rules.CorrectedEnv(seed=int(spec["seed_decimal"])).initial_state()
    state.start_player = state.current_player = spec["start_player"]
    value = rules.serialize_snapshot(state)
    guard.decode_training_snapshot(value)
    return value


def build_plan():
    specs = [dict(game_id=f"g{i:03d}",game_index=i,seed_decimal=seed_for(i),start_player=i%2,
                  quota=512,world_pool=3,cpuct=.4) for i in range(GAMES)]
    seeds = {row["seed_decimal"] for row in specs}
    prior = []
    for relative in PRIOR_SCHEDULES:
        values = c.read(c.ROOT/relative)
        c.require(type(values) is list and not seeds & {str(row["seed_decimal"]) for row in values},
                  "fresh seeds disjoint prior registered schedules")
        prior.append(c.artifact(c.ROOT/relative))
    old = c.read(c.ROOT/PRIOR_GAME_PLAN)
    c.require(not seeds & {row["seed_decimal"] for row in old["games"]["specifications"]},
              "fresh games disjoint reused opening diagnostic")
    prior.append(c.artifact(c.ROOT/PRIOR_GAME_PLAN))
    return dict(schema="gpu_throughput_fresh_eight_games_v1",specifications=specs,
        initial_states=[dict(game_id=spec["game_id"],initial=initial_snapshot(spec)) for spec in specs],
        seed_domain=SEED_DOMAIN,prior_schedules=prior,games_per_route=8,route_order=["cpu","gpu"],
        admitted_concurrencies=[1,2,4,8],concurrency_source="independently accepted full opening-panel rate",
        permanent_slot="game_index0..7",schedule="successive consecutive fixed waves; no replacement or refill",
        maximum_plies=MAX_PLIES,maximum_gameplay_seconds=GAMEPLAY_SECONDS,
        both_seats_same_backend=True,same_initial_states_across_routes=True,
        internal_seconds=c.INTERNAL_SECONDS,outer_seconds=c.OUTER_SECONDS,execution_policy_id=c.EXECUTION_POLICY_ID,
        descriptive_latency_thresholds_seconds=[1.,9.,10.],retries=0,replacements=0,
        full_independent_transition_replay=True,stop_route_on_decision_deadline=True,
        technical_or_verification_failure_stops_study=True,
        model_replay_selection="first neural decision per game in bands0..19,20..39,40+",
        model_replay_slot="(game_index+1)%8; other clients inactive",model_replay_maximum_per_route=24,
        within_route_full_result_equality_required=True,
        throughput_includes="first wave release through last durable natural result; wave gaps, tails, all move checks and writes",
        throughput_excludes="model/native startup, warmup, final cohort serialization, saved audits and model replays",
        strength_test=False,training_data=False)


def wave_groups(concurrency):
    c.require(type(concurrency) is int and concurrency in (1,2,4,8), "one prospectively selected concurrency")
    return [list(range(i,i+concurrency)) for i in range(0,GAMES,concurrency)]


def actual_after(before, action):
    state = guard.decode_training_snapshot(before)
    chosen = rules.deserialize_action(action)
    environment = rules.CorrectedEnv()
    c.require(chosen in environment.legal_actions(state), "complete corrected action legal")
    after = rules.serialize_snapshot(environment.step(state,chosen))
    c.require(after == ledger.transition(before,action), "actual corrected transition equals independent ledger")
    guard.check_training_transition(before,action,after)
    return after


def validate_clock(clock, *, completed=True):
    c.require(clock["internal_seconds"] == c.INTERNAL_SECONDS and clock["outer_seconds"] == c.OUTER_SECONDS and
        clock["execution_policy_id"] == c.EXECUTION_POLICY_ID and clock["no_retry"] is True,
        "actual fixed60/75 execution identity")
    c.require(math.isfinite(clock["elapsed_seconds"]) and clock["elapsed_seconds"] >= 0, "finite original response clock")
    if "ended_perf_counter" in clock:
        c.require(clock["elapsed_seconds"] == clock["ended_perf_counter"]-clock["started_perf_counter"], "original clock arithmetic")
    if completed:
        c.require(clock["status"] == "ok" and clock["completed"] is True and
                  clock["elapsed_seconds"] < c.OUTER_SECONDS, "complete response before stuck safeguard")
    else:
        c.require(clock["status"] in ("error","timeout") and clock["completed"] is False, "preserved failed response clock")


def validate_move(row, spec, route, expected_before, ply):
    from . import scientific
    c.require(row["game_id"] == spec["game_id"] and row["route"] == route and row["ply"] == ply and
              row["before"] == expected_before, "contiguous scheduled game snapshot")
    state = guard.decode_training_snapshot(expected_before)
    observer = rules.observation_payload(state)
    c.require(row["observer"] == observer and row["actor"] == state.current_player, "lawful public acting observation")
    validate_clock(row["clock"])
    scientific.validate_result(row["result"],observer,"search",expected_quota=512)
    scientific.audit_decision(observer,row["result"]["action"],row["result"]["diagnostics"],"search",
        before=expected_before,after=row["after"],expected_quota=512)
    c.require(row["after"] == actual_after(expected_before,row["result"]["action"]), "full independent transition reconstruction")
    return row["after"]


def replay_selection(rows):
    selected = []
    for lower,upper in PHASES:
        found = next((row for row in rows if lower <= row["ply"] <= upper and
                      row["result"]["diagnostics"]["inference_calls"] > 0),None)
        if found is not None: selected.append(dict(ply=found["ply"],band=[lower,upper]))
    return selected


def percentile(values, fraction):
    c.require(bool(values) and 0 <= fraction <= 1, "finite empirical percentile")
    ordered = sorted(values); position = (len(ordered)-1)*fraction
    low,high = math.floor(position),math.ceil(position)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def latency_summary(values):
    c.require(values and all(math.isfinite(x) and x >= 0 for x in values), "actual completed response durations")
    return dict(count=len(values),mean=statistics.fmean(values),p50=statistics.median(values),
        p95=percentile(values,.95),maximum=max(values),
        over_seconds={str(int(limit)):sum(value>limit for value in values) for limit in (1.,9.,10.)},
        latency_thresholds_are_descriptive=True)


def _play_one(plan, runtime, folder, spec, adapter, barrier, stopped, boundary, wave_index):
    from . import qualification as q, cell
    game_id = spec["game_id"]; client = "game_"+game_id
    before = initial_snapshot(spec); rows = []; previous_pass = False; ply = 0
    barrier.wait(timeout=c.OUTER_SECONDS)
    start = dict(started_perf_counter=time.perf_counter(),started_unix=time.time(),wave_index=wave_index,
                 game_id=game_id,slot=spec["game_index"])
    c.save(folder/"started.json",start)
    try:
        for ply in range(MAX_PLIES):
            boundary()
            state = guard.decode_training_snapshot(before)
            if state.terminal: break
            if stopped.is_set():
                return dict(game_id=game_id,case_id=game_id,index=spec["game_index"],status="cancelled_between_decisions",
                    completed_moves=len(rows),next_ply=ply,next_snapshot_sha256=c.digest(before),no_retry=True),None
            observer = rules.observation_payload(state)
            request = dict(player="search",observer=observer)
            name = f"{ply:06d}.json"
            def operation(started):
                c.save(folder/"requests"/name,request)
                c.save(folder/"attempts"/name,dict(game_id=game_id,ply=ply,wave_index=wave_index,
                    client_id=client,slot=spec["game_index"],started_perf_counter=started,observed_unix=time.time(),
                    before_sha256=c.digest(before),request_sha256=c.digest(request),no_retry=True))
                return q.ordinary(runtime,{"search":adapter},request,started=started,client_ids={"search":client},
                    evidence_callback=lambda raw:c.save(folder/"raw_results"/name,raw))
            try:
                value,clock = q.timed_call(operation,abort=runtime.abort,
                    on_receipt=lambda receipt:c.save(folder/"clocks"/name,receipt))
            finally:
                c.save(folder/"service_calls"/name,list(adapter.last_service_calls))
            after = actual_after(before,value["action"])
            row = dict(game_id=game_id,route=runtime.route,ply=ply,actor=state.current_player,
                       before=before,observer=observer,result=value,clock=clock,after=after)
            c.save(folder/"moves"/name,row)
            validate_move(row,spec,runtime.route,before,ply)
            rows.append(row)
            is_pass = value["action"]["action_type"] == "PASS"
            c.require(after != before and not(previous_pass and is_pass), "actual deadlock remains incomplete; no synthetic terminal")
            previous_pass = is_pass; before = after
        final = guard.decode_training_snapshot(before)
        c.require(final.terminal and bool(rows), "natural game terminal within fixed ply cap")
        winner = guard.require_official_terminal_target(before,"completed")
        summary = dict(status="naturally_completed",route=runtime.route,game_id=game_id,specification=spec,
            plies=len(rows),initial_sha256=c.digest(initial_snapshot(spec)),final=before,winner=winner,
            complete_response_seconds=[row["clock"]["elapsed_seconds"] for row in rows],
            inference_calls=sum(row["result"]["diagnostics"]["inference_calls"] for row in rows),
            simulations=sum(phase["simulations"] for row in rows for phase in row["result"]["diagnostics"]["phases"]),
            model_replay_selection=replay_selection(rows),training_data=False)
        c.save(folder/"result.json",summary)
        # The result is already fsynced. This observes its durable completion;
        # writing this timing receipt is outside that measured endpoint.
        finished = dict(game_id=game_id,ended_perf_counter=time.perf_counter(),ended_unix=time.time(),
                        result_sha256=c.digest(summary),scope="observed after durable natural result")
        c.save(folder/"finished.json",finished)
        return dict(game_id=game_id,case_id=game_id,index=spec["game_index"],status="completed",
                    completed_moves=len(rows),result_sha256=c.digest(summary),finished=finished,no_retry=True),None
    except BaseException as error:
        stopped.set()
        path = folder/"clocks"/f"{ply:06d}.json"
        clock = c.read(path) if path.exists() else None
        status = cell.classify_decision_error(error)
        if clock is None or clock.get("status") == "ok" or clock.get("completed") is True: status="integrity_failed"
        failure = dict(game_id=game_id,case_id=game_id,index=spec["game_index"],route=runtime.route,
            status=status,failed_ply=ply,completed_moves=len(rows),next_snapshot_sha256=c.digest(before),
            error_chain=cell.error_chain(error),error=traceback.format_exc(),clock=clock,no_retry=True)
        c.save(folder/"failure.json",failure)
        return failure,error


def resolve_wave_outcomes(rows, errors):
    """Keep between-move stops separate from failed original decision clocks."""
    from . import cell
    waiting = {index:row for index,row in rows.items() if row["status"] == "cancelled_between_decisions"}
    active = {index:row for index,row in rows.items() if index not in waiting}
    status = cell.resolve_group_outcomes(active,errors)
    triggers = [row["game_id"] for row in active.values() if row["status"] == "deadline_failed"]
    for row in waiting.values():
        row["status"] = "cancelled" if status == "deadline_failed" and triggers else "integrity_failed"
        row["initial_status"] = "cancelled_between_decisions"
        row["cancellation_kind"] = "between_decisions"
        row["cancellation_trigger_case_ids"] = triggers
    if any(row["status"] == "integrity_failed" for row in rows.values()): return "integrity_failed"
    if not errors and not all(row["status"] == "completed" for row in rows.values()): return "integrity_failed"
    return status


def run_games(plan, runtime, folder, boundary=lambda:None, *, concurrency):
    from . import qualification as q, cell
    q.validate_runtime(runtime)
    c.require(plan["games"] == build_plan(), "fixed fresh eight-game plan")
    groups = wave_groups(concurrency); specs = plan["games"]["specifications"]
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=False)
    adapters={}; all_outcomes={}; wave_rows=[]; first={}; status="passed"; all_threads=[]
    def gameplay_boundary():
        boundary()
        if first:
            c.require(time.perf_counter()-first["started_perf_counter"] < GAMEPLAY_SECONDS and
                      time.time()-first["started_unix"] < GAMEPLAY_SECONDS, "actual two-hour gameplay dual-clock cap")
    try:
        for spec in specs:
            boundary(); game_folder=folder/spec["game_id"]
            c.save(game_folder/"spec.json",spec); c.save(game_folder/"initial.json",initial_snapshot(spec))
            adapters[spec["game_index"]]=runtime.make_adapter("game_"+spec["game_id"],spec["game_index"],512)
        with runtime.service.lock:
            counters_start=dict(batch_count=runtime.service.batch_count,completed_requests=runtime.service.completed_requests)
        for wave_index,indexes in enumerate(groups):
            gameplay_boundary(); origin={}; stopped=threading.Event(); results=queue.Queue(); rows={}; errors={}; threads=[]
            wave_folder=folder/"waves"/f"{wave_index:03d}"
            def release():
                origin.update(started_perf_counter=time.perf_counter(),started_unix=time.time())
                if not first: first.update(origin)
                c.save(wave_folder/"start.json",dict(wave_index=wave_index,game_indexes=indexes,**origin))
            barrier=threading.Barrier(len(indexes)+1,action=release)
            def launch(index):
                spec=specs[index]
                try:
                    row,error=_play_one(plan,runtime,folder/spec["game_id"],spec,adapters[index],barrier,stopped,gameplay_boundary,wave_index)
                except BaseException as error:
                    stopped.set()
                    row=dict(game_id=spec["game_id"],case_id=spec["game_id"],index=index,status="integrity_failed",
                        error_chain=cell.error_chain(error),error=traceback.format_exc(),clock=None,no_retry=True)
                    try:c.save(folder/spec["game_id"]/"worker_failure.json",row)
                    except BaseException:pass
                    results.put((index,row,error)); return
                results.put((index,row,error))
            for index in indexes:
                thread=threading.Thread(target=launch,args=(index,),daemon=True,name="throughput-game-"+specs[index]["game_id"])
                thread.start();threads.append(thread);all_threads.append(thread)
            barrier.wait(timeout=c.OUTER_SECONDS)
            failure_observed=None
            while len(rows)<len(indexes):
                gameplay_boundary()
                if failure_observed is not None:
                    c.require(time.monotonic()-failure_observed < c.OUTER_SECONDS+5., "failed-wave peers did not return after original guards")
                try:index,row,error=results.get(timeout=.1)
                except queue.Empty:continue
                c.require(index not in rows, "one original game outcome")
                rows[index]=row
                if error is not None: errors[index]=error
                if row["status"] != "completed" and failure_observed is None:
                    failure_observed=time.monotonic();stopped.set()
            for thread in threads:thread.join(timeout=.1)
            c.require(not any(thread.is_alive() for thread in threads), "all wave game threads returned")
            wave_status=resolve_wave_outcomes(rows,errors)
            for index in indexes:c.save(folder/specs[index]["game_id"]/"outcome.json",rows[index])
            all_outcomes.update(rows)
            wave=dict(wave_index=wave_index,game_indexes=indexes,game_ids=[specs[i]["game_id"] for i in indexes],
                status=wave_status,**origin,ended_perf_counter=time.perf_counter(),ended_unix=time.time(),
                outcomes=[rows[i] for i in indexes],no_retry=True)
            wave["wall_seconds"]=wave["ended_perf_counter"]-wave["started_perf_counter"]
            c.save(wave_folder/"summary.json",wave);wave_rows.append(wave)
            if wave_status != "passed": status=wave_status;break
        gameplay_boundary()
        with runtime.service.lock:
            counters_end=dict(batch_count=runtime.service.batch_count,completed_requests=runtime.service.completed_requests)
        complete=[index for index,row in sorted(all_outcomes.items()) if row["status"]=="completed"]
        full=status=="passed" and len(complete)==8
        summaries=[c.read(folder/specs[i]["game_id"]/"result.json") for i in complete]
        moves=[c.read(path) for index in sorted(all_outcomes)
               for path in sorted((folder/specs[index]["game_id"]/"moves").glob("*.json"))]
        durations=[row["clock"]["elapsed_seconds"] for row in moves]
        end=max((all_outcomes[i]["finished"]["ended_perf_counter"] for i in complete),default=None) if full else None
        end_wall=max((all_outcomes[i]["finished"]["ended_unix"] for i in complete),default=None) if full else None
        elapsed=None if not full else end-first["started_perf_counter"]
        if full:c.require(elapsed>0, "positive complete-cohort wall")
        cohort=dict(status=status,route=runtime.route,concurrency=concurrency,game_ids=[spec["game_id"] for spec in specs],
            planned_games=8,started_game_ids=[specs[i]["game_id"] for i in sorted(all_outcomes)],
            completed_game_ids=[specs[i]["game_id"] for i in complete],
            never_started_game_ids=[spec["game_id"] for spec in specs if spec["game_index"] not in all_outcomes],
            finished_games=len(complete),full_cohort_completed=full,waves=wave_rows,**first,
            ended_perf_counter=end,ended_unix=end_wall,wall_seconds=elapsed,
            games_per_hour=None if not full else 8*3600/elapsed,
            decisions_per_second=None if not full else len(durations)/elapsed,
            total_plies=len(moves),completed_game_plies=sum(row["plies"] for row in summaries),
            latency=None if not durations else latency_summary(durations),
            inference_calls=sum(row["result"]["diagnostics"]["inference_calls"] for row in moves),
            simulations=sum(phase["simulations"] for row in moves for phase in row["result"]["diagnostics"]["phases"]),
            move_counts_scope="all durable validated moves, including prefixes of unfinished games",
            backend_counter_start=counters_start,backend_counter_end=counters_end,
            throughput_scope=plan["games"]["throughput_includes"],latency_thresholds_are_descriptive=True,
            retries=0,replacements=0,training_data=False,strength_claim=False)
        c.save(folder/"cohort.json",cohort)
    except BaseException:
        runtime.abort()
        c.save(folder/"failure.json",dict(error=traceback.format_exc(),route=runtime.route,no_retry=True))
        raise
    finally:
        for adapter in adapters.values():adapter.close()
    if status == "integrity_failed":
        raise RuntimeError("Game wave integrity failure; preserve evidence and stop study")
    proof=audit_games(plan,folder,concurrency=concurrency,allow_partial=status!="passed")
    c.save(folder/"saved_transition_audit.json",proof)
    replay=None
    if status=="passed": replay=replay_games(plan,runtime,folder,boundary)
    return dict(status=status,cohort=cohort,audit=proof,replay=replay)


def audit_games(plan, folder, *, concurrency, allow_partial=False):
    """Rebuild every saved transition and natural finish; no model calls.

The separate independent auditor additionally binds service journals, ownership,
error causes and supervisor-terminated prefixes before admitting a route result.
"""
    from . import scientific
    folder=Path(folder);selected=plan["games"];cohort=c.read(folder/"cohort.json")
    c.require(selected==build_plan() and cohort["concurrency"]==concurrency and cohort["route"] in ("cpu","gpu"),
              "exact fixed fresh game plan and selected concurrency")
    route=cohort["route"];specs=selected["specifications"];groups=wave_groups(concurrency)
    complete=[];started=[];moves=[];finished=[];outcomes={};rows_by_game={}
    for spec in specs:
        game_id=spec["game_id"];path=folder/game_id;before=initial_snapshot(spec)
        c.require(c.read(path/"spec.json")==spec and c.read(path/"initial.json")==before,"prospective prepared game identity")
        paths=sorted((path/"moves").glob("*.json"))
        c.require([item.name for item in paths]==[f"{i:06d}.json" for i in range(len(paths))],"exact contiguous move prefix")
        rows=[]
        for ply,item in enumerate(paths):
            row=c.read(item);name=item.name;clock=c.read(path/"clocks"/name);request=c.read(path/"requests"/name)
            attempt=c.read(path/"attempts"/name);raw=c.read(path/"raw_results"/name)
            c.require(clock==row["clock"] and request==dict(player="search",observer=row["observer"]) and
                attempt["game_id"]==game_id and attempt["ply"]==ply and attempt["slot"]==spec["game_index"] and
                attempt["client_id"]=="game_"+game_id and attempt["started_perf_counter"]==clock["started_perf_counter"] and
                attempt["before_sha256"]==c.digest(before) and attempt["request_sha256"]==c.digest(request),
                "original whole-response request, attempt and clock")
            c.require(raw["result"]==row["result"] and raw["observer"]==row["observer"] and
                raw["action"]==row["result"]["action"] and raw["diagnostics"]==row["result"]["diagnostics"],
                "complete response saved before validation")
            c.require(type(c.read(path/"service_calls"/name)) is list,"per-decision returned callback receipts retained")
            before=validate_move(row,spec,route,before,ply);rows.append(row)
        rows_by_game[game_id]=rows;moves.extend(rows)
        if not(path/"started.json").exists():
            c.require(not rows and not(path/"outcome.json").exists() and not(path/"result.json").exists() and
                      not list((path/"attempts").glob("*.json")),"prepared game was never started")
            continue
        start=c.read(path/"started.json");started.append(game_id)
        c.require(start["game_id"]==game_id and start["slot"]==spec["game_index"] and
                  spec["game_index"] in groups[start["wave_index"]],"actual game wave and permanent slot")
        outcome=c.read(path/"outcome.json");outcomes[spec["game_index"]]=outcome
        c.require(outcome["game_id"]==outcome["case_id"]==game_id and outcome["index"]==spec["game_index"] and
                  outcome["completed_moves"]==len(rows) and outcome["no_retry"] is True,"final game outcome accounts move prefix")
        if outcome["status"]=="completed":
            ending=c.read(path/"result.json");done=c.read(path/"finished.json")
            winner=guard.require_official_terminal_target(before,"completed")
            c.require(ending["status"]=="naturally_completed" and ending["game_id"]==game_id and ending["route"]==route and
                ending["specification"]==spec and ending["initial_sha256"]==c.digest(initial_snapshot(spec)) and
                ending["final"]==before and ending["winner"]==winner and ending["plies"]==len(rows)>0 and
                ending["model_replay_selection"]==replay_selection(rows),"complete natural game and deterministic replay selection")
            c.require(ending["complete_response_seconds"]==[row["clock"]["elapsed_seconds"] for row in rows] and
                ending["inference_calls"]==sum(row["result"]["diagnostics"]["inference_calls"] for row in rows) and
                ending["simulations"]==sum(phase["simulations"] for row in rows for phase in row["result"]["diagnostics"]["phases"]),
                "natural game totals recomputed")
            c.require(done==outcome["finished"] and done["game_id"]==game_id and
                done["result_sha256"]==outcome["result_sha256"]==c.digest(ending) and
                done["ended_perf_counter"]>=rows[-1]["clock"]["ended_perf_counter"]>=start["started_perf_counter"],
                "durable natural completion observed after final response")
            complete.append(game_id);finished.append(done)
            expected=len(rows)
        else:
            c.require(allow_partial and outcome["status"] in ("deadline_failed","cancelled"),"only preserved incomplete deadline route")
            c.require(not(path/"result.json").exists() and not(path/"finished.json").exists(),"unfinished game has no natural terminal receipt")
            if outcome.get("cancellation_kind")=="between_decisions":
                c.require(outcome["initial_status"]=="cancelled_between_decisions" and outcome["next_ply"]==len(rows) and
                          outcome["next_snapshot_sha256"]==c.digest(before),"peer stopped between complete decisions")
                expected=len(rows)
            else:
                failure=c.read(path/"failure.json");clock=c.read(path/"clocks"/f"{len(rows):06d}.json")
                c.require(failure["failed_ply"]==len(rows) and failure["clock"]==clock and
                    failure["next_snapshot_sha256"]==c.digest(before),"original failed next decision retained")
                validate_clock(clock,completed=False)
                name=f"{len(rows):06d}.json";request=c.read(path/"requests"/name);attempt=c.read(path/"attempts"/name)
                observer=rules.observation_payload(guard.decode_training_snapshot(before))
                c.require(request==dict(player="search",observer=observer) and attempt["started_perf_counter"]==clock["started_perf_counter"] and
                    attempt["request_sha256"]==c.digest(request) and attempt["before_sha256"]==c.digest(before),"failed request keeps original snapshot and clock")
                if (path/"raw_results"/name).exists():
                    raw=c.read(path/"raw_results"/name)
                    c.require(raw["observer"]==observer,"late complete response original observer")
                    scientific.validate_result(raw["result"],observer,"search",expected_quota=512)
                    after=actual_after(before,raw["result"]["action"])
                    scientific.audit_decision(observer,raw["result"]["action"],raw["result"]["diagnostics"],"search",
                        before=before,after=after,expected_quota=512)
                expected=len(rows)+1
        expected_names=[f"{i:06d}.json" for i in range(expected)]
        for name in ("requests","attempts","clocks","service_calls"):
            c.require([item.name for item in sorted((path/name).glob("*.json"))]==expected_names,"no missing or replacement decision attempts: "+name)
        raw_names=[item.name for item in sorted((path/"raw_results").glob("*.json"))]
        c.require(set(raw_names)<=set(expected_names) and set(item.name for item in paths)<=set(raw_names),
                  "raw complete responses belong only to original attempted decisions")
    waves=cohort["waves"]
    c.require(waves and len(waves)<=len(groups),"scheduled nonempty wave prefix")
    expected_started=[]
    for index,wave in enumerate(waves):
        indexes=groups[index];expected_started.extend(specs[i]["game_id"] for i in indexes)
        path=folder/"waves"/f"{index:03d}"
        c.require(wave==c.read(path/"summary.json") and wave["wave_index"]==index and wave["game_indexes"]==indexes and
            wave["game_ids"]==[specs[i]["game_id"] for i in indexes] and wave["outcomes"]==[outcomes[i] for i in indexes],
            "every fixed wave preserves all original game outcomes")
        c.require(c.read(path/"start.json")==dict(wave_index=index,game_indexes=indexes,
            started_perf_counter=wave["started_perf_counter"],started_unix=wave["started_unix"]),"actual fixed barrier origin")
        c.require(wave["wall_seconds"]==wave["ended_perf_counter"]-wave["started_perf_counter"]>0 and
            wave["status"]==("passed" if all(outcomes[i]["status"]=="completed" for i in indexes) else "deadline_failed") and
            (index==len(waves)-1 or wave["status"]=="passed"),"stop after failed wave without replacing games")
        if index:c.require(wave["started_perf_counter"]>=waves[index-1]["ended_perf_counter"],"fixed waves never overlap")
        for i in indexes:
            game_folder=folder/specs[i]["game_id"]
            start=c.read(game_folder/"started.json")
            c.require(wave["started_perf_counter"]<=start["started_perf_counter"]<=wave["ended_perf_counter"],"game starts inside its fixed wave")
            endpoint=wave["ended_perf_counter"]
            if outcomes[i]["status"]=="completed":
                done=c.read(game_folder/"finished.json")
                c.require(start["started_perf_counter"]<=done["ended_perf_counter"]<=endpoint,
                          "durable natural finish stays inside original wave")
                endpoint=done["ended_perf_counter"]
            for clock_path in sorted((game_folder/"clocks").glob("*.json")):
                clock=c.read(clock_path)
                ended=clock.get("ended_perf_counter",clock["started_perf_counter"]+clock["elapsed_seconds"])
                c.require(start["started_perf_counter"]<=clock["started_perf_counter"]<=ended<=endpoint,
                          "every original decision clock stays inside game and wave")
    full=cohort["status"]=="passed" and len(complete)==8
    c.require(started==expected_started==cohort["started_game_ids"] and complete==cohort["completed_game_ids"] and
        cohort["never_started_game_ids"]==[spec["game_id"] for spec in specs if spec["game_id"] not in started] and
        cohort["finished_games"]==len(complete) and cohort["full_cohort_completed"] is full and
        cohort["started_perf_counter"]==waves[0]["started_perf_counter"] and cohort["started_unix"]==waves[0]["started_unix"],
        "complete started/finished/unstarted inventory and cohort origin")
    durations=[row["clock"]["elapsed_seconds"] for row in moves]
    c.require(cohort["total_plies"]==len(moves) and cohort["latency"]==(latency_summary(durations) if durations else None) and
        cohort["inference_calls"]==sum(row["result"]["diagnostics"]["inference_calls"] for row in moves) and
        cohort["simulations"]==sum(phase["simulations"] for row in moves for phase in row["result"]["diagnostics"]["phases"]),
        "all validated move counts and descriptive latencies recomputed")
    if full:
        elapsed=max(done["ended_perf_counter"] for done in finished)-cohort["started_perf_counter"]
        c.require(cohort["wall_seconds"]==elapsed>0 and cohort["ended_perf_counter"]==max(done["ended_perf_counter"] for done in finished) and
            cohort["games_per_hour"]==8*3600/elapsed and cohort["decisions_per_second"]==len(moves)/elapsed and
            cohort["backend_counter_end"]["completed_requests"]-cohort["backend_counter_start"]["completed_requests"]==cohort["inference_calls"],
            "complete-cohort wall throughput and exact callback accounting")
    else:
        c.require(allow_partial and cohort["status"]=="deadline_failed" and all(cohort[key] is None for key in
            ("wall_seconds","games_per_hour","decisions_per_second","ended_perf_counter","ended_unix")),"failed route has no complete-cohort rate")
    return dict(status="all_saved_games_independently_replayed" if full else "saved_game_transition_prefixes_verified",
        route=route,concurrency=concurrency,started_games=len(started),natural_games=len(complete),transitions=len(moves),
        full_cohort_completed=full,cohort_sha256=c.digest(cohort),model_calls=0,training_data=False)


def replay_games(plan, runtime, folder, boundary=lambda:None):
    """Exact selected same-route decisions with shifted slot and no companions."""
    from . import qualification as q
    folder=Path(folder);receipts=[]
    for spec in plan["games"]["specifications"]:
        game_id=spec["game_id"];ending=c.read(folder/game_id/"result.json")
        client="replay_"+game_id;slot=(spec["game_index"]+1)%8
        adapter=runtime.make_adapter(client,slot,512)
        try:
            for selected in ending["model_replay_selection"]:
                boundary();name=f"{game_id}_{selected['ply']:06d}.json"
                source=c.read(folder/game_id/"moves"/f"{selected['ply']:06d}.json")
                request=dict(player="search",observer=source["observer"])
                def operation(started):
                    c.save(folder/"model_replay_requests"/name,request)
                    c.save(folder/"model_replay_attempts"/name,dict(game_id=game_id,ply=selected["ply"],client_id=client,
                        slot=slot,started_perf_counter=started,request_sha256=c.digest(request),source_result_sha256=c.digest(source["result"])))
                    return q.ordinary(runtime,{"search":adapter},request,started=started,client_ids={"search":client},
                        evidence_callback=lambda raw:c.save(folder/"model_replay_raw"/name,raw))
                try:
                    value,clock=q.timed_call(operation,abort=runtime.abort,
                        on_receipt=lambda row:c.save(folder/"model_replay_clocks"/name,row))
                finally:c.save(folder/"model_replay_service_calls"/name,list(adapter.last_service_calls))
                receipt=dict(game_id=game_id,ply=selected["ply"],route=runtime.route,original_slot=spec["game_index"],replay_slot=slot,
                    other_game_clients_inactive=True,result=value,source_result_sha256=c.digest(source["result"]),
                    exact_full_scientific_result=c.canonical(value)==c.canonical(source["result"]),clock=clock)
                c.save(folder/"model_replays"/name,receipt)
                c.require(receipt["exact_full_scientific_result"],"same-route full result differs under changed slot/companions")
                receipts.append(receipt)
        finally:adapter.close()
    c.require(0<len(receipts)<=24,"bounded prospectively selected model replays")
    summary=dict(status="exact_within_route_replays_passed",route=runtime.route,count=len(receipts),
        receipts_sha256=c.digest(receipts),changed_slots=True,changed_companions=True)
    c.save(folder/"model_replay_summary.json",summary)
    c.save(folder/"model_replay_saved_audit.json",verify_model_replays(plan,folder))
    return summary


def verify_model_replays(plan, folder):
    from . import scientific
    folder=Path(folder);rows=[];expected=[]
    for spec in plan["games"]["specifications"]:
        ending=c.read(folder/spec["game_id"]/"result.json")
        for selected in ending["model_replay_selection"]:
            name=f"{spec['game_id']}_{selected['ply']:06d}.json";expected.append(name)
            source=c.read(folder/spec["game_id"]/"moves"/f"{selected['ply']:06d}.json")
            row=c.read(folder/"model_replays"/name);clock=c.read(folder/"model_replay_clocks"/name)
            attempt=c.read(folder/"model_replay_attempts"/name);request=c.read(folder/"model_replay_requests"/name)
            raw=c.read(folder/"model_replay_raw"/name)
            c.require(row["game_id"]==spec["game_id"] and row["ply"]==selected["ply"] and row["route"]==ending["route"] and
                row["original_slot"]==spec["game_index"] and row["replay_slot"]==(spec["game_index"]+1)%8 and
                row["other_game_clients_inactive"] is True and row["exact_full_scientific_result"] is True and
                row["source_result_sha256"]==c.digest(source["result"]) and row["result"]==source["result"],"exact shifted-slot full replay")
            c.require(row["clock"]==clock and request==dict(player="search",observer=source["observer"]) and
                attempt["started_perf_counter"]==clock["started_perf_counter"] and attempt["slot"]==row["replay_slot"] and
                attempt["client_id"]=="replay_"+spec["game_id"] and attempt["request_sha256"]==c.digest(request) and
                attempt["source_result_sha256"]==c.digest(source["result"]) and
                raw["observer"]==source["observer"] and raw["result"]==row["result"],"replay request and result keep original clock")
            validate_clock(clock)
            scientific.validate_result(row["result"],source["observer"],"search",expected_quota=512)
            c.require(type(c.read(folder/"model_replay_service_calls"/name)) is list,"replay returned callback receipts retained")
            rows.append(row)
    for name in ("model_replays","model_replay_attempts","model_replay_requests","model_replay_clocks","model_replay_raw","model_replay_service_calls"):
        c.require([path.name for path in sorted((folder/name).glob("*.json"))]==sorted(expected),"exact selected replay inventory: "+name)
    summary=c.read(folder/"model_replay_summary.json")
    c.require(summary["status"]=="exact_within_route_replays_passed" and summary["count"]==len(rows) and
        0<len(rows)<=24 and summary["receipts_sha256"]==c.digest(rows),"all exact replays bound by summary")
    return dict(status="saved_exact_replays_verified",route=summary["route"],count=len(rows),model_calls=0,
                changed_slots=True,changed_companions=True)


def compare_game_rates(cpu, gpu):
    left,right=cpu["cohort"],gpu["cohort"]
    c.require(left["route"]=="cpu" and right["route"]=="gpu" and left["game_ids"]==right["game_ids"] and
        left["full_cohort_completed"] is right["full_cohort_completed"] is True and
        left["finished_games"]==right["finished_games"]==8,"two completed matched eight-game cohorts")
    c.require(left["ended_perf_counter"]<=right["started_perf_counter"],"route game measurements do not overlap")
    return dict(status="matched_finite_cohort_throughput_verified",games_each=8,cpu_concurrency=left["concurrency"],
        gpu_concurrency=right["concurrency"],cpu_wall_seconds=left["wall_seconds"],gpu_wall_seconds=right["wall_seconds"],
        cpu_games_per_hour=left["games_per_hour"],gpu_games_per_hour=right["games_per_hour"],
        cpu_over_gpu_wall_ratio=left["wall_seconds"]/right["wall_seconds"],
        same_seed_trajectories_may_differ=True,steady_state_claim=False,strength_claim=False,training_data=False)
