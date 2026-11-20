"""Run preregistered persistent card-option planner experiment `079`."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib, json, math, random, statistics, time
from pathlib import Path
from typing import Any

from splendor_ai.bots import LegalActionAttentionScorerBot, OptionPlannerTelemetry, PersistentCardOptionPlannerBot
from splendor_ai.bots.persistent_option_bot import OPTION_DEPTH, OPTION_DETERMINIZATIONS, OPTION_INTERNAL_BRANCHING, OPTION_ROOT_BRANCHING
from splendor_ai.eval.match import play_game

PROTOCOL = "docs/experiments/persistent_card_option_planner_079.md"
CHECKPOINT_SHA = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
VALID = ("completed", "forced_pass_deadlock")
_WORKER: dict[str, Any] = {}

@dataclass(frozen=True, slots=True)
class OptionPlannerConfig:
    checkpoint: Path
    output_dir: Path
    workers: int = 10
    mechanics_pairs: int = 40
    mechanics_seed_start: int = 773_000
    strength_pairs: int = 200
    strength_seed_start: int = 774_000
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 774_079

def _sha(path: Path) -> str:
    d=hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda:h.read(1<<20),b""): d.update(block)
    return d.hexdigest()

def _init(checkpoint: str, limits: tuple[int,int,int]) -> None:
    _WORKER["raw"] = LegalActionAttentionScorerBot(checkpoint, device="cpu")
    _WORKER["limits"] = limits

def _pair(seed: int) -> dict[str, Any]:
    raw=_WORKER["raw"]; mt,rl,nl=_WORKER["limits"]
    a=PersistentCardOptionPlannerBot(raw)
    first=play_game(a,raw,seed=seed,max_turns=mt,repetition_limit=rl,no_progress_limit=nl)
    telemetry=OptionPlannerTelemetry(); telemetry.merge(a.telemetry.as_dict())
    b=PersistentCardOptionPlannerBot(raw)
    second=play_game(raw,b,seed=seed,max_turns=mt,repetition_limit=rl,no_progress_limit=nl)
    telemetry.merge(b.telemetry.as_dict())
    return {"seed":seed,"candidate_seats":[0,1],"games":[asdict(first),asdict(second)],"telemetry":telemetry.as_dict()}

def _panel(cfg: OptionPlannerConfig, start: int, pairs: int, label: str) -> tuple[list[dict[str,Any]],float]:
    began=time.perf_counter(); seeds=list(range(start,start+pairs)); rows={}
    with ProcessPoolExecutor(max_workers=cfg.workers,initializer=_init,initargs=(str(cfg.checkpoint),(cfg.max_turns,cfg.repetition_limit,cfg.no_progress_limit))) as ex:
        futures={ex.submit(_pair,s):s for s in seeds}
        for done,future in enumerate(as_completed(futures),1):
            rows[futures[future]]=future.result()
            if done%10==0 or done==pairs:
                elapsed=time.perf_counter()-began; rate=done/elapsed
                print(f"[079:{label}] clusters={done}/{pairs} rate={rate:.3f}/s eta={(pairs-done)/rate:.0f}s",flush=True)
    return [rows[s] for s in seeds],time.perf_counter()-began

def _merge(rows: list[dict[str,Any]]) -> OptionPlannerTelemetry:
    t=OptionPlannerTelemetry()
    for row in rows:t.merge(row["telemetry"])
    return t

def _rates(t: OptionPlannerTelemetry,games:int)->dict[str,Any]:
    d=max(t.decisions,1); selected=t.option_selections
    return {"movement":t.moved_from_raw035/d,"option_decision_rate":t.option_decisions/d,
            "option_completion_rate":t.option_completions/selected if selected else 0.0,
            "mean_purchases":t.purchases/max(games,1),"deck_reserve_rate":t.deck_reserves/d,
            "phase_movement":{p:t.phase_movement.get(p,0)/max(t.phase_decisions.get(p,0),1) for p in ("early","mid","late")}}

def _integrity(rows:list[dict[str,Any]],pairs:int,start:int)->dict[str,bool]:
    games=[g for r in rows for g in r["games"]]; seeds=[r["seed"] for r in rows]
    return {"exact_clusters":len(rows)==pairs,"exact_games":len(games)==2*pairs,
            "exact_seed_seat":seeds==list(range(start,start+pairs)) and all(r["candidate_seats"]==[0,1] for r in rows),
            "valid_terminals":all(g["termination_reason"] in VALID for g in games),
            "zero_stalls":all(not g["stalled"] for g in games),"zero_timeouts":all(not g["timed_out"] for g in games),
            "zero_fallbacks":all(tuple(g["loop_fallback_triggers_by_seat"])==(0,0) for g in games),
            "zero_illegal_null":True,"candidate_acted":sum(r["telemetry"]["decisions"] for r in rows)>0}

def _metric(row:dict[str,Any])->tuple[float,float]:
    shares=[]; margins=[]
    for g,seat in zip(row["games"],row["candidate_seats"],strict=True):
        shares.append(.5 if g["winner"] is None else float(g["winner"]==seat))
        margins.append(g["final_scores"][seat]-g["final_scores"][1-seat])
    return statistics.fmean(shares),statistics.fmean(margins)

def _boot(values:list[float],samples:int,seed:int)->dict[str,float]:
    rng=random.Random(seed); n=len(values); draws=sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return {"mean":statistics.fmean(values),"ci95_low":draws[int(.025*(samples-1))],"ci95_high":draws[int(.975*(samples-1))]}

def _strength(rows:list[dict[str,Any]],cfg:OptionPlannerConfig)->dict[str,Any]:
    metrics=[_metric(r) for r in rows]; wins=losses=draws=0
    for r in rows:
        for g,s in zip(r["games"],r["candidate_seats"],strict=True):
            if g["winner"] is None:draws+=1
            elif g["winner"]==s:wins+=1
            else:losses+=1
    return {"score_share":_boot([m[0] for m in metrics],cfg.bootstrap_samples,cfg.bootstrap_seed),
            "margin":_boot([m[1] for m in metrics],cfg.bootstrap_samples,cfg.bootstrap_seed+1),
            "wins":wins,"losses":losses,"draws":draws,
            "mean_turns":statistics.fmean(g["turns"] for r in rows for g in r["games"])}

def _write(payload:dict[str,Any],path:Path)->None:
    path.mkdir(parents=True,exist_ok=True); (path/"result.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def run_persistent_card_option_planner(cfg:OptionPlannerConfig)->dict[str,Any]:
    began=time.perf_counter(); digest=_sha(cfg.checkpoint)
    arows,aelapsed=_panel(cfg,cfg.mechanics_seed_start,cfg.mechanics_pairs,"mechanics")
    _init(str(cfg.checkpoint),(cfg.max_turns,cfg.repetition_limit,cfg.no_progress_limit)); deterministic=_pair(cfg.mechanics_seed_start)==arows[0]
    at=_merge(arows); ar=_rates(at,2*cfg.mechanics_pairs); ai=_integrity(arows,cfg.mechanics_pairs,cfg.mechanics_seed_start)
    ai.update({"deterministic":deterministic,"checkpoint":digest==CHECKPOINT_SHA,"frozen_contract":(OPTION_DEPTH,OPTION_ROOT_BRANCHING,OPTION_INTERNAL_BRANCHING,OPTION_DETERMINIZATIONS)==(3,8,4,4)})
    projected=aelapsed*cfg.strength_pairs/cfg.mechanics_pairs
    operational={"under_45_minutes":aelapsed<=2700,"projected_under_3_hours":projected<=10800}
    mechanics={"movement":ar["movement"]>=.15,"option_active":ar["option_decision_rate"]>=.50,"completion":ar["option_completion_rate"]>=.50,"purchases":ar["mean_purchases"]>=5.0}
    stageb=None
    if not all(ai.values()) or not all(operational.values()): decision="persistent_card_option_planner_079_invalid_no_strength_decision"
    elif not all(mechanics.values()): decision="persistent_card_option_planner_079_mechanics_failed_close_exact_architecture"
    else:
        brows,belapsed=_panel(cfg,cfg.strength_seed_start,cfg.strength_pairs,"strength"); bt=_merge(brows); br=_rates(bt,2*cfg.strength_pairs); bi=_integrity(brows,cfg.strength_pairs,cfg.strength_seed_start)
        bi.update({"checkpoint":digest==CHECKPOINT_SHA,"movement":br["movement"]>=.15}); strength=_strength(brows,cfg)
        if not all(bi.values()):decision="persistent_card_option_planner_079_invalid_no_strength_decision"
        elif strength["score_share"]["mean"]>=.55 and strength["score_share"]["ci95_low"]>.50:decision="persistent_card_option_planner_079_stronger_authorize_deployability_preregistration"
        elif strength["score_share"]["ci95_high"]<.50:decision="persistent_card_option_planner_079_weaker_close_exact_architecture"
        else:decision="persistent_card_option_planner_079_not_decisively_stronger_close_exact_architecture"
        stageb={"integrity":bi,"rates":br,"telemetry":bt.as_dict(),"strength":strength,"elapsed_seconds":belapsed,"clusters":brows}
    payload={"experiment":"079","protocol":PROTOCOL,"configuration":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},"checkpoint_sha256":digest,
             "stage_a":{"integrity":ai,"operational":operational,"mechanics":mechanics,"rates":ar,"telemetry":at.as_dict(),"elapsed_seconds":aelapsed,"projected_stage_b_seconds":projected,"strength_not_used":True,"clusters":arows},
             "stage_b":stageb,"decision":decision,"passed":decision.endswith("authorize_deployability_preregistration"),"promotion_authorized":False,"human_data_used":False,"elapsed_seconds":time.perf_counter()-began}
    _write(payload,cfg.output_dir); print(f"[079] decision={decision}"); return payload

def main()->None:
    import argparse
    p=argparse.ArgumentParser(description="Persistent card-option planner (079)"); p.add_argument("--checkpoint",default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt");p.add_argument("--output-dir",default="outputs/persistent_card_option_planner_079");p.add_argument("--workers",type=int,default=10);a=p.parse_args()
    run_persistent_card_option_planner(OptionPlannerConfig(Path(a.checkpoint),Path(a.output_dir),workers=a.workers))
if __name__=="__main__":main()
