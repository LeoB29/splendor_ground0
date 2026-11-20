"""Prospective local-opponent panel; isolated from frozen Experiment 162.

No training, promotion, automatic launch, or changes to the teacher. All nine
players are retained; bootstrap units contain both seats and every matchup.
"""
from collections import Counter
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import student_refresh_162 as parent
from splendor_ai.bots.model_bot import CanonicalEntityActionScorerBot, LegalActionAttentionScorerBot
from splendor_ai.bots.search_bot import ShallowSearchBot

old, student = parent.old, parent.f
OUTPUT = Path("outputs/student_generalization_panel")
ARMS = ("anchor", "refresh", "replay")
OPPONENTS = ("035", "shallow", "023")
SEEDS = (162001, 162002, 162003)
CONTRASTS = (("refresh_minus_anchor", 1, 0), ("refresh_minus_replay", 1, 2),
             ("replay_minus_anchor", 2, 0))
CONFIG = {"main_clusters": 100, "probe_clusters": 1, "workers": 8,
          "bootstrap_samples": 50000, "bootstrap_seed": 2000300000,
          "main_seed": 2000000000, "probe_seed": 2000100000,
          "stream_tag": "student-generalization-panel-v1", "torch_threads": 1,
          "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
          "device": "cpu", "promotion": False, "automatic_resume": False}
SHALLOW = {"depth": 1, "max_branching": 6, "max_buy_actions": 4,
           "max_reserve_actions": 2, "max_take_actions": 2, "seed": 23,
           "information_legal": True, "determinizations": 1, "determinization_seed": 29}
CHECKPOINT_023 = Path("outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt")
SHA_023 = "29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070"
MODELS = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def schedule(stage):
    require(stage in ("probe", "main"), "Unknown panel stage")
    rows = []
    for cluster in range(CONFIG[f"{stage}_clusters"]):
        for arm in ARMS:
            for seed in SEEDS:
                for opponent in OPPONENTS:
                    for seat in (0, 1):
                        rows.append({"index": len(rows), "cluster": cluster, "arm": arm,
                                     "seed": seed, "opponent": opponent, "seat": seat,
                                     "environment_seed": CONFIG[f"{stage}_seed"] + cluster,
                                     "stream": old.digest([CONFIG["stream_tag"], stage, cluster])})
    return rows


def initialize(models):
    old.initialize_worker()
    MODELS.clear()
    for key, item in models.items():
        require(old.file_hash(Path(item["path"])) == item["sha256"], "Student checkpoint changed")
        MODELS[key] = student.StrongPolicyInference(item["path"]).model


def opponent_bot(name):
    # New instances for each game AND replay: 023 and shallow track game history.
    if name == "035":
        return LegalActionAttentionScorerBot(old.CHECKPOINT)
    if name == "023":
        require(old.file_hash(CHECKPOINT_023) == SHA_023, "023 checkpoint changed")
        return CanonicalEntityActionScorerBot(CHECKPOINT_023)
    if name == "shallow":
        return ShallowSearchBot(**SHALLOW)
    raise ValueError("Unknown opponent")


def game_path(stage, index, base=OUTPUT):
    require(stage in ("test", "probe", "main"), "Unknown game stage")
    base = Path(base)
    if stage == "test":
        require(base.resolve() != OUTPUT.resolve() and OUTPUT.resolve() not in base.resolve().parents,
                "Test records cannot enter production")
    return base / stage / "games" / f"game_{index:04d}.json"


def play_game(stage, spec, binding, base=OUTPUT, test_initial_snapshot=None):
    path = game_path(stage, spec["index"], base)
    require(not path.exists(), "Game already exists; no retry or replacement")
    require(test_initial_snapshot is None or stage == "test", "Production snapshot override")
    if stage != "test":
        require(schedule(stage)[spec["index"]] == spec, "Unexpected schedule")
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state() if test_initial_snapshot is None else old.deserialize_state_snapshot(test_initial_snapshot)
    record = {"manifest": binding, "spec": spec, "trace": []}
    if test_initial_snapshot is not None:
        record["test_initial_snapshot"] = test_initial_snapshot
    wall = time.perf_counter()
    reason, winner = "completed", None
    seen, passes, idle = Counter(), 0, 0
    try:
        bot = opponent_bot(spec["opponent"])
        while not state.terminal:
            seen[old.state_signature(state)] += 1
            require(seen[old.state_signature(state)] < 4, "repetition_cutoff")
            require(state.turn_index < 400, "max_turns")
            legal = env.legal_actions(state)
            require(bool(legal), "stalled")
            candidate = state.current_player == spec["seat"]
            key = f"{spec['arm']}_{spec['seed']}" if candidate else spec["opponent"]
            row = {"turn": state.turn_index, "actor": state.current_player, "model": key,
                   "state_sha256": old.digest(old.serialize_state_snapshot(state)), "legal_support": len(legal)}
            sw, sc = time.perf_counter(), time.process_time()
            if candidate:
                action, row["evidence"] = student.choose(state, spec["stream"], MODELS[key])
            else:
                action = bot.choose_action(env, state, legal)
            row["timing"] = {"wall": time.perf_counter()-sw, "cpu": time.process_time()-sc}
            require(action is not None and action in legal, "illegal_or_missing_action")
            row["action"] = old.serialize_action(action)
            record["trace"].append(row)
            nxt = env.step(state, action)
            passes = passes+1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state, nxt, action) else idle+1
            state = nxt
            if passes >= 2:
                reason, winner = "forced_pass_deadlock", old.adjudicate_scores(state)
                break
            require(idle < 60, "no_progress_cutoff")
        if reason == "completed":
            winner = state.winner
        record.update(reason=reason, winner=winner)
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record.update(final_state=old.serialize_state_snapshot(state), elapsed_seconds=time.perf_counter()-wall)
        old.save_new(path, record)
    return parent.artifact(path, index=spec["index"])


def statistics(rows):
    """Paired panel changes; strict complete Cartesian panel, no missing games."""
    specs = schedule("main")
    lookup = {r["index"]: r for r in rows}
    require(len(rows) == len(lookup) == len(specs), "Incomplete/duplicate panel")
    for spec in specs:
        row = lookup[spec["index"]]
        require(all(row.get(k) == v for k, v in spec.items()), "Row schedule mismatch")
        require(row["score"] in (0., .5, 1.), "Invalid score")
        require(row["reason"] in old.VALID_NATURAL_TERMINATIONS, "Invalid termination")
    # cluster, arm, fit, opponent, seat: each component equally weighted.
    values = np.array([lookup[i]["score"] for i in range(len(specs))]).reshape(100, 3, 3, 3, 2)
    panel = values.mean(axis=(2, 3, 4))
    opponent_values = values.mean(axis=(2, 4))
    deltas = np.column_stack([panel[:, a]-panel[:, b] for _, a, b in CONTRASTS])
    per_opponent_deltas = np.stack([opponent_values[:, a]-opponent_values[:, b]
                                    for _, a, b in CONTRASTS], axis=1)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    resampled, opp_resampled = [], []
    for start in range(0, CONFIG["bootstrap_samples"], 1000):
        size = min(1000, CONFIG["bootstrap_samples"]-start)
        ids = rng.integers(100, size=(size, 100))
        resampled.append(deltas[ids].mean(axis=1))
        opp_resampled.append(per_opponent_deltas[ids].mean(axis=1))
    ci = np.percentile(np.concatenate(resampled), [2.5, 97.5], axis=0)
    opp_ci = np.percentile(np.concatenate(opp_resampled), [2.5, 97.5], axis=0)
    comparisons = {}
    for j, (name, _, _) in enumerate(CONTRASTS):
        comparisons[name] = {"delta": float(deltas[:, j].mean()), "ci95": ci[:, j].tolist(),
                             "per_opponent": {o: {"delta": float(per_opponent_deltas[:, j, k].mean()),
                                                    "ci95": opp_ci[:, j, k].tolist()}
                                              for k, o in enumerate(OPPONENTS)}}
    scores = {}
    for a, arm in enumerate(ARMS):
        group = [r for r in rows if r["arm"] == arm]
        scores[arm] = {"score": float(panel[:, a].mean()),
                       "wins": sum(r["score"] == 1 for r in group),
                       "draws": sum(r["score"] == .5 for r in group),
                       "losses": sum(r["score"] == 0 for r in group),
                       "per_opponent": {o: float(opponent_values[:, a, i].mean()) for i, o in enumerate(OPPONENTS)},
                       "per_fit": {str(s): float(values[:, a, i].mean()) for i, s in enumerate(SEEDS)},
                       "per_seat": [float(values[:, a, :, :, i].mean()) for i in (0, 1)]}
    return {"games": len(rows), "clusters": 100, "primary": "refresh_minus_anchor",
            "secondary": ["refresh_minus_replay", "replay_minus_anchor"],
            "comparisons": comparisons, "scores": scores, "promotion": False,
            "interpretation": "Local panel calibration; not all-opponent, equal-time, or human superiority."}


def decision_map(result):
    """Interpret completed audited 162 results without changing their gates."""
    require(result.get("all_prescribed_audits_pass") is True, "162 is not audited")
    stats = result["statistics"]
    transfer, progress = stats["transfer"]["pass"], stats["progress"]["pass"]
    require(type(transfer) is bool and type(progress) is bool, "Invalid gate values")
    require(stats["both_pass"] == (transfer and progress), "Inconsistent162 gates")
    return {(True, True): "Fresh feedback adds value and advances161; broader transfer remains untested.",
            (True, False): "Fresh feedback beats replay; required progress beyond161 is unestablished.",
            (False, True): "The update advances161; superiority due to fresh feedback is unestablished.",
            (False, False): "This bounded refresh did not establish both benefits; inspect estimates and uncertainty."}[transfer, progress]
