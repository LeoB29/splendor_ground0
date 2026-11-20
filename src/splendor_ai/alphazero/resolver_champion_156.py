"""Observation-only instrumentation around the unchanged serial084 player."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path
import time
import traceback

from splendor_ai.alphazero import run_rollout_selector_gameplay as p155
from splendor_ai.eval import run_paired_public_belief_resolver_gameplay as r084

old = p155.old
BELIEF_START = {"main": 1_564_000_000, "probe": 1_569_400_000}
CONTINUATION_START = {"main": 1_560_000_000_000_000_000, "probe": 1_569_000_000_000_000_000}
FALLBACKS = {"fewer_than_four_root_actions", "root_mapping_changed", "root_action_terminal",
             "fewer_than_four_opponent_actions", "opponent_action_set_unstable"}


def config(stage):
    assert stage in ("probe", "main")
    # engine_seed in the legacy interface is a logical RNG index, never the
    # environment seed. Only namespace/operational fields differ from084.
    return r084.ResolverGameplayConfig(
        smoke_seed=0, belief_root_index_start=BELIEF_START[stage],
        continuation_seed_start=CONTINUATION_START[stage], workers=8)


def frozen_contract_matches(stage):
    cfg = config(stage)
    original = r084.ResolverGameplayConfig()
    # The historical checker also binds its ten-game-worker deployment.
    #156 changes only the approved RNG namespaces and outer pool size; the
    # bot itself still runs serial continuations. Check every other field.
    normalized = replace(cfg, smoke_seed=original.smoke_seed,
                         belief_root_index_start=original.belief_root_index_start,
                         continuation_seed_start=original.continuation_seed_start,
                         workers=original.workers)
    return cfg.workers == 8 and normalized == original and r084._frozen_contract_matches_049(normalized)


class TraceEnv:
    def __init__(self, delegate):
        self.delegate = delegate
        self.hasher = hashlib.sha256()
        self.moves = 0
        self.final = None

    def legal_actions(self, state):
        return self.delegate.legal_actions(state)

    def step(self, state, action):
        self.hasher.update(old.digest(old.serialize_action(action)).encode())
        successor = self.delegate.step(state, action)
        self.moves += 1
        self.final = successor
        return successor


def traced_branch(original, env, scorer, state, action, actor, seed,
                  epsilon, temperature, top_k, max_turns, scale, max_fallbacks):
    traced = TraceEnv(env)
    outcome = original(traced, scorer, state, action, actor, seed,
                       epsilon, temperature, top_k, max_turns, scale, max_fallbacks)
    row = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(state)),
           "action": old.serialize_action(action), "actor": actor, "outcome": asdict(outcome),
           "moves": traced.moves, "trace_sha256": traced.hasher.hexdigest(),
           "final_state_sha256": old.digest(old.serialize_state_snapshot(traced.final))}
    return outcome, row


@contextmanager
def capture_calls(record):
    """Process-local, serial observation hooks; originals restored on any exit."""
    original_worlds = r084.deterministic_belief_range
    original_branch = r084.rollout_score_margin_loop_safe
    def worlds(state, observer, *, root_index, count=3, seed_start=49_000_000):
        samples = original_worlds(state, observer, root_index=root_index, count=count, seed_start=seed_start)
        assert not record["worlds"]
        record["worlds"] = [{"seed": seed_start+root_index*10003+i,
                             "state_sha256": old.digest(old.serialize_state_snapshot(s))}
                            for i, s in enumerate(samples)]
        return samples
    def branch(*args):
        outcome, row = traced_branch(original_branch, *args)
        record["branches"].append(row)
        return outcome
    r084.deterministic_belief_range = worlds
    r084.rollout_score_margin_loop_safe = branch
    try:
        yield
    finally:
        r084.deterministic_belief_range = original_worlds
        r084.rollout_score_margin_loop_safe = original_branch


def score_rows(rows):
    return [{"action": old.serialize_action(r.action), "action_index": r.action_index,
             "score": r.score} for r in rows]


class RecordedChampion(r084.FrozenPublicBeliefResolverBot):
    def __init__(self, stage, stream_index, seat, **kwargs):
        super().__init__(config(stage), stream_index, seat, **kwargs)
        assert self._continuation_workers == 1, "156 preserves serial084; no nested pool"
        self.record = None

    def _on_resolved_decision(self, *, state, legal_actions, all_scores, parent, selected,
                              decision_index, resolution):
        self.record.update(kind="search", scores=score_rows(all_scores),
                           parent=old.serialize_action(parent), action=old.serialize_action(selected),
                           resolution=resolution.to_payload())

    def _on_fallback_decision(self, *, state, legal_actions, all_scores, parent, decision_index, reason):
        self.record.update(kind="fallback", scores=score_rows(all_scores),
                           parent=old.serialize_action(parent), action=old.serialize_action(parent), reason=reason)

    def choose_recorded(self, state, path, binding):
        snapshot = old.serialize_state_snapshot(state)
        if path.exists():
            row = old.read_json(path)
            assert row["manifest"] == binding and "failure" not in row
            assert old.digest(row["snapshot"]) == old.digest(snapshot)
            assert row["stream_index"] == self._engine_seed and row["seat"] == self._actor_seat
            assert row["decision_index"] == self._decisions and row["before"] == self.telemetry
            validate_record(row)
            self.restore(row["after"])
            return old.decode_action(row["action"]), row
        wall, cpu = time.perf_counter(), time.process_time()
        row = {"manifest": binding, "time": old.now(), "snapshot": snapshot,
               "stream_index": self._engine_seed, "seat": self._actor_seat,
               "decision_index": self._decisions, "before": self.telemetry,
               "worlds": [], "branches": [], "privacy": p155.privacy(state)}
        self.record = row
        try:
            with capture_calls(row):
                action = super().choose_action(old.SplendorEnv(), state)
            row["after"] = self.telemetry
            assert row["action"] == old.serialize_action(action)
            validate_record(row)
        except Exception:
            row["failure"] = traceback.format_exc()
            raise
        finally:
            row["timing"] = {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}
            old.save_new(path, row)
            self.record = None
        return action, row

    def restore(self, telemetry):
        from collections import Counter
        assert telemetry["engine_seed"] == self._engine_seed and telemetry["actor_seat"] == self._actor_seat
        for field, name in (("_decisions", "decisions"), ("_searches", "completed_searches"),
                            ("_switches", "switches"), ("_continuations", "continuations"),
                            ("_resolved_continuations", "resolved_continuations"),
                            ("_hard_caps", "hard_caps"), ("_loop_adjudications", "loop_adjudications"),
                            ("_rollout_fallback_triggers", "rollout_fallback_triggers")):
            setattr(self, field, telemetry[name])
        self._fallback_reasons = Counter(telemetry["fallback_reasons"])
        self._by_phase = {k: Counter(v) for k, v in telemetry["by_phase"].items()}
        assert self.telemetry == telemetry


def validate_record(row):
    assert all(math.isfinite(r["score"]) for r in row["scores"])
    assert row["after"]["decisions"] == row["before"]["decisions"]+1
    if row["kind"] == "fallback":
        assert row["reason"] in FALLBACKS and not row["branches"]
        assert row["action"] == row["parent"]
    else:
        assert row["kind"] == "search" and len(row["branches"]) == 192 and len(row["worlds"]) == 3
    for branch in row["branches"]:
        outcome = branch["outcome"]
        assert outcome["resolved"] and not outcome["hard_cap_reached"] and math.isfinite(outcome["value"])
        assert outcome["terminal"] or outcome["loop_adjudicated"]
        assert branch["moves"] == outcome["turns"]+1


def summary(row, index, turn):
    branches = row["branches"]
    return {"index": index, "seat": row["seat"], "turn": turn,
            "kind": row["kind"], "reason": row.get("reason"),
            "changed": row["after"]["switches"] > row["before"]["switches"], "privacy": row["privacy"],
            "timing": row["timing"], "continuations": len(branches),
            "moves": sum(b["moves"] for b in branches),
            "loop_adjudications": sum(b["outcome"]["loop_adjudicated"] for b in branches),
            "rollout_fallback_triggers": sum(b["outcome"]["fallback_triggers"] for b in branches),
            "hard_caps": sum(b["outcome"]["hard_cap_reached"] for b in branches),
            "resolved": sum(b["outcome"]["resolved"] for b in branches)}
