from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from splendor_ai.encoding import canonical_public_belief_key, canonical_successor_action_key
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card
from splendor_ai.eval import run_independent_world_root_validation as audit
from splendor_ai.training.replay import serialize_state_snapshot
from splendor_ai.training.run_action_preference_repair import rollout_score_margin_loop_safe


class OrderedScorer:
    def score_actions(self, state, legal_actions):
        return tuple(SimpleNamespace(action=a, score=float(len(legal_actions)-i), action_index=i)
                     for i, a in enumerate(legal_actions))


def sample_for(state=None):
    state = state or SplendorEnv(seed=138).initial_state()
    return {"sample_index": 12, "source_seed": 860012, "turn_index": state.turn_index,
            "actor_seat": state.current_player, "state_snapshot": serialize_state_snapshot(state)}


def test_selection_is_balanced_and_ignores_outcomes_decisions_and_source_order():
    rows = []
    for cell in audit.CELLS:
        seat, phase = cell.split("|")
        for i in range(12):
            rows.append({"source_seed": len(rows), "turn_index": i, "actor_seat": int(seat[-1]),
                         "phase": phase, "selection_cell": cell, "sample_index": len(rows)})
    selected = audit.select_states({"selected": rows})
    poisoned = [{**r, "winner": 999, "payoff": float("nan"), "expanded_action": "arbitrary"} for r in reversed(rows)]
    assert [s["sample_index"] for s in selected] == [s["sample_index"] for s in audit.select_states({"selected": poisoned})]
    assert len({s["source_seed"] for s in selected}) == 60
    assert all(sum(s["selection_cell"] == cell for s in selected) == 10 for cell in audit.CELLS)


def test_continuation_seed_slots_unique_and_common_across_conditions():
    seeds = {audit.continuation_seed(s, b, w, r) for s in range(300)
             for b in range(4) for w in range(3) for r in range(4)}
    assert len(seeds) == 14_400
    assert min(seeds) == 1_380_000_000_000_000_000
    assert max(seeds) < 1_389_000_000_000_000_000
    with pytest.raises(ValueError):
        audit.continuation_seed(300, 0, 0, 0)


def test_worlds_hold_observer_information_but_refresh_whole_deck_and_private_pool():
    env = SplendorEnv(seed=138)
    state = env.initial_state()
    blind = next(a for a in env.legal_actions(state) if a.action_type == ActionType.RESERVE_DECK)
    state = env.step(state, blind)  # current player now observes an opponent private reserve
    sample = sample_for(state)
    original = audit.worlds_for(sample, "original_worlds", 0)
    assert [serialize_state_snapshot(w) for w in original] == [serialize_state_snapshot(w) for w in audit.worlds_for(sample, "original_worlds", 3)]
    fresh0 = audit.worlds_for(sample, "fresh_worlds", 0)
    fresh1 = audit.worlds_for(sample, "fresh_worlds", 1)
    assert [serialize_state_snapshot(w) for w in original] != [serialize_state_snapshot(w) for w in fresh0]
    assert [serialize_state_snapshot(w) for w in fresh0] != [serialize_state_snapshot(w) for w in fresh1]
    key = canonical_public_belief_key(state, state.current_player)
    assert all(canonical_public_belief_key(w, state.current_player) == key for w in fresh0+fresh1)
    swapped = state.copy_shallow()
    for tier in (1, 2, 3):
        swapped.hidden_tier_decks[tier].reverse()
    changed = {**sample, "state_snapshot": serialize_state_snapshot(swapped)}
    assert [serialize_state_snapshot(w) for w in fresh0] == [serialize_state_snapshot(w) for w in audit.worlds_for(changed, "fresh_worlds", 0)]


def forced_pass_state():
    state = SplendorEnv(seed=138).initial_state()
    state.bank_tokens.update({c: 0 for c in ALL_TOKEN_TYPES})
    state.players[0].score = 2
    for pid, player in enumerate(state.players):
        player.reserved_cards = [Card(card_id=f"test-{pid}-{i}", tier=1, bonus_color="white",
                                      points=0, cost={"blue": 10}) for i in range(3)]
    return state


def test_fewer_than_four_replies_are_evaluated_without_world_replacement():
    env = SplendorEnv()
    state = forced_pass_state()
    action = env.legal_actions(state)[0]
    key = canonical_successor_action_key(state, action)
    branches = audit.prepare_branches([state.copy_shallow() for _ in range(3)], (key,), OrderedScorer())
    assert len(branches) == 3
    assert all(len(b["replies"]) == 1 for b in branches)


def test_terminal_roots_do_not_request_policy_or_continuations():
    state = SplendorEnv(seed=138).initial_state()
    state.pending_round_end = True
    state.current_player = 1
    state.players[0].score = 15
    env = SplendorEnv()
    key = canonical_successor_action_key(state, env.legal_actions(state)[0])
    class ForbiddenScorer:
        def score_actions(self, *args):
            raise AssertionError("terminal root requested a policy")
    branches = audit.prepare_branches([state.copy_shallow() for _ in range(3)], (key,), ForbiddenScorer())
    assert all(b["after"].terminal and b["replies"] == [] for b in branches)


def test_shared_infoset_minimum_and_wdl_under_margin_reply():
    root = ("root",)
    records = [{"world": w, "root": root, "info": ("shared",), "reply": (reply,),
                "margin": margin, "wdl": wdl}
               for w in range(3) for reply, margin, wdl in (("x", .2, 1.), ("y", .5, -1.))]
    values, changes, choices = audit.resolve_records(records, (root,))
    assert values[repr(root)] == pytest.approx({"margin": .2, "wdl_min": -1., "wdl_under_margin": 1.})
    assert changes == choices == 1
    # Cannot minimize separately inside each hidden world of the same infoset.
    for row in records:
        row["margin"] = (1. if row["world"] != 1 else -1.) * (1 if row["reply"] == ("x",) else -1)
    values, _, _ = audit.resolve_records(records, (root,))
    assert values[repr(root)]["margin"] == pytest.approx(-1/3)
    with pytest.raises(ValueError, match="unstable semantic actions"):
        audit.resolve_records(records[:-1], (root,))


def test_tracking_rollout_preserves_legacy_behavior_and_records_deadlock_wdl():
    state = forced_pass_state()
    env, scorer = SplendorEnv(), OrderedScorer()
    action = env.legal_actions(state)[0]
    legacy = rollout_score_margin_loop_safe(env, scorer, state, action, 0, 138,
                                           .1, .5, 3, 400, 20., 64)
    observed = audit.observe_rollout(state, action, 0, 138, scorer)
    assert {k: observed[k] for k in asdict(legacy)} == asdict(legacy)
    assert observed["wdl"] == 1
    assert observed["termination_reason"] == "forced_pass_deadlock_adjudicated"
    assert observed["final_scores"] == [2, 0]


def test_true_wdl_uses_card_count_tiebreak_even_when_margin_zero():
    state = SplendorEnv(seed=138).initial_state()
    state.players[0].purchased_cards = [state.visible_tier_cards[1][0]]
    record = audit.terminal_record(state, 0)
    assert record["wdl"] == -1 and record["final_scores"] == [0, 0]


def test_paired_bootstrap_weights_games_not_blocks_and_preserves_constant_difference():
    rows = []
    for i in range(60):
        old = i / 100
        rows.append({"selection_cell": audit.CELLS[i//10], "source_seed": i,
                     "gains": {"original_worlds": {"margin": old}, "fresh_worlds": {"margin": old-.03}},
                     "difference": -.03})
    summary = audit.paired_summary(rows, samples=1000)
    assert summary == audit.paired_summary(list(reversed(rows)), samples=1000)
    assert summary["fresh_minus_original"]["mean"] == pytest.approx(-.03)
    assert summary["fresh_minus_original"]["ci95"] == pytest.approx([-.03, -.03])
    assert summary["original_gain"]["mean"] == pytest.approx(.295)


def test_decision_requires_reproduced_control_and_respects_inconclusive_branches():
    summary = {"original_gain": {"mean": .05, "ci95": [.02, .08]},
               "fresh_gain": {"mean": 0., "ci95": [-.02, .02]},
               "fresh_minus_original": {"mean": -.05, "ci95": [-.08, -.02]}}
    assert audit.classify(summary) == "material_world_attenuation_supported"
    summary["original_gain"] = {"mean": -.01, "ci95": [-.03, .01]}
    assert audit.classify(summary) == "conditional_gain_not_reproduced"
    summary["original_gain"] = {"mean": .01, "ci95": [-.01, .03]}
    assert audit.classify(summary) == "original_world_gain_uncertain"
    summary["original_gain"] = {"mean": .05, "ci95": [.02, .08]}
    summary["fresh_gain"] = {"mean": .05, "ci95": [.02, .08]}
    summary["fresh_minus_original"] = {"mean": 0., "ci95": [-.01, .01]}
    assert audit.classify(summary) == "conditional_gain_survives_fresh_worlds"
    assert audit.classify(summary, False) == "independent_world_validation_integrity_failed"


def test_saved_state_cannot_resume_with_changed_identity_or_incomplete_blocks():
    sample = {"some": "fixed input"}
    row = {"sample_sha256": audit.digest(sample),
           "blocks": [{"condition": c, "block": b} for c in audit.CONDITIONS for b in range(4)]}
    payload = {"identity": {"hash": "fixed"}, "row": row}
    assert audit.validate_saved_state(payload, sample, payload["identity"]) == row
    with pytest.raises(ValueError, match="identity mismatch"):
        audit.validate_saved_state(payload, {"some": "changed"}, payload["identity"])
    broken = deepcopy(payload)
    broken["row"]["blocks"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        audit.validate_saved_state(broken, sample, payload["identity"])


def test_process_lock_excludes_concurrent_runner_and_releases_after_exception(tmp_path):
    path = tmp_path / "process.lock"
    with pytest.raises(RuntimeError):
        with audit.process_lock(path):
            with pytest.raises(OSError):
                with audit.process_lock(path):
                    pass
            raise RuntimeError("simulate crash")
    with audit.process_lock(path):
        pass
