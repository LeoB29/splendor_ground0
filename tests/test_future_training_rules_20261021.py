"""Future pipeline safeguards; no model loading or training."""
from copy import deepcopy
import hashlib

import pytest

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from splendor_ai.engine.env import SplendorEnv


def metadata(data=b"fixture"):
    return {"schema": "corrected_training_data_v1", "ruleset_id": guard.RULESET_ID,
            "catalogue_sha256": guard.CATALOGUE_SHA256, "turn_contract": guard.TURN_CONTRACT,
            "origin": "generated_under_declared_rules", "data_sha256": hashlib.sha256(data).hexdigest()}


def test_factory_uses_corrected_faces_without_patching_old_environment():
    state = guard.make_environment(seed=817).initial_state()
    assert type(guard.make_environment()) is rules.CorrectedEnv
    assert rules.serialize_snapshot(state)["catalogue_sha256"] == guard.CATALOGUE_SHA256
    with pytest.raises(ValueError, match="noncorrected"):
        rules.serialize_snapshot(SplendorEnv(seed=817).initial_state())


def test_old_snapshot_shape_is_rejected_instead_of_silently_converted():
    snapshot = rules.serialize_snapshot(guard.make_environment(seed=1).initial_state())
    with pytest.raises(ValueError, match="envelope"):
        guard.decode_training_snapshot(snapshot["state"])


@pytest.mark.parametrize("key,value", [
    ("schema", "legacy"), ("ruleset_id", "historical"), ("catalogue_sha256", "0" * 64),
    ("turn_contract", "sequential_unqualified"), ("origin", "legacy_relabelled"),
    ("data_sha256", "not-a-hash"), ("origin", None),
])
def test_historical_missing_or_incompatible_data_contract_fails(key, value):
    info = metadata()
    if value is None:
        info.pop(key)
    else:
        info[key] = value
    with pytest.raises(ValueError):
        guard.require_data_contract(info)


def test_data_hash_is_bound_to_actual_bytes(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"fixture")
    assert guard.verify_data_file(path, metadata()) == metadata()["data_sha256"]
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="data bytes"):
        guard.verify_data_file(path, metadata())


def test_decoder_checks_terminal_ledger_not_only_faces():
    snapshot = rules.serialize_snapshot(guard.make_environment(seed=1).initial_state())
    snapshot["state"]["terminal"] = True
    with pytest.raises(ValueError, match="round completion"):
        guard.decode_training_snapshot(snapshot)


def test_initial_transitions_and_official_target_rejection():
    env = guard.make_environment(seed=5)
    snapshot = rules.serialize_snapshot(env.initial_state())
    state = guard.decode_training_snapshot(snapshot)
    action = guard.legal_actions_for_snapshot(snapshot)[0]
    after = rules.serialize_snapshot(env.step(state, action))
    assert guard.check_training_transition(snapshot, rules.serialize_action(action), after)
    for reason in ("completed", "forced_pass_deadlock", "max_turns", "no_progress_cutoff"):
        with pytest.raises(ValueError, match="actual completed game"):
            guard.require_official_terminal_target(after, reason)


def test_malformed_turn_cannot_pass_transition_check():
    env = guard.make_environment(seed=5)
    before = rules.serialize_snapshot(env.initial_state())
    action = guard.legal_actions_for_snapshot(before)[0]
    after = rules.serialize_snapshot(env.step(guard.decode_training_snapshot(before), action))
    wrong = deepcopy(after)
    wrong["state"]["bank_tokens"]["gold"] += 1
    with pytest.raises(ValueError):
        guard.check_training_transition(before, rules.serialize_action(action), wrong)


def test_official_completed_target_is_accepted_but_deadlock_label_is_not():
    state = guard.make_environment(seed=5).initial_state()
    chosen = [c for c in rules.corrected_cards_by_id().values() if c.points == 5][:3]
    ids = {c.card_id for c in chosen}
    for tier in (1, 2, 3):
        state.visible_tier_cards[tier] = [c for c in state.visible_tier_cards[tier] if c.card_id not in ids]
        state.hidden_tier_decks[tier] = [c for c in state.hidden_tier_decks[tier] if c.card_id not in ids]
        while len(state.visible_tier_cards[tier]) < 4 and state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop(0))
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    player = state.players[0]
    player.purchased_cards = chosen
    player.score = 15
    for card in chosen:
        player.bonuses[card.bonus_color] += 1
    state.turn_index = 50
    state.pending_round_end = state.terminal = True
    state.winner = 0
    snapshot = rules.serialize_snapshot(state)
    assert guard.require_official_terminal_target(snapshot, "completed") == 0
    with pytest.raises(ValueError, match="actual completed game"):
        guard.require_official_terminal_target(snapshot, "forced_pass_deadlock")


def test_changed_rules_source_rejected(monkeypatch):
    monkeypatch.setattr(guard, "sha256", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="source changed"):
        guard.make_environment()


def test_helper_selfcheck_is_not_a_training_launch_qualification():
    receipt = guard.self_check()
    assert receipt["status"] == "helper_self_check_passed"
    assert receipt["verified_source_files"] == 8
    assert receipt["checked_initial_actions"] == 30
    assert receipt["training_pipeline_qualified"] is False
