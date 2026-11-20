"""Corrected planner synthetic fixtures: no checkpoint inference or played games."""
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from tools.corrected_planner_qualification_20261018 import local
from tools.external_strength_screen_20261018 import rules
from splendor_ai.alphazero import strong_policy
from splendor_ai.encoding import encode_legal_action_features, encode_public_observation_tensor
from splendor_ai.engine.data import build_base_deck_by_tier


class FakeRaw:
    def score_actions(self, state, legal):
        return tuple(-float(i) for i in range(len(legal)))


class FakeProposal:
    config = SimpleNamespace(observation_size=500, action_feature_size=200)
    def __call__(self, observation, features, mask):
        return torch.arange(features.shape[1], dtype=torch.float32)[None], torch.zeros(1)


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(local, "_INITIALIZED", True)
    monkeypatch.setattr(local, "_RAW", FakeRaw())
    monkeypatch.setattr(local, "_MODELS", {f"anchor_{f}": FakeProposal() for f in local.FITS})
    monkeypatch.setattr(local, "_ARTIFACTS", {f"anchor_{f}": {"sha256": str(f), "source_fit": f} for f in local.FITS})


def context(state, fit=161001):
    pair = next(p for p in range(201) if local.historical._public_fit_and_stream(p)[0] == fit)
    found, stream = local.historical._public_fit_and_stream(pair)
    return {"fit": found, "stream": stream, "identity": ["main", 2 * pair, state.turn_index]}


def observe(state, fit=161001):
    return local.make_observer_payload(rules.serialize_snapshot(state), context(state, fit))


def remove_cards(state, chosen):
    ids = {c.card_id for c in chosen}
    for tier in (1, 2, 3):
        state.visible_tier_cards[tier] = [c for c in state.visible_tier_cards[tier] if c.card_id not in ids]
        state.hidden_tier_decks[tier] = [c for c in state.hidden_tier_decks[tier] if c.card_id not in ids]
        while len(state.visible_tier_cards[tier]) < 4 and state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop())
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])


def give_cards(state, seat, chosen):
    remove_cards(state, chosen)
    for card in chosen:
        player = state.players[seat]
        player.purchased_cards.append(card)
        player.bonuses[card.bonus_color] += 1
        player.score += card.points


def near_terminal():
    state = rules.CorrectedEnv(seed=911).initial_state()
    give_cards(state, 0, [c for c in rules.corrected_cards_by_id().values() if c.points == 5][:3])
    state.current_player, state.turn_index, state.pending_round_end = 1, 1, True
    rules.validate_state(state)
    return state


def noble_pass_fixture():
    """Component-conserving control-flow edge; not an admissible match root.

    Its pending flag intentionally delays an already-earned finish to expose
    two PASS turns in the termination counter. Full replay fixtures below use
    separately validated pending/round accounting.
    """
    state = rules.CorrectedEnv(seed=911).initial_state()
    noble = state.nobles[0]
    all_cards = list(rules.corrected_cards_by_id().values())
    owned = []
    for color, count in noble.requirement.items():
        owned += sorted([c for c in all_cards if c.bonus_color == color and c.points == 0], key=lambda c: c.card_id)[:count]
    own_ids = {c.card_id for c in owned}
    give_cards(state, 0, owned)
    give_cards(state, 1, [c for c in all_cards if c.card_id not in own_ids])
    state.players[1].nobles = state.nobles[1:]
    state.players[1].score += sum(n.points for n in state.players[1].nobles)
    state.nobles = [noble]
    for seat, counts in ((0, {"white": 4, "blue": 4, "green": 2}), (1, {"green": 2, "red": 4, "black": 4})):
        for color, amount in counts.items():
            state.players[seat].tokens[color] = amount
            state.bank_tokens[color] -= amount
    rules.validate_state(state)
    return state


def noble_finish_fixture():
    """Admissible counterfactual ledger: one noble triggers the final round."""
    state = rules.CorrectedEnv(seed=911).initial_state()
    cards = list(rules.corrected_cards_by_id().values())
    chosen = [c for c in cards if c.points == 5][:2] + [c for c in cards if c.points == 4][:1]
    give_cards(state, 0, chosen)
    needed = state.nobles[0].requirement
    zeros = []
    for color, count in needed.items():
        n = max(count - state.players[0].bonuses[color], 0)
        zeros += sorted([c for c in cards if c.points == 0 and c.bonus_color == color], key=lambda c: c.card_id)[:n]
    give_cards(state, 0, zeros)
    assert state.players[0].score == 14
    from tools.external_strength_screen_20261018 import audit
    audit.validate_ledger(rules.serialize_snapshot(state))
    return state


def test_corrected_observer_roundtrip_and_historical_objects_unchanged():
    historical = {c.card_id: c for cards in build_base_deck_by_tier().values() for c in cards}
    state = near_terminal()
    rebuilt = local.state_from_observer_payload(observe(state))
    local.assert_corrected_cards(rebuilt)
    assert historical["T1_05"].cost == {"blue": 2, "green": 2}
    assert rules.corrected_cards_by_id()["T1_05"].cost == {"blue": 2, "black": 2}
    bad = deepcopy(observe(state)); bad["catalogue_sha256"] = "wrong"
    with pytest.raises(ValueError): local.state_from_observer_payload(bad)
    bad = deepcopy(observe(state)); bad["observation"]["hidden_tier_decks"] = {}
    with pytest.raises(ValueError): local.state_from_observer_payload(bad)


def test_dummy_hidden_and_all_world_objects_keep_corrected_faces():
    state = rules.CorrectedEnv(seed=911).initial_state()
    card = state.hidden_tier_decks[1].pop(); state.deck_counts[1] -= 1
    state.players[1].reserved_cards.append(card); state.players[1].reserved_card_sources.append("deck")
    variant = state.copy_shallow()
    variant.players[1].reserved_cards[0], variant.hidden_tier_decks[1][0] = variant.hidden_tier_decks[1][0], card
    for deck in variant.hidden_tier_decks.values(): deck.reverse()
    first, second = observe(state), observe(variant)
    assert first == second
    a, b = local.state_from_observer_payload(first), local.state_from_observer_payload(second)
    for seed in (0, 1, "world-synthetic"):
        wa = local.determinize_for_observer(a, 0, local.random.Random(seed))
        wb = local.determinize_for_observer(b, 0, local.random.Random(seed))
        assert local.serialize_state_snapshot(wa) == local.serialize_state_snapshot(wb)
        local.assert_corrected_cards(wa)
    bad = deepcopy(first); bad["observation"]["players"][1]["reserves"][0]["card_id"] = card.card_id
    with pytest.raises(ValueError, match="private"): local.state_from_observer_payload(bad)


def test_corrected_costs_reach_observation_and_action_features():
    state = rules.CorrectedEnv(seed=911).initial_state()
    card_id = "T1_05"
    location = next((group, i) for group in (state.visible_tier_cards[1], state.hidden_tier_decks[1])
                    for i, c in enumerate(group) if c.card_id == card_id)
    group, index = location
    group[index], state.visible_tier_cards[1][0] = state.visible_tier_cards[1][0], group[index]
    historical = local.old.deserialize_state_snapshot(local.serialize_state_snapshot(state))
    legal = rules.legal_actions(state)
    target = next(a for a in legal if a.action_type.name == "RESERVE_VISIBLE" and a.tier == 1 and a.market_index == 0)
    before = encode_legal_action_features(historical, [target], 0)[0].vector
    after = encode_legal_action_features(state, [target], 0)[0].vector
    assert before != after
    assert encode_public_observation_tensor(historical, 0).vector != encode_public_observation_tensor(state, 0).vector


def test_compulsory_noble_pass_is_progress_not_a_deadlock(fake_models):
    state = noble_pass_fixture()
    legal = rules.legal_actions(state)
    assert len(legal) == 1 and legal[0].action_type.name == "PASS" and legal[0].noble_id is not None
    result, steps = local.trajectory(state, legal[0], capture=True)
    assert result["reason"] == "completed" and result["turns"] == 2
    assert steps[0]["forced"] is True and steps[0]["scores_sha256"] is None
    assert steps[1]["forced"] is False and steps[1]["scores_sha256"] == local.old.digest([-0.0])
    assert steps[0]["action"]["noble_id"] is not None and steps[1]["action"]["noble_id"] is None


@pytest.mark.parametrize("fit", local.FITS)
def test_all_three_fits_exact_four_by128_serial_path_and_ties(fake_models, monkeypatch, fit):
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical engine/default reconstruction reached")
    monkeypatch.setattr(local.old, "SplendorEnv", forbidden)
    monkeypatch.setattr(strong_policy, "SplendorEnv", forbidden)
    monkeypatch.setattr(local.historical, "build_base_deck_by_tier", forbidden)
    old_raw, old_models = local.old._RAW, dict(local.arena._MODELS)
    traces = []
    decision = local.choose(observe(near_terminal(), fit), trace_sink=traces.append)
    record = decision["record"]
    assert record["proposal"]["fit"] == fit
    assert len(record["worlds"]) == 128 and len(record["candidates"]["actions"]) == 4
    assert len(traces) == 512 and all(t["turns"] == 1 for t in traces)
    assert [(t["world_index"], t["candidate_index"]) for t in traces] == [(w, c) for w in range(128) for c in range(4)]
    assert record["choice"] == 0 and record["means"] == [-1.] * 4
    assert decision["action"] == record["candidates"]["actions"][0]
    assert record["proposal"]["candidates"]["keys"][:3] == record["proposal"]["reference"]["keys"][:3]
    assert local.old._RAW is old_raw and local.arena._MODELS == old_models
    assert record["profile"] == local.PROFILE and record["ruleset_id"] == rules.RULESET_ID


def test_small_support_is_complete_and_invalid_breadth_rejected(fake_models):
    state = noble_pass_fixture()
    decision = local.choose(observe(state))
    assert len(decision["record"]["candidates"]["actions"]) == 1
    assert all(len(w["outcomes"]) == 1 for w in decision["record"]["worlds"])
    with pytest.raises(ValueError, match="four-candidate"): local.choose(observe(state), breadth=True)


def test_boundary_failure_cannot_return_partial_decision(fake_models):
    class Stopped(Exception): pass
    checks = []
    def boundary():
        checks.append(1)
        if len(checks) == 4: raise Stopped()
    with pytest.raises(Stopped): local.choose(observe(near_terminal()), boundary=boundary)


def test_nonfinite_or_incomplete_raw_scores_fail_closed(fake_models, monkeypatch):
    state = near_terminal(); legal = rules.legal_actions(state)
    monkeypatch.setattr(local._RAW, "score_actions", lambda *a: [float("nan")] * len(legal))
    with pytest.raises(ValueError, match="complete035"): local.score_actions(state, legal)


@pytest.mark.parametrize("factory", [near_terminal, noble_finish_fixture])
def test_complete_adapter_record_and_sidecar_pass_independent_replay(fake_models, tmp_path, factory):
    from tools.corrected_planner_qualification_20261018 import verify
    observer = observe(factory())
    traces = []
    decision = local.choose(observer, trace_sink=traces.append)
    path = tmp_path / "continuations.jsonl.gz"
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=3, mtime=0) as output:
            for row in traces:
                output.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    result = {"decision": decision, "trace": {"records": len(traces), "bytes": path.stat().st_size,
              "sha256": verify.file_hash(path)}, "ruleset_id": rules.RULESET_ID,
              "catalogue_sha256": rules.CATALOGUE_SHA256, "profile_id": local.PROFILE,
              "observer_sha256": local.old.digest(observer)}
    proof = verify.audit_decision(observer, result, path)
    assert proof["status"] == "passed" and proof["audited_continuations"] == len(traces)
    assert proof["raw035_choices_reexecuted"] == sum(t["turns"] - 1 for t in traces)
