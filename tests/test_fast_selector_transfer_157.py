import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor_ai.alphazero import fast_selector_157 as f
from splendor_ai.alphazero import run_fast_selector_transfer as p

spec = importlib.util.spec_from_file_location("audit157_test", "scripts/analysis/verify_fast_selector_transfer_157.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


@pytest.fixture(scope="module", autouse=True)
def initialize():
    f.old.initialize_worker()


def synthetic_rows():
    generator = torch.Generator().manual_seed(157999)
    config = f.old._RAW._model.config
    result = []
    for i in range(12):
        n = 3+i%3
        groups = torch.tensor([0, 1, 0]+list(range(2, n-1)))
        members = [torch.where(groups == j)[0] for j in range(min(3, n-1))]
        result.append({"observation": torch.rand(config.observation_size, generator=generator),
                       "features": torch.rand((n, config.action_feature_size), generator=generator),
                       "groups": groups, "group_count": n-1, "members": members,
                       "id": f"synthetic:{i}", "pair": "155:1" if i < 8 else "155:2", "split": "train" if i < 8 else "validation",
                       "teacher": i%len(members), "seat": i%2, "phase": i%3,
                       "privacy": {"own": i%2, "opponent": (i//2)%2}})
    return result


def test_split_pair_integrity_and_exposure():
    split = f.split_membership()
    assert split == a.independent_split() and len(split) == 400 and split["156:0"] == "train"
    for exp in (155, 156):
        assert sum(split[f"{exp}:{i}"] == "validation" for i in range(200)) == 40


def test_schedules_couple_only_proposer_and_use_fresh_seeds():
    for stage, n in (("probe", 48), ("main", 4800)):
        schedules = p.schedule(stage)
        assert len(schedules) == n
        assert schedules == [a.independent_spec(stage, i) for i in range(n)]
        assert len({s["stream"] for s in schedules[:12]}) == 2
        assert len({s["environment_seed"] for s in schedules}) == n//12
    assert {s["environment_seed"] for s in p.schedule("main")}.isdisjoint({s["environment_seed"] for s in p.schedule("probe")})


def test_candidate_restricted_loss_and_alias_tie_gradient():
    # Greedy concrete B precedes tied last A, while first canonical group is A.
    row = {"members": [torch.tensor([1]), torch.tensor([0, 2])], "groups": torch.tensor([0, 1, 0, 2]), "teacher": 1}
    logits = torch.tensor([[0., 1., 1., 100.]], requires_grad=True)
    loss, chosen = f.objective(logits, [row], "treatment")
    assert chosen == [0]  # Baseline-first offered order, not canonical-first A.
    assert float(loss.detach()[0]) == pytest.approx(np.log(2))
    loss.mean().backward()
    assert logits.grad[0, 3] == 0 and logits.grad[0, 2] < 0 and logits.grad[0, 1] > 0
    other = logits.detach().clone().requires_grad_()
    audit_loss, _ = a.audit_objective(other, [row], "treatment")
    audit_loss.mean().backward()
    assert torch.equal(logits.grad, other.grad)


def test_single_support_retained_zero_loss():
    row = {"members": [torch.tensor([0, 1])], "teacher": 0}
    logits = torch.tensor([[.1, .2]], requires_grad=True)
    losses, choice = f.objective(logits, [row], "treatment")
    assert losses.item() == 0 and choice == [0]
    losses.mean().backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_batching_padding_all_rows_and_paired_sequence():
    rows = synthetic_rows()*13
    for epoch in range(4):
        batches = f.batches(rows, 157001, epoch)
        assert batches == a.audit_batches(rows, 157001, epoch)
        assert sorted(i for b in batches for i in b) == list(range(len(rows)))
    batch = synthetic_rows()[:5]
    assert all(torch.equal(x, y) for x, y in zip(f.collate(batch), a.audit_batch_tensors(batch)))
    assert f.checkpoint_updates(19456)[0] == 0 and f.checkpoint_updates(19456)[-1] == 1216


def test_warm_controller_exact_and_hidden_inputs():
    env = f.old.SplendorEnv(seed=1579800000)
    state = env.initial_state()
    model = f.StrongPolicyInference(f.old.CHECKPOINT).model
    # Both players take a blind reserve, exercising own-visible/other-hidden inputs.
    for _ in range(2):
        action = next(x for x in env.legal_actions(state) if x.action_type.name == "RESERVE_DECK")
        state = env.step(state, action)
    own_changed = state.copy_shallow()
    card = own_changed.players[state.current_player].reserved_cards[0]
    own_changed.players[state.current_player].reserved_cards[0], own_changed.hidden_tier_decks[card.tier][0] = (
        own_changed.hidden_tier_decks[card.tier][0], card)
    before = a.independent_inputs(state, model.config)
    after = a.independent_inputs(own_changed, model.config)
    assert not torch.equal(before[0], after[0]) or not torch.equal(before[1], after[1])
    for _ in range(8):
        raw = f.old._RAW.choose_action(env, state)
        action, evidence = f.choose(state, "157-test-stream", model)
        replayed, alternate = a.independent_decision(state, "157-test-stream", model)
        assert action == replayed == raw
        a.compare(evidence, alternate)
        state = env.step(state, raw)


@pytest.mark.parametrize("arm", f.ARMS)
def test_complete_small_fit_independent_replay_and_head_unchanged(tmp_path, monkeypatch, arm):
    rows = synthetic_rows()
    # Split labels independently reproduced without accessing protected source labels.
    for row in rows:
        row["pair"] = next(key for key, value in a.independent_split().items() if value == row["split"])
    monkeypatch.setattr(f, "OUTPUT", tmp_path)
    monkeypatch.setattr(a, "OUTPUT", tmp_path)
    monkeypatch.setattr(f, "load_data", lambda: copy.deepcopy(rows))
    f.fit_one(arm, 157001)
    a.replay_fit(arm, 157001)
    report = f.old.read_json(tmp_path/"fits"/f"{arm}_157001"/"audit.json")
    assert report["complete_fit_identical"] and len(report["curves"]) == 5


def test_selection_pair_weighting_not_turn_weighting():
    class Fixed(torch.nn.Module):
        def forward(self, observation, features, mask):
            return features[:, :, 0], torch.zeros(len(observation))
    rows = synthetic_rows()[:5]
    for i, row in enumerate(rows):
        row["pair"] = "a" if i < 4 else "b"
    report = f.validation(Fixed(), rows, "treatment")
    groups = [[r["loss"] for r in report["rows"] if r["pair"] == pair] for pair in ("a", "b")]
    expected = sum(sum(group)/len(group) for group in groups)/2
    assert report["pair_loss"] == expected
    assert report["pair_loss"] != pytest.approx(sum(r["loss"] for r in report["rows"])/5)


def test_complete_recorded_source_export_and_private_replay(tmp_path, monkeypatch):
    # Only already exposed156 pair0; no protected validation labels inspected.
    module = f.SOURCES[156]
    path = module.game_path("main", 0)
    game = f.old.read_json(path)
    source = {"experiment": 156, "index": 0, "pair": "156:0", "path": path.as_posix(),
              "sha256": f.old.file_hash(path), "parent_binding": game["manifest"],
              "roots": [{"turn": m["turn"], "sha256": m["root_sha256"]} for m in game["trace"] if m["actor"] == game["spec"]["candidate_seat"]]}
    monkeypatch.setattr(f, "OUTPUT", tmp_path)
    exported = f.export_game(source, {"156:0": "train"})
    audit = a.source_game(source, exported)
    assert audit["coverage"]["rows"] == exported["rows"] > 0


def test_full_fast_game_replay_and_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(a, "OUTPUT", tmp_path)
    model = f.StrongPolicyInference(f.old.CHECKPOINT).model
    monkeypatch.setattr(p, "_MODELS", {"treatment_157001": model})
    spec = {**p.schedule("probe")[0], "environment_seed": 1579800001, "stream": "157-test-game"}
    monkeypatch.setattr(a, "independent_spec", lambda stage, index: spec)
    f.old.save_new(tmp_path/"manifest.json", {"test": True})
    f.old.save_new(tmp_path/"test_launch.json", {"test": True})
    binding = f.old.digest({"test": True})
    played = p.play_game("test", spec, binding)
    replay = a.replay_game("test", 0, played["sha256"])
    assert f.old.read_json(Path(replay["path"]))["game"]["reason"] in f.old.VALID_NATURAL_TERMINATIONS
    path = p.game_path("test", 0)
    game = f.old.read_json(path)
    game["trace"][0]["evidence"]["offered_scores"][0] += .1
    f.old.atomic_write(path, game)
    with pytest.raises(AssertionError):
        a.replay_game("test", 0, f.old.file_hash(path))


def fake_games(treatment, control):
    return [{**s, "score": float(s["pair"] < (treatment if s["arm"] == "treatment" else control)),
             "margin": 1 if s["pair"] < treatment else -1} for s in p.schedule("main")]


def test_cluster_statistics_and_no_damaged_control_pass():
    rows = fake_games(260, 180)
    stats = p.statistics(rows)
    a.compare(stats, a.independent_statistics(rows))
    assert stats["practical_pass"] and stats["attribution_pass"]
    assert stats["treatment_ci"][1]-stats["treatment_ci"][0] > .07  # 400 clusters, not4800 independent games.
    failure = p.statistics(fake_games(192, 120))
    assert failure["attribution_pass"] and not failure["practical_pass"]
    with pytest.raises(AssertionError):
        p.statistics(rows[:-1]+[rows[0]])


def test_no_silent_game_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    path = p.game_path("main", 0)
    path.parent.mkdir(parents=True)
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(AssertionError, match="reuse"):
        p.play_game("main", p.schedule("main")[0], "test")
    assert path.read_text() == "preserve"


def test_launch_refuses_changed_test_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "identity", lambda: {"current": True})
    f.old.save_new(tmp_path/"preparation"/"tests.json", {"identity": {"wrong": True}})
    with pytest.raises(AssertionError):
        p.verify_test_evidence()
