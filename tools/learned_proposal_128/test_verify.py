"""Independent sampled-audit contracts and bounded complete replay fixtures."""
import copy
from pathlib import Path
import sys
import time

import pytest

from tools.learned_proposal_128 import experiment as p
from tools.learned_proposal_128 import verify as a

BINDING = "proposal128-isolated-test"


@pytest.fixture(scope="module", autouse=True)
def models():
    p.initialize_models()


def private_state():
    env = p.old.SplendorEnv(seed=2069800000)
    state = env.initial_state()
    for _ in range(2):
        move = next(v for v in env.legal_actions(state)
                    if v.action_type.name == "RESERVE_DECK" and v.tier == 2)
        state = env.step(state, move)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


def synthetic_inventory():
    rows = []
    for arm in ("learned", "original"):
        for fit in a.FITS:
            for phase in range(3):
                for actor in (0, 1):
                    for privacy in range(4):
                        for repeat in range(3):
                            index, turn = len(rows), (0, 20, 40)[phase]+repeat
                            rows.append({"root_id": a.root_id(index, turn), "index": index, "pair": index//2,
                                         "fit": fit, "arm": arm, "phase": phase, "actor": actor,
                                         "turn": turn, "privacy_category": privacy,
                                         "privacy": {"own": privacy & 1, "opponent": (privacy >> 1) & 1},
                                         "support": repeat+1, "worlds": 128, "sha256": "a"*64})
    return rows


def test_identity_hash_sample_stress_missing_cells_and_overlap_accounting():
    inventory = synthetic_inventory()
    plan = a.sample_plan("main", inventory, BINDING, "b"*64)
    assert plan["coverage"]["stress_cells"] == 162
    assert plan["coverage"]["missing_stress_cells"] == []
    assert 144 <= plan["coverage"]["stress_roots"] <= 162
    assert plan["coverage"]["stress_selection_overlaps"] == 162-len(plan["full_stress_roots"])
    for row, selected in zip(inventory, plan["selections"]):
        expected = sorted(sorted(range(128), key=lambda wi: (
            p.old.digest(["learned-proposal-128-audit-world-v1", 2064000000,
                          "main", row["index"], row["turn"], wi]), wi))[:8])
        assert selected["ordinary_world_indices"] == expected and len(set(expected)) == 8
        assert selected["replay_world_indices"] == (list(range(128)) if selected["stress"] else expected)
    for cell in plan["stress_cells"]:
        selected = next(row for row in inventory if row["root_id"] == cell["selected_root"])
        parts = cell["cell"].split("|")
        eligible = [row for row in inventory if row["arm"] == parts[1] and row["fit"] == int(parts[2])]
        if parts[0] == "short_support":
            eligible = [row for row in eligible if row["support"] == int(parts[3])]
        else:
            eligible = [row for row in eligible if (row["phase"], row["actor"], row["privacy_category"]) == tuple(map(int, parts[3:]))]
        priority = lambda row: (p.old.digest(["learned-proposal-128-audit-stress-v1", 2064000000,
                                             "main", row["index"], row["turn"]]), row["root_id"])
        assert selected == min(eligible, key=priority)
    changed = copy.deepcopy(inventory)
    for row in changed:
        row.update(sha256="f"*64, outcome=1, choice=3, elapsed=999.)
    altered = a.sample_plan("main", changed, "changed-binding", "c"*64)
    assert altered["selections"] == plan["selections"] and altered["stress_cells"] == plan["stress_cells"]
    partial = a.sample_plan("main", inventory[:2], BINDING, "d"*64)
    assert partial["coverage"]["missing_stress_cells"] and partial["coverage"]["stress_roots"] <= 2
    with pytest.raises(AssertionError):
        a.sample_plan("main", [inventory[0], inventory[0]], BINDING, "d"*64)
    with pytest.raises(AssertionError):
        a.sample_plan("main", list(reversed(inventory)), BINDING, "d"*64)


@pytest.fixture(scope="module")
def tiny_games(tmp_path_factory):
    base = tmp_path_factory.mktemp("proposal128_games")
    snapshot = p.old.serialize_state_snapshot(private_state())
    p.old.save_new(base / "test_launch.json", {"manifest": BINDING})
    games = [p.play_game("test", spec, BINDING, base, snapshot) for spec in p.specs("test")]
    p.old.save_new(base / "test_games_lock.json", {"manifest": BINDING, "files": games})
    a.seal_sample("test", games, BINDING, base)
    a.validate_sample("test", BINDING, base)
    return base, snapshot, games


def test_all_fits_both_seats_private_refills_and_complete128_replay(tiny_games):
    base, snapshot, games = tiny_games
    widths, branches, recorded = set(), 0, 0
    receipts = []
    for item in games:
        receipt = a.audit_game("test", item["index"], item["sha256"], BINDING, base, snapshot)
        receipts.append(receipt)
        assert {r["planner"] for r in receipt["roots"]} == {"candidate", "reference"}
        assert receipt["roots"][0]["privacy"] == {"own": 1, "opponent": 1}
        # Each tiny root occupies a distinct stress cell, so the ordinary API
        # happens to replay all worlds without claiming an exhaustive protocol.
        assert not receipt["full_inner_replay"] and all(root["stress"] for root in receipt["roots"])
        for root in receipt["roots"]:
            assert root["continuations"] == root["replayed_branches"] == 128*root["support"]
            assert root["replay_world_indices"] == list(range(128))
            branches += root["replayed_branches"]
            recorded += root["continuations"]
            saved = p.old.read_json(p.root_path("test", item["index"], root["turn"], base))
            widths.add(saved["candidates"]["legal_support"])
            assert root["audit_timing"]["ordinary_replay_wall"] > 0
            assert root["audit_timing"]["stress_extra_replay_wall"] > 0
            assert any(v["action_type"] in ("BUY_VISIBLE", "RESERVE_VISIBLE") for v in saved["candidates"]["actions"])
    assert len(widths) > 1 and branches == recorded
    p.checked_audit_coverage(receipts, a.load_sample("test", BINDING, base),
                             p.old.file_hash(base / "test/audit_sample.json"))


def root_copy(tiny_games, target, index=0):
    base, snapshot, _ = tiny_games
    game = p.old.read_json(p.game_path("test", index, base))
    move = game["trace"][0]
    state = p.old.deserialize_state_snapshot(snapshot)
    p.old.save_new(target / "test_launch.json", {"manifest": BINDING})
    path = p.root_path("test", index, move["turn"], target)
    root = p.old.read_json(p.root_path("test", index, move["turn"], base))
    p.old.save_new(path, root)
    ordinary = a.ordinary_indices("test", index, move["turn"])
    selection = {"root_id": a.root_id(index, move["turn"]), "ordinary_world_indices": ordinary,
                 "replay_world_indices": ordinary, "stress": False}
    return game["spec"], state, move, root, path, selection


def test_ordinary_sample_replays_only_eight_whole_worlds_and_reports_limits(tiny_games, tmp_path):
    spec, state, move, root, path, selection = root_copy(tiny_games, tmp_path)
    # A valid-looking trace hash outside the replay sample cannot be verified.
    # This deliberately demonstrates the scope of the sampled assurance claim.
    wi = next(i for i in range(128) if i not in selection["replay_world_indices"])
    root["worlds"][wi]["outcomes"][0]["trace_sha256"] = "0"*64
    p.old.atomic_write(path, root)
    move["root_sha256"] = p.old.file_hash(path)
    result = a.audit_root("test", 0, state, move, spec, BINDING, tmp_path, selection)
    assert result["continuations"] == 128*result["support"]
    assert result["replayed_branches"] == 8*result["support"] and not result["full_inner_replay"]
    assert result["audit_timing"]["stress_extra_replay_wall"] == 0


@pytest.mark.parametrize("damage", ["world", "sampled_trace", "means", "choice", "winner", "proposal",
                                     "last_entry_missing", "last_entry_extra", "bad_timing", "bad_wdl"])
def test_root_semantic_corruption_rejected_after_rehashing(tiny_games, tmp_path, damage):
    spec, state, move, root, path, selection = root_copy(tiny_games, tmp_path)
    if damage == "world":
        root["worlds"][-1]["state_sha256"] = "0"*64
    elif damage == "sampled_trace":
        root["worlds"][selection["replay_world_indices"][0]]["outcomes"][0]["trace_sha256"] = "0"*64
    elif damage == "means":
        root["means"][0] += .125
    elif damage == "choice":
        root["choice"] = (root["choice"]+1) % len(root["means"])
    elif damage == "winner":
        row = root["worlds"][-1]["outcomes"][0]
        row["winner"] = 1-row["winner"] if row["winner"] is not None else 0
    elif damage == "proposal":
        root["proposal"]["learned_logits"][0] += .5
    elif damage == "last_entry_missing":
        root["worlds"][-1]["outcomes"].pop()
    elif damage == "last_entry_extra":
        root["worlds"][-1]["outcomes"].append(root["worlds"][-1]["outcomes"][0])
    elif damage == "bad_timing":
        root["proposal"]["timing"]["learned_extra_wall"] = root["proposal"]["timing"]["wall"]+1
    else:
        root["worlds"][-1]["outcomes"][0]["wdl"] = .25
    p.old.atomic_write(path, root)
    move["root_sha256"] = p.old.file_hash(path)
    with pytest.raises((AssertionError, ValueError)):
        a.audit_root("test", 0, state, move, spec, BINDING, tmp_path, selection)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "biased", "stress", "inventory", "unsealed"])
def test_sample_manifest_tamper_rejected(tiny_games, tmp_path, damage):
    base, _, _ = tiny_games
    # Copy only the small seals; metadata validation below uses original paths.
    sample_path = base / "test/audit_sample.json"
    value = p.old.read_json(sample_path)
    if damage == "missing":
        value["selections"].pop()
    elif damage == "duplicate":
        value["selections"].append(value["selections"][0])
    elif damage == "biased":
        value["selections"][0]["ordinary_world_indices"] = list(range(8))
    elif damage == "stress":
        value["selections"][0]["stress"] = not value["selections"][0]["stress"]
    elif damage == "inventory":
        value["inventory"][0]["privacy_category"] ^= 1
    else:
        value["manifest"] = "unsealed"
    p.old.save_new(tmp_path / "test/audit_sample.json", value)
    lock = p.old.read_json(base / "test_games_lock.json")
    p.old.save_new(tmp_path / "test_games_lock.json", lock)
    seal = p.old.read_json(base / "test/audit_sample_seal.json")
    if damage != "unsealed":
        seal["sample_sha256"] = p.old.file_hash(tmp_path / "test/audit_sample.json")
    p.old.save_new(tmp_path / "test/audit_sample_seal.json", seal)
    if damage == "unsealed":
        with pytest.raises(AssertionError):
            a.load_sample("test", BINDING, tmp_path)
    else:
        actual = a.load_sample("test", BINDING, tmp_path)
        expected = a.sample_plan("test", a.metadata_inventory("test", lock["files"], BINDING, base),
                                 BINDING, p.old.file_hash(base / "test_games_lock.json"))
        with pytest.raises((AssertionError, ValueError)):
            a.compare(actual, expected)


def test_inventory_requires_every_declared_game_and_unique_roots(tiny_games):
    base, _, games = tiny_games
    with pytest.raises(AssertionError):
        a.metadata_inventory("test", games[:-1], BINDING, base)
    with pytest.raises(AssertionError):
        a.metadata_inventory("test", games[:-1]+[games[0]], BINDING, base)
    changed = copy.deepcopy(games)
    changed[0]["sha256"] = "0"*64
    with pytest.raises(AssertionError):
        a.metadata_inventory("test", changed, BINDING, base)


@pytest.mark.parametrize("field,value", [("reason", "max_turns"), ("turns", 0), ("turns", True),
                                         ("turns", 402), ("winner", True), ("winner", 2),
                                         ("wdl", float("nan")), ("trace_sha256", "not-a-hash")])
def test_all_recorded_outcomes_require_complete_natural_consistent_fields(field, value):
    outcome = {"reason": "completed", "turns": 1, "winner": 0, "wdl": 1.,
               "trace_sha256": "a"*64, "final_state_sha256": "b"*64}
    a._check_outcome(outcome, 0, 399)
    outcome[field] = value
    with pytest.raises(AssertionError):
        a._check_outcome(outcome, 0, 399)


@pytest.mark.parametrize("damage", ["outer_state", "outer_winner", "duplicate_root", "missing_root"])
def test_actual_game_or_inventory_corruption_rejected(tiny_games, tmp_path, damage):
    source, snapshot, games = tiny_games
    p.old.save_new(tmp_path / "test_launch.json", {"manifest": BINDING})
    changed = []
    for item in games:
        game = p.old.read_json(p.game_path("test", item["index"], source))
        for move in game["trace"]:
            if damage == "missing_root" and item["index"] == 0 and move == game["trace"][0]:
                continue
            root = p.old.read_json(p.root_path("test", item["index"], move["turn"], source))
            p.old.save_new(p.root_path("test", item["index"], move["turn"], tmp_path), root)
        if item["index"] == 0:
            if damage == "outer_state":
                game["trace"][0]["state_sha256"] = "0"*64
            elif damage == "outer_winner":
                game["winner"] = 1-game["winner"] if game["winner"] is not None else 0
            elif damage == "duplicate_root":
                game["trace"].append(game["trace"][0])
        path = p.game_path("test", item["index"], tmp_path)
        p.old.save_new(path, game)
        changed.append({"index": item["index"], "sha256": p.old.file_hash(path)})
    p.old.save_new(tmp_path / "test_games_lock.json", {"manifest": BINDING, "files": changed})
    with pytest.raises((AssertionError, ValueError, FileNotFoundError)):
        a.seal_sample("test", changed, BINDING, tmp_path)
        a.validate_sample("test", BINDING, tmp_path)
        a.audit_game("test", 0, changed[0]["sha256"], BINDING, tmp_path, snapshot)


def short_state(support, seat):
    state = p.old.SplendorEnv(seed=2069800000).initial_state()
    # Preserve the full card catalog while exposing only token/pass choices.
    for tier in state.visible_tier_cards:
        state.players[0].purchased_cards.extend(state.visible_tier_cards[tier])
        state.players[0].purchased_cards.extend(state.hidden_tier_decks[tier])
        state.visible_tier_cards[tier], state.hidden_tier_decks[tier], state.deck_counts[tier] = [], [], 0
    state.bank_tokens = {color: 0 for color in state.bank_tokens}
    colors = list(state.players[0].bonuses)
    if support >= 2:
        state.bank_tokens[colors[0]] = 4
    if support == 3:
        state.bank_tokens[colors[1]] = 4
    state.current_player, state.start_player = seat, 1-seat
    state.pending_round_end = True
    state.players[1-seat].score = 15
    state.turn_index = 399
    return state


@pytest.mark.parametrize("support", [1, 2, 3])
@pytest.mark.parametrize("candidate", [False, True])
def test_short_support_exact_ties_forced_moves_and_last_turn_fully_replayed(tmp_path, support, candidate):
    spec = p.specs("test")[0]
    state = short_state(support, 0 if candidate else 1)
    p.old.save_new(tmp_path / "test_launch.json", {"manifest": BINDING})
    path = p.root_path("test", 0, 399, tmp_path)
    action, root = p.choose_root(state, spec["search_stream"], path, BINDING, spec["fit"] if candidate else None)
    assert len(root["candidates"]["actions"]) == support
    assert root["choice"] == 0 and len(set(root["means"])) == 1
    move = {"turn": 399, "actor": state.current_player, "state_sha256": p.old.digest(root["snapshot"]),
            "action": p.old.serialize_action(action), "root_sha256": p.old.file_hash(path)}
    selection = {"root_id": a.root_id(0, 399), "ordinary_world_indices": a.ordinary_indices("test", 0, 399),
                 "replay_world_indices": list(range(128)), "stress": True}
    receipt = a.audit_root("test", 0, state, move, spec, BINDING, tmp_path, selection, full_fixture=True)
    assert receipt["replayed_branches"] == 128*support and receipt["moves"] == 128*support
    assert receipt["reasons"] == {"completed": 128*support}


def test_two_hidden_spawn_pools_collect_seal_then_full_fixture_replay(tmp_path):
    base = tmp_path / "synthetic"
    base.mkdir()
    fixture = {"kind": "learned-proposal-128-synthetic-v1", "base": str(base), "binding": BINDING,
               "snapshot": p.old.serialize_state_snapshot(private_state()), "full_fixture_replay": True}
    path = tmp_path / "fixture.json"
    p.old.save_new(path, fixture)
    launch = p.windows.detached_launch([sys.executable, "-u", "-m", "tools.learned_proposal_128.verify", "--fixture", path],
                                       base / "out.txt", base / "err.txt", p.ROOT)
    result_path = base / "fixture_result.json"
    deadline = time.monotonic()+180
    while not result_path.exists():
        if time.monotonic() > deadline:
            logs = [path.read_text(encoding="utf-8") for path in base.rglob("*err*.txt")]
            pytest.fail(f"Synthetic hidden pools timed out PID {launch['pid']}: {logs}")
        time.sleep(.05)
    result = p.old.read_json(result_path)
    assert result["passed"] and result["games"] == 6 and result["roots"] >= 12
    assert result["continuations"] == result["replayed_branches"] and result["full_fixture_replay"]
    for kind in ("test_games", "test_audits"):
        healths = list((base / "workers" / kind).glob("*_health.json"))
        assert healths
        for path in healths:
            health = p.old.read_json(path)
            if sys.platform == "win32":
                assert Path(health["executable"]).name == Path(health["parent_executable"]).name == "pythonw.exe"
                assert health["console_window"] == health["console_process_count"] == 0
