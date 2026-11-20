"""Independent source, RNG, fixed-choice arithmetic and sampled replay for159.

Source prefixes are reconstructed from recorded legal actions, without rerunning
158 policies. Every new world is independently reconstructed. Only the sealed
world indices (0,16 selection;0,64 evaluation) receive complete trajectory
replay; this audit must not be described as all-trajectory verification.
"""
from collections import Counter
from pathlib import Path
import math
import re

import numpy as np
from scipy.stats import t as student_t

from splendor_ai.alphazero import learned_continuation_158 as h
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from scripts.analysis import verify_learned_continuation_158 as a158

old = h.old
SOURCE_BINDING = "06fcd7d298a3099b11206ff3cc5c4343b5c9c1fb01b3ef975c87fafa015d40ab"
POLICIES = ("GG", "LL")
WORLD_COUNTS = {"selection": 32, "evaluation": 128}
REPLAY_WORLDS = {"selection": (0, 16), "evaluation": (0, 64)}


def _sha(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), value


def _integer(value, low=0, high=None):
    assert type(value) is int and value >= low and (high is None or value <= high), value


def independent_stream(sample, stage):
    assert stage in WORLD_COUNTS
    return old.digest(["159-root-stream-v1", stage, sample["pair"]])


def independent_seeds(sample, stage):
    state = old.deserialize_state_snapshot(sample["snapshot"])
    assert state.current_player == sample["actor"]
    belief = old.digest(canonical_public_belief_key(state, state.current_player))
    stream = independent_stream(sample, stage)
    return [f"159-world-v1|{stage}|{stream}|{belief}|{i:04d}"
            for i in range(WORLD_COUNTS[stage])]


def verify_sample(sample):
    """Bind a sampled position to the frozen source, independently replaying its prefix."""
    _integer(sample["index"])
    _integer(sample["source_index"], high=401)
    _integer(sample["turn"], high=399)
    spec = a158.independent_spec("main", sample["source_index"])
    assert sample["source_spec"] == spec
    assert sample["pair"] == spec["pair"] and sample["model_id"] == spec["model_id"]
    game_path, root_path = Path(sample["source_game_path"]), Path(sample["source_root_path"])
    assert game_path.name == f"game_{sample['source_index']:04d}.json"
    assert root_path.name == f"turn_{sample['turn']:03d}.json"
    assert root_path.parent.name == f"game_{sample['source_index']:04d}"
    assert old.file_hash(game_path) == sample["source_game_sha256"]
    assert old.file_hash(root_path) == sample["source_root_sha256"]
    game, root = old.read_json(game_path), old.read_json(root_path)
    assert game["manifest"] == root["manifest"] == SOURCE_BINDING
    assert "failure" not in game and "failure" not in root and game["spec"] == spec
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state()
    assert old.digest(old.serialize_state_snapshot(state)) == game["initial_state_sha256"]
    visits, passes, idle, selected_move = Counter(), 0, 0, None
    for offset, move in enumerate(game["trace"]):
        assert not state.terminal and state.turn_index < 400
        assert move["turn"] == state.turn_index == offset
        assert move["actor"] == state.current_player
        assert move["state_sha256"] == old.digest(old.serialize_state_snapshot(state))
        visits[old.state_signature(state)] += 1
        assert visits[old.state_signature(state)] < 4
        legal = env.legal_actions(state)
        assert legal and old.decode_action(move["action"]) in legal
        if state.turn_index == sample["turn"]:
            selected_move = move
            break
        action = old.decode_action(move["action"])
        nxt = env.step(state, action)
        passes = passes + 1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if old.is_progress_transition(state, nxt, action) else idle + 1
        assert passes < 2 and idle < 60
        state = nxt
    assert selected_move is not None, "Sample turn absent from its source game"
    assert selected_move["root_sha256"] == sample["source_root_sha256"]
    snapshot = old.serialize_state_snapshot(state)
    assert old.digest(snapshot) == old.digest(sample["snapshot"]) == old.digest(root["snapshot"])
    assert state.current_player == sample["actor"]
    phase = 0 if state.turn_index < 20 else 1 if state.turn_index < 40 else 2
    assert sample["phase"] == phase
    planner = "LL" if state.current_player == spec["candidate_seat"] else "GG"
    assert sample["source_planner"] == planner
    if planner == "LL":
        assert root["continuation_model"] == sample["model_id"]
    else:
        assert "continuation_model" not in root
    cell = ((a158.FIXED_MODELS.index(spec["model_id"])*3+phase)*2+state.current_player)*2+(planner == "LL")
    assert sample["cell"] == cell
    schedule = a158.independent_schedule(state, spec["search_stream"])
    assert root["schedule"] == sample["source_schedule"] == schedule
    offered = a158.a154.independent_candidates(state, "155-live-v1", schedule["candidate_nonce"])
    assert root["candidates"] == sample["candidates"] == offered
    alternative = a158.hidden_alternative(state)
    assert a158.independent_schedule(alternative, spec["search_stream"]) == schedule
    assert a158.a154.independent_candidates(alternative, "155-live-v1", schedule["candidate_nonce"]) == offered
    _integer(root["choice"], high=len(offered["actions"])-1)
    assert offered["actions"][root["choice"]] == selected_move["action"]
    return {"index": sample["index"], "pair": sample["pair"], "turn": sample["turn"],
            "prefix_moves": sample["turn"], "support": len(offered["actions"]),
            "source_game_sha256": sample["source_game_sha256"],
            "source_root_sha256": sample["source_root_sha256"]}


def _outcome(outcome, actor, policy, source_turn):
    common = {"reason", "winner", "turns", "trace_sha256", "final_state_sha256", "wdl"}
    assert set(outcome) == common | ({"decision_sha256", "learned_decisions"} if policy == "LL" else set())
    assert outcome["reason"] in ("completed", "forced_pass_deadlock")
    assert outcome["winner"] is None or type(outcome["winner"]) is int and outcome["winner"] in (0, 1)
    _integer(outcome["turns"], 1, 400-source_turn)
    assert type(outcome["wdl"]) in (int, float) and math.isfinite(outcome["wdl"])
    assert outcome["wdl"] == (0. if outcome["winner"] is None else 1. if outcome["winner"] == actor else -1.)
    for field in ("trace_sha256", "final_state_sha256"):
        _sha(outcome[field])
    if policy == "LL":
        _sha(outcome["decision_sha256"])
        _integer(outcome["learned_decisions"])
        assert outcome["learned_decisions"] == outcome["turns"]-1


def verify_record(sample, record, stage, selection=None, replay=False):
    """Audit every world/outcome and optionally replay the fixed branch subset.

    File hashes, global selection-lock ordering and manifest identity must also
    be checked by the calling pipeline. This pure-record utility permits a null
    evaluation selection_sha256 for isolated in-memory test fixtures only.
    """
    assert stage in WORLD_COUNTS
    assert record["stage"] == stage and record["index"] == sample["index"]
    assert record["pair"] == sample["pair"] and "failure" not in record
    _sha(record["manifest"])
    state = old.deserialize_state_snapshot(sample["snapshot"])
    assert not state.terminal and state.current_player == sample["actor"] and state.turn_index == sample["turn"]
    assert sample["model_id"] in a158.FIXED_MODELS
    actions = sample["candidates"]["actions"]
    assert 1 <= len(actions) <= 4
    assert record["stream"] == independent_stream(sample, stage)
    assert set(record["timing"]) == {"wall", "cpu"}
    assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in record["timing"].values())
    if stage == "selection":
        assert selection is None and record["selection_sha256"] is None
        expected_indices = list(range(len(actions)))
        same_choice = False
    else:
        assert selection is not None
        verify_record(sample, selection, "selection", replay=False)
        assert record["manifest"] == selection["manifest"]
        if record["selection_sha256"] is not None:
            _sha(record["selection_sha256"])
        expected_indices = list(dict.fromkeys([selection["choices"]["LL"], selection["choices"]["GG"]]))
        same_choice = selection["choices"]["LL"] == selection["choices"]["GG"]
        assert type(record["same_choice"]) is bool and record["same_choice"] == same_choice
    assert record["action_indices"] == expected_indices
    assert len(record["worlds"]) == (0 if same_choice else WORLD_COUNTS[stage])
    totals = {policy: [0.] * len(expected_indices) for policy in POLICIES}
    moves, continuations, replayed = 0, 0, 0
    reasons = Counter()
    alternative = a158.hidden_alternative(state)
    for wi, (seed, row) in enumerate(zip(independent_seeds(sample, stage), record["worlds"])):
        assert row["seed"] == seed and set(row["outcomes"]) == set(POLICIES)
        world = a158.a154.independent_world(state, seed)
        world_sha = old.digest(old.serialize_state_snapshot(world))
        assert row["state_sha256"] == world_sha
        assert old.digest(old.serialize_state_snapshot(a158.a154.independent_world(alternative, seed))) == world_sha
        legal = old.SplendorEnv().legal_actions(world)
        for policy in POLICIES:
            outcomes = row["outcomes"][policy]
            assert len(outcomes) == len(expected_indices)
            for j, (action_index, outcome) in enumerate(zip(expected_indices, outcomes)):
                _integer(action_index, high=len(actions)-1)
                action = old.decode_action(actions[action_index])
                assert action in legal
                _outcome(outcome, state.current_player, policy, state.turn_index)
                totals[policy][j] += outcome["wdl"]
                reasons[outcome["reason"]] += 1
                moves += outcome["turns"]
                continuations += 1
                if replay and wi in REPLAY_WORLDS[stage]:
                    model = h.MODELS_CACHE[sample["model_id"]] if policy == "LL" else None
                    expected = a158.replay_branch(world, action, state.current_player, record["stream"], wi, model)
                    assert outcome == expected, (sample["index"], stage, wi, policy, action_index, outcome, expected)
                    replayed += 1
    if stage == "selection":
        assert set(record["means"]) == set(record["choices"]) == set(POLICIES)
        for policy in POLICIES:
            means = [total/WORLD_COUNTS[stage] for total in totals[policy]]
            a158.compare(record["means"][policy], means)
            choice = max(range(len(means)), key=lambda i: means[i])
            _integer(record["choices"][policy], high=len(actions)-1)
            assert record["choices"][policy] == choice
    return {"index": sample["index"], "stage": stage, "worlds": len(record["worlds"]),
            "continuations": continuations, "moves": moves, "reasons": dict(reasons),
            "replayed_branches": replayed, "same_choice": same_choice}


def independent_metrics(selection, evaluation):
    """Rebuild fixed-move effects directly from winner identities, in score units."""
    left, right = selection["choices"]["LL"], selection["choices"]["GG"]
    same = left == right
    assert evaluation["same_choice"] == same
    result = {"changed": not same}
    for policy in POLICIES:
        result[f"in_sample_d_{policy}"] = (selection["means"][policy][left]-selection["means"][policy][right])/2
        # WDL has already been independently checked against winner and root
        # actor by verify_record. Divide each difference by2, not the final CI.
        if same:
            assert evaluation["worlds"] == []
            differences = [0.]*128
        else:
            assert len(evaluation["worlds"]) == 128
            positions = [evaluation["action_indices"].index(i) for i in (left, right)]
            differences = [(world["outcomes"][policy][positions[0]]["wdl"]-
                            world["outcomes"][policy][positions[1]]["wdl"])/2
                           for world in evaluation["worlds"]]
        result[f"d_{policy}"] = sum(differences)/128
        result[f"first_half_d_{policy}"] = sum(differences[:64])/64
        result[f"second_half_d_{policy}"] = sum(differences[64:])/64
    result["interaction"] = result["d_LL"]-result["d_GG"]
    result["first_half_interaction"] = result["first_half_d_LL"]-result["first_half_d_GG"]
    result["second_half_interaction"] = result["second_half_d_LL"]-result["second_half_d_GG"]
    result["interaction_crossproduct"] = result["first_half_interaction"]*result["second_half_interaction"]
    return result


def _quantile(values, q):
    """Independent linear order-statistic interpolation."""
    ordered = sorted(values)
    at = (len(ordered)-1)*q
    low, high = math.floor(at), math.ceil(at)
    return float(ordered[low]+(at-low)*(ordered[high]-ordered[low]))


def independent_classification(estimates):
    g, l = estimates["d_GG"]["simultaneous_ci95"], estimates["d_LL"]["simultaneous_ci95"]
    precise = all(estimates[f"d_{p}"]["half_width"] <= .025 for p in POLICIES)
    if not precise:
        label = "inconclusive_precision_target_not_met"
    elif g[1] < -.025 and l[0] > .025:
        label = "material_conditional_preference_reversal"
    elif g[0] > .025 and l[1] < -.025:
        label = "material_reverse_direction_conditional_preference_reversal"
    elif min(g[0], l[0]) > .025:
        label = "learned_selected_moves_better_under_both_conditional_objectives"
    elif max(g[1], l[1]) < -.025:
        label = "greedy_selected_moves_better_under_both_conditional_objectives"
    elif min(g[0], l[0]) > -.025 and max(g[1], l[1]) < .025:
        label = "small_average_fixed_choice_effects_only"
    else:
        label = "inconclusive_for_material_classification"
    return {"label": label,
            "precision_target_met": precise,
            "caution": "Conditional mean effects only; no gameplay oracle, local equivalence or causal explanation of158."}


def independent_statistics(rows):
    """Reconstruct primary Welch intervals and exploratory bootstrap independently.

    Primary variances use explicit squared deviations and finite-cell degrees
    of freedom. Exploratory intervals use the same declared NumPy RNG, with
    frequency weights and order-statistic quantiles rather than the collector's
    gather/mean/quantile implementation. Centered bootstrap variation receives
    sqrt(5/4) calibration. No production159 statistical code is imported. The
    full fixed panel is mandatory, including unchanged choices.
    """
    assert len(rows) == 180
    assert {r["index"] for r in rows} == set(range(180))
    assert len({r["pair"] for r in rows}) == 180
    assert Counter(r["cell"] for r in rows) == {i: 5 for i in range(36)}
    keys = ["d_GG", "d_LL", "interaction", "interaction_crossproduct", "changed"]
    for row in rows:
        assert type(row["changed"]) is bool
        expected = ((a158.FIXED_MODELS.index(row["model_id"])*3+row["phase"])*2+row["actor"])*2+POLICIES.index(row["source_planner"])
        assert row["cell"] == expected
        assert all(math.isfinite(row[key]) for key in keys)
        assert all(-1 <= row[f"d_{p}"] <= 1 for p in POLICIES)
        assert abs(row["interaction"]-(row["d_LL"]-row["d_GG"])) < 1e-12
        assert -4 <= row["interaction_crossproduct"] <= 4
        if not row["changed"]:
            assert all(row[key] == 0 for key in keys)
    blocks = [[r for r in rows if r["cell"] == cell] for cell in range(36)]
    values = np.array([[[row[key] for key in keys] for row in block] for block in blocks], dtype=float)
    rng = np.random.default_rng(1593000000)
    draws = rng.integers(5, size=(50000, 36, 5))
    bootstrap = np.zeros((50000, len(keys)))
    for item in range(5):
        frequencies = (draws == item).sum(axis=2)
        # Elementwise accumulation avoids activating another BLAS/OpenMP
        # runtime alongside the already initialized frozen Torch models.
        for cell in range(36):
            bootstrap += frequencies[:, cell, None]*values[cell, item, :]
    bootstrap /= 180
    observed = np.array([sum(row[key] for row in rows)/180 for key in keys])
    bootstrap = observed+(bootstrap-observed)*math.sqrt(5/4)
    estimates = {}
    for column, key in enumerate(keys):
        estimate = sum(row[key] for row in rows)/180
        estimates[key] = {"mean": estimate}
        if key in ("d_GG", "d_LL"):
            components = []
            for block in blocks:
                mean = sum(row[key] for row in block)/5
                components.append(sum((row[key]-mean)**2 for row in block)/20)
            variance = sum(components)/(36**2)
            se = math.sqrt(variance)
            df = None if variance == 0 else sum(components)**2/sum(v*v/4 for v in components)
            width = 0. if variance == 0 else float(student_t.ppf(.9875, df))*se
            estimates[key].update(simultaneous_ci95=[estimate-width, estimate+width], half_width=width,
                                  standard_error=se, degrees_of_freedom=df,
                                  method="Bonferroni_pair_of_97.5_percent_marginal_Welch_Satterthwaite_t_intervals")
        else:
            estimates[key]["exploratory_ci95"] = [_quantile(bootstrap[:, column], q) for q in (.025, .975)]
    strata = {}
    for name in ("phase", "model_id", "actor", "source_planner"):
        strata[name] = {}
        for value in sorted({row[name] for row in rows}):
            part = [row for row in rows if row[name] == value]
            strata[name][str(value)] = {"roots": len(part), **{key: sum(row[key] for row in part)/len(part) for key in keys}}
    return {"roots": 180, "independent_source_pairs": 180, "estimates": estimates,
            "classification": independent_classification(estimates), "descriptive_subgroups": strata,
            "units": "win1_draw0.5_loss0; action difference learned-selected minus greedy-selected",
            "uncertainty": "Source-pair stratified resampling, conditional on3 fixed models and the balanced visited-state panel."}
