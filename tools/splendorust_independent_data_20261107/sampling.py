"""Fixed, outcome-blind breadth comparison on a new accepted corrected corpus.

Only identities and ordered usable-root indices enter selection. Held-out targets
are never loaded by the fitting path. Corpus acceptance authenticates each game;
the runtime loader rechecks complete per-game root identities before release.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

from tools.splendorust_policy_head_20261104 import data as accepted
from tools.splendorust_policy_head_20261104 import training as accepted_training
from . import common as c
from . import schedule

DOMAIN = "splendorust_independent_data_20261107_v1"
STUDY = DOMAIN
IDENTITY = deepcopy(accepted.IDENTITY)
GAME_COUNTS = {"train": 1024, "validation": 64, "offline": 64}
ARMS = ("narrow", "broad")
SCHEMA = "independent_data_selection_v1"
SEEDS = tuple(schedule.FIT_SEEDS[f"r{r}"] for r in (0, 1))
BOOTSTRAP_SEED = schedule.BOOTSTRAP_SEEDS["offline"]
require, canonical, digest = c.require, c.canonical, c.digest


def fits():
    return [{"fit_id": f"r{r}_{a}_joint", "repetition": r, "breadth": a, "scope": "joint"}
            for r in (0, 1) for a in ARMS]


def fit_identity(repetition, breadth, scope="joint"):
    require(type(repetition) is int and repetition in (0, 1), "fixed integer repetition")
    require(type(breadth) is str and breadth in ARMS and scope == "joint", "fixed joint arm")
    return {"fit_id": f"r{repetition}_{breadth}_joint", "repetition": repetition,
            "breadth": breadth, "scope": scope}


def _sha(value):
    require(type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef"), "full SHA256")
    return value


def root_identity(example):
    require(type(example) is dict and type(example.get("game_index")) is int and
            type(example.get("ply")) is int and 0 <= example["game_index"] < 1152 and
            0 <= example["ply"] < 2000 and example.get("split") in GAME_COUNTS, "strict root identity")
    return {"game_index": example["game_index"], "ply": example["ply"], "split": example["split"],
            "row_sha256": _sha(example["row_sha256"]), "root_input_sha256": _sha(example["root_input_sha256"])}


def relative_thirds(roots):
    require(type(roots) is list and len(roots) >= 3, "all three relative thirds exist")
    n = len(roots)
    return roots[:n // 3], roots[n // 3:2 * n // 3], roots[2 * n // 3:]


def build_selection(games, source_receipt):
    """Pure constructor. Raw targets, outcomes and losses are not ranking inputs."""
    require(type(games) is list and len(games) == 1152, "complete new 1152-game catalogue")
    require(type(source_receipt) is dict, "authenticated corpus receipt")
    require([g["spec"]["game_index"] for g in games] == list(range(1152)), "canonical unique game order")
    roots_by_game, counts, totals = {}, Counter(), Counter()
    for g in games:
        spec, roots = g["spec"], g["roots"]
        game, split = spec["game_index"], spec["split"]
        require(type(game) is int and split in GAME_COUNTS, "strict source game identity")
        require(type(spec.get("quota")) is int and spec["quota"] == 512, "fixed teacher512")
        require(canonical(spec) == canonical(schedule.collection_spec(game)), "exact reserved fresh source game")
        require(type(roots) is list and roots and all(root_identity(x) == x for x in roots), "minimal root identities only")
        require(all(x["game_index"] == game and x["split"] == split for x in roots), "root belongs to exact game/split")
        plies = [x["ply"] for x in roots]
        require(plies == sorted(set(plies)), "usable roots have unique ascending plies")
        roots_by_game[game] = roots; counts[split] += 1; totals[split] += len(roots)
    require(dict(counts) == GAME_COUNTS, "fixed independent partition sizes")
    train = [g["spec"]["game_index"] for g in games if g["spec"]["split"] == "train"]
    rank = sorted(train, key=lambda game: (digest([DOMAIN, "nested_game", game]), game))
    narrow = schedule.narrow_ids()
    narrow_rank = [game for game in rank if game in narrow]
    extra = {game: i % 3 for i, game in enumerate(narrow_rank)}
    require(Counter(extra.values()) == {0: 64, 1: 64, 2: 64}, "balanced fixed6/5/5 assignment")
    cells = []
    for rep in (0, 1):
        selected = {arm: [] for arm in ARMS}; phases = {arm: [0, 0, 0] for arm in ARMS}
        for game in sorted(train):
            parts = relative_thirds(roots_by_game[game])
            for third, part in enumerate(parts):
                ordered = sorted(part, key=lambda row: (digest([DOMAIN, "root", SEEDS[rep], game, row["ply"]]), row["ply"]))
                require(len(ordered) >= 1, "broad game has a root in every third")
                selected["broad"].append([game, ordered[0]["ply"]]); phases["broad"][third] += 1
                if game in extra:
                    quota = 6 if extra[game] == third else 5
                    require(len(ordered) >= quota, "nested game supports prescribed unique relative-third quota")
                    selected["narrow"].extend([game, row["ply"]] for row in ordered[:quota]); phases["narrow"][third] += quota
        for arm in ARMS:
            refs = sorted(selected[arm]); ids = sorted({x[0] for x in refs})
            require(len(refs) == len({tuple(x) for x in refs}) == 3072 and phases[arm] == [1024] * 3, "matched unique root exposure")
            require(ids == (narrow if arm == "narrow" else sorted(train)), "exact prospectively fixed arm games")
            order = sorted(range(3072), key=lambda i: (digest([DOMAIN, "batch_order", SEEDS[rep], arm, i]), i))
            batches = [order[i:i + 32] for i in range(0, 3072, 32)]
            cells.append({**fit_identity(rep, arm), "cell_id": f"r{rep}_{arm}", "game_ids": ids,
                          "root_ids": refs, "phase_counts": phases[arm], "permutations": [order],
                          "batches": batches, "probe_batches": [], "roots": 3072,
                          "optimizer_steps": 96, "presentations": 3072, "rng_seed": SEEDS[rep],
                          "training_weights": "one_per_record_equal_within_arm_game_quotas"})
    return {"schema": SCHEMA, "study_id": STUDY, "domain": DOMAIN, **IDENTITY,
            "games": deepcopy(games), "source_receipt": deepcopy(source_receipt), "game_rank": rank,
            "narrow_game_ids": narrow, "narrow_extra_thirds": [[g, extra[g]] for g in narrow],
            "heldout_game_ids": [g["spec"]["game_index"] for g in games if g["spec"]["split"] != "train"],
            "cells": cells, "game_counts": dict(counts), "root_counts": dict(totals),
            "relative_third_convention": "ordered_usable_roots[:n//3],[n//3:2*n//3],[2*n//3:]",
            "selection_uses_outcomes": False, "selection_uses_targets": False,
            "historical_data_relabelled": False, "all_fits_required_before_heldout_predictions": True}


def validate_selection(selection, *, authenticate=True, boundary=lambda: None):
    require(type(selection) is dict and selection.get("schema") == SCHEMA, "new corrected selection")
    boundary()
    expected = build_selection(selection["games"], selection["source_receipt"])
    require(canonical(selection) == canonical(expected), "exact frozen selector and complete batch order")
    if authenticate:
        from . import corpus
        corpus.authenticate_selection_source(selection, boundary=boundary)
    return selection


def prepare_selection(acceptance, *, boundary=lambda: None):
    from . import corpus
    games, receipt = corpus.selection_catalogue(acceptance, boundary=boundary)
    result = build_selection(games, receipt)
    return validate_selection(result, boundary=boundary)


def load_selected(selection, repetition, breadth, *, boundary=lambda: None):
    """Train-only loader: held-out games never enter this path."""
    from . import corpus
    fit_identity(repetition, breadth)
    validate_selection(selection, boundary=boundary)
    cell = next(x for x in selection["cells"] if x["repetition"] == repetition and x["breadth"] == breadth)
    wanted = {tuple(x) for x in cell["root_ids"]}; examples = []
    for game in selection["games"]:
        if game["spec"]["game_index"] not in cell["game_ids"]: continue
        require(game["spec"]["split"] == "train", "held-out embargo in actual fitting loader")
        boundary(); decoded = corpus.load_selected_game(game, selection["source_receipt"], boundary=boundary)
        require([root_identity(x) for x in decoded["examples"]] == game["roots"], "actual decoded roots match accepted complete inventory")
        examples.extend(x for x in decoded["examples"] if (x["game_index"], x["ply"]) in wanted)
    examples.sort(key=lambda x: (x["game_index"], x["ply"]))
    require([[x["game_index"], x["ply"]] for x in examples] == cell["root_ids"], "no missing/substituted selected examples")
    counts = Counter(x["game_index"] for x in examples)
    require(set(counts.values()) == {16 if breadth == "narrow" else 3}, "fixed within-arm game exposure")
    return {"train_examples": examples, "train_weights": [1.] * 3072, "batches": deepcopy(cell["batches"]),
            "probe_batches": [], "train_game_ids": cell["game_ids"], "train_counts": dict(counts),
            "selection_cell": deepcopy(cell), "source_receipt": deepcopy(selection["source_receipt"])}


def load_heldout(selection, commitment, *, boundary=lambda: None):
    from . import corpus
    from . import fit
    fit.validate_commitment(selection, commitment, boundary=boundary)
    validate_selection(selection, boundary=boundary)
    examples = []
    for game in selection["games"]:
        if game["spec"]["split"] == "train": continue
        boundary(); decoded = corpus.load_selected_game(game, selection["source_receipt"], boundary=boundary)
        require([root_identity(x) for x in decoded["examples"]] == game["roots"], "complete held-out root inventory")
        examples.extend(decoded["examples"])
    examples.sort(key=lambda x: (x["game_index"], x["ply"]))
    require(sorted({x["game_index"] for x in examples}) == selection["heldout_game_ids"], "all128 independent held-out games")
    return examples


def collate_inputs(examples, weights):
    return accepted_training.collate_inputs(examples, weights)


def collate(examples, weights):
    return accepted_training.collate(examples, weights)
