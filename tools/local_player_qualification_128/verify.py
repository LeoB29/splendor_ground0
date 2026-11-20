"""Independent arithmetic projection and recursive comparison of saved decisions."""
from copy import deepcopy
import math


def expected(root, breadth):
    if root.get("failure") or root["planner"] != "candidate":
        raise ValueError("Incomplete or non-candidate fixture")
    record = {key: deepcopy(root[key]) for key in
              ("root_identity", "planner", "privacy", "schedule", "candidates",
               "proposal", "scores", "worlds")}
    record["proposal"].pop("timing", None)
    offered = record["proposal"]["anchor_candidates"]
    count = len(record["candidates"]["actions"]) if breadth else len(offered["actions"])
    if not 1 <= count <= (8 if breadth else 4):
        raise ValueError("Invalid support")
    if record["candidates"]["actions"][:len(offered["actions"])] != offered["actions"]:
        raise ValueError("Broken preserved four-action prefix")
    if not breadth:
        record["planner"] = "reference"
        record["candidates"] = deepcopy(offered)
        record["proposal"]["candidates"] = deepcopy(offered)
        record["proposal"]["breadth"] = False
        record["proposal"]["added_indices"] = {"ranked": [], "random": []}
    if len(record["worlds"]) != 128:
        raise ValueError("Incomplete original world budget")
    sums = [0.0] * count
    for index, world in enumerate(record["worlds"]):
        if world["seed"] != record["schedule"]["world_seeds"][index]:
            raise ValueError("Original world schedule mismatch")
        world["outcomes"] = world["outcomes"][:count]
        if len(world["outcomes"]) != count:
            raise ValueError("Incomplete original outcomes")
        for j, outcome in enumerate(world["outcomes"]):
            if (type(outcome["wdl"]) not in (int, float) or outcome["wdl"] not in (-1, 0, 1)
                    or outcome["reason"] not in ("completed", "forced_pass_deadlock")):
                raise ValueError("Invalid original payoff")
            sums[j] += outcome["wdl"]
    means = [value / 128 for value in sums]
    choice = means.index(max(means))
    if breadth and (root["choice"] != choice or root["means"] != means):
        raise ValueError("Original choice/mean mismatch")
    record.update(means=means, choice=choice)
    return {"record": record, "action": deepcopy(record["candidates"]["actions"][choice])}


def compare(actual, wanted, path="result"):
    if isinstance(wanted, dict):
        if not isinstance(actual, dict) or set(actual) != set(wanted):
            raise ValueError(f"{path}: object fields differ")
        for key in wanted:
            compare(actual[key], wanted[key], f"{path}.{key}")
    elif isinstance(wanted, list):
        if not isinstance(actual, list) or len(actual) != len(wanted):
            raise ValueError(f"{path}: list length differs")
        for index, (a, b) in enumerate(zip(actual, wanted)):
            compare(a, b, f"{path}[{index}]")
    elif isinstance(wanted, float):
        model_value = any(f".{name}[" in path for name in ("scores", "learned_logits", "learned_group_scores"))
        equal = (math.isclose(actual, wanted, rel_tol=1e-6, abs_tol=1e-6) if model_value
                 and type(actual) in (int, float) else actual == wanted)
        if type(actual) not in (int, float) or not math.isfinite(actual) or not equal:
            raise ValueError(f"{path}: numeric value differs: {actual!r} vs {wanted!r}")
    elif type(actual) is not type(wanted) or actual != wanted:
        raise ValueError(f"{path}: value differs: {actual!r} vs {wanted!r}")


def completed_work(result):
    worlds = result["record"]["worlds"]
    return {"worlds": len(worlds), "continuations": sum(len(w["outcomes"]) for w in worlds)}
