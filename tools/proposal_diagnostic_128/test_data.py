"""Source selection and saved evidence tests; no model inference or rollouts."""
import copy
from collections import Counter

import pytest

from tools.proposal_diagnostic_128 import data as d


def synthetic_records():
    result = {}
    for stage, count in (("stage_a", 402), ("probe_a", 12)):
        entries = []
        for index in range(count):
            pair = index//2
            game = {"spec": {"index": index, "pair": pair, "fit": d.FITS[pair % 3], "candidate_seat": index % 2},
                    "trace": [{"turn": turn, "actor": turn % 2} for turn in range(50)],
                    "winner": 0, "score": 999}
            entries.append({"game": game})
        result[stage] = entries
    return result


def signature(selected):
    return {stage: [(entry["game"]["spec"]["pair"], entry["game"]["spec"]["fit"],
                     entry["game"]["spec"]["index"], move["turn"], role, phase)
                    for entry, move, role, phase in rows] for stage, rows in selected.items()}


def test_identity_selection_independent_quotas_and_one_state_per_pair():
    records = synthetic_records()
    selected = d.selected_identities(records)
    assert signature(selected) == signature(d.independent_identities(records))
    assert len(selected["main"]) == 201 and len(selected["probe"]) == 12
    assert len({entry["game"]["spec"]["pair"] for entry, *_ in selected["main"]}) == 201
    for fit in d.FITS:
        cells = Counter((role, phase) for entry, _, role, phase in selected["main"] if entry["game"]["spec"]["fit"] == fit)
        assert sorted(cells.values()) == [11, 11, 11, 11, 11, 12]
    assert set(Counter((role, phase) for _, _, role, phase in selected["probe"]).values()) == {2}
    assert Counter(entry["game"]["spec"]["fit"] for entry, *_ in selected["probe"]) == dict.fromkeys(d.FITS, 4)


def test_outcomes_logits_and_container_order_cannot_select_positions():
    records = synthetic_records()
    original = signature(d.selected_identities(records))
    for entries in records.values():
        entries.reverse()
        for entry in entries:
            entry["game"].update(winner=1, score=-999, logits=[float("nan")])
            entry["game"]["trace"].reverse()
            for move in entry["game"]["trace"]:
                move.update(wdl=123456, choice=9, disagreement=True)
    assert signature(d.selected_identities(records)) == original
    assert signature(d.independent_identities(records)) == original


def test_assigned_missing_phase_holds_without_fallback():
    records = synthetic_records()
    for entry in records["stage_a"]:
        entry["game"]["trace"] = entry["game"]["trace"][:40]
    with pytest.raises(ValueError, match="missing"):
        d.selected_identities(records)
    with pytest.raises(ValueError, match="missing"):
        d.independent_identities(records)


def real_entry(index=0):
    games = d.old.read_json(d.SOURCE / "stage_a_games_lock.json")
    audits = d.old.read_json(d.SOURCE / "stage_a_audits_lock.json")
    gi, ai = games["files"][index], audits["files"][index]
    return {"game": d.verified(gi), "audit": d.verified(ai), "game_item": gi, "audit_item": ai}


def test_selected_actual_source_replay_and_metadata_tamper():
    entry = real_entry()
    move = entry["game"]["trace"][43]
    role = d._role(entry["game"], move)
    spec = d._spec(0, "main", (entry, move, role, 2))
    snapshot = d.replay_selected_state(spec)
    assert d.old.digest(snapshot) == spec["snapshot_sha256"]
    for field, bad in (("role", "treatment" if role == "anchor" else "anchor"),
                       ("phase", 0), ("fit", 161002), ("pair", 99),
                       ("snapshot_sha256", "0"*64), ("canonical_support", 999)):
        changed = copy.deepcopy(spec)
        changed[field] = bad
        with pytest.raises(ValueError):
            d.replay_selected_state(changed)


def opening_roots():
    entries = [real_entry(i) for i in (0, 1)]
    roots = [d.verified(d._root_item("stage_a", i, entry["game"]["trace"][0])) for i, entry in enumerate(entries)]
    return roots


def test_saved_matched_root_exact_invariance_and_no_independent_assessment():
    treatment, anchor = opening_roots()
    report = d.describe_matched_roots(treatment, anchor, 161001)
    assert report["turn"] == 0 and not report["independent_assessment"]
    assert report["fixed_fourth_wdl_difference"] == treatment["means"][3]-anchor["means"][3]
    assert report["proposal_changed"] == (treatment["candidates"]["keys"][3] != anchor["candidates"]["keys"][3])
    same = copy.deepcopy(treatment)
    same["proposal"]["model_id"] = "anchor_161001"
    identical = d.describe_matched_roots(treatment, same, 161001)
    assert not identical["proposal_changed"] and not identical["decision_changed"]
    assert identical["fixed_fourth_wdl_difference"] == 0


@pytest.mark.parametrize("bad", ["snapshot", "world", "common_outcome", "legal", "logit", "choice"])
def test_saved_matching_rejects_snapshot_world_input_and_arithmetic_corruption(bad):
    treatment, anchor = opening_roots()
    anchor = copy.deepcopy(anchor)
    if bad == "snapshot":
        anchor["snapshot"]["turn_index"] += 1
    elif bad == "world":
        anchor["worlds"][0]["state_sha256"] = "0"*64
    elif bad == "common_outcome":
        # Keep WDL/means/choice unchanged so the shared-trajectory check detects it.
        anchor["worlds"][0]["outcomes"][0]["trace_sha256"] = "0"*64
    elif bad == "legal":
        anchor["proposal"]["legal_actions"].reverse()
    elif bad == "logit":
        anchor["proposal"]["learned_logits"][0] += 10
    elif bad == "choice":
        anchor["choice"] = (anchor["choice"]+1) % 4
    with pytest.raises(ValueError):
        d.describe_matched_roots(treatment, anchor, 161001)


def test_summary_equal_pair_means_and_empty_later_prefix():
    rows = [{"pair": pair, "fit": 161001, "phase": 0, "has_fourth": True,
             "proposal_changed": True, "decision_changed": False,
             "fixed_fourth_wdl_difference": value,
             "treatment_selected_fourth": False, "anchor_selected_fourth": True}
            for pair, value in [(0, 1.), (0, 1.), (0, 1.), (1, -1.)]]
    result = d._summary(rows)
    assert result["fixed_fourth_wdl_mean_by_root"] == .5
    assert result["fixed_fourth_wdl_mean_by_pair"] == 0
    assert result["source_pairs"] == 2
    assert d._summary([])["fixed_fourth_wdl_mean_by_pair"] is None
