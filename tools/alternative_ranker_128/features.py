"""Small deterministic ranker features from observation295/action79 only.

No snapshots, outcomes, model predictions, card IDs or private state enter this
module. Canonical successor classes are represented by the mean of their
concrete member features, so redundant action aliases receive no extra weight.
"""
from __future__ import annotations

import numpy as np

FEATURE_CONFIG = {
    "version": "alternative_ranker_lawful465_v1",
    "size": 465,
    "observation_size": 295,
    "action_size": 79,
    "layout": {"observation": [0, 295], "action": [295, 374],
               "afterstate": [374, 387], "known_cards": [387, 459],
               "nobles": [459, 465]},
    "afterstate": ["six_tokens_div10", "five_bonuses_div7", "score_div20", "reserve_count_div3"],
    "known_cards": "18_before_identity_slots_each_shortfall_before_div10_shortfall_after_div10_available_after_self_reserved_after",
    "nobles": "3_before_identity_slots_each_total_bonus_deficit_before_div20_after_div20",
    "shortfall": "max(sum(max(cost_colored-token_colored-bonus_colored,0))-one_shared_gold_pool,0)",
    "unknown_cards": "zero; no refill or blind-reserve identity is inferred",
    "canonical_reduction": "arithmetic_mean_of_concrete_member_vectors_float64_then_float32",
    "concrete_chunk": 32,
    "information": "original_lawful_observation295_and_action79_only",
}
FEATURE_SIZE = FEATURE_CONFIG["size"]


def _array(value):
    # Accept saved CPU tensors without importing Torch or copying whole pools.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def concrete_features(observation, actions):
    """Encode a bounded batch of concrete legal actions; return A x465 float32."""
    obs = _array(observation).astype(np.float64, copy=False)
    action = _array(actions).astype(np.float64, copy=False)
    if obs.shape != (295,) or action.ndim != 2 or action.shape[1] != 79:
        raise ValueError("Expected observation295 and A x79 actions")
    if not np.isfinite(obs).all() or not np.isfinite(action).all():
        raise ValueError("Nonfinite lawful features")
    count = len(action)
    out = np.zeros((count, FEATURE_SIZE), dtype=np.float64)
    out[:, :295], out[:, 295:374] = obs, action
    tokens0 = np.rint(obs[189:195] * 10)
    bonus0 = np.rint(obs[195:200] * 7)
    scales = np.array([4., 4., 4., 4., 4., 5.])
    flow = np.rint(action[:, 54:60] * scales)
    decoded_flow = (np.rint(action[:, 36:42] * scales)
                    - np.rint(action[:, 42:48] * scales)
                    - np.rint(action[:, 48:54] * scales))
    if not np.array_equal(flow, decoded_flow):
        raise ValueError("Action net flow disagrees with take/return/payment")
    tokens1 = tokens0 + flow
    buying = action[:, 4] + action[:, 5]
    bonus1 = bonus0 + buying[:, None] * action[:, 19:24]
    reserves1 = np.rint(obs[201] * 3) + action[:, 2] + action[:, 3] - action[:, 5]
    score1 = np.rint(obs[200] * 20) + np.rint(action[:, 61] * 8)
    if np.any(tokens1 < 0) or np.any(bonus1 < 0) or np.any((reserves1 < 0) | (reserves1 > 3)):
        raise ValueError("Impossible decoded legal afterstate")
    out[:, 374:387] = np.column_stack((tokens1 / 10, bonus1 / 7, score1 / 20, reserves1 / 3))
    starts = list(range(33, 189, 13)) + list(range(203, 242, 13)) + list(range(256, 295, 13))
    for slot, start in enumerate(starts):
        card = obs[start:start+13]
        if card[0] <= .5:
            continue
        costs = np.rint(card[8:13] * 7)
        before = max(0., np.maximum(costs-tokens0[:5]-bonus0, 0).sum()-tokens0[5])
        after = np.maximum(np.maximum(costs-tokens1[:, :5]-bonus1, 0).sum(axis=1)-tokens1[:, 5], 0)
        if slot < 12:
            target = ((action[:, 2] + action[:, 4])
                      * action[:, 6+slot//4] * action[:, 9+slot % 4])
        elif slot < 15:
            target = action[:, 5] * action[:, 13+slot-12]
        else:
            target = np.zeros(count)
        available = 1. - target * buying
        own_after = (float(12 <= slot < 15) * (1. - target * buying)
                     + target * action[:, 2])
        out[:, 387+slot*4:391+slot*4] = np.column_stack((
            np.full(count, before / 10), after / 10, available, own_after))
    for slot in range(3):
        noble = obs[12+7*slot:19+7*slot]
        if noble[0] <= .5:
            continue
        need = np.rint(noble[2:7] * 4)
        out[:, 459+2*slot] = np.maximum(need-bonus0, 0).sum() / 20
        out[:, 460+2*slot] = np.maximum(need-bonus1, 0).sum(axis=1) / 20
        # A legally claimed noble has zero remaining deficit. Its before identity
        # stays in its slot, just as purchased cards do; no replacement is guessed.
    return out.astype(np.float32)


def group_features(observation, action_features, members):
    """Return one465-vector per offered group, touching only its concrete rows."""
    observation = _array(observation)
    if len(getattr(action_features, "shape", ())) != 2 or action_features.shape[1] != 79:
        raise ValueError("Expected legal action79 matrix")
    result = np.empty((len(members), FEATURE_SIZE), dtype=np.float32)
    for group, indices in enumerate(members):
        values = _array(indices).tolist()
        if not values or any(type(i) is not int or i < 0 or i >= len(action_features) for i in values):
            raise ValueError("Invalid canonical member indices")
        if len(set(values)) != len(values):
            raise ValueError("Repeated concrete member")
        total = np.zeros(FEATURE_SIZE, dtype=np.float64)
        for lo in range(0, len(values), FEATURE_CONFIG["concrete_chunk"]):
            selected = values[lo:lo+FEATURE_CONFIG["concrete_chunk"]]
            # Do not convert the complete full-legal matrix to float64.
            block = np.asarray([_array(action_features[index]) for index in selected])
            total += concrete_features(observation, block).sum(axis=0, dtype=np.float64)
        result[group] = total / len(values)
    return result


def first_argmax(scores):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid ranker scores")
    return int(np.argmax(values)) if len(values) else None
