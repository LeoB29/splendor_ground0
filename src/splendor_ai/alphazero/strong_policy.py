"""Experiment 142 bridge from the intact 035 legal-action policy to PUCT.

This is a separate branch: it does not change the random-initialized 090 model.
035 attention always sees the COMPLETE legal action set. Search and its loss
use the maximum logit in each canonical successor class, preserving greedy
preferences without awarding extra prior mass to redundant concrete aliases.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch

from splendor_ai.alphazero.network import NetworkPrediction
from splendor_ai.alphazero.search import AlphaZeroSearch, RootEdgeResult, SearchResult
from splendor_ai.encoding import (
    actions_by_canonical_successor_key, canonical_successor_action_key,
    encode_legal_action_features, encode_public_observation_tensor, fit_feature_vector,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel

MODEL_KIND = "strong_035_policy_value_142_v1"


def encode_position(state, config, actions=None):
    actions = SplendorEnv().legal_actions(state) if actions is None else actions
    if not actions:
        raise ValueError("Cannot encode an empty legal action set")
    keys = list(actions_by_canonical_successor_key(state, actions))
    lookup = {key: i for i, key in enumerate(keys)}
    obs = encode_public_observation_tensor(state, state.current_player)
    features = encode_legal_action_features(state=state, legal_actions=actions, player_id=state.current_player)
    return {
        "observation": torch.tensor(fit_feature_vector(obs.vector, config.observation_size), dtype=torch.float32),
        "features": torch.tensor([fit_feature_vector(f.vector, config.action_feature_size) for f in features], dtype=torch.float32),
        "groups": torch.tensor([lookup[canonical_successor_action_key(state, a)] for a in actions], dtype=torch.long),
        "group_count": len(keys),
    }, keys, actions


def grouped_logits(logits, groups, count):
    """Differentiable canonical maximum, with no padding entering any group."""
    return logits.new_full((count,), -torch.inf).scatter_reduce(0, groups, logits, reduce="amax", include_self=True)


class StrongPolicyInference:
    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["model_kind"] not in ("legal_action_attention_scorer", MODEL_KIND):
            raise ValueError("Unsupported strong-policy checkpoint")
        self.model = LegalActionAttentionScorerModel(LegalActionAttentionScorerConfig(**payload["model_config"]))
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.checkpoint = SimpleNamespace(path=Path(path), generation=payload.get("generation", -1))

    @torch.inference_mode()
    def full_prediction(self, state, actions=None):
        row, keys, actions = encode_position(state, self.model.config, actions)
        logits, value = self.model(row["observation"][None], row["features"][None], torch.ones((1, len(actions)), dtype=torch.bool))
        return logits[0], float(value[0]), row, keys, actions

    def predict_one(self, state, actions):
        logits, value, row, keys, _ = self.full_prediction(state)
        values = grouped_logits(logits, row["groups"], row["group_count"]).tolist()
        by_key = dict(zip(keys, values, strict=True))
        return NetworkPrediction(tuple(by_key[canonical_successor_action_key(state, a)] for a in actions), value)


def save_model(path, model, generation, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_kind": MODEL_KIND, "model_config": asdict(model.config),
                "model_state_dict": model.state_dict(), "generation": generation, "metadata": metadata}, temporary)
    temporary.replace(path)


class BudgetSearch(AlphaZeroSearch):
    """Original PUCT semantics, with an optional inclusive wall-clock budget.

    Encoding/root expansion count against the budget. At least one simulation
    completes; after that no new simulation starts at/after the deadline.
    Overshoot is bounded by one simulation plus result assembly, and logged.
    Fixed-count mode is used for reproducible self-play and equivalence tests.
    """
    def run(self, env, state, legal_actions=None, *, self_play, seed=None, seconds=None):
        if seconds is None:
            return super().run(env, state, legal_actions, self_play=self_play, seed=seed)
        if seconds <= 0:
            raise ValueError("Decision budget must be positive")
        started = time.perf_counter()
        actions = env.legal_actions(state) if legal_actions is None else legal_actions
        if not actions:
            raise ValueError("Cannot search without legal actions")
        self._tree = {}
        self._expansions = self._inference_calls = self._max_depth_reached = 0
        root_player = state.current_player
        search_seed = self.config.seed if seed is None else int(seed)
        rng = random.Random(search_seed)
        sample = self._sample_state(state, root_player, rng, simulation=-1)
        root_actions = list(actions_by_canonical_successor_key(sample, env.legal_actions(sample)).values())
        node = self._expand(sample, root_actions, self._node_key(sample, root_actions))
        if self_play and self.config.root_noise_fraction > 0:
            self._apply_root_noise(node, rng)
        simulations = 0
        # No fixed-simulation ceiling: the two players receive the same deadline.
        while simulations == 0 or time.perf_counter() - started < seconds:
            sample = self._sample_state(state, root_player, rng, simulation=simulations)
            value, path = self._select_and_evaluate(env, sample, root_player=root_player,
                                                   search_seed=search_seed, simulation=simulations)
            self._backup(path, value)
            simulations += 1
        original = actions_by_canonical_successor_key(state, actions)
        available = [(key, edge, original[key]) for key, edge in node.edges.items() if key in original]
        total = sum(edge.visits for _, edge, _ in available)
        if total != simulations:
            raise RuntimeError("Timed root visit accounting mismatch")
        probs = [edge.visits / total for _, edge, _ in available]
        selected = self._select_root_row(probs, available, state.turn_index, self_play, rng)
        edges = tuple(RootEdgeResult(key, action, edge.visits, probability,
                                    edge.prior if edge.network_prior is None else edge.network_prior,
                                    edge.prior, edge.q)
                      for (key, edge, action), probability in zip(available, probs, strict=True))
        return SearchResult(available[selected][2], edges, node.network_value,
                            sum(edge.q * edge.visits for _, edge, _ in available) / total,
                            simulations, self._expansions, self._inference_calls,
                            self._max_depth_reached, time.perf_counter() - started,
                            self.config.mode, selected)
