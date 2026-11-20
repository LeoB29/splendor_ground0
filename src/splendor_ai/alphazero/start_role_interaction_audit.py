"""No-training start-role interaction and gradient audit for experiment 119."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch import nn

from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.encoding.entity_encoding import ENTITY_GLOBAL
from splendor_ai.training.model import CanonicalEntityActionScorerModel

from .network import load_checkpoint
from .replay import read_replay_jsonl
from .scaled_continuation import _sha256, _write_json
from .start_role_pilot import (
    _target_manifest_hash,
    _training_rows_from_records,
    reconstruct_role_panel,
    split_pair_games,
    summarize_target_records,
)


PROTOCOL_SHA256 = "9f5179d270438ee13514931a741da141d3de6478f336b9c9c5e8aca07f83b558"
EXPECTED_HASHES = {
    "result_118": "4846b79e1fbcd6344e759155befc469463a9100423f8785ff0e352fa9127fbc1",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "train_replay": "59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a",
    "target_manifest": "278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f",
}
EXPECTED_118_DECISION = (
    "minimal_role_bridge_failed_direct_uptake_review_richer_interaction_no_gameplay"
)


@dataclass(frozen=True, slots=True)
class StartRoleInteractionAuditConfig:
    output_dir: Path
    result_118: Path
    generation_11_checkpoint: Path
    train_replay: Path
    target_dir: Path
    device: str = "cpu"
    torch_threads: int = 1
    batch_pairs: int = 32


class _FrozenParentRolePrototype(nn.Module):
    name: str

    def __init__(self, parent: CanonicalEntityActionScorerModel) -> None:
        super().__init__()
        self.parent = parent
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.parent.eval()
        return self

    def endpoint_parameters(self, endpoint: Literal["policy", "value"]) -> list[nn.Parameter]:
        raise NotImplementedError

    def other_endpoint_parameters(
        self, endpoint: Literal["policy", "value"]
    ) -> list[nn.Parameter]:
        raise NotImplementedError

    def _validate_and_inputs(self, tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        self.parent._validate_inputs(
            tensors["entity_type_ids"],
            tensors["entity_card_ids"],
            tensors["entity_noble_ids"],
            tensors["entity_numeric_features"],
            tensors["entity_mask"],
            tensors["action_type_ids"],
            tensors["action_card_ids"],
            tensors["action_noble_ids"],
            tensors["action_numeric_features"],
            tensors["action_mask"],
        )
        role = tensors["observer_is_start_player"]
        if role.ndim != 1 or role.shape[0] != tensors["entity_type_ids"].shape[0]:
            raise ValueError("119 start-role tensor must have shape [batch].")
        if not bool((tensors["entity_type_ids"][:, 0] == ENTITY_GLOBAL).all()):
            raise ValueError("119 requires the canonical global entity first.")
        signed = role.to(tensors["entity_numeric_features"].dtype).mul(2.0).sub(1.0)
        entity_input = torch.cat(
            [
                tensors["entity_numeric_features"],
                self.parent.entity_type_embedding(tensors["entity_type_ids"]),
                self.parent.card_embedding(tensors["entity_card_ids"]),
                self.parent.noble_embedding(tensors["entity_noble_ids"]),
            ],
            dim=-1,
        )
        return self.parent.entity_input(entity_input), signed

    def _encode_actions(
        self,
        tensors: dict[str, torch.Tensor],
        policy_state: torch.Tensor,
    ) -> torch.Tensor:
        action_input = torch.cat(
            [
                tensors["action_numeric_features"],
                self.parent.action_type_embedding(tensors["action_type_ids"]),
                self.parent.card_embedding(tensors["action_card_ids"]),
                self.parent.noble_embedding(tensors["action_noble_ids"]),
            ],
            dim=-1,
        )
        action_hidden = self.parent.action_input(action_input)
        action_hidden = self.parent.action_encoder(
            action_hidden + self.parent.state_to_action_context(policy_state).unsqueeze(1),
            src_key_padding_mask=~tensors["action_mask"],
        )
        action_hidden = self.parent.action_layer_norm(action_hidden)
        return action_hidden.masked_fill(~tensors["action_mask"].unsqueeze(-1), 0.0)

    def _heads(
        self,
        tensors: dict[str, torch.Tensor],
        policy_state: torch.Tensor,
        value_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_hidden = self._encode_actions(tensors, policy_state)
        expanded = policy_state.unsqueeze(1).expand(-1, action_hidden.shape[1], -1)
        scores = self.parent.score_head(torch.cat([expanded, action_hidden], dim=-1)).squeeze(-1)
        value = self.parent.value_head(value_state).squeeze(-1)
        return scores, value


class GlobalAntisymmetricRoleChannel(_FrozenParentRolePrototype):
    """The antisymmetric role-contrast channel present inside experiment 118."""

    name = "global_antisymmetric_118_channel"

    def __init__(self, parent: CanonicalEntityActionScorerModel) -> None:
        super().__init__(parent)
        self.role_direction = nn.Parameter(torch.zeros(parent.config.entity_hidden_size))

    def endpoint_parameters(self, endpoint: Literal["policy", "value"]) -> list[nn.Parameter]:
        return [self.role_direction]

    def other_endpoint_parameters(
        self, endpoint: Literal["policy", "value"]
    ) -> list[nn.Parameter]:
        return []

    def forward(self, observer_is_start_player: torch.Tensor, **tensors: torch.Tensor):
        tensors = dict(tensors, observer_is_start_player=observer_is_start_player)
        transformer_input, signed = self._validate_and_inputs(tensors)
        global_delta = torch.zeros_like(transformer_input)
        global_delta[:, 0, :] = signed.unsqueeze(-1) * self.role_direction
        hidden = self.parent.entity_encoder(
            transformer_input + global_delta,
            src_key_padding_mask=~tensors["entity_mask"],
        )
        hidden = hidden.masked_fill(~tensors["entity_mask"].unsqueeze(-1), 0.0)
        counts = tensors["entity_mask"].sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        state = hidden.sum(dim=1) / counts
        return self._heads(tensors, state, state)


class SplitReadoutAdditiveRole(_FrozenParentRolePrototype):
    """Independent antisymmetric additive role vectors at policy/value readouts."""

    name = "split_readout_additive"

    def __init__(self, parent: CanonicalEntityActionScorerModel) -> None:
        super().__init__(parent)
        width = parent.config.entity_hidden_size
        self.policy_role = nn.Parameter(torch.zeros(width))
        self.value_role = nn.Parameter(torch.zeros(width))

    def endpoint_parameters(self, endpoint: Literal["policy", "value"]) -> list[nn.Parameter]:
        return [self.policy_role if endpoint == "policy" else self.value_role]

    def other_endpoint_parameters(
        self, endpoint: Literal["policy", "value"]
    ) -> list[nn.Parameter]:
        return [self.value_role if endpoint == "policy" else self.policy_role]

    def forward(self, observer_is_start_player: torch.Tensor, **tensors: torch.Tensor):
        tensors = dict(tensors, observer_is_start_player=observer_is_start_player)
        transformer_input, signed = self._validate_and_inputs(tensors)
        hidden = self.parent.entity_encoder(
            transformer_input,
            src_key_padding_mask=~tensors["entity_mask"],
        )
        hidden = hidden.masked_fill(~tensors["entity_mask"].unsqueeze(-1), 0.0)
        counts = tensors["entity_mask"].sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        state = hidden.sum(dim=1) / counts
        policy_state = state + signed.unsqueeze(-1) * self.policy_role
        value_state = state + signed.unsqueeze(-1) * self.value_role
        return self._heads(tensors, policy_state, value_state)


class SplitReadoutFiLMRole(_FrozenParentRolePrototype):
    """Independent antisymmetric feature-wise gain/bias at both readouts."""

    name = "split_readout_film"

    def __init__(self, parent: CanonicalEntityActionScorerModel) -> None:
        super().__init__(parent)
        width = parent.config.entity_hidden_size
        self.policy_gain = nn.Parameter(torch.zeros(width))
        self.policy_bias = nn.Parameter(torch.zeros(width))
        self.value_gain = nn.Parameter(torch.zeros(width))
        self.value_bias = nn.Parameter(torch.zeros(width))

    def endpoint_parameters(self, endpoint: Literal["policy", "value"]) -> list[nn.Parameter]:
        if endpoint == "policy":
            return [self.policy_gain, self.policy_bias]
        return [self.value_gain, self.value_bias]

    def other_endpoint_parameters(
        self, endpoint: Literal["policy", "value"]
    ) -> list[nn.Parameter]:
        if endpoint == "policy":
            return [self.value_gain, self.value_bias]
        return [self.policy_gain, self.policy_bias]

    def forward(self, observer_is_start_player: torch.Tensor, **tensors: torch.Tensor):
        tensors = dict(tensors, observer_is_start_player=observer_is_start_player)
        transformer_input, signed = self._validate_and_inputs(tensors)
        hidden = self.parent.entity_encoder(
            transformer_input,
            src_key_padding_mask=~tensors["entity_mask"],
        )
        hidden = hidden.masked_fill(~tensors["entity_mask"].unsqueeze(-1), 0.0)
        counts = tensors["entity_mask"].sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        state = hidden.sum(dim=1) / counts
        role = signed.unsqueeze(-1)
        policy_state = state * (1.0 + role * self.policy_gain) + role * self.policy_bias
        value_state = state * (1.0 + role * self.value_gain) + role * self.value_bias
        return self._heads(tensors, policy_state, value_state)


def run_start_role_interaction_audit_119(
    config: StartRoleInteractionAuditConfig,
) -> dict[str, Any]:
    """Run the frozen no-training audit and seal its result."""

    torch.set_num_threads(config.torch_threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    rows = read_replay_jsonl(config.train_replay)
    states, reconstruction = reconstruct_role_panel(
        rows,
        expected_games=224,
        label="119 training-target reconstruction",
    )
    records = _load_target_records(config.target_dir)
    oriented = _training_rows_from_records(states, records)
    pairs = [(oriented[index], oriented[index + 1]) for index in range(0, len(oriented), 2)]
    split = split_pair_games(states)
    validation_games = set(split["validation_games"])
    train_pairs = [pair for pair in pairs if pair[0].game_seed not in validation_games]
    validation_pairs = [pair for pair in pairs if pair[0].game_seed in validation_games]
    if (len(train_pairs), len(validation_pairs), len(pairs)) != (358, 90, 448):
        raise ValueError("119 pair split does not reproduce 118.")

    factories = (
        GlobalAntisymmetricRoleChannel,
        SplitReadoutAdditiveRole,
        SplitReadoutFiLMRole,
    )
    architectures: dict[str, dict[str, Any]] = {}
    for factory in factories:
        parent, identity = load_checkpoint(config.generation_11_checkpoint, device=config.device)
        if identity.generation != 11:
            raise ValueError("119 requires exact generation 11.")
        model = factory(parent).to(config.device)
        model.eval()
        architectures[model.name] = _audit_architecture(
            model,
            pairs=pairs,
            train_pairs=train_pairs,
            validation_pairs=validation_pairs,
            device=config.device,
            batch_pairs=config.batch_pairs,
        )

    baseline = architectures[GlobalAntisymmetricRoleChannel.name]
    additive = architectures[SplitReadoutAdditiveRole.name]
    film = architectures[SplitReadoutFiLMRole.name]
    additive_checks = _qualification_checks(additive, baseline)
    film_checks = _qualification_checks(film, baseline)
    additive["qualification_checks"] = additive_checks
    additive["qualified"] = all(additive_checks.values())
    film["qualification_checks"] = film_checks
    film["qualified"] = all(film_checks.values())
    if additive["qualified"]:
        recommendation = SplitReadoutAdditiveRole.name
        decision = (
            "direct_split_additive_role_gradient_qualifies_freeze_minimal_pilot_no_training"
        )
    elif film["qualified"]:
        recommendation = SplitReadoutFiLMRole.name
        decision = "direct_split_film_role_gradient_qualifies_freeze_pilot_no_training"
    else:
        recommendation = None
        decision = "direct_role_readout_gradient_not_qualified_close_start_role_only_repair"

    result = {
        "kind": "alphazero_start_role_interaction_audit_119",
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "input_hashes": {
            "result_118": _sha256(config.result_118),
            "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
            "train_replay": _sha256(config.train_replay),
            "target_manifest": _target_manifest_hash(config.target_dir),
        },
        "reconstruction": reconstruction,
        "split": split,
        "target_summary": summarize_target_records(records),
        "architectural_observation": {
            "118_role_path": "single_global_entity_then_two_post_norm_transformer_layers_then_entity_mean_pool",
            "global_injection_entities": 1,
            "entity_count": _entity_count_summary(pairs, config.device, config.batch_pairs),
            "readout_candidate_path": "post_pool_direct_antisymmetric_split_policy_value",
        },
        "architectures": architectures,
        "recommended_architecture": recommendation,
        "decision": decision,
        "training_performed": False,
        "optimizer_instantiated": False,
        "parameters_mutated": False,
        "checkpoint_written": False,
        "protected_confirmation_read": False,
        "search_targets_generated": False,
        "self_play_performed": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
    }
    _write_json(config.output_dir / "result.json", result)
    return result


def _verify_inputs(config: StartRoleInteractionAuditConfig) -> None:
    observed = {
        "result_118": _sha256(config.result_118),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "train_replay": _sha256(config.train_replay),
        "target_manifest": _target_manifest_hash(config.target_dir),
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"119 frozen input mismatch: {observed}")
    result = json.loads(config.result_118.read_text(encoding="utf-8"))
    if (
        result.get("status") != "complete"
        or result.get("decision") != EXPECTED_118_DECISION
        or result.get("promotion_authorized") is not False
        or result.get("training", {}).get("target_manifest_sha256") != EXPECTED_HASHES["target_manifest"]
    ):
        raise ValueError("119 does not reproduce the finalized 118 decision lineage.")


def _load_target_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shard in range(4):
        payload = json.loads((directory / f"shard_{shard}.json").read_text(encoding="utf-8"))
        if payload.get("completed") is not True or payload.get("completed_pairs") != 112:
            raise ValueError(f"119 target shard {shard} is incomplete.")
        if payload.get("identity", {}).get("protocol_sha256") != (
            "b7b92bf53ab1801bef96fbdb4f9544a81943889c82a243c20b4349cb4b2bf186"
        ):
            raise ValueError(f"119 target shard {shard} has wrong 118 identity.")
        records.extend(payload["rows"])
    if len(records) != 448:
        raise ValueError("119 requires exactly 448 paired target records.")
    return records


def _audit_architecture(
    model: _FrozenParentRolePrototype,
    *,
    pairs,
    train_pairs,
    validation_pairs,
    device: str,
    batch_pairs: int,
) -> dict[str, Any]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    zero_before = max(float(parameter.detach().abs().max().item()) for parameter in trainable)
    identity = _identity_audit(model, pairs, device=device, batch_pairs=batch_pairs)
    gradients = {}
    for endpoint in ("policy", "value"):
        train = _gradient_metrics(model, train_pairs, endpoint, device, batch_pairs)
        validation = _gradient_metrics(model, validation_pairs, endpoint, device, batch_pairs)
        all_pairs = _gradient_metrics(model, pairs, endpoint, device, batch_pairs)
        gradients[endpoint] = {
            "train": _without_vector(train),
            "validation": _without_vector(validation),
            "all": _without_vector(all_pairs),
            "train_validation_cosine": _cosine(train["vector"], validation["vector"]),
        }
    zero_after = max(float(parameter.detach().abs().max().item()) for parameter in trainable)
    declared = {
        GlobalAntisymmetricRoleChannel.name: 96,
        SplitReadoutAdditiveRole.name: 192,
        SplitReadoutFiLMRole.name: 384,
    }[model.name]
    integrity = {
        "declared_parameter_count": sum(parameter.numel() for parameter in trainable) == declared,
        "parent_frozen": all(not parameter.requires_grad for parameter in model.parent.parameters()),
        "parameters_zero_before": zero_before == 0.0,
        "parameters_zero_after": zero_after == 0.0,
        "identity_logits_within_1e_7": identity["max_logit_abs_diff"] <= 1e-7,
        "identity_values_within_1e_7": identity["max_value_abs_diff"] <= 1e-7,
        "deterministic_repeat": identity["max_repeat_abs_diff"] == 0.0,
        "finite": _all_finite({"identity": identity, "gradients": gradients}),
    }
    return {
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "endpoint_parameter_counts": {
            endpoint: sum(parameter.numel() for parameter in model.endpoint_parameters(endpoint))
            for endpoint in ("policy", "value")
        },
        "identity": identity,
        "gradients": gradients,
        "integrity_checks": integrity,
        "integrity_passed": all(integrity.values()),
    }


def _identity_audit(model, pairs, *, device: str, batch_pairs: int) -> dict[str, float | int]:
    max_logit = 0.0
    max_value = 0.0
    max_repeat = 0.0
    rows = 0
    with torch.no_grad():
        for chunk in _chunks(pairs, batch_pairs):
            batch = _collate_pairs(chunk, device)
            candidate_logits, candidate_values = model(**batch["inputs"])
            repeat_logits, repeat_values = model(**batch["inputs"])
            parent_inputs = {
                key: value
                for key, value in batch["inputs"].items()
                if key != "observer_is_start_player"
            }
            parent_logits, parent_values = model.parent(**parent_inputs)
            mask = batch["inputs"]["action_mask"]
            max_logit = max(
                max_logit,
                float((candidate_logits[mask] - parent_logits[mask]).abs().max().item()),
            )
            max_value = max(max_value, float((candidate_values - parent_values).abs().max().item()))
            max_repeat = max(
                max_repeat,
                float((candidate_logits[mask] - repeat_logits[mask]).abs().max().item()),
                float((candidate_values - repeat_values).abs().max().item()),
            )
            rows += len(chunk) * 2
    return {
        "rows": rows,
        "max_logit_abs_diff": max_logit,
        "max_value_abs_diff": max_value,
        "max_repeat_abs_diff": max_repeat,
    }


def _gradient_metrics(model, pairs, endpoint: str, device: str, batch_pairs: int) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    total_pairs = len(pairs)
    mean_loss = 0.0
    for chunk in _chunks(pairs, batch_pairs):
        batch = _collate_pairs(chunk, device)
        policy_loss, value_loss = _contrast_losses(model, batch)
        loss = policy_loss if endpoint == "policy" else value_loss
        weight = len(chunk) / total_pairs
        (loss * weight).backward()
        mean_loss += float(loss.detach().item()) * weight
    endpoint_parameters = model.endpoint_parameters(endpoint)
    vector = _gradient_vector(endpoint_parameters)
    other = _gradient_vector(model.other_endpoint_parameters(endpoint))
    norm = float(torch.linalg.vector_norm(vector).item())
    count = int(vector.numel())
    return {
        "pairs": total_pairs,
        "contrast_loss": mean_loss,
        "parameter_count": count,
        "gradient_l2": norm,
        "gradient_rms": norm / math.sqrt(count),
        "gradient_nonzero": norm > 0.0,
        "gradient_finite": bool(torch.isfinite(vector).all()),
        "cross_endpoint_gradient_max_abs": (
            float(other.abs().max().item()) if other.numel() else None
        ),
        "vector": vector.detach().cpu(),
    }


def _contrast_losses(model, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    logits, values = model(**batch["inputs"])
    mask = batch["inputs"]["action_mask"]
    probabilities = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    pair_count = probabilities.shape[0] // 2
    width = probabilities.shape[1]
    roles = batch["inputs"]["observer_is_start_player"].to(probabilities.dtype)
    signed = roles.mul(2.0).sub(1.0).reshape(pair_count, 2, 1)
    predicted_policy_delta = (probabilities.reshape(pair_count, 2, width) * signed).sum(dim=1)
    target_policy_delta = (batch["target_policy"].reshape(pair_count, 2, width) * signed).sum(dim=1)
    pair_mask = mask.reshape(pair_count, 2, width).all(dim=1)
    policy_error = (predicted_policy_delta - target_policy_delta).square()
    policy_loss = (
        (policy_error * pair_mask).sum(dim=1)
        / pair_mask.sum(dim=1).clamp_min(1).to(policy_error.dtype)
    ).mean()
    value_signed = signed.squeeze(-1)
    predicted_value_delta = (values.reshape(pair_count, 2) * value_signed).sum(dim=1)
    target_value_delta = (
        batch["target_value"].reshape(pair_count, 2) * value_signed
    ).sum(dim=1)
    value_loss = (predicted_value_delta - target_value_delta).square().mean()
    return policy_loss, value_loss


def _collate_pairs(pairs, device: str) -> dict[str, Any]:
    rows = [row for pair in pairs for row in pair]
    inputs = collate_entity_action_batch(
        [(row.state, list(row.actions)) for row in rows],
        device=device,
    )
    width = int(inputs["action_mask"].shape[1])
    target_policy = torch.zeros((len(rows), width), dtype=torch.float32, device=device)
    for index, row in enumerate(rows):
        target_policy[index, : len(row.policy_target)] = torch.tensor(
            row.policy_target, dtype=torch.float32, device=device
        )
    inputs["observer_is_start_player"] = torch.tensor(
        [row.observer_is_start_player for row in rows],
        dtype=torch.long,
        device=device,
    )
    return {
        "inputs": inputs,
        "target_policy": target_policy,
        "target_value": torch.tensor(
            [row.value_target for row in rows], dtype=torch.float32, device=device
        ),
    }


def _qualification_checks(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    checks = {"integrity": bool(candidate["integrity_passed"])}
    for endpoint in ("policy", "value"):
        gradients = candidate["gradients"][endpoint]
        baseline_all = baseline["gradients"][endpoint]["all"]
        checks[f"{endpoint}_gradient_nonzero"] = bool(gradients["all"]["gradient_nonzero"])
        checks[f"{endpoint}_train_validation_cosine_positive"] = (
            gradients["train_validation_cosine"] > 0.0
        )
        checks[f"{endpoint}_gradient_rms_exceeds_118_channel"] = (
            gradients["all"]["gradient_rms"] > baseline_all["gradient_rms"]
        )
        checks[f"{endpoint}_cross_endpoint_leakage_zero"] = (
            gradients["all"]["cross_endpoint_gradient_max_abs"] == 0.0
        )
    return checks


def _gradient_vector(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    pieces = []
    for parameter in parameters:
        pieces.append(
            torch.zeros_like(parameter).reshape(-1)
            if parameter.grad is None
            else parameter.grad.reshape(-1)
        )
    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces)


def _without_vector(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "vector"}


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator.item()) == 0.0:
        return 0.0
    return float(torch.dot(first, second).div(denominator).item())


def _chunks(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _entity_count_summary(pairs, device: str, batch_pairs: int) -> dict[str, float | int]:
    counts: list[int] = []
    for chunk in _chunks(pairs, batch_pairs):
        batch = _collate_pairs(chunk, device)
        counts.extend(int(value) for value in batch["inputs"]["entity_mask"].sum(dim=1).tolist())
    return {
        "minimum": min(counts),
        "mean": sum(counts) / len(counts),
        "maximum": max(counts),
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True
