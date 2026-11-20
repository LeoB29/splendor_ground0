"""Matched observer-legal action/card residuals with a fully trainable 161 policy.

The only information source is the original observation295/action79 encoding.
Known cards retain their *before-action identity* in all 18 rows: a visible
reservation changes its ownership; a purchase removes its availability. Unknown
refills and blind-reserved identities never enter the branch. After-action
deficits describe the same known card, with availability recorded separately.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from splendor_ai.alphazero.strong_policy import StrongPolicyInference
from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel

MODEL_KIND = "structured_proposer_128_v1"
ARMS = ("raw", "relations")
TRAINABLE_PARAMETERS = 9441
MODEL_CONFIG = {
    "observation_size": 295, "action_feature_size": 79, "known_card_slots": 18,
    "pair_common_size": 43, "pair_contrast_size": 12, "pair_size": 55,
    "pair_hidden": 32, "pair_layers": 2, "pooling": "masked_mean_and_max",
    "public_context_size": 61, "head_input_size": 204, "head_hidden": 32,
    "activation": "relu", "branch_trainable_parameters": TRAINABLE_PARAMETERS,
    "final_output_initialization": "exact_zero_weight_and_bias",
    "base_frozen_and_eval": False, "value_head_only_frozen": True, "dropout_free_eval": True, "feature_version": "known_card_relation55_v1",
}
PAIR_LAYOUT = {
    "card": (0, 13), "owner_before": (13, 16), "owner_after": (16, 19),
    "target": (19, 20), "available_after": (20, 21),
    "tokens_before": (21, 27), "tokens_after": (27, 33),
    "bonuses_before": (33, 38), "bonuses_after": (38, 43),
    "colored_before": (43, 48), "colored_after": (48, 53),
    "gold_or_total_shortfall_before": (53, 54), "gold_or_total_shortfall_after": (54, 55),
}
MODEL_CONFIG["pair_channel_layout"] = {key: list(value) for key, value in PAIR_LAYOUT.items()}
MODEL_CONFIG["contrast_raw"] = "colored_tokens_plus_discounts_before_after_div7_and_gold_before_after_div10"
MODEL_CONFIG["contrast_relations"] = "positive_colored_deficits_before_after_div7_and_shared_gold_total_shortfall_before_after_div10"
MODEL_CONFIG["known_card_identity"] = "before_action_identity_with_explicit_after_ownership_and_availability"


def _inputs(observation, actions):
    if observation.ndim != 2 or observation.shape[-1] != 295:
        raise ValueError("Relations require original observation295")
    if actions.ndim != 3 or actions.shape[-1] != 79 or actions.shape[0] != observation.shape[0]:
        raise ValueError("Relations require matching action79 batches")
    if observation.dtype != actions.dtype or observation.device != actions.device:
        raise ValueError("Observation and actions must share dtype and device")


def pair_components(observation, actions):
    """Return exact integer consequences decoded only from legal input tensors.

    Ownership channel order is market, self-reserved, opponent-public-reserved.
    No slot index is an identity: targets are derived from the action's supplied
    tier/market/reserved location, preserving concrete payments and returns.
    """
    _inputs(observation, actions)
    batch, count, _ = actions.shape
    cards = torch.cat((observation[:, 33:189].reshape(batch, 12, 13),
                       observation[:, 203:242].reshape(batch, 3, 13),
                       observation[:, 256:295].reshape(batch, 3, 13)), dim=1)
    known = cards[..., 0] > .5
    owner = observation.new_zeros((18, 3))
    owner[:12, 0], owner[12:15, 1], owner[15:, 2] = 1., 1., 1.
    before_owner = owner[None, None].expand(batch, count, -1, -1) * known[:, None, :, None]
    visible_location = (actions[..., 6:9, None] * actions[..., None, 9:13]).reshape(batch, count, 12)
    visible_target = visible_location * (actions[..., 2] + actions[..., 4])[..., None]
    own_target = actions[..., 13:16] * actions[..., 5, None]
    target = torch.cat((visible_target, own_target, actions.new_zeros((batch, count, 3))), dim=-1)
    target = target * known[:, None]
    purchase = (actions[..., 4] + actions[..., 5])[..., None]
    reservation = actions[..., 2, None]
    after_owner = before_owner * (1. - target[..., None] * (purchase + reservation)[..., None])
    after_owner = after_owner + target[..., None] * reservation[..., None] * owner.new_tensor([0., 1., 0.])
    token_before = torch.round(observation[:, 189:195] * 10.)[:, None].expand(-1, count, -1)
    # Action net flow already includes reserve gold, payments and forced returns.
    token_after = token_before + torch.round(actions[..., 54:60] * actions.new_tensor([4., 4., 4., 4., 4., 5.]))
    bonus_before = torch.round(observation[:, 195:200] * 7.)[:, None].expand(-1, count, -1)
    bonus_after = bonus_before + purchase * actions[..., 19:24]
    return {"cards": cards, "known": known, "owner_before": before_owner,
            "owner_after": after_owner, "target": target, "available_after": after_owner.sum(-1),
            "tokens_before": token_before, "tokens_after": token_after,
            "bonuses_before": bonus_before, "bonuses_after": bonus_after}


def pair_features(observation, actions, arm):
    """Build B x A x18 x55 features and B x18 lawful known-card mask.

    Both arms retain identical complete raw card/resources/discount channels.
    The final12 channels are either redundant raw colored purchasing resources
    (tokens+discounts, /7), gold (/10), or explicit positive colored deficits
    (/7), then total colored deficit minus the ONE shared gold pool (/10).
    Thus neither treatment nor control loses the original constituents.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown residual arm: {arm}")
    p = pair_components(observation, actions)
    _, count, _ = actions.shape
    cards = p["cards"][:, None].expand(-1, count, -1, -1)
    def repeat(value):
        return value[:, :, None].expand(-1, -1, 18, -1)
    common = [cards, p["owner_before"], p["owner_after"], p["target"][..., None],
              p["available_after"][..., None], repeat(p["tokens_before"] / 10.),
              repeat(p["tokens_after"] / 10.), repeat(p["bonuses_before"] / 7.),
              repeat(p["bonuses_after"] / 7.)]
    available_before = repeat(p["tokens_before"][..., :5] + p["bonuses_before"])
    available_after = repeat(p["tokens_after"][..., :5] + p["bonuses_after"])
    gold_before = repeat(p["tokens_before"][..., 5:6])
    gold_after = repeat(p["tokens_after"][..., 5:6])
    if arm == "raw":
        contrasts = [available_before / 7., available_after / 7., gold_before / 10., gold_after / 10.]
    else:
        costs = torch.round(cards[..., 8:13] * 7.)
        deficit_before = torch.clamp(costs - available_before, min=0.)
        deficit_after = torch.clamp(costs - available_after, min=0.)
        contrasts = [deficit_before / 7., deficit_after / 7.,
                     torch.clamp(deficit_before.sum(-1, keepdim=True) - gold_before, min=0.) / 10.,
                     torch.clamp(deficit_after.sum(-1, keepdim=True) - gold_after, min=0.) / 10.]
    result = torch.cat(common + contrasts, dim=-1)
    result = result * p["known"][:, None, :, None]
    return result, p["known"]


def independent_pair_features(observation, actions, arm):
    """Scalar audit implementation, independent of all production arithmetic.

    Inputs are decoded as Python numbers, actions classified, and each known
    card transformed separately. Only schema constants are shared. This is an
    audit path, never a differentiable or production inference implementation.
    """
    if arm not in ARMS:
        raise ValueError("Unknown audit arm")
    if observation.ndim != 2 or observation.shape[-1] != 295 or actions.ndim != 3 or actions.shape[-1] != 79:
        raise ValueError("Audit expects original tensor layout")
    if observation.shape[0] != actions.shape[0]:
        raise ValueError("Audit batch differs")
    all_features, all_masks = [], []
    for obs, action_rows in zip(observation.tolist(), actions.tolist(), strict=True):
        cards = [obs[start:start+13] for start in list(range(33, 189, 13)) + list(range(203, 242, 13)) + list(range(256, 295, 13))]
        masks = [card[0] > .5 for card in cards]
        tokens0, bonuses0 = [round(x*10) for x in obs[189:195]], [round(x*7) for x in obs[195:200]]
        action_outputs = []
        for action in action_rows:
            # Decode three flow blocks separately; do not trust the net block.
            scales = [4, 4, 4, 4, 4, 5]
            tokens1 = [tokens0[i] + round(action[36+i]*scales[i]) - round(action[42+i]*scales[i]) - round(action[48+i]*scales[i]) for i in range(6)]
            is_buy = action[4] > .5 or action[5] > .5
            bonuses1 = [bonuses0[i] + int(is_buy and action[19+i] > .5) for i in range(5)]
            output = []
            for slot, (card, known) in enumerate(zip(cards, masks, strict=True)):
                if not known:
                    output.append([0.] * 55)
                    continue
                owner0 = [int(slot < 12), int(12 <= slot < 15), int(slot >= 15)]
                visible_target = slot < 12 and (action[2] > .5 or action[4] > .5) and action[6+slot//4] > .5 and action[9+slot%4] > .5
                reserved_target = 12 <= slot < 15 and action[5] > .5 and action[13+slot-12] > .5
                target = visible_target or reserved_target
                owner1 = ([0, 0, 0] if is_buy else [0, 1, 0]) if target else list(owner0)
                available0 = [tokens0[i] + bonuses0[i] for i in range(5)]
                available1 = [tokens1[i] + bonuses1[i] for i in range(5)]
                if arm == "relations":
                    before = [max(0, round(card[8+i]*7)-available0[i]) for i in range(5)]
                    after = [max(0, round(card[8+i]*7)-available1[i]) for i in range(5)]
                    totals = [max(0, sum(before)-tokens0[5]), max(0, sum(after)-tokens1[5])]
                else:
                    before, after, totals = available0, available1, [tokens0[5], tokens1[5]]
                output.append(card + owner0 + owner1 + [float(target), float(sum(owner1))]
                    + [v/10 for v in tokens0] + [v/10 for v in tokens1]
                    + [v/7 for v in bonuses0] + [v/7 for v in bonuses1]
                    + [v/7 for v in before] + [v/7 for v in after] + [v/10 for v in totals])
            action_outputs.append(output)
        all_features.append(action_outputs)
        all_masks.append(masks)
    return (torch.tensor(all_features, dtype=observation.dtype, device=observation.device),
            torch.tensor(all_masks, dtype=torch.bool, device=observation.device))


class RelationModel(nn.Module):
    def __init__(self, base, arm, seed):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unknown residual arm")
        if base.config.observation_size != 295 or base.config.action_feature_size != 79:
            raise ValueError("Residual requires the unchanged observation295/action79 interface")
        self.config, self.arm, self.seed = base.config, arm, int(seed)
        self.base = base
        self.base.requires_grad_(True)
        self.base.value_head.requires_grad_(False)
        self.base.eval()
        # Do not alter the caller's RNG, and pair arm initializations exactly.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.pair_encoder = nn.Sequential(nn.Linear(55, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.residual_head = nn.Sequential(nn.Linear(204, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        if sum(p.numel() for m in (self.pair_encoder,self.residual_head) for p in m.parameters()) != TRAINABLE_PARAMETERS:
            raise AssertionError("Residual parameter accounting changed")

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def residual(self, observation, actions, legal_action_mask=None):
        features, mask = pair_features(observation, actions, self.arm)
        hidden = self.pair_encoder(features)
        active = mask[:, None, :, None]
        mean = (hidden * active).sum(-2) / active.sum(-2).clamp(min=1)
        maximum = hidden.masked_fill(~active, -torch.inf).amax(-2)
        maximum = torch.where(mask.any(-1)[:, None, None], maximum, torch.zeros_like(maximum))
        # Global/bank/deck/noble data and both public player summaries. All card
        # data already occur in the shared pair stream. Full action79 is kept.
        context = torch.cat((observation[:, :33], observation[:, 189:203], observation[:, 242:256]), dim=-1)
        context = context[:, None].expand(-1, actions.shape[1], -1)
        correction = self.residual_head(torch.cat((mean, maximum, actions, context), dim=-1)).squeeze(-1)
        if legal_action_mask is not None:
            correction = correction.masked_fill(~legal_action_mask, 0.)
        return correction

    def forward(self, observation, legal_action_features, legal_action_mask=None):
        logits, value = self.base(observation, legal_action_features, legal_action_mask)
        return logits + self.residual(observation, legal_action_features, legal_action_mask), value


def create_model(initializer_path, arm, seed):
    return RelationModel(StrongPolicyInference(initializer_path).model, arm, seed)


def save_model(model, path, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    payload = {"model_kind": MODEL_KIND, "model_config": asdict(model.config),
               "residual_config": dict(MODEL_CONFIG), "arm": model.arm, "seed": model.seed,
               "model_state_dict": model.state_dict(), "metadata": metadata,
               "generation": int(metadata.get("update", -1))}
    torch.save(payload, temporary)
    temporary.replace(path)


class RelationInference(StrongPolicyInference):
    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("model_kind") != MODEL_KIND or payload.get("residual_config") != MODEL_CONFIG:
            raise ValueError("Unsupported or changed relation checkpoint")
        base = LegalActionAttentionScorerModel(LegalActionAttentionScorerConfig(**payload["model_config"]))
        self.model = RelationModel(base, payload["arm"], payload["seed"])
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.eval()
        self.checkpoint = SimpleNamespace(path=Path(path), generation=payload.get("generation", -1))

    @classmethod
    def from_path(cls, path, expected_sha=None):
        if expected_sha is not None and hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha:
            raise ValueError("Relation checkpoint hash differs")
        return cls(path)


def load_model(path):
    return RelationInference(path).model
