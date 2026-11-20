"""Lossless legal-feature restructuring and shared-card learner for 151."""
import math
import torch

from splendor_ai.alphazero.afterstate_149 import CARDS
from splendor_ai.engine.constants import TOKEN_COLORS

FLAT_WIDTH = 1085
CONTEXT_WIDTH = 455
CARD_WIDTH = 22
CARD_COUNT = 90
PARAMETERS = 327297


def catalog_tensor():
    values = [[*[c.cost.get(color, 0)/7 for color in TOKEN_COLORS],
               *[float(c.bonus_color == color) for color in TOKEN_COLORS], c.points/5,
               *[float(c.tier == tier) for tier in (1, 2, 3)]] for c in CARDS]
    if len(values) != CARD_COUNT or len({tuple(v) for v in values}) != CARD_COUNT:
        raise ValueError("Card attributes must identify the public catalog uniquely")
    return torch.tensor(values, dtype=torch.float32)


def restructure(flat):
    """No simulator or hidden state: transform the frozen 149 observable vector."""
    if flat.shape[-1] != FLAT_WIDTH:
        raise ValueError("151 requires the exact 149 feature schema")
    status = flat[..., 295:835].reshape(*flat.shape[:-1], CARD_COUNT, 6)
    pool = flat[..., 835:925].unsqueeze(-1)
    target = flat[..., -101:-11].unsqueeze(-1)
    attributes = catalog_tensor().to(flat).expand(*flat.shape[:-1], CARD_COUNT, 14)
    cards = torch.cat((attributes, status, pool, target), dim=-1)
    context = torch.cat((flat[..., :295], flat[..., 925:]), dim=-1)
    return context, cards


def reconstruct(context, cards):
    """Exact audit inverse in canonical catalog order."""
    if context.shape[-1] != CONTEXT_WIDTH or cards.shape[-2:] != (CARD_COUNT, CARD_WIDTH):
        raise ValueError("Invalid structured feature dimensions")
    expected = catalog_tensor().to(cards).expand(*cards.shape[:-2], CARD_COUNT, 14)
    if not torch.equal(cards[..., :14], expected):
        raise ValueError("Inverse requires canonical catalog rows")
    if not torch.equal(cards[..., 21], context[..., -101:-11]):
        raise ValueError("Card target tags differ from the complete action")
    return torch.cat((context[..., :295], cards[..., 14:20].flatten(-2),
                      cards[..., 20], context[..., 295:]), dim=-1)


class SharedCardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.card_encoder = torch.nn.Sequential(torch.nn.Linear(22, 32), torch.nn.GELU(),
            torch.nn.LayerNorm(32), torch.nn.Linear(32, 32), torch.nn.GELU())
        self.context_encoder = torch.nn.Sequential(torch.nn.Linear(455, 256), torch.nn.LayerNorm(256),
            torch.nn.GELU(), torch.nn.Linear(256, 128), torch.nn.LayerNorm(128), torch.nn.GELU())
        self.query = torch.nn.Linear(128, 32)
        self.key = torch.nn.Linear(32, 32)
        self.value = torch.nn.Linear(32, 64)
        self.head = torch.nn.Sequential(torch.nn.Linear(224, 448), torch.nn.LayerNorm(448),
            torch.nn.GELU(), torch.nn.Linear(448, 128), torch.nn.LayerNorm(128), torch.nn.GELU(),
            torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Linear(64, 1), torch.nn.Tanh())
        if sum(p.numel() for p in self.parameters()) != PARAMETERS:
            raise ValueError("151 parameter contract changed")

    def forward(self, context, cards):
        encoded = self.card_encoder(cards)
        summary = self.context_encoder(context)
        logits = (self.key(encoded)*self.query(summary).unsqueeze(-2)).sum(-1)/math.sqrt(32)
        pooled = (logits.softmax(-1).unsqueeze(-1)*self.value(encoded)).sum(-2)
        scores = self.head(torch.cat((summary, pooled, encoded.mean(-2)), dim=-1)).squeeze(-1)
        return scores-scores[..., :1]
