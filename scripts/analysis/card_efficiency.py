"""Card efficiency and the reserve-gating mechanic.

Supports `docs/project_review_response_070.md` section 4 and the premises of
experiments `072`/`073`:

* points per token is monotone in cost *concentration* within each tier;
* for every colour X exactly two tier-3 cards cost `7X`, worth 4 and 5 points,
  and both grant the same bonus colour;
* `BANK_GEMS_PER_COLOR_2P = 4`, so a `7X` cost can never be paid from tokens of
  colour X alone -- at least 3 must come from gold or discounts, and gold is
  obtainable only by reserving.

Run from the repository root:

    python scripts/analysis/card_efficiency.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from splendor_ai.engine.constants import (
    BANK_GEMS_PER_COLOR_2P,
    GOLD_TOKENS,
    TOKEN_COLORS,
)


CARDS = Path("docs/splendor_base_cards.csv")


def load() -> list[dict[str, object]]:
    rows = []
    with CARDS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cost = {color: int(row[f"cost_{color}"]) for color in TOKEN_COLORS}
            total = int(row["total_cost_tokens"])
            points = int(row["prestige_points"])
            rows.append(
                {
                    "id": row["card_id"],
                    "tier": int(row["tier"]),
                    "bonus": row["bonus_color"],
                    "points": points,
                    "cost": cost,
                    "total": total,
                    "colours": sum(1 for value in cost.values() if value > 0),
                    "peak": max(cost.values()),
                    "ppt": points / total if total else 0.0,
                }
            )
    return rows


def main() -> None:
    rows = load()
    print(f"{len(rows)} cards; bank {BANK_GEMS_PER_COLOR_2P}/colour, {GOLD_TOKENS} gold\n")

    print("points per token by tier and number of distinct cost colours")
    grouped = defaultdict(list)
    for row in rows:
        if row["points"] > 0:
            grouped[(row["tier"], row["colours"])].append(row["ppt"])
    for key in sorted(grouped):
        values = grouped[key]
        print(
            f"  tier {key[0]}  {key[1]} colour(s):"
            f" mean {sum(values)/len(values):.3f}"
            f"  range {min(values):.3f}-{max(values):.3f}  n={len(values)}"
        )

    print("\nmono-colour 7-cost tier-3 pairs (the reserve-gated cards)")
    mono = [row for row in rows if row["peak"] == 7]
    by_colour = defaultdict(list)
    for row in mono:
        pay = next(c for c, v in row["cost"].items() if v == 7)
        by_colour[pay].append(row)
    for pay in sorted(by_colour):
        cards = sorted(by_colour[pay], key=lambda r: r["points"])
        bonuses = {card["bonus"] for card in cards}
        summary = ", ".join(
            f"{card['id']} {card['points']}pts/{card['total']}tok" for card in cards
        )
        print(f"  pay 7 {pay:6}: {summary}   grants {sorted(bonuses)}")
        assert len(bonuses) == 1, "the pair should grant one shared bonus colour"

    print("\nreserve gating")
    print(
        f"  a 7-of-a-colour cost exceeds the {BANK_GEMS_PER_COLOR_2P}-token bank"
        f" stack, so at least {7 - BANK_GEMS_PER_COLOR_2P} must come from gold or"
        " discounts."
    )
    print(
        f"  gold is obtainable only by reserving, {GOLD_TOKENS} exist, one per"
        " reserve, so the most efficient cards are structurally reserve-gated"
        " before a discount stack exists."
    )

    print("\nefficiency filter used by 072/073 (effective value >= 0.40, zero bonus)")
    admitted = [row for row in rows if row["ppt"] >= 0.40]
    per_tier: dict[int, int] = {}
    for row in admitted:
        per_tier[row["tier"]] = per_tier.get(row["tier"], 0) + 1
    print(f"  admits {len(admitted)} of {len(rows)} cards, by tier {dict(sorted(per_tier.items()))}")


if __name__ == "__main__":
    main()
