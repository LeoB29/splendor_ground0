"""Read primary saved components as data; never import or modify a game engine.

The licensed BGA tutorial provides the literal component database. Its old
static assets currently return HTTP 403, so the explicit color decoder comes
from the separately saved current BGA release. This version distinction remains
in the receipt, together with one directly inspected publisher illustration.
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/component_adjudication_20261018_01"
COLORS = ("white", "blue", "green", "red", "black")
TUTORIAL_URL = "https://en.boardgamearena.com/tutorial?game=splendor"
CURRENT_BASE = "https://x.boardgamearena.net/data/themereleases/current/games/splendor/260627-0953/"
UPSTREAM = ROOT / ("outputs/external_strength_preparation_20261018_01/qualification_sources/"
                   "roeey777/src/splendor/splendor/splendor_utils.py")
PDF_URL = "https://cdn.svc.asmodee.net/production-unboxnowcom/uploads/2022/02/Splendor-EN.pdf"
PINS = {
    "bga_tutorial.html": "4848130623139505046540f5ec1e6964c34e7fb1a0a5625a0209aa47af74b0f8",
    "bga_splendor.js": "8a93f7a4c816c89c0bc2f024856474184ed55c617f0df303aea23f968651cfb4",
    "bga_splendor.css": "4edbf971e5f755a977030bbba95241146a1090c849531037e3fd8d51e06ade4b",
    "publisher_rulebooks/splendor_classic_rules.pdf": "42626f3cd76d41abfd25eac1b765cccf5650a98e6b23ba37e5a29690c941396c",
}
LOCAL_PIN = "8af40dec97cd7f5aff6a0b1606568fd8788327aec23aebf7a4030f3cafed66fb"
UPSTREAM_PIN = "f4318e25da33ff14acab12cdad3e04377f1b2f5f89ac674dbebe07d118631444"
DISPUTED = {"T1_05": "2", "T1_07": "1", "T2_10": "42",
            "T2_12": "44", "T2_13": "45", "T2_14": "46"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def artifact(path, source_url=None):
    data = path.read_bytes()
    result = {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": digest_bytes(data)}
    if source_url:
        result["source_url"] = source_url
    return result


def embedded_object(html, key):
    matches = list(re.finditer(r'"' + re.escape(key) + r'"\s*:', html))
    require(len(matches) == 1, f"Ambiguous embedded {key}")
    value, _ = json.JSONDecoder().raw_decode(html[matches[0].end():].lstrip())
    require(isinstance(value, dict), f"{key} is not an object")
    return value


def literal_assignments(path, names):
    result = {}
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                require(target.id not in result, "Repeated literal assignment")
                result[target.id] = ast.literal_eval(node.value)
    require(set(result) == set(names), "Missing literal catalogue assignment")
    return result


def cost_tuple(cost):
    require(set(cost) <= set(COLORS), "Unknown cost color")
    require(all(type(value) is int and value >= 0 for value in cost.values()), "Invalid cost amount")
    return tuple(cost.get(color, 0) for color in COLORS)


def face(tier, bonus, points, cost):
    require(type(tier) is int and tier in (1, 2, 3), "Invalid tier")
    require(bonus in COLORS and type(points) is int and points >= 0, "Invalid bonus/points")
    return tier, bonus, points, cost_tuple(cost)


def show_face(value):
    tier, bonus, points, cost = value
    return {"tier": tier, "bonus": bonus, "points": points,
            "cost": {color: n for color, n in zip(COLORS, cost) if n}}


def check_cards(cards, name):
    require(len(cards) == len(set(cards.values())) == 90, f"{name} needs 90 unique faces")
    require(Counter(value[0] for value in cards.values()) == {1: 40, 2: 30, 3: 20}, f"{name} tier counts")


def inspect():
    for relative, wanted in PINS.items():
        require(digest_bytes((BASE / relative).read_bytes()) == wanted, "Changed pinned evidence: " + relative)
    local_path = ROOT / "src/splendor_ai/engine/data.py"
    require(digest_bytes(local_path.read_bytes()) == LOCAL_PIN, "Historical local catalogue changed")
    require(digest_bytes(UPSTREAM.read_bytes()) == UPSTREAM_PIN, "Pinned external catalogue changed")
    html = (BASE / "bga_tutorial.html").read_text(encoding="utf-8")
    carddb, nobledb = embedded_object(html, "carddb"), embedded_object(html, "nobledb")
    require(carddb == json.loads((BASE / "bga_tutorial_carddb.json").read_text(encoding="utf-8")),
            "Independent raw-HTML extraction disagrees with initial extraction")
    require(set(carddb) == {str(i) for i in range(1, 91)}, "Expected base card IDs 1 through 90")
    js = (BASE / "bga_splendor.js").read_text(encoding="utf-8")
    css = (BASE / "bga_splendor.css").read_text(encoding="utf-8")
    type_function = js.split("cardtypetoletter:function", 1)[1].split("letterToCardType", 1)[0]
    type_letters = {int(index): letter for index, letter in re.findall(r'case (\d+):return"([CSERO])"', type_function)}
    letter_colors = dict(re.findall(r'\.type_([CSERO]) \.spl_cardheader\{background-image:url\(img/ch_(white|blue|green|red|black)\.png\)', css))
    require(type_letters == {0: "C", 1: "S", 2: "E", 3: "R", 4: "O"}, "Unexpected BGA type map")
    require(letter_colors == {"C": "white", "S": "blue", "E": "green", "R": "red", "O": "black"}, "Unexpected CSS color map")

    def bga_cost(text):
        require(type(text) is str and set(text) <= set(letter_colors), "Invalid BGA cost string")
        return {letter_colors[letter]: amount for letter, amount in Counter(text).items()}

    bga_cards = {key: face(v["lvl"], letter_colors[type_letters[v["type"]]], v["points"], bga_cost(v["cost"]))
                 for key, v in carddb.items()}
    bga_nobles = {(value["points"], cost_tuple(bga_cost(value["cost"]))) for key, value in nobledb.items()
                  if value["promo"] is False}
    require(len(bga_nobles) == 10, "Expected ten distinct non-promo nobles")
    require(all(points == 3 for points, _ in bga_nobles), "Unexpected noble points")
    local = literal_assignments(local_path, {"_RAW_CARD_ROWS", "_RAW_NOBLE_ROWS", "_LEGACY_GEM_INDEX_TO_COLOR"})
    cmap = local["_LEGACY_GEM_INDEX_TO_COLOR"]
    local_cards = {}
    for index, row in enumerate(local["_RAW_CARD_ROWS"], 1):
        tier0, points, bonus, *cost = row
        require(len(cost) == 5, "Invalid local card row")
        local_cards[f"T{tier0 + 1}_{index:02d}"] = face(tier0 + 1, cmap[bonus], points, {cmap[i]: value for i, value in enumerate(cost)})
    local_nobles = {(3, cost_tuple({cmap[i]: n for i, n in enumerate(row)})) for row in local["_RAW_NOBLE_ROWS"]}
    upstream = literal_assignments(UPSTREAM, {"CARDS", "NOBLES"})
    upstream_cards = {key: face(tier, bonus, points, cost) for key, (bonus, cost, tier, points) in upstream["CARDS"].items()}
    upstream_nobles = {(3, cost_tuple(cost)) for _, cost in upstream["NOBLES"]}
    for name, cards in (("BGA", bga_cards), ("local", local_cards), ("upstream", upstream_cards)):
        check_cards(cards, name)
    require(set(upstream_cards.values()) == set(bga_cards.values()), "Upstream differs from licensed primary components")
    require(local_nobles == upstream_nobles == bga_nobles, "Noble catalogue discrepancy")
    inverse_bga = {value: key for key, value in bga_cards.items()}
    inverse_upstream = {value: key for key, value in upstream_cards.items()}
    unmatched = {key for key, value in local_cards.items() if value not in inverse_bga}
    require(unmatched == set(DISPUTED), "Unexpected local component discrepancies")
    reconciled = dict(local_cards)
    differences = []
    for local_id, bga_id in DISPUTED.items():
        before, after = local_cards[local_id], bga_cards[bga_id]
        require(before[:3] == after[:3], "Proposed ID correspondence changes tier/bonus/points")
        reconciled[local_id] = after
        differences.append({"local_card_id": local_id, "bga_card_id": bga_id,
                            "bga_literal": carddb[bga_id], "historical_local": show_face(before),
                            "licensed_primary_face": show_face(after), "upstream_id": inverse_upstream[after]})
    require(set(reconciled.values()) == set(bga_cards.values()) and len(set(reconciled.values())) == 90,
            "Six proposed correspondences do not form a complete bijection")
    sources = [artifact(BASE / "bga_tutorial.html", TUTORIAL_URL),
               artifact(BASE / "bga_splendor.js", CURRENT_BASE + "splendor.js"),
               artifact(BASE / "bga_splendor.css", CURRENT_BASE + "splendor.css"),
               artifact(BASE / "publisher_rulebooks/splendor_classic_rules.pdf", PDF_URL),
               artifact(local_path), artifact(UPSTREAM,
                   "https://raw.githubusercontent.com/roeey777/Splendor-AI/95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b/src/splendor/splendor/splendor_utils.py")]
    return {"schema": "independent_component_adjudication_v1", "time": datetime.now(timezone.utc).isoformat(),
            "checker": artifact(Path(__file__).resolve()), "sources": sources,
            "primary_provenance": "Licensed BGA implementation, developed jointly with Space Cowboys; corroborated by one publisher-hosted physical-card illustration",
            "publisher_authorization_source": "https://www.asmodee.co.uk/blogs/news/splendor-released-on-board-game-arena",
            "color_decoding": {"numeric_type_to_letter": type_letters, "letter_to_color": letter_colors,
                               "tutorial_release": "210916-2237", "decoder_release": "260627-0953",
                               "version_gap": "Tutorial-specific JS and CSS URLs returned HTTP403. Applying the current explicit BGA decoder to the older tutorial is a documented inference, corroborated by the publisher illustration and full90 semantic correspondence."},
            "direct_publisher_visual_corroboration": {
                "source_url": PDF_URL, "pdf_page_one_based": 2, "location": "Game setup illustration; middle tier, leftmost face-up card",
                "visible_face": {"tier": 2, "bonus": "white", "points": 2, "cost": {"red": 5}},
                "bga_card_id": "44", "local_card_id": "T2_12", "unprinted_cost_colors": ["white", "blue", "green", "black"],
                "inspection": "Human-readable image visually inspected, not inferred from either AI repository",
                "derived_image": artifact(BASE / "publisher_rulebooks/classic_setup_detail-2.png"),
                "render_command": "pdftoppm -f 2 -l 2 -scale-to 4400 -x 1980 -y 2350 -W 1150 -H 1040 -png splendor_classic_rules.pdf classic_setup_detail"},
            "counts": {"bga_cards": 90, "tier_counts": {"1": 40, "2": 30, "3": 20},
                       "bga_base_nobles": 10, "upstream_cards_matching_bga": 90,
                       "historical_local_cards_matching_bga": 84, "local_and_upstream_nobles_matching_bga": 10,
                       "after_six_correspondences_cards_matching_bga": 90},
            "disputed_faces": differences,
            "full_upstream_to_bga_join": {key: inverse_bga[value] for key, value in upstream_cards.items()},
            "full_proposed_local_to_bga_join": {key: inverse_bga[value] for key, value in reconciled.items()},
            "conclusion": "All six licensed-primary faces agree with pinned external PPO catalogue. The historical local catalogue has six cost errors relative to that base-game component set.",
            "changes_applied": False, "external_checkpoint_loaded": False, "games_run": 0,
            "limits": ["Six physical card photographs have not all been individually inspected; all90 faces are verified against licensed primary implementation data.",
                       "Catalogue agreement alone does not qualify external observations, actions, checkpoint execution, clocks or playing strength.",
                       "Existing local-rule experiment evidence is preserved and remains scoped to its historical rules."]}


if __name__ == "__main__":
    receipt = inspect()
    path = BASE / "independent_component_adjudication.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"receipt": str(path), "counts": receipt["counts"], "games_run": 0}, sort_keys=True))
