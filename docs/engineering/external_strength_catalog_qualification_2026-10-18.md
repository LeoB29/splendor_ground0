# External PPO catalog qualification: blocked before inference

The pinned `roeey777/Splendor-AI` opponent cannot presently qualify as a
same-rules external calibration opponent. A semantic join matches **84 of 90
cards and all 10 nobles**. No upstream source was imported or executed, no model
was deserialized, and no model inference or arena game was run in this check.

The local source is `src/splendor_ai/engine/data.py`, which the authoritative
environment uses through `build_base_deck_by_tier()` and `build_base_nobles()`.
The external source is `src/splendor/splendor/splendor_utils.py` at commit
`95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b`. Both were parsed with Python's AST
literal evaluator; neither source module was executed. Card signatures include
tier, bonus, points and all five colored costs. Noble signatures include points
and all five requirements. Both sources contain 90 distinct card faces with
tier counts 40/30/20 and 10 distinct noble faces.

The following pairs illustrate the disagreement; **they are not an accepted
mapping or a proposed correction**. Every listed card has a white bonus.

| Local ID | Tier / points | Local cost | External code | External cost |
|---|---|---|---|---|
| T1_05 | 1 / 0 | blue 2, green 2 | 2b2B | blue 2, black 2 |
| T1_07 | 1 / 0 | blue 1, black 1 | 1b1B3w | blue 1, black 1, white 3 |
| T2_10 | 2 / 1 | white 2, blue 3, red 3, green 4 | 3b3r2w | white 2, blue 3, red 3 |
| T2_12 | 2 / 2 | red 5, green 4 | 5r | red 5 |
| T2_13 | 2 / 2 | red 5, black 3, green 4 | 5r3B | red 5, black 3 |
| T2_14 | 2 / 3 | white 6, green 4 | 6w | white 6 |

An identifier translation cannot remove these differences. The pinned PPO's
`features.py` function `vectorize_card()` puts the actual five costs into its
input. `missing_gems_to_card()` also uses those costs to compute distance
features. Supplying each local card's real face therefore changes the network
input relative to the illustrated external face. Substituting the external
face would instead describe a different cost than the authoritative engine
charges. Neither option establishes a faithful same-rules comparison. Reusing
the network on a different component set would require an explicitly different
research question; this check does not silently authorize that change.

This evidence establishes disagreement, **not which catalog is officially
correct**. Resolve that question independently before proposing any change to
the local rules or component data. Historical frozen sources and evidence stay
intact. Even after a catalog match, observation equivalence, action coverage,
empty-mask behavior, safe checkpoint loading, output equivalence and the
one-second execution limit remain unpassed gates. In particular, source rules
omit blind reserve, cap purchased cards at seven per color and prune some
take/return combinations; these were not waived by this catalog check.

`tools/external_strength_qualification/adapter.py` exposes
`qualify_sources(repo_root)` for the static receipt. `create_adapter(repo_root)`
fails closed and cannot create a playing opponent. It does not import Torch or
the external project. A matching catalog by itself also cannot enable play.
The scoped synthetic checks passed: **15 tests in 0.12 seconds** using
`python -m pytest -q tests/test_external_strength_adapter.py`.

The complete receipt, including all matched identities, all unmatched semantic
rows, exact source hashes, nine additional pinned source-file identities and
qualification-code hashes, is
`outputs/external_strength_preparation_20261018_01/qualification_catalog_check_01.json`.
The added sources are in the new `qualification_sources/` directory; the
previously frozen `upstream/` files and preparation receipts were preserved.

Primary source: [pinned external component catalog](https://github.com/roeey777/Splendor-AI/blob/95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b/src/splendor/splendor/splendor_utils.py),
[pinned feature extraction](https://github.com/roeey777/Splendor-AI/blob/95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b/src/splendor/splendor/features.py).
