# Component Data Provenance

## Rules Authority

The authoritative rules source for this repository is:

- `c:\Users\leoba\Desktop\splendor_ai\docs\rulebook\ba-splendor-rulebook.pdf`

The following gameplay behaviors are directly anchored to that PDF:

- 2-player setup counts
- reveal 4 cards per level
- reveal `players + 1` nobles
- reserve / buy / token-take action categories
- token cap of 10
- immediate replacement of reserved or purchased face-up cards
- noble timing and single-noble-per-turn resolution
- end-of-round procedure after reaching 15 prestige
- tie-break by fewest development cards purchased

## Component List Source

The rulebook PDF specifies component counts, but it does not enumerate the exact faces of all 90 development cards and 10 noble tiles in machine-readable text.

For this repository, the full component list was transcribed from the complete local legacy dataset at:

- `c:\Users\leoba\Desktop\splendor_ai\utils.py`

That legacy file contains:

- 90 encoded development cards
- 10 encoded noble tiles

The new repository decodes that legacy representation into explicit `Card` and `Noble` records with stable ids.

## Important Note

If you want a different component-list authority than the legacy local dataset, provide it and this repository should be updated to match that source exactly.
