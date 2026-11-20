# Expanded-Root Resolver Challenge 087 — Strategic Stop Record

Date: 2026-08-11  
Stop verified: approximately 17:51 Europe/Paris  
Status: owner-authorized early termination; no strength conclusion

## Decision

The owner explicitly authorized termination of the post-hoc 087 continuation
and requested that the decision be recorded. The reason is strategic
reprioritization: the programme has moved to the AlphaZero loop, the original
087 formal experiment was already invalid at its smoke gate, and completing
the separately owner-waived strength block could not change the current
research direction or either champion.

At the verified stop boundary:

- 102 of 200 paired seeds were durably complete;
- 98 paired seeds remained;
- recorded active wall time was 64.7432 hours;
- runner stderr was zero bytes;
- progress last write was 2026-08-11 17:18:22 Europe/Paris;
- progress SHA-256 was
  `d6a85ba75d4fee94f10fbace76115e4b2f74e88b71410004df439264ad0a36ce`.

The exact process tree rooted at parent PID 35528 was verified before stopping.
It contained the parent, ten Python workers, and one console host. All twelve
processes were terminated and no listed process survived. The independent 095
process, PID 11400, was explicitly verified to be outside this tree and
remained alive with zero stderr.

## Evidence boundary

No 087 game outcomes were inspected before or during the stop decision. The
partial progress artifact is preserved unchanged at:

`outputs/expanded_root_resolver_champion_challenge_087/strength_vs_research_champion.progress.json`

Its SHA-256 reproduced unchanged after termination. The immutable original
formal `result.json`, failed smoke record, owner-waiver record, and partial
progress must not be rewritten as a completed experiment.

Because the frozen dose was 200 pairs and only 102 completed, the partial panel
cannot produce a formal strength decision, cannot be used for promotion, and
should remain outcome-sealed. There is no champion change. Any future decision
to inspect these partial outcomes would require a separately labelled
exploratory audit and would not revive the prospective 087 claim.

## Strategic rationale

Stopping avoids spending roughly another two to three days of ten-worker
compute on a post-hoc result with little decision value. The released compute
is assigned to the active AlphaZero programme, beginning with the unchanged
095 confirmation. This is a prospective resource-allocation decision, not an
inference about the unobserved 087 outcome.
