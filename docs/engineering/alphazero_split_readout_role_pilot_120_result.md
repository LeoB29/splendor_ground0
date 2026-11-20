# AlphaZero Split-Readout Start-Role Pilot 120 Result

Date completed: 2026-08-28

Experiment 120 completed normally at its development fitting gate. It reused
the sealed 448-pair / 896-row exact-gen11 teacher panel from 118, kept all
356,450 parent parameters frozen, and trained only the separate 96-parameter
policy-role and value-role vectors. It generated no new fitting target or
self-play.

Epoch 6 was selected. Development weighted objective improved from 2.087908 at
epoch zero to 2.083031, a relative improvement of 0.2335%. Policy loss improved
from 1.907804 to 1.907127 and value MSE from 0.360207 to 0.351809. The policy
and value role-vector norms were 0.051812 and 0.092197, both nonzero, while the
parent remained frozen and every metric stayed finite.

This is materially more uptake than Experiment 118's early global-entity
bridge, which improved the same development objective by only 0.0480%.
Nevertheless, it remains below the prospectively required 0.5%. The sole failed
selection condition was the minimum improvement threshold.

The runner therefore stopped before accessing the protected 117 confirmation
targets, the reachable-state safety panel, or gameplay. It ran no external
`035` comparison and performed no installation, promotion, or automatic
continuation. Exact generation 2 remains installed.

Decision:
`split_readout_role_failed_development_uptake_no_confirmation`.

This result closes the exact constant split-additive start-role repair under its
frozen six-epoch protocol. It does not show that start role is irrelevant: both
117's teacher contrast and 120's larger uptake remain real. It shows that two
constant role offsets are still not expressive enough to justify opening the
protected panel. Do not extend the epoch count, lower the gate, or play this
checkpoint after seeing the result. The next review should return to broader,
state-dependent representation—especially how role interacts with round phase,
score, and remaining-turn semantics—rather than another constant role-only
adapter.

Candidate checkpoint SHA-256:
`97f1745307a4bab17dabfcda3cffffe6a817ddbf954d6f79907d83fdbf0c7822`.

Raw result SHA-256:
`3e1ef85c2b4a8231694a60e089e0b366ceba299c5a1925de116657d420ada3cd`.
