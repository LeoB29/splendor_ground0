# AlphaZero State-Dependent Start-Role FiLM Pilot 121 Result

Date completed: 2026-08-28

Experiment 121 completed normally at its development fitting gate. It reused
the sealed 448-pair / 896-row teacher panel, froze every generation-11
parameter, and trained only 384 split policy/value FiLM gain and bias
parameters. It generated no new fitting target or self-play.

Epoch 6 was selected. Development weighted objective improved from 2.087908 to
2.081728, a relative improvement of 0.2960%. Policy loss improved from 1.907804
to 1.906937 and value MSE from 0.360207 to 0.349583. All four FiLM groups were
nonzero: policy gain/bias norms 0.046721/0.051029 and value gain/bias norms
0.087288/0.091213. The parent stayed frozen and all metrics were finite.

State-dependent modulation learned slightly more than Experiment 120's constant
split offsets (0.2960% versus 0.2335%) and substantially more than Experiment
118's early global bridge (0.0480%). It nevertheless remained below the frozen
0.5% requirement. The minimum-improvement threshold was the only failed
selection check.

The runner therefore stopped before accessing the protected 117 confirmation
targets, reachable-state safety, or gameplay. It ran no external `035`
comparison and performed no installation, promotion, or automatic continuation.
Exact generation 2 remains installed.

Decision:
`state_dependent_role_film_failed_development_close_small_role_adapters`.

This closes the bounded small role-adapter line. Start role remains a real
teacher-sensitive variable, but neither early injection, constant direct
offsets, nor feature-wise state-dependent modulation of a completely frozen
parent learned enough to justify protected evaluation. Do not add epochs,
lower the gate, or play this checkpoint. The next architecture review should
consider an explicit start-role-aware encoding with a limited jointly trainable
state representation, rather than another adapter around a frozen role-blind
trunk.

Frozen protocol SHA-256:
`8fd2b022926b1e2e1d65200e53690381de570c2a2a4e3e9974056b2857beb0f4`.

Candidate checkpoint SHA-256:
`06c964057ed9d190b97d409a356dfbec496af9aa1e4b3698e6090ee90d837104`.

Raw result SHA-256:
`f871427980697f4036c84b7617ba671db801563af6d6a0434f698a7d473d4e66`.
