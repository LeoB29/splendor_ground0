# AlphaZero Start-Role Interaction Audit 119 Result

Date completed: 2026-08-28

Experiment 119 completed normally as a no-training architecture and gradient
audit. It reused the sealed 448-pair / 896-row 118 fitting panel and exact
179/45 whole-game split. It did not read the protected 117 confirmation panel,
generate targets, instantiate an optimizer, mutate a parameter, write a
candidate checkpoint, run self-play, or enter gameplay.

The architecture audit identified a concrete attenuation path in 118. Its role
vector affects one global entity, then passes through two post-normalized frozen
transformer layers and is averaged with 52--71 entities (mean 62.663) before
either head sees it. All three 119 prototypes reproduced exact gen11 logits and
values on all 896 oriented rows with maximum error exactly zero and deterministic
repeatability.

For the diagnostic antisymmetric version of 118's global role channel, all-pair
policy contrast-gradient RMS was `2.417669e-6` and value contrast-gradient RMS
was `0.004041563`. Train/validation gradient cosines were 0.918566 and 0.999880.

The 192-parameter `split_readout_additive` candidate applies independent signed
96-vectors after entity pooling: one directly to the policy state before both
action context and scoring, and one directly to the value state. It passed every
frozen gate:

- policy contrast-gradient RMS `3.224555e-5`, 13.337 times the 118 channel;
- value contrast-gradient RMS `0.040341865`, 9.982 times the 118 channel;
- train/validation gradient cosine 0.445825 for policy and 0.999924 for value;
- finite, nonzero gradients for both endpoints;
- exactly zero policy-to-value and value-to-policy gradient leakage;
- 192 total trainable parameters, all still exactly zero after the audit.

The 384-parameter split FiLM fallback also passed, but its per-parameter gradient
RMS was lower than the additive candidate for both endpoints. Under the frozen
simplicity-first decision rule it is not preferred.

Decision:
`direct_split_additive_role_gradient_qualifies_freeze_minimal_pilot_no_training`.

The recommended next step is to design and preflight one bounded
identity-preserving `split_readout_additive` joint policy/value pilot using the
existing sealed 118 targets and protected confirmation/safety structure. The
audit does not authorize fitting, new target generation, confirmation access,
self-play, gameplay, external `035`, installation, promotion, or automatic
continuation. A new pilot requires a separately frozen protocol and explicit
launch authorization.

Frozen protocol SHA-256:
`9f5179d270438ee13514931a741da141d3de6478f336b9c9c5e8aca07f83b558`.

Raw result SHA-256:
`6256cc23f24e7c6b601884bbdbbe0fb9f9c4126e64700818db03f692f6fd1828`.
