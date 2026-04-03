# Splendor Environment Specification

## Scope

This document defines the environment target for the first implementation phase.

- Game: base Splendor only
- Players: exactly 2
- Observation: strictly legal public observation
- Objective: environment suitable for both baseline bots and neural self-play

## Non-Negotiable Properties

- Deterministic under a fixed seed
- No hidden-information leakage in the public observation encoder
- Full legal action coverage
- Action masking available from state
- Environment transitions are pure and testable
- Replayable trajectories for debugging and training

## Game State Decomposition

The canonical internal state should contain:

- current player to act
- round / ply counters
- bank token counts for five gem colors plus gold
- tier decks, including hidden deck order
- visible face-up market cards for each tier
- noble tiles currently available
- per-player token holdings
- per-player purchased cards
- per-player reserved cards
- per-player score
- terminal / winner metadata

Internal state may track hidden deck order, but the legal observation encoder must not expose hidden future cards.

## Legal Observation

Each acting player may observe:

- all public token stacks
- all visible market cards
- all available nobles
- both players' public token holdings
- both players' purchased cards and permanent bonuses
- counts and identities of each player's own reserved cards
- count of opponent reserved cards
- public score, turn information, and remaining deck sizes

The acting player may not observe:

- future deck order
- identities of opponent reserved cards
- hidden top-of-deck cards not yet revealed

## Action Model

The full legal action space must cover:

1. Take tokens
- up to three distinct non-gold colors; if fewer than three colors remain in the bank, the player takes the maximum available distinct count instead
- two of one color, subject to bank threshold rules
- token-return combinations if the player would exceed the hand limit

5. Pass
- if no take, reserve, or buy action is legal, the player must pass

2. Reserve
- reserve one visible market card
- reserve the top card of a tier deck
- take a gold token if available
- token-return combinations if the player would exceed the hand limit

3. Buy
- buy one visible market card
- buy one reserved card
- spend colored and gold tokens according to discounts from permanent bonuses

4. Noble resolution
- when one or more nobles become claimable at end of action, resolve according to the official 2-player rules
- in engine actions, noble choice is attached to the buy action via `noble_id` rather than represented as a separate turn action

## Fixed Action Codec

The implementation should use a fixed action index space instead of variable-length action objects during training.

Recommendation:

- Define a structured `Action` dataclass for engine clarity.
- Define a deterministic `ActionCodec` that maps every possible legal move template to a fixed integer index.
- Use an action mask to mark legal indices from the current state.
- Use slot-based target encoding for board cards, reserved cards, and noble choice so the index space stays stable across states.
- Use a flat legal-observation tensor for the first model iteration; current scaffold uses a 256-float layout with explicit sections for global state, bank, decks, nobles, board, self summary, self reserved cards, and opponent summary.

This separates engine correctness from model representation.

## Rules To Nail Down During Engine Implementation

The following items must be confirmed against the official base-game rulebook and encoded with regression tests:

- exact setup for 2-player token counts and nobles
- end-of-round triggering when a player reaches 15 prestige
- winner selection and tie-break semantics
- noble acquisition timing
- behavior when multiple nobles are simultaneously claimable
- reserve limit and reserve-from-deck handling
- two-of-a-kind token availability threshold
- legal token return combinations when above the token cap
- unresolved exact tie after prestige and development-card tie-break
- behavior when a non-terminal state has no legal actions under the strict action definitions

## Reward / Outcome Interface

The engine should expose:

- terminal winner
- final scores
- optional shaped debugging metrics for evaluation only
- exported stalled-game traces when a non-terminal state yields no legal actions under the current rule interpretation

Training targets should still be able to use sparse win/loss outcomes. Any shaped rewards must be optional and isolated from the canonical environment result.

## Performance Expectations

The simulator will be used for self-play and search, so it should eventually support:

- cheap state cloning or undo
- vectorizable encodings
- efficient legal action enumeration
- reproducible batched rollouts

Correctness comes first; performance work comes after baseline validation.
