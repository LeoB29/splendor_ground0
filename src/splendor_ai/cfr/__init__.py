"""Counterfactual-regret solvers and Splendor game adapters."""

from .analysis import (
    discrepant_action_value_analysis,
    exact_best_response,
    profile_exploitability,
)
from .full_game import (
    DeepCFRSchemaRow,
    FullGameTraversalProbe,
    PublicHistoryEncoding,
    ReservoirMemory,
    build_schema_row,
    collate_deep_cfr_rows,
    encode_public_history,
)
from .full_game_outcome import (
    FINITE_ADJUDICATIONS,
    NATURAL_TERMINALS,
    FrozenEntityTargetPolicy,
    FullGameOutcomeSampler,
    finite_learning_terminal_reason,
)
from .outcome_sampling import (
    OutcomeSamplingAdvantageRow,
    OutcomeSamplingMCCFRSolver,
    OutcomeSamplingUpdate,
    behavior_mixture,
    exact_fixed_policy_regret_increments,
    sample_reduced_outcome_updates,
)
from .public_belief_resolving import (
    ResolvedLeaf,
    TwoPlyResolution,
    deterministic_belief_range,
    resolve_two_ply,
)
from .reduced_splendor import ReducedSplendorCFRGame, ReducedSplendorCFRState
from .tabular import (
    CorrectedExternalSamplingMCCFRSolver,
    ExactCFRSolver,
    ExternalSamplingMCCFRSolver,
)

__all__ = [
    "ExactCFRSolver",
    "ExternalSamplingMCCFRSolver",
    "CorrectedExternalSamplingMCCFRSolver",
    "ReducedSplendorCFRGame",
    "ReducedSplendorCFRState",
    "discrepant_action_value_analysis",
    "exact_best_response",
    "profile_exploitability",
    "DeepCFRSchemaRow",
    "FullGameTraversalProbe",
    "PublicHistoryEncoding",
    "ReservoirMemory",
    "build_schema_row",
    "collate_deep_cfr_rows",
    "encode_public_history",
    "FINITE_ADJUDICATIONS",
    "NATURAL_TERMINALS",
    "FrozenEntityTargetPolicy",
    "FullGameOutcomeSampler",
    "finite_learning_terminal_reason",
    "OutcomeSamplingAdvantageRow",
    "OutcomeSamplingMCCFRSolver",
    "OutcomeSamplingUpdate",
    "behavior_mixture",
    "exact_fixed_policy_regret_increments",
    "sample_reduced_outcome_updates",
    "ResolvedLeaf",
    "TwoPlyResolution",
    "deterministic_belief_range",
    "resolve_two_ply",
]
