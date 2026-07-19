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
]
