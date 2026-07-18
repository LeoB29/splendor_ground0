"""Counterfactual-regret solvers and Splendor game adapters."""

from .analysis import (
    discrepant_action_value_analysis,
    exact_best_response,
    profile_exploitability,
)
from .reduced_splendor import ReducedSplendorCFRGame, ReducedSplendorCFRState
from .tabular import ExactCFRSolver, ExternalSamplingMCCFRSolver

__all__ = [
    "ExactCFRSolver",
    "ExternalSamplingMCCFRSolver",
    "ReducedSplendorCFRGame",
    "ReducedSplendorCFRState",
    "discrepant_action_value_analysis",
    "exact_best_response",
    "profile_exploitability",
]
