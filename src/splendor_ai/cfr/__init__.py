"""Counterfactual-regret solvers and Splendor game adapters."""

from .reduced_splendor import ReducedSplendorCFRGame, ReducedSplendorCFRState
from .tabular import ExactCFRSolver, ExternalSamplingMCCFRSolver

__all__ = [
    "ExactCFRSolver",
    "ExternalSamplingMCCFRSolver",
    "ReducedSplendorCFRGame",
    "ReducedSplendorCFRState",
]
