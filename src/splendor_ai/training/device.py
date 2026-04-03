"""Backend selection for training and evaluation.

This module intentionally avoids importing PyTorch at import time so the core
package stays lightweight until ML dependencies are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BackendName = Literal["cuda", "cpu", "directml"]


@dataclass(frozen=True, slots=True)
class TrainingBackend:
    name: BackendName
    description: str


def resolve_training_backend(
    prefer_cuda: bool = True,
    allow_directml: bool = True,
) -> TrainingBackend:
    """Resolve the preferred backend policy for later PyTorch initialization.

    Current behavior is policy-only. Runtime hardware probing will be added when
    the training stack is introduced.
    """

    if prefer_cuda:
        return TrainingBackend(
            name="cuda",
            description="Primary training path for the RTX 5070.",
        )
    if allow_directml:
        return TrainingBackend(
            name="directml",
            description="Compatibility fallback for Windows accelerator support.",
        )
    return TrainingBackend(
        name="cpu",
        description="Portable CPU fallback.",
    )
