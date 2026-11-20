"""Bounded integration check of the actual hidden single-process runtime."""
from pathlib import Path
import sys
import traceback
import torch
from . import common as c
from splendor_ai.alphazero.strong_policy import StrongPolicyInference


def main(destination):
    base = Path(destination).resolve()
    c.require(not base.is_relative_to(c.OUTPUT.resolve()), "No production fixture")
    health = c.windows.assert_windowless_worker(base/"stdout.txt", base/"stderr.txt")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ref = c.old.read_json(base/"fixture_model.json")
    model = StrongPolicyInference(c.verify_hash(ref)).model
    c.old.save_new(base/"complete.json", {"health": health, "threads": torch.get_num_threads(),
                  "deterministic": torch.are_deterministic_algorithms_enabled(), "model": ref,
                  "parameters": sum(p.numel() for p in model.parameters()), "eval": not model.training})


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try:
        main(destination)
    except BaseException:
        c.old.save_new(destination/"failure.json", {"traceback": traceback.format_exc()})
        raise
