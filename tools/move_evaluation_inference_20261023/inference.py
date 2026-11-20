"""Named inference scopes preserving the authenticated CPU float32 pipeline.

This is an instrumentation refactor, not an arithmetic optimization. Helpers
have distinct Python code objects for the selected-region timer; descendants,
kernel work, waits and callback overhead belong to their selected scope. No
Torch import, model construction or inference occurs at module import.
"""
from tools.move_evaluation_reuse_20261023.greedy_adapter import GreedyOpponent

FACTORED_ID = 'entity_inference_scopes_cpu_float32_20261023_v1'


def infer_input_shape(tokens, batch):
    if batch not in (1, 2, 8) or len(tokens) != batch * 31 * 48:
        raise ValueError('Malformed public entity tokens')


def infer_tensor_input(torch, tokens, batch):
    return torch.tensor(tokens, dtype=torch.float32).reshape(batch, 31, 48)


def infer_input_finite(torch, x):
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Nonfinite model input')


def infer_project(model, x):
    return model.project(x) + model.identity


def infer_transformer(model, h):
    return model.blocks(h)


def infer_pool_norm(model, h):
    return model.norm(h[:, 0])


def infer_heads(torch, model, h):
    return torch.cat((model.policy(h), model.value(h).tanh()), dim=1)


def infer_output_finite(torch, out):
    if not bool(torch.isfinite(out).all()):
        raise ValueError('Nonfinite model output')


def infer_output_list(out):
    return out.tolist()


class FactoredOpponent(GreedyOpponent):
    """All construction, policy selection, transport and public reuse inherited."""

    def _infer(self, tokens, batch):
        import torch
        infer_input_shape(tokens, batch)
        with torch.inference_mode():
            x = infer_tensor_input(torch, tokens, batch)
            infer_input_finite(torch, x)
            h = infer_project(self.model, x)
            h = infer_transformer(self.model, h)
            h = infer_pool_norm(self.model, h)
            out = infer_heads(torch, self.model, h)
            infer_output_finite(torch, out)
            return infer_output_list(out)


def fine_targets():
    """Fresh label mapping; never patch model methods or the frozen monitor."""
    return {
        'infer_input_shape': infer_input_shape,
        'infer_tensor_input': infer_tensor_input,
        'infer_input_finite': infer_input_finite,
        'infer_project': infer_project,
        'infer_transformer': infer_transformer,
        'infer_pool_norm': infer_pool_norm,
        'infer_heads': infer_heads,
        'infer_output_finite': infer_output_finite,
        'infer_output_list': infer_output_list,
    }
