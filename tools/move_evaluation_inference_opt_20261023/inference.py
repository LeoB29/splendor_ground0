"""Prebind the exact six eligible Torch 2.4.1 CPU fused-layer calls.

Only Python dispatch/eligibility assembly is removed. Original Parameter objects,
native operators, surrounding arithmetic and all scientific outputs are retained.
No Torch import, model construction or inference occurs at module import.
"""
import hashlib
import json
from pathlib import Path

from tools.move_evaluation_inference_20261023.inference import (
    FactoredOpponent, infer_input_shape, infer_tensor_input, infer_input_finite,
    infer_project, infer_pool_norm, infer_heads, infer_output_finite, infer_output_list,
)
from tools.move_evaluation_20261023.greedy_adapter import CHECKPOINT_SHA256, MODEL_DEFINITION_SHA256

IMPLEMENTATION_ID = 'entity_prebound_fused_transformer_cpu_float32_20261023_v1'
TORCH_VERSION = '2.4.1+cpu'
RUNTIME_HASHES = {
    'nn/modules/transformer.py': 'cd8000fc1cea3022f2dfb87f99517484dc02ebf54717232e2a0fdd987a50ec8e',
    'nn/modules/module.py': '4a884b3a62c0795dc52d09fa3e11d7fbab9d733a38d65cc2c6ce291a2d6bfb19',
    'nn/modules/activation.py': '874378929737a0a48eb7bc2c890fd6b5f98ccbacf1a6c6a0b388a09c7d449b45',
    '_C.cp312-win_amd64.pyd': '629c7e69d3928778f36bf2def7821f41040a009bc414b20d0ccb2f9ee22611b8',
    'lib/torch_cpu.dll': 'acf2eafe40886ef4b7afe4f76a2b544a85ddfa36b57fa5029b8aa2d82dfbc3f9',
}
ARGUMENT_ORDER = ('embed_dim', 'num_heads', 'in_proj_weight', 'in_proj_bias',
    'out_proj_weight', 'out_proj_bias', 'use_gelu', 'norm_first', 'eps',
    'norm1_weight', 'norm1_bias', 'norm2_weight', 'norm2_bias',
    'linear1_weight', 'linear1_bias', 'linear2_weight', 'linear2_bias', 'merged_mask', 'mask_type')
PARAMETER_SHAPES = {'in_proj_weight': (768, 256), 'in_proj_bias': (768,),
    'out_proj_weight': (256, 256), 'out_proj_bias': (256,),
    'norm1_weight': (256,), 'norm1_bias': (256,), 'norm2_weight': (256,), 'norm2_bias': (256,),
    'linear1_weight': (1024, 256), 'linear1_bias': (1024,),
    'linear2_weight': (256, 1024), 'linear2_bias': (256,)}
SCALARS = {'embed_dim': 256, 'num_heads': 8, 'use_gelu': True, 'norm_first': True,
           'eps': 1e-5, 'merged_mask': None, 'mask_type': None}
GLOBAL_HOOKS = ('_global_forward_pre_hooks', '_global_forward_hooks',
                '_global_backward_pre_hooks', '_global_backward_hooks')
LOCAL_HOOKS = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')
CONTRACT = {'jit': False, 'prediction_cache': False, 'native_kernel_changed': False,
            'parameter_copy_or_conversion': False, 'parameter_references_immutable_after_binding': True,
            'input_masks': False, 'nested_tensors': False, 'dynamic_global_guard_each_callback': True,
            'local_module_eligibility_checked_at_startup': True}


def require(value, message):
    if not value:
        raise ValueError('PREBOUND_INFERENCE: ' + message)


def _same(value, expected):
    return json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def authenticate_runtime(torch):
    require(torch.__version__ == TORCH_VERSION, 'Torch version')
    root = Path(torch.__file__).resolve().parent
    result = []
    for relative, expected in RUNTIME_HASHES.items():
        path = root/relative
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        require(actual == expected, 'Torch runtime source changed: ' + relative)
        result.append({'relative_path': relative, 'absolute_path': str(path),
                       'sha256': actual, 'bytes': path.stat().st_size})
    return result


def dynamic_eligibility(torch, module_runtime, *, inference):
    require(torch.backends.mha.get_fastpath_enabled() is True, 'MHA fast path disabled')
    require(not torch.is_autocast_enabled() and not torch.is_autocast_enabled('cpu'), 'Autocast enabled')
    require(torch._C._get_tracing_state() is None and not torch.jit.is_scripting(), 'Tracing or scripting enabled')
    require(all(not getattr(module_runtime, name) for name in GLOBAL_HOOKS), 'Global module hooks')
    if inference:
        require(not torch.is_grad_enabled() and torch.is_inference_mode_enabled(), 'Inference mode required')


def _layer_arguments(layer):
    # Exact positional order of the pinned TransformerEncoderLayer fast branch.
    return (layer.self_attn.embed_dim, layer.self_attn.num_heads,
        layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias,
        layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias,
        layer.activation_relu_or_gelu == 2, layer.norm_first, layer.norm1.eps,
        layer.norm1.weight, layer.norm1.bias, layer.norm2.weight, layer.norm2.bias,
        layer.linear1.weight, layer.linear1.bias, layer.linear2.weight, layer.linear2.bias,
        None, None)


def bind_eligible_layers(model, torch, module_runtime):
    """Validate static eligibility and retain immutable tuples of original refs."""
    dynamic_eligibility(torch, module_runtime, inference=False)
    require(model.has_history is False, 'History architecture')
    require(type(model.blocks) is torch.nn.TransformerEncoder and model.blocks.num_layers == 6 and
            len(model.blocks.layers) == 6, 'Exactly six encoder layers required')
    require(model.blocks.norm is None and model.blocks.enable_nested_tensor is False and
            model.blocks.use_nested_tensor is False, 'Encoder norm or nested conversion')
    for module in model.modules():
        require(module.training is False and getattr(module, '_compiled_call_impl', None) is None,
                'Training or compiled module')
        require(all(not getattr(module, name) for name in LOCAL_HOOKS), 'Local module hooks')
    kernel = torch._transformer_encoder_layer_fwd
    require(callable(kernel) and kernel.__name__ == '_transformer_encoder_layer_fwd' and
            kernel.__module__ == 'torch', 'Native fused operator identity')
    bound, descriptions = [], []
    for index, layer in enumerate(model.blocks.layers):
        require(type(layer) is torch.nn.TransformerEncoderLayer and
                type(layer.self_attn) is torch.nn.MultiheadAttention, 'Unexpected layer or attention subclass')
        require(layer.self_attn.batch_first is True and layer.self_attn._qkv_same_embed_dim is True and
                layer.activation_relu_or_gelu == 2 and layer.norm_first is True and
                layer.norm1.eps == layer.norm2.eps == 1e-5, 'Layer fast-path constants')
        args = _layer_arguments(layer)
        values = dict(zip(ARGUMENT_ORDER, args, strict=True))
        require(_same({k: values[k] for k in SCALARS}, SCALARS), 'Layer scalar arguments')
        parameters = []
        for name, shape in PARAMETER_SHAPES.items():
            value = values[name]
            require(type(value) is torch.nn.Parameter and value.device.type == 'cpu' and
                    value.dtype == torch.float32 and value.layout == torch.strided and not value.is_nested and
                    tuple(value.shape) == shape and value.is_contiguous(), 'Parameter eligibility: ' + name)
            require(not torch.overrides.has_torch_function((value,)), 'Parameter tensor override')
            parameters.append({'name': name, 'shape': list(shape), 'stride': list(value.stride()),
                               'dtype': 'torch.float32', 'device': 'cpu'})
        bound.append(args)
        descriptions.append({'index': index, 'scalars': SCALARS.copy(), 'parameters': parameters})
    return kernel, tuple(bound), descriptions


def validate_eligibility_receipt(value):
    """Model-free receipt schema check; surrounding freeze authenticates its file."""
    require(type(value) is dict and set(value) == {'schema', 'status', 'implementation_id', 'torch_version',
        'kernel', 'layer_count', 'static_arguments_per_layer', 'argument_order', 'runtime_inputs',
        'checkpoint_sha256', 'model_definition_sha256', 'original_parameter_references', 'layer_arguments', 'contract'},
        'Eligibility receipt schema')
    require(value['schema'] == 'entity_prebound_fused_layers_eligibility_v1' and value['status'] == 'passed' and
            value['implementation_id'] == IMPLEMENTATION_ID and value['torch_version'] == TORCH_VERSION and
            value['kernel'] == 'torch._transformer_encoder_layer_fwd' and
            type(value['layer_count']) is int and value['layer_count'] == 6 and
            type(value['static_arguments_per_layer']) is int and value['static_arguments_per_layer'] == 19 and
            value['argument_order'] == list(ARGUMENT_ORDER), 'Eligibility identity')
    require(value['checkpoint_sha256'] == CHECKPOINT_SHA256 and value['model_definition_sha256'] == MODEL_DEFINITION_SHA256
            and value['original_parameter_references'] is True and _same(value['contract'], CONTRACT), 'Eligibility contract')
    runtime = value['runtime_inputs']
    require(type(runtime) is list and len(runtime) == len(RUNTIME_HASHES), 'Runtime input count')
    for item, (relative, expected) in zip(runtime, RUNTIME_HASHES.items(), strict=True):
        require(type(item) is dict and set(item) == {'relative_path', 'absolute_path', 'sha256', 'bytes'} and
                item['relative_path'] == relative and item['sha256'] == expected and
                type(item['absolute_path']) is str and Path(item['absolute_path']).is_absolute() and
                type(item['bytes']) is int and item['bytes'] > 0, 'Runtime input identity')
    layers = value['layer_arguments']
    require(type(layers) is list and len(layers) == 6, 'Layer binding count')
    for index, layer in enumerate(layers):
        require(type(layer) is dict and set(layer) == {'index', 'scalars', 'parameters'} and
                type(layer['index']) is int and layer['index'] == index and _same(layer['scalars'], SCALARS),
                'Layer binding identity')
        parameters = layer['parameters']
        require(type(parameters) is list and len(parameters) == len(PARAMETER_SHAPES), 'Layer parameter count')
        for item, (name, shape) in zip(parameters, PARAMETER_SHAPES.items(), strict=True):
            stride = [shape[1], 1] if len(shape) == 2 else [1]
            require(_same(item, {'name': name, 'shape': list(shape), 'stride': stride,
                             'dtype': 'torch.float32', 'device': 'cpu'}), 'Layer parameter descriptor')
    return value


def infer_transformer(opponent, h):
    torch = opponent._prebound_torch
    dynamic_eligibility(torch, opponent._module_runtime, inference=True)
    require(torch._transformer_encoder_layer_fwd is opponent._prebound_kernel, 'Native operator changed after binding')
    for args in opponent._prebound_layers:
        h = opponent._prebound_kernel(h, *args)
    return h


class PreboundOpponent(FactoredOpponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import torch
            from torch.nn.modules import module as module_runtime
            runtime_inputs = authenticate_runtime(torch)
            kernel, layers, descriptions = bind_eligible_layers(self.model, torch, module_runtime)
            self._prebound_torch = torch
            self._module_runtime = module_runtime
            self._prebound_kernel, self._prebound_layers = kernel, layers
            self.eligibility_receipt = validate_eligibility_receipt({
                'schema': 'entity_prebound_fused_layers_eligibility_v1', 'status': 'passed',
                'implementation_id': IMPLEMENTATION_ID, 'torch_version': TORCH_VERSION,
                'kernel': 'torch._transformer_encoder_layer_fwd', 'layer_count': 6,
                'static_arguments_per_layer': 19, 'argument_order': list(ARGUMENT_ORDER),
                'runtime_inputs': runtime_inputs, 'checkpoint_sha256': CHECKPOINT_SHA256,
                'model_definition_sha256': MODEL_DEFINITION_SHA256,
                'original_parameter_references': True, 'layer_arguments': descriptions, 'contract': CONTRACT.copy()})
        except BaseException:
            self.close()
            raise

    def _infer(self, tokens, batch):
        import torch
        infer_input_shape(tokens, batch)
        with torch.inference_mode():
            x = infer_tensor_input(torch, tokens, batch)
            infer_input_finite(torch, x)
            h = infer_project(self.model, x)
            h = infer_transformer(self, h)
            h = infer_pool_norm(self.model, h)
            out = infer_heads(torch, self.model, h)
            infer_output_finite(torch, out)
            return infer_output_list(out)


def fine_targets():
    return {name: globals()[name] for name in ('infer_input_shape', 'infer_tensor_input', 'infer_input_finite',
        'infer_project', 'infer_transformer', 'infer_pool_norm', 'infer_heads', 'infer_output_finite', 'infer_output_list')}
