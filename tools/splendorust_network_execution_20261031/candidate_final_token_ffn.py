"""One exactness-qualified candidate: only encoder6's FFN processes token0.

The factory copies/replaces IR and constructs a ScriptFunction. It never calls a
model, traces, warms, times candidates, changes a parameter or selects a fallback.
Mathematical equivalence does not guarantee bitwise MKL parity when M changes;
the separately frozen actual qualification is authoritative.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


CANDIDATE_ID = 'final_token_only_last_ffn_v1'
REFERENCE_GRAPH_SHA256 = '87b5d357c610929fc527155c0f2d7c09f6367e74c6556e3d5c7167844fb0341c'
MAX_GRAPH_BYTES = 128*1024
_ATTEMPTED = False
ENCODER = 'aten::_transformer_encoder_layer_fwd'

# Exactly the native pre-norm/GELU encoder sequence. norm_first is checked true
# before insertion. Full attention, both norms and the first residual retain all
#31 tokens. Only the two FFN matrix operations have fewer rows.
REPLACEMENT_IR = r'''graph(%src : Tensor, %embed : int, %heads : int,
    %qkv_weight : Tensor, %qkv_bias : Tensor, %proj_weight : Tensor, %proj_bias : Tensor,
    %use_gelu : bool, %norm_first : bool, %eps : float,
    %norm_weight_1 : Tensor, %norm_bias_1 : Tensor,
    %norm_weight_2 : Tensor, %norm_bias_2 : Tensor,
    %ffn_weight_1 : Tensor, %ffn_bias_1 : Tensor,
    %ffn_weight_2 : Tensor, %ffn_bias_2 : Tensor,
    %mask : Tensor?, %mask_type : int?):
  %zero : int = prim::Constant[value=0]()
  %one : int = prim::Constant[value=1]()
  %false : bool = prim::Constant[value=0]()
  %true : bool = prim::Constant[value=1]()
  %shape : int[] = prim::ListConstruct(%embed)
  %normalized_1 : Tensor = aten::layer_norm(%src, %shape, %norm_weight_1, %norm_bias_1, %eps, %true)
  %attention : Tensor, %attention_weights : Tensor = aten::_native_multi_head_attention(%normalized_1, %normalized_1, %normalized_1, %embed, %heads, %qkv_weight, %qkv_bias, %proj_weight, %proj_bias, %mask, %false, %true, %mask_type)
  %residual : Tensor = aten::add_(%attention, %src, %one)
  %normalized_2 : Tensor = aten::layer_norm(%residual, %shape, %norm_weight_2, %norm_bias_2, %eps, %true)
  %root_input : Tensor = aten::select(%normalized_2, %one, %zero)
  %weight_1_t : Tensor = aten::t(%ffn_weight_1)
  %hidden : Tensor = aten::_addmm_activation(%ffn_bias_1, %root_input, %weight_1_t, %one, %one, %use_gelu)
  %weight_2_t : Tensor = aten::t(%ffn_weight_2)
  %ffn : Tensor = aten::addmm(%ffn_bias_2, %hidden, %weight_2_t, %one, %one)
  %root_residual : Tensor = aten::select(%residual, %one, %zero)
  %root_output : Tensor = aten::add_(%ffn, %root_residual, %one)
  %output : Tensor = aten::unsqueeze(%root_output, %one)
  return (%output)
'''


def require(condition, message):
    if not condition:
        raise ValueError('FINAL_TOKEN_FFN: ' + message)


def _text(graph):
    return str(graph)+'\n'


def _hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _constant(value, expected, label):
    require(value.node().kind() == 'prim::Constant', label+' must be frozen constant')
    actual=value.toIValue()
    require(type(actual) is type(expected) and actual == expected, label+' differs')


def _only_use(value, kind, label):
    uses=list(value.uses())
    require(len(uses)==1 and uses[0].offset==0 and uses[0].user.kind()==kind,
            label+' must have its one exact downstream use')
    return uses[0].user


def _validate_final_node(graph):
    nodes=list(graph.nodes())
    require(not any(list(n.blocks()) for n in nodes), 'flat frozen graph only')
    blocks=[n for n in nodes if n.kind()==ENCODER]
    require(len(blocks)==6, 'exactly six original fused encoder nodes')
    for before,after in zip(blocks,blocks[1:]):
        require(after.inputsAt(0)==before.output(), 'unchanged six-block chain')
        _only_use(before.output(),ENCODER,'earlier encoder output')
    last=blocks[-1]
    require(last.inputsSize()==20 and last.outputsSize()==1, 'native encoder signature')
    args=list(last.inputs())
    for index,expected,label in ((1,256,'embedding'),(2,8,'attention heads'),
        (7,True,'GELU'),(8,True,'pre-norm'),(9,1e-5,'double epsilon'),
        (18,None,'mask'),(19,None,'mask type')):
        _constant(args[index],expected,label)
    expected_shapes={3:(768,256),4:(768,),5:(256,256),6:(256,),10:(256,),11:(256,),
                     12:(256,),13:(256,),14:(1024,256),15:(1024,),16:(256,1024),17:(256,)}
    import torch
    for index,shape in expected_shapes.items():
        value=args[index]
        require(value.node().kind()=='prim::Constant', 'all last-block weights are frozen constants')
        tensor=value.toIValue()
        require(isinstance(tensor,torch.Tensor) and tuple(tensor.shape)==shape and
                tensor.dtype==torch.float32 and tensor.device.type=='cpu' and
                tensor.is_contiguous() and not tensor.requires_grad,
                'unchanged CPU float32 frozen weight shape/layout')
    batch_slice=_only_use(last.output(),'aten::slice','final encoder output')
    require(batch_slice.inputsSize()==5, 'exact unchanged batch-slice signature')
    for index,expected in ((1,0),(2,0),(3,9223372036854775807),(4,1)):
        _constant(batch_slice.inputsAt(index),expected,'batch slice argument')
    token_select=_only_use(batch_slice.output(),'aten::select','full batch slice')
    require(token_select.inputsSize()==3, 'exact final token-select signature')
    _constant(token_select.inputsAt(1),1,'token dimension')
    _constant(token_select.inputsAt(2),0,'token zero')
    _only_use(token_select.output(),'aten::layer_norm','selected output')
    return nodes,blocks,last,batch_slice


def rewrite_graph(original_graph):
    """Pure graph construction; tensor constants are inspected, never evaluated."""
    import torch
    before_text=_text(original_graph)
    graph=original_graph.copy()
    nodes,blocks,last,batch_slice=_validate_final_node(graph)
    index=nodes.index(last)
    prefix=[str(n) for n in nodes[:index]]
    unchanged_tail=[str(n) for n in nodes[index+2:]]
    require(nodes[index+1]==batch_slice, 'final encoder followed immediately by batch slice')
    first_five=[str(n) for n in blocks[:5]]
    constant_nodes=[str(n) for n in nodes if n.kind()=='prim::Constant']
    replacement=torch._C.parse_ir(REPLACEMENT_IR)
    replacement.lint()
    with graph.insert_point_guard(last):
        outputs=graph.insertGraph(replacement,list(last.inputs()))
    require(len(outputs)==1, 'one replacement output')
    last.output().replaceAllUsesWith(outputs[0])
    last.destroy()
    graph.lint()
    new_nodes=list(graph.nodes())
    require([str(n) for n in new_nodes[:index]]==prefix, 'first five blocks and original prefix unchanged')
    new_slice_index=new_nodes.index(batch_slice)
    require([str(n) for n in new_nodes[new_slice_index+1:]]==unchanged_tail,
            'existing token select, final norm and both heads unchanged')
    require([str(n) for n in new_nodes if n.kind()==ENCODER]==first_five,
            'exact first five fused nodes preserved')
    require([str(n) for n in new_nodes if n.kind()=='prim::Constant'][:len(constant_nodes)]==constant_nodes,
            'all original frozen constants preserved')
    require(_text(original_graph)==before_text, 'original compiled graph was not modified')
    # Frozen ScriptModules carry an unused self input; a ScriptFunction accepts
    # only the original tensor. No method or model is called to remove it.
    inputs=list(graph.inputs())
    if len(inputs)==2:
        require(inputs[0].type().kind()=='ClassType' and not list(inputs[0].uses()),
                'only an unused frozen self input may be removed')
        graph.eraseInput(0)
    require(len(list(graph.inputs()))==1 and list(graph.inputs())[0].type().kind()=='TensorType',
            'one original tensor input')
    graph.lint()
    return graph


def build_candidate(compiled, model, folder, save_json):
    """Construct the sole selected executable without any forward invocation."""
    global _ATTEMPTED
    require(not _ATTEMPTED,'candidate factory may be attempted only once')
    _ATTEMPTED=True
    original=compiled.inlined_graph
    original_text=_text(original)
    require(_hash(original_text)==REFERENCE_GRAPH_SHA256,'exact accepted reference graph')
    save_json(Path(folder)/'candidate_build_attempt.json',dict(candidate_id=CANDIDATE_ID,
        reference_graph_sha256=REFERENCE_GRAPH_SHA256,forward_calls=0,trace_calls=0,warmup_calls=0,
        automatic_fallback=False,additional_candidates=0))
    rewritten=rewrite_graph(original)
    graph_text=_text(rewritten)
    require(len(graph_text.encode('utf-8'))<=MAX_GRAPH_BYTES,'bounded rewritten graph evidence')
    import torch
    executable=torch._C._create_function_from_graph('search256_final_token_ffn_v1',rewritten)
    require(callable(executable),'compiled sole candidate callable')
    require(_text(original)==original_text,'original reference graph preserved after compilation')
    save_json(Path(folder)/'candidate_graph.json',dict(candidate_id=CANDIDATE_ID,
        schema='splendorust_final_token_ffn_graph_v1',reference_graph_sha256=REFERENCE_GRAPH_SHA256,
        graph_sha256=_hash(graph_text),graph_ir=graph_text,first_five_fused_blocks_unchanged=True,
        original_constants_unchanged=True,full_final_attention_and_both_norms_unchanged=True,
        only_final_ffn_rows_narrowed=True,returned_token_shape=['batch',1,256],supported_batches=[1,2,8],
        final_heads_unchanged=True,weights_changed=False,precision='float32',
        forward_calls_during_factory=0,trace_calls=0,warmup_calls=0,automatic_fallback=False,
        exact_float32_parity_established=False,
        numerical_risk='Different GEMM M/strides can change MKL accumulation; original exact qualification is mandatory.'))
    return executable
