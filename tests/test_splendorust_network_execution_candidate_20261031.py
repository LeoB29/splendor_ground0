"""Graph construction only: no checkpoint, model forward, tracing or warmups."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tools.splendorust_network_execution_20261031 import candidate_final_token_ffn as candidate


def original_graph():
    names=['qkvw','qkvb','projw','projb','nw1','nb1','nw2','nb2','fw1','fb1','fw2','fb2']
    shapes=[(768,256),(768,),(256,256),(256,),(256,),(256,),(256,),(256,),
            (1024,256),(1024,),(256,1024),(256,)]
    signature=', '.join('%'+name+' : Tensor' for name in names)
    lines=[f'graph(%input : Tensor, {signature}):',
        '  %embed : int = prim::Constant[value=256]()',
        '  %heads : int = prim::Constant[value=8]()',
        '  %yes : bool = prim::Constant[value=1]()',
        '  %eps : float = prim::Constant[value=1.0000000000000001e-05]()',
        '  %none : NoneType = prim::Constant()',
        '  %zero : int = prim::Constant[value=0]()',
        '  %one : int = prim::Constant[value=1]()',
        '  %max : int = prim::Constant[value=9223372036854775807]()',
        '  %shape : int[] = prim::Constant[value=[256]]()']
    previous='%input'
    for i in range(6):
        lines.append(f'  %block{i} : Tensor = aten::_transformer_encoder_layer_fwd({previous}, %embed, %heads, %qkvw, %qkvb, %projw, %projb, %yes, %yes, %eps, %nw1, %nb1, %nw2, %nb2, %fw1, %fb1, %fw2, %fb2, %none, %none)')
        previous=f'%block{i}'
    lines += ['  %sliced : Tensor = aten::slice(%block5, %zero, %zero, %max, %one)',
              '  %selected : Tensor = aten::select(%sliced, %one, %zero)',
              '  %output : Tensor = aten::layer_norm(%selected, %shape, %nw1, %nb1, %eps, %yes)',
              '  return (%output)']
    graph=torch._C.parse_ir('\n'.join(lines))
    inputs=list(graph.inputs())
    with graph.insert_point_guard(next(graph.nodes())):
        for value,shape in zip(inputs[1:],shapes,strict=True):
            constant=graph.insertConstant(torch.zeros(shape,dtype=torch.float32))
            value.replaceAllUsesWith(constant)
    for index in reversed(range(1,len(inputs))):graph.eraseInput(index)
    graph.lint()
    return graph


def final_encoder(graph):
    return [n for n in graph.nodes() if n.kind()==candidate.ENCODER][-1]


def test_graph_rewrite_preserves_original_prefix_and_tail_without_execution():
    original=original_graph();snapshot=str(original)
    # Graph.copy renumbers synthetic inserted constants; compare against its
    # deterministic copy baseline, while separately preserving original text.
    copied_baseline=original.copy()
    oldblocks=[str(n) for n in copied_baseline.nodes() if n.kind()==candidate.ENCODER]
    rewritten=candidate.rewrite_graph(original)
    assert str(original)==snapshot
    blocks=[str(n) for n in rewritten.nodes() if n.kind()==candidate.ENCODER]
    assert blocks==oldblocks[:5]
    nodes=list(rewritten.nodes());ops=[n.kind() for n in nodes]
    assert ops.count('aten::_native_multi_head_attention')==1
    assert ops.count('aten::_addmm_activation')==ops.count('aten::addmm')==1
    assert ops.count('aten::add_')==2 and ops.count('aten::layer_norm')==3
    assert ops.count('aten::unsqueeze')==1
    assert 'prim::CallFunction' not in ops and 'prim::CallMethod' not in ops
    fn=torch._C._create_function_from_graph('model_free_final_token_compile_check',rewritten)
    assert callable(fn)  # Deliberately never invoked.


def test_native_attention_norm_and_residual_operand_identity():
    graph=candidate.rewrite_graph(original_graph());nodes=list(graph.nodes())
    mha=next(n for n in nodes if n.kind()=='aten::_native_multi_head_attention')
    args=list(mha.inputs());assert args[0]==args[1]==args[2]
    norm1=args[0].node();assert norm1.kind()=='aten::layer_norm'
    assert norm1.inputsAt(0).node().kind()==candidate.ENCODER
    assert args[10].toIValue() is False and args[11].toIValue() is True
    assert args[9].toIValue() is None and args[12].toIValue() is None
    adds=[n for n in nodes if n.kind()=='aten::add_'];attention_add,ffn_add=adds
    assert attention_add.inputsAt(0)==mha.outputsAt(0)
    assert attention_add.inputsAt(1)==norm1.inputsAt(0)
    norm2=next(n for n in nodes if n.kind()=='aten::layer_norm' and n.inputsAt(0)==attention_add.output())
    selected=next(n for n in nodes if n.kind()=='aten::select' and n.inputsAt(0)==norm2.output())
    assert selected.inputsAt(1).toIValue()==1 and selected.inputsAt(2).toIValue()==0
    activated=next(n for n in nodes if n.kind()=='aten::_addmm_activation')
    assert activated.inputsAt(1)==selected.output()
    assert activated.inputsAt(3).toIValue()==activated.inputsAt(4).toIValue()==1
    assert activated.inputsAt(5).toIValue() is True
    assert activated.inputsAt(2).node().kind()=='aten::t'
    linear=next(n for n in nodes if n.kind()=='aten::addmm')
    assert linear.inputsAt(1)==activated.output() and linear.inputsAt(2).node().kind()=='aten::t'
    assert linear.inputsAt(3).toIValue()==linear.inputsAt(4).toIValue()==1
    assert ffn_add.inputsAt(0)==linear.output()
    residual=ffn_add.inputsAt(1).node()
    assert residual.kind()=='aten::select' and residual.inputsAt(0)==attention_add.output()
    assert residual.inputsAt(1).toIValue()==1 and residual.inputsAt(2).toIValue()==0
    output=next(n for n in nodes if n.kind()=='aten::unsqueeze')
    assert output.inputsAt(0)==ffn_add.output() and output.inputsAt(1).toIValue()==1


def test_dynamic_batch_preserved_without_flatten_or_hardcoded_batch():
    graph=candidate.rewrite_graph(original_graph())
    newops=[n for n in graph.nodes() if n.kind().startswith('aten::')]
    assert not any(n.kind() in ('aten::reshape','aten::view','aten::contiguous') for n in newops)
    # select(dim1,0) retains every batch, even when the selected view is strided;
    # unsqueeze(dim1) restores[B,1,256]. Runtime batch1/2/8 parity remains mandatory.
    assert all(n.inputsAt(1).toIValue()==1 for n in newops if n.kind()=='aten::select')


@pytest.mark.parametrize('index,value',[(1,128),(2,4),(7,False),(8,False),(9,1e-4),(18,False),(19,0)])
def test_unapproved_encoder_configuration_rejected(index,value):
    graph=original_graph();last=final_encoder(graph)
    with graph.insert_point_guard(last):replacement=graph.insertConstant(value)
    last.replaceInput(index,replacement)
    with pytest.raises(ValueError):candidate.rewrite_graph(graph)


@pytest.mark.parametrize('kind',['wrong_shape','half','requires_grad','noncontiguous','not_constant'])
def test_weights_must_be_original_frozen_cpu_float32_shapes(kind):
    graph=original_graph();last=final_encoder(graph)
    if kind=='not_constant':replacement=list(graph.inputs())[0]
    else:
        if kind=='wrong_shape':tensor=torch.zeros((1023,256))
        elif kind=='half':tensor=torch.zeros((1024,256),dtype=torch.float16)
        elif kind=='requires_grad':tensor=torch.zeros((1024,256),requires_grad=True)
        else:tensor=torch.zeros((256,1024)).t()
        if kind=='requires_grad':
            # The graph compiler itself refuses this unfrozen tensor before
            # candidate validation; preserve that earlier fail-closed boundary.
            with graph.insert_point_guard(last), pytest.raises(RuntimeError,match='Unsupported value kind'):
                graph.insertConstant(tensor)
            return
        with graph.insert_point_guard(last):replacement=graph.insertConstant(tensor)
    last.replaceInput(14,replacement)
    with pytest.raises(ValueError):candidate.rewrite_graph(graph)


def test_extra_final_token_use_rejected():
    graph=original_graph();last=final_encoder(graph)
    extra=graph.create('aten::alias',[last.output()],1);extra.insertAfter(last)
    with pytest.raises(ValueError,match='one exact downstream use'):candidate.rewrite_graph(graph)


@pytest.mark.parametrize('change',['index','dimension','batch_slice','earlier_chain'])
def test_only_original_token_zero_consumption_can_be_pruned(change):
    graph=original_graph();last=final_encoder(graph)
    batch_slice=list(last.output().uses())[0].user;selected=list(batch_slice.output().uses())[0].user
    with graph.insert_point_guard(last):two=graph.insertConstant(2)
    if change=='index':selected.replaceInput(2,two)
    elif change=='dimension':selected.replaceInput(1,two)
    elif change=='batch_slice':batch_slice.replaceInput(2,two)
    else:last.replaceInput(0,list(graph.inputs())[0])
    with pytest.raises(ValueError):candidate.rewrite_graph(graph)


def test_factory_only_compiles_once_and_records_unproven_parity(monkeypatch,tmp_path):
    graph=original_graph();calls=[]
    class Reference:
        inlined_graph=graph
        def __call__(self,*args,**kwargs):raise AssertionError('Forbidden forward')
    class NoModelCalls:
        def __getattr__(self,name):raise AssertionError('Model access '+name)
    def no_trace(*args,**kwargs):raise AssertionError('Forbidden trace/script/warmup')
    monkeypatch.setattr(candidate,'_ATTEMPTED',False)
    monkeypatch.setattr(candidate,'REFERENCE_GRAPH_SHA256',candidate._hash(candidate._text(graph)))
    monkeypatch.setattr(torch.jit,'trace',no_trace)
    monkeypatch.setattr(torch.jit,'script',no_trace)
    fn=candidate.build_candidate(Reference(),NoModelCalls(),tmp_path,lambda path,value:calls.append((path,deepcopy(value))))
    assert callable(fn) and len(calls)==2
    assert calls[0][0].name=='candidate_build_attempt.json'
    identity=calls[1][1]
    assert calls[1][0].name=='candidate_graph.json'
    assert identity['forward_calls_during_factory']==identity['trace_calls']==identity['warmup_calls']==0
    assert identity['supported_batches']==[1,2,8] and not identity['exact_float32_parity_established']
    assert identity['first_five_fused_blocks_unchanged'] and identity['only_final_ffn_rows_narrowed']
    with pytest.raises(ValueError,match='only once'):candidate.build_candidate(Reference(),None,tmp_path,lambda *args:None)


def test_unaccepted_reference_graph_stops_before_build_and_cannot_retry(monkeypatch,tmp_path):
    monkeypatch.setattr(candidate,'_ATTEMPTED',False)
    graph=original_graph();reference=SimpleNamespace(inlined_graph=graph)
    with pytest.raises(ValueError,match='accepted reference'):candidate.build_candidate(reference,None,tmp_path,lambda *a:None)
    with pytest.raises(ValueError,match='only once'):candidate.build_candidate(reference,None,tmp_path,lambda *a:None)
