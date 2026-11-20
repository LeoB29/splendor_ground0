"""Saved-only restart of the original runtime and accepted-qualification checks.

No experimental worker, inference, training, game or reproduction is dispatched.
The original immutable validators run in the isolated recovery auditor with its
new clock guard; their original source/input scope and comparisons are retained.
"""
from importlib import import_module
import inspect
from pathlib import Path
import sys
import time
import types

from . import common as c

ORIGINAL_INPUTS = 9330
PRIOR_INPUTS = 4885
PRIOR_SEAL = 4392
ORIGINAL_PACKAGE = 'tools.splendorust_isolated_reliability_20261026'
MODEL_MODULES = ('torch', 'tensorflow', 'jax', 'jaxlib', 'onnxruntime', 'tensorrt')


def _assert_model_free():
    c.require(not any(name == prefix or name.startswith(prefix+'.') for name in sys.modules
        for prefix in MODEL_MODULES), 'A model runtime was imported during saved-only recovery')


def _original_inventory(original_freeze, envelope):
    envelope.guard()
    item=c.artifact(c.ORIGINAL/'freeze.json',envelope.guard)
    c.require(item['sha256']==c.ORIGINAL_SHA,'Original reliability freeze changed')
    c.require(c.canonical(c.read(c.ORIGINAL/'freeze.json'))==c.canonical(original_freeze),
        'Provided original freeze differs from its immutable saved bytes')
    inputs=original_freeze['inputs'];index={row['path']:row for row in inputs}
    c.require(len(inputs)==len(index)==ORIGINAL_INPUTS,'Complete original9330input inventory required')
    c.require(original_freeze['placement_mode']=='isolated','Original isolated placement contract differs')
    for native in original_freeze['native_artifacts']:
        c.require(index.get(native['path'])==native,'Original native runtime artifact absent from full input inventory')
    accepted=original_freeze['qualification_acceptance']
    c.require(len(accepted['frozen_inputs'])==PRIOR_INPUTS and len(accepted['sealed_evidence'])==PRIOR_SEAL,
        'Exact preceding qualification inventory required')
    return index,item


def _prebind_python_sources(index,envelope):
    """Authenticate frozen repository Python before importing original validators."""
    sources=[item for name,item in index.items() if name.endswith('.py') and not name.startswith('outputs/')]
    c.require(sources,'Original frozen source inventory is empty')
    for item in sources:
        envelope.guard()
        c.require(c.artifact(c.ROOT/item['path'],envelope.guard)==item,'Changed imported frozen source: '+item['path'])
    return sources


def _load_validators():
    runner=import_module(ORIGINAL_PACKAGE+'.runner')
    preparation=import_module(ORIGINAL_PACKAGE+'.preparation')
    compression=import_module('tools.splendorust_search_scale_20261024.compression')
    sampler=import_module('tools.splendorust_timing_opt_20261024.sampler')
    return runner,preparation,compression,sampler


def _bind_function(function,module,index,compiled,source_inventory,envelope):
    """Bind actual callable ownership and code to its frozen, unexecuted source."""
    c.require(inspect.isfunction(function) and function.__module__==module.__name__ and
        function.__globals__ is vars(module),'Imported validator function owner differs')
    filename=Path(inspect.getsourcefile(function)).resolve()
    c.require(filename==Path(module.__file__).resolve() and filename==Path(function.__code__.co_filename).resolve(),
        'Imported validator source/code path differs')
    c.require(filename.is_relative_to(c.ROOT),'Frozen validator outside repository')
    relative=filename.relative_to(c.ROOT).as_posix()
    c.require(relative in index,'Imported validator is not bound by original freeze: '+relative)
    if relative not in compiled:
        envelope.guard();artifact=c.artifact(filename,envelope.guard)
        c.require(artifact==index[relative],'Imported validator source bytes changed')
        # Compile but never execute the source. This also rejects replacement
        # callables whose claimed module/file still names a genuine source.
        code=compile(filename.read_bytes(),function.__code__.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize)
        compiled[relative]=[item for item in code.co_consts if isinstance(item,types.CodeType)]
        source_inventory[relative]=artifact
    candidates=[code for code in compiled[relative] if code.co_name==function.__name__]
    c.require(len(candidates)==1 and function.__code__==candidates[0],
        'Loaded validator code does not match frozen source: '+module.__name__+'.'+function.__name__)
    return dict(module=module.__name__,function=function.__name__,source=index[relative])


def _bind_validators(runner,preparation,compression,sampler,index,envelope):
    c.require(runner.c is preparation.c and runner.placement is preparation.placement,
        'Original runtime and acceptance do not share their original modules')
    c.require(runner.snapshot_api_modules is sampler.snapshot_api_modules,
        'Original runtime snapshot function alias differs')
    specs=[(runner,('verify_runtime','runtime_placement_mode')),
        (preparation,('acceptance','validate_accepted_documents','validate_review_bindings')),
        (runner.c,('verify','artifact','file_hash','read','require')),
        (runner.placement,('verify_runtime','verify_host','plan','_planned','_process_masks','_thread_rows')),
        (runner.placement.clock,('processor_topology',)),
        (sampler,('snapshot_api_modules',)),(compression,('runtime_contract',))]
    compiled={};sources={};functions=[]
    for module,names in specs:
        for name in names:
            functions.append(_bind_function(getattr(module,name),module,index,compiled,sources,envelope))
    # Every already imported repository module that the original freeze owns is
    # checked too; new recovery modules are instead bound by the recovery freeze.
    for module in tuple(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if not value:continue
        path=Path(value).resolve()
        if not path.is_relative_to(c.ROOT):continue
        relative=path.relative_to(c.ROOT).as_posix()
        if relative in index and relative not in sources:
            envelope.guard();artifact=c.artifact(path,envelope.guard)
            c.require(artifact==index[relative],'Loaded frozen module differs: '+relative)
            sources[relative]=artifact
    return dict(status='passed',functions=functions,sources=sorted(sources.values(),key=lambda row:row['path']))


class _RuntimeBoundary:
    """No resource overrides or writes; delegate the new clock and progress only."""
    placement_mode='isolated'
    def __init__(self,envelope):self.envelope=envelope;self.calls=0
    def guard(self):
        self.envelope.guard();self.calls+=1
        # The original loop checks the boundary BEFORE each next input hash.
        # Only a full return below accepts the entire input/runtime check.
        complete_before_next=min(self.calls-1,ORIGINAL_INPUTS)
        if self.calls<=ORIGINAL_INPUTS and complete_before_next and complete_before_next%1024==0:
            self.envelope.checkpoint('original_runtime_input_progress',dict(
                inputs_verified_before_next_boundary=complete_before_next,total=ORIGINAL_INPUTS,
                complete_runtime_check_accepted=False))


def verify_runtime_and_prior(original_freeze,envelope):
    """Run the complete frozen checks from their beginning; no old clock resume."""
    begun=time.perf_counter();_assert_model_free();envelope.guard()
    envelope.checkpoint('runtime_recovery_begin',dict(original_inputs=ORIGINAL_INPUTS,
        preceding_qualification_inputs=PRIOR_INPUTS,preceding_qualification_seal=PRIOR_SEAL,
        restarting_unfinished_check_from_index_zero=True))
    index,freeze_item=_original_inventory(original_freeze,envelope)
    prebound=_prebind_python_sources(index,envelope)
    runner,preparation,compression,sampler=_load_validators();_assert_model_free()
    bindings=_bind_validators(runner,preparation,compression,sampler,index,envelope)
    runtime_started=time.perf_counter();boundary=_RuntimeBoundary(envelope)
    envelope.checkpoint('original_runtime_full_restart',dict(inputs=ORIGINAL_INPUTS,placement_mode='isolated'))
    runner.verify_runtime(original_freeze,boundary)
    c.require(boundary.calls==ORIGINAL_INPUTS+len(original_freeze['external_runtime']),
        'Original complete input/external runtime boundary inventory did not finish')
    _assert_model_free();envelope.guard()
    runtime_seconds=time.perf_counter()-runtime_started
    envelope.checkpoint('original_runtime_passed',dict(inputs=ORIGINAL_INPUTS,
        python_torch_numpy_native_external_compression_host_and_isolated_checks='passed',seconds=runtime_seconds))
    acceptance_started=time.perf_counter()
    envelope.checkpoint('preceding_qualification_acceptance_begin',dict(inputs=PRIOR_INPUTS,sealed_artifacts=PRIOR_SEAL))
    accepted=preparation.acceptance(boundary=envelope.guard)
    c.require(c.canonical(accepted)==c.canonical(original_freeze['qualification_acceptance']),
        'Exact frozen preceding-qualification acceptance comparison failed')
    _assert_model_free();envelope.guard()
    acceptance_seconds=time.perf_counter()-acceptance_started
    envelope.checkpoint('preceding_qualification_acceptance_passed',dict(inputs=PRIOR_INPUTS,sealed_artifacts=PRIOR_SEAL,
        seconds=acceptance_seconds))
    # Reauthenticate the actual function bindings after both validators return.
    final_bindings=_bind_validators(runner,preparation,compression,sampler,index,envelope)
    c.require(final_bindings['functions']==bindings['functions'] and
        all(item in final_bindings['sources'] for item in bindings['sources']),
        'Imported validator ownership/source changed during verification')
    c.require(c.artifact(c.ORIGINAL/'freeze.json',envelope.guard)==freeze_item,'Original freeze changed during recovery')
    _assert_model_free();envelope.guard()
    return dict(status='passed_runtime_and_prior_recovery',original_freeze=freeze_item,
        original_inputs_rechecked_from_start=ORIGINAL_INPUTS,previous_partial_loop_reused=False,
        python_and_numerical_packages='passed',torch_runtime_files=len(original_freeze['torch_runtime']),
        external_runtime_files=len(original_freeze['external_runtime']),native_artifacts=len(original_freeze['native_artifacts']),
        compression_runtime='passed',sampler_api_owners='passed',physical_host_and_isolated_supervisor='passed',
        preceding_qualification_exact_acceptance='passed',preceding_frozen_inputs=PRIOR_INPUTS,
        preceding_sealed_artifacts=PRIOR_SEAL,preimport_frozen_python_sources=len(prebound),
        imported_validator_bindings=final_bindings,runtime_seconds=runtime_seconds,acceptance_seconds=acceptance_seconds,
        elapsed_seconds=time.perf_counter()-begun,model_imports=0,model_calls=0,workers_launched=0,
        fresh_decisions=0,fresh_games=0,corrected_startup_check_owned_by_outer_pipeline=True,
        original_outputs_written=False,original_incomplete_status_unchanged=True,
        overall_recovered_acceptance_requires_other_saved_evidence_checks=True)
