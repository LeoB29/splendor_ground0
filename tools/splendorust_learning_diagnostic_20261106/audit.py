"""Saved-evidence acceptance: authenticate source targets, fits and predictions."""
import gc
import math
from pathlib import Path
from . import common as c, data, analysis
from tools.splendorust_gpu_probe_20261105.control import require_absent


def verify_stage(container, mode, spec, plan_sha, boundary):
    boundary()
    owner = c.read(container/'owner.json')
    execution = c.read(container/'execution.json')
    result = c.read(container/'worker/result.json')
    c.require(owner['mode'] == execution['mode'] == result['mode'] == mode and
        owner['fit'] == execution['fit'] == result['fit'] == spec and
        owner['identity'] == execution['identity'] == result['identity'] and
        owner['plan_sha256'] == execution['plan_sha256'] == result['plan_sha256'] == plan_sha,
        'Exact stage identity across owner, actual exit and result')
    c.require(execution['status'] == result['status'] == 'passed' and execution['worker_exit_code'] == 0 and
              not execution['cleanup_errors'] and not execution['error'] and execution['retries'] == 0,
              'Passed actual worker exit and cleanup')
    claim = c.read(c.OUT/'claim.json')
    c.require(owner['windows_job_assigned'] is True and owner['global_claim'] == c.artifact(c.OUT/'claim.json') and
              owner['maximum_seconds'] == execution['maximum_seconds'] and
              owner['monotonic_origin'] >= claim['monotonic_origin'] and owner['unix_origin'] >= claim['unix_origin'],
              'Stage ownership and original clock binding')
    if mode in ('qualification', 'fit'):
        c.require(owner['maximum_seconds'] == (c.QUALIFICATION_SECONDS if mode == 'qualification' else c.FIT_SECONDS),
                  'Frozen one-hour worker ceiling')
    else:
        c.require(mode == 'diagnostics' and 0 < owner['maximum_seconds'] <= c.EVALUATION_SECONDS,
                  'Frozen maximum evaluator allowance')
    c.require(owner['monotonic_origin'] == execution['monotonic_origin'] and
              owner['unix_origin'] == execution['unix_origin'] and
              0 <= execution['monotonic_end']-execution['monotonic_origin'] < owner['maximum_seconds'] and
              0 <= execution['unix_end']-execution['unix_origin'] < owner['maximum_seconds'], 'Unextended actual worker clocks')
    if mode == 'diagnostics':
        clock = c.read(c.OUT/'evaluation_start.json')
        c.require(owner['evaluation_clock'] == clock and
            execution['monotonic_end'] < clock['monotonic_origin'] + c.EVALUATION_SECONDS and
            execution['unix_end'] < clock['unix_origin'] + c.EVALUATION_SECONDS,
            'All evaluators finished inside one original aggregate allowance')
    c.require(execution['monotonic_end'] < claim['monotonic_origin'] + c.MODEL_SECONDS and
              execution['unix_end'] < claim['unix_origin'] + c.MODEL_SECONDS, 'Original model cutoff')
    for identity in [execution['identity'], *execution['observed_owned_identities']]: require_absent(identity)
    c.require(not (container/'worker/failure.json').exists(), 'No suppressed failed worker')
    return dict(mode=mode, fit=spec, owner=c.artifact(container/'owner.json'),
        execution=c.artifact(container/'execution.json'), result=c.artifact(container/'worker/result.json'),
        observed_owned_identities=execution['observed_owned_identities'])


def bind_prediction_rows(rows, examples):
    """Independently require exact root IDs, masks and genuine source targets."""
    c.require(len(rows) == len(examples), 'Every planned source root appears once')
    for row, example in zip(rows, examples, strict=True):
        expected = dict(game_index=example['game_index'], ply=example['ply'], split=example['split'],
            root_input_sha256=example['root_input_sha256'], row_sha256=example['row_sha256'],
            legal_mask81=example['mask'], policy_target81=example['policy'], value_target2=example['value'])
        c.require(analysis.row_identity(row) == expected, 'Prediction root/mask/targets equal authenticated actual source')


def verify_qualification_sources(saved, rechecked):
    from . import training
    examples = rechecked['qualification_examples']
    c.require(saved['root_identities'] == [[x['game_index'], x['ply'], x['row_sha256']] for x in examples] and
              saved['data_qualification'] == {'proof': rechecked['proof']}, 'Engineering update source identities and actual loader proof')
    for row in saved['scopes']:
        update = row['step']
        metrics = [analysis.prediction_metrics(training.prediction_row(example, output))
                   for example, output in zip(examples, update['outputs83'], strict=True)]
        for saved_key, metric in [('loss', 'combined'), ('policy_mean_unweighted', 'policy'), ('value_mean_unweighted', 'value')]:
            c.require(math.isclose(update[saved_key], math.fsum(x[metric] for x in metrics)/32,
                                  rel_tol=1e-6, abs_tol=1e-5), 'Engineering losses reproduce actual official source targets')


def run(plan, folder, *, boundary=lambda: None):
    from . import training
    # Tensor archive/collator inspection only. Never call engine/load_parent or forward.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    c.verify(plan['inputs'], boundary=boundary)
    selection = c.read(c.resolve(plan['selection']))
    data.validate_selection(selection, boundary=boundary)
    plan_sha = c.sha(c.PLAN)
    proof = dict(schema='saved_learning_diagnostic_audit_v1', status='audit_in_progress',
        plan=c.artifact(c.PLAN), selection=plan['selection'], new_games=0,
        new_model_instances=0, network_forward_calls=0, optimizer_calls=0,
        stages=[], fits=[], source_bindings=[], all_heldout_exploratory=True)
    proof['stages'].append(verify_stage(c.OUT/'qualification', 'qualification', None, plan_sha, boundary))
    qualification_folder = c.OUT/'qualification/worker'
    rechecked = data.qualify_data(selection, Path(folder)/'loader_recheck', boundary=boundary)
    saved = c.read(qualification_folder/'data/data_qualification.json')
    c.require(rechecked['proof'] == saved, 'Actual loader/collator qualification independently reproduced without models')
    verify_qualification_sources(c.read(qualification_folder/'qualification_result.json'), rechecked)
    proof['qualification'] = training.verify_qualification(selection, qualification_folder, boundary=boundary)
    del rechecked
    diagnostics = {}
    loaded = None
    previous_cell = None
    # Saved audit order groups identical subsets to avoid redundant source decoding.
    # Scientific fitting and evaluator orders remain the frozen plan order.
    for spec in sorted(plan['fits'], key=lambda x: (x['repetition'], x['breadth'], x['scope'])):
        boundary()
        cell = (spec['repetition'], spec['breadth'])
        if cell != previous_cell:
            del loaded
            gc.collect()
            loaded = data.load_selected(selection, *cell, boundary=boundary)
            previous_cell = cell
        fit_folder = c.OUT/'fits'/spec['fit_id']/'worker'
        diagnostic_folder = c.OUT/'diagnostics'/spec['fit_id']/'worker'
        proof['stages'].append(verify_stage(fit_folder.parent, 'fit', spec, plan_sha, boundary))
        proof['stages'].append(verify_stage(diagnostic_folder.parent, 'diagnostics', spec, plan_sha, boundary))
        verified_fit = training.verify_fit(selection, spec['repetition'], spec['breadth'], spec['scope'],
            fit_folder, loaded=loaded, boundary=boundary)
        proof['fits'].append(verified_fit)
        bundle, provenance = analysis.load_bundle(plan, spec, diagnostic_folder, boundary=boundary)
        diagnostic_result = c.read(diagnostic_folder/'diagnostics_result.json')
        c.require(diagnostic_result['fit_result'] == c.artifact(fit_folder/'fit_result.json'), 'Evaluator uses exact original fit')
        fit_result = c.read(fit_folder/'fit_result.json')
        original, _ = training.checkpoint_payload(fit_result['checkpoints']['0'], boundary=boundary)
        for checkpoint in diagnostic_result['checkpoints']:
            index = checkpoint['checkpoint_step']
            c.require(checkpoint['checkpoint_manifest'] == fit_result['checkpoints'][str(index)], 'Evaluator fixed checkpoint from this fit')
            current = original if index == 0 else training.checkpoint_payload(checkpoint['checkpoint_manifest'], boundary=boundary)[0]
            displacement = training.parameter_displacement(current['state_dict'], original['state_dict'])
            c.require(displacement == checkpoint['group_displacement'], 'Parameter displacement recomputed from actual saved tensors')
            if index != 0: del current
            for probe_index, descriptor in enumerate(checkpoint['gradient_probes']):
                row = c.read(c.resolve(descriptor))
                c.require(row['example_indices'] == loaded['probe_batches'][probe_index], 'Exact predetermined gradient source batch')
            for split in ('train', 'validation', 'offline'):
                examples = loaded['train_examples'] if split == 'train' else [x for x in loaded['heldout_examples'] if x['split'] == split]
                bind_prediction_rows(bundle['evaluations'][str(index)][split], examples)
        proof['source_bindings'].append(dict(fit=spec, source_root_counts={'train': 3072, 'validation': 1802, 'offline': 1755},
            checkpoints=[0,334,668], evaluator=c.artifact(diagnostic_folder/'diagnostics_result.json'),
            prediction_root_target_binding='all_exact', gradient_batches='all_exact', backward_probes=diagnostic_result['backward_probes']))
        diagnostics[spec['fit_id']] = diagnostic_folder
        del bundle, provenance, diagnostic_result, fit_result, original
    del loaded
    gc.collect()
    c.require(sum(x['optimizer_calls'] for x in proof['fits']) == 5344 and
        sum(x['replay_optimizer_calls'] for x in proof['fits']) == 8 and
        proof['qualification']['optimizer_calls'] == 2 and
        sum(x['backward_probes'] for x in proof['source_bindings']) == 144, 'Complete fixed work inventory')
    proof['status'] = 'saved_evidence_verified_pending_independent_acceptance'
    proof['total_verified_optimizer_calls'] = 5354
    proof['verified_per_loss_gradient_probes'] = 144
    proof['inputs'] = plan['inputs']
    evidence_item = c.save(Path(folder)/'saved_evidence.json', proof)
    report = analysis.run_analysis(plan, diagnostics, c.OUT/'report', boundary=boundary)
    c.verify(plan['inputs'], boundary=boundary)
    boundary()
    return dict(status='saved_audit_and_analysis_passed_pending_independent_acceptance',
        saved_evidence=evidence_item, analysis=report, scientific_fits=8,
        new_games=0, network_forward_calls=0, optimizer_calls=0, automatic_promotion=False)
