"""Seven fixed post-commitment predictors; saved verification uses no models.

The caller owns the one startup source guard.  This module never opens assessment
inputs before the independent all-endpoint commitment gate has succeeded.
"""
from collections import Counter
from collections.abc import Sequence
import math
from pathlib import Path
import struct

from . import common as c

MODEL_IDS = ('parent', 'historical_r0', 'historical_r1',
             'larger_r0', 'larger_r1', 'control_r0', 'control_r1')
SPLITS = ('train', 'validation', 'offline')
SOURCES = ('prior_20261107', 'fresh_20261115')
ROW_FIELDS = {'source', 'game_index', 'ply', 'split', 'row_sha256',
              'root_input_sha256', 'legal_mask81', 'policy_target81',
              'value_target2', 'root_q_win_credit', 'output83'}
require = c.require


def identity(row):
    require(row.get('source') in SOURCES and type(row.get('game_index')) is int
            and row['game_index'] >= 0 and type(row.get('ply')) is int
            and row['ply'] >= 0 and row.get('split') in SPLITS,
            'Namespaced prediction identity')
    return row['source'], row['game_index'], row['ply'], row['split']


def target_identity(row):
    return {key: row[key] for key in sorted(ROW_FIELDS - {'output83'})}


def _sha(value):
    require(type(value) is str and len(value) == 64 and
            set(value) <= set('0123456789abcdef'), 'Complete lowercase SHA256')


def strict_prediction(row):
    """Validate labels and exact float32 outputs before any saved arithmetic."""
    require(type(row) is dict and set(row) == ROW_FIELDS, 'Exact prediction schema')
    identity(row)
    for key in ('row_sha256', 'root_input_sha256'):
        _sha(row[key])
    mask, policy, value, outputs = (row[key] for key in
        ('legal_mask81', 'policy_target81', 'value_target2', 'output83'))
    require(type(mask) is list and len(mask) == 81 and any(mask)
            and all(type(x) is bool for x in mask), 'Explicit complete legal mask')
    require(type(policy) is list and len(policy) == 81 and
            all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in policy)
            and abs(math.fsum(policy) - 1.) <= 1e-7
            and all(legal or probability == 0 for legal, probability in zip(mask, policy)),
            'Normalized original policy on legal support')
    require(type(value) is list and len(value) == 2
            and all(type(x) in (int, float) and x in (-1., 0., 1.) for x in value)
            and sum(value) == 0., 'Official actor-oriented terminal labels')
    q = row['root_q_win_credit']
    require(type(q) in (int, float) and math.isfinite(q) and 0 <= q <= 1,
            'Authenticated actor-oriented search win credit')
    require(type(outputs) is list and len(outputs) == 83 and
            all(type(x) is float and math.isfinite(x) and abs(x) <= 3.4028234663852886e38
                and struct.unpack('<f', struct.pack('<f', x))[0] == x for x in outputs),
            'Exact finite float32 prediction outputs')
    require(all(-1 <= x <= 1 for x in outputs[81:]), 'Bounded signed value outputs')
    return row


def predict_outputs(model, examples):
    """Unchanged qualified forward mathematics, also usable on structural fixtures."""
    import torch
    from tools.splendorust_full_corpus_20261110 import sampling
    from tools.splendorust_policy_head_20261104 import training as accepted
    tokens = sampling.collate_inputs(examples, [1.] * len(examples))['tokens']
    with torch.no_grad():
        outputs = accepted.tensor_forward(model, tokens).tolist()
    return outputs


def prediction_row(example, output):
    """Use official labels for diagnostics even when the cache's fit value is mixed."""
    require('terminal_value' in example, 'Keep official outcome distinct from mixed fit value')
    row = dict(source=example['source'], game_index=example['game_index'],
        ply=example['ply'], split=example['split'], row_sha256=example['row_sha256'],
        root_input_sha256=example['root_input_sha256'], legal_mask81=list(example['mask']),
        policy_target81=list(example['policy']), value_target2=list(example['terminal_value']),
        root_q_win_credit=example['root_q_win_credit'], output83=list(output))
    return strict_prediction(row)


def predict_batch(model, examples):
    """Production route plus exact historical-row compatibility for qualification."""
    outputs = predict_outputs(model, examples)
    if all('source' in example for example in examples):
        return [prediction_row(x, y) for x, y in zip(examples, outputs, strict=True)]
    require(all('source' not in example for example in examples), 'One explicit prediction schema per batch')
    from tools.splendorust_full_corpus_20261110 import fit as previous_fit, offline as previous
    rows = [previous_fit.prediction_row(x, y) for x, y in zip(examples, outputs, strict=True)]
    for row in rows:
        previous.strict_prediction(row)
    return rows


class _View(Sequence):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, tuple(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.dataset[self.indices[index]]


def _read(item, boundary):
    c.verify([item], boundary=boundary)
    return c.read(c.resolve(item))


def _startup_proof(item, boundary):
    from . import fit
    return fit.verify_startup(item, boundary=boundary)


def consumed_inputs():
    """Only unprotected prepared metadata; the fit qualifier owns old examples."""
    proposal = c.read(c.PROPOSAL)
    return [c.artifact(c.PROPOSAL), c.artifact(c.BINDINGS), proposal['training_panel']['order']]


def _registry(committed):
    values = committed['models']
    require(type(values) is list and len(values) == len(MODEL_IDS), 'Exactly seven fixed predictors')
    result = {row['id']: row for row in values}
    require(set(result) == set(MODEL_IDS) and len(result) == len(values), 'No omitted or duplicate fixed model')
    for model_id, row in result.items():
        expected = 'parent' if model_id == 'parent' else ('historical' if model_id.startswith('historical') else 'new')
        require(row['kind'] == expected, 'Fixed checkpoint loader kind')
        require(type(row['checkpoint']) is dict, 'Committed checkpoint descriptor')
        _sha(row['weights_sha256'])
    return result


def _evaluation_context(boundary=lambda: None):
    """Gate first, before any assessment path hash, open, or cache construction."""
    from . import seal, corpus, selection
    committed = seal.validate_commitment(boundary=boundary)
    registry = _registry(committed)
    proposal = c.read(c.PROPOSAL)
    require(committed['evaluation'] == proposal['evaluation'], 'Exact committed prospective evaluation recipe')
    commitment = c.artifact(c.COMMITMENT)
    training = corpus.training_dataset(boundary=boundary)
    assessment = corpus.assessment_dataset(boundary=boundary)
    panel = [int(i) for i in selection.panel_indices()]
    require(len(panel) == len(set(panel)) == 16384 and
            all(type(i) is int and 0 <= i < len(training) for i in panel), 'Complete prepared training panel')
    indices = {'train': sorted(panel, key=lambda i: identity(training.index[i]))}
    for split in ('validation', 'offline'):
        indices[split] = sorted([i for i, row in enumerate(assessment.index) if row['split'] == split],
                                key=lambda i: identity(assessment.index[i]))
    require(sum(len(indices[s]) for s in ('validation', 'offline')) == len(assessment),
            'Only both complete fresh assessment partitions')
    groups = {'train': _View(training, indices['train'])}
    groups.update({s: _View(assessment, indices[s]) for s in ('validation', 'offline')})
    _validate_group_inventory(groups)
    provenance = dict(commitment=commitment, proposal=c.artifact(c.PROPOSAL),
        training_panel=proposal['training_panel']['order'],
        training_cache=c.artifact(c.TRAIN_CACHE/'cache_acceptance.json'),
        assessment_cache=c.artifact(c.ASSESS_CACHE/'cache_acceptance.json'),
        assessment_binding=committed['assessment_binding'], evaluation_recipe=committed['evaluation'])
    c.verify([provenance[k] for k in ('training_panel', 'training_cache', 'assessment_cache')], boundary=boundary)
    return registry, groups, provenance


def _validate_group_inventory(groups):
    require(set(groups) == set(SPLITS), 'Training panel and both fresh partitions only')
    all_games = set()
    for split, examples in groups.items():
        # Metadata only here: the dataset maps each source row lazily.
        rows = [examples.dataset.index[i] for i in examples.indices]
        ids = [identity(row) for row in rows]
        require(ids == sorted(set(ids)) and all(row['split'] == split for row in rows),
                'Complete unique ordered evaluation identities')
        counts = Counter((row['source'], row['game_index']) for row in rows)
        require(not all_games.intersection(counts), 'Disjoint game memberships across evaluation groups')
        all_games.update(counts)
        if split == 'train':
            require(len(rows) == 16384 and len(counts) == 4096 and set(counts.values()) == {4},
                    'Exactly four descriptive roots per training game')
            require(Counter(source for source, _ in counts) == {'prior_20261107': 1024, 'fresh_20261115': 3072},
                    'Separate prior/new training-game panels')
        else:
            require(len(counts) == 128 and set(source for source, _ in counts) == {'fresh_20261115'},
                    'All 128 genuinely fresh games in each assessment partition')


def _load_model(binding, boundary):
    from . import fit
    if binding['kind'] == 'parent':
        c.verify([binding['checkpoint']], boundary=boundary)
        require(binding['checkpoint']['sha256'] == c.PARENT_CHECKPOINT_SHA, 'Original historical EF8A archive')
        model = fit.accepted.load_parent()
    elif binding['kind'] == 'historical':
        model, _, _ = fit.old.load_checkpoint(binding['checkpoint'], boundary=boundary)
    else:
        model, _, _ = fit.load_checkpoint(binding['checkpoint'], boundary=boundary)
    model.eval()
    require(fit.accepted.state_digest(model) == binding['weights_sha256'], 'Exact committed predictor tensors')
    return model


def run_evaluation(folder, boundary=lambda: None, runtime_sources=None):
    from . import fit
    folder = Path(folder)
    require(folder.resolve() == (c.OUT/'jobs/offline/worker').resolve(), 'Owned fixed evaluation output')
    require(type(runtime_sources) is dict and runtime_sources, 'Caller supplies the single verified startup source receipt')
    registry, groups, provenance = _evaluation_context(boundary)
    fit.engine(folder, runtime_sources=runtime_sources)
    results = []
    for model_id in MODEL_IDS:
        boundary()
        binding = registry[model_id]
        model = _load_model(binding, boundary)
        batches = []
        for split in SPLITS:
            examples = groups[split]
            for number, offset in enumerate(range(0, len(examples), 32)):
                boundary()
                rows = predict_batch(model, examples[offset:offset+32])
                require(all('source' in row for row in rows), 'Only namespaced production prediction rows')
                batches.append(c.save(folder/'predictions'/model_id/split/f'{number:06d}.json',
                    dict(schema='training4096_prediction_batch_v1', model_id=model_id, split=split,
                         batch_index=number, predictor=binding, provenance=provenance, rows=rows)))
        require(fit.accepted.state_digest(model) == binding['weights_sha256'], 'Prediction leaves all weights unchanged')
        results.append(dict(model_id=model_id, predictor=binding, batches=batches,
                            root_counts={s: len(groups[s]) for s in SPLITS}, forward_calls=len(batches)))
        del model
    document = dict(schema='training4096_offline_evaluation_v1', status='all_seven_fixed_predictions_complete',
        provenance=provenance, models=results, forward_calls=sum(x['forward_calls'] for x in results),
        optimizer_calls=0, historical_predictions_reused=False, all_seven_predictions_fresh=True,
        fresh_assessment_games=256, reused_development_games_used=0,
        training_panel_roots=16384, all_endpoints_committed_before_any_assessment_read=True,
        checkpoint_selection=False, reference_promoted=False, runtime_sources=runtime_sources,
        startup=c.artifact(folder/'startup.json'), **c.IDENTITY)
    item = c.save(folder/'evaluation_result.json', document)
    boundary()
    return item


def load_predictions(folder=None, boundary=lambda: None):
    """Authenticate every saved input/label/batch against gated accepted caches."""
    folder = Path(folder) if folder is not None else c.OUT/'jobs/offline/worker'
    require(folder.resolve() == (c.OUT/'jobs/offline/worker').resolve(), 'Only this study evaluation')
    registry, groups, provenance = _evaluation_context(boundary)
    document = c.read(folder/'evaluation_result.json')
    require(document['schema'] == 'training4096_offline_evaluation_v1' and
            document['status'] == 'all_seven_fixed_predictions_complete' and
            document['provenance'] == provenance, 'Exact post-commitment saved evaluation')
    require(document['startup'] == c.artifact(folder/'startup.json'), 'Exact evaluator startup receipt')
    startup = _startup_proof(document['startup'], boundary)
    require(document['runtime_sources'] == startup['runtime_sources'], 'Same single corrected evaluator startup guard')
    for key, expected in dict(optimizer_calls=0, historical_predictions_reused=False,
            all_seven_predictions_fresh=True, fresh_assessment_games=256,
            reused_development_games_used=0, training_panel_roots=16384,
            all_endpoints_committed_before_any_assessment_read=True,
            checkpoint_selection=False, reference_promoted=False, **c.IDENTITY).items():
        require(document[key] == expected, 'Fixed evaluation claim '+key)
    require([x['model_id'] for x in document['models']] == list(MODEL_IDS), 'All seven fixed models in order')
    bundles, all_paths = {}, []
    for entry in document['models']:
        model_id = entry['model_id']
        require(entry['predictor'] == registry[model_id], 'Saved predictor matches exact committed endpoint')
        require(entry['root_counts'] == {s: len(groups[s]) for s in SPLITS}, 'Complete root counts')
        bundles[model_id] = {s: [] for s in SPLITS}
        expected_order = [(s, n) for s in SPLITS for n in range(math.ceil(len(groups[s])/32))]
        require(len(entry['batches']) == entry['forward_calls'] == len(expected_order), 'Exact full forward count')
        for item, (split, number) in zip(entry['batches'], expected_order, strict=True):
            boundary()
            expected_path = folder/'predictions'/model_id/split/f'{number:06d}.json'
            require(c.resolve(item) == expected_path.resolve(), 'Canonical unique prediction batch path')
            batch = _read(item, boundary)
            require(set(batch) == {'schema','model_id','split','batch_index','predictor','provenance','rows'}
                    and batch['schema'] == 'training4096_prediction_batch_v1'
                    and batch['model_id'] == model_id and batch['split'] == split
                    and type(batch['batch_index']) is int and batch['batch_index'] == number
                    and batch['predictor'] == registry[model_id] and batch['provenance'] == provenance,
                    'Exact saved batch model/source/commitment bindings')
            examples = groups[split][number*32:(number+1)*32]
            require(len(batch['rows']) == len(examples), 'No missing or padded prediction rows')
            for row, example in zip(batch['rows'], examples, strict=True):
                strict_prediction(row)
                require(target_identity(row) == target_identity(prediction_row(example, row['output83'])),
                        'Every saved label/input matches its authenticated original source')
            bundles[model_id][split].extend(batch['rows'])
            all_paths.append(expected_path.resolve())
    require(len(all_paths) == len(set(all_paths)) == document['forward_calls'], 'Unique exact total forward inventory')
    actual_paths = {p.resolve() for p in (folder/'predictions').rglob('*') if p.is_file()}
    require(actual_paths == set(all_paths), 'No missing or extra prediction evidence')
    return bundles, document


def verify_evaluation(folder=None, boundary=lambda: None):
    bundles, document = load_predictions(folder, boundary=boundary)
    result = dict(status='training4096_saved_evaluation_verified', model_calls=0, optimizer_calls=0,
        models=list(MODEL_IDS), forward_calls=document['forward_calls'],
        every_original_label_and_input_checked=True, all_seven_predictions_fresh=True,
        evaluation=c.artifact((Path(folder) if folder is not None else c.OUT/'jobs/offline/worker')/'evaluation_result.json'))
    del bundles
    return result


def run_analysis(folder, boundary=lambda: None):
    from . import analysis
    return analysis.run_analysis(folder, boundary=boundary)


def verify_analysis(folder=None, boundary=lambda: None):
    from . import analysis
    return analysis.verify_analysis(c.OUT/'jobs/analysis/worker' if folder is None else folder,
                                    boundary=boundary)
