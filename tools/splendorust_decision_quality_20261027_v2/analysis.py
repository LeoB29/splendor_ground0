"""Descriptive analysis of a fixed saved-position panel; no quality oracle."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
import statistics

from .audit import initial_network_win_credit, original, same

CONDITIONS = ((64, 'A'), (256, 'A'), (1024, 'A'), (1024, 'B'), (1024, 'C'))


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, median=None, p95=None, maximum=None)
    return dict(n=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                p95=values[max(0, math.ceil(.95*len(values))-1)], maximum=values[-1])


def _main(result):
    return result['diagnostics']['phases'][0]['action']


def _measure(row):
    result = row['clock']['result']
    phase = result['diagnostics']['phases'][0]
    telemetry = result['research']['phases'][0]['telemetry']
    raw83 = phase['inference'][0]['output83']
    logits = original.canonical_logits(raw83[:81])
    index = original.canonical_index(phase['action'])
    return dict(main_action=deepcopy(phase['action']), complete_action=deepcopy(result['action']),
                selected_index=index, root_visits=telemetry['visits'], root_q=telemetry['q'],
                root_visit_weighted_value=telemetry['root_value'],
                selected_edge_value=telemetry['selected_edge_value'],
                initial_network_win_credit=initial_network_win_credit(raw83),
                post_search_root_node_value=telemetry['network_value'],
                upstream_network_value_semantics='post_search_dynamic_fpu_running_mean',
                canonical_raw_logits=logits,
                selected_raw_logit=logits[index], raw_logits_are_normalized_priors=False,
                callback_count=result['diagnostics']['inference_calls'],
                leaf_calls=sum(p['leaf_calls'] for p in result['diagnostics']['phases']),
                simulations=sum(p['simulations'] for p in result['diagnostics']['phases']),
                complete_response_seconds=row['clock']['receipt_elapsed_seconds'],
                parent_elapsed_seconds=row['clock']['elapsed_seconds'])


def fixed_pair_comparison(original_action, alternative_action, references):
    """A fixes the pair. B/C may assess it, never select another alternative.

    This is same-network search stability. In particular, positive Q differences
    are not factual regret, a blunder label, or independent evaluation.
    """
    old = original.canonical_index(original_action)
    alt = original.canonical_index(alternative_action)
    result = dict(original_index=old, alternative_index=alt,
                  alternative_selected_from='1024-A_only', same_action=old == alt,
                  references={}, eligible=False, consensus_preference=None)
    for label in ('B', 'C'):
        observed = references.get(label)
        if observed is None:
            result['references'][label] = dict(available=False, both_edges_visited=False,
                                               alternative_minus_original_q=None)
            continue
        visits, qs = observed['root_visits'], observed['root_q']
        both = visits[old] > 0 and visits[alt] > 0 and qs[old] is not None and qs[alt] is not None
        result['references'][label] = dict(available=True, both_edges_visited=both,
            original_visits=visits[old], alternative_visits=visits[alt],
            original_q=qs[old], alternative_q=qs[alt],
            alternative_minus_original_q=qs[alt]-qs[old] if both else None)
    eligible = old != alt and all(r['both_edges_visited'] for r in result['references'].values())
    result['eligible'] = eligible
    if eligible:
        diffs = [r['alternative_minus_original_q'] for r in result['references'].values()]
        result['consensus_preference'] = ('alternative' if min(diffs) > 0 else
                                         'original' if max(diffs) < 0 else
                                         'tie' if diffs == [0, 0] else 'mixed_or_tied')
    return result


def analyze(records, panel, matrix, *, complete):
    """Only reads already-audited records; preserves every missing denominator."""
    research = defaultdict(dict)
    baseline = {}
    cost = defaultdict(list)
    callbacks = defaultdict(list)
    failures = []
    for row in records:
        spec = row['spec']
        if not row['completed']:
            failures.append(dict(sequence=spec['sequence'], root_id=spec['root_id'],
                                 stage=spec['stage'], status=row['clock']['status']))
            continue
        if spec['stage'] == 'frozen_reference_reproduction':
            baseline[spec['root_id']] = row['clock']['result']
        elif spec['stage'] in ('research_pilot', 'research_main'):
            key = (spec['quota'], spec['seed_family'])
            measured = _measure(row)
            research[spec['root_id']][key] = measured
            label = str(key[0])+'-'+key[1]
            cost[label].append(measured['complete_response_seconds'])
            callbacks[label].append(measured['callback_count'])

    # Materialize ALL pairs using original64 and A only before examining B/C.
    fixed_pairs = {}
    for descriptor in panel['roots']:
        rid = descriptor['root_id']
        available = research[rid]
        if rid in baseline and (1024, 'A') in available:
            fixed_pairs[rid] = (deepcopy(_main(baseline[rid])),
                               deepcopy(available[(1024, 'A')]['main_action']))

    roots = []
    transitions = {name: dict(available=0, main_changes=0, complete_action_changes=0)
                   for name in ('64A_to_256A', '256A_to_1024A', '64A_to_1024A')}
    stability = Counter()
    cells = defaultdict(Counter)
    preferences = Counter()
    for descriptor in panel['roots']:
        rid = descriptor['root_id']; available = research[rid]
        cell = str(descriptor['phase'])+'_seat'+str(descriptor['seat'])
        entry = dict(root_id=rid, cell=cell, ply=descriptor['ply'], seat=descriptor['seat'],
                     original64_reproduced=rid in baseline,
                     conditions={str(q)+'-'+s: available[(q,s)] for q,s in CONDITIONS if (q,s) in available},
                     missing_conditions=[str(q)+'-'+s for q,s in CONDITIONS if (q,s) not in available],
                     fixed_pair=None, seed_agreement=None)
        cells[cell]['planned_roots'] += 1
        cells[cell]['complete_research_roots'] += int(len(available) == 5)
        for name, left, right in (('64A_to_256A', (64,'A'), (256,'A')),
                                  ('256A_to_1024A', (256,'A'), (1024,'A')),
                                  ('64A_to_1024A', (64,'A'), (1024,'A'))):
            if left in available and right in available:
                transitions[name]['available'] += 1
                transitions[name]['main_changes'] += int(not same(available[left]['main_action'], available[right]['main_action']))
                transitions[name]['complete_action_changes'] += int(not same(available[left]['complete_action'], available[right]['complete_action']))
        if all((1024,s) in available for s in ('A','B','C')):
            actions = [available[(1024,s)]['main_action'] for s in ('A','B','C')]
            entry['seed_agreement'] = dict(A_equals_B=same(actions[0],actions[1]),
                                          A_equals_C=same(actions[0],actions[2]),
                                          B_equals_C=same(actions[1],actions[2]),
                                          all_equal=same(actions[0],actions[1]) and same(actions[1],actions[2]))
            stability['available_roots'] += 1
            stability['all_three_agree'] += int(entry['seed_agreement']['all_equal'])
            stability['B_C_disagree'] += int(not entry['seed_agreement']['B_equals_C'])
        if rid in fixed_pairs:
            old, alt = fixed_pairs[rid]
            entry['fixed_pair'] = fixed_pair_comparison(old, alt,
                {s: available[(1024,s)] for s in ('B','C') if (1024,s) in available})
            preferences['fixed_pairs_available'] += 1
            preferences['different_actions'] += int(not entry['fixed_pair']['same_action'])
            preferences['both_edges_visited_in_both_references'] += int(entry['fixed_pair']['eligible'])
            if entry['fixed_pair']['eligible']:
                preferences[entry['fixed_pair']['consensus_preference']] += 1
        roots.append(entry)
    if not complete:
        next_study = 'Resolve the recorded operational or evidence limit; this incomplete panel cannot settle the diagnostic.'
    else:
        next_study = ('A single next hypothesis requires joint interpretation of the budget-action changes, '
                      'A/B/C agreement, both-reference visited-edge coverage and measured cost. '
                      'No automatic numerical trigger or successor selection was prepared; '
                      'these descriptive patterns alone cannot establish improved play.')
    return dict(schema='splendorust_saved_decision_descriptive_analysis_v2',
                complete=bool(complete), planned_roots=24, roots_with_any_research=len([v for v in research.values() if v]),
                all_five_conditions_roots=sum(len(v)==5 for v in research.values()),
                first_failure=failures[0] if failures else None, stage='descriptive_diagnostic',
                budget_action_changes=transitions, high_budget_seed_agreement=dict(stability),
                A_fixed_pair_assessment=dict(preferences), cell_coverage={k:dict(v) for k,v in cells.items()},
                timing={k:distribution(v) for k,v in cost.items()},
                callbacks={k:distribution(v) for k,v in callbacks.items()}, roots=roots,
                next_study_recommendation=next_study,
                caveats=['Fixed small historical panel, not a population estimate.',
                         'A selects the pair before B/C are inspected; B/C are the same network and jointly vary search/world randomness.',
                         'Missing Q is absent, never zero; pair preference requires both edges visited in both B/C.',
                         'Q and network values are acting-player estimates, not observed returns or factual regret.',
                         'The initial network prediction is derived from the first callback; the upstream network_value field is the post-search dynamic-FPU root node mean.',
                         'Recorded policy values are raw logits, not normalized root priors.',
                         'No playing-strength, Elo, blunder-rate, causal bottleneck or automatic promotion claim.'])
