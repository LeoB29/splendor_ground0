"""Model-free synthesis of three already accepted, prospectively fixed matchups.

This reader does not pool different opponents into a win rate, choose a seed,
rerun a model, or replace each group's independent game/bootstrap audit.
"""
from copy import deepcopy
import math
from pathlib import Path
import sys

from . import common as c

GROUPS = ('mixed_terminal', 'mixed_parent', 'terminal_parent')
ARMS = {
    'mixed_terminal': ('mixed', 'terminal'),
    'mixed_parent': ('mixed', 'parent'),
    'terminal_parent': ('terminal', 'parent'),
}
LABELS = {
    'mixed_terminal': 'Mixed targets vs terminal targets',
    'mixed_parent': 'Mixed targets vs original',
    'terminal_parent': 'Terminal targets vs original',
}


def _interval(value, label):
    c.require(type(value) is list and len(value) == 2 and
              all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in value)
              and value[0] <= value[1], 'Bounded ordered ' + label)
    return value


def _score(score):
    """Recompute integer adoption conditions, including both uncertainty levels."""
    fields = ('games', 'pairs', 'numerator', 'denominator', 'seat_denominator', 'wins', 'draws', 'losses')
    c.require(all(type(score[k]) is int for k in fields), 'Exact score integer types')
    c.require(score['games'] == 400 and score['pairs'] == 200 and
              score['denominator'] == 800 and score['seat_denominator'] == 400,
              'Each group contains exactly400 main games and200 pairs')
    c.require(all(score[k] >= 0 for k in ('wins', 'draws', 'losses')) and
              score['wins'] + score['draws'] + score['losses'] == 400 and
              score['numerator'] == 2 * score['wins'] + score['draws'], 'Integer score arithmetic')
    c.require(type(score['mean_score']) in (int, float) and
              score['mean_score'] == score['numerator'] / 800, 'Exact full-sample score')
    seats = score['seat_numerators']
    c.require(type(seats) is dict and set(seats) == {'0', '1'} and
              all(type(x) is int and 0 <= x <= 400 for x in seats.values()) and
              sum(seats.values()) == score['numerator'], 'Both balanced seats retained')
    c.require(score['by_candidate_seat'] == {seat: value / 400 for seat, value in seats.items()},
              'Exact seat scores')
    pairs = score['pair_numerators']
    c.require(type(pairs) is list and len(pairs) == 200 and
              all(type(x) is int and 0 <= x <= 4 for x in pairs) and
              sum(pairs) == score['numerator'], 'All200 pair scores retained')
    ci95 = _interval(score['paired_ci95'], '95% interval')
    ci975 = _interval(score['paired_ci975'], '97.5% interval')
    c.require(ci975[0] <= ci95[0] and ci975[1] >= ci95[1], 'Adjusted interval contains unadjusted interval')
    results = {}
    for level, interval, log_factor, gates_key, useful_key in (
        ('95', ci95, 40, 'useful_gain_criteria', 'useful_gain'),
        ('975', ci975, 80, 'familywise_parent_criteria', 'familywise_parent_useful_gain'),
    ):
        radius = math.sqrt(math.log(log_factor) / 400)
        bound = [max(0., score['mean_score'] - radius), min(1., score['mean_score'] + radius)]
        c.require(score['bounded_pair_hoeffding_' + level] == bound, 'Unchanged bounded-pair safeguard')
        gates = dict(score_at_least_55pct=100 * score['numerator'] >= 55 * score['denominator'],
            each_seat_score_at_least_50pct=all(2 * n >= 400 for n in seats.values()),
            paired_lower_bound_above_50pct=interval[0] > .5,
            degenerate_interval_safeguard=interval[0] != interval[1] or bound[0] > .5)
        c.require(c.canonical(score[gates_key]) == c.canonical(gates) and
                  type(score[useful_key]) is bool and score[useful_key] == all(gates.values()),
                  'Preserved score, seat, uncertainty and degenerate criteria')
        results[level] = all(gates.values())
    c.require(score['bootstrap_degenerate'] is (ci95[0] == ci95[1]), 'Exact degenerate flag')
    return results


def validate_group(group, analysis, acceptance, *, analysis_descriptor=None):
    c.require(group in GROUPS and analysis['matchup'] == group and acceptance['matchup'] == group,
              'No substituted or renamed matchup')
    candidate, reference = ARMS[group]
    c.require(analysis.get('schema') == 'full_corpus_pooled_strength_analysis_v1' and
              analysis.get('status') == 'complete_verified_both_repetitions' and
              analysis['candidate_arm'] == acceptance['candidate_arm'] == candidate and
              analysis['reference_arm'] == acceptance['reference_arm'] == reference and
              analysis['control_arm'] == reference, 'Exact candidate/reference direction')
    c.require(analysis['checkpoint_names'] == [f'r{rep}_{candidate}_joint' for rep in (0, 1)] and
              analysis['reference_checkpoints'] ==
              [('original_ef8a' if reference == 'parent' else f'r{rep}_{reference}_joint') for rep in (0, 1)],
              'Both prospectively fixed checkpoints, no best-seed choice')
    c.require(analysis['primary_estimand'] == 'equal_mean_of_two_fixed_seed_pairs_for_' + group and
              analysis['games'] == 400 and type(analysis['games']) is int and
              analysis['pairs'] == 200 and type(analysis['pairs']) is int and
              analysis['allocation_each'] == dict(games=200, pairs=100), 'Exact fixed-repetition allocation')
    c.require(analysis['candidate_simulations'] == analysis['reference_simulations'] == 128 and
              analysis['world_pool_each'] == 3 and
              analysis['research_guard_seconds'] == dict(internal=9., complete_response=10.),
              'Matched search and unchanged research guards')
    c.require(analysis['complete_verification'] is True and analysis['evidence_gates_passed'] is True and
              analysis['automatic_promotion'] is False and analysis['individual_checkpoint_selection'] is False and
              analysis['operational_games_pooled'] is False and analysis['other_study_games_pooled'] is False and
              analysis['no_sample_extension'] is True, 'Complete frozen assessment with no selection or pooling')
    c.require(acceptance.get('schema') == 'full_corpus_strength_independent_acceptance_v1' and
              acceptance['approved'] is True and acceptance['blocking_findings'] == [] and
              type(acceptance['fixed_main_games']) is int and acceptance['fixed_main_games'] == 400 and
              type(acceptance['operational_games']) is int and acceptance['operational_games'] == 12 and
              acceptance['all_completed_segment_exits_observed'] is True and
              acceptance['automatic_promotion'] is False and acceptance['model_calls'] == acceptance['new_games'] == 0,
              'Each exact group has independent acceptance and observed stage exits')
    proof = acceptance['independent_pooled_arithmetic']
    c.require(proof['status'] == 'passed' and proof['verified_result_sha256'] == c.digest(analysis),
              'Independent arithmetic authenticates this exact analysis')
    if analysis_descriptor is not None:
        c.require(acceptance['analysis'] == analysis_descriptor, 'Acceptance binds physical analysis bytes')
    official, conservative = _score(analysis['official']), _score(analysis['conservative'])
    raw95 = official['95'] and conservative['95']
    adjusted = official['975'] and conservative['975']
    primary = group == 'mixed_terminal'
    c.require(analysis['unadjusted_95_criteria_met'] is raw95 and
              analysis['useful_gain_established'] is (raw95 if primary else False) and
              analysis['familywise_parent_useful_gain'] is (False if primary else adjusted) and
              analysis['uncertainty_role'] == ('primary_target_contrast95' if primary else
                                               'secondary_parent_family_bonferroni97.5'),
              'Primary95 and two-parent Bonferroni97.5 roles cannot be exchanged')
    interval_key = 'paired_ci95' if primary else 'paired_ci975'
    return dict(matchup=group, label=LABELS[group], candidate_arm=candidate, reference_arm=reference,
        role='primary_target_contrast' if primary else 'secondary_parent_comparison',
        main_games=400, operational_games=12, paired_setups=200, fixed_training_repetitions=2,
        interval_confidence=.95 if primary else .975,
        official=dict(score=analysis['official']['mean_score'], interval=deepcopy(analysis['official'][interval_key]),
                      wins=analysis['official']['wins'], draws=analysis['official']['draws'], losses=analysis['official']['losses']),
        conservative=dict(score=analysis['conservative']['mean_score'], interval=deepcopy(analysis['conservative'][interval_key])),
        useful_gain_criteria_met=raw95 if primary else adjusted,
        administrative_games=analysis['administrative_sensitivity']['administrative_games'],
        original_reference_remains=True, automatic_promotion=False)


def synthesize(records):
    c.require(type(records) is dict and set(records) == set(GROUPS), 'All and only three complete fixed matchups')
    rows = []
    for group in GROUPS:
        row = records[group]
        rows.append(validate_group(group, row['analysis'], row['acceptance'],
                                  analysis_descriptor=row.get('analysis_descriptor')))
    return dict(schema='full_corpus_family_summary_v1', status='three_matchups_independently_accepted',
        main_games=1200, operational_games=36, pairs=600, comparisons=rows,
        primary_target_comparison='mixed_terminal',
        secondary_parent_family=dict(comparisons=['mixed_parent', 'terminal_parent'],
            method='Bonferroni_two_two_sided_97.5pct_paired_intervals', family_alpha=.05),
        combined_win_rate_across_opponents=None, training_repetitions_are_not_extra_independent_games=True,
        no_best_seed_selection=True, no_automatic_promotion=True, original_model_remains_reference=True,
        complete_owner_exit_still_requires_observation=True,
        inference_limits=['400 games per contrast remains a coarse learning screen, not a well-powered test of a three-point gain.',
            'Two fixed training seeds do not establish generalization over arbitrary training seeds.',
            'Reused offline positions are diagnostics; only these fresh games are independent assessment.',
            'No historical game result is a concurrent control.',
            'No superhuman, independent-opponent or strict one-second deployment claim follows.'])


def chart(summary, path):
    c.require('torch' not in sys.modules, 'Family figure loads no model runtime')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = summary['comparisons']
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(left=.30, right=.77, bottom=.26, top=.83)
    all_bounds = [x for row in rows for score in ('official', 'conservative') for x in row[score]['interval']]
    lo, hi = max(0, min(35, 100 * min(all_bounds) - 5)), min(100, max(65, 100 * max(all_bounds) + 5))
    for y, row in enumerate(rows):
        for name, offset, color, marker in (('official', -.09, '#087f8c', 'o'), ('conservative', .09, '#607080', 's')):
            score = row[name]; mean = 100 * score['score']; lower, upper = [100*x for x in score['interval']]
            ax.errorbar(mean, y + offset, xerr=[[mean - lower], [upper - mean]], fmt=marker,
                color=color, markerfacecolor=color if name == 'official' else 'white',
                linewidth=2 if name == 'official' else 1.25, markersize=7, capsize=5,
                label=name.capitalize() if y == 0 else None)
        main = row['official']; secondary = row['conservative']
        score_text = f"{100*main['score']:.2f}%  [{100*main['interval'][0]:.2f}, {100*main['interval'][1]:.2f}]"
        if main['score'] != secondary['score'] or main['interval'] != secondary['interval']:
            score_text += f"\nConservative: {100*secondary['score']:.2f}%"
        ax.text(1.035, y, score_text, transform=ax.get_yaxis_transform(), va='center', fontsize=10)
    ax.set_yticks(range(3), [row['label'] + ('\nPrimary · 95% interval' if row['role'].startswith('primary')
                                         else '\nSecondary · adjusted 97.5% interval') for row in rows])
    ax.set_ylim(2.55, -.55); ax.set_xlim(lo, hi)
    ax.axvline(50, color='#555555', linewidth=1.2, label='Equal score: 50%')
    ax.axvline(55, color='#b77800', linewidth=1.2, linestyle='--', label='Useful-gain score threshold: 55%')
    ax.set_xlabel('Score of the first player named (%) · win 1, draw ½, loss 0')
    ax.grid(axis='x', alpha=.15); ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=14)
    fig.suptitle('Does a full training pass improve play?', x=.03, y=.97, ha='left', fontsize=20, fontweight='bold')
    fig.text(.03, .90, '57,022 training positions per model · same policy targets and training effort · 1,200 fresh games', fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.53, .105), ncol=2, frameon=False, fontsize=10)
    fig.text(.03, .025, 'Each row: 400 games, 200 swapped-seat pairs, equal average of two fixed training seeds.\n'
             'Official AND conservative results must meet score, seat, uncertainty and safeguard criteria; no automatic promotion.',
             fontsize=10, color='#555555')
    fig.savefig(path, dpi=170, bbox_inches='tight'); plt.close(fig)


def markdown(summary):
    lines = ['# Full-corpus learning screen', '',
        'All three fixed matchups completed independent verification: **1,200 main games**, plus **36 excluded operational games**.', '',
        '| Comparison | Role / interval | Official score | Conservative score | Useful-gain criteria |',
        '|---|---|---:|---:|---|']
    for row in summary['comparisons']:
        official, conservative = row['official'], row['conservative']
        bounds = official['interval']
        role = 'Primary, 95%' if row['role'].startswith('primary') else 'Secondary, adjusted 97.5%'
        lines.append(f"| {row['label']} | {role} | {100*official['score']:.3f}% [{100*bounds[0]:.2f}, {100*bounds[1]:.2f}] | "
                     f"{100*conservative['score']:.3f}% | {'Met' if row['useful_gain_criteria_met'] else 'Not established'} |")
    lines += ['', 'Scores count a win as one and a draw as one half. Different opponents are not pooled into a single win rate.', '',
        '## What this result establishes', '',
        'The primary comparison isolates the value-target choice at matched full-corpus training effort. The two secondary comparisons assess '
        'each fixed training approach against original EF8A, with Bonferroni-adjusted intervals for the two-parent family. '
        'Useful gain requires both official and conservative score ≥55%, lower interval bound >50%, both seats ≥50%, and the degenerate-interval safeguard. '
        'A failure to meet these criteria does not establish equivalence or rule out a smaller improvement. '
        'Each contrast still contains 400 games; the total of 1,200 does not make every contrast precise enough for a three-point gain.', '',
        '## The next step', '',
        'Use the complete effect sizes, uncertainty and matched diagnostic changes to recommend a separate confirmation or training revision. '
        'Preserve all fixed-seed results; do not select the better checkpoint or promote a model automatically. Original EF8A remains the reference.', '',
        '## The broader picture', '',
        'This study uses the full existing corpus to give learning a substantially larger opportunity than the previous 96-update fits. '
        'Keeping data, policy targets and training exposure matched lets the direct comparison test whether the outcome/search mixture helps. '
        'The reused offline data are diagnostic; the fresh paired games test actual playing strength. '
        'This does not establish performance against independent opponents, superhuman play or strict one-second deployment.', '',
        'The independent group audits passed. Final whole-study acceptance, actual outer-owner exit and final visual inspection remain separate completion requirements.', '']
    return '\n'.join(lines)


def finalize(folder, boundary=lambda: None):
    folder = Path(folder).resolve()
    c.require(folder.is_relative_to(c.OUT), 'Family outputs stay in this new study')
    folder.mkdir(parents=True, exist_ok=True)
    c.require(not (folder/'family_summary.json').exists(), 'No replacement family report')
    records, artifacts = {}, {}
    for group in GROUPS:
        boundary(); base = c.OUT/'strength'/group/'report'
        paths = {name: base/(name + '.json') for name in ('analysis', 'independent_acceptance')}
        items = {name: c.artifact(path) for name, path in paths.items()}
        c.verify(list(items.values()), boundary=boundary)
        records[group] = dict(analysis=c.read(paths['analysis']), acceptance=c.read(paths['independent_acceptance']),
                              analysis_descriptor=items['analysis'])
        artifacts[group] = items
    summary = synthesize(records); summary['accepted_group_artifacts'] = artifacts
    boundary(); summary_item = c.save(folder/'family_summary.json', summary)
    figure = folder/'family_strength.png'; c.require(not figure.exists(), 'No overwrite of family figure')
    chart(summary, figure)
    document = folder/'results.md'
    with document.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(markdown(summary))
    if c.WRITE_OBSERVER:
        for path in (figure, document): c.WRITE_OBSERVER(path, path.stat().st_size)
    boundary()
    return dict(status='family_summary_complete', summary=summary_item, figure=c.artifact(figure),
                report=c.artifact(document), comparisons=3, main_games=1200, operational_games=36,
                model_calls=0, new_games=0, automatic_promotion=False, charts_require_visual_inspection=True)
