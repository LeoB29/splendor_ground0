"""Saved-only pooled analysis, standalone chart and independent final reader."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import psutil
from . import common as c


def main_games():
    result = {}
    for rep in (0, 1):
        folder = c.OUT/f'r{rep}'/'assessment'
        paths = sorted((folder/'games/main').glob('g*/result.json'))
        c.require(len(paths) == 200, 'Both complete fixed main cohorts required')
        result[rep] = [c.read(path) for path in paths]
    return result


def chart(result, path):
    c.require('torch' not in sys.modules, 'Plotting isolated from tensor runtime')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [result['descriptive_individuals'][str(rep)]['official'] for rep in (0, 1)] + [result['official']]
    labels = ['One-pass model 1\n200 games', 'One-pass model 2\n200 games', 'Fixed equal average\n400 games']
    colors = ['#4975A5', '#4975A5', '#007C78']
    fig, ax = plt.subplots(figsize=(10., 4.8))
    fig.subplots_adjust(left=.25, right=.98, bottom=.27, top=.85)
    for y, (row, color) in enumerate(zip(rows, colors)):
        value = row['mean_score']*100
        lo, hi = [x*100 for x in row['paired_ci95']]
        ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt='o', color=color,
                    capsize=6, markersize=9, linewidth=2)
        ax.annotate(f'{value:.2f}%  [{lo:.2f}, {hi:.2f}]', (hi, y), xytext=(10, 0),
                    textcoords='offset points', va='center', fontsize=10)
    low = min(35., *(row['paired_ci95'][0]*100-4 for row in rows))
    high = max(70., *(row['paired_ci95'][1]*100+20 for row in rows))
    ax.set_xlim(max(0, low), min(115, high))
    ax.axvline(50, color='#555555', linewidth=1.2, label='Equal score: 50%')
    ax.axvline(55, color='#b77800', linewidth=1.2, linestyle='--', label='Useful-gain score threshold: 55%')
    ax.set_yticks(range(3), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Score against original EF8A (%) — win 1, draw ½, loss 0')
    ax.set_title('Does one pass of training improve play?', loc='left', fontsize=16, fontweight='bold')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.grid(axis='x', alpha=.13)
    # A separate footer band keeps the legend clear of every result annotation.
    legend_axis = fig.add_axes([.25, .13, .73, .07])
    legend_axis.axis('off')
    handles, legend_labels = ax.get_legend_handles_labels()
    legend_axis.legend(handles, legend_labels, loc='center', ncol=2, frameon=False, fontsize=8.5)
    fig.text(.015, .025, '95% paired intervals. Individual rows are descriptive; the fixed average is primary.\n'
             'Matched 128 simulations, three worlds. Additional seat, uncertainty and conservative-scoring criteria apply.',
             fontsize=9, color='#555555')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def aggregate(envelope):
    from .assessment import analysis
    games = main_games()
    for rep in (0,1):
        receipt = c.read(c.OUT/'segments'/f'saved_r{rep}'/'execution.json')
        c.require(receipt['status'] == 'passed' and receipt['worker_exit_code'] == 0 and
                  not receipt['cleanup_errors'], 'Saved scientific audits passed with observed exits')
    value = analysis.analyze_pooled(games, True, boundary=envelope.guard)
    proof = analysis.verify_pooled(games, value, boundary=envelope.guard)
    result = c.save(c.OUT/'report/analysis.json',value)
    c.save(c.OUT/'report/arithmetic_verification.json',proof)
    path = c.OUT/'report/strength.png'
    c.require(not path.exists(), 'No chart overwrite')
    chart(value,path)
    if c.WRITE_OBSERVER: c.WRITE_OBSERVER(path,path.stat().st_size)
    official, conservative = value['official'], value['conservative']
    text = ('# One pass training and fresh playing strength\n\n'
            'Both fixed one-pass models faced original EF8A at128 simulations each. '
            'The fixed400 main games follow12 excluded operational games.\n\n'
            f"Official average score: **{official['mean_score']*100:.3f}%**, paired95% interval "
            f"**{official['paired_ci95'][0]*100:.2f}–{official['paired_ci95'][1]*100:.2f}%**. "
            f"Conservative score: **{conservative['mean_score']*100:.3f}%**.\n\n"
            f"Frozen useful-gain criteria met: **{value['useful_gain_established']}**. "
            'The original checkpoint remains the reference; no automatic promotion occurs.\n\n'
            '## What this result establishes\n\n'
            'This tests the average of these two fixed trained models against their original parent. '
            'Individual intervals are descriptive. It does not establish performance across arbitrary '
            'training seeds, independent opponents, superhuman strength or one-second deployability. '
            'Late completed moves were accepted under the unchanged research guards.\n\n'
            '## The next step\n\n'
            'Use the complete effect size, uncertainty, individual results and conservative analysis '
            'to recommend a separate confirmation or a specific training change. No new study launches automatically.\n\n'
            '## The broader picture\n\n'
            'Search improvements supplied stronger guidance; the saved-corpus diagnostics suggested '
            'shorter training can retain a modest policy signal. This fresh paired comparison tests '
            'whether that signal improves actual play. Larger-data, external-opponent and six-world work remain deferred.\n')
    with (c.OUT/'report/results.md').open('x',encoding='utf-8',newline='\n') as stream: stream.write(text)
    if c.WRITE_OBSERVER: c.WRITE_OBSERVER(c.OUT/'report/results.md',(c.OUT/'report/results.md').stat().st_size)
    return dict(status='aggregate_complete',analysis=result,figure=c.artifact(path),summary=c.artifact(c.OUT/'report/results.md'))


def independent_reader(envelope, freeze):
    from .assessment import analysis
    from .owner import segment_id
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    expected = [('qualification',rep,None) for rep in (0,1)]
    expected += [('operational',rep,None) for rep in (0,1)]
    expected += [('main_block',rep,start) for start in range(0,200,10) for rep in (0,1)]
    expected += [('replay',rep,None) for rep in (0,1)]
    expected += [('saved',rep,None) for rep in (0,1)] + [('aggregate',0,None)]
    expected += [('independent',rep,None) for rep in (0,1)]
    actual = sorted(p.parent.name for p in (c.OUT/'segments').glob('*/execution.json'))
    c.require(actual == sorted(segment_id(*item) for item in expected), 'Exact completed segment inventory')
    previous_end = None
    for mode, rep, start in expected:
        envelope.guard()
        folder = c.OUT/'segments'/segment_id(mode,rep,start)
        receipt, owner = c.read(folder/'execution.json'), c.read(folder/'owner.json')
        c.require(receipt['status'] == 'passed' and receipt['worker_exit_code'] == 0 and
                  receipt['cleanup_errors'] == [] and receipt['error'] is None, 'All actual segment exits passed')
        c.require(receipt['identity'] == owner['identity'], 'Exact actual owner')
        if previous_end is not None:
            c.require(owner['created_monotonic'] >= previous_end, 'No overlapping segment controllers')
        previous_end = receipt['ended_monotonic']
        for item in receipt['observed_owned_identities']: require_absent(item)
        require_absent(receipt['identity'])
        segment = c.read(folder/'claim.json')
        phase = c.read(c.ROOT/segment['phase_clock']['path'])
        c.require(receipt['ended_monotonic'] < phase['monotonic_deadline'] and
                  receipt['ended_unix'] < phase['unix_deadline'], 'Actual exit inside inherited original clocks')
    value = c.read(c.OUT/'report/analysis.json')
    arithmetic = analysis.verify_pooled(main_games(), value, boundary=envelope.guard)
    for item in freeze['inputs']:
        envelope.guard(); c.verify([item])
    inventory=[]
    # Seal finished scientific evidence. Exclude currently open owner/reader logs.
    for base in (c.OUT/'r0',c.OUT/'r1',c.OUT/'report'):
        for path in sorted(base.rglob('*')):
            if path.is_file(): envelope.guard(); inventory.append(c.artifact(path))
    inventory_item = c.save(c.OUT/'final_inventory.json',dict(schema='onepass_completed_science_inventory_v1',
                         artifacts=inventory, files=len(inventory), bytes=sum(x['bytes'] for x in inventory)))
    for item in inventory: envelope.guard(); c.verify([item])
    workers = list(c.OUT.glob('r*/assessment/workers/*/initialization.json'))
    c.require(len(workers) <= 812 <= freeze['maximum_live_lifecycles'], 'Bounded complete lifecycle accounting')
    for path in workers:
        init = c.read(path); require_absent(init['identity'])
    value = dict(schema='onepass_strength_independent_acceptance_v1', approved=True, blocking_findings=[],
        fixed_main_games=400, operational_games=12, compared_checkpoints=2,
        independent_per_repetition_saved_reviews=[c.artifact(c.OUT/'segments'/f'independent_r{rep}'/'result.json') for rep in (0,1)],
        independent_pooled_arithmetic=arithmetic, completed_segments=len(expected),
        exact_worker_lifecycles=len(workers), all_completed_segment_exits_observed=True,
        saved_inventory=inventory_item, frozen_inputs_unchanged=len(freeze['inputs']),
        reference_unchanged=True, automatic_promotion=False, actual_outer_owner_exit_pending=True,
        charts_require_final_visual_inspection=True, model_calls=0, new_games=0)
    return c.save(c.OUT/'report/independent_acceptance.json', value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('aggregate','reader'),required=True)
    parser.add_argument('--segment',required=True);parser.add_argument('--freeze-sha256',required=True)
    args=parser.parse_args()
    folder=Path(args.segment).resolve()
    c.require(folder.parent==c.OUT/'segments','Exact saved-only segment')
    c.require(c.file_hash(c.OUT/'freeze.json')==args.freeze_sha256,'Frozen saved-only reader')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(folder/'claim.json')
    from .owner import validate_segment_claim
    validate_segment_claim(claim,folder,args.mode,0,None,args.freeze_sha256)
    c.require(claim['mode']==args.mode and claim['repetition']==0,'Saved-only fixed owner role')
    c.require(claim['freeze_sha256']==args.freeze_sha256 and
              c.read(c.ROOT/claim['claim']['path'])['freeze']==c.artifact(c.OUT/'freeze.json'),
              'Original owner and saved-only segment freeze binding')
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placement.configure_execution(freeze,'isolated')
    from .envelope import SegmentEnvelope
    envelope=SegmentEnvelope(claim['claim'],claim['phase_clock'],c.artifact(folder/'claim.json'))
    status,error,value='failed',None,None
    try:
        c.require('torch' not in sys.modules,'No models in saved-only report process')
        value=aggregate(envelope) if args.mode=='aggregate' else independent_reader(envelope,freeze)
        envelope.finish();status='passed'
    except BaseException: error=traceback.format_exc()
    c.save(folder/'result.json',dict(status=status,mode=args.mode,repetition=0,result=value,error=error,
        pid=os.getpid(),created=psutil.Process().create_time(),new_games=0,model_calls=0))
    if error: print(error,file=sys.stderr)
    return 0 if status=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
