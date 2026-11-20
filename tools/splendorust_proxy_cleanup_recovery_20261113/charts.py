"""Identical chart function; separate permitted output directory."""
from pathlib import Path
from tools.splendorust_proxy_calibration_20261113.analysis import QUOTAS,ADJACENT
from . import common as c
require=c.require

def render_charts(result,folder,historical_controls=None,*,boundary=lambda:None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    folder=Path(folder)
    require(folder.resolve().is_relative_to(c.OUT),'New chart output only')
    folder.mkdir(parents=True,exist_ok=True)
    colors=('#0f766e','#c2410c');x=np.arange(4);artifacts=[]
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.8))
    primary=result['primary_contrasts'];est=result['primary_estimates']
    if est['64']['mean'] is not None:
        means=[100*est[str(q)]['mean'] for q in QUOTAS]
        lower=[100*est[str(q)]['ci95'][0] for q in QUOTAS]
        upper=[100*est[str(q)]['ci95'][1] for q in QUOTAS]
        axes[0].plot(x,means,'o-',color=colors[0],lw=2.5)
        axes[0].fill_between(x,lower,upper,color=colors[0],alpha=.15)
    else:axes[0].text(.5,.5,'Insufficient shared Q support',transform=axes[0].transAxes,ha='center')
    agreement=result['agreement_estimates'];mean=[100*agreement[str(q)]['mean'] for q in QUOTAS]
    lo=[100*agreement[str(q)]['ci95'][0] for q in QUOTAS];hi=[100*agreement[str(q)]['ci95'][1] for q in QUOTAS]
    axes[1].plot(x,mean,'o-',color=colors[1],lw=2.5);axes[1].fill_between(x,lo,hi,color=colors[1],alpha=.15)
    for ax in axes:
        ax.set_xticks(x,QUOTAS);ax.set_xlabel('EF8A simulations per decision');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('Deep-reference credit of selected actions',fontsize=11)
    axes[0].set_ylabel('Reference credit points (higher is better)')
    axes[1].set_title('Agreement with independent deep searches',fontsize=11);axes[1].set_ylabel('Matching Main actions (%)')
    fig.suptitle('Can the cheaper diagnostic recognize known search improvements?',fontsize=14)
    fig.text(.5,.025,f"Q uses {result['coverage']['supported_roots']}/512 jointly supported roots; agreement uses all 512. Two control seeds, two 2048-simulation references.\n"
        'Same-network surrogate, not measured win probability. Nominal 95% intervals; paired contrasts appear in the companion chart.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.12,1,.93));path=folder/'proxy_budget_calibration.png';require(not path.exists(),'Exclusive chart')
    fig.savefig(path,dpi=160);plt.close(fig);artifacts.append(c.artifact(path));boundary()
    if historical_controls is not None:
        if type(historical_controls) is dict:
            historical_controls=historical_controls['comparisons']
        require(type(historical_controls) is list and len(historical_controls)==3,'Three historical adjacent controls')
        fig,axes=plt.subplots(1,2,figsize=(11.2,4.8));labels=[]
        for j,(record,pair) in enumerate(zip(historical_controls,ADJACENT,strict=True)):
            require((record['reference_simulations'],record['candidate_simulations'])==pair and 0<=record['score']<=1 and
                len(record['paired_ci95'])==2 and 0<=record['paired_ci95'][0]<=record['paired_ci95'][1]<=1,'Exact historical score context')
            labels.append(f'{pair[0]} to {pair[1]}')
            axes[0].plot([100*v for v in record['paired_ci95']],[j,j],color=colors[0],lw=2)
            axes[0].plot(100*record['score'],j,'o',color=colors[0])
            row=primary[j]
            if row['difference'] is not None:
                axes[1].plot([100*v for v in row['family_ci']],[j,j],color=colors[1],lw=2)
                axes[1].plot(100*row['difference'],j,'o',color=colors[1])
        for ax in axes:
            ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        if all(row['difference'] is None for row in primary):
            axes[1].text(.5,.5,'No supported primary contrasts',transform=axes[1].transAxes,ha='center')
        axes[0].axvline(50,color='#64748b',ls='--');axes[1].axvline(0,color='#64748b',ls='--')
        axes[0].set_title('Previously verified live comparisons');axes[0].set_xlabel('Higher-budget player score (%) | 95% intervals')
        axes[1].set_title('New saved-position Q contrasts');axes[1].set_xlabel('Difference in Q credit points | 98.33% intervals')
        fig.suptitle('Positive controls: agreement of direction, not an exchange rate',fontsize=14)
        fig.text(.5,.025,'Historical live games and this reused saved-position panel are different evidence.\n'
            'These plots do not convert proxy differences into win rates or Elo, and do not validate changed network weights.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.13,1,.93));path=folder/'proxy_historical_controls.png';require(not path.exists(),'Exclusive chart')
        fig.savefig(path,dpi=160);plt.close(fig);artifacts.append(c.artifact(path));boundary()
    return artifacts
