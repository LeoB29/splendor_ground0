"""At most two standalone scientific figures, drawn only after terminal audit."""
from . import common as c

def render(report,folder):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 folder.mkdir(parents=True,exist_ok=False)
 looks=[c.read(c.resolve(i)) for i in report['interims']]
 x=[r['games'] for r in looks];y=[r['look']['score']*100 for r in looks]
 lower=[r['look']['lower']*100 for r in looks];upper=[r['look']['upper']*100 for r in looks]
 fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
 ax.plot(x,y,'o-',color='#176b91',label='Six-world score')
 ax.fill_between(x,lower,upper,color='#176b91',alpha=.15,label='Sequentially adjusted interval')
 ax.axhline(50,color='#555555',linestyle='--',label='Equal score / player threshold')
 ax.axhline(48,color='#b87313',linestyle=':',label='Teacher lower-bound threshold')
 ax.set(xlabel='Verified games',ylabel='Six-world score (%)',title='Six worlds versus three — 512 simulations each')
 ax.legend(fontsize=9);ax.grid(alpha=.15)
 path=folder/'score.png';fig.savefig(path,dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
 data=[report['timings'][p] for p in ('3','6')]
 values=[[np.median(v),np.percentile(v,95),max(v)] if v else [float('nan')]*3 for v in data]
 x=np.arange(3)
 for i,(label,color) in enumerate((('Three worlds','#7b8792'),('Six worlds','#176b91'))):
  ax.bar(x+(i-.5)*.34,values[i],.34,label=label,color=color)
 ax.set_xticks(x,['Median','95th percentile','Longest completed move'])
 ax.set(ylabel='Complete response time (seconds)',title=f"Timing reported separately; {len(report['failed_decisions'])} deadline forfeits")
 ax.legend();ax.grid(axis='y',alpha=.15)
 if any(not v for v in data):
  ax.text(.01,.98,'Missing bars: no completed responses for that player',transform=ax.transAxes,va='top',fontsize=9)
 path2=folder/'timing.png';fig.savefig(path2,dpi=160);plt.close(fig)
 return [c.artifact(path),c.artifact(path2)]
