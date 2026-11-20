"""Final engineering metrics, including separately timed final verification."""
from copy import deepcopy
import math
from . import common as c

def final_metrics(acceptance,execution,resources,logical_processors):
 charges=acceptance['final_verification_seconds_by_arm']
 elapsed=execution['monotonic_end']-execution['monotonic_start']
 c.require(set(charges)=={'A1','A4','B-control','B-candidate'} and math.isfinite(elapsed) and elapsed>0
  and all(math.isfinite(x) and x>=0 for x in charges.values()) and sum(charges.values())<=elapsed+1e-6,
  'Finite nonnegative final charges within actual acceptance duration')
 unassigned=max(0.,elapsed-sum(charges.values()))
 rows={};live=acceptance['reconstructed_live'];collection=acceptance['reconstructed_collection']
 for arm in ('A1','A4','B-control','B-candidate'):
  source=live[arm] if arm.startswith('A') else collection[arm[2:]]
  duration=source['verified_seconds']+charges[arm]+unassigned/4
  final_charge=charges[arm]+unassigned/4
  cpu=resources['arm_cpu_seconds'].get(arm,0)+execution.get('cpu_seconds',0)*final_charge/elapsed
  rows[arm]=dict(games=source['games'],decisions=source['decisions'],verified_seconds=duration,
   verified_games_per_hour=source['games']*3600/duration,verified_decisions_per_hour=source['decisions']*3600/duration,
   response_seconds=source.get('response_seconds',source.get('complete_response_seconds')),
   final_verification_seconds_charged=charges[arm]+unassigned/4,
   cpu_seconds_approximate=cpu,cpu_core_equivalents_approximate=cpu/duration,
   cpu_percent_machine_approximate=cpu/duration/logical_processors*100,
   peak_rss_bytes_sampled=max(resources['arm_peak_rss'].get(arm,0),execution.get('peak_tree_rss_bytes',0)),technical_failures=0)
  if arm.startswith('B'):
   rows[arm]['cell_completion_cadence_seconds']=source.get('cell_completion_cadence_seconds')
   rows[arm]['cadence_scope']=source.get('cadence_scope','One observed X-to-Y transition; not asymptotic steady state')
 live_gain=rows['A4']['verified_games_per_hour']/rows['A1']['verified_games_per_hour']
 collection_gain=rows['B-candidate']['verified_games_per_hour']/rows['B-control']['verified_games_per_hour']
 return dict(arms=rows,live_speedup=live_gain,collection_speedup=collection_gain,
  live_adopt=live_gain>=2 and rows['A4']['response_seconds']['maximum']<60,
  collection_adopt=collection_gain>=1.3,all_identity_checks_exact=True,all_audits_passed=True,
  final_verification_accounting='Arm-specific final audit time; shared work split equally between relevant arms; independent-worker initialization, source preservation, exit/cleanup overhead split equally across all four arms',
  engineering_only=True,strength_claim=False)

def write():
 complete=c.read(c.OUT/'complete.json');final=c.read(c.OUT/'owner_final.json')
 observed=c.read(c.OUT/'observed_owner_exit.json')
 c.require(observed['status']=='passed' and observed['actual_root_exit']==0 and not observed['cleanup_errors'],'Report after actual successful owner exit')
 receipt=complete['payload']['independent_acceptance'];c.verify([receipt]);execution=c.read(c.resolve(receipt))
 c.verify([execution['result']]);acceptance=c.read(c.resolve(execution['result']))['payload']
 plan=c.read(c.PLAN);metrics=final_metrics(acceptance,execution,final,plan['topology']['validation']['logical_processor_count'])
 result=c.save(c.OUT/'results.json',dict(status='verified_engineering_result',metrics=metrics,
  independent_acceptance=receipt,actual_root_exit=c.artifact(c.OUT/'observed_owner_exit.json'),
  live=complete['payload']['live'],collection=complete['payload']['collection'],time=c.now()))
 lines=['# Parallel-throughput pilot result','','Engineering comparison only; all96 matched engineering games remain excluded from every corpus.','',
  '| Arm | Verified games/h | Decisions/h | Response median / p95 / max (s) | Approx. CPU cores | Sampled peak RSS (GiB) | Failures |',
  '|---|---:|---:|---|---:|---:|---:|']
 for arm,row in metrics['arms'].items():
  timing=row['response_seconds'];maximum=timing.get('maximum',timing.get('max'))
  median=timing.get('median',timing.get('p50'))
  lines.append(f"| {arm} | {row['verified_games_per_hour']:.2f} | {row['verified_decisions_per_hour']:.1f} | {median:.3f} / {timing['p95']:.3f} / {maximum:.3f} | {row['cpu_core_equivalents_approximate']:.2f} | {row['peak_rss_bytes_sampled']/c.GiB:.2f} | 0 |")
 lines+=['',f"Live: {metrics['live_speedup']:.3f}×; {'adopt A4' if metrics['live_adopt'] else 'retain A1'}. Required2.0× and all responses below60s.",
  f"Collection: {metrics['collection_speedup']:.3f}×; {'adopt four processes' if metrics['collection_adopt'] else 'retain current CPU8'}. Required1.3×.",'',
  'Every scientific identity and audit check passed. Final independent verification time and worker exit/cleanup are included. CPU and RSS are sampled owned-process measurements, not a host-saturation benchmark.',
  f"Candidate collection X-to-Y verified completion interval: {metrics['arms']['B-candidate']['cell_completion_cadence_seconds']}s. This is one observed transition, not established long-run steady-state throughput.",
  'First intended use: the fresh live strength test after the4,096-game training study. Collection recommendation applies only to a later corpus. No running study is changed.',
  '',f"Machine-readable receipts: `{result['path']}`."]
 path=c._destination(c.OUT/'results.md');path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,3.5));values=[metrics['live_speedup'],metrics['collection_speedup']]
 ax.bar(['Live: A4 / A1','Collection: four processes / CPU8'],values,color=['#287f8e','#5a66a3'])
 for i,bar in enumerate((2.,1.3)):ax.plot([i-.35,i+.35],[bar,bar],color='#b55038',linestyle='--',linewidth=2)
 ax.axhline(1,color='gray',linewidth=.8);ax.set_ylabel('Verified throughput ratio');ax.set_title('CPU orchestration: speed after all verification')
 ax.set_ylim(0,max(2.3,max(values)*1.2));fig.tight_layout();fig.savefig(c.OUT/'throughput.png',dpi=160);plt.close(fig)
 return result
