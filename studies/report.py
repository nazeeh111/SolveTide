"""Make tables/plots from frozen executed reports; no numerical or model tuning."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
confirmation=json.loads((ROOT/'experiments/confirmation.json').read_text())
optimization=json.loads((ROOT/'experiments/optimization.json').read_text())
memory=json.loads((ROOT/'experiments/memory.json').read_text())
policy=json.loads((ROOT/'solvetide/data/policy.json').read_text())
rows=confirmation['summary']
lines=['# Executed measurements','',
       'Local CPU research results, three chronological repeats per held-out trajectory. Timings include assembly, feature/model/basis work, classical setup/solves and residual verification. Independent direct-reference checks are recorded outside those timings. Scheduler load varied; these finite examples establish no universal acceleration or latency guarantee.','',
       'The trained paired quota/refresh model is '+('enabled' if policy['enabled'] else '**disabled**')+' by development validation. The learned rows below explicitly force that frozen model for measurement; deployed CLI/editor use ordinary warm starts when disabled. Confirmation does not enable or tune it.','',
       '| Family | Method | Median sequence(s) | Failed steps /120 | Retained MiB max | Forward error max | Gradient error max |',
       '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for family in ['heat','elasticity']:
    for r in sorted([r for r in rows if r['family']==family],key=lambda r:r['median_sequence_seconds']):
        refs=[step['reference_errors'] for run in confirmation['runs'] if run['family']==family and run['strategy']==r['strategy'] for step in run['steps'] if step['reference_errors']]
        f=max((v['forward_relative_l2'] for v in refs if v['forward_relative_l2'] is not None),default=0)
        g=max((v['gradient_relative_l2'] for v in refs if v['gradient_relative_l2'] is not None),default=0)
        lines.append(f"| {family} | {r['strategy']} | {r['median_sequence_seconds']:.4f} | {r['failures']} | {r['max_retained_storage_bytes']/2**20:.2f} | {f:.2g} | {g:.2g} |")
lines+=['','All retained basis controls have total rank 12, including separate 6+6 banks. Factor/AMG storage differs and is reported honestly. A failed step is retained in totals; a cheaper failed run is not a successful speedup. `gcrot` iterations are outer iterations; matrix-vector counts expose inner work. AMG includes symmetric smoothing and elasticity rigid-body candidates. The implemented memory correction is an initial projection, not full augmented CG.','',
        '## Complete inverse design','',
        'Identical target, start, bounds and stopping criteria per method; target generation and final direct verification included. Reported successful objectives below come from final direct solves. Failed attempts retain their consumed time and last accepted state in raw traces; they are not successful optimization speedups.','',
        '| Family | Method | Successful runs /3 | Median observed time(s) | Max verified objective among successes | Median evaluations |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
for family in ['heat','elasticity']:
    for method in ['direct','warm','fixed','joint','learned','amg_reuse']:
        rs=[r for r in optimization['runs'] if r['family']==family and r['strategy']==method]
        final_objectives=[r['direct_checked_objective'] for r in rs if r['success'] and r['direct_checked_objective'] is not None]
        objective_text=f'{max(final_objectives):.3g}' if final_objectives else 'unavailable'
        lines.append(f"| {family} | {method} | {sum(r['success'] for r in rs)} | {np.median([r['seconds'] for r in rs]):.4f} | {objective_text} | {np.median([r['evaluations'] for r in rs]):.0f} |")
lines+=['','## Isolated process memory','',
        'One separate process per family/method, n40. Process peak RSS includes Python, imports, native libraries, assembly and temporary workspaces; retained storage is a separate metric. These are measurements, not hard OS allocation limits.','',
        '| Family | Method | Process peak RSS(MiB) | Retained storage(MiB) |',
        '| --- | --- | ---: | ---: |']
for r in memory['runs']:lines.append(f"| {r['family']} | {r['strategy']} | {r['process_peak_rss_bytes']/2**20:.1f} | {r['max_retained_storage_bytes']/2**20:.2f} |")
latency=json.loads((ROOT/'experiments/editor-latency.json').read_text())
lines+=['','## Measured editor response tiers','','Actual changed-input loopback requests include computation, serialization, transport and parse. Browser paint is excluded. Three local observations per row; maximum is observed, not a latency bound. No universal real-time claim.','','| Family | Edge divisions | Median complete response(s) | Max observed(s) | Refused /3 |','| --- | ---: | ---: | ---: | ---: |']
for r in latency['summary']:lines.append(f"| {r['family']} | {r['n']} | {r['median_seconds']:.4f} | {r['max_observed_seconds']:.4f} | {r['failures']} |")
lines+=['','Raw [confirmation](../experiments/confirmation.json), [optimization](../experiments/optimization.json), [memory](../experiments/memory.json), [training](../experiments/training.json), and [independent numerical validation](../experiments/numerical-validation.json) retain statuses and traces. Source hashes, protocol and model hashes bind each study.','',
        'PCGBandit, G-RANS and SOLVAX were studied as references and were not benchmarked here. No superiority over them is claimed. Fixed chronological order does not use SKR offline sorting. Small PDE examples, one machine, original synthetic data and a simple nodal objective limit generalization. Solver convergence does not certify discretization or model truth.']
(ROOT/'docs/measurements.md').write_text('\n'.join(lines)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f5ef','axes.facecolor':'#f6f5ef','text.color':'#253d3a','axes.labelcolor':'#253d3a','xtick.color':'#253d3a','ytick.color':'#253d3a'})
fig,axes=plt.subplots(1,2,figsize=(12,6))
fig.subplots_adjust(left=.09,right=.99,bottom=.19,top=.84,wspace=.35)
for ax,family in zip(axes,['heat','elasticity']):
    rs=sorted([r for r in rows if r['family']==family],key=lambda r:r['median_sequence_seconds'])
    y=np.arange(len(rs));colors=['#cf714d' if r['strategy']=='learned' else '#32665e' for r in rs]
    ax.barh(y,[r['median_sequence_seconds'] for r in rs],color=colors,height=.65)
    ax.set_yticks(y,[r['strategy']+(' *' if r['failures'] else '') for r in rs]);ax.invert_yaxis()
    ax.set_xlabel('Median complete sequence, seconds');ax.set_title(family.capitalize());ax.grid(axis='x',alpha=.15)
fig.suptitle('Held-out chronological edits · all setup and verification costs',fontsize=15)
fig.text(.5,.03,'* Failures retained, not successful speedups. Learned forced for study; development gate controls deployment.',ha='center',fontsize=9)
fig.savefig(ROOT/'docs/benchmarks.next.png',dpi=160)
(ROOT/'docs/benchmarks.next.png').replace(ROOT/'docs/benchmarks.png')
plt.close(fig)
print('\n'.join(lines[:12]))
