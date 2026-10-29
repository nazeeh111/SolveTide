"""Isolated process high-water RSS, separate from retained NumPy/SciPy bytes.

No tracemalloc instrumentation in timed confirmation runs. RSS includes Python,
imports, native libraries, assembly and temporary workspaces; it is not a hard
allocation limit. Baseline high-water is recorded before assembling a sequence.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter


def child(family,strategy):
    from dataclasses import replace
    from solvetide import Runtime,SolverConfig,assemble
    from solvetide.policy import RankPolicy
    from studies.run import trajectories
    peak=lambda:resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    baseline=peak();start=perf_counter()
    cases=next(cases for name,cases in trajectories('confirmation') if family in name)
    cfg=SolverConfig(strategy='fixed' if strategy=='separate' else strategy,memory_budget=6 if strategy=='separate' else 12)
    r=Runtime(cfg,RankPolicy.load(force=True) if strategy=='learned' else None)
    other=Runtime(cfg) if strategy=='separate' else None
    stored=0;failures=0
    for case in cases:
        s=assemble(family,40,case['parameters'],case['boundary'])
        f=r.solve(s)
        if other and f.accepted:
            _,dj=s.objective(f.x,.12 if family=='heat' else .04)
            accepted=other.solve(s,rhs=dj,role='adjoint').accepted
        else:accepted=r.gradient(s,f,target=.12 if family=='heat' else .04).accepted
        failures+=not accepted
        stored=max(stored,r.state()['memory_bytes']+(other.state()['memory_bytes'] if other else 0))
    return {'family':family,'strategy':strategy,'n':40,'basis_budget':12,'baseline_process_peak_bytes':baseline,
            'process_peak_rss_bytes':peak(),'max_retained_storage_bytes':stored,'failures':failures,'seconds':perf_counter()-start}


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--child':
        print(json.dumps(child(sys.argv[2],sys.argv[3])));return
    rows=[]
    for family in ['heat','elasticity']:
        for strategy in ['direct','warm','gcrot','fixed','joint','learned','amg_reuse','separate']:
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            p=subprocess.run([sys.executable,'-m','studies.memory','--child',family,strategy],capture_output=True,text=True,env=env,timeout=180)
            if p.returncode:raise RuntimeError(p.stderr)
            rows.append(json.loads(p.stdout));print(f'{family} {strategy} RSS={rows[-1]["process_peak_rss_bytes"]}',flush=True)
    path=Path(__file__).resolve().parents[1]/'experiments/memory.json'
    path.write_text(json.dumps({'schema_version':1,'platform':platform.platform(),'runs':rows,
                                'limits':'Separate process high-water RSS, one run per strategy and family; includes imports/assembly/native workspaces, no hard cap. Fixed12 retained vectors matched; total allocated memory differs and is measured, not equated.'},indent=2)+'\n')
if __name__=='__main__':main()
