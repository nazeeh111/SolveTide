"""A separate application: identify material coefficients from a target field."""
from __future__ import annotations
import argparse
import json
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from solvetide import Runtime,SolverConfig,assemble
from solvetide.policy import RankPolicy


class _SolverRefused(RuntimeError):
    """An optimization evaluation produced no accepted gradient."""


def optimize(family='heat',n=24,strategy='warm',maxiter=40,*,force_policy=False):
    started=perf_counter()
    truth=assemble(family,n,{'material':1.8,'contrast':4.2},boundary='right')
    target=spsolve(truth.A,truth.b)
    policy=RankPolicy.load(force=force_policy) if strategy=='learned' else None
    effective=strategy
    gate=None
    if policy is not None and not policy.enabled:
        effective='warm';policy=None
        gate='Development gate disabled learning; ordinary warm start without model/bank overhead.'
    runtime=Runtime(SolverConfig(strategy=effective,rtol=1e-9,atol=1e-11,memory_budget=12,max_seconds=15),policy)
    history=[]
    base={'schema_version':1,'application':'separately installed solvetide-design-example','family':family,'n':n,
          'strategy':strategy,'effective_strategy':effective,'policy_gate':gate,'forced_research_policy':bool(force_policy and strategy=='learned'),
          'initial_parameters':[.65,.6],'truth':[1.8,4.2],'stopping':{'maxiter':maxiter,'gtol':1e-8,'ftol':1e-12}}
    def evaluate(logp):
        p=dict(zip(('material','contrast'),np.exp(logp)))
        t=perf_counter();s=assemble(family,n,p,boundary='right')
        f=runtime.solve(s);g=runtime.gradient(s,f,target)
        row={'parameters':{k:float(v) for k,v in p.items()},'status':g.status,'objective':g.objective,
             'gradient_norm':None,'primal':f.summary(),'adjoint':g.adjoint.summary() if g.adjoint else None}
        if not g.accepted:
            row['seconds']=perf_counter()-t;history.append(row)
            raise _SolverRefused(f'Optimization stopped: {f.status}/{g.status}; no current gradient')
        grad=np.array([g.gradient[k]*p[k] for k in ('material','contrast')])
        row.update(gradient_norm=float(np.linalg.norm(grad)),seconds=perf_counter()-t)
        history.append(row)
        return g.objective,grad
    try:
        result=minimize(evaluate,np.log([.65,.6]),jac=True,method='L-BFGS-B',
                        bounds=[(np.log(.2),np.log(6)),(np.log(.1),np.log(15))],
                        options={'maxiter':maxiter,'gtol':1e-8,'ftol':1e-12,'maxls':25})
    except _SolverRefused as error:
        accepted=[h for h in history if h['status']=='converged']
        return {**base,'success':False,'status':'solver_failed','message':str(error),'iterations':None,
                'evaluations':len(history),'parameters':None,'objective':None,'direct_checked_objective':None,
                'failed_parameters':history[-1]['parameters'],
                'last_accepted_parameters':accepted[-1]['parameters'] if accepted else None,
                'last_accepted_objective':accepted[-1]['objective'] if accepted else None,
                'seconds':perf_counter()-started,'history':history,'memory':runtime.state()}
    final=assemble(family,n,dict(zip(('material','contrast'),np.exp(result.x))),boundary='right')
    checked=spsolve(final.A,final.b)
    checked_objective=final.objective(checked,target)[0]
    return {**base,'success':bool(result.success),'status':'converged' if result.success else 'optimizer_stopped',
            'message':str(result.message),'iterations':int(result.nit),'evaluations':int(result.nfev),
            'parameters':np.exp(result.x).tolist(),'objective':float(result.fun),'direct_checked_objective':checked_objective,
            'seconds':perf_counter()-started,'history':history,'memory':runtime.state()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=['heat','elasticity'],default='heat');p.add_argument('--n',type=int,default=24)
    p.add_argument('--strategy',choices=['cold','warm','fixed','joint','drift','learned','direct','gcrot','amg','amg_reuse'],default='warm')
    p.add_argument('--maxiter',type=int,default=40);p.add_argument('--output')
    a=p.parse_args()
    try:
        result=optimize(a.family,a.n,a.strategy,a.maxiter)
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        if a.output:
            with open(a.output,'x') as f:f.write(text)
        else:print(text)
        return 0 if result['success'] else 3
    except (ValueError,OSError) as error:
        print(json.dumps({'success':False,'status':'invalid_input','message':str(error)}));return 2
if __name__=='__main__':raise SystemExit(main())
