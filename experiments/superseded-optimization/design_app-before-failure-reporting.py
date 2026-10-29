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


def optimize(family='heat',n=24,strategy='warm',maxiter=40):
    started=perf_counter()
    truth=assemble(family,n,{'material':1.8,'contrast':4.2},boundary='right')
    target=spsolve(truth.A,truth.b)
    policy=RankPolicy.load(force=True) if strategy=='learned' else None
    runtime=Runtime(SolverConfig(strategy=strategy,rtol=1e-9,atol=1e-11,memory_budget=12,max_seconds=15),policy)
    history=[]
    def evaluate(logp):
        p=dict(zip(('material','contrast'),np.exp(logp)))
        t=perf_counter();s=assemble(family,n,p,boundary='right')
        f=runtime.solve(s);g=runtime.gradient(s,f,target)
        if not g.accepted:raise RuntimeError(f'Optimization stopped: {f.status}/{g.status}')
        grad=np.array([g.gradient[k]*p[k] for k in ('material','contrast')])
        history.append({'parameters':{k:float(v) for k,v in p.items()},'objective':g.objective,
                        'gradient_norm':float(np.linalg.norm(grad)),'seconds':perf_counter()-t,
                        'primal':f.summary(),'adjoint':g.adjoint.summary()})
        return g.objective,grad
    result=minimize(evaluate,np.log([.65,.6]),jac=True,method='L-BFGS-B',
                    bounds=[(np.log(.2),np.log(6)),(np.log(.1),np.log(15))],
                    options={'maxiter':maxiter,'gtol':1e-8,'ftol':1e-12,'maxls':25})
    final=assemble(family,n,dict(zip(('material','contrast'),np.exp(result.x))),boundary='right')
    checked=spsolve(final.A,final.b)
    checked_objective=final.objective(checked,target)[0]
    return {'schema_version':1,'application':'separately installed solvetide-design-example','family':family,'n':n,
            'strategy':strategy,'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),
            'evaluations':int(result.nfev),'initial_parameters':[.65,.6],'truth':[1.8,4.2],'parameters':np.exp(result.x).tolist(),
            'objective':float(result.fun),'direct_checked_objective':checked_objective,'seconds':perf_counter()-started,
            'stopping':{'maxiter':maxiter,'gtol':1e-8,'ftol':1e-12},'history':history,'memory':runtime.state()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=['heat','elasticity'],default='heat');p.add_argument('--n',type=int,default=24)
    p.add_argument('--strategy',default='warm');p.add_argument('--maxiter',type=int,default=40);p.add_argument('--output')
    a=p.parse_args();result=optimize(a.family,a.n,a.strategy,a.maxiter)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if a.output:
        with open(a.output,'x') as f:f.write(text)
    else:print(text)
    return 0 if result['success'] else 1
if __name__=='__main__':raise SystemExit(main())
