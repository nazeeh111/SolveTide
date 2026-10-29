"""Installed command line interface."""
from __future__ import annotations
import argparse
import json
import sys
from . import __version__, Runtime, SolverConfig, assemble
from .artifacts import bundle,save
from .checker import load,verify


def main(argv=None):
    p=argparse.ArgumentParser(prog='solvetide',description='Checked sparse simulations and discrete adjoint sensitivities.')
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('solve',help='assemble, solve, differentiate and optionally export original equations')
    s.add_argument('--family',choices=['heat','elasticity'],default='heat');s.add_argument('--n',type=int,default=20)
    s.add_argument('--boundary',choices=['left','right','both','all'],default='left')
    for name,default in [('material',1.),('source',1.),('contrast',1.),('boundary-value',0.)]:s.add_argument('--'+name,type=float,default=default)
    s.add_argument('--target',type=float,default=0.)
    s.add_argument('--strategy',choices=['cold','warm','fixed','joint','drift','learned','direct','gcrot','amg','amg_reuse'],default='warm')
    s.add_argument('--maxiter',type=int,default=2000);s.add_argument('--output')
    v=sub.add_parser('verify',help='independently recompute exported equation residuals');v.add_argument('artifact')
    u=sub.add_parser('studio',help='open the loopback field editor');u.add_argument('--port',type=int,default=8765)
    a=p.parse_args(argv)
    try:
        if a.command=='studio':
            from .server import serve
            serve(port=a.port);return 0
        if a.command=='verify':
            report=verify(load(a.artifact));print(json.dumps(report,indent=2));return 0 if report['status']=='checked' else 3
        policy=None;effective=a.strategy;gate=None
        if a.strategy=='learned':
            from .policy import RankPolicy
            policy=RankPolicy.load()
            if not policy.enabled:
                effective='warm';policy=None;gate='Development validation disabled learning; ordinary warm start avoids its overhead.'
        config=SolverConfig(strategy=effective,maxiter=a.maxiter)
        runtime=Runtime(config,policy)
        system=assemble(a.family,a.n,{'material':a.material,'source':a.source,'contrast':a.contrast,'boundary_value':a.boundary_value},a.boundary)
        f=runtime.solve(system);g=runtime.gradient(system,f,target=a.target)
        report={'status':g.status,'requested_strategy':a.strategy,'effective_strategy':effective,'policy_gate':gate,
                'forward':f.summary(),'objective':g.objective,'gradient':g.gradient,'adjoint':g.adjoint.summary() if g.adjoint else None}
        if g.accepted and a.output:
            save(a.output,bundle(system,f,g,config,a.target));report['artifact']=a.output
        print(json.dumps(report,indent=2,allow_nan=False));return 0 if g.accepted else 3
    except (ValueError,TypeError,OSError) as e:
        print(json.dumps({'status':'invalid','message':str(e)}),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
