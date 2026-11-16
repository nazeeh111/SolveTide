"""Standalone standard-library check of exported ORIGINAL discrete equations.

This checks stored float64 equations and adjoint contraction, not the continuous
model, solution-error enclosure, or a signature. Run this file without installing
SolveTide: python checker.py result.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(document):
    body={k:v for k,v in document.items() if k!='content_sha256'}
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite(value):
    if type(value) not in (int,float):raise ValueError('finite real numbers required')
    try: result=float(value)
    except OverflowError: raise ValueError('finite real numbers required') from None
    if not math.isfinite(result):raise ValueError('finite real numbers required')
    return result


def csr(matrix,n):
    if not isinstance(matrix,dict):raise ValueError('sparse matrix must be a map')
    shape=matrix.get('shape')
    if not isinstance(shape,list) or len(shape)!=2 or any(type(v)!=int for v in shape) or shape!=[n,n]:raise ValueError('sparse shape mismatch')
    ip,ix,data=matrix['indptr'],matrix['indices'],matrix['data']
    if not all(isinstance(v,list) for v in (ip,ix,data)):raise ValueError('sparse arrays required')
    if len(ip)!=n+1 or ip[0]!=0 or ip[-1]!=len(ix) or len(ix)!=len(data) or len(ix)>2000000:raise ValueError('invalid sparse shape')
    if any(type(v)!=int for v in ip+ix) or any(a>b for a,b in zip(ip,ip[1:])):raise ValueError('invalid sparse indices')
    data=[finite(v) for v in data]
    for i in range(n):
        cols=ix[ip[i]:ip[i+1]]
        if any(j<0 or j>=n for j in cols) or any(a>=b for a,b in zip(cols,cols[1:])):raise ValueError('canonical sparse columns required')
    return ip,ix,data


def residual(matrix,x,b,*,transpose=False):
    n=len(b)
    ip,ix,data=csr(matrix,n)
    if len(x)!=n:raise ValueError('field shape mismatch')
    x=[finite(v) for v in x]; b=[finite(v) for v in b]
    if transpose:
        rows=[[] for _ in range(n)]
        sums=[[] for _ in range(n)]
        for i in range(n):
            for k in range(ip[i],ip[i+1]):
                rows[ix[k]].append(data[k]*x[i]); sums[ix[k]].append(abs(data[k]))
        ax=[math.fsum(row) for row in rows]
        anorm=max(math.fsum(row) for row in sums)
    else:
        ax=[math.fsum(data[k]*x[ix[k]] for k in range(ip[i],ip[i+1])) for i in range(n)]
        anorm=max(math.fsum(abs(data[k]) for k in range(ip[i],ip[i+1])) for i in range(n))
    ax=[finite(v) for v in ax]; anorm=finite(anorm)
    r=[finite(v-a) for v,a in zip(b,ax)]
    rnorm=finite(math.hypot(*r)); bnorm=finite(math.hypot(*b))
    denominator=finite(anorm*max(map(abs,x),default=0)+max(map(abs,b),default=0))
    return {'residual_norm':rnorm,'rhs_norm':bnorm,'backward_error':max(map(abs,r),default=0)/denominator if denominator else 0.0}


def verify(document):
    if not isinstance(document,dict):raise ValueError('artifact must be a map')
    if document.get('schema_version')!=1 or document.get('artifact')!='solvetide-discrete-result':raise ValueError('unsupported artifact')
    if document.get('content_sha256')!=digest(document):raise ValueError('content digest mismatch')
    eq=document['equations']
    if not isinstance(eq,dict):raise ValueError('equations must be a map')
    b=eq['b']; x=eq['x']; adjoint=eq['adjoint']; djdx=eq['dJdx']
    if not all(isinstance(v,list) for v in (b,x,adjoint,djdx)):raise ValueError('equation vectors must be arrays')
    n=len(b)
    if not 1<=n<=40000:raise ValueError('dimension limit exceeded')
    if any(len(v)!=n for v in (x,adjoint,djdx)):raise ValueError('equation vector shape mismatch')
    b=[finite(v) for v in b]; x=[finite(v) for v in x]
    adjoint=[finite(v) for v in adjoint]; djdx=[finite(v) for v in djdx]
    derivatives=eq['derivatives']; reported=document['gradient']
    if not isinstance(derivatives,dict) or not isinstance(reported,dict):raise ValueError('derivatives and gradient must be maps')
    if any(not isinstance(k,str) for k in derivatives) or any(not isinstance(k,str) for k in reported):raise ValueError('gradient keys must be strings')
    if derivatives.keys()!=reported.keys():raise ValueError('derivative and gradient keys must match exactly')
    reported={k:finite(v) for k,v in reported.items()}
    rtol=finite(document['acceptance']['rtol']); atol=finite(document['acceptance']['atol'])
    if min(rtol,atol)<0 or max(rtol,atol)<=0:raise ValueError('invalid tolerances')
    f=residual(eq['A'],x,b)
    a=residual(eq['A'],adjoint,djdx,transpose=True)
    # Extra rounding allowance only in checker comparison: fsum may expose a
    # final rounding difference. Never describe a failed runtime solve as valid.
    def accepted(r):return r['residual_norm']<=finite(max(atol,finite(rtol*r['rhs_norm']))*(1+1e-6))
    gradient={}
    for name,d in derivatives.items():
        if not isinstance(d,dict):raise ValueError('derivative must be a map')
        ip,ix,data=csr(d['A'],n)
        if not isinstance(d['b'],list):raise ValueError('derivative vector must be an array')
        db=[finite(v) for v in d['b']]
        if len(db)!=n:raise ValueError('derivative shape mismatch')
        v=[finite(db[i]-math.fsum(data[k]*x[ix[k]] for k in range(ip[i],ip[i+1]))) for i in range(n)]
        gradient[name]=finite(math.fsum(l*z for l,z in zip(adjoint,v)))
    gradients_ok=all(abs(v-reported[k])<=1e-10*max(1,abs(v)) for k,v in gradient.items())
    ok=accepted(f) and accepted(a) and gradients_ok
    return {'status':'checked' if ok else 'rejected','forward':f,'adjoint':a,'gradient_algebra_matches':gradients_ok,
            'scope':'Stored discrete residuals and implicit adjoint algebra only; no solution-error, discretization or physical-model certificate.'}


def load(path):
    path=Path(path)
    if path.stat().st_size>32*1024*1024:raise ValueError('artifact exceeds32MiB')
    def bad(value):raise ValueError('nonfinite JSON constant')
    return json.loads(path.read_text(),parse_constant=bad)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('artifact')
    a=p.parse_args(argv)
    try:
        report=verify(load(a.artifact));print(json.dumps(report,indent=2,allow_nan=False))
        return 0 if report['status']=='checked' else 3
    except (ValueError,KeyError,TypeError,OSError,OverflowError) as e:
        print(json.dumps({'status':'invalid','message':str(e)}));return 2

if __name__=='__main__':raise SystemExit(main())
