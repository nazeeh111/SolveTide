"""Reopen an editor export and check stored forward CSR equations, stdlib only.

This is not the richer CLI checker: studio exports contain no adjoint vector or
matrix derivative records. It does not establish gradient or physical accuracy.
"""
import json,math,sys
from pathlib import Path

def check(path):
 d=json.loads(Path(path).read_text());s=d['system'];r=[]
 for i,rhs in enumerate(s['b']):
  ax=math.fsum(s['data'][j]*s['x'][s['indices'][j]] for j in range(s['indptr'][i],s['indptr'][i+1]))
  r.append(rhs-ax)
 norm=math.hypot(*r);rhs=math.hypot(*s['b'])
 return {'file':Path(path).name,'revision':d['revision'],'field_values':len(d['field']),'triangles':len(d['cells'][0]),'residual_norm':norm,'relative_residual':norm/rhs if rhs else None,'scope':'Recomputed stored forward CSR residual only; no adjoint/gradient, solution-error or physical-model certificate.'}
if __name__=='__main__':print(json.dumps([check(p) for p in sys.argv[1:]],indent=2,allow_nan=False))
