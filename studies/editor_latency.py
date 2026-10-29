"""Real changed-input editor HTTP round trips; browser paint is not measured."""
import argparse
from hashlib import sha256
import http.client
import json
from pathlib import Path
from time import perf_counter
import statistics

p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);a=p.parse_args()
rows=[]
for family in ('heat','elasticity'):
    for n in (20,32,48):
        for repeat in range(3):
            payload={'family':family,'n':n,'boundary':'left','strategy':'warm','maxiter':2000,
                     'parameters':{'material':1.,'contrast':1.+.05*repeat,'source':1.+.02*repeat,'boundary_value':0.}}
            body=json.dumps(payload).encode();t=perf_counter()
            connection=http.client.HTTPConnection('127.0.0.1',a.port,timeout=40)
            connection.request('POST','/api/solve',body,{'Content-Type':'application/json'})
            response=connection.getresponse();raw=response.read();result=json.loads(raw);connection.close()
            seconds=perf_counter()-t
            if response.status!=200:raise RuntimeError(result)
            rows.append({'family':family,'n':n,'repeat':repeat,'http_roundtrip_seconds':seconds,
                         'server_compute_seconds':result['total_seconds'],'response_bytes':len(raw),
                         'status':result['status'],'field_present':result.get('field') is not None,
                         'identities':result['identities'],'input':payload})
summary=[]
for family in ('heat','elasticity'):
    for n in (20,32,48):
        selected=[r for r in rows if r['family']==family and r['n']==n]
        seconds=[r['http_roundtrip_seconds'] for r in selected]
        summary.append({'family':family,'n':n,'samples':3,'median_seconds':statistics.median(seconds),
                        'max_observed_seconds':max(seconds),'failures':sum(r['status']!='converged' for r in selected)})
root=Path(__file__).resolve().parents[1]
doc={'schema_version':1,'summary':summary,'requests':rows,
     'source_sha256':{name:sha256((root/name).read_bytes()).hexdigest() for name in ['solvetide/server.py','solvetide/runtime.py','solvetide/assembly.py']},
     'scope':'Actual loopback POST, changed source/contrast, assembly, forward+adjoint checks, JSON serialization, response transport and JSON parse. Browser paint is excluded. Three local observations per mesh/family under concurrent host load; no hard latency, real-time or universal interactive guarantee.'}
(root/'experiments/editor-latency.json').write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps(summary,indent=2))
