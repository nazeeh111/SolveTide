"""Run against a separately installed environment, from a fresh empty directory."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--output',required=True)
a=p.parse_args();python=str(Path(a.python).absolute())
with tempfile.TemporaryDirectory(prefix='solvetide-recipient-') as tmp:
    def run(*args):
        r=subprocess.run([python,'-I',*args],cwd=tmp,capture_output=True,text=True)
        if r.returncode:raise RuntimeError(r.stdout+r.stderr)
        return r.stdout
    versions=json.loads(run('-c','import solvetide,json; print(json.dumps({"version":solvetide.__version__,"path":solvetide.__file__}))'))
    first=Path(tmp)/'first.json';second=Path(tmp)/'changed.json'
    run('-m','solvetide','solve','--family','heat','--n','12','--contrast','2','--output',str(first))
    run('-m','solvetide','solve','--family','heat','--n','12','--contrast','2.4','--output',str(second))
    one,two=json.loads(first.read_text()),json.loads(second.read_text())
    assert one['identity']['operator']!=two['identity']['operator'] and one['field']!=two['field']
    verified=json.loads(run('-m','solvetide','verify',str(second)))
    assert verified['status']=='checked'
    third=Path(tmp)/'elasticity.json'
    run('-m','solvetide','solve','--family','elasticity','--n','10','--boundary','both','--output',str(third))
    # -I script path deliberately points to installed example app, not checkout.
    optimization=json.loads(run('-c','import design_app,json; print(json.dumps(design_app.optimize(n=12)))'))
    assert optimization['success'] and optimization['direct_checked_objective']<1e-8
    gated=json.loads(run('-c','import design_app,json; print(json.dumps(design_app.optimize(n=8,strategy="learned")))'))
    assert gated['success'] and gated['effective_strategy']=='warm' and not gated['forced_research_policy']
    report={'schema_version':1,'installed':versions,'changed_input_recomputed':True,'independent_check':verified,
            'elasticity_accepted':True,'external_application_success':optimization['success'],'external_deployment_gate_respected':True,
            'external_objective':optimization['direct_checked_objective'],
            'isolation':'Fresh empty working directory, interpreter -I, installed wheel and separately installed application; no private files or saved caches.'}
Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
