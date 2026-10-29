import json
import subprocess
import sys
import pytest
from solvetide import assemble,Runtime,SolverConfig
from solvetide.artifacts import bundle
from solvetide.checker import verify,digest


def test_independent_export_and_tamper(tmp_path):
    s=assemble('elasticity',8,{'contrast':3.,'boundary_value':.2})
    runtime=Runtime(SolverConfig(strategy='direct'))
    f=runtime.solve(s);g=runtime.gradient(s,f)
    d=bundle(s,f,g,runtime.config)
    assert verify(d)['status']=='checked'
    d['equations']['x'][0]+=.1
    with pytest.raises(ValueError,match='digest'):verify(d)
    d['content_sha256']=digest(d)
    assert verify(d)['status']=='rejected'


def test_failed_no_bundle():
    s=assemble('heat',12,{'contrast':20.})
    r=Runtime(SolverConfig(maxiter=1))
    f=r.solve(s);g=r.gradient(s,f)
    with pytest.raises(ValueError):bundle(s,f,g,r.config)
