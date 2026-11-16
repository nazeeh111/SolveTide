"""Detached artifact structure and finite-value refusal at real solver outputs."""
import copy
import json
import subprocess
import sys

import pytest
from solvetide import Runtime, SolverConfig, assemble
from solvetide.artifacts import bundle
from solvetide.checker import digest, verify

@pytest.fixture
def artifact():
    system = assemble('heat', 6, {'contrast': 2., 'boundary_value': .1})
    runtime = Runtime(SolverConfig(strategy='direct'))
    forward = runtime.solve(system)
    adjoint = runtime.gradient(system, forward, target=.2)
    return bundle(system, forward, adjoint, runtime.config, target=.2)

def reseal(doc):
    doc['content_sha256'] = digest(doc)
    return doc

@pytest.mark.parametrize('target', ['A', 'derivative_A'])
def test_declared_sparse_shape_must_match_equation_dimension(artifact, target):
    eq=artifact['equations']
    matrix=eq['A'] if target == 'A' else next(iter(eq['derivatives'].values()))['A']
    matrix['shape'][1] += 1
    with pytest.raises(ValueError, match='shape'):
        verify(reseal(artifact))

@pytest.mark.parametrize('vector', ['x', 'adjoint', 'dJdx'])
def test_equation_vectors_must_have_matching_length(artifact, vector):
    artifact['equations'][vector].pop()
    with pytest.raises(ValueError, match='shape'):
        verify(reseal(artifact))

def test_derivative_vector_length_must_match(artifact):
    next(iter(artifact['equations']['derivatives'].values()))['b'].pop()
    with pytest.raises(ValueError, match='shape'):
        verify(reseal(artifact))

@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_json_constants_refused_without_traceback(artifact, value, tmp_path):
    path=tmp_path/'bad.json'
    artifact['gradient'][next(iter(artifact['gradient']))]=value
    path.write_text(json.dumps(artifact))
    r=subprocess.run([sys.executable, '-m', 'solvetide', 'verify', str(path)],capture_output=True,text=True,timeout=15)
    assert r.returncode==2 and 'invalid' in r.stderr and 'Traceback' not in r.stderr

def test_missing_equations_key_produces_concise_invalid_cli(artifact, tmp_path):
    del artifact['equations']['dJdx']
    path=tmp_path/'bad.json'; path.write_text(json.dumps(reseal(artifact)))
    r=subprocess.run([sys.executable, '-m', 'solvetide', 'verify', str(path)],capture_output=True,text=True,timeout=15)
    assert r.returncode==2 and 'invalid' in r.stderr and 'Traceback' not in r.stderr

def test_finite_inputs_with_overflow_cannot_be_accepted(artifact):
    artifact['equations']['b']=[1e308]*len(artifact['equations']['b'])
    with pytest.raises((ValueError, OverflowError)):
        verify(reseal(artifact))


def test_top_level_nonmap_is_cleanly_invalid(tmp_path):
    path=tmp_path/'bad.json'; path.write_text('[]')
    r=subprocess.run([sys.executable, '-m', 'solvetide', 'verify', str(path)],capture_output=True,text=True,timeout=15)
    assert r.returncode==2 and 'artifact must be a map' in r.stderr and 'Traceback' not in r.stderr
