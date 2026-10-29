"""Real HTTP workflow checks, without a browser or private files."""
import json
import math
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from solvetide.server import make_server


@pytest.fixture
def editor():
    server = make_server(port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    thread.join(3)


def request(base, path, data=None, headers=None):
    payload = None if data is None else json.dumps(data).encode()
    req = Request(base + path, data=payload,
                  headers={'Content-Type': 'application/json', **(headers or {})})
    with urlopen(req, timeout=20) as result:
        return result.read()


def test_loopback_only():
    with pytest.raises(ValueError, match='loopback'):
        make_server('0.0.0.0', 0)


def test_actual_edit_export_invalidation_failure_and_fallback(editor):
    assert b'SOLVETIDE' in request(editor, '/')
    with pytest.raises(HTTPError) as error:
        request(editor, '/api/export')
    assert error.value.code == 409
    first = json.loads(request(editor, '/api/solve', {'n': 8}))
    assert first['field'] is not None
    assert first['input']['strategy'] == 'warm'
    assert first['gradient'] is not None
    changed = json.loads(request(editor, '/api/solve', {'n': 8, 'parameters': {'material': 1.05}}))
    assert first['identities']['operator'] != changed['identities']['operator']
    assert first['field'] != changed['field']
    exported = json.loads(request(editor, '/api/export'))
    assert exported['revision'] == changed['revision']
    # A separate standard-library row accumulation checks the exported equations.
    equations = exported['system']
    rows = []
    for row, b in enumerate(equations['b']):
        ax = math.fsum(equations['data'][j] * equations['x'][equations['indices'][j]]
                       for j in range(equations['indptr'][row], equations['indptr'][row + 1]))
        rows.append(b - ax)
    assert math.sqrt(math.fsum(r * r for r in rows)) < 1e-8
    remesh = json.loads(request(editor, '/api/solve', {'n': 10}))
    assert remesh['identities']['mapping'] != first['identities']['mapping']
    assert remesh['field'] is not None
    failed = json.loads(request(editor, '/api/solve', {'n': 16, 'maxiter': 1, 'strategy': 'cold'}))
    assert failed['field'] is None
    with pytest.raises(HTTPError) as error:
        request(editor, '/api/export')
    assert error.value.code == 409
    fallback = json.loads(request(editor, '/api/solve', {'n': 16, 'strategy': 'direct'}))
    assert fallback['field'] is not None
    assert fallback['primal']['relative_residual'] < 1e-8


def test_elasticity_and_rejection(editor):
    result = json.loads(request(editor, '/api/solve', {'family': 'elasticity', 'n': 6}))
    assert result['field'] is not None
    assert len(result['field']) == 2 * len(result['coordinates'][0])
    for data in ({'n': 1000}, {'maxiter': 0}, {'parameters': {'material': float('nan')}}, {'family': 'other'}):
        with pytest.raises(HTTPError) as error:
            request(editor, '/api/solve', data)
        assert error.value.code == 400
    with pytest.raises(HTTPError) as error:
        request(editor, '/api/reset', {}, {'Origin': 'https://example.com'})
    assert error.value.code == 403
    with pytest.raises(HTTPError) as error:
        request(editor, '/api/export')
    assert error.value.code == 409
    request(editor, '/api/reset', {})
    assert not json.loads(request(editor, '/api/state'))['has_export']


def test_learned_selection_honors_development_gate(editor):
    result = json.loads(request(editor, '/api/solve', {'n': 8, 'strategy': 'learned'}))
    assert result['field'] is not None
    assert result['requested_strategy'] == 'learned'
    assert isinstance(result['policy']['policy_disabled'], bool)
    if not result['policy']['development_enabled']:
        assert result['policy']['policy_disabled']
        assert 'fallback' in result['policy']['note'].lower()
        assert result['primal']['reused_rank'] == 0
        assert result['effective_strategy'] == 'warm'
        assert result['runtime_state']['strategy'] == 'warm'
    assert 'policy_action' in result['primal']


def test_disabled_model_uses_actual_warm_runtime(monkeypatch):
    from types import SimpleNamespace
    from solvetide import server
    # A parsed but development-disabled model must never run its planner.
    monkeypatch.setattr(server.RankPolicy, 'load', lambda: SimpleNamespace(enabled=False))
    editor = server.EditorState()
    result = editor.solve({'n': 8, 'strategy': 'learned'})
    assert result['field'] is not None
    assert result['policy']['available']
    assert result['policy']['policy_disabled']
    assert result['effective_strategy'] == 'warm'
    assert editor.runtime.config.strategy == 'warm'
    assert editor.runtime.policy is None
