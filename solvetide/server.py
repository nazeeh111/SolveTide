"""Loopback-only field editor. Every successful export comes from a fresh solve."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from threading import Event, Lock
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .assembly import assemble
from .runtime import Runtime, SolverConfig
from .policy import RankPolicy


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _report(result: Any) -> dict[str, Any]:
    names = ('status', 'iterations', 'residual_norm', 'relative_residual',
             'backward_error', 'elapsed_seconds', 'reused_rank', 'cache_event',
             'timings', 'memory_bytes', 'message', 'conditioning', 'policy_features', 'policy_action')
    return _plain({key: getattr(result, key, None) for key in names})


class EditorState:
    def __init__(self) -> None:
        self.solve_lock = Lock()
        self.state_lock = Lock()
        self.cancel = Event()
        self.revision = 0
        self.latest: dict[str, Any] | None = None
        self.runtime = Runtime(SolverConfig(strategy='warm'))
        self.strategy = 'warm'
        self.effective_strategy = 'warm'
        self.policy_available = False
        self.policy_enabled = False
        self.policy_note = 'Warm starts are the default. No general acceleration has been established.'

    def invalidate(self) -> tuple[int, Event]:
        with self.state_lock:
            self.cancel.set()
            self.cancel = Event()
            self.revision += 1
            self.latest = None
            return self.revision, self.cancel

    def solve(self, payload: dict[str, Any]) -> dict[str, Any]:
        revision, cancelled = self.invalidate()
        allowed = {'family', 'n', 'boundary', 'parameters', 'strategy', 'maxiter'}
        if set(payload) - allowed:
            raise ValueError('Unknown request fields.')
        family = payload.get('family', 'heat')
        if family not in ('heat', 'elasticity'):
            raise ValueError('Choose heat or elasticity.')
        n = payload.get('n', 20)
        if type(n) is not int or not 4 <= n <= 48:
            raise ValueError('Editor mesh resolution must be an integer from 4 to 48.')
        strategy = payload.get('strategy', 'warm')
        if strategy not in ('cold', 'warm', 'fixed', 'joint', 'drift', 'learned', 'direct'):
            raise ValueError('Unsupported editor strategy.')
        maxiter = payload.get('maxiter', 2000)
        if type(maxiter) is not int or not 1 <= maxiter <= 3000:
            raise ValueError('Iteration limit must be between 1 and 3000.')
        params = payload.get('parameters', {})
        if not isinstance(params, dict) or len(params) > 12:
            raise ValueError('parameters must be a small object of finite numbers.')
        for key, val in params.items():
            if not isinstance(key, str) or type(val) not in (int, float) or not math.isfinite(val):
                raise ValueError('Parameters must be finite numbers.')
        started = perf_counter()
        with self.solve_lock:
            if cancelled.is_set():
                return {'status': 'cancelled', 'revision': revision, 'field': None}
            if strategy != self.strategy:
                policy = None
                self.policy_note = 'No learned policy is used for this strategy.'
                if strategy == 'learned':
                    try:
                        policy = RankPolicy.load()
                        self.policy_note = ('Development gate enabled learned proposals; each solve still requires classical convergence.'
                                            if policy.enabled else
                                            'Development gate disabled learned corrections. Warm-start fallback is active.')
                    except (OSError, ValueError, TypeError, KeyError):
                        self.policy_note = 'A compatible trained policy is unavailable. Warm-start fallback is active.'
                self.policy_available = policy is not None
                self.policy_enabled = bool(getattr(policy, 'enabled', False))
                self.effective_strategy = 'warm' if strategy == 'learned' and not self.policy_enabled else strategy
                self.runtime = Runtime(SolverConfig(strategy=self.effective_strategy, maxiter=maxiter, max_seconds=10),
                                       policy=policy if self.policy_enabled else None)
                self.strategy = strategy
            else:
                self.runtime.config = SolverConfig(strategy=self.effective_strategy, maxiter=maxiter, max_seconds=10)
            system = assemble(family=family, n=n, boundary=payload.get('boundary', 'left'), parameters=params)
            forward = self.runtime.solve(system, cancel=cancelled.is_set)
            result: dict[str, Any] = {
                'schema_version': 1, 'revision': revision, 'status': forward.status,
                'requested_strategy': strategy, 'effective_strategy': self.effective_strategy,
                'input': {'family': family, 'n': n, 'boundary': system.boundary,
                          'parameters': system.parameters, 'strategy': self.effective_strategy, 'maxiter': maxiter},
                'identities': system.identity, 'primal': _report(forward), 'field': None,
                'assembly_seconds': system.assembly_seconds, 'runtime_state': self.runtime.state(),
            }
            if forward.accepted:
                sensitivity = self.runtime.gradient(system, forward, cancel=cancelled.is_set)
                result['adjoint'] = _report(sensitivity.adjoint) if sensitivity.adjoint is not None else None
                result['gradient'] = _plain(sensitivity.gradient)
                result['objective'] = sensitivity.objective
                result['status'] = sensitivity.status
                if sensitivity.gradient is not None:
                    full = system.expand(forward.x)
                    result['field'] = _plain(full)
                    result['coordinates'] = _plain(system.coordinates)
                    result['cells'] = _plain(system.cells)
                    result['system'] = {
                        'shape': list(system.A.shape), 'data': system.A.data.tolist(),
                        'indices': system.A.indices.tolist(), 'indptr': system.A.indptr.tolist(),
                        'b': system.b.tolist(), 'x': forward.x.tolist(),
                        'free': system.free.tolist(), 'fixed': system.fixed.tolist(),
                        'prescribed': system.prescribed.tolist(),
                    }
            result['runtime_state'] = self.runtime.state()
            result['policy'] = {
                'available': self.policy_available,
                'development_enabled': self.policy_enabled,
                'policy_disabled': strategy == 'learned' and (
                    not self.policy_enabled
                    or result['runtime_state'].get('policy_disabled', False)),
                'policy_action': result['runtime_state'].get('policy_action', {}),
                'requested_strategy': strategy, 'effective_strategy': self.effective_strategy,
                'note': self.policy_note,
            }
            if strategy == 'learned' and result['runtime_state'].get('policy_disabled', False):
                result['policy']['note'] = 'Runtime disabled learned corrections. Warm-start fallback is active.'
                result['effective_strategy'] = result['policy']['effective_strategy'] = 'warm'
            result['total_seconds'] = perf_counter() - started
            result = _plain(result)
            with self.state_lock:
                if revision != self.revision or cancelled.is_set():
                    return {'status': 'cancelled', 'revision': revision, 'field': None}
                if result['field'] is not None:
                    self.latest = result
            return result


class EditorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int]):
        self.editor = EditorState()
        super().__init__(address, EditorHandler)


class EditorHandler(BaseHTTPRequestHandler):
    server: EditorServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _trusted(self) -> bool:
        parsed = urlsplit('http://' + self.headers.get('Host', ''))
        if parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            return False
        try:
            if parsed.port != self.server.server_port:
                return False
        except ValueError:
            return False
        origin = self.headers.get('Origin')
        if origin:
            target = urlsplit(origin)
            if target.scheme != 'http' or target.netloc != self.headers.get('Host'):
                return False
        return True

    def _send(self, status: int, data: Any, content_type: str = 'application/json; charset=utf-8') -> None:
        body = data if isinstance(data, bytes) else json.dumps(_plain(data), allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._trusted():
            self._send(403, {'error': 'Only this loopback editor origin is allowed.'})
            return
        path = urlsplit(self.path).path
        if path == '/api/export':
            with self.server.editor.state_lock:
                latest = self.server.editor.latest
            if latest is None:
                self._send(409, {'error': 'No current accepted field. Compute a new field first.'})
            else:
                self._send(200, latest)
            return
        if path == '/api/state':
            state = self.server.editor
            with state.state_lock:
                self._send(200, {'revision': state.revision, 'has_export': state.latest is not None})
            return
        assets = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if path not in assets:
            self._send(404, {'error': 'Not found.'})
            return
        name, mime = assets[path]
        self._send(200, files('solvetide').joinpath('web', name).read_bytes(), mime)

    def do_POST(self) -> None:
        if not self._trusted():
            self._send(403, {'error': 'Only this loopback editor origin is allowed.'})
            return
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            self._send(415, {'error': 'Send application/json.'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise ValueError('Request must be at most 16 KiB.')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('Send a JSON object.')
            path = urlsplit(self.path).path
            if path == '/api/solve':
                self._send(200, self.server.editor.solve(data))
            elif path == '/api/cancel':
                self.server.editor.invalidate()
                self._send(200, {'status': 'cancelled'})
            elif path == '/api/reset':
                self.server.editor.invalidate()
                with self.server.editor.solve_lock:
                    self.server.editor.runtime.reset()
                self._send(200, {'status': 'reset'})
            else:
                self._send(404, {'error': 'Not found.'})
        except (ValueError, TypeError, KeyError) as error:
            self._send(400, {'error': str(error)})
        except Exception as error:
            # Invalid input or a numerical backend failure revokes the export.
            self.server.editor.invalidate()
            self._send(500, {'error': f'Computation failed ({type(error).__name__}). No field was accepted.'})


def make_server(host: str = '127.0.0.1', port: int = 8765) -> EditorServer:
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('The editor only binds to IPv4 loopback (127.0.0.1 or localhost).')
    return EditorServer((host, port))


def serve(host: str = '127.0.0.1', port: int = 8765) -> None:
    server = make_server(host, port)
    print(f'SolveTide editor: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.editor.cancel.set()
        server.server_close()
