"""Real local sockets: interrupt headers/body/SSE and keep one serial A client."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread
from types import SimpleNamespace
import time

import pytest

from sketchik.agent import ChatClient, ProviderError
from sketchik.async_planning import OverlapRequest
from sketchik.request_cancellation import RequestCancelled, RequestCancellation


@pytest.mark.parametrize('phase', ['headers', 'stream', 'json'])
def test_cancel_blocked_exchange_preserves_partial_attempt_and_client_can_recover(tmp_path, phase):
    entered, release, progress = Event(), Event(), Event()
    requests, pauses = [], []
    partial = b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"partial rationale"}}]}\n\n'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if len(requests) == 1:
                if phase != 'headers':
                    self.send_response(200); self.end_headers()
                    self.wfile.write(partial if phase == 'stream' else b'{"choices":[')
                    self.wfile.flush()
                entered.set()
                release.wait(10)
                return
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'recovery'}}]}).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    client = ChatClient(f'http://127.0.0.1:{server.server_port}/v1', 'unit-model', 'unit-secret',
        stream=phase != 'json', timeout=8, transport_dir=tmp_path/'transport')
    client.request_pause_handler = lambda *args: pauses.append('must not retry intentional cancel')
    client.on_stream_progress = lambda event: progress.set()
    attempts = []
    from contextlib import contextmanager
    @contextmanager
    def observe(client):
        try:
            yield
        finally:
            attempts.append(dict(client.last_response_metadata))
    client.attempt_observer = observe
    robot = SimpleNamespace(on_step=None)
    request = OverlapRequest(client, [{'role': 'user', 'content': 'prepare'}], {}, tmp_path/'draft')
    try:
        with request.during_execution(robot):
            assert entered.wait(5)
            if phase == 'stream':
                assert progress.wait(3)
            cancelled = time.perf_counter()
            request.cancel('execution_failed')
        assert time.perf_counter()-cancelled < 1.
        assert isinstance(request.error, RequestCancelled) and not request.thread.is_alive()
        assert client.request_cancellation is None and robot.on_step is None
        assert not pauses and len(requests) == 1
        assert attempts[0]['status'] == 'cancelled'
        assert not attempts[0]['upstream_cancellation_confirmed']
        if phase == 'stream':
            assert 'partial rationale' in attempts[0]['message']['reasoning_content']
            assert next((tmp_path/'transport'/'streams').glob('*.sse')).read_bytes() == partial
        # The obsolete worker has exited before the same A starts recovery.
        recovered = client.complete([{'role': 'user', 'content': 'actual stopped state'}])
        assert recovered['message']['content'] == 'recovery' and len(requests) == 2
        assert len(attempts) == 2 and not pauses
        records = [json.loads(line) for line in (tmp_path/'transport'/'responses.jsonl').read_text().splitlines()]
        assert len(records) == 2 and records[0]['status'] == 'cancelled'
    finally:
        release.set(); server.shutdown(); server.server_close()


def test_cancel_before_connect_never_opens_a_socket():
    token = RequestCancellation(); token.cancel('execution_failed')
    with pytest.raises(RequestCancelled):
        token.create_connection(('127.0.0.1', 1), timeout=1)


@pytest.mark.parametrize('error', [ValueError('program fault'),
    ProviderError('Provider HTTP 401', response_metadata={'http_status': 401})])
def test_cancel_race_does_not_hide_program_or_authentication_errors(error):
    client = ChatClient('http://127.0.0.1:1/v1', 'unit-model', 'unit-secret')
    client.request_cancellation = RequestCancellation()
    def fail(*args):
        client.request_cancellation.cancel('execution_failed')
        raise error
    client._complete_once = fail
    with pytest.raises(type(error)) as caught:
        client.complete([])
    assert caught.value is error and not isinstance(caught.value, RequestCancelled)
