import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from sketchik.usage_accounting import credit_snapshot, save_usage, transport_usage, usage_record


@pytest.fixture
def gateway(tmp_path):
    state = {'calls': [], 'payload': {'success': True, 'total_remain': 123.45,
        'details': [{'email': 'private@example.org', 'remaining': 123.45}]}, 'redirect': None}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state['calls'].append((self.path, self.headers.get('Authorization')))
            self.send_response(302 if state['redirect'] else 200)
            if state['redirect']:
                self.send_header('Location', state['redirect'])
            self.end_headers()
            self.wfile.write(json.dumps(state['payload']).encode())
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path/'private_config.json'
    config.write_text(json.dumps({'server_host': '127.0.0.1', 'server_port': server.server_port,
        'admin_token': 'private-test-token'}))
    yield config, state
    server.shutdown()
    server.server_close()
    thread.join()


def test_credit_snapshot_one_query_and_no_private_account_data(gateway):
    config, state = gateway
    result = credit_snapshot(config)
    assert result['status'] == 'available' and result['remaining_platform_credits'] == 123.45
    assert state['calls'] == [('/api/credits/simple', 'Bearer private-test-token')]
    assert 'private' not in json.dumps(result) and 'admin_token' not in json.dumps(result)


@pytest.mark.parametrize('payload', [
    {'success': False, 'total_remain': 9999},
    {'success': True, 'total_remain': -1},
    {'success': True, 'total_remain': 'NaN'},
    {'success': True, 'total_remain': 12, 'details': [{'error': 'private-provider-error'}]},
    {'success': True, 'total_remain': 0, 'details': []},
])
def test_unknown_or_partial_balance_is_not_zero_or_success(gateway, payload):
    config, state = gateway
    state['payload'] = payload
    record = credit_snapshot(config)
    assert record['status'] == 'unavailable' and record['remaining_platform_credits'] is None
    assert len(state['calls']) == 1 and 'private' not in json.dumps(record)


def test_admin_credentials_never_follow_redirect(gateway):
    config, state = gateway
    state['redirect'] = 'http://127.0.0.1:'+str(json.loads(config.read_text())['server_port'])+'/elsewhere'
    result = credit_snapshot(config)
    assert result['status'] == 'unavailable' and len(state['calls']) == 1


def test_no_remote_admin_requests_or_configuration_errors_exposed(tmp_path):
    config = tmp_path/'private.json'
    config.write_text(json.dumps({'server_host': 'example.org', 'server_port': 443,
        'admin_token': 'secret'}))
    assert credit_snapshot(config)['status'] == 'unavailable'
    missing = credit_snapshot(tmp_path/'nonexistent-secret-file')
    assert missing['status'] == 'unavailable' and 'secret' not in json.dumps(missing)
    assert credit_snapshot()['status'] == 'not_configured'


def prepare_transport(folder):
    path = folder/'a_transport'
    path.mkdir(parents=True)
    request = {'body_sha256': 'same-body', 'body_bytes': 1200, 'image_count': 4,
        'request_options': {'model': 'gpt-6.1-sol'}, 'messages': [{'content': 'private-context'}]}
    (path/'requests.jsonl').write_text((json.dumps(request)+'\n')*3)
    (path/'responses.jsonl').write_text('\n'.join(map(json.dumps, [
        {'usage': None}, {'status': 'http_error', 'usage': {'prompt_tokens': -1}},
        {'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}]))+'\n')
    (folder/'events.jsonl').write_text('\n'.join(map(json.dumps, [
        {'event': 'api_finished', 'wall_s': 2}, {'event': 'single_a_request_finished', 'wall_s': 3},
        {'event': 'motion_finished', 'wall_s': 100}]))+'\n')


def test_counts_repeated_requests_errors_and_unknown_token_values(tmp_path):
    prepare_transport(tmp_path)
    result = transport_usage(tmp_path)
    role = result['roles']['a']
    assert result['total_requests'] == 3 and role['http_errors'] == 1
    assert role['image_occurrences_sent'] == 12 and role['request_body_bytes'] == 3600
    assert role['tokens']['prompt_tokens'] == {'reported_sum': 100, 'responses_with_valid_value': 1}
    assert result['recorded_a_request_wall_s_sum'] == 5
    assert 'private-context' not in json.dumps(result)
    assert transport_usage(tmp_path/'missing')['total_requests'] == 0


@pytest.mark.parametrize('before,after,expected', [
    (100.10, 98.01, 2.09), (98.01, 100.10, -2.09), (None, 0., None), (0., 0., 0.),
])
def test_balance_difference_is_exact_account_delta_not_attributed_cost(tmp_path, before, after, expected):
    def balance(value):
        return {'status': 'available' if value is not None else 'unavailable',
                'remaining_platform_credits': value}
    result = usage_record(tmp_path, {}, balance(before), balance(after))
    assert result['account_credit_decrease'] == expected and result['attributed_trial_cost'] is None


def test_ledger_keeps_one_final_record_per_run_and_no_invented_historical_balance(tmp_path):
    folder = tmp_path/'run'
    folder.mkdir()
    prepare_transport(folder)
    record = usage_record(folder, {'status': 'finished', 'wall_s': 12.})
    for _ in range(2):
        save_usage(folder, tmp_path/'ledger', record)
    assert record['account_credit_decrease'] is None
    assert len((tmp_path/'ledger/trials.jsonl').read_text().splitlines()) == 1
    assert len((tmp_path/'ledger/trials.csv').read_text().splitlines()) == 2
    assert json.loads((folder/'usage.json').read_text())['total_requests'] == 3


def test_every_failed_and_resumed_attempt_is_measured_without_exposing_inputs(tmp_path, monkeypatch):
    from sketchik.agent import ChatClient, ProviderError
    from sketchik.usage_accounting import TrialUsage
    values = iter([1000., 1000., 990., 990., 975., 975.])
    def balance(_):
        return {'status': 'available', 'remaining_platform_credits': next(values)}
    monkeypatch.setattr('sketchik.usage_accounting.credit_snapshot', balance)
    usage = TrialUsage(tmp_path, tmp_path/'ledger')
    client = ChatClient('http://127.0.0.1:1/v1', 'fixture', 'private-test-key')
    client.attempt_observer = usage.measure_attempt
    calls = []
    def attempt(messages, tools):
        calls.append(1)
        client.last_request_metadata = {'body_sha256': 'same-body', 'body_bytes': 80}
        if len(calls) == 1:
            raise ProviderError('fixture disconnect')
        client.last_response_metadata = {'finish_reason': 'stop'}
        return {'message': {'role': 'assistant', 'content': 'ready'}}
    client._complete_once = attempt
    client.request_pause_handler = lambda *args: None  # Explicit test-only resume.
    client.complete([{'role': 'user', 'content': 'private-task-input'}])
    usage.finish({'status': 'finished'})
    records = [json.loads(x) for x in (tmp_path/'usage_attempts.jsonl').read_text().splitlines()]
    assert [r['account_credit_decrease'] for r in records] == [10., 15.]
    assert records[0]['failure_type'] == 'ProviderError' and records[1]['failure_type'] is None
    assert client.operator_resumed_requests == 1 and len(calls) == 2
    assert all(r['attributed_request_cost'] is None for r in records)
    assert 'private' not in (tmp_path/'usage_attempts.jsonl').read_text()
    final = json.loads((tmp_path/'usage.json').read_text())
    assert final['account_credit_decrease'] == 25.
    assert final['attempt_accounting']['failed_attempts'] == 1


def test_trial_failure_finalizes_usage_without_replacing_original_failure(tmp_path, monkeypatch):
    import sketchik.task_trial as trial
    calls = []
    class Usage:
        def __init__(self, folder, ledger, config):
            calls.append(('start', config))
        def finish(self, report):
            calls.append(('finish', report['status']))
            raise OSError('private-account-error')
    monkeypatch.setattr('sketchik.usage_accounting.TrialUsage', Usage)
    def unavailable(*_):
        raise RuntimeError('resource unavailable')
    monkeypatch.setattr('sketchik.runtime.configure', unavailable)
    folder = tmp_path/'run'
    trial.run(SimpleNamespace(output=folder, planner_layout='single', gpu_uuid='test', peer_pid=[],
        credits_config=tmp_path/'private.json'))
    result = json.loads((folder/'result.json').read_text())
    assert result['error'] == 'RuntimeError: resource unavailable'
    assert result['usage_accounting'] == {'status': 'unavailable', 'error_type': 'OSError'}
    assert calls == [('start', tmp_path/'private.json'), ('finish', 'failed')]
    assert 'private-account-error' not in json.dumps(result)
