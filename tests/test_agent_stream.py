"""Actual stream protocol assembly must finish before any action can be exposed."""
from copy import deepcopy
from io import BytesIO
import json

import pytest

from sketchik.agent import ChatClient, ProviderError


def sse(*events, done=True):
    lines = [b': keepalive\n\n']
    lines.extend(b'data: ' + json.dumps(event).encode() + b'\n\n' for event in events)
    if done:
        lines.append(b'data: [DONE]\n\n')
    return b''.join(lines)


def chunk(delta, finish=None):
    return {'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}


class Opener:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(json.loads(request.data))
        return BytesIO(self.reply)


def client_for(reply, tmp_path):
    client = ChatClient('http://127.0.0.1:1/v1', 'unit-model', 'unit-secret', 'low',
                        stream=True, max_tokens=8192, budget_field='max_completion_tokens',
                        transport_dir=tmp_path)
    client.opener = Opener(reply)
    return client


def test_stream_assembles_interleaved_tools_and_keeps_history_usage_and_audit(tmp_path):
    usage = {'prompt_tokens': 120, 'prompt_tokens_details': {'cached_tokens': 100},
             'completion_tokens': 30, 'completion_tokens_details': {'reasoning_tokens': 20}}
    body = sse(
        chunk({'role': 'assistant', 'reasoning_content': 'private deliberation '}),
        chunk({'reasoning_content': 'retained', 'content': 'ready'}),
        chunk({'tool_calls': [
            {'index': 1, 'id': 'second', 'function': {'name': 'wait', 'arguments': '{"steps":'}},
            {'index': 0, 'id': 'first', 'function': {'name': 'move', 'arguments': '{"x":'}}]}),
        chunk({'tool_calls': [
            {'index': 0, 'function': {'arguments': '1}'}},
            {'index': 1, 'function': {'arguments': '2}'}}]}),
        chunk({}, 'tool_calls'), {'choices': [], 'usage': usage})
    client = client_for(body, tmp_path)
    progress = []
    client.on_stream_progress = progress.append
    messages = [{'role': 'assistant', 'content': 'prior', 'reasoning_content': 'full history'},
                {'role': 'user', 'content': 'next'}]
    original = deepcopy(messages)
    tools = [{'type': 'function', 'function': {'name': 'move', 'parameters': {'type': 'object'}}}]
    result = client.complete(messages, tools)
    assert messages == original
    sent = client.opener.requests[0]
    assert sent['messages'] == original and sent['tools'] == tools
    assert sent['max_completion_tokens'] == 8192 and 'max_tokens' not in sent
    assert sent['stream'] is True and sent['stream_options'] == {'include_usage': True}
    assert sent['parallel_tool_calls'] is False and sent['reasoning_effort'] == 'low'
    assert result['usage'] == usage and result['finish_reason'] == 'tool_calls'
    assert result['message']['reasoning_content'] == 'private deliberation retained'
    assert result['message']['content'] == 'ready'
    assert [(c['id'], c['function']) for c in result['message']['tool_calls']] == [
        ('first', {'name': 'move', 'arguments': '{"x":1}'}),
        ('second', {'name': 'wait', 'arguments': '{"steps":2}'})]
    assert progress and 'private deliberation' not in json.dumps(progress)
    metrics = client.last_response_metadata['stream_metrics']
    assert metrics['stream_complete'] and metrics['first_generation_s'] >= 0
    assert metrics['reasoning_chars'] == len('private deliberation retained')
    record = json.loads((tmp_path/'requests.jsonl').read_text())
    assert record['request_options']['max_completion_tokens'] == 8192
    assert record['tools'] == tools and 'unit-secret' not in json.dumps(record)
    traces = list((tmp_path/'streams').glob('*.sse'))
    # The trace contains consumed bytes through [DONE]; no read past its terminal line.
    assert len(traces) == 1 and traces[0].read_bytes() == body.removesuffix(b'\n')
    assert json.loads((tmp_path/'responses.jsonl').read_text())['message'] == result['message']
    client.complete(messages, tools)
    assert len(list((tmp_path/'streams').glob('*.sse'))) == 2


@pytest.mark.parametrize('finish', [None, 'length', 'content_filter'])
def test_partial_or_truncated_tool_stream_is_archived_and_never_returned_or_retried(finish, tmp_path):
    events = [chunk({'tool_calls': [{'index': 0, 'id': 'one', 'function': {
        'name': 'move', 'arguments': '{"x":'}}]})]
    if finish:
        events.append(chunk({}, finish))
    client = client_for(sse(*events, done=finish is not None), tmp_path)
    with pytest.raises(ProviderError, match='Incomplete|Truncated|Filtered'):
        client.complete([], [])
    assert len(client.opener.requests) == 1
    archive = json.loads((tmp_path/'responses.jsonl').read_text())
    assert archive['finish_reason'] == finish
    assert archive['message']['tool_calls'][0]['function']['arguments'] == '{"x":'
    assert archive['stream_metrics']['stream_complete'] == (finish is not None)


def test_stream_provider_error_is_sanitized_and_trace_kept(tmp_path):
    client = client_for(sse({'error': {'message': 'unit-secret provider echo'}}), tmp_path)
    with pytest.raises(ProviderError) as caught:
        client.complete([])
    assert 'unit-secret' not in str(caught.value)
    assert len(client.opener.requests) == 1
    trace = next((tmp_path/'streams').glob('*.sse')).read_text()
    assert '[redacted]' in trace and 'unit-secret' not in trace


def test_malformed_stream_does_not_return_tools_or_retry(tmp_path):
    client = client_for(b'data: {broken JSON\n\n', tmp_path)
    with pytest.raises(ProviderError, match='transport/JSON'):
        client.complete([])
    assert len(client.opener.requests) == 1
    assert next((tmp_path/'streams').glob('*.sse')).read_bytes() == b'data: {broken JSON\n'


def test_gateway_json_fallback_preserves_complete_multiline_reply_and_trace(tmp_path):
    reply = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '{}'}}],
             'usage': {'prompt_tokens': 10}}
    body = json.dumps(reply, indent=2).encode()
    client = client_for(body, tmp_path)
    result = client.complete([])
    assert result['message'] == reply['choices'][0]['message']
    assert client.last_response_metadata['stream_metrics'] == {
        'mode': 'json_fallback', 'stream_complete': True}
    assert next((tmp_path/'streams').glob('*.sse')).read_bytes() == body


def test_json_fallback_without_finish_reason_cannot_expose_tool_calls(tmp_path):
    body = json.dumps({'choices': [{'message': {'role': 'assistant', 'tool_calls': [{
        'id': 'one', 'type': 'function', 'function': {'name': 'move', 'arguments': '{}'}}]}}]}).encode()
    client = client_for(body, tmp_path)
    with pytest.raises(ProviderError, match='Incomplete'):
        client.complete([])
    assert len(client.opener.requests) == 1
    assert not client.last_response_metadata['stream_metrics']['stream_complete']


@pytest.mark.parametrize('options', [{'stream': 1}, {'budget_field': 'unsupported'}])
def test_invalid_transport_options_rejected_before_network(options):
    with pytest.raises(ValueError, match='stream mode'):
        ChatClient('http://127.0.0.1:1/v1', 'unit-model', 'unit-secret', **options)
