from copy import deepcopy
import json

import pytest

from sketchik.json_tools import request_messages, response_message
from test_agent_stream import client_for, sse, chunk


def test_request_preserves_selected_history_images_and_active_tools():
    source = [{'role': 'system', 'content': 'persistent instructions'},
        {'role': 'assistant', 'reasoning_content': 'retained source reasoning',
         'tool_calls': [{'id': 'old', 'function': {'name': 'read_image', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'old', 'content': 'actual result'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'current state'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YQ=='}}]}]
    original = deepcopy(source)
    tools = [{'type': 'function', 'function': {'name': 'draft_next_sketch',
        'parameters': {'type': 'object'}}}]
    wire = request_messages(source, tools)
    assert source == original
    assert [m['role'] for m in wire] == ['system', 'user']
    assert wire[0]['content'] == 'persistent instructions'
    assert source[-1]['content'][-1] in wire[-1]['content']
    assert 'actual result' in json.dumps(wire)
    assert 'retained source reasoning' in json.dumps(wire)
    assert json.loads(wire[-1]['content'][-1]['text'])['available_tools'] == tools


def test_response_only_maps_envelope_and_preserves_repetitions_for_engine():
    call = {'name': 'execute_actions', 'arguments': {'actions': []}}
    source = {'role': 'assistant', 'content': json.dumps({'tool_calls': [call, call]}),
              'reasoning_content': 'original'}
    original = deepcopy(source)
    decoded, audit = response_message(source)
    assert source == original and audit['tool_dispatches'] == 0
    assert len(decoded['tool_calls']) == 2
    assert decoded['tool_calls'][0]['id'] != decoded['tool_calls'][1]['id']
    assert decoded['content'] == source['content']
    assert decoded['reasoning_content'] == source['reasoning_content']


@pytest.mark.parametrize('text', ['{}', '', '```json\n{}\n```',
    '{"tool_calls":[]}', '{"tool_calls":[{"name":"move","arguments":NaN}]}',
    '{"tool_calls":[{"name":"move","arguments":{},"arguments":{}}]}',
    '{"tool_calls":[{"name":"move","arguments":"{}"}]}'])
def test_invalid_text_remains_invalid_without_dispatch_or_repair(text):
    message = {'role': 'assistant', 'content': text}
    decoded, audit = response_message(message)
    assert decoded is message
    assert audit['status'] == 'invalid_json_envelope'
    assert not decoded.get('tool_calls')


def test_real_client_archives_raw_json_and_uses_no_native_tool_request(tmp_path):
    text = json.dumps({'tool_calls': [{'name': 'finish', 'arguments': {'reason': 'done'}}]})
    client = client_for(sse(chunk({'content': text}, 'stop')), tmp_path)
    client.json_tools = True
    tools = [{'type': 'function', 'function': {'name': 'finish', 'parameters': {'type': 'object'}}}]
    result = client.complete([{'role': 'system', 'content': 'instructions'}], tools)
    sent = client.opener.requests[0]
    assert 'tools' not in sent and 'tool_choice' not in sent and 'parallel_tool_calls' not in sent
    assert result['message']['tool_calls'][0]['function']['name'] == 'finish'
    raw = json.loads((tmp_path/'responses.jsonl').read_text())['message']
    assert raw['content'] == text and not raw.get('tool_calls')
    decoded = json.loads((tmp_path/'json_tool_conversions.jsonl').read_text())
    assert decoded['tool_dispatches'] == 0
