import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent, ModelRetry, ToolCallPart
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ModelMessagesTypeAdapter, RetryPromptPart, TextPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from primitive_guard import ResponseGuard, guarded_session_class


def call(name='move_to', args=None, ident='a'):
    return ToolCallPart(name, args if args is not None else {'xyz': [0, 0, 1]}, tool_call_id=ident)


def test_only_duplicate_physical_calls_reject_and_archive_entire_reply(tmp_path):
    guard = ResponseGuard({'move_to'}, tmp_path/'audit.jsonl')
    response = ModelResponse(parts=[TextPart('original explanation'), call(),
                                   call(args={'xyz': [0, 0, 1]}, ident='b')])
    with pytest.raises(ModelRetry, match='NO tools'):
        guard.check(response)
    record = json.loads((tmp_path/'audit.jsonl').read_text())
    archived = Path(record['rejected_response_archive'])
    assert archived.stat().st_mode & 0o777 == 0o600
    restored = ModelMessagesTypeAdapter.validate_json(archived.read_bytes())
    assert restored[0].parts == response.parts
    assert record['rejected_response_dispatched_tools'] == 0


@pytest.mark.parametrize('parts', [
    [call('read_text_file', {'path': 'guide'}), call('read_text_file', {'path': 'guide'}, 'b')],
    [call('segment', {'prompt': 'box'}), call('segment', {'prompt': 'box'}, 'b')],
    [call('back_project', {'row': 1}), call('back_project', {'row': 1}, 'b')],
    [call(), call(args={'xyz': [0, 0, 2]}, ident='b')],
    [call(), call('read_image', {'name': 'wrist.png'})],
    [call('finish', {}), call('read_text_file', {'path': 'guide'})],
    [call('memory_write', {'text': 'x'}), call('memory_write', {'text': 'x'}, 'b')],
    [call('unknown', {}), call('unknown', {}, 'b')],
    [call(args='not JSON'), call(args='not JSON', ident='b')],
])
def test_no_extra_interventions(tmp_path, parts):
    response = ModelResponse(parts=parts)
    assert ResponseGuard({'move_to'}, tmp_path/'audit.jsonl').check(response) is response


def test_duplicate_arg_key_order_and_cross_response_retries(tmp_path):
    guard = ResponseGuard({'move_to'}, tmp_path/'audit.jsonl')
    response = ModelResponse(parts=[call(args={'xyz': [0, 0, 1], 'gripper': -1}),
                                   call(args={'gripper': -1, 'xyz': [0, 0, 1]}, ident='b')])
    with pytest.raises(ModelRetry):
        guard.check(response)
    unique = ModelResponse(parts=[call()])
    assert guard.check(unique) is unique
    assert guard.check(unique) is unique  # Same target in a later response is allowed.


def capability(tmp_path):
    class NativeSession(AbstractCapability):
        def __init__(self):
            self.toolkit = SimpleNamespace(toolkit=SimpleNamespace(
                list_tools=lambda: [SimpleNamespace(name='move_to'), SimpleNamespace(name='read_text_file')]))
    cls = guarded_session_class(NativeSession, tmp_path/'audit.jsonl', lambda name: name == 'move_to')
    return cls()


@pytest.mark.parametrize('streaming', [False, True])
def test_real_sdk_atomic_rejection_same_history_then_single_execution(tmp_path, streaming):
    dispatched, histories = [], []
    def move_to(xyz: list[float]) -> str:
        dispatched.append(('move_to', xyz))
        return 'actual executed result'
    def read_text_file(path: str) -> str:
        dispatched.append(('read_text_file', path))
        return 'guide'
    def response(messages, _info):
        histories.append(list(messages))
        n = len(histories)
        if n == 1:
            return ModelResponse(parts=[call(), call(ident='b'), call('read_text_file', {'path': 'guide'}, 'c')])
        if n == 2:
            assert dispatched == []  # No sibling read or physical action ran.
            assert any(isinstance(p, RetryPromptPart) and 'NO tools' in str(p.content)
                       for m in messages for p in m.parts)
            assert any(isinstance(m, ModelResponse) and len(m.parts) == 3 for m in messages)
            return ModelResponse(parts=[call(ident='corrected')])
        assert dispatched == [('move_to', [0, 0, 1])]
        return ModelResponse(parts=[TextPart('done')])
    async def streamed(messages, info):
        result = response(messages, info)
        if isinstance(result.parts[0], TextPart):
            yield result.parts[0].content
        else:
            yield {i: DeltaToolCall(name=p.tool_name, json_args=json.dumps(p.args),
                                   tool_call_id=p.tool_call_id) for i, p in enumerate(result.parts)}
    async def consume(_ctx, events):
        async for _ in events:
            pass
    model = FunctionModel(response, stream_function=streamed)
    agent = Agent(model, tools=[move_to, read_text_file], capabilities=[capability(tmp_path)],
                  retries={'output': 3})
    async def run():
        return await agent.run('test', event_stream_handler=consume if streaming else None)
    result = asyncio.run(run())
    assert result.output == 'done'
    assert result.usage.requests == 3  # Rejected response was not a free/invisible retry.
    assert dispatched == [('move_to', [0, 0, 1])]
    assert [json.loads(x)['decision'] for x in (tmp_path/'audit.jsonl').read_text().splitlines()] == ['reject', 'pass', 'pass']


def test_rejections_keep_original_sdk_retry_budget(tmp_path):
    dispatched = []
    def move_to(xyz: list[float]) -> str:
        dispatched.append(xyz)
        return 'executed'
    def response(_messages, _info):
        return ModelResponse(parts=[call(), call(ident='b')])
    agent = Agent(FunctionModel(response), tools=[move_to], capabilities=[capability(tmp_path)],
                  retries={'output': 1})
    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync('test')
    assert dispatched == []
    assert len(list((tmp_path/'private/primitive_guard').glob('*.json'))) == 2


def test_launcher_changes_only_entry_and_can_select_original_baseline(tmp_path):
    from run_seed0 import configure_primitive_guard
    command = ['python', '-m', 'rpent.cli.main', '--model', 'openai-chat:gpt-6.1-sol',
               '--memory-profile', 'local', '--max-tokens', '65536']
    saved = list(command)
    env = {}
    guarded = configure_primitive_guard(command, env, tmp_path, 'reject-duplicate-primitives-v1')
    assert guarded[0] == command[0]
    assert guarded[2:] == command[3:]
    assert command == saved
    assert env == {'HARNESS_GUARD_AUDIT': str(tmp_path/'primitive_guard.jsonl')}
    assert configure_primitive_guard(command, {}, tmp_path, 'off') == command
