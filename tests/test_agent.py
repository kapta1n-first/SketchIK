from copy import deepcopy
import base64
from io import BytesIO
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import urllib.error

import numpy as np
import pytest
from PIL import Image

from sketchik.agent import ChatClient, ProviderError, ScriptedClient, load_settings, run_session
from sketchik.tools import ToolHub


class FakeRobot:
    """Only for protocol/unit tests; actual simulation is checked separately."""
    cameras = ("agentview", "robot0_eye_in_hand")
    steps = 0
    max_steps = 2000
    done = False
    metadata = {"task_language": "unit test, not a benchmark solution"}

    def state(self):
        return {"step": self.steps, "q": [0]*7, "position": [0,0,1], "quaternion_xyzw": [0,0,0,1],
                "finger_joint_positions": [.04,-.04], "gripper_command": -1,
                "official_success": True, "done": False}

    def camera(self, name):
        return {"rgb": np.zeros((16,16,3), dtype=np.uint8), "depth_m": np.ones((16,16)),
                "K": np.eye(3), "world_from_camera": np.eye(4), "step": self.steps,
                "camera": name, "convention": "unit fixture"}


class FakeExecutor:
    def __init__(self, robot):
        self.robot = robot
        self.calls = 0
        self.fail = False
        self.explode = False

    def move_relative(self, delta_m):
        self.calls += 1
        if self.explode:
            raise RuntimeError("unexpected backend failure")
        self.robot.steps += 3
        return {"converged": not self.fail, "state": self.robot.state(), "position_error_m": .001}

    def wait(self, steps=10):
        self.calls += 1
        self.robot.steps += steps
        return self.robot.state()


@pytest.fixture
def hub(tmp_path):
    robot = FakeRobot()
    records = []
    result = ToolHub(robot, FakeExecutor(robot), tmp_path, records.append)
    result.records = records
    return result


def call(name, args, identifier="id1"):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_observation_labels_depth_and_no_success_oracle(hub):
    description, parts = hub.observe()
    assert "official_success" not in json.dumps(description)
    assert sum(x["type"] == "image_url" for x in parts) == 2
    assert all(c["camera"] in hub.robot.cameras for c in description["cameras"])
    result, _ = hub.call("back_project", {"observation_id": "obs_00000", "camera": "agentview", "pixel_xy": [2,3]})
    assert result["result"]["world_xyz"] == [2,3,1]
    hub.robot.steps = 1
    result, _ = hub.call("back_project", {"observation_id": "obs_00000", "camera": "agentview", "pixel_xy": [2,3]})
    assert result["result"]["current"] is False
    assert result["result"]["source_step"] == 0


def test_stale_action_and_unknown_params_do_not_dispatch(hub):
    hub.robot.steps = 4
    result, _ = hub.call("move_relative", {"source_step": 0, "params": {"delta_m": [0,0,.02]}})
    assert not result["ok"] and result["action_count"] == 0
    result, _ = hub.call("move_relative", {"source_step": 4, "params": {"delta_m": [0,0,.02], "unsupported": True}})
    assert not result["ok"] and hub.executor.calls == 0


def test_bad_later_plan_syntax_rejected_before_first_action(hub):
    result, _ = hub.call("execute_plan", {"source_step": 0, "steps": [
        {"primitive": "move_relative", "params": {"delta_m": [0,0,.02]}},
        {"primitive": "made_up_tool", "params": {}}]})
    assert not result["ok"] and hub.executor.calls == 0


def test_nonconvergence_stops_plan_suffix_but_allows_replanning(hub):
    hub.executor.fail = True
    result, images = hub.call("execute_plan", {"source_step": 0, "steps": [
        {"primitive": "move_relative", "params": {"delta_m": [0,0,.02]}},
        {"primitive": "wait", "params": {"steps": 2}}]})
    assert result["ok"] and result["result"]["all_commands_completed"] is False
    assert hub.executor.calls == 1 and images
    hub.executor.fail = False
    result, _ = hub.call("move_relative", {"source_step": 3, "params": {"delta_m": [0,0,-.01]}})
    assert result["ok"] and hub.executor.calls == 2


def test_unexpected_failure_is_sticky_and_never_replays(hub):
    hub.executor.explode = True
    args = {"source_step": 0, "params": {"delta_m": [0,0,.02]}}
    first, _ = hub.call("move_relative", args)
    second, _ = hub.call("move_relative", args)
    assert first["fatal"] and second["fatal"] and hub.fatal
    assert hub.executor.calls == 1


def test_scripted_conversation_images_history_and_recovery(hub):
    class CheckingScript(ScriptedClient):
        def complete(self, messages, tools):
            assert any(t["function"]["name"] == "execute_plan" for t in tools)
            if self.round:
                assert any(m["role"] == "tool" for m in messages)
                assert sum(m["role"] == "user" for m in messages) >= 2
            return super().complete(messages, tools)
    result = run_session(hub, CheckingScript(), "unit instructions")
    assert result["status"] == "finished" and result["rounds"] == 3
    assert result["real_model_used"] is False
    assert hub.executor.calls == 2  # move + wait, stale action not dispatched
    transcript = (hub.run/"transcript.json").read_text()
    assert "data:image" not in transcript and "saved-observation-image-sha256" in transcript


def test_duplicate_id_and_malformed_batch_no_extra_motion(hub):
    class Client:
        metadata = {"kind": "scripted_transport_test"}
        def complete(self, messages, tools):
            one = call("move_relative", {"source_step": 0, "params": {"delta_m": [0,0,.02]}})
            return {"message": {"role": "assistant", "tool_calls": [one, deepcopy(one)]}}
    result = run_session(hub, Client(), "test")
    assert result["status"] == "provider_or_session_error" and hub.executor.calls == 0


def test_tool_budget_exhaustion_does_not_report_completion(hub):
    result = run_session(hub, ScriptedClient(), "test", max_tool_calls=1)
    assert result["status"] == "tool_budget_exhausted" and result["agent_assessment"] is None


def test_config_is_literal_not_shell(tmp_path, monkeypatch):
    for key in ("SKETCHIK_API_KEY", "SKETCHIK_BASE_URL", "SKETCHIK_MODEL", "SKETCHIK_REASONING"):
        monkeypatch.delenv(key, raising=False)
    file = tmp_path/"config.env"
    file.write_text("SKETCHIK_API_KEY='unit-secret'\nSKETCHIK_MODEL=unit-model\n")
    assert load_settings(file)["SKETCHIK_MODEL"] == "unit-model"
    file.write_text("PATH=$(touch should-not-exist)\n")
    with pytest.raises(ValueError):
        load_settings(file)


def test_http_chat_payload_and_truncated_response_no_tool_execution(tmp_path):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"finish_reason": "length" if len(requests)>1 else "stop",
                "message": {"role": "assistant", "content": "local protocol fixture"}}],
                "usage": {"prompt_tokens": 12}}).encode())
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "unit-model", "unit-secret", "low",transport_dir=tmp_path)
        result = client.complete([{"role": "user", "content": "test"}], [])
        assert result["usage"]["prompt_tokens"] == 12
        assert requests[0][0] == "/v1/chat/completions" and requests[0][2] == "Bearer unit-secret"
        assert requests[0][1]["reasoning_effort"] == "low"
        assert "unit-secret" not in json.dumps(client.metadata)
        with pytest.raises(ProviderError, match="Truncated"):
            client.complete([], [])
        assert len(requests) == 2
        archived=[json.loads(line) for line in (tmp_path/'responses.jsonl').read_text().splitlines()]
        assert [row['finish_reason'] for row in archived]==['stop','length']
        assert archived[1]['usage']=={'prompt_tokens':12}
        assert archived[1]['message']['content']=='local protocol fixture'
        assert client.last_response_metadata['finish_reason']=='length'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_plain_parameter_request_does_not_advertise_tools():
    captured = []
    class Opener:
        def open(self, request, timeout):
            captured.append(json.loads(request.data))
            return BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '{"steps":15}'}}]}).encode())
    client = ChatClient('http://127.0.0.1:1/v1', 'unit-model', 'unit-secret')
    client.opener = Opener()
    result = client.complete([{'role': 'user', 'content': 'Return selected-tool parameters'}])
    assert result['message']['content'] == '{"steps":15}'
    assert not {'tools', 'tool_choice', 'parallel_tool_calls'} & captured[0].keys()




@pytest.mark.parametrize('budget',[None,8192])
def test_explicit_output_budget_is_sent_and_audited_without_changing_history(budget,tmp_path):
    captured=[]
    class Opener:
        def open(self,request,timeout):
            captured.append(json.loads(request.data))
            return BytesIO(json.dumps({'choices':[{'finish_reason':'stop',
                'message':{'role':'assistant','content':'{}'}}]}).encode())
    client=ChatClient('http://127.0.0.1:1/v1','unit-model','unit-secret','low',
        max_tokens=budget,transport_dir=tmp_path)
    client.opener=Opener()
    messages=[{'role':'assistant','content':'prior result','reasoning_content':'full preserved source'},
        {'role':'user','content':'next stage'}]
    client.complete(messages,[])
    assert captured[0]['messages']==messages
    assert captured[0].get('max_tokens')==budget
    assert ('max_tokens' in captured[0])==(budget is not None)
    assert client.metadata['requested_max_tokens']==budget
    assert client.last_request_metadata['requested_max_tokens']==budget
    assert json.loads((tmp_path/'requests.jsonl').read_text())['requested_max_tokens']==budget


@pytest.mark.parametrize('budget',[0,-1,True,1.5])
def test_invalid_output_budget_is_rejected_before_request(budget):
    with pytest.raises(ValueError,match='max_tokens'):
        ChatClient('http://127.0.0.1:1/v1','unit-model','unit-secret',max_tokens=budget)


@pytest.mark.parametrize('reason', [TimeoutError('private unit-secret'),
    ConnectionResetError(104, 'private unit-secret'), 'private unit-secret'])
def test_transport_reason_class_retained_without_secret_or_retry(reason):
    class BrokenOpener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            assert timeout == 300
            raise urllib.error.URLError(reason)
    client = ChatClient('https://unit.invalid/v1', 'unit-model', 'unit-secret', timeout=300)
    client.opener = BrokenOpener()
    with pytest.raises(ProviderError) as result:
        client.complete([], [])
    assert f'reason_type={type(reason).__name__}' in str(result.value)
    assert 'unit-secret' not in str(result.value)
    assert client.opener.calls == 1


@pytest.mark.parametrize('detail,expected',[
    ({'code':'insufficient_quota','type':'invalid_request_error'},('insufficient_quota','invalid_request_error')),
    ({'code':'unit-secret','type':'private provider echo'},(None,None)),
])
def test_http_failure_archives_only_status_and_known_categories_without_retry(tmp_path,detail,expected):
    body=json.dumps({'error':{**detail,'message':'Bearer unit-secret private user input'},
        'authorization':'unit-secret'}).encode()
    class BrokenOpener:
        calls=0
        def open(self,request,timeout):
            self.calls+=1
            raise urllib.error.HTTPError('https://unit.invalid/private-unit-secret',403,
                'private unit-secret provider echo',{},BytesIO(body))
    client=ChatClient('https://unit.invalid/v1','unit-model','unit-secret',transport_dir=tmp_path)
    client.opener=BrokenOpener()
    with pytest.raises(ProviderError,match='HTTP 403; no automatic retry') as failure:
        client.complete([{'role':'user','content':'public protocol fixture'}])
    audit=json.loads((tmp_path/'responses.jsonl').read_text())
    assert audit==client.last_response_metadata==failure.value.response_metadata
    assert audit['http_status']==403 and audit['status']=='http_error'
    assert (audit['provider_error_code'],audit['provider_error_type'])==expected
    assert audit['request_body_sha256']==client.last_request_metadata['body_sha256']
    assert audit['error_body_bytes_read']==len(body) and not audit['error_body_truncated']
    assert client.opener.calls==1 and 'message' not in audit
    serialized=json.dumps(audit)+str(failure.value)
    assert 'unit-secret' not in serialized and 'private' not in serialized and 'Bearer' not in serialized


@pytest.mark.parametrize('body',[b'<html>private unit-secret gateway error</html>',b'[]',
    b'{"error":{"code":"permission_denied","message":"'+b'x'*20000+b'"}}'])
def test_unstructured_or_oversized_http_error_preserves_original_status(body,tmp_path):
    class BrokenOpener:
        calls=0
        def open(self,request,timeout):
            self.calls+=1
            raise urllib.error.HTTPError('https://unit.invalid/v1',403,'Forbidden',{},BytesIO(body))
    client=ChatClient('https://unit.invalid/v1','unit-model','unit-secret',transport_dir=tmp_path)
    client.opener=BrokenOpener()
    with pytest.raises(ProviderError,match='HTTP 403'):
        client.complete([])
    audit=client.last_response_metadata
    assert audit['provider_error_code'] is None and audit['provider_error_type'] is None
    assert audit['error_body_bytes_read']<=16_385
    assert audit['error_body_truncated']==(len(body)>16_384) and client.opener.calls==1


def test_proxy_bypass_is_explicit_and_defaults_remain(monkeypatch):
    monkeypatch.setattr(urllib.request, 'getproxies', lambda: {'https': 'http://127.0.0.1:7900'})
    normal = ChatClient('https://unit.invalid/v1', 'unit-model', 'unit-secret')
    direct = ChatClient('https://unit.invalid/v1', 'unit-model', 'unit-secret', use_env_proxy=False)
    local = ChatClient('http://127.0.0.1:18760/v1', 'unit-model', 'unit-secret')
    assert normal.metadata['proxy_mode'] == 'environment'
    assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies for h in normal.opener.handlers)
    assert direct.metadata['proxy_mode'] == local.metadata['proxy_mode'] == 'direct'
    assert not any(isinstance(h, urllib.request.ProxyHandler) for h in direct.opener.handlers)
    assert not any(isinstance(h, urllib.request.ProxyHandler) for h in local.opener.handlers)


def test_jpeg_wire_encoding_preserves_full_source_history_and_saves_exact_sent_images(tmp_path):
    source = BytesIO()
    Image.fromarray(np.arange(32*32*3, dtype=np.uint8).reshape(32,32,3)).save(source, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(source.getvalue()).decode()
    messages = [{'role': 'system', 'content': 'unchanged instructions'}, {'role': 'user', 'content': [
        {'type': 'text', 'text': 'same pixel coordinates and camera'},
        {'type': 'image_url', 'image_url': {'url': url, 'detail': 'high'}},
        {'type': 'image_url', 'image_url': {'url': url}}]}]
    original = deepcopy(messages)
    captured = []
    class Opener:
        def open(self, request, timeout):
            captured.append(json.loads(request.data))
            return BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'local fixture'}}]}).encode())
    client = ChatClient('https://unit.invalid/v1', 'unit-model', 'unit-secret',
                        image_jpeg_quality=95, transport_dir=tmp_path/'wire')
    client.opener = Opener()
    for _ in range(2):client.complete(messages, [])
    assert messages == original and client.last_request_metadata['image_count'] == 2
    assert captured[0] == captured[1] and len(client._jpeg_cache) == 1
    wire = captured[0]['messages']
    assert wire[0] == original[0] and wire[1]['content'][0] == original[1]['content'][0]
    for part in wire[1]['content'][1:]:
        sent = part['image_url']['url']
        assert sent.startswith('data:image/jpeg;base64,')
        with Image.open(BytesIO(base64.b64decode(sent.split(',',1)[1]))) as picture:
            assert picture.size == (32,32) and picture.format == 'JPEG'
    assert wire[1]['content'][1]['image_url']['detail'] == 'high'
    assert len(list((tmp_path/'wire/images').glob('*.jpg'))) == 1
    records = [json.loads(s) for s in (tmp_path/'wire/requests.jsonl').read_text().splitlines()]
    assert len(records) == 2 and records[0]['image_count'] == 2
    assert 'unit-secret' not in json.dumps(records)
    assert 'data:image' not in json.dumps(records)


@pytest.mark.parametrize('quality', [0, 101, True, 95.5])
def test_invalid_jpeg_transport_quality_is_rejected_before_network(quality):
    with pytest.raises(ValueError, match='JPEG transport quality'):
        ChatClient('https://unit.invalid/v1', 'unit-model', 'unit-secret', image_jpeg_quality=quality)


def test_exact_image_references_are_opt_in_and_reversible(tmp_path):
    image_a = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YQ==', 'detail': 'high'}}
    image_b = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,Yg==', 'detail': 'high'}}
    different_detail = {'type': 'image_url', 'image_url': {'url': image_a['image_url']['url'], 'detail': 'low'}}
    messages = [
        {'role': 'system', 'content': 'keep all context'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'first'}, deepcopy(image_a), deepcopy(image_b)]},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_1', 'type': 'function',
            'function': {'name': 'look', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"ok":true}'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'later'}, deepcopy(image_a),
            deepcopy(different_detail), {'type': 'text', 'text': 'tail'}, deepcopy(image_b)]},
    ]
    original = deepcopy(messages)
    captured = []
    class Opener:
        def open(self, request, timeout):
            captured.append(json.loads(request.data)['messages'])
            return BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'fixture'}}]}).encode())

    default = ChatClient('https://unit.invalid/v1', 'unit-model', 'unit-secret')
    default.opener = Opener()
    default.complete(messages)
    assert captured[-1] == original
    assert default.last_request_metadata['image_count'] == 5
    assert default.last_request_metadata['original_image_count'] == 5
    assert default.last_request_metadata['image_references'] == []

    client = ChatClient('https://unit.invalid/v1', 'unit-model', 'unit-secret',
                        reference_exact_images=True, transport_dir=tmp_path/'wire')
    client.opener = Opener()
    client.complete(messages)
    wire = captured[-1]
    assert messages == original
    assert [m['role'] for m in wire] == [m['role'] for m in original]
    assert wire[2:4] == original[2:4]
    assert wire[1]['content'][0] == original[1]['content'][0]
    assert wire[4]['content'][0] == original[4]['content'][0]
    assert wire[4]['content'][-2] == original[4]['content'][-2]
    sent_images = [part for message in wire if isinstance(message.get('content'), list)
                   for part in message['content'] if part['type'] == 'image_url']
    assert sent_images == [image_a, image_b, different_detail]
    metadata = client.last_request_metadata
    assert metadata['image_count'] == 3 and metadata['original_image_count'] == 5
    assert len(metadata['image_references']) == 2
    assert 'data:image' not in json.dumps(metadata)
    record = json.loads((tmp_path/'wire/requests.jsonl').read_text().splitlines()[0])
    assert record['image_references'] == metadata['image_references']
    assert 'data:image' not in json.dumps(record)

    # Reconstruct the pre-reference wire from first copies and recorded positions.
    canonical = {tuple(mapping['canonical_position']): mapping['image_id']
                 for mapping in metadata['image_references']}
    duplicates = {tuple(mapping['original_position']): mapping for mapping in metadata['image_references']}
    first_image_by_id = {}
    for message in wire:
        if not isinstance(message.get('content'), list):
            continue
        for before, after in zip(message['content'], message['content'][1:]):
            if before.get('type') == 'text' and before.get('text', '').startswith('Image ID '):
                image_id = before['text'].removeprefix('Image ID ').removesuffix(':')
                assert after['type'] == 'image_url'
                first_image_by_id[image_id] = after
    reconstructed = deepcopy(wire)
    for message_index, message in enumerate(reconstructed):
        if not isinstance(message.get('content'), list):
            continue
        restored = []
        original_parts = original[message_index]['content']
        wire_index = 0
        for part_index in range(len(original_parts)):
            location = (message_index, part_index)
            if location in duplicates:
                mapping = duplicates[location]
                assert message['content'][wire_index]['text'] == f"Image reference {mapping['image_id']}."
                restored.append(deepcopy(first_image_by_id[mapping['image_id']]))
                wire_index += 1
            elif location in canonical or original_parts[part_index]['type'] == 'image_url':
                assert message['content'][wire_index]['text'].startswith('Image ID ')
                restored.append(message['content'][wire_index + 1])
                wire_index += 2
            else:
                restored.append(message['content'][wire_index])
                wire_index += 1
        assert wire_index == len(message['content'])
        message['content'] = restored
    assert reconstructed == captured[0]


def test_run_files_create_read_list_and_explicit_overwrite(hub):
    result, _ = hub.call("write_text_file", {"path": "workspace/plan.json", "content": '{"steps": []}'})
    assert result["ok"]
    result, _ = hub.call("list_dir", {"path": "workspace"})
    assert result["result"]["entries"] == ["plan.json"]
    result, _ = hub.call("read_text_file", {"path": "workspace/plan.json"})
    assert result["result"]["content"] == '{"steps": []}'
    result, _ = hub.call("write_text_file", {"path": "workspace/plan.json", "content": "replacement"})
    assert not result["ok"]
    assert (hub.workspace/"plan.json").read_text() == '{"steps": []}'
    result, _ = hub.call("write_text_file", {"path": "workspace/plan.json", "content": "replacement", "overwrite": True})
    assert result["ok"] and (hub.workspace/"plan.json").read_text() == "replacement"
    assert hub.robot.steps == 0 and hub.executor.calls == 0


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "observations/../result.json", "result.json", ".runtime/agent.env"])
def test_file_tools_cannot_expose_private_or_evaluation_artifacts(hub, path):
    (hub.run/"result.json").write_text('{"official_success": true}')
    result, _ = hub.call("read_text_file", {"path": path})
    assert not result["ok"] and "official_success" not in json.dumps(result)
    assert not hub.fatal and hub.executor.calls == 0


def test_file_tools_read_docs_but_do_not_overwrite_them(hub, tmp_path):
    project = tmp_path/"project"
    (project/"prompts").mkdir(parents=True)
    (project/"prompts/system.md").write_text("tool instructions")
    hub.project_root = project
    result, _ = hub.call("read_text_file", {"path": "prompts/system.md", "max_chars": 4})
    assert result["result"]["content"] == "tool" and result["result"]["truncated"]
    result, _ = hub.call("write_text_file", {"path": "prompts/system.md", "content": "changed", "overwrite": True})
    assert not result["ok"] and (project/"prompts/system.md").read_text() == "tool instructions"
    (hub.workspace/"shortcut.txt").symlink_to(project/"prompts/system.md")
    result, _ = hub.call("read_text_file", {"path": "workspace/shortcut.txt"})
    assert not result["ok"]


def test_read_historical_image_preserves_frame_step_and_does_not_render(hub, monkeypatch):
    hub.observe()
    hub.robot.steps = 12
    def unexpected_render(name):
        raise AssertionError("read_image must not capture a new image")
    monkeypatch.setattr(hub.robot, "camera", unexpected_render)
    result, images = hub.call("read_image", {"observation_id": "obs_00000", "camera": "agentview"})
    assert result["ok"] and result["result"]["source_step"] == 0
    assert result["result"]["current"] is False and result["source_step"] == 12
    assert sum(p["type"] == "image_url" for p in images) == 1
    assert hub.executor.calls == 0


def test_file_tool_and_historical_image_in_full_conversation(hub):
    class Client:
        metadata = {"kind": "scripted_transport_test"}
        def __init__(self):
            self.round = 0
        def complete(self, messages, tools):
            self.round += 1
            if self.round == 1:
                request = call("read_image", {"observation_id": "obs_00000", "camera": "robot0_eye_in_hand"})
            else:
                # Historical RGB is still passed as a real multimodal image after tool replies.
                assert messages[-1]["role"] == "user"
                assert any(p["type"] == "image_url" for p in messages[-1]["content"])
                request = call("finish", {"source_step": 0, "assessment": "blocked", "reason": "IO test"}, "id2")
            return {"message": {"role": "assistant", "content": None, "tool_calls": [request]}}
    result = run_session(hub, Client(), "IO instructions")
    assert result["status"] == "finished" and hub.robot.steps == 0
