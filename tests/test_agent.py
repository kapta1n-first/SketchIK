from copy import deepcopy
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

import numpy as np
import pytest

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


def test_http_chat_payload_and_truncated_response_no_tool_execution():
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
        client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "unit-model", "unit-secret", "low")
        result = client.complete([{"role": "user", "content": "test"}], [])
        assert result["usage"]["prompt_tokens"] == 12
        assert requests[0][0] == "/v1/chat/completions" and requests[0][2] == "Bearer unit-secret"
        assert requests[0][1]["reasoning_effort"] == "low"
        assert "unit-secret" not in json.dumps(client.metadata)
        with pytest.raises(ProviderError, match="Truncated"):
            client.complete([], [])
        assert len(requests) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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
