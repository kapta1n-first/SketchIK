"""Synchronous vision/tool conversation. No automatic network retries or action replay."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .tools import ToolHub


class ProviderError(RuntimeError):
    pass


def load_settings(path=None):
    """Literal KEY=VALUE file only; never source shell code or serialize secrets."""
    values = {}
    if path:
        for raw in Path(path).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            key, sep, value = line.partition("=")
            if not sep or key not in {"SKETCHIK_API_KEY", "SKETCHIK_BASE_URL", "SKETCHIK_MODEL", "SKETCHIK_REASONING"}:
                raise ValueError("Agent config accepts only literal SKETCHIK_API_KEY/BASE_URL/MODEL/REASONING assignments")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key] = value
    for key in ("SKETCHIK_API_KEY", "SKETCHIK_BASE_URL", "SKETCHIK_MODEL", "SKETCHIK_REASONING"):
        if key in os.environ:
            values[key] = os.environ[key]
    return values


class ChatClient:
    def __init__(self, base_url, model, api_key, reasoning=None, timeout=90):
        parsed = urlparse(base_url or "")
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query:
            raise ValueError("Expected API base URL without embedded credentials or query")
        if not model or not api_key:
            raise ValueError("Model and API key are required; nothing is sent without configuration")
        self.url = base_url.rstrip("/") + "/chat/completions"
        # Local gateways/tests must not be sent through a workstation HTTP proxy.
        self.opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                       if parsed.hostname in ("localhost", "127.0.0.1", "::1")
                       else urllib.request.build_opener())
        self.model, self.api_key, self.reasoning, self.timeout = model, api_key, reasoning, timeout
        self.metadata = {"kind": "chat_completions", "base_url": base_url, "model": model,
                         "reasoning_effort": reasoning, "automatic_retries": 0, "timeout_s": timeout}

    def complete(self, messages, tools):
        payload = {"model": self.model, "messages": messages, "tools": tools,
                   "tool_choice": "auto", "parallel_tool_calls": False}
        if self.reasoning:
            payload["reasoning_effort"] = self.reasoning
        body = json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        # Do not log raw provider error bodies (they can echo auth or full inputs).
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"Provider HTTP {exc.code}; no automatic retry") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Provider transport/JSON failure ({type(exc).__name__}); no automatic retry") from None
        choices = reply.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("Provider returned no choices")
        choice = choices[0]
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise ProviderError("Truncated/filtered response; tools not executed")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ProviderError("Invalid assistant message")
        return {"message": message, "usage": reply.get("usage"), "finish_reason": choice.get("finish_reason")}


def transcript_copy(messages):
    """Save image references as hashes; full pixels are in observation PNG files."""
    data = deepcopy(messages)
    for message in data:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    part["image_url"]["url"] = "saved-observation-image-sha256:" + hashlib.sha256(url.encode()).hexdigest()
    return data


def run_session(hub: ToolHub, client, system, max_turns=40, max_tool_calls=100):
    if max_turns < 1 or max_tool_calls < 1:
        raise ValueError("Agent budgets must be positive")
    (hub.run / "system_prompt.md").write_text(system)
    (hub.run / "tools.json").write_text(json.dumps(hub.schemas, indent=2))
    (hub.run / "agent_config.json").write_text(json.dumps(client.metadata | {
        "max_turns": max_turns, "max_tool_calls": max_tool_calls,
        "history_policy": "full session, no silent truncation or compression"}, indent=2))
    _, observation = hub.observe()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": observation}]
    seen = set()
    calls = 0
    rounds = 0
    status, error = "turn_budget_exhausted", None
    api_seconds = 0.0
    try:
        for turn in range(max_turns):
            if hub.robot.done or hub.robot.steps >= hub.robot.max_steps:
                status = "episode_budget_exhausted"
                break
            if calls >= max_tool_calls:
                status = "tool_budget_exhausted"
                break
            start = time.perf_counter()
            rounds += 1
            try:
                response = client.complete(messages, hub.schemas)
            except Exception:
                elapsed = time.perf_counter()-start
                api_seconds += elapsed
                hub.emit({"event": "agent_request_failed", "round": turn+1,
                          "wall_s": elapsed})
                raise
            elapsed = time.perf_counter()-start
            api_seconds += elapsed
            message = response["message"]
            requests = message.get("tool_calls") or []
            if not isinstance(requests, list):
                raise ProviderError("tool_calls must be a list")
            # Validate entire provider batch before any robot action, including duplicate IDs.
            batch = set()
            parsed = []
            for item in requests:
                call_id = item.get("id")
                function = item.get("function", {})
                if item.get("type") != "function" or not isinstance(call_id, str) or not call_id or call_id in seen or call_id in batch:
                    raise ProviderError("Malformed or duplicate tool call ID; batch not executed")
                if not isinstance(function.get("name"), str):
                    raise ProviderError("Missing function name")
                try:
                    arguments = json.loads(function["arguments"], parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                except (KeyError, ValueError, TypeError):
                    raise ProviderError("Invalid function arguments JSON; batch not executed") from None
                batch.add(call_id)
                parsed.append((call_id, function["name"], arguments))
            if calls + len(parsed) > max_tool_calls:
                status = "tool_budget_exhausted"
                break
            # Retain provider reasoning_content/signatures when supplied for compatible gateways.
            kept = {k:v for k,v in message.items() if k not in ("refusal", "annotations", "audio")}
            messages.append(kept)
            hub.emit({"event": "agent_response", "round": turn+1, "wall_s": elapsed,
                      "usage": response.get("usage"), "finish_reason": response.get("finish_reason"),
                      "message": kept})
            if not parsed:
                status = "model_stopped_without_finish"
                break
            fresh_images = []
            for call_id, name, arguments in parsed:
                seen.add(call_id)
                calls += 1
                result, images = hub.call(name, arguments)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, allow_nan=False)})
                if images:
                    fresh_images.extend(images)
            # A user multimodal message follows tool replies; no image arrays in string-only tool content.
            if fresh_images:
                messages.append({"role": "user", "content": fresh_images})
            if hub.fatal:
                status = "execution_error"
                break
            if hub.finished:
                status = "finished"
                break
    except Exception as exc:
        status = "provider_or_session_error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        (hub.run / "transcript.json").write_text(json.dumps(transcript_copy(messages), indent=2))
    return {"status": status, "error": error, "rounds": rounds, "tool_calls": calls,
            "provider_wall_s": api_seconds, "agent_assessment": hub.finished,
            "real_model_used": client.metadata["kind"] != "scripted_transport_test"}


class ScriptedClient:
    """Deterministic protocol test; never described as LLM planning or paid API validation."""
    metadata = {"kind": "scripted_transport_test", "model": None, "network_requests": 0}

    def __init__(self):
        self.round = 0

    def complete(self, messages, tools):
        self.round += 1
        observations = [m for m in messages if m["role"] == "user"]
        latest = json.loads(observations[-1]["content"][0]["text"])
        step = latest["source_step"]
        if self.round == 1:
            name, args = "execute_plan", {"source_step": step, "steps": [
                {"primitive": "move_relative", "params": {"delta_m": [0,0,.02]}},
                {"primitive": "wait", "params": {"steps": 2}}]}
        elif self.round == 2:
            # Verify stale motion is rejected and the session can continue with fresh state.
            name, args = "move_relative", {"source_step": 0, "params": {"delta_m": [0,0,.02]}}
        else:
            name, args = "finish", {"source_step": step, "assessment": "completed",
                                    "reason": "Protocol smoke only: inspect real motion and stale-request logs; not task completion."}
        return {"message": {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"script_{self.round}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
                "usage": None, "finish_reason": "tool_calls"}
