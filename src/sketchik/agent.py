"""Synchronous vision/tool conversation. No automatic network retries or action replay."""
from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import http.client
from io import BytesIO
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from PIL import Image

from .tools import ToolHub
from .request_cancellation import RequestCancelled


class ProviderError(RuntimeError):
    def __init__(self, message, *, response_metadata=None):
        super().__init__(message)
        self.response_metadata = deepcopy(response_metadata or {})


def http_error_metadata(error, *, api_key, request_hash, elapsed_s):
    """Record bounded categorical diagnostics, never a raw provider echo."""
    known = {'insufficient_quota', 'invalid_api_key', 'authentication_error',
        'permission_denied', 'forbidden', 'model_not_found', 'rate_limit_exceeded',
        'rate_limit_error', 'content_policy_violation', 'context_length_exceeded',
        'invalid_request_error', 'billing_hard_limit_reached', 'account_deactivated'}
    limit = 16_384
    record = {'status': 'http_error', 'http_status': int(error.code),
        'request_body_sha256': request_hash, 'wall_s': elapsed_s,
        'provider_error_code': None, 'provider_error_type': None,
        'error_body_read_limit_bytes': limit, 'automatic_retries': 0}
    try:
        raw = error.read(limit + 1)
        record.update(error_body_bytes_read=len(raw), error_body_truncated=len(raw)>limit)
        value = json.loads(raw) if len(raw)<=limit else None
        detail = value.get('error') if isinstance(value, dict) else None
        if isinstance(detail, dict):
            for name in ('code', 'type'):
                item = detail.get(name)
                if isinstance(item, str) and item.lower() in known and api_key not in item:
                    record['provider_error_'+name] = item.lower()
    except (ValueError, OSError, http.client.HTTPException):
        # Diagnostic reads cannot replace the original HTTP status with a
        # JSON/transport error. Unknown text and unrecognised fields stay out.
        pass
    finally:
        error.close()
    return record


class RejectedResponse(ProviderError):
    """A complete transport reply was rejected; never a retry instruction."""
    def __init__(self, finish_reason):
        self.finish_reason=finish_reason
        category='Truncated' if finish_reason=='length' else 'Filtered'
        super().__init__(f'{category} response (finish_reason={finish_reason}); tools not executed')


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
    def __init__(self, base_url, model, api_key, reasoning=None, timeout=90, *, use_env_proxy=True,
                 image_jpeg_quality=None, transport_dir=None, reference_exact_images=False,
                 image_window=None, max_tokens=None, stage_context=None, stream=False,
                 budget_field='max_tokens'):
        parsed = urlparse(base_url or "")
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query:
            raise ValueError("Expected API base URL without embedded credentials or query")
        if not model or not api_key:
            raise ValueError("Model and API key are required; nothing is sent without configuration")
        if image_jpeg_quality is not None and (type(image_jpeg_quality) is not int or not 1 <= image_jpeg_quality <= 100):
            raise ValueError("JPEG transport quality must be an integer in [1,100], or None for original images")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ValueError("API max_tokens must be a positive integer or None for provider default")
        if type(stream) is not bool or budget_field not in ('max_tokens','max_completion_tokens'):
            raise ValueError('Invalid API stream mode or output budget field')
        self.url = base_url.rstrip("/") + "/chat/completions"
        # Local gateways/tests must not be sent through a workstation HTTP proxy.
        direct = not use_env_proxy or parsed.hostname in ("localhost", "127.0.0.1", "::1")
        self._direct = direct
        self.opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                       if direct
                       else urllib.request.build_opener())
        self.model, self.api_key, self.reasoning, self.timeout = model, api_key, reasoning, timeout
        self.image_jpeg_quality = image_jpeg_quality
        self.max_tokens = max_tokens
        self.stream, self.budget_field = stream, budget_field
        self.on_stream_progress = None
        self.request_pause_handler = None
        self.request_cancellation = None
        self.attempt_observer = None
        self.operator_resumed_requests = 0
        self._stream_metrics = {}
        self.reference_exact_images = reference_exact_images
        self.transport_dir = Path(transport_dir) if transport_dir is not None else None
        self._jpeg_cache = {}
        self.last_request_metadata = {}
        self.last_response_metadata = {}
        self._last_image_references = []
        self.image_window = image_window
        self.stage_context = stage_context
        self._last_working_context = {'mode': 'full_history'}
        self.metadata = {"kind": "chat_completions", "base_url": base_url, "model": model,
                         "reasoning_effort": reasoning, "automatic_retries": 0, "timeout_s": timeout,
                         "requested_max_tokens": max_tokens,
                         "stream":stream,"budget_field":budget_field,
                         "stage_context": stage_context.limits if stage_context is not None else None,
                         "proxy_mode": "direct" if direct else "environment",
                         "reference_exact_images": reference_exact_images,
                         "working_context": ({'mode': 'bounded_images', **image_window.limits}
                                             if image_window is not None else {'mode': 'full_history'}),
                         "image_transport": {"format": "jpeg" if image_jpeg_quality is not None else "original",
                             "quality": image_jpeg_quality, "resize": False, "source_images_unchanged": True}}

    def wire_messages(self, messages):
        """Explicit transport-only encoding; original history/RGB-D remain untouched."""
        self._last_image_references = []
        self._last_working_context = {'mode': 'full_history'}
        full_source = messages
        stage_audit = None
        if self.image_window is not None:
            self.image_window.register_source(full_source)
        if self.stage_context is not None:
            messages, stage_audit = self.stage_context.apply(full_source)
        if self.image_jpeg_quality is None and not self.reference_exact_images and self.image_window is None:
            if stage_audit is not None:
                self._last_working_context = stage_audit
            return messages
        encoded = deepcopy(messages)
        if self.image_jpeg_quality is not None:
            for message in encoded:
                if not isinstance(message.get("content"), list):
                    continue
                for part in message["content"]:
                    if part.get("type") != "image_url":
                        continue
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/png;base64,"):
                        continue
                    identity = hashlib.sha256(url.encode()).hexdigest()
                    if identity not in self._jpeg_cache:
                        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                        with Image.open(BytesIO(raw)) as source:
                            rgb = source.convert("RGB")
                            buffer = BytesIO()
                            rgb.save(buffer, format="JPEG", quality=self.image_jpeg_quality,
                                     subsampling=0, optimize=True)
                        data = buffer.getvalue()
                        wire_url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
                        self._jpeg_cache[identity] = wire_url
                        if self.transport_dir is not None:
                            images = self.transport_dir / "images"
                            images.mkdir(parents=True, exist_ok=True)
                            file = images / (hashlib.sha256(wire_url.encode()).hexdigest() + ".jpg")
                            if not file.exists():
                                file.write_bytes(data)
                    part["image_url"]["url"] = self._jpeg_cache[identity]
        if self.image_window is not None and getattr(self.image_window, 'active', True):
            try:
                encoded, self._last_working_context = self.image_window.apply(messages, encoded,
                    deduplicate=self.reference_exact_images)
            except ValueError as exc:
                raise ProviderError(str(exc)) from None
        if stage_audit is not None:
            self._last_working_context = {**stage_audit, 'image_window': self._last_working_context}
        if self.image_window is not None:
            self.image_window.register_source(full_source)
        if self.reference_exact_images:
            seen = {}
            for message_index, message in enumerate(encoded):
                if not isinstance(message.get("content"), list):
                    continue
                content = []
                for part_index, part in enumerate(message["content"]):
                    if part.get("type") != "image_url":
                        content.append(part)
                        continue
                    # Compare source parts, so JPEG encoding cannot merge distinct images.
                    source_part = messages[message_index]["content"][part_index]
                    identity = json.dumps(source_part, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    if identity not in seen:
                        image_id = f"image_ref_{len(seen) + 1:04d}"
                        seen[identity] = (image_id, [message_index, part_index])
                        content.append({"type": "text", "text": f"Image ID {image_id}:"})
                        content.append(part)
                    else:
                        image_id, canonical = seen[identity]
                        content.append({"type": "text", "text": f"Image reference {image_id}."})
                        self._last_image_references.append({"image_id": image_id,
                            "original_position": [message_index, part_index],
                            "canonical_position": canonical})
                message["content"] = content
        return encoded

    def _read_stream(self, response, started, trace_path):
        """Assemble one complete SSE reply before exposing any tool call."""
        chunks=0;reasoning=[];content=[];calls={};usage=None;finish=None;first=None;last_progress=0.
        trace=trace_path.open('wb') if trace_path is not None else None
        reasoning_chars=content_chars=0;fallback=False
        try:
            for line in response:
                if trace is not None:
                    trace.write(line.replace(self.api_key.encode(),b'[redacted]'));trace.flush()
                if self.request_cancellation is not None:
                    self.request_cancellation.check()
                if not line.strip() or line.startswith(b':'):
                    continue
                if not line.startswith(b'data:'):
                    # Some compatible gateways ignore stream=True and return a complete JSON body.
                    if not chunks and line.lstrip().startswith(b'{'):
                        tail=self._read_body(response, trace)
                        reply=json.loads(line+tail)
                        fallback=True
                        choices=reply.get('choices') or []
                        complete=bool(choices and choices[0].get('finish_reason') is not None)
                        self._stream_metrics={'mode':'json_fallback','stream_complete':complete}
                        return reply
                    continue
                raw=line[5:].strip()
                if raw==b'[DONE]':break
                event=json.loads(raw);chunks+=1
                if event.get('error'):
                    raise ProviderError('Provider returned an SSE error; no automatic retry')
                if event.get('usage') is not None:usage=event['usage']
                for choice in event.get('choices',[]):
                    if choice.get('index',0)!=0:continue
                    if choice.get('finish_reason') is not None:finish=choice['finish_reason']
                    delta=choice.get('delta',{})
                    generated=any(delta.get(k) for k in ('content','reasoning_content','tool_calls'))
                    if generated and first is None:first=time.perf_counter()-started
                    text=delta.get('reasoning_content') or '';reasoning.append(text);reasoning_chars+=len(text)
                    text=delta.get('content') or '';content.append(text);content_chars+=len(text)
                    for part in delta.get('tool_calls') or []:
                        call=calls.setdefault(part['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                        call['id']+=part.get('id') or ''
                        for key in ('name','arguments'):
                            call['function'][key]+=part.get('function',{}).get(key) or ''
                    elapsed=time.perf_counter()-started
                    if generated and self.on_stream_progress is not None and (not last_progress or elapsed-last_progress>=5):
                        self.on_stream_progress({'elapsed_s':elapsed,'first_generation_s':first,
                            'reasoning_chars':reasoning_chars,'content_chars':content_chars,'chunks':chunks})
                        last_progress=elapsed
        finally:
            if trace is not None:trace.close()
            if not fallback:
                self._stream_metrics={'mode':'sse','chunks':chunks,'first_generation_s':first,
                    'reasoning_chars':reasoning_chars,'content_chars':content_chars,'stream_complete':finish is not None}
            self._partial_stream = {'message': {'role': 'assistant', 'content': ''.join(content),
                'reasoning_content': ''.join(reasoning), 'tool_calls': [calls[i] for i in sorted(calls)]},
                'usage': usage}
        message={'role':'assistant','content':''.join(content)}
        if reasoning_chars:message['reasoning_content']=''.join(reasoning)
        if calls:message['tool_calls']=[calls[i] for i in sorted(calls)]
        return {'choices':[{'message':message,'finish_reason':finish}],'usage':usage}

    def _read_body(self, response, trace=None):
        chunks = []
        read = getattr(response, 'read1', response.read)
        while True:
            if self.request_cancellation is not None:
                self.request_cancellation.check()
            chunk = read(65536)
            if not chunk:
                break
            if trace is not None:
                trace.write(chunk.replace(self.api_key.encode(), b'[redacted]'))
                trace.flush()
            chunks.append(chunk)
        return b''.join(chunks)

    def complete(self, messages, tools=None):
        """Retry only after an installed handler receives an explicit operator resume."""
        while True:
            try:
                from contextlib import nullcontext
                self.last_request_metadata = {}
                self.last_response_metadata = {}
                self._stream_metrics = {}
                self._partial_stream = {}
                self._stream_trace_path = None
                with self.attempt_observer(self) if self.attempt_observer is not None else nullcontext():
                    try:
                        if self.request_cancellation is not None:
                            self.request_cancellation.check()
                        result = self._complete_once(messages, tools)
                        if self.request_cancellation is not None:
                            self.request_cancellation.check()
                        return result
                    except RequestCancelled:
                        token = self.request_cancellation
                        if token is None or not token.event.is_set():
                            raise
                        self.last_response_metadata = {'status': 'cancelled',
                            'reason': token.reason, 'automatic_retries': 0,
                            'request_body_sha256': self.last_request_metadata.get('body_sha256'),
                            'stream_metrics': self._stream_metrics,
                            'stream_trace_file': self._stream_trace_path,
                            'upstream_cancellation_confirmed': False,
                            **self._partial_stream}
                        if self.transport_dir is not None:
                            self.transport_dir.mkdir(parents=True, exist_ok=True)
                            with (self.transport_dir/'responses.jsonl').open('a') as archive:
                                archive.write(json.dumps(self.last_response_metadata, allow_nan=False)+'\n')
                        raise RequestCancelled('Obsolete advance planning request cancelled') from None
            except RejectedResponse:
                raise  # A completed length/filter refusal is not a broken connection.
            except ProviderError as exc:
                if self.request_pause_handler is None:
                    raise
                self.request_pause_handler(exc, messages, tools)
                self.operator_resumed_requests += 1

    def _complete_once(self, messages, tools=None):
        started = time.perf_counter()
        self.last_response_metadata = {}
        self._stream_metrics = {}
        wire = self.wire_messages(messages)
        payload = {"model": self.model, "messages": wire}
        if tools is not None:
            payload.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        if self.reasoning:
            payload["reasoning_effort"] = self.reasoning
        if self.max_tokens is not None:
            payload[self.budget_field] = self.max_tokens
        if self.stream:
            payload.update(stream=True,stream_options={'include_usage':True})
        body = json.dumps(payload, allow_nan=False).encode()
        original_image_count = sum(part.get("type") == "image_url" for message in messages
            if isinstance(message.get("content"), list) for part in message["content"])
        self.last_request_metadata = {"body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
            "image_count": sum(part.get("type") == "image_url" for message in wire
                if isinstance(message.get("content"), list) for part in message["content"]),
            "original_image_count": original_image_count,
            "image_references": self._last_image_references,
            "working_context": self._last_working_context,
            "reasoning_history_chars": sum(len(m.get('reasoning_content') or '') for m in wire),
            "requested_max_tokens": self.max_tokens,
            "stream":self.stream,"budget_field":self.budget_field,
            "image_encoding_and_serialization_s": time.perf_counter()-started,
            "image_transport": self.metadata["image_transport"]}
        if self.transport_dir is not None:
            self.transport_dir.mkdir(parents=True, exist_ok=True)
            with (self.transport_dir / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps({**self.last_request_metadata,
                    "messages": transcript_copy(wire),"tools":tools,
                    "request_options":{k:v for k,v in payload.items() if k not in ('messages','tools')}}, allow_nan=False) + "\n")
        request = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        # Provider bodies may echo auth/inputs; retain only whitelisted categories.
        try:
            sent=time.perf_counter()
            token = self.request_cancellation
            if token is not None:
                token.check()
            opener = token.opener(direct=self._direct) if token is not None else self.opener
            with opener.open(request, timeout=self.timeout) as response:
                path=None
                if self.stream and self.transport_dir is not None:
                    folder=self.transport_dir/'streams';folder.mkdir(exist_ok=True)
                    path=folder/(str(time.time_ns())+'_'+self.last_request_metadata['body_sha256']+'.sse')
                    self._stream_trace_path = str(path)
                if self.stream:
                    reply = self._read_stream(response, sent, path)
                elif token is not None and self.transport_dir is not None:
                    folder = self.transport_dir/'streams'; folder.mkdir(exist_ok=True)
                    path = folder/(str(time.time_ns())+'_'+self.last_request_metadata['body_sha256']+'.body')
                    self._stream_trace_path = str(path)
                    with path.open('wb') as trace:
                        reply = json.loads(self._read_body(response, trace))
                else:
                    reply = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            self.last_response_metadata = http_error_metadata(exc, api_key=self.api_key,
                request_hash=self.last_request_metadata['body_sha256'], elapsed_s=time.perf_counter()-sent)
            if self.transport_dir is not None:
                with (self.transport_dir/'responses.jsonl').open('a') as archive:
                    archive.write(json.dumps(self.last_response_metadata, allow_nan=False)+'\n')
            raise ProviderError(f"Provider HTTP {exc.code}; no automatic retry",
                response_metadata=self.last_response_metadata) from None
        except (urllib.error.URLError, OSError, json.JSONDecodeError, http.client.HTTPException) as exc:
            if self.request_cancellation is not None:
                self.request_cancellation.check()
            reason = getattr(exc, "reason", exc)
            # Only record exception classes/numeric errno, never raw network messages.
            # URLError.reason can contain URLs or provider-echoed private data.
            detail = f"{type(exc).__name__}; reason_type={type(reason).__name__}"
            if isinstance(reason, OSError) and isinstance(reason.errno, int):
                detail += f"; errno={reason.errno}"
            raise ProviderError(f"Provider transport/JSON failure ({detail}); no automatic retry") from None
        if self.request_cancellation is not None:
            self.request_cancellation.check()
        choices = reply.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("Provider returned no choices")
        choice = choices[0]
        self.last_response_metadata = {'finish_reason':choice.get('finish_reason'),
            'usage':reply.get('usage'),'request_body_sha256':self.last_request_metadata['body_sha256'],
            'stream_metrics':self._stream_metrics,'stream_trace_file':str(path) if self.stream and path is not None else None}
        if self.transport_dir is not None:
            with (self.transport_dir/'responses.jsonl').open('a') as stream:
                stream.write(json.dumps({**self.last_response_metadata,
                    'message':choice.get('message')},allow_nan=False)+'\n')
        if self.stream and not self._stream_metrics.get('stream_complete'):
            raise ProviderError('Incomplete streamed response; tools not executed; no automatic retry')
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise RejectedResponse(choice['finish_reason'])
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


class TrialClient:
    def __init__(self, client, emit):
        self.client, self.emit, self.round = client, emit, 0

    def complete(self, messages, tools):
        self.round += 1
        start = time.perf_counter()
        self.emit({'event': 'api_started', 'round': self.round})
        try:
            result = self.client.complete(messages, tools)
        except Exception as exc:
            self.emit({'event': 'api_finished', 'round': self.round, 'wall_s': time.perf_counter()-start,
                       'error_type': type(exc).__name__, 'request': self.client.last_request_metadata,
                       'provider_response':getattr(self.client,'last_response_metadata',{})})
            raise
        self.emit({'event': 'api_finished', 'round': self.round, 'wall_s': time.perf_counter()-start,
                   'request': self.client.last_request_metadata, 'usage': result.get('usage')})
        return result
