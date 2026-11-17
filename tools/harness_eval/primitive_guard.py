"""Whole-response validation before the native SDK dispatches any tool.

This is an explicit evaluation adapter, not a change to the native source or
to XML parsing. Rejected replies remain in SDK history and use ModelRetry.
"""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from pydantic_ai import ModelRetry, ToolCallPart
from pydantic_ai.messages import ModelMessagesTypeAdapter


class ResponseGuard:
    def __init__(self, primitive_names, audit_path):
        self.primitive_names = frozenset(primitive_names)
        self.audit_path = Path(audit_path)

    def check(self, response):
        calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
        issues, seen = [], set()
        for index, call in enumerate(calls):
            if call.tool_name not in self.primitive_names:
                continue
            try:
                arguments = call.args_as_dict(raise_if_invalid=True)
                encoded = json.dumps(arguments, sort_keys=True, separators=(',', ':'), allow_nan=False)
            except (ValueError, TypeError, AssertionError):
                # Leave malformed arguments to the original SDK/tool validator.
                continue
            key = (call.tool_name, encoded)
            if key in seen:
                issues.append(f'Call {index + 1}: duplicate {call.tool_name} with identical arguments.')
            seen.add(key)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        archived = None
        if issues:
            # Native SDK history preserves this response, but RPent's custom
            # transcript only sees dispatched CallToolsNode replies. Preserve a
            # complete private copy here too, without inventing tool receipts.
            private = self.audit_path.parent / 'private' / 'primitive_guard'
            private.mkdir(parents=True, exist_ok=True, mode=0o700)
            archived = private / ('rejected_' + uuid.uuid4().hex + '.json')
            with os.fdopen(os.open(archived, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'wb') as stream:
                stream.write(ModelMessagesTypeAdapter.dump_json([response]))
        record = {
            'at_unix': time.time(), 'profile': 'reject-duplicate-primitives-v1',
            'decision': 'reject' if issues else 'pass',
            'tool_names': [c.tool_name for c in calls],
            'tool_call_ids': [c.tool_call_id for c in calls],
            'issues': issues,
            'rejected_response_archive': str(archived) if archived else None,
            'response_parts_sha256': hashlib.sha256(repr(response.parts).encode()).hexdigest(),
            'guard_dispatches_tools': False,
            'rejected_response_dispatched_tools': 0 if issues else None,
        }
        with self.audit_path.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        if issues:
            message = ('Protocol rejection: NO tools from this response were executed. '
                       + ' '.join(issues)
                       + ' Resubmit without duplicate physical primitive calls in this response. '
                         'This is a format correction, not a network failure.')
            raise ModelRetry(message)
        return response


def guarded_session_class(native_session_class, audit_path, is_primitive_action):
    """Extend the native lifecycle hook; the SDK owns history and retry budgets."""
    class GuardedSession(native_session_class):
        async def after_model_request(self, ctx, *, request_context, response):
            response = await super().after_model_request(
                ctx, request_context=request_context, response=response)
            if not hasattr(self, '_primitive_response_guard'):
                primitives = {
                    item.name
                    for item in self.toolkit.toolkit.list_tools()
                    if is_primitive_action(item.name)
                }
                self._primitive_response_guard = ResponseGuard(
                    primitives, audit_path)
            return self._primitive_response_guard.check(response)
    return GuardedSession
