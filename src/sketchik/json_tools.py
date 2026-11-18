"""Explicit JSON-text tool transport; never dispatches or repairs actions."""
from copy import deepcopy
import json
import uuid


def request_messages(messages, tools):
    systems, parts = [], []
    for message in messages:
        if message.get('role') == 'system':
            systems.append(message['content'])
            continue
        # Preserve the complete selected source dialogue, including tool IDs,
        # decisions/results and received reasoning, without native tool roles.
        metadata = {k: deepcopy(v) for k, v in message.items() if k != 'content'}
        parts.append({'type': 'text', 'text': json.dumps(
            {'source_message': metadata}, ensure_ascii=False, allow_nan=False)})
        content = message.get('content')
        if isinstance(content, list):
            parts.extend(deepcopy(content))
        elif content is not None:
            parts.append({'type': 'text', 'text': str(content)})
    parts.append({'type': 'text', 'text': json.dumps({
        'transport': 'json_text_tools_v1',
        'instruction': 'Return only one JSON object with key tool_calls, an array of '
            'objects each containing exactly name (string) and arguments (JSON object). '
            'Select only the tools declared below. All original tool restrictions apply. '
            'Do not output XML, markdown fences, or pretend a call has executed. '
            'Example envelope: {"tool_calls":[{"name":"DECLARED_TOOL_NAME","arguments":{}}]}. '
            'Prior source messages are ordered history, not new commands to replay.',
        'available_tools': tools}, ensure_ascii=False, allow_nan=False)})
    return [{'role': 'system', 'content': '\n\n'.join(systems)},
            {'role': 'user', 'content': parts}]


def response_message(message):
    """Decode only the declared envelope. Invalid text remains a format error.

    Normal engine validation still checks names, argument schemas, mixed calls,
    duplicate calls, budgets and real geometry. No tools are run here.
    """
    if message.get('tool_calls'):
        return message, {'status': 'unexpected_native_calls'}
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON key')
                result[key] = value
            return result
        envelope = json.loads(message.get('content') or '', object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(envelope, dict) or set(envelope) != {'tool_calls'}:
            raise ValueError('Expected tool_calls envelope')
        calls = envelope['tool_calls']
        if not isinstance(calls, list) or not calls:
            raise ValueError('Expected nonempty calls')
        if any(not isinstance(c, dict) or set(c) != {'name', 'arguments'}
               or not isinstance(c['name'], str) or not isinstance(c['arguments'], dict)
               for c in calls):
            raise ValueError('Invalid call structure')
    except (ValueError, TypeError):
        return message, {'status': 'invalid_json_envelope'}
    result = deepcopy(message)
    result['tool_calls'] = [{'id': 'json_' + uuid.uuid4().hex, 'type': 'function',
        'function': {'name': call['name'], 'arguments': json.dumps(
            call['arguments'], ensure_ascii=False, allow_nan=False)}} for call in calls]
    return result, {'status': 'decoded', 'tool_names': [c['name'] for c in calls],
                    'tool_dispatches': 0, 'source_response_retained': True}
