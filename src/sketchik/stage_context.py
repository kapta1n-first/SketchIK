"""Audited API working contexts; source history is append-only.

No model summarizer or long-term memory. A completed tool dialogue can start a
new wire epoch from fixed instructions, initial evidence and program state.
Single-agent windows also retain recent complete tool dialogues, without
forcing a reset when acquisition ends. Retained messages are never edited.
Checkpoint markers make the active view recoverable from the source transcript.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


MARKER = 'sketchik_stage_context'
EPISODE_INSTRUCTIONS = 'SketchIK persistent episode instructions'
EPISODE_REFERENCES = 'single_A_episode_references'
ACTUAL_IMAGES = 'single_A_actual_images'
CURRENT_STATE = 'single_A_direct_execution'
ACQUISITION_RECORD = 'initial_acquisition_telemetry'


def message_phase(message):
    content = message.get('content')
    if message.get('role') != 'user' or not isinstance(content, list) or not content:
        return None
    first = content[0]
    try:
        value = json.loads(first.get('text', '')) if first.get('type') == 'text' else {}
    except (ValueError, TypeError):
        return None
    return value.get('phase') if isinstance(value, dict) else None


def archived_messages(messages):
    result = deepcopy(messages)
    for message in result:
        if isinstance(message.get('content'), list):
            for part in message['content']:
                if part.get('type') == 'image_url':
                    url = part['image_url']['url']
                    part['image_url']['url'] = 'saved-observation-image-sha256:' + hashlib.sha256(url.encode()).hexdigest()
    return result


def metrics(messages):
    public = archived_messages(messages)
    return {'messages': len(messages), 'text_chars': len(json.dumps(public, ensure_ascii=False)),
        'reasoning_chars': sum(len(m.get('reasoning_content') or '') for m in messages),
        'image_occurrences': sum(p.get('type') == 'image_url' for m in messages
            if isinstance(m.get('content'), list) for p in m['content'])}


def closed_tool_dialogue(messages):
    pending = set()
    for message in messages:
        role = message.get('role')
        if role == 'tool':
            identity = message.get('tool_call_id')
            if identity not in pending:
                raise ValueError('Cannot checkpoint an orphan tool result')
            pending.remove(identity)
        elif role == 'assistant':
            if pending:
                raise ValueError('Cannot checkpoint an incomplete tool batch')
            pending.update(call['id'] for call in message.get('tool_calls') or [])
        elif pending:
            raise ValueError('Cannot checkpoint before every tool result is recorded')
    if pending:
        raise ValueError('Cannot checkpoint unresolved tool calls')


class StageContext:
    def __init__(self, max_chars=120_000, max_messages=80, *, retain_robot_facts=True,
                 keep_messages=0, keep_chars=None, retain_episode_references=False):
        if any(type(value) is not int or value <= 0 for value in (max_chars, max_messages)):
            raise ValueError('Stage context limits must be positive')
        if any(type(value) is not bool for value in (retain_robot_facts, retain_episode_references)):
            raise ValueError('Context retention flags must be boolean')
        if (type(keep_messages) is not int or not 0 <= keep_messages < max_messages
                or (keep_chars is not None and (type(keep_chars) is not int
                    or not 0 < keep_chars < max_chars))):
            raise ValueError('Retained context limits must be below trigger limits')
        self.limits = {'max_chars': max_chars, 'max_messages': max_messages,
            'retain_robot_facts': retain_robot_facts,
            'retain_episode_references': retain_episode_references,
            'keep_messages': keep_messages, 'keep_chars': keep_chars,
            'measurement': 'serialized text characters with image payloads hashed; not a tokenizer'}

    def working_indices(self, source, indices):
        if not self.limits['retain_episode_references']:
            return indices
        # Full source remains append-only. Only the newest explicit numeric
        # state and scene binding snapshot belong in the working request.
        latest = {phase: max((i for i in indices if message_phase(source[i]) == phase), default=None)
            for phase in (CURRENT_STATE, EPISODE_REFERENCES)}
        execution = latest[CURRENT_STATE] is not None
        return [i for i in indices if
            not (execution and (message_phase(source[i]) == ACQUISITION_RECORD or
                (not self.limits['retain_robot_facts'] and source[i].get('role') == 'user'
                    and isinstance(source[i].get('content'), str)
                    and '"robot_physical_facts"' in source[i]['content'][:100])))
            and (message_phase(source[i]) not in latest or latest[message_phase(source[i])] == i)]

    def apply(self, source):
        latest = None
        for index, message in enumerate(source):
            value = message.get('content')
            if message.get('role') != 'user' or not isinstance(value, str) or MARKER not in value[:80]:
                continue
            try:
                candidate = json.loads(value)
            except ValueError:
                continue
            if isinstance(candidate, dict) and MARKER in candidate:
                latest = (index, candidate[MARKER])
        if latest is None:
            indices = self.working_indices(source, list(range(len(source))))
            active = [source[i] for i in indices]
            return active, {'mode': 'stage_context', 'epoch': 0,
                'source': metrics(source), 'active': metrics(active),
                'omitted_source_messages': len(source)-len(indices), 'active_source_indices': indices}
        boundary, marker = latest
        indices = marker['fixed_prefix_indices']
        tail = marker.get('retained_tail_indices', [])
        if (marker['boundary_message_index'] != boundary
                or any(type(i) is not int or i < 0 or i >= boundary for i in indices + tail)
                or len(indices + tail) != len(set(indices + tail))
                or tail != sorted(tail)):
            raise ValueError('Stage checkpoint does not match its source transcript')
        active_indices = self.working_indices(source, indices + tail + list(range(boundary, len(source))))
        active = [source[i] for i in active_indices]
        closed_tool_dialogue(active)
        return active, {'mode': 'stage_context', 'epoch': marker['epoch'],
            'boundary_source_step': marker['source_step'], 'source': metrics(source),
            'active': metrics(active), 'omitted_source_messages': len(source)-len(active_indices),
            'retained_tail_messages': len(tail), 'active_source_indices': active_indices,
            'source_history_changed': False, 'retained_tool_reasoning_unchanged': True}

    def prepare_boundary(self, messages, checkpoint, folder, emit, *, force=False):
        active, current = self.apply(messages)
        before = metrics(active)
        if not force and before['text_chars'] <= self.limits['max_chars'] and before['messages'] <= self.limits['max_messages']:
            return None
        closed_tool_dialogue(active)
        # Initial instructions, initial dual RGB, and known robot facts keep a
        # stable prefix. Dynamic phase prompts arrive after the checkpoint.
        indices = []
        roots=[i for i,m in enumerate(messages) if m.get('role')=='system'
            and isinstance(m.get('content'),str) and m['content'].startswith(EPISODE_INSTRUCTIONS+'\n')]
        if roots:
            indices.append(roots[-1])
        elif messages and messages[0].get('role') == 'system':
            indices.append(0)
        initial = next((i for i, m in enumerate(messages) if m.get('role') == 'user'), None)
        if initial is not None:
            indices.append(initial)
        robot = next((i for i, m in enumerate(messages) if m.get('role') == 'user'
            and isinstance(m.get('content'), str) and '"robot_physical_facts"' in m['content'][:100]), None)
        if robot is not None and self.limits['retain_robot_facts']:
            indices.append(robot)
        latest_actual_images = latest_current_state = latest_references = None
        if self.limits['retain_episode_references']:
            for index, item in enumerate(messages):
                content = item.get('content')
                if item.get('role') != 'user' or not isinstance(content, list) or not content:
                    continue
                first = content[0]
                try:
                    tag = json.loads(first.get('text', '')) if first.get('type') == 'text' else {}
                except (ValueError, TypeError):
                    continue
                if isinstance(tag, dict) and tag.get('phase') == EPISODE_REFERENCES:
                    latest_references = index
                if isinstance(tag, dict) and tag.get('phase') == ACTUAL_IMAGES:
                    latest_actual_images = index
                if isinstance(tag, dict) and tag.get('phase') == CURRENT_STATE:
                    latest_current_state = index
            if latest_references is not None:
                indices.append(latest_references)
        # The most recent permanent instructions lead the working request,
        # even when they were appended during a verified episode continuation.
        indices = list(dict.fromkeys(indices))
        tail = []
        if self.limits['keep_messages']:
            candidates = [i for i in current['active_source_indices'] if i not in indices
                and i != latest_current_state
                and messages[i].get('role') != 'system'
                and not (messages[i].get('role') == 'user'
                    and isinstance(messages[i].get('content'), str)
                    and MARKER in messages[i]['content'][:80])]
            # A suffix may start only outside a tool batch. Retain its whole
            # assistant request and all results even if it exceeds the target.
            pending = set(); starts = []; last_tool_start = None
            for offset, i in enumerate(candidates):
                if not pending:
                    starts.append(offset)
                message = messages[i]
                if message.get('role') == 'assistant':
                    if message.get('tool_calls'):
                        last_tool_start = offset
                    pending.update(call['id'] for call in message.get('tool_calls') or [])
                elif message.get('role') == 'tool':
                    pending.remove(message['tool_call_id'])
            if candidates:
                desired = max(0, len(candidates) - self.limits['keep_messages'])
                if last_tool_start is not None:
                    desired = min(desired, last_tool_start)
                cutoff = max(start for start in starts
                    if start <= desired)
                tail = candidates[cutoff:]
                if self.limits['keep_chars'] is not None:
                    for start in [start for start in starts if start >= cutoff
                            and (last_tool_start is None or start <= last_tool_start)]:
                        tail = candidates[start:]
                        if metrics([messages[i] for i in tail])['text_chars'] <= self.limits['keep_chars']:
                            break
        if latest_actual_images is not None and latest_actual_images not in indices:
            # This independent user message has no tool pairing. Keep its
            # original position relative to retained decisions/results.
            tail = sorted(set(tail) | {latest_actual_images})
        if latest_current_state is not None and latest_current_state not in indices:
            tail = sorted(set(tail) | {latest_current_state})
        checkpoint_state = checkpoint
        if latest_current_state is not None:
            # CURRENT already contains the full measured state once. Repeating
            # it here displaced earlier complete model decisions every epoch.
            checkpoint_state = {key: checkpoint[key] for key in
                ('source_step', 'observation_id', 'original_task') if key in checkpoint}
            checkpoint_state['current_state_source'] = 'Latest single_A_direct_execution CURRENT message and current.json'
        source_public = archived_messages(messages)
        marker = {'schema_version': 2, 'epoch': current['epoch'] + 1,
            'parent_epoch': current['epoch'], 'source_step': checkpoint['source_step'],
            'boundary_message_index': len(messages), 'fixed_prefix_indices': indices,
            'retained_tail_indices': tail,
            'source_sha256': hashlib.sha256(json.dumps(source_public, sort_keys=True).encode()).hexdigest(),
            'limits': self.limits, 'previous_active': before}
        message = {'role': 'user', 'content': json.dumps({MARKER: marker,
            'instruction': 'Explicit working-context transition within this same A episode. Older complete dialogues remain archived; the recent complete dialogues above are retained verbatim. The structured state below is program evidence, not a model reasoning transcript or task-success certification. Continue from current real outcome, never replay completed actions. Available interfaces are only those in the current tool schema. Old observations and known robot calibration remain in the episode archive; use the available history/file tools to retrieve them when needed.',
            'episode_state': checkpoint_state}, ensure_ascii=False, allow_nan=False)}
        retained = [messages[i] for i in indices + tail]
        closed_tool_dialogue(retained)
        after = metrics(retained + [message])
        if after['text_chars'] >= before['text_chars']:
            emit({'event': 'a_context_checkpoint_skipped', 'reason': 'Structured state would not reduce active text',
                'before': before, 'after': after})
            return None
        path = Path(folder) / f"epoch_{marker['epoch']:04d}"
        path.mkdir(parents=True, exist_ok=False)
        for name, value in [('source_before.json', source_public), ('active_before.json', archived_messages(active)),
                ('checkpoint.json', json.loads(message['content'])),
                ('active_after.json', archived_messages(retained + [message]))]:
            (path / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
        messages.append(message)
        emit({'event': 'a_context_checkpoint', **marker, 'after': after,
            'source_history_retained': True, 'model_summary_requests': 0})
        return marker

