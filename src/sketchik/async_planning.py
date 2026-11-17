"""One A, serial model dialogue, with one conditional draft during execution.

Only the API runs in the worker. Robot callbacks and request recovery stay on
the owner thread. Normal complete sketches continue without an approval turn;
actual execution or binding failures return to the same A.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from queue import Empty, Queue
from threading import Event, Thread, get_ident
import time

from jsonschema import Draft202012Validator, ValidationError

from .tools import obj
from .request_cancellation import RequestCancellation, RequestCancelled


def draft_tool(batch):
    text = {'type': 'string', 'minLength': 1}
    schema = obj({'status': {'enum': ['ready', 'finish_review']}, 'reason': text,
        'preconditions': text, 'intent': text, 'actions': deepcopy(batch['properties']['actions'])},
        ['status', 'reason'])
    schema['allOf'] = [{'if': {'properties': {'status': {'const': 'ready'}}},
        'then': {'required': ['preconditions', 'intent', 'actions']},
        'else': {'not': {'anyOf': [{'required': [key]} for key in ('preconditions', 'intent', 'actions')]}}}]
    return {'type': 'function', 'function': {'name': 'draft_next_sketch',
        'description': 'Save the complete next-subtask sketch using existing geometry; normal completed execution '
            'continues it directly, including gripper and articulated actions. Preconditions are planning notes, '
            'not an extra approval gate. Future outcomes remain unknown. Only when the executing sketch is '
            'expected to finish the entire task, use finish_review to request normal tools on the actual outcome '
            'for finish or repair. This is not defer or a success declaration; do not invent wait actions. '
            'No robot or file tools execute in this phase.', 'parameters': schema}}


def draft_context_start(messages, candidate):
    """Pin the original draft dialogue during recovery or terminal assessment."""
    for index in range(len(messages)-1, -1, -1):
        message = messages[index]
        if message.get('role') not in ('tool', 'user') or not isinstance(message.get('content'), str):
            continue
        try:
            value = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict) or not value.get('saved') or value.get('candidate_id') != candidate['candidate_id']:
            continue
        call_id = message.get('tool_call_id')
        for previous in range(index-1, -1, -1):
            source = messages[previous]
            if source.get('role') != 'assistant':
                continue
            calls = source.get('tool_calls') or []
            if call_id and not any(call.get('id') == call_id and
                    call.get('function', {}).get('name') == 'draft_next_sketch' for call in calls):
                continue
            return previous
    raise ValueError('Pending candidate has no original draft dialogue; no action dispatched')


def parse_draft(response, tool, source_step, batch, *, preparation_snapshot=None):
    message = response['message']
    calls = message.get('tool_calls') or []
    if calls:
        if len(calls) != 1 or calls[0]['function']['name'] != 'draft_next_sketch':
            raise ValueError('Draft phase permits one draft_next_sketch only; no tools dispatched')
        value = json.loads(calls[0]['function']['arguments'])
    else:
        value = json.loads(message.get('content') or '')
    Draft202012Validator(tool['function']['parameters']).validate(value)
    frozen = {**value, 'prepared_from_step': source_step, 'after_batch': batch}
    if preparation_snapshot is not None:
        frozen['preparation_snapshot'] = deepcopy(preparation_snapshot)
    identity = hashlib.sha256(json.dumps(frozen, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {**frozen, 'candidate_id': identity, 'executed': False}


def continuation_snapshot(hub):
    """Owner-thread snapshot; expectations never become actual A assessments."""
    from .carry_reference import report
    episode = hub.episode_state.data
    reference = getattr(hub, 'carried_surface_reference', None)
    return {'run': str(hub.run.resolve()), 'source_step': hub.robot.steps,
        'request_turn': hub.single_a_requests, 'grasp_epoch': episode.get('grasp_epoch', 0),
        'reference': report(reference) if reference else None}


def check_continuation(candidate, current, receipt, *, batch, snapshot, episode, consumed):
    """Normal execution permission, with only actual faults blocking dispatch.

    New grasp epochs, missing semantic certification, occlusion, and gripper or
    articulated actions do not add a model approval turn. prepare_actions and
    the real executor still reject any actually used stale/unavailable binding.
    """
    checks = []
    def add(name, passed, **evidence):
        checks.append({'check': name, 'status': 'pass' if passed else 'fail', **deepcopy(evidence)})
    before = candidate.get('preparation_snapshot') or {}
    identity = {k: v for k, v in candidate.items() if k not in ('candidate_id', 'executed')}
    intact = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    add('candidate_identity', intact == candidate['candidate_id'] and
        candidate['candidate_id'] not in consumed and not candidate.get('executed'))
    step = current.get('source_step')
    add('actual_boundary', bool(before) and before.get('run') == snapshot['run'] and
        candidate.get('after_batch') == batch and before.get('request_turn') == snapshot['request_turn'] and
        before.get('source_step') == candidate.get('prepared_from_step') and isinstance(step, int) and
        isinstance(before.get('source_step'), int) and
        before['source_step'] <= step == receipt.get('source_step') == snapshot['source_step'],
        source_step=step, batch=batch)
    add('execution', receipt.get('commands_completed') is True and
        not any(item.get('result', {}).get('converged') is False for item in receipt.get('steps', [])))
    add('contact_warning', not (current.get('contact_warning') or episode.get('contact_warning') or
        episode.get('carry_premise') == 'invalidated_by_contact_change'))
    held = current.get('geometry', {}).get('carried_observed_geometry') or {}
    support = held.get('visible_support') or {}
    # Missing/occluded support is unverified information. A measured current
    # contradiction is an actual reference fault, independently of semantics.
    add('current_depth_contradiction', support.get('status') != 'contradicted', measured_support=support)
    reference = snapshot.get('reference') or {}
    contradiction = (reference.get('validity') == 'suspended' and
        reference.get('suspension_kind') == 'depth_contradiction' and
        reference.get('reference_id') is not None and
        held.get('reference_id') == reference['reference_id'] and
        held.get('grasp_epoch') == reference.get('grasp_epoch') == snapshot.get('grasp_epoch') and
        isinstance(reference.get('suspended_at_step'), int) and isinstance(step, int) and
        reference['suspended_at_step'] <= step)
    add('retained_depth_contradiction', not contradiction,
        suspended_reference=reference if contradiction else None)
    return {'candidate_id': candidate['candidate_id'], 'source_step': step,
        'status': 'pass' if all(c['status'] == 'pass' for c in checks) else 'fail', 'checks': checks,
        'grasp_epoch_transition': {'prepared': before.get('grasp_epoch'), 'actual': snapshot.get('grasp_epoch'),
            'semantic_certification_required_for_dispatch': False},
        'holding_certified': False, 'task_success_certified': False, 'collision_checked': False,
        'scope': 'Normal continuation without A approval; actual used bindings and real IK are still checked.'}


class OverlapRequest:
    """A single worker request; all retry/pause handlers run on the owner thread."""
    def __init__(self, client, messages, tool, folder):
        self.client, self.messages, self.tool = client, deepcopy(messages), deepcopy(tool)
        self.folder = folder
        self.owner = get_ident()
        self.pauses = Queue()
        self.cancelled = Event()
        self.cancellation = RequestCancellation()
        self.previous_cancellation = getattr(client, 'request_cancellation', None)
        self.handler = getattr(client, 'request_pause_handler', None)
        self.started = self.ended = None
        self.response = self.error = None
        self.metadata = {}
        self.thread = Thread(target=self._run, name='single-A-next-sketch', daemon=True)

    def _run(self):
        self.started = time.perf_counter()
        try:
            self.response = self.client.complete(self.messages, [self.tool])
        except BaseException as exc:
            self.error = exc
        finally:
            self.metadata = deepcopy(getattr(self.client, 'last_request_metadata', {}))
            self.ended = time.perf_counter()

    def _pause(self, error, messages, tools):
        if self.cancelled.is_set():
            raise RequestCancelled('Draft cancelled after execution stopped')
        item = {'error': error, 'messages': messages, 'tools': tools, 'done': Event()}
        self.pauses.put(item)
        item['done'].wait()
        if 'exception' in item:
            raise item['exception']

    def pump(self, *, cancel=False):
        if get_ident() != self.owner:
            raise RuntimeError('Async request recovery must run on the robot owner thread')
        while True:
            try:
                item = self.pauses.get_nowait()
            except Empty:
                return
            if cancel:
                item['exception'] = RequestCancelled('Draft cancelled after execution stopped')
                item['done'].set()
                continue
            try:
                self.handler(item['error'], item['messages'], item['tools'])
            except BaseException as exc:
                item['exception'] = exc
            finally:
                item['done'].set()
            if 'exception' in item:
                raise item['exception']

    @contextmanager
    def during_execution(self, robot):
        old_callback = getattr(robot, 'on_step', None)
        def on_step(state):
            if old_callback is not None:
                old_callback(state)
            self.pump()
        if self.handler is not None:
            self.client.request_pause_handler = self._pause
        robot.on_step = on_step
        self.client.request_cancellation = self.cancellation
        execution_start = time.perf_counter()
        self.thread.start()
        execution_error = False
        try:
            yield
        except BaseException:
            execution_error = True
            self.cancel('execution_program_error')
            raise
        finally:
            execution_end = time.perf_counter()
            try:
                while self.thread.is_alive():
                    self.pump(cancel=execution_error or self.cancelled.is_set())
                    self.thread.join(.05)
            except BaseException:
                # A pause handler has signalled the waiting worker before raising.
                # Do not restore a handler while that same request can still use it.
                self.thread.join()
                raise
            finally:
                robot.on_step = old_callback
                self.client.request_cancellation = self.previous_cancellation
                if self.handler is not None:
                    self.client.request_pause_handler = self.handler
            self.metrics = {'request_wall_s': self.ended-self.started,
                'execution_wall_s': execution_end-execution_start,
                'overlap_wall_s': max(0., min(self.ended, execution_end)-max(self.started, execution_start)),
                'post_execution_wait_wall_s': max(0., self.ended-execution_end),
                'cancellation_requested': self.cancelled.is_set(),
                'cancellation_reason': self.cancellation.reason,
                'cancellation_cleanup_wall_s': (max(0., self.ended-self.cancellation.requested_at)
                    if self.cancellation.requested_at is not None else None),
                'scope': 'Request includes encoding/accounting/pause. Interval intersection is not speedup or physics-only time.'}
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder/'timing.json').write_text(json.dumps(self.metrics, indent=2))
            if self.response is not None:
                (self.folder/'response.json').write_text(json.dumps(self.response, indent=2))

    def cancel(self, reason):
        self.cancelled.set()
        self.cancellation.cancel(reason)


def execute_overlapped(hub, client, messages, tool, call, submission, batch, execute):
    """Accept dispatch, freeze the next request, execute, then collect the inert draft."""
    running = {'accepted': True, 'execution_status': 'starting', 'commands_completed': False,
        'source_step': hub.robot.steps, 'batch': batch,
        'instruction': 'Accepted dispatch, not a completion result. Future outcomes are unknown.'}
    messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(running)}
        if call.get('id') else {'role': 'user', 'content': json.dumps(running)})
    source_step = hub.robot.steps
    context = getattr(client, 'stage_context', None)
    if context is not None:
        from .timing import timings_for
        with timings_for(hub).scope('model_context_preparation'):
            context.prepare_boundary(messages, hub.single_a_artifacts.latest, hub.run/'a_context', hub.emit)
    messages.append({'role': 'user', 'content': json.dumps({
        'phase': 'async_prepare_next', 'source_step': source_step, 'executing_sketch': submission,
        'instruction': '同一个A：当前段正在执行，只提前规划紧接着的一个子任务。'
            '用已有历史和已读几何保存紧接一段的完整草图，不以defer跳过。'
            '未来结果未知，不得当作可见事实；把所需结果写成接续前提，不扩展成整个任务计划。'
            '沿用move/rotate/set_gripper/wait参数表达式，相对EEF仍到真实路点入口绑定。'
            '不填写未来held/changed/source_step；移动后会过期的世界像素不能假设仍有效。'
            '当前段正常结束就直接执行保存的下一段，包括抓放、夹爪和活动结构；不另发确认请求。'
            'preconditions是规划依据，不触发额外复核；语义结果仍未认证，实际失败和所用坐标失效会停下。'
            '仅当正在执行的草图预计完成整个任务时，用finish_review请求段末实际反馈后的正常finish或恢复；'
            '它不是未来成功声明，也不能用来泛化defer。不要为了提前规划或终态检查编造wait。'
            '缺少支持的未来条件保持为前提，不编造观测或坐标。此阶段只保存草图，不执行工具或动作。'
            '执行失败时立即作废本次候选并取消仍在生成的请求，再按实际停止状态画恢复段；'
            '旧候选只作意图参考，不能复用失效坐标。恢复段执行时同样提前准备后续段。'}, ensure_ascii=False)})
    hub.single_a_requests += 1
    snapshot = continuation_snapshot(hub)
    snapshot['executing_targets'] = sorted({a['target_id'] for a in submission['actions'] if a.get('target_id')})
    turn = hub.single_a_requests
    folder = hub.run/'single_a'/f'turn_{turn:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    request = OverlapRequest(client, messages, tool, folder)
    hub.emit({'event': 'async_draft_started', 'turn': turn, 'source_step': source_step, 'executing_batch': batch})
    with request.during_execution(hub.robot):
        receipt = execute()
        if not receipt.get('commands_completed'):
            request.cancel('execution_failed')
            hub.emit({'event': 'async_draft_cancel_requested', 'turn': turn,
                'executing_batch': batch, 'source_step': hub.robot.steps})
    candidate = None
    result = {'executed': False, 'robot_actions': 0}
    if request.error is None:
        messages.append(request.response['message'])
        calls = request.response['message'].get('tool_calls') or []
        hub.single_a_tool_calls += max(1, len(calls))
        try:
            if hub.single_a_tool_calls > hub.single_a_budget_limits['tool_calls']:
                raise ValueError('Draft tool budget exhausted; no candidate accepted')
            candidate = parse_draft(request.response, tool, source_step, batch, preparation_snapshot=snapshot)
            result.update(saved=candidate is not None, candidate_id=candidate['candidate_id'] if candidate else None)
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            result.update(saved=False, error=(exc.message if isinstance(exc, ValidationError) else str(exc))[:1800])
            hub.single_a_format_rejections += 1
        for item in calls:
            if item.get('id'):
                messages.append({'role': 'tool', 'tool_call_id': item['id'], 'content': json.dumps(result)})
        if not calls:
            messages.append({'role': 'user', 'content': json.dumps(result)})
    else:
        result.update(saved=False, error_type=type(request.error).__name__)
        cancelled = isinstance(request.error, RequestCancelled)
        if cancelled:
            result.update(cancelled=True, cancellation_reason=request.cancellation.reason)
        messages.append({'role': 'user', 'content': json.dumps({
            'phase': 'async_draft_cancelled' if cancelled else 'async_draft_failed', **result})})
    usable = candidate is not None and receipt.get('commands_completed') and not hub.robot.done
    if candidate is not None and not usable:
        hub.async_recovery_candidate = deepcopy(candidate)
        result['discarded'] = 'Execution failed or episode ended; review actual stop and replan with same A'
        messages.append({'role': 'user', 'content': json.dumps({'phase': 'async_candidate_discarded',
            'candidate_id': candidate['candidate_id'], 'reason': result['discarded'], 'robot_actions': 0,
            'instruction': 'Old candidate is non-executable reference only. Same A: repair from actual stop; '
                'during repair execution prepare a new successor. Never replay completed actions or revive stale points.'})})
    entry = {'turn': turn, 'executing_batch': batch, **request.metrics, **result, 'transport': request.metadata,
        'usage': request.response.get('usage') if request.response else None}
    (folder/'draft_result.json').write_text(json.dumps({**entry, 'candidate': candidate}, indent=2))
    hub.emit({'event': 'async_draft_finished', **entry, 'source_step': hub.robot.steps})
    # Intentional cancellation is a normal failed-motion -> same-A T2 boundary,
    # not a transport fault or a reason to stop/retry the task API.
    error = None if isinstance(request.error, RequestCancelled) else request.error
    return receipt, candidate if usable else None, entry, error
