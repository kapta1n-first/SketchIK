from copy import deepcopy
import json
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from sketchik.async_planning import (OverlapRequest, draft_tool, draft_context_start, parse_draft)
from sketchik.single_agent import tools_for


def test_draft_is_inert_and_retains_the_complete_original_actions():
    tool = draft_tool(tools_for({'cup'})[0]['function']['parameters'])
    value = {'status': 'ready', 'reason': 'Destination already observed',
        'preconditions': 'Current motion completes', 'intent': 'withdraw',
        'actions': [{'primitive': 'wait', 'target_id': 'cup', 'intent': 'settle', 'selection': {'steps': 2}}]}
    response = {'message': {'role': 'assistant', 'content': json.dumps(value)}}
    candidate = parse_draft(response, tool, 20, 1)
    assert not candidate['executed'] and candidate['prepared_from_step'] == 20
    assert candidate['actions'] == value['actions'] and candidate['preconditions'] == value['preconditions']
    assert 'held_object_id' not in candidate and 'visible_evidence' not in candidate
    with pytest.raises(ValueError, match='one draft_next_sketch'):
        parse_draft({'message': {'tool_calls': [{'function': {'name': 'execute_actions'}}]}}, tool, 20, 1)


def test_terminal_proposal_is_not_a_robot_action_or_future_success_claim():
    from jsonschema import ValidationError
    tool = draft_tool(tools_for({'cup'})[0]['function']['parameters'])
    proposal = {'status': 'finish_review', 'reason': 'Executing placement is expected to end the task'}
    parsed = parse_draft({'message': {'content': json.dumps(proposal)}}, tool, 20, 1)
    assert parsed['status'] == 'finish_review' and not parsed['executed']
    assert 'actions' not in parsed and 'task_success' not in parsed
    for extra in ({'actions': []}, {'success': True}, {'visible_evidence': 'future outcome'}):
        with pytest.raises(ValidationError):
            parse_draft({'message': {'content': json.dumps({**proposal, **extra})}}, tool, 20, 1)


def test_nonterminal_drafts_require_complete_actions_and_have_no_approval_gate():
    from jsonschema import ValidationError
    batch = tools_for({'cup'})[0]['function']['parameters']
    assert 'next_subtask' not in batch['properties']
    tool = draft_tool(batch)
    assert tool['function']['parameters']['type'] == 'object'
    for value in ({'status': 'defer', 'reason': 'need evidence'}, {'status': 'ready', 'reason': 'fixture'}):
        with pytest.raises(ValidationError):
            parse_draft({'message': {'content': json.dumps(value)}}, tool, 20, 1)
    value = {'status': 'ready', 'reason': 'existing geometry', 'preconditions': 'normal end', 'intent': 'continue',
        'actions': [{'primitive': 'wait', 'target_id': 'cup', 'intent': 'settle', 'selection': {'steps': 2}}]}
    for extra in ({'requires_visual_confirmation': True}, {'entry_conditions': {}}, {'held_object_id': 'cup'}):
        with pytest.raises(ValidationError):
            parse_draft({'message': {'content': json.dumps({**value, **extra})}}, tool, 20, 1)


def test_pending_candidate_reuses_original_complete_draft_dialogue():
    source = [{'role': 'assistant', 'tool_calls': [{'id': 'draft',
        'function': {'name': 'draft_next_sketch', 'arguments': '{"status":"ready"}'}}]},
        {'role': 'tool', 'tool_call_id': 'draft', 'content': json.dumps({'saved': True, 'candidate_id': 'pending'})},
        {'role': 'user', 'content': 'actual outcome'}]
    assert draft_context_start(source, {'candidate_id': 'pending'}) == 0
    with pytest.raises(ValueError, match='no original draft dialogue'):
        draft_context_start(source, {'candidate_id': 'missing'})
    source[1].pop('tool_call_id'); source[1]['role'] = 'user'
    assert draft_context_start(source, {'candidate_id': 'pending'}) == 0


def test_request_overlaps_motion_and_freezes_inputs_without_robot_access(tmp_path):
    owner = get_ident(); entered = Event(); moved = Event()
    source = [{'role': 'user', 'content': 'before'}]
    class Client:
        last_request_metadata = {}
        def complete(self, messages, tools):
            assert get_ident() != owner
            entered.set()
            assert moved.wait(5)
            assert messages == [{'role': 'user', 'content': 'before'}]
            return {'message': {'role': 'assistant', 'content': 'defer'}}
    seen = []
    callback = lambda state: seen.append((get_ident(), state))
    robot = SimpleNamespace(on_step=callback)
    request = OverlapRequest(Client(), source, {}, tmp_path)
    with request.during_execution(robot):
        assert entered.wait(5)
        source[0]['content'] = 'changed on owner'
        robot.on_step({'step': 1})
        moved.set()
    assert robot.on_step is callback and seen == [(owner, {'step': 1})]
    assert request.metrics['overlap_wall_s'] > 0
    assert not request.thread.is_alive()


@pytest.mark.parametrize('abort', [False, True])
def test_network_pause_is_transferred_to_owner_and_no_motion_is_replayed(tmp_path, abort):
    owner = get_ident(); paused = Event(); handlers = []
    def handler(error, messages, tools):
        handlers.append(get_ident())
        assert messages == [{'role': 'user', 'content': 'frozen'}]
        if abort:
            raise InterruptedError('stop')
    class Client:
        last_request_metadata = {}
        request_pause_handler = staticmethod(handler)
        def complete(self, messages, tools):
            paused.set()
            self.request_pause_handler(RuntimeError('network'), messages, tools)
            return {'message': {'role': 'assistant', 'content': 'finished'}}
    robot = SimpleNamespace(on_step=None)
    client = Client(); request = OverlapRequest(client, [{'role': 'user', 'content': 'frozen'}], {}, tmp_path)
    if abort:
        with pytest.raises(InterruptedError):
            with request.during_execution(robot):
                assert paused.wait(5)
                # The callback services the queued pause; no robot step is issued by recovery.
                while request.pauses.empty():
                    request.thread.join(.001)
                robot.on_step({'step': 1})
    else:
        with request.during_execution(robot):
            assert paused.wait(5)
    assert handlers == [owner] and client.request_pause_handler is handler
    assert not request.thread.is_alive() and robot.on_step is None


def test_execution_program_error_cancels_a_waiting_draft_without_resuming_API(tmp_path):
    entered = Event(); pauses = []
    class Client:
        last_request_metadata = {}
        request_pause_handler = staticmethod(lambda *args: pauses.append('unexpected retry'))
        def complete(self, messages, tools):
            entered.set()
            self.request_pause_handler(RuntimeError('network'), messages, tools)
            raise AssertionError('cancelled draft must not retry')
    request = OverlapRequest(Client(), [], {}, tmp_path)
    robot = SimpleNamespace(on_step=None)
    with pytest.raises(RuntimeError, match='executor fault'):
        with request.during_execution(robot):
            assert entered.wait(5)
            raise RuntimeError('executor fault')
    assert pauses == [] and isinstance(request.error, InterruptedError)
    assert robot.on_step is None and not request.thread.is_alive()


def test_failed_motion_receipt_cancels_draft_and_returns_normal_T2_boundary(tmp_path, monkeypatch):
    from sketchik.async_planning import execute_overlapped
    from sketchik.request_cancellation import RequestCancelled
    entered, exited = Event(), Event()
    class Client:
        stage_context = None
        last_request_metadata = {}
        def complete(self, messages, tools):
            entered.set()
            assert self.request_cancellation.event.wait(3), 'must not wait for a complete obsolete sketch'
            try:
                self.request_cancellation.check()
            finally:
                exited.set()
    robot = SimpleNamespace(steps=20, done=False, on_step=None)
    events = []
    hub = SimpleNamespace(robot=robot, run=tmp_path, emit=events.append, single_a_requests=0,
        single_a_tool_calls=0, single_a_budget_limits={'tool_calls': 10}, single_a_format_rejections=0)
    monkeypatch.setattr('sketchik.async_planning.continuation_snapshot', lambda hub: {'source_step': robot.steps})
    failed = {'source_step': 21, 'commands_completed': False, 'remaining_suffix_executed': False}
    def execute():
        assert entered.wait(3)
        robot.steps = 21
        return failed
    messages = []
    receipt, candidate, entry, error = execute_overlapped(hub, Client(), messages, {},
        {'id': 'real-dispatch'}, {'actions': []}, 1, execute)
    assert receipt is failed and candidate is None and error is None and exited.is_set()
    assert entry['error_type'] == RequestCancelled.__name__ and entry['cancellation_requested']
    assert hub.single_a_requests == 1 and hub.single_a_tool_calls == 0
    assert hub.single_a_format_rejections == 0 and robot.steps == 21
    assert any(event['event'] == 'async_draft_cancel_requested' for event in events)
    assert not any(message['role'] == 'assistant' for message in messages)


def continuation_fixture():
    from sketchik.async_planning import check_continuation
    batch = tools_for({'cup'})[0]['function']['parameters']
    reference = {'reference_id': 'original-points', 'grasp_epoch': 2, 'relation_revision': 0, 'validity': 'active'}
    claim = {'object_id': 'cup', 'source_step': 20, 'relative_pose_assessment': 'unchanged'}
    before = {'run': '/episode', 'source_step': 20, 'request_turn': 4, 'grasp_epoch': 2,
        'reference': reference.copy(), 'a_held_claim': claim, 'executing_targets': ['cup']}
    action = {'primitive': 'wait', 'target_id': 'cup', 'intent': 'settle', 'selection': {'steps': 2}}
    args = {'status': 'ready', 'reason': 'existing destination', 'preconditions': 'Same reviewed carry persists',
        'intent': 'continue', 'actions': [action]}
    def candidate():
        return parse_draft({'message': {'content': json.dumps(args)}}, draft_tool(batch), 20, 1,
            preparation_snapshot=before)
    support = {'status': 'supported', 'cameras': [{'camera': 'front', 'source_step': 40}],
        'testable_points': 100, 'supported_points': 75, 'support_fraction': .75}
    current = {'source_step': 40, 'robot_state': {'gripper_command': 1.},
        'geometry': {'carried_observed_geometry': {'object_id': 'cup', **reference, 'source_step': 40,
            'reference_available': True, 'visible_support': support},
            'robot_gripper_geometry': {'current_inner_jaw_gap_m': .04}}}
    kwargs = {'batch': 1, 'snapshot': {**before, 'source_step': 40},
        'episode': {'last_gripper_event': {'source_step': 15}},
        'consumed': set()}
    receipt = {'source_step': 40, 'commands_completed': True, 'steps': [{'result': {'converged': True}}]}
    return candidate, args, before, current, receipt, kwargs, check_continuation


def test_continuation_checks_reviewed_carry_without_mutation_or_success_certification():
    make, args, before, current, receipt, kwargs, check = continuation_fixture()
    candidate = make(); original = deepcopy((candidate, current, receipt, kwargs))
    report = check(candidate, current, receipt, **kwargs)
    assert report['status'] == 'pass'
    assert not report['holding_certified'] and not report['task_success_certified'] and not report['collision_checked']
    assert (candidate, current, receipt, kwargs) == original


@pytest.mark.parametrize('problem', ['new_grasp', 'release', 'occluded', 'stale_depth', 'no_claim',
    'relative_rebind', 'suspended_unused_reference', 'no_gap', 'candidate_release', 'candidate_close'])
def test_normal_contact_transitions_or_uncertified_semantics_do_not_force_A_review(problem):
    make, args, before, current, receipt, kwargs, check = continuation_fixture()
    held = current['geometry']['carried_observed_geometry']
    if problem in ('new_grasp', 'release'): kwargs['snapshot']['grasp_epoch'] += 1
    if problem == 'release':
        current['robot_state']['gripper_command'] = -1.
        current['geometry'].pop('carried_observed_geometry')
        kwargs['snapshot']['reference'] = None
    elif problem == 'occluded': held['visible_support']['status'] = 'insufficient_visibility'
    elif problem == 'stale_depth': held['visible_support']['cameras'][0]['source_step'] = 39
    elif problem == 'no_claim': before['a_held_claim'] = None
    elif problem == 'relative_rebind': kwargs['snapshot']['reference'] = {**before['reference'], 'relation_revision': 1}
    elif problem == 'suspended_unused_reference': held['validity'] = kwargs['snapshot']['reference']['validity'] = 'suspended'
    elif problem == 'no_gap': current['geometry']['robot_gripper_geometry'].clear()
    elif problem.startswith('candidate_'):
        args['actions'][0] = {'primitive': 'set_gripper', 'target_id': None,
            'intent': 'planned configuration change', 'selection': {
                'command': 'open' if problem == 'candidate_release' else 'close', 'steps': 20}}
    assert check(make(), current, receipt, **kwargs)['status'] == 'pass'


@pytest.mark.parametrize('problem', ['contact_warning', 'depth_contradiction', 'execution_failed',
    'wrong_batch', 'wrong_request', 'wrong_run', 'reused', 'tampered', 'stale_state'])
def test_actual_faults_and_obsolete_candidates_still_stop_dispatch(problem):
    make, args, before, current, receipt, kwargs, check = continuation_fixture()
    if problem == 'contact_warning': current['contact_warning'] = {'reason': 'jaw collapse'}
    elif problem == 'depth_contradiction': current['geometry']['carried_observed_geometry']['visible_support']['status'] = 'contradicted'
    elif problem == 'execution_failed': receipt['commands_completed'] = False
    elif problem == 'wrong_batch': kwargs['batch'] = 2
    elif problem == 'wrong_request': kwargs['snapshot']['request_turn'] += 1
    elif problem == 'wrong_run': kwargs['snapshot']['run'] = '/other'
    elif problem == 'stale_state': current['source_step'] = 39
    candidate = make()
    if problem == 'reused': kwargs['consumed'].add(candidate['candidate_id'])
    elif problem == 'tampered': candidate['actions'][0]['selection']['steps'] = 99
    assert check(candidate, current, receipt, **kwargs)['status'] == 'fail'


@pytest.mark.parametrize('current_epoch,expected', [(True, 'fail'), (False, 'pass')])
def test_recorded_depth_suspension_is_a_real_fault_only_for_the_current_reference_epoch(current_epoch, expected):
    make, args, before, current, receipt, kwargs, check = continuation_fixture()
    held = current['geometry']['carried_observed_geometry']
    held.pop('visible_support')  # Real runtime drops this field while a reference remains suspended.
    reference = kwargs['snapshot']['reference']
    reference.update(validity='suspended', suspension_kind='depth_contradiction',
        suspended_at_step=35, suspension_reason='Actual depth contradicted original points')
    held.update(reference)
    if not current_epoch:
        kwargs['snapshot']['grasp_epoch'] += 1  # Retired grasp evidence cannot block a new contact.
    gate = check(make(), current, receipt, **kwargs)
    assert gate['status'] == expected
    evidence = next(c for c in gate['checks'] if c['check'] == 'retained_depth_contradiction')
    if current_epoch:
        assert evidence['suspended_reference']['suspended_at_step'] == 35
        assert evidence['suspended_reference']['suspension_reason'] == reference['suspension_reason']
