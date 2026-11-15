"""Shared image/history and structured observation dialogue, without grasp services."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import time

from jsonschema import Draft202012Validator, ValidationError

from .agent import ProviderError, transcript_copy
from .tools import PERCEPTION_TOOLS


def image_part(path):
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()}}


def save_planning_history(run, messages):
    folder=run/'planning_images';folder.mkdir(exist_ok=True)
    for message in messages:
        if not isinstance(message.get('content'),list):continue
        for part in message['content']:
            if part.get('type')=='image_url':
                url=part['image_url']['url']
                if url.startswith('data:image/png;base64,'):
                    identity=hashlib.sha256(url.encode()).hexdigest()
                    target = folder/(identity+'.png')
                    if not target.exists():
                        target.write_bytes(base64.b64decode(url.split(',',1)[1],validate=True))
    (run/'planning_transcript.json').write_text(json.dumps(transcript_copy(messages),indent=2))


def planning_dialogue(client, hub, messages, submit_tool, *, max_turns=40, max_tool_calls=100,
                      allow_actions=False, extra_observation_tools=frozenset()):
    """Use phase-specific tools; action permission is explicit for initial observation."""
    if min(max_turns, max_tool_calls) < 1:
        raise ValueError('Planning budgets must be positive')
    observation_actions = {'move', 'move_relative', 'move_waypoints', 'wait'} | set(extra_observation_tools)
    names = {t['function']['name'] for t in hub.schemas}
    if not allow_actions and names - PERCEPTION_TOOLS:
        raise ProviderError('Read-only phase cannot expose motion or other tools')
    if allow_actions and names - PERCEPTION_TOOLS - observation_actions:
        raise ProviderError('Initial observation exposes only perception and observation motion')
    submission_name = submit_tool['function']['name']
    seen = {call['id'] for message in messages for call in (message.get('tool_calls') or [])
            if isinstance(call, dict) and isinstance(call.get('id'), str)}
    total_s = 0.
    call_count = 0
    for turn in range(1, max_turns + 1):
        active_submit=deepcopy(submit_tool)
        parameters=active_submit['function']['parameters']
        properties=parameters.get('properties',{})
        if 'source_step' in properties:
            properties['source_step']['const']=hub.robot.steps
        plan=properties.get('plan',{}).get('properties',{})
        if 'source_step' in plan:
            plan['source_step']['const']=hub.robot.steps
        tools=hub.schemas+[active_submit]
        validators={t['function']['name']:Draft202012Validator(t['function']['parameters']) for t in tools}
        start = time.perf_counter()
        response = client.complete(messages, tools)
        elapsed = time.perf_counter() - start
        total_s += elapsed
        message = response['message']
        calls = message.get('tool_calls') or []
        if not calls and not hub.schemas:
            # A single semantic-submission boundary may be returned as plain
            # JSON. Never interpret JSON content as a perception/motion tool.
            messages.append(message)
            hub.emit({'event':'grasp_planning_response','round':turn,'wall_s':elapsed,
                'usage':response.get('usage'),'finish_reason':response.get('finish_reason'),
                'tool_names':[],'submission_transport':'assistant_json'})
            try:
                args=json.loads(message.get('content') or '')
                validators[submission_name].validate(args)
            except (ValidationError,ValueError) as exc:
                messages.append({'role':'user','content':json.dumps({
                    'symbolic_submission':submission_name,'saved':False,'robot_actions':0,
                    'current_source_step':hub.robot.steps,
                    'validation_error':exc.message if isinstance(exc,ValidationError) else str(exc),
                    'instruction':'Return one JSON object matching the supplied submission schema, or use its native tool call. JSON content cannot invoke motion or perception.'})})
                hub.emit({'event':'submission_schema_rejected','round':turn,
                    'current_source_step':hub.robot.steps,'robot_actions':0,
                    'submission_transport':'assistant_json'})
                continue
            messages.append({'role':'user','content':json.dumps({
                'symbolic_submission':submission_name,'saved':True,'robot_actions':0,
                'submission_transport':'assistant_json'})})
            return args,response,{'api_wall_s':total_s,'rounds':turn,'tool_calls':call_count,
                'submission_transport':'assistant_json'}
        if not calls:
            raise ProviderError('No tool call or sketch submitted; no automatic API retry')
        if call_count + len(calls) > max_tool_calls:
            raise ProviderError('Grasp planning tool budget exhausted')
        parsed = []
        batch = set()
        submission_errors = {}
        # Validate the entire batch before even invoking a read-only service.
        for call in calls:
            identity = call.get('id')
            function = call.get('function', {})
            name = function.get('name')
            if call.get('type') != 'function' or not isinstance(identity, str) or not identity or identity in seen or identity in batch:
                raise ProviderError('Malformed or duplicate tool call ID')
            if name not in validators:
                raise ProviderError('Unknown planning tool; motion is not exposed in this phase')
            args=None
            try:
                args = json.loads(function['arguments'], parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                validators[name].validate(args)
            except (ValidationError,ValueError) as exc:
                submission_errors[identity] = {'error_type':'schema_validation' if isinstance(exc,ValidationError) else 'json_validation',
                    'error':exc.message if isinstance(exc,ValidationError) else str(exc),
                    'validation_path':list(exc.absolute_path) if isinstance(exc,ValidationError) else [],
                    'schema_path':list(exc.absolute_schema_path) if isinstance(exc,ValidationError) else []}
            if name == submission_name and identity not in submission_errors and args['source_step'] != hub.robot.steps:
                submission_errors[identity]={'error_type':'stale_source_step',
                    'error':'Submission source_step does not match the current physical step'}
            batch.add(identity)
            parsed.append((identity, name, args))
        submissions = [v for v in parsed if v[1] == submission_name]
        if len(submissions) > 1 or (submissions and parsed[-1][1] != submission_name):
            raise ProviderError('Submit at most one final sketch, after any read-only calls in the batch')
        messages.append(message)
        hub.emit({'event':'grasp_planning_response','round':turn,'wall_s':elapsed,
                  'usage':response.get('usage'),'finish_reason':response.get('finish_reason'),
                  'tool_names':[v[1] for v in parsed]})
        if submission_errors:
            # Preserve the entire attempted batch, but dispatch none of it. The
            # next model response must explicitly supply a corrected submission.
            for identity, name, args in parsed:
                seen.add(identity)
                call_count += 1
                error=submission_errors.get(identity, {
                    'error_type':'batch_not_dispatched',
                    'error':'This batch contains an invalid submission; this tool was not dispatched.'})
                messages.append({'role':'tool','tool_call_id':identity,'content':json.dumps({
                    'ok':False,'saved':False,'robot_actions':0,'actions_from_submission':0,
                    'current_source_step':hub.robot.steps, **error,
                    'instruction':'Correct the invalid tool arguments or submission with a new tool call ID. Both source_step and plan.source_step must equal current_source_step when present; historical images keep their own original steps. Camera, observation_id and pixel bounds must refer to the same saved image. No tools in this batch were dispatched.'},allow_nan=False)})
            hub.emit({'event':'submission_schema_rejected' if submissions else 'observation_schema_rejected','round':turn,
                'current_source_step':hub.robot.steps,'robot_actions':0,'errors':submission_errors})
            continue
        pictures = []
        for identity, name, args in parsed:
            seen.add(identity)
            call_count += 1
            if name == submission_name:
                if args['source_step'] != hub.robot.steps:
                    messages.append({'role':'tool','tool_call_id':identity,'content':json.dumps({
                        'saved':False,'actions_from_submission':0,'current_source_step':hub.robot.steps,
                        'error':'An earlier observation action changed source_step; resubmit using its real result.'})})
                    hub.emit({'event':'submission_rejected_after_observation_action','current_source_step':hub.robot.steps})
                    continue
                messages.append({'role':'tool','tool_call_id':identity,
                                 'content':json.dumps({'saved':True,'robot_actions':0})})
                if pictures:
                    messages.append({'role':'user','content':pictures})
                return args, response, {'api_wall_s':total_s,'rounds':turn,'tool_calls':call_count}
            result, images = hub.call(name, args)
            messages.append({'role':'tool','tool_call_id':identity,'content':json.dumps(result,allow_nan=False)})
            pictures.extend(images)
            if hub.fatal:
                if pictures:messages.append({'role':'user','content':pictures})
                raise ProviderError('Observation action result is failed or uncertain; no further action dispatched')
        if pictures:
            messages.append({'role':'user','content':pictures})
    raise ProviderError('Grasp planning reply budget exhausted without a sketch')
