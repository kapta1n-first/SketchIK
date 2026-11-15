"""Numerical recovery of the same target, then reconciliation by the same agent."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path

from .kinematics import IKError
from .tools import public
from .timing import timings_for


def repeats_stalled_goal(failure, compiled, current):
    """Equivalent expressions do not become a new physical recovery target."""
    if (failure.get('failure_stage')!='motion' or not failure.get('executed')
            or failure.get('actual_motion',{}).get('motion_progress',{}).get('recent_progress') is not False):
        return False
    from scipy.spatial.transform import Rotation
    import numpy as np
    goal=failure.get('original_absolute_goal')
    candidate=(compiled.get('audit') or {}).get('compiled_pose')
    if not goal or not candidate:return False
    if np.linalg.norm(np.asarray(candidate['position'])-goal['position'])>1e-4:return False
    rotation=Rotation.from_quat(candidate['quaternion_xyzw'])
    return any((rotation*Rotation.from_quat(pose['quaternion_xyzw']).inv()).magnitude()<=1e-4
               for pose in (goal,current))


def run_step(hub, grounder, stage, index, description, pictures, folder, *, subtask_index=None):
    from .task_execution import BindingUnavailable, POSE_ACTIONS
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    step=stage['plan']['steps'][index]
    trace=[];last_output=None;last_compiled=None;failure=None;t1_used=False
    actual_description,actual_pictures=description,pictures

    def record(tier, decision, **evidence):
        item={'tier':tier,'decision':decision,'source_step':hub.robot.steps,**public(evidence)}
        trace.append(item)
        (folder/'recovery.json').write_text(json.dumps(trace,indent=2,allow_nan=False))
        hub.emit({'event':'step_recovery','index':index,**item})

    def classify(exc):
        return deepcopy(getattr(exc,'failure',None) or {
            'failure_stage':'binding','error':str(exc),'executed':False})

    def summary():
        result=[]
        for item in trace:
            small={key:item[key] for key in ('tier','decision','source_step','reason',
                'absolute_goal_preserved','recent_progress') if key in item}
            if item.get('failure'):
                small['failure']={key:item['failure'][key] for key in
                    ('failure_stage','error','executed','physical_cause') if key in item['failure']}
                pose=(item['failure'].get('compiled_from_selection') or {}).get('compiled_pose')
                if pose:small['failure']['compiled_pose']=deepcopy(pose)
            result.append(small)
        return result

    def escalate(error):
        record('T3','return_to_A',failure=error)
        if last_output is not None:
            return {'compiled':last_compiled,'output':last_output,'recovery':summary(),'failure':error}
        exc=BindingUnavailable(json.dumps(error,allow_nan=False));exc.failure=error;exc.recovery=summary()
        raise exc

    def check_actual():
        nonlocal actual_description,actual_pictures
        if actual_description['source_step']!=hub.robot.steps:
            actual_description,actual_pictures=hub.observe()
        if hasattr(grounder,'check_actual_contact') and step['primitive'] in POSE_ACTIONS:
            grounder.check_actual_contact(stage,index,actual_description,
                folder/('contact_'+str(hub.robot.steps)))

    def dispatch(compiled):
        params=compiled['parameters']
        if hasattr(grounder, 'record_dispatch'):
            grounder.record_dispatch(stage, index, compiled)
        hub.emit({'event':'bound_step_execution_started','index':index,
            'subtask':subtask_index,'primitive':step['primitive'],'parameters':params,
            'parameter_source':compiled.get('parameter_source','agent_a'),
            'binding_source_step':stage.get('source_step')})
        route=compiled.get('checked_route') or getattr(hub,'_bound_checked_route',None)
        hub._bound_checked_route=None
        source=hub.robot.steps
        search=(hub.executor.ik.recovery_search(timeout_s=5.)
            if compiled.get('ik_recovery_search') else nullcontext())
        try:
            with search, timings_for(hub).scope('controller_IK_and_physics'):
                if compiled.get('compiled_motion') is not None:
                    return public(hub.executor.execute_prepared_motion(compiled['compiled_motion']))
                if step['primitive'] in POSE_ACTIONS and route:
                    options={'settle_steps':params['settle_steps']} if 'settle_steps' in params else {}
                    return public(hub.executor.move_waypoints(route['waypoints'],**options))
                return public(getattr(hub.executor,step['primitive'])(**params))
        except IKError as exc:
            if hub.robot.steps!=source:
                raise RuntimeError('IK exception after physical state advanced; no automatic replay') from exc
            blocked=BindingUnavailable(str(exc))
            blocked.failure={'failure_stage':'ik_path','error':str(exc),'executed':False,
                             'compiled_from_selection':compiled.get('audit')}
            raise blocked from exc

    try:
        compiled=grounder.compile_step(stage,index,description,folder)
    except BindingUnavailable as exc:
        failure=classify(exc)
        if failure.get('failure_stage') in ('carried_contact','stalled_goal'):
            return escalate(failure)
        if (step['primitive'] not in POSE_ACTIONS
                or failure.get('failure_stage') not in ('ik_endpoint','ik_path')):
            raise
        if hasattr(grounder,'recover_compile'):
            t1_used=True
            record('T1','retry_same_pose_IK',failure=failure,absolute_goal_preserved=True)
            try:
                compiled=grounder.recover_compile(exc,folder/'t1_ik')
                failure=None
                record('T1','same_pose_IK_passed',absolute_goal_preserved=True)
            except BindingUnavailable as retry:
                failure=classify(retry)
                record('T1','same_pose_IK_rejected',failure=failure)
        else:record('T1','unavailable',reason='No alternative-seed compilation backend')
    if failure is None:
        if compiled.get('needs_observation'):
            return {'compiled':compiled,'output':None,'recovery':summary()}
        last_compiled=compiled
        try:last_output=dispatch(compiled)
        except BindingUnavailable as exc:failure=classify(exc)
        if last_output is not None:
            if last_output.get('episode_ended') or hub.robot.done:
                return {'compiled':compiled,'output':last_output,'recovery':summary()}
            try:check_actual()
            except BindingUnavailable as exc:return escalate(classify(exc))
            if last_output.get('converged') is not False:
                return {'compiled':compiled,'output':last_output,'recovery':summary()}
            failure={'failure_stage':'motion','error':'Motion did not converge',
                'physical_cause':'not_determined','executed':True,
                'original_absolute_goal':(compiled.get('audit') or {}).get('compiled_pose') or
                    last_output.get('continuation_goal'),
                'parameters':compiled['parameters'],'actual_motion':last_output}
            progressing=last_output.get('motion_progress',{}).get('recent_progress') is True
            if (step['primitive'] in POSE_ACTIONS and not t1_used and progressing
                    and hasattr(hub.executor,'continue_motion')):
                t1_used=True
                record('T1','continue_same_absolute_goal',failure=failure)
                options = {'max_steps': 40}
                guard = grounder.continuation_guard() if hasattr(grounder, 'continuation_guard') else None
                if guard is not None:
                    options['stop_condition'] = guard
                continued=public(hub.executor.continue_motion(last_output,**options))
                from .motion_record import merge_tracking
                continued={**continued,
                    'tracking':merge_tracking(last_output.get('tracking'),continued.get('tracking')),
                    **({'compiled_motion_id':last_output['compiled_motion_id']} if last_output.get('compiled_motion_id') else {}),
                    'actions':last_output.get('actions',0)+continued.get('actions',0),
                    'execution_wall_s':last_output.get('execution_wall_s',0)+continued.get('execution_wall_s',0),
                    'motion_attempts':[last_output,deepcopy(continued)]}
                last_output=continued
                if continued.get('episode_ended') or hub.robot.done:
                    return {'compiled':compiled,'output':continued,'recovery':summary()}
                try:check_actual()
                except BindingUnavailable as exc:return escalate(classify(exc))
                if continued.get('continuation_interruption'):
                    return escalate({**continued['continuation_interruption'],
                        'executed': True, 'actual_motion': continued})
                if continued.get('converged'):
                    record('T1','continuation_finished',result=continued)
                    return {'compiled':compiled,'output':continued,'recovery':summary()}
                failure={**failure,'actual_motion':continued}
                record('T1','continuation_incomplete',failure=failure)
            else:
                record('T1','skip_continuation',recent_progress=progressing,
                    reason='No remaining T1 allowance or no measured recent progress; do not repeat a stalled command')
    return escalate(failure)
