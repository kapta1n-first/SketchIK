"""Compile and execute one agent's parameterized sketches using real IK."""
from __future__ import annotations

import json
from pathlib import Path
import time
from copy import deepcopy

from jsonschema import Draft202012Validator

from .planning import image_part
from .tools import obj, V3, Q4, PARAMS, public
from .timing import timed
from .pose_binding import BINDING, compile_binding, compile_rotation
from .episode_state import EpisodeState

ROTATION_ACTIONS = frozenset({'rotate_roll', 'rotate_pitch', 'rotate_yaw'})
POSE_ACTIONS = ROTATION_ACTIONS | {'move'}
POSE = obj({'position': V3, 'quaternion_xyzw': Q4}, ['position', 'quaternion_xyzw'])


def previous_stage_outcome(receipt):
    """Public failed goals and actual stop states, without old images or CoT."""
    if not receipt:return None
    def motion(result, goal=None):
        result=result or {}; state=result.get('state') or {}
        if not goal or not all(key in goal for key in ('position', 'quaternion_xyzw')):
            goal = result.get('continuation_goal') or goal or {}
        recorded_goal = {k:goal[k] for k in ('position','quaternion_xyzw') if k in goal}
        final_q = (result.get('continuation_goal') or {}).get('joint_target')
        if final_q is None:
            final_q = (result.get('tracking') or {}).get('goal_joint_positions_rad')
        if final_q is not None:
            recorded_goal['joint_target'] = deepcopy(final_q)
        return {'goal':recorded_goal,
            **{k:deepcopy(result[k]) for k in ('converged','position_error_m','rotation_error_deg',
                'physical_cause','tracking','compiled_motion_id') if k in result},
            'actual_stop':{k:deepcopy(state[k]) for k in ('step','q','position','quaternion_xyzw') if k in state}}
    motions=[];rejections=[]
    for step in receipt.get('steps',[]):
        if step.get('primitive') in POSE_ACTIONS:
            motions.append(motion(step.get('result'),step.get('parameters')))
        failure=step.get('recovery_failure') or {}
        if failure.get('actual_motion'):
            item=motion(failure['actual_motion'],failure.get('compiled_pose'))
            if item not in motions:motions.append(item)
    for event in [*receipt.get('recovery',[]),
            *(event for step in receipt.get('steps',[]) for event in step.get('recovery',[]))]:
        failure=event.get('failure') or {}
        if failure.get('compiled_pose'):
            item={'source_step':event.get('source_step'),'target_pose':failure['compiled_pose'],
                'failure_stage':failure.get('failure_stage'),'executed':False}
            if item not in rejections:rejections.append(item)
    # Older receipts kept this public compiler audit only in the final error
    # string. Recover that exact numerical evidence, never a model diagnosis.
    legacy=receipt.get('binding_failure')
    if isinstance(legacy,str) and '{' in legacy:
        try:legacy=json.loads(legacy[legacy.index('{'):])
        except (ValueError,TypeError):legacy=None
    if isinstance(legacy,dict) and legacy.get('failure_stage') in ('ik_endpoint','ik_path'):
        audit=legacy.get('compiled_from_selection') or {}
        if legacy.get('executed') is False and audit.get('compiled_pose'):
            item={'source_step':audit.get('execution_source_step',audit.get('source_step',receipt.get('source_step'))),
                'target_pose':audit['compiled_pose'],'failure_stage':legacy['failure_stage'],
                'executed':False,'audit_source':'recorded_binding_failure'}
            if not any(v['target_pose']==item['target_pose'] and v['source_step']==item['source_step']
                       for v in rejections):rejections.append(item)
    return {'stage_id':receipt.get('stage_id'),'plan_source_step':receipt.get('plan_source_step'),
        'source_step':receipt.get('source_step'),'commands_completed':receipt.get('commands_completed'),
        'remaining_suffix_executed':receipt.get('remaining_suffix_executed'),
        'failure':receipt.get('failure'),'binding_failure':receipt.get('binding_failure'),
        'motions':motions,'compile_rejections':rejections,
        'recovery_budget_scope':'historical_stage_only',
        'scope':'Recorded robot commands/results from the preceding stage; causes and holding are not certified. '
            'Do not repeat a stalled goal as a fresh route. Historical recovery usage does not consume the current request allowance.'}


@timed('hand_geometry_views')
def hand_observation_parts(hub, description, folder):
    """Current geometry for A's visual review, without adding a tool or action."""
    from .hand_observation import prepare_hand_observation
    card = prepare_hand_observation(hub, description, folder)
    if card is None:
        return []
    metadata = deepcopy(card['metadata'])
    geometry = metadata.get('known_robot_geometry')
    if geometry is not None:
        for key in ('actual_parts', 'recorded_goal_parts'):
            geometry[key+'_count'] = len(geometry.pop(key, []))
        geometry['full_record'] = 'The same card metadata JSON; CURRENT gives its public file path.'
    return [{'type': 'text', 'text':
        'CURRENT observed surfaces with the known complete robot hand envelope '
        '(blue) and finger pads (red). Same physical source as the current RGB. '
        'Magenta is a retained observed shape propagated by the current EEF, not a new measurement or verified attachment. '
        'If present, orange parts show the saved absolute FAILED goal at the current finger opening; '
        'cyan shows only an associated carried-point hypothesis at that goal. '
        'Actual and goal use the same part envelopes; their overlap with fixed observed surfaces '
        'is not collision/contact certification. Never reselect an old failure pixel in a new image. '
        'Unassigned surfaces do not certify a held object or rigid attachment. '
        'Consider the palm and fingers as well as the object when choosing the '
        'contact method and allowed orientation. The fourth panel also shows the known moving arm '
        'at actual joints and, only if recorded, the exact old target joint branch. It does not solve '
        'a new branch or certify collision. '+json.dumps(metadata)},
        image_part(Path(card['image_file']))]


class BindingUnavailable(ValueError):
    """Concrete model/geometry failures; A may revise the semantic stage."""


def check_pose(ik, state, pose):
    Draft202012Validator(POSE).validate(pose)
    from .kinematics import quaternion
    solution = ik.solve(pose['position'], quaternion(pose['quaternion_xyzw']), state['q'])
    return {'ik_pass': True, 'position_error_m': solution.position_error_m,
            'rotation_error_rad': solution.rotation_error_rad, 'ik_wall_s': solution.elapsed_s,
            'joint_positions': solution.q.tolist(), 'path_checked': False,
            'collision_checked': False, 'robot_actions': 0}


def parameter_schema(step):
    if step['primitive'] == 'move':
        return POSE
    if step['primitive'] == 'set_gripper':
        return obj({'command': {'const': step['command']}, 'steps': {'type': 'integer', 'minimum': 1, 'maximum': 200}},
                   ['command'])
    return PARAMS[step['primitive']]


class NumericalCompiler:
    """Numerical geometry, IK and contact checks; no model client or planning API."""

    def __init__(self, hub, handoff):
        self.hub, self.handoff = hub, handoff
        if not hasattr(hub, 'episode_state'):
            hub.episode_state = EpisodeState(hub.run)

    def record_execution(self, receipt):
        self.hub.episode_state.record_execution(receipt)

    def continuation_guard(self):
        """Robot-only T1 stop check; retained RGB-D does not become new evidence."""
        from .episode_state import carried_contact_change
        from .robot_geometry import measured_inner_jaw_gap
        episode = self.hub.episode_state
        baseline = deepcopy(episode.data.get('contact_baseline'))
        reference = getattr(self.hub, 'carried_surface_reference', None) or {}
        identity = getattr(self.hub, 'carried_target_id', None) or reference.get('object_id')
        if (not baseline or not identity or not hasattr(self.hub.executor.ik, 'model')
                or baseline.get('carried_observed_geometry', {}).get('object_id') != identity):
            return None
        boundary = episode.data.get('contact_boundary_step')

        def guard(state):
            if state['step'] != self.hub.robot.steps:
                raise RuntimeError('T1 contact check requires the actual physical-step state')
            if (state.get('gripper_command') != 1.
                    or episode.data.get('contact_boundary_step') != boundary):
                return {'failure_stage': 'carried_contact', 'executed': False,
                    'error': 'The contact premise changed during T1; review the actual stopped state.'}
            warning = episode.data.get('contact_warning')
            if warning and warning.get('object_id') == identity:
                return {'failure_stage': 'carried_contact', 'executed': False,
                    'error': warning['reason'], 'contact_change': deepcopy(warning)}
            gap = measured_inner_jaw_gap(self.hub.executor.ik, state)
            packet = {'source_step': state['step'],
                'robot_gripper_geometry': {'current_inner_jaw_gap_m': gap},
                'carried_observed_geometry': {'object_id': identity}}
            warning = carried_contact_change([baseline], packet, identity, 1.)
            if warning is None:
                return None
            warning['measurement_scope'] = 'Actual robot jaw geometry during T1; no new RGB-D or holding certification.'
            episode.data.update(source_step=state['step'],
                carry_premise='invalidated_by_contact_change', contact_warning=deepcopy(warning))
            episode.save()
            self.hub.emit({'event': 'carried_contact_change', **warning, 'check_phase': 'T1'})
            return {'failure_stage': 'carried_contact', 'executed': False,
                'error': warning['reason'], 'contact_change': warning}

        return guard

    def _check_contact_state(self, packet, previous_packets=(), *, folder=None):
        """Check current contact evidence before dispatch."""
        warning=self.hub.episode_state.observe_contact(packet,
            getattr(self.hub,'carried_target_id',None),
            self.hub.robot.state().get('gripper_command'),previous_packets)
        if warning is not None:
            if folder is not None:
                (Path(folder)/'carried_contact_change.json').write_text(json.dumps(warning,indent=2))
            self.hub.emit({'event':'carried_contact_change',**warning})
            blocked=BindingUnavailable(warning['reason'])
            blocked.failure={'failure_stage':'carried_contact','error':warning['reason'],
                'executed':False,'contact_change':warning}
            raise blocked

    def seed_scan_carry(self, target_id, folder):
        """Bind the selected observed component at actual closing, not a holding claim."""
        import numpy as np
        from scipy.spatial.transform import Rotation
        evidence=getattr(self.hub,'last_contact_binding',None)
        if evidence is None or evidence['target_id']!=target_id:return
        started=time.perf_counter();folder=Path(folder)
        record={'object_id':target_id,'source_kind':'scan_cloud','status':'unavailable',
            'identity_source':'Agent grasp target and selected contact point; not holding certification'}
        try:
            anchor=np.asarray(evidence['audit']['anchor_world_xyz_m'],float)
            views=evidence['views']
            if not views:raise ValueError('No point cloud for the actual contact binding')
            binding=evidence.get('geometry_binding') or {}
            if binding and not binding.get('carry_reference_eligible', False):
                raise ValueError('This current binding is localization-only or mixed. Use a clean current '
                    'object subset, or recover with current pixels and pad/EEF references.')
            selected_id=(evidence['audit'].get('binding') or {}).get('anchor',{}).get('view_id')
            selected_view=next((view for view in views if view.get('view_id')==selected_id),views[0])
            group_file=Path(selected_view['cloud_file']).parent/'extent_groups.npz'
            with np.load(group_file) as groups:
                ids=groups['point_group_ids'].copy();cloud_file=Path(str(groups['cloud_file']))
            with np.load(cloud_file) as cloud:
                xyz=cloud['world_xyz_m'];nearest=int(np.argmin(np.linalg.norm(xyz-anchor,axis=1)))
                if np.linalg.norm(xyz[nearest]-anchor)>.004 or ids[nearest]<=0:
                    raise ValueError('Selected contact has no supported observed component')
                group_id=int(ids[nearest]);mask=ids==group_id
                points=xyz[mask].astype(float);rgb=cloud['rgb'][mask].copy()
            if len(points)<25 or np.max(np.ptp(points,axis=0))>.3:
                raise ValueError('Selected component is not an isolated small carried shape')
            state=self.hub.robot.state()
            rotation=Rotation.from_quat(state['quaternion_xyzw']).as_matrix()
            local=(points-np.asarray(state['position']))@rotation
            record.update(status='predicted_rigid',source_step=state['step'],
                source_cloud_file=str(cloud_file),source_observations=deepcopy(selected_view.get("sources", [])),
                component_group_id=group_id,points=len(local),
                anchor_world_xyz_m=anchor.tolist(),rigid_attachment_verified=False)
            np.savez(folder/'scan_carry_reference.npz',eef_xyz_m=local,rgb=rgb,
                source_step=state['step'],source_cloud_file=str(cloud_file),point_ids=np.flatnonzero(mask))
            from .carry_reference import initialize_reference, report
            epoch=getattr(getattr(self.hub,'episode_state',None),'data',{}).get('grasp_epoch',0)
            record['reference_scope']=binding.get('selection_scope','initial_observed_component')
            self.hub.carried_surface_reference=initialize_reference(record,local,rgb,epoch,np.flatnonzero(mask),world_xyz_m=points)
            record.update(report(self.hub.carried_surface_reference))
        except (ValueError,KeyError,OSError) as exc:
            record['reason']=str(exc)
        record['wall_s']=time.perf_counter()-started
        (folder/'scan_carry_reference.json').write_text(json.dumps(record,indent=2))
        self.hub.emit({'event':'scan_carry_reference_bound',**record})

    @timed('runtime_point_cloud_and_dimensions')
    def _refresh_runtime(self, packet, step, description, folder):
        """Update real step-start geometry without rendering presentation views."""
        for key in ('carried_observed_geometry', 'current_vertical_envelope', 'observed_surface_clearance'):
            packet.pop(key, None)  # Frozen stage-start evidence must not masquerade as current evidence.
        state = {**description['state'], **self.hub.robot.state()}
        state.setdefault('step', description['source_step'])
        packet['eef']={k:state[k] for k in ('position','quaternion_xyzw')}
        clearance = None
        if hasattr(self.hub.executor.ik, 'model'):
            import numpy as np
            ik = self.hub.executor.ik
            for name, value in zip(state['finger_joint_names'], state['finger_joint_positions']):
                ik.data.qpos[ik.model.joint(name).qposadr[0]] = value
            origin, rotation = ik.fk(state['q'])
            pads = [index for index in range(ik.model.ngeom) if 'pad_collision' in (ik.model.geom(index).name or '')]
            if len(pads) == 2:
                from .robot_geometry import gripper_geometry_in_eef
                packet['robot_gripper_geometry'] = gripper_geometry_in_eef(ik,origin,rotation,state)
                packet['eef']['tool_axes_world']={'finger_direction':rotation[:,2].tolist(),'closing_axis':rotation[:,0].tolist()}
                facts=self.hub.run/'robot_geometry_evidence.json'
                if facts.exists():
                    maximum=json.loads(facts.read_text()).get('max_observed_open_jaw',{}).get('current_inner_jaw_gap_m')
                    if maximum is not None:packet['robot_gripper_geometry']['max_observed_inner_jaw_gap_m']=maximum
            reference=getattr(self.hub,'carried_surface_reference',None)
            if step['primitive'] in POSE_ACTIONS and (state['gripper_command'] == -1. or getattr(self.hub,'carried_target_id',None) or reference is not None):
                from .observed_clearance import ObservedClearance
                clearance = ObservedClearance(ik, margin_m=.004)
                for source_frame in self.hub.observations[description['observation_id']].values():
                    clearance.add_frame(source_frame, state)
                clearance_report=clearance.build()
                (folder/'observed_clearance.json').write_text(json.dumps(clearance_report,indent=2))
                packet['observed_surface_clearance']={'checked_by_program':not getattr(self.hub,'ik_only',False),
                    'geometry_extracted':True,'complete_collision_planning':False}
                if state['gripper_command'] != -1.:
                    carried_id=getattr(self.hub,'carried_target_id',None)
                    if reference and carried_id and reference['object_id']!=carried_id:
                        reference=None
                    carried_id=carried_id or (reference['object_id'] if reference else None)
                    from .carry_reference import can_test, report, suspend
                    if reference is not None and can_test(reference):
                        held=clearance.separate_carried_surfaces(state,
                            reference['local_xyz_m'],reference['source_step'],preserve_reference=True,
                            reference_rgb=reference.get('rgb'))
                        if not held.get('reference_available'):
                            suspend(reference,state['step'],held.get('reason','Retained point set is unsupported'))
                        elif reference.get('validity')=='suspended':
                            visible=held.get('visible_support') or {}
                            if visible.get('status')=='insufficient_visibility':
                                # Occlusion can maintain a valid hypothesis, but cannot
                                # clear a previously measured contradiction.
                                clearance.carried_local=None;clearance.carried_rgb=None
                                clearance._reset_carried_proxy('reference_suspended_after_review')
                                held={**held,'reference_available':False,
                                    'reason':'A reviewed the original reference, but current depth does not yet support it'}
                            else:
                                reference['validity']='active'
                                reference['restored_at_step']=state['step']
                    else:
                        held={'reference_available':False,'method':'retained_reference_unavailable',
                            'source_step':state['step'],'rigid_attachment_verified':False,
                            'reason':(reference.get('suspension_reason') if reference else
                                'No explicitly associated carried point set. Near-hand components are not object identity. '
                                'Current pixels with pad/EEF references remain available for recovery.')}
                    packet['carried_observed_geometry']={'object_id':carried_id,**held}
                    if reference:
                        packet['carried_observed_geometry'].update(report(reference))
                        packet['carried_observed_geometry']['reference_source_kind']=reference.get('source_kind','observed_component')
                    packet['carried_observed_geometry']['held_surface_collision_checked']=False
                    if clearance.carried_local is not None:
                        from .observed_dimensions import carried_dimensions,write_measurement
                        measured=carried_dimensions(clearance.carried_local,
                            reference['source_step'] if reference else state['step'])
                        if reference:
                            measured['dimension_checked_at_step']=state['step']
                            measured['dimension_scope']='Retained observed surfaces transformed by EEF; current visibility status: '+str(
                                (held.get('visible_support') or {}).get('status','legacy_3d_support'))+'; not complete shape or holding certification'
                        write_measurement(self.hub.run,carried_id,measured)
                        np.savez(folder/'carried_reference.npz',eef_xyz_m=clearance.carried_local,
                            source_step=state['step'],reference_step=reference['source_step'] if reference else state['step'])
                    else:
                        from .observed_dimensions import invalidate_carried_dimensions
                        invalidate_carried_dimensions(self.hub.run,carried_id,state['step'],
                            'Current observation did not support carried-surface measurement')
            bounds=packet.get('robot_gripper_geometry',{}).get('hand_bounds_in_eef_tool_m')
            if bounds is not None:
                from .vertical_envelope import current_vertical_envelope
                packet['current_vertical_envelope']=current_vertical_envelope(state,bounds,
                    clearance.carried_local if clearance is not None else None)
                facts=self.hub.run/'robot_geometry_evidence.json'
                evidence=json.loads(facts.read_text()) if facts.exists() else {}
                evidence['current_vertical_envelope']={'source_step':state['step'],
                    **packet['current_vertical_envelope']}
                facts.write_text(json.dumps(evidence,indent=2,allow_nan=False))
        return clearance

    def _compile_selection(self, step, selection, packet, cloud_views, clearance, description, folder, prefix):
        audit=None;check={};validation_stage='output';params=selection;checked_route=None;prepared_motion=None
        try:
            schema=BINDING if step['primitive']=='move' else parameter_schema(step)
            Draft202012Validator(schema).validate(selection)
            if step['primitive'] in POSE_ACTIONS:
                from .perception import saved_frame
                carried=clearance.carried_local if clearance is not None else None
                if step['primitive'] in ROTATION_ACTIONS:
                    params,pose,audit=compile_rotation(selection,packet,step['primitive'])
                else:
                    params,audit=compile_binding(selection,packet,
                        lambda identity,camera:saved_frame(self.hub.run,identity,camera),carried,cloud_views=cloud_views)
                    pose=params
                audit.update(grounding_source_step=packet.get('grounding_source_step',packet['source_step']),
                    execution_source_step=description['source_step'],execution_observation_id=description['observation_id'])
                prefix.with_suffix('.compiled.json').write_text(json.dumps(audit,indent=2))
                if getattr(self.hub,'ik_only',False) and hasattr(self.hub.executor,'prepare_motion'):
                    from .timing import timings_for
                    validation_stage='ik_path'
                    with timings_for(self.hub).scope('numerical_motion_compilation'):
                        prepared_motion=self.hub.executor.prepare_motion([pose])
                    recorded=prepared_motion['planned']
                    check={'ik_pass':True,'path_checked':True,'collision_checked':False,
                        'robot_actions':0,'joint_positions':recorded['joint_target_final'],
                        'position_error_m':recorded['ik_endpoint_position_error_m'],
                        'rotation_error_rad':recorded['ik_endpoint_rotation_error_rad'],
                        'ik_wall_s':recorded['ik_wall_s'],'compiled_motion_id':prepared_motion['artifact_id']}
                    (folder/'compiled_motion.json').write_text(json.dumps(prepared_motion,allow_nan=False))
                else:
                    validation_stage='ik_endpoint'
                    check=check_pose(self.hub.executor.ik,self.hub.robot.state(),pose)
                    validation_stage='ik_path'
                    self.hub.executor.ik.cartesian_path(self.hub.robot.state()['q'],pose['position'],pose['quaternion_xyzw'])
                    check['path_checked']=True
                if getattr(self.hub,'ik_only',False):
                    check.update(mode='ik_only',collision_checked=False,observed_surface_checked=False)
                elif clearance is not None:
                    validation_stage='observed_surface'
                    checked_route=clearance.plan(self.hub.robot.state(),pose)
                    check['observed_surface_check']=checked_route
            else:check={'parameters_valid':True}
        except Exception as exc:
            if type(exc).__name__ not in ('IKError','ValueError','ValidationError','JSONDecodeError'):raise
            value={'ik_pass':check.get('ik_pass',False if validation_stage=='ik_endpoint' else None),
                'path_checked':check.get('path_checked',False),'failure_stage':validation_stage,'error':str(exc),'executed':False}
            if validation_stage=='observed_surface':value['observed_clearance_passed']=False
            self.hub.journal.pose({'model':getattr(self,'parameter_source','agent_a'),'source_step':description['source_step'],
                'primitive':step['primitive'],'raw_answer':selection,**value})
            feedback={**value,'compiled_from_selection':audit}
            failure=BindingUnavailable(json.dumps(feedback));failure.failure=feedback;failure.audit=audit
            failure.retry_context=(step,selection,packet,cloud_views,clearance,description)
            raise failure from exc
        if self.hub.robot.steps!=description['source_step']:
            raise BindingUnavailable('State advanced during local compilation')
        self.hub._bound_checked_route=checked_route
        self.hub.journal.pose({'model':getattr(self,'parameter_source','agent_a'),'source_step':description['source_step'],
            'primitive':step['primitive'],'proposal':params,**check,'executed':False})
        (folder/'parameters.json').write_text(json.dumps(params,indent=2))
        return {'parameters':params,'needs_observation':False,'audit':audit,'check':check,
            'checked_route':checked_route,'compiled_motion':prepared_motion}

    def recover_compile(self, failure, folder):
        """T1 changes numerical joint seeds only; target/pose/reference stay exact."""
        if (getattr(failure,'failure',{}).get('failure_stage') not in ('ik_endpoint','ik_path')
                or not hasattr(failure,'retry_context')
                or not hasattr(self.hub.executor.ik,'recovery_search')):
            raise failure
        step,selection,packet,views,clearance,description=failure.retry_context
        if self.hub.robot.steps!=description['source_step']:
            raise BindingUnavailable('T1 cannot use a stale compilation state')
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        with self.hub.executor.ik.recovery_search(timeout_s=5.):
            result=self._compile_selection(step,selection,packet,views,clearance,description,folder,folder/'same_goal')
        if failure.audit and result['audit']['compiled_pose']!=failure.audit['compiled_pose']:
            raise RuntimeError('T1 changed the absolute target pose')
        result['ik_recovery_search']=True
        return result

    def compile_step(self, stage_binding, index, actual_description, folder):
        """Resolve one saved expression using fresh real geometry, never an API."""
        from copy import deepcopy
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        self.hub._bound_checked_route=None
        def blocked(reason):
            result={'parameters':None,'needs_observation':True,'reason':reason,'audit':None,'check':{},'checked_route':None}
            (folder/'needs_observation.json').write_text(json.dumps(result,indent=2))
            return result
        if stage_binding['needs_observation']:return blocked('A saved action selection is unavailable; observation is required')
        if self.hub.robot.steps!=actual_description['source_step']:
            raise BindingUnavailable('Local compilation requires the actual step-start observation')
        step=stage_binding['plan']['steps'][index];selection=stage_binding['selections'][index]
        condition=stage_binding['conditions'][index]
        opening=step['primitive']=='release' or (step['primitive']=='set_gripper' and step.get('command')=='open')
        kind=(condition or {}).get('kind','previous_motion_converged' if opening and index>0 else 'always')
        if kind=='previous_motion_converged':
            motions=[r for r in stage_binding.get('execution_receipts',[]) if r.get('primitive') in POSE_ACTIONS]
            if not motions or motions[-1].get('result',{}).get('converged') is not True:
                return blocked('The preceding actual motion has no converged execution receipt')
        elif kind!='always':return blocked('Condition requires new semantic/visual evidence: '+kind)
        packet=deepcopy(stage_binding['frozen_packets'][index])
        location=(stage_binding.get('locations') or [{}]*len(stage_binding['plan']['steps']))[index]
        grounding_step=location.get('grounding_source_step',stage_binding['source_step'])
        operation_target=packet.get('operation_reference_target_id',step['target_id'])
        if step['primitive']=='move' and selection['anchor']['kind'] in ('cloud_point','image_point'):
            record=self.hub.journal.data.get('objects',{}).get(operation_target,{})
            if (record.get('invalidated_at_step') is not None and
                    record['invalidated_at_step']>grounding_step):
                return blocked('The selected object moved after its frozen pixel evidence')
            if (operation_target==getattr(self.hub,'carried_target_id',None)
                    and actual_description['source_step']>grounding_step):
                return blocked('Held-object pixels are frozen before intervening motion; use a static destination or dynamic EEF reference')
        packet['grounding_source_step']=grounding_step;packet['source_step']=actual_description['source_step']
        held_id=getattr(self.hub,'carried_target_id',None)
        command=self.hub.robot.state().get('gripper_command')
        reference=getattr(self.hub,'carried_surface_reference',None)
        carry_hypothesis=held_id or ((reference or {}).get('object_id'))
        carry_check=bool(carry_hypothesis and command == 1.)
        runtime_step={**step,'primitive':'move'} if carry_check and step['primitive'] not in POSE_ACTIONS else step
        clearance=self._refresh_runtime(packet,runtime_step,actual_description,folder)
        (folder/'packet.json').write_text(json.dumps({'context':packet,'grounding_observation_id':stage_binding['observation_id'],
            'execution_observation_id':actual_description['observation_id']},indent=2))
        self._check_contact_state(packet,stage_binding['frozen_packets'],folder=folder)
        if step['primitive']=='move' and selection['reference'] in ('held_bottom','held_center') and (
                clearance is None or clearance.carried_local is None):
            return blocked('Actual carried surfaces are unavailable for this step reference')
        result=self._compile_selection(step,selection,packet,stage_binding['cloud_views'][index],clearance,
            actual_description,folder,folder/'local_compile')
        return result

    def check_actual_contact(self, stage, index, description, folder):
        """Fresh failure/outcome evidence uses the same gate as every route point."""
        from copy import deepcopy
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        packet=deepcopy(stage['frozen_packets'][index])
        packet['source_step']=description['source_step']
        self._refresh_runtime(packet,{**stage['plan']['steps'][index],'primitive':'move'},description,folder)
        self._check_contact_state(packet,stage['frozen_packets'],folder=folder)


def symbolic_schema_feedback(exc, schema, answer):
    """Report field paths/constraints without echoing a whole invalid plan."""
    import re
    pending=list(Draft202012Validator(schema).iter_errors(answer)) or [exc]
    issues=[]
    while pending:
        error=pending.pop(0)
        if error.context:
            pending[:0]=list(error.context)
            continue
        if error.validator=='type' and error.validator_value=='null' and error.instance is not None:
            continue  # The object branch gives the actionable plan errors.
        path='/'.join(map(str,error.absolute_path)) or '$'
        keyword=error.validator
        if keyword=='required':
            details=[f'{path}/{key}: required field is missing' for key in error.validator_value
                if key not in error.instance]
        elif keyword=='not' and isinstance(error.instance,dict) and isinstance(error.validator_value,dict):
            branches=error.validator_value.get('anyOf',[error.validator_value])
            forbidden=[key for branch in branches if branch.get('required')
                and Draft202012Validator(branch).is_valid(error.instance) for key in branch['required']]
            details=[f'{path}: forbidden field combination here: '+', '.join(forbidden)] if forbidden else [
                f'{path}: violates the supplied forbidden-combination constraint']
        elif keyword=='additionalProperties' and isinstance(error.instance,dict):
            properties=error.schema.get('properties',{})
            patterns=error.schema.get('patternProperties',{})
            extras=[key for key in error.instance if key not in properties
                and not any(re.search(pattern,key) for pattern in patterns)]
            details=[f'{path}/{key}: field is not allowed here' for key in extras]
        elif keyword=='enum':
            details=[f'{path}: choose a value from this field\'s supplied enum']
        elif keyword=='type' and error.validator_value in ('array','object') and isinstance(error.instance,str):
            expected=error.validator_value
            try:
                json.loads(error.instance)
            except json.JSONDecodeError as syntax:
                details=[f'{path}: expected a JSON {expected}; serialized text contains invalid JSON '
                    f'at line {syntax.lineno}, column {syntax.colno}, character {syntax.pos}: {syntax.msg}. '
                    'Check brackets and commas; submit the structured value, not a quoted string.']
            else:
                details=[f'{path}: expected a JSON {expected} value, received a string. '
                    'Submit the structured value, not quoted JSON text.']
        elif not isinstance(keyword,str):
            details=[f'{path}: {error.message[:500]}']
        else:
            expected=json.dumps(error.validator_value,ensure_ascii=False)[:160]
            details=[f'{path}: must satisfy {keyword}={expected}']
        for detail in details:
            if detail not in issues:issues.append(detail)
    # Repeated errors inside early array items must not hide independent
    # envelope errors until the last permitted correction has already ended.
    issues.sort(key=lambda detail: detail.split(':', 1)[0].count('/'))
    feedback = '; '.join(issues[:12])
    if len(issues) > 12 or len(feedback) > 1800:
        feedback = feedback[:1680]+'; Additional schema errors omitted; check the complete supplied schema.'
    return feedback or 'The submission does not match the supplied schema.'


def execute_sketch(hub, grounder, plan, index, *, frozen_binding):
    outputs = []
    description,pictures=hub.observe()
    if getattr(hub.robot,'done',False):
        return {'subtask':plan['subtask'],'steps':outputs,'commands_completed':False,
            'episode_ended':True,'source_step':description['source_step'],'remaining_suffix_executed':False}
    stage_folder=hub.run/'execution'/f'subtask_{index:02d}'
    try:
        if plan['source_step'] != description['source_step'] or hub.robot.steps != description['source_step']:
            raise BindingUnavailable('Sketch source state is stale; no binding or actions dispatched')
        if frozen_binding['semantic_plan'] != plan or frozen_binding['source_step'] != description['source_step']:
            raise ValueError('Saved action batch differs from its submitted plan or source step')
        stage_binding = deepcopy(frozen_binding)
        hub.episode_state.begin_stage(plan)
        folder = stage_folder/'binding'
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'stage_binding.json').write_text(json.dumps(stage_binding, indent=2))
        hub.emit({'event': 'action_batch_prepared', 'source_step': description['source_step'],
                  'future_ik_checked': False})
    except BindingUnavailable as exc:
        receipt = {'subtask': plan['subtask'], 'steps': outputs, 'commands_completed': False,
                   'binding_failure': str(exc), 'binding_scope': 'stage', 'failed_plan': plan,
                   'failed_step': plan['steps'][0] if len(plan['steps']) == 1 else None,
                   'source_step': hub.robot.steps, 'remaining_suffix_executed': False}
        hub.emit({'event': 'sketch_grounding_unavailable', 'subtask': index, 'receipt': receipt})
        return receipt
    if stage_binding.get('needs_observation'):
        return {'subtask':plan['subtask'],'steps':outputs,'commands_completed':False,
            'needs_observation':True,'reason':stage_binding.get('reason','Current observations do not support the complete action batch'),
            'unbound_step_assessments':stage_binding.get('unbound_step_assessments',[]),
            'source_step':hub.robot.steps,'remaining_suffix_executed':False}
    stage_binding.setdefault('plan',plan)
    stage_binding['execution_receipts']=outputs
    number=0
    while number<len(stage_binding['plan']['steps']):
        step=stage_binding['plan']['steps'][number]
        location=(stage_binding.get('locations') or [{}]*len(stage_binding['plan']['steps']))[number]
        if hub.robot.done:
            return {'subtask':plan['subtask'],'steps':outputs,'commands_completed':False,
                'episode_ended':True,'source_step':hub.robot.steps,'remaining_suffix_executed':False}
        if hub.robot.steps >= hub.robot.max_steps:
            raise RuntimeError('Episode action budget exhausted')
        description, pictures = hub.observe()
        folder=stage_folder/f'step_{number:02d}'
        try:
            from .step_recovery import run_step
            outcome=run_step(hub,grounder,stage_binding,number,description,pictures,folder,subtask_index=index)
            compiled=outcome['compiled']
            if compiled.get('needs_observation'):
                return {'subtask':plan['subtask'],'steps':outputs,'commands_completed':False,
                    'failed_step':step,'failed_step_index':number,'failed_location':location,'needs_observation':True,
                    'reason':compiled.get('reason','Current observations do not support this binding'),
                    'source_step':hub.robot.steps,'remaining_suffix_executed':False}
            params=compiled['parameters']
        except BindingUnavailable as exc:
            # Preserve the actual failure and return to the same agent.
            receipt = {'subtask':plan['subtask'], 'steps':outputs, 'commands_completed':False,
                'failed_step':step, 'binding_failure':str(exc),
                'failed_step_index':number,'failed_location':location,'recovery':getattr(exc,'recovery',[]),
                'source_step':hub.robot.steps, 'remaining_suffix_executed':False}
            hub.emit({'event':'sketch_grounding_unavailable', 'subtask':index, 'receipt':receipt})
            return receipt
        output=outcome['output']
        executed_primitive=compiled.get('executed_primitive',step['primitive'])
        if output.get('episode_ended') or hub.robot.done:
            actual,_=hub.observe()
            receipt={'primitive':executed_primitive,'parameters':params,'result':output,
                'expanded_step_index':number,'location':location,'intent':step.get('intent'),
                'source_step':actual['source_step'],'observation_id':actual['observation_id'],
                'recovery':outcome['recovery'],
                'episode_ended':True}
            outputs.append(receipt)
            grounder.record_execution(receipt)
            hub.emit({'event':'bound_step_episode_ended','subtask':index,'index':number,**receipt})
            return {'subtask':plan['subtask'],'steps':outputs,'commands_completed':False,
                'episode_ended':True,'source_step':actual['source_step'],
                'observation_id':actual['observation_id'],'remaining_suffix_executed':False}
        if output.get('converged') is False or outcome.get('failure'):
            actual,pictures=hub.observe()
            receipt={'primitive':executed_primitive,'parameters':params,'result':output,
                'expanded_step_index':number,'location':location,'intent':step.get('intent'),
                'source_step':actual['source_step'],'observation_id':actual['observation_id'],
                'recovery':outcome['recovery'],'recovery_failure':outcome.get('failure')}
            outputs.append(receipt)
            grounder.record_execution(receipt)
            hub.emit({'event':'bound_step_execution_failed','subtask':index,'index':number,**receipt})
            return {'subtask':plan['subtask'],'steps':outputs,'commands_completed':False,
                'failure':'Local recovery failed; suffix not executed' if outcome.get('failure') else 'Motion did not converge; suffix not executed',
                'remaining_suffix_executed':False,
                'failed_step_index':number,'failed_location':location,'recovery':outcome['recovery'],
                'source_step':actual['source_step'],'observation_id':actual['observation_id']}
        if step['primitive'] in ('release','set_gripper'):
            if hasattr(hub,'single_last_motion_failure'):
                hub.single_last_motion_failure=None
            from .observed_dimensions import invalidate_carried_dimensions
            reference=getattr(hub,'carried_surface_reference',None) or {}
            former_carried=getattr(hub,'carried_target_id',None) or reference.get('object_id')
            location_uncertain={former_carried}-{None}
            if (step['target_id'] is not None and
                    (step['primitive']=='release' or step.get('command')=='close')):
                location_uncertain.add(step['target_id'])
            for identity in location_uncertain:
                hub.journal.mark_moved(identity,hub.robot.steps)
            state=getattr(hub,'episode_state',None)
            if state is not None:
                state.record_gripper_event(hub.robot.steps, step.get('command','open'), step['target_id'])
                state.retire_route_failure(hub.robot.steps,'Actual gripper event changed the contact reference')
            for identity in {former_carried, reference.get('object_id'), step['target_id']}:
                if identity:
                    invalidate_carried_dimensions(hub.run,identity,hub.robot.steps,
                        'Gripper action changed the previous grasp reference')
            hub.carried_surface_reference=None
            if (step.get('command')=='close' and step['target_id'] is not None
                    and hasattr(grounder,'seed_scan_carry')):
                grounder.seed_scan_carry(step['target_id'],folder)
        if step['primitive']=='release' or (step['primitive']=='set_gripper' and step['command']=='open'):
            hub.carried_target_id=None
            hub.carried_surface_reference=None
            hub.last_contact_binding=None
        audit=compiled.get('audit') or {}
        if (step['primitive']=='move' and step.get('reference_target_id',step['target_id'])==step['target_id']
                and (audit.get('binding') or {}).get('anchor',{}).get('kind') in ('cloud_point','image_point')):
            hub.last_contact_binding={'target_id':step['target_id'],'audit':audit,
                'views':stage_binding['cloud_views'][number],
                'geometry_binding':deepcopy(next((item.get('geometry_binding',{}) for item in grounder.handoff.get('objects',[])
                    if item['object_id']==step['target_id']),{}))}
        actual, pictures = hub.observe()
        receipt = {'primitive': step['primitive'], 'parameters': params, 'result': output,
                   'expanded_step_index':number,'intent':step.get('intent'),
                   'source_step': actual['source_step'], 'observation_id': actual['observation_id'],
                   'recovery':outcome['recovery'],'completed':True,'location':location}
        outputs.append(receipt)
        grounder.record_execution(receipt)
        hub.emit({'event': 'bound_step_execution_finished', 'subtask': index, 'index': number, **receipt})
        hub.journal.sync_observations(hub.robot.steps)
        number+=1
    return {'subtask': plan['subtask'], 'steps': outputs, 'commands_completed': True,
            'source_step': actual['source_step'], 'observation_id': actual['observation_id']}
