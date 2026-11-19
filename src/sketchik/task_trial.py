"""One persistent agent: program observation, parameterized sketches and pure IK."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .agent import ChatClient, load_settings, transcript_copy, TrialClient
from .planning import planning_dialogue, save_planning_history, image_part
from .geometry_journal import GeometryJournal
from .scene_cloud import save_scene, region_points
from .perception import ObservationHistory, snapshot
from .task_execution import hand_observation_parts
from .timing import timed, timings_for
from .tools import ToolHub, public

PROTOCOL_VERSION = 72


def semantic_observation_description(description):
    """A's observation facts; complete calibration remains in the saved source."""
    result=deepcopy(description)
    for camera in result.get('cameras',[]):
        for key in ('K','world_from_camera'):
            camera.pop(key,None)
        camera['calibration_scope']='Saved with native RGB-D; the program performs coordinate transforms.'
    return result


def semantic_display_annotation(metadata):
    """The visible compass is sufficient for A's qualitative directions."""
    return {key:deepcopy(metadata[key]) for key in (
        'panel_bounds_xyxy','annotation_pixels_not_surface','projection',
        'observed_surface_evidence') if key in metadata} | {
        'direction_reference':'WORLD compass shown in this image; direction only, no point or metric scale.'}


class ObservationHub(ToolHub):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, allowed_tools=frozenset(), **kwargs)
        self.journal = GeometryJournal(self.run)
        self._surface_evidence = {}
        self._observation_cache = None
        self._display_cache = {}
        self.observation_reuse_count = 0

    def display_image(self, observation_id, camera, frame):
        key = (observation_id, camera)
        if key in self._display_cache:
            return deepcopy(self._display_cache[key])
        from PIL import Image
        from .image_axes import annotate_world_axes
        from .surface_evidence import summarize_horizontal_surfaces
        rgb, metadata = annotate_world_axes(frame['rgb'], frame['world_from_camera'],
            source_identity={'observation_id':observation_id,'camera':camera})
        if key not in self._surface_evidence:
            try:
                self._surface_evidence[key] = summarize_horizontal_surfaces(frame)
            except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
                self._surface_evidence[key] = {'scope': 'unavailable', 'error_type': type(exc).__name__}
        metadata['observed_surface_evidence'] = self._surface_evidence[key]
        folder = self.run/'observations'/observation_id/'a_display'
        folder.mkdir(exist_ok=True)
        picture = folder/(camera+'.png')
        Image.fromarray(rgb).save(picture, compress_level=1)
        (folder/(camera+'.json')).write_text(json.dumps(metadata, indent=2))
        display = semantic_display_annotation(metadata)
        evidence = metadata['observed_surface_evidence']
        display['observed_surface_evidence'] = {
            'scope': evidence['scope'], 'horizontal_patches': [
                {name: patch[name] for name in ('bounds_xyxy', 'world_height_m', 'normal_angle_to_z_deg')}
                for patch in evidence.get('horizontal_patches', [])]}
        result = display, image_part(picture)
        self._display_cache[key] = deepcopy(result)
        return result

    @timed('observation_acquisition')
    def observe(self):
        # This live hub owns all simulation advancement. Equal full public robot
        # state/step and camera configuration denotes the same physical instant.
        # Do not reuse across a step, even if the arm returns to identical joints.
        state = self.state()
        key = json.dumps({'state': state, 'cameras': list(self.robot.cameras),
            'size': getattr(self.robot, 'size', None)}, sort_keys=True, allow_nan=False)
        cached = self._observation_cache
        if cached is not None and cached[0] == key:
            self.observation_reuse_count += 1
            self.emit({'event': 'observation_reused', 'observation_id': cached[1]['observation_id'],
                'source_step': state['step'], 'new_camera_renders': 0, 'new_rgbd_files': 0})
            return deepcopy(cached[1]), deepcopy(cached[2])
        description,parts=super().observe()
        # Only the default model text changes. The full program return and
        # observation.json/NPZ are already saved with exact calibration.
        parts[0]={'type':'text','text':json.dumps(semantic_observation_description(description))}
        if description['state'] != state:
            raise RuntimeError('Robot state changed during RGB-D capture; no mixed-time observation reused')
        self._observation_cache = key, deepcopy(description), deepcopy(parts)
        return description,parts

    @timed('whole_scene_point_cloud_and_views')
    def make_scene(self, observation_id):
        existing = self.journal.data.get('scene_clouds', {}).get(observation_id)
        if existing is not None:
            self.journal.data['latest_scene'] = deepcopy(existing)
            self.emit({'event': 'whole_scene_reused', 'observation_id': observation_id})
            return deepcopy(existing)
        folder = self.run / 'scene_clouds' / observation_id
        scene = save_scene(self.observations[observation_id], folder, observation_id)
        self.journal.data['latest_scene'] = scene
        self.journal.data.setdefault('scene_clouds', {})[observation_id] = scene
        self.journal.save()
        self.emit({'event': 'whole_scene_saved', **scene})
        return scene

    def scene_picture(self, scene):
        return [{'type': 'text', 'text': 'Entire observed scene; world metres. No segmentation, hidden surfaces unknown.'},
                image_part(Path(scene['cloud_image']))]

    def observe_scene(self):
        description, images = self.observe()
        scene = self.make_scene(description['observation_id'])
        description['scene_cloud'] = {key: scene[key] for key in
            ('point_count', 'point_cloud_file', 'cloud_image', 'world_min_xyz_m', 'world_max_xyz_m', 'wall_s')}
        return description, images + self.scene_picture(scene)



def compact_view_feedback(result):
    if 'max_views_per_need' in result:
        return {key:result[key] for key in ('status','source_step','captured_views','wall_s',
            'observation_ids','max_views','semantic_coverage_certified') if key in result} | {
            'needs':[{'object_ids':attempt['object_ids'],'need':attempt['need'],
                'acquisition_status':attempt['acquisition_status'],
                'actual_views':[v.get('observation_id') for v in attempt['views'] if v.get('observation_id')],
                'ik_checks':sum(len(v['checks']) for v in attempt['views'])}
                for attempt in result['attempts']]}
    return {key: result[key] for key in ('status', 'source_step', 'wall_s', 'captured_views',
        'observation_ids', 'diagnostic_file') if key in result} | {'targets': [
        {'target_id': item['target']['target_id'], 'status': item['status'],
         'observation_id': item.get('captured_observation_id'),
         **({'request_feedback':item['request_feedback']} if item.get('request_feedback') else {}),
         **({'error': item['error']} if item.get('error') else {})}
        for item in result['attempts']]}


class ProgramObservationHub(ObservationHub):
    """A names missing surfaces; only Python chooses and executes camera poses."""
    def scene_picture(self, scene):
        # A judges visible surfaces from native RGB. Numeric cloud artifacts
        # remain complete locally for the execution grounding packet.
        return []


def reset_observation_arm(hub):
    """Check the configured motion route, then physically return home."""
    folder=hub.run/'observation_resets';folder.mkdir(exist_ok=True)
    path=folder/f'reset_{len(list(folder.glob("reset_*.json"))):03d}.json'
    begin=time.perf_counter();state=hub.robot.state()
    clearance=None
    if not getattr(hub,'ik_only',False):
        from .observed_clearance import ObservedClearance
        from .perception import saved_frame
        clearance=ObservedClearance(hub.executor.ik)
        sources=hub.journal.data.get('scene_clouds',{})
        for identity in sources:
            metadata=json.loads((hub.run/'observations'/identity/'observation.json').read_text())
            for camera in ('agentview','robot0_eye_in_hand'):
                frame=hub.observations.get(identity,{}).get(camera)
                if frame is None:frame=saved_frame(hub.run,identity,camera)
                clearance.add_frame(frame,metadata['state'])
        if not sources:
            for camera in ('agentview','robot0_eye_in_hand'):
                clearance.add_frame(hub.robot.camera(camera),state)
        clearance.build()
    home=hub.executor.initial_state
    record={'source_step':state['step'],'initial_source_step':home['step'],
        'environment_reset':False,'object_states_reset':False}
    try:
        from .motion_checks import motion_route,observation_return_route
        from .kinematics import IKError
        try:
            route=motion_route(hub,clearance,state,{key:home[key] for key in ('position','quaternion_xyzw')})
        except IKError as direct_error:
            if hub.robot.steps!=state['step']:
                raise RuntimeError('Reset planning advanced physical state; no automatic recovery') from direct_error
            previous=[json.loads(item.read_text()) for item in folder.glob('reset_*.json') if item!=path]
            since=max([home['step'],*[item['source_step'] for item in previous
                if item.get('execution',{}).get('converged') is True and item['source_step']<=state['step']]])
            record['direct_ik_rejection']=str(direct_error)
            route=observation_return_route(hub,clearance,state,home,since)
            hub.emit({'event':'observation_reset_return_route_planned','source_step':state['step'],
                'waypoints':len(route['waypoints']),'strategy':route['strategy'],
                'original_full_endpoint_preserved':True,'robot_actions_during_planning':0})
        record['checked_route']=route
        record['execution']=public(hub.executor.reset(checked_waypoints=route['waypoints']))
        actual,pictures=hub.observe_scene()
        pictures=[*pictures,*hand_observation_parts(hub,actual,path.with_suffix('')/'hand_geometry')]
        record.update(source_step=hub.robot.steps,observation_id=actual['observation_id'],
            wall_s=time.perf_counter()-begin)
        path.write_text(json.dumps(record,indent=2))
        hub.emit({'event':'observation_arm_reset',**record})
        if not record['execution']['converged']:
            raise ValueError('Observation arm reset did not converge; no manipulation dispatched')
        return record,pictures
    except Exception as exc:
        record.update(error=f'{type(exc).__name__}: {exc}',wall_s=time.perf_counter()-begin)
        path.write_text(json.dumps(record,indent=2))
        hub.emit({'event':'observation_arm_reset_failed',**record})
        hub.fatal=True
        raise


def append_robot_physical_facts(hub,messages):
    """Give fresh and resumed A the same robot-only geometry before planning."""
    from .robot_geometry import gripper_geometry_in_eef
    ik,state=hub.executor.ik,hub.robot.state()
    for name,value in zip(state['finger_joint_names'],state['finger_joint_positions']):
        ik.data.qpos[ik.model.joint(name).qposadr[0]]=value
    origin,rotation=ik.fk(state['q'])
    facts=gripper_geometry_in_eef(ik,origin,rotation,state)
    if facts is None:
        raise ValueError('Robot hand geometry unavailable before A planning')
    path=hub.run/'robot_geometry_evidence.json'
    evidence=json.loads(path.read_text()) if path.exists() else {}
    evidence['initial_hand_geometry']=facts
    control_context=getattr(hub.executor,'gripper_control_context',None)
    control_facts=control_context() if control_context is not None else None
    if control_facts is not None:evidence['gripper_control']=control_facts
    path.write_text(json.dumps(evidence,indent=2,allow_nan=False))
    messages.append({'role':'user','content':json.dumps({'robot_physical_facts':facts,
        **({'gripper_control':control_facts} if control_facts is not None else {}),
        'instruction':'Use individual finger lengths/thickness, pad offsets, current jaw gap and the entire palm/crossbar envelope when choosing grasp manner, approach, insertion and withdrawal. The current tool schema defines whether you submit semantics or actual parameters. All dimensions are in EEF tool axes and change world orientation with the hand. The palm may collide even when fingertips fit.'},allow_nan=False)})
    hub.emit({'event':'initial_robot_geometry_shared','source_step':state['step'],
        'hand_span_xyz_m':facts['hand_span_xyz_m'],'finger_count':len(facts['fingers']),
        'current_inner_jaw_gap_m':facts.get('current_inner_jaw_gap_m')})
    return facts


def table_observation(client,hub,messages,*,max_turns,max_tool_calls):
    from .table_scan import run_table_scan
    from .grounding_packet import prepare_table_context
    from .scene_catalog import (build_scene_catalog,public_catalog,binding_schema,bind_requirements)
    from .guided_observation import run_guided_observation,interior_patches
    from .stage_context import ACQUISITION_RECORD
    (hub.run/'a_scan_tools.json').write_text('[]')
    result,pictures=run_table_scan(hub)
    reset,home_pictures=reset_observation_arm(hub)
    pictures.extend(home_pictures)
    result.update(source_step=hub.robot.steps,reset=reset)
    table=prepare_table_context(hub)
    if table is None:raise ValueError('Fixed scan produced no measured table context')
    hub.scene_catalog=catalog=build_scene_catalog(hub)
    # An acquisition request is historical user context once the batch ends.
    # Keeping it as another system instruction made its obsolete tool/parameter
    # restrictions compete with the execution phase before the first checkpoint.
    messages.append({'role':'user','content':json.dumps({
        'phase':'initial_acquisition_request',
        'scope':'Only the following submit_scene_requirements request; expires when the acquisition batch ends. Retained later as historical context.',
        'instruction':(Path(__file__).resolve().parents[2]/'prompts/table_scan_review.md').read_text()},
        ensure_ascii=False)})
    messages.append({'role':'user','content':[{'type':'text','text':json.dumps({
        'phase':ACQUISITION_RECORD,'kind':'baseline_catalog','instruction':
        'Whole-table baseline scan is finished and the arm has returned to its initial full pose. '
        'The point-cloud images show one full-table oblique, then an opposite oblique and overhead view framed around tabletop objects when possible, followed by RGB observations. G IDs stay consistent across the views. '
        'All G groups and ungrouped tabletop geometry remain, including unrelated obstacles. '
        'G labels are observed geometric fragments, not object identities: touching objects can merge '
        'and one object can split. Bind ALL task contact objects/parts to one or more component_ids; '
        'several parts may share a G. Choose required_surfaces and missing_surfaces. '
        'Use source_regions only if a necessary part is missing from the catalog. No contact pixels, '
        'camera poses, plan or coverage certification now. The program will acquire complementary '
        'real views in one bounded batch, then return all target geometry for semantic review and sketch. '
        ,'catalog':public_catalog(catalog)},ensure_ascii=False)},
        *(image_part(Path(view['image_file'])) for view in catalog['views']),*pictures]})
    hub.schemas=[];hub.validators={}
    tool={'type':'function','function':{'name':'submit_scene_requirements',
        'description':'Bind all task contact parts to measured scene groups and name missing surfaces. Program chooses cameras.',
        'parameters':binding_schema(catalog)}}
    (hub.run/'a_tools.json').write_text(json.dumps([tool],indent=2))
    requirements,response,timing=planning_dialogue(client,hub,messages,tool,
        max_turns=max_turns,max_tool_calls=max_tool_calls)
    if requirements['source_step']!=hub.robot.steps:raise ValueError('Scene binding source state is stale')
    handoff=bind_requirements(requirements,catalog)
    batch,pictures=run_guided_observation(hub,catalog,handoff,**getattr(hub,'supplement_budgets',{}))
    if batch['status']=='stopped_after_motion_failure':
        raise RuntimeError('Supplemental camera motion did not converge; stop before manipulation')
    if batch['robot_actions']:
        reset,home=reset_observation_arm(hub)
        pictures.extend(home);timing['supplement_reset_wall_s']=reset['wall_s']
    if batch['observation_ids']:
        manifest=hub.run/'scan_evidence.json'
        evidence=json.loads(manifest.read_text())
        known={(v['observation_id'],v['camera']) for v in evidence['sources']}
        for identity in batch['observation_ids']:
            metadata=json.loads((hub.run/'observations'/identity/'observation.json').read_text())
            for camera in ('agentview','robot0_eye_in_hand'):
                if (identity,camera) not in known:
                    evidence['sources'].append({'observation_id':identity,'camera':camera,
                        'source_step':metadata['source_step']})
        manifest.write_text(json.dumps(evidence,indent=2))
        prepare_table_context(hub)
    messages.append({'role':'user','content':[{'type':'text','text':json.dumps({
        'phase':ACQUISITION_RECORD,'kind':'supplement_result','instruction':
        'The program acquisition batch has ended; the initial acquisition-only request is now historical. '
        'Use the tools supplied for the current execution phase, not submit_scene_requirements. '
        'These are actual new images. Acquisition quality '
        'and IK convergence do not certify an interior/contact surface. Geometry follows, then confirm '
        'actual coverage and choose the first stage together. No additional observation loop. '
        ,'feedback':compact_view_feedback(batch)},ensure_ascii=False)},*pictures]})
    hub.initial_preview_sources={}
    for item in handoff['objects']:
        components=[catalog['components'][key] for key in item['component_ids']]
        patches=interior_patches(components) if 'interior_support' in item['required_surfaces'] else []
        refs=patches or [ref for component in components for ref in component['sources']
            if ref['observation_id'] in batch['observation_ids']]
        if refs:
            ref=refs[0]
            hub.initial_preview_sources[item['object_id']]={**{key:ref[key] for key in
                ('observation_id','camera','bounds_xyxy')},'surface_role':'overall_shape',
                'visible_evidence':'Program-selected actual evidence window; A must confirm semantic surface.'}
    hub.initial_supplement_ids=batch['observation_ids']
    handoff['source_step']=hub.robot.steps
    (hub.run/'scene_catalog/catalog_after_acquisition.json').write_text(json.dumps(catalog,indent=2))
    interfaces=hub.run/'interfaces.json'
    if interfaces.exists():
        data=json.loads(interfaces.read_text());data['a_scene_requirements']=tool['function']['parameters']
        interfaces.write_text(json.dumps(data,indent=2))
    return handoff,response,{**timing,'mode':'full_table_catalog_and_adaptive_supplement',
        'supplement_batches':1,'supplement_wall_s':batch['wall_s'],
        'supplement_captured_views':batch['captured_views'],
        'fixed_scan_agent_requests':0,'fixed_scan_wall_s':result['wall_s'],
        'initial_reset_wall_s':result['reset']['wall_s'],'captured_table_views':result['captured_views']}


def write_dimensions(hub, expected_objects=()):
    identities = set(hub.journal.data['objects']) | {item['object_id'] for item in expected_objects}
    data = {'unit': 'm', 'coordinate_frame': 'world', 'objects': {}}
    for identity in sorted(identities):
        record = hub.journal.data['objects'].get(identity, {})
        summary = record.get('observed_geometry') or {}
        data['objects'][identity] = {
            'observed_span_xyz_m': summary.get('observed_span_xyz_m'),
            'observed_principal_spans_m': summary.get('observed_principal_spans_m'),
            'observed_principal_axes_world': summary.get('observed_principal_axes_world'),
            'principal_axis_gap_ratios': summary.get('principal_axis_gap_ratios'),
            'scope': record.get('scope', 'unknown'),
            'source_steps': record.get('source_steps', [record['source_step']] if 'source_step' in record else []),
            'status': record.get('geometry_status', 'not_measured'),
            'complete_dimensions_known': False, 'object_dimensions_m': None, 'usable_cavity_dimensions_m': None}
    (hub.run / 'dimensions.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def package_handoff(hub, handoff):
    ids = [item['object_id'] for item in handoff['objects']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate whole-task object identity')
    if handoff['source_step'] != hub.robot.steps:
        raise ValueError('Handoff source state is stale')
    for item in handoff['objects']:
        for evidence in item['evidence']:
            hub.observations[evidence['observation_id']][evidence['camera']]
    for item in handoff['objects']:
        regions = []
        for evidence in item['evidence']:
            if 'bounds_xyxy' in evidence:
                frame = hub.observations[evidence['observation_id']][evidence['camera']]
                xyz = region_points(frame, evidence['bounds_xyxy'])
                if len(xyz) >= 64:
                    regions.append((xyz, frame['step']))
        if regions:
            from .geometry_cache import observed_summary
            hub.journal.update_object(item['object_id'], {
                'description': item['description'], 'needed_geometry': item['needed_geometry'],
                'geometry_status': 'observed_coarse_region', 'scope': 'image rectangles; not object masks',
                'observed_geometry': observed_summary(np.concatenate([v[0] for v in regions])),
                'source_steps': sorted({v[1] for v in regions}), 'semantic_identity_verified': False})
    hub.journal.sync_observations(hub.robot.steps)
    if not (hub.run/'table_context/context.json').exists():
        from .grounding_packet import prepare_table_context
        table=prepare_table_context(hub)
        if table:hub.emit({'event':'multiview_table_handoff_prepared',**table})
    write_dimensions(hub, handoff['objects'])
    from .grounding_packet import prepare_cavity_dimensions
    prepare_cavity_dimensions(hub,handoff)
    packet = {'task': hub.robot.metadata['task_language'], **handoff,
              'geometry': hub.journal.data['objects'], 'observations': hub.journal.data['observations'],
              'coverage_checked_by': 'Agent A; semantic judgment, not simulator truth',
              'scene_clouds': hub.journal.data.get('scene_clouds', {})}
    packet['dimensions_file'] = 'dimensions.json'
    (hub.run / 'handoff.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2))
    return packet


def save_context_manifest(folder, messages, source, source_step):
    images = [part for message in messages if isinstance(message.get('content'), list)
              for part in message['content'] if part.get('type') == 'image_url']
    (folder/'context_manifest.json').write_text(json.dumps({
        'protocol_version': PROTOCOL_VERSION, 'episode_id': folder.name,
        'scope': 'Complete verified source prefix and this episode; no extra independent-trial histories',
        'verified_replay_source': str(source) if source else None, 'source_step': source_step,
        'source_messages': len(messages), 'source_image_occurrences': len(images),
        'source_unique_images': len({part['image_url']['url'] for part in images}),
        'additional_context_sources': [], 'source_history_loaded_completely': True}, indent=2))


def make_task_client(args, folder, settings=None):
    """Create the single episode client; no secondary model backend."""
    common = {'image_jpeg_quality': args.image_jpeg_quality,
              'reference_exact_images': args.reference_exact_images}
    if getattr(args, 'a_image_window', False) or getattr(args, 'a_context_window', False):
        from .image_context import ImageWindow
        common['image_window'] = ImageWindow(archive_reader='read_image')
        common['image_window'].active = False
    if getattr(args, 'a_context_window', False):
        from .stage_context import StageContext
        common['stage_context'] = StageContext(retain_robot_facts=False,
            keep_messages=40, keep_chars=60_000, retain_episode_references=True)
    settings = load_settings(args.env_file) if settings is None else settings
    return ChatClient(settings.get('SKETCHIK_BASE_URL'), settings.get('SKETCHIK_MODEL'),
        settings.get('SKETCHIK_API_KEY'), settings.get('SKETCHIK_REASONING'),
        timeout=getattr(args, 'api_timeout_s', 180.),
        max_tokens=getattr(args, 'api_max_tokens', None), stream=getattr(args, 'api_stream', False),
        budget_field=getattr(args, 'api_budget_field', 'max_tokens'),
        json_tools=getattr(args, 'api_json_tools', False),
        transport_dir=Path(folder)/'a_transport', **common)


def run(args):
    root = Path(__file__).resolve().parents[2]
    folder = args.output.resolve()
    folder.mkdir(exist_ok=False)
    begin = time.perf_counter()
    robot = history = video = hub = usage = request_recovery = None
    messages = []
    report = {'status': 'started', 'object_gt_used': False,
              'completion_diagnostic': getattr(args, 'completion_diagnostic', False),
              'motion_check_mode': 'ik_only', 'observed_collision_checks_enabled': False,
              'automatic_network_retries': 0, 'automatic_collision_planning': False,
              'a_per_primitive_approval': False, 'single_agent_direct_actions': True,
              'protocol_version': PROTOCOL_VERSION}
    (folder/'execution_mode.json').write_text(json.dumps({
        'motion_check_mode': 'ik_only', 'observed_collision_checks_enabled': False,
        'full_pose_ik': True, 'joint_limits_checked': True, 'strict_fk_retained': True,
        'task_success_from_commands': False}, indent=2))
    (folder/'session.json').write_text(json.dumps({
        'protocol_version': PROTOCOL_VERSION, 'episode_id': folder.name,
        'entrypoint': 'sketchik.task_trial',
        'context': {'a': 'Full source archive with recent complete dialogues and actual current state',
                    'a_stage_context_enabled': getattr(args, 'a_context_window', False),
                    'long_term_memory': False},
        'execution': {'simulator_owner': 'calling thread', 'binding': 'current sketch batch',
                      'synchronous': not getattr(args, 'async_planning', False),
                      'async_next_sketch_planning': getattr(args, 'async_planning', False),
                      'model_calls_serial': True,
                      'max_route_points': getattr(args, 'max_route_points', 6),
                      'motion_check_mode': 'ik_only', 'per_primitive_model_call': False,
                      'failure': 'stop unexecuted suffix'},
        'records': {'interfaces': 'interfaces.json', 'context': 'context_manifest.json',
                    'events': 'events.jsonl', 'outcome': 'result.json',
                    'source_history': 'planning_transcript.json', 'wire_history': ['a_transport/']}}, indent=2))
    (folder/'interfaces.json').write_text(json.dumps({
        'protocol_version': PROTOCOL_VERSION,
        'a_scene_requirements': 'Generated from the measured whole-scene catalog; frozen in a_tools.json'}, indent=2))
    def emit(event):
        with (folder/'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'wall_elapsed_s': time.perf_counter()-begin,
                                    **public(event)}, allow_nan=False)+'\n')
    try:
        from .usage_accounting import TrialUsage
        usage = TrialUsage(folder, root/'.runtime/accounting', getattr(args, 'credits_config', None))
        from .runtime import configure
        gpu = configure(root, args.gpu_uuid, 'pro')
        from .libero import LiberoRobot
        from .primitives import MotionExecutor
        import imageio.v2 as imageio
        robot_options = {'max_steps': args.max_episode_steps}
        if getattr(args, 'completion_diagnostic', False):
            robot_options['completion_diagnostic'] = True
        robot = LiberoRobot('pro', 'libero_10', args.task, args.seed, args.init_state,
                            size=1024, **robot_options)
        metadata = robot.metadata | {'gpu': gpu}
        (folder/'metadata.json').write_text(json.dumps(metadata, indent=2))
        video = imageio.get_writer(str(folder/'video.mp4'), fps=20, codec='libx264', quality=7)
        history = ObservationHistory(robot, folder, emit, interval_s=None, video=video)
        def on_step(state):
            emit({'event': 'physics_step', **state})
            history.on_step(state)
        robot.on_step = on_step
        q = robot.state()['q']
        for _ in range(30):
            robot.step(q, -1.)
        hub = ProgramObservationHub(robot, MotionExecutor(robot, emit, settle_gripper=True), folder, emit)
        hub.ik_only = True
        hub.max_route_points = args.max_route_points
        history.attach(hub)
        snapshot(robot, folder/'initial_observation', metadata, folder.name)
        client_a = make_task_client(args, folder)
        client_a.attempt_observer = usage.measure_attempt
        from .request_recovery import RequestRecovery
        request_recovery = RequestRecovery(folder, client_a, robot.state, emit)
        client_a.request_pause_handler = request_recovery.wait_for_resume
        if client_a.stream:
            client_a.on_stream_progress = lambda status: emit({
                'event': 'api_generation_progress', 'agent': 'a',
                'request_body_sha256': client_a.last_request_metadata.get('body_sha256'), **status})
        traced = TrialClient(client_a, emit)
        class RecordingClient:
            def complete(self, messages, tools):
                with timings_for(hub).scope('api_request'):
                    response = traced.complete(messages, tools)
                (folder/f'a_response_{traced.round:03d}.json').write_text(json.dumps(response, indent=2))
                hub.journal.round(traced.round, response['message'], robot.steps)
                return response
        hashes = {}
        for source in [*sorted((root/'src/sketchik').glob('*.py')), *sorted((root/'prompts').glob('*.md'))]:
            relative = source.relative_to(root)
            destination = folder/'source_snapshot'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        (folder/'source_sha256.json').write_text(json.dumps(hashes, indent=2))
        (folder/'config.json').write_text(json.dumps({**vars(args), 'a': client_a.metadata}, default=str, indent=2))
        from .stage_context import EPISODE_INSTRUCTIONS
        system = EPISODE_INSTRUCTIONS+'\n'+(root/'prompts/task_single.md').read_text()
        system += '\n本轮由用户明确选择IK-only对照：保留完整位姿IK、关节限位、路径IK和严格FK，关闭观测表面碰撞筛查及其绕行。点云仍用于定位和持物参考，不把无碰撞拒绝或运动收敛当成成功；以实际新图复核抓持、放置及关门。\n'
        (folder/'system_prompt.md').write_text(system)
        _, pictures = hub.observe_scene()
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': pictures}]
        save_context_manifest(folder, messages, None, robot.steps)
        append_robot_physical_facts(hub, messages)
        hub.supplement_budgets = {'max_views': args.supplement_max_views,
            'max_views_per_need': args.supplement_max_views_per_need,
            'max_ik_candidates': args.supplement_max_ik_candidates}
        with timings_for(hub).scope('initial_scan_and_acquisition'):
            handoff, _, timing = table_observation(RecordingClient(), hub, messages,
                max_turns=args.max_turns, max_tool_calls=args.max_tool_calls)
        with timings_for(hub).scope('initial_geometry_handoff'):
            packet = package_handoff(hub, handoff)
        report['a_observation_timing'] = timing
        (folder/'observation_history_policy.json').write_text(json.dumps({
            'shown_conversation_retained_completely': True, 'periodic_rgbd_enabled': False,
            'periodic_frames_added_to_a_api': False, 'video_recorded': True,
            'keyframes_rgbd_retained': True,
            'a_request_image_window_enabled': getattr(client_a, 'image_window', None) is not None,
            'a_stage_context_enabled': getattr(client_a, 'stage_context', None) is not None,
            'scope': 'Initial, observation-stop and action-end/failure RGB-D are archived; motion video is separate.'}, indent=2))
        save_planning_history(folder, messages)
        emit({'event': 'a_observation_handoff_saved', 'source_step': robot.steps,
              'coverage_complete': handoff['coverage_complete']})
        from .single_agent import run_single_agent
        report.update(run_single_agent(hub, client_a, messages, packet,
            max_turns=args.max_turns, max_batches=args.max_subtasks,
            max_points=args.max_route_points, max_tool_calls=args.max_tool_calls,
            max_invalid_replies=getattr(args, 'max_invalid_replies', 4),
            async_planning=getattr(args, 'async_planning', False)))
        report['operator_resumed_requests'] = getattr(client_a, 'operator_resumed_requests', 0)
    except KeyboardInterrupt:
        report.update(status='interrupted', interruption='Operator stopped the task; no automatic restart')
        emit({'event': 'trial_interrupted', 'source_step': robot.steps if robot else None})
    except InterruptedError:
        report.update(status='interrupted', interruption='Operator aborted the paused API request')
        emit({'event': 'trial_interrupted', 'source_step': robot.steps if robot else None})
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        provider_response = getattr(exc, 'response_metadata', None)
        if provider_response:
            report['provider_response'] = provider_response
        emit({'event': 'trial_exception', 'error': report['error'],
              **({'provider_response': provider_response} if provider_response else {})})
    finally:
        artifacts = getattr(hub, 'single_a_artifacts', None)
        if artifacts is not None:
            try:
                artifacts.live_cloud_files.close()
            except Exception as exc:
                report['background_cloud_error_type'] = type(exc).__name__
                report.setdefault('error', 'Background point-cloud processing failed')
                report['status'] = 'failed'
        if request_recovery is not None:
            report.update(operator_resumed_requests=request_recovery.resumes,
                api_operator_pause_wall_s=request_recovery.pause_wall_s)
        if robot is not None:
            report.update(actions=robot.steps, official_task_success=bool(robot.state()['official_success']))
            if getattr(args, 'completion_diagnostic', False):
                first = robot.first_official_success_step
                if first is None and report['official_task_success']:
                    first = robot.steps
                report.update(official_success_first_step=first, official_success_ever=first is not None,
                    evaluation_scope='Completion diagnostic: final success is evaluated after the actual stop; '
                                     'earlier success is separate and never supplied to the planner.')
            robot.on_step = None
            snapshot(robot, folder/'final_observation', metadata, folder.name)
        if history is not None:
            history.close()
        if video is not None:
            video.close()
        if hub is not None:
            report['geometry_journal_write_wall_s'] = hub.journal.write_wall_s
            report['observation_reuse_count'] = getattr(hub, 'observation_reuse_count', 0)
            from .single_agent import single_execution_report
            report.update(single_execution_report(hub))
        save_planning_history(folder, messages)
        (folder/'a_transcript.json').write_text(json.dumps(transcript_copy(messages), indent=2))
        report['wall_s'] = time.perf_counter()-begin
        if hub is not None:
            report['timing'] = timings_for(hub).report(report['wall_s'])
            (folder/'timing.json').write_text(json.dumps(report['timing'], indent=2))
        if usage is not None:
            try:
                report['usage_accounting'] = usage.finish(report)
            except Exception as exc:
                report['usage_accounting'] = {'status': 'unavailable', 'error_type': type(exc).__name__}
        report['wall_with_accounting_s'] = time.perf_counter()-begin
        (folder/'result.json').write_text(json.dumps(report, indent=2))
        if robot is not None:
            robot.close()
        print(json.dumps({'run': str(folder), **report}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init-state', type=int, default=0)
    parser.add_argument('--max-episode-steps', type=int, default=2000)
    parser.add_argument('--max-turns', type=int, default=20)
    parser.add_argument('--max-tool-calls', type=int, default=80)
    parser.add_argument('--max-invalid-replies', type=int, default=4,
        help='Consecutive invalid tool replies before stopping (default four, allowing three corrections). '
             'No robot actions are dispatched for invalid calls; all API/tool attempts remain counted.')
    parser.add_argument('--max-subtasks', type=int, default=6,
        help='Complete sketch-batch budget.')
    parser.add_argument('--async-planning', action='store_true',
        help='Same A conditionally drafts the next subtask during current execution; '
             'all normal sketches continue directly; actual failures return to A and terminal proposals receive actual feedback. '
             'Draft and review requests count toward existing budgets.')
    parser.add_argument('--completion-diagnostic', action='store_true',
        help='Fresh single-A diagnostic: continue past official success to verify actual release/withdrawal; '
             'retain physical/backend termination, record first and final evaluation separately, never show GT to A.')
    parser.add_argument('--supplement-max-views',type=int,default=4)
    parser.add_argument('--supplement-max-views-per-need',type=int,default=2)
    parser.add_argument('--supplement-max-ik-candidates',type=int,default=8)
    parser.add_argument('--max-route-points',type=int,default=6,
        help='Maximum waypoints per move in a submitted sketch.')
    parser.add_argument('--ik-only',action='store_true',default=True,
        help='Pure IK execution; observed geometry remains available without collision refusal or detours.')
    parser.add_argument('--image-jpeg-quality', type=int, choices=range(1,101), metavar='1..100',
        help='Explicitly authorized API transmission encoding only; original native history/RGB-D remain intact')
    parser.add_argument('--reference-exact-images', action='store_true',
        help='Wire-only references for exactly repeated image parts; preserve full source history and every unique image')
    parser.add_argument('--a-image-window', action='store_true',
        help='Explicit A request image window: 24/18 unique images and 8 MB encoded image URLs. '
             'Preserve source history, initial/current images, text and tool reasoning; allow archived-image retrieval.')
    parser.add_argument('--a-context-window', action=argparse.BooleanOptionalAction, default=True,
        help='Default A working context: checkpoint after 120,000 text characters or '
             '80 messages. Single retains up to 40 recent messages/60,000 text characters with complete '
             'tool batches and current evidence. Preserve the source archive. '
             'Use --no-a-context-window for the legacy full-text comparison.')
    parser.add_argument('--api-timeout-s',type=float,default=180.,
        help='Explicit per-request timeout; changes waiting only, never retries.')
    parser.add_argument('--api-max-tokens',type=int,
        help='Explicit API output budget including reasoning when supported by the provider. '
             'Default leaves the provider setting unchanged; truncated replies are rejected without actions/retries.')
    parser.add_argument('--api-stream',action='store_true',
        help='Use SSE with usage/progress records; assemble a complete reply before accepting tools. No retries.')
    parser.add_argument('--api-budget-field',choices=['max_tokens','max_completion_tokens'],default='max_tokens',
        help='Explicit compatible API field for the output budget; changes transport only.')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--api-json-tools', action='store_true',
        help='Explicit JSON-text tool envelope for APIs without compatible native tool history; original engine validation remains.')
    parser.add_argument('--credits-config', type=Path,
        help='Optional private local Flowith gateway config for before/after account credits. '
             'Operator-only usage records; no retries, account identities or credentials in public files.')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_route_points < 1:
        parser.error('--max-route-points must be positive')
    if args.env_file is None:
        parser.error('--env-file is required')
    if not np.isfinite(args.api_timeout_s) or args.api_timeout_s <= 0:
        parser.error('--api-timeout-s must be finite and positive')
    if args.api_max_tokens is not None and args.api_max_tokens < 1:
        parser.error('--api-max-tokens must be positive')
    if min(args.max_episode_steps-30, args.max_turns, args.max_tool_calls, args.max_subtasks, args.max_invalid_replies,
           args.supplement_max_views, args.supplement_max_views_per_need, args.supplement_max_ik_candidates) <= 0:
        parser.error('Invalid episode/API budget')
    run(args)


if __name__ == '__main__':
    main()
