"""One persistent agent with public files and direct, parameterized IK actions.

The existing numerical compiler/controller and contact checks are shared. There
is no semantic submission followed by another model playing a grounding role.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np

from jsonschema import Draft202012Validator, ValidationError

from .tools import obj, public, PARAMS
from .pose_binding import BINDING
from .local_route import route_schema, expand_stage
from .public_files import PublicFiles
from .timing import timed, timings_for
from .task_execution import (NumericalCompiler, BindingUnavailable, execute_sketch,
    hand_observation_parts, previous_stage_outcome, symbolic_schema_feedback)


TEXT = {'type': 'string', 'minLength': 1}


def native_camera_frame(camera):
    """Display known calibrated axes; never define a task object's front/back."""
    if 'world_from_camera' not in camera:
        return None
    transform = np.asarray(camera['world_from_camera'], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Native camera display frame must be a finite rigid calibration')
    return {'camera_position_world_m': transform[:3, 3].round(6).tolist(),
        'screen_right_world_unit': transform[:3, 0].round(6).tolist(),
        'screen_down_world_unit': transform[:3, 1].round(6).tolist(),
        'viewing_forward_world_unit': transform[:3, 2].round(6).tolist(),
        'scope': 'Known axes of this native camera at its recorded source_step, rounded for display only. '
            'Not object axes or a definition of task front/back. Exact calibration/point conversion stays in the program.'}


def projection_frame(view):
    """Known virtual renderer directions, not an estimated object frame."""
    eye = np.asarray(view.get('virtual_camera_eye_direction', []), dtype=float)
    if eye.shape != (3,) or not np.isfinite(eye).all() or np.linalg.norm(eye) < 1e-8:
        return {}
    forward = -eye/np.linalg.norm(eye)
    right = np.cross(forward, [0.,0.,1.])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.,0.,0.])
    right /= np.linalg.norm(right)
    return {'projection_frame': {'kind':'orthographic_virtual_view',
        'toward_viewer_world_unit':(-forward).tolist(), 'screen_right_world_unit':right.tolist(),
        'screen_down_world_unit':np.cross(forward,right).tolist(),
        'scope':'Known rendering directions, not the physical camera pose, object axes or a surface normal.'}}


def working_cloud_record(record, *, catalogue=False):
    """One point-source card; full immutable metadata remains publicly readable."""
    if not record.get('view_id'):
        return deepcopy(record)  # Native camera timing/axes and mixed hand layers stay intact.
    names = ('kind', 'view_id', 'object_id', 'path', 'image_path', 'mode',
        'width', 'height', 'group_ids', 'selectable', 'same_pixel_map_as',
        'point_time_status', 'point_cloud_path', 'point_index_path', 'point_groups_path',
        'historical_evidence_only', 'retired_reason', 'coordinate_estimate')
    names += ('point_footprint_pixels', 'point_footprint_half_width_m', 'pixels_per_metre')
    names += ('view_kind', 'snapshot_id', 'source_step', 'selection_scope',
        'carry_reference_eligible', 'slot_target_hint', 'coverage', 'framing_changed',
        'display_bounds_world_m', 'measurement_kind', 'requested_path', 'latest_entry')
    names += ('logical_image_slot', 'mixed_evidence_layers')
    result = {key: deepcopy(record[key]) for key in names if key in record}
    if not catalogue and 'projection_frame' in record:
        result['projection_frame'] = deepcopy(record['projection_frame'])
    if 'point_time_status' in result:
        result['point_time_status'] = {key: result['point_time_status'][key]
            for key in ('owner', 'invalidated_at_step', 'coordinate_estimated_at_step',
                'source_step', 'current_snapshot', 'coordinate_policy', 'reference_id',
                'relation_revision', 'pose_id', 'image_axes_are_historical',
                'height_colours_are_historical', 'shape_pose_rotation_world',
                'scope') if key in result['point_time_status']}
    sources = record.get('source_observations')
    path = result.get('image_path', result.get('path'))
    if sources is not None and path:
        result['source_observation_count'] = len(sources)
        result['source_metadata_file'] = str(Path(path).with_suffix('.json'))
    return result


def reference_working_card(packet, carried_local=None):
    """Expose existing compiler references, not a new geometry estimator/planner."""
    from scipy.spatial.transform import Rotation
    from .pose_binding import compile_binding
    from .cloud_extents import principal_extents
    origin = np.asarray(packet['eef']['position'], dtype=float)
    references = {}
    names = ['eef']
    if len((packet.get('robot_gripper_geometry') or {}).get('pad_centres_in_eef_tool_m', [])) == 2:
        names += ['pad_0', 'pad_1', 'pad_midpoint']
    held = packet.get('carried_observed_geometry') or {}
    has_held = bool(held.get('reference_available') and carried_local is not None and len(carried_local) >= 25)
    if has_held:
        names += ['held_center', 'held_bottom']
    for name in names:
        _, audit = compile_binding({'anchor': {'kind': 'current_eef'}, 'reference': name,
            'offset_m': [0., 0., 0.], 'offset_frame': 'world', 'orientation': 'preserve'},
            packet, lambda *args: None, carried_local=carried_local if has_held else None)
        offset = np.asarray(audit['reference_offset_world_m'])
        references[name] = {'offset_from_eef_world_m': offset.tolist(),
            'position_world_m': (origin+offset).tolist()}
    card = {'source_step': packet['source_step'], 'references_at_current_orientation': references,
        'held_reference_available': has_held, 'held_object_id': held.get('object_id') if has_held else None,
        'meaning': 'Offsets include X, Y and Z. These values describe the CURRENT orientation only. '
            'For a move the existing compiler recomputes the chosen reference at the TARGET orientation: '
            'EEF = anchor + offset - reference_offset. Do not add this compensation again. '
            'EEF is not the held centre or bottom. Held references use associated observed points, not verified attachment or complete shape.'}
    if has_held:
        local = np.asarray(carried_local, dtype=float)
        rotation = Rotation.from_quat(packet['eef']['quaternion_xyzw']).as_matrix()
        projection = principal_extents(local)
        axes = np.asarray(projection['observed_principal_axes_world'])
        card['held_observed_principal_projections'] = [
            {'direction_eef_tool_unit': axis.tolist(), 'direction_current_world_unit': (rotation@axis).tolist(),
             'projected_span_m': span, 'variance_gap_ratio': gap}
            for axis, span, gap in zip(axes, projection['observed_principal_spans_m'],
                projection['principal_axis_gap_ratios'])]
        card['held_projection_scope'] = ('Same compiler point set; axes ordered by sample variance, signs arbitrary, '
            'small variance gaps imply unstable directions. Not semantic axes or chosen grasp directions.')
    return card


def working_current(current):
    """A model-facing view only; current.json and the executor retain full evidence."""
    from .robot_geometry import pad_reference_geometry
    result = deepcopy(current)
    geometry = result.get('geometry', {})
    mechanical = pad_reference_geometry(geometry.get('robot_gripper_geometry'))
    if mechanical is not None:
        geometry.pop('robot_gripper_geometry')
        geometry['robot_hand_reference'] = mechanical
    sources = result.get('available_sources', {})
    sources['cloud_views'] = [working_cloud_record(view, catalogue=True) for view in sources.get('cloud_views', [])]
    def pair_projections(value):
        if isinstance(value, dict):
            axes, spans = value.get('observed_principal_axes_world'), value.get('observed_principal_spans_m')
            if axes is not None and spans is not None and len(axes) == len(spans):
                value['observed_principal_projections'] = [{'direction_world_unit': axis,
                    'projected_span_m': span} for axis, span in zip(axes, spans)]
                del value['observed_principal_axes_world'], value['observed_principal_spans_m']
            for item in value.values():
                pair_projections(item)
        elif isinstance(value, list):
            for item in value:
                pair_projections(item)
    pair_projections(result.get('dimensions', {}))
    if 'recent_execution_receipts' in result:
        count = len(result.get('execution_ledger', []))
        receipts = result['recent_execution_receipts']
        result['recent_execution_receipts'] = [
            {**previous_stage_outcome(receipt),
             **{key: deepcopy(receipt[key]) for key in ('execution_progress', 'failed_location') if key in receipt},
             'receipt_path': f'receipts/batch_{count-len(receipts)+index+1:03d}.json',
             'steps': [{'primitive': step['primitive'], 'completed': step.get('completed', False),
                 **{key: deepcopy(step[key]) for key in ('expanded_step_index', 'location', 'intent') if key in step},
                 'parameters': step.get('parameters'),
                 **{key: step.get('result', {}).get(key) for key in ('converged', 'position_error_m', 'rotation_error_deg')
                    if key in step.get('result', {})}}
                 for step in receipt.get('steps', [])]}
            for index, receipt in enumerate(receipts)]
    # Put the actual outcome before the large static geometry/source catalogue.
    front = ('original_task', 'task_referents', 'source_step', 'observation_id', 'robot_state', 'recent_execution_receipts',
             'failed_attempts_this_episode', 'known_robot_motion_geometry',
             'contact_warning', 'reference_geometry')
    result = {**{key: result[key] for key in front if key in result},
              **{key: value for key, value in result.items() if key not in front}}
    result['working_view_scope'] = ('Current evidence only. Full robot geometry/provenance in current.json '
        'and robot_geometry.json; full receipts at receipt_path. Principal projections are observed point spans, not a solid box or '
        'a reconstructed centre. Use exact pixels; program computes reference-to-EEF compensation.')
    return result


def tools_for(objects, max_points=6):
    identity = {'enum': sorted(objects)}
    binding = deepcopy(BINDING)
    fields = binding['properties']
    fields['anchor']['properties']['kind']['enum'].remove('opening')
    fields['anchor']['properties'].pop('opening_index')
    fields['offset_frame']['enum'].remove('opening')
    fields['anchor']['description'] = ('Observed surface point, resolved exactly from the saved pixel table at execution; '
        'do not estimate its XYZ. current_eef uses the actual EEF at each waypoint entry; '
        'ordinary relative EEF translations pair current_eef with reference=eef.')
    fields['reference']['description'] = ('Point placed at anchor+offset: eef = grip_site origin; '
        'pad_midpoint = midpoint between the two calibrated finger-pad centres; pad_0/pad_1 = one pad centre. '
        'Program handles the entire reference-to-EEF compensation at the requested orientation. '
        'Do not add that compensation again to offset_m. '+fields['reference']['description'])
    fields['offset_m'] = {**fields['offset_m'], 'description':
        'Desired displacement in metres from the observed anchor to the chosen reference point, not to EEF unless reference=eef.'}
    fields['offset_frame']['description'] = 'world uses world XYZ; tool uses the TARGET tool axes, not the current axes.'
    fields['orientation']['description'] = ('finger_direction = target tool +Z in world, closing_axis = target tool +X in world. '
        'Program computes the full quaternion. preserve retains all current axes.')
    referents = {'type': 'array', 'maxItems': 8, 'description':
        'Optional updates of task-part interpretations, keyed by object_id and a verbatim original-task phrase. '
        'Retained independently of route notes. Omit unchanged entries; do not solve unrelated future parts now.',
        'items': obj({'object_id': identity, 'task_phrase': {**TEXT, 'maxLength': 240},
            'status': {'enum': ['selected', 'unresolved']},
            'interpretation': {**TEXT, 'maxLength': 600},
            'image_paths': {'type': 'array', 'items': TEXT, 'minItems': 1, 'maxItems': 4},
            'revision_basis': {**TEXT, 'maxLength': 600}},
            ['object_id', 'task_phrase', 'status'])}
    referents['items']['allOf'] = [{'if': {'properties': {'status': {'const': 'selected'}}},
        'then': {'required': ['interpretation', 'image_paths']}}]
    actions = []
    for primitive in ('move', 'rotate_roll', 'rotate_pitch', 'rotate_yaw', 'set_gripper', 'wait'):
        selection = route_schema(binding, max_points) if primitive == 'move' else PARAMS[primitive]
        properties = {'primitive': {'const': primitive}, 'target_id': identity,
            'intent': TEXT, 'selection': deepcopy(selection)}
        if primitive == 'set_gripper':
            properties['target_id'] = {'anyOf': [identity, {'type': 'null'}],
                'description': 'Object whose grasp/contact is intentionally changed. Use null when '
                    'only changing the gripper configuration without a new object target. This does '
                    'not certify an empty hand; any existing carried reference is still retired.'}
        required = list(properties)
        if primitive == 'move' or primitive.startswith('rotate_'):
            properties['reference_target_id'] = {'enum': [*sorted(objects), 'stage_support']}
            properties['reference_target_id']['description'] = (
                'Owner of the observed anchor. May be omitted only when all move points are cloud_point '
                'views of one known object; the program then uses that registered owner. '
                'Native pixels, relative EEF moves and rotations require an explicit reference.')
            if primitive != 'move':
                required.append('reference_target_id')
        actions.append(obj(properties, required))
    batch = obj({'source_step': {'type': 'integer', 'minimum': 0,
        'description': 'Optional echo of the physical snapshot at request send. Omit to use that exact snapshot; a conflicting value is rejected.'},
        'intent': {**TEXT, 'description': 'Optional batch-purpose note. Action intents remain required; omission does not invent a model statement.'},
        'visible_evidence': TEXT,
        'task_context_update': {'type': 'string', 'minLength': 1, 'maxLength': 2400,
            'description': 'Optional short implementation/dependency note. Replaces only the previous note, never task_referents. Geometric fit and route estimates belong here, not as evidence of task-part identity.'},
        'task_referents': referents,
        'held_object_id': {'anyOf': [identity, {'type': 'null'}]},
        'held_pose_relation': {'enum': ['unchanged', 'changed', 'unknown', 'none']},
        'changed_objects': {'type': 'array', 'items': identity, 'uniqueItems': True,
            'description': 'Objects already observed to have changed position/contact. Not the objects you plan to move. Their previous world-point evidence becomes stale even when the proposed action is rejected.'},
        'actions': {'type': 'array', 'items': {'oneOf': actions}, 'minItems': 1, 'maxItems': 12,
            'description': 'Each item is a complete action: primitive, target_id, intent, selection, and reference_target_id for movement. Move points belong only inside that item’s selection.waypoints array; they are not action items or root fields.'}},
        ['visible_evidence', 'held_object_id', 'held_pose_relation', 'actions'])
    refresh_common = {'mode': {'enum': ['inspect', 'bind']}, 'object_id': identity,
        'source_step': {'type': 'integer', 'minimum': 0}}
    inspect_schema = obj({**refresh_common, 'mode': {'const': 'inspect'},
        'sources': {'type': 'array', 'minItems': 1, 'maxItems': 2,
            'description': 'Optional current already-viewed native rectangles. Omit for the complete current dual views. Rectangles are search windows, not semantic masks.',
            'items': obj({'observation_id': TEXT, 'camera': {'enum': ['agentview', 'robot0_eye_in_hand']},
                'bounds_xyxy': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 1024},
                    'minItems': 4, 'maxItems': 4}}, ['observation_id', 'camera', 'bounds_xyxy'])}}, ['mode', 'object_id'])
    bind_schema = obj({**refresh_common, 'mode': {'const': 'bind'},
        'evidence_id': {**TEXT, 'description': 'A displayed current inspection ID; initial_scene is only for correcting untouched initial G identities.'},
        'candidate_ids': {'type': 'array', 'items': TEXT, 'uniqueItems': True},
        'decision': {'enum': ['select', 'none']},
        'selection_scope': {'enum': ['whole_visible_object', 'visible_part']},
        'point_set_scope': {'enum': ['one_rigid_object', 'mixed_or_unknown'],
            'description': 'Whether ALL selected points belong to one rigid object, even for a visible part. '
                'Mixed or uncertain groups remain selectable for localization but cannot define a carried shape. '
                'Omission means mixed_or_unknown; a tight current inspect rectangle can isolate a clean subset.'},
        'rebind_held_pose': {'type':'boolean', 'description':'Optional explicit visual claim that this same object remains held but its pose relative to the hand changed. Suspends the old relation, then uses a clean one_rigid_object current patch to register the original frozen samples in this same call. Mixed or unsupported matches remain suspended. Does not certify holding or clear contact warnings. Default false.'},
        'description': TEXT, 'visible_evidence': TEXT},
        ['mode', 'object_id', 'evidence_id', 'candidate_ids', 'decision', 'visible_evidence'])
    bind_schema['allOf'] = [{'if': {'properties': {'decision': {'const': 'select'}}},
        'then': {'required': ['selection_scope', 'description'],
            'properties': {'candidate_ids': {'minItems': 1}}},
        'else': {'properties': {'candidate_ids': {'maxItems': 0}}}}]
    declarations = [
        ('execute_actions', 'Submit exactly ONE tool call in this reply, with one ordered sketch in the actions JSON array. Put all actions of this sketch in that array; do not repeat the call or mix it with other tool calls. Each move contains an ordered selection.waypoints array of complete pose objects. Execute actions and waypoints sequentially, preserve the completed prefix and stop the unexecuted suffix on failure. Return actual observations after the sketch or an early failure so the same agent can plan its next segment.', batch),
        ('list_dir', 'List only published files/folders in this episode, non-recursively. No motion.',
            obj({'path': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0}})),
        ('read_text_file', 'Read a published UTF-8 file, with explicit pagination. No arbitrary run or project access.',
            obj({'path': TEXT, 'offset': {'type': 'integer', 'minimum': 0},
                'max_chars': {'type': 'integer', 'minimum': 1, 'maximum': 40000}}, ['path'])),
        ('read_image', 'Optionally view an indexed published image, including views not yet in available_sources. A successful read registers its cloud view_id or native pixel source, subject to owner/time checks. latest resolves an immutable version; current local and _measured views provide current measured samples. Reading a rigid latest original-shape entry may attempt supported pose matching; the result states coordinate availability. Current context samples require explicit reference_target_id. No camera motion.',
            obj({'path': TEXT}, ['path'])),
        ('get_state', 'Read current robot and observed geometry facts. No physics or new camera motion.', obj({})),
        ('refresh_object', 'Optional object association capability. Submit exactly one tool call in this reply; do not mix it with other calls. inspect organizes saved current RGB-D into numbered fragments without camera motion, identity choice or whole-object registration. bind explicitly selects shown fragments or none and publishes current point views; it does not restore the old complete template. rebind_held_pose optionally attempts the same grasp\'s relative-pose recovery. initial_scene only corrects untouched initial identities. Does not certify holding or clear contact warnings.',
            {'oneOf': [inspect_schema, bind_schema]}),
        ('finish', 'Submit exactly one tool call in this reply; do not mix it with other calls. Stop with a visible-evidence assessment. blocked/failure are valid. The benchmark is evaluated separately.',
            obj({'status': {'enum': ['success', 'failure', 'blocked']}, 'summary': TEXT,
                'goal_evidence': {'type': 'array', 'minItems': 1, 'items': obj({
                    'goal': {**TEXT, 'description': 'Verbatim original_task text. At least one entry must quote the ENTIRE original_task; optional extra entries may quote its phrases. Put your interpretation separately.'},
                    'interpretation': {**TEXT, 'description': 'Why the observed object/part corresponds to that original phrase; separate from whether the physical relation was achieved.'},
                    'assessment': {'enum': ['satisfied', 'contradicted', 'unknown']},
                    'image_paths': {'type': 'array', 'items': TEXT, 'minItems': 1},
                    'visible_evidence': TEXT}, ['goal', 'assessment', 'image_paths', 'visible_evidence'])},
                'task_referents': referents},
                ['status', 'summary', 'goal_evidence']))]
    return [{'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': schema}} for name, description, schema in declarations]


class DirectExecutor(NumericalCompiler):
    """Numerical methods only: deliberately has no bind/repair/model client."""
    def __init__(self, hub, handoff, artifacts):
        from .episode_state import EpisodeState
        self.hub, self.handoff, self.artifacts = hub, handoff, artifacts
        if not hasattr(hub, 'episode_state'):
            hub.episode_state = EpisodeState(hub.run)
        self.parameter_source = 'single_agent_a'
        self.last_recorded_physical_step = hub.robot.steps

    def record_dispatch(self, stage, index, compiled):
        self.artifacts.record_dispatch(stage, index, compiled)

    def record_execution(self, receipt):
        if (receipt['source_step'] > self.last_recorded_physical_step and
                receipt['primitive'] in ('move', 'rotate_roll', 'rotate_pitch', 'rotate_yaw')):
            # Possible carried motion invalidates old world positions, not the
            # archived shape or the current contact-support hypothesis.
            reference = getattr(self.hub, 'carried_surface_reference', None) or {}
            for identity in {getattr(self.hub, 'carried_target_id', None), reference.get('object_id')}-{None}:
                self.hub.journal.mark_moved(identity, receipt['source_step'])
        self.last_recorded_physical_step = receipt['source_step']
        self.artifacts.record_execution_step(receipt)
        self.hub.episode_state.record_execution(receipt)

    def compile_step(self, stage, index, description, folder):
        step, selection = stage['plan']['steps'][index], stage['selections'][index]
        if step['primitive'] == 'move':
            try:
                self.artifacts.validate_point(selection, step['reference_target_id'], request_step=stage['source_step'])
            except (ValueError, ValidationError) as exc:
                raise BindingUnavailable(str(exc)) from exc
        result = super().compile_step(stage, index, description, folder)
        contact = getattr(self.hub, 'last_contact_binding', None)
        if (step['primitive'] == 'set_gripper' and step.get('command') == 'close'
                and contact and contact.get('target_id') == step.get('target_id')):
            contact['world_validity_before_close'] = deepcopy(
                self.hub.journal.data['objects'].get(step['target_id'], {}))
        if result.get('parameters') is not None:
            audit = result.get('audit') or {}
            estimate = audit.get('coordinate_estimate') or {}
            if step['primitive'] == 'move' and estimate.get('kind') == 'registered_rigid_point':
                binding = self.artifacts.rigid_view_bindings.get(audit['cloud_view_id'])
                if binding and binding['pose_id'] == estimate['pose_id'] and binding['owner'] == step['target_id']:
                    result['independent_rigid_template'] = deepcopy(binding)
                    (Path(folder)/'rigid_template_at_contact.json').write_text(json.dumps(binding, indent=2))
            from .step_recovery import repeats_stalled_goal
            failure = getattr(self.hub, 'single_last_motion_failure', None)
            if failure and repeats_stalled_goal(failure, result, self.hub.robot.state()):
                blocked = BindingUnavailable('This is the same stalled absolute goal; choose a supported change from the actual stopped state')
                blocked.failure = {'failure_stage': 'stalled_goal', 'executed': False, 'error': str(blocked)}
                raise blocked
        result['parameter_source'] = self.parameter_source
        return result

    def _compile_selection(self, step, selection, packet, cloud_views, clearance, description, folder, prefix):
        retained = packet.get('retained_cloud_point_bindings', [])
        if step['primitive'] == 'move' and any(value['view_id'] == selection['anchor'].get('view_id')
                and value['pixel_xy'] == selection['anchor'].get('pixel_xy') for value in retained):
            evidence = self.artifacts.validate_point(selection, step['reference_target_id'],
                request_step=packet['grounding_source_step'])
            current = evidence.get('retained_cloud_point_binding')
            held = any(value['kind'] == 'retained_grasp_point_current_eef' for value in retained
                if value['view_id'] == selection['anchor'].get('view_id')
                and value['pixel_xy'] == selection['anchor'].get('pixel_xy'))
            if current not in retained or (held and (clearance is None or clearance.carried_local is None)):
                raise BindingUnavailable('The frozen retained point no longer has the same supported grasp relation')
        return super()._compile_selection(step, selection, packet, cloud_views, clearance, description, folder, prefix)


class EpisodeArtifacts:
    def __init__(self, hub, handoff):
        self.hub, self.handoff = hub, handoff
        self.files = PublicFiles(hub.run/'agent_public')
        self.views, self.owners, self.seen_views, self.seen_rgb = {}, {}, set(), set()
        self.seen_paths = set()
        self.image_paths = {}
        self.point_data_paths = {}
        self.latest = None
        self.receipts = []
        self.hub.single_a_receipts = self.receipts
        self.ledger = {'version': 2, 'scope': 'This episode only. Original model expressions, dispatch evidence and actual results are separate; no automatic replay or task-success certification.',
            'batches': [], 'rejected_attempts': []}
        self.hub.single_a_execution_ledger = self.ledger
        self.save_execution_ledger()
        self.object_entries = {}
        self.scene_entry = None
        self.retired_views = {}
        self.binding_revisions = {}
        self.reference_reads = []
        self.geometry_inspections = {}
        self.registration_attempts = set()
        self.registration_templates = {}
        self.original_shape_views = {}
        self.rigid_view_bindings = {}
        self.pose_inspections = {}
        self.pose_attempts = set()
        from .live_cloud import LiveCloudFiles
        self.live_cloud_files = LiveCloudFiles(self)
        self.live_dispatches = []

    def execution_history_index(self):
        return [{'batch': item['batch'], 'request_turn': item['request_turn'],
            'history_path': f"receipts/sketch_{item['batch']:03d}.json",
            'receipt_path': f"receipts/batch_{item['batch']:03d}.json",
            'source_step': item['receipt']['source_step'], 'status': item['status'],
            'historical_model_intent': item['receipt'].get('subtask'),
            'commands_completed': item['receipt'].get('commands_completed'),
            'completed_primitives': [step['primitive'] for step in item['receipt'].get('steps', []) if step.get('completed')],
            'failed_step_index': item['receipt'].get('failed_step_index'),
            'failed_location': item['receipt'].get('failed_location')}
            for item in self.ledger['batches']]

    def save_execution_ledger(self, *, batch_changed=True):
        path = self.hub.run/'single_a/execution_ledger.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(public(self.ledger), ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(path)
        if batch_changed and self.ledger['batches']:
            batch = self.ledger['batches'][-1]
            self.files.publish(f"receipts/sketch_{batch['batch']:03d}.json", value=public(batch))
            self.files.publish(f"receipts/batch_{batch['batch']:03d}.json", value=public(batch['receipt']))
        self.files.publish('receipts/index.json', value={
            'scope': self.ledger['scope'], 'batches': self.execution_history_index(),
            'rejected_attempt_paths': [item['public_path'] for item in self.ledger['rejected_attempts']]})

    def publish_budget(self):
        if self.latest is not None:
            self.latest['budgets'] = single_budget(self.hub)
            self.publish_available_sources()

    def begin_execution(self, plan, stage, batch, *, submission, bound_submission, program_origin=None):
        self.hub.single_a_batches = batch
        self.ledger['batches'].append({'batch': batch, 'request_turn': self.hub.single_a_requests,
            'request_source_step': plan['source_step'], 'status': 'started',
            'model_submission': deepcopy(submission) if program_origin is None else None,
            'program_submission': deepcopy(submission) if program_origin is not None else None,
            'dispatch_origin': deepcopy(program_origin) if program_origin is not None else 'model_tool',
            'request_bound_submission': deepcopy(bound_submission),
            'expanded_steps': [{'expanded_step_index': index,
                'location': deepcopy(stage['locations'][index]),
                'primitive': step['primitive'], 'target_id': step.get('target_id'), 'intent': step.get('intent'),
                'selection_expression': deepcopy(stage['selections'][index]),
                'status': 'not_dispatched', 'dispatches': []}
                for index, step in enumerate(stage['plan']['steps'])],
            'planned_execution_steps': len(stage['plan']['steps']),
            'receipt': {'subtask': plan['subtask'], 'steps': [], 'source_step': plan['source_step'],
                'commands_completed': False, 'remaining_suffix_executed': False,
                'unrecorded_current_action_possible': True}})
        self.save_execution_ledger()

    def record_dispatch(self, stage, index, compiled):
        if not self.ledger['batches'] or self.ledger['batches'][-1]['status'] != 'started':
            return
        entry = self.ledger['batches'][-1]['expanded_steps'][index]
        step = stage['plan']['steps'][index]
        known = {item['object_id'] for item in self.handoff['objects']}
        targets = {step.get('target_id'), step.get('reference_target_id')}&known
        points = [self.hub.robot.state()['position']]
        for value in ((compiled.get('parameters') or {}).get('position'),
                (compiled.get('audit') or {}).get('anchor_world_xyz_m')):
            if value is not None and np.asarray(value).shape == (3,):
                points.append(value)
        held_id, held_bounds = self.live_cloud_files._held_bounds()
        if held_id in targets and held_bounds is not None:
            points.extend(held_bounds.tolist())
        self.live_dispatches.append({'source_step': self.hub.robot.steps,
            'targets': sorted(targets), 'framing_points': deepcopy(points),
            'batch': self.ledger['batches'][-1]['batch'], 'expanded_step_index': index,
            'physical_steps_advanced': 0})
        entry['dispatches'].append({'source_step': self.hub.robot.steps,
            'actual_entry_state': public(self.hub.robot.state()),
            'parameters': public(deepcopy(compiled['parameters'])),
            **({'compiled_motion_id': compiled['compiled_motion']['artifact_id'],
                'recorded_joint_target_rad': compiled['compiled_motion']['planned']['joint_target_final']}
                if compiled.get('compiled_motion') is not None else {})})
        entry['status'] = 'dispatch_started'
        self.save_execution_ledger()

    def execution_progress(self, batch):
        entries = batch['expanded_steps']
        prefix = []
        for entry in entries:
            if entry['status'] != 'completed':
                break
            prefix.append(deepcopy(entry['location']))
        def unresolved(entry):
            return {key: deepcopy(entry[key]) for key in ('expanded_step_index', 'location', 'primitive',
                'target_id', 'intent', 'selection_expression', 'status')}
        return {'history_path': f"receipts/sketch_{batch['batch']:03d}.json",
            'completed_prefix_locations': prefix,
            'failed_or_unknown_steps': [unresolved(entry) for entry in entries
                if entry['status'] not in ('completed', 'not_dispatched')],
            'unexecuted_suffix': [unresolved(entry) for entry in entries if entry['status'] == 'not_dispatched'],
            'scope': 'Historical expressions, not a recovery command. Undispatched relative EEF points have no resolved absolute target. Review the actual stop and plan the remaining work; never replay the completed prefix.'}

    def record_execution_step(self, receipt):
        if not self.ledger['batches'] or self.ledger['batches'][-1]['status'] != 'started':
            return
        pending = self.ledger['batches'][-1]['receipt']
        entry = self.ledger['batches'][-1]['expanded_steps'][receipt['expanded_step_index']]
        entry['actual_result'] = deepcopy(receipt)
        entry['source_step_end'] = receipt['source_step']
        entry['physical_steps_advanced'] = (receipt['source_step']-entry['dispatches'][0]['source_step']
            if entry['dispatches'] else 0)
        entry['status'] = ('completed' if receipt.get('completed') else
            'outcome_episode_ended' if receipt.get('episode_ended') else
            'attempted_failed' if entry['physical_steps_advanced'] > 0 else 'dispatched_no_physics')
        pending['steps'].append(deepcopy(receipt))
        pending['source_step'] = receipt['source_step']
        self.save_execution_ledger()

    def finish_execution(self, receipt):
        batch = self.ledger['batches'][-1]
        failed = receipt.get('failed_step_index')
        if failed is not None:
            entry = batch['expanded_steps'][failed]
            receipt.setdefault('failed_location', deepcopy(entry['location']))
            if entry['status'] == 'not_dispatched':
                entry.update(status='failed_before_dispatch', failure=deepcopy(receipt.get('binding_failure') or receipt.get('reason')))
            elif entry['status'] == 'dispatch_started':
                # A path-IK failure may happen after dispatch entry, before physics.
                advanced = self.hub.robot.steps-entry['dispatches'][0]['source_step']
                entry.update(status='outcome_unknown' if advanced else 'dispatched_no_physics',
                    source_step_end=self.hub.robot.steps, physical_steps_advanced=advanced)
        receipt['execution_progress'] = self.execution_progress(batch)
        batch.update(status='finished', receipt=deepcopy(receipt))
        for record in self.live_dispatches:
            if record.get('batch') == batch['batch']:
                entry = batch['expanded_steps'][record['expanded_step_index']]
                record['physical_steps_advanced'] = entry.get('physical_steps_advanced', 0)
        self.live_dispatches = [record for record in self.live_dispatches
            if record.get('batch') != batch['batch'] or record.get('physical_steps_advanced', 0) > 0]
        self.receipts.append(receipt)
        self.save_execution_ledger()

    def interrupt_execution(self, exc):
        pending = self.ledger['batches'][-1]
        receipt = pending['receipt']
        receipt.update(source_step=self.hub.robot.steps, actual_stop=public(self.hub.robot.state()),
            execution_interrupted=True, error=f'{type(exc).__name__}: {exc}')
        for entry in pending['expanded_steps']:
            if entry['status'] == 'dispatch_started':
                entry.update(status='outcome_unknown', source_step_end=self.hub.robot.steps,
                    actual_stop=public(self.hub.robot.state()))
        last_recorded = max([pending['request_source_step']]+[step['source_step'] for step in receipt['steps']])
        if (self.hub.robot.steps > last_recorded and not any(
                entry['status'] == 'outcome_unknown' for entry in pending['expanded_steps'])):
            receipt['unattributed_physical_advance'] = True
            for entry in pending['expanded_steps']:
                if entry['status'] == 'not_dispatched':
                    entry['status'] = 'dispatch_unverified'
        receipt['unrecorded_current_action_possible'] = (self.hub.robot.steps > last_recorded or
            any(entry['status'] == 'outcome_unknown' for entry in pending['expanded_steps']))
        receipt['execution_progress'] = self.execution_progress(pending)
        pending['status'] = 'interrupted'
        self.receipts.append(deepcopy(receipt))
        self.save_execution_ledger()

    def record_rejected_attempt(self, *, request_turn, request_source_step, name, arguments, result, assessment_committed):
        path = f"receipts/rejected_{len(self.ledger['rejected_attempts'])+1:03d}.json"
        item = {'public_path': path, 'request_turn': request_turn, 'tool_name': name,
            'request_source_step': request_source_step, 'actual_source_step': self.hub.robot.steps,
            'model_arguments': deepcopy(arguments), 'result': deepcopy(result),
            'robot_actions': 0, 'batch_accepted': False,
            'assessment_committed': bool(assessment_committed),
            'scope': 'Rejected reply/tool attempt, not an accepted sketch. A committed assessment can change reference validity without moving the robot.'}
        self.ledger['rejected_attempts'].append(item)
        self.files.publish(path, value=public(item))
        self.save_execution_ledger(batch_changed=False)

    def publish_object(self, item, index, description, *, revision=0):
        from .grounding_packet import build_focused_context
        from .cloud_selection import height_auxiliary, write_views
        target = item['object_id']
        suffix = f'/binding_{revision:03d}' if revision else ''
        relative = f'geometry/object_{index:02d}'+suffix
        folder = self.hub.run/'single_a_geometry'/f'object_{index:02d}'
        if revision:
            folder = folder/f'binding_{revision:03d}'
        options = {'view_prefix': target+f'_binding_{revision:03d}'} if revision else {}
        context = build_focused_context(self.hub, self.handoff, target, description,
            folder, active_views_only=True, whole_object=True, **options)
        views = list(context['selectable_views'])
        # Add a gravity projection of the SAME observed points. It supplies a
        # simple horizontal layout without choosing any semantic task part.
        # The original three obliques and their maps remain unchanged/readable.
        if len(views) >= 3:
            source = views[0]
            groups = None
            if source.get('point_group_ids_file'):
                with np.load(source['point_group_ids_file']) as data:
                    groups = data['point_group_ids'].copy()
            views += write_views(source['cloud_file'], source.get('sources', []), folder,
                source['view_id']+'_gravity', (0, 0, 1), role=source['role'],
                selectable=source['selectable'], group_ids=groups, directions=[('top', (0, 0, 1))])
        entries = []
        for view_index, view in enumerate(views):
            path = self.publish_view(view, target, relative)
            height = height_auxiliary(view, folder/'height')
            height_path = self.publish_view(height, target, relative)
            entry = {'rgb': path, 'height': height_path, 'view_id': view['view_id']}
            if view_index >= len(context['selectable_views']):
                entry['projection_role'] = 'gravity_top'
            entries.append(entry)
        self.object_entries[target] = {**deepcopy(item), 'binding_revision': revision, 'views': entries,
            'geometry_available': context['local_context_available'],
            'observed_extent_groups': context.get('observed_extent_groups'),
            'horizontal_layers': context.get('horizontal_layers'),
            'scope': 'Whole observed object and adjacent structure; rectangles/groups are not semantic masks or complete shape.'}
        self.original_shape_views[target] = [self.files.entries[path]['view_id']
            for entry in entries for path in (entry['rgb'], entry['height'])]

    def default_object_image_paths(self):
        entries = [self.object_entries.get(item['object_id'], {}).get('views', [])
            for item in self.handoff['objects']]
        # Same rule for every object: source-facing focus, one fixed side, top.
        # A side-opening object still has two obliques; the other side is indexed.
        ordered = []
        for views in entries:
            tops = [view for view in views if view.get('projection_role') == 'gravity_top']
            sides = [view for view in views if view.get('projection_role') != 'gravity_top']
            ordered.append([sides[0], sides[-1], tops[0]] if len(sides) >= 3 and tops else views)
        entries = ordered
        # Give every object its first view before adding second/third angles.
        # Reserve the existing 24-image window for six explicit reads, initial
        # dual RGB, supplemental wrists, scene, current dual/hand and one comparison pair.
        slots = max(len(entries), 10-len(set(getattr(self.hub, 'initial_supplement_ids', []))))
        return [views[index]['rgb'] for index in range(3) for views in entries
            if len(views) > index][:slots]

    def reference_image_paths(self):
        paths = (self.scene_entry or {}).get('image_paths', [])[:1]
        paths += self.default_object_image_paths()
        paths += [path for path in self.reference_reads
            if self.files.entries[path].get('view_id') not in self.retired_views and path not in paths]
        return paths

    def object_image_index(self):
        """List all target image paths before bulky measurements, without viewing them."""
        return [{'object_id':item['object_id'],
            'binding_revision':self.binding_revisions.get(item['object_id'],0),
            'views':[{**deepcopy(view), **projection_frame(self.views.get(view['view_id'],{}))}
                for view in self.object_entries.get(item['object_id'],{}).get('views',[])]}
            for item in self.handoff['objects']]

    def point_time_status(self, owner):
        invalidated = self.hub.journal.data['objects'].get(owner, {}).get('invalidated_at_step')
        return {'owner': owner, 'invalidated_at_step': invalidated,
            'rule': ('No recorded owner-motion invalidation; earlier capture time alone does not invalidate an anchor.'
                if invalidated is None else 'Selected point source_step must be at least invalidated_at_step.'),
            'scope': 'Recorded time/owner checks only, not semantic identity or unobserved-motion certification.'}

    def publish_scene(self):
        from .scene_catalog import public_catalog
        catalog = getattr(self.hub, 'scene_catalog', None)
        if catalog and self.scene_entry is not None:
            self.scene_entry['catalog'] = public_catalog(catalog,
                objects=self.handoff['objects'], journal=self.hub.journal.data)
        self.files.publish('scene.json', value={'original_task': self.handoff['task'],
            'object_image_index':self.object_image_index(),
            'objects': [self.object_entries[item['object_id']] for item in self.handoff['objects']],
            'whole_scene': self.scene_entry,
            'observation_coverage': {'assessment': self.handoff['whole_task_coverage'],
                'status': 'Earlier A observation claim; verify against actual evidence before acting'},
            'geometry': 'Observed surfaces only. Initial scans become historical after object motion.'})

    def validate_initial_binding(self, answer):
        if answer['source_step'] != self.hub.robot.steps:
            raise ValueError('source_step is stale; read current.json')
        catalog = getattr(self.hub, 'scene_catalog', None)
        if catalog is None:
            raise ValueError('No measured scene G catalogue is available for identity correction')
        target = answer['object_id']
        if target not in {item['object_id'] for item in self.handoff['objects']}:
            raise ValueError('Only an existing task object may be rebound')
        if any(key not in catalog['components'] for key in answer['component_ids']):
            raise ValueError('component_ids must name existing scene G groups from scene.json')
        components = set(answer['component_ids'])
        for item in self.handoff['objects']:
            if item['object_id'] != target and not components.intersection(item.get('component_ids', [])):
                continue
            record = self.hub.journal.data['objects'].get(item['object_id'], {})
            if record.get('invalidated_at_step') is not None or getattr(self.hub, 'carried_target_id', None) == item['object_id']:
                raise ValueError('Historical G binding cannot restore a moved/contacted object or shared group; use current evidence')
        if not any(path.startswith('scene/') for path in self.seen_paths):
            raise ValueError('Read the whole-scene G image before correcting a scene binding')

    def bind_initial_groups(self, answer):
        """A corrects identity; the program only rebuilds existing measured evidence."""
        from .scene_catalog import bind_requirements
        target = answer['object_id']
        self.registration_templates.pop(target, None)
        self.rigid_view_bindings = {key: value for key, value in self.rigid_view_bindings.items()
            if value['owner'] != target}
        self.pose_attempts = {key for key in self.pose_attempts if key[0] != target}
        index = next(i for i, item in enumerate(self.handoff['objects']) if item['object_id'] == target)
        previous = deepcopy(self.handoff['objects'][index])
        revision = self.binding_revisions.get(target, 0)+1
        folder = self.hub.run/'single_a_bindings'/f'object_{index:02d}'/f'revision_{revision:03d}'
        folder.mkdir(parents=True, exist_ok=False)
        dimension_path = self.hub.run/'dimensions.json'
        dimensions = json.loads(dimension_path.read_text()) if dimension_path.exists() else {'unit': 'm', 'objects': {}}
        (folder/'before.json').write_text(json.dumps({'object': previous,
            'dimensions': dimensions.get('objects', {}).get(target),
            'journal': self.hub.journal.data['objects'].get(target)}, indent=2))
        (folder/'request.json').write_text(json.dumps(answer, indent=2))
        requirements = {key: deepcopy(previous[key]) for key in
            ('object_id', 'interactions', 'required_surfaces', 'shape_motion') if key in previous}
        requirements.update(component_ids=answer['component_ids'], description=answer['description'], missing_surfaces=[])
        replacement = bind_requirements({'source_step': answer['source_step'], 'objects': [requirements]},
            self.hub.scene_catalog)['objects'][0]
        replacement['identity_correction'] = {'source_step': answer['source_step'],
            'previous_component_ids': previous.get('component_ids', []),
            'visible_evidence': answer['visible_evidence'], 'semantic_identity_verified': False}
        self.handoff['objects'][index] = replacement
        self.binding_revisions[target] = revision
        for key, owner in self.owners.items():
            if owner == target and key not in self.retired_views:
                self.retired_views[key] = 'Object binding superseded by A identity correction at step '+str(answer['source_step'])
                self.seen_views.discard(key)
        dimensions.setdefault('objects', {})[target] = {}
        dimension_path.write_text(json.dumps(dimensions, indent=2))
        self.hub.journal.data['objects'][target] = {'description': replacement['description'],
            'component_ids': replacement['component_ids'], 'geometry_status': 'identity_rebound_from_observed_groups',
            'semantic_identity_verified': False, 'object_motion_validity': 'unverified',
            'binding_revision': revision}
        self.hub.journal.save()
        description, _ = self.hub.observe()
        self.publish_object(replacement, index, description, revision=revision)
        self.live_cloud_files.rebind_static_shape(target, description)
        self.publish_scene()
        result = {'object_id': target, 'component_ids': replacement['component_ids'],
            'previous_component_ids': previous.get('component_ids', []), 'binding_revision': revision,
            'source_step': answer['source_step'], 'visible_evidence': answer['visible_evidence'],
            'semantic_identity_verified': False, 'robot_actions': 0, 'new_camera_observations': 0,
            'instruction': 'The corrected local views and dimensions supersede this object’s earlier binding. '
                'Old files remain historical and cannot be used as this object’s point anchors.'}
        (folder/'result.json').write_text(json.dumps(result, indent=2))
        self.hub.emit({'event': 'single_a_object_rebound', **result})
        return result

    def validate_refresh(self, answer, request_step):
        from .current_geometry import validate_rectangle, validate_selection
        from .perception import saved_frame
        validate_request_snapshot(self, answer, request_step)
        if answer['object_id'] not in {item['object_id'] for item in self.handoff['objects']}:
            raise ValueError('Refresh only an existing task object')
        if answer['mode'] == 'inspect':
            sources = answer.get('sources')
            if sources is None:
                sources = [{'observation_id': self.latest['observation_id'], 'camera': camera}
                    for camera in ('agentview', 'robot0_eye_in_hand')]
            inputs = []
            for value in sources:
                key = (value['observation_id'], value['camera'])
                if key not in self.seen_rgb:
                    raise ValueError('Read the current native image before selecting its inspection rectangle')
                frame = saved_frame(self.hub.run, *key)
                if frame['step'] != request_step:
                    raise ValueError('Inspection requires current native images from this request snapshot')
                height, width = frame['depth_m'].squeeze().shape
                bounds = value.get('bounds_xyxy', [0, 0, width, height])
                validate_rectangle(bounds, (height, width))
                inputs.append({'frame': frame, 'observation_id': key[0], 'bounds_xyxy': bounds})
            if len({value['frame']['camera'] for value in inputs}) != len(inputs):
                raise ValueError('Use at most one current rectangle per camera')
            return {'inputs': inputs}
        identity = answer['evidence_id']
        if answer.get('rebind_held_pose') is True and (identity == 'initial_scene' or answer['decision'] != 'select'):
            raise ValueError('rebind_held_pose requires selecting a current inspection, not initial_scene or NONE')
        if identity == 'initial_scene':
            if answer['decision'] == 'none':
                return {}
            if answer['selection_scope'] != 'whole_visible_object':
                raise ValueError('initial_scene only corrects whole initial identities; inspect current pixels to select a visible part')
            initial = {'source_step': request_step, 'object_id': answer['object_id'],
                'component_ids': answer['candidate_ids'], 'description': answer['description'],
                'visible_evidence': answer['visible_evidence']}
            self.validate_initial_binding(initial)
            return {'initial': initial}
        if identity not in self.geometry_inspections:
            raise ValueError('Unknown current inspection ID; inspect current native views first')
        inspection = self.geometry_inspections[identity]
        if inspection['source_step'] != request_step:
            raise ValueError('Inspection is historical; inspect the new current snapshot')
        if not any(path in self.seen_paths for path in inspection['image_paths']):
            raise ValueError('View the numbered current candidates before binding their identity')
        if answer['decision'] == 'select':
            validate_selection(inspection, answer['candidate_ids'], request_step)
        return {'inspection': inspection}

    def refresh_object(self, answer, prepared):
        """New per-episode evidence, never a grasp or object-identity oracle."""
        started = time.perf_counter()
        if answer['mode'] == 'inspect':
            from .current_geometry import inspect_snapshot
            index = len(self.geometry_inspections)+1
            identity = f'current_{index:03d}'
            folder = self.hub.run/'single_a_geometry/current_inspections'/identity
            mask = None
            if hasattr(self.hub.executor.ik, 'model'):
                from .observed_clearance import ObservedClearance
                geometry = ObservedClearance(self.hub.executor.ik)
                state = self.hub.robot.state()
                mask = lambda xyz: geometry.non_robot_mask(xyz, state, margin=.004)
            table_path = self.hub.run/'table_context/context.json'
            table = json.loads(table_path.read_text()) if table_path.exists() else {}
            inspection = inspect_snapshot(prepared['inputs'], self.hub.robot.steps, folder,
                non_robot_mask=mask, table_height=table.get('observed_table_height_m'))
            public_folder = 'current_geometry/'+identity
            image_paths = []
            for index, path in enumerate(inspection['candidate_image_files']):
                relative = public_folder+f'/candidates_{index}.png'
                self.files.publish(relative, source=path, metadata={'kind': 'current_geometry_candidates',
                    'source_step': inspection['source_step'], 'evidence_id': identity,
                    'semantic_identity_verified': False, 'selectable': False,
                    'sources': inspection['sources']})
                image_paths.append(relative)
            inspection.update(evidence_id=identity, image_paths=image_paths)
            self.geometry_inspections[identity] = inspection
            result = {key: deepcopy(inspection[key]) for key in ('evidence_id', 'source_step', 'sources',
                'image_paths', 'components', 'point_count', 'robot_filter', 'table_reference', 'scope', 'semantic_identity_verified',
                'complete_shape_known', 'old_world_points_merged', 'sam_requests', 'molmo_requests', 'wall_s')}
            for key, public_key in [('cloud_file', 'point_cloud_path'), ('point_group_ids_file', 'point_groups_path')]:
                relative = public_folder+'/data/'+Path(inspection[key]).name
                self.files.publish(relative, source=inspection[key], metadata={
                    'kind': 'current_candidate_data', 'format': 'numpy_npz',
                    'evidence_id': identity, 'source_step': inspection['source_step'], 'selectable': False})
                result[public_key] = relative
            result.update(candidate_ids=['G'+str(value['group_id']) for value in inspection['components']],
                instruction='G IDs are geometric fragments, not identities. Select the intended visible object/part, multiple fragments or none. No candidate has been chosen automatically.', robot_actions=0)
            parts = []
            for path in image_paths:
                # Inspection produces a current evidence set; it is not an
                # explicit selection of every candidate as a lasting reference.
                record, picture = self.read_image(path, remember=False)
                parts.extend([{'type': 'text', 'text': json.dumps(record)}, picture])
            result['geometry_wall_s'] = result['wall_s']
            result['wall_s'] = time.perf_counter()-started
            self.files.publish(public_folder+'/inspection.json', value=result)
            self.publish_available_sources()
            self.hub.emit({'event': 'single_a_current_geometry_inspected', **result})
            return result, parts, False
        if answer['decision'] == 'none':
            return {'object_id': answer['object_id'], 'evidence_id': answer['evidence_id'],
                'decision': 'none', 'binding_changed': False, 'robot_actions': 0}, [], False
        if 'initial' in prepared:
            return self.bind_initial_groups(prepared['initial']), [], True
        from .current_geometry import selected_geometry
        from .cloud_selection import height_auxiliary
        from .observed_dimensions import write_measurement
        target = answer['object_id']
        index = next(i for i, item in enumerate(self.handoff['objects']) if item['object_id'] == target)
        self.registration_template(target)  # Preserve original shape before replacing current position views.
        revision = self.binding_revisions.get(target, 0)+1
        folder = self.hub.run/'single_a_geometry'/f'object_{index:02d}'/f'current_binding_{revision:03d}'
        result = selected_geometry(prepared['inspection'], answer['candidate_ids'], self.hub.robot.steps,
            folder, f'object_{index:02d}_current_{revision:03d}')
        carried_pose_rebind = (self.rebind_carried_pose(target, answer, result, folder)
            if answer.get('rebind_held_pose') is True else None)
        before = deepcopy(self.object_entries[target])
        (folder/'previous_binding.json').write_text(json.dumps(before, ensure_ascii=False, indent=2))
        (folder/'binding_request.json').write_text(json.dumps(answer, ensure_ascii=False, indent=2))
        entries = []
        for view in result['selectable_views']:
            relative = f'geometry/object_{index:02d}/current_binding_{revision:03d}'
            path = self.publish_view(view, target, relative)
            height = self.publish_view(height_auxiliary(view, folder/'height'), target, relative)
            entries.append({'rgb': path, 'height': height, 'view_id': view['view_id']})
        # Never clear the actual-motion invalidation or original shape. New
        # source times pass independently; old point images remain historical.
        fresh_views = {view['view_id'] for view in result['selectable_views']}
        fresh_views.update(self.files.entries[entry['height']]['view_id'] for entry in entries)
        for key, owner in self.owners.items():
            if owner == target and key not in fresh_views:
                self.retired_views.setdefault(key, 'Superseded by an explicit current object/part binding')
                self.seen_views.discard(key)
        binding = {'evidence_id': answer['evidence_id'], 'source_step': self.hub.robot.steps,
            'candidate_ids': answer['candidate_ids'], 'selection_scope': answer['selection_scope'],
            'description': answer['description'], 'visible_evidence': answer['visible_evidence'],
            'semantic_identity_verified': False,
            'point_set_scope': answer.get('point_set_scope','mixed_or_unknown'),
            'carry_reference_eligible': answer.get('point_set_scope')=='one_rigid_object'}
        self.binding_revisions[target] = revision
        self.object_entries[target] = {**before, 'views': entries, 'binding_revision': revision,
            'geometry_available': True, 'geometry_binding': binding,
            'scope': 'Current selected visible geometry. Historical whole shape remains separate; contact is not certified.'}
        self.handoff['objects'][index]['geometry_binding'] = deepcopy(binding)
        self.hub.journal.update_object(target, {'binding_revision': revision, 'current_geometry_binding': binding})
        write_measurement(self.hub.run, target, {'current_observed_shape': {
            **result['observed_shape'], 'geometry_binding': binding}, 'current_shape_binding': binding})
        self.publish_scene()
        result = {'object_id': target, 'geometry_binding': binding, 'binding_revision': revision,
            'views': entries, 'source_step': self.hub.robot.steps,
            'semantic_identity_verified': False, 'contact_certified': False, 'robot_actions': 0,
            'wall_s': time.perf_counter()-started}
        if carried_pose_rebind is not None:
            result['carried_pose_rebind'] = carried_pose_rebind
        (folder/'binding_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        self.hub.emit({'event': 'single_a_current_geometry_bound', **result})
        return result, [], True

    def rebind_carried_pose(self, target, answer, selected, folder):
        """An explicit bind may revise the same grasp relation, never reseed it."""
        from .carry_reference import (pose_rebind_identity_error, relocalization_readiness,
            relocalize_reference, suspend)
        from .object_registration import register_object
        from .observed_dimensions import invalidate_carried_dimensions
        from .perception import saved_frame
        state = self.hub.robot.state()
        reference = getattr(self.hub, 'carried_surface_reference', None)
        epoch = self.hub.episode_state.data.get('grasp_epoch', 0)
        request = dict(object_id=target, grasp_epoch=epoch, explicit_pose_rebind=True,
            point_set_scope=answer.get('point_set_scope', 'mixed_or_unknown'), current_state=state)
        reason = pose_rebind_identity_error(reference, target, epoch, state)
        if getattr(self.hub, 'carried_target_id', None) != target:
            reason = 'current_held_identity_not_this_object'
        unavailable = lambda why: {'status':'unavailable', 'reason':why, 'robot_actions':0,
            'agent_requests':0, 'holding_certified':False}
        if reason:
            return unavailable(reason)
        current_frames = list(self.hub.observations.get(self.latest['observation_id'], {}).values())
        if not current_frames or any(frame['step']!=state['step'] for frame in current_frames):
            return unavailable('current_rgbd_snapshot_not_available')
        claim = {'source_step':int(state['step']), 'evidence_id':answer['evidence_id'],
            'candidate_ids':sorted(answer['candidate_ids']),
            'point_set_scope':answer.get('point_set_scope','mixed_or_unknown')}
        if (reference.get('validity')=='active' and reference.get('relocalized_at_step')==state['step']
                and reference.get('explicit_pose_rebind_claim')==claim):
            return {'status':'already_relocalized', 'reference_id':reference['reference_id'],
                'relation_revision':reference['relation_revision'], 'source_step':state['step'],
                'robot_actions':0, 'agent_requests':0, 'holding_certified':False}
        if reference.get('explicit_pose_rebind_claim') != claim:
            # The valid visual claim retires the old transform even when the
            # current patch is mixed or too small to establish a replacement.
            # Contact review is independent: do not call confirm_held_claim.
            suspend(reference,state['step'],'A explicitly requested current object-to-hand pose rebind',
                kind='relative_pose_changed')
            reference['explicit_pose_rebind_claim']=claim
            invalidate_carried_dimensions(self.hub.run,target,state['step'],'Explicit current pose-rebind claim')
            self.hub.journal.mark_moved(target,state['step'])
            self.hub.emit({'event':'single_a_carried_pose_rebind_claim', 'object_id':target,
                'grasp_epoch':epoch, **claim, 'contact_review_acknowledged':False})
        reason = relocalization_readiness(reference, **request)
        if reason:
            return unavailable(reason)
        # The matching template is this grasp's original sample set, not a later
        # partial view or an unrelated initial whole-object coordinate frame.
        world = reference['original_world_xyz_m']
        old = {'world_xyz_m':world, 'rgb':reference['rgb']}
        sources = reference.get('source_observations', [])
        if sources:
            with np.load(reference['source_cloud_file']) as source:
                ids = reference['point_ids']
                old['pixel_xy'] = source['pixel_xy'][ids].copy()
                key = 'source_index' if 'source_index' in source else 'source_id'
                old['source_index'] = source[key][ids].copy()
                capture_key = 'original_world_xyz_m' if 'original_world_xyz_m' in source else 'world_xyz_m'
                old['original_world_xyz_m'] = source[capture_key][ids].astype(float)
                if not np.array_equal(source['world_xyz_m'][ids].astype(float), world):
                    return unavailable('original_source_file_changed')
        with np.load(selected['cloud_file']) as source:
            count = len(source['world_xyz_m'])
        # A explicitly declared all selected fragments to belong to one rigid
        # object. G1 here names only their numerical union in this audit. It is
        # not an auto-selected scene candidate or a persistent semantic ID.
        group_file = Path(folder)/'pose_registration_selected_union.npz'
        np.savez_compressed(group_file, point_group_ids=np.ones(count, np.int32))
        inspection = {'cloud_file':selected['cloud_file'],
            'point_group_ids_file':str(group_file), 'source_step':state['step']}
        registration = register_object(old, sources, inspection, current_frames,
            lambda identity, camera: saved_frame(self.hub.run, identity, camera))
        after = self.hub.robot.state()
        if (after != state or self.hub.episode_state.data.get('grasp_epoch',0) != epoch
                or getattr(self.hub,'carried_target_id',None) != target
                or getattr(self.hub,'carried_surface_reference',None) is not reference):
            raise RuntimeError('Robot or grasp state changed during current registration; no binding committed')
        registration['selected_union_scope'] = {'G1':list(answer['candidate_ids']),
            'identity_source':'A explicit current clean patch; not semantic certification'}
        audit_file = Path(folder)/'carried_pose_registration.json'
        audit_file.write_text(json.dumps(registration, indent=2))
        public_audit = f"current_geometry/{answer['evidence_id']}/carried_pose_registration_{self.binding_revisions.get(target,0):03d}.json"
        self.files.publish(public_audit, source=audit_file)
        if registration.get('status') != 'accepted':
            return {**unavailable(registration.get('reason','no_supported_unique_registration')),
                'audit_file':public_audit, 'registration_wall_s':registration['wall_s']}
        # Save the original relation before the sole commit. A failed fit leaves
        # that relation suspended and preserves all point IDs/colors and warnings.
        np.savez(Path(folder)/'carried_relation_before.npz',
            original_local_xyz_m=reference['original_local_xyz_m'],
            original_world_xyz_m=reference['original_world_xyz_m'],
            local_xyz_m=reference['local_xyz_m'], point_ids=reference['point_ids'])
        result = relocalize_reference(reference, **request, registration=registration,
            frozen_world_xyz_m=world)
        if result['status']=='relocalized':
            np.savez(Path(folder)/'carried_relation_after.npz',
                local_xyz_m=reference['local_xyz_m'], point_ids=reference['point_ids'])
        return {**result, 'audit_file':public_audit, 'registration_wall_s':registration['wall_s']}

    def prepare_failure_geometry(self, stage, receipt):
        """Refresh changed geometry, without equating motor tracking with object movement."""
        index = receipt.get('failed_step_index')
        if (index is None or self.hub.robot.steps <= stage['source_step'] or
                index >= len(stage['plan']['steps'])):
            return None
        target = stage['plan']['steps'][index]['target_id']
        warning = self.hub.episode_state.data.get('contact_warning') or {}
        reference=getattr(self.hub,'carried_surface_reference',None) or {}
        if (warning.get('object_id') in self.object_entries
                and warning.get('source_step',-1)>stage['source_step']):
            target=warning['object_id'];reason='new_contact_change_in_failed_segment'
        elif (reference.get('object_id') in self.object_entries
                and reference.get('validity')=='suspended'
                and reference.get('suspended_at_step',-1)>stage['source_step']):
            target=reference['object_id'];reason='new_carried_reference_contradiction'
        else:
            self.hub.emit({'event':'single_a_failure_geometry_deferred',
                'source_step':self.hub.robot.steps,'object_id':target,
                'reason':'Tracking failure and retiring old world anchors do not establish a new '
                    'object-to-hand change. Current images and refresh_object remain available.',
                'robot_actions':0,'extra_a_requests':0})
            return None
        return self.prepare_geometry_recovery(target, reason)

    def failure_attempts(self, limit=12):
        """Mechanical index of this episode's failed attempts, not persistent experience."""
        rows = []
        for batch in self.ledger['batches']:
            for entry in batch['expanded_steps']:
                if entry['status'] in ('completed', 'not_dispatched', 'dispatch_started'):
                    continue
                actual = entry.get('actual_result') or {}
                result = actual.get('result') or {}
                goal = result.get('continuation_goal') or (entry.get('dispatches') or [{}])[-1].get('parameters')
                tracking = result.get('tracking') or {}
                rows.append({'batch': batch['batch'], 'location': deepcopy(entry['location']),
                    'status': entry['status'], 'target_id': entry.get('target_id'), 'intent': entry.get('intent'),
                    'source_step_end': entry.get('source_step_end'),
                    'original_expression': deepcopy(entry.get('selection_expression')),
                    'recorded_absolute_goal': deepcopy(goal),
                    'actual_stop': {key: deepcopy((result.get('state') or {})[key]) for key in
                        ('step', 'position', 'quaternion_xyzw') if key in (result.get('state') or {})},
                    'error': {key: deepcopy(result[key]) for key in
                        ('position_error_m', 'rotation_error_deg', 'physical_cause') if key in result},
                    'tracking': {key: deepcopy(tracking[key]) for key in
                        ('max_goal_joint_error_rad', 'max_error_joint_name', 'last_command_consecutive_steps',
                            'recent_measured_window') if key in tracking},
                    'history_path': f"receipts/sketch_{batch['batch']:03d}.json",
                    'receipt_path': f"receipts/batch_{batch['batch']:03d}.json"})
        return {'attempts': rows[-limit:], 'total_failed_attempts': len(rows),
            'omitted_older_attempts': max(0, len(rows)-limit), 'complete_history': 'receipts/index.json',
            'scope': 'This episode only, mechanically extracted from actual dispatch/results. '
                'Old goals and object/contact states are historical; no diagnosis, retry command or current reachability certification.'}

    def prepare_geometry_recovery(self, target, reason):
        """Prepare current fragments from saved dual depth, without an A call."""
        attempt = (target, self.hub.robot.steps)
        if attempt in self.registration_attempts:
            return None
        self.registration_attempts.add(attempt)
        started = time.perf_counter()
        for camera in ('agentview', 'robot0_eye_in_hand'):
            # These exact current images are included in the following actual
            # feedback; registering them does not acquire or move a camera.
            self.read_image('observations/'+self.latest['observation_id']+'/'+camera+'.png', remember=False)
        answer = {'mode': 'inspect', 'object_id': target}
        result, _, _ = self.refresh_object(answer, self.validate_refresh(answer, self.hub.robot.steps))
        registration = {'status': 'deferred', 'reason':
            'Rigid world-pose matching is performed on a requested latest shape read; '
            'articulated/unknown objects use current measurements. No automatic ICP here.'}
        result['old_cloud_relocalization'] = registration
        self.latest.setdefault('object_relocalizations', {})[target] = registration
        self.hub.emit({'event': 'single_a_failure_geometry_prepared', 'source_step': self.hub.robot.steps,
            'evidence_id': result['evidence_id'], 'wall_s': time.perf_counter()-started,
            'geometry_and_publication_wall_s': result['wall_s'],
            'trigger': reason, 'old_cloud_relocalization': registration,
            'sam_requests': 0, 'molmo_requests': 0, 'robot_actions': 0, 'extra_a_requests': 0})
        self.publish_available_sources()
        return result

    def registration_template(self, target):
        """Keep original named shape distinct from later local position bindings."""
        from .object_registration import catalog_template, expand_named_template
        if target not in self.registration_templates:
            item=next(value for value in self.handoff['objects'] if value['object_id']==target)
            catalog=getattr(self.hub,'scene_catalog',None)
            template=catalog_template(catalog,item.get('component_ids',[])) if catalog else None
            original_views=[self.views[key] for key in self.original_shape_views.get(target, [])]
            if original_views and template:
                template=expand_named_template(template,original_views[0])
            self.registration_templates[target]=template
        return self.registration_templates[target]

    def relocalize_object(self, target, inspection):
        from .object_registration import register_object, save_registered_cloud
        from .perception import saved_frame
        item = next(value for value in self.handoff['objects'] if value['object_id'] == target)
        if item.get('shape_motion', 'unknown') != 'rigid':
            return {'status': 'unavailable', 'reason': 'Whole referenced shape is not declared rigid',
                'robot_actions': 0, 'agent_requests': 0}
        object_key = f'object_{self.handoff["objects"].index(item):02d}'
        catalog = getattr(self.hub, 'scene_catalog', None)
        template = self.registration_template(target)
        if template is None:
            return {'status': 'unavailable', 'reason': 'no_named_native_template', 'agent_requests': 0}
        self.registration_templates[target] = template
        old, sources = template
        frames = [saved_frame(self.hub.run, source['observation_id'], source['camera']) for source in inspection['sources']]
        table_file = self.hub.run/'table_context/context.json'
        table = json.loads(table_file.read_text()) if table_file.exists() else {}
        bounds = None
        if 'display_xy_min_m' in table and 'display_xy_max_m' in table:
            diameter = np.linalg.norm(np.ptp(old['world_xyz_m'], axis=0))
            hand = np.asarray(self.hub.robot.state()['position'])[:2]
            bounds = np.stack([np.minimum(table['display_xy_min_m'], hand)-diameter,
                np.maximum(table['display_xy_max_m'], hand)+diameter])
        other_points = []
        other_bounds = []
        if catalog:
            for identity, component in catalog['components'].items():
                if identity in item.get('component_ids', []): continue
                if 'observed_bounds_world_m' in component:
                    other_bounds.append(component['observed_bounds_world_m'])
                with np.load(component['point_cloud_file']) as source:
                    xyz = source['world_xyz_m']
                    other_points.append(xyz[::max(1, len(xyz)//3000)].copy())
        registration = register_object(old, sources, inspection, frames,
            lambda identity, camera: saved_frame(self.hub.run, identity, camera), search_xy_bounds=bounds,
            other_observed_points=np.concatenate(other_points) if other_points else None,
            other_component_bounds=other_bounds)
        root = self.hub.run/'single_a_registration'/object_key
        root.mkdir(parents=True, exist_ok=True)
        index = len(list(root.glob('attempt_*.json')))+1
        audit = root/f'attempt_{index:03d}.json'
        audit.write_text(json.dumps(registration, indent=2))
        public_audit = f'geometry/registration/{object_key}/attempt_{index:03d}.json'
        self.files.publish(public_audit, source=audit)
        summary = {key: registration[key] for key in ('status', 'reason', 'source_step', 'wall_s',
            'feature_correspondences', 'current_groups_considered', 'search_complete', 'scope') if key in registration}
        summary.update(audit_file=public_audit, agent_requests=0, robot_actions=0,
            holding_certified=False, original_capture_times_preserved=True)
        if registration['status'] != 'accepted':
            self.hub.emit({'event': 'single_a_object_registration', 'object_id': target, **summary})
            return summary
        folder = root/f'accepted_{index:03d}'
        cloud = save_registered_cloud(old, sources, registration, folder)
        from .rigid_binding import save_bindings
        views = [self.views[key] for key in self.original_shape_views.get(target, [])]
        bindings = save_bindings(old, sources, registration, views, folder/'shape_pose_bindings', target)
        self.rigid_view_bindings.update(bindings)
        public_pose = f'geometry/registration/{object_key}/accepted_{index:03d}/shape_pose.json'
        self.files.publish(public_pose, source=folder/'shape_pose_bindings/bindings.json')
        summary.update(bitmap_reused=True, pose_binding_file=public_pose,
            reusable_view_ids=sorted(bindings), coordinates_available=bool(bindings))
        # Retain all registered XYZ as data, while keeping the original bitmap
        # and original pixel -> sample IDs. No reprojection or height recolour.
        self.files.publish(f'geometry/registration/{object_key}/accepted_{index:03d}/points.npz',
            source=cloud, metadata={'kind': 'registered_rigid_points', 'selectable_without_view': False})
        self.hub.journal.update_object(target, {'pose_registration': {
            'source_step': self.hub.robot.steps, 'audit_file': public_audit, 'bitmap_reused': True}})
        self.hub.emit({'event': 'single_a_object_registration', 'object_id': target, **summary})
        return summary

    def ensure_rigid_pose(self, target, snapshot):
        """On-demand world pose from saved current dual RGB-D, without plots."""
        from .carry_reference import active_reference
        from .rigid_binding import valid_binding
        active = active_reference(self.hub)
        if active and active['object_id'] == target:
            return {'status': 'held_coordinates_available', 'robot_actions': 0, 'agent_requests': 0}
        changed = self.hub.journal.data['objects'].get(target, {}).get('invalidated_at_step')
        if changed is None:
            return {'status': 'original_static_coordinates_available', 'robot_actions': 0, 'agent_requests': 0}
        if any(valid_binding(self.hub, value, target) for value in self.rigid_view_bindings.values()):
            return {'status': 'registered_coordinates_available', 'robot_actions': 0, 'agent_requests': 0}
        attempt = (target, snapshot['snapshot_id'])
        if attempt in self.pose_attempts:
            return {'status': 'coordinates_unavailable', 'reason': 'Matching already attempted on this snapshot'}
        self.pose_attempts.add(attempt)
        from .current_geometry import inspect_snapshot
        from .perception import saved_frame
        key = snapshot['snapshot_id']
        if key not in self.pose_inspections:
            inputs = []
            for source in snapshot['sources']:
                frame = saved_frame(self.hub.run, source['observation_id'], source['camera'])
                height, width = frame['depth_m'].shape
                inputs.append({'frame': frame, 'observation_id': source['observation_id'],
                    'bounds_xyxy': [0, 0, width, height]})
            mask = None
            if hasattr(self.hub.executor.ik, 'model'):
                from .observed_clearance import ObservedClearance
                geometry = ObservedClearance(self.hub.executor.ik)
                state = self.hub.robot.state()
                mask = lambda xyz: geometry.non_robot_mask(xyz, state, margin=.004)
            table_path = self.hub.run/'table_context/context.json'
            table = json.loads(table_path.read_text()) if table_path.exists() else {}
            self.pose_inspections[key] = inspect_snapshot(inputs, snapshot['source_step'],
                snapshot['folder']/'pose_inspection', non_robot_mask=mask,
                table_height=table.get('observed_table_height_m'), render_images=False)
        return self.relocalize_object(target, self.pose_inspections[key])

    def publish_point_data(self, view, owner, folder):
        """Retain exact binary geometry once; expose references, never XYZ dumps."""
        from .cloud_selection import public_view
        references = {}
        for field, public_key, kind in (
                ('cloud_file', 'point_cloud_path', 'point_cloud_data'),
                ('index_map_file', 'point_index_path', 'cloud_pixel_index'),
                ('point_group_ids_file', 'point_groups_path', 'cloud_point_groups')):
            if field not in view:
                continue
            source = Path(view[field]).resolve()
            if source not in self.point_data_paths:
                path = folder+'/data/'+source.name
                if path in self.files.entries:
                    raise ValueError('Distinct point data cannot share a public path')
                metadata = {'kind': kind, 'object_id': owner, 'format': 'numpy_npz'}
                if field == 'cloud_file':
                    metadata.update(coordinates='world_XYZ', units='m', point_count=view['point_count'],
                        source_observations=public_view(view)['source_observations'],
                        scope='All original points retained; observed surfaces only, not live object coordinates.')
                self.files.publish(path, source=source, metadata=metadata)
                self.point_data_paths[source] = path
            references[public_key] = self.point_data_paths[source]
        return references

    def publish_view(self, view, owner, folder):
        from .cloud_selection import public_view
        key = view['view_id']
        if key in self.views:
            raise ValueError('A view_id cannot be reused with a different point projection')
        point_data = self.publish_point_data(view, owner, folder)
        path = folder+'/'+key+'.png'
        self.files.publish(path, source=view['image_file'], metadata={
            'kind': 'point_cloud_projection', 'object_id': owner, **public_view(view),
            **projection_frame(view), **point_data})
        self.views[key], self.owners[key], self.image_paths[key] = deepcopy(view), owner, path
        self.files.publish(folder+'/'+key+'.json', value={**public_view(view), 'image_path': path,
            'object_id': owner, **projection_frame(view), **point_data,
            'point_lookup': 'Exact saved point table; use view_id and pixel_xy in move. No separate query needed.'})
        return path

    def read_image(self, path, *, remember=True):
        requested = self.files.canonical_name(path)
        if (requested in self.live_cloud_files.slots or
                requested == self.live_cloud_files.root+'/overview.png'):
            try:
                with timings_for(self.hub).scope('current_cloud_local_view'):
                    self.live_cloud_files.materialize(requested)
            except Exception as exc:
                raise RuntimeError('Failed to generate a registered latest cloud image') from exc
        record, picture = self.files.read_image(path)
        if requested.startswith(self.live_cloud_files.root+'/'):
            record['logical_image_slot'] = requested
        if requested in self.files.image_aliases:
            record['latest_entry'] = {key: value for key, value in self.files.entries[requested].items()
                if key in ('local_image_policy', 'shape_motion', 'pose_status', 'pose_source_step')}
        self.seen_paths.add(record['path'])
        record['public_artifact_read'] = bool(remember)
        record['display_origin'] = 'explicit_read_image' if remember else 'automatic_current_or_reference_display'
        record['capture_is_current_physical_step'] = (record.get('kind') == 'native_rgb'
            and record.get('source_step') == self.hub.robot.steps)
        current_hand = (record.get('kind') == 'hand_geometry'
            and record.get('source_step') == self.hub.robot.steps)
        current_candidates = (record.get('kind') == 'current_geometry_candidates'
            and record.get('source_step') == self.hub.robot.steps)
        current_cloud = self.current_context(record)
        record['capture_is_current_physical_step'] |= current_cloud
        record['historical_evidence_only'] = not (record['capture_is_current_physical_step'] or current_hand or current_candidates)
        if record.get('view_kind') == 'current_measurement_context' and not current_cloud:
            record.update(selectable=False, retired_reason='An older current snapshot; read the fixed latest entry')
        if record.get('kind') == 'hand_geometry':
            record['mixed_evidence_layers'] = True
            record['current_measured_layers_at_current_physical_step'] = current_hand
        if record.get('view_kind') == 'retained_scene_context':
            record.update(mixed_evidence_layers=True, selectable=False)
        if record.get('view_id') in self.owners:
            record['point_time_status'] = self.point_time_status(self.owners[record['view_id']])
        if record.get('view_id') in self.retired_views:
            record.update(selectable=False, retired_reason=self.retired_views[record['view_id']])
        elif record.get('view_id'):
            self.seen_views.add(record['view_id'])
        key = record.get('view_id')
        if key and self.retained_view_status(key):
            record.update(selectable=True, selection_scope='retained_grasp_points_only')
            record['point_time_status'].update(self.retained_view_status(key))
            record['selection_scope'] = 'independently_retained_template_points_only'
            if 'projection_frame' in record:
                record['projection_frame']['scope'] = ('Original bitmap axes refer to the capture world frame, '
                    'not the current object pose; use the separately supplied pose rotation and current RGB.')
            record.pop('retired_reason', None)
            self.seen_views.add(key)
        if remember and record.get('view_id') not in self.retired_views:
            # The bounded visual working set follows A's explicit image reads,
            # including native historical views needed to resolve structure.
            primary = set(self.default_object_image_paths())
            if (record.get('kind') in ('native_rgb', 'current_geometry_candidates') or
                    (record.get('kind') == 'point_cloud_projection' and record['path'] not in primary)):
                if record.get('view_kind') != 'current_measurement_context':
                    self.reference_reads = [old for old in self.reference_reads
                        if old != record['path'] and self.files.entries[old].get('view_id') not in self.retired_views]
                    self.reference_reads = (self.reference_reads+[record['path']])[-6:]
        if record.get('kind') == 'native_rgb' and record.get('observation_id'):
            self.seen_rgb.add((record['observation_id'], record['camera']))
        self.live_cloud_files.note_displayed(requested, record)
        if self.latest is not None:
            self.latest['current_clouds'] = self.live_cloud_files.status(self.live_cloud_files.description)
            self.publish_available_sources()
        return record, picture

    def current_context(self, view):
        snapshot = self.live_cloud_files.snapshot
        return bool(view.get('view_kind') == 'current_measurement_context' and snapshot
            and view.get('snapshot_id') == snapshot['snapshot_id']
            and view.get('source_step') == self.hub.robot.steps)

    def retained_view_status(self, key):
        from .carry_reference import retained_view_reference
        reference = retained_view_reference(self.hub, self.views[key], self.owners[key])
        common = {'image_axes_are_historical': True,
            'height_colours_are_historical': self.views[key].get('mode') == 'height',
            'scope': 'Bitmap/capture axes/height colours stay historical. Only independently retained sample IDs receive estimated current coordinates.'}
        if reference is not None:
            value = {**common, 'coordinate_estimated_at_step': self.hub.robot.steps,
                'coordinate_policy': 'retained_point_id_transformed_by_current_eef',
                'reference_id': reference['reference_id'], 'relation_revision': reference['relation_revision']}
            if reference.get('source_world_to_eef_rotation') is not None:
                from scipy.spatial.transform import Rotation
                value['shape_pose_rotation_world'] = (Rotation.from_quat(self.hub.robot.state()['quaternion_xyzw']).as_matrix()
                    @np.asarray(reference['source_world_to_eef_rotation'])).tolist()
            return value
        from .rigid_binding import valid_binding
        binding = self.rigid_view_bindings.get(key)
        if valid_binding(self.hub, binding, self.owners[key]):
            return {**common, 'coordinate_estimated_at_step': binding['source_step'],
                'coordinate_policy': 'registered_pose_on_original_point_ids', 'pose_id': binding['pose_id'],
                'shape_pose_rotation_world': binding['transform']['rotation']}
        return None

    def cloud_keys(self, reference):
        return [key for key in sorted(self.seen_views)
            if (self.views[key].get('view_kind') != 'current_measurement_context'
                and self.owners[key] == reference and (key not in self.retired_views or self.retained_view_status(key)))
                or self.current_context(self.views[key])]

    def publish_available_sources(self):
        """The catalogue and compiler registry describe the same read images."""
        from .cloud_selection import public_view
        cloud_views = []
        for key in sorted(self.seen_views):
            view = self.views[key]
            record = {**public_view(view),
                'object_id': self.owners[key], 'image_path': self.image_paths[key],
                'point_time_status': self.point_time_status(self.owners[key]),
                **{name: value for name, value in self.files.entries[self.image_paths[key]].items()
                    if name in ('point_cloud_path', 'point_index_path', 'point_groups_path', 'projection_frame')}}
            if view.get('view_kind') == 'current_measurement_context':
                current = self.current_context(view)
                record.update(historical_evidence_only=not current,
                    selectable=bool(view.get('selectable') and current),
                    point_time_status={'owner': None, 'source_step': view['source_step'],
                        'current_snapshot': current,
                        'scope': 'Only this explicitly selected sample; reference_target_id is required. '
                            'An older current snapshot is historical; read current_clouds/latest.'})
                if not current:
                    record['retired_reason'] = 'An older current snapshot; read the fixed latest entry'
            if key in self.retired_views:
                record.update(selectable=False, retired_reason=self.retired_views[key])
            retained = self.retained_view_status(key)
            if retained:
                record.update(selectable=True, selection_scope='independently_retained_template_points_only')
                record['point_time_status'].update(retained)
                record.pop('retired_reason', None)
            cloud_views.append(record)
        sources = {'cloud_views': cloud_views,
            'image_references': [{'observation_id': key, 'camera': camera,
                'image_path': 'observations/'+key+'/'+camera+'.png'}
                for key, camera in sorted(self.seen_rgb)]}
        self.latest['available_sources'] = sources
        self.latest['current_geometry_candidates'] = [
            {'evidence_id': value['evidence_id'], 'source_step': value['source_step'],
                'candidate_ids': ['G'+str(group['group_id']) for group in value['components']],
                'image_paths': value['image_paths'], 'identity_confirmed': False}
            for value in self.geometry_inspections.values() if value['source_step'] == self.hub.robot.steps]
        self.latest['geometry'].pop('operation_reference_target_id', None)
        self.latest['geometry'].pop('cloud_views', None)
        self.latest['geometry'].pop('image_references', None)
        self.files.publish('current.json', value=self.latest)
        self.files.publish('index.json', value={'allowed_root': 'agent_public',
            'files': [entry for name, entry in self.files.entries.items() if name != 'index.json'],
            'scope': 'Current episode only; no credentials, simulator object truth, evaluator or other projects.'})

    @timed('initial_object_artifacts')
    def prepare(self):
        from .scene_catalog import public_catalog
        started = time.perf_counter()
        description, _ = self.hub.observe()
        for index, item in enumerate(self.handoff['objects']):
            self.publish_object(item, index, description)
        catalog = getattr(self.hub, 'scene_catalog', None)
        scene = None
        if catalog:
            scene = public_catalog(catalog, objects=self.handoff['objects'], journal=self.hub.journal.data)
            scene_images, scene_data = [], []
            for view in catalog.get('views', []):
                # Overview is context; object-local exact maps are the selectable sources.
                path = 'scene/'+view['view_id']+'.png'
                point_data = self.publish_point_data(view, None, 'scene')
                self.files.publish(path, source=view['image_file'], metadata={'kind': 'scene_context',
                    'selectable': False, 'source_steps': catalog.get('source_steps', []), **point_data})
                scene_images.append(path)
                scene_data.append({'image_path': path, **point_data})
            scene = {'catalog': scene, 'image_paths': scene_images, 'point_data': scene_data}
        self.scene_entry = scene
        self.publish_scene()
        self.files.publish('usage.md', source=self.hub.project_root/'prompts/task_single.md')
        self.sync_observations()
        self.hub.emit({'event': 'single_a_artifacts_prepared', 'objects': len(self.object_entries),
            'cloud_views': len(self.views), 'wall_s': time.perf_counter()-started,
            'root': 'agent_public'})
        return self.reference_image_paths()

    def sync_observations(self):
        for folder in sorted((self.hub.run/'observations').glob('obs_*')):
            metadata_path = folder/'observation.json'
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text())
            for camera in ('agentview', 'robot0_eye_in_hand'):
                name = 'observations/'+folder.name+'/'+camera+'.png'
                if name not in self.files.entries:
                    frame = native_camera_frame(next((item for item in metadata.get('cameras', [])
                        if item.get('camera') == camera), {}))
                    self.files.publish(name, source=folder/(camera+'.png'), metadata={
                        'kind': 'native_rgb', 'observation_id': folder.name, 'camera': camera,
                        'source_step': metadata['source_step'], 'width': 1024, 'height': 1024,
                        **({'camera_frame': frame} if frame is not None else {})})

    def packet(self, description, reference):
        from .cloud_selection import public_view
        return {'source_step': description['source_step'], 'observation_id': description['observation_id'],
            'eef': {k: description['state'][k] for k in ('position', 'quaternion_xyzw')},
            'operation_reference_target_id': reference,
            'cloud_views': [public_view(self.views[key]) for key in self.cloud_keys(reference)],
            'image_references': [{'observation_id': key, 'camera': camera} for key, camera in sorted(self.seen_rgb)],
            'observed_opening_frames': [], 'motion_check_mode': 'ik_only' if self.hub.ik_only else 'observed_surfaces'}

    def validate_point(self, selection, reference, *, request_step=None):
        from .cloud_selection import resolve_cloud_point
        from .perception import saved_frame
        from .libero import back_project
        anchor = selection['anchor']
        if anchor['kind'] == 'current_eef':
            if reference == 'stage_support':
                raise ValueError('stage_support requires a current request native image point')
            return {'kind': 'current_eef', 'source_step': self.hub.robot.steps}
        if anchor['kind'] == 'opening':
            raise ValueError('Select an observed cloud/image point; no implicit opening candidates in this mode')
        if anchor['kind'] == 'cloud_point':
            key = anchor.get('view_id')
            retained_view = key in self.views and self.retained_view_status(key)
            if key in self.retired_views and not retained_view:
                raise ValueError('Selected point belongs to a superseded object binding; read the corrected local view')
            if key not in self.seen_views:
                raise ValueError('Read the exact point cloud image before selecting it')
            context = self.views[key].get('view_kind') == 'current_measurement_context'
            if context and not self.current_context(self.views[key]):
                # A frozen stage may use its request snapshot across static
                # destination waypoints; it may not revive a prior request view.
                snapshot = self.live_cloud_files.snapshot
                if (snapshot is None or self.views[key].get('snapshot_id') != snapshot['snapshot_id']
                        or self.views[key].get('source_step') != request_step):
                    raise ValueError('Earlier current cloud snapshot is historical; read current_clouds/latest')
            if context and reference not in {item['object_id'] for item in self.handoff['objects']}:
                raise ValueError('Current context requires an explicit known reference_target_id for this sample')
            if not context and self.owners[key] != reference:
                raise ValueError('Point cloud owner differs from reference_target_id')
            from .cloud_selection import public_view
            view = self.views[key]
            point, provenance = resolve_cloud_point(anchor, [public_view(view)], [view])
            if retained_view:
                from .carry_reference import retained_point_binding
                retained = retained_point_binding(self.hub, view, reference, point, provenance)
                if retained is not None and request_step is not None and request_step != self.hub.robot.steps:
                    raise ValueError('Retained held-point anchor predates intervening motion; use a static destination or dynamic EEF')
                if retained is None:
                    from .rigid_binding import point_binding
                    retained = point_binding(self.hub, self.rigid_view_bindings.get(key), reference, view, point, provenance)
                if retained is None:
                    raise ValueError('Retained point has no valid pose for this reference_target_id')
                return {'kind': 'cloud_point', 'owner': reference, 'source_step': retained['source_step'],
                    'retained_cloud_point_binding': retained}
            source = provenance.get('source_observation', {})
            source_step = source.get('source_step')
            if source_step is None and source.get('observation_id'):
                source_step = saved_frame(self.hub.run, source['observation_id'], source['camera'])['step']
            if source_step is None:
                raise ValueError('Cloud point has no observed-time provenance')
        else:
            key = (anchor.get('observation_id'), anchor.get('camera'))
            if key not in self.seen_rgb:
                raise ValueError('Read the native image before selecting its pixel')
            frame = saved_frame(self.hub.run, *key)
            back_project(frame, anchor['pixel_xy'])
            source_step = frame['step']
        if reference == 'stage_support':
            if anchor['kind'] != 'image_point' or source_step != (
                    self.hub.robot.steps if request_step is None else request_step):
                raise ValueError('stage_support requires a native image from the current request snapshot')
        carried = getattr(self.hub, 'carried_surface_reference', None) or {}
        if reference in {getattr(self.hub, 'carried_target_id', None), carried.get('object_id')}-{None}:
            if source_step < self.hub.robot.steps:
                raise ValueError('Held-object pixels precede the current physical state; use current evidence, a static destination or dynamic EEF')
        invalidated = self.hub.journal.data['objects'].get(reference, {}).get('invalidated_at_step')
        if invalidated is not None and source_step < invalidated:
            raise ValueError('Selected point predates actual object motion/contact change; use current native RGB, a static destination or dynamic EEF')
        return {'kind': anchor['kind'], 'owner': reference, 'source_step': source_step}

    @timed('current_state_packet')
    def current(self, executor, folder):
        from .observed_dimensions import dimension_context
        from .held_pose_review import held_pose_comparison
        description, _ = self.hub.observe()
        self.sync_observations()
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        target = self.handoff['objects'][0]['object_id']
        packet = self.packet(description, target)
        clearance = executor._refresh_runtime(packet, {'primitive': 'move', 'target_id': target}, description, folder)
        episode = executor.hub.episode_state.data
        carried_local = getattr(clearance, 'carried_local', None)
        if episode.get('contact_warning') or episode.get('carry_premise') == 'invalidated_by_contact_change':
            carried_local = None
        comparison, _, comparison_paths = held_pose_comparison(self.hub.run, description,
            self.receipts[-1] if self.receipts else None)
        comparison['image_paths'] = [str(Path(path).relative_to(self.hub.run)) for path in comparison_paths]
        facts = self.hub.run/'robot_geometry_evidence.json'
        value = {'original_task': self.handoff['task'], 'source_step': description['source_step'],
            'observation_id': description['observation_id'], 'robot_state': public(description['state']),
            'geometry': packet, 'dimensions': dimension_context(self.hub.run,
                {item['object_id'] for item in self.handoff['objects']}, current_source_step=description['source_step']),
            'reference_geometry': reference_working_card(packet, carried_local),
            'held_pose_comparison': comparison,
            'robot_geometry_file': 'robot_geometry.json',
            'held_claim': executor.hub.episode_state.data.get('a_held_claim'),
            'task_context': executor.hub.episode_state.data.get('a_task_context'),
            'task_referents': deepcopy(episode.get('a_task_referents', [])),
            'object_bindings': [{'object_id': item['object_id'], 'component_ids': item.get('component_ids', []),
                'binding_revision': self.binding_revisions.get(item['object_id'], 0),
                'description': item['description'], 'identity_correction': item.get('identity_correction'),
                'geometry_binding': item.get('geometry_binding'),
                'semantic_identity_verified': False} for item in self.handoff['objects']],
            'contact_warning': executor.hub.episode_state.data.get('contact_warning'),
            'initial_grasp_contact': deepcopy(executor.hub.episode_state.data.get('initial_grasp_contact')),
            'object_invalidations': {key: record.get('invalidated_at_step') for key, record in self.hub.journal.data['objects'].items()},
            'recent_execution_receipts': self.receipts[-2:],
            'failed_attempts_this_episode': self.failure_attempts(),
            'execution_ledger': self.execution_history_index(),
            'execution_history_file': 'receipts/index.json',
            'execution_ledger_scope': 'Intents are historical model claims, not evidence identifying task parts. '
                'Completion records commands/results only; it does not verify task meaning or fulfilment.',
            'task_success_certified': False}
        if hasattr(self.hub, 'single_a_budget_limits'):
            value['budgets'] = single_budget(self.hub)
        if facts.exists():
            self.files.publish('robot_geometry.json', source=facts)
        if (self.hub.run/'dimensions.json').exists():
            self.files.publish('dimensions.json', source=self.hub.run/'dimensions.json')
        hand_observation_parts(self.hub, description, folder/'hand')
        hand_metadata_file = folder/'hand'/'hand_observation.json'
        hand_metadata = json.loads(hand_metadata_file.read_text()) if hand_metadata_file.exists() else {}
        for number, image in enumerate(sorted((folder/'hand').glob('*.png'))):
            relative = 'hand/'+str(description['source_step'])+'_'+str(number)
            self.files.publish(relative+'.json', value=hand_metadata)
            self.files.publish('hand/'+str(description['source_step'])+'_'+str(number)+'.png', source=image,
                metadata={'kind': 'hand_geometry', 'source_step': description['source_step'],
                    'observation_id': description['observation_id'], 'selectable': False,
                    'evidence_layers': hand_metadata.get('evidence_layers', {}),
                    'metadata_file': relative+'.json'})
        goal = hand_metadata.get('failed_goal_projection')
        robot_geometry = hand_metadata.get('known_robot_geometry')
        if robot_geometry is not None:
            value['known_robot_motion_geometry'] = {key: deepcopy(robot_geometry[key]) for key in (
                'actual_source_step', 'actual_joint_positions_rad', 'recorded_goal_joint_positions_rad',
                'goal_failure_step', 'goal_status', 'finger_configuration_source_step', 'scope')}
            value['known_robot_motion_geometry']['full_geometry_file'] = 'hand/'+str(description['source_step'])+'_0.json'
        if goal is not None:
            value['recorded_goal_hand_geometry'] = {key: deepcopy(goal[key]) for key in (
                'pose', 'failure_observed_step', 'hand_configuration_source_step',
                'failure_at_current_physical_step', 'hand_world_bounds_m', 'part_world_bounds_m',
                'pad_centres_world_m', 'envelope_scope', 'new_observation', 'collision_checked',
                'contact_cause_certified') if key in goal}
            value['recorded_goal_hand_geometry']['is_current_pose'] = False
        value['hand_image_paths'] = [name for name, entry in self.files.entries.items()
            if name.startswith('hand/') and entry.get('kind') == 'hand_geometry'
                and entry.get('source_step') == description['source_step']]
        with timings_for(self.hub).scope('current_cloud_files'):
            self.live_cloud_files.update(description)
        value['current_clouds'] = self.live_cloud_files.status(description)
        recovery_candidate = getattr(self.hub, 'async_recovery_candidate', None)
        if recovery_candidate is not None:
            self.files.publish('sketch/recovery_reference.json', value={'candidate': recovery_candidate,
                'executable': False, 'scope': 'Historical intent only. Use actual stop to repair; stale coordinates stay invalid.'})
            value['async_recovery_reference'] = {'candidate_id': recovery_candidate['candidate_id'],
                'intent': recovery_candidate.get('intent', recovery_candidate['reason']), 'executable': False, 'path': 'sketch/recovery_reference.json'}
        self.latest = value
        self.publish_available_sources()
        return value

    def validate_semantic_assessment(self, answer, *, finish=False):
        """Check provenance/identity of statements, never their semantic truth."""
        def viewed(paths):
            for path in paths:
                self.files.path(path)
                canonical = self.files.canonical_name(path)
                canonical = self.files.image_aliases.get(canonical, canonical)
                if canonical not in self.seen_paths:
                    raise ValueError('Semantic evidence must cite image paths actually viewed by this A')
        previous = {(item['object_id'], item['task_phrase']): item
            for item in self.hub.episode_state.data.get('a_task_referents', [])}
        keys = set()
        for item in answer.get('task_referents', []):
            phrase = item['task_phrase']
            if not phrase.strip() or phrase not in self.handoff['task']:
                raise ValueError('task_referents.task_phrase must quote original_task verbatim; put your interpretation in interpretation')
            key = (item['object_id'], phrase)
            if key in keys:
                raise ValueError('Provide at most one task_referents update per object_id/task_phrase')
            keys.add(key)
            viewed(item.get('image_paths', []))
            old = previous.get(key)
            if old and any(old.get(name) != item.get(name) for name in ('interpretation', 'status')):
                if not item.get('revision_basis', '').strip():
                    raise ValueError('Revising an existing task referent requires revision_basis from identity/structural evidence; a blocked route or size estimate alone does not rename the target')
            previous[key] = item
        if not finish:
            return
        if not any(goal['goal'] == self.handoff['task'] for goal in answer['goal_evidence']):
            raise ValueError('goal_evidence must include the entire original_task verbatim; partial goals do not replace the full task')
        for goal in answer['goal_evidence']:
            if not goal['goal'].strip() or goal['goal'] not in self.handoff['task']:
                raise ValueError('goal_evidence.goal must be a verbatim original_task phrase; put any part interpretation in interpretation')
            viewed(goal['image_paths'])
        if answer['status'] == 'success':
            if any(goal['assessment'] != 'satisfied' or not goal.get('interpretation', '').strip()
                    for goal in answer['goal_evidence']):
                raise ValueError('Success requires satisfied evidence and a separate interpretation for every original-task goal; use blocked for unknown')
            if any(item['status'] == 'unresolved' for item in previous.values()):
                raise ValueError('Task-part interpretations remain unresolved; update them from structural evidence or finish blocked')

    @timed('model_reference_packaging')
    def reference_parts(self, paths):
        """A fixed historical visual reference, separate from changing robot state."""
        from .stage_context import EPISODE_REFERENCES
        parts = [{'type': 'text', 'text': json.dumps({'phase': EPISODE_REFERENCES,
            'assembled_at_source_step': self.hub.robot.steps, 'original_task': self.handoff['task'],
            'historical_evidence_only': True,
            'objects': [{key: item[key] for key in ('object_id', 'component_ids', 'description', 'interactions', 'required_surfaces', 'shape_motion', 'identity_correction')
                if key in item} for item in self.handoff['objects']],
            'object_image_index': [{**item, 'views': [{key: view[key]
                for key in ('rgb', 'height', 'view_id') if key in view} for view in item['views']]}
                for item in self.object_image_index()],
            'entry_files': ['scene.json', 'dimensions.json', 'current.json', 'index.json'],
            'instruction': 'Historical scene/shape references. Check CURRENT owner invalidations before selecting a point. '
                'Read an indexed image once before selecting it; additional angles and full metadata remain in scene.json.'},
            ensure_ascii=False)}]
        for path in dict.fromkeys(paths):
            record, picture = self.read_image(self.live_cloud_files.latest_alias(path), remember=False)
            parts.extend([{'type': 'text', 'text': json.dumps(working_cloud_record(record))}, picture])
        return parts

    def feedback_parts(self, extra_paths=()):
        current = self.latest
        comparison = current.get('held_pose_comparison', {})
        previous = comparison.get('image_paths', [])
        paths = [*extra_paths, *previous, *current.get('hand_image_paths', [])[:1], *['observations/'+current['observation_id']+'/'+camera+'.png'
            for camera in ('agentview', 'robot0_eye_in_hand')]]
        pictures = []
        # Register before serializing CURRENT; its own images must not be []
        # in the advertised source catalogue.
        for path in dict.fromkeys(paths):
            record, picture = self.read_image(path, remember=False)
            if path in previous:
                record['comparison_role'] = 'PREVIOUS actual at last held claim or actual close; not current pose or certified holding'
            elif record.get('kind') == 'native_rgb':
                record['comparison_role'] = 'CURRENT actual outcome'
            pictures.extend([{'type': 'text', 'text': json.dumps(record)}, picture])
        self.publish_available_sources()
        parts = [{'type': 'text', 'text': json.dumps({'phase': 'single_A_direct_execution',
            'initial_acquisition_finished': True, 'CURRENT': working_current(current),
            'public_folder': 'agent_public', 'entry_files': ['scene.json', 'dimensions.json', 'current.json', 'index.json'],
            'instruction': 'CURRENT is the latest measured state. available_sources lists already-read images; '
                'owner/time checks still apply. Pixel + offset + orientation compiles directly. '
                'Full evidence: current.json; all readable angles: scene.json.'}, ensure_ascii=False)}]
        return parts + pictures


    @timed('model_feedback_packaging')
    def feedback_messages(self):
        from .stage_context import ACTUAL_IMAGES
        parts = self.feedback_parts()
        tag = {'phase': ACTUAL_IMAGES, 'source_step': self.latest['source_step'],
            'observation_id': self.latest['observation_id'],
            'instruction': 'Latest actual robot outcome images. Each record identifies its source step. '
                'When available, PREVIOUS actual dual RGB supplies the last held-claim or close-event comparison; '
                'compare the object against pads and fixed scene across the actual motion, not against a propagated prediction. '
                'The structured CURRENT/checkpoint gives current numeric state; older scene references remain historical.'}
        return [{'role': 'user', 'content': parts[:1]},
            {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(tag)}, *parts[1:]]}]


def record_task_referents(hub, updates):
    """Episode-local A claims survive route notes and text checkpoints."""
    if not updates:
        return
    active = hub.episode_state.data.setdefault('a_task_referents', [])
    for item in updates:
        key = (item['object_id'], item['task_phrase'])
        index = next((i for i, old in enumerate(active) if (old['object_id'], old['task_phrase']) == key), None)
        old = active[index] if index is not None else None
        if old and all(old.get(name) == value for name, value in item.items()):
            continue
        record = {**deepcopy(item), 'source_step': hub.robot.steps,
            'revision': old['revision']+1 if old else 1,
            'scope': 'A task-part interpretation, not verified identity, current coordinates or geometric fit. '
                'Source images are historical; motion invalidations still apply. Only new identity/structural evidence revises this interpretation.'}
        if old:
            hub.episode_state.data.setdefault('a_task_referent_history', []).append(deepcopy(old))
            active[index] = record
        else:
            active.append(record)
    hub.episode_state.save()


def apply_assessment(hub, answer):
    from .observed_dimensions import invalidate_carried_dimensions
    from .episode_state import new_held_relation_change
    held = answer['held_object_id']
    changed = new_held_relation_change(hub.episode_state, held, hub.robot.steps, answer['held_pose_relation'])
    previous = getattr(hub, 'carried_target_id', None)
    if changed or previous != held:
        if previous:
            invalidate_carried_dimensions(hub.run, previous, hub.robot.steps, 'Single A actual-image held assessment changed')
        reference = getattr(hub, 'carried_surface_reference', None)
        if not (held and reference and reference.get('object_id') == held):
            hub.carried_surface_reference = None
    reference = getattr(hub, 'carried_surface_reference', None)
    if (held and reference and reference.get('object_id') == held
            and (changed or answer['held_pose_relation']!='changed')):
        from .carry_reference import assess
        assess(reference, hub.robot.steps, answer['held_pose_relation'])
    for identity in answer.get('changed_objects', []):
        hub.journal.mark_moved(identity, hub.robot.steps)
    hub.carried_target_id = held
    record_task_referents(hub, answer.get('task_referents', []))
    if 'task_context_update' in answer:
        record = {'statement': answer['task_context_update'], 'source_step': hub.robot.steps,
            'scope': 'Latest A implementation/dependency note; not physical evidence, verified identity or executable geometry. '
                'Does not replace retained task_referents or original_task.'}
        previous = hub.episode_state.data.get('a_task_context')
        if previous is None or previous['statement'] != record['statement']:
            hub.episode_state.data.setdefault('a_task_context_history', []).append(deepcopy(record))
            hub.episode_state.data['a_task_context'] = record
    hub.episode_state.confirm_held_claim(held, hub.robot.steps, answer['held_pose_relation'])


def validate_request_snapshot(artifacts, answer, request_step):
    """A stale declaration cannot mutate the episode's current assessment."""
    if artifacts.hub.robot.steps != request_step:
        raise ValueError('Request snapshot changed while awaiting this reply; no action dispatched')
    if answer.get('source_step', request_step) != request_step:
        raise ValueError('source_step differs from the physical snapshot at request send')


def bind_request_fields(artifacts, answer, request_step):
    """Bind only program-known evidence; never fill an action or semantic intent."""
    validate_request_snapshot(artifacts, answer, request_step)
    bound = deepcopy(answer)
    bound['source_step'] = request_step
    for index, action in enumerate(bound['actions']):
        if action['primitive'] != 'move' or 'reference_target_id' in action:
            continue
        points = action['selection']['waypoints']
        owners = {artifacts.owners.get(point['anchor'].get('view_id')) for point in points
            if point['anchor']['kind'] == 'cloud_point' and point['anchor'].get('view_id') in artifacts.seen_views}
        if (len(owners) != 1 or None in owners or
                any(point['anchor']['kind'] != 'cloud_point' or point['anchor'].get('view_id') not in artifacts.seen_views
                    for point in points)):
            raise ValueError(f'actions/{index}/reference_target_id: required unless all points use viewed clouds of one known owner')
        action['reference_target_id'] = next(iter(owners))
    return bound


def single_budget(hub):
    limits = hub.single_a_budget_limits
    used = {'requests': getattr(hub, 'single_a_requests', 0),
        'tool_calls': getattr(hub, 'single_a_tool_calls', 0),
        'action_batches': getattr(hub, 'single_a_batches', 0), 'physical_steps': hub.robot.steps,
        'consecutive_invalid_replies': getattr(hub, 'single_a_invalid_replies', 0)}
    return {'limits': dict(limits), 'used': used,
        'remaining': {key: max(0, limit-used[key]) for key, limit in limits.items()},
        'scope': 'Execution phase request/tool/batch limits; physical steps cover the whole episode including acquisition. '
            'Each attempted tool call consumes a tool slot; over-budget replies are rejected before any dispatch.'}


def single_execution_report(hub):
    return {'subtasks': [deepcopy(batch['receipt']) for batch in
            getattr(hub, 'single_a_execution_ledger', {}).get('batches', [])],
        'a_execution_requests': getattr(hub, 'single_a_requests', 0),
        'a_execution_tool_calls': getattr(hub, 'single_a_tool_calls', 0),
        'tool_format_invalid_replies': getattr(hub, 'single_a_format_rejections', 0),
        'binding_or_file_invalid_replies': getattr(hub, 'single_a_binding_rejections', 0),
        'action_batches_accepted': getattr(hub, 'single_a_batches', 0),
        'execution_ledger_file': ('single_a/execution_ledger.json'
            if hasattr(hub, 'single_a_execution_ledger') else None),
        **({'execution_budgets': single_budget(hub)} if hasattr(hub, 'single_a_budget_limits') else {})}


def prepare_actions(executor, artifacts, answer):
    hub = executor.hub
    if answer['source_step'] != hub.robot.steps:
        raise ValueError('source_step is stale; read current.json')
    description, _ = hub.observe()
    steps, selections, packets, views = [], [], [], []
    for action in answer['actions']:
        step = {key: deepcopy(value) for key, value in action.items() if key != 'selection'}
        step['camera'] = 'agentview'
        if step['primitive'] == 'set_gripper':
            step['command'] = action['selection']['command']
        reference = step.get('reference_target_id', step['target_id'])
        packet = artifacts.packet(description, reference)
        if step['primitive'] == 'move':
            for point in action['selection']['waypoints']:
                evidence = artifacts.validate_point(point, reference)
                if evidence.get('retained_cloud_point_binding'):
                    packet.setdefault('retained_cloud_point_bindings', []).append(
                        evidence['retained_cloud_point_binding'])
        steps.append(step); selections.append(action['selection']); packets.append(packet)
        views.append([artifacts.views[key] for key in artifacts.cloud_keys(reference)])
    plan = {'source_step': hub.robot.steps, 'subtask': answer.get('intent'), 'steps': steps,
        'hard_constraints': [], 'preferences': [], 'expected_effects': [], 'object_goals': []}
    stage = expand_stage(plan, selections, [{'kind': 'always'} for _ in steps], packets, views,
        description['source_step'], description['observation_id'])
    for index, step in enumerate(stage['plan']['steps']):
        if step['primitive'] == 'move':
            evidence = artifacts.validate_point(stage['selections'][index], step['reference_target_id'],
                request_step=description['source_step'])
            stage['locations'][index]['grounding_source_step'] = evidence['source_step']
    return plan, stage


def run_single_agent(hub, client, messages, handoff, *, max_turns, max_batches, max_points=6,
                     max_tool_calls=100, max_invalid_replies=4, async_planning=False):
    from .agent import transcript_copy
    from .planning import save_planning_history
    from .stage_context import EPISODE_INSTRUCTIONS
    if min(max_turns, max_batches, max_tool_calls, max_invalid_replies) < 1:
        raise ValueError('Single-agent execution budgets must be positive')
    hub.single_a_budget_limits = {'requests': max_turns, 'tool_calls': max_tool_calls,
        'action_batches': max_batches, 'physical_steps': hub.robot.max_steps,
        'consecutive_invalid_replies': max_invalid_replies}
    hub.single_a_requests = hub.single_a_tool_calls = hub.single_a_batches = 0
    hub.single_a_invalid_replies = 0
    hub.single_a_format_rejections = hub.single_a_binding_rejections = 0
    async_records = []; async_decisions = []; continuation_checks = []; consumed_candidates = set()
    def terminal(status, **fields):
        return {**single_execution_report(hub), 'status': status, **fields,
            'async_planning': {'enabled': async_planning, 'requests': async_records,
                'preparation_decisions': async_decisions, 'continuation_checks': continuation_checks,
                'program_continuations': len(consumed_candidates),
                'scope': 'Same A; normal sketches continue directly. Actual failures return to A; terminal proposals use actual feedback. Overlap is not measured speedup.'}}
    artifacts = EpisodeArtifacts(hub, handoff)
    hub.single_a_artifacts = artifacts
    executor = DirectExecutor(hub, handoff, artifacts)
    initial = artifacts.prepare()
    from .stage_context import archived_messages
    artifacts.files.publish('acquisition/history.json', value={
        'scope': 'Complete acquisition conversation, historical robot states and model judgments; not current evidence.',
        'messages': archived_messages(messages)})
    supplemental = []
    evidence = []
    for identity in getattr(hub, 'initial_supplement_ids', []):
        path = 'observations/'+identity+'/robot0_eye_in_hand.png'
        if path in artifacts.files.entries:
            record, picture = artifacts.read_image(path, remember=False)
            evidence.extend([{'type':'text','text':json.dumps(record)}, picture])
            supplemental.append({key: record[key] for key in ('path','observation_id','camera','source_step')})
    acquisition_roles = {}
    for scan_path in sorted((hub.run/'table_scan').glob('scan_*.json')):
        for attempt in json.loads(scan_path.read_text()).get('attempts', []):
            if attempt.get('captured_observation_id'):
                acquisition_roles[attempt['captured_observation_id']] = attempt['target']['target_id']
    artifacts.files.publish('acquisition/index.json', value={
        'history_file':'acquisition/history.json','supplemental_wrist_evidence':supplemental,
        'native_image_index': [{**{key: entry[key] for key in ('path','observation_id','camera','source_step')},
                'acquisition_role': acquisition_roles.get(entry['observation_id'])}
            for entry in artifacts.files.entries.values() if entry.get('kind') == 'native_rgb'],
        'native_observations_folder':'observations', 'scene_catalog_file':'scene.json',
        'scope':'Raw acquisition telemetry is historical. All native images and full source data remain readable; no camera motion.'})
    prompt = (hub.project_root/'prompts/task_single.md').read_text()
    permanent = EPISODE_INSTRUCTIONS+'\n'+prompt
    if not any(message.get('role') == 'system' and isinstance(message.get('content'), str)
            and message['content'].startswith(permanent) for message in messages):
        messages.append({'role': 'system', 'content': permanent})
    context = getattr(client, 'stage_context', None)
    window = getattr(client, 'image_window', None)
    if window is not None:
        window.register_source(messages); window.active = True
    artifacts.current(executor, hub.run/'single_a/current_000')
    messages.append({'role': 'user', 'content': artifacts.reference_parts(initial)})
    if evidence:
        messages.append({'role':'user','content':[{'type':'text','text':json.dumps({
            'phase':'single_A_acquisition_evidence', 'historical_evidence_only':True,
            'instruction':'Distinct supplemental wrist views before reset, not current outcomes. '
                'All other acquisition images/numeric telemetry remain in acquisition/index.json and observations/.'})}, *evidence]})
    messages.extend(artifacts.feedback_messages())
    tools = tools_for({item['object_id'] for item in handoff['objects']}, max_points)
    from .async_planning import (draft_tool, draft_context_start, execute_overlapped,
        continuation_snapshot, check_continuation)
    draft_schema = draft_tool(tools[0]['function']['parameters'])
    (hub.run/'a_tools.json').write_text(json.dumps(tools, indent=2))
    interfaces = {key: value for key, value in json.loads((hub.run/'interfaces.json').read_text()).items()
        if key in ('protocol_version', 'a_observation', 'a_scene_requirements')}
    interfaces.update(single_agent_tools=tools, single_agent_has_separate_grounding_phase=False)
    interfaces['async_planning'] = {'enabled': async_planning, 'same_persistent_A': True,
        'draft_tool': draft_schema if async_planning else None, 'actual_outcome_approval_required': False,
        'program_continuation': 'all_normal_sketches', 'program_checks': ['pass', 'fail'],
        'preparation_every_segment_with_budget': True, 'draft_may_defer': False,
        'finish_review_is_proposal_only': True}
    (hub.run/'interfaces.json').write_text(json.dumps(interfaces, indent=2))
    def retire_candidate(candidate, reason):
        if candidate is not None:
            artifacts.files.publish('sketch/prepared_next.json', value={
                'candidate': candidate, 'pending': False, 'retired_reason': reason,
                'scope': 'Historical conditional proposal; not an execution command or current geometry.'})
            hub.emit({'event': 'async_candidate_retired', 'candidate_id': candidate['candidate_id'],
                'reason': reason, 'source_step': hub.robot.steps})
    batches = 0; invalid = 0; pending_candidate = None
    while hub.single_a_requests < max_turns:
        turn = hub.single_a_requests+1
        if hub.single_a_tool_calls >= max_tool_calls:
            return terminal('single_agent_tool_budget_exhausted')
        program_dispatch = None
        program_plan = None
        if pending_candidate is not None:
            gate = check_continuation(pending_candidate, artifacts.latest, hub.previous_stage_receipt,
                batch=batches, snapshot=continuation_snapshot(hub), episode=hub.episode_state.data,
                consumed=consumed_candidates)
            if gate['status'] == 'pass' and pending_candidate['status'] == 'ready':
                proposed = {'intent': pending_candidate['intent'], 'actions': deepcopy(pending_candidate['actions']),
                    'source_step': hub.robot.steps}
                try:
                    proposed = bind_request_fields(artifacts, proposed, hub.robot.steps)
                    program_plan = prepare_actions(executor, artifacts, proposed)
                except ValueError as exc:
                    gate['status'] = 'fail'
                    gate['checks'].append({'check': 'current_geometry_preflight', 'status': 'fail', 'reason': str(exc)[:1800]})
                else:
                    program_dispatch = {'candidate_id': pending_candidate['candidate_id'],
                        'authorization': 'conditional_draft', 'continuation_check': deepcopy(gate),
                        'model_request_issued_for_dispatch': False}
            continuation_checks.append(gate)
            artifacts.files.publish('sketch/continuation_check.json', value=gate)
            hub.emit({'event': 'async_continuation_checked', **gate})
            messages.append({'role': 'user', 'content': json.dumps({'phase': 'async_continuation_check', **gate})})
            if gate['status'] == 'fail':
                hub.async_recovery_candidate = deepcopy(pending_candidate)
                retire_candidate(pending_candidate, 'program_check_failed')
                pending_candidate = None
            elif pending_candidate['status'] == 'finish_review':
                hub.async_recovery_candidate = deepcopy(pending_candidate)
                retire_candidate(pending_candidate, 'terminal_proposal_actual_result')
                messages.append({'role': 'user', 'content': json.dumps({
                    'phase': 'async_terminal_actual', 'source_step': hub.robot.steps,
                    'candidate_id': pending_candidate['candidate_id'],
                    'instruction': 'This was only your proposal to end after the executing sketch. '
                        'Use the latest ACTUAL feedback to finish if satisfied or repair if needed. '
                        'All normal tools are available. Do not invent a settling wait for another review.'})})
                pending_candidate = None
        if program_dispatch is not None:
            # This is a program event using an already saved A draft, never a new A reply.
            request_step = hub.robot.steps
            folder = hub.run/'program_continuations'/f'batch_{batches+1:03d}'
            folder.mkdir(parents=True, exist_ok=False)
            (folder/'authorization.json').write_text(json.dumps(program_dispatch, indent=2))
            messages.append({'role': 'user', 'content': json.dumps({'phase': 'program_continuation_dispatch',
                **program_dispatch, 'instruction': 'Program is attempting your saved conditional sketch. '
                    'No new A holding/semantic assessment was made.'})})
            calls = [{'function': {'name': 'program_prepared_sketch', 'arguments': json.dumps(proposed)}}]
            validators = {'program_prepared_sketch': Draft202012Validator(obj({
                'intent': {'type': 'string'}, 'source_step': {'type': 'integer'},
                'actions': tools[0]['function']['parameters']['properties']['actions']},
                ['intent', 'source_step', 'actions']))}
        else:
            reference_candidate = getattr(hub, 'async_recovery_candidate', None)
            retained_draft = draft_context_start(messages, reference_candidate) if reference_candidate is not None else None
            active_tools = tools
            validators = {tool['function']['name']: Draft202012Validator(tool['function']['parameters']) for tool in active_tools}
            if context is not None:
                with timings_for(hub).scope('model_context_preparation'):
                    context.prepare_boundary(messages, artifacts.latest, hub.run/'a_context', hub.emit,
                        **({'retain_from': retained_draft} if retained_draft is not None else {}))
            hub.single_a_requests = turn
            request_step = hub.robot.steps
            artifacts.publish_budget()
            folder = hub.run/'single_a'/f'turn_{turn:03d}'; folder.mkdir(parents=True, exist_ok=True)
            hub.emit({'event': 'single_a_request_started', 'turn': turn, 'source_step': hub.robot.steps})
            started = time.perf_counter()
            with timings_for(hub).scope('api_request'):
                response = client.complete(messages, active_tools)
            (folder/'response.json').write_text(json.dumps(response, indent=2))
            messages.append(response['message'])
            hub.emit({'event': 'single_a_request_finished', 'turn': turn, 'wall_s': time.perf_counter()-started,
                'usage': response.get('usage'), 'transport': client.last_request_metadata})
            calls = response['message'].get('tool_calls') or []
            if not calls:
                # Some compatible providers use a JSON tool envelope instead of native tool calls.
                try:
                    value = json.loads(response['message'].get('content') or '')
                    if 'actions' in value:
                        name, arguments = 'execute_actions', value
                    else:
                        name, arguments = value['name'], value['arguments']
                    calls = [{'function': {'name': name, 'arguments': json.dumps(arguments)}}]
                except (ValueError, TypeError, KeyError):
                    messages.append({'role': 'user', 'content': 'Use one of the supplied tools; no robot action was executed.'})
                    artifacts.record_rejected_attempt(request_turn=turn, request_source_step=request_step,
                        name=None, arguments=response['message'].get('content'),
                        result={'error_kind': 'tool_format', 'error': 'No valid tool call or JSON envelope'},
                        assessment_committed=False)
                    invalid += 1
                    hub.single_a_invalid_replies = invalid
                    hub.single_a_format_rejections += 1
                    artifacts.publish_budget()
                    if invalid >= max_invalid_replies:
                        return terminal('single_agent_invalid_output')
                    continue
        over_budget = program_dispatch is None and hub.single_a_tool_calls+len(calls) > max_tool_calls
        if not over_budget and program_dispatch is None:
            hub.single_a_tool_calls += len(calls)
        artifacts.publish_budget()
        batch_has_mutation = any(call['function']['name'] in ('execute_actions', 'finish', 'refresh_object') for call in calls)
        batch_invalid = batch_has_mutation and len(calls) != 1
        image_parts = []; finished = None; acted = False; rebound = False; references_changed = False
        reply_invalid = False; assessed = False
        for call in calls:
            name = call['function']['name']
            early_tool_receipt = False
            dispatch_started = False
            error_kind = 'tool_format'
            try:
                if over_budget:
                    raise ValueError('Execution tool-call budget exhausted; entire reply rejected, no calls dispatched')
                if name not in validators:
                    raise ValueError('Unknown tool; use the published tool list')
                batch_error = ('Send exactly one execute_actions, refresh_object or finish call; '
                    'do not mix it with other calls. Put multiple robot actions in one execute_actions.actions array.'
                    if batch_invalid else '')
                try:
                    arguments = json.loads(call['function']['arguments'])
                    validators[name].validate(arguments)
                except (json.JSONDecodeError, ValidationError) as exc:
                    detail = (symbolic_schema_feedback(exc, validators[name].schema, arguments)
                        if isinstance(exc, ValidationError) else
                        f'arguments: invalid JSON at line {exc.lineno}, column {exc.colno}, '
                        f'character {exc.pos}: {exc.msg}. Check brackets and commas.')
                    raise ValueError((batch_error+' ' if batch_error else '')+detail) from None
                if batch_invalid:
                    raise ValueError(batch_error)
                error_kind = 'parameter_binding' if name in ('execute_actions', 'program_prepared_sketch', 'refresh_object', 'finish') else 'file_access'
                if name in ('execute_actions', 'program_prepared_sketch'):
                    if batches >= max_batches:
                        raise ValueError('Action-batch budget exhausted; finish with an honest assessment')
                    if program_dispatch is None:
                        artifacts.validate_semantic_assessment(arguments)
                    original_submission = deepcopy(arguments)
                    (folder/'submitted_actions.json').write_text(json.dumps(arguments, indent=2))
                    validate_request_snapshot(artifacts, arguments, request_step)
                    if program_dispatch is None:
                        apply_assessment(hub, arguments)
                        assessed = True
                    arguments = bind_request_fields(artifacts, arguments, request_step)
                    (folder/'request_bound_actions.json').write_text(json.dumps(arguments, indent=2))
                    plan, stage = program_plan if program_dispatch is not None else prepare_actions(executor, artifacts, arguments)
                    dispatch_started = True
                    artifacts.begin_execution(plan, stage, batches+1,
                        submission=original_submission, bound_submission=arguments, program_origin=program_dispatch)
                    if program_dispatch is not None:
                        consumed_candidates.add(pending_candidate['candidate_id'])
                        pending_candidate['executed'] = True
                    hub.async_recovery_candidate = None
                    if pending_candidate is not None:
                        retire_candidate(pending_candidate,
                            'program_continuation' if program_dispatch is not None else 'A_replanned')
                        pending_candidate = None
                    preparation_budget = (batches+1 < max_batches and
                        hub.single_a_requests+2 <= max_turns and hub.single_a_tool_calls+2 <= max_tool_calls)
                    can_prepare = async_planning and preparation_budget
                    if async_planning:
                        decision = {'batch': batches+1, 'source_step': hub.robot.steps,
                            'request_started': can_prepare,
                            'reason': 'every_segment' if can_prepare else 'budget_reserved_for_next_actual_request'}
                        async_decisions.append(decision)
                        hub.emit({'event': 'async_preparation_decision', **decision})
                    draft_error = None
                    try:
                        if can_prepare:
                            early_tool_receipt = True
                            receipt, pending_candidate, entry, draft_error = execute_overlapped(
                                hub, client, messages, draft_schema, call, original_submission, batches+1,
                                lambda: execute_sketch(hub, executor, plan, batches, frozen_binding=stage))
                            async_records.append(entry)
                            if pending_candidate is not None:
                                artifacts.files.publish('sketch/prepared_next.json', value={
                                    'candidate': pending_candidate, 'pending': True,
                                    'scope': 'Conditional proposal only; no action dispatched or future state certified.'})
                        else:
                            receipt = execute_sketch(hub, executor, plan, batches, frozen_binding=stage)
                    except BaseException as exc:
                        artifacts.interrupt_execution(exc)
                        raise
                    batches += 1; acted = True
                    artifacts.finish_execution(receipt)
                    hub.previous_stage_receipt = receipt
                    if draft_error is not None:
                        from .agent import RejectedResponse
                        if not isinstance(draft_error, RejectedResponse):
                            artifacts.current(executor, folder/'actual')
                            save_planning_history(hub.run, messages)
                            raise draft_error
                    if not receipt.get('commands_completed'):
                        hub.episode_state.remember_route_failure(previous_stage_outcome(receipt))
                    for item in receipt.get('steps', []):
                        if item.get('result', {}).get('converged') is False:
                            goal = item['result'].get('continuation_goal') or item.get('parameters', {})
                            if 'position' in goal and 'quaternion_xyzw' in goal:
                                hub.single_last_motion_failure = {'failure_stage': 'motion', 'executed': True,
                                    'actual_motion': item['result'], 'original_absolute_goal': goal}
                    result = public(receipt)
                    hub.emit({'event': 'single_a_actions_finished', 'batch': batches,
                        'dispatch_origin': 'program_continuation' if program_dispatch is not None else 'model_tool',
                        'source_step': hub.robot.steps, 'commands_completed': receipt.get('commands_completed')})
                elif name == 'list_dir':
                    result = artifacts.files.list_dir(**arguments)
                elif name == 'read_text_file':
                    result = artifacts.files.read_text_file(**arguments)
                elif name == 'read_image':
                    before_references = list(artifacts.reference_reads)
                    result, picture = artifacts.read_image(**arguments)
                    references_changed |= before_references != artifacts.reference_reads
                    image_record = (working_cloud_record(result)
                        if result.get('kind') == 'point_cloud_projection' else result)
                    image_parts += [{'type': 'text', 'text': json.dumps(image_record)}, picture]
                elif name == 'get_state':
                    dispatch_started = True  # Internal evidence errors are program faults, not argument corrections.
                    result = artifacts.current(executor, folder/'state')
                elif name == 'refresh_object':
                    prepared = artifacts.validate_refresh(arguments, request_step)
                    dispatch_started = True
                    before_references = list(artifacts.reference_reads)
                    result, parts, rebound = artifacts.refresh_object(arguments, prepared)
                    retire_candidate(pending_candidate, 'explicit_object_refresh')
                    pending_candidate = None
                    image_parts.extend(parts)
                    references_changed |= before_references != artifacts.reference_reads
                else:
                    artifacts.validate_semantic_assessment(arguments, finish=True)
                    record_task_referents(hub, arguments.get('task_referents', []))
                    result = {'original_task': handoff['task'], 'agent_assessment': arguments,
                        'semantic_identity_certified': False, 'official_success_visible_to_agent': False}
                    finished = arguments
                    retire_candidate(pending_candidate, 'A_finished')
                    pending_candidate = None
            except (ValidationError, ValueError) as exc:
                # Physical execution exceptions must never be recast as a harmless tool typo.
                if acted or dispatch_started:
                    raise
                error = (symbolic_schema_feedback(exc, validators[name].schema, arguments)
                    if isinstance(exc, ValidationError) else str(exc)[:1800])
                result = {'executed': False, 'error': error, 'robot_actions': 0,
                    'error_kind': error_kind, 'physical_execution_failed': False,
                    'correction': 'Resubmit a corrected tool call using the same current state. '
                        'No robot action or action-batch slot was consumed; API/tool limits still apply.'}
                if not reply_invalid:
                    hub.single_a_invalid_replies = invalid+1
                    if error_kind == 'tool_format':
                        hub.single_a_format_rejections += 1
                    else:
                        hub.single_a_binding_rejections += 1
                reply_invalid = True
                artifacts.record_rejected_attempt(request_turn=turn, request_source_step=request_step,
                    name=name, arguments=call['function'].get('arguments'), result=result,
                    assessment_committed=assessed)
                hub.emit({'event': 'single_a_tool_rejected', 'turn': turn, 'name': name, **result})
            # Archive the full result below; present the same canonical cards as
            # automatic feedback instead of repeating expanded source metadata.
            presented = result
            if name == 'get_state' and 'robot_state' in result:
                presented = working_current(result)
            elif name == 'read_image' and result.get('kind') == 'point_cloud_projection':
                presented = working_cloud_record(result)
            presented = {**presented, 'execution_budgets': single_budget(hub)}
            if call.get('id') and not early_tool_receipt:
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(presented, ensure_ascii=False)})
            else:
                messages.append({'role': 'user', 'content': json.dumps({'tool': name, 'result': presented}, ensure_ascii=False)})
            hub.emit({'event': 'single_a_tool_result', 'turn': turn, 'name': name,
                'source_step': hub.robot.steps, 'result': result})
        if image_parts:
            messages.append({'role': 'user', 'content': image_parts})
        if references_changed:
            messages.append({'role': 'user', 'content': artifacts.reference_parts(artifacts.reference_image_paths())})
        invalid = invalid+1 if reply_invalid else 0
        hub.single_a_invalid_replies = invalid
        artifacts.publish_budget()
        with timings_for(hub).scope('history_archive_write'):
            save_planning_history(hub.run, messages)
        (folder/'transcript.json').write_text(json.dumps(transcript_copy(messages), indent=2))
        if finished:
            return terminal('a_reports_full_task_completed' if finished['status'] == 'success' else 'single_agent_'+finished['status'],
                agent_assessment=finished, task_success_certified_by_commands=False)
        if over_budget:
            return terminal('single_agent_tool_budget_exhausted')
        if invalid >= max_invalid_replies:
            return terminal('single_agent_invalid_output')
        if hub.robot.done:
            return terminal('simulator_episode_ended')
        if acted or rebound or assessed:
            artifacts.current(executor, folder/'actual')
            if window is not None:
                window.register_source(messages)
            prepared_failure = (artifacts.prepare_failure_geometry(stage, receipt)
                if acted and not receipt.get('commands_completed') else None)
            # Assessment declarations already invalidate old world anchors.
            # They are not requests to rigid-register every changed object
            # after the new sketch; current native/pad/EEF recovery stays usable.
            if rebound or prepared_failure:
                messages.append({'role': 'user', 'content': artifacts.reference_parts(artifacts.reference_image_paths())})
            messages.extend(artifacts.feedback_messages())
    return terminal('single_agent_request_budget_exhausted')
