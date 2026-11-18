from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from jsonschema import Draft202012Validator, ValidationError

from sketchik.geometry_journal import GeometryJournal
from sketchik.public_files import PublicFiles
from sketchik.single_agent import (EpisodeArtifacts, DirectExecutor, prepare_actions,
    run_single_agent, apply_assessment, tools_for)
from sketchik.task_execution import execute_sketch, BindingUnavailable


def test_native_camera_frame_uses_saved_axes_without_inferring_object_directions():
    from scipy.spatial.transform import Rotation
    from sketchik.single_agent import native_camera_frame
    transform = np.eye(4); transform[:3, :3] = Rotation.from_euler('xyz', [.4, -.2, 1.1]).as_matrix()
    transform[:3, 3] = [.45, -.1, 1.6]
    camera = {'world_from_camera': transform.tolist()}; original = deepcopy(camera)
    result = native_camera_frame(camera)
    for index, name in enumerate(('screen_right_world_unit', 'screen_down_world_unit', 'viewing_forward_world_unit')):
        np.testing.assert_allclose(result[name], transform[:3, index], atol=5.1e-7)
    assert result['camera_position_world_m'] == [.45, -.1, 1.6]
    assert camera == original and native_camera_frame({}) is None
    assert 'Not object axes' in result['scope']
    invalid = transform.copy(); invalid[1, 1] += .1
    with pytest.raises(ValueError, match='rigid calibration'):
        native_camera_frame({'world_from_camera': invalid.tolist()})


def test_acquisition_images_keep_display_provenance_without_authorizing_unshown_frames(episode):
    from sketchik.planning import image_part
    hub, handoff, actions, _ = episode
    description, _ = hub.observe()
    identity = description['observation_id']
    frame = hub.run/'observations'/identity
    shown = image_part(frame/'agentview.png')
    # A saved/cached frame alone must not count as displayed. Use a distinct
    # wrist bitmap so byte equality cannot stand in for the missing display.
    Image.new('RGB', (32, 32), 'red').save(frame/'robot0_eye_in_hand.png')
    unshown = image_part(frame/'robot0_eye_in_hand.png')
    hub._display_cache = {(identity, 'agentview'): ({}, shown),
        (identity, 'robot0_eye_in_hand'): ({}, unshown),
        ('missing', 'agentview'): ({}, shown)}
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Also claim to have seen the wrist image'}, shown]}]
    original = deepcopy(messages)
    artifacts = EpisodeArtifacts(hub, handoff)
    inherited = artifacts.inherit_shown_native_images(messages)
    assert artifacts.seen_rgb == {(identity, 'agentview')}
    assert inherited == [{'observation_id': identity, 'camera': 'agentview',
        'source_step': 30, 'path': f'observations/{identity}/agentview.png'}]
    assert artifacts.inherit_shown_native_images(messages) == inherited
    assert messages == original and actions == [] and hub.robot.steps == 30
    assert not artifacts.inherit_shown_native_images([
        {'role': 'assistant', 'content': [unshown]}])
    assert (identity, 'robot0_eye_in_hand') not in artifacts.seen_rgb


def test_native_image_records_keep_their_own_calibration_and_original_pixels(episode):
    import hashlib
    hub, handoff, actions, _ = episode
    description, _ = hub.observe()
    folder = hub.run/'observations'/description['observation_id']
    path = folder/'observation.json'; metadata = json.loads(path.read_text())
    for index, camera in enumerate(metadata['cameras']):
        transform = np.eye(4); transform[:3, 3] = [index, .2, 1.4]
        camera['world_from_camera'] = transform.tolist()
    path.write_text(json.dumps(metadata))
    before = hashlib.sha256((folder/'agentview.png').read_bytes()).hexdigest()
    artifacts = EpisodeArtifacts(hub, handoff); artifacts.sync_observations()
    for index, camera in enumerate(('agentview', 'robot0_eye_in_hand')):
        record, _ = artifacts.read_image(f'observations/{description["observation_id"]}/{camera}.png')
        assert record['camera_frame']['camera_position_world_m'] == [index, .2, 1.4]
        assert record['camera_frame']['viewing_forward_world_unit'] == [0., 0., 1.]
        assert record['source_step'] == 30 and record['capture_is_current_physical_step']
    assert hashlib.sha256((folder/'agentview.png').read_bytes()).hexdigest() == before
    assert actions == [] and hub.robot.steps == 30


@pytest.mark.parametrize('eye', [[1.,2.,3.],[0.,0.,1.],[0.,0.,-1.]])
def test_cloud_projection_directions_match_the_existing_rasterizer(eye):
    from sketchik.single_agent import projection_frame
    result = projection_frame({'virtual_camera_eye_direction':eye})['projection_frame']
    toward = np.asarray(result['toward_viewer_world_unit'])
    right = np.asarray(result['screen_right_world_unit'])
    down = np.asarray(result['screen_down_world_unit'])
    np.testing.assert_allclose(toward, np.asarray(eye)/np.linalg.norm(eye))
    np.testing.assert_allclose(np.stack([right,down,-toward])@np.stack([right,down,-toward]).T,
        np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(-toward,right),down,atol=1e-12)
    from sketchik.cloud_selection import rasterize
    points = np.random.default_rng(2).normal(size=(64,3))
    _, _, metadata = rasterize(points, np.zeros((64,3),np.uint8), mode='rgb', eye=eye, size=256)
    for index, axis in enumerate('XYZ'):
        np.testing.assert_allclose(metadata['world_positive_axes_screen_xy'][axis],
            [right[index], down[index]], atol=1e-12)
    assert projection_frame({}) == {}


def test_single_tool_schema_describes_compensation_and_excludes_unavailable_openings():
    from sketchik.pose_binding import BINDING
    from sketchik.local_route import route_schema
    tools = tools_for({'object'})
    move = tools[0]['function']['parameters']['properties']['actions']['items']['oneOf'][0]
    def contract(schema):
        if isinstance(schema, dict):
            return {key:contract(value) for key,value in schema.items() if key != 'description'}
        return [contract(item) for item in schema] if isinstance(schema,list) else schema
    expected = deepcopy(BINDING)
    expected['properties']['anchor']['properties']['kind']['enum'].remove('opening')
    expected['properties']['anchor']['properties'].pop('opening_index')
    expected['properties']['offset_frame']['enum'].remove('opening')
    assert contract(move['properties']['selection']) == contract(route_schema(expected))
    assert 'opening' in BINDING['properties']['anchor']['properties']['kind']['enum']  # AB unchanged.
    for point_binding in [binding({'kind': 'opening', 'opening_index': 0}),
            {**binding(), 'offset_frame': 'opening'}]:
        with pytest.raises(ValidationError):
            Draft202012Validator(move['properties']['selection']).validate({'waypoints': [point_binding]})
    point = move['properties']['selection']['properties']['waypoints']['items']['properties']
    assert 'pad_midpoint = midpoint' in point['reference']['description']
    assert 'not to EEF unless reference=eef' in point['offset_m']['description']


def test_working_current_pairs_projections_and_retains_full_public_evidence():
    from sketchik.single_agent import working_current
    full = {'source_step':30,'robot_state':{'q':[1]*7},'geometry':{'robot_gripper_geometry':{}},
        'dimensions':{'object':{'observed_principal_spans_m':[.2,.1,.03],
            'observed_principal_axes_world':[[0,0,1],[0,1,0],[1,0,0]],'principal_axis_gap_ratios':[.8,.6]}},
        'available_sources':{'cloud_views':[{'view_id':'exact','image_path':'geometry/view.png',
            'source_observations':[{'source_step':12}],'point_index_path':'data/exact.npz'}]}}
    original = deepcopy(full); work = working_current(full)
    assert full == original and work['robot_state'] == full['robot_state']
    assert work['geometry']['robot_gripper_geometry'] == {}  # Missing calibration not invented.
    assert work['dimensions']['object']['observed_principal_projections'][2] == {
        'direction_world_unit':[1,0,0], 'projected_span_m':.03}
    assert work['dimensions']['object']['principal_axis_gap_ratios'] == [.8,.6]
    view = work['available_sources']['cloud_views'][0]
    assert view['view_id'] == 'exact' and view['point_index_path'] == 'data/exact.npz'
    assert view['source_metadata_file'] == 'geometry/view.json'
    assert full['available_sources']['cloud_views'][0]['source_observations'] == [{'source_step':12}]


def test_compact_point_cards_keep_exact_sources_and_do_not_compact_actual_camera_evidence():
    from sketchik.single_agent import working_cloud_record, working_current
    native = {'kind': 'native_rgb', 'observation_id': 'obs_00007', 'source_step': 80,
        'camera': 'robot0_eye_in_hand', 'camera_frame': {'screen_right_world_unit': [0, 1, 0]},
        'capture_is_current_physical_step': True}
    assert working_cloud_record(native) == native
    source = {'view_id': 'part_exact', 'object_id': 'part', 'image_path': 'geometry/part.png',
        'width': 1024, 'height': 1024, 'group_ids': [0, 2], 'selectable': True,
        'source_observations': [{'observation_id': 'obs_00001', 'source_step': 20}],
        'point_index_path': 'geometry/data/part_exact.npz',
        'projection_frame': {'screen_right_world_unit': [1, 0, 0]},
        'point_time_status': {'owner': 'part', 'invalidated_at_step': 70, 'rule': 'source-time guard'},
        'group_scope': 'Expanded provenance in the canonical public metadata'}
    original = deepcopy(source)
    shown = working_cloud_record(source)
    card = working_current({'available_sources': {'cloud_views': [source]}})['available_sources']['cloud_views'][0]
    assert source == original
    assert shown['projection_frame'] == source['projection_frame']
    assert 'projection_frame' not in card  # Display axes appear once with the image, not in every source catalogue.
    for value in (shown, card):
        assert value['group_ids'] == [0, 2] and value['point_index_path'] == source['point_index_path']
        assert value['point_time_status'] == {'owner': 'part', 'invalidated_at_step': 70}
        assert value['source_metadata_file'] == 'geometry/part.json'
        assert value['width'] == value['height'] == 1024


def test_reference_card_uses_exact_compiler_offsets_in_all_three_axes():
    from itertools import product
    from scipy.spatial.transform import Rotation
    from sketchik.single_agent import reference_working_card
    from sketchik.pose_binding import compile_binding
    local = np.asarray(list(product([-.02,.02], [.01,.12], [-.10,.03]))*4)
    packet = {'source_step': 55, 'eef': {'position':[.2,.3,1.],
        'quaternion_xyzw':Rotation.from_euler('xyz',[.4,.2,.7]).as_quat().tolist()},
        'robot_gripper_geometry': {'pad_centres_in_eef_tool_m':[[-.02,0,-.004],[.02,0,-.004]]},
        'carried_observed_geometry': {'reference_available':True,'object_id':'book'}}
    card = reference_working_card(packet,local)
    points = local@Rotation.from_quat(packet['eef']['quaternion_xyzw']).as_matrix().T+packet['eef']['position']
    centre = (points.min(axis=0)+points.max(axis=0))/2
    np.testing.assert_allclose(card['references_at_current_orientation']['held_center']['position_world_m'], centre)
    np.testing.assert_allclose(card['references_at_current_orientation']['held_bottom']['position_world_m'],
        [centre[0],centre[1],points[:,2].min()])
    reference = card['references_at_current_orientation']['held_center']
    assert all(abs(x)>.001 for x in reference['offset_from_eef_world_m'])
    target = {'anchor':{'kind':'current_eef'},'reference':'held_bottom','offset_m':[.1,.2,-.1],
        'offset_frame':'world','orientation':{'rpy_world_rad':[0.,.8,1.]}}
    pose,audit = compile_binding(target,packet,lambda *a:None,carried_local=local)
    target_points = local@Rotation.from_quat(pose['quaternion_xyzw']).as_matrix().T+pose['position']
    wanted = np.asarray(packet['eef']['position'])+target['offset_m']
    np.testing.assert_allclose((target_points.min(axis=0)+target_points.max(axis=0))[:2]/2,wanted[:2])
    assert target_points[:,2].min() == pytest.approx(wanted[2])
    for row in card['held_observed_principal_projections']:
        assert np.linalg.norm(row['direction_eef_tool_unit']) == pytest.approx(1.)
        projected = local@row['direction_eef_tool_unit']
        assert projected.ptp() == pytest.approx(row['projected_span_m'])
    packet['carried_observed_geometry']['reference_available'] = False
    unavailable = reference_working_card(packet,local)
    assert not unavailable['held_reference_available']
    assert 'held_center' not in unavailable['references_at_current_orientation']
    assert 'held_observed_principal_projections' not in unavailable


def test_current_hand_mixed_layers_are_not_blanket_historical_or_selectable(episode, tmp_path):
    hub,handoff,_,view = episode
    artifacts = EpisodeArtifacts(hub,handoff)
    artifacts.files.publish('hand/30_0.png',source=view['image_file'],metadata={
        'kind':'hand_geometry','source_step':30,'observation_id':'obs_00030','selectable':False,
        'evidence_layers':{'magenta_shape':{'new_observation':False,'attachment_certified':False}}})
    record,_ = artifacts.read_image('hand/30_0.png')
    assert record['historical_evidence_only'] is False
    assert record['capture_is_current_physical_step'] is False
    assert record['mixed_evidence_layers'] is True
    assert record['current_measured_layers_at_current_physical_step'] is True
    assert record['selectable'] is False and artifacts.seen_rgb == set()
    assert record['evidence_layers']['magenta_shape']['new_observation'] is False
    hub.robot.steps += 1
    old,_ = artifacts.read_image('hand/30_0.png')
    assert old['historical_evidence_only'] and not old['current_measured_layers_at_current_physical_step']


def test_working_receipt_keeps_failure_actual_stop_and_unexecuted_suffix_without_changing_archive():
    from sketchik.single_agent import working_current
    failure = {'source_step':42,'commands_completed':False,'failed_step_index':0,
        'remaining_suffix_executed':False,'failure':'motion_not_converged','steps':[
            {'primitive':'move','parameters':{'position':[1,2,3],'quaternion_xyzw':[0,0,0,1]},
             'result':{'converged':False,'position_error_m':.12,'rotation_error_deg':33.,
                 'state':{'step':42,'position':[.9,2,3],'quaternion_xyzw':[0,0,0,1]},
                 'q_history':[0]*1000}}]}
    full = {'source_step':42,'robot_state':{'position':[.9,2,3]},'geometry':{},
        'recent_execution_receipts':[failure],'execution_ledger':[{'batch':1}]}
    original = deepcopy(full); work = working_current(full)
    assert full == original
    assert list(work).index('recent_execution_receipts') < list(work).index('geometry')
    receipt = work['recent_execution_receipts'][0]
    assert receipt['remaining_suffix_executed'] is False and receipt['commands_completed'] is False
    assert receipt['receipt_path'] == 'receipts/batch_001.json'
    assert receipt['motions'][0]['actual_stop']['position'] == [.9,2,3]
    assert receipt['motions'][0]['position_error_m'] == .12
    assert 'q_history' not in str(receipt)


def test_public_files_are_explicit_paged_and_cannot_escape(tmp_path):
    store = PublicFiles(tmp_path/'public')
    secret = tmp_path/'agent.env'; secret.write_text('PRIVATE_TOKEN')
    store.publish('geometry/info.json', value={'observed': True})
    (store.root/'not_published.txt').write_text('not public')
    for name in ('../agent.env', str(secret), 'not_published.txt', 'geometry/../../agent.env'):
        with pytest.raises(ValueError):
            store.read_text_file(name)
    (store.root/'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        store.list_dir('escape')
    assert [item['path'] for item in store.list_dir()['entries']] == ['geometry']
    store.publish('long.txt', text='abcdefgh')
    page = store.read_text_file('long.txt', offset=2, max_chars=3)
    assert page['content'] == 'cde' and page['next_offset'] == 5
    assert 'PRIVATE_TOKEN' not in json.dumps(store.list_dir())


def test_point_image_cannot_change_after_publication(tmp_path):
    original = tmp_path/'rgb.png'; Image.new('RGB', (32, 32)).save(original)
    store = PublicFiles(tmp_path/'public')
    store.publish('cloud.png', source=original)
    Image.new('RGB', (32, 32), 'red').save(store.root/'cloud.png')
    with pytest.raises(ValueError, match='changed'):
        store.read_image('cloud.png')


def test_public_root_prefix_resolves_same_artifact_without_broadening_access(tmp_path):
    original = tmp_path/'rgb.png'; Image.new('RGB', (32, 32)).save(original)
    store = PublicFiles(tmp_path/'agent_public')
    store.publish('geometry/view.png', source=original)
    store.publish('geometry/info.json', value={'observed': True})
    bare, picture = store.read_image('geometry/view.png')
    rooted, rooted_picture = store.read_image('agent_public/geometry/view.png')
    assert rooted == bare and rooted_picture == picture
    assert store.read_text_file('agent_public/geometry/info.json') == store.read_text_file('geometry/info.json')
    assert store.list_dir('agent_public/geometry') == store.list_dir('geometry')
    assert store.list_dir('agent_public') == store.list_dir()
    for name in ('agent_public/../agent.env', 'agent_public//geometry/view.png',
                 'agent_public/./geometry/view.png', 'agent_public/geometry/../view.png',
                 'agent_public/agent_public/geometry/view.png', 'agent_public/not_registered.png'):
        with pytest.raises(ValueError):
            store.read_image(name)
    (store.root/'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        store.list_dir('agent_public/escape')


@pytest.fixture
def episode(tmp_path, monkeypatch):
    state = {'step': 30, 'q': [0.]*7, 'position': [0., 0., 1.],
        'quaternion_xyzw': [0., 0., 0., 1.], 'gripper_command': -1.}
    robot = SimpleNamespace(steps=30, done=False, max_steps=2000, state=lambda: deepcopy(state))
    executed = []
    def action(primitive, **parameters):
        executed.append(primitive)
        state['step'] += 1; robot.steps = state['step']
        if primitive == 'move':
            state.update(position=parameters['position'], quaternion_xyzw=parameters['quaternion_xyzw'])
        if primitive == 'set_gripper':
            state['gripper_command'] = 1. if parameters['command'] == 'close' else -1.
        return {'converged': True}
    ik = SimpleNamespace(solve=lambda *args: SimpleNamespace(q=np.zeros(7),
        position_error_m=0., rotation_error_rad=0., elapsed_s=0.), cartesian_path=lambda *args: [])
    executor = SimpleNamespace(ik=ik, move=lambda **kw: action('move', **kw),
        set_gripper=lambda **kw: action('set_gripper', **kw))
    hub = SimpleNamespace(run=tmp_path, project_root=Path(__file__).resolve().parents[1],
        robot=robot, executor=executor, journal=GeometryJournal(tmp_path), ik_only=True,
        carried_target_id=None, carried_surface_reference=None, emit=lambda event: None,
        _fixture_state=state)
    def observe():
        identity = f'obs_{state["step"]:05d}'
        folder = tmp_path/'observations'/identity; folder.mkdir(parents=True, exist_ok=True)
        (folder/'observation.json').write_text(json.dumps({'source_step': state['step'],
            'cameras': [{'camera': camera} for camera in ('agentview', 'robot0_eye_in_hand')]}))
        for camera in ('agentview', 'robot0_eye_in_hand'):
            Image.new('RGB', (32, 32), 'blue').save(folder/(camera+'.png'))
            if not (folder/(camera+'.npz')).exists():
                np.savez(folder/(camera+'.npz'), depth_m=np.ones((32, 32), np.float32),
                    K=np.asarray([[80., 0., 16.], [0., 80., 16.], [0., 0., 1.]]),
                    world_from_camera=np.eye(4))
        return {'source_step': state['step'], 'observation_id': identity, 'state': deepcopy(state)}, []
    hub.observe = observe
    cloud = tmp_path/'points.npz'
    np.savez(cloud, world_xyz_m=np.tile([[.2, .3, .9]], (30, 1)), rgb=np.zeros((30, 3), np.uint8),
        source_index=np.zeros(30, int), pixel_xy=np.zeros((30, 2), int))
    mapping = tmp_path/'point_index.npz'; np.savez(mapping, point_index=np.zeros((32, 32), int))
    picture = tmp_path/'cloud.png'; Image.new('RGB', (32, 32)).save(picture)
    view = {'view_id': 'object_face', 'cloud_file': str(cloud), 'index_map_file': str(mapping),
        'image_file': str(picture), 'sources': [{'source_step': 10, 'observation_id': 'scan', 'camera': 'agentview'}],
        'selectable': True, 'role': 'recorded object', 'width': 32, 'height': 32,
        'point_count': 30, 'mode': 'rgb'}
    handoff = {'task': 'place the object', 'source_step': 30, 'whole_task_coverage': 'visible surfaces',
        'objects': [{'object_id': 'object', 'description': 'object', 'interactions': ['grasp'],
            'needed_geometry': 'shape', 'evidence': [], 'required_surfaces': ['overall_shape']}]}
    monkeypatch.setattr('sketchik.grounding_packet.build_focused_context', lambda *a, **kw: {
        'local_context_available': True, 'selectable_views': [view]})
    monkeypatch.setattr('sketchik.cloud_selection.height_auxiliary', lambda v, folder: {
        **v, 'view_id': v['view_id']+'_height', 'mode': 'height', 'same_pixel_map_as': v['view_id']})
    (tmp_path/'interfaces.json').write_text('{}')
    return hub, handoff, executed, view


def binding(anchor=None):
    return {'anchor': anchor or {'kind': 'current_eef'}, 'reference': 'eef',
        'offset_m': [0, 0, .1], 'offset_frame': 'world', 'orientation': 'preserve'}


def live_fixture(episode):
    hub, handoff, actions, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    artifacts.prepare()
    executor = DirectExecutor(hub, handoff, artifacts)
    description, _ = hub.observe()
    folder = hub.run/'observations'/description['observation_id']
    for camera in ('agentview', 'robot0_eye_in_hand'):
        transform = np.eye(4); transform[:3, 3] = [.2, .3, 0.]
        np.savez(folder/(camera+'.npz'), depth_m=np.full((32, 32), .9, np.float32),
            K=np.asarray([[80., 0., 16.], [0., 80., 16.], [0., 0., 1.]]),
            world_from_camera=transform)
    artifacts.current(executor, hub.run/'live_initial')
    return hub, artifacts, executor


def advance_live_fixture(hub, artifacts, executor, *, operate=True, drain=True):
    before = hub.robot.steps
    hub._fixture_state['step'] += 1
    hub.robot.steps = hub._fixture_state['step']
    description, _ = hub.observe()
    source = hub.run/'observations'/f'obs_{before:05d}'
    folder = hub.run/'observations'/description['observation_id']
    import shutil
    for camera in ('agentview', 'robot0_eye_in_hand'):
        shutil.copyfile(source/(camera+'.npz'), folder/(camera+'.npz'))
    if operate:
        artifacts.live_dispatches.append({'source_step': before, 'targets': ['object'],
            'framing_points': [[.2, .3, .9]], 'physical_steps_advanced': 1})
    result = artifacts.current(executor, hub.run/f'live_current_{hub.robot.steps}')
    if drain:
        artifacts.live_cloud_files.collect_ready(wait=True)
    return result


def test_background_cloud_does_not_block_A_but_explicit_read_waits_for_exact_snapshot(episode, monkeypatch):
    from threading import Event, Thread
    import sketchik.live_cloud as cloud
    started, release, returned = Event(), Event(), Event()
    real = cloud.prepare_snapshot
    def slow(*args):
        started.set()
        assert release.wait(5)
        return real(*args)
    monkeypatch.setattr(cloud, 'prepare_snapshot', slow)
    hub, artifacts, executor = live_fixture(episode)
    try:
        current = advance_live_fixture(hub, artifacts, executor, drain=False)
        assert started.wait(2) and current['current_clouds']['overview_pending']
        assert artifacts.latest['source_step'] == hub.robot.steps
        assert sum(p.get('type') == 'image_url' for p in artifacts.feedback_parts()) == 2
        result = []
        def read():
            result.append(artifacts.read_image('current_clouds/latest/overview.png')[0])
            returned.set()
        reader = Thread(target=read); reader.start()
        assert not returned.wait(.05)
        release.set(); reader.join(5)
        assert returned.is_set() and result[0]['source_step'] == hub.robot.steps
        assert result[0]['view_kind'] == 'retained_scene_context' and not result[0]['selectable']
        assert result[0]['logical_image_slot'] == 'current_clouds/latest/overview.png'
    finally:
        release.set(); artifacts.live_cloud_files.close()


def test_background_worker_error_is_not_downgraded_to_bad_view(episode, monkeypatch):
    def broken(*args):
        raise OSError('fixture disk failure')
    monkeypatch.setattr('sketchik.live_cloud.prepare_snapshot', broken)
    hub, artifacts, executor = live_fixture(episode)
    try:
        with pytest.raises(OSError, match='disk failure'):
            advance_live_fixture(hub, artifacts, executor)
    finally:
        # The failed future remains available until its fault is inspected.
        with pytest.raises(OSError, match='disk failure'):
            artifacts.live_cloud_files.close()


def test_older_background_job_cannot_restore_retired_latest_pixels(episode, monkeypatch):
    from threading import Event
    import sketchik.live_cloud as cloud
    started, release = Event(), Event()
    real = cloud.prepare_snapshot
    def delayed(description, *args):
        if description['source_step'] == 31:
            started.set()
            assert release.wait(5)
        return real(description, *args)
    monkeypatch.setattr(cloud, 'prepare_snapshot', delayed)
    hub, artifacts, executor = live_fixture(episode)
    try:
        advance_live_fixture(hub, artifacts, executor, drain=False)
        assert started.wait(2)
        advance_live_fixture(hub, artifacts, executor, drain=False)
        assert artifacts.files.entries['current_clouds/latest/overview.png']['source_step'] == 32
        release.set()
        artifacts.live_cloud_files.collect_ready(wait=True)
        artifacts.live_cloud_files.collect_ready()
        record, _ = artifacts.read_image('current_clouds/latest/overview.png')
        assert record['source_step'] == 32
        assert artifacts.live_cloud_files.snapshot['source_step'] == 32
        assert artifacts.live_cloud_files.last_update['source_step'] == 32
    finally:
        release.set(); artifacts.live_cloud_files.close()


def test_latest_listing_does_not_render_or_view_and_feedback_does_not_add_cloud_images(episode):
    hub, artifacts, executor = live_fixture(episode)
    count = lambda parts: sum(part.get('type') == 'image_url' for part in parts)
    before_count = count(artifacts.feedback_parts())
    original_references = artifacts.reference_image_paths()
    current = advance_live_fixture(hub, artifacts, executor)
    before = set(artifacts.views)
    listing = artifacts.files.list_dir(current['current_clouds']['directory'])
    slot = next(v for v in listing['entries'] if v['path'].endswith('object_00_face.png'))
    assert slot['deferred'] and slot['resolved_path'] is None
    assert set(artifacts.views) == before and not any(artifacts.current_context(v) for key, v in artifacts.views.items()
        if key in artifacts.seen_views)
    assert count(artifacts.feedback_parts()) == before_count
    assert artifacts.reference_image_paths() == original_references
    assert not list((artifacts.live_cloud_files.snapshot['folder']).glob('slot_*/*.png'))


def test_changed_geometry_paths_are_explicit_and_display_step_updates_only_after_read(episode):
    hub, artifacts, executor = live_fixture(episode)
    current = advance_live_fixture(hub, artifacts, executor)
    entry = current['current_clouds']['updated_object_views'][0]
    assert entry['object_id'] == 'object'
    assert entry['latest_displayed_step'] < hub.robot.steps
    assert entry['paths'] == ['current_clouds/latest/object_00_face.png']  # This fixture has one view.
    alias = 'current_clouds/latest/object_00_face.png'
    assert alias in entry['paths']
    artifacts.files.list_dir('current_clouds/latest')
    assert artifacts.live_cloud_files.status(hub.observe()[0])['updated_object_views'][0]['latest_displayed_step'] < hub.robot.steps
    record, _ = artifacts.read_image(alias)
    after = artifacts.live_cloud_files.status(hub.observe()[0])['updated_object_views'][0]
    assert after['latest_displayed_step'] == record['source_step'] == hub.robot.steps
    advance_live_fixture(hub, artifacts, executor)
    updated = artifacts.live_cloud_files.status(hub.observe()[0])['updated_object_views'][0]
    assert updated['latest_displayed_step'] < hub.robot.steps


def test_read_latest_materializes_one_exact_selectable_view_without_implicit_binding(episode):
    from sketchik.single_agent import bind_request_fields
    hub, artifacts, executor = live_fixture(episode)
    advance_live_fixture(hub, artifacts, executor)
    alias = 'current_clouds/latest/object_00_face.png'
    before = set(artifacts.views)
    record, _ = artifacts.read_image(alias)
    assert record['requested_path'] == alias and record['path'] != alias
    assert record['capture_is_current_physical_step'] and not record['historical_evidence_only']
    assert record['view_kind'] == 'current_measurement_context'
    assert artifacts.owners[record['view_id']] is None
    assert set(artifacts.views)-before == {record['view_id']}
    assert not artifacts.reference_reads  # Not silently resent with fixed historical references.
    view = artifacts.views[record['view_id']]
    with np.load(view['index_map_file']) as data:
        y, x = np.argwhere(data['point_index'] >= 0)[0]
    anchor = {'kind': 'cloud_point', 'view_id': record['view_id'], 'pixel_xy': [int(x), int(y)]}
    answer = submission(hub.robot.steps, anchor)
    plan, stage = prepare_actions(executor, artifacts, answer)
    from sketchik.pose_binding import compile_binding
    pose, audit = compile_binding(binding(anchor), stage['frozen_packets'][0], lambda *args: None,
        cloud_views=stage['cloud_views'][0])
    assert pose['position'][2] == pytest.approx(1., abs=1e-6)
    assert audit['carry_reference_eligible'] is False and audit['binding_scope'] == 'single_sample_context'
    answer['actions'][0].pop('reference_target_id')
    with pytest.raises(ValueError, match='required unless'):
        bind_request_fields(artifacts, answer, hub.robot.steps)


def test_latest_pointer_advances_without_changing_old_pixels_or_resurrecting_coordinates(episode):
    import hashlib
    hub, artifacts, executor = live_fixture(episode)
    advance_live_fixture(hub, artifacts, executor)
    alias = 'current_clouds/latest/object_00_face.png'
    old, _ = artifacts.read_image(alias)
    view = artifacts.views[old['view_id']]
    hashes = {name: hashlib.sha256(Path(view[name]).read_bytes()).hexdigest()
        for name in ('image_file', 'index_map_file', 'cloud_file')}
    with np.load(view['index_map_file']) as data:
        y, x = np.argwhere(data['point_index'] >= 0)[0]
    selection = binding({'kind': 'cloud_point', 'view_id': old['view_id'], 'pixel_xy': [int(x), int(y)]})
    artifacts.validate_point(selection, 'object')
    advance_live_fixture(hub, artifacts, executor)
    assert artifacts.files.image_aliases[alias] is None
    assert old['path'] in artifacts.seen_paths and alias not in artifacts.seen_paths
    for name, digest in hashes.items():
        assert hashlib.sha256(Path(view[name]).read_bytes()).hexdigest() == digest
    historical, _ = artifacts.read_image(old['path'])
    assert historical['historical_evidence_only'] and not historical['selectable']
    catalogued = next(v for v in artifacts.latest['available_sources']['cloud_views']
        if v['view_id'] == old['view_id'])
    assert not catalogued['selectable'] and catalogued['historical_evidence_only']
    assert catalogued['point_time_status']['current_snapshot'] is False
    with pytest.raises(ValueError, match='Earlier current cloud'):
        artifacts.validate_point(selection, 'object')
    current, _ = artifacts.read_image(alias)
    assert current['view_id'] != old['view_id'] and current['source_step'] == hub.robot.steps


def test_same_snapshot_reuses_files_and_static_shape_keeps_original_source_time(episode):
    hub, artifacts, executor = live_fixture(episode)
    alias = 'current_clouds/latest/object_00_face.png'
    original = artifacts.files.image_aliases[alias]
    assert artifacts.files.entries[alias]['source_steps'] == [10]
    advance_live_fixture(hub, artifacts, executor, operate=False)
    assert artifacts.files.image_aliases[alias] == original
    snapshot = artifacts.live_cloud_files.snapshot
    table = Path(snapshot['cloud_file']); modified = table.stat().st_mtime_ns
    count = len(artifacts.views)
    artifacts.current(executor, hub.run/'repeat_without_motion')
    assert artifacts.live_cloud_files.snapshot is snapshot
    assert table.stat().st_mtime_ns == modified and len(artifacts.views) == count
    assert artifacts.files.entries[alias]['source_steps'] == [10]


def test_current_static_point_survives_its_frozen_route_but_not_a_new_request(episode):
    hub, artifacts, executor = live_fixture(episode)
    advance_live_fixture(hub, artifacts, executor)
    record, _ = artifacts.read_image('current_clouds/latest/object_00_face.png')
    view = artifacts.views[record['view_id']]
    with np.load(view['index_map_file']) as data:
        y, x = np.argwhere(data['point_index'] >= 0)[0]
    selection = binding({'kind': 'cloud_point', 'view_id': record['view_id'],
        'pixel_xy': [int(x), int(y)]})
    request_step = hub.robot.steps
    hub.robot.steps += 1
    # A frozen route still references the measured static destination. This
    # does not grant a later planning request access to that old snapshot.
    assert artifacts.validate_point(selection, 'object', request_step=request_step)['source_step'] == request_step
    with pytest.raises(ValueError, match='Earlier current cloud'):
        artifacts.validate_point(selection, 'object', request_step=hub.robot.steps)
    hub.journal.mark_moved('object', hub.robot.steps)
    with pytest.raises(ValueError, match='predates actual object'):
        artifacts.validate_point(selection, 'object', request_step=request_step)


def test_held_coordinate_framing_transforms_original_points_without_redrawing(episode):
    from scipy.spatial.transform import Rotation
    from sketchik.carry_reference import initialize_reference, suspend
    hub, artifacts, executor = live_fixture(episode)
    original = np.array([[-.02, -.01, .03], [.03, .01, .11], [0., -.02, .07]])
    reference = initialize_reference({'object_id': 'object'}, original, None, 2, [4, 8, 12])
    hub.carried_surface_reference = reference
    images_before = set(artifacts.views)
    hub._fixture_state.update(position=[.2, -.3, 1.1],
        quaternion_xyzw=Rotation.from_euler('xyz', [.2, -.5, .7]).as_quat().tolist())
    target, bounds = artifacts.live_cloud_files._held_bounds()
    expected = Rotation.from_quat(hub._fixture_state['quaternion_xyzw']).apply(original)+hub._fixture_state['position']
    assert target == 'object'
    np.testing.assert_allclose(bounds, [expected.min(axis=0), expected.max(axis=0)])
    np.testing.assert_array_equal(reference['local_xyz_m'], original)
    np.testing.assert_array_equal(reference['point_ids'], [4, 8, 12])
    assert set(artifacts.views) == images_before
    suspend(reference, hub.robot.steps, 'observed slip', kind='relative_pose_changed')
    assert artifacts.live_cloud_files._held_bounds() == (None, None)


def test_rigid_held_shape_reuses_bitmap_but_selected_sample_gets_current_xyz(episode, monkeypatch):
    import hashlib
    from scipy.spatial.transform import Rotation
    from sketchik.carry_reference import initialize_reference, retained_point_binding
    from sketchik.pose_binding import compile_binding
    from sketchik.single_agent import working_cloud_record
    hub, artifacts, executor = live_fixture(episode)
    artifacts.handoff['objects'][0]['shape_motion'] = 'rigid'
    view = artifacts.views['object_face']
    image_hash = hashlib.sha256(Path(view['image_file']).read_bytes()).hexdigest()
    with np.load(view['cloud_file']) as data:
        world = data['world_xyz_m'][:25].astype(float)
    local = world-np.array([0., 0., 1.])
    hub.episode_state.data['grasp_epoch'] = 1
    hub.carried_target_id = 'object'
    hub.carried_surface_reference = initialize_reference({'object_id': 'object',
        'source_cloud_file': view['cloud_file'], 'source_step': 30}, local, None, 1,
        np.arange(25), world_xyz_m=world)
    hub._fixture_state.update(gripper_command=1., position=[.4, -.1, 1.2],
        quaternion_xyzw=Rotation.from_euler('z', .7).as_quat().tolist())
    hub.journal.mark_moved('object', 30)
    monkeypatch.setattr(artifacts, 'relocalize_object', lambda *args: pytest.fail('Normal rigid carry must not run ICP'))
    advance_live_fixture(hub, artifacts, executor)
    before = set(artifacts.views)
    record, _ = artifacts.read_image('current_clouds/latest/object_00_face.png')
    assert set(artifacts.views) == before and record['view_id'] == 'object_face'
    assert hashlib.sha256(Path(view['image_file']).read_bytes()).hexdigest() == image_hash
    status = working_cloud_record(record)['point_time_status']
    assert status['coordinate_policy'] == 'retained_point_id_transformed_by_current_eef'
    assert status['image_axes_are_historical'] and status['coordinate_estimated_at_step'] == hub.robot.steps
    anchor = {'kind': 'cloud_point', 'view_id': 'object_face', 'pixel_xy': [1, 1]}
    plan, stage = prepare_actions(executor, artifacts, submission(hub.robot.steps, anchor))
    pose, audit = compile_binding(binding(anchor), stage['frozen_packets'][0], lambda *args: None,
        cloud_views=stage['cloud_views'][0])
    expected = Rotation.from_quat(hub._fixture_state['quaternion_xyzw']).apply(local[0])+hub._fixture_state['position']
    np.testing.assert_allclose(pose['position'], expected+[0., 0., .1])
    assert audit['source_observation']['source_step'] == 10
    assert audit['coordinate_estimate']['source_step'] == hub.robot.steps
    with pytest.raises(ValueError, match='retained point IDs'):
        retained_point_binding(hub, view, 'object', world[0],
            {'cloud_point_index': 29, 'cloud_pixel_xy': [1, 1]})
    frozen_step = hub.robot.steps
    advance_live_fixture(hub, artifacts, executor)
    with pytest.raises(ValueError, match='intervening motion'):
        artifacts.validate_point(binding(anchor), 'object', request_step=frozen_step)
    hub.carried_surface_reference['validity'] = 'suspended'
    with pytest.raises(ValueError, match='Held-object pixels|predates actual object'):
        artifacts.validate_point(binding(anchor), 'object')


def test_rigid_world_recovery_runs_only_on_requested_read_and_keeps_shape_on_failed_fit(episode, monkeypatch):
    hub, artifacts, executor = live_fixture(episode)
    artifacts.handoff['objects'][0]['shape_motion'] = 'rigid'
    hub.journal.mark_moved('object', 30)
    calls = []
    monkeypatch.setattr(artifacts, 'relocalize_object', lambda *args:
        (calls.append(args) or {'status': 'unavailable', 'reason': 'ambiguous_pose_or_identity'}))
    advance_live_fixture(hub, artifacts, executor)
    assert calls == []
    alias = 'current_clouds/latest/object_00_face.png'
    record, _ = artifacts.read_image(alias)
    assert len(calls) == 1 and record['view_id'] == 'object_face'
    assert record['latest_entry']['pose_status']['status'] == 'unavailable'
    artifacts.read_image(alias)
    assert len(calls) == 1  # No repeated registration on the same evidence.
    anchor = {'kind': 'cloud_point', 'view_id': 'object_face', 'pixel_xy': [1, 1]}
    with pytest.raises(ValueError, match='predates actual object'):
        artifacts.validate_point(binding(anchor), 'object')
    measured, _ = artifacts.read_image(alias[:-4]+'_measured.png')
    assert measured['view_kind'] == 'current_measurement_context'
    assert artifacts.owners[measured['view_id']] is None


@pytest.mark.parametrize('motion', ['articulated', 'unknown'])
def test_changed_nonrigid_scope_reads_current_geometry_without_whole_object_registration(episode, monkeypatch, motion):
    hub, artifacts, executor = live_fixture(episode)
    artifacts.handoff['objects'][0]['shape_motion'] = motion
    hub.journal.mark_moved('object', 30)
    monkeypatch.setattr(artifacts, 'ensure_rigid_pose', lambda *args: pytest.fail('No whole-object rigid fit'))
    advance_live_fixture(hub, artifacts, executor)
    record, _ = artifacts.read_image('current_clouds/latest/object_00_face.png')
    assert record['view_kind'] == 'current_measurement_context'


def test_latest_alias_is_not_a_second_arbitrary_file_interface(tmp_path):
    files = PublicFiles(tmp_path/'public')
    image = tmp_path/'image.png'; Image.new('RGB', (8, 8)).save(image)
    files.publish('history/v1.png', source=image)
    files.image_alias('latest/face.png', 'history/v1.png')
    record, _ = files.read_image('latest/face.png')
    assert record['path'] == 'history/v1.png' and record['requested_path'] == 'latest/face.png'
    for target in ('../secret.png', '/tmp/secret.png', 'unpublished.png', 'latest/face.png'):
        with pytest.raises(ValueError):
            files.image_alias('latest/other.png', target)
    with pytest.raises(ValueError, match='cannot replace'):
        files.image_alias('history/v1.png', 'history/v1.png')
    files.image_alias('latest/deferred.png', metadata={'source_step': 31})
    assert next(v for v in files.list_dir('latest')['entries'] if v['path'].endswith('deferred.png'))['deferred']
    with pytest.raises(ValueError, match='deferred'):
        files.read_image('latest/deferred.png')


@pytest.mark.parametrize('invalidate_before_close', [False, True])
def test_context_contact_only_seeds_an_independently_valid_original_component(episode, invalidate_before_close):
    hub, artifacts, executor = live_fixture(episode)
    advance_live_fixture(hub, artifacts, executor)
    current, _ = artifacts.read_image('current_clouds/latest/object_00_face.png')
    context = artifacts.views[current['view_id']]
    original = artifacts.views['object_face']
    cloud = Path(original['cloud_file'])
    np.savez(cloud.parent/'extent_groups.npz', cloud_file=str(cloud), point_group_ids=np.ones(30, int))
    hub.last_contact_binding = {'target_id': 'object',
        'audit': {'anchor_world_xyz_m': [.2, .3, .9], 'binding_scope': 'single_sample_context',
            'carry_reference_eligible': False, 'binding': {'anchor': {'view_id': current['view_id']}}},
        'views': [context, original], 'geometry_binding': {},
        'world_validity_at_binding': {},
        'world_validity_before_close': {'invalidated_at_step': 20} if invalidate_before_close else {},
        'observed_source': {'observation_id': 'obs_00031', 'source_step': 31}}
    folder = hub.run/'carry_seed'; folder.mkdir()
    executor.seed_scan_carry('object', folder)
    record = json.loads((folder/'scan_carry_reference.json').read_text())
    if invalidate_before_close:
        assert record['status'] == 'unavailable' and hub.carried_surface_reference is None
        assert 'no unique independently bound' in record['reason']
    else:
        assert record['status'] == 'predicted_rigid'
        assert record['independent_template_evidence']['context_roi_used_as_template'] is False
        assert record['independent_template_evidence']['source_cloud_file'] == str(cloud)
        reference = hub.carried_surface_reference
        assert len(reference['local_xyz_m']) == 30  # Original template, not the 2048-point context.
        np.testing.assert_array_equal(reference['point_ids'], np.arange(30))
        assert reference['rigid_attachment_verified'] is False


def test_context_without_an_independent_template_never_becomes_whole_carried_region(episode):
    hub, artifacts, executor = live_fixture(episode)
    advance_live_fixture(hub, artifacts, executor)
    current, _ = artifacts.read_image('current_clouds/latest/object_00_face.png')
    context = artifacts.views[current['view_id']]
    # Even a neighbouring extent_groups file must not make a context eligible.
    cloud = Path(context['cloud_file'])
    np.savez(cloud.parent/'extent_groups.npz', cloud_file=str(cloud), point_group_ids=np.ones(context['point_count'], int))
    hub.last_contact_binding = {'target_id': 'object',
        'audit': {'anchor_world_xyz_m': [.2, .3, .9], 'binding_scope': 'single_sample_context',
            'carry_reference_eligible': False, 'binding': {'anchor': {'view_id': current['view_id']}}},
        'views': [context], 'geometry_binding': {},
        'observed_source': {'observation_id': 'obs_00031', 'source_step': 31}}
    folder = hub.run/'carry_seed'; folder.mkdir()
    executor.seed_scan_carry('object', folder)
    record = json.loads((folder/'scan_carry_reference.json').read_text())
    assert record['status'] == 'unavailable' and hub.carried_surface_reference is None
    assert 'no unique independently bound' in record['reason']


def test_relocalized_shape_can_be_regrasped_without_replacing_original_samples(episode, monkeypatch):
    from scipy.spatial.transform import Rotation
    from sketchik.carry_reference import retained_point_binding
    hub, handoff, actions, view = episode
    handoff['objects'][0]['shape_motion'] = 'rigid'
    artifacts = EpisodeArtifacts(hub, handoff); artifacts.prepare()
    executor = DirectExecutor(hub, handoff, artifacts)
    artifacts.read_image(artifacts.image_paths['object_face'])
    with np.load(view['cloud_file']) as data:
        original = data['world_xyz_m'][:25].astype(float)
    rotation = Rotation.from_euler('z', .4).as_matrix()
    translation = np.array([-.2, -.3, .1])
    world = original@rotation.T+translation
    members = hub.run/'registered_members.npz'
    np.savez(members, original_point_ids=np.arange(25))
    artifacts.rigid_view_bindings['object_face'] = {
        'owner': 'object', 'pose_id': 'independent_pose', 'source_step': 30,
        'original_cloud_file': str(Path(view['cloud_file']).resolve()),
        'members_file': str(members),
        'transform': {'rotation': rotation.tolist(), 'translation_m': translation.tolist()}}
    hub.journal.mark_moved('object', 20)
    monkeypatch.setattr('sketchik.observed_clearance.visible_reference_support',
        lambda *args: {'status': 'supported', 'supported_points': 25})
    anchor = {'kind': 'cloud_point', 'view_id': 'object_face', 'pixel_xy': [1, 1]}
    answer = submission(30, anchor)
    answer['actions'].append({'primitive': 'set_gripper', 'target_id': 'object',
        'intent': 'close at relocated contact', 'selection': {'command': 'close', 'steps': 1}})
    plan, stage = prepare_actions(executor, artifacts, answer)
    apply_assessment(hub, answer)
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    assert receipt['commands_completed'] and actions == ['move', 'set_gripper']
    reference = hub.carried_surface_reference
    assert reference['source_kind'] == 'registered_rigid_shape'
    np.testing.assert_array_equal(reference['point_ids'], np.arange(25))
    np.testing.assert_array_equal(reference['original_world_xyz_m'], original)
    np.testing.assert_allclose(reference['local_xyz_m'], world-hub.robot.state()['position'])
    # A later EEF motion changes coordinates, never pixel identity or the original samples.
    hub._fixture_state['position'][0] += .08
    mapped = retained_point_binding(hub, view, 'object', original[0],
        {'cloud_point_index': 0, 'cloud_pixel_xy': [1, 1]})
    np.testing.assert_allclose(mapped['world_xyz_m'], world[0]+[.08, 0., 0.])
    evidence = hub.last_contact_binding
    with pytest.raises(ValueError, match='invalid before'):
        executor._registered_contact_template({**evidence,
            'world_validity_before_close': {'invalidated_at_step': 31}}, world[0])
    with pytest.raises(ValueError, match='exact registered'):
        executor._registered_contact_template(evidence, world[0]+[.01, 0., 0.])
    monkeypatch.setattr('sketchik.observed_clearance.visible_reference_support',
        lambda *args: {'status': 'contradicted'})
    with pytest.raises(ValueError, match='does not support'):
        executor._registered_contact_template(evidence, world[0])


def submission(step, anchor=None):
    return {'source_step': step, 'intent': 'move from actual state', 'visible_evidence': 'fixture',
        'held_object_id': None, 'held_pose_relation': 'none', 'actions': [{
            'primitive': 'move', 'target_id': 'object', 'reference_target_id': 'object',
        'intent': 'raise', 'selection': {'waypoints': [binding(anchor)]}}]}


@pytest.mark.parametrize('failed,stale', [(False, False), (True, False), (False, True)])
def test_async_same_A_drafts_during_motion_and_normal_outcome_continues_directly(episode, monkeypatch, failed, stale):
    from threading import Event, get_ident
    from sketchik.stage_context import closed_tool_dialogue
    hub, handoff, actions, _ = episode
    # Even unknown/articulated semantic shape no longer adds an approval turn.
    handoff['objects'][0]['shape_motion'] = 'articulated'
    owner = get_ident(); entered = Event(); motion = Event()
    original_move = hub.executor.move
    def move(**kwargs):
        if not actions:
            assert entered.wait(5), 'draft must run during execution'
            motion.set()
        return original_move(**kwargs)
    hub.executor.move = move
    original_execute = execute_sketch
    if failed:
        def fail(*args, **kwargs):
            receipt = original_execute(*args, **kwargs)
            return {**receipt, 'commands_completed': False, 'failure': 'actual segment failure'}
        monkeypatch.setattr('sketchik.single_agent.execute_sketch', fail)
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, value = 'execute_actions', submission(hub.robot.steps)
            elif self.calls == 2:
                assert get_ident() != owner and [t['function']['name'] for t in tools] == ['draft_next_sketch']
                closed_tool_dialogue(messages)
                entered.set(); assert motion.wait(5)
                name, value = 'draft_next_sketch', {'status': 'ready', 'reason': 'existing geometry',
                    'preconditions': 'current motion ends normally', 'intent': 'next subtask',
                    'actions': submission(30)['actions']}
                if stale:
                    value['actions'][0]['selection']['waypoints'][0]['anchor'] = {
                        'kind': 'cloud_point', 'view_id': 'never_read', 'pixel_xy': [0, 0]}
            else:
                assert self.calls == 3 and get_ident() == owner
                assert {t['function']['name'] for t in tools} == {t['function']['name'] for t in tools_for({'object'})}
                assert actions == (['move', 'move'] if not failed and not stale else ['move'])
                name, value = 'finish', {'status': 'blocked', 'summary': 'fixture only',
                    'goal_evidence': [{'goal': handoff['task'], 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'], 'visible_evidence': 'fixture'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(value)}}]}}
    client = Client(); messages = []
    result = run_single_agent(hub, client, messages, handoff, max_turns=6, max_batches=2, async_planning=True)
    closed_tool_dialogue(messages)
    assert result['status'] == 'single_agent_blocked' and client.calls == 3
    assert result['a_execution_requests'] == 3 and result['async_planning']['requests'][0]['overlap_wall_s'] > 0
    assert len(result['subtasks']) == (2 if not failed and not stale else 1)
    assert result['binding_or_file_invalid_replies'] == 0  # Program preflight is not an invalid A reply.
    if stale:
        assert result['async_planning']['continuation_checks'][0]['status'] == 'fail'
    if failed:
        assert 'discarded' in result['async_planning']['requests'][0]


def test_async_does_not_spend_last_request_or_last_sketch_on_speculation(episode):
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            assert not any(t['function']['name'] == 'draft_next_sketch' for t in tools)
            if self.calls == 1:
                value, name = submission(hub.robot.steps), 'execute_actions'
            else:
                value, name = {'status': 'blocked', 'summary': 'fixture', 'goal_evidence': [
                    {'goal': handoff['task'], 'assessment': 'unknown', 'image_paths': ['observations/obs_00031/agentview.png'],
                        'visible_evidence': 'fixture'}]}, 'finish'
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(value)}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=2, max_batches=2, async_planning=True)
    assert result['async_planning']['requests'] == [] and actions == ['move']


@pytest.mark.parametrize('fault', ['contact', 'depth'])
def test_async_actual_fault_returns_same_A_to_normal_tools_without_dispatch(episode, monkeypatch, fault):
    from sketchik.stage_context import StageContext, closed_tool_dialogue
    hub, handoff, actions, _ = episode
    original_current = EpisodeArtifacts.current
    def current(artifacts, *args, **kwargs):
        value = original_current(artifacts, *args, **kwargs)
        if getattr(hub, 'single_a_batches', 0) == 1:
            if fault == 'contact': value['contact_warning'] = {'reason': 'actual contact changed'}
            else: value['geometry']['carried_observed_geometry'] = {'visible_support': {'status': 'contradicted'}}
        return value
    monkeypatch.setattr(EpisodeArtifacts, 'current', current)
    class Client:
        calls = 0
        last_request_metadata = {}
        stage_context = StageContext(max_chars=2000, keep_messages=1, keep_chars=1000, retain_episode_references=True)
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, value = 'execute_actions', submission(hub.robot.steps)
            elif self.calls == 2:
                name, value = 'draft_next_sketch', {'status': 'ready', 'reason': 'existing geometry',
                    'preconditions': 'normal motion', 'intent': 'next', 'actions': submission(30)['actions']}
            else:
                assert self.calls == 3 and actions == ['move'] and len(tools) == 7
                active, _ = self.stage_context.apply(messages); closed_tool_dialogue(active)
                assert 'draft_next_sketch' in json.dumps(active)  # Original inert intent retained for recovery.
                name, value = 'finish', {'status': 'blocked', 'summary': 'fixture only',
                    'goal_evidence': [{'goal': handoff['task'], 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'], 'visible_evidence': 'Actual fault'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(value)}}]}}
    source = []; client = Client()
    result = run_single_agent(hub, client, source, handoff, max_turns=6, max_batches=2, async_planning=True)
    assert result['status'] == 'single_agent_blocked' and actions == ['move']
    assert result['async_planning']['program_continuations'] == 0
    assert result['async_planning']['continuation_checks'][0]['status'] == 'fail'
    closed_tool_dialogue(source)


def test_saved_cloud_requires_actual_view_and_observed_time_not_new_request_time(episode):
    hub, handoff, _, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    path = artifacts.publish_view(view, 'object', 'geometry')
    anchor = {'kind': 'cloud_point', 'view_id': view['view_id'], 'pixel_xy': [1, 1]}
    with pytest.raises(ValueError, match='Read the exact'):
        artifacts.validate_point(binding(anchor), 'object')
    artifacts.read_image(path)
    artifacts.validate_point(binding(anchor), 'object')
    hub.journal.mark_moved('object', 20)
    # A new request at step30 cannot revive a step10 pixel.
    with pytest.raises(ValueError, match='predates'):
        artifacts.validate_point(binding(anchor), 'object')
    artifacts.validate_point(binding(), 'object')
    with pytest.raises(ValueError, match='owner'):
        artifacts.validate_point(binding(anchor), 'other')


def test_original_cloud_and_exact_maps_remain_referencable_without_reviving_old_position(episode):
    hub, handoff, _, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    image_path = artifacts.publish_view(view, 'object', 'geometry')
    second = {**view, 'view_id': 'object_other_angle'}
    artifacts.publish_view(second, 'object', 'geometry')
    metadata = json.loads(artifacts.files.read_text_file('geometry/object_face.json')['content'])
    cloud_path, map_path = metadata['point_cloud_path'], metadata['point_index_path']
    assert (artifacts.files.root/cloud_path).read_bytes() == Path(view['cloud_file']).read_bytes()
    assert (artifacts.files.root/map_path).read_bytes() == Path(view['index_map_file']).read_bytes()
    assert len([item for item in artifacts.files.entries.values() if item.get('kind') == 'point_cloud_data']) == 1
    assert metadata['source_observations'][0]['source_step'] == 10
    assert 'cloud_file' not in metadata  # No arbitrary private path access.
    assert not artifacts.seen_views  # Publishing data does not mean the model viewed it.
    with pytest.raises(ValueError, match='binary geometry'):
        artifacts.files.read_text_file(cloud_path)
    executor = DirectExecutor(hub, handoff, artifacts)
    artifacts.current(executor, hub.run/'current')
    record, _ = artifacts.read_image('agent_public/'+image_path)
    assert record['path'] == image_path and artifacts.seen_paths == {image_path}
    current = json.loads(artifacts.files.read_text_file('current.json')['content'])
    assert current['available_sources']['cloud_views'][0]['point_cloud_path'] == cloud_path
    hub.journal.mark_moved('object', 20)
    # Files stay readable as historical geometry; motion validation is unchanged.
    assert artifacts.read_image('agent_public/'+image_path)[0]['source_observations'][0]['source_step'] == 10
    anchor = {'kind': 'cloud_point', 'view_id': view['view_id'], 'pixel_xy': [1, 1]}
    with pytest.raises(ValueError, match='predates'):
        artifacts.validate_point(binding(anchor), 'object')


def test_one_A_parameter_submission_executes_shared_core_without_a_B_phase(episode):
    hub, handoff, actions, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    assert not hasattr(executor, 'bind_stage') and not hasattr(executor, 'repair_step')
    answer = submission(30)
    plan, stage = prepare_actions(executor, artifacts, answer)
    apply_assessment(hub, answer)
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    assert actions == ['move'] and receipt['commands_completed']
    assert not hasattr(executor, 'client') and hub.robot.state()['position'] == [0., 0., 1.1]
    log = json.loads((hub.run/'execution/subtask_00/step_00/local_compile.compiled.json').read_text())
    assert log['binding']['anchor']['kind'] == 'current_eef'


@pytest.mark.parametrize('grasp_target,expected_move', [(None, True), ('object', False)])
def test_gripper_configuration_without_object_keeps_static_point_but_real_close_retires_it(
        episode, grasp_target, expected_move):
    hub, handoff, actions, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    artifacts.read_image(artifacts.publish_view(view, 'object', 'geometry'))
    executor = DirectExecutor(hub, handoff, artifacts)
    answer = submission(30, {'kind': 'cloud_point', 'view_id': view['view_id'],
                             'pixel_xy': [1, 1]})
    answer['actions'].insert(0, {'primitive': 'set_gripper', 'target_id': grasp_target,
        'intent': 'change finger configuration', 'selection': {'command': 'close', 'steps': 1}})
    Draft202012Validator(tools_for({'object'})[0]['function']['parameters']).validate(answer)
    plan, stage = prepare_actions(executor, artifacts, answer)
    apply_assessment(hub, answer)
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    assert receipt['commands_completed'] is expected_move
    assert actions == (['set_gripper', 'move'] if expected_move else ['set_gripper'])
    assert None not in hub.journal.data['objects']
    assert ('null' not in hub.journal.data['objects'])
    invalidated = hub.journal.data['objects'].get('object', {}).get('invalidated_at_step')
    assert invalidated == (None if expected_move else 31)


def test_no_target_gripper_event_still_retires_existing_carried_location(episode):
    hub, handoff, actions, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    view = {**view, 'sources': [{**view['sources'][0], 'source_step': 30}]}
    artifacts.read_image(artifacts.publish_view(view, 'object', 'geometry'))
    executor = DirectExecutor(hub, handoff, artifacts)
    answer = submission(30, {'kind': 'cloud_point', 'view_id': view['view_id'],
                             'pixel_xy': [1, 1]})
    hub.carried_target_id = 'object'
    answer.update(held_object_id='object', held_pose_relation='unchanged')
    answer['actions'].insert(0, {'primitive': 'set_gripper', 'target_id': None,
        'intent': 'change configuration without naming a new grasp target',
        'selection': {'command': 'close', 'steps': 1}})
    seeded = []
    executor.seed_scan_carry = lambda *args: seeded.append(args)
    plan, stage = prepare_actions(executor, artifacts, answer)
    apply_assessment(hub, answer)
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    assert not receipt['commands_completed']
    assert actions == ['set_gripper'] and not seeded
    assert hub.carried_surface_reference is None
    assert hub.journal.data['objects']['object']['invalidated_at_step'] == 31


def test_null_gripper_target_does_not_allow_null_movement_target():
    answer = submission(30)
    answer['actions'][0]['target_id'] = None
    with pytest.raises(ValidationError):
        Draft202012Validator(tools_for({'object'})[0]['function']['parameters']).validate(answer)


@pytest.mark.parametrize('command', ['open', 'close'])
def test_no_target_event_retires_reference_only_carried_dimensions_before_a_claim(episode, command):
    from sketchik.observed_dimensions import write_measurement
    hub, handoff, actions, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    # A same-sketch close/lift can establish a point reference before a new A reply.
    assert hub.carried_target_id is None
    hub.carried_surface_reference = {'object_id': 'object', 'source_step': 30,
        'source_kind': 'scan_cloud', 'local_xyz_m': np.zeros((30, 3))}
    write_measurement(hub.run, 'object', {'dimensions_frame': 'eef_tool',
        'dimensions_kind': 'visible_surface_extent_estimate', 'dimensions_valid': True,
        'object_dimensions_m': [.05, .04, .1], 'dimension_source_step': 30})
    answer = submission(30)
    answer['actions'] = [{'primitive': 'set_gripper', 'target_id': None,
        'intent': 'change configuration without a new object target',
        'selection': {'command': command, 'steps': 1}}]
    plan, stage = prepare_actions(executor, artifacts, answer)
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    assert receipt['commands_completed'] and actions == ['set_gripper']
    assert hub.carried_surface_reference is None
    assert hub.journal.data['objects']['object']['invalidated_at_step'] == 31
    dimensions = json.loads((hub.run/'dimensions.json').read_text())['objects']['object']
    assert dimensions['dimensions_valid'] is False
    assert dimensions['object_dimensions_m'] is None
    assert dimensions['dimension_invalidated_at_step'] == 31


def test_release_then_old_point_stops_suffix_but_dynamic_eef_can_withdraw(episode):
    hub, handoff, actions, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    view = {**view, 'sources': [{**view['sources'][0], 'source_step': hub.robot.steps}]}
    artifacts.read_image(artifacts.publish_view(view, 'object', 'geometry'))
    executor = DirectExecutor(hub, handoff, artifacts)
    hub.carried_target_id = 'object'
    answer = submission(30, {'kind': 'cloud_point', 'view_id': view['view_id'], 'pixel_xy': [1, 1]})
    answer['held_object_id'] = 'object'; answer['held_pose_relation'] = 'unchanged'
    answer['actions'].insert(0, {'primitive': 'set_gripper', 'target_id': 'object',
        'intent': 'open fixture', 'selection': {'command': 'open', 'steps': 1}})
    plan, stage = prepare_actions(executor, artifacts, answer)
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    assert actions == ['set_gripper'] and receipt['commands_completed'] is False
    plan, stage = prepare_actions(executor, artifacts, submission(hub.robot.steps))
    assert execute_sketch(hub, executor, plan, 1, frozen_binding=stage)['commands_completed']
    assert actions == ['set_gripper', 'move']


@pytest.mark.parametrize('status', ['contradicted', 'unknown', 'contact_geometry_supported'])
def test_first_close_contact_result_controls_suffix_without_replaying_gripper(episode, status):
    hub, handoff, actions, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    checked = []
    def check(target_id, before, after, folder):
        checked.append((target_id, before['source_step'], after['source_step']))
        # The real event and epoch must exist before any contact interpretation.
        assert hub.episode_state.data['last_gripper_event']['command'] == 'close'
        assert hub.episode_state.data['grasp_epoch'] == 1
        result = {'status': status, 'reason': 'frozen local contact evidence',
            'source_step': after['source_step'], 'active': True, 'grasp_epoch': 1,
            'holding_certified': False, 'target_identity_certified': False}
        hub.episode_state.data['initial_grasp_contact'] = result
        return result
    executor.check_initial_grasp_contact = check
    answer = submission(30)
    answer['actions'].insert(0, {'primitive': 'set_gripper', 'target_id': 'object',
        'intent': 'grasp before lifting', 'selection': {'command': 'close', 'steps': 1}})
    plan, stage = prepare_actions(executor, artifacts, answer)
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    assert checked == [('object', 30, 31)]
    assert receipt['steps'][0]['completed'] is True  # Close really happened.
    assert receipt['steps'][0]['initial_grasp_contact']['status'] == status
    if status == 'contradicted':
        assert actions == ['set_gripper'] and receipt['commands_completed'] is False
        assert receipt['steps'][0]['recovery_failure']['failure_stage'] == 'grasp_contact'
        assert receipt['recovery'][-1]['tier'] == 'T2'
        assert receipt['remaining_suffix_executed'] is False
        current = artifacts.current(executor, hub.run/'contact_failure_current')
        assert current['initial_grasp_contact']['status'] == 'contradicted'
        # A fresh recovery can retreat from the actual stop; the old close is not replayed.
        repair, fresh = prepare_actions(executor, artifacts, submission(hub.robot.steps))
        assert execute_sketch(hub, executor, repair, 1, frozen_binding=fresh)['commands_completed']
        assert actions == ['set_gripper', 'move']
    else:
        assert receipt['commands_completed'] and actions == ['set_gripper', 'move']


def test_configuration_close_skips_grasp_check_and_retires_old_contact_evidence(episode):
    hub, handoff, actions, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    hub.episode_state.data['initial_grasp_contact'] = {
        'status': 'contact_geometry_supported', 'source_step': 29, 'active': True}
    executor.check_initial_grasp_contact = lambda *args: pytest.fail('No intended grasp target')
    answer = submission(30)
    answer['actions'].insert(0, {'primitive': 'set_gripper', 'target_id': None,
        'intent': 'configure fingers for pushing', 'selection': {'command': 'close', 'steps': 1}})
    plan, stage = prepare_actions(executor, artifacts, answer)
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    assert receipt['commands_completed'] and actions == ['set_gripper', 'move']
    assert hub.episode_state.data['initial_grasp_contact']['active'] is False
    assert hub.episode_state.data['initial_grasp_contact']['retired_at_step'] == 31


def test_single_review_resends_actual_close_and_current_pairs_across_context_epoch(episode):
    from sketchik.stage_context import StageContext, ACTUAL_IMAGES
    hub,handoff,actions,_ = episode
    artifacts = EpisodeArtifacts(hub,handoff);executor = DirectExecutor(hub,handoff,artifacts)
    artifacts.current(executor,hub.run/'before_close')
    answer = submission(hub.robot.steps)
    answer['actions'].insert(0,{'primitive':'set_gripper','target_id':'object','intent':'close fixture',
        'selection':{'command':'close','steps':1}})
    plan,stage=prepare_actions(executor,artifacts,answer);apply_assessment(hub,answer)
    receipt=execute_sketch(hub,executor,plan,0,frozen_binding=stage)
    artifacts.receipts.append(receipt)
    current=artifacts.current(executor,hub.run/'after_lift')
    comparison=current['held_pose_comparison']
    assert comparison['reference_basis']=='actual_close_event_not_certified_holding'
    assert comparison['reference_source_step']==31 and comparison['current_source_step']==32
    assert comparison['reference_image_count']==2 and not comparison['holding_or_rigid_attachment_certified']
    assert actions==['set_gripper','move'] and not hasattr(executor, 'client')
    previous_reads=list(artifacts.reference_reads)
    messages=[{'role':'system','content':'same agent'}, {'role':'user','content':'original fixture task'},
        {'role':'assistant','content':'x'*10000}, *artifacts.feedback_messages()]
    context=StageContext(max_chars=1000,keep_messages=2,keep_chars=500,retain_robot_facts=False)
    context.prepare_boundary(messages,{'source_step':32},hub.run/'comparison_context',lambda event:None)
    active,metadata=context.apply(messages)
    assert metadata['epoch']==1
    image_message=next(msg for msg in active if isinstance(msg.get('content'),list)
        and json.loads(msg['content'][0].get('text','{}')).get('phase')==ACTUAL_IMAGES)
    records=[json.loads(part['text']) for part in image_message['content'][1:] if part['type']=='text']
    assert [row['source_step'] for row in records]==[31,31,32,32]
    assert all('PREVIOUS actual' in row['comparison_role'] and row['historical_evidence_only'] for row in records[:2])
    assert all('CURRENT actual' in row['comparison_role'] and row['capture_is_current_physical_step'] for row in records[2:])
    assert sum(part['type']=='image_url' for part in image_message['content'])==4
    assert artifacts.reference_reads==previous_reads
    # Reading state again without any physical motion keeps the comparison and does not create a new contact.
    again=artifacts.current(executor,hub.run/'same_pose')
    assert again['held_pose_comparison']['reference_source_step']==31
    opened=deepcopy(receipt);opened['steps'].append({'primitive':'set_gripper','parameters':{'command':'open'},
        'completed':True,'source_step':32})
    artifacts.receipts[-1]=opened
    retired=artifacts.current(executor,hub.run/'after_open')
    assert retired['held_pose_comparison']['reference_image_count']==0
    assert retired['held_pose_comparison']['image_paths']==[]


def test_new_held_claim_preserves_matching_scan_reference(episode):
    hub, handoff, _, _ = episode
    DirectExecutor(hub, handoff, EpisodeArtifacts(hub, handoff))
    reference = {'object_id': 'object', 'local_xyz_m': np.zeros((30, 3))}
    hub.carried_surface_reference = reference
    answer = submission(30); answer.update(held_object_id='object', held_pose_relation='unchanged')
    apply_assessment(hub, answer)
    assert hub.carried_surface_reference is reference
    answer['held_pose_relation'] = 'changed'
    apply_assessment(hub, answer)
    assert hub.carried_surface_reference is reference
    assert reference['validity']=='suspended'
    assert reference['suspension_kind']=='relative_pose_changed'
    # Subsequent unchanged claims do not erase the observed slip or choose
    # another nearby fragment as the same material.
    answer['held_pose_relation']='unchanged'
    apply_assessment(hub, answer)
    from sketchik.carry_reference import can_test
    assert not can_test(reference)
    np.testing.assert_array_equal(reference['local_xyz_m'],np.zeros((30,3)))


@pytest.mark.parametrize('prefix', ['', 'agent_public/'])
def test_persistent_A_can_read_directory_then_execute_then_honestly_stop(episode, prefix):
    from sketchik.stage_context import StageContext
    hub, handoff, actions, _ = episode
    earlier_decision = {'role': 'assistant', 'content': 'Visible fixture object requires a new angle',
        'reasoning_content': 'Retained earlier decision from this same episode'}
    class Client:
        last_request_metadata = {}
        calls = 0
        stage_context = StageContext(keep_messages=40, keep_chars=60_000, retain_robot_facts=False)
        def complete(self, messages, tools):
            self.calls += 1
            active, audit = self.stage_context.apply(messages)
            assert earlier_decision in active and audit['epoch'] == 0
            names = {tool['function']['name'] for tool in tools}
            assert 'execute_actions' in names and 'move' not in names and 'submit_actual_stage_result' not in names
            if self.calls == 1:
                name, arguments = 'list_dir', {'path': prefix+'geometry'}
            elif self.calls == 2:
                assert any(message.get('role') == 'tool' and 'entries' in message.get('content', '') for message in messages)
                name, arguments = 'execute_actions', submission(hub.robot.steps)
            else:
                assert actions == ['move']
                name, arguments = 'finish', {'status': 'blocked', 'summary': 'fixture has no placement evidence',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': [prefix+f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'no real task evidence in unit fixture'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}}
    client = Client(); messages = [earlier_decision]
    result = run_single_agent(hub, client, messages, handoff, max_turns=6, max_batches=2)
    assert result['status'] == 'single_agent_blocked'
    assert client.calls == 3 and actions == ['move']
    assert (hub.run/'agent_public/current.json').exists()
    assert not (hub.run/'b_transport').exists()


def test_one_sketch_batch_runs_multiple_physical_actions_before_asking_A_again(episode):
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, arguments = 'execute_actions', submission(hub.robot.steps)
                arguments['actions'].append(deepcopy(arguments['actions'][0]))
                Draft202012Validator(next(tool['function']['parameters'] for tool in tools
                    if tool['function']['name'] == name)).validate(arguments)
            else:
                assert actions == ['move', 'move']
                name, arguments = 'finish', {'status': 'blocked', 'summary': 'fixture has no task proof',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'fixture only'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}}
    client = Client()
    result = run_single_agent(hub, client, [], handoff, max_turns=4, max_batches=2)
    assert client.calls == 2 and actions == ['move', 'move']
    assert len(result['subtasks']) == 1
    assert len(result['subtasks'][0]['steps']) == 2


def test_program_error_after_dispatch_is_fatal_not_a_format_correction(episode, monkeypatch):
    hub, handoff, _, _ = episode
    calls = []
    class Client:
        last_request_metadata = {}
        def complete(self, messages, tools):
            calls.append(1)
            return {'message': {'role': 'assistant', 'content': json.dumps({'name': 'execute_actions',
                'arguments': submission(hub.robot.steps)})}}
    def broken(*args, **kwargs):
        hub.robot.steps += 1
        raise ValueError('program error after physics')
    monkeypatch.setattr('sketchik.single_agent.execute_sketch', broken)
    with pytest.raises(ValueError, match='program error'):
        run_single_agent(hub, Client(), [], handoff, max_turns=6, max_batches=2)
    assert calls == [1]






def test_truncated_single_A_reply_has_no_action_or_automatic_request_retry(episode):
    from sketchik.agent import RejectedResponse
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        def complete(self, messages, tools):
            self.calls += 1
            raise RejectedResponse('length')
    client = Client()
    with pytest.raises(RejectedResponse):
        run_single_agent(hub, client, [], handoff, max_turns=6, max_batches=2)
    assert client.calls == hub.single_a_requests == 1
    assert actions == [] and not (hub.run/'b_transport').exists()


def test_single_schema_feedback_names_field_paths_and_preserves_raw_reply(episode):
    hub, handoff, actions, _ = episode
    bad = submission(hub.robot.steps)
    bad.update(primitive='move', target_id='object', reference_target_id='object',
        selection={'waypoints':[binding()]})
    bad['actions'] = [binding(), binding(), binding()]
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, args = 'execute_actions', bad
            elif self.calls == 2:
                assert actions == []
                reply = json.loads(messages[-1]['content'])
                assert reply['robot_actions'] == 0 and not reply['executed']
                assert 'actions/0/primitive: required field is missing' in reply['error']
                for field in ('primitive','target_id','reference_target_id','selection'):
                    assert '$/'+field+': field is not allowed here' in reply['error']
                assert 'Failed validating' not in reply['error'] and 'properties' not in reply['error']
                name, args = 'execute_actions', submission(hub.robot.steps)
            else:
                name, args = 'finish', {'status': 'blocked', 'summary': 'fixture lacks task evidence',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'fixture only'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}}
    client = Client()
    result = run_single_agent(hub, client, [], handoff, max_turns=4, max_batches=2)
    assert client.calls == 3 and actions == ['move']
    archived = json.loads((hub.run/'single_a/turn_001/response.json').read_text())
    assert json.loads(archived['message']['tool_calls'][0]['function']['arguments']) == bad


def test_single_schema_feedback_allows_three_corrections_then_stops_after_four_invalid_replies(episode):
    hub, handoff, actions, _ = episode
    bad = submission(hub.robot.steps); del bad['visible_evidence']
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 2:
                assert json.loads(messages[-1]['content'])['error'] == '$/visible_evidence: required field is missing'
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': 'execute_actions', 'arguments': json.dumps(bad)}}]}}
    client = Client()
    result = run_single_agent(hub, client, [], handoff, max_turns=6, max_batches=2)
    assert result['status'] == 'single_agent_invalid_output' and client.calls == 4
    assert result['tool_format_invalid_replies'] == 4
    assert result['action_batches_accepted'] == 0
    assert actions == []


@pytest.mark.parametrize('malformed_root', [False, True])
def test_duplicate_calls_report_json_error_in_same_correction_without_repair(episode, malformed_root):
    hub, handoff, actions, _ = episode
    bad = submission(hub.robot.steps)
    # Reproduce a complete XML parameter whose JSON array has an extra closing brace.
    bad['actions'] = json.dumps(bad['actions'])[:-1]+'}]'
    raw = '{"actions": [}' if malformed_root else json.dumps(bad)
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return {'message': {'role': 'assistant', 'tool_calls': [
                    {'id': str(index), 'type': 'function', 'function': {
                        'name': 'execute_actions', 'arguments': raw}} for index in range(2)]}}
            if self.calls == 2:
                assert actions == [] and hub.robot.steps == 30
                feedback = [json.loads(message['content']) for message in messages[-2:]]
                for reply in feedback:
                    assert reply['robot_actions'] == 0 and not reply['executed']
                    assert 'exactly one execute_actions' in reply['error']
                    assert ('arguments: invalid JSON' if malformed_root else 'actions: expected a JSON array') in reply['error']
                    assert 'column ' in reply['error'] and 'character ' in reply['error']
                    assert 'Check brackets and commas' in reply['error']
                name, args = 'execute_actions', submission(hub.robot.steps)
            else:
                name, args = 'finish', {'status': 'blocked', 'summary': 'fixture lacks task evidence',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'fixture only'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls),
                'function': {'name': name, 'arguments': json.dumps(args)}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=4, max_batches=2)
    assert result['status'] == 'single_agent_blocked' and actions == ['move']
    archived = json.loads((hub.run/'single_a/turn_001/response.json').read_text())
    assert [call['function']['arguments'] for call in archived['message']['tool_calls']] == [raw, raw]


def test_identical_valid_sketch_calls_are_not_automatically_deduplicated_or_executed(episode):
    hub, handoff, actions, _ = episode
    raw = json.dumps(submission(hub.robot.steps))
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            assert actions == [] and hub.robot.steps == 30
            return {'message': {'role': 'assistant', 'tool_calls': [
                {'id': str(index), 'function': {'name': 'execute_actions', 'arguments': raw}}
                for index in range(2)]}}
    client = Client()
    result = run_single_agent(hub, client, [], handoff, max_turns=4, max_batches=2)
    assert result['status'] == 'single_agent_invalid_output' and client.calls == 4
    assert actions == [] and hub.robot.steps == 30


def test_three_format_errors_can_be_corrected_without_consuming_an_action_batch(episode):
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls <= 3:
                name, args = 'execute_actions', submission(hub.robot.steps)
                del args['visible_evidence']
            elif self.calls == 4:
                assert hub.robot.steps == 30 and actions == []
                error = json.loads(messages[-1]['content'])
                assert error['error_kind'] == 'tool_format' and not error['physical_execution_failed']
                assert error['execution_budgets']['used']['action_batches'] == 0
                name, args = 'execute_actions', submission(hub.robot.steps)
            else:
                name, args = 'finish', {'status': 'blocked', 'summary': 'fixture lacks task evidence',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'fixture only'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls),
                'function': {'name': name, 'arguments': json.dumps(args)}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=6, max_batches=1)
    assert result['status'] == 'single_agent_blocked' and actions == ['move']
    assert result['tool_format_invalid_replies'] == 3 and result['action_batches_accepted'] == 1
    assert result['execution_budgets']['used']['consecutive_invalid_replies'] == 0


def test_explicit_api_resume_continues_same_A_after_completed_prefix_without_replay(episode):
    from io import BytesIO
    import threading
    import time
    from sketchik.agent import ChatClient
    from sketchik.request_recovery import RequestRecovery, submit_command
    hub, handoff, actions, _ = episode
    client = ChatClient('http://127.0.0.1:1/v1', 'fixture', 'fixture-secret')
    class Opener:
        calls = 0
        pending = None
        def open(self, request, timeout):
            self.calls += 1
            if self.calls == 1:
                name, args = 'execute_actions', submission(hub.robot.steps)
            elif self.calls == 2:
                assert actions == ['move']
                self.pending = request.data
                raise OSError('fixture connection broke')
            else:
                assert self.calls == 3 and self.pending == request.data
                assert actions == ['move']
                name, args = 'finish', {'status': 'blocked', 'summary': 'fixture lacks task evidence',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'fixture only'}]}
            return BytesIO(json.dumps({'choices': [{'finish_reason': 'tool_calls',
                'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls),
                    'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}).encode())
    opener = client.opener = Opener()
    recovery = RequestRecovery(hub.run, client, hub.robot.state, hub.emit, poll_s=.01)
    client.request_pause_handler = recovery.wait_for_resume
    operator_errors = []
    def resume():
        try:
            deadline = time.monotonic()+8
            while not recovery.state_path.exists():
                if time.monotonic() >= deadline: raise AssertionError('No pause checkpoint')
                time.sleep(.01)
            assert actions == ['move'] and opener.calls == 2
            submit_command(hub.run, 'resume')
        except BaseException as exc:
            operator_errors.append(exc)
    operator = threading.Thread(target=resume, daemon=True); operator.start()
    result = run_single_agent(hub, client, [], handoff, max_turns=4, max_batches=1)
    operator.join(5)
    assert not operator_errors and not operator.is_alive()
    assert result['status'] == 'single_agent_blocked' and actions == ['move']
    assert result['a_execution_requests'] == 2 and opener.calls == 3
    assert result['action_batches_accepted'] == 1 and client.operator_resumed_requests == 1


def test_first_feedback_registers_its_own_sources_and_file_reads_keep_catalogue_current(episode):
    hub, handoff, _, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    first = artifacts.publish_view(view, 'object', 'geometry')
    other = {**view, 'view_id': 'object_other_angle'}
    later = artifacts.publish_view(other, 'object', 'geometry')
    artifacts.current(executor, hub.run/'current')
    parts = artifacts.feedback_parts([first])
    payload = json.loads(parts[0]['text'])['CURRENT']
    available = payload['available_sources']
    assert [item['view_id'] for item in available['cloud_views']] == ['object_face']
    assert {item['camera'] for item in available['image_references']} == {'agentview', 'robot0_eye_in_hand'}
    assert not {'cloud_views', 'image_references'} & payload['geometry'].keys()
    full = json.loads((artifacts.files.root/'current.json').read_text())
    assert full['source_step'] == payload['source_step']
    assert full['available_sources']['cloud_views'][0]['source_observations']
    assert 'source_observations' not in available['cloud_views'][0]
    assert available['cloud_views'][0]['source_metadata_file'] == 'geometry/object_face.json'
    assert payload['robot_state'] == full['robot_state']
    assert later not in artifacts.seen_paths
    artifacts.read_image(later)
    current = json.loads((artifacts.files.root/'current.json').read_text())
    assert {item['view_id'] for item in current['available_sources']['cloud_views']} == artifacts.seen_views
    indexed = json.loads((artifacts.files.root/'index.json').read_text())['files']
    assert next(item for item in indexed if item['path'] == 'current.json') == artifacts.files.entries['current.json']
    assert not any(item['path'] == 'index.json' for item in indexed)


def test_episode_reference_message_is_historical_and_has_no_old_current_robot_state(episode):
    from sketchik.stage_context import EPISODE_REFERENCES, ACTUAL_IMAGES
    hub, handoff, _, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    path = artifacts.publish_view(view, 'object', 'geometry')
    executor = DirectExecutor(hub, handoff, artifacts)
    artifacts.current(executor, hub.run/'current')
    parts = artifacts.reference_parts([path])
    header = json.loads(parts[0]['text'])
    assert header['phase'] == EPISODE_REFERENCES and header['historical_evidence_only']
    assert header['original_task'] == handoff['task']
    assert header['objects'][0]['object_id'] == 'object'
    assert not {'CURRENT', 'robot_state', 'eef', 'held_claim'} & header.keys()
    assert json.loads(parts[1]['text'])['view_id'] == view['view_id']
    assert view['view_id'] in artifacts.seen_views
    current = json.loads(artifacts.feedback_parts()[0]['text'])['CURRENT']
    assert current['available_sources']['cloud_views'][0]['view_id'] == view['view_id']
    dynamic, actual = artifacts.feedback_messages()
    assert 'CURRENT' in json.loads(dynamic['content'][0]['text'])
    actual_header = json.loads(actual['content'][0]['text'])
    assert actual_header['phase'] == ACTUAL_IMAGES and actual_header['source_step'] == hub.robot.steps
    assert 'CURRENT' not in actual_header and sum(p['type'] == 'image_url' for p in actual['content']) == 2
    hub.journal.mark_moved('object', 20)
    with pytest.raises(ValueError, match='predates'):
        artifacts.validate_point(binding({'kind': 'cloud_point', 'view_id': view['view_id'], 'pixel_xy': [1, 1]}), 'object')


def test_current_uses_actual_hand_once_and_full_past_calibration_remains_readable(episode, monkeypatch):
    hub, handoff, _, _ = episode
    past = {'initial_hand_geometry': {'source_step': 10, 'marker': 'old_hand'},
        'max_observed_open_jaw': {'current_inner_jaw_gap_m': .08}}
    (hub.run/'robot_geometry_evidence.json').write_text(json.dumps(past))
    def actual_runtime(self, packet, step, description, folder):
        packet['robot_gripper_geometry'] = {'source_step': description['source_step'], 'marker': 'actual_hand'}
    monkeypatch.setattr(DirectExecutor, '_refresh_runtime', actual_runtime)
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    current = artifacts.current(executor, hub.run/'current')
    assert current['geometry']['robot_gripper_geometry']['marker'] == 'actual_hand'
    assert 'old_hand' not in json.dumps(current)
    assert 'old_hand' in artifacts.files.read_text_file(current['robot_geometry_file'])['content']


def add_binding_catalog(hub, handoff, view, monkeypatch):
    """A misnamed local view with two real measured component alternatives."""
    handoff['objects'][0]['component_ids'] = ['G1']
    source = {'observation_id': 'scan', 'camera': 'agentview', 'source_step': 10,
        'bounds_xyxy': [0, 0, 32, 32], 'point_count': 30}
    component = {'observed_bounds_world_m': [[0, 0, .8], [1, 1, 1]],
        'observed_span_xyz_m': [1, 1, .2], 'point_count': 30, 'source_steps': [10],
        'last_observed_step': 10, 'semantic_identity_verified': False, 'complete_dimensions_known': False,
        'sources': [source], 'patches': []}
    hub.scene_catalog = {'source_step': 10, 'source_steps': [10], 'scope': 'observed test components',
        'components': {key: {**deepcopy(component), 'component_id': key} for key in ('G1', 'G2')},
        'views': [{**view, 'view_id': 'whole_scene', 'selectable': False}]}
    def context(*args, **kwargs):
        prefix = kwargs.get('view_prefix')
        if prefix:
            from sketchik.observed_dimensions import write_measurement
            write_measurement(hub.run, 'object', {'initial_observed_shape': {
                'source_steps': [10], 'test_component_ids': ['G2']}})
        return {'local_context_available': True,
            'selectable_views': [{**view, 'view_id': prefix+'_focus' if prefix else view['view_id']}]}
    monkeypatch.setattr('sketchik.grounding_packet.build_focused_context', context)


def test_identity_correction_rebuilds_sources_and_retires_wrong_points_without_physics(episode, monkeypatch):
    hub, handoff, actions, view = episode
    handoff['objects'][0]['shape_motion'] = 'rigid'
    add_binding_catalog(hub, handoff, view, monkeypatch)
    (hub.run/'dimensions.json').write_text(json.dumps({'objects': {'object': {
        'initial_observed_shape': {'test_component_ids': ['G1']}, 'wrong_height': .99}}}))
    artifacts = EpisodeArtifacts(hub, handoff)
    initial = artifacts.prepare()
    artifacts.live_cloud_files.setup(hub.observe()[0])
    latest_alias = 'current_clouds/latest/object_00_face.png'
    artifacts.reference_parts(initial)
    original_image = (artifacts.files.root/'geometry/object_00/object_face.png').read_bytes()
    answer = {'source_step': 30, 'object_id': 'object', 'component_ids': ['G2'],
        'description': 'The actual flat object, not the round neighbouring cup',
        'visible_evidence': 'G2 is on the flat object in the overview; the G1 local image is a cup'}
    artifacts.validate_initial_binding(answer)
    result = artifacts.bind_initial_groups(answer)
    assert result['component_ids'] == ['G2'] and result['previous_component_ids'] == ['G1']
    assert result['robot_actions'] == result['new_camera_observations'] == 0
    assert hub.robot.steps == 30 and actions == []
    assert handoff['objects'][0]['component_ids'] == ['G2']
    assert handoff['objects'][0]['shape_motion'] == 'rigid'
    slot = artifacts.live_cloud_files.slots[latest_alias]
    assert 'binding_001' in slot['original_view_id']
    assert artifacts.files.image_aliases[latest_alias] == slot['original_path']
    latest_record, _ = artifacts.read_image(latest_alias)
    assert latest_record['view_id'] == slot['original_view_id'] and latest_record['selectable']
    dimensions = json.loads((hub.run/'dimensions.json').read_text())['objects']['object']
    assert dimensions['initial_observed_shape']['test_component_ids'] == ['G2']
    assert 'wrong_height' not in dimensions
    scene = json.loads((artifacts.files.root/'scene.json').read_text())
    assert scene['objects'][0]['component_ids'] == ['G2']
    assert scene['whole_scene']['catalog']['components'][1]['semantic_aliases'] == ['object']
    old_path = 'geometry/object_00/object_face.png'
    assert (artifacts.files.root/old_path).read_bytes() == original_image
    old_record, _ = artifacts.read_image(old_path)
    assert not old_record['selectable'] and 'retired_reason' in old_record
    with pytest.raises(ValueError, match='superseded'):
        artifacts.validate_point(binding({'kind': 'cloud_point', 'view_id': 'object_face', 'pixel_xy': [1, 1]}), 'object')
    corrected = artifacts.reference_image_paths()
    assert 'binding_001' in corrected[1]
    artifacts.reference_parts(corrected)
    new_view = next(key for key in artifacts.seen_views if key.endswith('binding_001_focus'))
    artifacts.validate_point(binding({'kind': 'cloud_point', 'view_id': new_view, 'pixel_xy': [1, 1]}), 'object')
    current = artifacts.current(DirectExecutor(hub, handoff, artifacts), hub.run/'current')
    assert current['object_bindings'][0]['identity_correction']['previous_component_ids'] == ['G1']
    assert current['object_bindings'][0]['binding_revision'] == 1
    assert 'object_face' not in {item['view_id'] for item in current['available_sources']['cloud_views']}
    archived = json.loads((hub.run/'single_a_bindings/object_00/revision_001/before.json').read_text())
    assert archived['dimensions']['wrong_height'] == .99


@pytest.mark.parametrize('invalid', ['stale', 'unknown_group', 'moved', 'shared_group'])
def test_identity_correction_cannot_revive_old_object_positions(episode, monkeypatch, invalid):
    hub, handoff, actions, view = episode
    add_binding_catalog(hub, handoff, view, monkeypatch)
    artifacts = EpisodeArtifacts(hub, handoff)
    artifacts.reference_parts(artifacts.prepare())
    answer = {'source_step': 30, 'object_id': 'object', 'component_ids': ['G2'],
        'description': 'corrected object', 'visible_evidence': 'the overview label is G2'}
    if invalid == 'stale': answer['source_step'] = 29
    elif invalid == 'unknown_group': answer['component_ids'] = ['missing']
    elif invalid == 'moved': hub.journal.mark_moved('object', 20)
    else:
        handoff['objects'].append({'object_id': 'other', 'component_ids': ['G2']})
        hub.journal.mark_moved('other', 20)
    before = deepcopy(handoff)
    with pytest.raises(ValueError): artifacts.validate_initial_binding(answer)
    assert handoff == before and hub.robot.steps == 30 and actions == []
    assert not (hub.run/'single_a_bindings').exists()


def test_same_A_corrects_binding_then_receives_corrected_views_before_batch_execution(episode, monkeypatch):
    hub, handoff, actions, view = episode
    add_binding_catalog(hub, handoff, view, monkeypatch)
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, answer = 'refresh_object', {'mode': 'bind', 'evidence_id': 'initial_scene',
                    'decision': 'select', 'selection_scope': 'whole_visible_object',
                    'source_step': 30, 'object_id': 'object',
                    'candidate_ids': ['G2'], 'description': 'corrected object',
                    'visible_evidence': 'G2 in the whole-scene image, unlike the old local cup view'}
            elif self.calls == 2:
                assert actions == [] and hub.robot.steps == 30
                assert 'object_binding_001_focus' in json.dumps(messages[-3:])
                current = json.loads(messages[-2]['content'][0]['text'])['CURRENT']
                assert current['object_bindings'][0]['component_ids'] == ['G2']
                name, answer = 'execute_actions', submission(30)
                answer['actions'].append(deepcopy(answer['actions'][0]))
            else:
                assert actions == ['move', 'move']
                name, answer = 'finish', {'status': 'blocked', 'summary': 'Only a fixture execution',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'No task evaluation in this fixture'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(answer)}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=4, max_batches=1)
    assert result['status'] == 'single_agent_blocked'
    assert len(result['subtasks']) == 1 and len(result['subtasks'][0]['steps']) == 2


def test_complete_execution_ledger_survives_beyond_recent_two_batches(episode):
    hub, handoff, _, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    for i in range(5):
        answer = submission(hub.robot.steps)
        plan, stage = prepare_actions(executor, artifacts, answer)
        hub.single_a_requests = i+1
        artifacts.begin_execution(plan, stage, i+1, submission=answer, bound_submission=answer)
        receipt = execute_sketch(hub, executor, plan, i, frozen_binding=stage)
        artifacts.finish_execution(receipt)
    current = artifacts.current(executor, hub.run/'current')
    assert len(current['recent_execution_receipts']) == 2
    assert [item['batch'] for item in current['execution_ledger']] == list(range(1, 6))
    assert all(item['completed_primitives'] == ['move'] for item in current['execution_ledger'])


def test_selected_extra_cloud_views_survive_reference_refresh_without_retaining_all_history(episode):
    hub, handoff, actions, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    primary = artifacts.prepare()
    extras = [artifacts.publish_view({**view, 'view_id': f'object_side_{i}'}, 'object', 'geometry')
        for i in range(8)]
    for path in extras: artifacts.read_image(path)
    assert artifacts.reference_reads == extras[-6:]
    before = list(artifacts.reference_reads)
    paths = artifacts.reference_image_paths()
    assert paths == primary+extras[-6:]
    parts = artifacts.reference_parts(paths)
    assert artifacts.reference_reads == before  # Publishing the snapshot cannot evict A's own choices.
    assert sum(part['type'] == 'image_url' for part in parts) == len(paths)
    artifacts.retired_views['object_side_7'] = 'identity corrected'
    assert extras[-1] not in artifacts.reference_image_paths()
    assert not artifacts.read_image(extras[-1])[0]['selectable']
    assert actions == [] and hub.robot.steps == 30
    assert EpisodeArtifacts(hub, handoff).reference_reads == []


def test_task_context_is_optional_persistent_model_judgment_and_never_compiled_as_geometry(episode):
    hub, handoff, actions, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    answer = submission(30)
    original_plan, original_stage = prepare_actions(executor, artifacts, answer)
    text = 'Selected the task-named inner section using visible structural adjacency; keep that goal during recovery.'
    answer['task_context_update'] = text
    schema = next(tool['function']['parameters'] for tool in tools_for({'object'})
        if tool['function']['name'] == 'execute_actions')
    Draft202012Validator(schema).validate(answer)
    plan, stage = prepare_actions(executor, artifacts, answer)
    assert plan == original_plan and stage == original_stage
    apply_assessment(hub, answer)
    current = artifacts.current(executor, hub.run/'current_context')
    assert current['task_context']['statement'] == text
    assert current['task_context']['source_step'] == 30
    assert 'not physical evidence' in current['task_context']['scope']
    assert not current['task_success_certified'] and not actions
    apply_assessment(hub, submission(30))
    assert hub.episode_state.data['a_task_context']['statement'] == text
    revised = submission(30); revised['task_context_update'] = 'New actual evidence withdraws the prior section interpretation; placement is unresolved.'
    apply_assessment(hub, revised)
    assert len(hub.episode_state.data['a_task_context_history']) == 2
    assert hub.episode_state.data['a_task_context']['statement'] == revised['task_context_update']
    Draft202012Validator(schema).validate(submission(30))  # Existing action batches stay valid.
    invalid = submission(30); invalid['task_context_update'] = {'not': 'a statement'}
    with pytest.raises(ValidationError): Draft202012Validator(schema).validate(invalid)


def test_task_part_basis_survives_route_notes_and_checkpoint_without_changing_motion(episode):
    from sketchik.single_agent import working_current
    from sketchik.stage_context import StageContext
    hub, handoff, actions, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    path = artifacts.publish_view(view, 'object', 'geometry')
    artifacts.read_image(path)
    answer = submission(30)
    plan, stage = prepare_actions(executor, artifacts, answer)
    entry = {'object_id': 'object', 'task_phrase': 'the object', 'status': 'selected',
        'interpretation': 'The item adjacent to the visible divider', 'image_paths': [path]}
    answer.update(task_referents=[entry], task_context_update='Current implementation estimate only')
    Draft202012Validator(tools_for({'object'})[0]['function']['parameters']).validate(answer)
    artifacts.validate_semantic_assessment(answer)
    assert prepare_actions(executor, artifacts, answer) == (plan, stage)
    apply_assessment(hub, answer)
    later = submission(30); later['task_context_update'] = 'Route blocked; reconsider orientation, not identity'
    apply_assessment(hub, later)
    current = artifacts.current(executor, hub.run/'retained_part')
    assert current['task_context']['statement'] == later['task_context_update']
    assert current['task_referents'][0]['interpretation'] == entry['interpretation']
    assert current['task_referents'][0]['image_paths'] == entry['image_paths']
    assert current['task_referents'][0]['source_step'] == 30
    messages = artifacts.feedback_messages()
    context = StageContext(retain_robot_facts=False, retain_episode_references=True)
    context.prepare_boundary(messages, current, hub.run/'part_boundary', lambda x: None, force=True)
    active, _ = context.apply(messages)
    payloads = [json.loads(m['content'][0]['text']) for m in active if isinstance(m.get('content'), list)]
    retained = next(p['CURRENT'] for p in payloads if 'CURRENT' in p)
    assert retained['original_task'] == handoff['task']
    assert retained['task_referents'] == current['task_referents']
    assert working_current(current)['task_referents'] == current['task_referents']
    assert actions == [] and hub.robot.steps == 30


def test_semantic_updates_validate_all_sources_and_revision_before_mutation(episode):
    hub, handoff, actions, view = episode
    DirectExecutor(hub, handoff, None)
    artifacts = EpisodeArtifacts(hub, handoff)
    path = artifacts.publish_view(view, 'object', 'geometry')
    answer = submission(30)
    entry = {'object_id': 'object', 'task_phrase': 'the object', 'status': 'selected',
        'interpretation': 'Chosen visible item by the divider', 'image_paths': [path]}
    answer['task_referents'] = [entry]
    before = deepcopy(hub.episode_state.data)
    with pytest.raises(ValueError, match='actually viewed'):
        artifacts.validate_semantic_assessment(answer)
    assert hub.episode_state.data == before
    artifacts.read_image(path)
    artifacts.validate_semantic_assessment(answer)
    apply_assessment(hub, answer)
    unchanged = deepcopy(hub.episode_state.data)
    revised = deepcopy(answer); revised['task_referents'][0]['interpretation'] = 'Another visible item by the second divider'
    with pytest.raises(ValueError, match='revision_basis'):
        artifacts.validate_semantic_assessment(revised)
    assert hub.episode_state.data == unchanged
    revised['task_referents'][0]['revision_basis'] = 'A second visible divider distinguishes the frame'
    artifacts.validate_semantic_assessment(revised); apply_assessment(hub, revised)
    assert hub.episode_state.data['a_task_referents'][0]['revision'] == 2
    assert hub.episode_state.data['a_task_referent_history'][0]['interpretation'] == entry['interpretation']
    revised['task_referents'][0]['task_phrase'] = 'the easier neighbouring object'
    with pytest.raises(ValueError, match='verbatim'):
        artifacts.validate_semantic_assessment(revised)
    revised['task_referents'] = [dict(entry, revision_basis='Return to the original structural interpretation')]*2
    with pytest.raises(ValueError, match='at most one'):
        artifacts.validate_semantic_assessment(revised)
    assert actions == [] and hub.robot.steps == 30


def test_finish_keeps_entire_original_goal_separate_and_can_resolve_without_motion(episode):
    hub, handoff, actions, view = episode
    DirectExecutor(hub, handoff, None)
    artifacts = EpisodeArtifacts(hub, handoff)
    path = artifacts.publish_view(view, 'object', 'geometry'); artifacts.read_image(path)
    answer = submission(30)
    answer['task_referents'] = [{'object_id': 'object', 'task_phrase': 'the object', 'status': 'unresolved'}]
    schema = tools_for({'object'})[0]['function']['parameters']
    Draft202012Validator(schema).validate(answer)  # No forced speculative future explanation.
    artifacts.validate_semantic_assessment(answer); apply_assessment(hub, answer)
    finish = {'status': 'success', 'summary': 'Model assessment only', 'goal_evidence': [{
        'goal': 'the object', 'interpretation': 'A model interpretation', 'assessment': 'satisfied',
        'image_paths': [path], 'visible_evidence': 'Fixture picture'}]}
    with pytest.raises(ValueError, match='entire original_task'):
        artifacts.validate_semantic_assessment(finish, finish=True)
    finish['goal_evidence'][0]['goal'] = handoff['task']
    with pytest.raises(ValueError, match='unresolved'):
        artifacts.validate_semantic_assessment(finish, finish=True)
    finish['task_referents'] = [{'object_id': 'object', 'task_phrase': 'the object', 'status': 'selected',
        'interpretation': 'The item by the divider', 'image_paths': [path],
        'revision_basis': 'The divider distinguishes it from the competing item'}]
    Draft202012Validator(tools_for({'object'})[-1]['function']['parameters']).validate(finish)
    artifacts.validate_semantic_assessment(finish, finish=True)  # Same-call resolution, zero new actions.
    finish['goal_evidence'][0]['goal'] += ' (my renamed target)'
    with pytest.raises(ValueError, match='original_task'):
        artifacts.validate_semantic_assessment(finish, finish=True)
    assert hub.episode_state.data['a_task_referents'][0]['status'] == 'unresolved'
    assert actions == [] and hub.robot.steps == 30


def test_finish_dispatch_updates_part_interpretation_without_an_execution_batch(episode):
    hub, handoff, actions, _ = episode
    DirectExecutor(hub, handoff, None)
    answer = submission(30)
    answer['task_referents'] = [{'object_id': 'object', 'task_phrase': 'the object', 'status': 'unresolved'}]
    apply_assessment(hub, answer)
    path = 'observations/obs_00030/agentview.png'
    class Client:
        last_request_metadata = {}
        def complete(self, messages, tools):
            args = {'status': 'blocked', 'summary': 'Identity clarified but fixture has no placement proof',
                'goal_evidence': [{'goal': handoff['task'], 'assessment': 'unknown', 'image_paths': [path],
                    'interpretation': 'Item adjacent to the divider', 'visible_evidence': 'No observed placement'}],
                'task_referents': [{'object_id': 'object', 'task_phrase': 'the object', 'status': 'selected',
                    'interpretation': 'Item adjacent to the divider', 'image_paths': [path],
                    'revision_basis': 'Viewed divider distinguishes the item'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': 'finish', 'type': 'function',
                'function': {'name': 'finish', 'arguments': json.dumps(args)}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=1, max_batches=1)
    assert result['status'] == 'single_agent_blocked'
    assert hub.episode_state.data['a_task_referents'][0]['status'] == 'selected'
    assert hub.episode_state.data['a_task_referent_history'][0]['status'] == 'unresolved'
    assert result['subtasks'] == [] and actions == [] and hub.robot.steps == 30


def test_A_chosen_side_view_and_task_interpretation_remain_visible_after_a_batch(episode, monkeypatch):
    from sketchik.stage_context import StageContext, EPISODE_REFERENCES
    hub, handoff, actions, view = episode
    side = {**view, 'view_id': 'object_side'}
    monkeypatch.setattr('sketchik.grounding_packet.build_focused_context', lambda *a, **kw: {
        'local_context_available': True, 'selectable_views': [view, side]})
    statement = 'The intended task part remains the inner section, not an alternate section after a blocked route.'
    class Client:
        calls = 0
        last_request_metadata = {}
        stage_context = StageContext(retain_robot_facts=False, retain_episode_references=True)
        def complete(self, messages, tools):
            self.calls += 1
            active, _ = self.stage_context.apply(messages)
            refs = [m for m in active if isinstance(m.get('content'), list) and
                EPISODE_REFERENCES in m['content'][0].get('text', '')]
            assert len(refs) == 1
            if self.calls == 1:
                name, args = 'read_image', {'path': 'geometry/object_00/object_side.png'}
            elif self.calls == 2:
                assert 'geometry/object_00/object_side.png' in json.dumps(refs[0])
                name, args = 'execute_actions', submission(30)
                args['task_context_update'] = statement
            else:
                assert actions == ['move']
                assert 'geometry/object_00/object_side.png' in json.dumps(refs[0])
                current = json.loads(messages[-2]['content'][0]['text'])['CURRENT']
                assert current['task_context']['statement'] == statement
                name, args = 'finish', {'status': 'blocked', 'summary': 'Fixture has no task completion proof',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'Only the fixture move was executed'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=4, max_batches=1)
    assert len(result['subtasks']) == 1


def test_selected_native_images_share_the_bounded_reference_set_and_survive_automatic_feedback(episode):
    hub, handoff, actions, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    artifacts.prepare()
    artifacts.current(DirectExecutor(hub, handoff, artifacts), hub.run/'current')
    native = 'observations/obs_00030/robot0_eye_in_hand.png'
    artifacts.read_image(native)
    sides = [artifacts.publish_view({**view, 'view_id': f'other_{i}'}, 'object', 'geometry')
        for i in range(5)]
    for path in sides: artifacts.read_image(path)
    assert artifacts.reference_reads == [native, *sides]
    artifacts.feedback_parts()
    references = artifacts.reference_parts(artifacts.reference_image_paths())
    assert artifacts.reference_reads == [native, *sides]
    assert native in json.dumps(references)
    artifacts.read_image(sides[0])
    assert artifacts.reference_reads == [native, *sides[1:], sides[0]]
    assert not actions and hub.robot.steps == 30


def test_default_RGB_angles_and_selected_height_share_exact_point_data(episode, monkeypatch):
    hub, handoff, actions, view = episode
    views = [{**view, 'view_id':f'object_{name}'} for name in ('focus','left','right')]
    monkeypatch.setattr('sketchik.grounding_packet.build_focused_context', lambda *a, **kw: {
        'local_context_available':True,'selectable_views':views})
    artifacts = EpisodeArtifacts(hub, handoff)
    initial = artifacts.prepare()
    assert initial == [f'geometry/object_00/object_{name}.png' for name in ('focus','right','focus_gravity_top')]
    assert 'geometry/object_00/object_left.png' in artifacts.files.entries
    artifacts.reference_parts(initial)
    assert artifacts.reference_reads == []
    height = 'geometry/object_00/object_focus_height.png'
    assert 'object_focus_height' not in artifacts.seen_views
    record, _ = artifacts.read_image(height)
    assert record['same_pixel_map_as'] == 'object_focus'
    assert artifacts.reference_reads == [height]
    assert height in artifacts.reference_image_paths()
    artifacts.validate_point(binding({'kind':'cloud_point','view_id':'object_focus_height','pixel_xy':[1,1]}), 'object')
    assert not actions


def test_default_angles_budget_gives_all_objects_first_views_before_extra_angles(episode):
    hub, handoff, _, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    handoff['objects'] = [{'object_id':f'object_{i}'} for i in range(5)]
    artifacts.object_entries = {f'object_{i}':{'views':[{'rgb':f'{i}_{j}.png'} for j in range(3)]}
        for i in range(5)}
    paths = artifacts.default_object_image_paths()
    assert len(paths) == 10
    assert paths[:5] == [f'{i}_0.png' for i in range(5)]
    assert paths[5:] == [f'{i}_1.png' for i in range(5)]


def test_gravity_projection_uses_original_point_ids_groups_and_motion_invalidation(episode, monkeypatch):
    import hashlib
    from sketchik.cloud_selection import resolve_cloud_point, public_view
    hub, handoff, actions, view = episode
    xy = np.asarray([(x, y) for x in np.linspace(.1, .2, 11) for y in np.linspace(.2, .35, 13)])
    points = np.concatenate([np.column_stack([xy, np.full(len(xy), z)]) for z in (.8, .95)])
    cloud = Path(view['cloud_file'])
    np.savez(cloud, world_xyz_m=points, rgb=np.zeros((len(points), 3), np.uint8),
        source_index=np.zeros(len(points), int), pixel_xy=np.zeros((len(points), 2), int))
    ids = np.repeat([0, 1], len(xy)); groups = hub.run/'original_groups.npz'
    np.savez(groups, point_group_ids=ids)
    originals = [{**view, 'view_id': 'object_'+name, 'point_count': len(points),
        'point_group_ids_file': str(groups), 'group_ids': [0, 1]}
        for name in ('focus', 'left', 'right')]
    monkeypatch.setattr('sketchik.grounding_packet.build_focused_context', lambda *a, **kw: {
        'local_context_available': True, 'selectable_views': originals})
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cloud, groups, Path(view['image_file']))}
    artifacts = EpisodeArtifacts(hub, handoff); defaults = artifacts.prepare()
    top = artifacts.views['object_focus_gravity_top']
    assert top['cloud_file'] == str(cloud.resolve())
    assert top['sources'] == view['sources']
    assert len(defaults) == 3 and defaults[0].endswith('object_focus.png')
    assert defaults[1].endswith('object_right.png') and defaults[2].endswith('object_focus_gravity_top.png')
    assert 'object_left' in artifacts.views and 'object_left_height' in artifacts.views
    with np.load(top['index_map_file']) as data:
        mapping = data['point_index']; ys, xs = np.nonzero(mapping >= 0)
    assert len(xs) > 100
    np.testing.assert_allclose(points[mapping[ys, xs], 2], .95)  # Topmost observed layer, not hidden completion.
    pixel = [int(xs[len(xs)//2]), int(ys[len(ys)//2])]
    anchor = {'kind': 'cloud_point', 'view_id': top['view_id'], 'pixel_xy': pixel, 'group_id': 1}
    expected = points[mapping[pixel[1], pixel[0]]]
    actual, source = resolve_cloud_point(anchor, [public_view(top)], [top])
    np.testing.assert_array_equal(actual, expected)
    assert source['cloud_group_id'] == 1 and source['source_observation']['source_step'] == 10
    with pytest.raises(ValueError, match='Read the exact'):
        artifacts.validate_point(binding(anchor), 'object')
    artifacts.read_image(defaults[2]); artifacts.validate_point(binding(anchor), 'object')
    hub.journal.mark_moved('object', 20)
    with pytest.raises(ValueError, match='predates'):
        artifacts.validate_point(binding(anchor), 'object')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    assert actions == [] and hub.robot.steps == 30


def test_all_object_image_paths_precede_large_measurements_without_registering_unread_points(episode, monkeypatch):
    hub, handoff, actions, view = episode
    handoff['objects'].append({**deepcopy(handoff['objects'][0]), 'object_id':'destination'})
    def context(*args, **kwargs):
        target = args[2]
        return {'local_context_available':True, 'selectable_views':[
            {**view, 'view_id':target+'_focus'}, {**view, 'view_id':target+'_side'}],
            'horizontal_layers': {'large_measurements':'x'*20000}}
    monkeypatch.setattr('sketchik.grounding_packet.build_focused_context', context)
    artifacts = EpisodeArtifacts(hub, handoff)
    artifacts.prepare()
    text = (artifacts.files.root/'scene.json').read_text()
    assert text.index('destination_side.png') < 6000
    assert text.index('object_image_index') < text.index('large_measurements')
    assert artifacts.seen_views == set()
    with pytest.raises(ValueError, match='Read the exact'):
        artifacts.validate_point(binding({'kind':'cloud_point','view_id':'destination_side','pixel_xy':[1,1]}), 'destination')
    artifacts.read_image('geometry/object_01/destination_side.png')
    artifacts.validate_point(binding({'kind':'cloud_point','view_id':'destination_side','pixel_xy':[1,1]}), 'destination')
    hub.journal.mark_moved('destination', 20)
    record, _ = artifacts.read_image('geometry/object_01/destination_side.png')
    assert record['point_time_status']['invalidated_at_step'] == 20
    with pytest.raises(ValueError, match='predates'):
        artifacts.validate_point(binding({'kind':'cloud_point','view_id':'destination_side','pixel_xy':[1,1]}), 'destination')
    assert not actions


def test_A_selected_native_history_is_visible_after_the_actual_context_checkpoint(episode):
    from sketchik.stage_context import StageContext, EPISODE_REFERENCES
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        stage_context = StageContext(retain_robot_facts=False, retain_episode_references=True)
        def complete(self, messages, tools):
            self.calls += 1
            active, _ = self.stage_context.apply(messages)
            refs = [m for m in active if isinstance(m.get('content'), list) and
                EPISODE_REFERENCES in m['content'][0].get('text', '')]
            if self.calls == 1:
                name, args = 'read_image', {'path':'observations/obs_00030/robot0_eye_in_hand.png'}
            elif self.calls == 2:
                name, args = 'execute_actions', submission(30)
            else:
                assert actions == ['move']
                assert 'observations/obs_00030/robot0_eye_in_hand.png' in json.dumps(refs)
                name, args = 'finish', {'status':'blocked','summary':'Fixture only',
                    'goal_evidence':[{'goal':'place the object','assessment':'unknown',
                        'image_paths':['observations/obs_00031/agentview.png'],'visible_evidence':'Only one fixture move'}]}
            return {'message':{'role':'assistant','tool_calls':[{'id':str(self.calls),'type':'function',
                'function':{'name':name,'arguments':json.dumps(args)}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=4, max_batches=1)
    index = json.loads((hub.run/'agent_public/acquisition/index.json').read_text())
    assert any(item['path'] == 'observations/obs_00030/robot0_eye_in_hand.png' for item in index['native_image_index'])


def test_held_native_pixel_keeps_capture_time_and_temporary_support_is_request_current(episode, monkeypatch):
    hub, handoff, _, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    artifacts.seen_rgb.add(('older', 'agentview'))
    monkeypatch.setattr('sketchik.perception.saved_frame', lambda *args: {'step': 10})
    monkeypatch.setattr('sketchik.libero.back_project', lambda *args: np.array([.2, .3, .9]))
    point = binding({'kind': 'image_point', 'observation_id': 'older',
        'camera': 'agentview', 'pixel_xy': [1, 1]})
    # An older static destination remains useful; a carried object's old world
    # coordinates and an anonymous old support do not become current on reread.
    assert artifacts.validate_point(point, 'object')['source_step'] == 10
    with pytest.raises(ValueError, match='current.*request|request.*current'):
        artifacts.validate_point(point, 'stage_support', request_step=30)
    hub.carried_target_id = 'object'
    with pytest.raises(ValueError, match='Held|held'):
        artifacts.validate_point(point, 'object')


def test_frozen_single_cloud_uses_actual_pixel_capture_not_submission_step(episode):
    hub, handoff, _, view = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    artifacts.read_image(artifacts.publish_view(view, 'object', 'geometry'))
    executor = DirectExecutor(hub, handoff, artifacts)
    answer = submission(30, {'kind': 'cloud_point', 'view_id': view['view_id'], 'pixel_xy': [1, 1]})
    _, stage = prepare_actions(executor, artifacts, answer)
    assert stage['source_step'] == 30
    assert stage['locations'][0]['grounding_source_step'] == 10


def test_actual_held_motion_retires_world_position_even_if_next_A_claim_is_empty(episode):
    hub, handoff, _, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    hub.carried_target_id = 'object'
    hub._fixture_state['gripper_command'] = 1.
    executor.record_execution({'primitive': 'move', 'source_step': 31,
        'parameters': {'position': [0., 0., 1.1]}, 'result': {'converged': False}})
    assert hub.journal.data['objects']['object']['invalidated_at_step'] == 31
    hub.carried_target_id = None
    assert artifacts.point_time_status('object')['invalidated_at_step'] == 31


def test_missing_redundant_batch_fields_bind_to_request_without_inventing_intent(episode):
    hub, handoff, actions, _ = episode
    answer = submission(30, {'kind': 'cloud_point', 'view_id': 'object_face', 'pixel_xy': [1, 1]})
    for key in ('source_step', 'intent'):
        del answer[key]
    del answer['actions'][0]['reference_target_id']
    raw = deepcopy(answer)
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls),
                'type': 'function', 'function': {'name': 'execute_actions', 'arguments': json.dumps(answer)}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=1, max_batches=1)
    assert actions == ['move'] and len(result['subtasks']) == 1
    assert result['subtasks'][0]['subtask'] is None  # No model intent fabricated by the program.
    assert answer == raw
    folder = hub.run/'single_a/turn_001'
    assert json.loads((folder/'submitted_actions.json').read_text()) == raw
    bound = json.loads((folder/'request_bound_actions.json').read_text())
    assert bound['source_step'] == 30 and 'intent' not in bound
    assert bound['actions'][0]['reference_target_id'] == 'object'


@pytest.mark.parametrize('missing_owner', [False, True])
def test_changed_assessment_is_recorded_before_old_points_are_rejected_without_batch(episode, missing_owner):
    hub, handoff, actions, _ = episode
    add_current_depth_fixture(hub)
    first = submission(30, {'kind': 'cloud_point', 'view_id': 'object_face', 'pixel_xy': [1, 1]})
    first['changed_objects'] = ['object']
    if missing_owner:
        first['actions'][0]['selection']['waypoints'] = [binding()]
        del first['actions'][0]['reference_target_id']
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 2:
                assert actions == []
                assert hub.journal.data['objects']['object']['invalidated_at_step'] == 30
                assert hub.single_a_batches == 0
            answer = first if self.calls == 1 else submission(30)
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls),
                'type': 'function', 'function': {'name': 'execute_actions', 'arguments': json.dumps(answer)}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=2, max_batches=1)
    assert actions == ['move'] and len(result['subtasks']) == 1
    attempt = hub.single_a_execution_ledger['rejected_attempts'][0]
    assert attempt['assessment_committed'] and attempt['robot_actions'] == 0
    assert not attempt['batch_accepted'] and attempt['actual_source_step'] == 30


def test_single_tool_budget_rejects_entire_over_budget_reply_without_partial_calls(episode):
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            return {'message': {'role': 'assistant', 'tool_calls': [
                {'id': str(i), 'type': 'function', 'function': {'name': 'list_dir',
                    'arguments': json.dumps({'path': 'geometry'})}} for i in range(2)]}}
    client = Client()
    result = run_single_agent(hub, client, [], handoff, max_turns=5, max_batches=1, max_tool_calls=1)
    assert result['status'] == 'single_agent_tool_budget_exhausted'
    assert result['a_execution_tool_calls'] == 0 and client.calls == 1 and actions == []


def test_published_current_budget_matches_the_current_read_and_stops_at_the_limit(episode):
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            current = json.loads((hub.run/'agent_public/current.json').read_text())
            assert current['budgets']['used']['requests'] == self.calls
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls),
                'type': 'function', 'function': {'name': 'read_text_file',
                    'arguments': json.dumps({'path': 'current.json', 'max_chars': 40000})}}]}}
    client = Client(); messages = []
    result = run_single_agent(hub, client, messages, handoff, max_turns=5, max_batches=1, max_tool_calls=2)
    assert result['status'] == 'single_agent_tool_budget_exhausted' and client.calls == 2
    replies = [json.loads(m['content']) for m in messages if m.get('role') == 'tool']
    for index, reply in enumerate(replies, 1):
        current = json.loads(reply['content'])
        assert current['budgets']['used']['tool_calls'] == index
        assert reply['execution_budgets'] == current['budgets']
    assert actions == []


def test_single_invalid_streak_counts_whole_replies_and_good_read_does_not_hide_bad_read(episode):
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            paths = {1: ['missing_a', 'missing_b'], 2: ['usage.md'],
                3: ['missing_c', 'usage.md'], 4: ['missing_d']}[self.calls]
            return {'message': {'role': 'assistant', 'tool_calls': [
                {'id': f'{self.calls}_{i}', 'type': 'function', 'function': {'name': 'read_text_file',
                    'arguments': json.dumps({'path': path})}} for i, path in enumerate(paths)]}}
    client = Client()
    result = run_single_agent(hub, client, [], handoff, max_turns=6, max_batches=1, max_invalid_replies=2)
    assert result['status'] == 'single_agent_invalid_output' and client.calls == 4
    assert result['a_execution_tool_calls'] == 6 and actions == []


def test_execution_ledger_survives_later_provider_failure_without_retry_or_lost_prefix(episode):
    from sketchik.agent import ProviderError
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls > 1:
                raise ProviderError('Provider HTTP 500; fixture')
            answer = submission(30)
            answer['actions'].append(deepcopy(answer['actions'][0]))
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': 'move', 'type': 'function',
                'function': {'name': 'execute_actions', 'arguments': json.dumps(answer)}}]}}
    client = Client()
    with pytest.raises(ProviderError):
        run_single_agent(hub, client, [], handoff, max_turns=5, max_batches=2)
    assert client.calls == 2 and actions == ['move', 'move']
    ledger = json.loads((hub.run/'single_a/execution_ledger.json').read_text())
    assert len(ledger['batches']) == 1 and ledger['batches'][0]['status'] == 'finished'
    assert len(ledger['batches'][0]['receipt']['steps']) == 2
    assert len(hub.single_a_receipts) == 1


def test_interrupted_batch_records_known_prefix_and_unknown_partial_current_action(episode, monkeypatch):
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            answer = submission(30); answer['actions'].append(deepcopy(answer['actions'][0]))
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': 'move', 'type': 'function',
                'function': {'name': 'execute_actions', 'arguments': json.dumps(answer)}}]}}
    def interrupted(hub, executor, plan, index, *, frozen_binding):
        params = {'position': [0., 0., 1.1], 'quaternion_xyzw': [0., 0., 0., 1.]}
        executor.record_dispatch(frozen_binding, 0, {'parameters': params})
        output = hub.executor.move(**params)
        executor.record_execution({'primitive': 'move', 'parameters': params,
            'result': output, 'source_step': hub.robot.steps, 'completed': True,
            'expanded_step_index': 0, 'location': frozen_binding['locations'][0]})
        executor.record_dispatch(frozen_binding, 1, {'parameters': params})
        hub.executor.move(**params)
        raise ValueError('program fault after unreceipted physics')
    monkeypatch.setattr('sketchik.single_agent.execute_sketch', interrupted)
    with pytest.raises(ValueError, match='program fault'):
        run_single_agent(hub, Client(), [], handoff, max_turns=5, max_batches=2)
    assert actions == ['move', 'move']
    ledger = json.loads((hub.run/'single_a/execution_ledger.json').read_text())
    batch = ledger['batches'][0]
    assert batch['status'] == 'interrupted' and batch['receipt']['source_step'] == 32
    assert len(batch['receipt']['steps']) == 1
    assert batch['receipt']['unrecorded_current_action_possible'] is True
    assert batch['receipt']['commands_completed'] is False
    assert [entry['status'] for entry in batch['expanded_steps']] == ['completed', 'outcome_unknown']
    assert batch['receipt']['execution_progress']['unexecuted_suffix'] == []


def add_current_depth_fixture(hub):
    original = hub.observe
    def observe():
        description, pictures = original()
        depth = np.full((32, 32), np.nan)
        depth[6:26, 6:26] = 1.
        for camera in ('agentview', 'robot0_eye_in_hand'):
            np.savez(hub.run/'observations'/description['observation_id']/(camera+'.npz'),
                depth_m=depth, K=np.array([[500., 0., 16.], [0., 500., 16.], [0., 0., 1.]]),
                world_from_camera=np.eye(4))
        return description, pictures
    hub.observe = observe


def current_refresh_fixture(episode):
    hub, handoff, actions, _ = episode
    add_current_depth_fixture(hub)
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    artifacts.reference_parts(artifacts.prepare())
    hub.journal.mark_moved('object', 20)
    artifacts.current(executor, hub.run/'current')
    artifacts.feedback_parts()
    request = {'mode': 'inspect', 'object_id': 'object'}
    result, parts, changed = artifacts.refresh_object(request, artifacts.validate_refresh(request, 30))
    return hub, artifacts, executor, result, parts


def test_inspection_candidates_do_not_become_explicit_protected_reference_reads(episode):
    import hashlib
    hub, artifacts, executor, result, parts = current_refresh_fixture(episode)
    paths = result['image_paths']
    before = {path: hashlib.sha256(artifacts.files.path(path).read_bytes()).hexdigest()
              for path in paths}
    assert artifacts.reference_reads == []
    assert set(paths).issubset(artifacts.seen_paths)
    records = [json.loads(part['text']) for part in parts if part['type'] == 'text']
    assert all(record['public_artifact_read'] is False for record in records)
    assert all(record['display_origin'] == 'automatic_current_or_reference_display'
               for record in records)
    # The current set remains bindable without disguising all its pictures as
    # explicit choices to preserve in later episode-reference messages.
    request = {'mode': 'bind', 'object_id': 'object', 'evidence_id': result['evidence_id'],
        'candidate_ids': ['G1'], 'decision': 'select', 'selection_scope': 'visible_part',
        'description': 'visible fragment', 'visible_evidence': 'G1 in the current displayed candidates'}
    assert artifacts.validate_refresh(request, hub.robot.steps)['inspection']['evidence_id'] == result['evidence_id']
    chosen, _ = artifacts.read_image(paths[0])
    assert chosen['public_artifact_read'] is True
    assert chosen['display_origin'] == 'explicit_read_image'
    assert artifacts.reference_reads == [paths[0]]
    # Another automatically displayed set must not evict A's actual selection.
    inspect = {'mode': 'inspect', 'object_id': 'object'}
    fresh, _, _ = artifacts.refresh_object(inspect, artifacts.validate_refresh(inspect, hub.robot.steps))
    assert fresh['evidence_id'] != result['evidence_id']
    assert artifacts.reference_reads == [paths[0]]
    assert not any(path in artifacts.reference_image_paths() for path in fresh['image_paths'])
    assert all(hashlib.sha256(artifacts.files.path(path).read_bytes()).hexdigest() == digest
               for path, digest in before.items())
    assert hub.robot.steps == 30


def test_registered_cloud_uses_new_coordinate_epoch_without_rewriting_old_capture_or_clearing_warning(episode, monkeypatch):
    import hashlib
    from sketchik.pose_binding import compile_binding
    from sketchik.single_agent import working_cloud_record
    hub, artifacts, executor, result, _ = current_refresh_fixture(episode)
    artifacts.handoff['objects'][0]['shape_motion'] = 'rigid'
    view = artifacts.views['object_face']
    with np.load(view['cloud_file']) as data:
        template = {key: data[key][:25].copy() for key in ('world_xyz_m', 'rgb', 'pixel_xy', 'source_index')}
    xyz = template['world_xyz_m'].copy()
    template.update(template_source_cloud_file=np.asarray(view['cloud_file']), template_point_ids=np.arange(25))
    sources = view['sources']
    hashes = {name: hashlib.sha256(Path(view[name]).read_bytes()).hexdigest()
        for name in ('image_file', 'index_map_file', 'cloud_file')}
    hub.scene_catalog = {'components': {}}
    monkeypatch.setattr('sketchik.object_registration.catalog_template', lambda *a: (template, sources))
    registration = {'status': 'accepted', 'source_step': 30, 'reason': 'supported_unique_fit', 'wall_s': .1,
        'transform': {'rotation': np.eye(3).tolist(), 'translation_m': [.03, -.02, .01]},
        'feature_correspondences': 12, 'current_groups_considered': 1, 'search_complete': True,
        'scope': 'Fixture numerical pose estimate, not holding certification'}
    monkeypatch.setattr('sketchik.object_registration.register_object', lambda *a, **kw: registration)
    warning = {'source_step': 30, 'reason': 'contact changed'}
    hub.episode_state.data['contact_warning'] = deepcopy(warning)
    summary = artifacts.relocalize_object('object', artifacts.geometry_inspections[result['evidence_id']])
    assert summary['status'] == 'accepted' and summary['agent_requests'] == summary['robot_actions'] == 0
    assert hub.episode_state.data['contact_warning'] == warning
    assert hub.carried_target_id is None and hub.carried_surface_reference is None
    assert summary['bitmap_reused'] and summary['coordinates_available']
    record, _ = artifacts.read_image(artifacts.image_paths['object_face'])
    status = working_cloud_record(record)['point_time_status']
    assert status['coordinate_estimated_at_step'] == 30 and status['image_axes_are_historical']
    anchor = {'kind': 'cloud_point', 'view_id': 'object_face', 'pixel_xy': [1, 1]}
    checked = artifacts.validate_point(binding(anchor), 'object', request_step=30)
    assert checked['source_step'] == 30
    answer = submission(30, anchor)
    plan, stage = prepare_actions(executor, artifacts, answer)
    pose, audit = compile_binding(binding(anchor), stage['frozen_packets'][0], lambda *args: None,
        cloud_views=stage['cloud_views'][0])
    np.testing.assert_allclose(pose['position'], xyz[0]+[.03, -.02, .11])
    assert audit['source_observation']['source_step'] == 10
    assert audit['coordinate_estimate']['source_step'] == 30
    assert audit['carry_reference_eligible'] is False
    for name, digest in hashes.items():
        assert hashlib.sha256(Path(view[name]).read_bytes()).hexdigest() == digest
    hub.journal.mark_moved('object', 31)
    with pytest.raises(ValueError, match='predates actual object motion'):
        artifacts.validate_point(binding(anchor), 'object', request_step=31)
    assert hub.episode_state.data['contact_warning'] == warning


def test_current_binding_replaces_stale_positions_without_erasing_shape_or_contact_warning(episode):
    hub, artifacts, executor, result, parts = current_refresh_fixture(episode)
    from sketchik.observed_dimensions import write_measurement
    old_shape = {'source_steps': [10], 'span_world_m': [.2, .1, .04], 'complete_shape_known': False}
    write_measurement(hub.run, 'object', {'initial_observed_shape': old_shape})
    hub.episode_state.data['contact_warning'] = {'reason': 'same actual contact remains under review'}
    assert result['candidate_ids'] == ['G1'] and len(parts) == 4
    assert artifacts.binding_revisions == {}  # One candidate is not automatic identity selection.
    request = {'mode': 'bind', 'object_id': 'object', 'evidence_id': result['evidence_id'],
        'candidate_ids': ['G1'], 'decision': 'select', 'selection_scope': 'visible_part',
        'description': 'currently visible fragment', 'visible_evidence': 'G1 matches the intended visible region'}
    original_template=({'world_xyz_m':np.arange(180,dtype=float).reshape(60,3)},['original sources'])
    artifacts.registration_templates['object']=original_template
    bound, _, changed = artifacts.refresh_object(request, artifacts.validate_refresh(request, 30))
    assert artifacts.registration_template('object') is original_template
    assert not bound['geometry_binding']['carry_reference_eligible']
    assert bound['geometry_binding']['point_set_scope']=='mixed_or_unknown'
    assert changed and bound['binding_revision'] == 1
    assert hub.journal.data['objects']['object']['invalidated_at_step'] == 20
    assert hub.episode_state.data['contact_warning']['reason'] == 'same actual contact remains under review'
    dimensions = json.loads((hub.run/'dimensions.json').read_text())['objects']['object']
    assert dimensions['initial_observed_shape'] == old_shape
    assert dimensions['current_observed_shape']['geometry_binding']['selection_scope'] == 'visible_part'
    assert 'object_face' in artifacts.retired_views
    fresh = bound['views'][0]['view_id']
    view = artifacts.views[fresh]
    with np.load(view['index_map_file']) as data:
        y, x = np.argwhere(data['point_index'] >= 0)[50]
    point = binding({'kind': 'cloud_point', 'view_id': fresh, 'pixel_xy': [int(x), int(y)], 'group_id': 1})
    with pytest.raises(ValueError, match='Read the exact'):
        artifacts.validate_point(point, 'object')
    artifacts.reference_parts(artifacts.reference_image_paths())
    assert artifacts.validate_point(point, 'object')['source_step'] == 30
    with pytest.raises(ValueError, match='superseded'):
        artifacts.validate_point(binding({'kind': 'cloud_point', 'view_id': 'object_face', 'pixel_xy': [1, 1]}), 'object')
    assert hub.robot.steps == 30


@pytest.mark.parametrize('problem', ['stale_candidate', 'unknown_id', 'unseen_image', 'old_native'])
def test_current_geometry_invalid_binding_and_old_source_do_not_rebind_or_move(episode, problem):
    hub, artifacts, executor, result, _ = current_refresh_fixture(episode)
    request = {'mode': 'bind', 'object_id': 'object', 'evidence_id': result['evidence_id'],
        'candidate_ids': ['G1'], 'decision': 'select', 'selection_scope': 'whole_visible_object',
        'description': 'visible object', 'visible_evidence': 'candidate image'}
    step = 30
    if problem == 'stale_candidate':
        hub.robot.steps = hub._fixture_state['step'] = step = 31
    elif problem == 'unknown_id': request['candidate_ids'] = ['G99']
    elif problem == 'unseen_image': artifacts.seen_paths.difference_update(result['image_paths'])
    else:
        hub.robot.steps = hub._fixture_state['step'] = step = 31
        request = {'mode': 'inspect', 'object_id': 'object', 'sources': [
            {'observation_id': 'obs_00030', 'camera': 'agentview', 'bounds_xyxy': [0, 0, 32, 32]}]}
    with pytest.raises(ValueError): artifacts.validate_refresh(request, step)
    assert artifacts.binding_revisions == {} and not artifacts.retired_views


def test_current_candidate_none_does_not_choose_the_only_group_or_change_any_binding(episode):
    hub, artifacts, _, result, _ = current_refresh_fixture(episode)
    before = deepcopy(artifacts.object_entries)
    request = {'mode': 'bind', 'object_id': 'object', 'evidence_id': result['evidence_id'],
        'candidate_ids': [], 'decision': 'none', 'visible_evidence': 'Not enough identity evidence'}
    value, parts, changed = artifacts.refresh_object(request, artifacts.validate_refresh(request, 30))
    assert value['decision'] == 'none' and not changed and parts == []
    assert artifacts.object_entries == before and artifacts.binding_revisions == {}
    assert hub.robot.steps == 30


def test_same_A_inspects_current_depth_binds_exact_points_and_resumes_segment_after_checkpoint(episode):
    from sketchik.stage_context import StageContext
    hub, handoff, actions, _ = episode
    add_current_depth_fixture(hub)
    hub.journal.mark_moved('object', 20)
    class Client:
        calls = 0
        last_request_metadata = {}
        stage_context = StageContext(max_chars=1000, max_messages=5,
            retain_robot_facts=False, retain_episode_references=True, keep_messages=4, keep_chars=500)
        def complete(self, messages, tools):
            self.calls += 1
            active, _ = self.stage_context.apply(messages)
            names = {value['function']['name'] for value in tools}
            assert len(names) == 7 and 'refresh_object' in names and 'rebind_object' not in names
            if self.calls == 1:
                name, answer = 'refresh_object', {'mode': 'inspect', 'object_id': 'object'}
            elif self.calls == 2:
                assert 'current_geometry/current_001/candidates_0.png' in json.dumps(active)
                name, answer = 'refresh_object', {'mode': 'bind', 'object_id': 'object',
                    'evidence_id': 'current_001', 'candidate_ids': ['G1'], 'decision': 'select',
                    'selection_scope': 'visible_part', 'description': 'the chosen visible fragment',
                    'visible_evidence': 'G1 in current candidates corresponds to intended visible region'}
            elif self.calls == 3:
                current = json.loads((hub.run/'agent_public/current.json').read_text())
                view = next(value for value in current['available_sources']['cloud_views'] if '_current_' in value['view_id'])
                assert view['image_path'] in json.dumps(active)
                with np.load(hub.run/'agent_public'/view['point_index_path']) as data:
                    y, x = np.argwhere(data['point_index'] >= 0)[50]
                name, answer = 'execute_actions', submission(30, {'kind': 'cloud_point',
                    'view_id': view['view_id'], 'pixel_xy': [int(x), int(y)], 'group_id': 1})
            else:
                assert actions == ['move']
                name, answer = 'finish', {'status': 'blocked', 'summary': 'Fixture only',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': ['observations/obs_00031/agentview.png'], 'visible_evidence': 'One actual fixture move'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls),
                'type': 'function', 'function': {'name': name, 'arguments': json.dumps(answer)}}]}}
    client = Client()
    result = run_single_agent(hub, client, [], handoff, max_turns=5, max_batches=1)
    assert result['status'] == 'single_agent_blocked'
    assert client.calls == 4 and actions == ['move'] and result['a_execution_tool_calls'] == 4
    assert hub.robot.steps == 31


def test_changed_geometry_failure_prepares_candidates_without_an_extra_A_inspection_request(episode):
    hub, handoff, actions, _ = episode
    add_current_depth_fixture(hub)
    hub.journal.mark_moved('object', 20)
    events = []; hub.emit = events.append
    original_move = hub.executor.move
    moves = []
    def move(**parameters):
        output = original_move(**parameters)
        moves.append(1)
        if len(moves) == 1:
            hub.journal.mark_moved('object', hub.robot.steps)
            hub.carried_surface_reference={'object_id':'object','validity':'suspended',
                'suspended_at_step':hub.robot.steps,'source_step':20,'local_xyz_m':np.zeros((30,3))}
        return {**output, 'converged': len(moves) > 1,
            'position_error_m': .02 if len(moves) == 1 else 0., 'state': hub.robot.state()}
    hub.executor.move = move
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, answer = 'execute_actions', submission(30)
            elif self.calls == 2:
                current = json.loads((hub.run/'agent_public/current.json').read_text())
                assert current['current_geometry_candidates'][0]['evidence_id'] == 'current_001'
                assert 'current_geometry/current_001/candidates_0.png' in json.dumps(messages)
                # Preparing fresh evidence does not assert new object motion.
                assert hub.journal.data['objects']['object']['invalidated_at_step'] == 31
                name, answer = 'refresh_object', {'mode': 'bind', 'object_id': 'object',
                    'evidence_id': 'current_001', 'candidate_ids': ['G1'], 'decision': 'select',
                    'selection_scope': 'visible_part', 'description': 'chosen visible fragment',
                    'visible_evidence': 'G1 in the current post-failure candidates'}
            elif self.calls == 3:
                name, answer = 'execute_actions', submission(31)
            else:
                name, answer = 'finish', {'status': 'blocked', 'summary': 'Fixture only',
                    'goal_evidence': [{'goal': 'place the object', 'assessment': 'unknown',
                        'image_paths': ['observations/obs_00032/agentview.png'], 'visible_evidence': 'Fixture only'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(answer)}}]}}
    client = Client()
    result = run_single_agent(hub, client, [], handoff, max_turns=5, max_batches=2)
    assert result['status'] == 'single_agent_blocked'
    assert client.calls == 4 and actions == ['move', 'move']
    assert len(result['subtasks']) == 2 and result['subtasks'][0]['commands_completed'] is False
    event = next(value for value in events if value['event'] == 'single_a_failure_geometry_prepared')
    assert event['robot_actions'] == event['extra_a_requests'] == event['sam_requests'] == 0


def test_tracking_failure_alone_does_not_block_on_object_relocalization(episode):
    hub, handoff, _, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    DirectExecutor(hub, handoff, artifacts)
    hub.journal.mark_moved('object', 20)  # Old invalidation is not a new physical change.
    events = []; hub.emit = events.append
    artifacts.prepare_geometry_recovery = lambda *args: pytest.fail('Tracking alone must not start registration')
    stage = {'source_step': hub.robot.steps, 'plan': {'steps': [{'target_id': 'object'}]}}
    hub.robot.steps += 1
    # Normal carried motion invalidates world pixels at this new step; it
    # does not imply that a fresh object-to-hand registration is needed.
    hub.journal.mark_moved('object',hub.robot.steps)
    assert artifacts.prepare_failure_geometry(stage, {'failed_step_index': 0}) is None
    assert events[-1]['event'] == 'single_a_failure_geometry_deferred'
    assert hub.journal.data['objects']['object']['invalidated_at_step'] == hub.robot.steps


def test_failure_table_keeps_current_episode_pose_evidence_and_unresolved_statuses(episode):
    hub, handoff, _, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    for number in range(15):
        artifacts.ledger['batches'].append({'batch': number+1, 'expanded_steps': [
            {'status': 'attempted_failed', 'location': {'action_index': 0, 'waypoint_index': 0},
                'intent': 'fixture attempt', 'target_id': 'object',
                'selection_expression': {'offset_m': [number*.001, 0, 0]},
                'dispatches': [{'parameters': {'position': [number*.001, 0, 1]}}],
                'actual_result': {'result': {'state': {'step': number, 'position': [0, 0, 1]},
                    'position_error_m': .02, 'tracking': {'last_command_consecutive_steps': 81}}}},
            {'status': 'not_dispatched', 'selection_expression': {'anchor': 'current_eef'}}]})
    before = deepcopy(artifacts.ledger)
    table = artifacts.failure_attempts()
    assert table['total_failed_attempts'] == 15 and table['omitted_older_attempts'] == 3
    assert len(table['attempts']) == 12 and table['attempts'][0]['batch'] == 4
    last = table['attempts'][-1]
    assert last['tracking']['last_command_consecutive_steps'] == 81
    assert last['recorded_absolute_goal']['position'] == [.014, 0, 1]
    assert last['actual_stop']['step'] == 14
    assert last['history_path'] == 'receipts/sketch_015.json'
    table['attempts'][-1]['original_expression']['offset_m'][0] = 999
    assert artifacts.ledger == before


def begin_history_fixture(episode, answer=None):
    hub, handoff, actions, _ = episode
    artifacts = EpisodeArtifacts(hub, handoff)
    executor = DirectExecutor(hub, handoff, artifacts)
    answer = submission(hub.robot.steps) if answer is None else answer
    plan, stage = prepare_actions(executor, artifacts, answer)
    hub.single_a_requests = 1
    artifacts.begin_execution(plan, stage, 1, submission=answer, bound_submission=answer)
    return hub, artifacts, executor, plan, stage, actions


def test_failed_middle_waypoint_preserves_original_expressions_and_unexecuted_suffix(episode):
    from sketchik.single_agent import working_current
    answer = submission(30); answer.pop('intent')
    answer['actions'][0]['selection']['waypoints'] = [binding(), binding(), binding()]
    answer['actions'].append({'primitive': 'set_gripper', 'target_id': 'object',
        'intent': 'close after reaching the rim', 'selection': {'command': 'close', 'steps': 20}})
    hub, artifacts, executor, plan, stage, actions = begin_history_fixture(episode, answer)
    move = hub.executor.move
    def fail_second(**params):
        value = move(**params)
        if len(actions) == 2:
            value.update(converged=False, position_error_m=.02, motion_progress={'recent_progress': False})
        return value
    hub.executor.move = fail_second
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    artifacts.finish_execution(receipt)
    assert actions == ['move', 'move'] and receipt['failed_location']['waypoint_index'] == 1
    history = json.loads(artifacts.files.read_text_file('receipts/sketch_001.json')['content'])
    assert history['model_submission'] == answer and 'intent' not in history['model_submission']
    assert [x['status'] for x in history['expanded_steps']] == [
        'completed', 'attempted_failed', 'not_dispatched', 'not_dispatched']
    assert history['expanded_steps'][1]['physical_steps_advanced'] == 1
    assert not history['expanded_steps'][2]['dispatches']
    progress = receipt['execution_progress']
    assert progress['unexecuted_suffix'][0]['selection_expression'] == binding()
    assert 'parameters' not in progress['unexecuted_suffix'][0]
    assert progress['unexecuted_suffix'][1]['intent'] == 'close after reaching the rim'
    working = working_current({'execution_ledger': artifacts.execution_history_index(),
        'recent_execution_receipts': [receipt]})
    assert working['recent_execution_receipts'][0]['execution_progress'] == progress
    assert working['recent_execution_receipts'][0]['steps'][1]['location']['waypoint_index'] == 1
    assert artifacts.files.list_dir('receipts')['total'] == 3


def test_post_motion_contact_check_does_not_erase_actual_dispatch(episode):
    hub, artifacts, executor, plan, stage, actions = begin_history_fixture(episode)
    def changed(*args):
        error = BindingUnavailable('contact changed after movement')
        error.failure = {'failure_stage': 'carried_contact', 'executed': False}
        raise error
    executor.check_actual_contact = changed
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    artifacts.finish_execution(receipt)
    entry = artifacts.ledger['batches'][0]['expanded_steps'][0]
    assert actions == ['move'] and entry['status'] == 'attempted_failed'
    assert entry['physical_steps_advanced'] == 1
    assert entry['actual_result']['result']['converged'] is True
    assert entry['actual_result']['recovery_failure']['executed'] is False


def test_compile_rejection_keeps_failed_expression_without_invented_motion(episode):
    hub, artifacts, executor, plan, stage, actions = begin_history_fixture(episode)
    def unavailable(*args):
        raise BindingUnavailable('current point does not support this binding')
    executor.compile_step = unavailable
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    artifacts.finish_execution(receipt)
    entry = artifacts.ledger['batches'][0]['expanded_steps'][0]
    assert actions == [] and entry['status'] == 'failed_before_dispatch'
    assert entry['dispatches'] == [] and 'parameters' not in entry
    assert receipt['execution_progress']['failed_or_unknown_steps'][0]['location']['semantic_step_index'] == 0


def test_format_rejection_is_public_attempt_not_an_accepted_sketch(episode):
    hub, handoff, actions, _ = episode
    class Client:
        last_request_metadata = {}
        def complete(self, messages, tools):
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': 'bad', 'type': 'function',
                'function': {'name': 'execute_actions', 'arguments': '{invalid'}}]}}
    result = run_single_agent(hub, Client(), [], handoff, max_turns=1, max_batches=1, max_invalid_replies=1)
    ledger = hub.single_a_execution_ledger
    assert result['status'] == 'single_agent_invalid_output' and actions == []
    assert ledger['batches'] == [] and len(ledger['rejected_attempts']) == 1
    attempt = ledger['rejected_attempts'][0]
    assert attempt['model_arguments'] == '{invalid'
    assert attempt['robot_actions'] == 0 and not attempt['assessment_committed'] and not attempt['batch_accepted']
    assert (hub.run/'agent_public'/attempt['public_path']).exists()


def test_T1_continuation_stays_one_expanded_step_with_two_numeric_results(episode):
    hub, artifacts, executor, plan, stage, actions = begin_history_fixture(episode)
    move = hub.executor.move
    def incomplete(**params):
        result = move(**params)
        result.update(converged=False, position_error_m=.01,
            motion_progress={'recent_progress': True}, continuation_goal=deepcopy(params))
        return result
    def continue_same(failed, *, max_steps):
        assert max_steps == 40
        return move(**failed['continuation_goal'])
    hub.executor.move = incomplete
    hub.executor.continue_motion = continue_same
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    artifacts.finish_execution(receipt)
    entries = artifacts.ledger['batches'][0]['expanded_steps']
    assert actions == ['move', 'move'] and len(entries) == 1
    assert entries[0]['status'] == 'completed' and len(entries[0]['dispatches']) == 1
    assert entries[0]['physical_steps_advanced'] == 2
    assert len(entries[0]['actual_result']['result']['motion_attempts']) == 2


def test_unattributed_physics_is_not_labelled_as_an_unexecuted_suffix(episode):
    hub, artifacts, executor, plan, stage, actions = begin_history_fixture(episode)
    hub.executor.move(position=[0, 0, 1.1], quaternion_xyzw=[0, 0, 0, 1])
    artifacts.interrupt_execution(RuntimeError('unexpected advance outside dispatch recording'))
    batch = artifacts.ledger['batches'][0]
    assert actions == ['move'] and batch['receipt']['unattributed_physical_advance']
    assert batch['receipt']['unrecorded_current_action_possible']
    assert batch['expanded_steps'][0]['status'] == 'dispatch_unverified'
    assert batch['receipt']['execution_progress']['unexecuted_suffix'] == []


def test_historical_sketch_is_readable_after_current_context_reduction(episode):
    from sketchik.stage_context import StageContext
    hub, artifacts, executor, plan, stage, actions = begin_history_fixture(episode)
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    artifacts.finish_execution(receipt)
    current = artifacts.current(executor, hub.run/'reduced_current')
    messages = [{'role': 'system', 'content': 'persistent instructions'},
        {'role': 'user', 'content': 'initial'},
        {'role': 'assistant', 'content': 'old visible planning text'*20000},
        {'role': 'user', 'content': 'current follow-up'}]
    context = StageContext(max_chars=100, max_messages=2, keep_messages=1, keep_chars=30)
    context.prepare_boundary(messages, current, hub.run/'a_context_history_test', hub.emit, force=True)
    active, _ = context.apply(messages)
    assert len(json.dumps(active)) < len(json.dumps(messages))
    path = current['execution_ledger'][0]['history_path']
    old = json.loads(artifacts.files.read_text_file(path)['content'])
    assert old['model_submission']['actions'][0]['intent'] == 'raise'
    assert old['expanded_steps'][0]['location']['waypoint_index'] == 0
    assert old['expanded_steps'][0]['status'] == 'completed' and actions == ['move']


def test_history_dispatch_path_ik_failure_advances_no_physics(episode):
    from sketchik.kinematics import IKError
    hub, artifacts, executor, plan, stage, actions = begin_history_fixture(episode)
    def reject(**params):
        raise IKError('mock execution path IK unavailable')
    hub.executor.move = reject
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    artifacts.finish_execution(receipt)
    entry = artifacts.ledger['batches'][0]['expanded_steps'][0]
    assert actions == [] and hub.robot.steps == 30
    assert entry['status'] == 'dispatched_no_physics'
    assert len(entry['dispatches']) == 1 and entry['physical_steps_advanced'] == 0
    assert receipt['execution_progress']['unexecuted_suffix'] == []
    assert artifacts.live_dispatches == []


def test_history_episode_end_preserves_undispatched_suffix_and_filters_oracle(episode):
    answer = submission(30)
    answer['actions'][0]['selection']['waypoints'] = [binding(), binding()]
    hub, artifacts, executor, plan, stage, actions = begin_history_fixture(episode, answer)
    move = hub.executor.move
    def ended(**params):
        out = move(**params)
        hub.robot.done = True
        return {**out, 'episode_ended': True, 'official_success': True}
    hub.executor.move = ended
    receipt = execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    artifacts.finish_execution(receipt)
    batch = artifacts.ledger['batches'][0]
    assert actions == ['move'] and hub.robot.steps == 31
    assert [entry['status'] for entry in batch['expanded_steps']] == ['outcome_episode_ended', 'not_dispatched']
    assert receipt['episode_ended'] and not receipt['commands_completed']
    assert len(receipt['execution_progress']['unexecuted_suffix']) == 1
    assert 'official_success' not in artifacts.files.read_text_file('receipts/sketch_001.json')['content']


def test_history_t1_exception_is_unknown_not_an_unexecuted_action(episode):
    hub, artifacts, executor, plan, stage, actions = begin_history_fixture(episode)
    move = hub.executor.move
    def incomplete(**params):
        return {**move(**params), 'converged': False, 'position_error_m': .01,
            'motion_progress': {'recent_progress': True}, 'continuation_goal': deepcopy(params)}
    def interrupted(failed, **kwargs):
        move(**failed['continuation_goal'])
        raise RuntimeError('mock continuation interrupted after physical advance')
    hub.executor.move = incomplete
    hub.executor.continue_motion = interrupted
    with pytest.raises(RuntimeError) as caught:
        execute_sketch(hub, executor, plan, 0, frozen_binding=stage)
    artifacts.interrupt_execution(caught.value)
    batch = artifacts.ledger['batches'][0]; entry = batch['expanded_steps'][0]
    assert actions == ['move', 'move'] and hub.robot.steps == 32
    assert len(entry['dispatches']) == 1 and entry['status'] == 'outcome_unknown'
    assert entry['dispatches'][0]['source_step'] == 30 and entry['source_step_end'] == 32
    assert batch['receipt']['unrecorded_current_action_possible']
    assert batch['receipt']['execution_progress']['unexecuted_suffix'] == []


@pytest.mark.parametrize('has_goal', [False, True])
def test_current_keeps_recorded_goal_envelope_as_hypothesis_without_new_camera_or_action(episode, monkeypatch, has_goal):
    from sketchik.single_agent import working_current
    hub, handoff, actions, _ = episode
    goal = {'pose': {'position': [0, 0, 1], 'quaternion_xyzw': [0, 0, 0, 1]},
        'failure_observed_step': 20, 'hand_configuration_source_step': 30,
        'failure_at_current_physical_step': False,
        'hand_world_bounds_m': [[-.1, -.1, .8], [.1, .1, 1.2]],
        'part_world_bounds_m': [[[-.1, -.1, .8], [.1, .1, 1.2]]],
        'pad_centres_world_m': [[0, 0, 1]],
        'envelope_scope': 'known hand bounds only; not contact certification',
        'new_observation': False, 'collision_checked': False, 'contact_cause_certified': False}
    def saved_hand(hub, description, folder):
        folder.mkdir(parents=True)
        metadata = {'source_step': 30, 'evidence_layers': {}}
        if has_goal: metadata['failed_goal_projection'] = goal
        (folder/'hand_observation.json').write_text(json.dumps(metadata))
        Image.new('RGB', (32, 32), 'white').save(folder/'hand_observation.png')
        return []
    monkeypatch.setattr('sketchik.single_agent.hand_observation_parts', saved_hand)
    artifacts = EpisodeArtifacts(hub, handoff); executor = DirectExecutor(hub, handoff, artifacts)
    current = artifacts.current(executor, hub.run/'goal_card')
    assert hub.robot.steps == 30 and actions == [] and len(current['hand_image_paths']) == 1
    assert ('recorded_goal_hand_geometry' in current) == has_goal
    if has_goal:
        card = current['recorded_goal_hand_geometry']
        assert card['is_current_pose'] is False and card['new_observation'] is False
        assert card['failure_observed_step'] == 20 and card['hand_configuration_source_step'] == 30
        assert card['hand_world_bounds_m'] == goal['hand_world_bounds_m']
        assert card['collision_checked'] is False and card['contact_cause_certified'] is False
        assert working_current(current)['recorded_goal_hand_geometry'] == card


def test_declared_motion_change_does_not_auto_register_after_current_eef_recovery(episode,monkeypatch):
    hub,handoff,actions,_=episode
    monkeypatch.setattr(EpisodeArtifacts,'prepare_geometry_recovery',
        lambda *a:pytest.fail('A changed_objects is not a request to rigid-register every named object'))
    class Client:
        calls=0
        last_request_metadata={}
        def complete(self,messages,tools):
            self.calls+=1
            if self.calls==1:
                name,answer='execute_actions',submission(30)
                answer['changed_objects']=['object']
            else:
                assert actions==['move'] and hub.robot.steps==31
                name,answer='finish',{'status':'blocked','summary':'Fixture finished one recovery movement',
                    'goal_evidence':[{'goal':'place the object','assessment':'unknown',
                        'image_paths':['observations/obs_00031/agentview.png'],'visible_evidence':'Current fixture image'}]}
            return {'message':{'role':'assistant','tool_calls':[{'id':str(self.calls),'type':'function',
                'function':{'name':name,'arguments':json.dumps(answer)}}]}}
    client=Client();result=run_single_agent(hub,client,[],handoff,max_turns=3,max_batches=2)
    assert result['status']=='single_agent_blocked' and client.calls==2
    assert hub.journal.data['objects']['object']['invalidated_at_step']==30
    assert actions==['move']


def pose_rebind_case(episode):
    from sketchik.carry_reference import initialize_reference, suspend
    hub, artifacts, executor, result, parts = current_refresh_fixture(episode)
    with np.load(artifacts.geometry_inspections[result['evidence_id']]['cloud_file']) as data:
        visible = data['world_xyz_m'].astype(float)
    # Keep extra originally observed samples hidden by the current view.
    old = np.concatenate([visible-[.01, .02, 0.], visible-[.01, .02, .018]])
    rgb = np.full((len(old),3),80,np.uint8)
    ref = initialize_reference({'object_id':'object','source_step':10,'source_cloud_file':'unused.npz'},
        old-[0,0,1],rgb,3,np.arange(len(old)),world_xyz_m=old)
    suspend(ref,20,'actual relative-pose change',kind='relative_pose_changed')
    hub.carried_surface_reference=ref;hub.carried_target_id='object'
    hub._fixture_state['gripper_command']=1.
    from sketchik.perception import saved_frame
    hub.observations={artifacts.latest['observation_id']:{camera:saved_frame(hub.run,artifacts.latest['observation_id'],camera)
        for camera in ('agentview','robot0_eye_in_hand')}}
    hub.episode_state.data['grasp_epoch']=3
    hub.episode_state.data['contact_warning']={'reason':'keep actual contact review','source_step':20}
    request={'mode':'bind','object_id':'object','evidence_id':result['evidence_id'],
        'candidate_ids':['G1'],'decision':'select','selection_scope':'visible_part',
        'point_set_scope':'one_rigid_object','rebind_held_pose':True,
        'description':'current clean patch of the same held object','visible_evidence':'visible local surface'}
    registration={'status':'accepted','source_step':30,'search_complete':True,'wall_s':.01,
        'reason':'supported_unique_fit','transform':{'rotation':np.eye(3).tolist(),'translation_m':[.01,.02,0.]}}
    return hub,artifacts,request,ref,old,visible,registration


def test_explicit_current_patch_rebuilds_same_grasp_full_reference_and_keeps_old_anchors_stale(episode,monkeypatch):
    hub,artifacts,request,ref,old,visible,registration=pose_rebind_case(episode)
    before=deepcopy(ref);warning=deepcopy(hub.episode_state.data['contact_warning']);calls=[]
    def register(template,sources,inspection,*args,**kwargs):
        calls.append(template['world_xyz_m'].copy())
        with np.load(inspection['cloud_file']) as data:assert len(data['world_xyz_m'])==len(visible)
        np.testing.assert_array_equal(template['world_xyz_m'],old)
        return registration
    monkeypatch.setattr('sketchik.object_registration.register_object',register)
    bound,_,changed=artifacts.refresh_object(request,artifacts.validate_refresh(request,30))
    assert changed and len(calls)==1
    assert bound['carried_pose_rebind']['status']=='relocalized'
    assert not bound['carried_pose_rebind']['holding_certified']
    assert hub.carried_surface_reference is ref
    np.testing.assert_allclose(ref['local_xyz_m'],old+[.01,.02,-1.])
    np.testing.assert_array_equal(ref['point_ids'],before['point_ids'])
    np.testing.assert_array_equal(ref['rgb'],before['rgb'])
    np.testing.assert_array_equal(ref['original_local_xyz_m'],before['original_local_xyz_m'])
    assert len(ref['local_xyz_m'])==2*len(visible)
    assert ref['reference_id']==before['reference_id'] and ref['grasp_epoch']==3 and ref['relation_revision']==1
    assert hub.episode_state.data['contact_warning']==warning
    # The explicit current rebind is a new visual declaration of pose change;
    # its own current points remain usable, earlier world anchors do not.
    assert hub.journal.data['objects']['object']['invalidated_at_step']==30
    with pytest.raises(ValueError,match='superseded'):
        artifacts.validate_point(binding({'kind':'cloud_point','view_id':'object_face','pixel_xy':[1,1]}),'object')
    assert hub.robot.steps==30


@pytest.mark.parametrize('problem',['not_requested','mixed','unavailable_fit','snapshot_changed'])
def test_current_pose_bind_does_not_guess_or_erase_suspended_reference(episode,monkeypatch,problem):
    hub,artifacts,request,ref,old,visible,registration=pose_rebind_case(episode)
    if problem=='not_requested':request.pop('rebind_held_pose')
    if problem=='mixed':request['point_set_scope']='mixed_or_unknown'
    before=deepcopy(ref);warning=deepcopy(hub.episode_state.data['contact_warning']);calls=[]
    def register(*args,**kwargs):
        calls.append(1)
        if problem=='unavailable_fit':return {**registration,'status':'unavailable','transform':None,'reason':'ambiguous_pose_or_identity'}
        if problem=='snapshot_changed':hub._fixture_state['step']=31;hub.robot.steps=31
        return registration
    monkeypatch.setattr('sketchik.object_registration.register_object',register)
    if problem=='snapshot_changed':
        with pytest.raises(RuntimeError,match='state changed'):
            artifacts.refresh_object(request,artifacts.validate_refresh(request,30))
        bound={};changed=False
    else:
        bound,_,changed=artifacts.refresh_object(request,artifacts.validate_refresh(request,30))
        assert changed
    if problem=='snapshot_changed':assert bound=={}
    elif problem=='not_requested':assert 'carried_pose_rebind' not in bound
    else:assert bound['carried_pose_rebind']['status']=='unavailable'
    assert bool(calls)==(problem in ['unavailable_fit','snapshot_changed'])
    assert ref['validity']=='suspended' and ref['relation_revision']==0
    np.testing.assert_array_equal(ref['local_xyz_m'],before['local_xyz_m'])
    np.testing.assert_array_equal(ref['point_ids'],before['point_ids'])
    assert hub.episode_state.data['contact_warning']==warning


def test_explicit_assessment_after_pose_rebind_retires_contact_warning_without_resuspending_reference(episode,monkeypatch):
    hub,artifacts,request,ref,old,visible,registration=pose_rebind_case(episode)
    monkeypatch.setattr('sketchik.object_registration.register_object',lambda *a,**k:registration)
    warning=deepcopy(hub.episode_state.data['contact_warning'])
    result,_,_=artifacts.refresh_object(request,artifacts.validate_refresh(request,30))
    assert result['carried_pose_rebind']['status']=='relocalized'
    assert hub.episode_state.data['contact_warning']==warning  # Binding itself cannot acknowledge it.
    apply_assessment(hub,{'held_object_id':'object','held_pose_relation':'changed'})
    assert ref['validity']=='active' and ref['relation_revision']==1
    assert 'contact_warning' not in hub.episode_state.data  # Independent explicit A review retires contact baseline.
    # A genuinely new physical change still invalidates this revised relation.
    hub.robot.steps=hub._fixture_state['step']=31
    apply_assessment(hub,{'held_object_id':'object','held_pose_relation':'changed'})
    assert ref['validity']=='suspended' and ref['suspended_at_step']==31


@pytest.mark.parametrize('fit',['accepted','unavailable','mixed','missing_capture'])
def test_current_pose_rebind_declares_active_relation_changed_without_an_action_round(episode,monkeypatch,fit):
    from sketchik.observed_dimensions import carried_dimensions, write_measurement
    hub,artifacts,request,ref,old,visible,registration=pose_rebind_case(episode)
    ref['validity']='active'
    for key in ('suspended_at_step','suspension_kind','suspension_reason'):ref.pop(key,None)
    if fit=='mixed':request['point_set_scope']='mixed_or_unknown'
    if fit=='missing_capture':ref['original_world_xyz_m']=None
    write_measurement(hub.run,'object',carried_dimensions(ref['local_xyz_m'],10))
    hub.journal.update_object('unrelated',{'invalidated_at_step':12})
    hub.episode_state.data['contact_baseline']={'source_step':20,'object_id':'object','gap_m':.02}
    before=deepcopy(ref);episode_before=deepcopy(hub.episode_state.data);calls=[]
    def register(*args,**kwargs):
        calls.append(1)
        if fit=='unavailable':return {**registration,'status':'unavailable','transform':None,'reason':'ambiguous_pose_or_identity'}
        return registration
    monkeypatch.setattr('sketchik.object_registration.register_object',register)
    result,_,_=artifacts.refresh_object(request,artifacts.validate_refresh(request,30))
    rebuilt=result['carried_pose_rebind']
    assert rebuilt['status']==('relocalized' if fit=='accepted' else 'unavailable')
    assert ref['validity']==('active' if fit=='accepted' else 'suspended')
    assert ref['relation_revision']==(1 if fit=='accepted' else 0)
    assert bool(calls)==(fit in ('accepted','unavailable'))
    assert hub.journal.data['objects']['object']['invalidated_at_step']==30
    assert hub.journal.data['objects']['unrelated']['invalidated_at_step']==12
    assert hub.episode_state.data==episode_before  # No implicit held/contact review.
    dimensions=json.loads((hub.run/'dimensions.json').read_text())['objects']['object']
    assert dimensions['dimensions_valid'] is False and dimensions['object_dimensions_m'] is None
    assert ref['reference_id']==before['reference_id'] and ref['grasp_epoch']==3
    np.testing.assert_array_equal(ref['point_ids'],before['point_ids'])
    assert hub.robot.steps==30


def test_repeating_same_current_pose_rebind_does_not_refit_or_retire_new_relation(episode,monkeypatch):
    hub,artifacts,request,ref,old,visible,registration=pose_rebind_case(episode)
    calls=[]
    monkeypatch.setattr('sketchik.object_registration.register_object',lambda *a,**k:(calls.append(1) or registration))
    first,_,_=artifacts.refresh_object(request,artifacts.validate_refresh(request,30))
    assert first['carried_pose_rebind']['status']=='relocalized'
    local=ref['local_xyz_m'].copy();revision=ref['relation_revision']
    second,_,_=artifacts.refresh_object(request,artifacts.validate_refresh(request,30))
    assert second['carried_pose_rebind']['status']=='already_relocalized'
    assert calls==[1] and ref['validity']=='active' and ref['relation_revision']==revision
    np.testing.assert_array_equal(ref['local_xyz_m'],local)


@pytest.mark.parametrize('problem',['wrong_held_identity','wrong_epoch','open','stale_snapshot'])
def test_invalid_pose_rebind_premise_cannot_suspend_current_reference(episode,monkeypatch,problem):
    hub,artifacts,request,ref,old,visible,registration=pose_rebind_case(episode)
    ref['validity']='active'
    for key in ('suspended_at_step','suspension_kind','suspension_reason'):ref.pop(key,None)
    if problem=='wrong_held_identity':hub.carried_target_id='another'
    if problem=='wrong_epoch':hub.episode_state.data['grasp_epoch']=4
    if problem=='open':hub._fixture_state['gripper_command']=-1.
    if problem=='stale_snapshot':request['source_step']=29
    episode_before=deepcopy(hub.episode_state.data)
    monkeypatch.setattr('sketchik.object_registration.register_object',lambda *a,**k:pytest.fail('Invalid identity must not start registration'))
    if problem=='stale_snapshot':
        with pytest.raises(ValueError,match='source_step'):
            artifacts.validate_refresh(request,30)
    else:
        result,_,_=artifacts.refresh_object(request,artifacts.validate_refresh(request,30))
        assert result['carried_pose_rebind']['status']=='unavailable'
    assert ref['validity']=='active' and ref['relation_revision']==0
    assert hub.journal.data['objects']['object']['invalidated_at_step']==20
    assert hub.episode_state.data==episode_before


def test_pose_rebind_retains_native_capture_frame_of_an_already_registered_cloud(episode,monkeypatch):
    hub,artifacts,request,ref,old,visible,registration=pose_rebind_case(episode)
    source=hub.run/'previously_registered.npz'
    captured=old-[.2,0,0]
    np.savez(source,world_xyz_m=old,original_world_xyz_m=captured,
        pixel_xy=np.zeros((len(old),2),int),source_index=np.zeros(len(old),np.uint16))
    ref['source_cloud_file']=str(source)
    ref['source_observations']=[{'observation_id':'original_capture','camera':'agentview'}]
    def register(template,*args,**kwargs):
        np.testing.assert_array_equal(template['world_xyz_m'],old)
        np.testing.assert_array_equal(template['original_world_xyz_m'],captured)
        return registration
    monkeypatch.setattr('sketchik.object_registration.register_object',register)
    result,_,_=artifacts.refresh_object(request,artifacts.validate_refresh(request,30))
    assert result['carried_pose_rebind']['status']=='relocalized'
    np.testing.assert_array_equal(ref['original_world_xyz_m'],old)  # Frozen grasp frame is separate.
    np.testing.assert_allclose(ref['local_xyz_m'],old+[.01,.02,-1.])


def test_program_continuation_has_no_fabricated_A_assessment_or_extra_request(episode, monkeypatch):
    hub, handoff, actions, _ = episode
    handoff['objects'][0]['shape_motion'] = 'rigid'
    assessments = []
    original_assessment = apply_assessment
    def assess(hub, answer):
        assessments.append(deepcopy(answer))
        original_assessment(hub, answer)
    monkeypatch.setattr('sketchik.single_agent.apply_assessment', assess)
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, args = 'execute_actions', submission(30)
            elif self.calls == 2:
                assert tools[0]['function']['name'] == 'draft_next_sketch'
                name, args = 'draft_next_sketch', {'status': 'ready', 'reason': 'same known geometry',
                    'preconditions': 'previous motion completed', 'intent': 'second motion',
                    'actions': submission(30)['actions']}
            else:
                assert self.calls == 3 and actions == ['move', 'move']
                assert len(assessments) == 1
                name, args = 'finish', {'status': 'blocked', 'summary': 'Two fixture motions only',
                    'goal_evidence': [{'goal': handoff['task'], 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'No fixture task completion evidence'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls),
                'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}}
    client = Client(); messages = []
    result = run_single_agent(hub, client, messages, handoff, max_turns=5, max_batches=2, async_planning=True)
    assert result['async_planning']['program_continuations'] == 1
    assert result['a_execution_tool_calls'] == 3 and hub.single_a_requests == 3
    ledger = hub.single_a_artifacts.ledger['batches'][1]
    assert ledger['model_submission'] is None
    assert ledger['dispatch_origin']['authorization'] == 'conditional_draft'
    assert not ledger['dispatch_origin']['model_request_issued_for_dispatch']
    assert 'visible_evidence' not in ledger['program_submission'] and 'held_object_id' not in ledger['program_submission']
    assert not any(c['function']['name'] == 'program_prepared_sketch' for m in messages
        if m['role'] == 'assistant' for c in m.get('tool_calls', []))


def test_failed_execution_waits_for_late_draft_and_prepares_successor_during_repair(episode):
    from threading import Event
    hub, handoff, actions, _ = episode
    handoff['objects'][0]['shape_motion'] = 'rigid'
    moved = Event(); draft_finished = Event(); worker_started = Event()
    original_move = hub.executor.move
    def move(**params):
        assert worker_started.wait(5)
        result = original_move(**params)
        if len(actions) == 1:
            moved.set()
            return {**result, 'converged': False, 'position_error_m': .02, 'state': hub.robot.state()}
        return result
    hub.executor.move = move
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            turn = self.calls
            if turn in (1, 3):
                if turn == 3:
                    assert draft_finished.is_set() and actions == ['move']
                    assert hub.async_recovery_candidate is not None
                    assert 'async_recovery_reference' in json.loads((hub.run/'agent_public/current.json').read_text())
                    assert any('async_candidate_discarded' in m.get('content', '') for m in messages if isinstance(m.get('content'), str))
                name, args = 'execute_actions', submission(hub.robot.steps)
                args['intent'] = 'repair from actual stop' if turn == 3 else 'first attempt'
            elif turn in (2, 4):
                assert tools[0]['function']['name'] == 'draft_next_sketch'
                worker_started.set()
                if turn == 2:
                    assert moved.wait(5)
                    draft_finished.set()
                else:
                    # Same persistent A, now preparing the successor while the repair is running.
                    assert any('repair from actual stop' in m.get('content', '') for m in messages if isinstance(m.get('content'), str))
                name, args = 'draft_next_sketch', {'status': 'ready', 'reason': 'existing geometry',
                    'preconditions': 'actual prior motion completed', 'intent': 'new successor' if turn == 4 else 'old nonexecuted successor',
                    'actions': submission(hub.robot.steps)['actions']}
            else:
                assert turn == 5 and actions == ['move']*3
                name, args = 'finish', {'status': 'blocked', 'summary': 'Only fixture execution',
                    'goal_evidence': [{'goal': handoff['task'], 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                        'visible_evidence': 'Fixture only'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(turn), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}}
    client = Client(); messages = []
    result = run_single_agent(hub, client, messages, handoff, max_turns=8, max_batches=3, async_planning=True)
    assert client.calls == 5 and len(result['async_planning']['requests']) == 2
    assert 'discarded' in result['async_planning']['requests'][0]
    assert result['async_planning']['program_continuations'] == 1
    ledger = hub.single_a_artifacts.ledger['batches']
    assert ledger[0]['receipt']['commands_completed'] is False
    assert ledger[1]['model_submission']['intent'] == 'repair from actual stop'
    assert ledger[2]['program_submission']['intent'] == 'new successor'
    old = json.loads((hub.run/'agent_public/sketch/recovery_reference.json').read_text())
    assert not old['executable'] and old['candidate']['intent'] == 'old nonexecuted successor'


@pytest.mark.parametrize('repair_at_terminal', [False, True])
def test_async_grasp_release_continues_without_approval_then_uses_actual_terminal_feedback(episode, repair_at_terminal):
    from sketchik.stage_context import StageContext, closed_tool_dialogue
    hub, handoff, actions, _ = episode
    handoff['objects'][0]['shape_motion'] = 'articulated'  # No extra semantic gate.
    class Client:
        calls = 0
        last_request_metadata = {}
        stage_context = StageContext(max_chars=2000, keep_messages=1, keep_chars=1000, retain_episode_references=True)
        def complete(self, messages, tools):
            self.calls += 1
            names = [t['function']['name'] for t in tools]
            if self.calls == 1:
                name, value = 'execute_actions', submission(hub.robot.steps)
                value['actions'] += [
                    {'primitive': 'set_gripper', 'target_id': 'object', 'intent': 'planned grasp',
                        'selection': {'command': 'close', 'steps': 20}},
                    deepcopy(value['actions'][0])]
            elif self.calls == 2:
                assert names == ['draft_next_sketch']
                name, value = 'draft_next_sketch', {'status': 'ready', 'intent': 'place and release',
                    'reason': 'existing geometry', 'preconditions': 'normal current execution',
                    'actions': [*submission(30)['actions'],
                        {'primitive': 'set_gripper', 'target_id': None, 'intent': 'release',
                            'selection': {'command': 'open', 'steps': 20}},
                        *submission(30)['actions']]}
            elif self.calls == 3:
                assert names == ['draft_next_sketch']
                # Current draft request follows actual close, without a new A holding declaration.
                assert hub.episode_state.data['a_held_claim']['object_id'] is None
                name, value = 'draft_next_sketch', {'status': 'finish_review',
                    'reason': 'The executing release and withdrawal should finish the task'}
            else:
                assert len(tools) == 7
                if self.calls == 4:
                    assert any('async_terminal_actual' in m.get('content', '') for m in messages if isinstance(m.get('content'), str))
                    assert hub.robot.state()['gripper_command'] == -1. and actions.count('set_gripper') == 2
                    active, _ = self.stage_context.apply(messages); closed_tool_dialogue(active)
                    assert 'finish_review' in json.dumps(active)
                if self.calls == 4 and repair_at_terminal:
                    name, value = 'execute_actions', submission(hub.robot.steps)
                else:
                    assert self.calls == (5 if repair_at_terminal else 4)
                    name, value = 'finish', {'status': 'blocked', 'summary': 'No fixture semantic success certification',
                        'goal_evidence': [{'goal': handoff['task'], 'assessment': 'unknown',
                            'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'],
                            'visible_evidence': 'Actual release complete; fixture does not prove the task'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(value)}}]}}
    messages=[]; client=Client()
    result=run_single_agent(hub, client, messages, handoff, max_turns=8, max_batches=3, async_planning=True)
    assert result['status'] == 'single_agent_blocked'
    assert result['async_planning']['program_continuations'] == 1
    assert result['a_execution_requests'] == (5 if repair_at_terminal else 4)
    assert actions == ['move', 'set_gripper', 'move', 'move', 'set_gripper', 'move'] + (['move'] if repair_at_terminal else [])
    ledger=hub.single_a_artifacts.ledger['batches']
    assert ledger[1]['model_submission'] is None and ledger[1]['dispatch_origin']['authorization'] == 'conditional_draft'
    assert 'visible_evidence' not in ledger[1]['program_submission'] and 'held_pose_relation' not in ledger[1]['program_submission']
    epochs=result['async_planning']['continuation_checks'][0]['grasp_epoch_transition']
    assert epochs['prepared'] != epochs['actual']  # Actual new grasp epoch is allowed.
    assert len(result['async_planning']['requests']) == 2
    assert all(m.get('function', {}).get('name') != 'accept_prepared_sketch' for msg in messages
        for m in msg.get('tool_calls', []))
    closed_tool_dialogue(messages)


def test_auto_candidate_cannot_revive_world_points_retired_by_actual_close(episode):
    hub, handoff, actions, _ = episode
    class Client:
        calls = 0
        last_request_metadata = {}
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, value = 'execute_actions', submission(30)
                value['actions'].append({'primitive': 'set_gripper', 'target_id': 'object', 'intent': 'grasp',
                    'selection': {'command': 'close', 'steps': 20}})
            elif self.calls == 2:
                name, value = 'draft_next_sketch', {'status': 'ready', 'reason': 'fixture',
                    'preconditions': 'normal close completes', 'intent': 'use old world point',
                    'actions': submission(30, {'kind': 'cloud_point', 'view_id': 'object_face', 'pixel_xy': [0, 0]})['actions']}
            else:
                assert self.calls == 3 and len(tools) == 7 and actions == ['move', 'set_gripper']
                assert hub.async_recovery_candidate is not None
                name, value = 'finish', {'status': 'blocked', 'summary': 'Old coordinates unavailable',
                    'goal_evidence': [{'goal': handoff['task'], 'assessment': 'unknown',
                        'image_paths': [f'observations/obs_{hub.robot.steps:05d}/agentview.png'], 'visible_evidence': 'Actual close occurred'}]}
            return {'message': {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(value)}}]}}
    result=run_single_agent(hub, Client(), [], handoff, max_turns=6, max_batches=2, async_planning=True)
    gate=result['async_planning']['continuation_checks'][0]
    assert gate['status'] == 'fail'
    assert any(c['check'] == 'current_geometry_preflight' and 'predates' in c['reason'] for c in gate['checks'])
    assert result['action_batches_accepted'] == 1 and result['async_planning']['program_continuations'] == 0
    assert result['binding_or_file_invalid_replies'] == 0
