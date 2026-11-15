import json

import numpy as np
import pytest
from PIL import Image

from sketchik.hand_observation import render_hand_observation


def test_rotated_hand_and_all_observed_points_are_preserved(tmp_path):
    points = np.array([[.2, .3, .4], [.25, .31, .42]])
    rgb = np.array([[10, 20, 30], [100, 110, 120]], dtype=np.uint8)
    before_points, before_rgb = points.copy(), rgb.copy()
    state = {'position': [1., 2., 3.], 'quaternion_xyzw': [0., 0., np.sqrt(.5), np.sqrt(.5)], 'step': 42}
    picture = render_hand_observation(points, rgb, state, tmp_path, [[0, 0, 0], [1, 2, 3]], [[1, 0, 0]])
    assert Image.open(picture).size == (1024, 1024)
    with np.load(tmp_path / 'hand_observation.npz') as data:
        np.testing.assert_array_equal(data['points_world_m'], before_points)
        np.testing.assert_array_equal(data['rgb'], before_rgb)
        np.testing.assert_allclose(data['hand_corners_world_m'].min(axis=0), [-1, 2, 3])
        np.testing.assert_allclose(data['hand_corners_world_m'].max(axis=0), [1, 3, 6])
        np.testing.assert_allclose(data['pad_centres_world_m'], [[1, 3, 3]])
    np.testing.assert_array_equal(points, before_points)
    np.testing.assert_array_equal(rgb, before_rgb)
    meta = json.loads((tmp_path / 'hand_observation.json').read_text())
    assert meta['source_step'] == 42 and meta['point_count'] == 2
    assert meta['frame'] == 'world' and meta['selectable'] is False
    assert meta['complete_object_shape_known'] is False


@pytest.mark.parametrize('count', [0, 1])
def test_empty_or_sparse_surface_still_renders_only_known_hand(tmp_path, count):
    state = {'position': [0, 0, 1], 'quaternion_xyzw': [0, 0, 0, 1], 'source_step': 4}
    picture = render_hand_observation(np.zeros((count, 3)), np.zeros((count, 3), dtype=np.uint8),
                                      state, tmp_path, [[-.01, -.02, -.03], [.01, .02, .03]])
    assert picture.exists()
    with np.load(tmp_path / 'hand_observation.npz') as data:
        assert len(data['points_world_m']) == count
        assert len(data['hand_corners_world_m']) == 8


@pytest.mark.parametrize('field,bad', [('points', [[np.nan, 0, 0]]),
                                        ('rgb', [[np.inf, 0, 0]]),
                                        ('bounds', [[0, 0, 0], [np.inf, 1, 1]]),
                                        ('pads', [[0, np.nan, 0]])])
def test_rejects_nonfinite(tmp_path, field, bad):
    args = dict(points_world=[[0, 0, 0]], rgb=[[0, 0, 0]],
                state={'position': [0, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1], 'step': 0},
                folder=tmp_path, hand_bounds=[[0, 0, 0], [1, 1, 1]], pad_centres=None)
    args[{'points': 'points_world', 'rgb': 'rgb', 'bounds': 'hand_bounds', 'pads': 'pad_centres'}[field]] = bad
    with pytest.raises(ValueError):
        render_hand_observation(**args)


def test_prepare_reuses_current_cloud_and_crops_without_projection(tmp_path):
    from types import SimpleNamespace
    from sketchik.hand_observation import prepare_hand_observation
    state = {'step': 9, 'q': [0] * 7, 'position': [0, 0, 1],
             'quaternion_xyzw': [0, 0, 0, 1]}
    hub = SimpleNamespace(robot=SimpleNamespace(state=lambda: state), executor=None,
                          observations={})
    description = {'source_step': 9, 'observation_id': 'obs_9', 'state': state}
    cloud = tmp_path / 'current_surfaces.npz'
    xyz = np.array([[0, 0, 1], [.22, 0, 1.25], [.221, 0, 1], [0, 0, 1.251]])
    np.savez(cloud, world_xyz_m=xyz, rgb=np.tile([100, 110, 120], (4, 1)))
    result = prepare_hand_observation(hub, description, tmp_path / 'card', cloud_file=cloud,
        gripper_geometry={'hand_bounds_in_eef_tool_m': [[-.01] * 3, [.01] * 3]})
    assert result['metadata']['point_count'] == 2
    assert result['metadata']['source_kind'] == 'current focused cloud'
    assert result['metadata']['observation_id'] == 'obs_9'
    with np.load(tmp_path / 'card' / 'hand_observation.npz') as data:
        np.testing.assert_array_equal(data['points_world_m'], xyz[:2])


def test_prepare_rejects_stale_source_before_writing(tmp_path):
    from types import SimpleNamespace
    from sketchik.hand_observation import prepare_hand_observation
    state = {'step': 10, 'position': [0, 0, 1], 'quaternion_xyzw': [0, 0, 0, 1]}
    hub = SimpleNamespace(robot=SimpleNamespace(state=lambda: state), executor=None)
    description = {'source_step': 9, 'observation_id': 'obs_9', 'state': {'step': 9}}
    with pytest.raises(ValueError, match='current physical'):
        prepare_hand_observation(hub, description, tmp_path,
            gripper_geometry={'hand_bounds_in_eef_tool_m': [[0] * 3, [1] * 3]})
    assert not tmp_path.joinpath('hand_observation.png').exists()


def test_prepare_fake_model_unavailable_without_side_effect(tmp_path):
    from types import SimpleNamespace
    from sketchik.hand_observation import prepare_hand_observation
    state = {'step': 2, 'position': [0, 0, 1], 'quaternion_xyzw': [0, 0, 0, 1]}
    fake_ik = SimpleNamespace(model=SimpleNamespace(), data=None)
    hub = SimpleNamespace(robot=SimpleNamespace(state=lambda: state),
                          executor=SimpleNamespace(ik=fake_ik))
    assert prepare_hand_observation(hub, {'source_step': 2, 'observation_id': 'obs',
        'state': state}, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_prepare_missing_model_does_not_read_description_or_robot(tmp_path):
    from types import SimpleNamespace
    from sketchik.hand_observation import prepare_hand_observation
    assert prepare_hand_observation(SimpleNamespace(executor=SimpleNamespace(ik=SimpleNamespace())),
                                    {}, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_predicted_reference_is_archived_and_labelled_separately_from_observed_surfaces(tmp_path):
    import json
    from sketchik.hand_observation import render_hand_observation
    observed=np.array([[0.,0.,1.]])
    predicted=np.array([[.1,.1,.9],[.1,.1,1.1]])
    state={'position':[0.,0.,1.],'quaternion_xyzw':[0.,0.,0.,1.],'step':2}
    picture=render_hand_observation(observed,np.zeros((1,3)),state,tmp_path,
        [[-.03,-.03,-.02],[.03,.03,.05]],predicted_points_world=predicted)
    assert picture.exists()
    with np.load(tmp_path/'hand_observation.npz') as data:
        np.testing.assert_array_equal(data['points_world_m'],observed)
        np.testing.assert_array_equal(data['predicted_reference_world_m'],predicted)
    meta=json.loads((tmp_path/'hand_observation.json').read_text())
    assert meta['panel_layout'] == ['current_world_XY','current_world_XZ','current_world_YZ','current_world_3D']
    assert meta['evidence_layers']['rgb_surfaces']['source_step'] == 2
    assert meta['evidence_layers']['magenta_shape']['present'] is True
    assert meta['evidence_layers']['magenta_shape']['new_observation'] is False
    assert meta['point_count']==1 and meta['predicted_reference_point_count']==2
    assert meta['predicted_reference_is_new_observation'] is False and meta['carried_attachment_certified'] is False


def test_failed_goal_projects_same_robot_parts_without_moving_observed_scene(tmp_path):
    from itertools import product
    from scipy.spatial.transform import Rotation
    observed=np.array([[.1,.2,.3],[.4,.5,.6]])
    rgb=np.array([[1,2,3],[4,5,6]],np.uint8)
    state={'step':20,'position':[0,0,1],'quaternion_xyzw':[0,0,0,1]}
    parts=np.array([[[-.1,-.1,0],[.1,.1,.02]],
                    [[-.04,0,.02],[-.02,.01,.08]],[[.02,0,.02],[.04,.01,.08]]])
    local=np.array([[.01,.02,.03]])
    goal={'pose':{'position':[1,2,3],'quaternion_xyzw':[0,0,np.sqrt(.5),np.sqrt(.5)]},
          'failure_observed_step':15,'source':'current_recovery_failure'}
    picture=render_hand_observation(observed,rgb,state,tmp_path,[[-.1,-.1,0],[.1,.1,.08]],
        [[-.03,0,.06],[.03,0,.06]],hand_parts=parts,failed_goal=goal,carried_local=local)
    assert Image.open(picture).size==(1024,1024)
    rot=Rotation.from_quat(goal['pose']['quaternion_xyzw']).as_matrix()
    corners=np.array([list(product(*zip(part[0],part[1]))) for part in parts])
    with np.load(tmp_path/'hand_observation.npz') as data:
        np.testing.assert_array_equal(data['points_world_m'],observed)
        np.testing.assert_array_equal(data['rgb'],rgb)
        np.testing.assert_allclose(data['actual_hand_parts_world_m'],corners+[0,0,1])
        np.testing.assert_allclose(data['failed_goal_hand_parts_world_m'],corners@rot.T+[1,2,3])
        np.testing.assert_allclose(data['failed_goal_carried_hypothesis_world_m'],local@rot.T+[1,2,3])
        np.testing.assert_allclose(data['failed_goal_pad_centres_world_m'],
                                   np.array([[-.03,0,.06],[.03,0,.06]])@rot.T+[1,2,3])
    metadata = json.loads((tmp_path/'hand_observation.json').read_text())
    assert metadata['panel_layout'][-1] == 'unreached_goal_world_3D'
    assert metadata['evidence_layers']['orange_hand_cyan_shape']['is_current_pose'] is False
    assert metadata['evidence_layers']['known_robot_hand_and_pads']['source_step'] == 20
    meta=metadata['failed_goal_projection']
    assert meta['hand_part_count']==3 and meta['carried_hypothesis_point_count']==1
    assert metadata['panel_layout'][:3] == ['current_and_recorded_goal_world_'+plane for plane in ('XY','XZ','YZ')]
    assert meta['hand_configuration_source_step'] == 20
    assert meta['failure_at_current_physical_step'] is False
    expected = corners@rot.T+[1,2,3]
    np.testing.assert_allclose(meta['hand_world_bounds_m'], [expected.reshape(-1,3).min(0), expected.reshape(-1,3).max(0)])
    np.testing.assert_allclose(meta['part_world_bounds_m'], [[box.min(0), box.max(0)] for box in expected])
    assert 'excludes arm ancestors' in meta['envelope_scope']
    assert meta['pose']==goal['pose'] and meta['failure_observed_step']==15
    assert all(meta[key] is False for key in ('observed_scene_moved','new_observation','selectable',
                                             'collision_checked','contact_cause_certified'))


def test_failed_goal_without_associated_object_does_not_transform_scene_as_carried(tmp_path):
    render_hand_observation([[0,0,1]],[[1,2,3]],
        {'step':2,'position':[0,0,1],'quaternion_xyzw':[0,0,0,1]},tmp_path,
        [[-.01]*3,[.01]*3],failed_goal={'pose':{'position':[1,2,3],'quaternion_xyzw':[0,0,0,1]}})
    with np.load(tmp_path/'hand_observation.npz') as data:
        assert data['failed_goal_carried_hypothesis_world_m'].shape==(0,3)
        np.testing.assert_array_equal(data['points_world_m'],[[0,0,1]])


@pytest.mark.parametrize('pose',[
    {'position':[np.nan,0,0],'quaternion_xyzw':[0,0,0,1]},
    {'position':[0,0,0],'quaternion_xyzw':[0,0,0,2]}])
def test_failed_goal_rejects_invalid_pose_without_writing(tmp_path,pose):
    with pytest.raises(ValueError,match='Failed goal'):
        render_hand_observation([[0,0,1]],[[1,2,3]],
            {'step':2,'position':[0,0,1],'quaternion_xyzw':[0,0,0,1]},tmp_path,
            [[-.01]*3,[.01]*3],failed_goal={'pose':pose})
    assert not (tmp_path/'hand_observation.npz').exists()


def test_failed_goal_context_uses_saved_absolute_goal_and_actual_failure_not_old_selector():
    from sketchik.hand_observation import failed_goal_context
    current={'position':[1,2,3],'quaternion_xyzw':[0,0,0,1]}
    old={'position':[9,8,7],'quaternion_xyzw':[0,0,0,1]}
    history={'motions':[{'converged':True,'goal':old,'actual_stop':{'step':100}},
                        {'converged':False,'goal':old,'actual_stop':{'step':90}}],
             'compile_rejections':[{'target_pose':current,'source_step':95,'failure_stage':'ik_path'}]}
    assert failed_goal_context(outcome=history)['pose']==current
    failure={'original_absolute_goal':old,'compiled_from_selection':{'compiled_pose':current},
             'actual_motion':{'state':{'step':101}},'failure_stage':'motion','executed':True}
    chosen=failed_goal_context(failure,history)
    assert chosen['pose']==old and chosen['failure_observed_step']==101
    assert chosen['source']=='current_recovery_failure'
    assert failed_goal_context(outcome={'motions':[{'converged':True,'goal':old}]}) is None


def test_full_robot_context_reuses_one_card_and_keeps_local_and_scene_points_exact(tmp_path):
    from itertools import product
    from PIL import Image
    corners = np.asarray(list(product((-.1, .1), (-.2, .2), (.5, 1.))))
    geometry = {'actual_parts': [{'body': 'known_arm', 'corners_world_m': corners.tolist()}],
        'recorded_goal_parts': [{'body': 'known_arm', 'corners_world_m': (corners+[.2, 0, 0]).tolist()}],
        'actual_source_step': 2, 'goal_status': 'exact_recorded_joint_branch'}
    points, colors = np.array([[0., 0., 1.]]), np.array([[1, 2, 3]])
    scene = np.array([[.3, .3, .8], [.1, -.1, .7]])
    rgb = np.array([[4, 5, 6], [7, 8, 9]])
    picture = render_hand_observation(points, colors,
        {'step': 2, 'position': [0, 0, 1], 'quaternion_xyzw': [0, 0, 0, 1]}, tmp_path,
        [[-.01]*3, [.01]*3], robot_geometry=geometry, scene_points_world=scene, scene_rgb=rgb)
    assert Image.open(picture).size == (1024, 1024)
    assert len(list(tmp_path.glob('*.png'))) == 1
    metadata = json.loads((tmp_path/'hand_observation.json').read_text())
    assert metadata['known_robot_geometry'] == geometry
    assert metadata['panel_layout'][-1] == 'actual_and_recorded_joint_goal_full_robot_world_3D'
    assert metadata['evidence_layers']['known_full_robot']['collision_checked'] is False
    with np.load(tmp_path/'hand_observation.npz') as data:
        np.testing.assert_array_equal(data['points_world_m'], points)
        np.testing.assert_array_equal(data['rgb'], colors)
        np.testing.assert_array_equal(data['robot_context_points_world_m'], scene)
        np.testing.assert_array_equal(data['robot_context_rgb'], rgb)
        np.testing.assert_array_equal(data['actual_robot_parts_world_m'][0], corners)


@pytest.mark.parametrize('invalidated',[False,True])
@pytest.mark.parametrize('source_kind',['scan_cloud','observed_component'])
def test_prepare_projects_only_valid_associated_carried_hypothesis(tmp_path,invalidated,source_kind):
    from types import SimpleNamespace
    from sketchik.hand_observation import prepare_hand_observation
    state={'step':9,'position':[0,0,1],'quaternion_xyzw':[0,0,0,1],'gripper_command':1.}
    pose={'position':[1,2,3],'quaternion_xyzw':[0,0,0,1]}
    local=np.array([[.01,.02,.03]])
    episode={'last_failed_stage_outcome':{'motions':[{'converged':False,'goal':pose,
                                                    'actual_stop':{'step':8}}]},
             'carry_premise':'invalidated_by_contact_change' if invalidated else 'supported_hypothesis'}
    hub=SimpleNamespace(robot=SimpleNamespace(state=lambda:state),executor=None,carried_target_id='mug',
        episode_state=SimpleNamespace(data=episode),carried_surface_reference={
            'source_kind':source_kind,'object_id':'mug','source_step':7,'local_xyz_m':local})
    cloud=tmp_path/'cloud.npz';np.savez(cloud,world_xyz_m=[[0,0,1]],rgb=[[1,2,3]])
    result=prepare_hand_observation(hub,{'source_step':9,'observation_id':'obs','state':state},
        tmp_path/'card',cloud_file=cloud,
        gripper_geometry={'hand_bounds_in_eef_tool_m':[[-.01]*3,[.01]*3]},carried_local=local)
    meta=result['metadata']['failed_goal_projection']
    assert meta['pose']==pose
    assert meta['carried_hypothesis_point_count']==(0 if invalidated else 1)
    with np.load(tmp_path/'card/hand_observation.npz') as data:
        assert len(data['predicted_reference_world_m'])==(0 if invalidated else 1)
    if not invalidated:
        assert result['metadata']['predicted_reference_source']['kind']==source_kind
        assert result['metadata']['predicted_reference_source']['source_step']==7
