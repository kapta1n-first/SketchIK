from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from sketchik.observed_clearance import ObservedClearance
import sketchik.observed_clearance as observed_clearance
from sketchik.program_scan import describe_opening


@pytest.mark.parametrize('measured,expected_status,expected_fraction',[
    (1.,'supported',1.),(.5,'insufficient_visibility',None),(2.,'contradicted',0.)])
def test_visible_support_excludes_hidden_back_and_nearer_occluders(measured,expected_status,expected_fraction):
    from sketchik.observed_clearance import visible_reference_support
    x,y=np.meshgrid(np.arange(30,70,4),np.arange(30,70,4))
    front=np.column_stack([(x.ravel()-50)/100,(y.ravel()-50)/100,np.ones(x.size)])
    points=np.vstack([front,front*2])  # rear surfaces project behind the same front pixels
    original=points.copy()
    frame={'depth_m':np.full((100,100),measured),'world_from_camera':np.eye(4),
        'K':np.array([[100,0,50],[0,100,50],[0,0,1]]),'camera':'fixture','step':10}
    report=visible_reference_support(points,[frame])
    assert report['status']==expected_status and report['support_fraction']==expected_fraction
    assert report['testable_points']==(100 if expected_fraction is not None else 0)
    np.testing.assert_array_equal(points,original)


def test_visible_support_accepts_either_current_camera_and_real_camera_transform():
    from sketchik.observed_clearance import visible_reference_support
    from scipy.spatial.transform import Rotation
    x,y=np.meshgrid(np.arange(30,70,4),np.arange(30,70,4))
    local=np.column_stack([(x.ravel()-50)/100,(y.ravel()-50)/100,np.ones(x.size)])
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',[.2,-.4,.8]).as_matrix()
    transform[:3,3]=[.3,-.6,1.2];points=local@transform[:3,:3].T+transform[:3,3]
    frame={'depth_m':np.ones((100,100)),'world_from_camera':transform,
        'K':np.array([[100,0,50],[0,100,50],[0,0,1]]),'camera':'fixture','step':10}
    hidden={**frame,'depth_m':np.full((100,100),.5)}
    report=visible_reference_support(points,[hidden,frame])
    assert report['status']=='supported' and report['support_fraction']==1.


def test_convex_mesh_inside_matches_dense_planes_with_boundary_and_margin():
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(17)
    vertices = rng.normal(size=(80, 3))
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    equations = ConvexHull(vertices).equations
    boundary = vertices[0]
    points = np.vstack((rng.uniform(-1.4, 1.4, (700, 3)),
                        [0, 0, 0], boundary, boundary * 1.02, boundary * 1.08))
    checker = ObservedClearance.__new__(ObservedClearance)
    checker.ik = SimpleNamespace(data=SimpleNamespace(
        geom_xpos=np.zeros((1, 3)), geom_xmat=np.eye(3).reshape(1, 9)))
    shape = {'index': 0, 'kind': 7, 'size': np.zeros(3), 'equations': equations}
    for margin in (0., .02, .1):
        expected = np.all(points @ equations[:, :3].T + equations[:, 3] <= margin, axis=1)
        np.testing.assert_array_equal(checker.inside(shape, points, margin), expected)


def test_convex_mesh_inside_bounds_each_point_plane_block(monkeypatch):
    rng = np.random.default_rng(18)
    points = rng.uniform(-.1, .1, (9001, 3))
    equations = np.column_stack((rng.normal(size=(257, 3)), np.full(257, -1.)))
    seen = []
    original = observed_clearance._halfspace_block_inside

    def record_block(local, planes, margin):
        seen.append((len(local), len(planes)))
        assert len(local) <= 4096 and len(planes) <= 64
        return original(local, planes, margin)

    monkeypatch.setattr(observed_clearance, '_halfspace_block_inside', record_block)
    actual = observed_clearance._convex_mesh_inside(points, equations, .012)
    expected = np.all(points @ equations[:, :3].T + equations[:, 3] <= .012, axis=1)
    np.testing.assert_array_equal(actual, expected)
    assert len(seen) >= 3 * 5
    assert any(count == 4096 for count, _ in seen)
    assert any(count == 64 for _, count in seen)


def test_opening_plane_uses_labelled_rim_and_rejects_collinearity():
    R = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [0, -1., 1.]
    frame = {'rgb': np.zeros((100, 100, 3), np.uint8), 'depth_m': np.ones((100, 100)),
             'K': np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1.]]), 'world_from_camera': T, 'step': 0}
    geometry, _ = describe_opening(frame, [10, 10, 90, 90], [[20, 20], [80, 20], [20, 80], [80, 80]])
    np.testing.assert_allclose(geometry['proposed_surface_normal_world'], [0, -1, 0], atol=1e-7)
    assert geometry['rim_plane_max_residual_m'] < 1e-6
    with pytest.raises(ValueError, match='collinear'):
        describe_opening(frame, [10, 10, 90, 90], [[20, 20], [30, 30], [40, 40]])


@pytest.fixture(scope='module')
def robot_ik():
    from robosuite.models.robots import Panda
    from robosuite.models.grippers import PandaGripper
    from sketchik.kinematics import RobotIK
    arm = Panda(); gripper = PandaGripper()
    arm.add_gripper(gripper)
    return RobotIK(arm.get_xml(), arm.joints, gripper.important_sites['grip_site'])


def test_fixed_hand_descendant_and_intermediate_motion_are_checked(robot_ik):
    checker = ObservedClearance(robot_ik)
    q0 = np.array([-.03, -.13, 0, -2.44, 0, 2.2, .78])
    q1 = q0.copy(); q1[0] += 1.
    robot_ik.fk((q0+q1)/2)
    hand = next(s for s in checker.shapes if 'hand_collision' in (mujoco.mj_id2name(
        robot_ik.model, mujoco.mjtObj.mjOBJ_GEOM, s['index']) or ''))
    body = int(robot_ik.model.geom_bodyid[hand['index']])
    assert body in checker.moving_bodies
    centre = robot_ik.data.geom_xpos[hand['index']].copy()
    rotation = robot_ik.data.geom_xmat[hand['index']].reshape(3, 3)
    checker.points = centre+np.array([[-.006, 0, 0], [.006, 0, 0], [0, .006, 0]]) @ rotation.T
    checker.build()
    assert checker.check_q(q0)['observed_clearance_passed']
    assert checker.check_q(q1)['observed_clearance_passed']
    result = checker.check_path(q0, [SimpleNamespace(q=q1)])
    assert not result['observed_clearance_passed']
    assert not result['continuous_collision_certified']


def test_known_occluder_is_not_counted_as_visible_target():
    checker = ObservedClearance.__new__(ObservedClearance)
    checker.points = np.array([[0., 0., 1.]])
    K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    assert checker.visibility(np.eye(4), K, np.array([[0., 0., 2.]]), 100, 100) == 0
    assert checker.visibility(np.eye(4), K, np.array([[.3, 0., 2.]]), 100, 100) == 1


def test_carried_visible_component_is_removed_from_static_obstacles(robot_ik):
    checker=ObservedClearance(robot_ik)
    state={'q':[-.03,-.13,0,-2.44,0,2.2,.78],'step':7}
    origin,rotation=robot_ik.fk(state['q'])
    x,y,z=np.meshgrid(np.linspace(.02,.05,8),np.linspace(-.02,.02,8),np.linspace(.03,.07,8))
    local=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    far=np.array([[.2,.2,.3],[.21,.2,.3],[.2,.21,.3]])
    checker.points=np.r_[local,far]@rotation.T+origin
    checker.build()
    result=checker.separate_carried_surfaces(state)
    assert result['isolated'] and result['observed_points']>=25
    assert not result['rigid_attachment_verified'] and not result['complete_object_dimensions_known']
    assert len(checker.points)==3
    np.testing.assert_allclose(np.max(checker.carried_local,axis=0),[.05,.02,.07],atol=1e-12)


def test_carry_match_does_not_remove_connected_appliance_or_accept_missing_reference(robot_ik):
    checker=ObservedClearance(robot_ik)
    state={'q':[-.03,-.13,0,-2.44,0,2.2,.78],'step':8}
    origin,rotation=robot_ik.fk(state['q'])
    x,y,z=np.meshgrid(np.linspace(.02,.05,8),np.linspace(-.02,.02,8),np.linspace(.03,.07,8))
    local=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    # The visible object touches a long connected obstacle. Connected-component
    # removal alone would either reject the object or erase the whole obstacle.
    wall=np.column_stack([np.arange(.055,.6,.005),np.full(109,.02),np.full(109,.07)])
    world=np.r_[local,wall]@rotation.T+origin
    checker.points=world.copy();checker.build()
    old=checker.separate_carried_surfaces(state)
    assert not old['reference_available']
    result=checker.separate_carried_surfaces(state,local,7)
    assert result['reference_available'] and not result['isolated']
    assert result['method']=='rigid_observed_match'
    assert not result['rigid_attachment_verified']
    # Only the small measured-point matching neighbourhood is removed.
    remaining=(checker.points-origin)@rotation
    assert np.all(remaining[:,0]>.05)
    assert len(remaining)>=len(wall)-3
    checker.points=world.copy();checker.build();checker.carried_local=None
    result=checker.separate_carried_surfaces(state,local+[0,.5,0],7)
    assert not result['reference_available'] and checker.carried_local is None


@pytest.mark.parametrize('near_count', [2, 3, 4])
def test_carried_near_point_witnesses_preserve_threshold_and_report_real_pairs(near_count):
    from scipy.spatial import cKDTree

    origin=np.array([.5,-.25,0.])
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    local=np.column_stack([np.arange(6)*.1,np.zeros(6),np.zeros(6)])
    predicted=local@rotation.T+origin
    obstacles=predicted.copy()
    obstacles[:,2]=.01
    obstacles[:near_count,2]=np.arange(1,near_count+1)*.0004
    # The existing strict <2mm rule must exclude this boundary point.
    obstacles[-1,2]=.002
    checker=ObservedClearance.__new__(ObservedClearance)
    checker.ik=SimpleNamespace(fk=lambda q:(origin,rotation))
    checker.shapes=[]
    checker.carried_local=local.copy()
    # Shuffle the obstacle table so a carried index is not its nearest-point ID.
    checker.points=obstacles[[4,2,0,5,1,3]].copy()
    checker.tree=cKDTree(checker.points)
    distances,_=checker.tree.query(predicted)
    previous_passed=np.count_nonzero(distances<.002)<3
    result=checker.check_q(np.zeros(7))
    assert result['observed_clearance_passed']==previous_passed==(near_count<3)
    np.testing.assert_array_equal(checker.carried_local,local)
    np.testing.assert_array_equal(checker.points,obstacles[[4,2,0,5,1,3]])
    if previous_passed:
        assert result['observed_clearance_passed']
        assert not result['carried_volume_proxy_checked']
        assert not result['carried_volume_proxy']['available']
        return
    assert result['near_points']==near_count
    assert result['near_point_distance_threshold_m']==.002
    assert len(result['near_point_witnesses'])==3
    for index,pair in enumerate(result['near_point_witnesses']):
        np.testing.assert_array_equal(pair['predicted_carried_world_xyz_m'],predicted[index])
        np.testing.assert_array_equal(pair['observed_obstacle_world_xyz_m'],obstacles[index])
        distance=np.linalg.norm(np.asarray(pair['predicted_carried_world_xyz_m'])-
                                pair['observed_obstacle_world_xyz_m'])
        assert pair['distance_m']==pytest.approx(distance,abs=1e-14)
        assert pair['distance_m']<result['near_point_distance_threshold_m']


def observed_box_surface():
    from itertools import product
    points=[]
    for axis in range(3):
        for face in (-.04,.04):
            for first,second in product(np.linspace(-.04,.04,5),repeat=2):
                point=[first,second];point.insert(axis,face);points.append(point)
    return np.unique(points,axis=0)


def proxy_checker(local,obstacles,poses=None):
    from scipy.spatial import cKDTree
    checker=ObservedClearance.__new__(ObservedClearance)
    poses=poses or [(np.zeros(3),np.eye(3))]
    checker.ik=SimpleNamespace(fk=lambda q:poses[int(np.asarray(q)[0])])
    checker.shapes=[];checker.carried_local=np.asarray(local);checker.carried_rgb=None
    checker.points=np.asarray(obstacles);checker.tree=cKDTree(checker.points)
    checker._build_carried_proxy()
    return checker


@pytest.mark.parametrize('count',[2,3])
def test_carried_proxy_blocks_interior_points_at_three_without_changing_surface_distance_rule(count):
    local=observed_box_surface()
    obstacles=np.array([[0.,0.,0.],[.005,0.,0.],[0.,.005,0.]])[:count]
    checker=proxy_checker(local,obstacles)
    distances,_=checker.tree.query(local)
    assert np.count_nonzero(distances<.002)==0
    result=checker.check_q(np.zeros(7))
    assert result['observed_clearance_passed']==(count<3)
    assert result['carried_volume_proxy_checked'] and result['carried_volume_proxy']['available']
    info=result['carried_volume_proxy']
    assert info['fills_concavities_and_hollow_interiors']
    assert not info['true_object_shape'] and not info['hidden_surface_reconstruction']
    assert not info['rigid_attachment_verified']
    if count==3:
        assert result['carried_convex_proxy_contains_obstacle']
        assert result['observed_points_inside_carried_proxy']==3
        assert result['obstacle_point_rejection_threshold']==3
        np.testing.assert_allclose(result['example_world_xyz_m'],obstacles)


def test_carried_proxy_follows_current_tool_rotation_and_translation():
    from scipy.spatial.transform import Rotation
    local=observed_box_surface()+[.1,.02,.04]
    origin=np.array([.5,-.4,1.2]);rotation=Rotation.from_euler('xyz',[.4,-.6,.7]).as_matrix()
    interior=np.array([[.1,.02,.04],[.105,.02,.04],[.1,.025,.04]])
    world=interior@rotation.T+origin
    poses=[(np.zeros(3),np.eye(3)),(origin,rotation),(origin+[2.,0.,0.],rotation)]
    checker=proxy_checker(local,world,poses)
    assert checker.check_q(np.zeros(7))['observed_clearance_passed']
    result=checker.check_q(np.ones(7))
    assert result['carried_convex_proxy_contains_obstacle']
    assert not result['observed_clearance_passed']
    assert checker.check_q(np.full(7,2.))['observed_clearance_passed']
    np.testing.assert_array_equal(checker.carried_local,local)


def test_coplanar_proxy_is_explicitly_unavailable_and_original_near_pair_check_remains():
    from scipy.spatial import cKDTree
    x,y=np.meshgrid(np.linspace(-.03,.03,7),np.linspace(-.03,.03,7))
    local=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    checker=proxy_checker(local,np.array([[0.,0.,.05],[.01,0.,.05],[0.,.01,.05]]))
    result=checker.check_q(np.zeros(7))
    assert result['observed_clearance_passed'] and not result['carried_volume_proxy_checked']
    assert result['carried_volume_proxy']['status']=='unavailable'
    assert 'degenerate' in result['carried_volume_proxy']['reason']
    checker.points=local[:3]+[0,0,.001];checker.tree=cKDTree(checker.points)
    result=checker.check_q(np.zeros(7))
    assert not result['observed_clearance_passed'] and result['near_points']>=3
    assert result['near_point_distance_threshold_m']==.002 and not result['carried_volume_proxy_checked']


def test_failed_new_carried_separation_discards_previous_proxy():
    from scipy.spatial import cKDTree
    checker=proxy_checker(observed_box_surface(),np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]]))
    assert checker.carried_proxy is not None
    checker.set_state=lambda state:None
    checker.points=np.array([[1.,1.,1.],[1.01,1.,1.],[1.,1.01,1.]])
    checker.tree=cKDTree(checker.points)
    result=checker.separate_carried_surfaces({'q':np.zeros(7),'step':2})
    assert checker.carried_local is None and checker.carried_proxy is None
    assert result['reference_available'] is False
    assert not result['carried_volume_proxy']['available']
    assert checker.check_q(np.zeros(7))['observed_clearance_passed']


def test_previous_rigid_reference_is_checked_even_beyond_eef_origin_neighbourhood():
    local=observed_box_surface()+[.2,0,0]
    checker=proxy_checker(local,local.copy());checker.set_state=lambda state:None;checker.colors=None
    report=checker.separate_carried_surfaces({'q':np.zeros(7),'step':20},local,10)
    assert report['reference_available'] is True
    assert report['reference_visible_fraction']==1.
    assert report['observed_points']==len(local)
    np.testing.assert_allclose(np.sort(checker.carried_local,axis=0),np.sort(local,axis=0))


@pytest.mark.parametrize('empty',[False,True])
def test_missing_current_surface_and_jaw_collapse_reach_episode_contact_gate(tmp_path,empty):
    from scipy.spatial import cKDTree
    from sketchik.episode_state import EpisodeState
    local=observed_box_surface()
    points=np.empty((0,3)) if empty else local+[1,1,1]
    checker=proxy_checker(local,points);checker.set_state=lambda state:None
    current=checker.separate_carried_surfaces({'q':np.zeros(7),'step':20},local,10)
    assert current['reference_available'] is False and checker.carried_local is None
    state=EpisodeState(tmp_path)
    previous={'source_step':10,'carried_observed_geometry':{'object_id':'object','reference_available':True},
              'robot_gripper_geometry':{'current_inner_jaw_gap_m':.071}}
    state.observe_contact(previous,'object',1.)
    packet={'source_step':20,'carried_observed_geometry':{'object_id':'object',**current},
            'robot_gripper_geometry':{'current_inner_jaw_gap_m':.0018}}
    assert state.observe_contact(packet,'object',1.)['object_id']=='object'
    assert state.data['carry_premise']=='invalidated_by_contact_change'
    assert state.data['holding_or_task_success_certified'] is False
    state.reset_contact(30,'new contact')
    state.observe_contact(previous,'object',1.)
    packet['robot_gripper_geometry']['current_inner_jaw_gap_m']=.071
    assert state.observe_contact(packet,'object',1.) is None  # missing depth alone remains insufficient
