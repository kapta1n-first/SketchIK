"""CPU-only robot geometry checks; no task scene or renderer is created."""
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sketchik.robot_geometry import hand_bounds_in_eef


def test_fast_jaw_gap_matches_full_known_robot_geometry_for_actual_finger_states():
    from robosuite.models.robots import Panda
    from robosuite.models.grippers import PandaGripper
    from sketchik.kinematics import RobotIK
    from sketchik.robot_geometry import measured_inner_jaw_gap, gripper_geometry_in_eef
    arm = Panda()
    gripper = PandaGripper()
    arm.add_gripper(gripper)
    ik = RobotIK(arm.get_xml(), arm.joints, gripper.important_sites['grip_site'])
    for yaw in [-.4, 0., .7]:
        for opening in [.0001, .001, .01, .04]:
            q = np.array([yaw, -.13, 0., -2.44, 0., 2.2, .78])
            values = [opening, -opening]
            state = {'step': 10, 'q': q.tolist(), 'finger_joint_names': gripper.joints,
                     'finger_joint_positions': values}
            for name, value in zip(gripper.joints, values):
                ik.data.qpos[ik.model.joint(name).qposadr[0]] = value
            origin, rotation = ik.fk(q)
            expected = gripper_geometry_in_eef(ik, origin, rotation, state)['current_inner_jaw_gap_m']
            actual = measured_inner_jaw_gap(ik, state)
            assert actual == pytest.approx(expected, abs=1e-12)


def test_pad_reference_presentation_preserves_full_bounds_and_direction_signs():
    from copy import deepcopy
    from sketchik.robot_geometry import pad_reference_geometry
    raw = {'source_step':123,'axis_meaning':{'x':'closing','y':'transverse','z':'forward'},
        'pad_centres_in_eef_tool_m':[[-.04,0,-.004],[.04,0,-.004]],
        'hand_bounds_in_eef_tool_m':[[-.1,-.02,-.12],[.1,.02,.01]],
        'palm_and_fixed_hand':{'bounds_in_eef_tool_m':[[-.1,-.02,-.12],[.1,.02,-.03]]},
        'current_inner_jaw_gap_m':.079,
        'fingers':[{'length_along_tool_z_m':.055,
            'bounds_in_eef_tool_m':[[-.06,-.01,-.045],[-.04,.01,.01]],
            'pads':[{'bounds_in_eef_tool_m':[[-.05,-.01,-.012],[-.04,.01,.004]]}]}]}
    original = deepcopy(raw); view = pad_reference_geometry(raw)
    assert raw == original and view['origin'] == 'pad_midpoint' and view['source_step'] == 123
    np.testing.assert_allclose(view['palm_bounds_relative_pad_midpoint_tool_m'],
        [[-.1,-.02,-.116],[.1,.02,-.026]])
    np.testing.assert_allclose(view['hand_bounds_relative_pad_midpoint_tool_m'],
        [[-.1,-.02,-.116],[.1,.02,.014]])
    np.testing.assert_allclose(view['fingers'][0]['pad_bounds_relative_pad_midpoint_tool_m'][0],
        [[-.05,-.01,-.008],[-.04,.01,.008]])
    assert view['fingers'][0]['length_along_tool_z_m'] == .055
    assert view['current_inner_jaw_gap_m'] == .079
    palm = view['palm_extent_along_negative_tool_z_from_pad_m']
    assert palm['nearest_signed_distance_m'] == pytest.approx(.026)
    assert palm['farthest_signed_distance_m'] == pytest.approx(.116)
    assert pad_reference_geometry({}) is None
    assert pad_reference_geometry({**raw, 'pad_centres_in_eef_tool_m':[[0,0,0]]}) is None


@pytest.fixture(params=[False, True], ids=['site_on_hand', 'fixed_marker_leaf'])
def hand_ik(request):
    from sketchik.kinematics import RobotIK

    # Seven arm joints satisfy RobotIK; sliders and nested pads model real FK
    # descendants instead of putting all the geometry on the EEF site's body.
    links = ''.join(
        f'<body name="link{i}" pos=".08 0 0">'
        f'<joint name="joint{i}" type="hinge" axis="0 0 1" '
        'limited="true" range="-3 3"/>'
        '<geom type="capsule" size=".02 .03" pos="0 0 .04"/>'
        for i in range(7))
    fingers = ''.join(
        f'<body name="finger{side}" pos="{sign * .05} 0 .015">'
        f'<joint name="finger_joint{side}" type="slide" axis="{sign} 0 0" '
        'limited="true" range="0 .04"/>'
        '<geom type="box" size=".008 .009 .03" pos="0 0 .03"/>'
        f'<body name="pad{side}" pos="0 0 .075">'
        '<geom type="sphere" size=".012"/></body></body>'
        for side, sign in enumerate((-1, 1)))
    site = '<site name="eef" pos=".01 -.005 .01" quat="1 0 0 1"/>'
    if request.param:
        site = f'<body name="eef_marker">{site}</body>'
    xml = (
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<geom name="static_base" type="box" size="10 10 10"/>'
        f'{links}<body name="hand">'
        f'{site}'
        '<geom name="palm" type="box" size=".04 .03 .02" pos="0 0 -.03"/>'
        '<geom name="visual" type="box" size="5 5 5" contype="0" conaffinity="0"/>'
        f'{fingers}</body>{"</body>" * 7}</worldbody></mujoco>')
    ik = RobotIK(xml, [f'joint{i}' for i in range(7)], 'eef')
    ik.fk(np.zeros(7))
    return ik


def test_hand_bounds_include_articulated_finger_descendants_only(hand_ik):
    ik = hand_ik
    origin, rotation = ik.fk(np.zeros(7))
    bounds = hand_bounds_in_eef(ik, origin, rotation)
    # Pads at hand z=.09 extend to .102; the EEF site is .01 above hand origin.
    # The static base, seven arm geoms, and the 10m-wide visual are all excluded.
    np.testing.assert_allclose(bounds, [[-.025, -.052, -.06], [.035, .072, .092]], atol=1e-12)
    assert isinstance(bounds, list) and all(isinstance(row, list) for row in bounds)
    assert ik.model.body('pad0').parentid[0] == ik.model.body('finger0').id
    assert ik.model.body('finger0').parentid[0] == ik.model.body('hand').id


def test_finger_opening_updates_current_fk_bounds(hand_ik):
    ik = hand_ik
    origin, rotation = ik.fk(np.zeros(7))
    closed = np.asarray(hand_bounds_in_eef(ik, origin, rotation))
    for name in ('finger_joint0', 'finger_joint1'):
        ik.data.qpos[ik.model.joint(name).qposadr[0]] = .04
    origin, rotation = ik.fk(np.zeros(7))
    opened = np.asarray(hand_bounds_in_eef(ik, origin, rotation))
    np.testing.assert_allclose(opened, [[-.025, -.092, -.06], [.035, .112, .092]], atol=1e-12)
    assert opened[1, 1] - opened[0, 1] == pytest.approx(closed[1, 1] - closed[0, 1] + .08)


def test_hand_bounds_are_invariant_under_full_world_frame_motion(hand_ik):
    ik = hand_ik
    origin, rotation = ik.fk(np.zeros(7))
    initial = hand_bounds_in_eef(ik, origin, rotation)
    # A rigid motion of the whole recorded robot must preserve tool-frame bounds.
    from scipy.spatial.transform import Rotation
    motion = Rotation.from_rotvec([.7, -.3, .9]).as_matrix()
    shift = np.array([1.2, -.7, .4])
    moved = SimpleNamespace(
        model=ik.model, site_id=ik.site_id,
        data=SimpleNamespace(
            geom_xpos=ik.data.geom_xpos @ motion.T + shift,
            geom_xmat=np.asarray([motion @ matrix.reshape(3, 3)
                                  for matrix in ik.data.geom_xmat])))
    np.testing.assert_allclose(
        hand_bounds_in_eef(moved, motion @ origin + shift, motion @ rotation), initial, atol=1e-12)


def test_mesh_uses_compiled_vertices_and_geom_pose():
    import mujoco

    xml = ('<mujoco><asset><mesh name="shape" '
           'vertex="0 0 0 .02 0 0 0 .03 0 0 0 .04"/></asset><worldbody>'
           '<body name="hand" pos=".4 -.2 .3"><site name="eef"/>'
           '<geom type="mesh" mesh="shape" pos=".01 .02 -.03" quat="1 0 0 1"/>'
           '</body></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    ik = SimpleNamespace(model=model, data=data, site_id=model.site('eef').id)
    # The original tetrahedron rotates 90 degrees about z and translates in hand.
    # Compiled mesh vertices have their own centered/principal-axis representation.
    np.testing.assert_allclose(
        hand_bounds_in_eef(ik, data.site_xpos[ik.site_id], data.site_xmat[ik.site_id].reshape(3, 3)),
        [[-.02, .02, -.03], [.01, .04, .01]], atol=1e-8)


@pytest.mark.parametrize('shape,size,expected', [
    ('sphere', '.02', [.02, .02, .02]),
    ('capsule', '.02 .04', [.02, .02, .06]),
    ('cylinder', '.02 .04', [.02, .02, .04]),
    ('ellipsoid', '.01 .02 .03', [.01, .02, .03]),
])
def test_curved_primitives_have_conservative_local_bounds(shape, size, expected):
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="hand"><site name="eef"/>'
        f'<geom type="{shape}" size="{size}"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    ik = SimpleNamespace(model=model, data=data, site_id=model.site('eef').id)
    np.testing.assert_allclose(hand_bounds_in_eef(ik, np.zeros(3), np.eye(3)),
                               [-np.asarray(expected), expected], atol=1e-12)


def test_no_collision_hand_returns_none_and_does_not_change_fk(hand_ik):
    ik = hand_ik
    origin, rotation = ik.fk(np.zeros(7))
    saved_qpos = ik.data.qpos.copy()
    saved_xpos = ik.data.geom_xpos.copy()
    ik.model.geom_contype[:] = 0
    ik.model.geom_conaffinity[:] = 0
    assert hand_bounds_in_eef(ik, origin, rotation) is None
    np.testing.assert_array_equal(ik.data.qpos, saved_qpos)
    np.testing.assert_array_equal(ik.data.geom_xpos, saved_xpos)


def test_marker_root_search_does_not_cross_arm_joint(hand_ik):
    ik = hand_ik
    origin, rotation = ik.fk(np.zeros(7))
    hand_root = ik.model.body('hand').id
    for geom in range(ik.model.ngeom):
        body = int(ik.model.geom_bodyid[geom])
        while body and body != hand_root:
            body = int(ik.model.body_parentid[body])
        if body == hand_root:
            ik.model.geom_contype[geom] = 0
            ik.model.geom_conaffinity[geom] = 0
    # Arm links still collide; an unbounded ancestor search would include them.
    assert np.any(ik.model.geom_contype)
    assert hand_bounds_in_eef(ik, origin, rotation) is None


def test_actual_panda_marker_resolves_palm_and_both_articulated_fingers():
    from robosuite.models.robots import Panda
    from robosuite.models.grippers import gripper_factory
    from sketchik.kinematics import RobotIK

    arm = Panda(idn=0)
    gripper = gripper_factory('PandaGripper', idn=0)
    arm.add_gripper(gripper)
    ik = RobotIK(arm.get_xml(), arm.joints, gripper.important_sites['grip_site'])
    seed = np.array([0., -.5, 0., -2., 0., 1.5, .7])
    origin, rotation = ik.fk(seed)
    # Current robosuite has the EEF marker and finger bodies as siblings.
    marker = int(ik.model.site_bodyid[ik.site_id])
    assert not np.any(ik.model.geom_bodyid == marker)
    closed_hand = np.asarray(hand_bounds_in_eef(ik, origin, rotation))
    assert closed_hand.shape == (2, 3) and np.isfinite(closed_hand).all()
    assert np.all(closed_hand[1] > closed_hand[0])
    # The actual palm is wider than the fingers, even at full opening. Isolate
    # finger collision volumes to verify they were included through marker siblings.
    palm = int(ik.model.body_parentid[marker])
    palm_geoms = np.flatnonzero(ik.model.geom_bodyid == palm)
    masks = (ik.model.geom_contype[palm_geoms].copy(), ik.model.geom_conaffinity[palm_geoms].copy())
    ik.model.geom_contype[palm_geoms] = 0
    ik.model.geom_conaffinity[palm_geoms] = 0
    closed_fingers = np.asarray(hand_bounds_in_eef(ik, origin, rotation))
    for name in gripper.joints:
        joint = ik.model.joint(name)
        # Panda's two slider coordinates use opposite signed joint ranges.
        limits = ik.model.jnt_range[joint.id]
        ik.data.qpos[joint.qposadr[0]] = limits[np.argmax(np.abs(limits))]
    origin, rotation = ik.fk(seed)
    opened_fingers = np.asarray(hand_bounds_in_eef(ik, origin, rotation))
    # Site +X is finger separation: both outside faces shift by exactly .04m.
    assert opened_fingers[0, 0] == pytest.approx(closed_fingers[0, 0] - .04)
    assert opened_fingers[1, 0] == pytest.approx(closed_fingers[1, 0] + .04)
    np.testing.assert_allclose(opened_fingers[:, 1:], closed_fingers[:, 1:], atol=1e-9)
    ik.model.geom_contype[palm_geoms], ik.model.geom_conaffinity[palm_geoms] = masks
    opened_hand = np.asarray(hand_bounds_in_eef(ik, origin, rotation))
    assert np.all(opened_hand[0] <= opened_fingers[0] + 1e-9)
    assert np.all(opened_hand[1] >= opened_fingers[1] - 1e-9)
    from sketchik.robot_geometry import gripper_geometry_in_eef
    state = {'step': 30, 'finger_joint_names': gripper.joints,
             'finger_joint_positions': [float(ik.data.qpos[ik.model.joint(name).qposadr[0]])
                                        for name in gripper.joints]}
    saved_qpos, saved_xpos = ik.data.qpos.copy(), ik.data.geom_xpos.copy()
    facts = gripper_geometry_in_eef(ik, origin, rotation, state)
    assert len(facts['fingers']) == 2
    assert facts['current_inner_jaw_gap_m'] == pytest.approx(.079, abs=1e-6)
    np.testing.assert_allclose(facts['hand_span_xyz_m'], opened_hand[1]-opened_hand[0])
    for finger in facts['fingers']:
        assert .04 < finger['length_along_tool_z_m'] < .10
        assert len(finger['pads']) == 1
        assert np.all(np.asarray(finger['pads'][0]['span_xyz_m']) > 0)
        assert np.all(np.asarray(finger['bounds_in_eef_tool_m'])[0] >= opened_hand[0]-1e-9)
        assert np.all(np.asarray(finger['bounds_in_eef_tool_m'])[1] <= opened_hand[1]+1e-9)
    np.testing.assert_array_equal(ik.data.qpos, saved_qpos)
    np.testing.assert_array_equal(ik.data.geom_xpos, saved_xpos)


def test_import_does_not_load_mujoco_or_robosuite():
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    subprocess.run(
        [sys.executable, '-c', 'import sys; import sketchik.robot_geometry; '
         'assert "mujoco" not in sys.modules; assert "robosuite" not in sys.modules'],
        env=env, check=True, capture_output=True, text=True)


def test_recorded_full_robot_uses_exact_q_without_ik_and_restores_all_caches(hand_ik):
    from sketchik.robot_geometry import recorded_robot_geometry
    ik = hand_ik
    cached = np.full(7, -.04)
    ik.fk(cached)
    before_qpos = ik.data.qpos.copy()
    before_xpos = ik.data.geom_xpos.copy()
    actual = np.zeros(7)
    target = np.full(7, .08)
    state = {'step': 42, 'q': actual.tolist(),
        'finger_joint_names': ['finger_joint0', 'finger_joint1'],
        'finger_joint_positions': [.02, .03]}
    ik.solve = lambda *args, **kwargs: pytest.fail('Display must not solve a new IK branch')
    result = recorded_robot_geometry(ik, state,
        {'pose': {'joint_target': target.tolist()}, 'failure_observed_step': 40})
    assert result['goal_status'] == 'exact_recorded_joint_branch'
    assert result['actual_joint_positions_rad'] == actual.tolist()
    assert result['recorded_goal_joint_positions_rad'] == target.tolist()
    names = {part['body'] for part in result['actual_parts']}
    assert {'link0', 'link6', 'finger0', 'finger1', 'hand'} <= names
    assert 'world' not in names
    assert len(result['actual_parts']) == len(result['recorded_goal_parts'])
    assert result['actual_parts'] != result['recorded_goal_parts']
    np.testing.assert_array_equal(ik.data.qpos, before_qpos)
    np.testing.assert_array_equal(ik.data.geom_xpos, before_xpos)
    missing = recorded_robot_geometry(ik, state, {'pose': {'position': [0, 0, 1]}})
    assert missing['goal_status'] == 'no_recorded_joint_branch'
    assert missing['recorded_goal_parts'] == []
    np.testing.assert_array_equal(ik.data.qpos, before_qpos)
    np.testing.assert_array_equal(ik.data.geom_xpos, before_xpos)
