import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.kinematics import IKError, RobotIK, joint_trajectory, rotation_from_directions
from sketchik.libero import back_project


@pytest.fixture(scope="module")
def ik():
    # Standalone robot asset, no LIBERO environment, object state or GPU context.
    from robosuite.models.robots import Panda
    from robosuite.models.grippers import gripper_factory
    arm = Panda(idn=0)
    gripper = gripper_factory("PandaGripper", idn=0)
    arm.add_gripper(gripper)
    return RobotIK(arm.get_xml(), arm.joints, gripper.important_sites["grip_site"])


@pytest.mark.parametrize("shift", [
    [.04, -.03, .02, .04, -.04, .03, -.04],
    [-.05, .02, -.03, -.04, .03, -.02, .05],
    [0, 0, 0, 0, 0, 0, .25],
])
def test_full_pose_ik_round_trip(ik, shift):
    seed = np.array([0., -.5, 0., -2., 0., 1.5, .7])
    goal = seed + shift
    p, r = ik.fk(goal)
    sol = ik.solve(p, Rotation.from_matrix(r).as_quat(), seed)
    recovered_p, recovered_r = ik.fk(sol.q)
    assert np.linalg.norm(recovered_p-p) < .0005
    assert (Rotation.from_matrix(recovered_r) * Rotation.from_matrix(r).inv()).magnitude() < np.deg2rad(.5)
    assert np.all(sol.q >= ik.limits[:, 0]) and np.all(sol.q <= ik.limits[:, 1])


def test_unreachable_ik_rejects(ik):
    seed = [0., -.5, 0., -2., 0., 1.5, .7]
    _, r = ik.fk(seed)
    with pytest.raises(IKError):
        ik.solve([2., 2., 2.], Rotation.from_matrix(r).as_quat(), seed)


def test_seed_joint_limits_reject(ik):
    with pytest.raises(IKError):
        ik.solve([0, 0, 0], [0, 0, 0, 1], [10.]*7)


def test_quintic_commands_obey_velocity_and_acceleration():
    q0 = np.zeros(7)
    target = np.array([.6, -.4, .1, -.2, .8, -.15, .2])
    samples = joint_trajectory(q0, [target, q0])
    padded = np.vstack([q0, q0, samples, q0, q0])
    assert np.max(np.abs(np.diff(padded, axis=0)/.05)) <= .6 + 1e-9
    assert np.max(np.abs(np.diff(padded, n=2, axis=0)/.05**2)) <= 1.2 + 1e-9
    np.testing.assert_allclose(samples[-1], q0, atol=1e-12)


def test_direction_frame_and_parallel_rejection():
    z = np.array([.2, .1, -1.])
    q = rotation_from_directions(z, [0, 1, 0])
    r = Rotation.from_quat(q).as_matrix()
    np.testing.assert_allclose(r[:, 2], z/np.linalg.norm(z))
    np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.)
    with pytest.raises(ValueError):
        rotation_from_directions([0, 0, 1], [0, 0, -1])


def test_metric_depth_world_frame():
    # Pixel (2,1), z=2m. Camera translation and 90-degree rotation matter.
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    t[:3, 3] = [1, 2, 3]
    frame = {"depth_m": np.full((3, 4), 2.), "K": np.array([[2,0,1],[0,2,1],[0,0,1]]),
             "world_from_camera": t}
    np.testing.assert_allclose(back_project(frame, [2, 1]), [1, 3, 5], atol=1e-12)
    with pytest.raises(ValueError):
        back_project(frame, [-1, 0])
    frame["depth_m"][1, 2] = np.nan
    with pytest.raises(ValueError):
        back_project(frame, [2, 1])


def test_adapter_refreshes_post_integration_pose_before_callbacks(ik):
    import mujoco
    from types import SimpleNamespace
    from sketchik.libero import LiberoRobot
    model=ik.model;data=mujoco.MjData(model)
    seed=np.array([0.,-.5,0.,-2.,0.,1.5,.7]);goal=seed+np.array([.02,0,0,0,0,0,0])
    data.qpos[ik.q_index]=seed;mujoco.mj_forward(model,data)
    old_position=data.site_xpos[ik.site_id].copy()
    forward_calls=[]
    def forward():
        forward_calls.append(True);mujoco.mj_forward(model,data)
    controller=SimpleNamespace(goal_qpos=seed.copy())
    def integrate_without_final_forward(action):
        data.qpos[ik.q_index]=action[:7]
        controller.goal_qpos=action[:7].copy()
        np.testing.assert_array_equal(data.site_xpos[ik.site_id],old_position)
        return None,0.,False,{}
    fingers=[model.joint(i).name for i in range(model.njnt) if 'finger_joint' in model.joint(i).name]
    robot=LiberoRobot.__new__(LiberoRobot)
    robot.env=SimpleNamespace(sim=SimpleNamespace(data=data,forward=forward,
        model=SimpleNamespace(jnt_range=model.jnt_range,
            get_joint_qpos_addr=lambda n:int(model.joint(n).qposadr[0]))),
        step=integrate_without_final_forward,_check_success=lambda:False)
    robot.robot=SimpleNamespace(arm_joint_indexes=[model.joint(model.joint(i).name).id for i in range(7)],
        part_controllers={'right':controller},gripper={'right':SimpleNamespace(joints=fingers)})
    robot.q_index=ik.q_index;robot.site_id=ik.site_id
    robot.done=False;robot.steps=0;robot.max_steps=100;robot.gripper_command=-1.
    callbacks=[];robot.on_step=callbacks.append
    result=robot.step(goal)
    expected_p,expected_r=ik.fk(goal)
    assert len(forward_calls)==1 and len(callbacks)==1 and robot.steps==1
    np.testing.assert_array_equal(data.qpos[ik.q_index],goal)
    np.testing.assert_allclose(result['position'],expected_p,atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(callbacks[0]['quaternion_xyzw']).as_matrix(),expected_r,atol=1e-12)
