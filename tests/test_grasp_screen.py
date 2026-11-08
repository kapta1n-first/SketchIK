import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from sketchik.grasp_screen import approach_poses, proximity_check


def test_approach_is_in_grasp_frame_and_keeps_final_pose():
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    grasp[:3, 3] = [.2, -.1, .5]
    poses, distances = approach_poses(grasp, .1)
    np.testing.assert_allclose(poses[0, :3, 3], [.1, -.1, .5])
    np.testing.assert_allclose(poses[-1], grasp)
    assert distances[0] == .1 and distances[-1] == 0
    assert np.max(abs(np.diff(distances))) <= .01000001


def test_clear_endpoint_can_still_have_colliding_approach():
    grasp = np.eye(4)
    poses, _ = approach_poses(grasp, .1)
    tree = cKDTree([[0., 0., -.05]])
    samples = np.zeros((1, 3))
    _, endpoint_collision = proximity_check(tree, samples, [grasp], .005)
    distances, approach_collision = proximity_check(tree, samples, poses, .005)
    assert not endpoint_collision and approach_collision
    assert len(distances) < len(poses)
    assert min(distances) < 1e-9


def test_bounded_query_does_not_change_clearance_decision():
    rng = np.random.default_rng(1)
    scene = rng.uniform(-.2, .2, (500, 3))
    samples = rng.uniform(-.02, .02, (100, 3))
    tree = cKDTree(scene)
    for clearance in [.001, .01, .03]:
        _, rejected = proximity_check(tree, samples, [np.eye(4)], clearance)
        brute_min = np.linalg.norm(samples[:, None]-scene[None], axis=-1).min()
        assert rejected == (brute_min < clearance)
