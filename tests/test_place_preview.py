import numpy as np
from scipy.spatial.transform import Rotation

from sketchik.place_preview import target_eef_pose, transform_points


def test_object_delta_preserves_measured_object_in_hand_transform():
    current = np.eye(4)
    current[:3, :3] = Rotation.from_euler('xyz', [.4, -.2, .8]).as_matrix()
    current[:3, 3] = [.1, .2, .9]
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler('xyz', [-.1, .5, .2]).as_matrix()
    delta[:3, 3] = [-.1, .3, .2]
    points = np.array([[.12, .21, .82], [.08, .24, .86]])
    target = target_eef_pose(delta, current[:3, 3], Rotation.from_matrix(current[:3, :3]).as_quat())
    before = transform_points(points, np.linalg.inv(current))
    after = transform_points(transform_points(points, delta), np.linalg.inv(target))
    np.testing.assert_allclose(before, after, atol=1e-12)
    # Reversing transform order does not describe the same held-object motion.
    assert not np.allclose(target, current @ delta)


def test_world_delta_identity_keeps_full_pose():
    quat = Rotation.from_euler('xyz', [1., -.3, .8]).as_quat()
    pose = target_eef_pose(np.eye(4), [.1, -.2, .9], quat)
    np.testing.assert_allclose(pose[:3, 3], [.1, -.2, .9])
    np.testing.assert_allclose(pose[:3, :3], Rotation.from_quat(quat).as_matrix())
