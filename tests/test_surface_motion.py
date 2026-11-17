import numpy as np

from sketchik.surface_motion import static_camera_pairs, track_samples


def frame(depth=1.):
    return {'camera': 'front', 'step': 20, 'depth_m': np.full((256, 256), depth),
        'rgb': np.full((256, 256, 3), 100, np.uint8),
        'K': np.array([[240., 0., 128.], [0., 240., 128.], [0., 0., 1.]]),
        'world_from_camera': np.eye(4)}


def patch(x):
    a, b = np.meshgrid(np.linspace(x-.04, x+.04, 35), np.linspace(-.04, .04, 35))
    return np.c_[a.ravel(), b.ravel(), np.ones(a.size)]


def evidence(x):
    a, b = np.meshgrid(np.linspace(x-.045, x+.045, 8), np.linspace(-.045, .045, 8))
    return np.c_[a.ravel(), b.ravel(), np.ones(a.size)]


def test_two_surface_motions_preserve_original_rows_and_unobserved_neighbour():
    static, moving, unknown = patch(-.15), patch(.1), patch(.32)
    original = np.r_[static, moving, unknown]
    before = original.copy()
    a, b = evidence(-.15), evidence(.1)
    shifted = b+[.065, 0, 0]
    current, supported, report = track_samples(original, np.r_[a, b], np.r_[a, shifted],
        [frame()], budget_s=10.)
    first, second, third = np.split(supported, 3)
    assert first.sum() > 100 and second.sum() > 100 and not third.any()
    np.testing.assert_allclose(current[:len(static)][first], static[first], atol=1e-7)
    np.testing.assert_allclose(current[len(static):2*len(static)][second], moving[second]+[.065, 0, 0], atol=1e-7)
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(current[-len(unknown):], unknown)
    assert report['moved_samples'] > 100 and not report['selectable']
    assert report['original_samples'] == len(original) and report['original_points_deleted'] == 0


def test_conflicting_local_motions_do_not_choose_a_whole_object_pose():
    original, a = patch(0), evidence(0)
    current, supported, report = track_samples(original, np.r_[a, a],
        np.r_[a+[.06, 0, 0], a+[-.06, 0, 0]], [frame()], budget_s=10.)
    assert not supported.any() and report['status'] == 'unavailable'
    np.testing.assert_array_equal(current, original)


def test_duplicate_and_collinear_correspondences_are_not_independent_support():
    original = patch(0)
    a = np.array([[-.02, 0, 1], [0, 0, 1], [.02, 0, 1.]])
    a = np.repeat(a, 30, axis=0)
    current, supported, report = track_samples(original, a, a+[.06, 0, 0], [frame()])
    assert not supported.any() and report['distinct_feature_pairs'] == 3
    np.testing.assert_array_equal(current, original)


def test_feature_consensus_in_measured_free_space_is_rejected():
    a, original = evidence(0), patch(0)
    current, supported, report = track_samples(original, a, a+[.06, 0, 0],
        [frame(depth=1.2)], budget_s=10.)
    assert not supported.any() and report['status'] == 'unavailable'
    np.testing.assert_array_equal(current, original)


def test_moved_wrist_is_not_treated_as_a_static_camera():
    before, now = frame(), frame()
    before['step'] = 10
    now['world_from_camera'][0, 3] = .2
    old = {'world_xyz_m': patch(0), 'source_index': np.zeros(len(patch(0)), int),
        'pixel_xy': np.zeros((len(patch(0)), 2), int)}
    a, b = static_camera_pairs(old, [{'observation_id': 'old', 'camera': 'front'}],
        [now], lambda *_: before)
    assert a.shape == b.shape == (0, 3)


def test_missing_or_mixed_depth_snapshots_are_rejected():
    import pytest
    a = evidence(0)
    newer = frame(); newer['step'] += 1
    for frames in ([], [frame(), newer]):
        with pytest.raises(ValueError, match='one current physical snapshot'):
            track_samples(patch(0), a, a, frames)



def test_another_identical_textureless_plane_is_not_the_original_surface():
    original = patch(0)
    # A second plane fits geometry/colour, but has no original feature identity.
    updated, supported, report = track_samples(original, np.empty((0, 3)),
        np.empty((0, 3)), [frame(1.04)])
    assert not supported.any() and report['status'] == 'unavailable'
    np.testing.assert_array_equal(updated, original)
