from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from sketchik.observed_clearance import ObservedClearance


@pytest.mark.parametrize('distance', [.012 - 1e-12, .012, .012 + 1e-12, .5])
@pytest.mark.parametrize('duplicate', [False, True])
def test_retained_reference_matching_keeps_strict_boundary_and_all_source_points(distance, duplicate):
    # Separated sources make the nearest match and strict threshold unambiguous.
    reference = np.column_stack([np.arange(30), np.zeros(30), np.zeros(30)])
    points = reference + [0., distance, 0.]
    if duplicate:
        points = np.repeat(points, 2, axis=0)
    colors = np.column_stack([np.arange(len(points)), np.zeros(len(points)),
                              np.full(len(points), 120)]).astype(np.uint8)
    checker = ObservedClearance.__new__(ObservedClearance)
    checker.ik = SimpleNamespace(fk=lambda q: (np.zeros(3), np.eye(3)))
    checker.set_state = lambda state: None
    checker.points = points.copy()
    checker.colors = colors.copy()
    checker.tree = cKDTree(checker.points)
    checker.source_frames = []
    checker.carried_local = checker.carried_rgb = None
    report = checker.separate_carried_surfaces(
        {'q': np.zeros(7), 'step': 10}, reference, 7, preserve_reference=True)
    expected_match = distance < .012
    assert report['reference_available'] is expected_match
    if expected_match:
        assert report['reference_visible_fraction'] == 1.
        assert report['observed_points'] == len(points)
        assert len(checker.points) == len(checker.colors) == 0
        np.testing.assert_array_equal(checker.carried_local, reference)
        assert not report['rigid_attachment_verified']
    else:
        np.testing.assert_array_equal(checker.points, points)
        np.testing.assert_array_equal(checker.colors, colors)
        assert checker.carried_local is None
    # The archived input is never rewritten or thinned by matching.
    np.testing.assert_array_equal(reference[:, 1:], np.zeros((30, 2)))


def test_empty_current_scene_cannot_support_a_retained_reference():
    checker = ObservedClearance.__new__(ObservedClearance)
    checker.ik = SimpleNamespace(fk=lambda q: (np.zeros(3), np.eye(3)))
    checker.set_state = lambda state: None
    checker.points = np.empty((0, 3))
    checker.colors = np.empty((0, 3), np.uint8)
    checker.tree = cKDTree(checker.points)
    checker.source_frames = []
    checker.carried_local = checker.carried_rgb = None
    reference = np.column_stack([np.arange(30), np.zeros((30, 2))])
    report = checker.separate_carried_surfaces(
        {'q': np.zeros(7), 'step': 10}, reference, 7, preserve_reference=True)
    assert report['reference_available'] is False
    assert checker.carried_local is None
    assert checker.points.shape == (0, 3)
