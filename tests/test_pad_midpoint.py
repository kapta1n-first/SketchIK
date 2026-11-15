import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.pose_binding import compile_binding


def packet(pads):
    return {
        'eef': {'position': [0.2, -0.1, 0.7], 'quaternion_xyzw': [0, 0, 0, 1]},
        'source_step': 4,
        'robot_gripper_geometry': {
            'pad_centres_in_eef_tool_m': pads,
            'pad_midpoint_in_eef_tool_m': [99, 99, 99],
        },
    }


def binding(reference='pad_midpoint'):
    return {
        'anchor': {'kind': 'current_eef'}, 'reference': reference,
        'offset_m': [0.03, -0.02, 0.04], 'offset_frame': 'world',
        'orientation': {'rpy_world_rad': [0.3, -0.4, 0.8]},
    }


def compile_for(reference, pads):
    return compile_binding(binding(reference), packet(pads), lambda *_: None)


def test_midpoint_aligns_anchor_after_full_rotation_and_translation():
    pads = [[0.04, 0.012, -0.01], [-0.02, -0.006, -0.03]]
    pose, audit = compile_for('pad_midpoint', pads)
    rotation = Rotation.from_euler('xyz', [0.3, -0.4, 0.8])
    midpoint = np.mean(pads, axis=0)
    expected = np.array([0.2, -0.1, 0.7]) + [0.03, -0.02, 0.04]
    np.testing.assert_allclose(np.array(pose['position']) + rotation.apply(midpoint), expected)
    np.testing.assert_allclose(audit['reference_offset_world_m'], rotation.apply(midpoint))
    np.testing.assert_allclose(pose['quaternion_xyzw'], rotation.as_quat())


def test_midpoint_uses_current_asymmetric_pad_geometry_each_compile():
    current = packet([[0.05, 0.01, 0], [-0.03, 0.03, -0.02]])
    first, _ = compile_binding(binding(), current, lambda *_: None)
    current['robot_gripper_geometry']['pad_centres_in_eef_tool_m'] = [[0.07, 0.02, 0.01], [-0.01, 0.04, -0.01]]
    second, _ = compile_binding(binding(), current, lambda *_: None)
    rotation = Rotation.from_euler('xyz', [0.3, -0.4, 0.8])
    np.testing.assert_allclose(np.array(second['position']) - first['position'],
                               -rotation.apply([0.02, 0.01, 0.01]))


@pytest.mark.parametrize('pads', [None, [], [[0, 0, 0]], [[0, 0, 0], [1, 2]],
                                     [[0, 0, 0], [float('nan'), 0, 0]],
                                     [[0, 0, 0], [float('inf'), 0, 0]]])
def test_midpoint_rejects_missing_or_invalid_pair(pads):
    with pytest.raises(ValueError, match='geometry unavailable|pad centre'):
        compile_for('pad_midpoint', pads)


@pytest.mark.parametrize('reference,index', [('pad_0', 0), ('pad_1', 1)])
def test_legacy_pad_bindings_still_select_individual_pad(reference, index):
    pads = [[0.05, 0.01, 0.02], [-0.03, -0.02, -0.01]]
    pose, audit = compile_for(reference, pads)
    expected = Rotation.from_euler('xyz', [0.3, -0.4, 0.8]).apply(pads[index])
    np.testing.assert_allclose(audit['reference_offset_world_m'], expected)
    np.testing.assert_allclose(pose['position'], np.array([0.23, -0.12, 0.74]) - expected)
