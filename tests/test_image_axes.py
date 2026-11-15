import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.image_axes import annotate_world_axes


def transform(rotation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def test_native_rgb_is_copied_without_resize_or_outside_panel_changes():
    rgb = np.random.default_rng(8).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    original = rgb.copy()
    annotated, metadata = annotate_world_axes(rgb, transform())
    assert annotated.shape == rgb.shape
    assert annotated.dtype == np.uint8
    np.testing.assert_array_equal(rgb, original)
    assert not np.shares_memory(rgb, annotated)
    x0, y0, x1, y1 = metadata["panel_bounds_xyxy"]
    outside = np.ones(rgb.shape[:2], dtype=bool)
    outside[y0:y1, x0:x1] = False
    np.testing.assert_array_equal(annotated[outside], rgb[outside])
    assert np.any(annotated[y0:y1, x0:x1] != rgb[y0:y1, x0:x1])
    assert metadata["annotation_pixels_not_surface"] is True
    assert metadata["metric_scale"] is False
    assert metadata["source_calibration"]["world_from_camera"] == np.eye(4).tolist()


@pytest.mark.parametrize("size", [(64, 80), (128, 160), (300, 240)])
def test_general_image_sizes(size):
    annotated, metadata = annotate_world_axes(np.zeros((*size, 3), np.uint8), transform())
    assert annotated.shape == (*size, 3)
    assert metadata["image_size_wh"] == [size[1], size[0]]
    x0, y0, x1, y1 = metadata["panel_bounds_xyxy"]
    assert 0 <= x0 < x1 <= size[1]
    assert 0 <= y0 < y1 <= size[0]


def test_identity_is_right_down_away_not_screen_y_flipped():
    _, metadata = annotate_world_axes(np.zeros((256, 256, 3), np.uint8), transform())
    axes = metadata["axes"]
    assert axes["+X"]["screen_delta_px"][0] > 0
    assert axes["+X"]["screen_delta_px"][1] == 0
    assert axes["+Y"]["screen_delta_px"][1] > 0
    assert axes["+Z"]["screen_delta_px"] == [0.0, 0.0]
    assert axes["+Z"]["near_optical_axis"] is True
    assert axes["+Z"]["viewer_relation"] == "away from viewer"


@pytest.mark.parametrize("angles", [(0, 0, 90), (180, 0, 0), (0, 180, 0), (25, -35, 45)])
def test_camera_rotations_use_inverse_rotation(angles):
    rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    calibration = transform(rotation)
    _, metadata = annotate_world_axes(np.zeros((256, 256, 3), np.uint8), calibration)
    for index, label in enumerate(("+X", "+Y", "+Z")):
        expected = rotation.T @ np.eye(3)[:, index]
        np.testing.assert_allclose(metadata["world_axes_camera"][label], expected, atol=1e-12)
        delta = np.array(metadata["axes"][label]["screen_delta_px"])
        np.testing.assert_allclose(
            delta, expected[:2] * metadata["compass_arm_length_px"], atol=1e-12,
        )


def test_quarter_turn_camera_puts_world_x_up_and_world_y_right():
    calibration = transform(np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    _, metadata = annotate_world_axes(np.zeros((256, 256, 3), np.uint8), calibration)
    np.testing.assert_array_equal(metadata["world_axes_camera"]["+X"], [0, -1, 0])
    np.testing.assert_array_equal(metadata["world_axes_camera"]["+Y"], [1, 0, 0])


def test_camera_z_flip_marks_axis_toward_viewer():
    calibration = transform(np.diag([1.0, -1.0, -1.0]))
    _, metadata = annotate_world_axes(np.zeros((256, 256, 3), np.uint8), calibration)
    axis = metadata["axes"]["+Z"]
    assert axis["near_optical_axis"] is True
    assert axis["viewer_relation"] == "toward viewer"
    assert metadata["axes"]["+Y"]["screen_delta_px"][1] < 0


def test_slightly_slanted_optical_axis_retains_projected_direction():
    calibration = transform(Rotation.from_euler("y", 5, degrees=True).as_matrix())
    _, metadata = annotate_world_axes(np.zeros((256, 256, 3), np.uint8), calibration)
    axis = metadata["axes"]["+Z"]
    assert axis["near_optical_axis"] is True
    assert axis["screen_delta_px"][0] < 0
    assert axis["viewer_relation"] == "away from viewer"


@pytest.mark.parametrize("camera_z", [-1.52e-7, 1.52e-7, -0.01, 0.01])
def test_near_image_plane_relation_tolerates_small_tilt_without_snapping(camera_z):
    calibration = transform(Rotation.from_euler("y", np.arcsin(camera_z)).as_matrix())
    _, metadata = annotate_world_axes(np.zeros((256, 256, 3), np.uint8), calibration)
    direction = metadata["world_axes_camera"]["+X"]
    assert direction[2] == pytest.approx(camera_z, abs=1e-15)
    assert metadata["axes"]["+X"]["viewer_relation"] == "in image plane"
    assert metadata["approximate_image_plane_abs_camera_z_threshold"] == 0.05
    assert metadata["axes"]["+X"]["screen_delta_px"][0] == pytest.approx(
        np.sqrt(1 - camera_z ** 2) * metadata["compass_arm_length_px"], abs=1e-12,
    )


@pytest.mark.parametrize("camera_z,relation", [(0.1, "away from viewer"), (-0.1, "toward viewer")])
def test_meaningful_depth_tilt_is_not_labelled_in_image_plane(camera_z, relation):
    calibration = transform(Rotation.from_euler("y", np.arcsin(camera_z)).as_matrix())
    _, metadata = annotate_world_axes(np.zeros((256, 256, 3), np.uint8), calibration)
    assert metadata["world_axes_camera"]["+X"][2] == pytest.approx(camera_z)
    assert metadata["axes"]["+X"]["viewer_relation"] == relation


def test_translation_does_not_change_direction_or_picture():
    rgb = np.zeros((256, 256, 3), np.uint8)
    calibration = transform(Rotation.from_euler("xyz", [20, 40, 15], degrees=True).as_matrix())
    first, first_meta = annotate_world_axes(rgb, calibration)
    calibration[:3, 3] = [2.0, -3.0, 5.0]
    second, second_meta = annotate_world_axes(rgb, calibration)
    np.testing.assert_array_equal(first, second)
    assert first_meta["world_axes_camera"] == second_meta["world_axes_camera"]
    assert first_meta["source_calibration"] != second_meta["source_calibration"]


@pytest.mark.parametrize("bad", [
    np.full((4, 4), np.nan), np.full((4, 4), np.inf), np.eye(3),
    np.diag([2.0, 1.0, 1.0, 1.0]), np.diag([-1.0, 1.0, 1.0, 1.0]),
    np.diag([1.0, 1.0, 1.0, 2.0]),
])
def test_invalid_transforms_rejected(bad):
    with pytest.raises(ValueError):
        annotate_world_axes(np.zeros((256, 256, 3), np.uint8), bad)


@pytest.mark.parametrize("bad", [
    np.zeros((256, 256), np.uint8), np.zeros((256, 256, 4), np.uint8),
    np.zeros((256, 256, 3), float), np.zeros((32, 256, 3), np.uint8),
])
def test_invalid_images_rejected(bad):
    with pytest.raises(ValueError):
        annotate_world_axes(bad, transform())


def test_source_identity_is_visible_in_existing_non_surface_panel_and_preserves_raw_rgb():
    source={'observation_id':'obs_00003','camera':'robot0_eye_in_hand'}
    rgb=np.zeros((1024,1024,3),np.uint8)
    plain,metadata=annotate_world_axes(rgb,transform())
    labelled,info=annotate_world_axes(rgb,transform(),source_identity=source)
    assert info['source_image']==source
    assert info['panel_bounds_xyxy']==metadata['panel_bounds_xyxy']
    x0,y0,x1,y1=info['panel_bounds_xyxy']
    changed=np.any(plain!=labelled,axis=-1)
    assert changed.any() and not changed[y1:,:].any() and not changed[:,x1:].any()
    assert not rgb.any()
