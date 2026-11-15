import json

import numpy as np

from sketchik.surface_evidence import summarize_horizontal_surfaces


def frame_for_planes(heights, *, rotation=None, holes=None, slope=0.0):
    h, w = heights.shape
    k = np.array([[130.0, 0, w / 2], [0, 130.0, h / 2], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1, -1, -1]) if rotation is None else rotation
    pose[:3, 3] = [0.1, -0.2, 1.5]
    yy, xx = np.indices((h, w))
    rays = np.stack(((xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)), -1)
    world_rays = rays @ pose[:3, :3].T
    depth = (heights + slope * pose[0, 3] - pose[2, 3]) / (world_rays[..., 2] - slope * world_rays[..., 0])
    if holes is not None:
        depth[holes] = np.nan
    return {"depth_m": depth, "K": k, "world_from_camera": pose}


def test_horizontal_world_plane_survives_camera_rotation_and_serializes():
    angle = np.deg2rad(25)
    tilt = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])
    rotation = np.diag([1, -1, -1]) @ tilt
    frame = frame_for_planes(np.full((160, 200), 0.94), rotation=rotation)
    before = frame["depth_m"].copy()
    result = summarize_horizontal_surfaces(frame)
    json.dumps(result)
    patch = result["horizontal_patches"][0]
    assert abs(patch["world_height_m"] - 0.94) < 1e-4
    assert patch["visible_pixel_sample_count"] > 1000
    assert patch["normal_angle_to_z_deg"] < 0.1
    np.testing.assert_array_equal(frame["depth_m"], before)


def test_distinct_height_levels_and_depth_hole_remain_separate():
    heights = np.full((180, 220), 0.9)
    heights[:, :90] = 1.1
    holes = np.zeros_like(heights, dtype=bool)
    holes[50:80, 20:40] = True
    frame = frame_for_planes(heights, holes=holes)
    result = summarize_horizontal_surfaces(frame)
    levels = [p["world_height_m"] for p in result["horizontal_patches"]]
    assert any(abs(z - 0.9) < 1e-3 for z in levels)
    assert any(abs(z - 1.1) < 1e-3 for z in levels)
    assert all(p["height_residual_rmse_m"] < 0.01 for p in result["horizontal_patches"])
    assert not np.isfinite(frame["depth_m"][holes]).any()


def test_inclined_plane_does_not_claim_horizontal_patch():
    frame = frame_for_planes(np.full((180, 220), 0.9), slope=0.35)
    assert summarize_horizontal_surfaces(frame)["horizontal_patches"] == []


def test_small_image_and_bad_calibration_are_handled():
    frame = frame_for_planes(np.full((2, 2), 0.9))
    assert summarize_horizontal_surfaces(frame)["horizontal_patches"] == []
    try:
        summarize_horizontal_surfaces(frame, max_patches=0)
    except ValueError as error:
        assert "max_patches" in str(error)
    else:
        raise AssertionError("invalid max_patches accepted")
    frame["K"][:] = 0
    try:
        summarize_horizontal_surfaces(frame)
    except ValueError as error:
        assert "invertible" in str(error)
    else:
        raise AssertionError("singular K accepted")
