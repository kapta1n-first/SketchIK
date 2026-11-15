import numpy as np
import pytest

from sketchik.libero import back_project
from sketchik.scene_cloud import frame_points, save_scene, region_points


def frame(step=4):
    transform = np.eye(4)
    transform[:3,3] = [.1,.2,.3]
    depth = np.arange(1,31,dtype=float).reshape(5,6) / 5
    depth[0,:3] = [0, np.nan, np.inf]
    return {'depth_m': depth, 'rgb': np.arange(90,dtype=np.uint8).reshape(5,6,3),
            'K': np.array([[20.,0,3],[0,20.,2],[0,0,1]]),
            'world_from_camera': transform, 'step': step}


def test_full_frame_preserves_all_positive_depth_and_source_pixels_including_far_points():
    data = frame()
    xyz, rgb, pixels = frame_points(data)
    assert len(xyz) == 27 and len(rgb) == 27
    assert xyz[:,2].max() > 3  # No silent workspace/depth crop.
    for point, color, pixel in zip(xyz, rgb, pixels):
        np.testing.assert_allclose(point, back_project(data,pixel),atol=4e-7)
        np.testing.assert_array_equal(color,data['rgb'][pixel[1],pixel[0]])


def test_whole_scene_file_keeps_provenance_and_is_not_object_dimensions(tmp_path):
    frames = {'front': frame(), 'wrist': frame()}
    report = save_scene(frames,tmp_path,'saved')
    assert report['point_count'] == 54
    assert report['sam_requests'] == report['molmo_requests'] == 0
    assert not report['scene_span_is_object_dimensions']
    with np.load(report['point_cloud_file']) as data:
        assert len(data['world_xyz_m']) == 54
        np.testing.assert_array_equal(data['source_id'],[0]*27+[1]*27)
        assert data['pixel_xy'].shape == (54,2)
    assert (tmp_path/'scene_3d.png').exists()
    with pytest.raises(ValueError,match='one refreshed'):
        save_scene({'front':frame(), 'wrist':frame(5)},tmp_path/'mismatch','saved')


def test_optional_image_rectangle_is_not_required_for_scene_or_movement():
    points = region_points(frame(),[3,0,6,5])
    assert len(points) == 15
    with pytest.raises(ValueError):
        region_points(frame(),[3,0,8,5])
