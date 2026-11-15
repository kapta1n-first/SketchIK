import numpy as np
import pytest
from PIL import Image

from sketchik.cloud_selection import height_auxiliary, public_view, rasterize, resolve_cloud_point, write_views


def _project(points, eye, size):
    forward = -np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    h, v, depth = points @ right, points @ down, points @ forward
    scale = (size - 120) / max(np.ptp(h), np.ptp(v), 1e-6)
    x = np.rint((h - (h.min() + h.max()) / 2) * scale + size / 2).astype(int)
    y = np.rint((v - (v.min() + v.max()) / 2) * scale + size / 2 + 20).astype(int)
    return x, y, depth


@pytest.mark.parametrize('footprint', [3, 5, 9])
def test_raster_matches_brute_force_zbuffer_including_edges_and_ties(footprint):
    rng = np.random.default_rng(14)
    points = rng.uniform(-1, 1, (90, 3))
    points[1] = points[0]  # Equal depth and projected centre: lower ID wins.
    points[3, :2] = points[2, :2]
    points[3, 2] = points[2, 2] - .3  # Farther point at same centre loses.
    rgb = rng.integers(0, 256, (len(points), 3), dtype=np.uint8)
    size = 160
    eye = (.3, -.8, 1.)
    image, actual, meta = rasterize(points, rgb, 'rgb', eye, size=size,
                                    footprint_pixels=footprint)
    x, y, depth = _project(points, eye, size)
    expected = np.full((size, size), -1, np.int32)
    radius = footprint // 2
    for py in range(45, size - 25):
        for px in range(size):
            candidates = [i for i in range(len(points))
                          if abs(x[i] - px) <= radius and abs(y[i] - py) <= radius]
            if candidates:
                expected[py, px] = min(candidates, key=lambda i: (depth[i], i))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.asarray(image)[actual >= 0], rgb[actual[actual >= 0]])
    assert meta['point_footprint_pixels'] == footprint
    assert np.all(actual[:45] == -1)
    assert np.all(actual[size - 25:] == -1)


def test_new_views_and_height_use_same_nine_pixel_footprint(tmp_path):
    rng = np.random.default_rng(4)
    points = rng.uniform(-.3, .3, (35, 3))
    rgb = rng.integers(0, 256, (35, 3), dtype=np.uint8)
    cloud = tmp_path / 'cloud.npz'
    np.savez_compressed(cloud, world_xyz_m=points, rgb=rgb)
    views = write_views(cloud, [], tmp_path, 'sample', (.4, -1., .7), role='test')
    assert len(views) == 3
    for view in views:
        assert view['point_footprint_pixels'] == 9
        assert public_view(view)['point_footprint_half_width_m'] == 4/view['pixels_per_metre']
        auxiliary = height_auxiliary(view, tmp_path)
        assert auxiliary['point_footprint_pixels'] == 9
        with np.load(view['index_map_file']) as data:
            mapping = data['point_index']
        with np.load(auxiliary['index_map_file']) as data:
            np.testing.assert_array_equal(mapping, data['point_index'])
        assert np.any(mapping >= 0)
        assert np.all(mapping[:45] == -1)
        assert np.all(mapping[-25:] == -1)
        bounds = view['legend_pixel_bounds']
        x0, y0, x1, y1 = bounds
        assert np.all(mapping[y0:y1, x0:x1] == -1)
        py, px = np.argwhere(mapping >= 0)[0]
        point, provenance = resolve_cloud_point(
            {'view_id': view['view_id'], 'pixel_xy': [int(px), int(py)]}, views, views)
        np.testing.assert_array_equal(point, points[mapping[py, px]])
        assert provenance['cloud_point_index'] == mapping[py, px]
        with pytest.raises(ValueError, match='unknown/background'):
            resolve_cloud_point({'view_id': view['view_id'], 'pixel_xy': [0, 0]}, views, views)
        with pytest.raises(ValueError, match='unknown/background'):
            resolve_cloud_point({'view_id': view['view_id'], 'pixel_xy': [x0, y0]}, views, views)


def test_footprint_requires_positive_odd_integer():
    points = np.zeros((30, 3))
    rgb = np.zeros((30, 3), np.uint8)
    for invalid in (0, 2, -3, 3.0, True):
        with pytest.raises(ValueError, match='positive odd'):
            rasterize(points, rgb, 'rgb', footprint_pixels=invalid)


def test_group_labels_mask_exact_pixels_in_rgb_and_height(tmp_path):
    rng=np.random.default_rng(21)
    first=rng.uniform(-.02,.02,(45,3))+[-.15,0,.6]
    second=rng.uniform(-.02,.02,(45,3))+[.15,0,.6]
    points=np.vstack((first,second))
    rgb=rng.integers(0,256,(len(points),3),dtype=np.uint8)
    cloud=tmp_path/'groups.npz'
    np.savez_compressed(cloud,world_xyz_m=points,rgb=rgb)
    group_ids=np.r_[np.ones(45,np.int32),np.full(45,2,np.int32)]
    views=write_views(cloud,[],tmp_path,'groups',(0,-1,.7),role='test',group_ids=group_ids)
    for view in views:
        assert view['point_footprint_pixels']==9
        labels=view['component_label_bounds_xyxy']
        assert {label['group_id'] for label in labels}=={1,2}
        assert public_view(view)['component_label_bounds_xyxy']==labels
        auxiliary=height_auxiliary(view,tmp_path)
        with np.load(view['index_map_file']) as data:mapping=data['point_index']
        with np.load(auxiliary['index_map_file']) as data:
            np.testing.assert_array_equal(mapping,data['point_index'])
        for label in labels:
            x0,y0,x1,y1=label['bounds_xyxy']
            assert np.all(mapping[y0:y1,x0:x1]==-1)
            with pytest.raises(ValueError,match='unknown/background'):
                resolve_cloud_point({'view_id':view['view_id'],
                                     'pixel_xy':[x0,y0]},views,views)
        image=np.asarray(Image.open(view['image_file']))
        selectable=mapping>=0
        assert np.any(selectable)
        np.testing.assert_array_equal(image[selectable],rgb[mapping[selectable]])
        py,px=np.argwhere(selectable)[0]
        point,_=resolve_cloud_point({'view_id':view['view_id'],
                                     'pixel_xy':[int(px),int(py)],
                                     'group_id':int(group_ids[mapping[py,px]])},views,views)
        np.testing.assert_array_equal(point,points[mapping[py,px]])
