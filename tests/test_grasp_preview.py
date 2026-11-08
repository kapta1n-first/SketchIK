import base64

import numpy as np
from PIL import Image
import pytest

from sketchik.grasp_preview import mask_cloud
from sketchik.libero import back_project


def make_observation(tmp_path):
    folder = tmp_path / 'obs'
    folder.mkdir()
    Image.new('RGB', (4, 4), 'red').save(folder/'agentview.png')
    depth = np.full((4, 4), .6)
    depth[0, 0] = np.nan
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [.1, -.2, .3]
    frame = dict(depth_m=depth, K=np.array([[3., 0, 1], [0, 4., 1], [0, 0, 1]]),
                 world_from_camera=transform)
    np.savez(folder/'agentview.npz', **frame)
    return folder, frame


def candidate(mask, index, score):
    return dict(id=index, score=score, box_xyxy=[0, 0, 3, 3], mask_shape=list(mask.shape),
                mask_packbits_base64=base64.b64encode(np.packbits(mask.ravel())).decode())


def test_mask_retains_all_valid_points_in_calibrated_world_frame(tmp_path, monkeypatch):
    folder, frame = make_observation(tmp_path)
    mask = np.ones((4, 4), dtype=bool)
    mask[3, 3] = False
    monkeypatch.setattr('sketchik.grasp_preview.request', lambda *a: dict(
        candidates=[candidate(mask, 0, .7)], inference_wall_s=.1))
    output = tmp_path/'result'
    output.mkdir()
    points, colors, _ = mask_cloud(folder, 'agentview', 'observed object', None, '', output)
    pixels = [(x, y) for y in range(4) for x in range(4) if mask[y, x] and np.isfinite(frame['depth_m'][y, x])]
    assert len(points) == 14
    np.testing.assert_allclose(points, [back_project(frame, xy) for xy in pixels])
    np.testing.assert_allclose(colors, np.tile([1., 0, 0], (14, 1)))


def test_multiple_masks_do_not_silently_select_highest_score(tmp_path, monkeypatch):
    folder, _ = make_observation(tmp_path)
    masks = [candidate(np.ones((4, 4), dtype=bool), i, score) for i, score in enumerate([.99, .6])]
    monkeypatch.setattr('sketchik.grasp_preview.request', lambda *a: dict(candidates=masks, inference_wall_s=.1))
    output = tmp_path/'result'
    output.mkdir()
    with pytest.raises(ValueError, match='no top-score selection'):
        mask_cloud(folder, 'agentview', 'observed object', None, '', output)
    assert (output/'sam_masks.png').exists()
    assert not (output/'observed_object.npz').exists()
