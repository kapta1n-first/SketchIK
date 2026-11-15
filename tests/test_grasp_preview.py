import base64

import numpy as np
from PIL import Image
import pytest

from sketchik.grasp_preview import CAMERAS, mask_clouds, points_from_mask
from sketchik.libero import back_project


def frame_for_test():
    depth = np.full((4, 4), .6)
    depth[0, 0] = np.nan
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [.1, -.2, .3]
    return dict(depth_m=depth, K=np.array([[3., 0, 1], [0, 4., 1], [0, 0, 1]]),
                world_from_camera=transform)


def test_mask_retains_all_valid_points_in_calibrated_world_frame():
    frame = frame_for_test()
    mask = np.ones((4, 4), dtype=bool)
    mask[3, 3] = False
    points, colors = points_from_mask(frame, mask, Image.new('RGB', (4, 4), 'red'))
    pixels = [(x, y) for y in range(4) for x in range(4) if mask[y, x] and np.isfinite(frame['depth_m'][y, x])]
    assert len(points) == 14
    np.testing.assert_allclose(points, [back_project(frame, xy) for xy in pixels])
    np.testing.assert_allclose(colors, np.tile([1., 0, 0], (14, 1)))


def test_rgb_depth_mask_must_have_same_resolution():
    with pytest.raises(ValueError, match='resolutions must match'):
        points_from_mask(frame_for_test(), np.ones((3, 3), dtype=bool), Image.new('RGB', (4, 4)))


def make_observation(tmp_path):
    folder = tmp_path/'obs'
    folder.mkdir()
    for index, camera in enumerate(CAMERAS):
        Image.new('RGB', (1024, 1024), 'red').save(folder/(camera+'.png'))
        transform = np.eye(4)
        transform[0, 3] = index
        np.savez(folder/(camera+'.npz'), depth_m=np.ones((1024, 1024)), K=np.eye(3),
                 world_from_camera=transform)
    return folder


def candidate(index, score):
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[2, 3] = True
    return dict(id=index, score=score, box_xyxy=[0, 0, 4, 4], mask_shape=list(mask.shape),
                mask_packbits_base64=base64.b64encode(np.packbits(mask.ravel())).decode())


def response(views):
    return dict(images=views, inference_wall_s=.1, model_pid=123, request_index=1)


def test_two_views_use_their_own_calibration_even_if_response_order_changes(tmp_path, monkeypatch):
    folder = make_observation(tmp_path)
    views = [dict(camera=camera, candidates=[candidate(0, .7)]) for camera in reversed(CAMERAS)]
    payloads = []
    def call(url, payload):
        payloads.append(payload)
        return response(views)
    monkeypatch.setattr('sketchik.grasp_preview.request', call)
    output = tmp_path/'result'
    output.mkdir()
    points, _, source = mask_clouds(folder, 'observed object', {}, '', output)
    assert len(payloads) == 1
    assert [v['camera'] for v in payloads[0]['images']] == list(CAMERAS)
    np.testing.assert_allclose(points, [[3, 2, 1], [4, 2, 1]])
    np.testing.assert_array_equal(np.load(output/'observed_object.npz')['camera_index'], [0, 1])
    assert source['cross_view_identity_certified'] is False


def test_highest_score_selection_is_explicitly_recorded_and_can_be_overridden(tmp_path, monkeypatch):
    folder = make_observation(tmp_path)
    views = [dict(camera=camera, candidates=[candidate(0, .99), candidate(1, .6)]) for camera in CAMERAS]
    monkeypatch.setattr('sketchik.grasp_preview.request', lambda *a: response(views))
    output = tmp_path/'result'
    output.mkdir()
    _, _, source = mask_clouds(folder, 'observed object', {CAMERAS[1]: 1}, '', output)
    assert source['views'][0]['selected_candidate_id'] == 0
    assert source['views'][0]['selected_score'] == .99
    assert source['views'][0]['selection'] == 'highest_sam_score_per_view'
    assert source['views'][1]['selected_candidate_id'] == 1
    assert source['views'][1]['selection'] == 'explicit_candidate_id'
    assert (output/'agentview/sam_masks.png').exists()
    assert (output/'observed_object.npz').exists()


def test_unobserved_view_does_not_get_a_fabricated_mask(tmp_path, monkeypatch):
    folder = make_observation(tmp_path)
    views = [dict(camera=CAMERAS[0], candidates=[]), dict(camera=CAMERAS[1], candidates=[candidate(0, .7)])]
    monkeypatch.setattr('sketchik.grasp_preview.request', lambda *a: response(views))
    output = tmp_path/'result'
    output.mkdir()
    points, _, source = mask_clouds(folder, 'observed object', {}, '', output)
    assert len(points) == 1
    assert source['views'][0]['status'] == 'no_detection'
    assert not (output/CAMERAS[0]/'selected_mask.png').exists()


def test_saved_256_images_are_not_silently_upscaled(tmp_path, monkeypatch):
    folder = make_observation(tmp_path)
    Image.new('RGB', (256, 256)).save(folder/'agentview.png')
    def unexpected(*a):
        raise AssertionError('must reject before invoking model')
    monkeypatch.setattr('sketchik.grasp_preview.request', unexpected)
    with pytest.raises(ValueError, match='do not upscale'):
        mask_clouds(folder, 'object', {}, '', tmp_path)


def test_part_query_reaches_sam_and_mask_selector_without_replacing_parent_identity(tmp_path, monkeypatch):
    folder = make_observation(tmp_path)
    views = [dict(camera=c, candidates=[candidate(0, .7)]) for c in CAMERAS]
    calls = []
    def call(url, payload):
        calls.append(payload)
        if len(calls) == 1:
            return response(views)
        return {'answer': '{"selected_ids": [1], "reason": "requested visible part"}'}
    monkeypatch.setattr('sketchik.grasp_preview.request', call)
    output = tmp_path/'result'
    output.mkdir()
    _, _, source = mask_clouds(folder, 'yellow and white mug', {}, 'sam', output,
                               sam_prompt='yellow handle of the yellow and white mug', molmo_url='molmo')
    assert calls[0]['prompt'] == 'yellow handle of the yellow and white mug'
    import json
    intent = json.loads(calls[1]['prompt'].splitlines()[-1])
    assert intent['target_description'] == 'yellow and white mug'
    assert intent['segmentation_description'] == calls[0]['prompt']
    assert source['views'][1]['status'] == 'molmo_none'


def test_image_region_hint_reaches_molmo_without_removing_sam_candidates(tmp_path, monkeypatch):
    folder = make_observation(tmp_path)
    views = [dict(camera=c, candidates=[candidate(0,.99), candidate(1,.4)]) for c in CAMERAS]
    region = {'camera': 'agentview', 'bounds_xyxy': [0,0,10,10],
              'description': 'visible support behind the partition', 'visible_evidence': 'separating edges visible'}
    calls = []
    def call(url, payload):
        calls.append(payload)
        if len(calls) == 1:
            return response(views)
        return {'answer': '{"selected_ids": [2], "reason": "visible support"}'}
    monkeypatch.setattr('sketchik.grasp_preview.request', call)
    output = tmp_path/'result'; output.mkdir()
    _, _, source = mask_clouds(folder, 'container', {}, 'sam', output,
        sam_prompt='support inside identified region', molmo_url='molmo', selection_context=region)
    import json
    intent = json.loads(calls[1]['prompt'].splitlines()[-1])
    assert intent['high_level_image_region'] == region
    assert len(intent['candidate_mapping']) == 4
    assert all(set(value) == {'camera'} for value in intent['candidate_mapping'].values())
    assert intent['allowed_ids'] == [1,2,3,4]
    assert source['views'][0]['selected_candidate_id'] == 1
    assert source['views'][0]['selected_score'] == .4
    assert source['views'][1]['status'] == 'molmo_none'
