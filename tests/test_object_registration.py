import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.object_registration import (rigid_fit, correspondence_seeds, fit_evidence,
    depth_checks, register_object, save_registered_cloud)


def shape():
    rng = np.random.default_rng(8)
    xyz = rng.uniform(-1, 1, (900, 3))*[.08, .045, .025]
    # An asymmetric extension constrains shape pose independently of PCA sign.
    xyz[:200] += [.07, .025, .028]
    rgb = np.clip((xyz+[.08, .045, .025])*[1200, 1800, 2300], 0, 255).astype(np.uint8)
    return xyz, rgb


def test_rigid_rotation_and_translation_recovered_despite_outlier_correspondences():
    old, _ = shape()
    r = Rotation.from_euler('xyz', [74, -31, 123], degrees=True).as_matrix()
    t = np.array([.23, -.11, .94])
    current = old@r.T+t
    current[:140] += .2
    seeds = correspondence_seeds(old[:500], current[:500])
    assert seeds
    assert any(np.max(np.linalg.norm(old@a.T+b-(old@r.T+t), axis=1)) < 1e-6 for a, b in seeds)
    with pytest.raises(ValueError, match='Collinear'):
        rigid_fit(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))


def test_occlusion_is_not_free_space_and_missing_depth_is_not_support():
    xy = np.stack(np.meshgrid(np.arange(-10, 11), np.arange(-10, 11)), -1).reshape(-1, 2)/1000
    points = np.column_stack([xy, np.ones(len(xy))])
    frame = {'K': np.array([[1000., 0., 32.], [0., 1000., 32.], [0., 0., 1.]]),
        'world_from_camera': np.eye(4), 'depth_m': np.ones((64, 64)), 'camera': 'agentview', 'step': 40}
    report = depth_checks(points, [frame])[0]
    assert report['supported_pixels'] == len(points)
    frame['depth_m'][:] = .7
    report = depth_checks(points, [frame])[0]
    assert report['supported_pixels'] == report['free_space_contradictions'] == 0
    frame['depth_m'][:] = 1.2
    assert depth_checks(points, [frame])[0]['contradiction_fraction'] == 1
    frame['depth_m'][:] = np.nan
    report = depth_checks(points, [frame])[0]
    assert report['known_depth_pixels'] == report['supported_pixels'] == 0


def test_identical_geometry_with_wrong_appearance_is_rejected():
    xyz, rgb = shape()
    good = fit_evidence(xyz, rgb, xyz, rgb, np.eye(3), np.zeros(3))
    assert not good['rejection_reasons']
    bad = fit_evidence(xyz, np.full_like(rgb, 10), xyz, np.full_like(rgb, 240), np.eye(3), np.zeros(3))
    assert 'appearance_not_supported' in bad['rejection_reasons']


def test_exact_appearance_ranks_above_increasing_color_error():
    xyz, _ = shape()
    rgb = np.full((len(xyz), 3), 100, np.uint8)
    evidence = [fit_evidence(xyz, rgb, xyz, rgb + difference,
        np.eye(3), np.zeros(3)) for difference in (0, 1, 5, 20)]
    assert all(not item['rejection_reasons'] for item in evidence)
    assert evidence[0]['median_rgb_distance'] == 0.
    assert all(first['score'] > second['score']
        for first, second in zip(evidence, evidence[1:])), evidence


def test_pipeline_rejects_two_indistinguishable_locations(monkeypatch, tmp_path):
    xyz, rgb = shape()
    current = np.concatenate([xyz+[.2, 0, 1], xyz+[-.2, 0, 1]])
    cloud = tmp_path/'current.npz'; np.savez(cloud, world_xyz_m=current, rgb=np.tile(rgb, (2, 1)))
    groups = tmp_path/'groups.npz'; np.savez(groups, point_group_ids=np.repeat([1, 2], len(xyz)))
    inspection = {'cloud_file': str(cloud), 'point_group_ids_file': str(groups), 'source_step': 40}
    old = {'world_xyz_m': xyz, 'rgb': rgb}
    monkeypatch.setattr('sketchik.object_registration.feature_correspondences', lambda *args: (xyz, xyz+[.2, 0, 1]))
    # Native checks are tested independently above; this tests pose competition.
    monkeypatch.setattr('sketchik.object_registration.depth_checks', lambda *args: [
        {'supported_pixels': 100, 'support_fraction_of_testable': 1., 'known_depth_pixels': 100, 'contradiction_fraction': 0.}])
    result = register_object(old, [], inspection, [], lambda *args: None, budget_s=20)
    assert result['status'] == 'unavailable' and result['reason'] == 'ambiguous_pose_or_identity'
    assert result['transform'] is None and result['agent_requests'] == result['robot_actions'] == 0


def test_budget_exhaustion_cannot_accept_an_unsearched_winner(monkeypatch, tmp_path):
    xyz, rgb = shape()
    cloud = tmp_path/'current.npz'; np.savez(cloud, world_xyz_m=xyz, rgb=rgb)
    groups = tmp_path/'groups.npz'; np.savez(groups, point_group_ids=np.ones(len(xyz), int))
    monkeypatch.setattr('sketchik.object_registration.feature_correspondences', lambda *args: (np.empty((0, 3)), np.empty((0, 3))))
    result = register_object({'world_xyz_m': xyz, 'rgb': rgb}, [],
        {'cloud_file': str(cloud), 'point_group_ids_file': str(groups), 'source_step': 40}, [], lambda *args: None, budget_s=0)
    assert result['status'] == 'unavailable' and result['reason'] == 'search_budget_exhausted'


def test_other_known_instance_cannot_be_mistaken_for_the_slipped_object(monkeypatch, tmp_path):
    xyz, rgb = shape()
    other = xyz+[.2, 0, 1]
    target = xyz+[-.2, 0, 1]
    cloud = tmp_path/'current.npz'; np.savez(cloud, world_xyz_m=np.concatenate([other, target]), rgb=np.tile(rgb, (2, 1)))
    groups = tmp_path/'groups.npz'; np.savez(groups, point_group_ids=np.repeat([1, 2], len(xyz)))
    monkeypatch.setattr('sketchik.object_registration.feature_correspondences', lambda *args: (xyz, other))
    monkeypatch.setattr('sketchik.object_registration.depth_checks', lambda *args: [
        {'supported_pixels': 100, 'support_fraction_of_testable': 1., 'known_depth_pixels': 100, 'contradiction_fraction': 0.}])
    result = register_object({'world_xyz_m': xyz, 'rgb': rgb}, [],
        {'cloud_file': str(cloud), 'point_group_ids_file': str(groups), 'source_step': 40}, [], lambda *args: None,
        budget_s=20, other_observed_points=other)
    assert result['status'] == 'accepted' and result['best_hypothesis']['candidate_id'] == 'G2'
    assert result['excluded_identity_conflicts'][0]['candidate_id'] == 'G1'
    r, t = np.asarray(result['transform']['rotation']), np.asarray(result['transform']['translation_m'])
    assert np.percentile(np.linalg.norm(xyz@r.T+t-target, axis=1), 95) < .003


def test_registered_table_preserves_all_original_points_and_capture_provenance(tmp_path):
    xyz, rgb = shape()
    old = {'world_xyz_m': xyz.copy(), 'rgb': rgb, 'pixel_xy': np.arange(len(xyz)*2).reshape(-1, 2),
        'source_index': np.zeros(len(xyz), np.uint16)}
    r = Rotation.from_euler('y', 83, degrees=True).as_matrix(); t = np.array([.1, -.2, 1.])
    registration = {'status': 'accepted', 'source_step': 40, 'transform': {'rotation': r.tolist(), 'translation_m': t.tolist()}}
    sources = [{'observation_id': 'old', 'source_step': 10, 'camera': 'agentview'}]
    path = save_registered_cloud(old, sources, registration, tmp_path/'accepted')
    with np.load(path) as data:
        np.testing.assert_array_equal(data['original_world_xyz_m'], xyz)
        np.testing.assert_allclose(data['world_xyz_m'], xyz@r.T+t)
        for key in ('rgb', 'pixel_xy', 'source_index'): np.testing.assert_array_equal(data[key], old[key])
    np.testing.assert_array_equal(old['world_xyz_m'], xyz)
    assert json.loads((path.parent/'registration.json').read_text())['original_sources'] == sources
    with pytest.raises(ValueError, match='Unavailable'):
        save_registered_cloud(old, sources, {'status': 'unavailable'}, tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


def test_repeated_registration_preserves_native_capture_coordinates(tmp_path):
    xyz,rgb=shape()
    old={'world_xyz_m':xyz.copy(),'rgb':rgb,'pixel_xy':np.zeros((len(xyz),2),int),
        'source_index':np.zeros(len(xyz),np.uint16)}
    def fitted(translation,step):
        return {'status':'accepted','source_step':step,'transform':{
            'rotation':np.eye(3).tolist(),'translation_m':translation}}
    first=save_registered_cloud(old,[],fitted([.3,0,0],20),tmp_path/'first')
    with np.load(first) as data:updated={key:data[key].copy() for key in data.files}
    second=save_registered_cloud(updated,[],fitted([.1,.2,0],30),tmp_path/'second')
    with np.load(second) as data:
        np.testing.assert_array_equal(data['original_world_xyz_m'],xyz)
        np.testing.assert_allclose(data['world_xyz_m'],xyz+[.4,.2,0])
        np.testing.assert_array_equal(data['pixel_xy'],old['pixel_xy'])


def test_native_features_map_capture_pixels_to_the_revised_point_coordinates():
    from sketchik.object_registration import feature_correspondences
    # Real SIFT and calibrated RGB-D, with a transported texture. The stored
    # original pixels still refer to capture coordinates after a pose update.
    rng=np.random.default_rng(2026)
    texture=rng.integers(0,256,(160,160,3),dtype=np.uint8)
    original_rgb=np.zeros((1024,1024,3),np.uint8)
    original_rgb[300:460,350:510]=texture
    current_rgb=np.zeros_like(original_rgb)
    current_rgb[300:460,710:870]=texture
    pixels=np.stack(np.meshgrid(np.arange(350,510),np.arange(300,460)),axis=-1).reshape(-1,2)
    captured=np.column_stack([(pixels[:,0]-512)/600,(pixels[:,1]-512)/600,np.ones(len(pixels))])
    old={'world_xyz_m':captured+[.3,0,0],'original_world_xyz_m':captured,
        'pixel_xy':pixels,'source_index':np.zeros(len(pixels),np.uint16)}
    calibration={'K':np.array([[600.,0,512],[0,600.,512],[0,0,1.]]),
        'depth_m':np.ones((1024,1024),np.float32),'world_from_camera':np.eye(4),'camera':'agentview'}
    source_frame={**calibration,'rgb':original_rgb,'step':10}
    current_frame={**calibration,'rgb':current_rgb,'step':30}
    a,b=feature_correspondences(old,[{'observation_id':'capture','camera':'agentview'}],
        lambda *args:source_frame,[current_frame])
    assert len(a)>=8
    from scipy.spatial import cKDTree
    assert cKDTree(old['world_xyz_m']).query(a)[0].max()<1e-12
    # SIFT locations can differ by a few pixels even for translated texture;
    # point identity and the robust metric displacement must still be correct.
    np.testing.assert_allclose(np.median(b-a,axis=0),[.3,0,0],atol=1e-12)
    assert correspondence_seeds(a,b)


def test_degenerate_observed_feature_fit_does_not_abort_geometry_fallback(monkeypatch,tmp_path):
    # Frozen native RGB-D feature correspondences, v174 step2368. Repeated
    # texture matches passed the minimal RANSAC sample but its final inlier
    # refit was collinear. No simulation object state or model calls are used.
    from pathlib import Path
    with np.load(Path(__file__).parent/'data/observed_duplicate_feature_pairs.npz') as data:
        pairs_old,pairs_new=data['old_xyz_m'].copy(),data['new_xyz_m'].copy()
    xyz,rgb=shape();current=xyz+[.2,0,1]
    cloud=tmp_path/'current.npz';np.savez(cloud,world_xyz_m=current,rgb=rgb)
    groups=tmp_path/'groups.npz';np.savez(groups,point_group_ids=np.ones(len(xyz),int))
    monkeypatch.setattr('sketchik.object_registration.feature_correspondences',lambda *a:(pairs_old,pairs_new))
    monkeypatch.setattr('sketchik.object_registration.depth_checks',lambda *a:[
        {'supported_pixels':100,'support_fraction_of_testable':1.,'known_depth_pixels':100,'contradiction_fraction':0.}])
    result=register_object({'world_xyz_m':xyz,'rgb':rgb},[],
        {'cloud_file':str(cloud),'point_group_ids_file':str(groups),'source_step':40},[],lambda *a:None,budget_s=20)
    assert result['status']=='accepted'
    rotation=np.asarray(result['transform']['rotation']);translation=np.asarray(result['transform']['translation_m'])
    assert np.percentile(np.linalg.norm(xyz@rotation.T+translation-current,axis=1),95)<.003
    assert result['agent_requests']==result['robot_actions']==0
