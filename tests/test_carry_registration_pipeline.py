"""Synthetic RGB-D regression; no simulator or task-object ground truth inputs."""
import numpy as np
from scipy.spatial.transform import Rotation

from sketchik.scene_cloud import frame_points
from sketchik.object_registration import register_object
from sketchik.carry_reference import initialize_reference, suspend, relocalize_reference


def native_frame(xyz, rgb, camera_rotation, camera_origin, camera):
    size=1024
    k=np.array([[700.,0.,512.],[0.,700.,512.],[0.,0.,1.]])
    local=(xyz-camera_origin)@camera_rotation
    projection=local@k.T
    uv=np.rint(projection[:,:2]/projection[:,2,None]).astype(int)
    valid=(local[:,2]>0)&np.all((uv>=0)&(uv<size),axis=1)
    indices=np.flatnonzero(valid);flat=uv[indices,1]*size+uv[indices,0]
    order=np.lexsort((local[indices,2],flat));flat=flat[order];indices=indices[order]
    chosen=np.r_[True,flat[1:]!=flat[:-1]];indices=indices[chosen]
    depth=np.full((size,size),np.nan);image=np.zeros((size,size,3),np.uint8)
    depth[uv[indices,1],uv[indices,0]]=local[indices,2]
    image[uv[indices,1],uv[indices,0]]=rgb[indices]
    transform=np.eye(4);transform[:3,:3]=camera_rotation;transform[:3,3]=camera_origin
    return dict(depth_m=depth,rgb=image,K=k,world_from_camera=transform,step=24,camera=camera)


def test_actual_rigid_registration_of_occluded_dual_rgbd_rebuilds_original_full_reference(tmp_path):
    rng=np.random.default_rng(8)
    old=rng.uniform(-1,1,(1600,3))*[.08,.045,.025]
    old[:360]+=[.07,.025,.028]
    rgb=np.clip((old+[.08,.045,.025])*[1200,1800,2300],0,255).astype(np.uint8)
    old+=[0.,0.,.95]
    rot=Rotation.from_euler('xyz',[.1,-.07,.31]).as_matrix()
    shift=np.array([.025,-.015,.01])
    centre=np.mean(old,axis=0);translation=centre+shift-rot@centre
    moved=old@rot.T+translation
    keep=old[:,0]>-.035
    frames=[native_frame(moved[keep],rgb[keep],np.eye(3),np.zeros(3),'agentview'),
        native_frame(moved[keep],rgb[keep],Rotation.from_euler('y',np.pi).as_matrix(),np.array([0.,0.,1.9]),'robot0_eye_in_hand')]
    measured=[frame_points(frame) for frame in frames]
    current=np.concatenate([value[0] for value in measured]);colors=np.concatenate([value[1] for value in measured])
    cloud=tmp_path/'current.npz';groups=tmp_path/'groups.npz'
    np.savez(cloud,world_xyz_m=current,rgb=colors)
    np.savez(groups,point_group_ids=np.ones(len(current),np.int32))
    registration=register_object({'world_xyz_m':old,'rgb':rgb},[],
        {'cloud_file':str(cloud),'point_group_ids_file':str(groups),'source_step':24},
        frames,lambda *args:None,budget_s=8.)
    import json
    (tmp_path/'registration.json').write_text(json.dumps(registration,indent=2))
    assert registration['status']=='accepted',registration['reason']
    estimate=old@np.array(registration['transform']['rotation']).T+registration['transform']['translation_m']
    assert np.percentile(np.linalg.norm(estimate-moved,axis=1),95)<.003
    ref=initialize_reference({'object_id':'object','source_step':4,'source_cloud_file':'synthetic_original.npz'},
        old-[0,0,1.],rgb,2,np.arange(len(old)),world_xyz_m=old)
    frozen=ref['local_xyz_m'].copy();identity=ref['reference_id']
    suspend(ref,23,'A observed changed relation',kind='relative_pose_changed')
    state={'step':24,'gripper_command':1.,'position':[.01,.02,1.02],
        'quaternion_xyzw':Rotation.from_euler('xyz',[.3,-.1,.2]).as_quat().tolist()}
    result=relocalize_reference(ref,object_id='object',grasp_epoch=2,explicit_pose_rebind=True,
        point_set_scope='one_rigid_object',registration=registration,frozen_world_xyz_m=old,current_state=state)
    assert result['status']=='relocalized'
    np.testing.assert_allclose(ref['local_xyz_m'],(estimate-state['position'])@Rotation.from_quat(state['quaternion_xyzw']).as_matrix())
    np.testing.assert_array_equal(ref['original_local_xyz_m'],frozen)
    np.testing.assert_array_equal(ref['point_ids'],np.arange(len(old)))
    assert len(ref['local_xyz_m'])==1600 and ref['reference_id']==identity
    assert result['holding_certified'] is False and result['agent_requests']==result['robot_actions']==0


def test_uniform_or_repeated_rgb_never_invents_point_correspondence():
    from sketchik.object_registration import color_correspondences
    points=np.arange(90).reshape(30,3)
    a,b=color_correspondences(points,np.full((30,3),120),points+.1,np.full((30,3),120))
    assert len(a)==len(b)==0
    colors=np.repeat(np.array([[10,20,30],[100,110,120],[200,210,220]]),10,axis=0)
    a,b=color_correspondences(points,colors,points+.1,colors)
    assert len(a)==len(b)==0


def test_mutual_unique_rgb_only_initializes_metric_correspondence():
    from sketchik.object_registration import color_correspondences
    rng=np.random.default_rng(78)
    xyz=rng.normal(size=(80,3))
    rgb=rng.integers(0,256,(80,3),dtype=np.uint8)
    keep=np.arange(0,80,2)
    a,b=color_correspondences(xyz,rgb,xyz[keep]+[.1,-.2,.3],rgb[keep])
    assert len(a)==len(keep)
    np.testing.assert_allclose(b-a,np.broadcast_to([.1,-.2,.3],a.shape))


def test_color_initialization_cannot_choose_between_two_indistinguishable_objects(tmp_path,monkeypatch):
    rng=np.random.default_rng(8)
    xyz=rng.uniform(-1,1,(900,3))*[.08,.045,.025];xyz[:200]+=[.07,.025,.028]
    rgb=np.clip((xyz+[.08,.045,.025])*[1200,1800,2300],0,255).astype(np.uint8)
    cloud=tmp_path/'ambiguous.npz';groups=tmp_path/'ambiguous_groups.npz'
    np.savez(cloud,world_xyz_m=np.concatenate([xyz+[.2,0,1],xyz+[-.2,0,1]]),rgb=np.tile(rgb,(2,1)))
    np.savez(groups,point_group_ids=np.repeat([1,2],len(xyz)))
    monkeypatch.setattr('sketchik.object_registration.depth_checks',lambda *args:[
        {'supported_pixels':100,'support_fraction_of_testable':1.,'known_depth_pixels':100,'contradiction_fraction':0.}])
    result=register_object({'world_xyz_m':xyz,'rgb':rgb},[],
        {'cloud_file':str(cloud),'point_group_ids_file':str(groups),'source_step':24},[],lambda *args:None,budget_s=8.)
    assert result['status']=='unavailable' and result['reason']=='ambiguous_pose_or_identity'
    assert result['transform'] is None
    assert all(item['seeds']>0 for item in result['color_initializations_without_image_features'])
