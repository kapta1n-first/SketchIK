import numpy as np
import pytest
import os
from pathlib import Path
import subprocess
import sys
from scipy.spatial.transform import Rotation

from sketchik.pose_binding import compile_binding, compile_rotation


def test_binding_import_leaves_renderer_configuration_before_mujoco():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-c','import sys; import sketchik.task_trial; assert "mujoco" not in sys.modules'],
        env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def evidence():
    return {'eef':{'position':[0.,0.,1.2],'quaternion_xyzw':[1.,0.,0.,0.]},'source_step':8,
        'image_references':[{'observation_id':'frame','camera':'agentview'}],
        'robot_gripper_geometry':{'pad_centres_in_eef_tool_m':[[.04,0.,-.0036],[-.04,0.,-.0036]]}}


def frame(*_):
    return {'rgb':np.zeros((32,32,3),np.uint8),'depth_m':np.full((32,32),.9),
        'K':np.array([[100.,0.,16.],[0.,100.,16.],[0.,0.,1.]]),'world_from_camera':np.eye(4)}


@pytest.mark.parametrize('primitive,axis',[
    ('rotate_roll',[1,0,0]),('rotate_pitch',[0,1,0]),('rotate_yaw',[0,0,1])])
def test_single_angle_binding_rotates_complete_actual_pose_about_fixed_world_axis(primitive,axis):
    packet=evidence()
    packet['eef']={'position':[.23,-.31,1.17],
                   'quaternion_xyzw':Rotation.from_euler('xyz',[.4,-.3,.7]).as_quat().tolist()}
    entry=Rotation.from_quat(packet['eef']['quaternion_xyzw'])
    value=.27
    selection={'angle_rad':value,'settle_steps':19}
    params,pose,audit=compile_rotation(selection,packet,primitive)
    expected=Rotation.from_rotvec(np.array(axis)*value)*entry
    assert params==selection
    np.testing.assert_array_equal(pose['position'],packet['eef']['position'])
    np.testing.assert_allclose(Rotation.from_quat(pose['quaternion_xyzw']).as_matrix(),expected.as_matrix(),atol=1e-12)
    assert np.linalg.norm(pose['quaternion_xyzw'])==pytest.approx(1.)
    assert audit['compiled_pose']==pose and audit['rotation_tool']==primitive
    assert audit['source_step']==8 and audit['executed'] is False
    # The same saved relative angle resolves from the later actual entry.
    packet['eef']={'position':[-.12,.21,1.03],
                   'quaternion_xyzw':Rotation.from_euler('xyz',[-.6,.25,-.4]).as_quat().tolist()}
    _,later,_=compile_rotation(selection,packet,primitive)
    expected=Rotation.from_rotvec(np.array(axis)*value)*Rotation.from_quat(packet['eef']['quaternion_xyzw'])
    np.testing.assert_array_equal(later['position'],packet['eef']['position'])
    np.testing.assert_allclose(Rotation.from_quat(later['quaternion_xyzw']).as_matrix(),expected.as_matrix(),atol=1e-12)


@pytest.mark.parametrize('primitive',['rotate_roll','rotate_pitch','rotate_yaw'])
@pytest.mark.parametrize('selection',[{}, {'angle_rad':None}, {'angle_rad':True}, {'angle_rad':'0.2'},
    {'delta_yaw':.2}, {'target_pitch':.5}, {'angle_rad':.2,'target_yaw':.5},
    {'angle_rad':.2,'axis':[1,0,0]}, {'angle_rad':.2,'frame':'tool'}])
def test_single_angle_binding_rejects_invalid_or_missing_angle(primitive,selection):
    from jsonschema import ValidationError
    with pytest.raises(ValidationError):compile_rotation(selection,evidence(),primitive)


def test_contact_pixel_aligns_real_pad_after_rotation_not_eef_origin():
    packet=evidence()
    quat=Rotation.from_euler('z',.7).as_quat().tolist()
    binding={'anchor':{'kind':'image_point','observation_id':'frame','camera':'agentview','pixel_xy':[16,16]},
        'reference':'pad_0','offset_m':[0.,0.,.005],'offset_frame':'world','orientation':quat}
    pose,audit=compile_binding(binding,packet,frame)
    pad=np.array(pose['position'])+Rotation.from_quat(pose['quaternion_xyzw']).apply([.04,0.,-.0036])
    np.testing.assert_allclose(pad,[0.,0.,.905],atol=1e-12)
    assert not np.allclose(pose['position'],pad)
    assert audit['anchor_world_xyz_m']==[0.,0.,.9]
    binding['anchor']['observation_id']='unseen'
    with pytest.raises(ValueError,match='supplied native'):compile_binding(binding,packet,frame)


def test_held_bottom_places_observed_shape_on_support_with_clearance():
    x,y,z=np.meshgrid(np.linspace(.01,.06,3),np.linspace(-.02,.04,3),np.linspace(-.01,.09,3))
    local=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    binding={'anchor':{'kind':'image_point','observation_id':'frame','camera':'agentview','pixel_xy':[16,16]},
        'reference':'held_bottom','offset_m':[0.,0.,.005],'offset_frame':'world','orientation':'preserve'}
    pose,_=compile_binding(binding,evidence(),frame,local)
    transformed=Rotation.from_quat(pose['quaternion_xyzw']).apply(local)+pose['position']
    assert transformed[:,2].min()==pytest.approx(.905)
    np.testing.assert_allclose((transformed[:,:2].min(axis=0)+transformed[:,:2].max(axis=0))/2,[0,0],atol=1e-12)
    with pytest.raises(ValueError,match='unavailable'):compile_binding(binding,evidence(),frame)


def test_positive_opening_offset_remains_outside_observed_plane():
    packet=evidence()
    packet['observed_opening_frames']=[{'anchor_world_m':[.2,.3,1.],
        'proposed_surface_normal_world':[0.,-1.,0.]}]
    binding={'anchor':{'kind':'opening','opening_index':0},'reference':'eef',
        'offset_m':[0,0,.1],'offset_frame':'opening','orientation':'preserve'}
    pose,_=compile_binding(binding,packet,frame)
    np.testing.assert_allclose(pose['position'],[.2,.2,1.])


def test_opening_anchor_must_belong_to_operation_reference_target():
    packet=evidence()
    packet['operation_reference_target_id']='microwave'
    packet['observed_opening_frames']=[
        {'target_id':'cabinet','anchor_world_m':[.1,.2,1.],
         'proposed_surface_normal_world':[0.,-1.,0.]},
        {'target_id':'microwave','anchor_world_m':[.3,.4,1.],
         'proposed_surface_normal_world':[0.,-1.,0.]}]
    binding={'anchor':{'kind':'opening','opening_index':0},'reference':'eef',
        'offset_m':[0,0,.1],'offset_frame':'opening','orientation':'preserve'}
    with pytest.raises(ValueError,match='target_id.*operation_reference_target_id'):
        compile_binding(binding,packet,frame)
    binding['anchor']['opening_index']=1
    pose,_=compile_binding(binding,packet,frame)
    np.testing.assert_allclose(pose['position'],[.3,.3,1.])
    del packet['observed_opening_frames'][1]['target_id']
    with pytest.raises(ValueError,match='target_id.*operation_reference_target_id'):
        compile_binding(binding,packet,frame)


def test_opening_offset_frame_checks_owner_even_with_another_anchor():
    packet=evidence()
    packet['operation_reference_target_id']='microwave'
    packet['observed_opening_frames']=[{'target_id':'cabinet','anchor_world_m':[.2,.3,1.],
        'proposed_surface_normal_world':[0.,-1.,0.]}]
    binding={'anchor':{'kind':'current_eef'},'reference':'eef',
        'offset_m':[0,0,.1],'offset_frame':'opening','orientation':'preserve'}
    with pytest.raises(ValueError,match='target_id.*operation_reference_target_id'):
        compile_binding(binding,packet,frame)
    packet['observed_opening_frames'][0]['target_id']='microwave'
    pose,_=compile_binding(binding,packet,frame)
    np.testing.assert_allclose(pose['position'],[0.,-.1,1.2])


def test_world_direction_pair_rotates_full_tool_and_rebinds_pad_offset():
    packet=evidence()
    binding={'anchor':{'kind':'image_point','observation_id':'frame','camera':'agentview','pixel_xy':[16,16]},
        'reference':'pad_0','offset_m':[0,0,.01],'offset_frame':'world',
        'orientation':{'finger_direction':[0,2,0],'closing_axis':[0,0,3]}}
    pose,audit=compile_binding(binding,packet,frame)
    rotation=Rotation.from_quat(pose['quaternion_xyzw'])
    np.testing.assert_allclose(rotation.apply([0,0,1]),[0,1,0],atol=1e-12)
    np.testing.assert_allclose(rotation.apply([1,0,0]),[0,0,1],atol=1e-12)
    # Changing orientation must also move the EEF so the chosen pad stays on
    # the requested world target, rather than copying the old EEF position.
    np.testing.assert_allclose(np.array(pose['position'])+rotation.apply([.04,0,-.0036]),[0,0,.91],atol=1e-12)
    binding['orientation']['closing_axis']=[0,1,0]
    with pytest.raises(ValueError,match='parallel'):compile_binding(binding,packet,frame)
    binding['orientation']['finger_direction']=[0,0,0]
    with pytest.raises(ValueError,match='nonzero'):compile_binding(binding,packet,frame)


def test_angles_use_declared_frame_and_keep_selected_contact_in_world():
    from jsonschema import Draft202012Validator, ValidationError
    from sketchik.pose_binding import BINDING
    binding={'anchor':{'kind':'image_point','observation_id':'frame','camera':'agentview','pixel_xy':[16,16]},
        'reference':'pad_0','offset_m':[0,0,0],'offset_frame':'world',
        'orientation':{'rpy_world_rad':[0,0,np.pi/2]}}
    pose,_=compile_binding(binding,evidence(),frame)
    world_rotation=Rotation.from_quat(pose['quaternion_xyzw'])
    np.testing.assert_allclose(world_rotation.apply([1,0,0]),[0,1,0],atol=1e-12)
    np.testing.assert_allclose(np.array(pose['position'])+world_rotation.apply([.04,0,-.0036]),[0,0,.9],atol=1e-12)
    binding['orientation']={'rpy_delta_tool_rad':[0,0,np.pi/2]}
    pose,_=compile_binding(binding,evidence(),frame)
    tool_rotation=Rotation.from_quat(pose['quaternion_xyzw'])
    # The starting tool is flipped about X: local +Z points along world -Z,
    # so a positive local yaw turns X toward world -Y, not world +Y.
    np.testing.assert_allclose(tool_rotation.apply([1,0,0]),[0,-1,0],atol=1e-12)
    np.testing.assert_allclose(np.array(pose['position'])+tool_rotation.apply([.04,0,-.0036]),[0,0,.9],atol=1e-12)
    binding['orientation']=[0,0,0,1]
    with pytest.raises(ValidationError):Draft202012Validator(BINDING).validate(binding)


def test_cloud_selected_pixel_binds_exact_xyz_and_preserves_original_source(tmp_path):
    from sketchik.cloud_selection import write_views, public_view
    x,y=np.meshgrid(np.linspace(-.1,.1,10),np.linspace(-.1,.1,10))
    points=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)]).astype(np.float32)
    uv=np.column_stack([np.arange(len(points)),np.arange(len(points))+1]).astype(np.uint16)
    cloud=tmp_path/'cloud.npz'
    np.savez(cloud,world_xyz_m=points,rgb=np.full((len(points),3),80,np.uint8),
        pixel_xy=uv,source_index=np.zeros(len(points),np.uint16))
    source={'observation_id':'scan_3','camera':'agentview','source_step':3}
    views=write_views(cloud,[source],tmp_path,'target',(.2,-1.,.5),role='recorded')
    packet=evidence();packet['cloud_views']=[public_view(v) for v in views]
    selected=[]
    for view in views:
        with np.load(view['index_map_file']) as data:mapping=data['point_index']
        # Find the same source point in every projection, even when its image
        # coordinates differ. The ID is shared, never independently regenerated.
        selected.append(set(mapping[mapping>=0].tolist()))
    point_id=min(set.intersection(*selected))
    for view in views:
        with np.load(view['index_map_file']) as data:yy,xx=np.nonzero(data['point_index']==point_id)
        binding={'anchor':{'kind':'cloud_point','view_id':view['view_id'],'pixel_xy':[int(xx[0]),int(yy[0])]},
            'reference':'eef','offset_m':[0,0,.02],'offset_frame':'world','orientation':'preserve'}
        pose,audit=compile_binding(binding,packet,frame,cloud_views=views)
        np.testing.assert_allclose(pose['position'],points[point_id]+[0,0,.02])
        assert audit['cloud_point_index']==point_id
        assert audit['source_observation']==source
        assert audit['source_pixel_xy']==uv[point_id].tolist()
    binding['anchor']['pixel_xy']=[0,0]
    with pytest.raises(ValueError,match='no snapping'):compile_binding(binding,packet,frame,cloud_views=views)
    binding['anchor']['view_id']='unprovided_scatter_plot'
    with pytest.raises(ValueError,match='not supplied'):compile_binding(binding,packet,frame,cloud_views=views)
    binding['anchor']['view_id']=views[0]['view_id']
    packet['cloud_views'][0]['selectable']=False
    with pytest.raises(ValueError,match='stale'):compile_binding(binding,packet,frame,cloud_views=views)
