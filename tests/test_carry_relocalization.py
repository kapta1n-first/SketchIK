from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.carry_reference import initialize_reference, suspend, relocalize_reference


def fixture():
    # Synthetic asymmetric samples; no simulator, planner GT or robot action.
    rng = np.random.default_rng(48)
    world = rng.normal(size=(240, 3))*[.04, .025, .012] + [.15, -.13, .94]
    original_r = Rotation.from_euler('xyz', [.1, -.2, .6]).as_matrix()
    local = (world-[.16, -.12, 1.02])@original_r
    ref = initialize_reference({'object_id':'object', 'source_step':4, 'source_cloud_file':'frozen_observed.npz'},
        local, np.full((240, 3), 127, np.uint8), 3, np.arange(240)*2, world_xyz_m=world)
    suspend(ref, 11, 'A observed slip', kind='relative_pose_changed')
    r = Rotation.from_euler('xyz', [.15, .05, -.3]).as_matrix()
    t = np.array([.02, -.03, .015])
    state = {'step':12, 'position':[.17, -.10, 1.01],
        'quaternion_xyzw':Rotation.from_euler('xyz', [-.25, .4, .1]).as_quat().tolist(),
        'gripper_command':1.}
    registration = {'status':'accepted', 'source_step':12, 'search_complete':True,
        'transform':{'rotation':r.tolist(), 'translation_m':t.tolist()}}
    kwargs = dict(object_id='object', grasp_epoch=3, explicit_pose_rebind=True,
        point_set_scope='one_rigid_object', registration=registration,
        frozen_world_xyz_m=world, current_state=state)
    return ref, world, kwargs


def test_changed_relation_can_relocalize_the_same_frozen_samples_without_a_new_close():
    ref, world, kwargs = fixture()
    identity, ids, rgb, original = ref['reference_id'], ref['point_ids'].copy(), ref['rgb'].copy(), ref['local_xyz_m'].copy()
    before = deepcopy(kwargs)
    report = relocalize_reference(ref, **kwargs)
    r = np.array(kwargs['registration']['transform']['rotation'])
    t = np.array(kwargs['registration']['transform']['translation_m'])
    eef_r = Rotation.from_quat(kwargs['current_state']['quaternion_xyzw']).as_matrix()
    expected = (world@r.T+t-kwargs['current_state']['position'])@eef_r
    np.testing.assert_allclose(ref['local_xyz_m'], expected, atol=1e-12)
    np.testing.assert_array_equal(ref['original_local_xyz_m'], original)
    np.testing.assert_array_equal(ref['point_ids'], ids)
    np.testing.assert_array_equal(ref['rgb'], rgb)
    assert ref['reference_id']==identity and ref['grasp_epoch']==3
    assert ref['relation_revision']==1 and ref['validity']=='active'
    assert report['status']=='relocalized' and report['holding_certified'] is False
    assert report['robot_actions']==report['agent_requests']==0
    np.testing.assert_array_equal(kwargs['frozen_world_xyz_m'], before['frozen_world_xyz_m'])


@pytest.mark.parametrize('kind', ['mixed', 'not_explicit', 'object', 'epoch', 'stale', 'unavailable', 'incomplete', 'open', 'different_shape'])
def test_unverified_rebinding_never_replaces_or_unsuspends_the_original_reference(kind):
    ref, world, kwargs = fixture()
    if kind=='mixed':kwargs['point_set_scope']='mixed_or_unknown'
    elif kind=='not_explicit':kwargs['explicit_pose_rebind']=False
    elif kind=='object':kwargs['object_id']='another'
    elif kind=='epoch':kwargs['grasp_epoch']=4
    elif kind=='stale':kwargs['registration']['source_step']=11
    elif kind=='unavailable':kwargs['registration'].update(status='unavailable', transform=None)
    elif kind=='incomplete':kwargs['registration']['search_complete']=False
    elif kind=='open':kwargs['current_state']['gripper_command']=-1.
    elif kind=='different_shape':kwargs['frozen_world_xyz_m']=world*np.array([1.,1.,1.1])
    before = deepcopy(ref)
    report = relocalize_reference(ref, **kwargs)
    assert report['status']=='unavailable'
    assert ref.keys()==before.keys()
    for key, value in before.items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(ref[key],value)
        else:assert ref[key]==value


def test_second_relocalization_is_another_relation_of_the_original_point_set():
    ref, world, kwargs = fixture()
    original = ref['local_xyz_m'].copy()
    relocalize_reference(ref, **kwargs)
    suspend(ref,13,'A observed another slip',kind='relative_pose_changed')
    kwargs['registration']['source_step']=14;kwargs['current_state']['step']=14
    kwargs['registration']['transform']['translation_m']=[.01, .02, -.01]
    report = relocalize_reference(ref, **kwargs)
    assert report['status']=='relocalized' and ref['relation_revision']==2
    np.testing.assert_array_equal(ref['original_local_xyz_m'],original)
    np.testing.assert_array_equal(ref['point_ids'],np.arange(240)*2)
    assert len(ref['relation_history'])==2


def test_same_shape_from_another_world_capture_cannot_replace_the_original_capture():
    ref, world, kwargs = fixture()
    kwargs['frozen_world_xyz_m'] = world + [.1, 0., 0.]
    before = deepcopy(ref)
    report = relocalize_reference(ref, **kwargs)
    assert report['status']=='unavailable'
    np.testing.assert_array_equal(ref['local_xyz_m'],before['local_xyz_m'])
    assert ref['relation_revision']==0
