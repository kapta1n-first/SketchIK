"""Reference-supported carried fragments must not become self-obstacles."""
import numpy as np
import pytest

from sketchik.observed_clearance import ObservedClearance


@pytest.fixture(scope='module')
def robot_ik():
    from robosuite.models.robots import Panda
    from robosuite.models.grippers import PandaGripper
    from sketchik.kinematics import RobotIK
    arm=Panda(); gripper=PandaGripper(); arm.add_gripper(gripper)
    return RobotIK(arm.get_xml(),arm.joints,gripper.important_sites['grip_site'])


def scene(robot_ik, *, wall=False, slipped=False):
    q=np.array([-.03,-.13,0,-2.44,0,2.2,.78])
    origin,rotation=robot_ik.fk(q)
    x,y,z=np.meshgrid(np.linspace(.02,.05,8),np.linspace(-.02,.02,8),np.linspace(.03,.07,8))
    main=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    # Three points form a separate visible strip, with only two inside the
    # existing 12 mm rigid-reference matching distance.
    fragment=np.array([[.061,.003,.04],[.061,.003,.046],[.064,.003,.052]])
    # A nearby separate obstacle is still in the hand neighbourhood, beyond
    # the measured-reference match; association must not clear that region.
    external=np.array([[.078,.004,.04],[.079,.008,.045],[.08,.003,.05]])
    parts=[main,fragment,external]
    if wall:
        parts.append(np.column_stack([np.arange(.055,.6,.005),np.full(109,.02),np.full(109,.07)]))
    checker=ObservedClearance(robot_ik)
    checker.points=np.concatenate(parts)@rotation.T+origin
    checker.build()
    reference=main+([0,.5,0] if slipped else [0,0,0])
    return checker,{'q':q,'step':8},reference,fragment@rotation.T+origin,external@rotation.T+origin


def test_reference_associates_disconnected_carried_strip_and_keeps_external(robot_ik):
    checker,state,fragment,strip,external=scene(robot_ik)
    result=checker.separate_carried_surfaces(state,fragment,7)
    assert result['reference_available']
    assert result['associated_reference_fragments']==1
    from scipy.spatial import cKDTree
    assert np.count_nonzero(cKDTree(checker.points).query(strip)[0]>.005)>=2
    assert np.all(cKDTree(checker.points).query(external)[0]<1e-9)


def test_without_valid_reference_fragments_are_not_removed(robot_ik):
    from scipy.spatial import cKDTree
    checker,state,reference,strip,_=scene(robot_ik)
    result=checker.separate_carried_surfaces(state)
    assert result['associated_reference_fragments']==0
    assert np.all(cKDTree(checker.points).query(strip)[0]<1e-9)
    checker,state,reference,strip,_=scene(robot_ik,slipped=True)
    result=checker.separate_carried_surfaces(state,reference,7)
    assert result['reference_available'] is False
    assert checker.carried_local is None
    assert np.all(cKDTree(checker.points).query(strip)[0]<1e-9)


def test_connected_appliance_kept(robot_ik):
    checker,state,reference,_,_=scene(robot_ik,wall=True)
    result=checker.separate_carried_surfaces(state,reference,7)
    assert result['reference_available']
    assert result['method']=='rigid_observed_match'
    assert len(checker.points)>90


def test_rigid_prediction_preserves_reference_and_validates_current_depth(robot_ik):
    checker,state,reference,_,external=scene(robot_ik)
    result=checker.separate_carried_surfaces(state,reference,7,preserve_reference=True,
        reference_rgb=np.zeros((len(reference),3),np.uint8))
    assert result['reference_available'] and result['method']=='rigid_reference_prediction'
    assert result['rigid_attachment_verified'] is False
    np.testing.assert_array_equal(checker.carried_local,reference)
    assert result['predicted_points']==len(reference)
    assert result['reference_visible_fraction']>=.25
    from scipy.spatial import cKDTree
    assert np.all(cKDTree(checker.points).query(external)[0]<1e-9)


def test_unsupported_scan_prediction_cannot_be_used_as_current_carried_geometry(robot_ik):
    checker,state,reference,_,_=scene(robot_ik,slipped=True)
    before=checker.points.copy()
    result=checker.separate_carried_surfaces(state,reference,7,preserve_reference=True)
    assert result['reference_available'] is False and checker.carried_local is None
    np.testing.assert_array_equal(checker.points,before)


@pytest.mark.parametrize('source_kind',['scan_cloud','observed_component'])
def test_runtime_keeps_reference_bottom_when_current_view_hides_half_the_shape(robot_ik,tmp_path,monkeypatch,source_kind):
    """Occlusion changes evidence, never the point B was asked to position."""
    from copy import deepcopy
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from sketchik.task_execution import NumericalCompiler
    checker,state,reference,_,_=scene(robot_ik)
    origin,rotation=robot_ik.fk(state['q'])
    # Only the near half is currently visible. It still supports the rigid
    # association, but its own AABB centre/bottom differ from the saved shape.
    visible=reference[reference[:,2]<.05]
    checker.points=visible@rotation.T+origin
    checker.colors=np.zeros((len(visible),3),np.uint8)
    checker.build()
    monkeypatch.setattr(checker,'add_frame',lambda *args:None)
    monkeypatch.setattr(checker,'build',lambda:{'obstacle_points':len(checker.points)})
    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance',lambda *args,**kwargs:checker)
    finger_names=[robot_ik.model.joint(i).name for i in range(robot_ik.model.njnt)
        if 'finger_joint' in (robot_ik.model.joint(i).name or '')]
    state.update(position=origin.tolist(),quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
        finger_joint_names=finger_names,finger_joint_positions=[.01,-.01],gripper_command=1.)
    retained={'object_id':'cup','source_kind':source_kind,'source_step':7,
        'local_xyz_m':reference.copy(),'rgb':np.zeros((len(reference),3),np.uint8)}
    hub=SimpleNamespace(run=tmp_path,robot=SimpleNamespace(state=lambda:deepcopy(state)),
        executor=SimpleNamespace(ik=robot_ik),carried_target_id='cup',
        carried_surface_reference=retained,observations={'current':{'camera':{}}},ik_only=True)
    grounder=NumericalCompiler.__new__(NumericalCompiler);grounder.hub=hub
    packet={};description={'observation_id':'current','source_step':8,'state':state}
    actual=grounder._refresh_runtime(packet,{'primitive':'move'},description,tmp_path)
    np.testing.assert_array_equal(actual.carried_local,reference)
    assert hub.carried_surface_reference is retained
    report=packet['carried_observed_geometry']
    assert report['predicted_points']==len(reference)>report['observed_points']
    assert report['reference_step']==7 and report['reference_source_kind']==source_kind
    assert not report['rigid_attachment_verified'] and not report['held_surface_collision_checked']
    import json
    measured=json.loads((tmp_path/'dimensions.json').read_text())['objects']['cup']
    assert measured['dimension_source_step']==7 and measured['dimension_checked_at_step']==8
    assert 'Retained observed surfaces' in measured['dimension_scope']


@pytest.mark.parametrize('point_set_scope', [None, 'one_rigid_object', 'mixed_or_unknown'])
def test_scan_binding_uses_actual_selected_contact_component_and_actual_closing_pose(tmp_path, point_set_scope):
    import json
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from sketchik.task_execution import NumericalCompiler
    cloud=tmp_path/'scan.npz'
    points=np.column_stack([np.linspace(0,.03,40),np.linspace(.02,.04,40),np.linspace(.9,.98,40)])
    background=points+[1,0,0]
    np.savez(cloud,world_xyz_m=np.r_[points,background],rgb=np.zeros((80,3),np.uint8))
    np.savez(tmp_path/'extent_groups.npz',point_group_ids=np.r_[np.ones(40,int),np.full(40,2)],cloud_file=str(cloud))
    state={'step':12,'position':[.01,.02,1.],'quaternion_xyzw':Rotation.from_euler('xyz',[.2,.1,.4]).as_quat().tolist()}
    events=[]
    grounder=NumericalCompiler.__new__(NumericalCompiler)
    grounder.hub=SimpleNamespace(robot=SimpleNamespace(state=lambda:state),emit=events.append,
        last_contact_binding={'target_id':'cup','audit':{'anchor_world_xyz_m':points[5].tolist()},
            'views':[{'cloud_file':str(cloud)}]})
    if point_set_scope is not None:
        grounder.hub.last_contact_binding['geometry_binding']={
            'selection_scope':'visible_part','point_set_scope':point_set_scope,
            'carry_reference_eligible':point_set_scope=='one_rigid_object'}
    grounder.seed_scan_carry('cup',tmp_path)
    if point_set_scope=='mixed_or_unknown':
        assert not hasattr(grounder.hub,'carried_surface_reference')
        assert 'localization-only or mixed' in events[-1]['reason']
        return
    ref=grounder.hub.carried_surface_reference
    recovered=ref['local_xyz_m']@Rotation.from_quat(state['quaternion_xyzw']).as_matrix().T+state['position']
    np.testing.assert_allclose(recovered,points,atol=1e-12)
    assert len(ref['local_xyz_m'])==40 and ref['component_group_id']==1
    assert ref['source_step']==12 and ref['rigid_attachment_verified'] is False
    assert len(ref['reference_id'])==64 and ref['relation_revision']==0
    np.testing.assert_array_equal(ref['point_ids'],np.arange(40))
    assert not hasattr(grounder.hub,'carried_target_id')
    assert json.loads((tmp_path/'scan_carry_reference.json').read_text())['status']=='predicted_rigid'


@pytest.mark.parametrize('review', [False, True])
def test_runtime_contradiction_cannot_silently_reseed_on_same_snapshot(robot_ik, tmp_path, monkeypatch, review):
    """Regression for v173 step1346: a second refresh must not change the held reference."""
    from copy import deepcopy
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from sketchik.task_execution import NumericalCompiler

    _, state, old_points, _, _ = scene(robot_ik, slipped=True)
    origin, rotation = robot_ik.fk(state['q'])
    finger_names = [robot_ik.model.joint(i).name for i in range(robot_ik.model.njnt)
        if 'finger_joint' in (robot_ik.model.joint(i).name or '')]
    state.update(position=origin.tolist(), quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
        finger_joint_names=finger_names, finger_joint_positions=[.01, -.01], gripper_command=1.)
    retained = {'object_id': 'cup', 'source_kind': 'scan_cloud', 'source_step': 7,
        'local_xyz_m': old_points.copy(), 'rgb': np.zeros((len(old_points), 3), np.uint8)}

    def current_checker(*args, **kwargs):
        checker, _, _, _, _ = scene(robot_ik)
        checker.colors = np.zeros((len(checker.points), 3), np.uint8)
        checker.add_frame = lambda *args: None
        checker.build = lambda: {'obstacle_points': len(checker.points)}
        return checker

    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance', current_checker)
    hub = SimpleNamespace(run=tmp_path, robot=SimpleNamespace(state=lambda: deepcopy(state)),
        executor=SimpleNamespace(ik=robot_ik), carried_target_id='cup',
        carried_surface_reference=retained, observations={'current': {'camera': {}}},
        ik_only=True, emit=lambda event: None)
    compiler = NumericalCompiler(hub, {})
    compiler.hub.episode_state.reset_contact(7, 'Actual close in this fixture')
    description = {'observation_id': 'current', 'source_step': 8, 'state': state}
    reports = []
    for number in range(3):
        folder = tmp_path/str(number); folder.mkdir()
        packet = {}
        checker = compiler._refresh_runtime(packet, {'primitive': 'move'}, description, folder)
        reports.append(packet['carried_observed_geometry'])
        assert not packet['carried_observed_geometry']['reference_available'], (
            'A normal repeated refresh must not replace a contradicted carry point set with its nearest component')
        assert checker.carried_local is None
        if review:
            from sketchik.carry_reference import assess
            assess(hub.carried_surface_reference,state['step'],'unchanged')
    assert hub.carried_surface_reference is not None, 'Retain the old point set as suspended evidence'
    np.testing.assert_array_equal(hub.carried_surface_reference['local_xyz_m'], old_points)
    assert all(value['object_id'] == 'cup' for value in reports)


def test_no_reference_current_near_hand_component_stays_unassigned(robot_ik,tmp_path,monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from sketchik.task_execution import NumericalCompiler
    checker,state,_,_,_=scene(robot_ik)
    origin,rotation=robot_ik.fk(state['q'])
    names=[robot_ik.model.joint(i).name for i in range(robot_ik.model.njnt)
        if 'finger_joint' in (robot_ik.model.joint(i).name or '')]
    state.update(position=origin.tolist(),quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
        finger_joint_names=names,finger_joint_positions=[.01,-.01],gripper_command=1.)
    checker.add_frame=lambda *a:None
    checker.build=lambda:{'points':len(checker.points)}
    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance',lambda *a,**kw:checker)
    hub=SimpleNamespace(run=tmp_path,robot=SimpleNamespace(state=lambda:deepcopy(state)),
        executor=SimpleNamespace(ik=robot_ik),carried_target_id='cup',carried_surface_reference=None,
        observations={'current':{'camera':{}}},ik_only=True,emit=lambda e:None)
    compiler=NumericalCompiler(hub,{})
    packet={'source_step':8}
    current=compiler._refresh_runtime(packet,{'primitive':'move'},
        {'state':state,'source_step':8,'observation_id':'current'},tmp_path)
    assert not packet['carried_observed_geometry']['reference_available']
    assert hub.carried_surface_reference is None and current.carried_local is None
    # Recovery references do not depend on held surface assignment.
    from sketchik.pose_binding import compile_binding
    selection={'anchor':{'kind':'current_eef'},'reference':'eef',
        'offset_m':[0,0,.03],'offset_frame':'world','orientation':'preserve'}
    pose,_=compile_binding(selection,packet,lambda *a:None,current.carried_local)
    np.testing.assert_allclose(pose['position'],np.asarray(state['position'])+[0,0,.03])
