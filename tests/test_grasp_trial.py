from copy import deepcopy
import json
import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError
from scipy.spatial.transform import Rotation
from sketchik.grasp_trial import SKETCH_SCHEMA, parse_choice, bind_plan
from sketchik.perception_service import filter_observed_outliers


def sketch():
    return {'source_step':0,'subtask':'grasp','target_description':'visible object','grasp_preference':'secure grip',
        'steps':[{'primitive':'move','params':{'pose_ref':'selected_grasp.pregrasp'}},
                 {'primitive':'move','params':{'pose_ref':'selected_grasp.grasp'}},
                 {'primitive':'set_gripper','params':{'command':'close','steps':20}}]}


def test_schema_does_not_let_agent_guess_grounded_coordinates():
    good=sketch();Draft202012Validator(SKETCH_SCHEMA).validate(good)
    bad=deepcopy(good);bad['steps'][0]['params']={'position':[0,0,1]}
    with pytest.raises(ValidationError):Draft202012Validator(SKETCH_SCHEMA).validate(bad)


def test_local_selection_can_reject_every_pose():
    assert parse_choice('{"candidate_id":null,"reason":"target outside fingers"}',[3,5])['candidate_id'] is None
    assert parse_choice('{"candidate_id":5,"reason":"encloses body"}',[3,5])['candidate_id']==5
    for identity in [True,2,'5',5.0]:
        with pytest.raises(ValueError):parse_choice(json.dumps({'candidate_id':identity,'reason':'x'}),[3,5])
    with pytest.raises(ValueError):parse_choice('{"candidate_id":5}',[5])


def test_binding_uses_local_approach_axis_and_native_tcp_full_rotation():
    g=np.eye(4);g[:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix();g[:3,3]=[.2,.3,.4]
    tcp=np.eye(4);tcp[2,3]=.097
    source=sketch();frozen=deepcopy(source);bound=bind_plan(source,g,tcp,.1)
    np.testing.assert_allclose(bound['steps'][0]['params']['position'],[.197,.3,.4],atol=1e-10)
    np.testing.assert_allclose(bound['steps'][1]['params']['position'],[.297,.3,.4],atol=1e-10)
    np.testing.assert_allclose(Rotation.from_quat(bound['steps'][0]['params']['quaternion_xyzw']).as_matrix(),g[:3,:3],atol=1e-10)
    assert bound['steps'][2]==source['steps'][2] and source==frozen


def test_cpu_knn_matches_upstream_l1_outlier_rule_including_duplicates():
    rng=np.random.default_rng(7);pts=rng.normal(0,.001,(100,3)).astype(np.float32)
    pts=np.concatenate([pts,pts[:3],[[.1,.1,.1]]]).astype(np.float32)
    distances=np.abs(pts[:,None]-pts[None,:]).sum(axis=-1)
    np.fill_diagonal(distances,np.inf)
    expected=np.sort(distances,axis=1)[:,:20].mean(axis=1)<.014
    filtered,keep=filter_observed_outliers(pts)
    np.testing.assert_array_equal(keep,expected);np.testing.assert_array_equal(filtered,pts[expected])
    assert not keep[-1] and keep[:-1].all()


def test_mask_selection_rejects_wrong_camera_pair_or_unprovided_ids():
    from sketchik.grasp_preview import validate_mask_choice
    mapping={1:{'camera':'agentview','local_id':2},2:{'camera':'agentview','local_id':3},3:{'camera':'robot0_eye_in_hand','local_id':0}}
    assert validate_mask_choice('{"selected_ids":[1,3],"reason":"same target"}',mapping)=={'agentview':2,'robot0_eye_in_hand':0}
    assert validate_mask_choice('{"selected_ids":[],"reason":"neither visible"}',mapping)=={}
    for ids in [[1,2],[99],[True],[1,1]]:
        with pytest.raises(ValueError):validate_mask_choice(json.dumps({'selected_ids':ids,'reason':'x'}),mapping)
