from copy import deepcopy
from pathlib import Path
import json

from PIL import Image
import pytest

from sketchik.planning import image_part
from sketchik.held_pose_review import held_pose_comparison
from sketchik.stage_context import StageContext


def save_pair(root, name, step, color):
    folder=root/'observations'/name;folder.mkdir(parents=True)
    (folder/'observation.json').write_text(json.dumps({'source_step':step,
        'state':{'step':step,'position':[0,0,1],'quaternion_xyzw':[0,0,0,1]}}))
    for camera in ('agentview','robot0_eye_in_hand'):
        Image.new('RGB',(32,32),color).save(folder/(camera+'.png'))
    return folder


def claim(root,step):
    (root/'episode_state.json').write_text(json.dumps({'a_held_claim':{
        'object_id':'object','source_step':step,'relative_pose_assessment':'unknown'}}))


def test_comparison_uses_the_claim_time_and_never_the_latest_current_pose(tmp_path):
    previous=save_pair(tmp_path,'old',10,'red');save_pair(tmp_path,'new',20,'blue');claim(tmp_path,10)
    before={p:p.read_bytes() for p in previous.glob('*.png')}
    meta,parts,paths=held_pose_comparison(tmp_path,{'source_step':20})
    assert meta['reference_source_step']==10 and meta['current_source_step']==20
    assert meta['reference_observation_id']=='old' and meta['reference_image_count']==2
    assert all(Path(path).parent==previous for path in paths)
    assert sum(part['type']=='image_url' for part in parts)==2
    assert not meta['holding_or_rigid_attachment_certified']
    assert all(p.read_bytes()==raw for p,raw in before.items())


@pytest.mark.parametrize('claim_step,status',[(20,'same_physical_step'),(30,'invalid_future_reference'),
    (5,'reference_images_unavailable')])
def test_unavailable_or_same_snapshot_is_explicit_without_wrong_historical_images(tmp_path,claim_step,status):
    save_pair(tmp_path,'current',20,'blue');claim(tmp_path,claim_step)
    meta,parts,paths=held_pose_comparison(tmp_path,{'source_step':20})
    assert meta['status']==status and meta['reference_image_count']==0
    assert not parts and not paths


def test_no_previous_holding_claim_needs_no_extra_images(tmp_path):
    meta,parts,paths=held_pose_comparison(tmp_path,{'source_step':20})
    assert meta['status']=='no_prior_held_claim' and not parts and not paths


def test_first_lift_can_compare_with_actual_close_without_certifying_grasp(tmp_path):
    save_pair(tmp_path,'after_close',10,'red')
    close={'primitive':'set_gripper','parameters':{'command':'close'},'completed':True,'source_step':10}
    meta,parts,paths=held_pose_comparison(tmp_path,{'source_step':20},{'steps':[close]})
    assert meta['reference_source_step']==10 and meta['reference_image_count']==2
    assert meta['reference_basis']=='actual_close_event_not_certified_holding'
    assert 'holding was not certified' in json.dumps(parts)
    assert not meta['holding_or_rigid_attachment_certified']
    opened={'primitive':'release','source_step':15}
    meta,parts,paths=held_pose_comparison(tmp_path,{'source_step':20},{'steps':[close,opened]})
    assert meta['reference_image_count']==0 and not parts


