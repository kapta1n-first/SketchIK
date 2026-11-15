from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from sketchik.perception import verify_replay_observation










def observation(tmp_path):
    folder = tmp_path / 'observations/obs_00004'
    folder.mkdir(parents=True)
    state = {'step': 30, 'q': [0.]*7, 'position': [0., 0., 1.],
             'quaternion_xyzw': [0., 0., 0., 1.], 'finger_joint_positions': [.04, -.04]}
    description = {'source_step': 30, 'state': state, 'cameras': [
        {'camera': 'agentview'}, {'camera': 'robot0_eye_in_hand'}]}
    (folder / 'observation.json').write_text(json.dumps(description))
    frames = {}
    for camera in ['agentview', 'robot0_eye_in_hand']:
        rgb = np.full((4, 4, 3), 128, np.uint8)
        Image.fromarray(rgb).save(folder / (camera+'.png'))
        arrays = {'depth_m': np.full((4, 4), .3), 'K': np.eye(3), 'world_from_camera': np.eye(4)}
        np.savez(folder / (camera+'.npz'), **arrays)
        frames[camera] = {'rgb': rgb, **arrays}
    return state, frames


def test_replay_requires_robot_state_and_exact_native_rgbd(tmp_path):
    state, frames = observation(tmp_path)
    class Robot:
        cameras = ('agentview', 'robot0_eye_in_hand')
        def state(self): return state
        def camera(self, name): return frames[name]
    robot = Robot()
    assert verify_replay_observation(robot, tmp_path, 'obs_00004')['native_rgbd_and_calibration_exact']
    frames['agentview']['depth_m'][0, 0] += 1e-8
    with pytest.raises(ValueError, match='native RGB-D'):
        verify_replay_observation(robot, tmp_path, 'obs_00004')
    frames['agentview']['depth_m'][0, 0] = .3
    frames['agentview']['rgb'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='native RGB-D'):
        verify_replay_observation(robot, tmp_path, 'obs_00004')
    checked = verify_replay_observation(robot, tmp_path, 'obs_00004', rgb_roundoff_pixels=1)
    assert checked['native_geometry_and_calibration_exact']
    assert not checked['native_rgbd_and_calibration_exact']
    frames['agentview']['depth_m'][0, 0] += 1e-8
    with pytest.raises(ValueError, match='native RGB-D'):
        verify_replay_observation(robot, tmp_path, 'obs_00004', rgb_roundoff_pixels=1)
    frames['agentview']['depth_m'][0, 0] = .3
    frames['agentview']['rgb'][0, 0, 1] += 2
    with pytest.raises(ValueError, match='native RGB-D'):
        verify_replay_observation(robot, tmp_path, 'obs_00004', rgb_roundoff_pixels=1)
    frames['agentview']['rgb'][0, 0] = 128
    state['position'][2] += .001
    with pytest.raises(ValueError, match='robot state mismatch'):
        verify_replay_observation(robot, tmp_path, 'obs_00004')




def test_transient_rgb_draw_can_be_rechecked_without_changing_physics(tmp_path):
    state, frames = observation(tmp_path)
    calls = {'agentview': 0, 'robot0_eye_in_hand': 0}
    class Robot:
        cameras = ('agentview', 'robot0_eye_in_hand')
        def state(self): return state
        def camera(self, name):
            calls[name] += 1
            result = deepcopy(frames[name])
            if name == 'agentview' and calls[name] == 1: result['rgb'][0, 0, 0] += 1
            return result
    result = verify_replay_observation(Robot(), tmp_path, 'obs_00004', diagnostic_dir=tmp_path/'diagnostic')
    assert state['step'] == 30
    assert result['native_rgbd_and_calibration_exact']
    assert result['rgb_differences']['agentview']['same_physical_step_rerender']
    assert (tmp_path/'diagnostic/obs_00004_agentview_first.png').exists()


@pytest.mark.parametrize('settles',[True,False])
def test_redraws_are_bounded_and_save_every_comparison(tmp_path,settles):
    state,frames=observation(tmp_path)
    calls={'agentview':0,'robot0_eye_in_hand':0}
    class Robot:
        cameras=('agentview','robot0_eye_in_hand')
        def state(self):return state
        def camera(self,name):
            calls[name]+=1
            result=deepcopy(frames[name])
            if name=='agentview' and (not settles or calls[name]<3):
                result['rgb'][0,0,0]+=1
            return result
    if settles:
        result=verify_replay_observation(Robot(),tmp_path,'obs_00004',diagnostic_dir=tmp_path/'draws')
        assert result['native_rgbd_and_calibration_exact']
        assert len(result['rgb_differences']['agentview']['same_physical_step_redraws'])==2
        assert calls['agentview']==3
    else:
        with pytest.raises(ValueError,match='/rgb'):
            verify_replay_observation(Robot(),tmp_path,'obs_00004',diagnostic_dir=tmp_path/'draws')
        assert calls['agentview']==4
    audit=json.loads((tmp_path/'draws/obs_00004_agentview_redraws.json').read_text())
    assert len(audit['redraws'])==(2 if settles else 3)
    assert audit['redraws'][-1]['accepted']==settles
    assert state['step']==30


def test_redraw_cannot_hide_changed_depth(tmp_path):
    state,frames=observation(tmp_path)
    calls=0
    class Robot:
        cameras=('agentview','robot0_eye_in_hand')
        def state(self):return state
        def camera(self,name):
            nonlocal calls
            calls+=1
            result=deepcopy(frames[name])
            if calls==1:result['rgb'][0,0,0]+=1
            else:result['depth_m'][0,0]+=.001
            return result
    with pytest.raises(ValueError,match='repeated render geometry mismatch'):
        verify_replay_observation(Robot(),tmp_path,'obs_00004')
    assert calls==2
