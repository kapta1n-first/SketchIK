from copy import deepcopy
import base64
from io import BytesIO
import json
import os

import numpy as np
from PIL import Image
import pytest

from sketchik.task_trial import ProgramObservationHub
from sketchik.planning import save_planning_history
from sketchik.timing import Timings


class Robot:
    cameras = ('agentview', 'robot0_eye_in_hand')
    size, steps, max_steps = 64, 30, 1000
    metadata = {'task_language': 'observation unit fixture'}
    def __init__(self):
        self.renders = 0
        self.q = [0.]*7
    def state(self):
        return {'step': self.steps, 'q': self.q[:], 'position': [0., 0., 1.],
            'quaternion_xyzw': [0., 0., 0., 1.], 'gripper_command': -1., 'official_success': False}
    def camera(self, name):
        self.renders += 1
        transform = np.eye(4); transform[:3, 3] = [self.q[0], 0., 1.]
        return {'rgb': np.full((self.size, self.size, 3), self.steps, np.uint8),
            'depth_m': np.ones((self.size, self.size)), 'K': np.eye(3),
            'world_from_camera': transform, 'camera': name, 'step': self.steps, 'convention': 'unit fixture'}


@pytest.fixture
def observation_hub(tmp_path):
    events = []
    hub = ProgramObservationHub(Robot(), object(), tmp_path, events.append)
    return hub, events


def test_same_state_renders_once_and_returns_independent_values(observation_hub):
    hub, events = observation_hub
    first, images = hub.observe(); original = deepcopy(first)
    first['state']['q'][0] = 99; images[0]['text'] = 'caller modification'
    second, parts = hub.observe()
    assert second == original and parts[0]['text'] != 'caller modification'
    assert hub.robot.renders == 2 and len(hub.observations) == 1 and hub.observation_reuse_count == 1
    assert events[-1]['event'] == 'observation_reused' and events[-1]['new_camera_renders'] == 0
    folder = hub.run/'observations'/original['observation_id']
    metadata = json.loads((folder/'observation.json').read_text())
    assert metadata == original and 'official_success' not in json.dumps(metadata)
    for camera in hub.robot.cameras:
        frame = hub.observations[original['observation_id']][camera]
        np.testing.assert_array_equal(np.asarray(Image.open(folder/(camera+'.png'))), frame['rgb'])
        with np.load(folder/(camera+'.npz')) as data:
            np.testing.assert_array_equal(data['depth_m'], frame['depth_m'])
            np.testing.assert_array_equal(data['world_from_camera'], frame['world_from_camera'])


def test_new_step_forces_capture_even_if_arm_did_not_move(observation_hub):
    hub, _ = observation_hub
    first, _ = hub.observe(); hub.robot.steps += 1; second, _ = hub.observe()
    assert first['observation_id'] != second['observation_id'] and hub.robot.renders == 4
    record, _ = hub.read_image(first['observation_id'], 'agentview')
    assert record['source_step'] == 30 and not record['current'] and hub.robot.renders == 4
    assert len(list((hub.run/'observations').glob('obs_*'))) == 2


def test_state_or_image_configuration_change_invalidates_reuse(observation_hub):
    hub, _ = observation_hub
    first, _ = hub.observe(); hub.robot.q[0] = .02; moved, _ = hub.observe()
    assert first['observation_id'] != moved['observation_id']
    hub.robot.size = 80; resized, _ = hub.observe()
    assert resized['cameras'][0]['width'] == 80 and hub.robot.renders == 6
    assert len(hub.observations) == 3


def test_same_observation_whole_scene_is_not_rebuilt(observation_hub, monkeypatch):
    hub, _ = observation_hub
    record, _ = hub.observe(); calls = []
    def build(frames, folder, identity):
        calls.append(identity)
        return {'observation_id': identity, 'point_count': 123}
    monkeypatch.setattr('sketchik.task_trial.save_scene', build)
    first = hub.make_scene(record['observation_id']); second = hub.make_scene(record['observation_id'])
    assert first == second and calls == [record['observation_id']]
    hub.robot.steps += 1; new, _ = hub.observe(); hub.make_scene(new['observation_id'])
    assert calls == [record['observation_id'], new['observation_id']]


def test_history_archive_does_not_rewrite_old_images_and_adds_new_images(tmp_path):
    def part(color):
        buffer = BytesIO(); Image.new('RGB', (3, 3), color).save(buffer, format='PNG')
        return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+
            base64.b64encode(buffer.getvalue()).decode()}}
    messages = [{'role': 'user', 'content': [part('red')]}]; save_planning_history(tmp_path, messages)
    saved = next((tmp_path/'planning_images').glob('*.png'))
    original = saved.read_bytes(); os.utime(saved, ns=(1000000000, 1000000000))
    messages.append({'role': 'user', 'content': [part('blue')]}); save_planning_history(tmp_path, messages)
    assert saved.stat().st_mtime_ns == 1000000000 and saved.read_bytes() == original
    assert len(list((tmp_path/'planning_images').glob('*.png'))) == 2
    assert len(json.loads((tmp_path/'planning_transcript.json').read_text())) == 2


def test_nested_timing_counts_once_and_closes_on_error():
    ticks = iter([0., 1., 3., 5.]); timings = Timings(clock=lambda: next(ticks))
    with timings.scope('outer'):
        with pytest.raises(ValueError):
            with timings.scope('inner'):
                raise ValueError('fixture')
    report = timings.report(7.)
    assert report['phases']['outer']['inclusive_wall_s'] == 5.
    assert report['phases']['outer']['exclusive_wall_s'] == 3.
    assert report['phases']['inner']['exclusive_wall_s'] == 2.
    assert report['measured_exclusive_wall_s'] == 5. and report['outside_instrumented_phases_wall_s'] == 2.
    assert timings.stack == []
