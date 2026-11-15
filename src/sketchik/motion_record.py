"""Robot-command provenance and measured tracking; no motion policy or oracle."""
from collections import deque
from copy import deepcopy
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation


def _json_numbers(value):
    """Preserve numeric values while canonicalising internal arrays for records."""
    if isinstance(value, np.ndarray):
        return _json_numbers(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_numbers(item) for item in value]
    return value


def source_state(state):
    return {key: _json_numbers(state[key]) for key in ('step', 'q', 'position',
        'quaternion_xyzw', 'finger_joint_names', 'finger_joint_positions',
        'gripper_command') if key in state}


def record_digest(value):
    return hashlib.sha256(json.dumps(_json_numbers(value), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def seal_motion(value):
    value = deepcopy(_json_numbers(value))
    value['artifact_id'] = record_digest(value)
    return value


def validate_motion(value, state, executor_identity):
    payload = {key: item for key, item in value.items() if key != 'artifact_id'}
    if value.get('artifact_id') != record_digest(payload):
        raise ValueError('Compiled motion record changed; no action dispatched')
    if value['executor_identity'] != executor_identity:
        raise ValueError('Compiled motion belongs to another executor; no action dispatched')
    if value['source_state'] != source_state(state):
        raise ValueError('Compiled motion entrance state changed; no action dispatched')


class MotionTrace:
    """Bounded per-step measurements; never classifies a collision or stops motion."""
    def __init__(self, state, final_q, joint_names=()):
        self.start_step = int(state['step'])
        self.final_q = np.asarray(final_q, float).copy()
        self.joint_names = list(joint_names)
        self.recent = deque(maxlen=10)
        self.samples = 0
        self.peak_setpoint_error = 0.
        self.first_joint_tolerance_exceedance_step = None
        self.last_setpoint = None
        self.same_command_since = None
        self.same_command_steps = 0
        self.phase = None

    def sample(self, state, target, phase):
        actual = np.asarray(state['q'], float)
        target = np.asarray(target, float)
        error = float(np.max(np.abs(actual - target)))
        self.peak_setpoint_error = max(self.peak_setpoint_error, error)
        if error > .02 and self.first_joint_tolerance_exceedance_step is None:
            self.first_joint_tolerance_exceedance_step = int(state['step'])
        if self.last_setpoint is not None and np.array_equal(target, self.last_setpoint):
            self.same_command_steps += 1
        else:
            self.same_command_since = int(state['step'])
            self.same_command_steps = 1
        self.last_setpoint = target.copy()
        self.phase = phase
        self.samples += 1
        self.recent.append(source_state(state))

    def summary(self, state):
        actual = np.asarray(state['q'], float)
        errors = actual - self.final_q
        worst = int(np.argmax(np.abs(errors)))
        recent = list(self.recent)
        window = {'samples': len(recent)}
        if recent:
            rotations = Rotation.from_quat([item['quaternion_xyzw'] for item in recent])
            angles = (rotations * rotations[0].inv()).magnitude()
            window.update(first_step=recent[0]['step'], last_step=recent[-1]['step'],
                position_axis_ranges_m=np.ptp([item['position'] for item in recent], axis=0).tolist(),
                joint_ranges_rad=np.ptp([item['q'] for item in recent], axis=0).tolist(),
                maximum_rotation_from_first_rad=float(np.max(angles)))
        return {'source_step_start': self.start_step, 'source_step_end': int(state['step']),
            'samples': self.samples, 'phase_at_stop': self.phase,
            'goal_joint_positions_rad': self.final_q.tolist(),
            'actual_joint_positions_rad': actual.tolist(),
            'signed_goal_joint_errors_rad': errors.tolist(),
            'max_goal_joint_error_rad': float(abs(errors[worst])),
            'max_error_joint_index': worst,
            'max_error_joint_name': self.joint_names[worst] if len(self.joint_names) == len(actual) else None,
            'last_sent_joint_target_rad': None if self.last_setpoint is None else self.last_setpoint.tolist(),
            'final_goal_commanded': bool(self.last_setpoint is not None and np.array_equal(self.last_setpoint, self.final_q)),
            'last_command_held_since_step': self.same_command_since,
            'last_command_consecutive_steps': self.same_command_steps,
            'peak_current_setpoint_joint_error_rad': self.peak_setpoint_error,
            'first_joint_tolerance_exceedance_step': self.first_joint_tolerance_exceedance_step,
            'recent_measured_window': window,
            'scope': 'Measured robot tracking only. Tolerance exceedance during a trajectory can be ordinary lag. No collision cause, attachment, task success or new stopping policy inferred.'}


def merge_tracking(initial, continuation):
    """Aggregate a continuous command boundary without relabelling its last window."""
    if not initial or not continuation:
        return deepcopy(continuation or initial or {})
    out=deepcopy(continuation)
    out.update(source_step_start=initial['source_step_start'],
        samples=initial['samples']+continuation['samples'], aggregation='whole_motion',
        attempt_count=initial.get('attempt_count',1)+continuation.get('attempt_count',1),
        recent_window_scope='Final attempt only; bounded last measured window',
        peak_current_setpoint_joint_error_rad=max(initial['peak_current_setpoint_joint_error_rad'],
            continuation['peak_current_setpoint_joint_error_rad']))
    first=[value.get('first_joint_tolerance_exceedance_step') for value in (initial,continuation)]
    out['first_joint_tolerance_exceedance_step']=min((value for value in first if value is not None),default=None)
    if (initial.get('last_sent_joint_target_rad') is not None
            and initial['last_sent_joint_target_rad']==continuation.get('last_sent_joint_target_rad')
            and initial['source_step_end']==continuation['source_step_start']
            and continuation['last_command_consecutive_steps']==continuation['samples']):
        out['last_command_consecutive_steps']+=initial['last_command_consecutive_steps']
        out['last_command_held_since_step']=initial['last_command_held_since_step']
    return out
