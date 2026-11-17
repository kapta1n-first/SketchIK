import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from sketchik.episode_state import EpisodeState
from sketchik.geometry_journal import GeometryJournal
from sketchik.task_execution import BindingUnavailable, NumericalCompiler


def packet(step, supported, gap):
    return {'source_step':step,
        'robot_gripper_geometry':{'current_inner_jaw_gap_m':gap},
        'carried_observed_geometry':{'object_id':'held','reference_available':supported,
            'reason':'Current depth contradicts the rigid hypothesis' if supported is False else 'visible support'}}


@pytest.mark.parametrize('status', ['contradicted', 'unknown', 'contact_geometry_supported'])
def test_initial_close_check_persists_local_evidence_and_only_suspends_on_contradiction(
        tmp_path, monkeypatch, status):
    frames = {'before': {'front': {'marker': 'before'}}, 'after': {'front': {'marker': 'after'}}}
    events = []
    hub = SimpleNamespace(run=tmp_path, executor=SimpleNamespace(ik=SimpleNamespace(model=object())),
        observations=frames, emit=events.append,
        carried_surface_reference={'object_id': 'held', 'validity': 'active'})
    compiler = NumericalCompiler(hub, {})
    hub.episode_state.record_gripper_event(11, 'close', 'held')
    before = {'state': {'step': 10}, 'source_step': 10, 'observation_id': 'before'}
    after = {'state': {'step': 11}, 'source_step': 11, 'observation_id': 'after'}
    def inspect(ik, before_state, before_frames, after_state, after_frames):
        assert before_state == {'step': 10} and after_state == {'step': 11}
        assert before_frames == [{'marker': 'before'}] and after_frames == [{'marker': 'after'}]
        return {'status': status, 'reason': 'native local evidence', 'source_step': 11,
            'holding_certified': False, 'target_identity_certified': False}
    monkeypatch.setattr('sketchik.grasp_contact.inspect_initial_contact', inspect)
    result = compiler.check_initial_grasp_contact('held', before, after, tmp_path)
    assert result['grasp_epoch'] == 1 and result['active'] is True
    assert result['object_id'] == 'held' and not result['holding_certified']
    assert EpisodeState(tmp_path).data['initial_grasp_contact'] == result
    assert json.loads((tmp_path/'initial_grasp_contact.json').read_text()) == result
    assert events[-1]['event'] == 'initial_grasp_contact_checked'
    assert (hub.carried_surface_reference['validity'] == 'suspended') == (status == 'contradicted')


def test_active_stage_persists_the_structured_object_relations(tmp_path):
    state=EpisodeState(tmp_path)
    plan={'source_step':1,'subtask':'carry','object_goals':[{'target_id':'mug','relation':'held'}]}
    state.begin_stage(plan)
    plan['object_goals'].clear()
    assert EpisodeState(tmp_path).data['active_stage']['object_goals']==[{'target_id':'mug','relation':'held'}]


def test_failed_route_survives_successful_withdrawal_and_reload_but_not_changed_contact(tmp_path):
    state=EpisodeState(tmp_path)
    failure={'source_step':10,'commands_completed':False,
        'motions':[{'converged':False,'goal':{'position':[1,2,3]}}]}
    state.remember_route_failure(failure)
    state.begin_stage({'source_step':20,'subtask':'successful withdrawal'})
    state.remember_route_failure({'source_step':20,'commands_completed':True,
        'motions':[{'converged':True}]})
    reloaded=EpisodeState(tmp_path)
    assert reloaded.data['last_failed_stage_outcome']['source_step']==10
    # Rechecking an unchanged empty hand does not erase a pre-grasp IK failure.
    reloaded.confirm_held_claim(None,20,'none')
    assert 'last_failed_stage_outcome' in reloaded.data
    reloaded.retire_route_failure(30,'actual gripper event')
    assert 'last_failed_stage_outcome' not in EpisodeState(tmp_path).data
    reloaded.remember_route_failure(failure)
    reloaded.confirm_held_claim('held',40,'changed')
    assert 'last_failed_stage_outcome' not in reloaded.data


def test_contact_baseline_survives_unknown_visibility_and_stage_boundaries(tmp_path):
    state=EpisodeState(tmp_path)
    assert state.observe_contact(packet(10,True,.018),'held',1.) is None
    assert state.observe_contact(packet(20,None,.018),'held',1.) is None
    # A stage boundary must not replace measured support with a model claim.
    state.confirm_held_claim('held',20)
    state.observe_contact(packet(21,True,.018),'held',1.)
    resumed=EpisodeState(tmp_path)
    warning=resumed.observe_contact(packet(30,False,.001),'held',1.)
    assert warning['previous_jaw_gap_m']==.018
    assert resumed.data['carry_premise']=='invalidated_by_contact_change'
    assert not resumed.data['holding_or_task_success_certified']
    assert resumed.data['a_held_claim']['source_step']==20
    assert json.loads((tmp_path/'episode_state.json').read_text())['source_step']==30


def test_missing_geometry_alone_is_not_empty_and_new_grasp_retires_old_baseline(tmp_path):
    state=EpisodeState(tmp_path)
    state.observe_contact(packet(10,True,.018),'held',1.)
    unknown=packet(20,False,.018)
    unknown['carried_observed_geometry']['geometry_status']='not_supplied'
    assert state.observe_contact(unknown,'held',1.) is None
    assert state.observe_contact(packet(21,False,.018),'held',1.) is None
    state.reset_contact(30,'New actual close event')
    # A thin contact is not compared with a previous grasp's larger gap.
    assert state.observe_contact(packet(31,False,.001),'held',1.) is None
    state.observe_contact(packet(32,True,.001),'held',1.)
    state.observe_contact(packet(33,False,.001),'held',-1.)
    assert state.data['contact_baseline'] is None


@pytest.mark.parametrize('supported,status',[(True,'measured'),(False,'measured'),
    (None,'not_supplied')])
def test_jaw_collapse_is_independent_of_cloud_support(tmp_path,supported,status):
    state=EpisodeState(tmp_path)
    state.observe_contact(packet(1898,True,.028363),'held',1.)
    for step,gap in [(2027,.028362),(2172,.028362),(2300,.0042)]:
        assert state.observe_contact(packet(step,True,gap),'held',1.) is None
    current=packet(2378,supported,0.)
    current['carried_observed_geometry'].update(geometry_status=status,
        visible_support={'fraction':.2876688753,'min_support_fraction':.25})
    warning=state.observe_contact(current,'held',1.)
    assert warning['baseline_source_step']==1898
    assert warning['previous_jaw_gap_m']==.028363
    assert warning['current_carried_report']['visible_support']['fraction']>.25
    assert state.data['contact_baseline']['source_step']==1898
    assert state.data['current_contact']['jaw_gap_m']==0.
    assert state.data['holding_or_task_success_certified'] is False


def test_same_contact_baseline_is_not_overwritten_by_gradual_narrowing(tmp_path):
    state=EpisodeState(tmp_path)
    state.observe_contact(packet(10,True,.020),'held',1.)
    original=deepcopy(state.data['contact_baseline'])
    for step,gap in [(20,.015),(30,.010),(40,.005)]:
        assert state.observe_contact(packet(step,True,gap),'held',1.) is None
        assert state.data['contact_baseline']==original
    warning=EpisodeState(tmp_path).observe_contact(packet(50,True,.002),'held',1.)
    assert warning['previous_jaw_gap_m']==.020


def test_contact_warning_survives_refresh_until_contact_epoch_is_retired(tmp_path):
    state=EpisodeState(tmp_path)
    state.observe_contact(packet(10,True,.018),'held',1.)
    warning=state.observe_contact(packet(20,True,.001),'held',1.)
    assert warning is not None
    assert state.observe_contact(packet(20,True,.001),'held',1.)==warning
    state.confirm_held_claim('held',20,'unchanged')
    # A repeated unchanged claim and a higher later similarity cannot silently
    # turn the recorded mechanical change back into a valid old contact.
    assert state.observe_contact(packet(21,True,.018),'held',1.)==warning
    state.confirm_held_claim('held',21,'changed')
    assert 'contact_warning' not in state.data
    assert state.observe_contact(packet(21,True,.018),'held',1.) is None


def test_new_grasp_does_not_reuse_frozen_packets_from_previous_contact(tmp_path):
    state=EpisodeState(tmp_path)
    old=packet(10,True,.018)
    state.observe_contact(old,'held',1.)
    state.reset_contact(30,'Actual new close event')
    assert state.observe_contact(packet(31,True,.001),'held',1.,[old]) is None
    assert state.data['contact_baseline']['source_step']==31
    assert state.data['contact_baseline']['robot_gripper_geometry']['current_inner_jaw_gap_m']==.001


def test_new_contact_hypothesis_can_be_monitored_before_a_stage_result(tmp_path):
    state=EpisodeState(tmp_path)
    # Supports grasp->carry within one stage, without certifying an A held ID.
    state.observe_contact(packet(10,True,.018),None,1.)
    warning=state.observe_contact(packet(20,False,.001),None,1.)
    assert warning['object_id']=='held'
    assert state.data['a_held_claim'] is None


@pytest.mark.parametrize('supported',[False,True])
def test_motion_failure_checks_current_contact_without_any_model_client(tmp_path,supported):
    state={'step':30,'position':[0,0,1.],'quaternion_xyzw':[0,0,0,1],
        'q':[0]*7,'gripper_command':1.}
    events=[]
    hub=SimpleNamespace(run=tmp_path,project_root=Path(__file__).resolve().parents[1],
        journal=GeometryJournal(tmp_path),carried_target_id='held',emit=events.append,
        executor=SimpleNamespace(ik=SimpleNamespace()),
        robot=SimpleNamespace(steps=30,state=lambda:state))
    grounder=NumericalCompiler(hub,{'task':'carry the held object','objects':[{'object_id':'held'}]})
    hub.episode_state.observe_contact(packet(10,True,.018),'held',1.)
    current=packet(30,supported,.001)
    def refresh(target,*args):
        target.update(deepcopy(current))
    grounder._refresh_runtime=refresh
    step={'primitive':'move','target_id':'held','reference_target_id':'held',
        'camera':'agentview','intent':'continue the carry'}
    stage={'plan':{'source_step':10,'subtask':'carry held object','steps':[step],'expected_effects':[]},'selections':[{}], 'frozen_packets':[packet(10,True,.018)]}
    description={'source_step':30,'observation_id':'actual','state':state}
    with pytest.raises(BindingUnavailable) as caught:
        grounder.check_actual_contact(stage,0,description,tmp_path/'contact')
    assert caught.value.failure['failure_stage']=='carried_contact'
    assert not hasattr(grounder,'client') and hub.robot.steps==30
    assert events[-1]['event']=='carried_contact_change'


def test_repeated_same_snapshot_slip_review_preserves_new_measurement_and_failure(tmp_path):
    state=EpisodeState(tmp_path)
    state.observe_contact(packet(10,True,.02),'held',1.)
    state.confirm_held_claim('held',20,'changed')
    assert state.data['contact_baseline'] is None
    state.observe_contact(packet(20,True,.018),'held',1.)
    state.remember_route_failure({'source_step':20,'commands_completed':False,
        'motions':[{'converged':False,'actual_stop':{'step':20},'goal':{'position':[0,0,1]}}]})
    fresh=deepcopy(state.data['contact_baseline'])
    reloaded=EpisodeState(tmp_path)
    reloaded.confirm_held_claim('held',20,'changed')
    assert reloaded.data['contact_baseline']==fresh
    assert reloaded.data['carry_premise']=='supported_hypothesis'
    assert 'last_failed_stage_outcome' in reloaded.data
    reloaded.confirm_held_claim('held',21,'changed')
    assert reloaded.data['contact_baseline'] is None
    assert 'last_failed_stage_outcome' not in reloaded.data
