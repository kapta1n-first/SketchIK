from copy import deepcopy
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError

from sketchik.local_route import route_schema,expand_stage
from sketchik.pose_binding import BINDING
from sketchik.task_execution import execute_sketch,BindingUnavailable
from sketchik.episode_state import EpisodeState


def point(delta):
    return {'anchor':{'kind':'current_eef'},'reference':'eef',
        'offset_m':[delta,0,0],'offset_frame':'world','orientation':'preserve'}


def stage():
    step=lambda primitive:{'primitive':primitive,'target_id':'object',
        'reference_target_id':'object','intent':'same selected contact method','camera':'agentview'}
    plan={'source_step':30,'subtask':'contact stage','steps':[step('wait'),step('move'),step('release')],
        'expected_effects':['reach support then release'],'final_task':False}
    return expand_stage(plan,[{'steps':1},{'waypoints':[point(.1),point(.2),point(.8)]},{}],
        [{'kind':'always'},{'kind':'always'},{'kind':'previous_motion_converged'}],
        [{},{},{}],[[],[],[]],30,'initial')




def test_route_schema_bounds_points_and_never_accepts_gripper_events():
    validator=Draft202012Validator(route_schema(BINDING,2))
    validator.validate({'waypoints':[point(.1),point(.2)]})
    for bad in ({'waypoints':[]},{'waypoints':[point(.1)]*3},
                {'waypoints':[{'primitive':'release'}]}, {'preparation':point(.1),'selection':point(.2)}):
        with pytest.raises(ValidationError):validator.validate(bad)
    with pytest.raises(ValueError):route_schema(BINDING,0)




def execute_fixture(tmp_path, *, repair_fails=False,contact_invalid=False):
    bound=stage();events=[];calls=[];position=[0.,0.,1.]
    robot=SimpleNamespace(steps=30,max_steps=1000,done=False)
    robot.state=lambda:{'position':position.copy(),'quaternion_xyzw':[0,0,0,1],'gripper_command':1.}
    def observe():return {'source_step':robot.steps,'observation_id':str(robot.steps)},[]
    def move(**params):
        calls.append(('move',params['position'][0]));robot.steps+=1
        position[:]=params['position']
        count=sum(name=='move' for name,*_ in calls)
        return {'converged':count!=2 and not (repair_fails and count==3),
            'motion_progress':{'recent_progress':False},'actions':1}
    def wait(**params):calls.append(('wait',));robot.steps+=1;return {}
    def release(**params):calls.append(('release',));robot.steps+=1;return {}
    hub=SimpleNamespace(run=tmp_path,robot=robot,observe=observe,emit=events.append,
        episode_state=EpisodeState(tmp_path),
        executor=SimpleNamespace(move=move,wait=wait,release=release),
        journal=SimpleNamespace(sync_observations=lambda step:None,mark_moved=lambda *args:None))
    class Grounder:
        requests=0
        def compile_step(self,stage,index,description,folder):
            primitive=stage['plan']['steps'][index]['primitive']
            params={'steps':1} if primitive=='wait' else {}
            if primitive=='move':params={'position':[position[0]+stage['selections'][index]['offset_m'][0],0,1.],
                'quaternion_xyzw':[0,0,0,1]}
            return {'parameters':params,'audit':{'compiled_pose':params}}
        def check_actual_contact(self,stage,index,description,folder):
            if contact_invalid and sum(name=='move' for name,*_ in calls)==2:
                error=BindingUnavailable('Actual support and jaw evidence invalidate the contact premise')
                error.failure={'failure_stage':'carried_contact','executed':False}
                raise error
        def record_execution(self,*args):pass
    return hub,Grounder(),bound,calls






def test_contact_loss_after_actual_failure_stops_without_replanning(tmp_path):
    hub,b,bound,calls=execute_fixture(tmp_path,contact_invalid=True)
    result=execute_sketch(hub,b,bound['semantic_plan'],0,frozen_binding=bound)
    assert not result['commands_completed'] and b.requests==0
    assert [item[0] for item in calls]==['wait','move','move']
    assert result['steps'][-1]['recovery_failure']['failure_stage']=='carried_contact'




def test_unbound_future_without_explicit_prefix_review_still_stops_all_actions():
    original=stage();plan=original['semantic_plan']
    bound=expand_stage(plan,[{'steps':1},{'waypoints':[point(.1)]},None],
        [{'kind':'always'}]*3,[{}]*3,[[]]*3,30,'initial')
    assert bound['needs_observation'] is True






@pytest.mark.parametrize('held_id,rigid_reference,expected',[
    ('object',None,{'object'}),(None,{'object_id':'object'},{'object'}),(None,None,set())])
def test_open_retires_carried_location_but_empty_initial_open_keeps_static_scan(
        tmp_path,held_id,rigid_reference,expected):
    hub,b,_,calls=execute_fixture(tmp_path)
    hub.carried_target_id=held_id;hub.carried_surface_reference=rigid_reference
    invalidations=[]
    hub.journal.mark_moved=lambda identity,step:invalidations.append((identity,step))
    def open_gripper(**params):
        calls.append(('open',));hub.robot.steps+=1;return {}
    hub.executor.set_gripper=open_gripper
    plan={'source_step':30,'subtask':'release contact','steps':[{
        'primitive':'set_gripper','target_id':'object','command':'open','camera':'agentview'}]}
    bound={'semantic_plan':plan,'plan':plan,'source_step':30,'needs_observation':False}
    b.compile_step=lambda *args:{'parameters':{'command':'open'}}
    receipt=execute_sketch(hub,b,plan,0,frozen_binding=bound)
    assert receipt['commands_completed'] and calls==[('open',)]
    assert {identity for identity,_ in invalidations}==expected
    assert all(step==31 for _,step in invalidations)
    assert hub.carried_surface_reference is None and hub.carried_target_id is None




def test_route_expansion_preserves_order_events_and_each_original_point():
    bound=stage()
    assert [s['primitive'] for s in bound['plan']['steps']]==['wait','move','move','move','release']
    assert [s['offset_m'][0] for s in bound['selections'][1:4]]==[.1,.2,.8]
    assert all(v['grounding_source_step']==30 for v in bound['locations'])
    assert len(bound['semantic_plan']['steps'])==3
    validator=Draft202012Validator(route_schema(BINDING))
    with pytest.raises(ValidationError):
        validator.validate({'waypoints':[point(.1)],'review_after_route':True,'review_reason':'retired B prefix'})
