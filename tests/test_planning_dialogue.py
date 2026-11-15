from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from sketchik.agent import ProviderError
from sketchik.planning import planning_dialogue
from sketchik.tools import obj
HANDOFF = obj({'source_step': {'type': 'integer'}, 'coverage_complete': {'type': 'boolean'},
    'objects': {'type': 'array', 'items': {'type':'object','properties':{'missing':{'type':'array'}},'required':['missing']}},
    'whole_task_coverage': {'type': 'string'},
    'plan': {'type':'object','properties':{'source_step':{'type':'integer'}},'required':['source_step']}},
    ['source_step', 'coverage_complete', 'objects', 'whole_task_coverage', 'plan'])
# The nested fields above exercise generic dialogue transport, not a robot planning interface.

from sketchik.tools import tool_schemas


def handoff():
    return {'source_step':0,'coverage_complete':True,
        'objects':[{'object_id':'mug','description':'visible mug','interactions':['grasp'],
            'needed_geometry':'visible grasp surface','evidence':[{'observation_id':'obs_00000',
                'camera':'agentview','visible_evidence':'mug on table'}],'missing':[]}],
        'whole_task_coverage':'Mug observed; hidden surfaces unknown.',
        'plan':{'source_step':0,'subtask':'grasp mug','steps':[{'primitive':'move',
            'target_id':'mug','intent':'approach visible mug','camera':'agentview'}],
            'expected_effects':['approach mug'],'final_task':False}}


def tool_call(name,args,identity):
    return {'type':'function','id':identity,'function':{'name':name,'arguments':json.dumps(args)}}


def response(calls):
    return {'message':{'role':'assistant','tool_calls':calls},'finish_reason':'tool_calls'}


def fixture_hub(schemas=()):
    events=[];dispatched=[]
    def forbidden_dispatch(name,args):
        dispatched.append(name)
        raise AssertionError('An invalid submission batch must not dispatch tools')
    return SimpleNamespace(schemas=list(schemas),robot=SimpleNamespace(steps=0),
        call=forbidden_dispatch,emit=events.append,events=events,dispatched=dispatched)


SUBMIT={'type':'function','function':{'name':'submit_handoff','parameters':HANDOFF}}


def test_missing_handoff_field_is_tool_feedback_and_model_can_explicitly_correct_it():
    hub=fixture_hub();good=handoff();bad=deepcopy(good);del bad['objects'][0]['missing']
    messages=[{'role':'user','content':'Retained public observation history'}];initial=deepcopy(messages)
    class Client:
        turns=0
        def complete(self,messages,tools):
            self.turns+=1
            if self.turns==1:return response([tool_call('submit_handoff',bad,'invalid')])
            feedback=json.loads(messages[-1]['content'])
            assert messages[-1]['role']=='tool' and messages[-1]['tool_call_id']=='invalid'
            assert feedback['saved'] is False and feedback['error_type']=='schema_validation'
            assert feedback['validation_path']==['objects',0] and 'missing' in feedback['error']
            assert feedback['robot_actions']==0 and feedback['current_source_step']==0
            return response([tool_call('submit_handoff',good,'corrected')])
    client=Client();actual,_,timing=planning_dialogue(client,hub,messages,SUBMIT)
    assert actual==good and client.turns==2 and timing['rounds']==2 and timing['tool_calls']==2
    assert messages[:1]==initial and messages[1]['tool_calls'][0]['id']=='invalid'
    assert messages[-1]['tool_call_id']=='corrected' and json.loads(messages[-1]['content'])['saved']
    assert hub.dispatched==[] and hub.robot.steps==0
    assert any(event['event']=='submission_schema_rejected' for event in hub.events)


def test_single_semantic_boundary_accepts_plain_json_without_inventing_tool_calls():
    from sketchik.stage_context import closed_tool_dialogue
    hub=fixture_hub();raw={'role':'assistant','content':json.dumps(handoff())}
    class Client:
        def complete(self,*args):return {'message':raw,'finish_reason':'stop'}
    messages=[{'role':'user','content':'same public observation history'}]
    actual,response,timing=planning_dialogue(Client(),hub,messages,SUBMIT)
    assert actual==handoff() and response['message']==raw
    assert timing['tool_calls']==0 and timing['submission_transport']=='assistant_json'
    assert messages[1] is raw and not any(m.get('tool_calls') or m['role']=='tool' for m in messages)
    assert json.loads(messages[-1]['content'])['saved'] is True
    assert hub.dispatched==[]
    closed_tool_dialogue(messages)


def test_plain_json_never_dispatches_observation_motion():
    motion=next(tool for tool in tool_schemas() if tool['function']['name']=='move_relative')
    hub=fixture_hub([motion])
    class Client:
        def complete(self,*args):return {'message':{'role':'assistant','content':json.dumps({
            'source_step':0,'params':{'delta_m':[0,0,.02]}})}}
    with pytest.raises(ProviderError,match='No tool call'):
        planning_dialogue(Client(),hub,[],SUBMIT,allow_actions=True)
    assert hub.dispatched==[] and hub.robot.steps==0


def test_plain_semantic_data_still_requires_all_schema_fields():
    hub=fixture_hub();bad=handoff();del bad['coverage_complete']
    class Client:
        def complete(self,*args):return {'message':{'role':'assistant','content':json.dumps(bad)}}
    messages=[]
    with pytest.raises(ProviderError,match='reply budget exhausted'):
        planning_dialogue(Client(),hub,messages,SUBMIT,max_turns=1)
    assert json.loads(messages[-1]['content'])['saved'] is False
    assert not any(m.get('tool_calls') or m['role']=='tool' for m in messages) and hub.dispatched==[]


def test_invalid_handoff_prevents_same_batch_observation_motion_from_dispatching():
    motion=next(tool for tool in tool_schemas() if tool['function']['name']=='move_relative')
    hub=fixture_hub([motion]);bad=handoff();del bad['objects'][0]['missing']
    class Client:
        turns=0
        def complete(self,messages,tools):
            self.turns+=1
            if self.turns==1:return response([
                tool_call('move_relative',{'source_step':0,'params':{'delta_m':[0,0,.02]}},'motion'),
                tool_call('submit_handoff',bad,'invalid')])
            replies={message['tool_call_id']:json.loads(message['content'])
                     for message in messages if message['role']=='tool'}
            assert replies['motion']['error_type']=='batch_not_dispatched'
            assert replies['invalid']['error_type']=='schema_validation'
            assert all(reply['robot_actions']==0 for reply in replies.values())
            return response([tool_call('submit_handoff',handoff(),'corrected')])
    actual,_,timing=planning_dialogue(Client(),hub,[],SUBMIT,allow_actions=True)
    assert actual==handoff() and timing['tool_calls']==3
    assert hub.dispatched==[] and hub.robot.steps==0


def test_schema_feedback_does_not_retry_provider_failure_or_exceed_turn_budget():
    bad=handoff();del bad['objects'][0]['missing']
    class Client:
        turns=0
        def complete(self,messages,tools):
            self.turns+=1
            if self.turns==1:return response([tool_call('submit_handoff',bad,'invalid')])
            raise ProviderError('Provider transport failed; no automatic retry')
    hub=fixture_hub();client=Client();messages=[]
    with pytest.raises(ProviderError,match='Provider transport failed'):
        planning_dialogue(client,hub,messages,SUBMIT)
    assert client.turns==2 and json.loads(messages[-1]['content'])['saved'] is False
    client=Client()
    with pytest.raises(ProviderError,match='reply budget exhausted'):
        planning_dialogue(client,hub,[],SUBMIT,max_turns=1)
    assert client.turns==1 and hub.dispatched==[] and hub.robot.steps==0


@pytest.mark.parametrize('invalid',['outer_step','plan_step','json'])
def test_stale_and_malformed_handoff_are_corrected_without_actions(invalid):
    hub=fixture_hub();hub.robot.steps=1625
    good=handoff();good['source_step']=good['plan']['source_step']=1625
    bad=deepcopy(good)
    if invalid=='outer_step':bad['source_step']=30
    elif invalid=='plan_step':bad['plan']['source_step']=30
    messages=[];original_schema=deepcopy(SUBMIT)
    class Client:
        turns=0
        def complete(self,messages,tools):
            self.turns+=1;p=tools[-1]['function']['parameters']['properties']
            assert p['source_step']['const']==p['plan']['properties']['source_step']['const']==1625
            if self.turns==1:
                call=tool_call('submit_handoff',bad,'invalid')
                if invalid=='json':call['function']['arguments']='{broken JSON'
                return response([call])
            feedback=json.loads(messages[-1]['content'])
            assert feedback['current_source_step']==1625 and feedback['robot_actions']==0
            assert not feedback['saved']
            return response([tool_call('submit_handoff',good,'corrected')])
    actual,_,timing=planning_dialogue(Client(),hub,messages,SUBMIT)
    assert actual==good and timing['rounds']==2 and hub.dispatched==[]
    assert SUBMIT==original_schema and messages[0]['tool_calls'][0]['id']=='invalid'


def test_submission_schema_tracks_step_after_observation_action():
    motion=next(t for t in tool_schemas() if t['function']['name']=='move_relative')
    hub=fixture_hub([motion]);hub.fatal=False
    def move(name,args):
        hub.dispatched.append(name);hub.robot.steps=7
        return {'ok':True,'current_source_step':7},[]
    hub.call=move
    class Client:
        turns=0
        def complete(self,messages,tools):
            self.turns+=1;p=tools[-1]['function']['parameters']['properties']
            if self.turns==1:
                assert p['source_step']['const']==0
                return response([tool_call('move_relative',{'source_step':0,'params':{'delta_m':[0,0,.02]}},'move')])
            assert p['source_step']['const']==p['plan']['properties']['source_step']['const']==7
            good=handoff();good['source_step']=good['plan']['source_step']=7
            return response([tool_call('submit_handoff',good,'submit')])
    actual,_,_=planning_dialogue(Client(),hub,[],SUBMIT,allow_actions=True)
    assert actual['source_step']==7 and hub.dispatched==['move_relative']


def test_invalid_observation_source_arguments_reject_the_whole_batch_then_correct_in_same_history(tmp_path):
    from sketchik.tools import obj
    original={'role':'user','content':'Original observed images retained'}
    hub=SimpleNamespace(robot=SimpleNamespace(steps=0),fatal=False,emit=lambda event:None)
    view={'type':'function','function':{'name':'request_views','parameters':obj({
        'observation_id':{'enum':['seen']},'camera':{'const':'agentview'}},['observation_id','camera'])}}
    hub.schemas=[view];executed=[]
    hub.call=lambda name,args:(executed.append((name,args)) or {'ok':True},[])
    requests=[]
    class Client:
        def complete(self,messages,tools):
            requests.append(deepcopy(messages))
            if len(requests)==1:
                return response([tool_call('request_views',{'camera':'agentview'},'bad'),
                    tool_call('submit_handoff',handoff(),'blocked_submission')])
            assert executed==[]
            return response([tool_call('request_views',{'observation_id':'seen','camera':'agentview'},'good'),
                tool_call('submit_handoff',handoff(),'accepted_submission')])
    messages=[original]
    result,_,timing=planning_dialogue(Client(),hub,messages,SUBMIT,
        allow_actions=True,extra_observation_tools=frozenset({'request_views'}))
    assert result==handoff() and len(requests)==2
    assert len(executed)==1 and executed[0][1]['observation_id']=='seen'
    assert requests[1][0]==original
    errors=[json.loads(m['content']) for m in requests[1] if m['role']=='tool']
    assert len(errors)==2 and all(error['robot_actions']==0 for error in errors)
    assert timing['tool_calls']==4
