import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from sketchik.task_trial import ProgramObservationHub, compact_view_feedback

@pytest.mark.parametrize('existing', [False, True], ids=['fresh_start', 'resumed'])
def test_initial_a_receives_detailed_geometry_and_preserves_existing_evidence(tmp_path,monkeypatch,existing):
    from sketchik.task_trial import append_robot_physical_facts
    facts={'source_step':30,'hand_span_xyz_m':[.20,.06,.13],
        'fingers':[{'length_along_tool_z_m':.06}]*2,'current_inner_jaw_gap_m':.079}
    ik=SimpleNamespace(fk=lambda q:([0,0,0],[[1,0,0],[0,1,0],[0,0,1]]))
    state={'step':30,'q':[0]*7,'finger_joint_names':[],'finger_joint_positions':[]}
    control={'target_gap_control_available':False,'steps_role':'minimum_dwell_then_finger_settling'}
    hub=SimpleNamespace(run=tmp_path,executor=SimpleNamespace(ik=ik,gripper_control_context=lambda:control),
        robot=SimpleNamespace(state=lambda:state),emit=lambda event:None)
    path=tmp_path/'robot_geometry_evidence.json'
    if existing:path.write_text(json.dumps({'prior_evidence':{'source_step':10}}))
    monkeypatch.setattr('sketchik.robot_geometry.gripper_geometry_in_eef',lambda *args:facts)
    messages=[];append_robot_physical_facts(hub,messages)
    value=json.loads(messages[0]['content'])
    assert value['robot_physical_facts']['fingers'][0]['length_along_tool_z_m']==.06
    assert value['gripper_control']==control
    assert json.loads(path.read_text())['gripper_control']==control
    assert json.loads(path.read_text())['initial_hand_geometry']==facts
    assert ('prior_evidence' in json.loads(path.read_text()))==existing


def test_view_feedback_keeps_provenance_without_numeric_camera_search():
    result = compact_view_feedback({'status': 'completed', 'source_step': 80,
        'attempts': [{'target': {'target_id': 'opening'}, 'status': 'view_captured',
            'captured_observation_id': 'saved', 'geometry': {'anchor_world_m': [1, 2, 3]},
            'candidates': [{'quaternion_xyzw': [0, 0, 0, 1]}]}]})
    assert result['targets'] == [{'target_id': 'opening', 'status': 'view_captured', 'observation_id': 'saved'}]
    assert 'geometry' not in json.dumps(result) and 'quaternion' not in json.dumps(result)


@pytest.fixture
def catalog_stub(tmp_path,monkeypatch):
    from PIL import Image
    photos=[]
    for name,color in [('overview','red'),('opposite','green'),('top','blue')]:
        photo=tmp_path/f'{name}.png'
        Image.new('RGB',(1024,1024),color).save(photo)
        photos.append({'image_file':str(photo)})
    catalog={'source_step':0,'source_steps':[10],'table_height_m':.8,
        'table_sources':[{'observation_id':'source','camera':'agentview','source_step':10}],
        'scope':'whole observed table','components':{'G1':{'component_id':'G1',
            'observed_bounds_world_m':[[0,0,.81],[.2,.2,1.]],'observed_span_xyz_m':[.2,.2,.19],
            'point_count':100,'sources':[{'observation_id':'source','camera':'agentview',
                'source_step':10,'bounds_xyxy':[1,1,50,50]}],
            'patches':[],'source_steps':[10],'last_observed_step':10,
            'semantic_identity_verified':False,'complete_dimensions_known':False}},
        'views':photos}
    (tmp_path/'scene_catalog').mkdir()
    (tmp_path/'scan_evidence.json').write_text(json.dumps({'sources':catalog['table_sources']}))
    def build(hub):catalog['source_step']=hub.robot.steps;return catalog
    monkeypatch.setattr('sketchik.scene_catalog.build_scene_catalog',build)
    monkeypatch.setattr('sketchik.grounding_packet.prepare_table_context',lambda *args:{'present':True})
    return catalog


def test_fixed_table_scan_and_reset_finish_before_first_a_request(tmp_path,monkeypatch,catalog_stub):
    from sketchik.task_trial import table_observation
    robot=SimpleNamespace(steps=30)
    hub=ProgramObservationHub(robot,SimpleNamespace(),tmp_path,lambda event:None)
    order=[]
    def scan(actual_hub):
        order.append('five_direction_scan');robot.steps=60
        return {'status':'completed','source_step':60,'wall_s':3.,'captured_views':5,
            'observation_ids':['scan'],'attempts':[]},[{'type':'text','text':'all actual scan pictures'}]
    def reset(actual_hub):
        order.append('reset');robot.steps=70
        return {'wall_s':1.,'execution':{'converged':True}},[{'type':'text','text':'actual home pictures'}]
    def dialogue(client,actual_hub,messages,submit,**kwargs):
        from sketchik.planning import image_part
        order.append('first_a_request')
        assert robot.steps==70 and actual_hub.schemas==[]
        assert submit['function']['name']=='submit_scene_requirements'
        assert 'all actual scan pictures' in json.dumps(messages)
        assert 'actual home pictures' in json.dumps(messages)
        assert [part for part in messages[-1]['content'] if part['type']=='image_url']==[
            image_part(Path(view['image_file'])) for view in catalog_stub['views']]
        return {'source_step':70,'objects':[{'object_id':'item','component_ids':['G1'],
            'description':'item','interactions':['grasp'],'required_surfaces':['contact_surface'],
            'missing_surfaces':[]}]},None,{'rounds':1,'api_wall_s':.1,'tool_calls':1}
    def supplement(*args,**kwargs):
        return {'status':'completed','source_step':70,'robot_actions':0,'observation_ids':[],
            'wall_s':0.,'captured_views':0,'attempts':[]},[]
    monkeypatch.setattr('sketchik.table_scan.run_table_scan',scan)
    monkeypatch.setattr('sketchik.task_trial.reset_observation_arm',reset)
    monkeypatch.setattr('sketchik.task_trial.planning_dialogue',dialogue)
    monkeypatch.setattr('sketchik.guided_observation.run_guided_observation',supplement)
    result,_,timing=table_observation(SimpleNamespace(),hub,[],max_turns=5,max_tool_calls=8)
    assert order==['five_direction_scan','reset','first_a_request']
    assert not result['coverage_complete'] and result['source_step']==70
    assert result['objects'][0]['component_ids']==['G1']


def test_supplement_is_one_program_batch_after_requirements_and_resets_once(tmp_path,monkeypatch,catalog_stub):
    from sketchik.task_trial import table_observation
    robot=SimpleNamespace(steps=0)
    hub=ProgramObservationHub(robot,SimpleNamespace(),tmp_path,lambda event:None)
    order=[];messages=[]
    def scan(*args):
        robot.steps=10;order.append('fixed_scan')
        return {'status':'completed','source_step':10,'wall_s':1.,'captured_views':5,'attempts':[]},[]
    def reset(*args):
        robot.steps+=5;order.append('reset')
        return {'wall_s':.1},[]
    def dialogue(client,actual_hub,current,tool,**kwargs):
        order.append('A_requirements')
        return {'source_step':15,'objects':[{'object_id':'container','component_ids':['G1'],
            'description':'container','interactions':['place'],'required_surfaces':['interior_support'],
            'missing_surfaces':['interior_support']}]},None,{'rounds':1,'api_wall_s':.1,'tool_calls':1}
    def supplement(actual_hub,catalog,handoff,**kwargs):
        order.append('one_batch');assert handoff['source_step']==15
        assert handoff['objects'][0]['acquisition_needs']==['interior_support']
        robot.steps=25
        observation=tmp_path/'observations/new';observation.mkdir(parents=True)
        (observation/'observation.json').write_text(json.dumps({'source_step':25}))
        return {'status':'completed','attempts':[],'observation_ids':['new'],'source_step':25,
            'robot_actions':10,'captured_views':1,'wall_s':.1},[{'type':'text','text':'new actual image'}]
    monkeypatch.setattr('sketchik.table_scan.run_table_scan',scan)
    monkeypatch.setattr('sketchik.task_trial.reset_observation_arm',reset)
    monkeypatch.setattr('sketchik.task_trial.planning_dialogue',dialogue)
    monkeypatch.setattr('sketchik.guided_observation.run_guided_observation',supplement)
    handoff,_,timing=table_observation(SimpleNamespace(),hub,messages,max_turns=5,max_tool_calls=8)
    assert order==['fixed_scan','reset','A_requirements','one_batch','reset']
    assert handoff['source_step']==30 and handoff['coverage_complete'] is False
    assert hub.initial_supplement_ids==['new'] and timing['supplement_batches']==1
    manifest=json.loads((tmp_path/'scan_evidence.json').read_text())
    assert len(manifest['sources'])==3  # Both real cameras joined the final global fusion.
    assert 'new actual image' in json.dumps(messages) and hub.schemas==[]

