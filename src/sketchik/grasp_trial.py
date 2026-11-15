"""One authorized high-level grasp sketch, followed by local candidate selection. No actuation."""
from __future__ import annotations

import argparse
import base64
import hashlib
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError

from .agent import ChatClient, load_settings, transcript_copy, ProviderError
from .planning import image_part, save_planning_history, planning_dialogue
from .tools import obj, V3, STEPS, public, tool_schemas, ToolHub, PERCEPTION_TOOLS
from .perception import SavedObservationRobot, ObservationHistory
from .grasp_preview import request


POSE = obj({'pose_ref':{'enum':['selected_grasp.pregrasp','selected_grasp.grasp']}},['pose_ref'])
SKETCH_PARAMS = {'move':POSE,
    'set_gripper':obj({'command':{'enum':['open','close']},'steps':STEPS},['command']),
    'move_relative':obj({'delta_m':V3},['delta_m']),'wait':obj({'steps':STEPS},['steps'])}
SKETCH_SCHEMA = obj({'source_step':{'type':'integer','minimum':0},'subtask':{'type':'string','minLength':1},
    'target_description':{'type':'string','minLength':1}, 'sam_prompt':{'type':'string','minLength':1},
    'grasp_review':obj({'previous_candidate_matches':{'type':['boolean','null']},
        'observed_grasp_region':{'type':'string','minLength':1},'intended_contact_region':{'type':'string','minLength':1},
        'clarification':{'type':'string','minLength':1}},['previous_candidate_matches','observed_grasp_region','intended_contact_region','clarification']),
    'grasp_preference' :{'type':'string','minLength':1},
    'steps':{'type':'array','minItems':1,'maxItems':20,'items':{'oneOf':[
        obj({'primitive':{'const':name},'params':params},['primitive','params']) for name,params in SKETCH_PARAMS.items()]}}},
    ['source_step','subtask','target_description','grasp_preference','steps'])








def restore_planning_images(messages, roots):
    """Reload saved public RGBs when resuming a previous multi-tool review history."""
    pictures = {}
    for root in roots:
        for path in root.rglob('*.png'):
            if path.is_symlink():
                continue
            url = image_part(path)['image_url']['url']
            pictures['saved-observation-image-sha256:' + hashlib.sha256(url.encode()).hexdigest()] = url
    for message in messages:
        if isinstance(message.get('content'), list):
            for part in message['content']:
                if part.get('type') == 'image_url':
                    reference = part['image_url']['url']
                    if reference not in pictures:
                        raise ValueError('Saved planning image is missing; cannot silently omit history')
                    part['image_url']['url'] = pictures[reference]
    return messages


def planning_instructions(text, allow_perception=True, *, active_observation=False):
    if allow_perception and not active_observation:
        return text
    head, separator, _ = text.partition('\n本轮开放submit_grasp_sketch')
    _, heading, tail = text.partition('\n## 读图、对象身份与状态')
    if not separator or not heading:
        raise ValueError('Grasp guide structure changed; cannot prepare symbolic-only instructions')
    if active_observation:
        middle='\n\n'+(Path(__file__).resolve().parents[2]/'prompts/initial_observation.md').read_text()+'\n'
    else:
        middle = '''

本次是独立静态抓取诊断：只看程序提供的保存图像、任务和状态，生成符号抓取草图；不调用感知工具，不求像素或三维坐标。当前没有后台Agent或放置路线分析。

本轮只开放submit_grasp_sketch。当前入口仍只接通抓取草图，按完整任务选择对象、分割范围和抓取意图，不编造未实现的放置动作。SAM3/GraspGen由执行器调用，Mask和抓取位姿仍由常驻Molmo选择ID或NONE；你不代替Molmo选候选。不需要先自行测量尺寸或定位后续放置点才能交草图。当前图片来自保存观测，没有主动移动相机。
'''
    return head + middle + heading + tail


def generate_plan(observation, env_file, review_trial=None, *, max_turns=40, max_tool_calls=100,
                  additional_instructions='', allow_perception=True):
    root=Path(__file__).resolve().parents[2];run=root/'runs'/(datetime.now().strftime('%Y%m%dT%H%M%S')+'_grasp_trial');run.mkdir()
    settings=load_settings(env_file)
    client=ChatClient(settings.get('SKETCHIK_BASE_URL'),settings.get('SKETCHIK_MODEL'),settings.get('SKETCHIK_API_KEY'),settings.get('SKETCHIK_REASONING'),timeout=120)
    meta=json.loads((observation/'metadata.json').read_text());state=public(json.loads((observation/'initial_state.json').read_text()))
    system=planning_instructions((root/'prompts/grasp_trial.md').read_text(), allow_perception)
    if additional_instructions:system += '\n\n' + additional_instructions
    (run/'system_prompt.md').write_text(system)
    tool={'type':'function','function':{'name':'submit_grasp_sketch','description':'Save a symbolic current-grasp sketch; no motion or candidate selection yet.','parameters':SKETCH_SCHEMA}}
    def emit(record):
        with (run/'planning_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
    hub=ToolHub(SavedObservationRobot(observation),None,run,emit,
                allowed_tools=PERCEPTION_TOOLS if allow_perception else frozenset())
    initial=hub.observe()[0]
    content=[{'type':'text','text':json.dumps({'task':meta['task_language'],'source_step':state['step'],
        'observation_id':initial['observation_id'],'robot_state':initial['state'],
        'static_observation':True,'phase':'Read-only current grasp planning; grounded poses are filled later.'})}]
    for camera in ['agentview','robot0_eye_in_hand']:
        content.extend([{'type':'text','text':f'Current calibrated camera: {camera}, 1024x1024'},image_part(observation/(camera+'.png'))])
    messages=[{'role':'system','content':system},{'role':'user','content':content}]
    if review_trial is not None:
        previous=review_trial.resolve()
        previous_cfg=json.loads((previous/'config.json').read_text());old_observation=Path(previous_cfg['observation'])
        old_meta=json.loads((old_observation/'metadata.json').read_text());old_state=public(json.loads((old_observation/'initial_state.json').read_text()))
        old_content=[{'type':'text','text':json.dumps({'task':old_meta['task_language'],'source_step':old_state['step'],'robot_state':old_state,'phase':'Previous grasp planning turn'})}]
        for camera in ['agentview','robot0_eye_in_hand']:
            old_content.extend([{'type':'text','text':f'Previous initial camera: {camera}'},image_part(old_observation/(camera+'.png'))])
        old_reply=json.loads((previous/'high_level_response.json').read_text())['message']
        calls=[c for c in old_reply.get('tool_calls',[]) if c.get('function',{}).get('name')=='submit_grasp_sketch']
        if len(calls)!=1:raise ValueError('Review requires a saved actual sketch submission')
        selected=json.loads((previous/'selection_result.json').read_text())
        timing=json.loads((previous/'full_run_timing.json').read_text());execution=Path(timing['execution_run'])
        executed=json.loads((execution/'execution_summary.json').read_text())
        executed.pop('official_task_success',None)
        receipt={'selection':selected,'execution':executed,'note':'This is the actual previous attempt, not the new current scene. No hidden object GT.'}
        followup='用户要求：请复核你上次想要的抓法。下面是Molmo选中的原候选及其实际停下画面。它是否符合你原先grasp_preference所说的夹把手？依据可见图像回答；如果不是，请说明实际是哪种接触，以及你本意要求两指分别接触哪一段材料/哪两面、哪一侧需要净空。不要只重复“抓把手”，不要把一指在杯内就自动称为抓把手；如果图像不足以判断请明确不确定，不要求你接受或否决某个编号。请在grasp_review中记录复核，并在grasp_preference里写清下一次选择需要满足的接触意图；然后基于最后提供的新重置观测重新提交完整抓取草图。新观测source_step才是本次计划的source_step，旧执行状态仅用于复核。'
        review_content=[{'type':'text','text':followup}]
        for label,path in [('Previous selected pose on both calibrated cameras',Path(selected['selected_image'])),
            ('Previous selected pose solid gripper and observed target points',previous/'candidates'/f"rank_{selected['candidate_id']:03d}_solid.png"),
            ('Previous real stopped scene, not a current input',execution/'step_002'/'agentview.png')]:
            review_content.extend([{'type':'text','text':label},image_part(path)])
        messages=[{'role':'system','content':system},{'role':'user','content':old_content},old_reply,
            {'role':'tool','tool_call_id':calls[0]['id'],'content':json.dumps(public(receipt))},
            {'role':'user','content':review_content},
            {'role':'user','content':[{'type':'text','text':'NEW RESET CURRENT OBSERVATION FOR THE NEXT GRASP:'},*content]}]
        if (previous/'planning_transcript.json').exists():
            history=restore_planning_images(json.loads((previous/'planning_transcript.json').read_text()),
                [old_observation,previous/'observations',previous/'planning_images'])
            receipts=[m for m in history if m.get('role')=='tool' and m.get('tool_call_id')==calls[0]['id']]
            if len(receipts)!=1:
                raise ValueError('Previous transcript is missing its actual sketch submission receipt')
            history[0]={'role':'system','content':system}
            receipts[0]['content']=json.dumps(public(receipt))
            messages=history+messages[-2:]
        (run/'review_source.json').write_text(json.dumps({'previous_trial':str(previous),'previous_execution':str(execution),
            'same_model':client.model==previous_cfg['provider']['model'],'user_requested':True},indent=2))
    (run/'planning_input.json').write_text(json.dumps(transcript_copy(messages),indent=2))
    config={'observation':str(observation.resolve()),'provider':client.metadata,'robot_actions':0,
            'planning_perception_tools': sorted(PERCEPTION_TOOLS) if allow_perception else []}
    (run/'config.json').write_text(json.dumps(config,indent=2));(run/'planning_tool.json').write_text(json.dumps(tool,indent=2))
    (run/'planning_tools.json').write_text(json.dumps(hub.schemas+[tool],indent=2))
    start=time.perf_counter()
    try:
        sketch,response,timing=planning_dialogue(client,hub,messages,tool,max_turns=max_turns,max_tool_calls=max_tool_calls)
        elapsed=timing['api_wall_s']
        (run/'high_level_response.json').write_text(json.dumps({'wall_s':elapsed,'planning':timing,**response},indent=2))
        Draft202012Validator(SKETCH_SCHEMA).validate(sketch)
        if sketch['source_step']!=state['step']:raise ValueError('Stale sketch source_step')
        if review_trial is not None and 'grasp_review' not in sketch:raise ValueError('Requested grasp review missing from submission')
        (run/'sketch.json').write_text(json.dumps(sketch,indent=2))
        print(json.dumps({'trial':str(run),'model':client.model,'reasoning':client.reasoning,'api_wall_s':elapsed,'sketch':sketch},indent=2),flush=True)
    except Exception as e:
        (run/'planning_error.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'wall_s':time.perf_counter()-start,'automatic_retry':False},indent=2))
        print(json.dumps({'trial':str(run),'planning_error':type(e).__name__}),flush=True);raise
    finally:
        save_planning_history(run,messages)
    return run


def replay_observation_prelude(robot, metadata, source, *, allow_recorded_closing=False):
    """Replay only recorded robot commands, never object state, then match RGB-D."""
    prelude=metadata.get('observation_prelude')
    if not prelude:return
    count=int(prelude['steps'])
    if prelude.get('kind')=='recorded_observation_controls':
        name=prelude['controls_file']
        if Path(name).name!=name or (source/name).is_symlink():
            raise ValueError('Recorded observation controls must be a local plain file')
        controls=[json.loads(line) for line in (source/name).read_text().splitlines()]
        if not 0<=count<robot.max_steps or len(controls)!=count:
            raise ValueError('Recorded observation control count/budget mismatch')
        for i,control in enumerate(controls,1):
            q=control.get('joint_target')
            if (set(control)!={'source_step','joint_target','gripper_command'} or control['source_step']!=i
                    or not isinstance(q,list) or len(q)!=7
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in q)
                    or control['gripper_command'] not in ((-1.0, 1.0) if allow_recorded_closing else (-1.0,))):
                raise ValueError('Invalid recorded open-gripper observation control; no replay dispatched')
        for control in controls:robot.step(control['joint_target'],control['gripper_command'])
    else:
        if not 0<count<=200 or prelude['gripper']!=-1.0:
            raise ValueError('Unsupported replay prelude')
        for _ in range(count):robot.step(prelude['joint_target'],prelude['gripper'])


def generate_observing_plan(source, run, env_file, gpu_uuid, peer_pids, *,
                             max_turns=12, max_tool_calls=60, history_interval_s=1.):
    """A can move the empty hand to collect real temporal history before its sketch."""
    import shutil
    root=Path(__file__).resolve().parents[2];run.mkdir(exist_ok=False)
    metadata=json.loads((source/'metadata.json').read_text())
    messages=[];robot=None;history=None;video=None
    def emit(record):
        from datetime import timezone
        with (run/'planning_events.jsonl').open('a') as stream:
            stream.write(json.dumps({**public(record),'utc':datetime.now(timezone.utc).isoformat()},allow_nan=False)+'\n')
    begin=time.perf_counter()
    try:
        settings=load_settings(env_file)
        client=ChatClient(settings.get('SKETCHIK_BASE_URL'),settings.get('SKETCHIK_MODEL'),
                          settings.get('SKETCHIK_API_KEY'),settings.get('SKETCHIK_REASONING'),timeout=120)
        from .runtime import configure
        gpu=configure(root,gpu_uuid,metadata['variant'],peer_pids)
        from .libero import LiberoRobot
        from .primitives import MotionExecutor
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
        robot=LiberoRobot(metadata['variant'],metadata['suite'],metadata['task'],metadata['seed'],metadata['init_state_id'],
                          size=metadata['image_size'],max_steps=metadata['max_steps'])
        video=imageio.get_writer(str(run/'observation_video.mp4'),fps=20,codec='libx264',quality=7)
        history=ObservationHistory(robot,run,emit,history_interval_s,video)
        robot.on_step=history.on_step
        replay_observation_prelude(robot,metadata,source)
        allowed=PERCEPTION_TOOLS|{'move','move_relative','move_waypoints','wait'}
        hub=ToolHub(robot,MotionExecutor(robot,emit),run,emit,allowed_tools=allowed)
        history.attach(hub)
        initial,pictures=hub.observe()
        system=planning_instructions((root/'prompts/grasp_trial.md').read_text(),active_observation=True)
        tool={'type':'function','function':{'name':'submit_grasp_sketch',
              'description':'Submit the grasp sketch after useful initial observation; executor and Molmo bind/select grasp later.',
              'parameters':SKETCH_SCHEMA}}
        messages=[{'role':'system','content':system},{'role':'user','content':[
            {'type':'text','text':json.dumps({'phase':'A initial observation then symbolic grasp sketch',
                'history_interval_s':history.interval_s,'history_purpose':'Saved calibrated observation history for independent grasp diagnosis; no background agent',
                'sampling_clock':'simulation time, not API/wall time'})},*pictures]}]
        final=run/'final_observation'
        config={'observation':str(final.resolve()),'initial_observation_source':str(source.resolve()),
                'provider':client.metadata,'mode':'active_initial_observation',
                'planning_perception_tools':sorted(PERCEPTION_TOOLS),'observation_motion_tools':sorted(allowed-PERCEPTION_TOOLS),
                'history_interval_simulation_s':history.interval_s}
        (run/'config.json').write_text(json.dumps(config,indent=2))
        (run/'system_prompt.md').write_text(system)
        (run/'planning_tools.json').write_text(json.dumps(hub.schemas+[tool],indent=2))
        (run/'planning_tool.json').write_text(json.dumps(tool,indent=2))
        (run/'planning_input.json').write_text(json.dumps(transcript_copy(messages),indent=2))
        sketch,response,timing=planning_dialogue(client,hub,messages,tool,max_turns=max_turns,
                                                max_tool_calls=max_tool_calls,allow_actions=True)
        if sketch['source_step']!=robot.steps:raise ValueError('Submitted observation state is stale')
        if robot.done or robot.steps>=robot.max_steps:raise ValueError('Observation used the remaining episode budget')
        if history.control_count!=robot.steps:raise ValueError('Observation control trace is incomplete')
        final.mkdir()
        state=robot.state()
        (final/'initial_state.json').write_text(json.dumps(state,indent=2))
        (final/'robot_only.xml').write_text(robot.robot.robot_model.get_xml())
        prelude={'kind':'recorded_observation_controls','steps':robot.steps,'controls_file':'observation_controls.jsonl'}
        (final/'metadata.json').write_text(json.dumps({**robot.metadata,'gpu':gpu,'observation_prelude':prelude,
             'capture_scope':'Final state after A observation motion; diagnostic replay only, not a fresh reset image'},indent=2))
        history.stream.flush()
        shutil.copyfile(run/'observation_controls.jsonl',final/'observation_controls.jsonl')
        for camera in robot.cameras:
            frame=robot.camera(camera)
            Image.fromarray(frame['rgb']).save(final/(camera+'.png'))
            np.savez_compressed(final/(camera+'.npz'),**{k:frame[k] for k in ('depth_m','K','world_from_camera')})
        (run/'sketch.json').write_text(json.dumps(sketch,indent=2))
        (run/'high_level_response.json').write_text(json.dumps({'wall_s':timing['api_wall_s'],'planning':timing,**response},indent=2))
        emit({'event':'a_observation_completed','source_step':robot.steps,'periodic_frames':len(history.periodic_ids),
              'initial_settle_steps':metadata.get('observation_prelude',{}).get('steps',0),
              'observation_wall_s':time.perf_counter()-begin,'api_wall_s':timing['api_wall_s']})
        print(json.dumps({'trial':str(run),'source_step':robot.steps,'planning':timing,'sketch':sketch},indent=2),flush=True)
    except Exception as exc:
        (run/'planning_error.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc),
            'wall_s':time.perf_counter()-begin,'automatic_retry':False,'physical_steps':robot.steps if robot else 0},indent=2))
        raise
    finally:
        if robot is not None:robot.on_step=None
        if history is not None:history.close()
        if video is not None:video.close()
        save_planning_history(run,messages)
        if robot is not None:robot.close()
    return run


def parse_choice(answer, allowed):
    raw=answer.strip()
    if raw.startswith('```json') and raw.endswith('```'):raw=raw[7:-3].strip()
    result=json.loads(raw)
    if not isinstance(result,dict) or set(result)-{'candidate_id','reason'} or 'candidate_id' not in result:raise ValueError('Expected candidate_id JSON')
    identity=result['candidate_id']
    if identity is not None and (type(identity) is not int or identity not in allowed):raise ValueError('Molmo returned an unprovided candidate ID')
    if not isinstance(result.get('reason'),str) or not result['reason'].strip():raise ValueError('A visible-evidence reason is required')
    return result


def bind_plan(sketch, grasp_base_world, base_to_site, approach_m):
    import numpy as np
    from scipy.spatial.transform import Rotation
    Draft202012Validator(SKETCH_SCHEMA).validate(sketch)
    g=np.asarray(grasp_base_world,dtype=float);calibration=np.asarray(base_to_site,dtype=float)
    if g.shape!=(4,4) or calibration.shape!=(4,4) or not np.isfinite(g).all() or not np.isfinite(calibration).all():
        raise ValueError('Expected finite calibrated 4x4 grasp transforms')
    pre=g.copy();pre[:3,3]-=float(approach_m)*g[:3,2]
    poses={'selected_grasp.pregrasp':pre@calibration,'selected_grasp.grasp':g@calibration}
    steps=[]
    for step in sketch['steps']:
        params=dict(step['params'])
        if 'pose_ref' in params:
            pose=poses[params.pop('pose_ref')]
            params.update(position=pose[:3,3].tolist(),quaternion_xyzw=Rotation.from_matrix(pose[:3,:3]).as_quat().tolist())
        steps.append({'primitive':step['primitive'],'params':params})
    return {'version':1,'steps':steps}


def run_selection(trial, gpu_uuid, peer_pids, perception_dir=None):
    root=Path(__file__).resolve().parents[2];trial=trial.resolve()
    if (trial/'selection_result.json').exists():raise ValueError('Trial already has a result; keep original evidence')
    sketch=json.loads((trial/'sketch.json').read_text());Draft202012Validator(SKETCH_SCHEMA).validate(sketch)
    cfg=json.loads((trial/'config.json').read_text());observation=Path(cfg['observation'])
    state=json.loads((observation/'initial_state.json').read_text())
    if state['step']!=sketch['source_step']:raise ValueError('Sketch observation mismatch')
    begin=time.perf_counter();stages={}
    previous_stages=json.loads((trial/'stage_timing.json').read_text()) if (trial/'stage_timing.json').exists() else {}
    env={**os.environ,'PYTHONPATH':str(root/'src'),'USE_FLAX':'0','USE_TF':'0','PYTHONDONTWRITEBYTECODE':'1'}
    def stage(name, module, arguments, marker, cpu=False):
        destination=trial/name
        if (destination/marker).exists():
            stages[name]={**previous_stages.get(name,{}),'reused_saved_evidence':True,'console_log':str(trial/(name+'_console.log'))};return destination
        if destination.exists():raise ValueError(f'Incomplete {name} evidence exists; inspect it rather than overwrite')
        start=time.perf_counter()
        with (trial/(name+'_console.log')).open('w') as f:
            proc=subprocess.run([str(root/'.venv-grasp/bin/python'),'-m',module,*arguments,'--output',str(destination)],cwd=root,
                env={**env,**({'CUDA_VISIBLE_DEVICES':''} if cpu else {})},stdout=f,stderr=subprocess.STDOUT)
        stages[name]={'wall_s':time.perf_counter()-start,'exit_code':proc.returncode}
        (trial/'stage_timing.json').write_text(json.dumps(stages,indent=2))
        if proc.returncode or not (destination/marker).exists():raise RuntimeError(f'{name} failed; see saved console log')
        return destination
    try:
        if perception_dir is None:
            preview=stage('perception','sketchik.grasp_preview',['--observation-dir',str(observation),'--target',sketch['target_description'],'--sam-prompt',sketch.get('sam_prompt',sketch['target_description']),'--molmo-url','http://127.0.0.1:18762'],'candidates.npz')
        else:
            preview=perception_dir.resolve()
            saved=json.loads((preview/'perception.json').read_text())
            if Path(saved['observation_directory']).resolve()!=observation.resolve() or saved['target_description']!=sketch['target_description'] or not (preview/'candidates.npz').exists():
                raise ValueError('Diagnostic preview does not match current saved observation')
            stages['perception']={'reused_saved_evidence':True,'source':str(preview),'actual_sam_prompt':saved.get('actual_sam_prompt',saved['target_description']),
                'high_level_description_unchanged':sketch['target_description']}
            (trial/'perception_source.json').write_text(json.dumps(stages['perception'],indent=2))
        cache=request('http://127.0.0.1:18760/release_cache',{})
        (trial/'graspgen_cache_release.json').write_text(json.dumps(cache,indent=2))
        screen=stage('screen','sketchik.grasp_screen',['--preview',str(preview)],'result.json',cpu=True)
        screening=json.loads((screen/'result.json').read_text());ranks=screening['retained_original_ranks']
        if not ranks:
            report={'candidate_id':None,'reason':'No candidate passed observed-scene and IK screening','robot_actions':0,'retained':0}
            (trial/'selection_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
        render_args=['--screen',str(screen),'--gpu-uuid',gpu_uuid,'--stills-only']
        for pid in peer_pids:render_args+=['--peer-pid',str(pid)]
        pictures=stage('candidates','sketchik.grasp_render',render_args,'render_metadata.json')
        meta=json.loads((observation/'metadata.json').read_text())
        prompt=(root/'prompts/grasp_selector.md').read_text()
        intent={'full_task':meta['task_language'],'current_subtask':sketch['subtask'],
            'target_description':sketch['target_description'],'grasp_preference':sketch['grasp_preference']}
        if 'grasp_review' in sketch:intent['grasp_contact_review']=sketch['grasp_review']
        choices=[];selection_start=time.perf_counter()
        def select_group(ids,label):
            base_label=label
            cached=trial/(label+'_response.json')
            if cached.exists():
                response=json.loads(cached.read_text())
                try:
                    parsed=parse_choice(response['answer'],ids)
                except (ValueError,TypeError):
                    suffix=2
                    while (trial/(base_label+f'_attempt{suffix}_response.json')).exists():suffix+=1
                    label=base_label+f'_attempt{suffix}'
                else:
                    choices.append({'label':label,**parsed,'reused_saved_response':True,'client_wall_s':response['client_wall_s'],'inference_wall_s':response['inference_wall_s']})
                    (trial/'molmo_choices.json').write_text(json.dumps(choices,indent=2));return parsed
            current_prompt=prompt+'\n'+json.dumps({**intent,'allowed_candidate_ids':ids})
            images=[{'label':camera+' current scene','path':observation/(camera+'.png')} for camera in ['agentview','robot0_eye_in_hand']]
            images += [{'label':f'Candidate ID {identity}, two calibrated views','path':pictures/f'rank_{identity:03d}_photos.png'} for identity in ids]
            # Persist inputs as file references, never duplicate large encoded images in logs.
            (trial/(label+'_input.json')).write_text(json.dumps({'prompt':current_prompt,'images':[{'label':v['label'],'path':str(v['path'])} for v in images]},indent=2))
            payload={'prompt':current_prompt,'images':[{'label':v['label'],'image_base64':base64.b64encode(v['path'].read_bytes()).decode()} for v in images]}
            start=time.perf_counter();response=request('http://127.0.0.1:18762/infer',payload,timeout=180)
            response['client_wall_s']=time.perf_counter()-start
            (trial/(label+'_response.json')).write_text(json.dumps(response,indent=2))
            parsed=parse_choice(response['answer'],ids)
            choices.append({'label':label,**parsed,'client_wall_s':response['client_wall_s'],'inference_wall_s':response['inference_wall_s']})
            (trial/'molmo_choices.json').write_text(json.dumps(choices,indent=2));return parsed
        winners=[]
        # All retained candidates are shown, at most fourteen paired candidate pictures per request.
        for i in range(0,len(ranks),14):
            chosen=select_group(ranks[i:i+14],f'molmo_batch_{i//14:02d}')
            if chosen['candidate_id'] is not None:winners.append(chosen)
        choice=({'candidate_id':None,'reason':'No group contained a suitable grasp'} if not winners else winners[0])
        if len(winners)>1:choice=select_group([v['candidate_id'] for v in winners],'molmo_final')
        report={**choice,'all_retained_original_ranks':ranks,'all_candidates_shown':True,'retained':len(ranks),
            'generated':screening['candidates'],'robot_actions':0,'object_gt_used':False,'physical_grasp_tested':False,
            'molmo_model':'Molmo2-ER (4B), resident FP16','molmo_selection_wall_s':time.perf_counter()-selection_start,
            'selection_pipeline_wall_s':time.perf_counter()-begin,'stages':stages,
            'limitations':screening['limitations']}
        if choice['candidate_id'] is not None:
            import numpy as np
            data=np.load(preview/'candidates.npz');index=choice['candidate_id']-1;g=data['grasps'][index]
            calibration=screening['native_gripper_calibration']['base_to_grip_site']
            numeric=bind_plan(sketch,g,calibration,screening['approach_m'])
            (trial/'bound_numeric_plan.json').write_text(json.dumps(numeric,indent=2))
            selected={'candidate_id':index+1,'grasp_base_world':g.tolist(),'base_to_grip_site':calibration,
                'approach_m':screening['approach_m'],'score':float(data['scores'][index]),'source_step':sketch['source_step'],
                'image':str(pictures/f'rank_{index+1:03d}_photos.png'),'robot_actions':0}
            (trial/'selected_grasp.json').write_text(json.dumps(selected,indent=2));report['selected_image']=selected['image']
        (trial/'selection_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    except Exception as e:
        (trial/'selection_error.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'wall_s':time.perf_counter()-begin,'stages':stages,'robot_actions':0},indent=2))
        raise


def execute_selected(trial, gpu_uuid, peer_pids, lift_test_m=0., *, on_execution_ready=None):
    """Execute an unchanged selected grasp in a matching reset, optionally test lift.

    Uses observed RGB-D and robot state for reset matching, never object poses.
    """
    import hashlib
    import numpy as np
    from PIL import Image
    root=Path(__file__).resolve().parents[2];trial=trial.resolve()
    if not 0<=lift_test_m<=.10:raise ValueError('Lift diagnostic must be 0..0.10m')
    cfg=json.loads((trial/'config.json').read_text());source=Path(cfg['observation'])
    metadata=json.loads((source/'metadata.json').read_text());old_state=json.loads((source/'initial_state.json').read_text())
    selected=json.loads((trial/'selected_grasp.json').read_text())
    prelude=metadata.get('observation_prelude')
    expected_step=int(prelude['steps']) if prelude else 0
    if selected['source_step']!=old_state['step'] or old_state['step']!=expected_step:
        raise ValueError('Selected source step does not match the replayable reset prelude')
    original=json.loads((trial/'bound_numeric_plan.json').read_text())
    if set(original)!={'version','steps'} or original['version']!=1:raise ValueError('Invalid selected numeric plan')
    tool_schema=next(t['function']['parameters'] for t in tool_schemas() if t['function']['name']=='execute_plan')
    Draft202012Validator(tool_schema).validate({'source_step':old_state['step'],'steps':original['steps']})
    run=root/'runs'/(datetime.now().strftime('%Y%m%dT%H%M%S')+'_grasp_execute');run.mkdir()
    provenance={'source_trial':str(trial),'source_observation':str(source),'candidate_id':selected['candidate_id'],
        'selected_pose_unchanged':True,'source_plan_sha256':hashlib.sha256((trial/'bound_numeric_plan.json').read_bytes()).hexdigest(),
        'extra_diagnostic_lift_m':lift_test_m,'object_gt_used':False,'assessment':'pending_real_execution'}
    (run/'execution_provenance.json').write_text(json.dumps(provenance,indent=2))
    (run/'source_bound_plan.json').write_bytes((trial/'bound_numeric_plan.json').read_bytes())
    from .runtime import configure
    gpu=configure(root,gpu_uuid,metadata['variant'],peer_pids)
    from .libero import LiberoRobot
    from .validation import run_validation
    start=time.perf_counter();robot=None
    try:
        robot=LiberoRobot(metadata['variant'],metadata['suite'],metadata['task'],metadata['seed'],metadata['init_state_id'],
            size=metadata['image_size'],max_steps=metadata['max_steps'])
        environment_create_s=time.perf_counter()-start
        replay_start=time.perf_counter()
        replay_observation_prelude(robot,metadata,source)
        replay_wall_s=time.perf_counter()-replay_start
        (run/'metadata.json').write_text(json.dumps({**robot.metadata,'gpu':gpu,'observation_prelude':prelude,
            'environment_create_wall_s':environment_create_s,'observation_replay_wall_s':replay_wall_s,
            'initialization_wall_s':time.perf_counter()-start},indent=2))
        fresh=robot.state();(run/'initial_state.json').write_text(json.dumps(fresh,indent=2))
        checks={'robot_state':{},'cameras':{},'robot_actions_before_match':robot.steps}
        for key in ['q','position','quaternion_xyzw','finger_joint_positions','gripper_command']:
            checks['robot_state'][key]=bool(np.allclose(fresh[key],old_state[key],rtol=0,atol=1e-8))
        for camera in robot.cameras:
            frame=robot.camera(camera);Image.fromarray(frame['rgb']).save(run/(camera+'.png'))
            np.savez_compressed(run/(camera+'.npz'),depth_m=frame['depth_m'],K=frame['K'],world_from_camera=frame['world_from_camera'])
            rgb=np.asarray(Image.open(source/(camera+'.png')).convert('RGB'));saved=np.load(source/(camera+'.npz'))
            checks['cameras'][camera]={'rgb_identical':bool(np.array_equal(rgb,frame['rgb'])),
                'rgb_max_difference':int(np.abs(rgb.astype(int)-frame['rgb'].astype(int)).max()),
                'depth_matches':bool(np.allclose(saved['depth_m'],frame['depth_m'],rtol=0,atol=1e-7,equal_nan=True)),
                'K_matches':bool(np.allclose(saved['K'],frame['K'],rtol=0,atol=1e-8)),
                'extrinsics_match':bool(np.allclose(saved['world_from_camera'],frame['world_from_camera'],rtol=0,atol=1e-8))}
        checks['matched']=all(checks['robot_state'].values()) and all(
            v['rgb_identical'] and v['depth_matches'] and v['K_matches'] and v['extrinsics_match'] for v in checks['cameras'].values())
        (run/'initial_observation_match.json').write_text(json.dumps(checks,indent=2))
        if not checks['matched']:raise ValueError('Fresh reset differs from selected observation; no stale pose dispatched')
        execution=json.loads(json.dumps(original))
        if lift_test_m:execution['steps'].append({'primitive':'move_relative','params':{'delta_m':[0.,0.,lift_test_m]}})
        (run/'execution_plan.json').write_text(json.dumps(execution,indent=2))
        print(json.dumps({'run':str(run),'candidate_id':selected['candidate_id'],'initial_observation_matched':True,'extra_lift_m':lift_test_m}),flush=True)
        if on_execution_ready is not None:
            on_execution_ready(run)
        result=run_validation(robot,run,'execute',run/'execution_plan.json')
        (run/'execution_summary.json').write_text(json.dumps({'candidate_id':selected['candidate_id'],'actions':result['actions'],
            'all_commands_completed':result['control_validation_passed'],'error':result['error'],'wall_seconds':result['wall_seconds'],
            'initialization_actions':expected_step,'grasp_actions':result['actions']-expected_step,'grasp_success':None,'grasp_assessment':'Review real RGB before/during closure and after lift; no object GT.',
            'official_task_success':result['official_task_success'],'extra_lift_m':lift_test_m},indent=2))
        print(json.dumps({'run':str(run),'actions':result['actions'],'all_commands_completed':result['control_validation_passed'],
            'error':result['error'],'wall_seconds':result['wall_seconds'],'grasp_success':None}),flush=True)
        return run
    except Exception as exc:
        (run/'execution_error.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc),
            'actions':robot.steps if robot else 0,'wall_s':time.perf_counter()-start},indent=2))
        print(json.dumps({'run':str(run),'error':str(exc)}),flush=True);raise
    finally:
        if robot is not None:robot.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='phase',required=True)
    p=sub.add_parser('plan');p.add_argument('--observation',type=Path,required=True);p.add_argument('--env-file',type=Path,required=True);p.add_argument('--review-trial',type=Path)
    p.add_argument('--max-turns',type=int,default=40);p.add_argument('--max-tool-calls',type=int,default=100)
    p=sub.add_parser('select');p.add_argument('--trial',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--peer-pid',type=int,action='append',default=[]);p.add_argument('--perception-dir',type=Path)
    p=sub.add_parser('execute');p.add_argument('--trial',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--peer-pid',type=int,action='append',default=[]);p.add_argument('--lift-test-m',type=float,default=0.)
    p=sub.add_parser('observe-plan');p.add_argument('--source-observation',type=Path,required=True)
    p.add_argument('--output-run',type=Path,required=True);p.add_argument('--env-file',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--peer-pid',type=int,action='append',default=[])
    p.add_argument('--history-interval-s',type=float,default=1.)
    p.add_argument('--max-turns',type=int,default=12);p.add_argument('--max-tool-calls',type=int,default=60)
    args=parser.parse_args()
    if args.phase=='plan':generate_plan(args.observation,args.env_file,args.review_trial,max_turns=args.max_turns,max_tool_calls=args.max_tool_calls)
    elif args.phase=='select':run_selection(args.trial,args.gpu_uuid,args.peer_pid,args.perception_dir)
    elif args.phase=='observe-plan':generate_observing_plan(args.source_observation,args.output_run,args.env_file,
        args.gpu_uuid,args.peer_pid,max_turns=args.max_turns,max_tool_calls=args.max_tool_calls,
        history_interval_s=args.history_interval_s)
    else:execute_selected(args.trial,args.gpu_uuid,args.peer_pid,args.lift_test_m)


if __name__=='__main__':main()
