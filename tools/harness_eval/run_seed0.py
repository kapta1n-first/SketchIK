"""Prepared future native HarnessVLA runner; requires separate launch authorization.
External launch/accounting and an explicitly labeled duplicate-primitive adapter.
Frozen RPent files and ordinary planning policy stay unchanged.
"""
import os,sys,json,time,hashlib,subprocess,socket,threading,shutil,urllib.request,urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'.runtime/harness_original_control_v168'
OUT=Path(os.environ.get('HARNESS_OUTPUT_DIR','/var/tmp/harnessvla_sol61_low_seed0_effective30m_next'))
STATE=Path(os.environ.get('HARNESS_STATE_FILE',str(ROOT/'.runtime/harness_sol61_low_seed0_effective30m_next.json')))
PYTHON=os.environ.get('HARNESS_PYTHON',str(ROOT.parent/'AsyncSketch/.venv-rpent/bin/python'))
GPU=os.environ.get('HARNESS_GPU_UUID','GPU-963d3440-dfe0-be7d-7fc2-69cc6d7acf00')
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sketchik.usage_accounting import credit_snapshot
from network_recovery import make_server
from empty_memory import configure_empty_local_memory,selected_tasks
CREDITS=ROOT/'.runtime/flowith_probe/private/data/config.json'
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def read(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def write(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def port():
 with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def health(endpoint):
 try:
  req=urllib.request.Request(endpoint+'/call',data=b'{"method":"healthz","args":[],"kwargs":{}}',headers={'Content-Type':'application/json'})
  with OPENER.open(req,timeout=2) as r:return json.load(r).get('ok') is True
 except (OSError,ValueError):return False

def hashes():
 return {str(p.relative_to(BASE/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.yaml','.yml','.toml')}

def selected_gpu_index(gpu_rows, gpu_uuid, minimum_free_mib=18000):
 for row in gpu_rows.splitlines():
  fields=[item.strip() for item in row.split(',')]
  if len(fields)>=5 and fields[1]==gpu_uuid:
   if int(fields[3].split()[0])<minimum_free_mib:
    raise RuntimeError('Selected GPU has insufficient free memory; no model service started')
   return fields[0]
 raise RuntimeError('Selected GPU UUID not found; refusing fallback to another GPU')

def configure_primitive_guard(command, environment, run, profile):
 if profile=='off':return command
 if profile!='reject-duplicate-primitives-v1':raise ValueError('Unknown primitive guard profile')
 if command[1:3]!=['-m','rpent.cli.main']:
  raise ValueError('Primitive guard requires the prepared native module entry')
 environment['HARNESS_GUARD_AUDIT']=str(Path(run)/'primitive_guard.jsonl')
 return [command[0],str(ROOT/'tools/harness_eval/guarded_native.py'),*command[3:]]

def guard_hashes(profile):
 if profile=='off':return {}
 return {name:hashlib.sha256((ROOT/'tools/harness_eval'/name).read_bytes()).hexdigest()
         for name in ('primitive_guard.py','guarded_native.py')}

def main():
 tasks=selected_tasks(os.environ.get('HARNESS_TASKS','0'))
 guard_profile=os.environ.get('HARNESS_PRIMITIVE_GUARD','reject-duplicate-primitives-v1')
 if guard_profile not in ('off','reject-duplicate-primitives-v1'):raise ValueError('Unknown primitive guard profile')
 guard_source_hash=guard_hashes(guard_profile)
 if OUT.exists() or STATE.exists():raise RuntimeError('Refusing overwrite of previous run')
 OUT.mkdir(mode=0o700);(OUT/'private/data').mkdir(parents=True,mode=0o700)
 cfg=read(BASE/'prepared_command.json');source_hash=hashes();write(OUT/'source_hashes.json',source_hash)
 env=os.environ.copy();env.update(cfg['environment']);env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',MUJOCO_EGL_DEVICE_ID='0',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost',HF_HUB_OFFLINE='1')
 state={'status':'starting_services','pid':os.getpid(),'started_at_unix':time.time(),'source_commit':cfg['source_commit'],'scope':'New separately authorized full-initial-state experiment only; 1800s effective budget excludes confirmed network failures and retry backoff. No old process continuation or action replay. Empty per-task local memory as explicitly requested, matching the earlier DS setup; Sol low/65536 retained.','model':'gpt-6.1-sol','reasoning_effort':'low','max_tokens':65536,'planner_timeout_s':1800,'native_max_turns':100,'authorized_tasks':tasks,'stop_after_selected_tasks':True,'memory_profile':'local','evaluation_label':'native-local-prompt-empty-external-memory','strict_zero_shot':False,'built_in_task_priors_retained':True,'tool_protocol':'flowith_xml_bridge','memory_initial_files':'zero-byte MEMORY.md only; empty global/task-family/task-specific','native_wall_watchdog_s':1000000000,'budget_policy':'1800s effective at model-response boundary; confirmed failed network attempts and retry backoff excluded; pending/unknown model wait counted','gpu_uuid':GPU,'services':{},'jobs':[],'output':str(OUT)};write(STATE,state)
 state.update(primitive_guard_profile=guard_profile,primitive_guard_source_hashes=guard_source_hash)
 if guard_profile!='off':state['evaluation_label']+='-duplicate-primitive-rejection-adapter'
 processes=[];child=None;servers=[]
 def save():write(STATE,state)
 try:
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
  write(OUT/'gpu_before.json',{'gpus':gpu.splitlines(),'processes':apps.splitlines()})
  gpu_index=selected_gpu_index(gpu,GPU)
  env['MUJOCO_EGL_DEVICE_ID']=gpu_index
  state['gpu_physical_index']=gpu_index;save()
  for name,rel,extra in [('vla','rpent/robots/components/pi05_vla_server.py',['--embodiment','libero']),('sam3','rpent/robots/components/sam3_server.py',[])]:
   server_port=port();command=[PYTHON,str(BASE/'source'/rel),'--transport','http','--host','127.0.0.1','--port',str(server_port),'--cuda-device',gpu_index,*extra]
   with (OUT/f'{name}.log').open('xb') as log:p=subprocess.Popen(command,cwd=BASE/'source',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   processes.append(p);state['services'][name]={'pid':p.pid,'endpoint':f'http://127.0.0.1:{server_port}','command':command};save()
  deadline=time.monotonic()+600
  while not all(health(v['endpoint']) for v in state['services'].values()):
   if any(p.poll() is not None for p in processes):raise RuntimeError('Dedicated model service exited before healthy')
   if time.monotonic()>deadline:raise RuntimeError('Model service startup timeout')
   time.sleep(2)
  # Separate large-body gateway; original Mac gateway is never changed.
  gateway_port=port();gateway_config=read(CREDITS);gateway_config.update(server_host='127.0.0.1',server_port=gateway_port)
  conf=OUT/'private/data/config.json';conf.write_text(json.dumps(gateway_config));conf.chmod(0o600)
  gateway_env=env.copy();gateway_env['FLOWITH2API_DATA_DIR']=str(OUT/'private/data');gateway_env['FLOWITH2API_STREAM_AUDIT_DIR']=str(OUT/'private/stream_audit')
  binary=BASE/'flowith2api_large_body'
  with (OUT/'gateway.log').open('xb') as log:g=subprocess.Popen([str(binary),'-host','127.0.0.1','-port',str(gateway_port)],cwd=OUT/'private',env=gateway_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  processes.append(g);state['gateway']={'pid':g.pid,'port':gateway_port,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()};save()
  deadline=time.monotonic()+15
  while True:
   if g.poll() is not None:raise RuntimeError('Dedicated gateway exited')
   try:
    with OPENER.open(f'http://127.0.0.1:{gateway_port}/health',timeout=1) as response:assert json.load(response)['status']=='ok'
    break
   except (OSError,ValueError):
    if time.monotonic()>deadline:raise RuntimeError('Gateway health timeout')
    time.sleep(.2)
  private={}
  for line in (ROOT/'.runtime/flowith_probe/private/gpt.env').read_text().splitlines():
   if '=' in line and not line.lstrip().startswith('#'):
    k,v=line.split('=',1);private[k.strip().removeprefix('export ')]=v.strip().strip('\"\'')
  state.update(status='running',balance_before=credit_snapshot(CREDITS));save()
  results=[]
  for task in tasks:
   if hashes()!=source_hash:raise RuntimeError('Frozen native source changed')
   if guard_hashes(guard_profile)!=guard_source_hash:raise RuntimeError('Frozen primitive guard changed')
   if any(p.poll() is not None for p in processes):raise RuntimeError('Dedicated service exited')
   if shutil.disk_usage(OUT).free<10*1024**3:raise RuntimeError('Less than 10GiB on output filesystem')
   run=OUT/f'task_{task:02d}';run.mkdir();(run/'api').mkdir()
   ledger_lock=threading.Lock();attempts=[]
   class Ledger(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args):pass
    def do_POST(self):
     body=self.rfile.read(int(self.headers.get('Content-Length','0')))
     try:
      data=json.loads(body)
      assert data['model']=='gpt-6.1-sol' and data.get('reasoning_effort')=='low'
      assert data.get('max_completion_tokens',data.get('max_tokens'))==65536
     except (ValueError,KeyError,AssertionError):
      self.send_error(400,'Configured model/reasoning/token-budget mismatch');return
     balance_started=time.monotonic();before=credit_snapshot(CREDITS);balance_before_s=time.monotonic()-balance_started
     start=time.time();stream_started=time.monotonic();status=None;failure=None
     timing={'headers_s':None,'first_response_byte_s':None,'first_generation_s':None,'stream_complete_s':None};pending=b'';event_count=0
     try:
      request=urllib.request.Request(f'http://127.0.0.1:{gateway_port}'+self.path,data=body,headers={k:v for k,v in self.headers.items() if k.lower() not in ['host','connection','content-length']},method='POST')
      try:response=OPENER.open(request,timeout=360)
      except urllib.error.HTTPError as e:response=e
      timing['headers_s']=time.monotonic()-stream_started
      status=response.status;self.send_response(status);self.send_header('Content-Type',response.headers.get('Content-Type','text/event-stream'));self.send_header('Transfer-Encoding','chunked');self.end_headers()
      with response:
       while True:
        block=response.read1(65536)
        if not block:break
        elapsed=time.monotonic()-stream_started
        if timing['first_response_byte_s'] is None:timing['first_response_byte_s']=elapsed
        pending+=block
        while b'\n' in pending:
         line,pending=pending.split(b'\n',1)
         if not line.startswith(b'data:'):continue
         raw=line[5:].strip()
         if raw==b'[DONE]':timing['stream_complete_s']=elapsed;continue
         try:event=json.loads(raw)
         except (ValueError,UnicodeError):continue
         event_count+=1
         if timing['first_generation_s'] is None and isinstance(event,dict) and any(any(c.get('delta',{}).get(k) for k in ('content','reasoning_content','tool_calls')) for c in event.get('choices',[]) if isinstance(c,dict)):
          timing['first_generation_s']=elapsed
          with (run/'usage_stream_progress.jsonl').open('a') as progress:progress.write(json.dumps({'task':task,'started_at_unix':start,'request_sha256':hashlib.sha256(body).hexdigest(),'first_generation_s':elapsed})+'\n')
        self.wfile.write(f'{len(block):X}\r\n'.encode()+block+b'\r\n');self.wfile.flush()
      self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
     except Exception as exc:
      failure=type(exc).__name__;self.close_connection=True
     finally:
      stream_wall_s=time.monotonic()-stream_started
      balance_started=time.monotonic();after=credit_snapshot(CREDITS);balance_after_s=time.monotonic()-balance_started
      record={'task':task,'started_at_unix':start,'ended_at_unix':time.time(),'model':data.get('model'),'reasoning_effort':data.get('reasoning_effort'),'request_sha256':hashlib.sha256(body).hexdigest(),'request_bytes':len(body),'http_status':status,'failure_type':failure,'balance_before':before,'balance_after':after,'balance_query_before_s':balance_before_s,'balance_query_after_s':balance_after_s,'upstream_stream_wall_s':stream_wall_s,'stream_timing':timing,'sse_event_count':event_count,'scope':'Account-wide balance snapshots, not exclusive per-request billing. First-generation timing is gateway-visible content/reasoning/tool delta; not the provider internal first reasoning token.'}
      with ledger_lock:
       attempts.append(record)
       with (run/'usage_attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   ledger=ThreadingHTTPServer(('127.0.0.1',0),Ledger);ledger.daemon_threads=False;servers.append(ledger);threading.Thread(target=ledger.serve_forever,daemon=True).start()
   proxy=make_server(f'http://127.0.0.1:{ledger.server_port}/v1',run/'api',timeout=360,effective_budget_s=1800)
   proxy.daemon_threads=False;servers.append(proxy);threading.Thread(target=proxy.serve_forever,daemon=True).start()
   runenv=env.copy();runenv.update(OPENAI_API_KEY=private['SKETCHIK_API_KEY'],OPENAI_BASE_URL=f'http://127.0.0.1:{proxy.server_port}')
   config_dir=run/'libero_config';config_dir.mkdir();shutil.copy2(ROOT/'.runtime/libero_pro/config.yaml',config_dir/'config.yaml');runenv['LIBERO_CONFIG_PATH']=str(config_dir)
   command=list(cfg['command']);command[command.index('--task')+1]=str(task);command[command.index('--cuda-device')+1]=gpu_index;command[-1]=str(run)
   command+=['--planner-timeout-s','1000000000']
   command=configure_empty_local_memory(command,run)
   for name,service in state['services'].items():command.extend([f'--{name}-endpoint',service['endpoint']])
   command=configure_primitive_guard(command,runenv,run,guard_profile)
   job={'task':task,'seed':0,'init':0,'status':'running','run':str(run),'started_at_unix':time.time(),'command':command,'balance_before':credit_snapshot(CREDITS)}
   with (run/'console.log').open('xb') as log:child=subprocess.Popen(command,cwd=cfg['cwd'],env=runenv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   job['pid']=child.pid;state['jobs'].append(job);save();print(json.dumps({'event':'task_started','task':task,'pid':child.pid}),flush=True)
   rc=child.wait();job.update(returncode=rc,ended_at_unix=time.time(),balance_after=credit_snapshot(CREDITS))
   # All API threads must be quiescent before the next task closes over a new run.
   proxy.shutdown();ledger.shutdown();proxy.server_close();ledger.server_close();servers.remove(proxy);servers.remove(ledger)
   transcripts=list(run.glob('transcript*.json'));transcript=read(transcripts[0]) if transcripts else {}
   api=[read(p) for p in sorted((run/'api').glob('*/summary.json'))];excluded=sum(x.get('excluded_network_s',0) for x in api)
   terminated=any(bool(x.get('terminated')) for x in read(run/'states.json').get('steps',[]));error=transcript.get('error');budget_timeout=isinstance(error,str) and 'planner timed out after 1800s' in error
   effective_timeout=any(x.get('state')=='effective_budget_exhausted' for x in api)
   bad=not transcript or (rc!=0 and not budget_timeout and not effective_timeout) or (bool(error) and not budget_timeout and not effective_timeout) or hashes()!=source_hash or guard_hashes(guard_profile)!=guard_source_hash
   job.update(status='stopped_for_inspection' if bad else 'finished',official_success=transcript.get('environment_success'),native_error=error)
   row={'task':task,'seed':0,'init':0,'official_success':transcript.get('environment_success') if transcript.get('environment_success') is not None else (terminated if not bad or terminated else None),'raw_environment_success':transcript.get('environment_success'),'native_termination_success':terminated,'native_finish':transcript.get('finish'),'native_error':error,'budget_exhausted':budget_timeout or effective_timeout,'native_elapsed_s':transcript.get('elapsed_s'),'wall_s':job['ended_at_unix']-job['started_at_unix'],'excluded_network_s':excluded,'effective_wall_s':max(0,job['ended_at_unix']-job['started_at_unix']-excluded),'planner_requests':len(api),'upstream_attempts':len(attempts),'balance_before':job['balance_before'],'balance_after':job['balance_after'],'program_or_api_error':bad,'run':str(run)}
   row.update(primitive_guard_profile=guard_profile,primitive_guard_source_hashes=guard_source_hash)
   results.append(row);write(run/'control_summary.json',row);write(OUT/'results.json',{'status':'running' if not bad else 'stopped_for_inspection','tasks':results,'shared_account_not_exclusive_cost':True});save();print(json.dumps({'event':'task_finished','task':task,'official':row['official_success'],'wall_s':row['wall_s'],'error':bool(error)}),flush=True)
   if bad:state['status']='stopped_for_inspection';save();break
  else:state.update(status='finished',ended_at_unix=time.time());save()
  data=read(OUT/'results.json');data['status']=state['status'];write(OUT/'results.json',data)
 except BaseException as exc:
  state.update(status='stopped_for_inspection',error_type=type(exc).__name__,live_process_preserved=child is not None and child.poll() is None);save();raise
 finally:
  # Keep a genuinely live simulation and its services on a supervision failure.
  if child is None or child.poll() is not None:
   for server in servers:server.shutdown()
   for p in reversed(processes):
    if p.poll() is None:p.terminate()
   for p in reversed(processes):
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()
   state.update(dedicated_services_stopped=True,balance_after=credit_snapshot(CREDITS));save()

if __name__=='__main__':
 if '--execute' not in sys.argv:
  print('Prepared only. Requires a separately authorized experiment launch with --execute. Effective budget 1800s; empty per-task local memory; native wall deadline disabled; no services started.')
 else:main()
