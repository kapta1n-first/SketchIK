"""External transport only: freeze requests and commit complete SSE responses."""
import hashlib
from schema_adapter import adapt_request
import json
import socket
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def inspect_sse(status, body):
    if status is None or 300 <= status < 500:
        return 'terminal_http'
    events=[]
    try:
        for line in body.decode('utf-8').splitlines():
            if line.startswith('data:'):
                value=line[5:].strip()
                events.append(value if value=='[DONE]' else json.loads(value))
    except (ValueError, UnicodeError):
        return 'terminal_format'
    if not events:
        return 'terminal_format'
    if any(not isinstance(e,dict) and e!='[DONE]' for e in events):
        return 'terminal_format'
    errors=[e['error'] for e in events if isinstance(e,dict) and 'error' in e]
    if errors:
        text=json.dumps(errors).lower()
        if any(x in text for x in ('auth','quota','credit','rate limit','invalid','parameter','unsupported')):
            return 'terminal_error'
        if ('flowith stream ended inside an unclosed think block; no completed answer' in text or any(x in text for x in ('eof','timeout','timed out','connection','broken pipe','reset by peer'))):
            return 'network'
        return 'terminal_error'
    done=bool(events) and events[-1]=='[DONE]'
    finishes=[c.get('finish_reason') for e in events if isinstance(e,dict)
              for c in e.get('choices',[]) if c.get('finish_reason') is not None]
    if done and finishes and all(x in ('stop','tool_calls','function_call','length','content_filter') for x in finishes):
        return 'complete'  # length remains a complete original response, never retried.
    if not done and status == 200 and any(isinstance(e,dict) and isinstance(e.get('choices'),list) and e['choices'] for e in events):
        return 'network'
    return 'terminal_format'


def make_server(upstream, archive_root, retry_delay=None, heartbeat=.5, timeout=1200., max_attempts=128, network_budget_s=7200., effective_budget_s=1200., budget_clock=time.monotonic, adapt_schema=True):
    parsed=urllib.parse.urlsplit(upstream)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password:
        raise ValueError('Upstream must be an explicit local HTTP gateway')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    root=Path(archive_root);root.mkdir(parents=True,exist_ok=True)
    lock=threading.Lock();counter=0
    budget_started=budget_clock(); confirmed_network_total=0.
    budget_audit=root/'effective_budget.json'
    def budget_state():
        elapsed=budget_clock()-budget_started
        return {'wall_since_control_start_s':elapsed,'confirmed_failed_network_and_backoff_s':confirmed_network_total,'effective_s':elapsed-confirmed_network_total,'effective_budget_s':effective_budget_s,'budget_origin':'proxy creation before original process launch; conservatively includes startup','unknown_wait_deducted':False}
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def do_POST(self):
            nonlocal counter,confirmed_network_total
            with lock:counter+=1;index=counter
            directory=root/f'{index:04d}';directory.mkdir()
            original_body=self.rfile.read(int(self.headers.get('Content-Length','0')))
            body=(json.dumps(adapt_request(json.loads(original_body)),ensure_ascii=False,separators=(',',':')).encode()
                  if adapt_schema else original_body)
            (directory/'original_client_request.json').write_bytes(original_body)
            (directory/'schema_adapter.json').write_text(json.dumps({'change':'schema union type array to equivalent anyOf only' if adapt_schema else 'none; native request bytes preserved','original_sha256':hashlib.sha256(original_body).hexdigest(),'forwarded_sha256':hashlib.sha256(body).hexdigest(),'arguments_modified':False},indent=2))
            digest=hashlib.sha256(body).hexdigest()
            (directory/'request.json').write_bytes(body)
            original_headers={k:v for k,v in self.headers.items()
                              if k.lower() not in ('host','connection','content-length')}
            started=time.monotonic();wall=time.time();failed=0.;attempts=[];disconnected=False;exhausted=False
            deadline=started+network_budget_s
            self.send_response(200)
            self.send_header('Content-Type','text/event-stream')
            self.send_header('Transfer-Encoding','chunked');self.end_headers()
            def send(data):
                self.wfile.write(f'{len(data):X}\r\n'.encode()+data+b'\r\n');self.wfile.flush()
            def summary(state):
                elapsed=time.monotonic()-started
                snapshot=json.dumps({
                    'state':state,'request_sha256':digest,'request_bytes':len(body),
                    'started_wall':wall,'total_s':elapsed,'excluded_network_s':failed,
                    'effective_s':elapsed-failed,'attempts':attempts,
                    'unknown_cancelled_wait_s':sum(a.get('elapsed_s',0) for a in attempts if a.get('classification')=='client_cancelled_transport_wait'),
                    'limits':{'max_upstream_attempts':max_attempts,'transport_wait_budget_s':network_budget_s,'backoff_s':[0,5,15,30]},
                    'upstream_proxy':'explicit_none','redirects':'disabled',
                    'external_transport_recovery':True,'response_buffered':True,'effective_budget':budget_state()},indent=2)
                temporary=directory/'summary.json.tmp'
                temporary.write_text(snapshot)
                temporary.replace(directory/'summary.json')
            def wait_alive(event, period):
                nonlocal disconnected,exhausted
                while not event.wait(period):
                    if time.monotonic()>=deadline:
                        exhausted=True;return False
                    try:send(b': transport waiting\n\n')
                    except (BrokenPipeError,ConnectionResetError,OSError):
                        disconnected=True;return False
                return True
            def budget_error():
                try:
                    send(b'data: {"error":{"type":"infrastructure_budget_exhausted","message":"External network recovery limit reached; no action response committed"}}\n\n')
                    self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError,OSError):pass
                summary('infrastructure_budget_exhausted')
            if budget_state()['effective_s'] >= effective_budget_s:
                send(b'data: '+json.dumps({'error':{'type':'control_effective_budget_exhausted','message':f'Effective task budget {effective_budget_s:g}s exhausted before request dispatch'}}).encode()+b'\n\n')
                self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
                summary('effective_budget_exhausted');return
            number=0
            while not disconnected:
                number+=1;begin=time.monotonic();begin_wall=time.time();result={};done=threading.Event()
                target=directory/f'attempt_{number:04d}'
                def fetch():
                    status=None;data=b'';err=None
                    target.with_suffix('.response').touch()
                    try:
                        request=urllib.request.Request(upstream.rstrip('/')+self.path,
                            data=body,headers=original_headers,method='POST')
                        try:response=opener.open(request,timeout=min(timeout,max(.01,deadline-time.monotonic())))
                        except urllib.error.HTTPError as exc:response=exc
                        status=response.status
                        with response, target.with_suffix('.response').open('wb') as output:
                            while True:
                                block=response.read1(65536)
                                if not block:break
                                output.write(block);output.flush();data+=block
                    except (urllib.error.URLError,TimeoutError,ConnectionError,OSError,http_error) as exc:
                        text=str(exc).lower()
                        err=type(exc).__name__
                        result['network_exception']=isinstance(exc,(TimeoutError,ConnectionError,http_error)) or any(x in text for x in ('timed out','timeout','eof','connection refused','connection reset','broken pipe','network is unreachable'))
                    finally:
                        result.update(status=status,body=data,error=err);done.set()
                # A cancelled client cannot cause another upstream request. The current
                # daemon fetch may finish archiving; its content is never delivered.
                threading.Thread(target=fetch,daemon=True).start()
                if not wait_alive(done,heartbeat):
                    cancelled=time.monotonic()-begin  # Pending cancelled request is unknown, not confirmed network failure.
                    attempts.append({'attempt':number,'classification':'client_cancelled_transport_wait',
                        'started_wall':begin_wall,'ended_wall':time.time(),'elapsed_s':cancelled,
                        'request_sha256':digest})
                    if exhausted:budget_error()
                    else:summary('client_disconnected')
                    return
                elapsed=time.monotonic()-begin
                kind=('network' if result.get('network_exception') else 'terminal_transport') if result['error'] else inspect_sse(result['status'],result['body'])
                record={'attempt':number,'elapsed_s':elapsed,'status':result['status'],
                        'started_wall':begin_wall,'ended_wall':time.time(),
                        'classification':kind,'error_type':result['error'],
                        'request_sha256':digest,'body_bytes':len(result['body'])}
                attempts.append(record);target.with_suffix('.json').write_text(json.dumps(record,indent=2))
                if kind=='network':
                    failed+=elapsed
                    confirmed_network_total+=elapsed
                    if number>=max_attempts or time.monotonic()>=deadline:
                        budget_error();return
                    delay_start=time.monotonic();delay=threading.Event()
                    backoff=retry_delay if retry_delay is not None else (0,5,15,30)[min(number-1,3)]
                    record['backoff_requested_s']=backoff
                    threading.Timer(backoff,delay.set).start()
                    if not wait_alive(delay,heartbeat):
                        record['backoff_actual_s']=time.monotonic()-delay_start
                        failed+=record['backoff_actual_s']
                        confirmed_network_total+=record['backoff_actual_s']
                        target.with_suffix('.json').write_text(json.dumps(record,indent=2))
                        if exhausted:budget_error()
                        else:summary('client_disconnected')
                        return
                    record['backoff_actual_s']=time.monotonic()-delay_start
                    failed+=record['backoff_actual_s']
                    confirmed_network_total+=record['backoff_actual_s']
                    target.with_suffix('.json').write_text(json.dumps(record,indent=2))
                    summary('retrying');continue
                audit=budget_state()
                audit.update(request_index=index,response_classification=kind)
                budget_audit.write_text(json.dumps(audit,indent=2))
                if kind=='complete' and audit['effective_s']>effective_budget_s:
                    try:
                        send(b'data: '+json.dumps({'error':{'type':'control_effective_budget_exhausted','message':f'Effective task budget {effective_budget_s:g}s exhausted; complete response archived but not committed'}}).encode()+b'\n\n')
                        self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
                    except (BrokenPipeError,ConnectionResetError,OSError):pass
                    summary('effective_budget_exhausted');return
                payload=result['body']
                if kind.startswith('terminal'):
                    payload=b'data: '+json.dumps({'error':{'type':kind,'upstream_status':result['status']}}).encode()+b'\n\n'
                try:
                    send(payload);self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError,OSError):
                    summary('client_disconnected');return
                summary(kind);return
    # HTTPException includes incomplete chunk framing / unexpected EOF.
    from http.client import HTTPException as http_error
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.effective_budget_state=budget_state
    return server
