import sys,time,json,threading,unittest,tempfile,urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/harness_eval'))
from network_recovery import make_server,inspect_sse
COMPLETE=b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
NETWORK=b'data: {"error":"unexpected EOF"}\n\n'
AUTH=b'data: {"error":"authentication invalid"}\n\n'
class Tests(unittest.TestCase):
 def run_case(self,sequence,budget=2,clock=time.monotonic,before=None,adapt_schema=True,payload=b'{"model":"offline-test","messages":[]}'):
  calls=[]
  class H(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    calls.append(self.rfile.read(int(self.headers['Content-Length'])));delay,body=sequence[min(len(calls)-1,len(sequence)-1)];time.sleep(delay);self.send_response(200);self.end_headers();self.wfile.write(body)
  upstream=ThreadingHTTPServer(('127.0.0.1',0),H)
  threading.Thread(target=upstream.serve_forever,daemon=True).start()
  with tempfile.TemporaryDirectory() as td:
   proxy=make_server(f'http://127.0.0.1:{upstream.server_port}',td,retry_delay=.02,heartbeat=.01,effective_budget_s=budget,budget_clock=clock,adapt_schema=adapt_schema)
   threading.Thread(target=proxy.serve_forever,daemon=True).start()
   try:
    if before:before()
    req=urllib.request.Request(f'http://127.0.0.1:{proxy.server_port}/chat/completions',data=payload,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=3) as response:body=response.read()
    p=Path(td)/'0001/summary.json'
    for _ in range(100):
     if p.exists():break
     time.sleep(.005)
    summary=json.loads(p.read_text());return calls,body,summary
   finally:proxy.shutdown();proxy.server_close();upstream.shutdown();upstream.server_close()
 def test_failed_network_does_not_exhaust_effective_budget(self):
  calls,body,s=self.run_case([(.55,NETWORK),(.01,COMPLETE)],budget=.4)
  self.assertEqual(len(calls),2);self.assertEqual(calls[0],calls[1]);self.assertEqual(s['state'],'complete');self.assertGreater(s['total_s'],.4);self.assertGreater(s['excluded_network_s'],.55);self.assertLess(s['effective_s'],.4);self.assertIn(b'"ok"',body)
 def test_completed_model_time_is_counted_and_not_dispatched_when_expired(self):
  calls,body,s=self.run_case([(.12,COMPLETE)],budget=.07)
  self.assertEqual(s['state'],'effective_budget_exhausted');self.assertEqual(s['excluded_network_s'],0);self.assertNotIn(b'"ok"',body);self.assertIn(b'0.07s',body)
 def test_authentication_is_not_retried(self):
  calls,body,s=self.run_case([(0,AUTH)])
  self.assertEqual(len(calls),1);self.assertEqual(s['state'],'terminal_error');self.assertEqual(s['excluded_network_s'],0)
 def test_already_expired_budget_sends_no_upstream_request(self):
  tick=[0.];calls,body,s=self.run_case([(0,COMPLETE)],budget=2,clock=lambda:tick[0],before=lambda:tick.__setitem__(0,3.))
  self.assertEqual(calls,[]);self.assertEqual(s['state'],'effective_budget_exhausted')
 def test_native_endpoint_preserves_request_bytes_including_union_schema(self):
  payload=b'{ "model":"offline-test", "messages":[], "tools":[{"type":"function","function":{"name":"echo","parameters":{"type":"object","properties":{"value":{"type":["string","null"]}}}}}] }'
  calls,body,s=self.run_case([(0,COMPLETE)],adapt_schema=False,payload=payload)
  self.assertEqual(calls,[payload]);self.assertEqual(s['state'],'complete')
if __name__=='__main__':unittest.main()
