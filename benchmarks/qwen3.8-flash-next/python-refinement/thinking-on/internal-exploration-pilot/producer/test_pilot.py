import copy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent/'artifacts/cluster/arc-flash-python-nothink-v2'))
import controller
from protocol_pilot import *
import launch

class Words:
    def encode(self,text,**_): return list(text.encode())
    def decode(self,ids,**_): return bytes(ids).decode()

class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.visible={'train':[{'input':[[1]],'output':[[1]]}],'test':[{'input':[[4]]}]}
        self.s=new_state('sample','B','test')
    def test_probe_and_revision(self):
        calls=[]
        def execute(code,data):
            calls.append(data)
            return dict(executable=True,stdout=('x'*300 if isinstance(data,dict) else ''),output=[[0]])
        probe='```probe\nprint(train)\n```'
        finish_turn(self.s,probe,'eos',controller,self.visible,execute,Words())
        s=json.loads(json.dumps(self.s))
        self.assertEqual(s['probes'][0]['observation'],'x'*256)
        self.assertIn('x'*256,messages(controller,self.visible,s)[-1]['content'])
        code='```python\ndef solve(grid): return [[0]]\n```'
        finish_turn(s,code,'eos',controller,self.visible,execute,Words())
        self.assertNotIn('x'*256,str(messages(controller,self.visible,s)))
        self.assertEqual(calls,[self.visible,[[1]]])
    def test_python_fence_dispatch(self):
        text = '\n'.join(['```python', 'print(test[0]["input"])', '```'])
        solver = '\n'.join(['```python','def solve(grid):','    print(grid)','    return grid','```'])
        self.assertEqual(probe_body(text), 'print(test[0]["input"])')
        self.assertIsNone(probe_body(solver))
        self.assertIsNone(probe_body(text + solver))
        calls = []
        def execute(code, data):
            calls.append(data)
            return dict(executable=True, stdout='[[4]]', output=data)
        finish_turn(self.s,text,'eos',controller,self.visible,execute,Words())
        self.assertEqual(calls,[self.visible])
        self.assertEqual(self.s['probes'][0]['observation'],'[[4]]')
        self.assertEqual(self.s['attempts'],[])
        finish_turn(self.s,solver,'eos',controller,self.visible,execute,Words())
        self.assertTrue(self.s['done'])
        self.assertEqual(len(self.s['attempts']),1)

    def test_limits_and_selection(self):
        for _ in range(2):
            finish_turn(self.s,'```probe','probe_limit',controller,self.visible,None,Words())
        finish_turn(self.s,'```probe','probe_limit',controller,self.visible,None,Words())
        self.assertEqual(len(self.s['attempts']),1)
        self.assertEqual(len(self.s['probes']),2)
        self.assertEqual(allowance(self.s,[0]*18431),1)
        self.assertEqual(len(set(PANEL['X']+PANEL['Y'])),8)
    def test_budget_and_duplicate_submission(self):
        with TemporaryDirectory() as d, patch.object(launch,'OUT',Path(d)), patch.object(launch.subprocess,'check_output',return_value='1234\n') as submit:
            for block in RETRY_MINUTES:
                launch.launch()
                launch.launch()
                save(Path(d)/block/'allocation.json',dict(phase='ended',block=block,job='1234'))
                def output(cmd,**kwargs):
                    return 'COMPLETED|0:0\n' if cmd[0]=='sacct' else '1234\n'
                submit.side_effect=output
            launch.launch()
            self.assertEqual(sum(c.args[0][0]=='sbatch' and c.args[0][-1] in BLOCKS for c in submit.call_args_list),3)
            gpu_calls=[c.args[0] for c in submit.call_args_list if c.args[0][0]=="sbatch" and c.args[0][-1] in BLOCKS]
            self.assertEqual([(c[-1],c[2]) for c in gpu_calls], [("BX","--time=00:26:00"),("BY","--time=00:22:00"),("AY","--time=00:22:00")])
            self.assertLess((2935+60*sum(RETRY_MINUTES.values()))*4,8*3600)
    def test_timeout_advances_once_only_after_matching_end(self):
        with TemporaryDirectory() as d, patch.object(launch,'OUT',Path(d)):
            save(Path(d)/'BX-submission.json',dict(id='1234'))
            end_file=Path(d)/'BX'/'allocation.json'
            def output(cmd,**kwargs):
                if cmd[0]=='sacct':
                    return 'TIMEOUT|0:15'
                return '2345' if '--dependency=afterany:2345' not in cmd else '3456'
            with patch.object(launch.subprocess,'check_output',side_effect=output) as calls:
                for phase,job in [('panel','1234'),('ended','9999')]:
                    save(end_file,dict(phase=phase,job=job,block='BX'))
                    launch.launch()
                    self.assertFalse((Path(d)/'BY-submission.json').exists())
                save(end_file,dict(phase='ended',job='1234',block='BX',finished=[]))
                launch.launch()
                launch.launch()
                submissions=[c.args[0] for c in calls.call_args_list if c.args[0][0]=='sbatch']
                self.assertEqual(len(submissions),2)
                self.assertEqual(submissions[0][-1],'BY')
                self.assertIn('--dependency=afterany:2345',submissions[1])

if __name__=='__main__':
    unittest.main()

class DecodeTest(unittest.TestCase):
    def test_replay_identical_and_switch(self):
        import torch
        from decoding import decode
        class Tok:
            eos_token_id=0
            pad_token_id=0
            def encode(self,*args,**kwargs): return [2]
            def decode(self,*args,**kwargs): return 'code'
        class Toy:
            def get_input_embeddings(self): return torch.nn.Embedding(5,4)
            def generate(self,input_ids,logits_processor,stopping_criteria,generation_config,**kwargs):
                for _ in range(generation_config.max_new_tokens):
                    scores=torch.tensor([[-100.,1.,-100.,1.,1.]])
                    scores=logits_processor(input_ids,scores)
                    input_ids=torch.cat([input_ids,scores.argmax(-1,keepdim=True)],dim=1)
                    if stopping_criteria(input_ids,scores).all(): break
        def generate(s,n):
            decode(Toy(),Tok(),[(s,[1])],lambda _:None,lambda:False,chunk=n,reasoning_cap=3)
        s=new_state('abc','A','test')
        full=copy.deepcopy(s)
        generate(full,10)
        generate(s,4)
        s=json.loads(json.dumps(s))
        generate(s,6)
        self.assertEqual(full,s)
        self.assertEqual(s['prefix'][3],2)
        self.assertEqual(s['reasoning'],3)
