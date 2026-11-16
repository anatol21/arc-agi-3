"""Replay a real saved probe through the deployed tokenizer and CPU isolation."""
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'arc-flash-program-agent-v4')]
from decoding import final_text
from protocol_pilot import run_probe
from sandbox import execute_case
from transformers import AutoTokenizer

work=HERE.parent
prior=work/'runs/arc-python-exploration-pilot-v1/AX/control/probe.json'
state=json.loads(prior.read_text())
tok=AutoTokenizer.from_pretrained(work/'runs/arc-flash-v6/checkpoint-660',local_files_only=True)
text=final_text(tok,state)
assert text.strip()=='```python\nprint(test[0]["input"])\n```'
for grid in ([[8,1,3,0,5,2]], [[4,7,0,9,8,6]]):
    observed=run_probe(text,dict(train=[],test=[dict(input=grid)]),execute_case,tok)
    assert json.loads(observed)==grid,repr(observed)
print('SAVED_PROBE_REPLAY_PASSED',repr(text),flush=True)
