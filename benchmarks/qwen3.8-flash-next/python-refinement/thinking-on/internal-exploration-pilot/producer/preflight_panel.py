"""Use the real staged panel and tokenizer; exercise batched sampling and saves on CPU."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from worker import BLOCKS, NATIVE, PANEL, decode, prepare_panel, prepare_rows, ready, save
import torch
from transformers import AutoTokenizer


class OneStepModel:
    def get_input_embeddings(self):
        return torch.nn.Embedding(1,1)
    def generate(self, *, input_ids, attention_mask, logits_processor, stopping_criteria, **kwargs):
        assert input_ids.shape==attention_mask.shape and input_ids.shape[0]==4
        scores=torch.zeros(4,tokenizer.eos_token_id+1)
        tokens=logits_processor(input_ids,scores).argmax(-1)
        stopping_criteria(torch.cat([input_ids,tokens[:,None]],dim=1),scores)


if __name__=='__main__':
    torch.set_num_threads(1)
    _,recipe,modules=ready(NATIVE)
    tokenizer=AutoTokenizer.from_pretrained(recipe['checkpoint'],local_files_only=True)
    print('recipe IDs:',[key for key in recipe if 'benchmark_id' in key],flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        for block in BLOCKS:
            path=Path(tmp)/block
            tasks,states=prepare_panel(block,recipe,modules,path)
            persist=lambda s:save(path/'tasks'/(s['task']+'.json'),s)
            rows=prepare_rows(states,tasks,tokenizer,persist)
            assert {s['task'] for s,_ in rows}==set(PANEL[block[1]])
            decode(OneStepModel(),tokenizer,rows,persist,lambda:False,chunk=1)
            _,restored=prepare_panel(block,recipe,modules,path)
            assert all(s['total']==1 and len(s['prefix'])==1 and s['rng'] for s in restored.values())
            print('BATCH_PREFLIGHT_OK',block,[(s['task'],len(ids)) for s,ids in rows],flush=True)
