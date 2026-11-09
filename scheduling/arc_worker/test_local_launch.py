"""Exercise the documented worker command on a tiny local checkpoint without mocking the loader."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


class LocalLaunchTest(unittest.TestCase):
    def test_native_loader_and_scheduler_wrapper(self):
        example = Path(__file__).resolve().parent
        wrapper = example.parent / 'slurm' / 'worker.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / 'model'
            model_path.mkdir()
            tokenizer = Tokenizer(WordLevel({'<pad>': 0, '<unk>': 1, '<eos>': 2, 'x': 3}, unk_token='<unk>'))
            tokenizer.pre_tokenizer = Whitespace()
            hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>', eos_token='<eos>',
                                         pad_token='<pad>', chat_template='{% for m in messages %}{{ m.content }}{% endfor %}')
            hf.save_pretrained(model_path)
            config = Qwen3Config(vocab_size=4, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=8, intermediate_size=32, max_position_embeddings=1024,
                                 eos_token_id=2, pad_token_id=0)
            Qwen3ForCausalLM(config).save_pretrained(model_path)
            campaign = json.loads((example / 'campaign.example.json').read_text())
            campaign['worker_command'] = [sys.executable, str(example / 'worker.py')]
            campaign['arc'].update(model=str(model_path), model_revision='0' * 40, context=1024, max_new_tokens=3,
                                   bands=[1024], checkpoint_tokens=2, tasks_file=str(example / 'tasks.example.jsonl'))
            cfg = root / 'campaign.json'
            cfg.write_text(json.dumps(campaign))
            cmd = [sys.executable, str(wrapper), str(cfg)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
            first = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn('ARC_BATCH 2', first.stdout)
            files = list((root / '.campaigns').glob('*/items/*.json'))
            saved = {path: path.read_bytes() for path in files}
            self.assertEqual(set(path.stem for path in files), set(campaign['items']))
            self.assertTrue(all(json.loads(raw)['status'] == 'finished' for raw in saved.values()))
            second = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertNotIn('ARC_BATCH', second.stdout)
            self.assertEqual(saved, {path: path.read_bytes() for path in files})
