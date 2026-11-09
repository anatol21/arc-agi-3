"""CPU staging: same public tasks and prompt text, distinct aggressive batching variant."""
import json
import os
from pathlib import Path
from bridge import NAME, atomic, dependencies, digest, output, sha
import policy
import records


def stage(folder):
    parent, native, prompts, modules, _ = dependencies(folder, require_ready=False)
    modules['common'].verify_training_package(folder.parent, parent)
    modules['common'].verify_loader_sources(parent)
    with modules['common'].output_lock(output(folder)):
        if (folder / 'READY.json').exists():
            dependencies(folder)
            print('UTILIZATION PACKAGE ALREADY STAGED; verified unchanged', flush=True)
            return
        if any((output(folder) / 'batches').glob('*.json')):
            raise RuntimeError('Existing output requires its original readiness record')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(native['checkpoint'], local_files_only=True)
        if (tokenizer.eos_token_id != native['eos_token_id'] or
                tokenizer.pad_token_id != native['pad_token_id'] or
                tokenizer.pad_token_id == tokenizer.eos_token_id):
            raise RuntimeError('Tokenizer identity or PAD/EOS contract changed')
        tokenized = {}
        for prompt in prompts:
            ids = records.render(tokenizer, prompt['messages'])
            if not ids or len(ids) >= records.CONTEXT:
                raise RuntimeError('Full prompt exhausts context; no truncation')
            tokenized[prompt['task_id']] = ids
        # Answers remain in the parent scorer file: generation never parses them.
        atomic(folder / 'tokenized-prompts.json', tokenized)
        cfg = json.loads((parent / 'inference-config.json').read_text())
        cfg['output_dir'] = str(output(folder) / 'loader')
        atomic(folder / 'inference-config.json', cfg)
        modules['configuration'].schema_preflight(folder / 'inference-config.json')
        recipe = dict(name=NAME, source_benchmark_id=native['benchmark_id'],
            checkpoint=native['checkpoint'], adapter_sha256=native['adapter_sha256'],
            global_step=660, tasks=138, query_grids=185, arms=['finetuned', 'base'],
            generation=native['generation'], eos_token_id=native['eos_token_id'],
            pad_token_id=native['pad_token_id'], policy=policy.POLICY,
            batch_size=policy.MAX_BATCH, seed=42,
            prompts_sha256=sha(parent / 'prompts.jsonl'),
            schedule='length-sorted waves; alternate same-arm groups; no answer-based selection',
            output_reuse='this identity only; no serial/debug prediction imports',
            reserve_seconds=90, optimizer_updates=0, python_execution=False,
            numerical_scope='batched/replayed native variant; not numerically identical to serial',
            interpretation='partial coverage is scheduling-biased; score paired tasks, no full accuracy until complete')
        recipe['benchmark_id'] = digest(recipe)
        atomic(folder / 'recipe.json', recipe)
        receipt = dict(own_manifest_sha256=sha(folder / 'package-manifest.json'),
            parent_ready_sha256=sha(parent / 'READY.json'),
            files_sha256={n: sha(folder / n) for n in
                          ('recipe.json', 'tokenized-prompts.json', 'inference-config.json')})
        atomic(folder / 'READY.json', receipt)
    print('VALIDATION UTILIZATION STAGED: all138 public tasks; both models; direct JSON; thinking off', flush=True)
    print('3 H100s /30min debug; batch cap64;90k padded input tokens;512-token checkpoints', flush=True)
    print('OOM is permitted: completed journals retained; no automatic retry or full-job submission', flush=True)


if __name__ == '__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
    stage(Path(__file__).resolve().parent)
