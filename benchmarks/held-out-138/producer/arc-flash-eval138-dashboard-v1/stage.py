"""CPU staging: same public tasks and prompt text, distinct aggressive batching variant."""
import json
import os
from pathlib import Path
from bridge import NAME, atomic, dependencies, digest, output, sha
import policy
import records
import eligibility


def stage(folder):
    from score import verify_runtime
    verify_runtime()
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
        proof=eligibility.snapshot(folder.parent,prompts)
        atomic(folder/'eligibility.json',proof)
        recipe = dict(name=NAME, source_benchmark_id=native['benchmark_id'],
            checkpoint=native['checkpoint'], adapter_sha256=native['adapter_sha256'],
            global_step=660, tasks=len(prompts), query_grids=sum(p['query_count'] for p in prompts),
            cohorts={'public_validation':[p['task_id']for p in prompts]},
            eligible=proof['eligible'],eligibility_sha256=sha(folder/'eligibility.json'),
            arms=['finetuned', 'base'],
            generation=native['generation'], eos_token_id=native['eos_token_id'],
            pad_token_id=native['pad_token_id'], policy=policy.POLICY,
            batch_size=policy.MAX_BATCH, seed=42,
            prompts_sha256=sha(parent / 'prompts.jsonl'),
            schedule='length-aware24/32/48/64 ladder, floor18; alternate arms; no correctness feedback',
            output_reuse='finished pairs excluded by frozen staging snapshot; no prior response/prefix imports',
            reserve_seconds=90, optimizer_updates=0, python_execution=False,
            numerical_scope='batched/replayed native variant; not numerically identical to serial',
            interpretation='unfinished-pair diagnostic; incomplete length-selected subset; not full138-task accuracy')
        recipe['benchmark_id'] = digest(recipe)
        atomic(folder / 'recipe.json', recipe)
        receipt = dict(own_manifest_sha256=sha(folder / 'package-manifest.json'),
            parent_ready_sha256=sha(parent / 'READY.json'),
            files_sha256={n: sha(folder / n) for n in
                          ('recipe.json', 'tokenized-prompts.json', 'inference-config.json','eligibility.json')})
        atomic(folder / 'READY.json', receipt)
    print('UNFINISHED PUBLIC VALIDATION STAGED '+json.dumps({a:len(v)for a,v in proof['eligible'].items()}),flush=True)
    print('DASHBOARD READY: 3H100; targets24/32/48/64; floor18; sealed per-run journals; no GPU job submitted.',flush=True)
    print('Only unfinished cluster direct-answer pairs. No prior answers imported, no job submitted.',flush=True)


if __name__ == '__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
    stage(Path(__file__).resolve().parent)
