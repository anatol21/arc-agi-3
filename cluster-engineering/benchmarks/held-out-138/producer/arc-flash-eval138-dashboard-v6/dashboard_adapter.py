"""One logical dashboard worker owns the three-GPU native batch runner."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def counters(states, batches):
    return (sum(row['generated_tokens'] for row in states.values()),
            sum(r['finish_reason'] in ('eos', 'token_limit') for r in states.values()))


def checkpoint_names(root):
    return ['workers/0/checkpoint.json', 'native/source-receipt.json'] + [
        str(p.relative_to(root)) for pattern in ('native/batches/batch-*.json',
                                               'native/oom-attempts/*.json', 'native/terminals/*.json')
        for p in sorted(root.glob(pattern))]


def main():
    folder = Path(__file__).resolve().parent
    root = Path(os.environ['WM_RUN_DIR']).resolve()
    sys.path.insert(0, str(root))
    from sdk import Adapter
    adapter = Adapter()
    if adapter.worker != 0 or adapter.workers != 1:
        raise RuntimeError('This model spans three GPUs in one logical worker, not three model replicas')
    sys.path.insert(0, str(folder))
    from bridge import dependencies, output, atomic, sha
    import journal
    import run
    with adapter.phase('preparation'):
        _, _, prompts, modules, recipe = dependencies(folder)
        manifest = json.loads(Path(os.environ['WM_CONFIG']).read_text())
        expected = {'package': sha(folder/'package-manifest.json'),
                    'recipe': sha(folder/'recipe.json'),
                    'eligibility': sha(folder/'eligibility.json'),
                    'adapter': recipe['adapter_sha256']}
        if manifest['config']['fingerprints'] != expected:
            raise RuntimeError('Dashboard recipe/source/eligibility fingerprints changed')
        required = run.required(recipe)
        if int(os.environ['WM_TASK_COUNT']) != required:
            raise RuntimeError('Declared dashboard inventory differs from frozen eligibility')
        if importlib.util.find_spec('causal_conv1d') is not None:
            raise RuntimeError('Native-kernel variant requires original environment without causal_conv1d')
        owned = output(folder)
        if owned != root/'native': raise RuntimeError('Output leaves dashboard run directory')
        receipt = dict(dashboard_identity=adapter.identity, fingerprints=expected,
                       benchmark_id=recipe['benchmark_id'], required_predictions=required)
        old = owned/'source-receipt.json'
        if old.exists() and json.loads(old.read_text()) != receipt:
            raise RuntimeError('Existing native journals belong to another dashboard recipe')
        atomic(old, receipt)
        tokenized = json.loads((folder/'tokenized-prompts.json').read_text())

    inherited_tokens=sum(len(r['prefix']) for r in json.loads((folder/'continuation.json').read_text())['prefixes'])
    def record(states, batches, allocation=None):
        tokens, done = counters(states, batches)
        tokens -= inherited_tokens
        state = dict(benchmark_id=recipe['benchmark_id'], tokens=tokens,
                     completed=done, required=required, committed_batches=len(batches))
        adapter.checkpoint(state)
        values = dict(progress=tokens, generated_tokens=tokens, completed=done,
                      required=required, batch_calls=len(batches), pending=len(states)-done)
        if batches:
            b=batches[-1]
            values.update(batch_size=b['actual_batch_size'],
                          tokens_per_second=b['tokens_per_second'])
            for entry in b.get('memory', []):
                values['gpu%d_peak_reserved_gib'%entry['device']]=entry['peak_reserved_bytes']/2**30
        if allocation: values['failed_batch_attempts']=allocation['failed_attempts']
        adapter.metric(**values)
        return tokens, done

    with adapter.phase('restore'):
        states, batches = journal.collect(owned, recipe, prompts, tokenized)
        prior = adapter.load_checkpoint()
        tokens, done = counters(states, batches)
        tokens -= inherited_tokens
        if prior and (prior['benchmark_id'] != recipe['benchmark_id'] or
                      prior['tokens'] > tokens or prior['completed'] > done):
            raise RuntimeError('Durable native progress is behind the dashboard checkpoint')
        os.environ['WM_PRIOR_COMPLETED'] = str(done)
        record(states, batches)
    baseline = done
    finished = set(journal.completed(states))

    def terminal(row):
        key = row['task_id'], row['arm']
        if key in finished:
            raise RuntimeError('Duplicate terminal response')
        journal.publish_terminal(owned, row)
        finished.add(key)
        elapsed = time.time() - float(os.environ['WM_ALLOCATION_START'])
        new = len(finished) - baseline
        adapter.metric(completed=len(finished), newly_finished=new,
                       terminal_task=row['task_id'], terminal_arm=row['arm'],
                       terminal_save_seconds=elapsed,
                       responses_per_allocated_gpu_hour=new*3600/(3*elapsed))
    run.dashboard_terminal = terminal
    run.dashboard_stopping = adapter.stopping
    run.dashboard_progress = record
    original_loader = run.load_model

    def load_model(folder, recipe, modules, output_root, allocation):
        from cold import local_snapshot

        source = Path(json.loads((folder / 'inference-config.json').read_text())['base_model']).resolve()
        staging = {}
        with adapter.phase('model_load'):
            with local_snapshot(source, modules, staging, workers=32):
                adapter.metric(weight_copy_verify_seconds=staging['preparation_seconds'],
                               weight_copy_workers=staging['copy_workers'],
                               verified_weight_files=staging['verified_files'],
                               staged_bytes=staging['staged_bytes'])
                return original_loader(folder, recipe, modules, output_root, allocation)

    run.load_model = load_model
    failure = None
    started = time.monotonic()
    try:
        with adapter.phase('execution'): run.run(folder)
    except BaseException as error:
        failure = error
    # Read only committed, identity-checked journals. Wrong/capped outputs are terminal.
    states, batches = journal.collect(owned, recipe, prompts, tokenized)
    allocation = json.loads((owned/'last-allocation.json').read_text()) if (owned/'last-allocation.json').exists() else None
    tokens, done = record(states, batches, allocation)
    adapter.metric(adapter_elapsed_seconds=time.monotonic()-started)
    result = 'failed' if failure else 'complete' if done == required else 'partial_checkpointed'
    adapter.finish(result, progress=tokens, completed=done,
                   checkpoint_files=checkpoint_names(root))
    if failure: raise failure
    print('DASHBOARD_ARC_RESULT '+json.dumps(dict(state=result, completed=done,
        required=required, generated_tokens=tokens, output=str(owned))), flush=True)


if __name__ == '__main__': main()
