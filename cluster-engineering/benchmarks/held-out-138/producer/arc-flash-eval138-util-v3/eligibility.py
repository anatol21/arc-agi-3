"""Snapshot finished cluster direct-answer pairs, never using correctness or query labels."""
import json
from pathlib import Path
from bridge import digest, sha

NATIVE_RUNS = ('arc-flash-eval138-v1', 'arc-flash-eval138-debug-v1',
               'arc-flash-eval138-pilot-v1')
BATCHED_RUNS = ('arc-flash-eval138-util-v1', 'arc-flash-eval138-util-v2')


def snapshot(workspace, prompts):
    wanted = {p['task_id'] for p in prompts}
    finished = set(); evidence = []; pending_tails = []

    def accept(row, path):
        if row.get('task_id') not in wanted or row.get('arm') not in ('finetuned','base'):
            raise RuntimeError('Unknown prior prediction pair: '+str(path))
        if row.get('enable_thinking', False) is not False:
            raise RuntimeError('Unexpected thinking response in prior direct-answer run')
        if row.get('finish_reason') in ('eos','token_limit'):
            finished.add((row['task_id'],row['arm']))

    for name in NATIVE_RUNS:
        path = Path(workspace)/'runs'/name/'predictions.jsonl'
        if not path.exists():
            continue
        raw = path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            cut = raw.rfind(b'\n')+1
            pending_tails.append(str(path.relative_to(workspace)))
            raw = raw[:cut]  # Preserve source bytes. An uncommitted tail is not finished.
        for line in raw.splitlines():
            if line.strip():
                accept(json.loads(line), path)
        evidence.append(dict(path=str(path.relative_to(workspace)),sha256=sha(path)))
    for name in BATCHED_RUNS:
        for seq,path in enumerate(sorted((Path(workspace)/'runs'/name/'batches').glob('batch-*.json')),1):
            value=json.loads(path.read_text())
            if value.get('sequence')!=seq or digest({k:v for k,v in value.items()if k!='sha256'})!=value.get('sha256'):
                raise RuntimeError('Corrupt prior batch journal: '+str(path))
            for row in value['rows']:
                accept(row,path)
            evidence.append(dict(path=str(path.relative_to(workspace)),sha256=sha(path)))
    if not evidence:
        raise RuntimeError('No prior direct-answer prediction evidence found; refusing an unintended fresh run')
    excluded=[dict(task_id=t,arm=a)for t,a in sorted(finished)]
    eligible={a:[p['task_id']for p in prompts if (p['task_id'],a)not in finished]for a in ('finetuned','base')}
    return dict(eligible=eligible,excluded_finished_pairs=excluded,sources=evidence,
        ignored_uncommitted_tails=pending_tails,selection_uses_correctness=False,
        prior_responses_imported=False,unfinished_prefixes_imported=False,
        scope='All138 public tasks, only cluster direct-answer task/model pairs not finished at staging')
