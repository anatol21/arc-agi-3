"""Verify published reports from their saved responses and score records, offline."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile


def main():
    benchmarks = Path(__file__).resolve().parent
    subprocess.run([sys.executable, benchmarks / 'qwen3.8-flash-next/greedy-no-thinking/internal-128/verify.py'], check=True)
    held_out = benchmarks / 'qwen3.8-flash-next/greedy-no-thinking/held-out-138'
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([sys.executable, held_out / 'code/summarize.py', '--out', folder], check=True)
        for name in ('summary.json', 'paired-results.csv'):
            if (Path(folder) / name).read_bytes() != (held_out / 'evidence' / name).read_bytes():
                raise ValueError(f'Held-out result does not reproduce: {name}')
    print('Held-out results match the published summary and all 138 paired rows.', flush=True)
    refinement = benchmarks / 'qwen3.8-flash-next/python-refinement'
    for variant in ('thinking-on', 'serial-no-thinking', 'length-batched-no-thinking'):
        folder = refinement / variant
        report = json.loads((folder / 'report.json').read_bytes())
        for stage in ('first', 'selected'):
            for arm in ('base', 'finetuned'):
                rows = [r for r in report['per_task'][stage] if r['arm'] == arm]
                summary = report[stage]['arms'][arm]
                if (len(set(r['task_id'] for r in rows)) != summary['processed_tasks']
                        or sum(r['task_correct'] for r in rows) != summary['solved_tasks']):
                    raise ValueError(f'Inconsistent report: {variant}, {arm}, {stage}')
                attempts = folder / 'attempted.json'
                if attempts.exists() and not {r['task_id'] for r in rows} <= set(json.loads(attempts.read_bytes())[arm]):
                    raise ValueError(f'Scored task missing from attempted IDs: {variant}, {arm}')
        for line in (folder / 'candidates.jsonl').read_text().splitlines():
            row = json.loads(line)
            if hashlib.sha256(row['code'].encode()).hexdigest() != row['sha256']:
                raise ValueError(f'Candidate checksum differs: {variant}, {row["file"]}')
        print(f'Verified partial Python refinement report: {variant}', flush=True)
    subprocess.run([sys.executable, benchmarks / 'qwen3.8-flash/openrouter-prompt-study/report.py', 'verify'], check=True)


if __name__ == '__main__':
    main()
