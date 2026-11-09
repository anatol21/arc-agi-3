"""Verify published reports from their saved responses and score records, offline."""
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    benchmarks = Path(__file__).resolve().parent
    subprocess.run([sys.executable, benchmarks / 'internal-128/verify.py'], check=True)
    held_out = benchmarks / 'held-out-138'
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([sys.executable, held_out / 'code/summarize.py', '--out', folder], check=True)
        for name in ('summary.json', 'paired-results.csv'):
            if (Path(folder) / name).read_bytes() != (held_out / 'evidence' / name).read_bytes():
                raise ValueError(f'Held-out result does not reproduce: {name}')
    print('Held-out results match the published summary and all 138 paired rows.', flush=True)
    subprocess.run([sys.executable, benchmarks / 'openrouter-138/report.py', 'verify'], check=True)


if __name__ == '__main__':
    main()
