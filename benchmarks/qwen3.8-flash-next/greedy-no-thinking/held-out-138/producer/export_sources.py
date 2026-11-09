"""Export deployed evaluation source from a private workspace and record original/public SHA-256s."""
import argparse
import hashlib
import json
from pathlib import Path

PACKAGES = ['arc-flash-eval138-v1', 'arc-flash-eval138-debug-v1', 'arc-flash-eval138-pilot-v1',
            'arc-flash-eval138-util-v1', 'arc-flash-eval138-util-v2', 'arc-flash-eval138-util-v3',
            'arc-flash-eval138-dashboard-v1', 'arc-flash-eval138-dashboard-v4',
            'arc-flash-eval138-dashboard-v5', 'arc-flash-eval138-dashboard-v6']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def export(workspace, destination, original_path):
    index = []
    for pkg in PACKAGES:
        src = workspace / pkg
        dst = destination / pkg
        dst.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((src / 'package-manifest.json').read_bytes())['files_sha256']
        for f in sorted(src.iterdir()):
            config = f.name in {'inference-config.json', 'benchmark-recipe.json'}
            if not f.is_file() or not (config or f.suffix in {'.py', '.sh', '.sbatch'} or f.name == 'instruction.txt'):
                continue
            raw = f.read_bytes()
            if not config and sha(raw) != manifest[f.name]:
                raise ValueError(f'Deployed file differs from package manifest: {pkg}/{f.name}')
            text = raw.decode()
            if config:
                text = text.replace(original_path, '/shared/arc-workspace')
            elif f.suffix == '.py':
                text = text.replace(f'Path("{original_path}")', 'Path(__file__).resolve().parents[1]')
            elif f.suffix in {'.sbatch', '.sh'}:
                lines = []
                for line in text.splitlines(keepends=True):
                    if line.startswith(('#SBATCH --account=', '#SBATCH --chdir=')):
                        continue
                    if line.strip() == 'cd ' + original_path:
                        line = 'cd "${ARC_WORKSPACE:?Set ARC_WORKSPACE to the shared workspace}"\n'
                    if line.startswith('export LD_LIBRARY_PATH="/opt/'):
                        line = '# Set the runtime library path for your local build before launch.\n'
                    if line.startswith('module load '):
                        line = 'if [[ -n "${ARC_PYTHON_MODULE:-}" ]]; then module load "$ARC_PYTHON_MODULE"; fi\n'
                    lines.append(line)
                text = ''.join(lines)
            if original_path in text:
                raise ValueError(f'Unmapped private path in {pkg}/{f.name}')
            data = text.encode()
            target = dst / f.name
            target.write_bytes(data)
            if f.suffix in {'.sh', '.sbatch'}:
                target.chmod(0o755)
            index.append({'path': f'{pkg}/{f.name}', 'deployed_sha256': sha(raw), 'published_sha256': sha(data),
                          'in_package_manifest': f.name in manifest and manifest[f.name] == sha(raw)})
    (destination / 'source-index.json').write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--original-workspace-path', help='Absolute workspace path embedded in the historical files')
    args = parser.parse_args()
    export(args.workspace, args.destination, args.original_workspace_path or str(args.workspace.resolve()))
