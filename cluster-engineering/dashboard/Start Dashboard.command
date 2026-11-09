#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import yaml' >/dev/null 2>&1; then
    .venv/bin/pip install -r requirements.txt
fi
print 'Open http://127.0.0.1:8766 in your browser. Press Control-C here to stop the local service.'
exec .venv/bin/python server.py --demo
