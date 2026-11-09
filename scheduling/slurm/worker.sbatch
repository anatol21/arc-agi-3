#!/usr/bin/env bash
set -euo pipefail
exec "${SCHEDULER_PYTHON:-python3}" "$2" "$1"
