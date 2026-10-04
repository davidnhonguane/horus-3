#!/usr/bin/env sh
# Horus one-command start (macOS / Linux)
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
echo "Checking Python packages (first time takes a minute) ..."
python -m pip install -q -r requirements.txt
python -m horus serve "$@"
