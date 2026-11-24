#!/bin/bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: Python 3 is not installed. Install it from https://www.python.org/downloads/"
  read -p "Press Enter to exit..."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "First-time setup: installing Product Brain..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

(sleep 2 && open "http://127.0.0.1:8000") &
uvicorn app:app --host 127.0.0.1 --port 8000