#!/usr/bin/env bash
# Runs the whole mini project on macOS / Linux, from setup to GitHub:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO_NAME="1MS25CI031-mlflow-champion-challenger"
TRAILER=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01GMoD1tzXsHMbPL9WKxts7e'
step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
commit() { git add -A; if ! git diff --cached --quiet; then git commit -q -m "$1$TRAILER"; echo "   committed: $1"; fi; }

step "Git repository"
[ -d .git ] || git init -q -b main
git config user.name >/dev/null || git config user.name "Bhavayraj Singhal"
if ! git config user.email >/dev/null; then read -rp "Email address of your GitHub account (for the commits): " e; git config user.email "$e"; fi
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git add README.md champion_rule.json requirements.txt .gitignore data/new_data.csv
  git commit -q -m "Project setup and Champion rule, written before any evaluation run$TRAILER"
  echo "   committed: Champion rule"
  commit "Add evaluation, registry, prediction and report scripts"
fi

step "Python environment (.venv)"
[ -x .venv/bin/python ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -c "import sys; assert (3,9) <= sys.version_info[:2] <= (3,13), 'Use Python 3.9 to 3.13'"
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r requirements.txt
$PY -m playwright install chromium

step "MLflow tracking server on http://127.0.0.1:5000"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
up() { curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5000/health 2>/dev/null | grep -q 200; }
if ! up; then
  nohup $PY -m mlflow server --host 127.0.0.1 --port 5000 > mlflow_server.log 2>&1 &
  for _ in $(seq 60); do up && break; sleep 2; done
  up || { echo "MLflow server did not start, see mlflow_server.log"; exit 1; }
fi
echo "   server is up"

step "Evaluation data (200 SST-2 sentences, seed 42)"
[ -f data/eval_inputs.csv ] || $PY prepare_data.py
commit "Add shared evaluation set data/eval_inputs.csv"

step "Running both models (first run downloads ~1.7 GB of models)"; $PY evaluate.py
step "Registering models and setting aliases"; $PY register_champion.py
step "Predicting on new data with @champion"; $PY predict_from_mlflow.py
step "Stretch goal: same script with @challenger"; $PY predict_from_mlflow.py --alias challenger
step "Screenshots"; $PY capture_screenshots.py
step "Writing results and decision into README.md"; $PY write_report.py
commit "Add results, plots, screenshots, registry aliases and Champion decision"

step "GitHub"
if git remote | grep -q origin; then
  git push -q origin main && echo "   pushed to origin"
elif command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" --public --source . --remote origin --push
else
  echo "   Create an EMPTY public repo named $REPO_NAME on https://github.com/new (no README), then run:"
  echo "   git remote add origin https://github.com/<your-username>/$REPO_NAME.git"
  echo "   git push -u origin main"
fi
printf '\n\033[32mDone. Open http://127.0.0.1:5000 to see the runs. Read README.md sections 4 and 5 before submitting.\033[0m\n'
