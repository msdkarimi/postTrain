#!/usr/bin/env bash
# eval.sh MODEL NAME [extra run_eval args]   ->  outputs/eval/NAME/{summary.md,summary.json,results.jsonl,failures.md}
#   bash eval.sh Qwen/Qwen3.5-0.8B-Base base
#   bash eval.sh outputs/sft_full_08b/final sft --per-family 4
# Meant for the (throwaway) GPU VM: it passes --allow-code-exec, so the model's own Python/bash runs on this machine.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: bash eval.sh MODEL NAME [extra args]"; exit 1; }
VENV_DIR="${VENV_DIR:-$HOME/venv}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
[ -f .env ] && { set -a; source .env; set +a; }
MODEL="$1"; NAME="$2"; shift 2
python -m evaluation.run_eval --model "$MODEL" --out "outputs/eval/$NAME" --allow-code-exec "$@"
