#!/usr/bin/env bash
# evaluate.sh MODEL NAME [quick|full] [extra run_eval args]
# Everything we compare models on, for ONE model, then refresh the plots:
#   1. held-out agent tasks  -> outputs/eval/NAME/
#   2. IFEval / GSM8K / MMLU -> outputs/bench/NAME/
#   3. charts + report.md    -> outputs/plots/   (every model found so far, base as reference)
# Run it for every model you compare, with the same preset:
#   bash evaluate.sh Qwen/Qwen3.5-0.8B-Base base full
#   bash evaluate.sh outputs/sft_full_08b/final sft full
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: bash evaluate.sh MODEL NAME [quick|full] [extra run_eval args]"; exit 1; }
VENV_DIR="${VENV_DIR:-$HOME/venv}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
[ -f .env ] && { set -a; source .env; set +a; }
MODEL="$1"; NAME="$2"; PRESET="${3:-full}"; shift $(( $# < 3 ? $# : 3 ))
python -m evaluation.run_eval --model "$MODEL" --out "outputs/eval/$NAME" --allow-code-exec "$@"
python -m evaluation.bench --model "$MODEL" --name "$NAME" --preset "$PRESET"
python -m evaluation.plot ${REFERENCE:+--reference "$REFERENCE"} ${WANDB_PROJECT:+--wandb "$WANDB_PROJECT"}
