#!/usr/bin/env bash
# teacher.sh [smoke|full]  - use Qwen3.8-27B as the TEACHER on the GPU box (run BEFORE training; it needs the whole GPU).
#   smoke: 64 prompts per job, a few minutes - check the pass rates and look at the samples first
#   full : IF_N instruction prompts + PARA_N rewrites per family, then builds the rewrite records
# Output: data/raw/if_verified.jsonl, data/raw/agent_teacher_model_rewrites.jsonl  (both picked up by etl/configs/sft_mix*.yaml)
set -euo pipefail
MODE="${1:-smoke}"
TEACHER="${TEACHER:-Qwen/Qwen3.8-27B}"
IF_N="${IF_N:-3000}"; PARA_N="${PARA_N:-60}"; BATCH="${BATCH:-32}"
VENV_DIR="${VENV_DIR:-$HOME/venv}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
[ -f .env ] && { set -a; source .env; set +a; }

FREE_GB=$(df -BG --output=avail "${HF_HOME:-$HOME/.cache}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
[ "${FREE_GB:-0}" -ge 70 ] || { echo "Need ~70 GB free for the 56 GB of teacher weights (have ${FREE_GB:-?} GB). Free disk or set HF_HOME."; exit 1; }
echo "=== downloading $TEACHER (56 GB, once) ==="
python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$TEACHER"))
PY

mkdir -p data/raw data/teacher
[ -f data/raw/if_prompts.jsonl ] || python -m distill.ifgen --n 6000 --out data/raw/if_prompts.jsonl
LIMIT=(); [ "$MODE" = "smoke" ] && LIMIT=(--limit 64)
[ "$MODE" = "full" ] && head -n "$IF_N" data/raw/if_prompts.jsonl > data/raw/if_prompts_run.jsonl || cp data/raw/if_prompts.jsonl data/raw/if_prompts_run.jsonl

echo "=== job 1: verified instruction-following answers ==="
python -m distill.teacher_sample if --teacher "$TEACHER" --prompts data/raw/if_prompts_run.jsonl --out data/raw/if_verified.jsonl --batch-size "$BATCH" "${LIMIT[@]}"
echo "=== job 2: model-written prompt rewrites ==="
python -m distill.teacher_sample paraphrase --teacher "$TEACHER" --out data/teacher/model_rewrites.tsv --n-per-family "$PARA_N" --batch-size "$BATCH" "${LIMIT[@]}"
python -m distill.rewrite apply --rewrites data/teacher/model_rewrites.tsv --suffix tm --out data/raw/agent_teacher_model_rewrites.jsonl
echo "done ($MODE). Look at a few samples before training:  head -n 3 data/raw/if_verified.jsonl | python -m json.tool | head -60"
