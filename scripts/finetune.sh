#!/usr/bin/env bash
# LoRA fine-tune the SLM on the constructed training set, then fuse + publish.
#
#   ./scripts/finetune.sh                      # train + fuse
#   HF_REPO=you/fraud-sentinel-1b ./scripts/finetune.sh   # train, fuse, upload
#
# Trains against a LOCAL model directory so nothing depends on Hub availability.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-models/Llama-3.2-1B-Instruct-4bit}"
ADAPTERS="${ADAPTERS:-finetune/adapters_v2}"
BEST="${BEST:-finetune/adapters_v2_best}"
FUSED="${FUSED:-finetune/fused-v2}"
BEST_ITER="${BEST_ITER:-150}"
ITERS="${ITERS:-300}"
PY=.venv/bin/python

if [ ! -f "$MODEL/model.safetensors" ] && [ ! -f "$MODEL/model-00001-of-00002.safetensors" ]; then
  echo "ERROR: no weights in $MODEL - download model.safetensors first." >&2
  exit 1
fi

echo "==> [1/3] LoRA training ($ITERS iters) on $MODEL"
$PY -m mlx_lm lora \
  --model "$MODEL" --train --data finetune/data --adapter-path "$ADAPTERS" \
  --batch-size 4 --iters "$ITERS" --num-layers 8 --learning-rate 1e-4 \
  --steps-per-report 25 --steps-per-eval 50 --save-every 50 --max-seq-length 1536

# Validation loss bottoms around iter 150 and rises after; the final checkpoint is
# overfit. Fuse the best one, not the last one.
echo "==> [2/3] fusing the iter-$BEST_ITER checkpoint (not the final, which overfits)"
mkdir -p "$BEST"
cp "$ADAPTERS/adapter_config.json" "$BEST/"
cp "$ADAPTERS/$(printf '%07d' "$BEST_ITER")_adapters.safetensors" "$BEST/adapters.safetensors"
$PY -m mlx_lm fuse --model "$MODEL" --adapter-path "$BEST" --save-path "$FUSED"

if [ -n "${HF_REPO:-}" ]; then
  echo "==> [3/3] uploading to https://huggingface.co/$HF_REPO"
  .venv/bin/hf upload "$HF_REPO" "$FUSED" . --repo-type model
  echo "    adapter-only (11 MB) copy:"
  .venv/bin/hf upload "${HF_REPO}-lora" "$BEST" . --repo-type model
else
  echo "==> [3/3] upload skipped (set HF_REPO=user/name to publish)"
fi

echo
echo "Done. Load the fused model in LM Studio:"
echo "  cp -r $FUSED ~/.lmstudio/models/local/fraud-sentinel-v2"
echo "  lms load fraud-sentinel-v2"
