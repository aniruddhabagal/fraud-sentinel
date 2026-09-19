#!/usr/bin/env bash
# LoRA fine-tune the SLM on the constructed training set, then fuse + publish.
#
#   ./scripts/finetune.sh                      # train + fuse
#   HF_REPO=you/fraud-sentinel-1b ./scripts/finetune.sh   # train, fuse, upload
#
# Trains against a LOCAL model directory so nothing depends on Hub availability.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-models/Llama-3.2-1B-Instruct-bf16}"
ADAPTERS="${ADAPTERS:-finetune/adapters}"
FUSED="${FUSED:-finetune/fused-model}"
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
  --steps-per-report 20 --steps-per-eval 150 --save-every 150 --max-seq-length 1536

echo "==> [2/3] fusing adapter into a standalone model"
$PY -m mlx_lm fuse --model "$MODEL" --adapter-path "$ADAPTERS" --save-path "$FUSED"

if [ -n "${HF_REPO:-}" ]; then
  echo "==> [3/3] uploading to https://huggingface.co/$HF_REPO"
  .venv/bin/hf upload "$HF_REPO" "$FUSED" . --repo-type model
  echo "    adapter-only (small) copy:"
  .venv/bin/hf upload "$HF_REPO" "$ADAPTERS" adapters --repo-type model
else
  echo "==> [3/3] upload skipped (set HF_REPO=user/name to publish)"
fi

echo
echo "Done. Load the fused model in LM Studio:"
echo "  cp -r $FUSED ~/.lmstudio/models/local/fraud-sentinel-1b"
echo "  lms load local/fraud-sentinel-1b"
