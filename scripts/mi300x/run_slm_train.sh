#!/usr/bin/env bash
set -euo pipefail

cd /workspace/slm_train
export HF_HUB_DISABLE_TELEMETRY=1

python3 train_pomegranate_slm_mi300x.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file data/slm_pomegranate_train.jsonl \
  --eval-file data/slm_pomegranate_eval.jsonl \
  --output-dir /workspace/slm_train/out_3b \
  --epochs 3 \
  --batch-size 16 \
  --lr 1.5e-4 \
  --max-length 768
