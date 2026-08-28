#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rocling-dsa-repro
source /workspace/venv/bin/activate

# DeepSeek SVR + test while Prover downloads/embeds next
nohup bash -lc '
  set -e
  cd /workspace/rocling-dsa-repro
  source /workspace/venv/bin/activate
  python -m src.train_svr --config configs/deepseek_r1.yaml --strategy train_full_dev
  python -m src.predict_test --config configs/deepseek_r1.yaml --strategy train_full_dev --run-official-scoring
  python -m src.embed_llm_full --config configs/deepseek_prover.yaml --model prover --split all --dtype float16 --batch-size 2
  python -m src.train_svr --config configs/deepseek_prover.yaml --strategy train_full_dev
  python -m src.predict_test --config configs/deepseek_prover.yaml --strategy train_full_dev --run-official-scoring
  python -m src.ensemble_encoders \
    --encoders \
      deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      deepseek-ai/DeepSeek-Prover-V1.5-RL \
      intfloat/multilingual-e5-large \
      intfloat/multilingual-e5-large-instruct \
    --strategy train_full_dev
  echo ALL_DONE > /workspace/pipeline_done.flag
' > /workspace/pipeline.log 2>&1 &

echo STARTED_PIPELINE_PID=$!
sleep 3
tail -n 25 /workspace/pipeline.log || true
ps aux | grep -E 'train_svr|embed_llm|ensemble' | grep -v grep | head -10
