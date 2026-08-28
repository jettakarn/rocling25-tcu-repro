#!/usr/bin/env bash
# Run on a ≥16GB GPU pod after the repo + processed CSVs + e5 .npy are present.
# Skips TAIDE. Produces Table 3 SVR rows for DeepSeek-R1 / Prover and a 4-encoder Table 4 mix.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.embed_llm_full --config configs/deepseek_r1.yaml --model deepseek --split all --dtype float16 --batch-size 1
python -m src.train_svr --config configs/deepseek_r1.yaml --strategy train_full_dev
python -m src.predict_test --config configs/deepseek_r1.yaml --strategy train_full_dev --run-official-scoring

python -m src.embed_llm_full --config configs/deepseek_prover.yaml --model prover --split all --dtype float16 --batch-size 1
python -m src.train_svr --config configs/deepseek_prover.yaml --strategy train_full_dev
python -m src.predict_test --config configs/deepseek_prover.yaml --strategy train_full_dev --run-official-scoring

python -m src.ensemble_encoders \
  --encoders \
    deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    deepseek-ai/DeepSeek-Prover-V1.5-RL \
    intfloat/multilingual-e5-large \
    intfloat/multilingual-e5-large-instruct \
  --strategy train_full_dev

echo "DONE — check results/ for deepseek_* and encoder_ensemble_* JSON"
