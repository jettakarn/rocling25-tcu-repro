#!/usr/bin/env bash
# Full Table 3 LLM embeds + Table 4 five-encoder ensemble on a ≥16GB GPU.
# Requires HF_TOKEN in the environment for gated TAIDE.
set -euo pipefail
cd /workspace/rocling-dsa-repro
source /workspace/venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1

run_one() {
  local model="$1"
  local cfg="$2"
  echo "===== EMBED $model ====="
  python -m src.embed_llm_full --config "$cfg" --model "$model" --split all --dtype float16 --batch-size 2
  echo "===== SVR $model ====="
  python -m src.train_svr --config "$cfg" --strategy train_full_dev
  python -m src.predict_test --config "$cfg" --strategy train_full_dev --run-official-scoring
}

# DeepSeek may already exist; re-run to ensure train/dev/test present
run_one deepseek configs/deepseek_r1.yaml
run_one prover configs/deepseek_prover.yaml

if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN required for gated TAIDE (taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)"
  echo "Set HF_TOKEN then re-run taide + ensemble, or export before this script."
  exit 2
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"

run_one taide configs/taide.yaml

echo "===== TABLE 4 ENCODERS (5-way) ====="
python -m src.ensemble_encoders \
  --encoders \
    deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    deepseek-ai/DeepSeek-Prover-V1.5-RL \
    taide/Llama3-TAIDE-LX-8B-Chat-Alpha1 \
    intfloat/multilingual-e5-large \
    intfloat/multilingual-e5-large-instruct \
  --strategy train_full_dev

echo ALL_DONE > /workspace/pipeline_done.flag
echo "DONE"
