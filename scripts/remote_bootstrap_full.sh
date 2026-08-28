#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rocling-dsa-repro

PY=/usr/local/bin/python
if [[ ! -x /workspace/venv/bin/python ]]; then
  "$PY" -m venv --system-site-packages /workspace/venv
fi
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
pip install -q -U pip
pip install -q "transformers>=4.44" accelerate sentence-transformers scikit-learn pandas pyyaml tqdm huggingface_hub
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"

mkdir -p results
nohup bash /workspace/rocling-dsa-repro/scripts/runpod_table3_encoders.sh \
  > /workspace/pipeline.log 2>&1 &
echo STARTED_PIPELINE_PID=$!
sleep 5
tail -n 30 /workspace/pipeline.log || true
