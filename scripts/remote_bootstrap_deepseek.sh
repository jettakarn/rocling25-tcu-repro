#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rocling-dsa-repro

PY=/usr/local/bin/python
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"

if [[ ! -x /workspace/venv/bin/python ]]; then
  "$PY" -m venv --system-site-packages /workspace/venv
fi
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

python -c "import torch; print('venv torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install -q -U pip
pip install -q transformers accelerate sentence-transformers scikit-learn pandas pyyaml tqdm

python -c "import transformers; print('transformers', transformers.__version__)"
mkdir -p results

# Kill any previous embed job
pkill -f 'src.embed_llm_full' 2>/dev/null || true

nohup python -m src.embed_llm_full \
  --config configs/deepseek_r1.yaml \
  --model deepseek \
  --split all \
  --dtype float16 \
  --batch-size 2 \
  > /workspace/embed_deepseek.log 2>&1 &

echo "STARTED_PID=$!"
sleep 8
tail -n 40 /workspace/embed_deepseek.log || true
ps aux | grep embed_llm_full | grep -v grep || true
