#!/bin/bash
# Login / datamover node only (needs internet). GPU jobs should stay offline.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
mkdir -p "$HF_HOME"

if [[ $# -eq 0 ]]; then
  set -- 30b 32b
fi

python scripts/download_qwen_vl.py --model "$@"
echo "Cache: $HF_HOME"
echo "GPU jobs: export HF_HOME=$HF_HOME HF_HUB_OFFLINE=1"
