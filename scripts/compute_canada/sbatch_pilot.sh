#!/bin/bash
#SBATCH --job-name=qwen-vl-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
# Edit account/partition to match your allocation:
# #SBATCH --account=def-XXXX
# #SBATCH --partition=gpubase_bygpu_b4

# 30B MoE or 32B dense in BF16 needs an 80 GB GPU. Request a100_80gb / h100 if
# your cluster distinguishes them. For 40 GB cards: QUANT=nf4 sbatch ...

set -euo pipefail
module load python/3.11 cuda/12.6 2>/dev/null || true

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

MODEL="${MODEL:-30b}"
if [[ "$MODEL" == "30b" ]]; then
  CONFIG=configs/pilot_qwen3vl_30b.json
  OUT=runs/pilot_text48_7view_qwen30b_a3b
else
  CONFIG=configs/pilot_qwen3vl_32b.json
  OUT=runs/pilot_text48_7view_qwen32b
fi

python scripts/run_pilot.py \
  --all \
  --pilot-config "$CONFIG" \
  --manifest manifests/pilot_text_all.jsonl \
  --out "$OUT" \
  --max-new-tokens 1024 \
  --analysis-tokens 1024 \
  --local-files-only \
  ${QUANT:+--quant "$QUANT"}
