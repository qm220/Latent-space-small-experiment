#!/usr/bin/env python3
"""Download Qwen3-VL checkpoints into the Hugging Face cache (Compute Canada scratch).

Run on a login node or datamover that has internet. GPU jobs should then set
HF_HUB_OFFLINE=1 and HF_HOME to the same cache.

  export HF_HOME=$SCRATCH/huggingface
  python scripts/download_qwen_vl.py --model 30b
  python scripts/download_qwen_vl.py --model 32b

30B is Qwen3-VL-30B-A3B-Instruct (MoE VLM, ~3B active, ~31B stored).
32B is Qwen3-VL-32B-Instruct (dense VLM). Both are vision-language; text-only
Qwen3-30B cannot run the 7-view CAD protocol.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "32b": "Qwen/Qwen3-VL-32B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Qwen3-VL Hugging Face checkpoint.")
    parser.add_argument("--model", choices=sorted(MODELS), nargs="+", required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face hub cache. Default: $HF_HOME/hub or ~/.cache/huggingface/hub.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Optional explicit folder copy in addition to the hub cache.",
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir
    if cache_dir is None and os.environ.get("HF_HOME"):
        cache_dir = Path(os.environ["HF_HOME"]) / "hub"
    for key in args.model:
        repo_id = MODELS[key]
        print(f"Downloading {repo_id} …", flush=True)
        kwargs = {
            "repo_id": repo_id,
            "ignore_patterns": ["*.bin", "*.pth", "*.onnx", "original/*"],
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        if args.local_dir is not None:
            kwargs["local_dir"] = str(args.local_dir / key)
        path = snapshot_download(**kwargs)
        print(f"Ready: {repo_id}\n  {path}", flush=True)


if __name__ == "__main__":
    main()
