#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_inv.dataset.manifest import default_manifest_path, load_jsonl
from knowledge_inv.model.qwen_vl import QwenVLRunner
from knowledge_inv.paths import PROJECT_ROOT, load_project_config
from knowledge_inv.pilot.prompts import decision_prompt
from knowledge_inv.pilot.runner import collect_original_views, load_view_images, save_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL on one verified manifest record.")
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--request-id", type=str, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "qwen_example")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--no-activations", action="store_true")
    args = parser.parse_args()

    config = load_project_config()
    records = load_jsonl(args.manifest)
    record = next(
        (item for item in records if item["request_id"] == args.request_id),
        records[0],
    )
    views = load_view_images(
        collect_original_views(record, config.get("standard_views")),
        config["image_max_side"],
    )
    images = [item["image"] for item in views]
    view_names = [item["view"] for item in views]
    prompt = decision_prompt(record["instruction_text"])

    runner = QwenVLRunner()
    capture_layers = None if args.no_activations else config["activation_layers"]
    result = runner.generate(
        images,
        prompt,
        max_new_tokens=args.max_new_tokens,
        capture_layers=capture_layers,
        view_names=view_names,
    )

    out_dir = args.out / record["request_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "input_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for item in views:
        item["image"].save(image_dir / f"{item['view']}.png")
    save_generation(
        out_dir,
        result,
        {
            "stage": "decision",
            "condition": "direct",
            "request_id": record["request_id"],
            "n_images": len(views),
            "image_views": view_names,
            "image_paths": [item["absolute_path"] for item in views],
            "image_sizes": [item["size"] for item in views],
            "image_max_side": config["image_max_side"],
            "evaluation_assets_excluded": True,
        },
    )
    print("\nMODEL RESPONSE:\n")
    print(result["answer"])
    print("\nSaved:", out_dir)
    print(json.dumps(
        {
            "request_id": record["request_id"],
            "revision": runner.revision,
            "runtime_sec": result["runtime_sec"],
            "peak_gpu_memory_gb": result["peak_gpu_memory_gb"],
            "activations": bool(capture_layers),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
