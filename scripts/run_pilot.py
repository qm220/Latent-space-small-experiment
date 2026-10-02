#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_inv.dataset.manifest import default_manifest_path
from knowledge_inv.model.layers import sample_language_layers
from knowledge_inv.model.qwen_vl import QwenVLRunner
from knowledge_inv.paths import PROJECT_ROOT, load_project_config
from knowledge_inv.pilot.prompts import CONDITIONS
from knowledge_inv.pilot.runner import run_pilot


def merge_pilot_config(overlay: Path | None) -> dict:
    config = load_project_config()
    if overlay is None:
        return config
    extra = load_project_config(overlay)
    config.update({key: value for key, value in extra.items() if not str(key).startswith("_")})
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three-condition knowledge pilot.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--request-id", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=None,
        help="Overlay JSON (model id, layers, tokens). Dataset paths stay in configs/project.json.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["direct", "plan", "fbs"],
        choices=list(CONDITIONS),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run every record in the manifest.")
    parser.add_argument("--no-activations", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--analysis-tokens", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--quant", choices=("nf4", "fp4", "prequantized"), default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config = merge_pilot_config(args.pilot_config)
    if args.manifest is not None:
        manifest = args.manifest
    elif config.get("manifest"):
        manifest = Path(config["manifest"])
        if not manifest.is_absolute():
            manifest = PROJECT_ROOT / manifest
    else:
        manifest = default_manifest_path()
    out_dir = args.out or PROJECT_ROOT / config.get("default_out", "runs/pilot")
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(config.get("max_new_tokens", 1024))
    analysis_tokens = (
        args.analysis_tokens if args.analysis_tokens is not None else int(config.get("analysis_tokens", max_new_tokens))
    )
    request_id = None if args.all else (args.request_id or config["primary_example_request_id"])
    model_id = args.model_id or config.get("model_id")
    quantization = args.quant if args.quant is not None else config.get("quantization")
    skip_modules = config.get("quantization_skip_modules")
    if skip_modules is not None:
        skip_modules = list(skip_modules)
    device_map = config.get("device_map")
    print(f"Loading {model_id} (quantization={quantization!r})…", flush=True)
    runner = QwenVLRunner(
        model_id=model_id,
        quantization=quantization,
        skip_modules=skip_modules,
        device_map=device_map,
        local_files_only=True if args.local_files_only else None,
    )
    expected = config.get("n_language_layers_expected")
    if expected and runner.n_layers != expected:
        print(
            f"Warning: expected {expected} language layers, found {runner.n_layers}.",
            flush=True,
        )
    if args.layers is not None:
        capture_layers = list(args.layers)
    elif config.get("activation_layers"):
        capture_layers = list(config["activation_layers"])
    else:
        n_sample = int(config.get("n_activation_layers", 8))
        capture_layers = sample_language_layers(runner.n_layers, n_sample)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PROJECT_ROOT / "src" / "knowledge_inv" / "pilot" / "prompts.py",
        out_dir / "prompts_snapshot.py",
    )
    summary = run_pilot(
        manifest_path=manifest,
        request_id=request_id,
        out_dir=out_dir,
        conditions=args.conditions,
        capture_activations=not args.no_activations,
        force=args.force,
        run_all=args.all,
        max_new_tokens=max_new_tokens,
        analysis_tokens=analysis_tokens,
        runner=runner,
        capture_layers=capture_layers,
        extra_batch={
            "max_new_tokens": max_new_tokens,
            "analysis_tokens": analysis_tokens,
            "pilot_config": str(args.pilot_config) if args.pilot_config else None,
            "device_map": device_map,
            "activation_layers": capture_layers,
            "architecture": config.get("architecture"),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
