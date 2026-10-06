#!/usr/bin/env python3
"""Calibrate pilot prompts against Qwen3-VL-30B using text answers only.

Runs three CAD requests by default, each under direct, plan, and FBS prompting.
Does not capture activations. Edit src/knowledge_inv/pilot/prompts.py and re-run.

On a 32 GB GPU this defaults to 4-bit NF4. On an 80 GB cluster card use --quant bf16.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_inv.dataset.manifest import default_manifest_path
from knowledge_inv.model.qwen_vl import QwenVLRunner
from knowledge_inv.paths import PROJECT_ROOT, load_project_config
from knowledge_inv.pilot.prompts import CONDITIONS
from knowledge_inv.pilot.runner import run_pilot

PROMPTS_PATH = PROJECT_ROOT / "src" / "knowledge_inv" / "pilot" / "prompts.py"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pilot_qwen3vl_30b.json"
DEFAULT_OUT = PROJECT_ROOT / "runs" / "prompt_calibrate_30b"


def merge_pilot_config(overlay: Path | None) -> dict:
    config = load_project_config()
    if overlay is None:
        return config
    extra = load_project_config(overlay)
    config.update({key: value for key, value in extra.items() if not str(key).startswith("_")})
    return config


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def collect_request_dirs(out_dir: Path, request_ids: list[str] | None) -> list[Path]:
    if request_ids:
        return [out_dir / rid for rid in request_ids]
    return sorted(path for path in out_dir.iterdir() if path.is_dir() and (path / "summary.json").exists())


def write_transcript(out_dir: Path, request_dirs: list[Path], conditions: list[str]) -> Path:
    lines = [
        "# Prompt calibration transcript",
        "",
        "Text answers only. No activations were captured.",
        "",
    ]
    for request_dir in request_dirs:
        summary_path = request_dir / "summary.json"
        instruction = ""
        if summary_path.exists():
            instruction = json.loads(summary_path.read_text(encoding="utf-8")).get("instruction_text", "")
        lines.extend(
            [
                f"## {request_dir.name}",
                "",
                f"Editing request: {instruction}",
                "",
            ]
        )
        for condition in conditions:
            cond_dir = request_dir / condition
            analysis = read_text(cond_dir / "analysis" / "answer.txt")
            decision = read_text(cond_dir / "answer.txt")
            lines.append(f"### {condition}")
            lines.append("")
            if analysis:
                lines.append("Analysis:")
                lines.append("")
                lines.append(analysis)
                lines.append("")
            lines.append("Decision:")
            lines.append("")
            lines.append(decision or "(missing answer.txt)")
            lines.append("")
    path = out_dir / "transcript.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def print_transcript(request_dirs: list[Path], conditions: list[str]) -> None:
    for request_dir in request_dirs:
        instruction = ""
        summary_path = request_dir / "summary.json"
        if summary_path.exists():
            instruction = json.loads(summary_path.read_text(encoding="utf-8")).get("instruction_text", "")
        print("\n" + "=" * 72, flush=True)
        print(f"REQUEST {request_dir.name}", flush=True)
        print(f"Editing request: {instruction}", flush=True)
        for condition in conditions:
            cond_dir = request_dir / condition
            analysis = read_text(cond_dir / "analysis" / "answer.txt")
            decision = read_text(cond_dir / "answer.txt")
            print("-" * 72, flush=True)
            print(f"CONDITION: {condition}", flush=True)
            if analysis:
                print("\n--- analysis ---\n", flush=True)
                print(analysis, flush=True)
            print("\n--- decision ---\n", flush=True)
            print(decision or "(missing answer.txt)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send current prompts.py to Qwen3-VL-30B and collect verbal answers only."
    )
    parser.add_argument("--pilot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--request-id",
        action="append",
        dest="request_ids",
        default=None,
        help="Repeat to pick specific requests. Default: first --limit eligible records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of CAD requests to run when --request-id is not set. Ignored with --all.",
    )
    parser.add_argument("--all", action="store_true", help="Run every eligible manifest record.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["direct", "plan", "fbs"],
        choices=list(CONDITIONS),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--quant",
        choices=("nf4", "fp4", "bf16"),
        default="nf4",
        help="nf4/fp4 for a 32 GB GPU. bf16 for an 80 GB cluster GPU.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--analysis-tokens", type=int, default=None)
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Skip a condition if metadata.json already exists. Default is to regenerate.",
    )
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
    request_ids = None if args.all else args.request_ids
    limit = None if args.all else args.limit
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(config.get("max_new_tokens", 1024))
    analysis_tokens = (
        args.analysis_tokens if args.analysis_tokens is not None else int(config.get("analysis_tokens", max_new_tokens))
    )
    quantization = None if args.quant == "bf16" else args.quant
    skip_modules = config.get("quantization_skip_modules")
    if skip_modules is not None:
        skip_modules = list(skip_modules)

    print(
        f"Loading {config.get('model_id')} (quant={args.quant}, activations=off)…",
        flush=True,
    )
    runner = QwenVLRunner(
        model_id=config.get("model_id"),
        quantization=quantization,
        skip_modules=skip_modules,
        device_map=config.get("device_map"),
        local_files_only=True,
    )
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROMPTS_PATH, out_dir / "prompts_snapshot.py")
    summary = run_pilot(
        manifest_path=manifest,
        request_id=None,
        out_dir=out_dir,
        conditions=args.conditions,
        capture_activations=False,
        force=not args.no_force,
        run_all=args.all,
        max_new_tokens=max_new_tokens,
        analysis_tokens=analysis_tokens,
        runner=runner,
        capture_layers=None,
        request_ids=request_ids,
        limit=limit,
        extra_batch={
            "task": "prompt_calibration",
            "capture_activations": False,
            "quant": args.quant,
            "prompts": str(PROMPTS_PATH),
        },
    )
    ran_ids = summary.get("request_ids") if isinstance(summary, dict) else [summary["request_id"]]
    request_dirs = collect_request_dirs(out_dir, ran_ids)
    transcript = write_transcript(out_dir, request_dirs, args.conditions)
    print_transcript(request_dirs, args.conditions)
    print("\n" + "=" * 72, flush=True)
    print(f"Transcript: {transcript}", flush=True)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
