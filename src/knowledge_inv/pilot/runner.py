from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from knowledge_inv.dataset.readonly_db import STANDARD_VIEWS
from knowledge_inv.dataset.manifest import load_jsonl
from knowledge_inv.model.qwen_vl import QwenVLRunner, load_image
from knowledge_inv.paths import load_project_config
from knowledge_inv.pilot.prompts import CONDITIONS, decision_prompt


def choose_image(record: dict[str, Any], preferred_views: list[str]) -> dict[str, Any]:
    """First existing view in preferred order (kept for single-image callers)."""
    packed = collect_original_views(record, preferred_views, require_all=False)
    return packed[0]


def collect_original_views(
    record: dict[str, Any],
    view_order: Sequence[str] | None = None,
    require_all: bool = True,
) -> list[dict[str, Any]]:
    """Return original-model views in `view_order` (dataset standard order by default)."""
    images = record.get("original_image_paths") or {}
    order = list(view_order) if view_order is not None else list(STANDARD_VIEWS)
    found: list[dict[str, Any]] = []
    missing: list[str] = []
    for view in order:
        item = images.get(view)
        if item and item.get("exists"):
            found.append({"view": view, **item})
        else:
            missing.append(view)
    if not found:
        raise FileNotFoundError(f"No usable original images for {record['request_id']}")
    if require_all and missing:
        raise FileNotFoundError(
            f"{record['request_id']} missing original views: {missing}. "
            "Pilot-eligible records must have all 7."
        )
    return found


def load_view_images(view_infos: list[dict[str, Any]], max_side: int) -> list[dict[str, Any]]:
    loaded = []
    for info in view_infos:
        image = load_image(info["absolute_path"], max_side)
        loaded.append({**info, "image": image, "size": list(image.size)})
    return loaded


def save_generation(out_dir: Path, result: dict[str, Any], extra: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answer.txt").write_text(result["answer"], encoding="utf-8")
    (out_dir / "prompt.txt").write_text(result["prompt"], encoding="utf-8")
    activations = result.pop("activations", None)
    metadata = {**extra, **{k: v for k, v in result.items() if k != "answer"}}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if activations:
        import torch

        torch.save(activations["tensors"], out_dir / "activations.pt")
        (out_dir / "activations.json").write_text(
            json.dumps(activations["metadata"], indent=2),
            encoding="utf-8",
        )
        tokens = activations.get("tokens") or []
        (out_dir / "tokens.json").write_text(json.dumps(tokens, indent=2), encoding="utf-8")
        output_tokens = [item for item in tokens if item.get("split") == "output"]
        (out_dir / "output_tokens.json").write_text(
            json.dumps(output_tokens, indent=2),
            encoding="utf-8",
        )


def run_condition(
    runner: QwenVLRunner,
    record: dict[str, Any],
    condition: str,
    views: list[dict[str, Any]],
    out_root: Path,
    max_new_tokens: int,
    analysis_tokens: int,
    capture_layers: list[int] | None,
    force: bool,
) -> dict[str, Any]:
    spec = CONDITIONS[condition]
    cond_dir = out_root / condition
    if (cond_dir / "metadata.json").exists() and not force:
        return json.loads((cond_dir / "metadata.json").read_text(encoding="utf-8"))

    images = [item["image"] for item in views]
    view_names = [item["view"] for item in views]
    image_paths = [item["absolute_path"] for item in views]
    image_sizes = [item["size"] for item in views]
    analysis_text = None
    analysis_meta = None
    if spec["analysis_stage"]:
        analysis_prompt = spec["analysis_prompt"].format(instruction=record["instruction_text"])
        analysis_result = runner.generate(
            images,
            analysis_prompt,
            max_new_tokens=analysis_tokens,
            capture_layers=capture_layers,
            view_names=view_names,
        )
        analysis_text = analysis_result["answer"]
        analysis_dir = cond_dir / "analysis"
        save_generation(
            analysis_dir,
            analysis_result,
            {
                "stage": "analysis_generation",
                "knowledge_use": "none",
                "knowledge_kind": spec["analysis_stage"],
                "knowledge_source": "model_inferred",
                "not_supplied_expert_fact": True,
                "request_id": record["request_id"],
                "condition": condition,
                "n_images": len(views),
                "image_views": view_names,
            },
        )
        analysis_meta = {
            "stage": "analysis_generation",
            "text": analysis_text,
            "runtime_sec": analysis_result["runtime_sec"],
        }

    prompt = decision_prompt(
        record["instruction_text"],
        analysis=analysis_text,
        analysis_label=spec.get("analysis_label"),
    )
    result = runner.generate(
        images,
        prompt,
        max_new_tokens=max_new_tokens,
        capture_layers=capture_layers,
        view_names=view_names,
    )
    save_generation(
        cond_dir,
        result,
        {
            "stage": "decision",
            "condition": condition,
            "condition_label": spec["label"],
            "request_id": record["request_id"],
            "n_images": len(views),
            "image_views": view_names,
            "image_paths": image_paths,
            "image_sizes": image_sizes,
            "image_max_side": load_project_config()["image_max_side"],
            "knowledge_use": spec["analysis_stage"] or "none",
            "knowledge_kind": spec["analysis_stage"],
            "knowledge_source": "model_inferred" if spec["analysis_stage"] else None,
            "not_supplied_expert_fact": True,
            "analysis": analysis_meta,
            "evaluation_assets_excluded": True,
        },
    )
    return json.loads((cond_dir / "metadata.json").read_text(encoding="utf-8"))


def run_one_record(
    runner: QwenVLRunner,
    record: dict[str, Any],
    out_dir: Path,
    conditions: list[str],
    capture_layers: list[int] | None,
    force: bool,
    max_new_tokens: int,
    analysis_tokens: int,
) -> dict[str, Any]:
    if not record.get("pilot_eligible"):
        raise ValueError(f"Record {record['request_id']} is not pilot-eligible.")
    config = load_project_config()
    view_infos = collect_original_views(record, config.get("standard_views") or list(STANDARD_VIEWS))
    views = load_view_images(view_infos, config["image_max_side"])
    run_root = out_dir / record["request_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    image_dir = run_root / "input_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for item in views:
        item["image"].save(image_dir / f"{item['view']}.png")
    for condition in conditions:
        print(f"  [{record['request_id']}] {condition}", flush=True)
        run_condition(
            runner,
            record,
            condition,
            views,
            run_root,
            max_new_tokens=max_new_tokens,
            analysis_tokens=analysis_tokens,
            capture_layers=capture_layers,
            force=force,
        )
    summary = {
        "request_id": record["request_id"],
        "instruction_text": record["instruction_text"],
        "n_images": len(views),
        "image_views": [item["view"] for item in views],
        "image_paths": [item["absolute_path"] for item in views],
        "conditions": conditions,
        "output_dir": str(run_root),
        "model_revision": runner.revision,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_pilot(
    manifest_path: Path,
    request_id: str | None,
    out_dir: Path,
    conditions: list[str],
    capture_activations: bool = True,
    force: bool = False,
    max_new_tokens: int = 1024,
    analysis_tokens: int = 1024,
    run_all: bool = False,
    runner: QwenVLRunner | None = None,
    capture_layers: list[int] | None = None,
    extra_batch: dict[str, Any] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    config = load_project_config()
    records = load_jsonl(manifest_path)
    if run_all:
        chosen = [item for item in records if item.get("pilot_eligible")]
    elif request_id:
        chosen = [next(item for item in records if item["request_id"] == request_id)]
    else:
        chosen = [records[0]]

    runner = runner or QwenVLRunner()
    if capture_activations:
        layers = list(capture_layers) if capture_layers is not None else list(config["activation_layers"])
        for idx in layers:
            if idx < 0 or idx >= runner.n_layers:
                raise IndexError(
                    f"Activation layer {idx} is outside 0..{runner.n_layers - 1} "
                    f"for {runner.model_id} ({runner.n_layers} language layers)."
                )
    else:
        layers = None
    summaries = []
    for record in chosen:
        print(f"Running {record['request_id']}", flush=True)
        summaries.append(
            run_one_record(
                runner,
                record,
                out_dir,
                conditions,
                layers,
                force,
                max_new_tokens,
                analysis_tokens,
            )
        )
    if run_all:
        batch = {
            "count": len(summaries),
            "conditions": conditions,
            "manifest": str(manifest_path),
            "output_dir": str(out_dir),
            "model_id": runner.model_id,
            "model_revision": runner.revision,
            "quantization": runner.quantization,
            "n_language_layers": runner.n_layers,
            "hidden_size": runner.hidden_size,
            "activation_layers": layers,
            "request_ids": [item["request_id"] for item in summaries],
        }
        if extra_batch:
            batch.update(extra_batch)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "batch_summary.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
        return batch
    return summaries[0]
