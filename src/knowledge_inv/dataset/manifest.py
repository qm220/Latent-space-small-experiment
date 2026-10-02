from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from knowledge_inv.dataset.readonly_db import STANDARD_VIEWS, ReadOnlyNeuralCAD
from knowledge_inv.paths import PROJECT_ROOT, load_project_config

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def build_record(db: ReadOnlyNeuralCAD, request: dict, users: dict[str, dict]) -> dict[str, Any]:
    info = db.instruction_fields(request)
    assets = db.brep_assets(request["brep_start"])
    edits = [db.classify_edit(edit, request, users) for edit in db.request_edits(request["_id"])]
    missing = list(assets.get("missing") or [])
    if not info["genuine_text_instruction"]:
        missing.append("no_genuine_text_instruction")
        if info["has_transcript"]:
            missing.append("transcript_available_but_not_used_as_instruction")

    images = {}
    for view in STANDARD_VIEWS:
        item = assets.get("images", {}).get(view)
        if item:
            images[view] = item

    step = (assets.get("cad") or {}).get("step")
    return {
        "request_id": request["_id"],
        "original_model_id": request.get("brep_start"),
        "request_modality": request.get("modality"),
        "request_type": request.get("request_type"),
        "difficulty": request.get("difficulty"),
        "assembly": request.get("assembly"),
        "parametric": request.get("parametric"),
        "source": request.get("source"),
        "instruction_text": info["instruction_text"],
        "instruction_source": info["instruction_source"],
        "original_image_paths": images,
        "original_cad_path": step,
        "original_cad_formats": assets.get("cad"),
        "reference_edits": edits,
        "evaluation_only": {
            "reference_edits": True,
            "edited_images": True,
            "ratings": True,
            "note": "Do not supply these to the model during decision or analysis generation.",
        },
        "provenance": {
            "dataset": "autodesk/neuralCAD-Edit",
            "archive": "edit_192_external",
            "storage_dir": str(db.root),
            "db_name": db.db_name,
            "official_repo": "https://github.com/AutodeskAILab/neuralCAD-Edit",
            "official_commit": load_project_config().get("official_commit"),
            "request_user": request.get("user"),
            "filename": request.get("filename"),
        },
        "missing_data_notes": missing,
        "pilot_eligible": bool(
            info["genuine_text_instruction"]
            and step
            and step.get("exists")
            and all(images.get(view, {}).get("exists") for view in STANDARD_VIEWS)
        ),
    }


def select_random_ids(records: list[dict[str, Any]], limit: int = 8, seed: int = 42) -> list[str]:
    import random

    eligible = [rec for rec in records if rec.get("pilot_eligible")]
    ids = [rec["request_id"] for rec in eligible]
    if limit > len(ids):
        raise ValueError(f"Requested {limit} random records but only {len(ids)} are eligible.")
    rng = random.Random(seed)
    return rng.sample(ids, limit)


def select_pilot_ids(records: list[dict[str, Any]], primary_id: str, limit: int = 8) -> list[str]:
    eligible = [rec for rec in records if rec.get("pilot_eligible")]
    by_id = {rec["request_id"]: rec for rec in eligible}
    selected: list[str] = []
    if primary_id in by_id:
        selected.append(primary_id)

    buckets: dict[tuple, list[dict]] = {}
    for rec in eligible:
        key = (rec["difficulty"], bool(rec["assembly"]), bool(rec["parametric"]))
        buckets.setdefault(key, []).append(rec)
    for recs in buckets.values():
        recs.sort(key=lambda item: (len(item.get("instruction_text") or ""), item["request_id"]))

    # Deterministic coverage across difficulty / assembly / parametric.
    for difficulty in ("easy", "medium", "hard"):
        for assembly in (False, True):
            for parametric in (False, True):
                pool = buckets.get((difficulty, assembly, parametric), [])
                for rec in pool:
                    if rec["request_id"] not in selected:
                        selected.append(rec["request_id"])
                        break
                if len(selected) >= limit:
                    return selected[:limit]

    for rec in sorted(
        eligible,
        key=lambda item: (
            DIFFICULTY_ORDER.get(item["difficulty"], 9),
            len(item.get("instruction_text") or ""),
            item["request_id"],
        ),
    ):
        if rec["request_id"] not in selected:
            selected.append(rec["request_id"])
        if len(selected) >= limit:
            break
    return selected[:limit]


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def default_manifest_path() -> Path:
    return PROJECT_ROOT / "manifests" / "pilot_text_v1.jsonl"
