from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from knowledge_inv.dataset.readonly_db import STANDARD_VIEWS, ReadOnlyNeuralCAD
from knowledge_inv.paths import PROJECT_ROOT


def _count_by(docs: list[dict], field: str) -> dict[str, int]:
    counter: collections.Counter[str] = collections.Counter()
    missing = 0
    for doc in docs:
        if field not in doc:
            missing += 1
        else:
            counter[str(doc[field])] += 1
    out = dict(counter)
    if missing:
        out["<missing>"] = missing
    return out


def inspect_dataset(db: ReadOnlyNeuralCAD) -> dict[str, Any]:
    requests = list(db.requests.find())
    users = db.user_map()
    edits = list(db.edits.find())
    brep_count = db.breps.count_documents({})
    rating_count = db.ratings.count_documents({})

    request_keys = collections.Counter()
    nonempty_keys = collections.Counter()
    for req in requests:
        for key, value in req.items():
            request_keys[key] += 1
            if value not in (None, "", [], {}):
                nonempty_keys[key] += 1

    start_image_ok = 0
    start_step_ok = 0
    unresolved: list[dict[str, Any]] = []
    for req in requests:
        assets = db.brep_assets(req["brep_start"])
        views_ok = all(
            assets.get("images", {}).get(view, {}).get("exists") for view in STANDARD_VIEWS
        )
        step = (assets.get("cad") or {}).get("step")
        if views_ok:
            start_image_ok += 1
        if step and step.get("exists"):
            start_step_ok += 1
        if assets.get("missing"):
            unresolved.append(
                {
                    "request_id": req["_id"],
                    "brep_start": req.get("brep_start"),
                    "missing": assets["missing"][:8],
                }
            )

    text_ready = []
    for req in requests:
        info = db.instruction_fields(req)
        assets = db.brep_assets(req["brep_start"])
        views_ok = all(
            assets.get("images", {}).get(view, {}).get("exists") for view in STANDARD_VIEWS
        )
        step = (assets.get("cad") or {}).get("step")
        if info["genuine_text_instruction"] and views_ok and step and step.get("exists"):
            text_ready.append(req["_id"])

    edits_per_request = collections.Counter(
        db.edits.count_documents({"request": req["_id"]}) for req in requests
    )
    representative = next((r for r in requests if r.get("modality") == "text"), requests[0])
    representative_keys = sorted(representative.keys())

    return {
        "storage_dir": str(db.root),
        "mongita_db": str(db.db_path),
        "db_name": db.db_name,
        "official_repo": str(PROJECT_ROOT / "external" / "neuralCAD-Edit"),
        "collection_counts": {
            "users": len(users),
            "breps": brep_count,
            "requests": len(requests),
            "edits": len(edits),
            "ratings": rating_count,
        },
        "request_modalities": _count_by(requests, "modality"),
        "request_difficulty": _count_by(requests, "difficulty"),
        "request_type": _count_by(requests, "request_type"),
        "assembly": _count_by(requests, "assembly"),
        "parametric": _count_by(requests, "parametric"),
        "instruction_fields": {
            "text_nonempty": sum(1 for r in requests if (r.get("text") or "").strip()),
            "prompt_nonempty": sum(1 for r in requests if r.get("prompt") not in (None, "", [])),
            "instructions_nonempty": sum(
                1 for r in requests if r.get("instructions") not in (None, "", [])
            ),
            "transcript_present": sum(1 for r in requests if r.get("transcript")),
            "corrected_transcript_present": sum(
                1 for r in requests if r.get("corrected_transcript_segments")
            ),
        },
        "representative_request_keys": representative_keys,
        "original_model_views": list(STANDARD_VIEWS),
        "original_cad_formats": ["step", "stl", "f3d", "smt"],
        "requests_with_all_original_views": start_image_ok,
        "requests_with_original_step": start_step_ok,
        "text_requests_with_images_and_step": text_ready,
        "text_pilot_eligible_count": len(text_ready),
        "edits_per_request": {str(k): v for k, v in sorted(edits_per_request.items())},
        "unresolved_asset_examples": unresolved[:12],
        "unresolved_asset_count": len(unresolved),
        "notes": [
            "This archive is the full 192-request / 384-human-edit release, not the 48-case hackathon slice.",
            "The 48 text-modality requests are one of four equal modality groups inside the 192-request set.",
            "requests.prompt and requests.instructions are empty. Genuine typed instructions live in requests.text for modality=text only.",
            "Non-text requests have spoken transcripts. Those are not treated as typed instructions in the first pilot.",
            "Reference edits, edited images, and ratings are evaluation-only and must not be shown to the model.",
            "Official DatabaseManager was not used because it creates extra folders under the dataset root.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset inspection",
        "",
        f"- Storage dir: `{report['storage_dir']}`",
        f"- Mongita dir: `{report['mongita_db']}`",
        f"- Database name: `{report['db_name']}`",
        "",
        "## Collection counts",
    ]
    for key, value in report["collection_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Request fields"])
    lines.append(f"- Modalities: `{report['request_modalities']}`")
    lines.append(f"- Difficulty: `{report['request_difficulty']}`")
    lines.append(f"- Request type: `{report['request_type']}`")
    lines.append(f"- Instruction field occupancy: `{report['instruction_fields']}`")
    lines.append(f"- Representative keys: `{report['representative_request_keys']}`")
    lines.extend(["", "## Assets"])
    lines.append(f"- Standard original views: `{report['original_model_views']}`")
    lines.append(f"- CAD formats: `{report['original_cad_formats']}`")
    lines.append(
        f"- Requests with all original views: {report['requests_with_all_original_views']}"
    )
    lines.append(f"- Requests with original STEP: {report['requests_with_original_step']}")
    lines.append(
        f"- Text requests eligible for the first pilot: {report['text_pilot_eligible_count']}"
    )
    lines.append(f"- Edits per request: `{report['edits_per_request']}`")
    lines.append(f"- Unresolved asset records (any missing CAD field/file): {report['unresolved_asset_count']}")
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset_inspection.json"
    md_path = out_dir / "dataset_inspection.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
