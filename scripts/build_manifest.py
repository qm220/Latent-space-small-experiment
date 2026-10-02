#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_inv.dataset.manifest import (
    build_record,
    default_manifest_path,
    select_pilot_ids,
    select_random_ids,
    write_jsonl,
)
from knowledge_inv.dataset.readonly_db import ReadOnlyNeuralCAD
from knowledge_inv.paths import load_project_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a verified text-request pilot manifest.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--all-eligible", action="store_true", help="Include every eligible text request.")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=default_manifest_path())
    args = parser.parse_args()

    config = load_project_config()
    with ReadOnlyNeuralCAD(config) as db:
        users = db.user_map()
        records = [build_record(db, req, users) for req in db.requests.find()]

    eligible = [rec for rec in records if rec["pilot_eligible"]]
    if args.all_eligible:
        selected_ids = [rec["request_id"] for rec in sorted(eligible, key=lambda item: item["request_id"])]
        selection = f"all {len(selected_ids)} eligible text requests"
    elif args.random:
        selected_ids = select_random_ids(eligible, limit=args.limit, seed=args.seed)
        selection = (
            f"random sample of {args.limit} eligible text requests "
            f"(seed={args.seed})"
        )
    else:
        selected_ids = select_pilot_ids(eligible, config["primary_example_request_id"], limit=args.limit)
        selection = "text modality with nonempty requests.text, all original views, and original STEP"
    selected = [next(rec for rec in eligible if rec["request_id"] == rid) for rid in selected_ids]
    write_jsonl(selected, args.out)
    meta = {
        "path": str(args.out),
        "count": len(selected),
        "eligible_text_requests": len(eligible),
        "total_requests_inspected": len(records),
        "selected_ids": selected_ids,
        "random": args.random,
        "all_eligible": args.all_eligible,
        "seed": args.seed if args.random else None,
        "selection": selection,
        "evaluation_only": True,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
