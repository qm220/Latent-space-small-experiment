#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_inv.env_check import collect_environment, render_environment_markdown
from knowledge_inv.paths import PROJECT_ROOT


def main() -> None:
    report = collect_environment()
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "environment.json"
    md_path = out_dir / "environment.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_environment_markdown(report), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    cuda = report.get("cuda_test", {})
    if not cuda.get("matmul_ok"):
        raise SystemExit("CUDA matrix multiply did not succeed.")
    if not report["paths"]["storage_exists"]:
        raise SystemExit(f"Dataset root missing: {report['paths']['storage_dir']}")
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
