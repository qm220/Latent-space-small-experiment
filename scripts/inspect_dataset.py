#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_inv.dataset.inspect import inspect_dataset, write_report
from knowledge_inv.dataset.readonly_db import ReadOnlyNeuralCAD
from knowledge_inv.paths import PROJECT_ROOT


def main() -> None:
    with ReadOnlyNeuralCAD() as db:
        report = inspect_dataset(db)
    json_path, md_path = write_report(report, PROJECT_ROOT / "reports")
    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
