from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from knowledge_inv.paths import PROJECT_ROOT, load_project_config, storage_dir


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return (result.stdout or result.stderr or "").strip()
    except FileNotFoundError:
        return ""


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    import sys

    config = load_project_config()
    data_root = storage_dir(config)
    disk = shutil.disk_usage(PROJECT_ROOT)
    env = {
        "wsl": {
            "uname": platform.uname()._asdict(),
            "os_release": Path("/etc/os-release").read_text(encoding="utf-8")
            if Path("/etc/os-release").exists()
            else "",
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
        },
        "packages": {
            name: _package_version(name)
            for name in [
                "torch",
                "transformers",
                "accelerate",
                "safetensors",
                "huggingface_hub",
                "pillow",
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "mongita",
                "jupyterlab",
                "ipykernel",
            ]
        },
        "disk": {
            "total_gb": round(disk.total / 1024**3, 2),
            "used_gb": round(disk.used / 1024**3, 2),
            "free_gb": round(disk.free / 1024**3, 2),
        },
        "nvidia_smi": _run(["nvidia-smi"]),
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "storage_dir": str(data_root),
            "storage_exists": data_root.is_dir(),
            "official_repo": str(PROJECT_ROOT / config["official_repo"]),
        },
        "cuda_test": {},
    }

    try:
        import torch

        env["cuda_test"] = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            left = torch.randn(1024, 1024, device="cuda")
            right = torch.randn(1024, 1024, device="cuda")
            product = left @ right
            env["cuda_test"].update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "total_vram_gb": round(props.total_memory / 1024**3, 2),
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                    "matmul_ok": bool(torch.isfinite(product).all().item()),
                    "matmul_shape": list(product.shape),
                }
            )
    except Exception as exc:
        env["cuda_test"]["error"] = f"{type(exc).__name__}: {exc}"
    return env


def render_environment_markdown(report: dict[str, Any]) -> str:
    python = report["python"]
    cuda = report.get("cuda_test", {})
    lines = [
        "# Environment verification",
        "",
        f"- Python: `{python['executable']}`",
        f"- Version: `{python['version'].splitlines()[0]}`",
        f"- VIRTUAL_ENV: `{python['virtual_env']}`",
        f"- CONDA_DEFAULT_ENV: `{python['conda_default_env']}`",
        f"- Disk free: `{report['disk']['free_gb']} GB`",
        f"- Dataset present: `{report['paths']['storage_exists']}`",
        "",
        "## Packages",
    ]
    for name, version in report["packages"].items():
        lines.append(f"- {name}: `{version}`")
    lines.extend(["", "## CUDA test"])
    for key, value in cuda.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)
