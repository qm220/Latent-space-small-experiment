#!/usr/bin/env python3
"""Plot last-prompt-token PC1 against prompt length."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_last_token import CONDITIONS, overview_layers
from plot_last_token_pca3d import COLORS, LEGEND_LABELS, MARKERS

IMAGE_PAD = "<|image_pad|>"


def pearson(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.std() < 1e-12 or ya.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def collect_rows(run_root: Path, pca_path: Path) -> list[dict]:
    pca = json.loads(pca_path.read_text(encoding="utf-8"))
    layers = sorted(int(key) for key in pca["layers"])
    by_key: dict[tuple[str, str], dict] = {}
    for layer in layers:
        for point in pca["layers"][str(layer)]["points"]:
            key = (point["request_id"], point["condition"])
            entry = by_key.setdefault(
                key,
                {
                    "request_id": point["request_id"],
                    "label": point["label"],
                    "condition": point["condition"],
                },
            )
            entry[f"pc1_l{layer}"] = point["pc1"]
            entry[f"pc2_l{layer}"] = point["pc2"]
            entry[f"pc3_l{layer}"] = point["pc3"]
    rows = []
    for (request_id, cond), entry in sorted(by_key.items()):
        run_dir = run_root / request_id / cond
        meta = json.loads((run_dir / "activations.json").read_text(encoding="utf-8"))
        tokens = json.loads((run_dir / "tokens.json").read_text(encoding="utf-8"))
        prompt = [item for item in tokens if item.get("split") == "prompt"]
        n_image = sum(1 for item in prompt if item.get("token") == IMAGE_PAD)
        entry["prompt_length"] = int(meta["prompt_length"])
        entry["n_image_pad"] = n_image
        entry["n_text_tokens"] = len(prompt) - n_image
        rows.append(entry)
    direct_len = {row["request_id"]: row["prompt_length"] for row in rows if row["condition"] == "direct"}
    direct_text = {row["request_id"]: row["n_text_tokens"] for row in rows if row["condition"] == "direct"}
    for row in rows:
        row["extra_tokens"] = row["prompt_length"] - direct_len[row["request_id"]]
        row["extra_text_tokens"] = row["n_text_tokens"] - direct_text[row["request_id"]]
    return rows


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 50)
    return xs, slope * xs + intercept


def row_layers(rows: list[dict]) -> list[int]:
    keys = [key for key in rows[0] if key.startswith("pc1_l")]
    return sorted(int(key.split("pc1_l", 1)[1]) for key in keys)


def plot_pc1_vs_length(rows: list[dict], out_path: Path) -> None:
    layers = overview_layers(row_layers(rows))
    fig, axes = plt.subplots(1, len(layers), figsize=(4.4 * len(layers), 4.4), sharey=False)
    if len(layers) == 1:
        axes = [axes]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[c],
            color="none",
            markerfacecolor=COLORS[c],
            markersize=8,
            label=LEGEND_LABELS[c],
        )
        for c in CONDITIONS
    ]
    for ax, layer in zip(axes, layers):
        key = f"pc1_l{layer}"
        x = np.array([row["prompt_length"] for row in rows], dtype=float)
        y = np.array([row[key] for row in rows], dtype=float)
        r = pearson(x.tolist(), y.tolist())
        for cond in CONDITIONS:
            sub = [row for row in rows if row["condition"] == cond]
            ax.scatter(
                [row["prompt_length"] for row in sub],
                [row[key] for row in sub],
                c=COLORS[cond],
                marker=MARKERS[cond],
                s=28,
                zorder=3,
            )
        xs, ys = fit_line(x, y)
        ax.plot(xs, ys, color="#374151", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Prompt length (tokens, incl. 441 image pads)")
        ax.set_ylabel(f"Layer {layer} PC1")
        ax.set_title(f"Layer {layer}  ·  r = {r:.3f}")
        ax.grid(alpha=0.25)
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Last-prompt-token PC1 vs prompt length · 48 text requests", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pc1_vs_extra(rows: list[dict], out_path: Path) -> None:
    layers = overview_layers(row_layers(rows))
    fig, axes = plt.subplots(1, len(layers), figsize=(4.4 * len(layers), 4.4), sharey=False)
    if len(layers) == 1:
        axes = [axes]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[c],
            color="none",
            markerfacecolor=COLORS[c],
            markersize=8,
            label=LEGEND_LABELS[c],
        )
        for c in CONDITIONS
    ]
    for ax, layer in zip(axes, layers):
        key = f"pc1_l{layer}"
        x = np.array([row["extra_tokens"] for row in rows], dtype=float)
        y = np.array([row[key] for row in rows], dtype=float)
        r = pearson(x.tolist(), y.tolist())
        for cond in CONDITIONS:
            sub = [row for row in rows if row["condition"] == cond]
            ax.scatter(
                [row["extra_tokens"] for row in sub],
                [row[key] for row in sub],
                c=COLORS[cond],
                marker=MARKERS[cond],
                s=28,
                zorder=3,
            )
        xs, ys = fit_line(x, y)
        ax.plot(xs, ys, color="#374151", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Extra tokens vs that request’s direct prompt")
        ax.set_ylabel(f"Layer {layer} PC1")
        ax.set_title(f"Layer {layer}  ·  r = {r:.3f}")
        ax.grid(alpha=0.25)
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("PC1 vs extra prompt tokens (direct = 0)", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def correlations(rows: list[dict]) -> dict:
    report: dict = {"n_points": len(rows), "n_image_pad": sorted({row["n_image_pad"] for row in rows})}
    for layer in row_layers(rows):
        key = f"pc1_l{layer}"
        block = {
            "vs_prompt_length": pearson([row["prompt_length"] for row in rows], [row[key] for row in rows]),
            "vs_text_tokens": pearson([row["n_text_tokens"] for row in rows], [row[key] for row in rows]),
            "vs_extra_tokens": pearson([row["extra_tokens"] for row in rows], [row[key] for row in rows]),
            "within_condition": {},
        }
        for cond in CONDITIONS:
            sub = [row for row in rows if row["condition"] == cond]
            block["within_condition"][cond] = {
                "vs_prompt_length": pearson([row["prompt_length"] for row in sub], [row[key] for row in sub]),
                "n": len(sub),
                "prompt_length_min": min(row["prompt_length"] for row in sub),
                "prompt_length_max": max(row["prompt_length"] for row in sub),
                "prompt_length_mean": float(np.mean([row["prompt_length"] for row in sub])),
            }
        report[f"layer_{layer}"] = block
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PC1 against prompt length.")
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "pilot_text48")
    parser.add_argument(
        "--pca",
        type=Path,
        default=ROOT / "reports" / "last_token_pca3d_text48" / "pca3d_coordinates.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "pc1_vs_prompt_length")
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.root, args.pca)
    plot_pc1_vs_length(rows, out_dir / "pc1_vs_prompt_length.png")
    plot_pc1_vs_extra(rows, out_dir / "pc1_vs_extra_tokens.png")
    report = {
        "source_runs": str(args.root),
        "pca": str(args.pca),
        "note": "prompt_length includes a constant 441 image-pad tokens; text-token count is prompt_length - 441.",
        "correlations": correlations(rows),
        "points": rows,
    }
    (out_dir / "pc1_vs_prompt_length.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_dir), "correlations": report["correlations"]}, indent=2))


if __name__ == "__main__":
    main()
