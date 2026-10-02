#!/usr/bin/env python3
"""Snapshot POD of last prompt-token activations (same math as PCA / SVD).

Classic POD figures: modal energy, cumulative energy, coefficient faces,
and a sample × mode coefficient heatmap. Original PCA reports are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_last_token import CONDITIONS, layers_from_run_root, overview_layers
from plot_last_token_pca3d import (
    COLORS,
    collect_vectors,
    plot_layer_2d,
    plot_pc2_vs_pc3_layers,
)

N_ENERGY = 40
N_HEATMAP = 12


def l2_unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def modes_for(cumulative: np.ndarray, threshold: float) -> int:
    hits = np.where(cumulative >= threshold)[0]
    return int(hits[0] + 1) if len(hits) else int(len(cumulative))


def snapshot_pod(unit: np.ndarray) -> dict:
    """Center, then SVD. Coefficients a = U S; spatial/feature modes = V^T rows."""
    centered = unit - unit.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    energy = s**2
    total = float(energy.sum())
    frac = energy / total
    return {
        "singular_values": s,
        "energy_frac": frac,
        "cumulative": np.cumsum(frac),
        "coeffs": u * s,
        "total_energy": total,
        "rank": int(s.shape[0]),
    }


def subtract_leading_mode(unit: np.ndarray) -> np.ndarray:
    centered = unit - unit.mean(axis=0)
    direction = np.linalg.svd(centered, full_matrices=False)[2][0]
    return centered - np.outer(centered @ direction, direction)


def points_from_coeffs(entry: dict, coeffs: np.ndarray, energy_frac: np.ndarray, prefix: str) -> dict:
    points = []
    for i, (req, cond, label) in enumerate(zip(entry["ids"], entry["conds"], entry["labels"])):
        points.append(
            {
                "request_id": req,
                "label": label,
                "condition": cond,
                "pc1": float(coeffs[i, 0]),
                "pc2": float(coeffs[i, 1]),
                "pc3": float(coeffs[i, 2]),
            }
        )
    n_requests = len(set(entry["ids"]))
    return {
        "n_requests": n_requests,
        "n_points": len(points),
        "axis_prefix": prefix,
        "explained_variance_ratio": [float(v) for v in energy_frac[:3]],
        "explained_variance_sum": float(energy_frac[:3].sum()),
        "points": points,
    }


def summarize_pod(pod: dict, leftover_of_original: float | None = None) -> dict:
    cum = pod["cumulative"]
    frac = pod["energy_frac"]
    out = {
        "rank": pod["rank"],
        "total_energy": pod["total_energy"],
        "mode_energy_percent": [round(float(v) * 100, 4) for v in frac[:N_ENERGY]],
        "cumulative_percent": [round(float(v) * 100, 4) for v in cum[:N_ENERGY]],
        "modes_for_50pct": modes_for(cum, 0.50),
        "modes_for_80pct": modes_for(cum, 0.80),
        "modes_for_90pct": modes_for(cum, 0.90),
        "modes_for_95pct": modes_for(cum, 0.95),
        "first3_percent": [round(float(v) * 100, 2) for v in frac[:3]],
        "first3_sum_percent": round(float(frac[:3].sum()) * 100, 2),
    }
    if leftover_of_original is not None:
        out["leftover_percent_of_original"] = round(leftover_of_original * 100, 2)
        out["mode_energy_percent_of_original"] = [
            round(float(v) * leftover_of_original * 100, 4) for v in frac[:N_ENERGY]
        ]
    return out


def plot_energy(layers: dict[int, dict], out_path: Path, title: str, leftover: bool) -> None:
    ids = overview_layers(sorted(layers))
    fig, axes = plt.subplots(1, len(ids), figsize=(4.8 * len(ids), 4.6), sharey=True)
    if len(ids) == 1:
        axes = [axes]
    x = np.arange(1, N_ENERGY + 1)
    for ax, layer in zip(axes, ids):
        frac = layers[layer]["energy_frac"][:N_ENERGY] * 100
        cum = layers[layer]["cumulative"][:N_ENERGY] * 100
        ax.bar(x, frac, color="#334155", width=0.8, label="Mode energy")
        ax.plot(x, cum, color="#b45309", linewidth=1.6, label="Cumulative")
        ax.axhline(90, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("POD mode")
        ax.set_xlim(0.5, N_ENERGY + 0.5)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
        k90 = modes_for(layers[layer]["cumulative"], 0.90)
        text_x = 18 if k90 > 32 else min(k90 + 5, N_ENERGY - 2)
        ax.annotate(
            f"90% by mode {k90}",
            xy=(min(k90, N_ENERGY), 90),
            xytext=(text_x, 68),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.8),
        )
    axes[0].set_ylabel("% of leftover energy" if leftover else "% of total energy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_coefficient_heatmap(entries: dict[int, dict], pods: dict[int, dict], out_path: Path) -> None:
    ids = overview_layers(sorted(entries))
    fig, axes = plt.subplots(1, len(ids), figsize=(4.8 * len(ids), 6.2))
    if len(ids) == 1:
        axes = [axes]
    cond_order = list(CONDITIONS)
    for ax, layer in zip(axes, ids):
        entry = entries[layer]
        coeffs = pods[layer]["coeffs"][:, :N_HEATMAP]
        order = []
        row_colors = []
        for cond in cond_order:
            idx = [i for i, c in enumerate(entry["conds"]) if c == cond]
            order.extend(idx)
            row_colors.extend([COLORS[cond]] * len(idx))
        mat = coeffs[order]
        vmax = np.percentile(np.abs(mat), 99)
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            interpolation="nearest",
        )
        for i, color in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle((-0.5, i - 0.5), 0.18, 1.0, color=color, clip_on=False, zorder=5)
            )
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("POD mode")
        ax.set_xticks(np.arange(N_HEATMAP))
        ax.set_xticklabels([str(i + 1) for i in range(N_HEATMAP)])
        ax.set_yticks([23.5, 71.5, 119.5])
        ax.set_yticklabels(["direct", "plan", "FBS"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Snapshots (48 per condition)")
    handles = [
        Patch(facecolor=COLORS[k], label=name)
        for k, name in (("direct", "direct"), ("plan", "plan"), ("fbs", "FBS"))
    ]
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("POD coefficients · 144 last-prompt tokens × first 12 modes", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot POD plots of last prompt-token activations.")
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "pilot_text48")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "last_token_pod_text48")
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    collected = collect_vectors(args.root)
    if not collected:
        raise SystemExit(f"No complete condition triples in {args.root}")

    original_pods: dict[int, dict] = {}
    residual_pods: dict[int, dict] = {}
    orig_report = {"layers": {}, "n_requests": None}
    res_report = {"layers": {}, "n_requests": None}
    summary = {
        "method": (
            "Snapshot POD: L2-normalize last prompt-token vectors, subtract the mean, "
            "SVD. Mode energy = s_k^2 / sum(s^2). Coefficients a = U S. "
            "This is the same decomposition as PCA; mode 1 is original PC1."
        ),
        "source_runs": str(args.root),
        "n_snapshots": None,
        "layers": {},
    }

    for layer, entry in collected.items():
        unit = l2_unit(np.stack(entry["vectors"], axis=0))
        orig = snapshot_pod(unit)
        leftover = subtract_leading_mode(unit)
        resid = snapshot_pod(leftover)
        leftover_frac = resid["total_energy"] / orig["total_energy"]
        original_pods[layer] = orig
        residual_pods[layer] = resid

        orig_pts = points_from_coeffs(entry, orig["coeffs"], orig["energy_frac"], "a")
        res_pts = points_from_coeffs(entry, resid["coeffs"], resid["energy_frac"], "a")
        orig_report["layers"][str(layer)] = orig_pts
        res_report["layers"][str(layer)] = res_pts
        orig_report["n_requests"] = orig_pts["n_requests"]
        res_report["n_requests"] = res_pts["n_requests"]
        summary["n_snapshots"] = orig_pts["n_points"]
        summary["layers"][str(layer)] = {
            "original": summarize_pod(orig),
            "after_removing_mode1": summarize_pod(resid, leftover_of_original=leftover_frac),
        }
        plot_layer_2d(layer, orig_pts, out_dir / f"layer_{layer}_pod_faces.png")
        plot_layer_2d(
            layer,
            res_pts,
            out_dir / f"layer_{layer}_pod_residual_faces.png",
            note="after removing mode 1",
        )

    plot_energy(
        original_pods,
        out_dir / "pod_energy_spectrum.png",
        "POD energy spectrum · last prompt token · 48 text requests × 3 conditions",
        leftover=False,
    )
    plot_energy(
        residual_pods,
        out_dir / "pod_energy_residual.png",
        "POD energy after removing mode 1 (prompt-length / extra text)",
        leftover=True,
    )
    plot_pc2_vs_pc3_layers(orig_report, out_dir / "pod_a2_vs_a3.png")
    plot_pc2_vs_pc3_layers(res_report, out_dir / "pod_residual_a2_vs_a3.png", note="after removing mode 1")
    plot_coefficient_heatmap(collected, original_pods, out_dir / "pod_coefficient_heatmap.png")
    (out_dir / "pod_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "wrote": str(out_dir),
            "n_snapshots": summary["n_snapshots"],
            "original_first3_percent": {
                layer: summary["layers"][str(layer)]["original"]["first3_percent"]
                for layer in summary["layers"]
            },
            "residual_first3_percent_of_leftover": {
                layer: summary["layers"][str(layer)]["after_removing_mode1"]["first3_percent"]
                for layer in summary["layers"]
            },
            "modes_for_90pct": {
                layer: summary["layers"][str(layer)]["original"]["modes_for_90pct"]
                for layer in summary["layers"]
            },
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
