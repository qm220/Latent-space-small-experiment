#!/usr/bin/env python3
"""Re-run PCA after removing original PC1 or regressing out prompt length.

Original 3-PC plots are left unchanged. Use --root/--out for a specific run folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_last_token import CONDITIONS, overview_layers
from plot_last_token_pca3d import (
    COLORS,
    LEGEND_LABELS,
    MARKERS,
    collect_vectors,
    plot_layer,
    plot_layer_2d,
    plot_pc2_vs_pc3_layers,
    plot_three_layers_together,
    report_layer_ids,
    write_interactive_html,
)
from plot_pc1_vs_prompt_length import pearson

NOTE = {
    "pc1": "after subtracting proj onto original PC1",
    "length": "after subtracting proj onto OLS prompt-length direction",
    "length_ols": "after LinearRegression of embeddings on prompt length",
}


def prompt_length(run_root: Path, request_id: str, condition: str) -> int:
    meta = json.loads((run_root / request_id / condition / "activations.json").read_text(encoding="utf-8"))
    return int(meta["prompt_length"])


def l2_unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def subtract_projection(centered: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = direction / np.clip(np.linalg.norm(direction), 1e-12, None)
    scores = centered @ v
    residual = centered - np.outer(scores, v)
    return residual, scores


def residualize(unit: np.ndarray, lengths: np.ndarray, method: str) -> dict:
    mean = unit.mean(axis=0)
    centered = unit - mean
    original_pca = PCA(n_components=3, random_state=42).fit(unit)
    pc1 = original_pca.components_[0]
    if method == "pc1":
        residual, scores = subtract_projection(centered, pc1)
        direction = pc1
        removed = "original_pc1"
        extra_stats: dict = {}
    elif method == "length":
        y = lengths - lengths.mean()
        direction = centered.T @ y
        residual, scores = subtract_projection(centered, direction)
        removed = "ols_prompt_length"
        extra_stats = {}
    elif method == "length_ols":
        # sklearn: each of the 4096 dims ~ intercept + slope * length, then PCA leftover.
        L = lengths.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(L, unit)
        predicted = reg.predict(L)
        residual = unit - predicted
        direction = np.asarray(reg.coef_, dtype=float).reshape(-1)
        scores = (predicted - reg.intercept_) @ (
            direction / np.clip(np.linalg.norm(direction), 1e-12, None)
        )
        removed = "ols_length_regression"
        extra_stats = {
            "ols_intercept_norm": float(np.linalg.norm(reg.intercept_)),
            "ols_coef_norm": float(np.linalg.norm(direction)),
            "ols_r2_uniform_average": float(reg.score(L, unit)),
        }
    else:
        raise ValueError(method)
    residual_pca = PCA(n_components=3, random_state=42).fit(residual)
    coords = residual_pca.transform(residual)
    orig_total = float(centered.var(axis=0).sum())
    res_centered = residual - residual.mean(axis=0)
    res_total = float(res_centered.var(axis=0).sum())
    direction_unit = direction / np.clip(np.linalg.norm(direction), 1e-12, None)
    out = {
        "removed": removed,
        "original_explained_variance_ratio": [float(v) for v in original_pca.explained_variance_ratio_],
        "residual_explained_variance_ratio": [float(v) for v in residual_pca.explained_variance_ratio_],
        "residual_explained_variance_sum": float(residual_pca.explained_variance_ratio_.sum()),
        "fraction_of_original_variance_remaining": res_total / orig_total if orig_total else 0.0,
        "removed_score_vs_length_r": pearson(np.asarray(scores).tolist(), lengths.tolist()),
        "coords": coords,
        "removed_scores": np.asarray(scores).tolist(),
        "direction_unit_norm": float(np.linalg.norm(direction_unit)),
        "pc1_vs_length_direction_cosine": float(np.abs(pc1 @ direction_unit)),
    }
    out.update(extra_stats)
    return out


def eta2(values: np.ndarray, groups: list[str]) -> float:
    values = np.asarray(values, float)
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    ss_b = 0.0
    for g in set(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        ss_b += len(idx) * (values[idx].mean() - grand) ** 2
    return float(ss_b / ss_tot)


def points_from_coords(entry: dict, coords: np.ndarray, extra: dict) -> dict:
    points = []
    for i, (req, cond, label) in enumerate(zip(entry["ids"], entry["conds"], entry["labels"])):
        points.append(
            {
                "request_id": req,
                "label": label,
                "condition": cond,
                "pc1": float(coords[i, 0]),
                "pc2": float(coords[i, 1]),
                "pc3": float(coords[i, 2]),
            }
        )
    n_requests = len(set(entry["ids"]))
    projected = {
        "n_requests": n_requests,
        "n_points": len(points),
        "axis_prefix": "rPC",
        "explained_variance_ratio": extra["residual_explained_variance_ratio"],
        "explained_variance_sum": extra["residual_explained_variance_sum"],
        "original_explained_variance_ratio": extra["original_explained_variance_ratio"],
        "fraction_of_original_variance_remaining": extra["fraction_of_original_variance_remaining"],
        "removed": extra["removed"],
        "removed_score_vs_length_r": extra["removed_score_vs_length_r"],
        "pc1_vs_length_direction_cosine": extra["pc1_vs_length_direction_cosine"],
        "points": points,
    }
    for key in ("ols_r2_uniform_average", "ols_coef_norm", "ols_intercept_norm"):
        if key in extra:
            projected[key] = extra[key]
    lengths = extra["lengths"]
    conds = extra["conds"]
    ids = extra["ids"]
    projected["residual_pc_vs_length_r"] = {
        f"rpc{i+1}": pearson(coords[:, i].tolist(), list(lengths)) for i in range(3)
    }
    projected["eta2"] = {}
    for i, name in enumerate(("pc1", "pc2", "pc3")):
        vals = coords[:, i]
        extra_only = [j for j, c in enumerate(conds) if c != "direct"]
        projected["eta2"][name] = {
            "condition": eta2(vals, conds),
            "request": eta2(vals, ids),
            "plan_vs_fbs_extra_only": eta2(vals[extra_only], [conds[j] for j in extra_only]),
        }
    return projected


def plot_rpc_vs_length(report: dict, lengths_by_key: dict[tuple[str, str], int], out_path: Path) -> None:
    layers = overview_layers(report_layer_ids(report))
    fig, axes = plt.subplots(1, len(layers), figsize=(4.4 * len(layers), 4.4))
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
        pts = report["layers"][str(layer)]["points"]
        r = report["layers"][str(layer)]["residual_pc_vs_length_r"]["rpc1"]
        for cond in CONDITIONS:
            sub = [p for p in pts if p["condition"] == cond]
            ax.scatter(
                [lengths_by_key[(p["request_id"], p["condition"])] for p in sub],
                [p["pc1"] for p in sub],
                c=COLORS[cond],
                marker=MARKERS[cond],
                s=28,
                zorder=3,
            )
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_ylabel("rPC1")
        ax.set_title(f"Layer {layer}  ·  r = {r:.3f}")
        ax.grid(alpha=0.25)
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Residual PC1 vs prompt length (length direction should be gone)", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_method(collected: dict, run_root: Path, method: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    note = NOTE[method]
    method_blurb = {
        "pc1": "original PC1",
        "length": "the OLS prompt-length direction (rank-1 projection)",
        "length_ols": "the LinearRegression fit of each embedding dim on prompt length",
    }
    report = {
        "method": (
            "L2-normalize last prompt-token vectors, subtract "
            f"{method_blurb[method]}, then PCA to 3 components on the residual. "
            "Original PC1 plots were not overwritten."
        ),
        "source_runs": str(run_root),
        "plot_title": f"Residual 3D PCA · {note} · 48 text requests",
        "layers": {},
    }
    lengths_by_key: dict[tuple[str, str], int] = {}
    for layer, entry in collected.items():
        matrix = np.stack(entry["vectors"], axis=0)
        unit = l2_unit(matrix)
        lengths = np.array(
            [prompt_length(run_root, req, cond) for req, cond in zip(entry["ids"], entry["conds"])],
            dtype=float,
        )
        for req, cond, length in zip(entry["ids"], entry["conds"], lengths):
            lengths_by_key[(req, cond)] = int(length)
        extra = residualize(unit, lengths, method)
        extra["lengths"] = lengths.tolist()
        extra["conds"] = list(entry["conds"])
        extra["ids"] = list(entry["ids"])
        projected = points_from_coords(entry, extra["coords"], extra)
        projected["axis_prefix"] = "rPC"
        report["layers"][str(layer)] = projected
        plot_layer(
            layer,
            projected,
            out_dir / f"layer_{layer}_pca3d.png",
            elev=22,
            azim=-55,
            note=note,
        )
        plot_layer(
            layer,
            projected,
            out_dir / f"layer_{layer}_pca3d_alt.png",
            elev=18,
            azim=130,
            note=note,
        )
        plot_layer_2d(layer, projected, out_dir / f"layer_{layer}_pca_faces.png", note=note)
    n = next(iter(report["layers"].values()))["n_requests"]
    report["n_requests"] = n
    report["n_points_per_layer"] = 3 * n
    plot_three_layers_together(report, out_dir / "layers_overview_pca3d.png")
    plot_three_layers_together(report, out_dir / "layers_8_17_35_pca3d.png")
    plot_pc2_vs_pc3_layers(report, out_dir / "pc2_vs_pc3_layers.png", note=note)
    write_interactive_html(report, out_dir / "layers_overview_pca3d.html")
    write_interactive_html(report, out_dir / "layers_8_17_35_pca3d.html")
    plot_rpc_vs_length(report, lengths_by_key, out_dir / "rpc1_vs_prompt_length.png")
    slim = {k: v for k, v in report.items()}
    (out_dir / "pca3d_coordinates.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA after removing PC1 or prompt-length direction.")
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "pilot_text48")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for a single --method. Required-style when not using default names.",
    )
    parser.add_argument(
        "--method",
        choices=("pc1", "length", "length_ols", "both"),
        default="both",
        help=(
            "pc1: subtract original PC1. "
            "length: project out the OLS length direction. "
            "length_ols: X := X - LinearRegression(length).predict (sklearn snippet). "
            "both: pc1 and length with default report names."
        ),
    )
    args = parser.parse_args()
    collected = collect_vectors(args.root)
    if not collected:
        raise SystemExit(f"No complete condition triples in {args.root}")
    methods = ["pc1", "length"] if args.method == "both" else [args.method]
    if args.out is not None and args.method == "both":
        raise SystemExit("--out cannot be used with --method both; run pc1, length, or length_ols separately.")
    summary = {"original_kept": str(ROOT / "reports" / "last_token_pca3d_text48"), "source_runs": str(args.root), "runs": {}}
    for method in methods:
        out_dir = args.out if args.out is not None else ROOT / "reports" / f"last_token_pca3d_text48_residual_{method}"
        report = run_method(collected, args.root, method, out_dir)
        layer_ids = report_layer_ids(report)
        summary["runs"][method] = {
            "wrote": str(out_dir),
            "explained_residual": {
                layer: report["layers"][str(layer)]["explained_variance_ratio"] for layer in layer_ids
            },
            "original_variance_remaining": {
                layer: report["layers"][str(layer)]["fraction_of_original_variance_remaining"]
                for layer in layer_ids
            },
            "rpc1_vs_length_r": {
                layer: report["layers"][str(layer)]["residual_pc_vs_length_r"]["rpc1"] for layer in layer_ids
            },
            "eta2_rpc1_condition": {
                layer: report["layers"][str(layer)]["eta2"]["pc1"]["condition"] for layer in layer_ids
            },
            "ols_r2": {
                layer: report["layers"][str(layer)].get("ols_r2_uniform_average") for layer in layer_ids
            },
            "pc1_vs_removed_direction_cosine": {
                layer: report["layers"][str(layer)]["pc1_vs_length_direction_cosine"] for layer in layer_ids
            },
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
