#!/usr/bin/env python3
"""Project last-prompt-token activations to 3D with PCA and plot directions."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_last_token import CONDITIONS, last_prompt_vector, layers_from_run_root, overview_layers

LABELS = {
    "3YH2WFSRM22W7DKT_1769175629.871246": "plant pots",
    "4E22SJE7QVTSP2E3_1758737060.465216": "slot through",
    "4E22SJE7QVTSP2E3_1758737118.288898": "insert screws",
    "4E22SJE7QVTSP2E3_1758737238.499021": "fixation rod",
    "4S7JQK6ZQMAD25GL_1758863286.214549": "click buttons",
    "F332D3FXML85WLR2_1769424096.996583": "edge rounds",
    "F332D3FXML85WLR2_1770205912.1187751": "add flange",
    "ZK22J6VYRKQ2RTFD_1758874422.1403751": "connecting hole",
}
COLORS = {"direct": "#1d4ed8", "plan": "#0f766e", "fbs": "#b45309"}
MARKERS = {"direct": "o", "plan": "s", "fbs": "^"}
LEGEND_LABELS = {
    "direct": "Blue circle: direct (image + request only)",
    "plan": "Green square: plan, then decide",
    "fbs": "Orange triangle: generated FBS, then decide",
}


def short_label(request_id: str, instruction: str, used: dict[str, str]) -> str:
    if request_id in LABELS:
        return LABELS[request_id]
    words = re.sub(r"\s+", " ", (instruction or "").strip())
    if not words:
        return request_id[:10]
    clip = words[:36].rstrip(" ,.;:")
    label = clip if clip not in used.values() else f"{clip} ({request_id[-4:]})"
    used[request_id] = label
    return label


def collect_vectors(root: Path) -> dict[int, dict]:
    payload = {}
    used_labels: dict[str, str] = {}
    layers = layers_from_run_root(root)
    for request_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not all((request_dir / cond / "activations.pt").exists() for cond in CONDITIONS):
            continue
        instruction = ""
        summary_path = request_dir / "summary.json"
        if summary_path.exists():
            instruction = json.loads(summary_path.read_text(encoding="utf-8")).get("instruction_text", "")
        label = short_label(request_dir.name, instruction, used_labels)
        for layer in layers:
            payload.setdefault(layer, {"ids": [], "conds": [], "vectors": [], "labels": []})
            for cond in CONDITIONS:
                vec = last_prompt_vector(request_dir / cond, layer).numpy()
                payload[layer]["ids"].append(request_dir.name)
                payload[layer]["conds"].append(cond)
                payload[layer]["vectors"].append(vec)
                payload[layer]["labels"].append(label)
    return payload


def report_layer_ids(report: dict) -> list[int]:
    return sorted(int(key) for key in report["layers"])


def project_layer(entry: dict) -> dict:
    matrix = np.stack(entry["vectors"], axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.clip(norms, 1e-8, None)
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(unit)
    points = []
    n_requests = len(set(entry["ids"]))
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
    return {
        "n_requests": n_requests,
        "n_points": len(points),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "points": points,
    }


def _style_for_count(n_requests: int) -> dict:
    dense = n_requests > 12
    return {
        "marker_size": 18 if dense else 55,
        "line_width": 0.5 if dense else 1.0,
        "line_alpha": 0.28 if dense else 0.7,
        "annotate": not dense,
    }


def _draw_3d(ax, projected: dict, elev: float, azim: float, title: str) -> None:
    style = _style_for_count(projected.get("n_requests", len(projected["points"]) // 3))
    by_req: dict[str, dict[str, dict]] = {}
    for point in projected["points"]:
        by_req.setdefault(point["request_id"], {})[point["condition"]] = point
        ax.scatter(
            point["pc1"],
            point["pc2"],
            point["pc3"],
            c=COLORS[point["condition"]],
            marker=MARKERS[point["condition"]],
            s=style["marker_size"],
            depthshade=False,
        )
    for conds in by_req.values():
        direct, plan, fbs = conds["direct"], conds["plan"], conds["fbs"]
        ax.plot(
            [direct["pc1"], plan["pc1"]],
            [direct["pc2"], plan["pc2"]],
            [direct["pc3"], plan["pc3"]],
            color=COLORS["plan"],
            linewidth=style["line_width"],
            alpha=style["line_alpha"],
        )
        ax.plot(
            [direct["pc1"], fbs["pc1"]],
            [direct["pc2"], fbs["pc2"]],
            [direct["pc3"], fbs["pc3"]],
            color=COLORS["fbs"],
            linewidth=style["line_width"],
            alpha=style["line_alpha"],
        )
        if style["annotate"]:
            ax.text(direct["pc1"], direct["pc2"], direct["pc3"], f"  {direct['label']}", fontsize=7)
    prefix = projected.get("axis_prefix", "PC")
    ev = projected["explained_variance_ratio"]
    ax.set_xlabel(f"{prefix}1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"{prefix}2 ({ev[1]*100:.1f}%)")
    ax.set_zlabel(f"{prefix}3 ({ev[2]*100:.1f}%)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)


def plot_layer(layer: int, projected: dict, out_path: Path, elev: float, azim: float, note: str = "") -> None:
    fig = plt.figure(figsize=(8.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    n = projected.get("n_requests", len(projected["points"]) // 3)
    prefix = projected.get("axis_prefix", "PC")
    if note:
        title = (
            f"Layer {layer} {note}\n"
            f"{n} requests × 3 conditions · "
            f"{projected['explained_variance_sum']*100:.1f}% residual variance in 3 {prefix}s"
        )
    else:
        title = (
            f"Layer {layer} last prompt token in 3D PCA\n"
            f"{n} requests × 3 conditions · L2-normalized · "
            f"{projected['explained_variance_sum']*100:.1f}% variance in 3 PCs"
        )
    _draw_3d(ax, projected, elev, azim, title)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[c],
            color="none",
            markerfacecolor=COLORS[c],
            markersize=10,
            label=LEGEND_LABELS[c],
        )
        for c in CONDITIONS
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=True,
        fontsize=8,
        title="Condition (not placed by hand)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_layer_2d(layer: int, projected: dict, out_path: Path, note: str = "") -> None:
    style = _style_for_count(projected.get("n_requests", len(projected["points"]) // 3))
    ev = projected["explained_variance_ratio"]
    prefix = projected.get("axis_prefix", "PC")
    faces = (("pc1", "pc2", 0, 1), ("pc1", "pc3", 0, 2), ("pc2", "pc3", 1, 2))
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2))
    by_req: dict[str, dict[str, dict]] = {}
    for point in projected["points"]:
        by_req.setdefault(point["request_id"], {})[point["condition"]] = point
    for ax, (xk, yk, xi, yi) in zip(axes, faces):
        for conds in by_req.values():
            direct, plan, fbs = conds["direct"], conds["plan"], conds["fbs"]
            ax.plot(
                [direct[xk], plan[xk]],
                [direct[yk], plan[yk]],
                color=COLORS["plan"],
                linewidth=style["line_width"],
                alpha=style["line_alpha"],
            )
            ax.plot(
                [direct[xk], fbs[xk]],
                [direct[yk], fbs[yk]],
                color=COLORS["fbs"],
                linewidth=style["line_width"],
                alpha=style["line_alpha"],
            )
        for point in projected["points"]:
            ax.scatter(
                point[xk],
                point[yk],
                c=COLORS[point["condition"]],
                marker=MARKERS[point["condition"]],
                s=style["marker_size"],
                zorder=3,
            )
        ax.set_xlabel(f"{prefix}{xk[-1]} ({ev[xi]*100:.1f}%)")
        ax.set_ylabel(f"{prefix}{yk[-1]} ({ev[yi]*100:.1f}%)")
        ax.set_title(f"{prefix}{xk[-1]} vs {prefix}{yk[-1]}")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
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
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(
        f"Layer {layer} · {projected.get('n_requests', '?')} requests · "
        f"{projected['explained_variance_sum']*100:.1f}% variance in 3 {prefix}s"
        + (f" · {note}" if note else ""),
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pc2_vs_pc3_layers(report: dict, out_path: Path, note: str = "") -> None:
    """One PC2 vs PC3 (or rPC2 vs rPC3) panel per captured layer."""
    layers = report_layer_ids(report)
    cols = min(4, max(1, len(layers)))
    rows = math.ceil(len(layers) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 5.2 * rows))
    axes_list = np.atleast_1d(axes).ravel()
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
    for ax, layer in zip(axes_list, layers):
        projected = report["layers"][str(layer)]
        style = _style_for_count(projected.get("n_requests", len(projected["points"]) // 3))
        ev = projected["explained_variance_ratio"]
        prefix = projected.get("axis_prefix", "PC")
        by_req: dict[str, dict[str, dict]] = {}
        for point in projected["points"]:
            by_req.setdefault(point["request_id"], {})[point["condition"]] = point
        for conds in by_req.values():
            direct, plan, fbs = conds["direct"], conds["plan"], conds["fbs"]
            ax.plot(
                [direct["pc2"], plan["pc2"]],
                [direct["pc3"], plan["pc3"]],
                color=COLORS["plan"],
                linewidth=style["line_width"],
                alpha=style["line_alpha"],
            )
            ax.plot(
                [direct["pc2"], fbs["pc2"]],
                [direct["pc3"], fbs["pc3"]],
                color=COLORS["fbs"],
                linewidth=style["line_width"],
                alpha=style["line_alpha"],
            )
        for point in projected["points"]:
            ax.scatter(
                point["pc2"],
                point["pc3"],
                c=COLORS[point["condition"]],
                marker=MARKERS[point["condition"]],
                s=style["marker_size"],
                zorder=3,
            )
        ax.set_xlabel(f"{prefix}2 ({ev[1]*100:.1f}%)")
        ax.set_ylabel(f"{prefix}3 ({ev[2]*100:.1f}%)")
        ax.set_title(f"Layer {layer}")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axes_list[len(layers):]:
        ax.axis("off")
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8)
    n = next(iter(report["layers"].values())).get("n_requests", "?")
    extra = f" · {note}" if note else ""
    prefix = next(iter(report["layers"].values())).get("axis_prefix", "PC")
    fig.suptitle(f"{prefix}2 vs {prefix}3 · {n} text requests{extra}", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def replay_2d_from_json(json_path: Path, out_dir: Path, note: str = "") -> None:
    """Redraw face plots from saved coordinates (no activation reload)."""
    report = json.loads(json_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer, projected in report["layers"].items():
        plot_layer_2d(
            int(layer),
            projected,
            out_dir / f"layer_{layer}_pca_faces.png",
            note=note,
        )
    plot_pc2_vs_pc3_layers(report, out_dir / "pc2_vs_pc3_layers.png", note=note)


def plot_three_layers_together(report: dict, out_path: Path) -> None:
    layers = overview_layers(report_layer_ids(report))
    fig = plt.figure(figsize=(5.5 * len(layers), 5.8))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[c],
            color="none",
            markerfacecolor=COLORS[c],
            markersize=10,
            label=LEGEND_LABELS[c],
        )
        for c in CONDITIONS
    ]
    for i, layer in enumerate(layers, start=1):
        projected = report["layers"][str(layer)]
        ax = fig.add_subplot(1, len(layers), i, projection="3d")
        _draw_3d(
            ax,
            projected,
            22,
            -55,
            f"Layer {layer}\n{projected['explained_variance_sum']*100:.1f}% variance in 3 {projected.get('axis_prefix', 'PC')}s",
        )
    n = next(iter(report["layers"].values())).get("n_requests", "?")
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(
        report.get(
            "plot_title",
            f"Last prompt token in 3D PCA · {n} text requests · layers {', '.join(str(i) for i in layers)}",
        ),
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_interactive_html(report: dict, out_path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    layers = overview_layers(report_layer_ids(report))
    titles = []
    for layer in layers:
        projected = report["layers"][str(layer)]
        ev = projected["explained_variance_ratio"]
        prefix = projected.get("axis_prefix", "PC")
        titles.append(
            f"Layer {layer}  ·  {prefix}1 {ev[0]*100:.1f}%  {prefix}2 {ev[1]*100:.1f}%  {prefix}3 {ev[2]*100:.1f}%"
        )
    fig = make_subplots(
        rows=1,
        cols=len(layers),
        specs=[[{"type": "scene"}] * len(layers)],
        subplot_titles=titles,
        horizontal_spacing=0.04,
    )
    legend_shown = {"direct": False, "plan": False, "fbs": False}
    for col, layer in enumerate(layers, start=1):
        projected = report["layers"][str(layer)]
        by_req: dict[str, dict[str, dict]] = {}
        ev = projected["explained_variance_ratio"]
        scene = f"scene{col}" if col > 1 else "scene"
        for point in projected["points"]:
            by_req.setdefault(point["request_id"], {})[point["condition"]] = point
        for cond in CONDITIONS:
            pts = [p for p in projected["points"] if p["condition"] == cond]
            fig.add_trace(
                go.Scatter3d(
                    x=[p["pc1"] for p in pts],
                    y=[p["pc2"] for p in pts],
                    z=[p["pc3"] for p in pts],
                    mode="markers",
                    name=LEGEND_LABELS[cond],
                    legendgroup=cond,
                    showlegend=not legend_shown[cond],
                    marker=dict(size=4, color=COLORS[cond], opacity=0.85),
                    text=[p["label"] for p in pts],
                    hovertemplate="%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra>"
                    + cond
                    + "</extra>",
                ),
                row=1,
                col=col,
            )
            legend_shown[cond] = True
        for req, conds in by_req.items():
            for other, color in (("plan", COLORS["plan"]), ("fbs", COLORS["fbs"])):
                fig.add_trace(
                    go.Scatter3d(
                        x=[conds["direct"]["pc1"], conds[other]["pc1"]],
                        y=[conds["direct"]["pc2"], conds[other]["pc2"]],
                        z=[conds["direct"]["pc3"], conds[other]["pc3"]],
                        mode="lines",
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
        prefix = projected.get("axis_prefix", "PC")
        fig.update_scenes(
            dict(
                xaxis_title=f"{prefix}1 ({ev[0]*100:.1f}%)",
                yaxis_title=f"{prefix}2 ({ev[1]*100:.1f}%)",
                zaxis_title=f"{prefix}3 ({ev[2]*100:.1f}%)",
            ),
            row=1,
            col=col,
        )
    n = next(iter(report["layers"].values())).get("n_requests", "?")
    fig.update_layout(
        title=report.get(
            "plot_title",
            f"Last prompt token in 3D PCA · {n} text requests · drag to rotate each layer",
        ),
        height=620,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA 3D plots of last prompt-token activations.")
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "pilot_random8")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "last_token_pca3d")
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    collected = collect_vectors(args.root)
    if not collected:
        raise SystemExit(f"No complete condition triples in {args.root}")
    report = {
        "method": "L2-normalize last prompt-token vectors, then PCA to 3 components, separately per layer",
        "source_runs": str(args.root),
        "layers": {},
    }
    n_points = None
    for layer, entry in collected.items():
        projected = project_layer(entry)
        n_points = projected["n_points"]
        report["n_requests"] = projected["n_requests"]
        report["n_points_per_layer"] = projected["n_points"]
        report["layers"][str(layer)] = projected
        plot_layer(layer, projected, out_dir / f"layer_{layer}_pca3d.png", elev=22, azim=-55)
        plot_layer(layer, projected, out_dir / f"layer_{layer}_pca3d_alt.png", elev=18, azim=130)
        plot_layer_2d(layer, projected, out_dir / f"layer_{layer}_pca_faces.png")
    plot_three_layers_together(report, out_dir / "layers_overview_pca3d.png")
    plot_three_layers_together(report, out_dir / "layers_8_17_35_pca3d.png")
    plot_pc2_vs_pc3_layers(report, out_dir / "pc2_vs_pc3_layers.png")
    write_interactive_html(report, out_dir / "layers_overview_pca3d.html")
    write_interactive_html(report, out_dir / "layers_8_17_35_pca3d.html")
    (out_dir / "pca3d_coordinates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "wrote": str(out_dir),
            "n_requests": report.get("n_requests"),
            "n_points_per_layer": n_points,
            "explained": {
                layer: report["layers"][str(layer)]["explained_variance_ratio"]
                for layer in report_layer_ids(report)
            },
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
