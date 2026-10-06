#!/usr/bin/env python3
"""Cosine distance of last-prompt-token states across pilot conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LAYERS = (8, 17, 35)
CONDITIONS = ("direct", "plan", "fbs")
PAIRS = (("direct", "plan"), ("direct", "fbs"), ("plan", "fbs"))


def layers_from_run_root(root: Path) -> tuple[int, ...]:
    """Read captured language-layer indices from the first complete request."""
    for request_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        meta_path = request_dir / "direct" / "activations.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        indices = meta.get("layer_indices")
        if indices:
            return tuple(int(idx) for idx in indices)
        keys = [
            int(key.split("_", 1)[1])
            for key in (meta.get("shapes") or {})
            if key.startswith("layer_")
        ]
        if keys:
            return tuple(sorted(keys))
    return LAYERS


def overview_layers(layers: tuple[int, ...] | list[int], n: int = 3) -> tuple[int, ...]:
    ordered = tuple(sorted(int(idx) for idx in layers))
    if len(ordered) <= n:
        return ordered
    return tuple(ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n))


def last_prompt_vectors(run_dir: Path, layers: tuple[int, ...] | list[int]) -> dict[int, torch.Tensor]:
    """Load one activations file and return a cloned last-prompt-token vector per layer.

    The on-disk tensor is [n_tokens, hidden]. Returning a slice would keep the
    full sequence alive for as long as the caller holds the vector.
    """
    meta = json.loads((run_dir / "activations.json").read_text(encoding="utf-8"))
    data = torch.load(run_dir / "activations.pt", map_location="cpu", weights_only=True)
    index = int(meta["prompt_length"]) - 1
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        hidden = data[f"layer_{layer}"].float()
        if index < 0 or index >= hidden.shape[0]:
            raise IndexError(f"{run_dir}: prompt index {index} outside {tuple(hidden.shape)}")
        out[layer] = hidden[index].detach().clone()
    return out


def last_prompt_vector(run_dir: Path, layer: int) -> torch.Tensor:
    return last_prompt_vectors(run_dir, (layer,))[layer]


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    sim = F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item()
    return 1.0 - sim


def collect_rows(root: Path, layers: tuple[int, ...] | None = None) -> list[dict]:
    layers = layers or layers_from_run_root(root)
    rows = []
    for request_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not all((request_dir / cond / "activations.pt").exists() for cond in CONDITIONS):
            continue
        instruction = ""
        summary_path = request_dir / "summary.json"
        if summary_path.exists():
            instruction = json.loads(summary_path.read_text(encoding="utf-8")).get(
                "instruction_text", ""
            )
        vectors = {cond: last_prompt_vectors(request_dir / cond, layers) for cond in CONDITIONS}
        for layer in layers:
            for left, right in PAIRS:
                rows.append(
                    {
                        "request_id": request_dir.name,
                        "instruction": instruction,
                        "layer": layer,
                        "pair": f"{left}-{right}",
                        "cosine_similarity": 1.0
                        - cosine_distance(vectors[left][layer], vectors[right][layer]),
                        "cosine_distance": cosine_distance(
                            vectors[left][layer], vectors[right][layer]
                        ),
                        "token_role": "last_prompt_token",
                    }
                )
    return rows


def mean_by(rows: list[dict], key_layer: bool = True) -> dict[tuple, float]:
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["layer"], row["pair"]) if key_layer else (row["pair"],)
        grouped.setdefault(key, []).append(row["cosine_distance"])
    return {key: sum(vals) / len(vals) for key, vals in grouped.items()}


def plot_mean(rows: list[dict], out_path: Path) -> None:
    layers = tuple(sorted({row["layer"] for row in rows}))
    pair_names = [f"{a}-{b}" for a, b in PAIRS]
    x = list(range(len(layers)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8.5, 1.1 * len(layers)), 4.8))
    for i, pair in enumerate(pair_names):
        means = [mean_by(rows)[(layer, pair)] for layer in layers]
        ax.bar([p + (i - 1) * width for p in x], means, width=width, label=pair)
    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {layer}" for layer in layers])
    ax.set_ylabel("Mean cosine distance (1 − cosine similarity)")
    ax.set_title("Last prompt-token cosine distance across conditions")
    ax.set_ylim(0, max(row["cosine_distance"] for row in rows) * 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_request(rows: list[dict], out_path: Path) -> None:
    requests = sorted({row["request_id"] for row in rows})
    short = [req.split("_")[0] + "…" + req[-6:] for req in requests]
    layers = overview_layers(tuple(sorted({row["layer"] for row in rows})))
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.6), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    pair_names = [f"{a}-{b}" for a, b in PAIRS]
    width = 0.24
    x = list(range(len(requests)))
    by_req = {(row["request_id"], row["layer"], row["pair"]): row["cosine_distance"] for row in rows}
    for ax, layer in zip(axes, layers):
        for i, pair in enumerate(pair_names):
            vals = [by_req[(req, layer, pair)] for req in requests]
            ax.bar([p + (i - 1) * width for p in x], vals, width=width, label=pair)
        ax.set_title(f"Layer {layer}")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Cosine distance")
            ax.legend(fontsize=8)
    fig.suptitle("Per-request last prompt-token cosine distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosine distance of last prompt-token states.")
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "pilot_random8")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "last_token_cosine")
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.root)
    if not rows:
        raise SystemExit(f"No complete condition triples in {args.root}")
    layers = tuple(sorted({row["layer"] for row in rows}))
    means = {
        f"layer_{layer}_{pair}": mean_by(rows)[(layer, pair)]
        for layer in layers
        for pair in (f"{a}-{b}" for a, b in PAIRS)
    }
    report = {
        "token_role": "last_prompt_token (index = prompt_length - 1)",
        "metric": "cosine_distance = 1 - cosine_similarity",
        "n_requests": len({row["request_id"] for row in rows}),
        "layers": list(layers),
        "pairs": [f"{a}-{b}" for a, b in PAIRS],
        "mean_cosine_distance": means,
        "rows": rows,
    }
    json_path = out_dir / "last_token_cosine.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_mean(rows, out_dir / "mean_cosine_distance.png")
    if report["n_requests"] <= 16:
        plot_per_request(rows, out_dir / "per_request_cosine_distance.png")
    print(json.dumps({"n_requests": report["n_requests"], "means": means, "wrote": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
