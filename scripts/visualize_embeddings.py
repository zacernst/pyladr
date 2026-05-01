#!/usr/bin/env python3
"""Visualize Tree2Vec clause embeddings using t-SNE.

Reads the JSON produced by --tree2vec-dump-embeddings and plots each
SOS clause as a 2-D point.  Points are coloured by whether the clause
appeared in a proof, and sized by clause weight.

Output formats:
  - PNG/PDF (default, matplotlib): static image
  - HTML (plotly): interactive with hover tooltips showing the clause text

Usage:
    uv run python scripts/visualize_embeddings.py sos_embeddings.json
    uv run python scripts/visualize_embeddings.py sos_embeddings.json --out plot.png
    uv run python scripts/visualize_embeddings.py sos_embeddings.json --out plot.html
    uv run python scripts/visualize_embeddings.py sos_embeddings.json --point-size 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="t-SNE plot of Tree2Vec clause embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to the embedding JSON file.")
    parser.add_argument(
        "--out", metavar="FILE",
        help="Save the plot to FILE instead of showing it interactively.",
    )
    parser.add_argument(
        "--point-size", type=float, default=20.0, metavar="S",
        help="Base marker size in points² (default: 20). "
             "Actual size scales with clause weight.",
    )
    parser.add_argument(
        "--goal-color", default="#2ca02c", metavar="COLOR",
        help="Colour for goal clauses (default: '#2ca02c', green). "
             "Accepts any matplotlib colour string.",
    )
    parser.add_argument(
        "--perplexity", type=float, default=None,
        help="t-SNE perplexity (default: min(30, n_points-1)).",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="t-SNE max iterations (default: 1000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────

    path = Path(args.input)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    clauses = data.get("clauses", [])
    model_meta = data.get("model", {})

    # Filter to clauses that have a non-null embedding
    valid = [c for c in clauses if c.get("embedding") is not None]
    if not valid:
        print("No clauses with embeddings found in the file.", file=sys.stderr)
        return 1

    import numpy as np

    X = np.array([c["embedding"] for c in valid], dtype=np.float32)
    weights = np.array([c.get("weight", 1.0) for c in valid], dtype=np.float32)
    in_proof = np.array([bool(c.get("in_proof", False)) for c in valid])
    is_goal  = np.array([bool(c.get("is_goal",  False)) for c in valid])

    n = len(valid)
    print(
        f"Loaded {n} clauses  |  dim={X.shape[1]}  |  "
        f"{in_proof.sum()} in proof  |  {is_goal.sum()} goal  |  "
        f"model update #{model_meta.get('update_number', '?')}, "
        f"vocab={model_meta.get('vocab_size', '?')}"
    )

    if n < 2:
        print("Need at least 2 clauses for t-SNE.", file=sys.stderr)
        return 1

    # ── t-SNE ───────────────────────────────────────────────────────────────

    from sklearn.manifold import TSNE

    perplexity = args.perplexity if args.perplexity is not None else min(30.0, n - 1)
    print(f"Running t-SNE: perplexity={perplexity}, max_iter={args.iterations} …")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=args.iterations,
        random_state=args.seed,
        init="pca" if n >= 4 else "random",
    )
    coords = tsne.fit_transform(X)   # (n, 2)

    # ── Shared metadata ──────────────────────────────────────────────────────

    update_num = model_meta.get("update_number", "?")
    vocab = model_meta.get("vocab_size", "?")
    dim = model_meta.get("embedding_dim", X.shape[1])
    ts = model_meta.get("timestamp", "")
    title = f"Tree2Vec SOS embeddings — update #{update_num}  (vocab={vocab}, dim={dim})"
    if ts:
        title += f"  {ts}"

    # Decide output format: HTML → plotly, anything else → matplotlib
    out_path = Path(args.out) if args.out else None
    use_plotly = out_path is not None and out_path.suffix.lower() == ".html"

    regular = ~in_proof & ~is_goal
    size_base = np.clip(weights, 1.0, 50.0)
    sizes = args.point_size * (0.5 + size_base / 50.0)

    if use_plotly:
        _plot_plotly(valid, coords, sizes, in_proof, is_goal, regular,
                     args.goal_color, title, out_path)
    else:
        _plot_matplotlib(coords, sizes, in_proof, is_goal, regular,
                         args.goal_color, title, out_path)

    return 0


# ── Plotly (HTML, interactive hover) ────────────────────────────────────────

def _plot_plotly(valid, coords, sizes, in_proof, is_goal, regular,
                 goal_color, title, out_path):
    import textwrap
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Error: plotly is required for HTML output.  "
              "Install it with:  pip install plotly", file=sys.stderr)
        raise SystemExit(1)

    fig = go.Figure()

    # Groups drawn back-to-front so proof/goal points appear on top.
    groups = [
        (regular,  "#4c72b0", "circle",          "Regular",  1.0, "white", 1),
        (is_goal,  goal_color, "diamond",         "Goal",     2.0, "black", 2),
        (in_proof, "#dd4444", "star",             "In proof", 1.5, "white", 1),
    ]
    for mask, colour, symbol, label, scale, edge_col, lw in groups:
        if not mask.any():
            continue
        idx = mask.nonzero()[0]
        hover = []
        for i in idx:
            c = valid[i]
            clause_text = c.get("clause", "")
            # Wrap long clauses for readability in the tooltip
            wrapped = "<br>".join(textwrap.wrap(clause_text, width=80))
            hover.append(
                f"<b>id={c.get('id', '?')}</b>  weight={c.get('weight', '?'):.1f}<br>"
                f"in_proof={c.get('in_proof', False)}  is_goal={c.get('is_goal', False)}<br>"
                f"{wrapped}"
            )
        fig.add_trace(go.Scatter(
            x=coords[idx, 0],
            y=coords[idx, 1],
            mode="markers",
            name=f"{label} ({mask.sum()})",
            hovertext=hover,
            hoverinfo="text",
            marker=dict(
                color=colour,
                size=sizes[idx] * scale * 0.5,  # plotly px, not pts²
                symbol=symbol,
                opacity=0.90,
                line=dict(color=edge_col, width=lw),
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="t-SNE dim 1",
        yaxis_title="t-SNE dim 2",
        legend=dict(font=dict(size=10)),
        hoverlabel=dict(font_size=11, font_family="monospace"),
        width=1100,
        height=820,
    )
    fig.add_annotation(
        text="Marker size ∝ clause weight",
        xref="paper", yref="paper",
        x=0.01, y=0.01, showarrow=False,
        font=dict(size=9, color="#777777"),
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved → {out_path}  (open in a browser for hover tooltips)")


# ── Matplotlib (PNG/PDF/interactive) ────────────────────────────────────────

def _plot_matplotlib(coords, sizes, in_proof, is_goal, regular,
                     goal_color, title, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 9))

    groups = [
        (regular,  "#4c72b0",  "o", 2, "Regular",  1.0, "white", 0.4),
        (is_goal,  goal_color, "D", 3, "Goal",     2.0, "black", 1.5),
        (in_proof, "#dd4444",  "*", 4, "In proof", 1.5, "white", 0.4),
    ]
    for mask, colour, marker, zorder, lab, scale, ecol, elw in groups:
        if mask.any():
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colour, s=sizes[mask] * scale,
                marker=marker, zorder=zorder,
                alpha=0.90, linewidths=elw, edgecolors=ecol,
                label=f"{lab} ({mask.sum()})",
            )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.legend(fontsize=9, loc="best")
    ax.tick_params(labelsize=8)
    ax.annotate(
        "Marker size ∝ clause weight",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=7, color="#555555",
    )

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
