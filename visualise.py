"""
visualise.py
------------
Simple matplotlib visualisations:
  1. Campus graph with highlighted route
  2. Occupancy bar chart for all lots
  3. BFS vs A* node-expansion comparison chart

Keep graphics functional, not fancy — per project spec.
"""

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from campus_graph import CAMPUS_GRAPH, NODES, PARKING_LOTS
from parking_agent import AgentDecision
from search_algorithms import bfs, astar, compare_algorithms


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
NODE_COLOURS = {
    "building": "#4A90D9",
    "lot":      "#E67E22",
    "junction": "#95A5A6",
}

EDGE_COL    = "#BDC3C7"
ROUTE_COL   = "#2ECC71"
BEST_LOT_COL= "#E74C3C"


# ---------------------------------------------------------------------------
# 1. Campus graph
# ---------------------------------------------------------------------------

def draw_campus_graph(decision: AgentDecision | None = None, save_path: str | None = None):
    """
    Draw the campus graph. If *decision* is provided, highlight the
    recommended lot and the A* route.
    """
    G   = CAMPUS_GRAPH
    pos = {n: data["pos"] for n, data in NODES.items()}

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor("#1C1C2E")
    fig.patch.set_facecolor("#1C1C2E")

    # --- draw all edges ---
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=EDGE_COL,
                           width=1.5, alpha=0.6)

    # --- draw edge weight labels ---
    edge_labels = {(u, v): f"{d['weight']}m" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=5, font_color="#888888",
        bbox=dict(boxstyle="round,pad=0.1", fc="#1C1C2E", ec="none", alpha=0.7)
    )

    # Highlight route
    if decision:
        route_path = decision.best.route_result.path
        route_edges = list(zip(route_path, route_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, ax=ax,
                               edge_color=ROUTE_COL, width=4, alpha=0.9)

    # --- draw nodes ---
    for node, data in NODES.items():
        colour = NODE_COLOURS.get(data["type"], "#AAA")
        if decision and node == decision.best.lot_id:
            colour = BEST_LOT_COL
        nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax,
                               node_color=colour, node_size=350)

    # --- labels ---
    labels = {n: NODES[n]["label"].replace(" ", "\n") for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=6, font_color="white")

    # Legend
    handles = [
        mpatches.Patch(color=NODE_COLOURS["building"], label="Building"),
        mpatches.Patch(color=NODE_COLOURS["lot"],      label="Parking Lot"),
        mpatches.Patch(color=NODE_COLOURS["junction"], label="Junction"),
        mpatches.Patch(color=BEST_LOT_COL,             label="Recommended Lot"),
        mpatches.Patch(color=ROUTE_COL,                label="A* Route"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              facecolor="#2C2C3E", edgecolor="#555", labelcolor="white")

    title = "Campus Parking Graph"
    if decision:
        title += f"  |  Route to {decision.destination_label}"
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Graph saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 2. Occupancy bar chart
# ---------------------------------------------------------------------------

def draw_occupancy_chart(occupancy: dict[str, float], save_path: str | None = None):
    """Bar chart of predicted occupancy for all lots."""
    lots  = sorted(occupancy)
    occ   = [occupancy[l] * 100 for l in lots]
    caps  = [PARKING_LOTS[l]["capacity"] for l in lots]
    free  = [int(PARKING_LOTS[l]["capacity"] * (1 - occupancy[l])) for l in lots]
    labels = [NODES[l]["label"] for l in lots]

    colours = ["#E74C3C" if o > 85 else "#E67E22" if o > 60 else "#2ECC71" for o in occ]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1C1C2E")
    ax.set_facecolor("#1C1C2E")

    bars = ax.bar(labels, occ, color=colours, width=0.55, edgecolor="#2C2C3E", linewidth=1.2)

    # Annotate bars
    for bar, f in zip(bars, free):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{f} free", ha="center", va="bottom",
                fontsize=8, color="white")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Predicted Occupancy (%)", color="white")
    ax.set_title("Predicted Lot Occupancy", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.axhline(85, color="#E74C3C", linewidth=1, linestyle="--", alpha=0.6, label="85% threshold")
    ax.legend(facecolor="#2C2C3E", edgecolor="#555", labelcolor="white", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Occupancy chart saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3. Algorithm comparison
# ---------------------------------------------------------------------------

def draw_algorithm_comparison(source: str, target: str, save_path: str | None = None):
    """
    Side-by-side bar chart comparing BFS, DFS, and A* on:
      - nodes explored
      - path distance
    """
    results = compare_algorithms(source, target)
    names   = list(results)
    explored= [results[n].nodes_explored for n in names]
    dists   = [results[n].total_distance for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#1C1C2E")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1C1C2E")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

    COLS = ["#4A90D9", "#9B59B6", "#2ECC71"]

    ax1.bar(names, explored, color=COLS, width=0.5)
    ax1.set_title("Nodes Explored", color="white")
    ax1.set_ylabel("Count", color="white")
    for i, v in enumerate(explored):
        ax1.text(i, v + 0.3, str(v), ha="center", color="white", fontsize=9)

    ax2.bar(names, dists, color=COLS, width=0.5)
    ax2.set_title("Path Distance (m)", color="white")
    ax2.set_ylabel("Metres", color="white")
    for i, v in enumerate(dists):
        ax2.text(i, v + 5, f"{v:.0f}", ha="center", color="white", fontsize=9)

    src_lbl = NODES[source]["label"]
    tgt_lbl = NODES[target]["label"]
    fig.suptitle(f"Algorithm Comparison: {src_lbl} → {tgt_lbl}",
                 color="white", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Comparison chart saved → {save_path}")
    else:
        plt.show()
