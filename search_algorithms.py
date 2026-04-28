"""
search_algorithms.py
--------------------
Implements BFS, DFS, and A* on the campus NetworkX graph.

All functions return a SearchResult namedtuple:
    path          : list of node IDs from source to destination
    total_distance: total edge-weight (metres)
    nodes_explored: number of nodes expanded during search
    algorithm     : name string

A* uses Euclidean distance between node positions as the admissible heuristic.
"""

import math
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from campus_graph import CAMPUS_GRAPH, NODES


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    path:           list[str]
    total_distance: float          # metres
    nodes_explored: int
    algorithm:      str
    walking_minutes: float = field(init=False)

    WALKING_SPEED_MPM = 80  # metres per minute (~4.8 km/h)

    def __post_init__(self):
        self.walking_minutes = round(self.total_distance / self.WALKING_SPEED_MPM, 1)

    def summary(self) -> str:
        steps = " → ".join(
            NODES[n]["label"] if n in NODES else n for n in self.path
        )
        return (
            f"[{self.algorithm}] {steps}\n"
            f"  Distance : {self.total_distance:.0f} m\n"
            f"  Walk time: {self.walking_minutes} min\n"
            f"  Explored : {self.nodes_explored} nodes"
        )


# ---------------------------------------------------------------------------
# Helper: path cost from node list
# ---------------------------------------------------------------------------

def _path_cost(G: nx.Graph, path: list[str]) -> float:
    cost = 0.0
    for u, v in zip(path, path[1:]):
        cost += G[u][v]["weight"]
    return cost


# ---------------------------------------------------------------------------
# BFS  (unweighted shortest path by hop count)
# ---------------------------------------------------------------------------

def bfs(source: str, target: str, G: nx.Graph = CAMPUS_GRAPH) -> Optional[SearchResult]:
    """
    Breadth-First Search.
    Finds the path with fewest hops (not necessarily shortest distance).
    """
    if source == target:
        return SearchResult([source], 0.0, 1, "BFS")

    visited   = {source}
    queue     = deque([(source, [source])])
    explored  = 0

    while queue:
        node, path = queue.popleft()
        explored += 1

        for neighbour in G.neighbors(node):
            if neighbour == target:
                full_path = path + [neighbour]
                return SearchResult(full_path, _path_cost(G, full_path), explored, "BFS")
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))

    return None  # no path


# ---------------------------------------------------------------------------
# DFS  (depth-first; not optimal — included for comparison)
# ---------------------------------------------------------------------------

def dfs(source: str, target: str, G: nx.Graph = CAMPUS_GRAPH) -> Optional[SearchResult]:
    """
    Depth-First Search (iterative with explicit stack).
    Not optimal; included to contrast with A*.
    """
    if source == target:
        return SearchResult([source], 0.0, 1, "DFS")

    visited  = set()
    stack    = [(source, [source])]
    explored = 0

    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        explored += 1

        if node == target:
            return SearchResult(path, _path_cost(G, path), explored, "DFS")

        for neighbour in G.neighbors(node):
            if neighbour not in visited:
                stack.append((neighbour, path + [neighbour]))

    return None


# ---------------------------------------------------------------------------
# A*  (optimal weighted path using Euclidean heuristic)
# ---------------------------------------------------------------------------

def _euclidean(a: str, b: str) -> float:
    """Straight-line distance between two node positions (grid units → metres)."""
    SCALE = 50  # 1 grid unit ≈ 50 metres
    ax, ay = NODES[a]["pos"]
    bx, by = NODES[b]["pos"]
    return math.hypot(ax - bx, ay - by) * SCALE


def astar(source: str, target: str, G: nx.Graph = CAMPUS_GRAPH) -> Optional[SearchResult]:
    """
    A* Search with Euclidean distance heuristic.
    Guarantees the shortest weighted path.
    """
    if source == target:
        return SearchResult([source], 0.0, 1, "A*")

    # Priority queue: (f, g, node, path)
    open_heap = [(0.0, 0.0, source, [source])]
    best_g    = {source: 0.0}
    explored  = 0

    while open_heap:
        f, g, node, path = heapq.heappop(open_heap)
        explored += 1

        if node == target:
            return SearchResult(path, g, explored, "A*")

        if g > best_g.get(node, float("inf")):
            continue  # stale entry

        for neighbour, edge_data in G[node].items():
            new_g = g + edge_data["weight"]
            if new_g < best_g.get(neighbour, float("inf")):
                best_g[neighbour] = new_g
                h = _euclidean(neighbour, target)
                heapq.heappush(open_heap, (new_g + h, new_g, neighbour, path + [neighbour]))

    return None


# ---------------------------------------------------------------------------
# Convenience: run all algorithms and return comparison dict
# ---------------------------------------------------------------------------

def compare_algorithms(source: str, target: str) -> dict[str, SearchResult]:
    results = {}
    for name, fn in [("BFS", bfs), ("DFS", dfs), ("A*", astar)]:
        r = fn(source, target)
        if r:
            results[name] = r
    return results


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Route: P_NORTH → B_SCI ===\n")
    for name, result in compare_algorithms("P_NORTH", "B_SCI").items():
        print(result.summary())
        print()
