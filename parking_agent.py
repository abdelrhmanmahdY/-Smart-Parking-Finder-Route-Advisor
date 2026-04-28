"""
parking_agent.py
----------------
The top-level agent that combines:
  1. Neural-network occupancy prediction
  2. A* / BFS route search
  3. A scoring function that ranks parking lots by user preference

User preferences
----------------
  "nearest"   : minimise walking distance to destination
  "fastest"   : minimise walking time (same as nearest on this graph)
  "available" : maximise predicted free spaces (highest chance of finding a spot)

The agent also supports optional flags:
  accessible  : only consider lots with accessible spaces
  staff_only  : only consider lots with reserved staff spaces > 0
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal

from campus_graph import CAMPUS_GRAPH, PARKING_LOTS, NODES
from occupancy_model import OccupancyPredictor
from search_algorithms import astar, bfs, SearchResult

PREFERENCE = Literal["nearest", "fastest", "available"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class LotRecommendation:
    lot_id:          str
    lot_label:       str
    predicted_occ:   float          # 0-1
    free_spaces:     int
    walk_distance:   float          # metres to destination
    walk_minutes:    float
    score:           float
    route_result:    SearchResult
    rank:            int


@dataclass
class AgentDecision:
    destination:         str
    destination_label:   str
    arrival_hour:        int
    arrival_weekday:     int
    preference:          str
    best:                LotRecommendation
    alternatives:        list[LotRecommendation]
    explanation:         str


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(lot_id: str, occ: float, distance: float, preference: str) -> float:
    """
    Combine occupancy and distance into a single score (higher = better).

    We normalise each signal to [0,1] using rough campus-scale constants,
    then blend them according to the stated preference.
    """
    MAX_DIST   = 1200.0   # metres — approximate campus diameter
    free_pct   = 1.0 - occ
    dist_score = 1.0 - min(distance / MAX_DIST, 1.0)

    if preference == "available":
        w_free, w_dist = 0.80, 0.20
    elif preference in ("nearest", "fastest"):
        w_free, w_dist = 0.25, 0.75
    else:
        w_free, w_dist = 0.50, 0.50

    return round(w_free * free_pct + w_dist * dist_score, 4)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ParkingAgent:
    """
    Autonomous parking recommendation agent.

    Typical usage
    -------------
        agent = ParkingAgent()
        agent.initialise()          # trains NN once
        decision = agent.recommend(
            destination = "B_SCI",
            arrival_hour    = 9,
            arrival_weekday = 1,    # Tuesday
            preference      = "nearest",
        )
        print(agent.format_report(decision))
    """

    def __init__(self):
        self.predictor   = OccupancyPredictor()
        self._ready      = False

    # ------------------------------------------------------------------
    def initialise(self, verbose: bool = True):
        """Train the occupancy model (call once before any recommend())."""
        self.predictor.train(verbose=verbose)
        self._ready = True

    # ------------------------------------------------------------------
    def recommend(
        self,
        destination:     str,
        arrival_hour:    int,
        arrival_weekday: int,
        preference:      PREFERENCE = "available",
        event:           int = 0,
        accessible_only: bool = False,
        staff_only:      bool = False,
    ) -> AgentDecision:
        """
        Run the full pipeline and return an AgentDecision.

        Parameters
        ----------
        destination     : node ID of the target building (e.g. "B_SCI")
        arrival_hour    : 0-23
        arrival_weekday : 0 (Mon) – 6 (Sun)
        preference      : "nearest" | "fastest" | "available"
        event           : 1 if campus event today
        accessible_only : restrict to lots with accessible_spaces > 0
        staff_only      : restrict to lots with reserved_staff > 0
        """
        if not self._ready:
            raise RuntimeError("Call initialise() first.")
        if destination not in CAMPUS_GRAPH:
            raise ValueError(f"Unknown destination node: {destination}")

        # --- Step 1: predict occupancy for all lots ---
        occupancy = self.predictor.predict_all_lots(arrival_hour, arrival_weekday, event)

        # --- Step 2: compute A* routes from each lot to destination ---
        scored: list[LotRecommendation] = []

        for lot_id, meta in PARKING_LOTS.items():
            # Apply optional filters
            if accessible_only and meta["accessible_spaces"] == 0:
                continue
            if staff_only and meta["reserved_staff"] == 0:
                continue

            occ      = occupancy.get(lot_id, 0.5)
            capacity = meta["capacity"]
            free     = max(0, round(capacity * (1 - occ)))

            # A* route
            route = astar(lot_id, destination)
            if route is None:
                continue

            sc = _score(lot_id, occ, route.total_distance, preference)

            scored.append(LotRecommendation(
                lot_id        = lot_id,
                lot_label     = NODES[lot_id]["label"],
                predicted_occ = occ,
                free_spaces   = free,
                walk_distance = route.total_distance,
                walk_minutes  = route.walking_minutes,
                score         = sc,
                route_result  = route,
                rank          = 0,  # set below
            ))

        if not scored:
            raise RuntimeError("No lots available after applying filters.")

        # --- Step 3: rank ---
        scored.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(scored):
            r.rank = i + 1

        best         = scored[0]
        alternatives = scored[1:]

        explanation = self._explain(best, preference, arrival_hour, arrival_weekday)

        return AgentDecision(
            destination       = destination,
            destination_label = NODES[destination]["label"],
            arrival_hour      = arrival_hour,
            arrival_weekday   = arrival_weekday,
            preference        = preference,
            best              = best,
            alternatives      = alternatives,
            explanation       = explanation,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _explain(rec: LotRecommendation, preference: str, hour: int, wd: int) -> str:
        days    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        avail   = f"{100*(1-rec.predicted_occ):.0f}%"
        occ_pct = f"{100*rec.predicted_occ:.0f}%"

        lines = [
            f"The neural network predicts that {rec.lot_label} will be "
            f"{occ_pct} occupied on {days[wd]} at {hour:02d}:00, "
            f"leaving approximately {rec.free_spaces} free space(s) ({avail} availability).",
        ]

        if preference == "available":
            lines.append(
                f"Since your priority is finding a space, this lot scored highest "
                f"on predicted availability while still being reachable in ~{rec.walk_minutes} min."
            )
        elif preference in ("nearest", "fastest"):
            lines.append(
                f"Since your priority is minimising walking, this lot is the closest "
                f"option at {rec.walk_distance:.0f} m (~{rec.walk_minutes} min walk)."
            )

        lines.append(
            f"A* search found the optimal route in {rec.route_result.nodes_explored} node expansions."
        )
        return " ".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def format_report(decision: AgentDecision, show_bfs_compare: bool = True) -> str:
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        sep  = "─" * 60

        lines = [
            sep,
            f"  CAMPUS PARKING RECOMMENDATION",
            sep,
            f"  Destination : {decision.destination_label}",
            f"  Arrival     : {days[decision.arrival_weekday]} {decision.arrival_hour:02d}:00",
            f"  Preference  : {decision.preference}",
            sep,
            f"  ★  RECOMMENDED: {decision.best.lot_label}",
            f"     Predicted occupancy : {decision.best.predicted_occ:.1%}",
            f"     Estimated free spots: {decision.best.free_spaces}",
            f"     Walking distance    : {decision.best.walk_distance:.0f} m",
            f"     Walking time        : {decision.best.walk_minutes} min",
            f"",
            f"  ROUTE (A*):",
            f"     {decision.best.route_result.summary()}",
            sep,
            f"  WHY THIS LOT?",
            f"  {decision.explanation}",
        ]

        if decision.alternatives:
            lines += [sep, "  ALTERNATIVES:"]
            for alt in decision.alternatives:
                lines.append(
                    f"  #{alt.rank} {alt.lot_label:15s}  "
                    f"occ={alt.predicted_occ:.0%}  "
                    f"free={alt.free_spaces:3d}  "
                    f"walk={alt.walk_minutes} min"
                )

        # BFS comparison
        if show_bfs_compare:
            bfs_result = bfs(decision.best.lot_id, decision.destination)
            lines += [
                sep,
                "  ALGORITHM COMPARISON (BFS vs A*): same lot → destination",
            ]
            if bfs_result:
                lines.append(f"  {bfs_result.summary()}")
            lines.append(f"  {decision.best.route_result.summary()}")
            lines.append(
                "  A* explores fewer nodes than BFS because it uses the "
                "Euclidean heuristic to guide search toward the goal."
            )

        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = ParkingAgent()
    agent.initialise()

    decision = agent.recommend(
        destination     = "B_SCI",
        arrival_hour    = 9,
        arrival_weekday = 1,   # Tuesday
        preference      = "available",
    )
    print(agent.format_report(decision))
