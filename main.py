"""
main.py
-------
Entry point for the Campus Parking Recommendation System.

Usage examples
--------------
  # Interactive mode (prompts for input):
      python main.py

  # CLI mode:
      python main.py --dest B_SCI --hour 9 --day 1 --pref available
      python main.py --dest B_LIB --hour 14 --day 3 --pref nearest --event
      python main.py --dest B_GYM --hour 18 --day 4 --pref fastest --accessible

Options
-------
  --dest        Node ID of destination building
                Choices: B_MAIN B_SCI B_ENG B_LIB B_GYM B_ADMIN
  --hour        Arrival hour 0-23
  --day         Weekday  0=Mon … 6=Sun
  --pref        nearest | fastest | available
  --event       Flag: campus event today
  --accessible  Flag: need accessible parking
  --staff       Flag: staff-only lots
  --no-charts   Skip saving PNG visualisations
"""

import argparse
import os
import sys

# Make sure the package root is on the path when running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from parking_agent   import ParkingAgent
from campus_graph     import NODES, PARKING_LOTS
from visualise         import (
    draw_campus_graph,
    draw_occupancy_chart,
    draw_algorithm_comparison,
)

BUILDING_NODES = {k: v for k, v in NODES.items() if v["type"] == "building"}
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ---------------------------------------------------------------------------
# Interactive prompt helpers
# ---------------------------------------------------------------------------

def _choose_building() -> str:
    print("\n  Available destinations:")
    items = list(BUILDING_NODES.items())
    for i, (nid, data) in enumerate(items, 1):
        print(f"    {i}. {data['label']}  [{nid}]")
    while True:
        try:
            idx = int(input("\n  Enter number: ").strip()) - 1
            if 0 <= idx < len(items):
                return items[idx][0]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def _choose_hour() -> int:
    while True:
        try:
            h = int(input("  Arrival hour (0-23): ").strip())
            if 0 <= h <= 23:
                return h
        except ValueError:
            pass
        print("  Please enter a number 0-23.")


def _choose_day() -> int:
    print("  Weekday: 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri 5=Sat 6=Sun")
    while True:
        try:
            d = int(input("  Weekday number: ").strip())
            if 0 <= d <= 6:
                return d
        except ValueError:
            pass
        print("  Please enter 0-6.")


def _choose_pref() -> str:
    prefs = ["nearest", "fastest", "available"]
    print("  Preference:")
    for i, p in enumerate(prefs, 1):
        print(f"    {i}. {p}")
    while True:
        try:
            idx = int(input("  Enter number: ").strip()) - 1
            if 0 <= idx < len(prefs):
                return prefs[idx]
        except ValueError:
            pass
        print("  Invalid, try again.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    dest:        str,
    hour:        int,
    day:         int,
    pref:        str,
    event:       bool = False,
    accessible:  bool = False,
    staff:       bool = False,
    save_charts: bool = True,
):
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  Campus Parking Recommendation System")
    print("  CET251 – Artificial Intelligence  |  El Sewedy UT")
    print("="*60)
    print("  Initialising neural network model …")

    agent = ParkingAgent()
    agent.initialise(verbose=True)

    print("\n  Running agent …\n")
    decision = agent.recommend(
        destination     = dest,
        arrival_hour    = hour,
        arrival_weekday = day,
        preference      = pref,
        event           = int(event),
        accessible_only = accessible,
        staff_only      = staff,
    )

    print(agent.format_report(decision))

    if save_charts:
        print("\n  Saving visualisations …")

        # 1. Campus graph with route
        draw_campus_graph(
            decision  = decision,
            save_path = os.path.join(OUT_DIR, "campus_graph.png"),
        )

        # 2. Occupancy chart
        occ_map = agent.predictor.predict_all_lots(hour, day, int(event))
        draw_occupancy_chart(
            occ_map,
            save_path = os.path.join(OUT_DIR, "occupancy_chart.png"),
        )

        # 3. BFS vs DFS vs A* comparison on the chosen route
        draw_algorithm_comparison(
            source    = decision.best.lot_id,
            target    = decision.destination,
            save_path = os.path.join(OUT_DIR, "algorithm_comparison.png"),
        )

        print(f"\n  Charts saved to: {OUT_DIR}/")

    return decision


def main():
    parser = argparse.ArgumentParser(description="Campus Parking Recommender")
    parser.add_argument("--dest",       default=None,
                        help="Destination node ID, e.g. B_SCI")
    parser.add_argument("--hour",       type=int, default=None,
                        help="Arrival hour 0-23")
    parser.add_argument("--day",        type=int, default=None,
                        help="Weekday 0=Mon…6=Sun")
    parser.add_argument("--pref",       default=None,
                        choices=["nearest", "fastest", "available"])
    parser.add_argument("--event",      action="store_true",
                        help="Campus event today")
    parser.add_argument("--accessible", action="store_true",
                        help="Require accessible spaces")
    parser.add_argument("--staff",      action="store_true",
                        help="Staff-only lots")
    parser.add_argument("--no-charts",  action="store_true",
                        help="Skip PNG chart generation")
    args = parser.parse_args()

    # Interactive mode if any required arg is missing
    interactive = (args.dest is None or args.hour is None
                   or args.day is None or args.pref is None)

    if interactive:
        print("\n=== Campus Parking Recommender – Interactive Mode ===")
        dest = args.dest or _choose_building()
        hour = args.hour if args.hour is not None else _choose_hour()
        day  = args.day  if args.day  is not None else _choose_day()
        pref = args.pref or _choose_pref()
        event      = args.event
        accessible = args.accessible
        staff      = args.staff
    else:
        dest, hour, day, pref = args.dest, args.hour, args.day, args.pref
        event, accessible, staff = args.event, args.accessible, args.staff

    run(dest, hour, day, pref, event, accessible, staff,
        save_charts=not args.no_charts)


if __name__ == "__main__":
    main()