# Campus Parking Recommendation System
### CET251: Artificial Intelligence — El Sewedy University of Technology

---

## Project Overview

An AI-powered campus parking assistant that combines neural network occupancy prediction with graph search algorithms to recommend the best parking lot for a given destination and arrival time.

---

## Architecture

```
campus_parking/
├── data/
│   ├── campus_graph.py      # Campus graph: 20 nodes, weighted edges
│   └── generate_data.py     # Synthetic occupancy data generator
├── models/
│   └── occupancy_model.py   # MLP neural network (scikit-learn)
├── search/
│   └── search_algorithms.py # BFS, DFS, A* implementations
├── agent/
│   └── parking_agent.py     # Decision-making agent + scoring
├── viz/
│   └── visualise.py         # Matplotlib charts
└── main.py                  # Entry point (CLI + interactive)
```

---

## Core AI Concepts

### 1. Agents & Environments
`ParkingAgent` is a **goal-based agent** operating in a campus environment. Its percepts are `(destination, hour, weekday, preference)` and its actions are lot recommendations with computed routes.

### 2. Graph Search: BFS
**Breadth-First Search** explores the campus graph level by level, finding the path with the fewest hops. Guaranteed to find *a* path but not the shortest *weighted* one.

### 3. Graph Search: A*
**A\* Search** uses `f(n) = g(n) + h(n)` where:
- `g(n)` = actual distance walked so far
- `h(n)` = Euclidean straight-line distance to goal (admissible heuristic)

A* is **optimal and complete**, and explores fewer nodes than BFS because the heuristic prunes unpromising directions.

### 4. Neural Network (MLP Regressor)
A small feed-forward network trained on synthetic data:

| Layer  | Neurons | Activation |
|--------|---------|------------|
| Input  | 5       | —          |
| Hidden | 64      | ReLU       |
| Hidden | 32      | ReLU       |
| Output | 1       | Linear     |

**Features**: `hour`, `weekday`, `lot_id`, `class_density`, `event`  
**Target**: `occupancy_pct` ∈ [0, 1]  
**Performance**: MAE ≈ 0.043, R² ≈ 0.95

---

## Campus Graph

- **5 parking lots**: North, East, Central, West, South
- **6 buildings**: Main Hall, Science Block, Engineering, Library, Sports Complex, Admin
- **8 junction nodes** connecting paths
- **Total**: 19 nodes, 25 weighted edges (metres)

---

## Scoring Function

Each lot is scored by blending two signals:

```
score = w_free × (1 - occupancy) + w_dist × (1 - distance/max_dist)
```

| Preference | w_free | w_dist |
|------------|--------|--------|
| available  | 0.80   | 0.20   |
| nearest    | 0.25   | 0.75   |
| fastest    | 0.25   | 0.75   |

---

## Installation & Usage

```bash
# Install dependencies
pip install networkx pandas matplotlib scikit-learn

# CLI usage
python main.py --dest B_SCI --hour 9 --day 1 --pref available
python main.py --dest B_LIB --hour 14 --day 3 --pref nearest
python main.py --dest B_GYM --hour 18 --day 4 --pref fastest --event

# Interactive mode
python main.py

# Stretch features
python main.py --dest B_MAIN --hour 10 --day 0 --pref available --accessible
python main.py --dest B_ENG  --hour 8  --day 2 --pref nearest   --staff
```

---

## Destination Node IDs

| ID       | Building         |
|----------|------------------|
| B_MAIN   | Main Hall        |
| B_SCI    | Science Block    |
| B_ENG    | Engineering      |
| B_LIB    | Library          |
| B_GYM    | Sports Complex   |
| B_ADMIN  | Admin Building   |

---

## Stretch Features Implemented

- ✅ **Accessible parking mode** (`--accessible`): filters lots by `accessible_spaces > 0`
- ✅ **Staff priority mode** (`--staff`): filters lots by `reserved_staff > 0`
- ✅ **Event flag** (`--event`): boosts predicted occupancy on event days
- ✅ **DFS comparison**: included alongside BFS and A* in algorithm comparison chart

---

## Output Files

After running, `output/` contains:
- `campus_graph.png` — campus topology with highlighted A* route
- `occupancy_chart.png` — predicted occupancy per lot with free-space annotations
- `algorithm_comparison.png` — BFS vs DFS vs A* node expansion and distance comparison
