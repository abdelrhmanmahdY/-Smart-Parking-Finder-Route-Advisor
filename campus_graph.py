

import networkx as nx


NODES = {
    "B_MAIN":   {"type": "building",  "label": "Main Hall",        "pos": (5, 9)},
    "B_SCI":    {"type": "building",  "label": "Science Block",    "pos": (8, 7)},
    "B_ENG":    {"type": "building",  "label": "Engineering",      "pos": (3, 7)},
    "B_LIB":    {"type": "building",  "label": "Library",          "pos": (5, 5)},
    "B_GYM":    {"type": "building",  "label": "Sports Complex",   "pos": (9, 3)},
    "B_ADMIN":  {"type": "building",  "label": "Admin Building",   "pos": (1, 5)},

    "P_NORTH":  {"type": "lot", "label": "North Lot",   "pos": (4, 11), "capacity": 80},
    "P_EAST":   {"type": "lot", "label": "East Lot",    "pos": (10, 7), "capacity": 60},
    "P_CENTRAL":{"type": "lot", "label": "Central Lot", "pos": (5, 7),  "capacity": 40},
    "P_WEST":   {"type": "lot", "label": "West Lot",    "pos": (0, 7),  "capacity": 50},
    "P_SOUTH":  {"type": "lot", "label": "South Lot",   "pos": (6, 1),  "capacity": 70},

    "J1": {"type": "junction", "label": "Junction 1", "pos": (5, 10)},
    "J2": {"type": "junction", "label": "Junction 2", "pos": (5, 8)},
    "J3": {"type": "junction", "label": "Junction 3", "pos": (7, 8)},
    "J4": {"type": "junction", "label": "Junction 4", "pos": (3, 8)},
    "J5": {"type": "junction", "label": "Junction 5", "pos": (2, 6)},
    "J6": {"type": "junction", "label": "Junction 6", "pos": (5, 6)},
    "J7": {"type": "junction", "label": "Junction 7", "pos": (8, 5)},
    "J8": {"type": "junction", "label": "Junction 8", "pos": (7, 2)},
}


EDGES = [
    # North lot → main entrance corridor
    ("P_NORTH", "J1",      80),
    ("J1",      "B_MAIN",  100),
    ("J1",      "J2",      90),

    # Central spine
    ("J2",      "B_MAIN",  70),
    ("J2",      "J3",      80),
    ("J2",      "J4",      70),
    ("J2",      "J6",      90),

    # East side
    ("J3",      "B_SCI",   60),
    ("J3",      "P_EAST",  120),
    ("J3",      "J7",      110),

    # West side
    ("J4",      "B_ENG",   80),
    ("J4",      "J5",      100),
    ("J5",      "P_WEST",  90),
    ("J5",      "B_ADMIN", 70),

    # Central lot
    ("P_CENTRAL","J2",     50),
    ("P_CENTRAL","J6",     60),

    # Library hub
    ("J6",      "B_LIB",   50),
    ("J6",      "J7",      130),
    ("J6",      "J8",      160),

    # South / gym
    ("J7",      "B_GYM",   90),
    ("J7",      "J8",      110),
    ("J8",      "P_SOUTH", 80),
    ("J8",      "B_GYM",   100),
    ("P_SOUTH", "J8",      80),
]


PARKING_LOTS = {
    "P_NORTH":   {"capacity": NODES["P_NORTH"]["capacity"],  "reserved_staff": 10, "accessible_spaces": 4},
    "P_EAST":    {"capacity": NODES["P_EAST"]["capacity"],  "reserved_staff": 8,  "accessible_spaces": 3},
    "P_CENTRAL": {"capacity": NODES["P_CENTRAL"]["capacity"],  "reserved_staff": 5,  "accessible_spaces": 2},
    "P_WEST":    {"capacity": NODES["P_WEST"]["capacity"],  "reserved_staff": 8,  "accessible_spaces": 3},
    "P_SOUTH":   {"capacity": NODES["P_SOUTH"]["capacity"],  "reserved_staff": 6,  "accessible_spaces": 4},
}

LOT_ID_MAP = {lot: idx for idx, lot in enumerate(sorted(PARKING_LOTS.keys()))}



def build_graph() -> nx.Graph:
    """Return a weighted undirected NetworkX graph of the campus."""
    G = nx.Graph()
    for node_id, attrs in NODES.items():
        G.add_node(node_id, **attrs)
    for u, v, w in EDGES:
        G.add_edge(u, v, weight=w)
    return G


CAMPUS_GRAPH: nx.Graph = build_graph()
