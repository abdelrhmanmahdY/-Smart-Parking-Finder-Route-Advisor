"""
gui.py
------
CustomTkinter GUI for the Campus Parking Recommendation System.
CET251 – Artificial Intelligence | El Sewedy University of Technology

Layout
------
  LEFT PANEL   : inputs (destination, time, preference, flags)
  CENTER PANEL : results (recommendation card + route + alternatives)
  RIGHT PANEL  : tabbed charts (campus graph | occupancy | algo comparison)

Run with:
    python gui.py
"""

import sys, os, threading, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import customtkinter as ctk
from tkinter import messagebox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import networkx as nx

from campus_graph   import CAMPUS_GRAPH, NODES, PARKING_LOTS
from occupancy_model import OccupancyPredictor
from search_algorithms import compare_algorithms
from parking_agent import ParkingAgent

# ── appearance ──────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT     = "#3B82F6"
ACCENT2    = "#10B981"
WARN       = "#F59E0B"
DANGER     = "#EF4444"
BG_DARK    = "#0F172A"
BG_CARD    = "#1E293B"
BG_INPUT   = "#334155"
TXT        = "#F1F5F9"
TXT_MUTED  = "#94A3B8"
FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_HEAD  = ("Segoe UI", 13, "bold")
FONT_BODY  = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 10)
FONT_MONO  = ("Courier New", 10)

BUILDINGS = {
    "B_MAIN":  "Main Hall",
    "B_SCI":   "Science Block",
    "B_ENG":   "Engineering",
    "B_LIB":   "Library",
    "B_GYM":   "Sports Complex",
    "B_ADMIN": "Admin Building",
}
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
PREFS = ["available", "nearest", "fastest"]

# node type colours for graph
NODE_COLOURS = {"building": "#3B82F6", "lot": "#F59E0B", "junction": "#475569"}


# ════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Campus Parking Recommender — CET251 AI")
        self.geometry("1380x820")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_DARK)

        self.agent   = ParkingAgent()
        self.decision = None
        self._model_ready = False
        self._train_thread = None

        self._build_header()
        self._build_body()
        self._build_statusbar()

        # train model in background on launch
        self._set_status("⏳  Training neural network…", WARN)
        self._train_thread = threading.Thread(target=self._bg_train, daemon=True)
        self._train_thread.start()

    # ── header ──────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=62)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        ctk.CTkLabel(hdr, text="🅿  Campus Parking Recommender",
                     font=FONT_TITLE, text_color=TXT).pack(side="left", padx=20, pady=10)
        ctk.CTkLabel(hdr, text="CET251 · El Sewedy University of Technology",
                     font=FONT_SMALL, text_color=TXT_MUTED).pack(side="left", padx=4)

        self.lbl_model = ctk.CTkLabel(hdr, text="● Model loading…",
                                       font=FONT_SMALL, text_color=WARN)
        self.lbl_model.pack(side="right", padx=20)

    # ── three-column body ────────────────────────────────────
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color=BG_DARK)
        body.pack(fill="both", expand=True, padx=10, pady=(6,0))
        body.columnconfigure(0, weight=0, minsize=270)
        body.columnconfigure(1, weight=1, minsize=380)
        body.columnconfigure(2, weight=2, minsize=520)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_center(body)
        self._build_right(body)

    # ── LEFT: inputs ─────────────────────────────────────────
    def _build_left(self, parent):
        frame = ctk.CTkScrollableFrame(parent, fg_color=BG_CARD,
                                        corner_radius=12, width=260)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0,6), pady=4)

        def section(text):
            ctk.CTkLabel(frame, text=text, font=FONT_HEAD,
                         text_color=ACCENT).pack(anchor="w", padx=14, pady=(14,4))

        # ── Destination ──
        section("🏛  Destination")
        self.var_dest = ctk.StringVar(value="B_SCI")
        dest_menu = ctk.CTkOptionMenu(frame, variable=self.var_dest,
                                       values=list(BUILDINGS.keys()),
                                       dynamic_resizing=False, width=230,
                                       fg_color=BG_INPUT, button_color=ACCENT,
                                       command=lambda _: self._update_dest_label())
        dest_menu.pack(padx=14, pady=2)
        self.lbl_dest_name = ctk.CTkLabel(frame, text=BUILDINGS["B_SCI"],
                                           font=FONT_SMALL, text_color=TXT_MUTED)
        self.lbl_dest_name.pack(anchor="w", padx=16)

        # ── Arrival time ──
        section("🕐  Arrival Time")
        ctk.CTkLabel(frame, text="Hour (0 – 23)", font=FONT_SMALL,
                     text_color=TXT_MUTED).pack(anchor="w", padx=16)
        self.var_hour = ctk.IntVar(value=9)
        hour_slider = ctk.CTkSlider(frame, from_=0, to=23, number_of_steps=23,
                                     variable=self.var_hour, width=230,
                                     button_color=ACCENT,
                                     command=lambda v: self._update_hour_label(int(v)))
        hour_slider.pack(padx=14, pady=4)
        self.lbl_hour = ctk.CTkLabel(frame, text="09:00",
                                      font=("Segoe UI", 16, "bold"), text_color=TXT)
        self.lbl_hour.pack()

        ctk.CTkLabel(frame, text="Day of week", font=FONT_SMALL,
                     text_color=TXT_MUTED).pack(anchor="w", padx=16, pady=(10,2))
        self.var_day = ctk.StringVar(value="Tuesday")
        ctk.CTkOptionMenu(frame, variable=self.var_day, values=DAYS,
                           dynamic_resizing=False, width=230,
                           fg_color=BG_INPUT, button_color=ACCENT).pack(padx=14)

        # ── Preference ──
        section("⚙  Preference")
        self.var_pref = ctk.StringVar(value="available")
        for p in PREFS:
            ctk.CTkRadioButton(frame, text=p.capitalize(),
                                variable=self.var_pref, value=p,
                                text_color=TXT, fg_color=ACCENT).pack(anchor="w", padx=18, pady=2)

        # ── Flags ──
        section("🔧  Options")
        self.var_event      = ctk.BooleanVar(value=False)
        self.var_accessible = ctk.BooleanVar(value=False)
        self.var_staff      = ctk.BooleanVar(value=False)
        for var, label in [(self.var_event,      "Campus event today"),
                            (self.var_accessible, "Accessible parking only"),
                            (self.var_staff,      "Staff lots only")]:
            ctk.CTkCheckBox(frame, text=label, variable=var,
                             text_color=TXT, fg_color=ACCENT,
                             checkmark_color="white").pack(anchor="w", padx=18, pady=3)

        # ── Run button ──
        self.btn_run = ctk.CTkButton(frame, text="🔍  Find Parking",
                                      font=("Segoe UI", 13, "bold"),
                                      fg_color=ACCENT, hover_color="#2563EB",
                                      height=42, corner_radius=10,
                                      command=self._run)
        self.btn_run.pack(padx=14, pady=18, fill="x")

    # ── CENTER: results ──────────────────────────────────────
    def _build_center(self, parent):
        self.center = ctk.CTkScrollableFrame(parent, fg_color=BG_DARK,
                                              corner_radius=0)
        self.center.grid(row=0, column=1, sticky="nsew", padx=(0,6), pady=4)

        # placeholder
        self.lbl_placeholder = ctk.CTkLabel(
            self.center,
            text="Select a destination and\nclick Find Parking ↙",
            font=("Segoe UI", 14), text_color=TXT_MUTED
        )
        self.lbl_placeholder.pack(expand=True, pady=80)

    def _clear_center(self):
        for w in self.center.winfo_children():
            w.destroy()

    def _show_results(self, decision):
        self._clear_center()
        d = decision
        b = d.best

        # ── top recommendation card ──
        card = ctk.CTkFrame(self.center, fg_color=BG_CARD, corner_radius=12)
        card.pack(fill="x", padx=6, pady=(6,4))

        occ_pct = b.predicted_occ
        if occ_pct < 0.6:
            occ_col = ACCENT2
        elif occ_pct < 0.85:
            occ_col = WARN
        else:
            occ_col = DANGER

        ctk.CTkLabel(card, text="★  RECOMMENDED LOT",
                     font=FONT_SMALL, text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(10,0))
        ctk.CTkLabel(card, text=b.lot_label,
                     font=("Segoe UI", 20, "bold"), text_color=TXT).pack(anchor="w", padx=14)

        stats = ctk.CTkFrame(card, fg_color="transparent")
        stats.pack(fill="x", padx=14, pady=8)
        stats.columnconfigure((0,1,2,3), weight=1)

        def stat_block(parent, col, icon, value, label, colour=TXT):
            f = ctk.CTkFrame(parent, fg_color=BG_INPUT, corner_radius=8)
            f.grid(row=0, column=col, padx=4, pady=2, sticky="ew")
            ctk.CTkLabel(f, text=icon, font=("Segoe UI", 18)).pack(pady=(8,0))
            ctk.CTkLabel(f, text=value, font=("Segoe UI", 15, "bold"),
                         text_color=colour).pack()
            ctk.CTkLabel(f, text=label, font=FONT_SMALL,
                         text_color=TXT_MUTED).pack(pady=(0,8))

        stat_block(stats, 0, "🅿", b.lot_label.split()[0], "Lot", ACCENT)
        stat_block(stats, 1, "📊", f"{occ_pct:.0%}", "Occupancy", occ_col)
        stat_block(stats, 2, "🚶", f"{b.walk_minutes} min", "Walk time", ACCENT2)
        stat_block(stats, 3, "📍", f"{b.free_spaces}", "Free spaces", ACCENT2)

        # occupancy progress bar
        bar_frame = ctk.CTkFrame(card, fg_color="transparent")
        bar_frame.pack(fill="x", padx=14, pady=(0,10))
        ctk.CTkLabel(bar_frame, text=f"Predicted occupancy: {occ_pct:.1%}",
                     font=FONT_SMALL, text_color=TXT_MUTED).pack(anchor="w")
        pb = ctk.CTkProgressBar(bar_frame, height=10, corner_radius=5,
                                  progress_color=occ_col, fg_color=BG_INPUT)
        pb.pack(fill="x", pady=3)
        pb.set(occ_pct)

        # ── A* Route ──
        route_card = ctk.CTkFrame(self.center, fg_color=BG_CARD, corner_radius=12)
        route_card.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(route_card, text="🗺  A* ROUTE",
                     font=FONT_HEAD, text_color=ACCENT).pack(anchor="w", padx=14, pady=(10,4))

        route_path = d.best.route_result.path
        route_row  = ctk.CTkFrame(route_card, fg_color="transparent")
        route_row.pack(fill="x", padx=14, pady=(0,6))

        for i, nid in enumerate(route_path):
            label = NODES[nid]["label"] if nid in NODES else nid
            ntype = NODES[nid].get("type","junction") if nid in NODES else "junction"
            icon  = "🅿" if ntype=="lot" else ("🏛" if ntype=="building" else "•")
            col   = ACCENT if ntype=="lot" else (ACCENT2 if ntype=="building" else TXT_MUTED)
            ctk.CTkLabel(route_row, text=f"{icon} {label}",
                         font=FONT_SMALL, text_color=col).pack(side="left")
            if i < len(route_path)-1:
                ctk.CTkLabel(route_row, text=" → ", font=FONT_SMALL,
                             text_color=TXT_MUTED).pack(side="left")

        ctk.CTkLabel(route_card,
                     text=f"  {d.best.route_result.total_distance:.0f} m  ·  "
                          f"{d.best.route_result.walking_minutes} min walk  ·  "
                          f"{d.best.route_result.nodes_explored} nodes explored (A*)",
                     font=FONT_SMALL, text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(0,10))

        # ── Explanation ──
        exp_card = ctk.CTkFrame(self.center, fg_color=BG_CARD, corner_radius=12)
        exp_card.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(exp_card, text="🤖  WHY THIS LOT?",
                     font=FONT_HEAD, text_color=ACCENT).pack(anchor="w", padx=14, pady=(10,4))
        ctk.CTkLabel(exp_card, text=d.explanation, font=FONT_SMALL,
                     text_color=TXT, wraplength=330, justify="left").pack(anchor="w", padx=14, pady=(0,12))

        # ── Alternatives ──
        alt_card = ctk.CTkFrame(self.center, fg_color=BG_CARD, corner_radius=12)
        alt_card.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(alt_card, text="🔄  ALTERNATIVES",
                     font=FONT_HEAD, text_color=ACCENT).pack(anchor="w", padx=14, pady=(10,4))

        for alt in d.alternatives:
            row = ctk.CTkFrame(alt_card, fg_color=BG_INPUT, corner_radius=8)
            row.pack(fill="x", padx=14, pady=3)
            row.columnconfigure(1, weight=1)

            oc = alt.predicted_occ
            ac = ACCENT2 if oc < 0.6 else (WARN if oc < 0.85 else DANGER)

            ctk.CTkLabel(row, text=f"#{alt.rank}", font=FONT_SMALL,
                         text_color=TXT_MUTED, width=28).grid(row=0, column=0, padx=8, pady=6)
            ctk.CTkLabel(row, text=alt.lot_label, font=FONT_BODY,
                         text_color=TXT, anchor="w").grid(row=0, column=1, sticky="w", padx=4)
            ctk.CTkLabel(row, text=f"occ {oc:.0%}", font=FONT_SMALL,
                         text_color=ac).grid(row=0, column=2, padx=8)
            ctk.CTkLabel(row, text=f"{alt.free_spaces} free", font=FONT_SMALL,
                         text_color=ACCENT2).grid(row=0, column=3, padx=4)
            ctk.CTkLabel(row, text=f"{alt.walk_minutes} min", font=FONT_SMALL,
                         text_color=TXT_MUTED).grid(row=0, column=4, padx=8)

        ctk.CTkFrame(alt_card, height=8, fg_color="transparent").pack()

        # ── Algorithm comparison text ──
        bfs_res = None
        from search_algorithms import bfs
        bfs_res = bfs(d.best.lot_id, d.destination)
        if bfs_res:
            algo_card = ctk.CTkFrame(self.center, fg_color=BG_CARD, corner_radius=12)
            algo_card.pack(fill="x", padx=6, pady=(4,10))
            ctk.CTkLabel(algo_card, text="🔬  ALGORITHM COMPARISON",
                         font=FONT_HEAD, text_color=ACCENT).pack(anchor="w", padx=14, pady=(10,4))

            astar_res = d.best.route_result
            rows_data = [
                ("BFS",  bfs_res.nodes_explored,  bfs_res.total_distance),
                ("A*",   astar_res.nodes_explored, astar_res.total_distance),
            ]
            hdr_row = ctk.CTkFrame(algo_card, fg_color="transparent")
            hdr_row.pack(fill="x", padx=14)
            for col, txt in enumerate(["Algorithm","Nodes Explored","Distance (m)"]):
                ctk.CTkLabel(hdr_row, text=txt, font=("Segoe UI",10,"bold"),
                             text_color=TXT_MUTED, width=120 if col>0 else 90,
                             anchor="w").grid(row=0, column=col, padx=4)

            for alg, nodes, dist in rows_data:
                r = ctk.CTkFrame(algo_card, fg_color=BG_INPUT, corner_radius=6)
                r.pack(fill="x", padx=14, pady=2)
                col_c = ACCENT2 if alg=="A*" else TXT
                ctk.CTkLabel(r, text=alg, font=("Segoe UI",11,"bold"),
                             text_color=col_c, width=90, anchor="w").grid(row=0,column=0,padx=8,pady=6)
                ctk.CTkLabel(r, text=str(nodes), font=FONT_BODY,
                             text_color=col_c, width=120, anchor="w").grid(row=0,column=1,padx=4)
                ctk.CTkLabel(r, text=f"{dist:.0f}", font=FONT_BODY,
                             text_color=col_c, width=120, anchor="w").grid(row=0,column=2,padx=4)

            ctk.CTkLabel(algo_card,
                         text="A* uses a Euclidean heuristic — finds optimal path with fewer node expansions than BFS.",
                         font=FONT_SMALL, text_color=TXT_MUTED,
                         wraplength=340, justify="left").pack(anchor="w", padx=14, pady=(4,12))

    # ── RIGHT: charts ────────────────────────────────────────
    def _build_right(self, parent):
        right = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        right.grid(row=0, column=2, sticky="nsew", padx=(0,0), pady=4)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="📊  Visualisations",
                     font=FONT_HEAD, text_color=ACCENT).grid(row=0, column=0, sticky="w",
                                                              padx=14, pady=(10,4))
        self.tabs = ctk.CTkTabview(right, fg_color=BG_DARK,
                                    segmented_button_fg_color=BG_INPUT,
                                    segmented_button_selected_color=ACCENT,
                                    segmented_button_selected_hover_color="#2563EB",
                                    text_color=TXT)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0,8))

        for tab in ("Campus Graph", "Occupancy", "Algo Comparison"):
            self.tabs.add(tab)

        # embed placeholder canvas in each tab
        self.fig_graph, self.ax_graph = plt.subplots(figsize=(6,5))
        self.fig_graph.patch.set_facecolor(BG_DARK)
        self._canvas_graph = self._embed_fig(self.fig_graph, self.tabs.tab("Campus Graph"))

        self.fig_occ, self.ax_occ = plt.subplots(figsize=(6,4))
        self.fig_occ.patch.set_facecolor(BG_DARK)
        self._canvas_occ = self._embed_fig(self.fig_occ, self.tabs.tab("Occupancy"))

        self.fig_algo, self.ax_algo = plt.subplots(1,2, figsize=(6,4))
        self.fig_algo.patch.set_facecolor(BG_DARK)
        self._canvas_algo = self._embed_fig(self.fig_algo, self.tabs.tab("Algo Comparison"))

        # draw initial empty campus graph
        self._draw_campus_graph(decision=None)

    def _embed_fig(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    # ── chart drawing ────────────────────────────────────────
    def _draw_campus_graph(self, decision=None):
        ax = self.ax_graph
        ax.clear()
        ax.set_facecolor(BG_DARK)
        self.fig_graph.patch.set_facecolor(BG_DARK)

        G   = CAMPUS_GRAPH
        pos = {n: data["pos"] for n, data in NODES.items()}

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#334155", width=1.5, alpha=0.8)

        edge_labels = {(u,v): f"{d['weight']}m" for u,v,d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                     font_size=5, font_color="#64748B",
                                     bbox=dict(boxstyle="round,pad=0.1",
                                               fc=BG_DARK, ec="none", alpha=0.7))

        if decision:
            rp = decision.best.route_result.path
            nx.draw_networkx_edges(G, pos, edgelist=list(zip(rp, rp[1:])),
                                   ax=ax, edge_color=ACCENT2, width=4, alpha=0.95)

        for node, data in NODES.items():
            col = NODE_COLOURS.get(data["type"], "#475569")
            if decision and node == decision.best.lot_id:
                col = DANGER
            elif decision and node == decision.destination:
                col = ACCENT2
            nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax,
                                   node_color=col, node_size=280)

        labels = {n: NODES[n]["label"].replace(" ","\n") for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                font_size=5.5, font_color="white")

        handles = [
            mpatches.Patch(color=NODE_COLOURS["building"], label="Building"),
            mpatches.Patch(color=NODE_COLOURS["lot"],      label="Parking Lot"),
            mpatches.Patch(color=DANGER,                   label="Recommended"),
            mpatches.Patch(color=ACCENT2,                  label="Destination / Route"),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=7,
                  facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
        ax.set_title("Campus Walking Graph", color=TXT, fontsize=9, pad=6)
        ax.axis("off")
        self.fig_graph.tight_layout()
        self._canvas_graph.draw()

    def _draw_occupancy_chart(self, occupancy: dict):
        ax = self.ax_occ
        ax.clear()
        ax.set_facecolor(BG_DARK)
        self.fig_occ.patch.set_facecolor(BG_DARK)

        lots   = sorted(occupancy)
        occ    = [occupancy[l]*100 for l in lots]
        free   = [int(PARKING_LOTS[l]["capacity"]*(1-occupancy[l])) for l in lots]
        labels = [NODES[l]["label"] for l in lots]
        colours= [DANGER if o>85 else WARN if o>60 else ACCENT2 for o in occ]

        bars = ax.bar(labels, occ, color=colours, width=0.55,
                      edgecolor=BG_DARK, linewidth=1)
        for bar, f in zip(bars, free):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                    f"{f} free", ha="center", va="bottom",
                    fontsize=7, color="white")

        ax.set_ylim(0, 115)
        ax.axhline(85, color=DANGER, lw=1, ls="--", alpha=0.6)
        ax.set_ylabel("Predicted Occupancy (%)", color=TXT_MUTED, fontsize=8)
        ax.set_title("Lot Occupancy Prediction", color=TXT, fontsize=9, pad=6)
        ax.tick_params(colors=TXT_MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#334155")
        self.fig_occ.tight_layout()
        self._canvas_occ.draw()

    def _draw_algo_chart(self, decision):
        from search_algorithms import bfs, dfs, astar
        for ax in self.ax_algo:
            ax.clear()
            ax.set_facecolor(BG_DARK)
        self.fig_algo.patch.set_facecolor(BG_DARK)

        src, tgt = decision.best.lot_id, decision.destination
        results  = {}
        for name, fn in [("BFS", bfs), ("DFS", dfs), ("A*", astar)]:
            r = fn(src, tgt)
            if r:
                results[name] = r

        if not results:
            return

        names    = list(results)
        explored = [results[n].nodes_explored  for n in names]
        dists    = [results[n].total_distance  for n in names]
        colours  = [ACCENT, "#8B5CF6", ACCENT2][:len(names)]

        ax1, ax2 = self.ax_algo
        ax1.bar(names, explored, color=colours, width=0.5, edgecolor=BG_DARK)
        ax1.set_title("Nodes Explored", color=TXT, fontsize=8)
        ax1.set_ylabel("Count", color=TXT_MUTED, fontsize=7)
        for i, v in enumerate(explored):
            ax1.text(i, v+0.2, str(v), ha="center", color="white", fontsize=8)

        ax2.bar(names, dists, color=colours, width=0.5, edgecolor=BG_DARK)
        ax2.set_title("Path Distance (m)", color=TXT, fontsize=8)
        ax2.set_ylabel("Metres", color=TXT_MUTED, fontsize=7)
        for i, v in enumerate(dists):
            ax2.text(i, v+3, f"{v:.0f}", ha="center", color="white", fontsize=8)

        for ax in self.ax_algo:
            ax.tick_params(colors=TXT_MUTED, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#334155")

        src_l = NODES[src]["label"]
        tgt_l = NODES[tgt]["label"]
        self.fig_algo.suptitle(f"{src_l} → {tgt_l}",
                                color=TXT, fontsize=8)
        self.fig_algo.tight_layout()
        self._canvas_algo.draw()

    # ── status bar ───────────────────────────────────────────
    def _build_statusbar(self):
        self.statusbar = ctk.CTkFrame(self, fg_color=BG_CARD,
                                       corner_radius=0, height=28)
        self.statusbar.pack(fill="x", side="bottom")
        self.statusbar.pack_propagate(False)
        self.lbl_status = ctk.CTkLabel(self.statusbar, text="Ready",
                                        font=FONT_SMALL, text_color=TXT_MUTED)
        self.lbl_status.pack(side="left", padx=14)

    def _set_status(self, msg, colour=TXT_MUTED):
        self.lbl_status.configure(text=msg, text_color=colour)

    # ── helpers ──────────────────────────────────────────────
    def _update_dest_label(self):
        key = self.var_dest.get()
        self.lbl_dest_name.configure(text=BUILDINGS.get(key, ""))

    def _update_hour_label(self, h: int):
        self.lbl_hour.configure(text=f"{h:02d}:00")

    # ── background model training ────────────────────────────
    def _bg_train(self):
        metrics = self.agent.predictor.train(verbose=False)
        self.agent._ready = True
        self._model_ready = True
        self.after(0, self._on_model_ready, metrics)

    def _on_model_ready(self, metrics):
        mae = metrics["MAE"]
        r2  = metrics["R²"]
        self.lbl_model.configure(
            text=f"● Neural Net Ready  MAE={mae}  R²={r2}",
            text_color=ACCENT2
        )
        self._set_status(f"Model trained — MAE: {mae} | R²: {r2} | Ready to search", ACCENT2)
        # draw default occupancy for Tue 9am
        occ = self.agent.predictor.predict_all_lots(9, 1, 0)
        self._draw_occupancy_chart(occ)

    # ── main run ─────────────────────────────────────────────
    def _run(self):
        if not self._model_ready:
            messagebox.showwarning("Not ready", "Neural network is still training. Please wait.")
            return

        dest   = self.var_dest.get()
        hour   = int(self.var_hour.get())
        day    = DAYS.index(self.var_day.get())
        pref   = self.var_pref.get()
        event  = int(self.var_event.get())
        acc    = self.var_accessible.get()
        staff  = self.var_staff.get()

        self.btn_run.configure(state="disabled", text="⏳  Searching…")
        self._set_status("Running agent…", WARN)

        def worker():
            try:
                decision = self.agent.recommend(
                    destination     = dest,
                    arrival_hour    = hour,
                    arrival_weekday = day,
                    preference      = pref,
                    event           = event,
                    accessible_only = acc,
                    staff_only      = staff,
                )
                self.after(0, self._on_result, decision)
            except Exception as e:
                self.after(0, self._on_error, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_result(self, decision):
        self.decision = decision
        self._show_results(decision)
        self._draw_campus_graph(decision)
        occ = self.agent.predictor.predict_all_lots(
            decision.arrival_hour, decision.arrival_weekday,
            getattr(decision, "event", 0)
        )
        self._draw_occupancy_chart(occ)
        self._draw_algo_chart(decision)
        self.btn_run.configure(state="normal", text="🔍  Find Parking")
        self._set_status(
            f"✔  Recommended: {decision.best.lot_label}  |  "
            f"{decision.best.walk_minutes} min walk  |  "
            f"{decision.best.free_spaces} free spaces", ACCENT2
        )

    def _on_error(self, msg):
        self.btn_run.configure(state="normal", text="🔍  Find Parking")
        self._set_status(f"Error: {msg}", DANGER)
        messagebox.showerror("Error", msg)


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()