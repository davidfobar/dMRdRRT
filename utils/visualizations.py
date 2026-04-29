"""
Two visualizations:
  1. plot_robustness_figure  — static two-panel robustness figure
  2. animate_algorithm       — three-phase MRdRRT algorithm animation
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from collections import Counter


def _copy_terrain_to_ax(field, ax):
    """
    Render the field into a temporary figure, copy the terrain imshow
    onto `ax` using the field's coordinate bounds, then close the temp figure.
    This avoids passing ax= to field.plot() which is unsupported.
    """
    xmin, xmax, ymin, ymax = field.bounds
    tmp_fig, tmp_ax = field.plot(show=False, title="", finalize=False)
    images = tmp_ax.get_images()
    if images:
        img   = images[0]
        ax.imshow(
            img.get_array(),
            cmap=img.cmap,
            norm=img.norm,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            aspect="auto",
            zorder=0,
        )
    plt.close(tmp_fig)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


# ===============================================================================
#  1. STATIC R-ROBUSTNESS FIGURE
# ===============================================================================

def plot_robustness_figure(
    field,
    *,
    prm,
    lcc: list,
    robot_paths,
    start_ids,
    goal_ids,
    output_path="robustness_figure.png",
    show=False,
):
    lcc_set = set(lcc)

    internal_degree = {
        nid: sum(1 for (nb, _) in prm.graph[nid] if nb in lcc_set)
        for nid in lcc
    }
    r_lb        = min(internal_degree.values())
    max_degree  = max(internal_degree.values())
    mean_degree = sum(internal_degree.values()) / len(internal_degree)
    bottlenecks = [nid for nid, d in internal_degree.items() if d == r_lb]

    coord_to_nid = {
        (round(prm.nodes[i].x, 1), round(prm.nodes[i].y, 1)): i
        for i in range(len(prm.nodes))
    }
    used_node_edges = set()
    for path in robot_paths:
        for k in range(len(path) - 1):
            a  = (round(float(path[k][0]),   1), round(float(path[k][1]),   1))
            b  = (round(float(path[k+1][0]), 1), round(float(path[k+1][1]), 1))
            na, nb_ = coord_to_nid.get(a), coord_to_nid.get(b)
            if na is not None and nb_ is not None:
                used_node_edges.add(frozenset([na, nb_]))

    fig = plt.figure(figsize=(16, 7), dpi=110, facecolor="#0d0d1a")
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.08)
    ax_map  = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # terrain
    _copy_terrain_to_ax(field, ax_map)

    # PRM edges
    for i, j in prm.roadmap_edges:
        p, q  = prm.nodes[i], prm.nodes[j]
        used  = frozenset([i, j]) in used_node_edges
        ax_map.plot(
            [p.x, q.x], [p.y, q.y],
            color="white" if used else "0.6",
            lw=1.8 if used else 0.35,
            alpha=0.9 if used else 0.18,
            zorder=2 if used else 1,
        )

    isolated = [i for i in range(len(prm.nodes)) if i not in lcc_set]
    if isolated:
        ax_map.scatter(
            [prm.nodes[i].x for i in isolated],
            [prm.nodes[i].y for i in isolated],
            c="crimson", s=4, alpha=0.45, zorder=3,
            label=f"Isolated ({len(isolated)})",
        )

    cmap = plt.cm.plasma
    norm = Normalize(vmin=r_lb, vmax=max_degree)
    sc   = ax_map.scatter(
        [prm.nodes[i].x for i in lcc],
        [prm.nodes[i].y for i in lcc],
        c=[internal_degree[i] for i in lcc],
        cmap=cmap, norm=norm, s=12, zorder=4, alpha=0.85,
    )
    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.03, pad=0.01)
    cbar.set_label("Degree within LCC", fontsize=9, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax_map.scatter(
        [prm.nodes[i].x for i in bottlenecks],
        [prm.nodes[i].y for i in bottlenecks],
        facecolors="none", edgecolors="red",
        s=140, linewidths=2.5, zorder=6,
        label=f"Bottleneck (deg={r_lb}, n={len(bottlenecks)})",
    )

    colors = ["cyan", "tomato", "lime", "yellow", "magenta", "orange"]
    for idx, path in enumerate(robot_paths):
        xs, ys = [p[0] for p in path], [p[1] for p in path]
        col = colors[idx % len(colors)]
        ax_map.plot(xs, ys, color=col, lw=2.5, zorder=7,
                    path_effects=[pe.Stroke(linewidth=4.5, foreground="black"),
                                  pe.Normal()],
                    label=f"Robot {idx+1}")
        ax_map.scatter(xs[0],  ys[0],  c="lime", s=90,  zorder=8, edgecolors="black")
        ax_map.scatter(xs[-1], ys[-1], c=col,    s=200, zorder=8,
                       marker="*", edgecolors="black")

    ax_map.set_title(
        f"r-Robustness Map  |  LCC: {len(lcc)}/{len(prm.nodes)} nodes  |  "
        f"r ≥ {r_lb}  |  {len(bottlenecks)} bottleneck(s)",
        fontsize=10, fontweight="bold", color="white", pad=8,
    )
    ax_map.tick_params(colors="white")
    ax_map.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # histogram
    ax_hist.set_facecolor("#1a1a2e")
    deg_counts = Counter(internal_degree.values())
    degrees    = sorted(deg_counts.keys())
    counts     = [deg_counts[d] for d in degrees]

    bars = ax_hist.bar(degrees, counts,
                       color=[cmap(norm(d)) for d in degrees],
                       edgecolor="black", linewidth=0.6, zorder=3)
    for bar, d in zip(bars, degrees):
        if d == r_lb:
            bar.set_edgecolor("red")
            bar.set_linewidth(2.5)

    ax_hist.axvline(r_lb,        color="red",   lw=2,   ls="--", zorder=4,
                    label=f"r lower bound = {r_lb}")
    ax_hist.axvline(mean_degree, color="white", lw=1.5, ls=":",  zorder=4,
                    label=f"mean = {mean_degree:.1f}")

    ax_hist.set_xlabel("Degree within LCC", fontsize=10, color="white")
    ax_hist.set_ylabel("Node count",        fontsize=10, color="white")
    ax_hist.set_title("Degree Distribution", fontsize=11,
                       fontweight="bold", color="white")
    ax_hist.tick_params(colors="white")
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.spines[["bottom", "left"]].set_color("0.5")
    ax_hist.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    ax_hist.text(
        0.97, 0.97,
        f"r-robustness ≥ {r_lb}\n\n"
        f"LCC:         {len(lcc)} / {len(prm.nodes)} nodes\n"
        f"Isolated:    {len(isolated)} nodes\n"
        f"Bottlenecks: {len(bottlenecks)}\n"
        f"Mean deg:    {mean_degree:.1f}\n"
        f"Max deg:     {max_degree}",
        transform=ax_hist.transAxes, fontsize=9, va="top", ha="right",
        color="white", family="monospace",
        bbox=dict(boxstyle="round", facecolor="#0d0d1a", alpha=0.9),
    )

    plt.savefig(output_path, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved -> {output_path}")
    return output_path


# ===============================================================================
#  2. ALGORITHM ANIMATION
# ===============================================================================

def animate_algorithm(
    field,
    *,
    prm,
    mr_planner=None,        # optional — only needed to show tree growth (Phase 1)
    robot_paths,
    start_ids,
    goal_ids,
    lcc: list,
    output_path="algorithm.gif",
    interval=250,
    max_tree_frames=60,
    show_top_n_tree=300,
):
    lcc_set   = set(lcc)
    colors    = ["cyan", "tomato", "lime", "yellow", "magenta", "orange"]
    n_agents  = len(robot_paths)
    max_steps = max(len(p) for p in robot_paths)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=90, facecolor="#0d0d1a")
    fig.patch.set_facecolor("#0d0d1a")
    _copy_terrain_to_ax(field, ax)
    ax.set_title("MRdRRT — Algorithm View", color="white", fontweight="bold")
    ax.tick_params(colors="white")

    # Static PRM skeleton (short edges only)
    for i, j in prm.roadmap_edges:
        p, q = prm.nodes[i], prm.nodes[j]
        if ((p.x-q.x)**2 + (p.y-q.y)**2)**0.5 < 35:
            ax.plot([p.x, q.x], [p.y, q.y],
                    color="0.7", lw=0.3, alpha=0.2, zorder=1)

    isolated = [i for i in range(len(prm.nodes)) if i not in lcc_set]
    if isolated:
        ax.scatter([prm.nodes[i].x for i in isolated],
                   [prm.nodes[i].y for i in isolated],
                   c="crimson", s=3, alpha=0.3, zorder=2)

    for idx, sid in enumerate(start_ids):
        n = prm.nodes[sid]
        ax.scatter(n.x, n.y, c="lime", s=120, zorder=10, edgecolors="black")
    for idx, gid in enumerate(goal_ids):
        n = prm.nodes[gid]
        ax.scatter(n.x, n.y, c=colors[idx % len(colors)],
                   s=300, zorder=10, edgecolors="black", marker="*")

    # Tree edge artists — skipped if mr_planner is not provided
    tree_coords, tree_artists = [], []
    if mr_planner is not None:
        tree_nodes = mr_planner.nodes
        step_size  = max(1, len(tree_nodes) // show_top_n_tree)
        for node in tree_nodes[::step_size]:
            if node.parent is not None:
                parent = tree_nodes[node.parent]
                cp = prm.nodes[node.states[0]]
                pp = prm.nodes[parent.states[0]]
                ln, = ax.plot([], [], color="deepskyblue", lw=0.8, alpha=0.5, zorder=3)
                tree_coords.append((pp.x, pp.y, cp.x, cp.y))
                tree_artists.append(ln)

    sol_artists = []
    for idx, path in enumerate(robot_paths):
        ln, = ax.plot([], [], color=colors[idx % len(colors)], lw=3, zorder=6,
                      path_effects=[pe.Stroke(linewidth=5, foreground="black"),
                                    pe.Normal()])
        sol_artists.append((ln, path))

    trails = [ax.plot([], [], color=colors[i % len(colors)], lw=2,
                      alpha=0.7, zorder=7)[0] for i in range(n_agents)]
    dots   = [ax.plot([], [], "o", color=colors[i % len(colors)], ms=12,
                      zorder=9, markeredgecolor="white")[0] for i in range(n_agents)]

    warn_patch = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.09, transform=ax.transAxes,
        boxstyle="round,pad=0.01", fc="red", alpha=0.0, zorder=20,
    )
    ax.add_patch(warn_patch)
    warn_text  = ax.text(0.5, 0.055, "", transform=ax.transAxes, fontsize=10,
                         ha="center", va="center", color="white",
                         fontweight="bold", zorder=21)
    phase_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=9,
                         va="top", color="white", fontweight="bold",
                         bbox=dict(boxstyle="round", fc="#111111", alpha=0.85))

    for i in range(n_agents):
        ax.plot([], [], color=colors[i % len(colors)], lw=2, label=f"Robot {i+1}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    n_tree_frames = min(max_tree_frames, len(tree_artists)) if tree_artists else 0
    n_sol_frames  = 15
    n_exec_frames = max_steps + 8
    total_frames  = n_tree_frames + n_sol_frames + n_exec_frames

    all_dynamic = (tree_artists + [ln for ln, _ in sol_artists] +
                   trails + dots + [warn_patch, warn_text, phase_text])

    def update(frame):
        # Phase 1
        if frame < n_tree_frames:
            reveal = int((frame + 1) / n_tree_frames * len(tree_artists))
            for k, (ln, (x1, y1, x2, y2)) in enumerate(zip(tree_artists, tree_coords)):
                if k < reveal:
                    ln.set_data([x1, x2], [y1, y2])
            phase_text.set_text(
                f"Phase 1 — Composite tree growing\n{reveal}/{len(tree_artists)} branches")
            warn_patch.set_alpha(0.0)
            warn_text.set_text("")
            return all_dynamic

        # Phase 2
        local = frame - n_tree_frames
        if local < n_sol_frames:
            t = (local + 1) / n_sol_frames
            for ln, path in sol_artists:
                n_show = max(2, int(t * len(path)))
                ln.set_data([p[0] for p in path[:n_show]],
                            [p[1] for p in path[:n_show]])
            phase_text.set_text("Phase 2 — Solution path extracted from tree")
            warn_patch.set_alpha(0.0)
            warn_text.set_text("")
            return all_dynamic

        # Phase 3
        step = min(frame - n_tree_frames - n_sol_frames, max_steps - 1)
        positions = []
        for i, (path, trail, dot) in enumerate(zip(robot_paths, trails, dots)):
            s = min(step, len(path) - 1)
            t_start = max(0, s - 6)
            trail.set_data([p[0] for p in path[t_start:s+1]],
                           [p[1] for p in path[t_start:s+1]])
            dot.set_data([path[s][0]], [path[s][1]])
            positions.append(np.array(path[s]))

        too_close = any(
            np.linalg.norm(positions[i] - positions[j]) < 30
            for i in range(n_agents) for j in range(i+1, n_agents)
        )
        warn_patch.set_alpha(0.35 if too_close else 0.0)
        warn_text.set_text(
            "⚠  Near collision — r≥2 guarantees a detour exists" if too_close else "")
        phase_text.set_text(
            f"Phase 3 — Executing plan\n"
            f"Step {step+1}/{max_steps}  |  r≥2 ensures collision-free routing")
        return all_dynamic

    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=interval, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=1000 // interval), dpi=90)
    plt.close(fig)
    print(f"Saved -> {output_path}")
    return output_path