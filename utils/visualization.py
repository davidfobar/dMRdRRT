from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from agent import Agent

def plot_agent(
    agent: Agent,
    output_path: Path | None = None,
    show: bool = True,
    planner_name: str = "RRT",
    title: str | None = None,
    include_grade_obstacles: bool = True,
) -> None:
    """Render a field plus one agent's obstacle overlays and planner state."""
    field_title = title if title is not None else agent.field.default_agent_plot_title(planner_name)

    fig, ax = agent.field.plot(show=False, title=field_title, finalize=False)
    _overlay_agent_constraints(agent, ax, include_grade_obstacles=include_grade_obstacles)
    _overlay_agent_planner_state(agent, ax, planner_name)
    agent.field.finalize_plot(fig, output_path, show)

def _overlay_agent_constraints(
    agent: Agent,
    ax: object,
    *,
    include_grade_obstacles: bool,
) -> None:
    max_grade = agent.max_grade if include_grade_obstacles else None
    agent.field.overlay_obstacle_regions(ax, max_grade=max_grade)

def _overlay_agent_planner_state(agent: Agent, ax: object, planner_name: str) -> None:
    nodes = agent.planner.nodes if agent.planner is not None else []
    roadmap_edges = getattr(agent.planner, "roadmap_edges", []) if agent.planner is not None else []
    tree_style = agent.field.planner_tree_style()
    path_style = agent.field.planner_path_style()

    if nodes and roadmap_edges:
        for i, j in roadmap_edges:
            p = nodes[i]
            q = nodes[j]
            ax.plot(
                [p.x, q.x],
                [p.y, q.y],
                color=tree_style["color"],
                linewidth=tree_style["linewidth"],
                alpha=tree_style["alpha"],
            )
    elif nodes:
        for node in nodes:
            if node.parent is None:
                continue
            parent = nodes[node.parent]
            ax.plot(
                [parent.x, node.x],
                [parent.y, node.y],
                color=tree_style["color"],
                linewidth=tree_style["linewidth"],
                alpha=tree_style["alpha"],
            )

    if agent.path:
        xs = [point[0] for point in agent.path]
        ys = [point[1] for point in agent.path]
        ax.plot(xs, ys, color=path_style["color"], linewidth=path_style["linewidth"], label=f"{planner_name} path")

    ax.scatter(agent.last_plan_start[0], agent.last_plan_start[1], c="tab:green", s=80, label="start", zorder=5)
    if agent.last_goal is not None:
        ax.scatter(agent.last_goal[0], agent.last_goal[1], c="tab:blue", s=80, label="goal", zorder=5)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left")
