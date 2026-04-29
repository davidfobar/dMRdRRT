# ──────────────────────────────────────────────────────────────────────────────
# File: graphkit/viz.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Optional
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
Tensor = torch.Tensor
#from swarm.agent import AgentType  # local import keeps viz decoupled if desired


def plot(
    A: Tensor,
    positions: Optional[Tensor] = None,
    *,
    title: str = "",
    with_nodes: bool = True,
    node_size: float = 50.0,
    edge_alpha: float = 0.4,
    text_offset: float = 0.06,       # distance of index labels from node
    font_size: int = 10,
):
    """
    Plot adjacency matrix as a 2D network diagram with optional node labels.
    """
    A = (A > 0).to(torch.float32)
    M = A.shape[0]
    if positions is None:
        # circular layout
        theta = torch.linspace(0, 2 * math.pi, steps=M + 1)[:-1]
        positions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    pos = positions.cpu().numpy()

    # plot edges
    i, j = torch.nonzero(A, as_tuple=True)
    xs = list(zip(pos[i, 0], pos[j, 0]))
    ys = list(zip(pos[i, 1], pos[j, 1]))

    plt.figure(figsize=(5, 5))
    for (x1, x2), (y1, y2) in zip(xs, ys):
        plt.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=edge_alpha, color="gray")

    # plot nodes
    if with_nodes:
        plt.scatter(pos[:, 0], pos[:, 1], s=node_size, color="steelblue", zorder=10)

        # add agent indices as labels
        for idx, (x, y) in enumerate(pos):
            angle = math.atan2(y, x)
            dx = text_offset * math.cos(angle)
            dy = text_offset * math.sin(angle)
            plt.text(
                x + dx,
                y + dy,
                str(idx),
                ha="center",
                va="center",
                fontsize=font_size,
                weight="bold",
                color="black",
                zorder=15,
            )

    if title:
        plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    return plt.gca()

def plot_swarm(
    swarm,
    *,
    title: str = "",
    layout: str = "circular",              # "field" | "circular" | "given"
    positions: Optional[Tensor] = None,    # only used when layout="given"
    with_nodes: bool = True,
    node_size: float = 60.0,
    edge_alpha: float = 0.4,
    adversary_marker: str = "s",
    normal_marker: str = "o",
):
    ...
    A = (swarm.AdjMatrix > 0).to(torch.float32)
    M = A.shape[0]

    # ---- derive positions ----------------------------------------------------
    if layout == "field":
        if not hasattr(swarm, "agents") or len(swarm.agents) != M:
            raise RuntimeError("plot_swarm(layout='field'): swarm.agents not built.")

        # Get datasets; treat non-sequence or wrong type as "missing"
        datasets = getattr(swarm, "datasets", None)
        import numpy as np  # if not already imported at top

        if not isinstance(datasets, (list, tuple, torch.Tensor, np.ndarray)):
            raise RuntimeError("plot_swarm(layout='field'): swarm.datasets missing.")

        if isinstance(datasets, (list, tuple)):
            if len(datasets) != M:
                raise RuntimeError("plot_swarm(layout='field'): swarm.datasets missing.")
            # Each entry: [N_i, D], use first two dims as XY
            pos = torch.stack(
                [torch.as_tensor(d)[:, :2].mean(dim=0) for d in datasets],
                dim=0,
            ).to(torch.float32)
        else:
            # Assume a single array/tensor of shape [M, N, D]
            data_t = torch.as_tensor(datasets)
            if data_t.ndim < 2 or data_t.shape[0] != M:
                raise RuntimeError("plot_swarm(layout='field'): swarm.datasets missing.")
            pos = data_t[..., :2].mean(dim=1).to(torch.float32)

        positions_t = pos

    elif layout == "circular":
        theta = torch.linspace(0, 2 * math.pi, steps=M + 1)[:-1]
        positions_t = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    elif layout == "given":
        if positions is None:
            raise ValueError("plot_swarm(layout='given') requires `positions`.")
        if positions.shape != (M, 2):
            raise ValueError(f"`positions` must be shape ({M}, 2), got {tuple(positions.shape)}")
        positions_t = positions.to(torch.float32)

    else:
        raise ValueError("layout must be one of {'field','circular','given'}")

    pos = positions_t.cpu().numpy()

    # ---- draw edges ----------------------------------------------------------
    i, j = torch.nonzero(A, as_tuple=True)
    xs = list(zip(pos[i, 0], pos[j, 0]))
    ys = list(zip(pos[i, 1], pos[j, 1]))

    plt.figure(figsize=(6, 6))
    for (x1, x2), (y1, y2) in zip(xs, ys):
        plt.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=edge_alpha)

    # ---- draw nodes by agent type --------------------------------------------
    normal_idx: list[int] = []
    adv_idx: list[int] = []
    # AgentType is expected on agent.agent_type

    #for k, agent in enumerate(swarm.agents):
    #    if getattr(agent, "agent_type", AgentType.NORMAL) == AgentType.ADVERSARIAL:
    #        adv_idx.append(k)
    #    else:
    #        normal_idx.append(k)

    if with_nodes:
        if normal_idx:
            plt.scatter(pos[normal_idx, 0], pos[normal_idx, 1], color="blue",
                        s=node_size, marker=normal_marker, zorder=10, label="Normal")
        if adv_idx:
            plt.scatter(pos[adv_idx, 0], pos[adv_idx, 1], color="red",
                        s=node_size*1.15, marker=adversary_marker, zorder=11, label="Adversarial")

    if title:
        plt.title(title)
    if with_nodes and (normal_idx or adv_idx):
        plt.legend(loc="upper right", frameon=False)

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    return plt.gca()
