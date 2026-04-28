from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import numpy as np
from noise import pnoise2
#Robbie Push ->Added Some libraries
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


class BaseFieldClass(ABC):
    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        robot_radius: float = 0.8,
    ) -> None:
        self.bounds = bounds
        self.robot_radius = robot_radius

    @staticmethod
    def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(p - q))

    def point_in_bounds(self, point: np.ndarray) -> bool:
        xmin, xmax, ymin, ymax = self.bounds
        return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax

    def default_agent_plot_title(self, planner_name: str) -> str:
        return f"{planner_name} Visualization"

    def planner_tree_style(self) -> dict[str, float | str]:
        return {"color": "0.75", "alpha": 1.0, "linewidth": 0.7}

    def planner_path_style(self) -> dict[str, float | str]:
        return {"color": "tab:red", "linewidth": 2.2}

    def overlay_obstacle_regions(self, ax: object, *, max_grade: float | None = None) -> None:
        """Draw field-defined obstacle overlays for a specific agent capability."""

    def segment_exceeds_capability(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        *,
        max_grade: float | None = None,
    ) -> bool:
        """Return True when this segment violates an agent-specific constraint."""
        return False

    @abstractmethod
    def edge_is_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Return True when segment p1->p2 is valid under field constraints."""

    @abstractmethod
    def plot(
        self,
        output_path: Optional[Path] = None,
        show: bool = True,
        title: Optional[str] = None,
        finalize: bool = True,
    ) -> tuple[object, object]:
        """Render the base field only, without agent-specific overlays."""

    @abstractmethod
    def plot_result(
        self,
        nodes: list[object],
        path: list[np.ndarray] | None,
        start: tuple[float, float],
        goal: tuple[float, float],
        output_path: Optional[Path] = None,
        show: bool = True,
        planner_name: str = "RRT",
        title: Optional[str] = None,
    ) -> None:
        """Render a planner result for this field implementation."""

    @staticmethod
    def finalize_plot(fig: object, output_path: Optional[Path], show: bool) -> None:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def overlay_planner_state(
        self,
        ax: object,
        *,
        nodes: list[object],
        roadmap_edges: list[tuple[int, int]] | None = None,
        path: list[np.ndarray] | None,
        start: tuple[float, float],
        goal: tuple[float, float],
        planner_name: str,
    ) -> None:
        tree_style = self.planner_tree_style()
        path_style = self.planner_path_style()

        if roadmap_edges:
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
        else:
            for node in nodes:
                if getattr(node, "parent", None) is None:
                    continue
                parent = nodes[node.parent]
                ax.plot(
                    [parent.x, node.x],
                    [parent.y, node.y],
                    color=tree_style["color"],
                    linewidth=tree_style["linewidth"],
                    alpha=tree_style["alpha"],
                )

        if path:
            xs = [point[0] for point in path]
            ys = [point[1] for point in path]
            ax.plot(
                xs,
                ys,
                color=path_style["color"],
                linewidth=path_style["linewidth"],
                label=f"{planner_name} path",
            )

        ax.scatter(start[0], start[1], c="tab:green", s=80, label="start", zorder=5)
        ax.scatter(goal[0], goal[1], c="tab:blue", s=80, label="goal", zorder=5)

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left")

    def render_planner_result(
        self,
        *,
        nodes: list[object],
        roadmap_edges: list[tuple[int, int]] | None = None,
        path: list[np.ndarray] | None,
        start: tuple[float, float],
        goal: tuple[float, float],
        output_path: Optional[Path] = None,
        show: bool = True,
        planner_name: str = "RRT",
        title: Optional[str] = None,
        max_grade: float | None = None,
    ) -> None:
        plot_title = title if title is not None else self.default_agent_plot_title(planner_name)
        fig, ax = self.plot(show=False, title=plot_title, finalize=False)
        self.overlay_obstacle_regions(ax, max_grade=max_grade)
        self.overlay_planner_state(
            ax,
            nodes=nodes,
            roadmap_edges=roadmap_edges,
            path=path,
            start=start,
            goal=goal,
            planner_name=planner_name,
        )
        self.finalize_plot(fig, output_path, show)


class ToyFieldClass(BaseFieldClass):
    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        obstacles: list[tuple[float, float, float]],
        robot_radius: float = 0.8,
    ) -> None:
        super().__init__(bounds=bounds, robot_radius=robot_radius)
        self.obstacles = obstacles

    def segment_circle_collision(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        center: np.ndarray,
        radius: float,
        margin: float = 0.0,
    ) -> bool:
        # Check whether the line segment p1->p2 gets within (radius + margin)
        # of the circle center by computing the closest point on the segment.
        seg = p2 - p1
        seg_len_sq = float(np.dot(seg, seg))

        # Degenerate segment: treat as a point-vs-circle distance test.
        if seg_len_sq < 1e-12:
            return self.euclidean_distance(p1, center) <= radius + margin

        # Project center onto the infinite line and clamp to [0, 1] so the
        # resulting closest point lies on the finite segment.
        t = float(np.dot(center - p1, seg) / seg_len_sq)
        t = max(0.0, min(1.0, t))
        closest = p1 + t * seg

        # Collision exists when the closest point is inside the inflated radius.
        return self.euclidean_distance(closest, center) <= radius + margin

    def edge_is_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        if not self.point_in_bounds(p2):
            return False

        for ox, oy, r in self.obstacles:
            center = np.array([ox, oy], dtype=float)
            if self.segment_circle_collision(p1, p2, center, r, margin=self.robot_radius):
                return False

        return True

    def default_agent_plot_title(self, planner_name: str) -> str:
        return f"Toy 2D {planner_name} Path Planning"

    def overlay_obstacle_regions(self, ax: object, *, max_grade: float | None = None) -> None:
        for ox, oy, r in self.obstacles:
            ax.add_patch(Circle((ox, oy), r, facecolor="none", edgecolor="black", hatch="///", linewidth=0.0))

    def plot(
        self,
        output_path: Optional[Path] = None,
        show: bool = True,
        title: Optional[str] = None,
        finalize: bool = True,
    ) -> tuple[object, object]:
        fig, ax = plt.subplots(figsize=(8, 8))
        xmin, xmax, ymin, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")

        for ox, oy, r in self.obstacles:
            ax.add_patch(Circle((ox, oy), r, facecolor="tab:gray", edgecolor="black", alpha=0.35))

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.2)
        ax.set_title(title if title else "Toy 2D Field")

        if finalize:
            self.finalize_plot(fig, output_path, show)
        return fig, ax

    def plot_result(
        self,
        nodes: list[object],
        path: list[np.ndarray] | None,
        start: tuple[float, float],
        goal: tuple[float, float],
        output_path: Optional[Path] = None,
        show: bool = True,
        planner_name: str = "RRT",
        title: Optional[str] = None,
    ) -> None:
        self.render_planner_result(
            nodes=nodes,
            roadmap_edges=None,
            path=path,
            start=start,
            goal=goal,
            output_path=output_path,
            show=show,
            planner_name=planner_name,
            title=title,
        )


class TerrainFieldClass(BaseFieldClass):
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        scale: float = 1.0,
        octaves: int = 8,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        seed: int = 42,
        water_threshold: float = 0.30,
        vehicle_footprint_m: float = 8.0,
        max_elevation: float = 100.0,
        robot_radius: float = 0.8,
    ) -> None:
        bounds = (0.0, float(width), 0.0, float(height))
        super().__init__(bounds=bounds, robot_radius=robot_radius)
        self.width = width
        self.height = height
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed
        self.water_threshold = water_threshold
        self.vehicle_footprint_m = vehicle_footprint_m
        self.max_elevation = max_elevation
        self.effective_water_level: float | None = None
        self.terrain: np.ndarray | None = None
        self.elevation: np.ndarray | None = None
        self.grade: np.ndarray | None = None
        self.water_mask: np.ndarray | None = None

        self.generate_field()

    def generate_field(self) -> np.ndarray:
        base = self.seed % 256
        terrain = np.array([
            pnoise2(
                x / self.width * self.scale,
                y / self.height * self.scale,
                octaves=self.octaves,
                persistence=self.persistence,
                lacunarity=self.lacunarity,
                base=base,
            )
            for y in range(self.height)
            for x in range(self.width)
        ]).reshape(self.height, self.width)

        # Use fixed remapping from approximate Perlin range [-1, 1] to [0, 1]
        # so each generated map keeps consistent relief scaling.
        terrain = np.clip(0.5 * (terrain + 1.0), 0.0, 1.0)
        self.terrain = terrain
        self.generate_obstacle_layers()
        return terrain

    def remap_elevation(self, terrain: np.ndarray, water_level: float | None = None, *, clip: bool = True) -> np.ndarray:
        # When clip=False the result is unclipped, which is used for grade
        # computation so the shoreline doesn't create artificial steep cliffs.
        level = self.water_threshold if water_level is None else water_level
        remapped = (terrain - level) / max(1.0 - level, 1e-9)
        if clip:
            remapped = np.clip(remapped, 0.0, 1.0)
        return remapped * self.max_elevation

    def compute_grade(self, elevation: np.ndarray) -> np.ndarray:
        x_spacing = self.width / max(self.width - 1, 1)
        y_spacing = self.height / max(self.height - 1, 1)
        gy, gx = np.gradient(elevation, y_spacing, x_spacing)
        return 100.0 * np.sqrt(gx**2 + gy**2)

    @staticmethod
    def box_filter(data: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return data

        k = 2 * radius + 1
        padded = np.pad(data, ((radius, radius), (radius, radius)), mode="edge")
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        return (
            integral[k:, k:]
            - integral[:-k, k:]
            - integral[k:, :-k]
            + integral[:-k, :-k]
        ) / float(k * k)

    def smooth_elevation_for_grade(self, elevation: np.ndarray) -> np.ndarray:
        x_spacing = self.width / max(self.width - 1, 1)
        y_spacing = self.height / max(self.height - 1, 1)
        min_spacing = min(x_spacing, y_spacing)
        footprint_cells = max(1, int(round(self.vehicle_footprint_m / max(min_spacing, 1e-9))))
        radius = max(0, footprint_cells // 2)
        return self.box_filter(elevation, radius)

    def generate_obstacle_layers(self) -> np.ndarray:
        terrain = self.terrain if self.terrain is not None else self.generate_field()

        # If terrain values never fall below the nominal threshold, treat
        # `water_threshold` as a quantile fallback to preserve water regions.
        water_level = self.water_threshold
        if not np.any(terrain <= water_level):
            q = float(np.clip(self.water_threshold, 0.0, 1.0))
            water_level = float(np.quantile(terrain, q))
        self.effective_water_level = water_level

        self.elevation = self.remap_elevation(terrain, water_level=water_level)
        continuous_elevation = self.remap_elevation(terrain, water_level=water_level, clip=False)
        smoothed_elevation = self.smooth_elevation_for_grade(continuous_elevation)
        self.grade = self.compute_grade(smoothed_elevation)
        self.water_mask = terrain <= water_level
        return self.water_mask

    @staticmethod
    def make_terrain_colormap() -> mcolors.LinearSegmentedColormap:
        colors = [
            (0.05, 0.15, 0.40),   # deep water
            (0.10, 0.35, 0.65),   # shallow water
            (0.76, 0.72, 0.50),   # sand
            (0.35, 0.60, 0.25),   # lowland grass
            (0.25, 0.45, 0.18),   # highland grass
            (0.50, 0.45, 0.38),   # rock
            (0.90, 0.90, 0.90),   # snow
        ]
        positions = [0.0, 0.25, 0.30, 0.40, 0.60, 0.75, 1.0]
        return mcolors.LinearSegmentedColormap.from_list(
            "terrain_custom",
            list(zip(positions, colors)),
        )

    def edge_is_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check only terrain-absolute obstacles (water). Grade limits are
        a vehicle capability and are enforced by the Agent."""
        if self.water_mask is None:
            self.generate_obstacle_layers()

        if not self.point_in_bounds(p2):
            return False

        # Brute-force: sample at ~1-pixel intervals along the segment and
        # reject if any sample lands in water.
        dist = float(np.linalg.norm(p2 - p1))
        n_samples = max(2, int(np.ceil(dist)) + 1)

        for t in np.linspace(0.0, 1.0, n_samples):
            pt = p1 + t * (p2 - p1)
            col = int(np.clip(round(pt[0]), 0, self.width - 1))
            row = int(np.clip(round(pt[1]), 0, self.height - 1))
            if self.water_mask[row, col]:  # type: ignore[index]
                return False

        return True

    def default_agent_plot_title(self, planner_name: str) -> str:
        return f"Perlin Terrain {planner_name} Visualization"

    def planner_tree_style(self) -> dict[str, float | str]:
        return {"color": "white", "alpha": 0.5, "linewidth": 0.7}

    def planner_path_style(self) -> dict[str, float | str]:
        return {"color": "#e63946", "linewidth": 2.2}

    def overlay_obstacle_regions(self, ax: object, *, max_grade: float | None = None) -> None:
        water_mask = self.water_mask if self.water_mask is not None else self.generate_obstacle_layers()
        obstacle_cmap = plt.get_cmap("Greys")

        water_overlay = np.ma.masked_where(~water_mask, water_mask)
        ax.imshow(water_overlay, origin="lower", cmap=obstacle_cmap, alpha=0.50, interpolation="nearest")
        ax.contourf(water_mask.astype(float), levels=[0.5, 1.5], colors="none", hatches=["///"], origin="lower")

        if max_grade is not None and self.grade is not None:
            steep_grade_mask = (self.grade > max_grade) & (~water_mask)
            steep_overlay = np.ma.masked_where(~steep_grade_mask, steep_grade_mask)
            ax.imshow(steep_overlay, origin="lower", cmap=obstacle_cmap, alpha=0.50, interpolation="nearest")
            ax.contourf(
                steep_grade_mask.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["\\\\"],
                origin="lower",
            )

    def segment_exceeds_capability(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        *,
        max_grade: float | None = None,
    ) -> bool:
        if max_grade is None or self.grade is None:
            return False

        dist = float(np.linalg.norm(p2 - p1))
        n_samples = max(2, int(np.ceil(dist)) + 1)

        for t in np.linspace(0.0, 1.0, n_samples):
            pt = p1 + t * (p2 - p1)
            col = int(np.clip(round(pt[0]), 0, self.width - 1))
            row = int(np.clip(round(pt[1]), 0, self.height - 1))
            if self.grade[row, col] > max_grade:
                return True

        return False

    def plot(
        self,
        output_path: Optional[Path] = None,
        show: bool = True,
        title: Optional[str] = None,
        finalize: bool = True,
    ) -> tuple[object, object]:
        terrain = self.terrain if self.terrain is not None else self.generate_field()
        elevation = self.elevation if self.elevation is not None else self.remap_elevation(terrain, clip=True)

        cmap = self.make_terrain_colormap()
        ls = mcolors.LightSource(azdeg=315, altdeg=35)
        hillshade = ls.shade(terrain, cmap=cmap, vert_exag=3.0, blend_mode="soft")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(hillshade, origin="lower", interpolation="bilinear")

        # Colorbar shows above-water elevation: 0 maps to effective shoreline.
        water_level = self.effective_water_level if self.effective_water_level is not None else self.water_threshold
        water_level = float(np.clip(water_level, 0.0, 1.0 - 1e-9))
        land_colors = cmap(np.linspace(water_level, 1.0, 256))
        land_cmap = mcolors.LinearSegmentedColormap.from_list("terrain_land", land_colors)

        elevation_norm = mcolors.Normalize(vmin=0.0, vmax=self.max_elevation)
        elevation_mappable = plt.cm.ScalarMappable(norm=elevation_norm, cmap=land_cmap)
        elevation_mappable.set_array(elevation)
        colorbar = fig.colorbar(elevation_mappable, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        colorbar.set_label("Elevation")

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15, color="white")
        ax.set_title(title if title else "Perlin Terrain")

        if finalize:
            self.finalize_plot(fig, output_path, show)
        return fig, ax

    def plot_result(
        self,
        nodes: list[object],
        path: list[np.ndarray] | None,
        start: tuple[float, float],
        goal: tuple[float, float],
        output_path: Optional[Path] = None,
        show: bool = True,
        planner_name: str = "RRT",
        title: Optional[str] = None,
        grade_limit: float | None = None,
    ) -> None:
        self.render_planner_result(
            nodes=nodes,
            roadmap_edges=None,
            path=path,
            start=start,
            goal=goal,
            output_path=output_path,
            show=show,
            planner_name=planner_name,
            title=title,
            max_grade=grade_limit,
        )

#Robbie Push -> Render multiagent paths
    def render_multi_agent_result(
        self,
        *,
        prm,
        robot_paths,
        start_states,
        goal_states,
        output_path=None,
        show=True,
        title="MRdRRT Multi-Agent",
    ):
        fig, ax = self.plot(show=False, title=title, finalize=False)

        # Draw PRM roadmap (optional but matches PRM visuals)
        for i, j in prm.roadmap_edges:
            p = prm.nodes[i]
            q = prm.nodes[j]
            ax.plot(
                [p.x, q.x],
                [p.y, q.y],
                color="0.75",
                linewidth=0.7,
                alpha=0.6,
            )

        colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown", "tab:orange", "tab:pink"]
        
        for i, path in enumerate(robot_paths):
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]

            ax.plot(
                xs,
                ys,
                color=colors[i % len(colors)],
                linewidth=2.2,
                label=f"Robot {i+1}",
            )

            ax.scatter(xs[0], ys[0], c="green", s=60)
            ax.scatter(xs[-1], ys[-1], c="blue", s=60)
            
        for i, coords in enumerate(goal_states):
            # Draw goal point star
            ax.scatter(coords[0], coords[1], c=colors[i% len(colors)], marker='*', s=250, zorder=5, edgecolors='black')
        
        ax.legend()
        self.finalize_plot(fig, output_path, show)


    def animate_multi_agent_result(
        self,
        *,
        prm,
        robot_paths,
        start_states,
        goal_states,
        output_path=None,    # .gif or .mp4
        show=True,
        title="MRdRRT Multi-Agent",
        interval=300,        # ms per frame
        trail_length=15,      # how many past positions to show as faded trail
        comm_range,
    ):
        fig, ax = self.plot(show=False, title=title, finalize=False)

        # Draw static PRM roadmap
        for i, j in prm.roadmap_edges:
            p, q = prm.nodes[i], prm.nodes[j]
            ax.plot([p.x, q.x], [p.y, q.y], color="0.75", linewidth=0.7, alpha=0.6)

        colors = ["tab:blue", "tab:red", "tab:green", "tab:purple",
                "tab:brown", "tab:orange", "tab:pink"]

        # Draw faded full paths as a guide
        for i, path in enumerate(robot_paths):
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=1.0,
                    alpha=0.2, linestyle="--")

        # Draw start markers
        for i, path in enumerate(robot_paths):
            ax.scatter(path[0][0], path[0][1], c="green", s=80, zorder=5,
                    edgecolors="black")

        # Draw goal stars
        for i, coords in enumerate(goal_states):
            ax.scatter(coords[0], coords[1], c=colors[i % len(colors)],
                    marker="*", s=250, zorder=5, edgecolors="black")

        n_agents = len(robot_paths)
        max_steps = max(len(p) for p in robot_paths)

        # Artists we update each frame: trail lines + robot dot + step counter
        trails   = [ax.plot([], [], color=colors[i % len(colors)], linewidth=2.2,
                            alpha=0.8, label=f"Robot {i+1}")[0]
                    for i in range(n_agents)]
        dots     = [ax.plot([], [], "o", color=colors[i % len(colors)],
                            markersize=10, zorder=6)[0]
                    for i in range(n_agents)]
        counter  = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        fontsize=10, va="top",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        # After your existing dots/trails setup, add:
        comm_circles = [
            plt.Circle(
                (path[0][0], path[0][1]),  # initial position
                comm_range,
                color=colors[i % len(colors)],
                fill=False,
                lw=2,
                alpha=1,
                linestyle="--",
                zorder=5,
            )
            for i, path in enumerate(robot_paths)
        ]
        for c in comm_circles:
            ax.add_patch(c)
        ax.legend(loc="upper right")
        
        def init():
            for trail, dot in zip(trails, dots):
                trail.set_data([], [])
                dot.set_data([], [])
            counter.set_text("")
            return trails + dots + [counter]

        def update(frame):
            for i, (path, trail, dot) in enumerate(zip(robot_paths, trails, dots)):
                # Clamp frame to last position if this robot's path is shorter
                step = min(frame, len(path) - 1)

                # Trail: last `trail_length` positions
                trail_start = max(0, step - trail_length)
                xs = [p[0] for p in path[trail_start : step + 1]]
                ys = [p[1] for p in path[trail_start : step + 1]]
                trail.set_data(xs, ys)

                # Robot dot at current position
                dot.set_data([path[step][0]], [path[step][1]])
                # Move each circle to follow its robot
                for i, (path, circle) in enumerate(zip(robot_paths, comm_circles)):
                    step = min(frame, len(path) - 1)
                    circle.center = (path[step][0], path[step][1])

            counter.set_text(f"Step {frame + 1} / {max_steps}")
            return trails + dots + [counter]

        anim = FuncAnimation(
            fig,
            update,
            frames=max_steps,
            init_func=init,
            interval=interval,
            blit=True,
        )

        if output_path:
            if output_path.endswith(".gif"):
                anim.save(output_path, writer=PillowWriter(fps=1000 // interval))
            elif output_path.endswith(".mp4"):
                anim.save(output_path, writer="ffmpeg", fps=1000 // interval)
            print(f"Saved animation to {output_path}")

        if show:
            plt.show()

        return anim  # keep reference alive to prevent GC freeze
