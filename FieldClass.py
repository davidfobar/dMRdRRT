from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import numpy as np
from noise import pnoise2


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

    @abstractmethod
    def edge_is_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Return True when segment p1->p2 is valid under field constraints."""

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
        fig, ax = plt.subplots(figsize=(8, 8))
        xmin, xmax, ymin, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")

        for ox, oy, r in self.obstacles:
            ax.add_patch(Circle((ox, oy), r, facecolor="tab:gray", edgecolor="black", alpha=0.35))

        for node in nodes:
            if node.parent is None:
                continue
            parent = nodes[node.parent]
            ax.plot([parent.x, node.x], [parent.y, node.y], color="0.75", linewidth=0.7)

        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="tab:red", linewidth=2.4, label=f"{planner_name} path")

        ax.scatter(start[0], start[1], c="tab:green", s=80, label="start", zorder=5)
        ax.scatter(goal[0], goal[1], c="tab:blue", s=80, label="goal", zorder=5)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Toy 2D {planner_name} Path Planning")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


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
        grade_threshold: float = 15.0,
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
        self.grade_threshold = grade_threshold
        self.vehicle_footprint_m = vehicle_footprint_m
        self.max_elevation = max_elevation
        self.effective_water_level: float | None = None
        self.terrain: np.ndarray | None = None
        self.elevation: np.ndarray | None = None
        self.grade: np.ndarray | None = None
        self.water_mask: np.ndarray | None = None
        self.steep_grade_mask: np.ndarray | None = None
        self.obstacle_mask: np.ndarray | None = None

    def generate_field(self) -> np.ndarray:
        base = self.seed % 256
        terrain = np.zeros((self.height, self.width), dtype=float)

        for y in range(self.height):
            for x in range(self.width):
                terrain[y, x] = pnoise2(
                    x / self.width * self.scale,
                    y / self.height * self.scale,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    base=base,
                )

        # Use fixed remapping from approximate Perlin range [-1, 1] to [0, 1]
        # so each generated map keeps consistent relief scaling.
        terrain = np.clip(0.5 * (terrain + 1.0), 0.0, 1.0)
        self.terrain = terrain
        self.generate_obstacle_layers()
        return terrain

    def remap_elevation(self, terrain: np.ndarray, water_level: float | None = None) -> np.ndarray:
        level = self.water_threshold if water_level is None else water_level
        denom = max(1.0 - level, 1e-9)
        remapped = (terrain - level) / denom
        remapped = np.clip(remapped, 0.0, 1.0)
        return remapped * self.max_elevation

    def remap_elevation_continuous(self, terrain: np.ndarray, water_level: float | None = None) -> np.ndarray:
        # Continuous remap used for derivatives. Avoid clipping here so shoreline
        # transitions do not create artificial vertical cliffs in grade.
        level = self.water_threshold if water_level is None else water_level
        denom = max(1.0 - level, 1e-9)
        remapped = (terrain - level) / denom
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

    def generate_obstacle_layers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        terrain = self.terrain if self.terrain is not None else self.generate_field()

        # If terrain values never fall below the nominal threshold, treat
        # `water_threshold` as a quantile fallback to preserve water regions.
        water_level = self.water_threshold
        if not np.any(terrain <= water_level):
            q = float(np.clip(self.water_threshold, 0.0, 1.0))
            water_level = float(np.quantile(terrain, q))
        self.effective_water_level = water_level

        self.elevation = self.remap_elevation(terrain, water_level=water_level)
        continuous_elevation = self.remap_elevation_continuous(terrain, water_level=water_level)
        smoothed_elevation = self.smooth_elevation_for_grade(continuous_elevation)
        grade = self.compute_grade(smoothed_elevation)
        water_mask = terrain <= water_level
        # Keep obstacle layers disjoint so each cell is either water or steep grade.
        steep_grade_mask = (grade >= self.grade_threshold) & (~water_mask)
        obstacle_mask = water_mask | steep_grade_mask

        self.grade = grade
        self.water_mask = water_mask
        self.steep_grade_mask = steep_grade_mask
        self.obstacle_mask = obstacle_mask
        return water_mask, steep_grade_mask, obstacle_mask

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

    @staticmethod
    def make_obstacle_colormap() -> mcolors.Colormap:
        return plt.get_cmap("Greys")

    def edge_is_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        raise NotImplementedError("Collision behavior is not implemented for TerrainFieldClass yet")

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
        terrain = self.terrain if self.terrain is not None else self.generate_field()
        elevation = self.elevation if self.elevation is not None else self.remap_elevation(terrain)
        water_mask, steep_grade_mask, _ = (
            self.generate_obstacle_layers()
            if self.water_mask is None or self.steep_grade_mask is None
            else (self.water_mask, self.steep_grade_mask, self.obstacle_mask)
        )
        cmap = self.make_terrain_colormap()
        obstacle_cmap = self.make_obstacle_colormap()
        ls = mcolors.LightSource(azdeg=315, altdeg=35)
        hillshade = ls.shade(terrain, cmap=cmap, vert_exag=3.0, blend_mode="soft")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(hillshade, origin="lower", interpolation="bilinear")

        elevation_norm = mcolors.Normalize(vmin=0.0, vmax=self.max_elevation)
        elevation_mappable = plt.cm.ScalarMappable(norm=elevation_norm, cmap=cmap)
        elevation_mappable.set_array(elevation)
        colorbar = fig.colorbar(elevation_mappable, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        colorbar.set_label("Elevation")

        # Overlay obstacle layers using a shared grayscale palette.
        water_overlay = np.ma.masked_where(~water_mask, water_mask)
        steep_overlay = np.ma.masked_where(~steep_grade_mask, steep_grade_mask)
        ax.imshow(water_overlay, origin="lower", cmap=obstacle_cmap, alpha=0.50, interpolation="nearest")
        ax.imshow(steep_overlay, origin="lower", cmap=obstacle_cmap, alpha=0.50, interpolation="nearest")

        # Add diagonal hatching to make obstacle regions readable in grayscale.
        water_levels = [0.5, 1.5]
        steep_levels = [0.5, 1.5]
        ax.contourf(
            water_mask.astype(float),
            levels=water_levels,
            colors="none",
            hatches=["///"],
            origin="lower",
        )
        ax.contourf(
            steep_grade_mask.astype(float),
            levels=steep_levels,
            colors="none",
            hatches=["\\\\"],
            origin="lower",
        )

        if nodes:
            for node in nodes:
                if getattr(node, "parent", None) is None:
                    continue
                parent = nodes[node.parent]
                ax.plot([parent.x, node.x], [parent.y, node.y], color="white", linewidth=0.6, alpha=0.5)

        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="#e63946", linewidth=2.2, label=f"{planner_name} path")

        ax.scatter(start[0], start[1], c="tab:green", s=80, label="start", zorder=5)
        ax.scatter(goal[0], goal[1], c="tab:blue", s=80, label="goal", zorder=5)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Perlin Terrain {planner_name} Visualization")
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.15, color="white")

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
