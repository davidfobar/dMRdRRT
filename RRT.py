from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from FieldClass import BaseFieldClass, ToyFieldClass


@dataclass
class Node:
    x: float
    y: float
    parent: int | None
    cost: float


class RRTPlanner:
    def __init__(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        field: BaseFieldClass,
        step_size: float = 3.0,
        max_iters: int = 3000,
        goal_bias: float = 0.08,
        goal_tolerance: float = 3.5,
        use_rrt_star: bool = False,
        rrt_star_radius: float | None = None,
        seed: int = 7,
    ) -> None:
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.field = field
        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.use_rrt_star = use_rrt_star
        self.rrt_star_radius = rrt_star_radius if rrt_star_radius is not None else max(6.0, 3.0 * step_size)
        self.rng = np.random.default_rng(seed)
        self.nodes: list[Node] = [Node(self.start[0], self.start[1], None, 0.0)]

    @staticmethod
    def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(p - q))

    def nearest_node_index(self, point: np.ndarray) -> int:
        # compute the distance from the point to each node in the tree and 
        # return the index of the closest one
        distances = [self.euclidean_distance(np.array([n.x, n.y]), point) for n in self.nodes]
        return int(np.argmin(distances))

    def nearby_node_indices(self, point: np.ndarray) -> list[int]:
        indices: list[int] = []
        for i, node in enumerate(self.nodes):
            node_pos = np.array([node.x, node.y], dtype=float)
            if self.euclidean_distance(node_pos, point) <= self.rrt_star_radius:
                indices.append(i)
        return indices

    def steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        direction = to_point - from_point
        length = np.linalg.norm(direction)

        # if the length is very small, just return the from_point. 
        if length < 1e-12:
            return from_point.copy()
        # if the target is within step_size, return the target. 
        if length <= self.step_size:
            return to_point.copy()
        # otherwise, return a point in the direction of the target, but only 
        # step_size away from the from_point.
        return from_point + (direction / length) * self.step_size

    def reconstruct_path(self, goal_idx: int) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        idx: int | None = goal_idx
        while idx is not None:
            node = self.nodes[idx]
            path.append(np.array([node.x, node.y], dtype=float))
            idx = node.parent
        path.reverse()
        return path
    
    def path_length(self, path: list[np.ndarray]) -> float:
        return sum(self.euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def choose_best_parent(self, candidate: np.ndarray, nearest_idx: int, near_indices: list[int]) -> tuple[int, float]:
        best_parent = nearest_idx
        nearest_pos = np.array([self.nodes[nearest_idx].x, self.nodes[nearest_idx].y], dtype=float)
        best_cost = self.nodes[nearest_idx].cost + self.euclidean_distance(nearest_pos, candidate)

        for idx in near_indices:
            node = self.nodes[idx]
            parent_pos = np.array([node.x, node.y], dtype=float)
            if not self.field.edge_is_collision_free(parent_pos, candidate):
                continue

            new_cost = node.cost + self.euclidean_distance(parent_pos, candidate)
            if new_cost < best_cost:
                best_cost = new_cost
                best_parent = idx

        return best_parent, best_cost

    def rewire_neighbors(self, new_idx: int, near_indices: list[int]) -> None:
        new_node = self.nodes[new_idx]
        new_pos = np.array([new_node.x, new_node.y], dtype=float)

        for idx in near_indices:
            if idx == new_idx:
                continue
            neighbor = self.nodes[idx]
            neighbor_pos = np.array([neighbor.x, neighbor.y], dtype=float)
            proposed_cost = new_node.cost + self.euclidean_distance(new_pos, neighbor_pos)

            if proposed_cost >= neighbor.cost:
                continue
            if not self.field.edge_is_collision_free(new_pos, neighbor_pos):
                continue

            self.nodes[idx].parent = new_idx
            self.nodes[idx].cost = proposed_cost

    def sample(self) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.field.bounds

        # check to see if we should bias towards the goal. This helps guide the 
        # search and can speed up convergence.
        if self.rng.random() < self.goal_bias:
            return self.goal
    
        # otherwise, sample uniformly from the bounds
        return np.array(
            [self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax)],
            dtype=float,
        )

    def plan(self) -> list[np.ndarray] | None:
        for _ in range(self.max_iters):
            # get a random sample and find the nearest node in the tree
            sample = self.sample()
            nearest_idx = self.nearest_node_index(sample)
            nearest = np.array([self.nodes[nearest_idx].x, self.nodes[nearest_idx].y], dtype=float)

            # create a new candidate node in the direction of the sample, but only up to 
            # step_size away from the nearest node
            candidate = self.steer(nearest, sample)

            # check if the edge from nearest to candidate is collision-free. If not, skip this iteration.
            if not self.field.edge_is_collision_free(nearest, candidate):
                continue

            parent_idx = nearest_idx
            parent_cost = self.nodes[nearest_idx].cost + self.euclidean_distance(nearest, candidate)
            near_indices: list[int] = []

            # In RRT* mode, choose the parent that minimizes path cost and then rewire.
            if self.use_rrt_star:
                near_indices = self.nearby_node_indices(candidate)
                parent_idx, parent_cost = self.choose_best_parent(candidate, nearest_idx, near_indices)

            # add the candidate node to the tree
            self.nodes.append(Node(candidate[0], candidate[1], parent_idx, parent_cost))
            new_idx = len(self.nodes) - 1

            if self.use_rrt_star:
                self.rewire_neighbors(new_idx, near_indices)

            # check if the candidate is close enough to the goal to attempt a connection
            if self.euclidean_distance(candidate, self.goal) <= self.goal_tolerance:
                if self.field.edge_is_collision_free(candidate, self.goal):
                    goal_cost = self.nodes[new_idx].cost + self.euclidean_distance(candidate, self.goal)
                    self.nodes.append(Node(self.goal[0], self.goal[1], new_idx, goal_cost))
                    return self.reconstruct_path(len(self.nodes) - 1)

        return None

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a toy 2D RRT planner demo.")
    parser.add_argument("--show", action="store_true", help="Display the plot window after running")
    parser.add_argument(
        "--rrt-star",
        action="store_true",
        help="Enable RRT* parent optimization and rewiring",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("images/rrt_toy_solution.png"),
        help="Where to save the generated path figure",
    )
    args = parser.parse_args()

    bounds = (0.0, 100.0, 0.0, 100.0)
    start = (8.0, 8.0)
    goal = (92.0, 90.0)

    # Fixed circular obstacles for a deterministic toy benchmark.
    obstacles = [
        (25.0, 30.0, 9.0),
        (37.0, 55.0, 11.0),
        (52.0, 30.0, 8.5),
        (64.0, 62.0, 10.0),
        (74.0, 28.0, 8.0),
        (84.0, 48.0, 7.5),
    ]

    field = ToyFieldClass(bounds=bounds, obstacles=obstacles, robot_radius=0.6)

    planner = RRTPlanner(
        start=start,
        goal=goal,
        field=field,
        step_size=3.2,
        max_iters=4500,
        goal_bias=0.1,
        goal_tolerance=3.8,
        use_rrt_star=args.rrt_star,
        rrt_star_radius=10.0,
        seed=7,
    )
    path = planner.plan()
    nodes = planner.nodes

    if path is None:
        print("No path found. Try increasing max_iters or adjusting obstacles.")
    else:
        print(f"Path found with {len(path)} waypoints")
        print(f"Path length: {planner.path_length(path):.2f}")

    field.plot_result(
        nodes=nodes,
        path=path,
        start=start,
        goal=goal,
        output_path=args.output,
        show=args.show,
        planner_name="RRT*" if args.rrt_star else "RRT",
    )

    if args.output is not None:
        print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
