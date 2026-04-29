from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Protocol

import numpy as np

from planners.RRT import PlannerSpace


@dataclass(slots=True)
class PRMNode:
    x: float
    y: float


@dataclass(slots=True)
class PRMParameters:
    n_samples: int = 450
    k_neighbors: int = 12
    connection_radius: float | None = None
    max_build_attempts: int = 3
    seed: int = 7


@dataclass(slots=True)
class PRMQueryParameters:
    k_neighbors: int = 10
    connection_radius: float | None = None


class PRMPlannerLike(Protocol):
    nodes: list[PRMNode]
    roadmap_edges: list[tuple[int, int]]

    def path_length(self, path: list[np.ndarray]) -> float:
        ...


class PRMQueryResult:
    """Agent-facing PRM planner-like view.

    Keeps the persistent roadmap geometry available for plotting while storing
    the path found for one start-goal query.
    """

    def __init__(
        self,
        *,
        nodes: list[PRMNode],
        roadmap_edges: list[tuple[int, int]],
        path: list[np.ndarray] | None,
    ) -> None:
        self.nodes = nodes
        self.roadmap_edges = roadmap_edges
        self.path = path

    @staticmethod
    def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(p - q))

    def path_length(self, path: list[np.ndarray]) -> float:
        return sum(self.euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))


class PRMRoadmap:
    """Persistent PRM graph that can be queried repeatedly by many agents."""

    def __init__(
        self,
        space: PlannerSpace,
        params: PRMParameters,
    ) -> None:
        self.space = space
        self.params = params
        self.rng = np.random.default_rng(self.params.seed)

        self.nodes: list[PRMNode] = []
        self.roadmap_edges: list[tuple[int, int]] = []
        self.graph: dict[int, list[tuple[int, float]]] = {}

    @staticmethod
    def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(p - q))

    def path_length(self, path: list[np.ndarray]) -> float:
        return sum(self.euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def sample_point(self) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.space.bounds
        return np.array(
            [self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax)],
            dtype=float,
        )

    def point_is_valid(self, point: np.ndarray) -> bool:
        # Reuse the edge collision checker for a point test via a degenerate segment.
        return self.space.edge_is_collision_free(point, point)

    def sample_valid_point(self, max_tries: int = 300) -> np.ndarray | None:
        for _ in range(max_tries):
            candidate = self.sample_point()
            if self.point_is_valid(candidate):
                return candidate
        return None

    def add_node(self, point: np.ndarray) -> int:
        self.nodes.append(PRMNode(float(point[0]), float(point[1])))
        node_index = len(self.nodes) - 1
        self.graph[node_index] = []
        return node_index

    def node_point(self, idx: int) -> np.ndarray:
        node = self.nodes[idx]
        return np.array([node.x, node.y], dtype=float)

    def _build_once(self) -> None:
        self.nodes = []
        self.roadmap_edges = []
        self.graph = {}

        for _ in range(self.params.n_samples):
            sample = self.sample_valid_point()
            if sample is not None:
                self.add_node(sample)

        radius = self.params.connection_radius if self.params.connection_radius is not None else math.inf

        for i in range(len(self.nodes)):
            p = self.node_point(i)
            candidates: list[tuple[float, int]] = []

            for j in range(len(self.nodes)):
                if i == j:
                    continue
                q = self.node_point(j)
                dist = self.euclidean_distance(p, q)
                if dist <= radius:
                    candidates.append((dist, j))

            candidates.sort(key=lambda item: item[0])

            for dist, j in candidates[: self.params.k_neighbors]:
                if j <= i:
                    continue
                q = self.node_point(j)
                if not self.space.edge_is_collision_free(p, q):
                    continue

                self.graph[i].append((j, dist))
                self.graph[j].append((i, dist))
                self.roadmap_edges.append((i, j))

    def build(self) -> None:
        attempts = max(1, self.params.max_build_attempts)
        for _ in range(attempts):
            self._build_once()
            if self.nodes and self.roadmap_edges:
                return
        # Keep the last generated graph even if sparse or disconnected.

    def _connect_query_node(
        self,
        *,
        point: np.ndarray,
        node_index: int,
        adjacency: dict[int, list[tuple[int, float]]],
        working_nodes: list[np.ndarray],
        space: PlannerSpace,
        params: PRMQueryParameters,
        query_edges: list[tuple[int, int]],
    ) -> None:
        radius = params.connection_radius if params.connection_radius is not None else math.inf

        candidates: list[tuple[float, int]] = []
        for i in range(len(self.nodes)):
            base_pt = working_nodes[i]
            dist = self.euclidean_distance(point, base_pt)
            if dist <= radius:
                candidates.append((dist, i))

        candidates.sort(key=lambda item: item[0])

        for dist, base_idx in candidates[: params.k_neighbors]:
            base_pt = working_nodes[base_idx]
            if not space.edge_is_collision_free(point, base_pt):
                continue

            adjacency[node_index].append((base_idx, dist))
            adjacency[base_idx].append((node_index, dist))
            edge = (base_idx, node_index) if base_idx < node_index else (node_index, base_idx)
            query_edges.append(edge)

    def _dijkstra(
        self,
        *,
        adjacency: dict[int, list[tuple[int, float]]],
        start_idx: int,
        goal_idx: int,
    ) -> list[int] | None:
        distances = {idx: math.inf for idx in adjacency}
        previous: dict[int, int | None] = {idx: None for idx in adjacency}
        distances[start_idx] = 0.0

        queue: list[tuple[float, int]] = [(0.0, start_idx)]

        while queue:
            current_dist, current = heapq.heappop(queue)
            if current_dist > distances[current]:
                continue
            if current == goal_idx:
                break

            for neighbor, weight in adjacency[current]:
                candidate_dist = current_dist + weight
                if candidate_dist < distances[neighbor]:
                    distances[neighbor] = candidate_dist
                    previous[neighbor] = current
                    heapq.heappush(queue, (candidate_dist, neighbor))

        if math.isinf(distances[goal_idx]):
            return None

        node_path: list[int] = []
        cur: int | None = goal_idx
        while cur is not None:
            node_path.append(cur)
            cur = previous[cur]
        node_path.reverse()
        return node_path

    def query(
        self,
        *,
        start: tuple[float, float],
        goal: tuple[float, float],
        traversal_space: PlannerSpace | None = None,
        query_params: PRMQueryParameters | None = None,
    ) -> PRMQueryResult:
        if not self.nodes:
            self.build()

        space = traversal_space if traversal_space is not None else self.space
        params = query_params if query_params is not None else PRMQueryParameters(
            k_neighbors=self.params.k_neighbors,
            connection_radius=self.params.connection_radius,
        )

        start_point = np.array(start, dtype=float)
        goal_point = np.array(goal, dtype=float)

        if not space.edge_is_collision_free(start_point, start_point):
            return PRMQueryResult(nodes=self.nodes, roadmap_edges=self.roadmap_edges, path=None)
        if not space.edge_is_collision_free(goal_point, goal_point):
            return PRMQueryResult(nodes=self.nodes, roadmap_edges=self.roadmap_edges, path=None)

        base_size = len(self.nodes)
        start_idx = base_size
        goal_idx = base_size + 1

        working_nodes = [self.node_point(i) for i in range(base_size)] + [start_point, goal_point]
        adjacency: dict[int, list[tuple[int, float]]] = {
            idx: list(neighbors)
            for idx, neighbors in self.graph.items()
        }
        adjacency[start_idx] = []
        adjacency[goal_idx] = []

        query_edges: list[tuple[int, int]] = []

        self._connect_query_node(
            point=start_point,
            node_index=start_idx,
            adjacency=adjacency,
            working_nodes=working_nodes,
            space=space,
            params=params,
            query_edges=query_edges,
        )
        self._connect_query_node(
            point=goal_point,
            node_index=goal_idx,
            adjacency=adjacency,
            working_nodes=working_nodes,
            space=space,
            params=params,
            query_edges=query_edges,
        )

        direct_dist = self.euclidean_distance(start_point, goal_point)
        if space.edge_is_collision_free(start_point, goal_point):
            adjacency[start_idx].append((goal_idx, direct_dist))
            adjacency[goal_idx].append((start_idx, direct_dist))
            query_edges.append((start_idx, goal_idx))

        node_path = self._dijkstra(adjacency=adjacency, start_idx=start_idx, goal_idx=goal_idx)
        if node_path is None:
            return PRMQueryResult(nodes=self.nodes, roadmap_edges=self.roadmap_edges, path=None)

        path = [working_nodes[idx] for idx in node_path]
        # Keep persistent roadmap edges plus query connectors for richer plot context.
        result_edges = self.roadmap_edges + query_edges
        result_nodes = self.nodes + [PRMNode(float(start_point[0]), float(start_point[1])), PRMNode(float(goal_point[0]), float(goal_point[1]))]
        return PRMQueryResult(nodes=result_nodes, roadmap_edges=result_edges, path=path)
