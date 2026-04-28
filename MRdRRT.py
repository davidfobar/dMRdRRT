from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

@dataclass
class MRParameters:
    max_iters_per_step: int = 50  # Incremental local iterations per timestep
    goal_bias: float = 0.20
    seed: int = 7
    comm_range: float = 30.0      # Maximum range for communication graph
    r_robustness_degree: int = 1  # 'r' neighbors required to move

@dataclass
class LocalNode:
    prm_id: int
    parent: LocalNode | None
    cost: float

class PRMLike(Protocol):
    """Abstraction for the shared PRM graph."""
    nodes: list          # list of node objects with .x, .y
    graph: dict[int, list[tuple[int, float]]]  # node_id -> list of (neighbor_id, weight)

class ReservationTable:
    """
    Handles collision avoidance by reserving nodes and edges prior to movement.
    Time is tracked in discrete steps.
    """
    def __init__(self):
        # time -> set of node_ids
        self.nodes: dict[int, set[int]] = {}
        # time -> set of sorted tuples (u, v) representing an undirected edge
        self.edges: dict[int, set[tuple[int, int]]] = {}

    def reserve_node(self, time: int, node_id: int):
        if time not in self.nodes:
            self.nodes[time] = set()
        self.nodes[time].add(node_id)

    def reserve_edge(self, time: int, u: int, v: int):
        if time not in self.edges:
            self.edges[time] = set()
        edge = tuple(sorted((u, v)))
        self.edges[time].add(edge)

    def is_node_reserved(self, time: int, node_id: int) -> bool:
        return node_id in self.nodes.get(time, set())

    def is_edge_reserved(self, time: int, u: int, v: int) -> bool:
        edge = tuple(sorted((u, v)))
        return edge in self.edges.get(time, set())

class DecentralizedAgent:
    """
    An individual agent executing dMRdRRT online.
    """
    def __init__(
        self,
        agent_id: int,
        start_id: int,
        goal_id: int,
        prm: PRMLike,
        reservation_table: ReservationTable,
        params: MRParameters
    ):
        self.agent_id = agent_id
        self.current_id = start_id
        self.goal_id = goal_id
        self.prm = prm
        self.reservations = reservation_table
        self.params = params
        self.rng = random.Random(self.params.seed + agent_id)
        
        self.path_history = [start_id]
        self.done = False

        # Build a heuristic lookup (Dijkstra distance to goal)
        self.heuristic = self._build_heuristic()

    def _build_heuristic(self) -> dict[int, float]:
        """Simple backward Dijkstra from the goal to guide the local dRRT."""
        import heapq
        distances = {nid: math.inf for nid in self.prm.graph}
        distances[self.goal_id] = 0.0
        queue = [(0.0, self.goal_id)]

        while queue:
            dist, curr = heapq.heappop(queue)
            if dist > distances[curr]:
                continue
            for neighbor, weight in self.prm.graph[curr]:
                cand = dist + weight
                if cand < distances[neighbor]:
                    distances[neighbor] = cand
                    heapq.heappush(queue, (cand, neighbor))
        return distances

    def check_r_robustness(self, all_positions: dict[int, int]) -> bool:
        """
        Evaluates whether current position maintains r-robustness of the 
        communication graph by counting neighbors within comm_range.
        """
        if self.params.r_robustness_degree <= 0:
            return True

        my_pos = self.prm.nodes[self.current_id]
        visible_neighbors = 0

        for other_id, other_node_id in all_positions.items():
            if other_id == self.agent_id:
                continue
            other_pos = self.prm.nodes[other_node_id]
            dist = math.hypot(my_pos.x - other_pos.x, my_pos.y - other_pos.y)
            
            if dist <= self.params.comm_range:
                visible_neighbors += 1

        return visible_neighbors >= self.params.r_robustness_degree

    def plan_next_step(self, current_time: int, all_positions: dict[int, int]) -> int:
        """Online planning step executed at each timestep."""
        if self.current_id == self.goal_id:
            self.done = True
            self.reservations.reserve_node(current_time + 1, self.current_id)
            return self.current_id

        # 1. r-Robustness Check
        # If not enough neighbors are visible, hold position and rebroadcast.
        if not self.check_r_robustness(all_positions):
            self.reservations.reserve_node(current_time + 1, self.current_id)
            return self.current_id

        # 2. Local dRRT extension
        next_hop = self._incremental_drrt(current_time)

        # 3. Reserve Edge and Node prior to movement
        self.reservations.reserve_node(current_time + 1, next_hop)
        self.reservations.reserve_edge(current_time + 1, self.current_id, next_hop)
        
        return next_hop

    def _incremental_drrt(self, current_time: int) -> int:
        """
        Builds a small local dRRT tree to find the next collision-free hop.
        """
        # The tree is rooted at the current position
        tree = [LocalNode(self.current_id, None, 0.0)]
        target_time = current_time + 1

        best_hop = self.current_id
        best_heuristic = self.heuristic.get(self.current_id, math.inf)

        for _ in range(self.params.max_iters_per_step):
            # Sample goal or random node
            if self.rng.random() < self.params.goal_bias:
                sample_id = self.goal_id
            else:
                sample_id = self.rng.choice(list(self.prm.graph.keys()))

            # Find nearest node in our local tree (based on graph distance)
            nearest_node = min(
                tree, 
                key=lambda n: self.heuristic.get(n.prm_id, math.inf)
            )

            # Steer: Look at neighbors of nearest_node to find best step toward sample
            # (Simplified here to checking direct PRM neighbors)
            neighbors = self.prm.graph[nearest_node.prm_id]
            if not neighbors:
                continue

            # Check reservations to avoid agent-agent collisions
            valid_candidates = []
            for nbr_id, weight in neighbors:
                if not self.reservations.is_node_reserved(target_time, nbr_id) and \
                   not self.reservations.is_edge_reserved(target_time, nearest_node.prm_id, nbr_id):
                    valid_candidates.append((nbr_id, weight))

            if not valid_candidates:
                continue

            # Pick the candidate closest to our sample
            sample_pos = self.prm.nodes[sample_id]
            best_candidate = None
            min_dist = math.inf
            
            for cand_id, weight in valid_candidates:
                cand_pos = self.prm.nodes[cand_id]
                dist = math.hypot(cand_pos.x - sample_pos.x, cand_pos.y - sample_pos.y)
                if dist < min_dist:
                    min_dist = dist
                    best_candidate = cand_id

            if best_candidate is not None:
                new_cost = nearest_node.cost + min_dist
                new_node = LocalNode(best_candidate, nearest_node, new_cost)
                tree.append(new_node)

                # Keep track of the first hop from the root that leads to the best heuristic score
                # If we expanded directly from the root, evaluate it
                if nearest_node.prm_id == self.current_id:
                    cand_heuristic = self.heuristic.get(best_candidate, math.inf)
                    if cand_heuristic < best_heuristic:
                        best_heuristic = cand_heuristic
                        best_hop = best_candidate

        # Return the best immediate hop found by the local tree
        return best_hop

    def execute_step(self, next_id: int):
        self.current_id = next_id
        self.path_history.append(self.current_id)


class OnlineMRSimulation:
    """
    The main driver that coordinates the decentralized agents online.
    """
    def __init__(self, start_states: tuple[int, ...], goal_states: tuple[int, ...], prm: PRMLike, params: MRParameters):
        self.prm = prm
        self.params = params
        self.reservation_table = ReservationTable()
        self.time = 0
        
        # Initialize independent agents
        self.agents = [
            DecentralizedAgent(
                agent_id=i, 
                start_id=start_states[i], 
                goal_id=goal_states[i], 
                prm=self.prm, 
                reservation_table=self.reservation_table, 
                params=self.params
            )
            for i in range(len(start_states))
        ]

    def run(self, max_time_steps=1000):
        while self.time < max_time_steps:
            # Broadcast positions to all agents for r-robustness checks
            all_positions = {agent.agent_id: agent.current_id for agent in self.agents}
            
            if all(agent.done for agent in self.agents):
                break

            # 1. Planning Phase (Agents compute intents and reserve slots)
            next_steps = []
            for agent in self.agents:
                next_id = agent.plan_next_step(self.time, all_positions)
                next_steps.append(next_id)

            # 2. Execution Phase (Agents actually move)
            for i, agent in enumerate(self.agents):
                agent.execute_step(next_steps[i])

            self.time += 1

        # Format output similar to old composite path structure for compatibility
        return self._extract_composite_path()

    def _extract_composite_path(self):
        # Transpose individual agent histories into a list of composite tuples
        max_len = max(len(a.path_history) for a in self.agents)
        composite_path = []
        for t in range(max_len):
            state_at_t = []
            for agent in self.agents:
                # If agent finished early, pad with its last position
                idx = min(t, len(agent.path_history) - 1)
                state_at_t.append(agent.path_history[idx])
            composite_path.append(tuple(state_at_t))
        return composite_path
