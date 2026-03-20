from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from FieldClass import BaseFieldClass
from PRM import PRMQueryParameters, PRMQueryResult, PRMRoadmap
from RRT import RRTParameters, RRTPlanner


@dataclass(slots=True)
class AgentPlannerSpace:
    """Concrete adapter between a field and the planner for one agent.

    This is where agent-specific traversal rules are combined with the
    field's absolute obstacle checks.
    """

    field: BaseFieldClass
    max_grade: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.field.bounds

    def edge_is_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        if not self.field.edge_is_collision_free(p1, p2):
            return False

        return not self.field.segment_exceeds_capability(p1, p2, max_grade=self.max_grade)


class Agent:
    """
    Virtual robot that owns its physical capabilities and planning state.

    The agent holds a reference to a shared field (global terrain knowledge)
    and an individual planner state. During planning, it creates an
    AgentPlannerSpace adapter that combines field constraints with the
    agent's physical capability limits.

    Multiple Agent instances can share the same field while enforcing
    different physical limits, e.g. one vehicle handles 20 % grade while
    another is limited to 10 %.
    """

    def __init__(
        self,
        field: BaseFieldClass,
        start: tuple[float, float],
        rrt_params: RRTParameters | None = None,
        prm_roadmap: PRMRoadmap | None = None,
        planner_type: str = "rrt",
        max_grade: float = 15.0,
    ) -> None:
        self.field = field
        self.max_grade = max_grade
        self.position: np.ndarray = np.array(start, dtype=float)
        self.last_plan_start: np.ndarray = self.position.copy()
        self.last_goal: np.ndarray | None = None
        self.rrt_params = rrt_params if rrt_params is not None else RRTParameters()
        self.prm_roadmap = prm_roadmap
        self.planner_type = planner_type.lower()
        self.space = AgentPlannerSpace(field=field, max_grade=max_grade)
        self.planner: RRTPlanner | PRMQueryResult | None = None
        self.path: list[np.ndarray] | None = None

    def update_prm_roadmap(self, prm_roadmap: PRMRoadmap | None) -> None:
        """Attach or replace the shared PRM roadmap used by this agent."""
        self.prm_roadmap = prm_roadmap

    # ------------------------------------------------------------------
    # Planning interface
    # ------------------------------------------------------------------

    def plan_to(
        self,
        goal: tuple[float, float],
        planner_type: str | None = None,
        **kwargs,
    ) -> list[np.ndarray] | None:
        """
        Plan a path from the agent's current position to *goal*.

        The agent passes itself as the field interface so the planner uses
        the agent's edge_is_collision_free (water + grade capability check).
        Planner type can be chosen with planner_type in {"rrt", "rrt*", "prm"}.
        PRM is expected to be built externally, attached to the agent, and
        reused across agents.

        Any RRTPlanner keyword argument can be overridden via **kwargs for
        this specific call without altering the agent's stored defaults.
        """
        self.last_plan_start = self.position.copy()
        self.last_goal = np.array(goal, dtype=float)

        active_planner_type = (planner_type or self.planner_type).lower()

        if active_planner_type in {"rrt", "rrt*", "rrt_star"}:
            params = replace(self.rrt_params, **kwargs)
            if active_planner_type in {"rrt*", "rrt_star"}:
                params = replace(params, use_rrt_star=True)

            self.planner = RRTPlanner(
                start=(float(self.position[0]), float(self.position[1])),
                goal=goal,
                space=self.space,
                params=params,
            )
        elif active_planner_type == "prm":
            if self.prm_roadmap is None:
                raise ValueError(
                    "planner_type='prm' requires an externally built PRMRoadmap. "
                    "Pass it at Agent initialization or via update_prm_roadmap()."
                )

            query_params = PRMQueryParameters(
                k_neighbors=self.prm_roadmap.params.k_neighbors,
                connection_radius=self.prm_roadmap.params.connection_radius,
            )
            if kwargs:
                query_params = replace(query_params, **kwargs)

            self.planner = self.prm_roadmap.query(
                start=(float(self.position[0]), float(self.position[1])),
                goal=goal,
                traversal_space=self.space,
                query_params=query_params,
            )
        else:
            raise ValueError(
                f"Unknown planner_type '{active_planner_type}'. Expected one of: rrt, rrt*, prm."
            )

        if active_planner_type == "prm":
            self.path = self.planner.path
        else:
            self.path = self.planner.plan()
        return self.path

    def move_to(
        self,
        goal: tuple[float, float],
        **kwargs,
    ) -> list[np.ndarray] | None:
        """
        Plan to *goal* and, if a path is found, advance the agent's position
        to the goal. Returns the path or None if planning failed.
        """
        path = self.plan_to(goal, **kwargs)
        if path is not None:
            self.position = path[-1].copy()
        return path
