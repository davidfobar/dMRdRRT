from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from FieldClass import BaseFieldClass
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
    and an individual RRTPlanner. During planning, it creates an
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
        rrt_params: RRTParameters,
        max_grade: float = 15.0,
    ) -> None:
        self.field = field
        self.max_grade = max_grade
        self.position: np.ndarray = np.array(start, dtype=float)
        self.last_plan_start: np.ndarray = self.position.copy()
        self.last_goal: np.ndarray | None = None
        self.rrt_params = rrt_params
        self.space = AgentPlannerSpace(field=field, max_grade=max_grade)
        self.planner: RRTPlanner | None = None
        self.path: list[np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Planning interface
    # ------------------------------------------------------------------

    def plan_to(
        self,
        goal: tuple[float, float],
        **kwargs,
    ) -> list[np.ndarray] | None:
        """
        Plan a path from the agent's current position to *goal*.

        The agent passes itself as the field interface so the planner uses
        the agent's edge_is_collision_free (water + grade capability check).
        Any RRTPlanner keyword argument can be overridden via **kwargs for
        this specific call without altering the agent's stored defaults.
        """
        self.last_plan_start = self.position.copy()
        self.last_goal = np.array(goal, dtype=float)
        params = replace(self.rrt_params, **kwargs)
        self.planner = RRTPlanner(
            start=(float(self.position[0]), float(self.position[1])),
            goal=goal,
            space=self.space,
            params=params,
        )
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
