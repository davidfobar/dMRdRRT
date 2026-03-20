# dMRdRRT

Toy implementation of sampling-based motion planning for 2D fields.

Currently supported planners:

- RRT
- RRT*
- PRM (Probabilistic Roadmap)

Both toy obstacle fields and Perlin terrain fields support plotting planner state
(RRT tree or PRM roadmap) and final path overlays.

## Run the demo

```bash
python main.py
```

This runs a deterministic toy planning problem with circular obstacles, prints path stats, and saves:

- `images/rrt_toy_solution.png`

To open the plot window while also saving the image:

```bash
python main.py --show
```

To enable RRT* (cost-based parent selection + local rewiring):

```bash
python main.py --rrt-star
```

## PRM usage (in code / notebook)

```python
from Agent import Agent
from PRM import PRMParameters, PRMRoadmap

roadmap = PRMRoadmap(
	space=field,
	params=PRMParameters(
		n_samples=500,
		k_neighbors=14,
		connection_radius=24.0,
		max_build_attempts=3,
		seed=7,
	),
)
roadmap.build()  # persistent graph, reused for many queries/agents

agent = Agent(
	field,
	start=(8.0, 8.0),
	planner_type="prm",
	prm_roadmap=roadmap,
)
path = agent.plan_to((92.0, 90.0))

# Another agent can reuse the same roadmap.
other_agent = Agent(field, start=(15.0, 12.0), planner_type="prm", prm_roadmap=roadmap)
other_path = other_agent.plan_to((84.0, 70.0))

# If needed later, swap to a different shared roadmap.
other_agent.update_prm_roadmap(roadmap)
```

Use `utils.visualization.plot_agent(agent, planner_name="PRM")` to render the
roadmap and resulting path.