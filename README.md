# dMRdRRT

This project runs a distributed multi-robot planning simulation using MPI. The main entry point is `r_robust_MR_RRT.py`, which simulates multiple agents moving toward assigned goals while trying to preserve communication connectivity.

## Requirements

- Python 3.12+
- `uv`
- An MPI runtime that provides `mpiexec`

## Installation

The Python dependencies are defined in `pyproject.toml` and should be installed with `uv sync` called in the same directory as `pyproject.toml`.

This creates a local virtual environment at `.venv/`.

## Running the Simulation

The simulation is designed to be launched through MPI. The examples below use 8 ranks because the current script defines 8 goal locations.

Run with replanning enabled:

```bash
mpiexec -n 8 .venv/bin/python r_robust_MR_RRT.py
```

Run with replanning disabled:

```bash
mpiexec -n 8 .venv/bin/python r_robust_MR_RRT.py --no-replan
```

The script writes simulation frames into the `sim/` directory.

## Creating an Animation

After the simulation finishes, you can stitch the saved frames into an animation with the helper script in `sim/`:

```bash
.venv/bin/python sim/stitch_sim_frames_to_gif.py
```

## Project Notes

- The `agent` and `graphkit` contains logic that is shared with other projects and experiments outside the scope of this repository. Some methods or utilities in that class may therefore appear more general than what this specific project strictly requires.
- The current simulation setup assumes the number of MPI ranks matches the number of hard-coded endpoints in `r_robust_MR_RRT.py`.
