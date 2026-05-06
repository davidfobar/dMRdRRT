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

## Running Batch Trials On Constant Terrain, Start and Stop Conditions

> **Warning:** Each trial runs a full RRT* simulation across 8 MPI ranks and can take several minutes. Running many trials multiplies this cost proportionally. Plan accordingly before launching large batches.

`run_trials.py` executes multiple independent simulations over different random seeds and collects connectivity metrics from each.

Run 10 trials with replanning enabled, 4 at a time using 8 ranks each:

```bash
python run_trials.py --n-trials 10 --n-procs 8 --base-seed 0 --parallel 4
```

Run 10 trials with replanning disabled:

```bash
python run_trials.py --n-trials 10 --n-procs 8 --base-seed 0 --parallel 4 --no-replan
```

Each trial writes its metrics to `sim/replan/metrics_seed*.jsonl` or `sim/no_replan/metrics_seed*.jsonl` depending on the condition. These files accumulate across runs and are safe to append to.

### Batch Trial Post-processing

After trials complete, generate summary plots with:

```bash
python plot_summary.py             # replan condition  → trials/replan/summary_replan.png
python plot_summary.py --no-replan # no-replan condition → trials/no_replan/summary_no_replan.png
```

The plot shows algebraic connectivity (λ₂) traces across all trials normalized to [0, 1] progress, along with a stats block showing trial count, failure count, and average minimum λ₂.

## Project Notes

- The `agent` and `graphkit` contains logic that is shared with other projects and experiments outside the scope of this repository. Some methods or utilities in that class may therefore appear more general than what this specific project strictly requires.
- The current simulation setup assumes the number of MPI ranks matches the number of hard-coded endpoints in `r_robust_MR_RRT.py`.
