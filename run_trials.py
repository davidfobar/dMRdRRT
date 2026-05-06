#!/usr/bin/env python
# Usage: python run_trials.py [--n-trials N] [--n-procs P] [--base-seed S] [--parallel W] [--no-replan]
#   --n-trials   number of independent runs (default 10)
#   --n-procs    MPI ranks per run, must match end_points length (default 8)
#   --base-seed  first seed; each trial uses base_seed + trial_index (default 0)
#   --parallel   number of trials to run concurrently (default 3)
#   --no-replan  pass --no-replan through to the simulator

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", type=int, default=10)
parser.add_argument("--n-procs", type=int, default=8)
parser.add_argument("--base-seed", type=int, default=0)
parser.add_argument("--parallel", type=int, default=3)
parser.add_argument("--no-replan", action="store_true")
args = parser.parse_args()

sim_script = Path(__file__).parent / "r_robust_MR_RRT.py"
sim_dir = Path(__file__).parent / "sim"
shared_metrics = sim_dir / "metrics.jsonl"


def run_trial(trial_idx: int, seed: int) -> dict | None:
    output_dir = sim_dir / ("no_replan" if args.no_replan else "replan")
    output_dir.mkdir(exist_ok=True)

    per_trial_metrics = output_dir / f"metrics_seed{seed:04d}.jsonl"
    per_trial_metrics.unlink(missing_ok=True)

    cmd = [
        "mpiexec", "--oversubscribe", "-n", str(args.n_procs),
        sys.executable, str(sim_script),
        "--seed", str(seed),
        "--no-plot",
        "--metrics-out", str(per_trial_metrics),
    ]
    if args.no_replan:
        cmd.append("--no-replan")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [WARN] trial {trial_idx+1} (seed={seed}) failed (exit {result.returncode}).", flush=True)
        print(result.stderr[-2000:] if result.stderr else "", flush=True)
        return None

    if not per_trial_metrics.exists():
        print(f"  [WARN] trial {trial_idx+1} (seed={seed}): metrics file missing.", flush=True)
        return None

    with open(per_trial_metrics) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    if not lines:
        print(f"  [WARN] trial {trial_idx+1} (seed={seed}): metrics file empty.", flush=True)
        return None

    m = json.loads(lines[-1])

    # append to shared log under a lock (ThreadPoolExecutor uses threads, so a
    # simple file append is safe as long as we write atomically per line)
    with open(shared_metrics, "a") as f:
        f.write(json.dumps(m) + "\n")

    print(
        f"  [done] trial {trial_idx+1} (seed={seed})  "
        f"steps={m['steps']}  "
        f"min_λ₂={m['min_fiedler']:.4f}  "
        f"mean_λ₂={m['mean_fiedler']:.4f}",
        flush=True,
    )
    return m


print(
    f"Running {args.n_trials} trials  "
    f"({args.parallel} parallel x {args.n_procs} ranks each)",
    flush=True,
)

trial_records: list[dict] = []
futures_map = {}

with ThreadPoolExecutor(max_workers=args.parallel) as pool:
    for i in range(args.n_trials):
        seed = args.base_seed + i
        print(f"  [submit] trial {i+1}/{args.n_trials}  seed={seed}", flush=True)
        f = pool.submit(run_trial, i, seed)
        futures_map[f] = (i, seed)

    for f in as_completed(futures_map):
        m = f.result()
        if m is not None:
            trial_records.append(m)

# sort by seed so summary output is deterministic
trial_records.sort(key=lambda r: r["seed"])

if not trial_records:
    print("No successful trials — nothing to report.")
    sys.exit(1)

print(f"\n{len(trial_records)}/{args.n_trials} trials completed. Run plot_summary.py to generate plots.")
