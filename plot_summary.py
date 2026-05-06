#!/usr/bin/env python
# Usage: python plot_summary.py [--no-replan] [--sim-dir PATH] [--out-dir PATH]

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--no-replan", action="store_true", help="Read from sim/no_replan instead of sim/replan")
parser.add_argument("--sim-dir", type=str, default="sim", help="Base sim directory (subdir replan/ or no_replan/ is appended automatically)")
parser.add_argument("--out-dir", type=str, default="trials", help="Directory to write summary outputs")
args = parser.parse_args()

condition = "no_replan" if args.no_replan else "replan"
sim_dir = Path(args.sim_dir) / condition
out_dir = Path(args.out_dir) / condition
out_dir.mkdir(parents=True, exist_ok=True)

metric_files = sorted(sim_dir.glob("metrics_seed*.jsonl"))
if not metric_files:
    print(f"No metrics_seed*.jsonl files found in {sim_dir}")
    sys.exit(1)

trial_records = []
for path in metric_files:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    for line in lines:
        trial_records.append(json.loads(line))

trial_records.sort(key=lambda r: r["seed"])
print(f"Loaded {len(trial_records)} trial(s) from {sim_dir}  [condition: {condition}]")

min_fiedlers = np.array([r["min_fiedler"] for r in trial_records])
mean_fiedlers = np.array([r["mean_fiedler"] for r in trial_records])
steps = np.array([r["steps"] for r in trial_records])
disconnected_runs = int(np.sum(min_fiedlers == 0.0))

print("\n" + "=" * 50)
print(f"Condition        : {condition}")
print(f"Trials           : {len(trial_records)}")
print(f"Steps   mean±std : {steps.mean():.1f} ± {steps.std():.1f}  (min={steps.min()}, max={steps.max()})")
print(f"min λ₂  mean±std : {min_fiedlers.mean():.4f} ± {min_fiedlers.std():.4f}  (min={min_fiedlers.min():.4f})")
print(f"mean λ₂ mean±std : {mean_fiedlers.mean():.4f} ± {mean_fiedlers.std():.4f}")
print(f"Runs with λ₂=0   : {disconnected_runs}/{len(trial_records)}")
print("=" * 50)

fig, ax = plt.subplots(figsize=(6, 4))
for r in trial_records:
    x = np.linspace(0, 1, len(r["fiedler_log"]))
    ax.plot(x, r["fiedler_log"], alpha=0.4, linewidth=0.9)
ax.set_xlabel("Normalized simulation progress")
ax.set_ylabel("Algebraic connectivity (λ₂)")
ax.set_title(f"λ₂ traces across all trials [{condition}]")
ax.set_ylim(bottom=-0.1)

textblock = (
    f"n trials : {len(trial_records)}\n"
    f"failed   : {disconnected_runs}\n"
    f"avg min λ₂: {min_fiedlers.mean():.4f}"
)
ax.text(
    0.02, 0.02, textblock,
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=9,
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
)

fig.tight_layout()
summary_plot = out_dir / f"summary_{condition}.png"
fig.savefig(summary_plot, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSummary plot saved to {summary_plot}")

summary_json = out_dir / "summary.json"
with open(summary_json, "w") as f:
    json.dump(
        {
            "condition": condition,
            "n_trials": len(trial_records),
            "min_fiedler": {"mean": float(min_fiedlers.mean()), "std": float(min_fiedlers.std()),
                            "min": float(min_fiedlers.min()), "max": float(min_fiedlers.max())},
            "mean_fiedler": {"mean": float(mean_fiedlers.mean()), "std": float(mean_fiedlers.std())},
            "steps": {"mean": float(steps.mean()), "std": float(steps.std()),
                      "min": int(steps.min()), "max": int(steps.max())},
            "disconnected_runs": disconnected_runs,
            "trials": trial_records,
        },
        f,
        indent=2,
    )
print(f"Summary JSON saved to {summary_json}")
