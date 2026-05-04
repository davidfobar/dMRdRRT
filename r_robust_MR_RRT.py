# mpiexec -n 8 python r_robust_MR_RRT.py

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import Agent, MPITags
from planners.RRT import RRTParameters
from utils.FieldClass import TerrainFieldClass
from graphkit import SwarmGraph
from mpi4py import MPI

_parser = argparse.ArgumentParser()
_parser.add_argument("--no-replan", action="store_true", help="Disable all replanning")
_args, _ = _parser.parse_known_args()
DISABLE_REPLAN = _args.no_replan

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

water_threshold = 0.50
terrain_field = TerrainFieldClass(
    seed=42,
    water_threshold=water_threshold,
)

end_points = [(220, 460),
              (250, 455),
              (280, 445),
              (310, 455),
              (340, 455),
              (370, 465),
              (400, 470),
              (425, 465)]

STEP_DIST = 5.0  # metres per simulation step 
K_LOOKAHEAD = 40
REPLAN_BUFFER_K = 8  # replan when violation predicted within this many steps
REPLAN_COOLDOWN = 15
CURR_POS_TAG = MPITags.CONTROL
LOOKAHEAD_TAG = MPITags.CONTROL + 1
DEGREE_TAG = MPITags.CONTROL + 2

swarm = SwarmGraph(type="m_step_path", num_nodes=size, m=3)
swarm.build(adversarial_nodes=None)
neighbors = swarm.neighbors(rank)
local_degree = swarm.degree(rank)
comm_edges = [(int(i), int(j)) for i, j in swarm.graph.edges()]

terrain_agent = Agent(
    rank=rank,
    neighbors=neighbors,
    device=torch.device("cpu"),
    dtype=torch.float64,
    field=terrain_field,
    start=(240.0 + rank*5.0, 25.0),
    rrt_params=RRTParameters(
        step_size=STEP_DIST,
        max_iters=10000,
        seed=rank + 42,
        use_rrt_star=True,
    ),
    max_grade=30.0,
 )

REPLAN_MAX_DIST = 0.75 * terrain_agent.comms_range

terrain_path = terrain_agent.plan_to(end_points[rank])

# reduce max_iters for replanning
terrain_agent.rrt_params.max_iters = 10000

# delete frames from previous runs (if any)
sim_dir = Path("sim")
if rank == 0:
    sim_dir.mkdir(parents=True, exist_ok=True)
    for f in sim_dir.glob("frame*.png"):
        f.unlink()

if terrain_path is not None and len(terrain_path) > 1:
    waypoints = np.array(terrain_path)
    path_length = float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))
else:
    path_length = float("inf") if terrain_path is None else 0.0

# Barrier so output isn't interleaved across ranks
comm.Barrier()
for r in range(size):
    if rank == r:
        goal = end_points[rank]
        print(
            f"[rank {rank}] start=({240.0 + rank*5.0:.1f}, 25.0) "
            f"goal={goal}  path_length={path_length:.4f}"
        )
    comm.Barrier()

# interpolate along the path at fixed step sizes for smooth simulation
def _advance_along_path(waypoints, current_pos, step_dist):
    """
    Walk `step_dist` metres along `waypoints` starting from `current_pos`.
    Returns (new_position, remaining_waypoints, done).
    `remaining_waypoints` still includes the segment we are currently on.
    """
    pos = current_pos.copy()
    remaining = step_dist
    wpts = list(waypoints)

    while wpts and remaining > 1e-9:
        target = wpts[0]
        to_target = target - pos
        dist = float(np.linalg.norm(to_target))
        if dist <= remaining:
            pos = target
            remaining -= dist
            wpts.pop(0)
        else:
            pos = pos + (to_target / dist) * remaining
            remaining = 0.0

    done = len(wpts) == 0
    return pos, np.array(wpts), done

def _compute_k_lookahead(waypoints, current_pos, step_dist, k):
    """
    Compute the next k predicted positions spaced by `step_dist` along path.
    Once an agent reaches the end, remaining lookahead entries repeat final pose.
    """
    trial_pos = current_pos.copy()
    trial_waypoints = np.array(waypoints)
    trial_done = len(trial_waypoints) == 0
    lookahead = []

    for _ in range(k):
        if trial_done:
            lookahead.append(trial_pos.copy())
            continue

        next_pos, next_waypoints, next_done = _advance_along_path(
            trial_waypoints, trial_pos, step_dist
        )
        lookahead.append(next_pos.copy())
        trial_pos = next_pos
        trial_waypoints = next_waypoints
        trial_done = next_done

    return np.array(lookahead, dtype=float)


if terrain_path is None or len(terrain_path) < 2:
    # No valid path — agent stays put
    sim_waypoints = np.array([terrain_agent.position])
    done = True
else:
    # Drop the start point (agent is already there)
    sim_waypoints = np.array(terrain_path)[1:]
    done = False

last_replan_step = -REPLAN_COOLDOWN
step = 0
while True:
    comm.Barrier()  # all ranks begin this step together

    # If already done, just stay put and keep sharing that fact with neighbors.
    if not done:
        planned_next_pos, planned_next_waypoints, planned_done = _advance_along_path(
            sim_waypoints, terrain_agent.position, STEP_DIST
        )
    else:
        planned_next_pos = terrain_agent.position.copy()
        planned_next_waypoints = sim_waypoints
        planned_done = True

    # Neighbor comms sync: share current position, lookahead, and degree for this step
    n_neighbors = len(neighbors)
    my_lookahead = _compute_k_lookahead(
        sim_waypoints,
        terrain_agent.position,
        STEP_DIST,
        K_LOOKAHEAD,
    )
    if n_neighbors > 0:
        curr_send = torch.as_tensor(
            np.tile(terrain_agent.position, (n_neighbors, 1)),
            device=terrain_agent.device,
            dtype=terrain_agent.dtype,
        )
        lookahead_send = torch.as_tensor(
            np.tile(my_lookahead, (n_neighbors, 1, 1)),
            device=terrain_agent.device,
            dtype=terrain_agent.dtype,
        )
        degree_send = torch.as_tensor(
            np.full((n_neighbors, 1), float(local_degree)),
            device=terrain_agent.device,
            dtype=terrain_agent.dtype,
        )
        neighbor_curr_positions = terrain_agent.send_recv_tensor(
            comm, curr_send, tag=CURR_POS_TAG
        )
        neighbor_lookahead_paths = terrain_agent.send_recv_tensor(
            comm, lookahead_send, tag=LOOKAHEAD_TAG
        )
        neighbor_degrees = terrain_agent.send_recv_tensor(
            comm, degree_send, tag=DEGREE_TAG
        )
    else:
        # fall back if no neighbors (shouldn't happen in this graph, but just in case)
        neighbor_curr_positions = torch.empty((0, 2), device=terrain_agent.device, dtype=terrain_agent.dtype)
        neighbor_lookahead_paths = torch.empty((0, K_LOOKAHEAD, 2), device=terrain_agent.device, dtype=terrain_agent.dtype)
        neighbor_degrees = torch.empty((0, 1), device=terrain_agent.device, dtype=terrain_agent.dtype)

    ########################################################################
    # Replanning logic:
    #   - Check if any neighbor's predicted lookahead path violates comms range within the next K steps.
    #   - If so, only the "more responsible" agent (lower degree, or tie-break by rank) replans.
    #   - Replan by trying to rendezvous at a point along the predicted path of the neighbor that would violate comms, then continue to goal.
    #   - To keep things simple, the agent only tries to replan once per predicted violation, and waits at least REPLAN_COOLDOWN steps between replans.
    ########################################################################
    out_of_range_reports = []
    replan_candidate_idx = None
    replan_candidate_key = None
    did_replan = False
    if not DISABLE_REPLAN:

        if n_neighbors > 0:
            neighbor_lookahead_np = neighbor_lookahead_paths.detach().cpu().numpy()
            neighbor_degrees_np = neighbor_degrees.detach().cpu().numpy()
            for i, nbr in enumerate(neighbors):
                # Compare each predicted step between agent and neighbors
                lookahead_dists = np.linalg.norm(
                    neighbor_lookahead_np[i] - my_lookahead,
                    axis=1,
                )
                max_predicted_dist = float(np.max(lookahead_dists))
                out_of_range_idx = np.where(lookahead_dists > 0.8 * terrain_agent.comms_range)[0]

                # build a report to log violations
                if out_of_range_idx.size > 0:
                    first_idx = int(out_of_range_idx[0])
                    out_of_range_reports.append(
                        (
                            nbr,
                            first_idx + 1,
                            float(lookahead_dists[first_idx]),
                            max_predicted_dist,
                        )
                    )

                # If there is a predicted violation, decide whether this agent owns the
                # replanning responsibility (vs. the neighbor) by degree/rank tie-break.
                first_violation_step_idx = int(out_of_range_idx[0]) if out_of_range_idx.size > 0 else None
                if first_violation_step_idx is not None and (step - last_replan_step) >= REPLAN_COOLDOWN:
                    # Lower-degree nodes own replanning; rank breaks ties.
                    neighbor_degree = int(round(float(neighbor_degrees_np[i, 0])))
                    this_agent_should_replan = (local_degree < neighbor_degree) or (local_degree == neighbor_degree and rank < nbr)

                    if this_agent_should_replan:
                        distance_at_violation = float(lookahead_dists[first_violation_step_idx])
                        replan_priority_key = (first_violation_step_idx, distance_at_violation)

                        # Update neighbor replan candidate if either wins on priority:
                        #   1) earliest violation step
                        #   2) smallest violation distance at that step
                        if replan_candidate_key is None or replan_priority_key < replan_candidate_key:
                            replan_candidate_key = replan_priority_key
                            replan_candidate_idx = i

        replan_needed = replan_candidate_idx is not None
        if replan_needed and not done:
            # get the neighbor's predicted path that we are trying to rendezvous with
            neighbor_lookahead_np = neighbor_lookahead_paths.detach().cpu().numpy()
            replan_neighbor = neighbors[replan_candidate_idx]

            # Index of the first predicted violation for the selected neighbor.
            violation_lookahead_idx = int(replan_candidate_key[0])
            violation_lookahead_idx = max(0, min(violation_lookahead_idx, K_LOOKAHEAD - 1))
            my_predicted_pos = my_lookahead[violation_lookahead_idx]
            neighbor_predicted_pos = neighbor_lookahead_np[replan_candidate_idx][violation_lookahead_idx]

            # Compute a degree-weighted rendezvous point at the violation step.
            # Higher neighbor degree pulls the point closer to the neighbor trajectory.
            neighbor_degree = int(round(float(neighbor_degrees_np[replan_candidate_idx, 0])))
            total_pair_degree = max(1, local_degree + neighbor_degree)
            neighbor_weight = neighbor_degree / total_pair_degree
            rendezvous_point = (1.0 - neighbor_weight) * my_predicted_pos + neighbor_weight * neighbor_predicted_pos

            # Preserve the currently committed path prefix up to the violation step.
            preserved_prefix_positions = []
            replan_anchor_pos = terrain_agent.position.copy()
            replan_anchor_waypoints = np.array(sim_waypoints)
            anchor_path_exhausted = len(replan_anchor_waypoints) == 0
            for _ in range(violation_lookahead_idx):
                if anchor_path_exhausted:
                    break
                replan_anchor_pos, replan_anchor_waypoints, anchor_path_exhausted = _advance_along_path(
                    replan_anchor_waypoints,
                    replan_anchor_pos,
                    STEP_DIST,
                )
                preserved_prefix_positions.append(replan_anchor_pos.copy())

            # Save and restore planner state because plan_to uses terrain_agent.position.
            original_agent_pos = terrain_agent.position.copy()

            # determine if agent is within comms range of the end goal at the violation step 
            # if so, can replan directly to goal instead of rendezvous + goal
            goals_within_comms_range = np.linalg.norm(
                np.array(end_points[rank], dtype=float) - np.array(end_points[replan_neighbor], dtype=float)
            ) <= terrain_agent.comms_range

            # we have to temporarily move the agent to the anchor position to plan from there,
            # it would be cleaner if the agent could plan from an arbitrary state without modifying its own position
            terrain_agent.position = replan_anchor_pos.astype(float)
            if goals_within_comms_range:
                # If the final goals are still mutually reachable, skip the rendezvous
                # detour and replan directly from the anchor point to this agent's goal.
                direct_goal_path = terrain_agent.plan_to(end_points[rank])
                terrain_agent.position = original_agent_pos

                if direct_goal_path is not None and len(direct_goal_path) > 1:
                    # Keep original prefix, then splice in a fresh path to goal.
                    stitched_path_segments = []
                    if len(preserved_prefix_positions) > 0:
                        stitched_path_segments.append(np.array(preserved_prefix_positions, dtype=float))
                    stitched_path_segments.append(np.array(direct_goal_path, dtype=float)[1:])

                    sim_waypoints = np.concatenate(stitched_path_segments, axis=0)
                    # Refresh the immediate next simulated move from the new waypoint list.
                    planned_next_pos, planned_next_waypoints, planned_done = _advance_along_path(
                        sim_waypoints, terrain_agent.position, STEP_DIST
                    )
                    last_replan_step = step
                    did_replan = True
            else:
                # Otherwise, first plan to the rendezvous point, then from rendezvous to goal.
                rendezvous_goal = tuple(rendezvous_point.astype(int))
                rendezvous_path = terrain_agent.plan_to(rendezvous_goal)
                if rendezvous_path is not None and len(rendezvous_path) > 1:
                    # The second planning leg starts from the rendezvous location itself.
                    terrain_agent.position = rendezvous_point.astype(float)
                    goal_path = terrain_agent.plan_to(end_points[rank])
                else:
                    goal_path = None
                terrain_agent.position = original_agent_pos

                if rendezvous_path is not None and len(rendezvous_path) > 1 and goal_path is not None and len(goal_path) > 1:
                    # Keep original prefix, then append rendezvous leg and goal leg.
                    stitched_path_segments = []
                    if len(preserved_prefix_positions) > 0:
                        stitched_path_segments.append(np.array(preserved_prefix_positions, dtype=float))
                    stitched_path_segments.append(np.array(rendezvous_path, dtype=float)[1:])
                    stitched_path_segments.append(np.array(goal_path, dtype=float)[1:])

                    sim_waypoints = np.concatenate(stitched_path_segments, axis=0)
                    # Refresh the immediate next simulated move from the new waypoint list.
                    planned_next_pos, planned_next_waypoints, planned_done = _advance_along_path(
                        sim_waypoints, terrain_agent.position, STEP_DIST
                    )
                    last_replan_step = step
                    did_replan = True

    # update state for this iteration
    if not done:
        terrain_agent.position = planned_next_pos
        sim_waypoints = planned_next_waypoints
        done = planned_done

    # prepare the path for visualization
    if sim_waypoints.size > 0:
        planned_path_for_plot = np.vstack([terrain_agent.position.copy(), sim_waypoints.copy()])
    else:
        planned_path_for_plot = np.array([terrain_agent.position.copy()])

    # Determine if agent is out of comms range with any neighbor within the next K_LOOKAHEAD steps, and share that info for visualization.
    if n_neighbors > 0:
        curr_nbr_np = neighbor_curr_positions.detach().cpu().numpy()
        curr_dists = np.linalg.norm(curr_nbr_np - terrain_agent.position, axis=1)
        has_comms_issue = bool(np.any(curr_dists > terrain_agent.comms_range))
    else:
        has_comms_issue = False
    gathered_state = comm.gather(
        (rank, terrain_agent.position.copy(), has_comms_issue, planned_path_for_plot),
        root=0,
    )

    # Visualization on rank 0 after gathering all states.
    if rank == 0:
        gathered_state.sort(key=lambda x: x[0])
        positions = np.array([entry[1] for entry in gathered_state], dtype=float)
        comms_flags = np.array([entry[2] for entry in gathered_state], dtype=bool)
        planned_paths = [entry[3] for entry in gathered_state]
        colors = np.where(comms_flags, "red", "blue")

        fig, ax = terrain_field.plot(
            show=False,
            title=f"Step {step}: Agent Comms Status",
            finalize=False,
        )
        terrain_field.overlay_obstacle_regions(
            ax,
            max_grade=terrain_agent.max_grade,
        )
        for i, j in comm_edges:
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color="white",
                linewidth=1.2,
                alpha=0.75,
                zorder=4,
            )
        for path in planned_paths:
            if path.shape[0] >= 2:
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    linestyle="--",
                    color="black",
                    alpha=0.5,
                    linewidth=1.2,
                    zorder=3,
                )
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=colors,
            s=55,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

        frame_path = sim_dir / f"frame_{step:04d}.png"
        fig.savefig(frame_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Serialised per-rank progress print
    for r in range(size):
        if rank == r:
            status = "DONE" if done else "moving"
            print(
                f"[step {step:4d}][rank {rank}] pos=({terrain_agent.position[0]:.2f}, "
                f"{terrain_agent.position[1]:.2f})  [{status}]"
            )
            for nbr, first_k, first_dist, max_dist in out_of_range_reports:
                print(
                    f"  [rank {rank}] neighbor {nbr} is out of range: "
                    f"first_violation_at_k={first_k} "
                    f"distance={first_dist:.2f} "
                    f"max_k_distance={max_dist:.2f} "
                    f"> comms_range={terrain_agent.comms_range:.2f}"
                )
            if not DISABLE_REPLAN and did_replan and not done:
                print(
                    f"  [rank {rank}] replanned due to lookahead violation within "
                    f"REPLAN_BUFFER_K={REPLAN_BUFFER_K}"
                )
        comm.Barrier()

    step += 1

    # Check whether every rank is finished (all_done == size means all done)
    local_done = np.array([1 if done else 0], dtype=np.int32)
    global_done = np.zeros(1, dtype=np.int32)
    comm.Allreduce(local_done, global_done, op=MPI.SUM)
    if global_done[0] == size:
        break

if rank == 0:
    print(f"\nSimulation complete after {step} steps.")