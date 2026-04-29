"""MPI entrypoint for running TE-ADMM experiments across many configurations."""

from pathlib import Path
from datetime import UTC, datetime
import argparse
import json
import hashlib
import sys

from consensus import TE_ADMM, ADMMParams
from consensus.BCM import BCM, BCMParams
from graphkit import SwarmGraph
from agent import GPAgent
from fields import FieldBuilder
from metrics import plot_admm_results

from mpi4py import MPI
import torch
import numpy as np
import gpytorch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TE-ADMM experiments with configurable datasets and seeds."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="SRT30_N17E073",
        help=(
            "Dataset identifier or path to a *_meta.json file. "
            "If a bare name is provided, it is resolved under --data-root."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Base directory for stored field metadata files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--n-samples-per-rank",
        type=int,
        default=1000,
        help="Number of training samples drawn per rank from the field.",
    )
    parser.add_argument(
        "--num-adversaries",
        type=int,
        default=1,
        help="Number of adversarial ranks (excluding rank 0).",
    )
    parser.add_argument(
        "--dist-method",
        type=str,
        default="slice_axis0",
        help="Partitioning method passed to FieldBuilder.gpytorch_data_for_rank.",
    )
    parser.add_argument(
        "--local-iters",
        type=int,
        default=100,
        help="Number of local training iterations per agent before consensus.",
    )
    parser.add_argument(
        "--local-lr", type=float, default=0.1, help="Learning rate for local GP training."
    )
    parser.add_argument(
        "--admm-iters",
        type=int,
        default=100,
        help="Maximum TE-ADMM iterations during consensus.",
    )
    parser.add_argument(
        "--rho", type=float, default=1.0, help="TE-ADMM rho penalty parameter."
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=20.0,
        help="TE-ADMM proximal/linearization coefficient (>0).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/te_admm",
        help="Root directory for saving experiment artifacts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional label for this run; defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating diagnostic plots (useful when running many experiments).",
    )
    parser.add_argument(
        "--adversary-noise-std",
        type=float,
        default=1.0,
        help=(
            "Multiplicative gain applied to adversarial y_ij (1.0 keeps magnitude, >1 amplifies, <1 attenuates)."
        ),
    )
    parser.add_argument(
        "--adversary-white-noise",
        type=float,
        default=0.25,
        help="Std dev of additive white noise appended after scaling the y-updates.",
    )
    parser.add_argument(
        "--num-pred-locs",
        type=int,
        default=200,
        help="Number of uniformly sampled locations for BCM consensus predictions.",
    )
    parser.add_argument(
        "--bcm-iters",
        type=int,
        default=200,
        help="Maximum BCM iterations for prediction consensus.",
    )
    parser.add_argument(
        "--bcm-tol",
        type=float,
        default=1e-4,
        help="BCM early-stop tolerance for prediction consensus.",
    )
    parser.add_argument(
        "--skip-te-admm",
        action="store_true",
        help="Skip TE-ADMM consensus; keep trust weights at 1.",
    )
    parser.add_argument(
        "--bias-scale",
        type=float,
        default=1.0,
        help="Multiplicative bias applied to adversarial y_ij when using 'stubborn_bias' attack.",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default="add_noise",
        choices=["add_noise", "stubborn_bias"],
        help="Type of adversarial attack to inject during TE-ADMM y-update.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save experiment artifacts. Overrides --output-root if provided.",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=1e-3,
        help=(
            "Observation (measurement) noise std to add to sampled training targets. "
            "If not provided, falls back to the field's recorded `params.noise` value (if any)."
        ),
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: str, data_root: str | Path) -> Path:
    """Resolve dataset identifier to an on-disk *_meta.json path."""
    candidate = Path(dataset_arg)
    if candidate.is_file():
        return candidate

    data_root = Path(data_root)
    if dataset_arg.endswith(".json"):
        candidate = data_root / dataset_arg
        if candidate.is_file():
            return candidate

    if not dataset_arg.endswith("_meta.json"):
        candidate = data_root / f"{dataset_arg}_meta.json"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Unable to resolve dataset '{dataset_arg}'. Provide a valid path or identifier."
    )


def field_label_from_dataset(dataset_path: Path) -> str:
    """Derive a stable field label from a dataset meta filename."""
    stem = dataset_path.stem
    if stem.endswith("_meta"):
        stem = stem[:-5]
    return stem


def prepare_output_directory(
    base_dir: str | Path, field_label: str, seed: int, n_samples: int, run_name: str | None
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = run_name or timestamp
    out_dir = Path(base_dir) / field_label / f"seed_{seed}" / f"samples_{n_samples}" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset, args.data_root)
    field_label = field_label_from_dataset(dataset_path)

    seed_tag = f"{field_label}|{args.adversary_white_noise:.6f}"
    seed_hash = int(hashlib.sha256(seed_tag.encode("utf-8")).hexdigest()[:8], 16)
    derived_seed = (int(args.seed) + seed_hash) % (2**32 - 1)

    torch.manual_seed(derived_seed)
    np.random.seed(derived_seed)
    device = torch.device("cpu")
    dtype = torch.float64

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    gp_field = FieldBuilder.load(dataset_path)
    # pass measurement noise (if the field builder records a noise level) to sampled observations
    # determine observation noise: CLI override (--obs-noise) takes precedence,
    # otherwise fall back to the field's internal `params.noise` when available
    if args.obs_noise is not None:
        noise_std = float(args.obs_noise)
    else:
        noise_std = 0.0
        if hasattr(gp_field, "params") and gp_field.params is not None:
            try:
                noise_std = float(getattr(gp_field.params, "noise", 0.0))
            except Exception:
                noise_std = 0.0

    train_x, train_y = gp_field.gpytorch_data_for_rank(
        rank=rank,
        world=size,
        method=args.dist_method,
        n=args.n_samples_per_rank,
        seed=derived_seed + rank,
        device=device,
        dtype=dtype,
        noise_std=noise_std,
    )

    if rank == 0:
        rng = np.random.default_rng(derived_seed)
        if args.num_adversaries > max(0, size - 1):
            raise ValueError("numAdversaries exceeds number of non-zero ranks")
        adversarial_nodes = set(
            rng.choice(
                np.arange(1, size), args.num_adversaries, replace=False
            ).tolist()
        )
    else:
        adversarial_nodes = None

    adversarial_nodes = comm.bcast(adversarial_nodes, root=0)

    swarm = SwarmGraph(type="m_step_path", num_nodes=size, m=3)
    swarm.build(adversarial_nodes=adversarial_nodes)
    neighbors = swarm.neighbors(rank)
    agent = GPAgent(
        rank=rank,
        neighbors=neighbors,
        train_x=train_x,
        train_y=train_y,
        device=device,
        dtype=dtype,
        I_am_adversary=(rank in adversarial_nodes),
        noise_gain=args.adversary_noise_std,
        white_noise_std=args.adversary_white_noise,
        bias_scale=args.bias_scale,
        attack_type=args.attack_type,
    )

    # Determine output directory (override if --output-dir is provided)
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        if rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = prepare_output_directory(
            args.output_root,
            field_label,
            args.seed,
            args.n_samples_per_rank,
            args.run_name,
        )

    # Local training to warm-start consensus
    for i in range(args.local_iters):
        loss_val = agent.train_local(iters=1, lr=args.local_lr)
        loss_mean = comm.allreduce(loss_val, op=MPI.SUM) / size
        if rank == 0 and (i % 10 == 0 or i == args.local_iters - 1):
            print(f"iter {i:03d}  mean_loss {loss_mean:.4f}")

    # Use fixed local test points per rank for consistent RMSE evaluation
    D = train_x.shape[1]
    mins = train_x.min(dim=0).values
    maxs = train_x.max(dim=0).values
    test_x = mins + (maxs - mins) * torch.rand(
        args.num_pred_locs,
        D,
        device=device,
        dtype=dtype,
    )
    test_y_np = gp_field.sample_field(test_x.cpu().numpy())
    test_y = torch.tensor(test_y_np, device=device, dtype=dtype)

    agent.model.eval()
    agent.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = agent.likelihood(agent.model(test_x))
        rmse = torch.mean((pred.mean - test_y) ** 2).sqrt().item()
    rmse_mean = comm.allreduce(rmse, op=MPI.SUM) / size
    if rank == 0:
        print(f"RMSE (mean across ranks): {rmse_mean:.4f}")

    init_params = comm.gather(
        {
            "mean": float(agent.model.mean_module.constant.detach().cpu().item()),
            "outputscale": float(agent.model.covar_module.outputscale.detach().cpu().item()),
            "noise": float(agent.likelihood.noise.detach().cpu().item()),
            "lengthscale": agent.model.covar_module.base_kernel.lengthscale.detach().cpu().view(-1).tolist(),
        },
        root=0,
    )

    theta0, labels = agent.theta_pack()
    theta0 = theta0.detach().to(device=device, dtype=dtype)

    admm = ADMMParams()
    admm.max_iters = args.admm_iters
    admm.rho = args.rho
    admm.kappa = 20 #len(agent.neighbors)*args.rho*2
    # enable trust-edge updates only when TE is not skipped
    admm.trusted_edge = not args.skip_te_admm
    admm.tau = 0.5
    admm.alpha = 0.5
    admm.beta = 0.05

    consensus = TE_ADMM(
        comm=comm,
        rank=rank,
        agent=agent,
        x0=theta0,
        neighbors=neighbors,
        params=admm,
        convex_func="gp",
        use_te=not args.skip_te_admm,
    )

    if args.skip_te_admm and rank == 0:
        print("[TE-ADMM] trust updates disabled; weights remain at 1.")

    for it in range(admm.max_iters):
        consensus.step()
        theta_i = consensus.get_primal()
        nll_i, _ = agent.local_nll_and_grad(theta_i)
        nll_mean = comm.allreduce(float(nll_i.item()), op=MPI.SUM) / size

        if rank == 0 and (it % 10 == 0 or it == admm.max_iters - 1):
            print(f"[TE-ADMM] iter {it:03d} mean_nll {nll_mean:.4f}")

    theta_star = consensus.get_primal()
    agent.theta_unpack(theta_star)

    local_failed = 0
    rmse2 = 0.0
    agent.model.eval()
    agent.likelihood.eval()
    try:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = agent.likelihood(agent.model(test_x))
            rmse2 = torch.mean((pred.mean - test_y) ** 2).sqrt().item()
    except Exception as e:
        local_failed = 1
        print(f"[rank {rank}] post-ADMM prediction failed: {type(e).__name__}", flush=True)

    n_failed = comm.allreduce(local_failed, op=MPI.SUM)
    if n_failed > 0:
        if rank == 0:
            print(
                f"[TE-ADMM] {n_failed}/{size} rank(s) raised NotPSD after consensus "
                f"(attack destabilized hyperparams). Saving failed status and exiting.",
                flush=True,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json(out_dir / "metrics.json", {
                "status": "failed_not_psd",
                "rmse_local_mean": rmse_mean,
                "n_failed_ranks": n_failed,
            })
        comm.Barrier()
        MPI.Finalize()
        sys.exit(0)

    rmse2_mean = comm.allreduce(rmse2, op=MPI.SUM) / size
    if rank == 0:
        print(f"[TE-ADMM] RMSE mean across ranks: {rmse2_mean:.4f}")

    # --- BCM consensus on GP predictions at shared locations ---
    # Use the same local RMSE test points from all ranks as the shared BCM grid.
    local_pred_locs = test_x.detach().cpu().numpy()
    all_pred_locs = comm.gather(local_pred_locs, root=0)
    if rank == 0:
        pred_locs_np = np.concatenate(all_pred_locs, axis=0)
    else:
        pred_locs_np = None

    pred_locs_np = comm.bcast(pred_locs_np, root=0)

    pred_locs = torch.tensor(pred_locs_np, device=device, dtype=dtype)
    agent.model.eval()
    agent.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = agent.likelihood(agent.model(pred_locs))
        mu_pred = pred_dist.mean
        var_pred = pred_dist.variance.clamp_min(1e-8)

    prior_var = float(agent.model.covar_module.outputscale.detach().cpu().item())
    prior_var = max(prior_var, 1e-8)
    bcm_params = BCMParams(
        mode="rbcm",
        prior_var=prior_var,
        alpha=1.0,
        max_iters=args.bcm_iters,
        tol=args.bcm_tol,
    )

    #increase tau to be more selective in pruning edges for the BCM consensus, since low-trust edges can be especially harmful when sharing raw predictions
    admm.tau = 0.85

    # Globally collect trust weights and neighbor lists; prune any edge where either
    # direction's trust weight is below admm.tau, then assign updated neighbor lists.
    trust_weights = consensus.w_ji.detach().clone().cpu()
    all_trust_weights = comm.allgather(trust_weights)   # list[size] of per-rank weight tensors
    all_neighbors = comm.allgather(neighbors)           # list[size] of per-rank neighbor lists

    pruned_neighbors = []
    for i in range(size):
        trusted = []
        for k, j in enumerate(all_neighbors[i]):
            w_i_trusts_j = float(all_trust_weights[i][k])
            # Find i in j's neighbor list to get the reverse trust weight
            if i in all_neighbors[j]:
                k_rev = all_neighbors[j].index(i)
                w_j_trusts_i = float(all_trust_weights[j][k_rev])
            else:
                w_j_trusts_i = 0.0
            if w_i_trusts_j > admm.tau and w_j_trusts_i > admm.tau:
                trusted.append(j)
        pruned_neighbors.append(trusted)

    agent.neighbors = pruned_neighbors[rank]

    bcm = BCM(
        comm=comm,
        rank=rank,
        agent=agent,
        mu_local=mu_pred,
        var_local=var_pred,
        neighbors=agent.neighbors,
        params=bcm_params,
    )
    bcm.run()

    mu_cons, var_cons = bcm.current_estimate()
    mu_all = comm.gather(mu_pred, root=0)
    var_all = comm.gather(var_pred, root=0)
    mu_cons_all = comm.gather(mu_cons, root=0)
    var_cons_all = comm.gather(var_cons, root=0)

    bcm_pred_payload = None
    mu_stack = None
    var_stack = None
    mu_cons_stack = None
    var_cons_stack = None
    y_true = None

    if rank == 0:
        mu_stack = torch.stack(mu_all, dim=0)
        var_stack = torch.stack(var_all, dim=0)
        mu_cons_stack = torch.stack(mu_cons_all, dim=0)
        var_cons_stack = torch.stack(var_cons_all, dim=0)
        y_true = torch.tensor(gp_field.sample_field(pred_locs_np), dtype=dtype)

        honest_mask = torch.ones(size, dtype=torch.bool)
        for adv in adversarial_nodes:
            honest_mask[adv] = False

        mu_honest = mu_stack[honest_mask].mean(dim=0)
        var_honest = var_stack[honest_mask].mean(dim=0)

        max_diff_mu = (mu_cons_stack - mu_cons_stack[0]).abs().max().item()
        max_diff_var = (var_cons_stack - var_cons_stack[0]).abs().max().item()
        diff_honest = (mu_cons_stack[0] - mu_honest).abs().max().item()

        # --- BCM prediction metrics ---
        # Per-rank squared errors: [size, N_preds]
        _se = (mu_cons_stack - y_true).pow(2)
        _rmse_per_rank = _se.mean(dim=1).sqrt()          # [size]
        bcm_rmse_mean = _rmse_per_rank.mean().item()
        bcm_rmse_std  = _rmse_per_rank.std().item()

        # nRMSE: normalize by y_true range (dataset-agnostic scale)
        _y_range = y_true.max() - y_true.min()
        y_range   = float(_y_range.item())
        y_std_val = float(y_true.std().item())
        bcm_nrmse_range    = float((_rmse_per_rank.mean() / _y_range).item())
        bcm_nrmse_std_norm = float((_rmse_per_rank.mean() / y_true.std()).item())

        # NLPD: -mean log N(y | mu, var) = 0.5 * mean[log(2π σ²) + (y-μ)²/σ²]
        _nlpd_per_rank = 0.5 * (
            torch.log(2 * torch.pi * var_cons_stack) + _se / var_cons_stack
        ).mean(dim=1)                                     # [size]
        bcm_nlpd_mean = _nlpd_per_rank.mean().item()
        bcm_nlpd_std  = _nlpd_per_rank.std().item()

        # Honest-agent subsets
        bcm_nrmse_honest = float((_rmse_per_rank[honest_mask].mean() / _y_range).item())
        bcm_nlpd_honest  = float(_nlpd_per_rank[honest_mask].mean().item())

        print("[BCM] prediction consensus")
        print(f"num_pred_locs={args.num_pred_locs}")
        print(f"mean RMSE:               {bcm_rmse_mean:.4f}")
        print(f"stdev RMSE:              {bcm_rmse_std:.4f}")
        print(f"mean nRMSE (range-norm): {bcm_nrmse_range:.4f}")
        print(f"mean nRMSE (std-norm):   {bcm_nrmse_std_norm:.4f}")
        print(f"mean NLPD:               {bcm_nlpd_mean:.4f}")
        print(f"honest nRMSE:            {bcm_nrmse_honest:.4f}")
        print(f"honest NLPD:             {bcm_nlpd_honest:.4f}")

    final_params = comm.gather(
        {
            "outputscale": float(agent.model.covar_module.outputscale.detach().cpu().item()),
            "noise": float(agent.likelihood.noise.detach().cpu().item()),
            "lengthscale": agent.model.covar_module.base_kernel.lengthscale.detach().cpu().view(-1).tolist(),
        },
        root=0,
    )

    local_hist = consensus.history
    T = consensus.iteration
    x_hist = local_hist["x_hist"][:T].detach().cpu()
    r_hist = local_hist["r_hist"][:T].detach().cpu()
    s_hist = local_hist["s_hist"][:T].detach().cpu()
    w_hist = local_hist["w_hist"][:T].detach().cpu()
    phi_hist = local_hist["phi_hist"][:T].detach().cpu()
    neighbors_list = consensus.neighbors

    x_all = comm.gather(x_hist, root=0)
    r_all = comm.gather(r_hist, root=0)
    s_all = comm.gather(s_hist, root=0)
    w_all = comm.gather(w_hist, root=0)
    phi_all = comm.gather(phi_hist, root=0)
    nbrs_all = comm.gather(neighbors_list, root=0)
    T_all = comm.gather(T, root=0)

    if rank == 0:
        T_use = int(min(T_all))
        history = {
            "num_iters": T_use,
            "x_hist": [x[:T_use] for x in x_all],
            "r_hist": [r[:T_use] for r in r_all],
            "s_hist": [s[:T_use] for s in s_all],
            "w_hist": [w[:T_use] for w in w_all],
            "phi_hist": [p[:T_use] for p in phi_all],
            "neighbors": nbrs_all,
            "labels": labels,
        }

        # Average residual histories over honest ranks only
        honest_indices = [i for i in range(size) if i not in adversarial_nodes]
        r_hist_honest_mean = torch.stack(
            [history["r_hist"][i] for i in honest_indices], dim=0
        ).mean(dim=0).tolist()
        s_hist_honest_mean = torch.stack(
            [history["s_hist"][i] for i in honest_indices], dim=0
        ).mean(dim=0).tolist()

        if mu_stack is not None:
            edge_weights = [w[-1].detach().cpu() for w in w_all]
            bcm_pred_payload = {
                "pred_locs": torch.tensor(pred_locs_np, dtype=dtype),
                "y_true": y_true.detach().cpu(),
                "mu_pred": mu_stack.detach().cpu(),
                "var_pred": var_stack.detach().cpu(),
                "mu_bcm": mu_cons_stack.detach().cpu(),
                "var_bcm": var_cons_stack.detach().cpu(),
                "adversarial_nodes": sorted(adversarial_nodes),
                "x_test": test_x.detach().cpu(),
                "y_test": test_y.detach().cpu(),
                "edge_weights": edge_weights,
                "neighbors": neighbors_list,
            }
            torch.save(bcm_pred_payload, out_dir / "bcm_predictions.pt")

        config_payload = {
            "dataset": str(dataset_path),
            "field_label": field_label,
            "seed": args.seed,
            "derived_seed": derived_seed,
            "n_samples_per_rank": args.n_samples_per_rank,
            "num_adversaries": args.num_adversaries,
            "dist_method": args.dist_method,
            "local_iters": args.local_iters,
            "local_lr": args.local_lr,
            "admm_iters": args.admm_iters,
            "rho": args.rho,
            "kappa": args.kappa,
            "obs_noise": args.obs_noise,
            "adversary_noise_std": args.adversary_noise_std,
            "adversary_white_noise": args.adversary_white_noise,
            "world_size": size,
            "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "run_directory": str(out_dir),
        }
        save_json(out_dir / "config.json", config_payload)

        metrics_payload = {
            "rmse_local_mean": rmse_mean,
            "rmse_consensus_mean": rmse2_mean,
            # BCM prediction quality
            "bcm_rmse_mean": bcm_rmse_mean,
            "bcm_rmse_std": bcm_rmse_std,
            "bcm_nrmse_range": bcm_nrmse_range,       # RMSE / (y_max - y_min)
            "bcm_nrmse_std_norm": bcm_nrmse_std_norm, # RMSE / std(y_true)
            "bcm_nlpd_mean": bcm_nlpd_mean,           # -mean log N(y|mu,var)
            "bcm_nlpd_std": bcm_nlpd_std,
            "bcm_nrmse_honest": bcm_nrmse_honest,     # honest agents only
            "bcm_nlpd_honest": bcm_nlpd_honest,
            "y_range": y_range,
            "y_std": y_std_val,
            "initial_params": init_params,
            "final_params": final_params,
            "adversarial_nodes": sorted(adversarial_nodes),
            "bcm_pred_locs": int(args.num_pred_locs),
            "r_hist_honest_mean": r_hist_honest_mean,
            "s_hist_honest_mean": s_hist_honest_mean,
        }
        save_json(out_dir / "metrics.json", metrics_payload)

        history_file = out_dir / "history.pt"
        torch.save(history, history_file)
        save_json(
            out_dir / "history_summary.json",
            {
                "num_iters": history["num_iters"],
                "neighbors": history["neighbors"],
                "labels": history["labels"],
            },
        )

        if not args.no_plots:
            plot_dir = out_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_admm_results(
                history,
                out_dir=str(plot_dir),
                expectedValues=None,
                top_k=3,
                adversaries=adversarial_nodes,
                x_dim_labels=labels,
            )


if __name__ == "__main__":
    main()