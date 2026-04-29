# agent/agent.py
from __future__ import annotations
import gpytorch
import torch
from typing import Any, List, Tuple
from mpi4py import MPI
import numpy as np


from dataclasses import dataclass, replace

from utils.FieldClass import BaseFieldClass
from planners.PRM import PRMQueryParameters, PRMQueryResult, PRMRoadmap
from planners.RRT import RRTParameters, RRTPlanner

class MPITags:
    """Common MPI tags for reuse across the codebase."""

    X_UPDATE = 100
    Y_UPDATE = 101
    Z_UPDATE = 102
    CONSENSUS = 110
    CONTROL = 200
    DAC_UPDATE = 210
    DAC_MU = 211
    DAC_BETA = 212


def _np(x: torch.Tensor) -> Any:
    if x.is_cuda:
        raise RuntimeError("This implementation assumes CPU tensors for mpi4py buffers.")
    return x.detach().numpy()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        D = train_x.size(-1)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=D)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )
    
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
    def __init__(
        self,
        rank: int,
        neighbors: List[int],
        device: torch.device,
        dtype: torch.dtype,
        field: BaseFieldClass,
        start: tuple[float, float],
        comms_range: float = 100.0, #km
        I_am_adversary: bool = False,
        attack_type: str = "add_noise",
        noise_gain: float = 1.0,
        white_noise_std: float = 0.05,
        bias_scale: float = 1.0,
        rrt_params: RRTParameters | None = None,
        prm_roadmap: PRMRoadmap | None = None,
        planner_type: str = "rrt",
        max_grade: float = 15.0,
    ):
        self.rank = rank
        self.neighbors = neighbors

        # GP consensus state
        self.device = device
        self.dtype = dtype
        self.I_am_adversary = I_am_adversary
        self.attack_type = attack_type
        self.noise_gain = noise_gain
        self.white_noise_std = white_noise_std
        self.bias_scale = bias_scale
        self.edge_trust = torch.ones(len(neighbors), dtype=self.dtype)
        self._last_recv: dict[tuple[int, int], torch.Tensor] = {}

        # Planner-specific state
        self.field = field
        self.max_grade = max_grade
        self.comms_range = float(comms_range)
        self.position: np.ndarray = np.array(start, dtype=float)
        self.last_plan_start: np.ndarray = self.position.copy()
        self.last_goal: np.ndarray | None = None
        self.rrt_params = rrt_params if rrt_params is not None else RRTParameters()
        self.prm_roadmap = prm_roadmap
        self.planner_type = planner_type.lower()
        self.space = AgentPlannerSpace(field=field, max_grade=max_grade)
        self.planner: RRTPlanner | PRMQueryResult | None = None
        self.path: list[np.ndarray] | None = None

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

    def adversarial_admm_y_update(self, y_ij: torch.Tensor, attack_type: str = "add_noise") -> torch.Tensor:
        """
        Adversarial modification of the ADMM y-update values.

        Args:
            y_ij: Tensor of shape (n_neighbors, d) containing the honest y values to be sent to neighbors.

        Returns:
            Tensor of shape (n_neighbors, d) containing the potentially modified y values.
        """
        if not self.I_am_adversary:
            return y_ij
        
        if attack_type == "add_noise":

            eps = 1e-12

            # optional multiplicative bias/attenuation
            y_adv = y_ij * self.noise_gain

            # robust per-dimension scale across neighbors
            s = y_adv.abs().median(dim=0).values.clamp_min(eps)  # (d,)

            # relative noise: std per dimension proportional to typical magnitude
            noise = torch.randn_like(y_adv) * (self.white_noise_std * s[None, :])

            return y_adv + noise
        
        elif attack_type == "stubborn_bias":
            if not hasattr(self, "_frozen_y"):
                self._frozen_y = self.bias_scale * y_ij.detach().clone()

            return self._frozen_y
        
    def flush_cache(self):
        """Force gpytorch to recompute cached training data kernel/Cholesky."""
        self.model.train()
        self.likelihood.train()
        with torch.no_grad():
            self.model(self.train_x)

    def adversarial_dac_x_update(self, x_ij: torch.Tensor) -> torch.Tensor:
        """
        Adversarial modification of DAC/BCM x-update values.

        Returns uniform samples using data-field bounds (per-dimension) when available.
        """
        if not self.I_am_adversary:
            return x_ij

        data_min = getattr(self, "data_min", None)
        data_max = getattr(self, "data_max", None)
        if data_min is None or data_max is None:
            return x_ij

        data_min = torch.as_tensor(data_min, device=x_ij.device, dtype=x_ij.dtype)
        data_max = torch.as_tensor(data_max, device=x_ij.device, dtype=x_ij.dtype)

        if data_min.numel() == 1 or x_ij.shape[-1] != data_min.numel():
            min_val = data_min.min().item()
            max_val = data_max.max().item()
            if max_val <= min_val:
                return x_ij
            return (max_val - min_val) * torch.rand_like(x_ij) + min_val

        view_shape = [1] * (x_ij.ndim - 1) + [data_min.numel()]
        span = (data_max - data_min).view(*view_shape)
        base = data_min.view(*view_shape)
        return span * torch.rand_like(x_ij) + base

    def send_recv_tensor(
        self,
        comm: MPI.Comm,
        send_tensor: torch.Tensor,
        tag: int = MPITags.X_UPDATE,
        debug: bool = False,
        debug_prefix: str = "",
        timeout_sec: float | None = None,
        reuse_last: bool = True,
    ) -> torch.Tensor:
        n_neighbors = len(self.neighbors)
        if n_neighbors == 0:
            return torch.empty((0, *send_tensor.shape[1:]), device=self.device, dtype=self.dtype)

        if send_tensor.shape[0] != n_neighbors:
            raise ValueError(
                f"send_tensor.shape[0] must match number of neighbors ({n_neighbors}), "
                f"got {send_tensor.shape[0]}"
            )

        x_send_cpu = send_tensor.detach().cpu().contiguous()
        payload_shape = tuple(x_send_cpu.shape[1:])
        x_recv_cpu = torch.empty((n_neighbors, *payload_shape), device="cpu", dtype=self.dtype)

        if debug:
            prefix = f"{debug_prefix} " if debug_prefix else ""
            print(
                f"{prefix}rank={self.rank} send_recv_tensor tag={tag} neighbors={self.neighbors} payload_shape={payload_shape}",
                flush=True,
            )

        if timeout_sec is None:
            for t, j in enumerate(self.neighbors):
                if debug:
                    prefix = f"{debug_prefix} " if debug_prefix else ""
                    print(f"{prefix}rank={self.rank} -> neighbor={j} sendrecv start", flush=True)
                comm.Sendrecv(
                    _np(x_send_cpu[t]),
                    dest=j,
                    sendtag=tag,
                    recvbuf=_np(x_recv_cpu[t]),
                    source=j,
                    recvtag=tag,
                )
                if debug:
                    prefix = f"{debug_prefix} " if debug_prefix else ""
                    print(f"{prefix}rank={self.rank} <- neighbor={j} sendrecv done", flush=True)

            for t, j in enumerate(self.neighbors):
                self._last_recv[(tag, j)] = x_recv_cpu[t].clone()
            return x_recv_cpu.to(device=self.device)

        recv_reqs: list[MPI.Request] = []
        send_reqs: list[MPI.Request] = []

        for t, j in enumerate(self.neighbors):
            if debug:
                prefix = f"{debug_prefix} " if debug_prefix else ""
                print(f"{prefix}rank={self.rank} -> neighbor={j} isend/irecv start", flush=True)
            recv_reqs.append(comm.Irecv(_np(x_recv_cpu[t]), source=j, tag=tag))
            send_reqs.append(comm.Isend(_np(x_send_cpu[t]), dest=j, tag=tag))

        completed = [False] * n_neighbors
        t_start = MPI.Wtime()
        while True:
            all_done = True
            for idx, req in enumerate(recv_reqs):
                if not completed[idx]:
                    flag = req.Test()
                    if flag:
                        completed[idx] = True
                    else:
                        all_done = False
            if all_done:
                break
            if MPI.Wtime() - t_start >= timeout_sec:
                break

        for idx, j in enumerate(self.neighbors):
            if not completed[idx]:
                try:
                    recv_reqs[idx].Cancel()
                    recv_reqs[idx].Free()
                except Exception:
                    pass

                if reuse_last and (tag, j) in self._last_recv:
                    x_recv_cpu[idx].copy_(self._last_recv[(tag, j)])
                else:
                    x_recv_cpu[idx].copy_(x_send_cpu[idx])

                if debug:
                    prefix = f"{debug_prefix} " if debug_prefix else ""
                    print(
                        f"{prefix}rank={self.rank} <- neighbor={j} timeout; reused last",
                        flush=True,
                    )
            else:
                self._last_recv[(tag, j)] = x_recv_cpu[idx].clone()

        if send_reqs:
            MPI.Request.Waitall(send_reqs)

        return x_recv_cpu.to(device=self.device)


class GPAgent(Agent):
    """
    Encapsulates local data + local GP model + local ops.
    Must NOT call comm.allgather/allreduce/bcast/etc.
    """
    def __init__(
        self,
        rank: int,
        neighbors: List[int],
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        I_am_adversary: bool = False,
        noise_gain: float = 1.0,
        white_noise_std: float = 0.05,
        attack_type: str = "add_noise",
        bias_scale: float = 1.0,
    ):
        super().__init__(
            rank,
            neighbors,
            device,
            dtype,
            I_am_adversary,
            noise_gain=noise_gain,
            white_noise_std=white_noise_std,
        )

        self.train_x = train_x.to(device=device, dtype=dtype)
        self.train_y = train_y.to(device=device, dtype=dtype)

        self.data_min = self.train_x.min(dim=0).values.detach().cpu()
        self.data_max = self.train_x.max(dim=0).values.detach().cpu()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(device=device, dtype=dtype)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    # ----- local training -----
    def train_local(self, iters: int, lr: float) -> float:
        self.model.train()
        self.likelihood.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        last_loss = None
        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            out = self.model(self.train_x)
            loss = -self.mll(out, self.train_y)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().item())
        return float(last_loss if last_loss is not None else 0.0)

    # ----- consensus variable definition (raw space) -----
    def theta_pack(self) -> Tuple[torch.Tensor, List[str]]:
        raw_ls = self.model.covar_module.base_kernel.raw_lengthscale.detach().reshape(-1)
        raw_os = self.model.covar_module.raw_outputscale.detach().reshape(-1)
        raw_n  = self.likelihood.raw_noise.detach().reshape(-1)

        labels = []
        D = raw_ls.numel()
        for d in range(D):
            labels.append(f"lengthscale_dim{d}")
        labels.append("outputscale")
        labels.append("noise")

        return (torch.cat([raw_ls, raw_os, raw_n], dim=0), labels)

    def theta_unpack(self, theta: torch.Tensor) -> None:
        D = self.model.covar_module.base_kernel.raw_lengthscale.numel()
        with torch.no_grad():
            self.model.covar_module.base_kernel.raw_lengthscale.copy_(
                theta[:D].view_as(self.model.covar_module.base_kernel.raw_lengthscale)
            )
            self.model.covar_module.raw_outputscale.copy_(
                theta[D:D+1].view_as(self.model.covar_module.raw_outputscale)
            )
            self.likelihood.raw_noise.copy_(
                theta[D+1:D+2].view_as(self.likelihood.raw_noise)
            )

    def theta_raw_to_actual(self, theta_raw: torch.Tensor):
        # D = number of raw lengthscale params (ARD dims)
        D = self.model.covar_module.base_kernel.raw_lengthscale.numel()

        raw_ls = theta_raw[:D]
        raw_os = theta_raw[D:D+1]
        raw_n  = theta_raw[D+1:D+2]

        ls = self.model.covar_module.base_kernel.raw_lengthscale_constraint.transform(raw_ls)
        os = self.model.covar_module.raw_outputscale_constraint.transform(raw_os)

        # GaussianLikelihood noise constraint lives here in many gpytorch versions:
        n  = self.likelihood.noise_covar.raw_noise_constraint.transform(raw_n)

        return ls, os, n
    
    # ----- local objective gradient for TE-ADMM -----
    def local_nll_and_grad(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        import torch
        if not torch.isfinite(theta).all():
            print(f"[GPAgent] ERROR: NaN or Inf detected in theta input to local_nll_and_grad. theta={theta}")
            raise ValueError("NaN or Inf in theta input to local_nll_and_grad.")

        RAW_CLAMP = 10.0
        theta = theta.clamp(-RAW_CLAMP, RAW_CLAMP)
        self.theta_unpack(theta)

        self.model.train()
        self.likelihood.train()
        self.model.zero_grad(set_to_none=True)
        self.likelihood.zero_grad(set_to_none=True)

        out = self.model(self.train_x)
        nll = -self.mll(out, self.train_y)
        nll.backward()

        raw_ls_g = self.model.covar_module.base_kernel.raw_lengthscale.grad.reshape(-1)
        raw_os_g = self.model.covar_module.raw_outputscale.grad.reshape(-1)
        raw_n_g  = self.likelihood.raw_noise.grad.reshape(-1)
        grad = torch.cat([raw_ls_g, raw_os_g, raw_n_g], dim=0)

        return nll.detach(), grad.detach()

    # ----- local eval -----
    def eval_rmse(self, gp_field: Any, num_test: int = 200) -> float:
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            D = self.train_x.size(-1)
            mins = self.train_x.min(dim=0).values
            maxs = self.train_x.max(dim=0).values
            test_x = mins + (maxs - mins) * torch.rand(num_test, D, device=self.device, dtype=self.dtype)

            pred = self.likelihood(self.model(test_x))
            test_y_np = gp_field.sample_field(test_x.cpu().numpy())
            test_y = torch.tensor(test_y_np, device=self.device, dtype=self.dtype)
            rmse = torch.mean((pred.mean - test_y) ** 2).sqrt().item()
        return float(rmse)