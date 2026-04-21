"""
vanilla_PINN_2D_rect.py
=======================
Physics-Informed Neural Network (PINN) for 2-D heterogeneous consolidation,
compatible with the QAPILM forward/inverse framework.

PDE
---
    u_t = C(x,z) * u_xx  +  C(x,z)*Rcv * u_zz
        + (∂C/∂x) * u_x  +  Rcv*(∂C/∂z) * u_z

where u(x,z,t) is excess pore-water pressure and C(x,z) > 0 is the
coefficient of consolidation (to be recovered).  Rcv is a constant
anisotropy ratio (default 1 = isotropic).

Two separate networks
---------------------
  UNet(x, z, t)  →  u           (full solution trajectory)
  CNet(x, z)     →  C > 0       (coefficient field, softplus output)

Key features
------------
  * Observation generation via the QAPILM FD forward solver
    (or a built-in minimal FD fallback when QAPILM is unavailable).
    Supports configurable measurement density (grid / random) and
    explicit location overrides.

  * Loss-history lists are initialised once in __init__ and appended
    to during training — they are never reset inside the training loop.

  * Boundary conditions are consistent with the QAPILM FD scheme:
      flag = 0  →  Dirichlet  u = 0
      flag = 1  →  Neumann    ∂u/∂n = 0

  * Timestamped model save  ``pinn_YYYYMMDD_HHMMSS.pt``  with a JSON
    metadata sidecar for later loading / plotting.

  * Built-in triple-panel comparison plot (Real / QAPILM / PINN)
    matching the format of RectangularQAPILM.triplot2D().

  * Standalone compare_and_plot() helper so a single import produces
    publication-quality comparison figures with minimal boilerplate.

Usage
-----
    # Quick start — generate observations, train, save, and plot
    python vanilla_PINN_2D_rect.py

    # Load a previously saved model and only produce comparison plots
    python vanilla_PINN_2D_rect.py --load pinn_models/pinn_20250101_120000.pt

    # Override output directory
    python vanilla_PINN_2D_rect.py --outdir ./my_pinn_models

Dependencies
------------
    pip install torch numpy scipy matplotlib seaborn pyyaml
    # qapilm_rect.py must be in the same directory for QAPILM integration
    # (optional — a minimal built-in FD solver is used as fallback)

Notes on computational cost
----------------------------
The default training uses standard autograd for second-order derivatives
(u_xx, u_zz).  For large networks or many collocation points this can be
slow.  A mixed formulation that replaces second derivatives with first
derivatives of auxiliary networks is 2–5 × faster; it can be enabled by
setting  net_cfg.use_mixed_formulation = True  (see PINNNetConfig).
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys
import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Scientific stack
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import diags as sp_diags

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    sys.exit(
        "PyTorch is required.  Install with:\n"
        "    pip install torch\n"
        "See https://pytorch.org for platform-specific instructions."
    )

# ---------------------------------------------------------------------------
# Optional QAPILM integration
# ---------------------------------------------------------------------------
_QAPILM_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from qapilm_rect import RectangularQAPILM  # noqa: F401
    _QAPILM_AVAILABLE = True
except ImportError:
    pass


# ===========================================================================
# CONFIG DATACLASSES
# ===========================================================================

@dataclass
class PINNGeomConfig:
    """Geometric and physical domain parameters.
    These must match the QAPILM ProblemConfig when QAPILM integration is used.
    """
    Lh: float = 10.0           # Horizontal domain length  (m)
    Lv: float = 5.0            # Vertical domain length    (m)
    Lt: float = 2.0            # Total simulation time     (yr)
    spatial_reso: tuple = (0.2, 0.2)  # (dx, dz) grid spacing for FD fallback
    u0: float = 1.0            # Uniform initial excess pore pressure
    Rcv: float = 1.0           # Anisotropy ratio  Ch/Cv   (1 = isotropic)
    # Boundary flags — order: [top, bottom, left, right]
    #   0 = Dirichlet  u = 0          (drainage boundary)
    #   1 = Neumann    ∂u/∂n = 0      (no-flux / impermeable boundary)
    bcs: tuple = (0, 1, 0, 1)


@dataclass
class PINNNetConfig:
    """Neural-network architecture."""
    u_hidden_layers: int = 4      # Depth of UNet
    u_hidden_width: int = 64      # Width of UNet
    c_hidden_layers: int = 3      # Depth of CNet
    c_hidden_width: int = 32      # Width of CNet
    activation: str = "tanh"      # "tanh" | "sin" | "gelu"
    # When True the PDE residual uses a mixed formulation: auxiliary
    # networks approximate u_x and u_z so only first-order derivatives
    # are needed in the PDE, reducing autograd cost by 2–5×.
    use_mixed_formulation: bool = False


@dataclass
class PINNTrainConfig:
    """Training hyper-parameters."""
    epochs: int = 5000
    lr: float = 1e-3
    lr_decay: float = 0.95         # Multiplicative LR decay factor
    lr_decay_every: int = 1000     # Apply decay every N epochs
    n_colloc_pde: int = 4000       # Collocation points per epoch (PDE)
    n_bc: int = 400                # Boundary points per epoch
    n_ic: int = 400                # IC points per epoch
    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_ic: float = 10.0
    w_data_u: float = 100.0        # Weight for u observation data loss
    w_data_c: float = 100.0        # Weight for C observation data loss
    grad_clip: float = 1.0         # Gradient clipping max-norm (0 = disabled)
    print_every: int = 100         # Print diagnostics every N epochs


@dataclass
class PINNObsConfig:
    """Observation (measurement) configuration for generating training data."""
    # Density mode when auto-sampling observation locations:
    #   "grid"   — regular sub-grid at stride ~1/sqrt(density)
    #   "random" — uniformly random spatial locations
    density_mode: str = "grid"
    # Fraction of spatial grid points used for u observations  (0 < f ≤ 1)
    u_density: float = 0.05
    # Fraction of spatial grid points used for C observations  (0 < f ≤ 1)
    c_density: float = 0.05
    # Explicit spatial index pairs [[row, col], …] for u observations.
    # When non-empty, density_mode / u_density are ignored for u.
    u_locations: Optional[List] = None
    # Explicit spatial index pairs for C observations (same convention).
    c_locations: Optional[List] = None
    # Time-step indices at which u is observed.
    # None → n_time_obs evenly-spaced indices are chosen automatically.
    time_indices: Optional[List] = None
    n_time_obs: int = 5            # Used only when time_indices is None


@dataclass
class PINNSaveConfig:
    """Model saving / loading options."""
    model_dir: str = "./pinn_models"
    save_metadata: bool = True     # Write JSON sidecar with training metadata


# ===========================================================================
# NEURAL NETWORKS
# ===========================================================================

class _SinActivation(nn.Module):
    """Sinusoidal activation function (useful for periodic solutions)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def _make_activation(name: str) -> nn.Module:
    activations = {"tanh": nn.Tanh(), "sin": _SinActivation(), "gelu": nn.GELU()}
    if name not in activations:
        raise ValueError(f"Unknown activation {name!r}.  Choose from {list(activations)}")
    return activations[name]


def _build_mlp(in_dim: int, out_dim: int, n_hidden: int,
               width: int, activation: str) -> nn.Sequential:
    act = _make_activation(activation)
    layers: list = [nn.Linear(in_dim, width), act]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(width, width), _make_activation(activation)]
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """Maps (x̄, z̄, t̄) ∈ [0,1]³ → u  (excess pore-water pressure)."""

    def __init__(self, hidden_layers: int = 4, hidden_width: int = 64,
                 activation: str = "tanh"):
        super().__init__()
        self.net = _build_mlp(3, 1, hidden_layers, hidden_width, activation)

    def forward(self, xzt: torch.Tensor) -> torch.Tensor:
        """xzt : (N, 3) tensor of normalised [x̄, z̄, t̄] → (N, 1) u"""
        return self.net(xzt)


class CNet(nn.Module):
    """Maps (x̄, z̄) ∈ [0,1]² → C(x,z) > 0.
    Softplus is applied to the raw output to guarantee strict positivity.
    """

    def __init__(self, hidden_layers: int = 3, hidden_width: int = 32,
                 activation: str = "tanh"):
        super().__init__()
        self.net = _build_mlp(2, 1, hidden_layers, hidden_width, activation)

    def forward(self, xz: torch.Tensor) -> torch.Tensor:
        """xz : (N, 2) tensor of normalised [x̄, z̄] → (N, 1) C > 0"""
        return torch.nn.functional.softplus(self.net(xz))


# Auxiliary networks for the mixed formulation (p ≈ u_x, q ≈ u_z)
class _AuxNet(nn.Module):
    """Small auxiliary network for the mixed-formulation PINN."""
    def __init__(self, hidden_layers: int = 3, hidden_width: int = 32,
                 activation: str = "tanh"):
        super().__init__()
        self.net = _build_mlp(3, 1, hidden_layers, hidden_width, activation)

    def forward(self, xzt: torch.Tensor) -> torch.Tensor:
        return self.net(xzt)


# ===========================================================================
# MINIMAL FINITE-DIFFERENCE FORWARD SOLVER  (standalone fallback)
# ===========================================================================

def minimal_fd_forward(
    chm: np.ndarray,
    dt: float,
    numt: int,
    u0_val: float,
    Rcv: float,
    dx: float,
    dz: float,
    top: int,
    bot: int,
    left: int,
    right: int,
) -> np.ndarray:
    """Explicit FD forward solver for 2-D consolidation.

    Solves:  u_t = C * u_xx + (C*Rcv) * u_zz + (∂C/∂x) * u_x + Rcv*(∂C/∂z) * u_z

    This replicates the operator structure of RectangularQAPILM.forward_solver()
    using only NumPy / SciPy so it can be used without a QAPILM instance.

    Parameters
    ----------
    chm     : (nz+1, nx+1) array of C(x,z) values (ground-truth)
    dt      : time step  (must satisfy the CFL stability condition)
    numt    : number of time steps
    u0_val  : scalar initial excess pore pressure
    Rcv     : anisotropy ratio
    dx, dz  : spatial grid spacings
    top, bot, left, right : boundary flags  (0 = Dirichlet, 1 = Neumann)

    Returns
    -------
    utens : (numt+1, nz+1, nx+1) pressure tensor
    """
    nz, nx = chm.shape
    chv = chm * Rcv

    def _tridiag(n: int, scale: float) -> object:
        return sp_diags(
            [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)],
            [0, 1, -1], shape=(n, n), format="csr",
        ) * scale

    def _first_diff(n: int, scale: float) -> object:
        return sp_diags(
            [np.zeros(n), np.ones(n - 1), -np.ones(n - 1)],
            [0, 1, -1], shape=(n, n), format="csr",
        ) * scale

    a1 = _tridiag(nx, dt / dx ** 2)
    a2 = _tridiag(nz, dt / dz ** 2)
    b1 = _first_diff(nx, dt / (2 * dx))
    b2 = _first_diff(nz, dt / (2 * dz))

    # Numerical C-gradient for advection terms
    d1 = np.zeros_like(chm)
    d2 = np.zeros_like(chm)
    d1[:, 1:-1] = (chm[:, 2:] - chm[:, :-2]) / (2 * dx)
    d2[1:-1, :] = (chm[2:, :] - chm[:-2, :]) / (2 * dz)

    utens = np.zeros((numt + 1, nz, nx), dtype=np.float64)
    utens[0] = u0_val

    for i in range(numt):
        cu = utens[i]
        unew = (
            cu
            + chm  * (a1.T.dot(cu.T)).T
            + chv  * a2.dot(cu)
            + d1   * (b1.T.dot(cu.T)).T
            + Rcv * d2 * b2.dot(cu)
        )
        unew[:, 0]  = left  * unew[:, 1]
        unew[:, -1] = right * unew[:, -2]
        unew[0, :]  = top   * unew[1, :]
        unew[-1, :] = bot   * unew[-2, :]
        utens[i + 1] = unew

    return utens


# ===========================================================================
# MAIN PINN CLASS
# ===========================================================================

class PINN2DConsolidation:
    """Physics-Informed Neural Network for 2-D heterogeneous consolidation.

    Two networks are trained jointly:
      * UNet(x̄, z̄, t̄) → u  approximates the pore-pressure field
      * CNet(x̄, z̄)     → C  approximates the consolidation coefficient

    Training data is generated from the FD forward solver (either via a
    RectangularQAPILM instance or the built-in minimal_fd_forward).

    Parameters
    ----------
    geom      : PINNGeomConfig
    net_cfg   : PINNNetConfig
    train_cfg : PINNTrainConfig
    obs_cfg   : PINNObsConfig
    save_cfg  : PINNSaveConfig
    device    : "cpu" | "cuda" | "mps"  (auto-detected if None)
    """

    def __init__(
        self,
        geom: Optional[PINNGeomConfig] = None,
        net_cfg: Optional[PINNNetConfig] = None,
        train_cfg: Optional[PINNTrainConfig] = None,
        obs_cfg: Optional[PINNObsConfig] = None,
        save_cfg: Optional[PINNSaveConfig] = None,
        device: Optional[str] = None,
    ):
        self.geom      = geom      or PINNGeomConfig()
        self.net_cfg   = net_cfg   or PINNNetConfig()
        self.train_cfg = train_cfg or PINNTrainConfig()
        self.obs_cfg   = obs_cfg   or PINNObsConfig()
        self.save_cfg  = save_cfg  or PINNSaveConfig()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print(f"[PINN] Device: {self.device}")

        self._build_networks()

        # Loss-history lists — initialised here, never reset during training.
        self.loss_pde_history:   List[float] = []
        self.loss_bc_history:    List[float] = []
        self.loss_ic_history:    List[float] = []
        self.loss_data_history:  List[float] = []
        self.loss_total_history: List[float] = []

        self._obs_data: Optional[dict] = None
        self._save_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_networks(self) -> None:
        nc = self.net_cfg
        self.u_net = UNet(
            hidden_layers=nc.u_hidden_layers,
            hidden_width=nc.u_hidden_width,
            activation=nc.activation,
        ).to(self.device)
        self.c_net = CNet(
            hidden_layers=nc.c_hidden_layers,
            hidden_width=nc.c_hidden_width,
            activation=nc.activation,
        ).to(self.device)
        # Auxiliary networks for mixed formulation (lazy init)
        self._p_net: Optional[_AuxNet] = None
        self._q_net: Optional[_AuxNet] = None
        if nc.use_mixed_formulation:
            self._p_net = _AuxNet(
                hidden_layers=nc.u_hidden_layers,
                hidden_width=nc.u_hidden_width,
                activation=nc.activation,
            ).to(self.device)
            self._q_net = _AuxNet(
                hidden_layers=nc.u_hidden_layers,
                hidden_width=nc.u_hidden_width,
                activation=nc.activation,
            ).to(self.device)

    def _setup_optimizer(self) -> None:
        tc = self.train_cfg
        params = list(self.u_net.parameters()) + list(self.c_net.parameters())
        if self._p_net is not None:
            params += list(self._p_net.parameters()) + list(self._q_net.parameters())
        self.optimizer = optim.Adam(params, lr=tc.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=tc.lr_decay_every, gamma=tc.lr_decay
        )

    # ------------------------------------------------------------------
    # Coordinate normalisation helpers
    # ------------------------------------------------------------------

    def _norm(self, x_arr: np.ndarray, L: float) -> np.ndarray:
        return x_arr / L

    def _to_tensor(self, arr, requires_grad: bool = False) -> torch.Tensor:
        return torch.tensor(
            np.asarray(arr, dtype=np.float32),
            device=self.device,
            requires_grad=requires_grad,
        )

    # ------------------------------------------------------------------
    # Observation generation
    # ------------------------------------------------------------------

    def generate_observations_from_qapilm(
        self, qapilm_solver: "RectangularQAPILM"
    ) -> dict:
        """Generate training observations by running the QAPILM FD forward solver.

        Parameters
        ----------
        qapilm_solver : RectangularQAPILM  (already initialised with ground-truth C)

        Returns
        -------
        obs : dict  with keys  "u_pts", "u_vals", "c_pts", "c_vals"
        """
        g = self.geom
        top, bot, left, right = g.bcs
        # forward_solver accepts both a scalar and a matrix u0; pass the
        # uniform-IC matrix so utens[0] is set correctly for all grid points.
        u0_scalar = float(g.u0)
        u0_mat = np.ones((qapilm_solver.numv + 1, qapilm_solver.numh + 1)) * u0_scalar
        numt = qapilm_solver.ilim

        print(f"[PINN] Running QAPILM forward solver  (numt={numt}) …")
        utens, _ = qapilm_solver.forward_solver(
            qapilm_solver.chm, numt, u0_mat, top, bot, left, right
        )

        dx = qapilm_solver.dx
        dz = qapilm_solver.dz
        dt = qapilm_solver.dt

        obs = self._sample_obs_points(
            utens, qapilm_solver.chm, dt, numt,
            dx, dz, qapilm_solver.numh, qapilm_solver.numv,
        )
        self._obs_data = obs
        return obs

    def generate_observations_from_fd(
        self,
        chm: np.ndarray,
        dx: Optional[float] = None,
        dz: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> dict:
        """Generate training observations using the built-in minimal FD solver.

        Parameters
        ----------
        chm : (nz+1, nx+1) ground-truth C field
        dx  : grid spacing in x  (default: Lh / (nx))
        dz  : grid spacing in z  (default: Lv / (nz))
        dt  : time step  (auto: CFL-based if None)

        Returns
        -------
        obs : dict  with keys  "u_pts", "u_vals", "c_pts", "c_vals"
        """
        g = self.geom
        nz, nx = chm.shape[0] - 1, chm.shape[1] - 1
        dx = dx if dx is not None else g.Lh / nx
        dz = dz if dz is not None else g.Lv / nz
        if dt is None:
            alpha = 0.05
            Cmax  = max(float(np.max(chm)), 1e-9)
            dt    = min(alpha * dx ** 2 / Cmax,
                        alpha * dz ** 2 / (Cmax * max(g.Rcv, 1e-9)))
        numt = max(1, int(g.Lt / dt))

        top, bot, left, right = g.bcs
        print(f"[PINN] Minimal FD solver  (numt={numt},  dt={dt:.5f}) …")
        utens = minimal_fd_forward(
            chm, dt, numt, float(g.u0), g.Rcv, dx, dz, top, bot, left, right
        )

        obs = self._sample_obs_points(utens, chm, dt, numt, dx, dz, nx, nz)
        self._obs_data = obs
        return obs

    def _sample_obs_points(
        self,
        utens: np.ndarray,
        chm: np.ndarray,
        dt: float,
        numt: int,
        dx: float,
        dz: float,
        numh: int,
        numv: int,
    ) -> dict:
        """Sample observation points from a FD-solver output.

        Returns
        -------
        dict with:
            u_pts  : (N_u, 3)  physical (x, z, t) measurement locations
            u_vals : (N_u,)    measured u values
            c_pts  : (N_c, 2)  physical (x, z) C measurement locations
            c_vals : (N_c,)    measured C values
        """
        obc     = self.obs_cfg
        row_idx = np.arange(numv + 1)
        col_idx = np.arange(numh + 1)

        # ---- C observation spatial locations ----
        if obc.c_locations and len(obc.c_locations) > 0:
            c_idx = [tuple(pt) for pt in obc.c_locations]
        else:
            n_c = max(1, int(round((numv + 1) * (numh + 1) * obc.c_density)))
            if obc.density_mode == "grid":
                step = max(1, int(round(1.0 / (obc.c_density ** 0.5 + 1e-12))))
                ri   = row_idx[::step]
                ci   = col_idx[::step]
                c_idx = [(int(r), int(c)) for r in ri for c in ci]
            else:
                rs    = np.random.randint(0, numv + 1, n_c)
                cs    = np.random.randint(0, numh + 1, n_c)
                c_idx = [(int(r), int(c)) for r, c in zip(rs, cs)]

        # (x, z) physical coords  — note: row index → z, col index → x
        c_pts_xz = np.array([[c * dx, r * dz] for r, c in c_idx], dtype=np.float32)
        c_vals   = np.array([chm[r, c] for r, c in c_idx], dtype=np.float32)

        # ---- Time indices for u observations ----
        if obc.time_indices and len(obc.time_indices) > 0:
            t_idx = [min(int(i), numt) for i in obc.time_indices]
        else:
            t_idx = list(
                np.linspace(0, numt, obc.n_time_obs + 2, dtype=int)[1:-1]
            )

        # ---- u observation spatial locations ----
        if obc.u_locations and len(obc.u_locations) > 0:
            u_spatial = [tuple(pt) for pt in obc.u_locations]
        else:
            n_u = max(1, int(round((numv + 1) * (numh + 1) * obc.u_density)))
            if obc.density_mode == "grid":
                step = max(1, int(round(1.0 / (obc.u_density ** 0.5 + 1e-12))))
                ri   = row_idx[::step]
                ci   = col_idx[::step]
                u_spatial = [(int(r), int(c)) for r in ri for c in ci]
            else:
                rs        = np.random.randint(0, numv + 1, n_u)
                cs        = np.random.randint(0, numh + 1, n_u)
                u_spatial = [(int(r), int(c)) for r, c in zip(rs, cs)]

        u_pts_list: List = []
        u_vals_list: List = []
        for ti in t_idx:
            t_val = ti * dt
            for r, c in u_spatial:
                u_pts_list.append([c * dx, r * dz, t_val])
                u_vals_list.append(float(utens[ti, r, c]))

        u_pts  = np.array(u_pts_list,  dtype=np.float32)
        u_vals = np.array(u_vals_list, dtype=np.float32)

        print(
            f"[PINN] Observations sampled:  "
            f"u_pts={len(u_pts)},  C_pts={len(c_pts_xz)}"
        )
        return {
            "u_pts":  u_pts,
            "u_vals": u_vals,
            "c_pts":  c_pts_xz,
            "c_vals": c_vals,
        }

    # ------------------------------------------------------------------
    # PDE residual
    # ------------------------------------------------------------------

    def _u_and_derivs(self, xzt: torch.Tensor):
        """Compute u and all required partial derivatives at collocation points.

        Parameters
        ----------
        xzt : (N, 3) tensor [x̄, z̄, t̄]  normalised, requires_grad=True

        Returns
        -------
        u, u_t, u_x, u_z, u_xx, u_zz : each (N, 1)
        C, C_x, C_z                   : each (N, 1)
        """
        Lh, Lv, Lt = self.geom.Lh, self.geom.Lv, self.geom.Lt

        u = self.u_net(xzt)

        # First-order derivatives (normalised)
        g1 = torch.autograd.grad(
            u, xzt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        u_x_n, u_z_n, u_t_n = g1[:, 0:1], g1[:, 1:2], g1[:, 2:3]

        # Physical (chain-rule) first derivatives
        u_x = u_x_n / Lh
        u_z = u_z_n / Lv
        u_t = u_t_n / Lt

        # Second-order derivatives (normalised → physical)
        u_xx_n = torch.autograd.grad(
            u_x_n, xzt,
            grad_outputs=torch.ones_like(u_x_n),
            create_graph=True,
        )[0][:, 0:1]
        u_xx = u_xx_n / (Lh * Lh)

        u_zz_n = torch.autograd.grad(
            u_z_n, xzt,
            grad_outputs=torch.ones_like(u_z_n),
            create_graph=True,
        )[0][:, 1:2]
        u_zz = u_zz_n / (Lv * Lv)

        # C and its spatial derivatives — use a separate input tensor so
        # the C-net gradient computation is independent of xzt.
        xz = xzt[:, :2].detach().clone().requires_grad_(True)
        C  = self.c_net(xz)
        gC = torch.autograd.grad(
            C, xz,
            grad_outputs=torch.ones_like(C),
            create_graph=True,
        )[0]
        C_x = gC[:, 0:1] / Lh
        C_z = gC[:, 1:2] / Lv

        return u, u_t, u_x, u_z, u_xx, u_zz, C, C_x, C_z

    def _u_and_derivs_mixed(self, xzt: torch.Tensor):
        """Mixed-formulation variant: avoids second-order autograd.

        p_net ≈ u_x,  q_net ≈ u_z  (auxiliary networks).
        The PDE is reformulated using only first derivatives of p and q.
        Returns the same tuple as _u_and_derivs, plus aux_loss for
        the auxiliary constraints  p - u_x = 0  and  q - u_z = 0.
        """
        Lh, Lv, Lt = self.geom.Lh, self.geom.Lv, self.geom.Lt

        u = self.u_net(xzt)
        p = self._p_net(xzt)   # approximates u_x (physical)
        q = self._q_net(xzt)   # approximates u_z (physical)

        # First-order derivatives of u (for auxiliary constraints)
        g1 = torch.autograd.grad(
            u, xzt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        u_x_n, u_z_n, u_t_n = g1[:, 0:1], g1[:, 1:2], g1[:, 2:3]
        u_x = u_x_n / Lh
        u_z = u_z_n / Lv
        u_t = u_t_n / Lt

        # First derivatives of auxiliary networks
        p_requires = p
        p_grad = torch.autograd.grad(
            p_requires, xzt,
            grad_outputs=torch.ones_like(p_requires),
            create_graph=True,
        )[0]
        p_x = p_grad[:, 0:1] / Lh   # ≈ u_xx

        q_requires = q
        q_grad = torch.autograd.grad(
            q_requires, xzt,
            grad_outputs=torch.ones_like(q_requires),
            create_graph=True,
        )[0]
        q_z = q_grad[:, 1:2] / Lv   # ≈ u_zz

        # C
        xz = xzt[:, :2].detach().clone().requires_grad_(True)
        C  = self.c_net(xz)
        gC = torch.autograd.grad(
            C, xz,
            grad_outputs=torch.ones_like(C),
            create_graph=True,
        )[0]
        C_x = gC[:, 0:1] / Lh
        C_z = gC[:, 1:2] / Lv

        # Auxiliary constraint losses (p ≈ u_x, q ≈ u_z)
        aux_loss = torch.mean((p - u_x) ** 2) + torch.mean((q - u_z) ** 2)

        return u, u_t, p, q, p_x, q_z, C, C_x, C_z, aux_loss

    def pde_residual(self, xzt: torch.Tensor):
        """Compute PDE residual at collocation points.

        Residual:  r = u_t − [C*(u_xx + Rcv*u_zz) + C_x*u_x + Rcv*C_z*u_z]

        Parameters
        ----------
        xzt : (N, 3) normalised tensor  [x̄, z̄, t̄],  requires_grad=True

        Returns
        -------
        res : (N, 1) residual tensor
        aux_loss : scalar  (0 unless use_mixed_formulation=True)
        """
        Rcv = self.geom.Rcv
        if self.net_cfg.use_mixed_formulation:
            u, u_t, u_x, u_z, u_xx, u_zz, C, C_x, C_z, aux = \
                self._u_and_derivs_mixed(xzt)
            res = u_t - (C * (u_xx + Rcv * u_zz) + C_x * u_x + Rcv * C_z * u_z)
            return res, aux
        else:
            u, u_t, u_x, u_z, u_xx, u_zz, C, C_x, C_z = self._u_and_derivs(xzt)
            res = u_t - (C * (u_xx + Rcv * u_zz) + C_x * u_x + Rcv * C_z * u_z)
            return res, torch.tensor(0.0, device=self.device)

    # ------------------------------------------------------------------
    # Boundary / initial condition loss
    # ------------------------------------------------------------------

    def _bc_ic_loss(self, n_bc: int, n_ic: int):
        """Sample random boundary/IC points and compute soft constraint losses.

        BCs:
          flag = 0  →  Dirichlet:  enforce  u = 0
          flag = 1  →  Neumann:    enforce  ∂u/∂n = 0
        IC:
          enforce  u(x, z, 0) = u0  everywhere
        """
        top, bot, left, right = self.geom.bcs
        u0   = self.geom.u0
        dev  = self.device
        bc_loss = torch.tensor(0.0, device=dev)
        n = max(1, n_bc // 4)

        def _rand_pts(n_pts, x_lo, x_hi, z_lo, z_hi, t_lo=0.0, t_hi=1.0):
            xs = torch.FloatTensor(n_pts, 1).uniform_(x_lo, x_hi).to(dev)
            zs = torch.FloatTensor(n_pts, 1).uniform_(z_lo, z_hi).to(dev)
            ts = torch.FloatTensor(n_pts, 1).uniform_(t_lo, t_hi).to(dev)
            pts = torch.cat([xs, zs, ts], dim=1)
            pts.requires_grad_(True)
            return pts

        def _dirichlet_loss(u_val: torch.Tensor) -> torch.Tensor:
            return torch.mean(u_val ** 2)

        def _neumann_loss_z(u_val: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
            g = torch.autograd.grad(
                u_val, pts,
                grad_outputs=torch.ones_like(u_val),
                create_graph=True,
            )[0]
            return torch.mean(g[:, 1:2] ** 2)   # (∂u/∂z̄)²

        def _neumann_loss_x(u_val: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
            g = torch.autograd.grad(
                u_val, pts,
                grad_outputs=torch.ones_like(u_val),
                create_graph=True,
            )[0]
            return torch.mean(g[:, 0:1] ** 2)   # (∂u/∂x̄)²

        # Top  (z̄ = 0)
        pts_top = _rand_pts(n, 0, 1, 0, 0)
        u_top   = self.u_net(pts_top)
        bc_loss = bc_loss + (_dirichlet_loss(u_top) if top == 0
                             else _neumann_loss_z(u_top, pts_top))

        # Bottom  (z̄ = 1)
        pts_bot = _rand_pts(n, 0, 1, 1, 1)
        u_bot   = self.u_net(pts_bot)
        bc_loss = bc_loss + (_dirichlet_loss(u_bot) if bot == 0
                             else _neumann_loss_z(u_bot, pts_bot))

        # Left  (x̄ = 0)
        pts_left = _rand_pts(n, 0, 0, 0, 1)
        u_left   = self.u_net(pts_left)
        bc_loss = bc_loss + (_dirichlet_loss(u_left) if left == 0
                             else _neumann_loss_x(u_left, pts_left))

        # Right  (x̄ = 1)
        pts_right = _rand_pts(n, 1, 1, 0, 1)
        u_right   = self.u_net(pts_right)
        bc_loss = bc_loss + (_dirichlet_loss(u_right) if right == 0
                             else _neumann_loss_x(u_right, pts_right))

        # Initial condition  (t̄ = 0)
        xs_ic = torch.FloatTensor(n_ic, 1).uniform_(0, 1).to(dev)
        zs_ic = torch.FloatTensor(n_ic, 1).uniform_(0, 1).to(dev)
        ts_ic = torch.zeros(n_ic, 1, device=dev)
        xzt_ic = torch.cat([xs_ic, zs_ic, ts_ic], dim=1)
        u_ic   = self.u_net(xzt_ic)
        ic_loss = torch.mean((u_ic - u0) ** 2)

        return bc_loss, ic_loss

    # ------------------------------------------------------------------
    # Data loss
    # ------------------------------------------------------------------

    def _data_loss(self, obs: dict):
        """Supervised MSE loss at observation points for both u and C."""
        Lh, Lv, Lt = self.geom.Lh, self.geom.Lv, self.geom.Lt

        # u observations
        u_pts_phys = self._to_tensor(obs["u_pts"])
        u_vals     = self._to_tensor(obs["u_vals"]).unsqueeze(1)
        u_pts_norm = torch.stack([
            u_pts_phys[:, 0] / Lh,
            u_pts_phys[:, 1] / Lv,
            u_pts_phys[:, 2] / Lt,
        ], dim=1)
        u_pred  = self.u_net(u_pts_norm)
        loss_u  = torch.mean((u_pred - u_vals) ** 2)

        # C observations
        c_pts_phys = self._to_tensor(obs["c_pts"])
        c_vals     = self._to_tensor(obs["c_vals"]).unsqueeze(1)
        c_pts_norm = torch.stack([
            c_pts_phys[:, 0] / Lh,
            c_pts_phys[:, 1] / Lv,
        ], dim=1)
        c_pred  = self.c_net(c_pts_norm)
        loss_c  = torch.mean((c_pred - c_vals) ** 2)

        return loss_u, loss_c

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, obs_data: Optional[dict] = None) -> None:
        """Train the PINN.

        Loss-history lists (``loss_pde_history``, ``loss_bc_history``, etc.)
        are **appended to** during training and are never reset.  This means
        calling train() multiple times accumulates the full history.

        Parameters
        ----------
        obs_data : dict, optional
            Pre-computed observation data dict.  When None the internal
            ``self._obs_data`` set during ``generate_observations_*()`` is used.
        """
        if obs_data is not None:
            self._obs_data = obs_data
        if self._obs_data is None:
            raise RuntimeError(
                "No observation data.  Call generate_observations_from_qapilm() "
                "or generate_observations_from_fd() first."
            )
        obs = self._obs_data

        self._setup_optimizer()
        tc  = self.train_cfg
        dev = self.device

        self.u_net.train()
        self.c_net.train()
        if self._p_net is not None:
            self._p_net.train()
            self._q_net.train()

        print(
            f"[PINN] Training  epochs={tc.epochs}  "
            f"lr={tc.lr}  mixed={'yes' if self.net_cfg.use_mixed_formulation else 'no'}"
        )

        for epoch in range(tc.epochs):
            self.optimizer.zero_grad()

            # ---- PDE residual ----
            x_c  = torch.FloatTensor(tc.n_colloc_pde, 1).uniform_(0, 1).to(dev)
            z_c  = torch.FloatTensor(tc.n_colloc_pde, 1).uniform_(0, 1).to(dev)
            t_c  = torch.FloatTensor(tc.n_colloc_pde, 1).uniform_(0, 1).to(dev)
            xzt_c = torch.cat([x_c, z_c, t_c], dim=1)
            xzt_c.requires_grad_(True)

            res, aux_loss = self.pde_residual(xzt_c)
            loss_pde = torch.mean(res ** 2) + aux_loss

            # ---- BC / IC ----
            loss_bc, loss_ic = self._bc_ic_loss(tc.n_bc, tc.n_ic)

            # ---- Data ----
            loss_u, loss_c = self._data_loss(obs)
            loss_data = tc.w_data_u * loss_u + tc.w_data_c * loss_c

            # ---- Total ----
            loss = (tc.w_pde  * loss_pde
                    + tc.w_bc * loss_bc
                    + tc.w_ic * loss_ic
                    + loss_data)

            loss.backward()

            if tc.grad_clip > 0:
                params = (
                    list(self.u_net.parameters()) +
                    list(self.c_net.parameters()) +
                    (list(self._p_net.parameters()) + list(self._q_net.parameters())
                     if self._p_net is not None else [])
                )
                nn.utils.clip_grad_norm_(params, tc.grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            # Accumulate history — never reset inside the loop
            self.loss_pde_history.append(float(loss_pde.item()))
            self.loss_bc_history.append(float(loss_bc.item()))
            self.loss_ic_history.append(float(loss_ic.item()))
            self.loss_data_history.append(float((loss_u + loss_c).item()))
            self.loss_total_history.append(float(loss.item()))

            if epoch % tc.print_every == 0:
                print(
                    f"  Epoch {epoch:5d}/{tc.epochs}  "
                    f"pde={loss_pde.item():.3e}  "
                    f"bc={loss_bc.item():.3e}  "
                    f"ic={loss_ic.item():.3e}  "
                    f"data={loss_data.item():.3e}  "
                    f"total={loss.item():.3e}"
                )

        self.u_net.eval()
        self.c_net.eval()
        if self._p_net is not None:
            self._p_net.eval()
            self._q_net.eval()
        print(
            f"[PINN] Training complete.  "
            f"Final total loss = {self.loss_total_history[-1]:.4e}"
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, metadata: Optional[dict] = None) -> str:
        """Save the trained PINN with a timestamped filename.

        Files written
        -------------
        ``<model_dir>/pinn_YYYYMMDD_HHMMSS.pt``    — model weights + history
        ``<model_dir>/pinn_YYYYMMDD_HHMMSS_meta.json`` — training metadata

        Parameters
        ----------
        metadata : additional key-value pairs to include in the JSON sidecar

        Returns
        -------
        save_path : full path of the .pt file
        """
        sc = self.save_cfg
        os.makedirs(sc.model_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname    = f"pinn_{ts}.pt"
        save_path = os.path.join(sc.model_dir, fname)

        payload: dict = {
            "u_net_state":        self.u_net.state_dict(),
            "c_net_state":        self.c_net.state_dict(),
            "geom_cfg":           asdict(self.geom),
            "net_cfg":            asdict(self.net_cfg),
            "train_cfg":          asdict(self.train_cfg),
            "obs_cfg":            asdict(self.obs_cfg),
            "loss_pde_history":   self.loss_pde_history,
            "loss_bc_history":    self.loss_bc_history,
            "loss_ic_history":    self.loss_ic_history,
            "loss_data_history":  self.loss_data_history,
            "loss_total_history": self.loss_total_history,
        }
        if self._p_net is not None:
            payload["p_net_state"] = self._p_net.state_dict()
            payload["q_net_state"] = self._q_net.state_dict()

        torch.save(payload, save_path)
        self._save_path = save_path

        if sc.save_metadata:
            meta: dict = {
                "timestamp":    ts,
                "model_file":   save_path,
                "geom":         asdict(self.geom),
                "net":          asdict(self.net_cfg),
                "training":     asdict(self.train_cfg),
                "final_loss":   (self.loss_total_history[-1]
                                 if self.loss_total_history else None),
                "n_epochs_run": len(self.loss_total_history),
            }
            if metadata:
                meta.update(metadata)
            meta_path = save_path.replace(".pt", "_meta.json")
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, default=str)
            print(f"[PINN] Metadata  → {meta_path}")

        print(f"[PINN] Model saved → {save_path}")
        return save_path

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PINN2DConsolidation":
        """Load a saved PINN from a ``.pt`` file.

        Parameters
        ----------
        path   : path returned by save() or the timestamped .pt filename
        device : target device ("cpu", "cuda", "mps"; auto-detected if None)

        Returns
        -------
        pinn : PINN2DConsolidation with restored weights and loss histories
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)

        # Restore configs — handle tuple ↔ list serialisation differences
        def _fix_tuples(d: dict, *tuple_keys: str) -> dict:
            d = dict(d)
            for k in tuple_keys:
                if k in d and isinstance(d[k], list):
                    d[k] = tuple(d[k])
            return d

        geom_dict  = _fix_tuples(payload["geom_cfg"],  "bcs", "spatial_reso")
        net_dict   = dict(payload["net_cfg"])
        train_dict = dict(payload["train_cfg"])
        obs_dict   = dict(payload["obs_cfg"])

        geom      = PINNGeomConfig(**geom_dict)
        net_cfg   = PINNNetConfig(**net_dict)
        train_cfg = PINNTrainConfig(**train_dict)
        obs_cfg   = PINNObsConfig(**obs_dict)

        pinn = cls(geom=geom, net_cfg=net_cfg, train_cfg=train_cfg,
                   obs_cfg=obs_cfg, device=device)
        pinn.u_net.load_state_dict(payload["u_net_state"])
        pinn.c_net.load_state_dict(payload["c_net_state"])
        pinn.u_net.eval()
        pinn.c_net.eval()

        if "p_net_state" in payload and pinn._p_net is not None:
            pinn._p_net.load_state_dict(payload["p_net_state"])
            pinn._q_net.load_state_dict(payload["q_net_state"])
            pinn._p_net.eval()
            pinn._q_net.eval()

        pinn.loss_pde_history   = payload.get("loss_pde_history",   [])
        pinn.loss_bc_history    = payload.get("loss_bc_history",    [])
        pinn.loss_ic_history    = payload.get("loss_ic_history",    [])
        pinn.loss_data_history  = payload.get("loss_data_history",  [])
        pinn.loss_total_history = payload.get("loss_total_history", [])
        pinn._save_path         = path

        print(f"[PINN] Loaded from {path}")
        return pinn

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_C_field(
        self,
        grid_x: np.ndarray,
        grid_z: np.ndarray,
    ) -> np.ndarray:
        """Predict C(x,z) on a 2-D coordinate grid.

        Parameters
        ----------
        grid_x : (nz, nx) array of physical x-coordinates
        grid_z : (nz, nx) array of physical z-coordinates

        Returns
        -------
        C_pred : (nz, nx) array
        """
        Lh, Lv = self.geom.Lh, self.geom.Lv
        x_n = (grid_x.flatten() / Lh).astype(np.float32)
        z_n = (grid_z.flatten() / Lv).astype(np.float32)
        xz_t = torch.tensor(
            np.column_stack([x_n, z_n]), device=self.device
        )
        with torch.no_grad():
            C_flat = self.c_net(xz_t).cpu().numpy().flatten()
        return C_flat.reshape(grid_x.shape)

    def predict_u_field(
        self,
        grid_x: np.ndarray,
        grid_z: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Predict u(x,z,t) on a 2-D grid at physical time t.

        Parameters
        ----------
        grid_x : (nz, nx) array of physical x-coordinates
        grid_z : (nz, nx) array of physical z-coordinates
        t      : physical time  (must be in [0, Lt])

        Returns
        -------
        u_pred : (nz, nx) array
        """
        Lh, Lv, Lt = self.geom.Lh, self.geom.Lv, self.geom.Lt
        x_n = (grid_x.flatten() / Lh).astype(np.float32)
        z_n = (grid_z.flatten() / Lv).astype(np.float32)
        t_n = np.full(len(x_n), float(t) / Lt, dtype=np.float32)
        xzt_t = torch.tensor(
            np.column_stack([x_n, z_n, t_n]), device=self.device
        )
        with torch.no_grad():
            u_flat = self.u_net(xzt_t).cpu().numpy().flatten()
        return u_flat.reshape(grid_x.shape)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_comparison(
        self,
        true_C: np.ndarray,
        qapilm_C: Optional[np.ndarray] = None,
        grid_x: Optional[np.ndarray] = None,
        grid_z: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        cmap_main: str = "viridis",
        cmap_err: str = "bwr",
        mode: int = 1,
    ):
        """Triple-panel comparison plot matching RectangularQAPILM.triplot2D().

        Layout
        ------
        If qapilm_C is provided (full comparison):
            [ Real C | QAPILM C | PINN C | log(Real/PINN) ]
        Otherwise (two-model comparison):
            [ Real C | PINN C | log(Real/PINN) ]

        Parameters
        ----------
        true_C    : (nz, nx) ground-truth C field
        qapilm_C  : (nz, nx) QAPILM-estimated C field (None → omit panel)
        grid_x    : physical x grid  (inferred from geom if None)
        grid_z    : physical z grid
        save_path : file path to save figure  (None → plt.show())
        cmap_main : colormap for C panels
        cmap_err  : colormap for error/ratio panel
        mode      : 1 → show log(Real/PINN) in last panel
                    0 → show (Real − PINN)

        Returns
        -------
        fig : matplotlib Figure
        """
        g  = self.geom
        nz, nx = true_C.shape

        if grid_x is None or grid_z is None:
            xs = np.linspace(0, g.Lh, nx)
            zs = np.linspace(0, g.Lv, nz)
            grid_x, grid_z = np.meshgrid(xs, zs)

        pinn_C = self.predict_C_field(grid_x, grid_z)
        err_C  = (np.log(np.clip(true_C, 1e-12, None) /
                         np.clip(pinn_C, 1e-12, None))
                  if mode == 1 else true_C - pinn_C)

        ncols = 4 if qapilm_C is not None else 3
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), dpi=100)

        # Shared colour scale for C panels
        datasets = [true_C, pinn_C] + ([qapilm_C] if qapilm_C is not None else [])
        vmin = float(min(d.min() for d in datasets))
        vmax = float(max(d.max() for d in datasets))
        norm_main = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Tick marks (matching triplot2D convention)
        xticks  = [0, nx // 4, nx // 2, 3 * nx // 4, nx - 1]
        xtlabel = [f"{v:.1f}" for v in np.array([0, .25, .5, .75, 1.]) * g.Lh]
        zticks  = [0, nz // 4, nz // 2, 3 * nz // 4, nz - 1]
        ztlabel = [f"{v:.1f}" for v in np.array([0, .25, .5, .75, 1.]) * g.Lv]

        def _set_ticks(ax):
            ax.set_xticks(xticks); ax.set_xticklabels(xtlabel)
            ax.set_yticks(zticks); ax.set_yticklabels(ztlabel)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")

        col = 0

        # Panel 1 — Real C
        axes[col].imshow(true_C, cmap=cmap_main, norm=norm_main,
                         origin="upper", aspect="auto", interpolation="bicubic")
        axes[col].set_title("Real C")
        _set_ticks(axes[col])
        col += 1

        # Panel 2 — QAPILM C (optional)
        if qapilm_C is not None:
            im2 = axes[col].imshow(qapilm_C, cmap=cmap_main, norm=norm_main,
                                   origin="upper", aspect="auto",
                                   interpolation="bicubic")
            axes[col].set_title("QAPILM C")
            _set_ticks(axes[col])
            divider = make_axes_locatable(axes[col])
            plt.colorbar(im2, cax=divider.append_axes("right", "5%", pad=0.1))
            col += 1

        # Panel 3 — PINN C
        im3 = axes[col].imshow(pinn_C, cmap=cmap_main, norm=norm_main,
                               origin="upper", aspect="auto",
                               interpolation="bicubic")
        axes[col].set_title("PINN C")
        _set_ticks(axes[col])
        divider = make_axes_locatable(axes[col])
        plt.colorbar(im3, cax=divider.append_axes("right", "5%", pad=0.1))
        col += 1

        # Panel 4 — Error / ratio
        abs_max = max(float(np.abs(err_C).max()), 1e-12)
        norm_err = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        err_title = "log(Real/PINN)" if mode == 1 else "Real − PINN"
        im4 = axes[col].imshow(err_C, cmap=cmap_err, norm=norm_err,
                               origin="upper", aspect="auto",
                               interpolation="bicubic")
        axes[col].set_title(err_title)
        _set_ticks(axes[col])
        divider = make_axes_locatable(axes[col])
        plt.colorbar(im4, cax=divider.append_axes("right", "5%", pad=0.1))

        plt.suptitle("Consolidation coefficient  C  comparison", fontsize=12)
        plt.tight_layout()

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[PINN] Comparison plot → {save_path}")
        return fig

    def plot_loss_history(self, save_path: Optional[str] = None):
        """Plot all loss component histories on a log scale.

        Parameters
        ----------
        save_path : file path (None → plt.show())
        """
        if not self.loss_total_history:
            print("[PINN] No loss history available.")
            return None

        fig, ax = plt.subplots(figsize=(9, 4))
        ep = np.arange(1, len(self.loss_total_history) + 1)
        ax.semilogy(ep, self.loss_total_history, lw=2,   label="total")
        ax.semilogy(ep, self.loss_pde_history,   lw=1.5, ls="--",         label="PDE")
        ax.semilogy(ep, self.loss_bc_history,    lw=1.5, ls="-.",         label="BC")
        ax.semilogy(ep, self.loss_ic_history,    lw=1.5, ls=":",          label="IC")
        ax.semilogy(ep, self.loss_data_history,  lw=1.5, ls=(0,(3,1,1,1)), label="data")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("PINN training loss history")
        ax.legend(loc="upper right")
        plt.tight_layout()

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[PINN] Loss history plot → {save_path}")
        return fig


# ===========================================================================
# STANDALONE COMPARISON HELPER
# ===========================================================================

def compare_and_plot(
    true_C_or_path,
    qapilm_C_or_path,
    pinn: "PINN2DConsolidation",
    save_path: Optional[str] = None,
    mode: int = 1,
):
    """Generate a Real / QAPILM / PINN comparison plot with minimal boilerplate.

    This function can be imported and used from any script without needing to
    re-instantiate a PINN — just load the model once and call this helper.

    Parameters
    ----------
    true_C_or_path     : (nz, nx) array  **or**  path to a .npy file
                         containing the ground-truth C field
    qapilm_C_or_path   : (nz, nx) array  **or**  path to a .npy file
                         containing the QAPILM-estimated C field
                         (pass None to skip the QAPILM panel)
    pinn               : trained PINN2DConsolidation instance
    save_path          : output figure path  (None → plt.show())
    mode               : 1 → log(Real/PINN) error,  0 → Real−PINN

    Returns
    -------
    fig : matplotlib Figure

    Example
    -------
    >>> from vanilla_PINN_2D_rect import PINN2DConsolidation, compare_and_plot
    >>> pinn = PINN2DConsolidation.load("pinn_models/pinn_20250101_120000.pt")
    >>> compare_and_plot("results/run/ch_true.npy",
    ...                  "results/run/ch_est.npy",
    ...                  pinn,
    ...                  save_path="comparison.png")
    """
    def _load(x):
        if isinstance(x, (str, os.PathLike)):
            return np.load(str(x))
        return np.asarray(x) if x is not None else None

    true_C    = _load(true_C_or_path)
    qapilm_C  = _load(qapilm_C_or_path)

    return pinn.plot_comparison(
        true_C, qapilm_C, save_path=save_path, mode=mode
    )


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def _make_synthetic_C(geom: PINNGeomConfig) -> np.ndarray:
    """Create a simple smooth synthetic C field for demonstration."""
    dx, dz = geom.spatial_reso
    nx = int(round(geom.Lh / dx))
    nz = int(round(geom.Lv / dz))
    xs = np.linspace(0, geom.Lh, nx + 1)
    zs = np.linspace(0, geom.Lv, nz + 1)
    XX, ZZ = np.meshgrid(xs, zs)
    C = (1.5
         + 0.5 * np.sin(np.pi * XX / geom.Lh)
         + 0.3 * np.cos(2 * np.pi * ZZ / geom.Lv)
         + 0.1 * np.sin(2 * np.pi * XX / geom.Lh) * np.cos(np.pi * ZZ / geom.Lv))
    return C.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PINN solver for 2-D heterogeneous consolidation.\n"
            "Generates observations from the FD forward solver, trains the PINN,\n"
            "saves the model, and produces comparison plots."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--load", default=None, metavar="PATH",
        help="Load an existing .pt model and produce plots only.",
    )
    parser.add_argument(
        "--outdir", default="./pinn_models", metavar="DIR",
        help="Directory for saved models and figures  (default: ./pinn_models)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cpu | cuda | mps  (auto-detected if omitted)",
    )
    parser.add_argument(
        "--use-qapilm", action="store_true",
        help="Use the full QAPILM FD solver if available (requires CSV files).",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load mode — just produce plots from a saved model
    # -----------------------------------------------------------------------
    if args.load:
        pinn = PINN2DConsolidation.load(args.load, device=args.device)
        pinn.plot_loss_history(
            save_path=os.path.join(args.outdir, "loss_history.png")
        )
        print("[PINN] Load mode — no ground-truth field available for comparison.")
        print("       Use compare_and_plot() to generate a comparison figure.")
        return

    # -----------------------------------------------------------------------
    # Train mode
    # -----------------------------------------------------------------------
    geom      = PINNGeomConfig()
    net_cfg   = PINNNetConfig()
    train_cfg = PINNTrainConfig(epochs=args.epochs or 3000)
    obs_cfg   = PINNObsConfig()
    save_cfg  = PINNSaveConfig(model_dir=args.outdir)

    pinn = PINN2DConsolidation(
        geom=geom, net_cfg=net_cfg, train_cfg=train_cfg,
        obs_cfg=obs_cfg, save_cfg=save_cfg, device=args.device,
    )

    # -----------------------------------------------------------------------
    # Generate observations
    # -----------------------------------------------------------------------
    chm_true: Optional[np.ndarray] = None

    if args.use_qapilm and _QAPILM_AVAILABLE:
        try:
            from qapilm_rect import (
                ProblemConfig, BasisConfig, ModelConfig, SolverConfig, RunConfig,
                RectangularQAPILM,
            )
            pconf = ProblemConfig(regen_fluc=True, fluc_seed=42)
            solver = RectangularQAPILM(
                pconf, BasisConfig(), ModelConfig(), SolverConfig(), RunConfig()
            )
            chm_true = solver.chm.copy()
            obs = pinn.generate_observations_from_qapilm(solver)
        except Exception as exc:
            print(f"[PINN] QAPILM init failed ({exc}); falling back to minimal FD.")
            chm_true = None

    if chm_true is None:
        print("[PINN] Using built-in minimal FD solver with synthetic C field.")
        chm_true = _make_synthetic_C(geom)
        obs = pinn.generate_observations_from_fd(chm_true)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    pinn.train(obs)

    # -----------------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------------
    model_path = pinn.save(metadata={"demo_run": True})
    np.save(os.path.join(args.outdir, "ch_true.npy"), chm_true)

    # -----------------------------------------------------------------------
    # Plot loss history
    # -----------------------------------------------------------------------
    pinn.plot_loss_history(
        save_path=os.path.join(args.outdir, "loss_history.png")
    )

    # -----------------------------------------------------------------------
    # Comparison plot
    # -----------------------------------------------------------------------
    compare_and_plot(
        chm_true,
        None,       # no QAPILM estimate in the quick-start demo
        pinn,
        save_path=os.path.join(args.outdir, "comparison_C.png"),
    )
    print(f"\n[PINN] Demo complete.  Outputs in:  {args.outdir}")


if __name__ == "__main__":
    main()
