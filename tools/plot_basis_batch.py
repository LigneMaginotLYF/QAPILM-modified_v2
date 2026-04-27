#!/usr/bin/env python3
"""
tools/plot_basis_batch.py
=========================
Compare batch experiment results across different basis-function configurations.
Produces **two** sets of figures:

**Set A** – Cross-section / temporal comparison (identical layout to
``plot_epsilon_batch.py``):

    - C-field cross sections at x = const and z = const
    - U-field: temporal evolution at fixed spatial point(s)  OR
               spatial line at a fixed timestep (set ``U_MODE``)

**Set B** – Parallel field plots with equal-sized panels:

    - C field: one row per basis type, three panels per row
      [truth | mean prediction | log(truth/pred) or (truth − pred)]
    - U field snapshots at configurable years: same 3-panel layout,
      one row per snapshot year, one column group per basis type.

Mean and confidence band
------------------------
The **mean estimate** is the pointwise mean of the C (or U) field over all
MC realizations within a run::

    mean(x) = (1/N_mc) * sum_i  field_i(x)

The **confidence band** is mean ± k·std, where *k* is set by ``BAND_K``
(default 1.0 → 68 % interval for approximately Gaussian ensembles)::

    lower(x) = mean(x) - BAND_K * std(x)
    upper(x) = mean(x) + BAND_K * std(x)

Usage
-----
1.  Edit the ``TUNABLE PARAMETERS`` block below.
2.  Run::

        python tools/plot_basis_batch.py
"""

import os
import json
import glob as _glob
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Apply Times New Roman font with larger sizes for readability
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         13,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "legend.fontsize":   11,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
})

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qapilm_rect import (
    ProblemConfig, BasisConfig, ModelConfig, SolverConfig, RunConfig,
    RectangularQAPILM, softplus,
)


# =============================================================================
# TUNABLE PARAMETERS  –  edit this section before running
# =============================================================================

# --- Results folder and run selection ----------------------------------------
RESULTS_DIR   = "./results"

GLOB_PATTERNS = [
    "baseline_*",
    "basis_dct_*",
    "basis_legendre_*",
    "basis_wavelet_*",
]

LABELS = [
    "poly+sin  (baseline)",
    "DCT",
    "Legendre",
    "Wavelet",
]

# --- Domain physical size ----------------------------------------------------
LH = 10.0      # horizontal length (x)
LV = 5.0       # vertical   length (z)

# --- Set A: cross-section parameters -----------------------------------------
C_XSEC_X_IDX = 25    # column idx for C cross section (fix x, vary z)
C_XSEC_Z_IDX = 10    # row idx    for C cross section (fix z, vary x)

U_MODE = "temporal"       # "temporal" or "spatial"
U_MONITOR_PT_IDX_1 = 0   # monitor point index for temporal panel 1
U_MONITOR_PT_IDX_2 = 1   # monitor point index for temporal panel 2
U_SPATIAL_T_IDX   = 500
U_SPATIAL_AXIS_1  = "x"
U_SPATIAL_FIX_1   = 10
U_SPATIAL_AXIS_2  = "z"
U_SPATIAL_FIX_2   = 25

# --- Confidence band ----------------------------------------------------------
# Confidence band = mean ± BAND_K * std  at each grid / time point.
# BAND_K = 1.0 → ±1σ → nominally 68 % interval for Gaussian ensembles.
# BAND_K = 2.0 → ±2σ → nominally 95 % interval.
BAND_K = 1.0

# --- Set B: parallel field parameters ----------------------------------------
# Snapshot years to use for U parallel plots.
# Only snapshots saved in u_true_snapshots.npy / u_snapshots_all.npy are used.
# These values are matched to the nearest saved snapshot year.
U_SNAP_YEARS_TO_PLOT = [0.1, 1.0, 2.0]

# Error mode for 3rd panel: "log_ratio" (log(truth/pred)) or "diff" (truth-pred)
ERROR_MODE = "log_ratio"

CMAP_FIELD = "viridis"
CMAP_ERROR = "bwr"

# --- Plot style ---------------------------------------------------------------
CONFIDENCE_ALPHA = 0.25
LINE_WIDTH       = 1.8
COLORMAP         = "tab10"
FIGSIZE_XSEC     = (7, 4.5)
DPI              = 150

OUTPUT_DIR = "./plots"

# --- MC reconstruction behavior ------------------------------------------------
# If True, load mc_weights.npy + run_config_resolved.json and reconstruct each
# MC realization as: coef -> C -> forward solve U, then compute mean/std bands.
# When False (default), the script reads pre-saved mean/std statistics from
# batch_stats.npz (or individual *_mean.npy / *_std.npy files) written by
# run_batch.py, avoiding any forward-solver recomputation at plot time.
# Set to True only when older run folders lack the precomputed stats files.
REBUILD_FROM_MC_WEIGHTS = False

# =============================================================================
# END OF TUNABLE PARAMETERS
# =============================================================================


# ---------------------------------------------------------------------------
# Data-loading utilities  (identical to plot_epsilon_batch.py)
# ---------------------------------------------------------------------------

def _try_load(run_dir: str, *filenames):
    for fn in filenames:
        path = os.path.join(run_dir, fn)
        if os.path.exists(path):
            try:
                return np.load(path, allow_pickle=False)
            except Exception as exc:
                print(f"  [WARNING] Failed to load '{path}': {exc}")
    return None


def _load_from_precomputed_stats(run_dir: str):
    """Load pre-saved mean/std statistics from batch_stats.npz or individual files.

    Primary loading path: reads arrays written by run_batch.py without
    requiring any forward-solver recomputation.  Falls back to loading
    individual ``*_mean.npy`` / ``*_std.npy`` files if the NPZ bundle is absent.

    Returns
    -------
    dict or None if nothing found.
    """
    d = {}
    npz_path = os.path.join(run_dir, "batch_stats.npz")
    if os.path.exists(npz_path):
        try:
            bundle = np.load(npz_path, allow_pickle=False)
            if "ch_est_mean" in bundle:
                d["ch_est_mean"] = bundle["ch_est_mean"]
                d["ch_est_std"]  = bundle["ch_est_std"] if "ch_est_std" in bundle \
                                   else np.zeros_like(d["ch_est_mean"])
            if "ch_true" in bundle:
                d["ch_true"] = bundle["ch_true"]
            if "u_temporal_mean" in bundle:
                d["u_temporal_mean"] = bundle["u_temporal_mean"]
                d["u_temporal_std"]  = bundle["u_temporal_std"] if "u_temporal_std" in bundle \
                                       else np.zeros_like(d["u_temporal_mean"])
            if "u_true_temporal" in bundle:
                d["u_true_temporal"] = bundle["u_true_temporal"]
            if "u_snapshots_mean" in bundle:
                d["u_snapshots_mean"] = bundle["u_snapshots_mean"]
                d["u_snapshots_std"]  = bundle["u_snapshots_std"] if "u_snapshots_std" in bundle \
                                        else np.zeros_like(d["u_snapshots_mean"])
            if "u_true_snapshots" in bundle:
                d["u_true_snapshots"] = bundle["u_true_snapshots"]
            if "t_coords" in bundle:
                d["t_coords"] = bundle["t_coords"]
            if "snapshot_years" in bundle:
                d["snapshot_years"] = bundle["snapshot_years"]
            if "monitor_points" in bundle:
                d["monitor_points"] = bundle["monitor_points"]
            if d:
                return d
        except Exception as exc:
            print(f"  [WARNING] Failed to load '{npz_path}': {exc}")

    # Fallback: individual precomputed stats files
    ch_mean = _try_load(run_dir, "ch_est.npy")
    ch_std  = _try_load(run_dir, "ch_est_std.npy")
    if ch_mean is not None:
        d["ch_est_mean"] = ch_mean
        d["ch_est_std"]  = ch_std if ch_std is not None else np.zeros_like(ch_mean)
    ch_true = _try_load(run_dir, "ch_true.npy")
    if ch_true is not None:
        d["ch_true"] = ch_true

    u_temp_m = _try_load(run_dir, "u_temporal_mean.npy")
    u_temp_s = _try_load(run_dir, "u_temporal_std.npy")
    if u_temp_m is not None:
        d["u_temporal_mean"] = u_temp_m
        d["u_temporal_std"]  = u_temp_s if u_temp_s is not None else np.zeros_like(u_temp_m)
    u_true_t = _try_load(run_dir, "u_true_temporal.npy")
    if u_true_t is not None:
        d["u_true_temporal"] = u_true_t

    u_snap_m = _try_load(run_dir, "u_snapshots_mean.npy")
    u_snap_s = _try_load(run_dir, "u_snapshots_std.npy")
    if u_snap_m is not None:
        d["u_snapshots_mean"] = u_snap_m
        d["u_snapshots_std"]  = u_snap_s if u_snap_s is not None else np.zeros_like(u_snap_m)
    u_true_s = _try_load(run_dir, "u_true_snapshots.npy")
    if u_true_s is not None:
        d["u_true_snapshots"] = u_true_s

    t_coords = _try_load(run_dir, "t_coords.npy")
    if t_coords is not None:
        d["t_coords"] = t_coords
    snap_yrs = _try_load(run_dir, "snapshot_years.npy")
    if snap_yrs is not None:
        d["snapshot_years"] = snap_yrs
    mon_pts = _try_load(run_dir, "monitor_points.npy")
    if mon_pts is not None:
        d["monitor_points"] = mon_pts

    return d if d else None


def _build_solver_from_resolved_config(run_dir: str):
    cfg_path = os.path.join(run_dir, "run_config_resolved.json")
    if not os.path.exists(cfg_path):
        return None, None
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:
        print(f"  [WARNING] Failed to parse '{cfg_path}': {exc}")
        return None, None

    try:
        pconf = ProblemConfig(**cfg.get("problem", {}))
        bconf = BasisConfig(**cfg.get("basis", {}))
        mconf = ModelConfig(**cfg.get("model", {}))
        sconf = SolverConfig(**cfg.get("solver", {}))
        rvals = dict(cfg.get("run", {}))
        rvals["results_dir"] = run_dir
        rvals["save_losses"] = False
        rconf = RunConfig(**rvals)
        solver = RectangularQAPILM(pconf, bconf, mconf, sconf, rconf)
    except Exception as exc:
        print(f"  [WARNING] Failed to initialize solver for '{run_dir}': {exc}")
        return None, None
    return solver, cfg


def _reconstruct_from_mc_weights(run_dir: str):
    mc_weights = _try_load(run_dir, "mc_weights.npy")
    if mc_weights is None:
        return None

    solver, cfg = _build_solver_from_resolved_config(run_dir)
    if solver is None:
        print(f"  [WARNING] Cannot reconstruct MC realizations for '{run_dir}' "
              f"(missing/invalid run_config_resolved.json).")
        return None

    mc_weights = np.asarray(mc_weights, dtype=np.float64)
    if mc_weights.ndim == 1:
        mc_weights = mc_weights[np.newaxis, :]

    numt = solver.ilim
    top, bot, left, right = solver.p.bcs
    u0_mat = np.ones((solver.numv + 1, solver.numh + 1), dtype=np.float64) * solver.p.u0

    meas = cfg.get("measurements", {})
    ukt = np.asarray(meas.get("ukt", [numt]), dtype=int)
    if ukt.size == 0:
        ukt = np.array([numt], dtype=int)
    ukt = np.clip(ukt, 0, numt)
    obs_last_t = int(np.max(ukt))

    plot_cfg = cfg.get("plot_output", {})
    monitor_points = [tuple(pt) for pt in plot_cfg.get("monitor_points", [[10, 25], [5, 10]])]
    monitor_points = [
        (min(int(pt[0]), solver.numv), min(int(pt[1]), solver.numh))
        for pt in monitor_points
    ]
    snapshot_years_req = list(plot_cfg.get("snapshot_years", [0.1, 1.0, 2.0]))
    t_coords = np.arange(numt + 1) * solver.dt
    snap_t_idxs = [min(int(round(y / solver.dt)), numt) for y in snapshot_years_req]
    snap_years_actual = np.array([t_coords[i] for i in snap_t_idxs], dtype=np.float64)

    ch_all = []
    u_est_all = []
    u_temporal_all = []
    u_snapshots_all = []
    n_mc = mc_weights.shape[0]
    print(f"  Reconstructing {n_mc} MC realization(s) for '{os.path.basename(run_dir)}' ...")
    for mc_idx, coef in enumerate(mc_weights, start=1):
        m_est = solver.vec2mat2(solver.basese @ coef)
        ch_est = softplus(m_est)
        utens_est, _ = solver.forward_solver(ch_est, numt, u0_mat, top, bot, left, right)
        ch_all.append(ch_est)
        u_est_all.append(utens_est[obs_last_t].copy())
        u_temporal_all.append(np.array([utens_est[:, r, c] for r, c in monitor_points]))
        u_snapshots_all.append(np.array([utens_est[t_i] for t_i in snap_t_idxs]))
        if mc_idx == n_mc or (mc_idx % max(1, n_mc // 5) == 0):
            print(f"    progress: {mc_idx}/{n_mc}")

    utens_true, _ = solver.forward_solver(solver.chm, numt, u0_mat, top, bot, left, right)
    u_true_temporal = np.array([utens_true[:, r, c] for r, c in monitor_points])
    u_true_snapshots = np.array([utens_true[t_i] for t_i in snap_t_idxs])

    ch_all = np.array(ch_all)
    u_temporal_arr  = np.array(u_temporal_all)
    u_snapshots_arr = np.array(u_snapshots_all)
    return {
        "ch_true": solver.chm,
        "ch_est_all": ch_all,
        "ch_est_mean": np.mean(ch_all, axis=0),
        "ch_est_std": np.std(ch_all, axis=0),
        "u_est_all": np.array(u_est_all),
        "u_temporal_all": u_temporal_arr,
        "u_temporal_mean": np.mean(u_temporal_arr, axis=0),
        "u_temporal_std":  np.std( u_temporal_arr, axis=0),
        "u_snapshots_all": u_snapshots_arr,
        "u_snapshots_mean": np.mean(u_snapshots_arr, axis=0),
        "u_snapshots_std":  np.std( u_snapshots_arr, axis=0),
        "u_true_temporal": u_true_temporal,
        "u_true_snapshots": u_true_snapshots,
        "t_coords": t_coords,
        "snapshot_years": snap_years_actual,
        "monitor_points": np.array(monitor_points, dtype=int),
    }


def load_run_data(run_dir: str) -> dict:
    """Load all relevant arrays from a single run directory.

    Loading priority
    ----------------
    1. Pre-saved field statistics (``batch_stats.npz`` or individual
       ``ch_est_mean/std.npy``, ``u_temporal_mean/std.npy``, etc.) written by
       ``run_batch.py``.  These files avoid any forward-solver recomputation.
    2. If ``REBUILD_FROM_MC_WEIGHTS`` is True **and** the precomputed stats are
       absent, reconstruct each MC realization from ``mc_weights.npy`` by
       decoding coef → C → forward-solve U (slow but fully reproducible).
    3. Fallback: load the per-MC ensemble arrays (``ch_est_all.npy``,
       ``u_temporal_all.npy``, etc.) directly and compute mean/std on the fly.
    """
    d = {}

    # 1. Prefer pre-saved stats (fastest; no solver needed)
    precomp = _load_from_precomputed_stats(run_dir)
    if precomp is not None and ("ch_est_mean" in precomp or "u_temporal_mean" in precomp):
        d.update(precomp)
        return d

    # 2. Optional: reconstruct from mc_weights (legacy fallback)
    if REBUILD_FROM_MC_WEIGHTS:
        recon = _reconstruct_from_mc_weights(run_dir)
        if recon is not None:
            d.update(recon)
            return d

    # 3. Load per-MC ensemble arrays and derive mean/std on the fly
    ch_true = _try_load(run_dir, "ch_true.npy")
    if ch_true is not None:
        d["ch_true"] = ch_true

    ch_all = _try_load(run_dir, "ch_est_all.npy")
    if ch_all is not None:
        d["ch_est_all"]  = ch_all
        d["ch_est_mean"] = np.mean(ch_all, axis=0)
        d["ch_est_std"]  = np.std(ch_all,  axis=0)
    else:
        ch_mean = _try_load(run_dir, "ch_est.npy")
        if ch_mean is not None:
            d["ch_est_mean"] = ch_mean
            d["ch_est_std"]  = np.zeros_like(ch_mean)

    t_coords = _try_load(run_dir, "t_coords.npy")
    if t_coords is not None:
        d["t_coords"] = t_coords

    u_temporal = _try_load(run_dir, "u_temporal_all.npy")
    if u_temporal is not None:
        d["u_temporal_mean"] = np.mean(u_temporal, axis=0)
        d["u_temporal_std"]  = np.std( u_temporal, axis=0)

    u_true_temp = _try_load(run_dir, "u_true_temporal.npy")
    if u_true_temp is not None:
        d["u_true_temporal"] = u_true_temp

    u_snaps = _try_load(run_dir, "u_snapshots_all.npy")
    if u_snaps is not None:
        d["u_snapshots_mean"] = np.mean(u_snaps, axis=0)
        d["u_snapshots_std"]  = np.std( u_snaps, axis=0)

    u_true_snaps = _try_load(run_dir, "u_true_snapshots.npy")
    if u_true_snaps is not None:
        d["u_true_snapshots"] = u_true_snaps

    snap_yrs = _try_load(run_dir, "snapshot_years.npy")
    if snap_yrs is not None:
        d["snapshot_years"] = snap_yrs

    mon_pts = _try_load(run_dir, "monitor_points.npy")
    if mon_pts is not None:
        d["monitor_points"] = mon_pts

    return d


def find_runs(results_dir: str, patterns: list, labels: list):
    found_dirs, found_labels = [], []
    for pattern, label in zip(patterns, labels):
        matches = sorted(_glob.glob(os.path.join(results_dir, pattern)))
        if matches:
            found_dirs.append(matches[-1])
            found_labels.append(label)
        else:
            print(f"  [WARNING] No run matched pattern '{pattern}' – skipping.")
    return found_dirs, found_labels


# ---------------------------------------------------------------------------
# Set-A helpers  (shared with plot_epsilon_batch)
# ---------------------------------------------------------------------------

def _run_colors(n: int):
    cmap = get_cmap(COLORMAP)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _set_axis_labels(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title,   fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)


def plot_c_xsec(runs_data, labels, idx, axis, colors=None,
                band_k=BAND_K, save_path=None):
    """Plot C cross section (mean ± band_k·std) for all runs, superposed."""
    fig, ax = plt.subplots(figsize=FIGSIZE_XSEC, dpi=DPI)
    if colors is None:
        colors = _run_colors(len(runs_data))

    truth_plotted = False
    phys = ""
    for data, lbl, col in zip(runs_data, labels, colors):
        if "ch_est_mean" not in data:
            continue
        mean = data["ch_est_mean"]
        std  = data.get("ch_est_std", np.zeros_like(mean))
        nv, nh = mean.shape[0] - 1, mean.shape[1] - 1

        if axis == "x":
            col_i = min(idx, nh)
            xs = np.linspace(0, LV, nv + 1)
            m, s = mean[:, col_i], std[:, col_i]
            xlabel = "z (m)"
            phys   = f"x = {idx * LH / max(nh, 1):.2f} m"
            if not truth_plotted and "ch_true" in data:
                ax.plot(xs, data["ch_true"][:, col_i],
                        "k--", lw=LINE_WIDTH, label="Ground truth", zorder=3)
                truth_plotted = True
        else:
            row_i = min(idx, nv)
            xs = np.linspace(0, LH, nh + 1)
            m, s = mean[row_i, :], std[row_i, :]
            xlabel = "x (m)"
            phys   = f"z = {idx * LV / max(nv, 1):.2f} m"
            if not truth_plotted and "ch_true" in data:
                ax.plot(xs, data["ch_true"][row_i, :],
                        "k--", lw=LINE_WIDTH, label="Ground truth", zorder=3)
                truth_plotted = True

        ax.plot(xs, m, color=col, lw=LINE_WIDTH, label=lbl, zorder=2)
        ax.fill_between(xs, m - band_k * s, m + band_k * s,
                        color=col, alpha=CONFIDENCE_ALPHA, zorder=1)

    title = f"C field — {phys}  [band = mean ± {band_k:.1f}σ]"
    _set_axis_labels(ax, xlabel, r"$C$  (m²/year)", title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    return fig


def plot_u_temporal(runs_data, labels, pt_idx, colors=None,
                    band_k=BAND_K, save_path=None):
    """Plot U temporal evolution (mean ± band_k·std) at monitor point *pt_idx*.

    Reads from pre-saved ``u_temporal_mean`` / ``u_temporal_std`` when available.
    Falls back to computing mean/std from the ``u_temporal_all`` ensemble array.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_XSEC, dpi=DPI)
    if colors is None:
        colors = _run_colors(len(runs_data))

    truth_plotted = False
    pt_info = ""
    any_data = False
    for data, lbl, col in zip(runs_data, labels, colors):
        t_coords = data.get("t_coords")

        # Prefer precomputed mean/std; fall back to ensemble array
        u_mean = data.get("u_temporal_mean")   # (n_pts, numt+1)
        u_std  = data.get("u_temporal_std")    # (n_pts, numt+1)
        if u_mean is None:
            u_all = data.get("u_temporal_all")  # (N_mc, n_pts, numt+1)
            if u_all is not None:
                u_mean = np.mean(u_all, axis=0)
                u_std  = np.std( u_all, axis=0)

        if u_mean is None or t_coords is None:
            print(f"  [WARNING] No U temporal data for '{lbl}'.")
            print(f"            (re-run with current run_batch.py to generate"
                  f" u_temporal_mean.npy / batch_stats.npz)")
            continue
        if u_std is None:
            u_std = np.zeros_like(u_mean)

        n_pts, numt1 = u_mean.shape
        if pt_idx >= n_pts:
            continue

        t = t_coords[:numt1]
        m = u_mean[pt_idx, :]
        s = u_std[pt_idx,  :]
        any_data = True

        if not truth_plotted and "u_true_temporal" in data:
            ut = data["u_true_temporal"]
            if pt_idx < ut.shape[0]:
                ax.plot(t, ut[pt_idx, :min(numt1, ut.shape[1])], "k--", lw=LINE_WIDTH,
                        label="Ground truth", zorder=3)
                truth_plotted = True

        ax.plot(t, m, color=col, lw=LINE_WIDTH, label=lbl, zorder=2)
        ax.fill_between(t, m - band_k * s, m + band_k * s,
                        color=col, alpha=CONFIDENCE_ALPHA, zorder=1)

        if not pt_info and "monitor_points" in data and pt_idx < len(data["monitor_points"]):
            r, c = data["monitor_points"][pt_idx]
            pt_info = f"  (row={r}, col={c})"

    if not any_data:
        print(f"  [WARNING] No data to plot for U temporal panel (pt_idx={pt_idx}).")

    title = f"u temporal evolution{pt_info}  [band = mean ± {band_k:.1f}σ]"
    _set_axis_labels(ax, "t (year)", "u  (pore pressure)", title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    return fig


def plot_u_spatial(runs_data, labels, t_idx, axis, fix_idx,
                   colors=None, band_k=BAND_K, save_path=None):
    """Plot U spatial cross section (mean ± band_k·std) at fixed timestep.

    Reads from pre-saved ``u_snapshots_mean`` / ``u_snapshots_std`` when
    available.  Falls back to computing mean/std from ``u_snapshots_all``.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_XSEC, dpi=DPI)
    if colors is None:
        colors = _run_colors(len(runs_data))

    truth_plotted = False
    t_yr_str = ""
    any_data = False
    for data, lbl, col in zip(runs_data, labels, colors):
        snap_years = data.get("snapshot_years")
        t_coords   = data.get("t_coords")

        # Prefer precomputed mean/std; fall back to ensemble array
        u_snap_mean = data.get("u_snapshots_mean")  # (n_snaps, nv+1, nh+1)
        u_snap_std  = data.get("u_snapshots_std")   # (n_snaps, nv+1, nh+1)
        if u_snap_mean is None:
            u_snaps = data.get("u_snapshots_all")   # (N_mc, n_snaps, nv+1, nh+1)
            if u_snaps is not None:
                u_snap_mean = np.mean(u_snaps, axis=0)
                u_snap_std  = np.std( u_snaps, axis=0)

        if u_snap_mean is None:
            print(f"  [WARNING] No U snapshot data for '{lbl}'.")
            print(f"            (re-run with current run_batch.py to generate"
                  f" u_snapshots_mean.npy / batch_stats.npz)")
            continue
        if u_snap_std is None:
            u_snap_std = np.zeros_like(u_snap_mean)

        # Find snapshot index closest to t_idx
        n_snaps = u_snap_mean.shape[0]
        if snap_years is not None and t_coords is not None and len(t_coords) > 1:
            dt = t_coords[1] - t_coords[0]
            snap_t_idxs = [int(round(y / dt)) for y in snap_years]
            snap_i = min(range(len(snap_t_idxs)),
                         key=lambda i: abs(snap_t_idxs[i] - t_idx))
        else:
            snap_i = min(t_idx, n_snaps - 1)

        nv = u_snap_mean.shape[1] - 1
        nh = u_snap_mean.shape[2] - 1
        any_data = True

        if axis == "x":
            row_i  = min(fix_idx, nv)
            xs     = np.linspace(0, LH, nh + 1)
            m      = u_snap_mean[snap_i, row_i, :]
            s      = u_snap_std[ snap_i, row_i, :]
            xlabel = "x (m)"
            if not truth_plotted and "u_true_snapshots" in data:
                us = data["u_true_snapshots"]
                if snap_i < us.shape[0]:
                    ax.plot(xs, us[snap_i, row_i, :],
                            "k--", lw=LINE_WIDTH, label="Ground truth", zorder=3)
                    truth_plotted = True
        else:
            col_i  = min(fix_idx, nh)
            xs     = np.linspace(0, LV, nv + 1)
            m      = u_snap_mean[snap_i, :, col_i]
            s      = u_snap_std[ snap_i, :, col_i]
            xlabel = "z (m)"
            if not truth_plotted and "u_true_snapshots" in data:
                us = data["u_true_snapshots"]
                if snap_i < us.shape[0]:
                    ax.plot(xs, us[snap_i, :, col_i],
                            "k--", lw=LINE_WIDTH, label="Ground truth", zorder=3)
                    truth_plotted = True

        ax.plot(xs, m, color=col, lw=LINE_WIDTH, label=lbl, zorder=2)
        ax.fill_between(xs, m - band_k * s, m + band_k * s,
                        color=col, alpha=CONFIDENCE_ALPHA, zorder=1)
        if not t_yr_str and snap_years is not None:
            t_yr_str = f"  (t ≈ {float(snap_years[snap_i]):.3f} yr)"

    if not any_data:
        print(f"  [WARNING] No data to plot for U spatial panel.")

    title = f"u spatial cross section{t_yr_str}  [band = mean ± {band_k:.1f}σ]"
    _set_axis_labels(ax, xlabel, "u  (pore pressure)", title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Set-B helpers: parallel field comparison
# ---------------------------------------------------------------------------

def _field_error(truth, pred, mode="log_ratio"):
    """Compute the 3rd-panel error field."""
    if mode == "log_ratio":
        safe_truth = np.where(np.abs(truth) < 1e-12, 1e-12, truth)
        safe_pred  = np.where(np.abs(pred)  < 1e-12, 1e-12, pred)
        return np.log(np.abs(safe_truth) / np.abs(safe_pred))
    return truth - pred


def _add_colorbar(ax, im, pad="5%"):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=pad, pad=0.08)
    plt.colorbar(im, cax=cax)


def _imshow_field(ax, data_2d, norm, cmap, lh, lv, title=""):
    """Show a 2D field with consistent tick labels."""
    nv, nh = data_2d.shape[0] - 1, data_2d.shape[1] - 1
    xticks  = [0, nh // 4, nh // 2, 3 * nh // 4, nh]
    xtlabel = [f"{v:.1f}" for v in np.array(xticks) * lh / max(nh, 1)]
    zticks  = [0, nv // 4, nv // 2, 3 * nv // 4, nv]
    ztlabel = [f"{v:.1f}" for v in np.array(zticks) * lv / max(nv, 1)]

    im = ax.imshow(data_2d, cmap=cmap, norm=norm,
                   origin="upper", aspect="auto", interpolation="bicubic")
    ax.set_xticks(xticks);  ax.set_xticklabels(xtlabel, fontsize=7)
    ax.set_yticks(zticks);  ax.set_yticklabels(ztlabel, fontsize=7)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("z (m)", fontsize=8)
    ax.set_title(title, fontsize=9, pad=3)
    return im


def plot_c_parallel(runs_data, labels, save_path=None):
    """
    Parallel C-field comparison.

    Layout: n_runs rows × 3 columns
      col 0: Ground truth  (same for all rows but repeated for readability)
      col 1: Mean prediction
      col 2: Error / log(truth/pred)

    All rows share the same colour scale for cols 0 & 1, and the same
    symmetric scale for col 2.
    """
    n = len(runs_data)
    if n == 0:
        return

    # Collect fields
    truths, means, errors = [], [], []
    for data, lbl in zip(runs_data, labels):
        ch_true = data.get("ch_true")
        ch_mean = data.get("ch_est_mean")
        if ch_true is None or ch_mean is None:
            print(f"  [WARNING] Missing C fields for '{lbl}' – skipping row.")
            truths.append(None); means.append(None); errors.append(None)
            continue
        err = _field_error(ch_true, ch_mean, ERROR_MODE)
        truths.append(ch_true)
        means.append(ch_mean)
        errors.append(err)

    # Global colour limits
    all_c = [v for v in truths + means if v is not None]
    vmin_c = min(a.min() for a in all_c)
    vmax_c = max(a.max() for a in all_c)
    all_e  = [v for v in errors if v is not None]
    vabs_e = max(abs(a).max() for a in all_e) if all_e else 1.0

    norm_c = mcolors.Normalize(vmin=vmin_c, vmax=vmax_c)
    norm_e = mcolors.Normalize(vmin=-vabs_e, vmax=vabs_e)

    fig, axes = plt.subplots(
        n, 3,
        figsize=(3 * 4.5, n * 3.6),
        constrained_layout=True,
        dpi=DPI,
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    err_title = "log(truth / pred)" if ERROR_MODE == "log_ratio" else "truth − pred"

    for row, (data, lbl, truth, mean, err) in enumerate(
            zip(runs_data, labels, truths, means, errors)):
        if truth is None:
            for col in range(3):
                axes[row, col].set_visible(False)
            continue

        im0 = _imshow_field(axes[row, 0], truth, norm_c, CMAP_FIELD,
                            LH, LV, title="Ground truth" if row == 0 else "")
        im1 = _imshow_field(axes[row, 1], mean,  norm_c, CMAP_FIELD,
                            LH, LV, title="Mean prediction" if row == 0 else "")
        im2 = _imshow_field(axes[row, 2], err,   norm_e, CMAP_ERROR,
                            LH, LV, title=err_title if row == 0 else "")

        axes[row, 0].set_ylabel(f"{lbl}\nz (m)", fontsize=8)

        # Colourbar on last column only
        if row == n - 1:
            _add_colorbar(axes[row, 1], im1)
            _add_colorbar(axes[row, 2], im2)

    fig.suptitle("C field  —  basis comparison", fontsize=12, y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    return fig


def _find_snap_idx(snap_years_arr, target_yr):
    """Return the index in snap_years_arr closest to target_yr."""
    if snap_years_arr is None or len(snap_years_arr) == 0:
        return 0
    diffs = np.abs(np.array(snap_years_arr) - target_yr)
    return int(np.argmin(diffs))


def plot_u_parallel(runs_data, labels, snap_year: float, save_path=None):
    """
    Parallel U-field comparison for one snapshot year.

    Layout: n_runs rows × 3 columns
      col 0: Ground truth u snapshot
      col 1: Mean estimated u snapshot
      col 2: Error
    """
    n = len(runs_data)
    if n == 0:
        return

    truths, means, errors, t_labels = [], [], [], []
    for data, lbl in zip(runs_data, labels):
        u_true_snaps  = data.get("u_true_snapshots")    # (n_snaps, nv+1, nh+1)
        u_snaps_mean  = data.get("u_snapshots_mean")    # (n_snaps, nv+1, nh+1)
        snap_years    = data.get("snapshot_years")

        # Fall back to computing mean from per-MC ensemble if pre-saved mean absent
        if u_snaps_mean is None:
            u_snaps_all = data.get("u_snapshots_all")   # (N_mc, n_snaps, nv+1, nh+1)
            if u_snaps_all is not None:
                u_snaps_mean = np.mean(u_snaps_all, axis=0)

        if u_true_snaps is None or u_snaps_mean is None:
            print(f"  [WARNING] Missing U snapshots for '{lbl}' – skipping row.")
            truths.append(None); means.append(None)
            errors.append(None); t_labels.append(lbl)
            continue

        snap_i = _find_snap_idx(snap_years, snap_year)
        truth  = u_true_snaps[snap_i]       # (nv+1, nh+1)
        mean   = u_snaps_mean[snap_i]       # (nv+1, nh+1)
        err    = _field_error(truth, mean, ERROR_MODE)

        t_yr_actual = float(snap_years[snap_i]) if snap_years is not None else snap_year
        truths.append(truth);  means.append(mean);  errors.append(err)
        t_labels.append(f"{lbl}\n(t≈{t_yr_actual:.2f} yr)")

    all_u = [v for v in truths + means if v is not None]
    if not all_u:
        return
    vmin_u = min(a.min() for a in all_u)
    vmax_u = max(a.max() for a in all_u)
    all_e  = [v for v in errors if v is not None]
    vabs_e = max(abs(a).max() for a in all_e) if all_e else 1.0

    norm_u = mcolors.Normalize(vmin=vmin_u, vmax=vmax_u)
    norm_e = mcolors.Normalize(vmin=-vabs_e, vmax=vabs_e)

    fig, axes = plt.subplots(
        n, 3,
        figsize=(3 * 4.5, n * 3.6),
        constrained_layout=True,
        dpi=DPI,
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    err_title = "log(truth / pred)" if ERROR_MODE == "log_ratio" else "truth − pred"

    for row, (truth, mean, err, tlbl) in enumerate(
            zip(truths, means, errors, t_labels)):
        if truth is None:
            for col in range(3):
                axes[row, col].set_visible(False)
            continue

        im0 = _imshow_field(axes[row, 0], truth, norm_u, CMAP_FIELD,
                            LH, LV, title="Ground truth" if row == 0 else "")
        im1 = _imshow_field(axes[row, 1], mean,  norm_u, CMAP_FIELD,
                            LH, LV, title="Mean prediction" if row == 0 else "")
        im2 = _imshow_field(axes[row, 2], err,   norm_e, CMAP_ERROR,
                            LH, LV, title=err_title if row == 0 else "")

        axes[row, 0].set_ylabel(tlbl, fontsize=8)

        if row == n - 1:
            _add_colorbar(axes[row, 1], im1)
            _add_colorbar(axes[row, 2], im2)

    fig.suptitle(f"u field  (t ≈ {snap_year:.2f} yr)  —  basis comparison",
                 fontsize=12, y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Scanning '{RESULTS_DIR}' for matching runs …")
    print(f"Confidence band: mean ± {BAND_K:.1f}σ  (BAND_K = {BAND_K})")
    run_dirs, labels = find_runs(RESULTS_DIR, GLOB_PATTERNS, LABELS)
    if not run_dirs:
        print("ERROR: No matching run directories found.  "
              "Check RESULTS_DIR and GLOB_PATTERNS.")
        return

    print(f"Found {len(run_dirs)} run(s):")
    for d, lbl in zip(run_dirs, labels):
        print(f"  {lbl:35s}  →  {os.path.basename(d)}")

    runs_data = [load_run_data(d) for d in run_dirs]
    colors    = _run_colors(len(runs_data))

    print("\n--- Set A: cross-section / temporal plots ---")

    # A1: C cross section fix-x, vary-z
    plot_c_xsec(
        runs_data, labels, idx=C_XSEC_X_IDX, axis="x",
        colors=colors, band_k=BAND_K,
        save_path=os.path.join(OUTPUT_DIR, "basis_C_xsec_fixedX.png"),
    )

    # A2: C cross section fix-z, vary-x
    plot_c_xsec(
        runs_data, labels, idx=C_XSEC_Z_IDX, axis="z",
        colors=colors, band_k=BAND_K,
        save_path=os.path.join(OUTPUT_DIR, "basis_C_xsec_fixedZ.png"),
    )

    # A3 & A4: U-field
    if U_MODE == "temporal":
        plot_u_temporal(
            runs_data, labels, pt_idx=U_MONITOR_PT_IDX_1,
            colors=colors, band_k=BAND_K,
            save_path=os.path.join(OUTPUT_DIR, "basis_U_temporal_pt0.png"),
        )
        plot_u_temporal(
            runs_data, labels, pt_idx=U_MONITOR_PT_IDX_2,
            colors=colors, band_k=BAND_K,
            save_path=os.path.join(OUTPUT_DIR, "basis_U_temporal_pt1.png"),
        )
    else:
        plot_u_spatial(
            runs_data, labels,
            t_idx=U_SPATIAL_T_IDX, axis=U_SPATIAL_AXIS_1, fix_idx=U_SPATIAL_FIX_1,
            colors=colors, band_k=BAND_K,
            save_path=os.path.join(OUTPUT_DIR, "basis_U_spatial_line1.png"),
        )
        plot_u_spatial(
            runs_data, labels,
            t_idx=U_SPATIAL_T_IDX, axis=U_SPATIAL_AXIS_2, fix_idx=U_SPATIAL_FIX_2,
            colors=colors, band_k=BAND_K,
            save_path=os.path.join(OUTPUT_DIR, "basis_U_spatial_line2.png"),
        )

    print("\n--- Set B: parallel field comparison ---")

    # B1: C field parallel plot
    plot_c_parallel(
        runs_data, labels,
        save_path=os.path.join(OUTPUT_DIR, "basis_C_parallel.png"),
    )

    # B2: U field parallel plots – one file per snapshot year
    for yr in U_SNAP_YEARS_TO_PLOT:
        yr_str = f"{yr:.2f}".replace(".", "p")
        plot_u_parallel(
            runs_data, labels, snap_year=yr,
            save_path=os.path.join(OUTPUT_DIR, f"basis_U_parallel_t{yr_str}yr.png"),
        )

    print(f"\nDone.  Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
