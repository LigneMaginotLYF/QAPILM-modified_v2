#!/usr/bin/env python3
"""
tools/plot_epsilon_batch.py
===========================
Compare batch experiment results across different epsilon-loss threshold
settings.  Reads multiple run-output directories produced by ``run_batch.py``
and superimposes confidence bands (mean ± k·std across MC ensemble) for:

  - Two C-field cross sections  (vertical slice: x = const,
                                  horizontal slice: z = const)
  - Two U-field panels  (temporal evolution at a fixed spatial point  OR
                          spatial line at a fixed timestep — set by U_MODE)

Mean and confidence band
------------------------
For each run the **mean estimate** is the pointwise mean of the C (or U)
field over all MC realizations::

    mean(x) = (1/N_mc) * sum_i  field_i(x)

The **confidence band** is mean ± k·std, where *k* is set by ``BAND_K``
(default 1.0 → 68 % interval for approximately Gaussian ensembles)::

    lower(x) = mean(x) - BAND_K * std(x)
    upper(x) = mean(x) + BAND_K * std(x)

Usage
-----
1.  Edit the ``TUNABLE PARAMETERS`` block below (results folder, glob patterns,
    labels, cross-section indices, U-field mode, band width, etc.).
2.  Run::

        python tools/plot_epsilon_batch.py

Output files are written to ``OUTPUT_DIR`` (default ``./plots``).
"""

import os
import json
import glob as _glob
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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
RESULTS_DIR   = "./results"        # parent folder that contains run sub-dirs

# Each entry in GLOB_PATTERNS is matched (in order) against sub-folders of
# RESULTS_DIR.  The most-recent match for each pattern is used.
# Use "*" as a wildcard, e.g. "eps_0.05_*" matches any timestamped variant.
GLOB_PATTERNS = [
    "eps_0.05_*",
    "baseline_*",
    "eps_0.20_*",
]

# Legend labels – one entry per pattern (same order)
LABELS = [
    "ε = 0.05",
    "ε = 0.10  (baseline)",
    "ε = 0.20",
]

# --- Domain physical size (metres / years) -----------------------------------
# These are used only for axis tick labels.  They do not affect the data loaded.
LH = 10.0      # horizontal domain length (x direction)
LV = 5.0       # vertical   domain length (z direction)

# --- C-field cross-section locations (grid indices) --------------------------
C_XSEC_X_IDX = 25    # column index (x = const slice) for C cross section
C_XSEC_Z_IDX = 10    # row index    (z = const slice) for C cross section

# --- U-field mode ------------------------------------------------------------
# "temporal" : plot u(x0, z0, t) over time at each monitor point stored in
#              the run directory (saved by run_batch.py as u_temporal_all.npy).
# "spatial"  : plot u along a spatial line at a fixed timestep T_IDX.
U_MODE = "temporal"

# For "temporal" mode: index into the monitor_points array saved by run_batch.
U_MONITOR_PT_IDX_1 = 0     # first U temporal panel
U_MONITOR_PT_IDX_2 = 1     # second U temporal panel

# For "spatial" mode:
U_SPATIAL_T_IDX  = 500     # time-step index for the spatial snapshot
U_SPATIAL_AXIS_1 = "x"    # "x" (vary x, fix z) or "z" (vary z, fix x) – panel 3
U_SPATIAL_FIX_1  = 10     # fixed row (z mode) or column (x mode) – panel 3
U_SPATIAL_AXIS_2 = "z"    # panel 4
U_SPATIAL_FIX_2  = 25     # panel 4

# --- Confidence band ----------------------------------------------------------
# Confidence band = mean ± BAND_K * std  at each grid / time point.
# BAND_K = 1.0 → ±1σ → nominally 68 % interval for Gaussian ensembles.
# BAND_K = 2.0 → ±2σ → nominally 95 % interval.
BAND_K = 1.0

# --- Plot style ---------------------------------------------------------------
CONFIDENCE_ALPHA = 0.25    # opacity of the confidence-band fill
LINE_WIDTH       = 1.8
COLORMAP         = "tab10" # matplotlib colormap for per-run colours
FIGSIZE          = (7, 4.5)
DPI              = 150

# --- Output directory ---------------------------------------------------------
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
# Data-loading utilities
# ---------------------------------------------------------------------------

def _try_load(run_dir: str, *filenames):
    """Try loading the first existing file from *filenames* in *run_dir*.
    Returns a numpy array or None."""
    for fn in filenames:
        path = os.path.join(run_dir, fn)
        if os.path.exists(path):
            try:
                arr = np.load(path, allow_pickle=False)
                return arr
            except Exception as exc:
                print(f"  [WARNING] Failed to load '{path}': {exc}")
    return None


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


def _load_from_precomputed_stats(run_dir: str):
    """Load pre-saved mean/std statistics from batch_stats.npz or individual files.

    This is the primary loading path introduced by the batch-analysis outputs PR.
    ``run_batch.py`` now saves a ``batch_stats.npz`` bundle containing::

        ch_true, ch_est_mean, ch_est_std,
        u_temporal_mean, u_temporal_std, u_true_temporal,
        u_snapshots_mean, u_snapshots_std, u_true_snapshots,
        x_coords, z_coords, t_coords, snapshot_years, monitor_points

    If the bundle is absent, fall back to loading individual ``*_mean.npy`` /
    ``*_std.npy`` files saved alongside it.

    Returns
    -------
    dict  (same schema as the rest of load_run_data) or None if nothing found.
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
    # Also expose mean/std under the new canonical key names
    return {
        "ch_true": solver.chm,
        "ch_est_all": ch_all,
        "ch_est_mean": np.mean(ch_all, axis=0),
        "ch_est_std": np.std(ch_all, axis=0),
        "u_est_all": np.array(u_est_all),
        "u_temporal_all": np.array(u_temporal_all),
        "u_temporal_mean": np.mean(np.array(u_temporal_all), axis=0),
        "u_temporal_std":  np.std( np.array(u_temporal_all), axis=0),
        "u_snapshots_all": np.array(u_snapshots_all),
        "u_snapshots_mean": np.mean(np.array(u_snapshots_all), axis=0),
        "u_snapshots_std":  np.std( np.array(u_snapshots_all), axis=0),
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

    Returns
    -------
    dict with keys (all optional; only present when the file was found):
        ch_true           : (nv+1, nh+1)
        ch_est_mean       : (nv+1, nh+1)         – mean C field across MC
        ch_est_std        : (nv+1, nh+1)         – std  C field across MC
        t_coords          : (numt+1,)             – physical time in years
        u_temporal_mean   : (n_pts, numt+1)       – mean U at monitor pts
        u_temporal_std    : (n_pts, numt+1)       – std  U at monitor pts
        u_true_temporal   : (n_pts, numt+1)
        u_snapshots_mean  : (n_snaps, nv+1, nh+1) – mean U at snapshot years
        u_snapshots_std   : (n_snaps, nv+1, nh+1) – std  U at snapshot years
        u_true_snapshots  : (n_snaps, nv+1, nh+1)
        snapshot_years    : (n_snaps,)
        monitor_points    : (n_pts, 2)
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
    """Match glob patterns against sub-directories; return (dirs, labels)."""
    found_dirs, found_labels = [], []
    for pattern, label in zip(patterns, labels):
        matches = sorted(_glob.glob(os.path.join(results_dir, pattern)))
        if matches:
            found_dirs.append(matches[-1])      # most recent match
            found_labels.append(label)
        else:
            print(f"  [WARNING] No run matched pattern '{pattern}' – skipping.")
    return found_dirs, found_labels


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _run_colors(n: int):
    cmap = get_cmap(COLORMAP)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _set_axis_labels(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title,   fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)


def plot_c_xsec(runs_data, labels, idx: int, axis: str,
                colors=None, lh=LH, lv=LV, band_k=BAND_K, save_path=None):
    """Plot C-field cross section (mean ± band_k·std) for all runs, superposed.

    Parameters
    ----------
    idx    : grid row (axis="x" → vary z) or grid column (axis="z" → vary x)
    axis   : "x" = fix x, vary z;  "z" = fix z, vary x
    band_k : confidence-band multiplier (default BAND_K = 1.0 → ±1σ)
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    if colors is None:
        colors = _run_colors(len(runs_data))

    truth_plotted = False
    location_label = ""
    for data, lbl, col in zip(runs_data, labels, colors):
        if "ch_est_mean" not in data:
            continue
        mean = data["ch_est_mean"]       # (nv+1, nh+1)
        std  = data.get("ch_est_std", np.zeros_like(mean))
        nv, nh = mean.shape[0] - 1, mean.shape[1] - 1

        if axis == "x":
            col_i = min(idx, nh)
            xs  = np.linspace(0, lv, nv + 1)
            m   = mean[:, col_i]
            s   = std[:, col_i]
            xlabel = "z (m)"
            location_label = f"x = {idx * lh / max(nh, 1):.2f} m"
            # ground truth (same for all runs → plot once)
            if not truth_plotted and "ch_true" in data:
                ax.plot(xs, data["ch_true"][:, col_i],
                        "k--", lw=LINE_WIDTH, label="Ground truth", zorder=3)
                truth_plotted = True
        else:
            row_i = min(idx, nv)
            xs  = np.linspace(0, lh, nh + 1)
            m   = mean[row_i, :]
            s   = std[row_i, :]
            xlabel = "x (m)"
            location_label = f"z = {idx * lv / max(nv, 1):.2f} m"
            if not truth_plotted and "ch_true" in data:
                ax.plot(xs, data["ch_true"][row_i, :],
                        "k--", lw=LINE_WIDTH, label="Ground truth", zorder=3)
                truth_plotted = True

        ax.plot(xs, m, color=col, lw=LINE_WIDTH, label=lbl, zorder=2)
        ax.fill_between(xs, m - band_k * s, m + band_k * s,
                        color=col, alpha=CONFIDENCE_ALPHA, zorder=1)

    # Add band-width note to title
    title = f"C field — {location_label}  [band = mean ± {band_k:.1f}σ]"
    _set_axis_labels(ax, xlabel, r"$C$  (m²/year)", title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    return fig


def plot_u_temporal(runs_data, labels, pt_idx: int,
                    colors=None, band_k=BAND_K, save_path=None):
    """Plot U temporal evolution (mean ± band_k·std) at monitor point *pt_idx*.

    Reads from pre-saved ``u_temporal_mean`` / ``u_temporal_std`` when available
    (written by run_batch.py as ``u_temporal_mean.npy`` / ``u_temporal_std.npy``
    or inside ``batch_stats.npz``).  Falls back to computing mean/std from the
    ``u_temporal_all`` ensemble array if only the legacy per-MC files exist.

    Parameters
    ----------
    pt_idx : index into the monitor_points array saved by run_batch.py
    band_k : confidence-band multiplier (default BAND_K = 1.0 → ±1σ)
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
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
            print(f"  [WARNING] pt_idx={pt_idx} ≥ n_pts={n_pts} for '{lbl}' – skipping.")
            continue

        t = t_coords[:numt1]
        m = u_mean[pt_idx, :]
        s = u_std[pt_idx,  :]
        any_data = True

        # Ground truth temporal trace (same for all runs → plot once)
        if not truth_plotted and "u_true_temporal" in data:
            ut = data["u_true_temporal"]           # (n_pts, numt+1)
            if pt_idx < ut.shape[0]:
                ax.plot(t, ut[pt_idx, :min(numt1, ut.shape[1])], "k--", lw=LINE_WIDTH,
                        label="Ground truth", zorder=3)
                truth_plotted = True

        ax.plot(t, m, color=col, lw=LINE_WIDTH, label=lbl, zorder=2)
        ax.fill_between(t, m - band_k * s, m + band_k * s,
                        color=col, alpha=CONFIDENCE_ALPHA, zorder=1)

        # Build info string from monitor_points
        if not pt_info and "monitor_points" in data:
            mp = data["monitor_points"]
            if pt_idx < len(mp):
                r, c = mp[pt_idx]
                pt_info = f"  (grid row={r}, col={c})"

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


def plot_u_spatial(runs_data, labels, t_idx: int, axis: str, fix_idx: int,
                   colors=None, lh=LH, lv=LV, band_k=BAND_K, save_path=None):
    """Plot U spatial cross section (mean ± band_k·std) at fixed timestep *t_idx*.

    Reads from pre-saved ``u_snapshots_mean`` / ``u_snapshots_std`` when
    available.  Falls back to computing mean/std from ``u_snapshots_all``.

    Parameters
    ----------
    t_idx  : time-step index used to find the nearest saved snapshot
    axis   : "x" = vary x at fixed z-row; "z" = vary z at fixed x-column
    fix_idx: fixed row (axis="x") or column (axis="z") index
    band_k : confidence-band multiplier (default BAND_K = 1.0 → ±1σ)
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
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
            xs     = np.linspace(0, lh, nh + 1)
            m      = u_snap_mean[snap_i, row_i, :]
            s      = u_snap_std[ snap_i, row_i, :]
            xlabel = "x (m)"
            if not truth_plotted and "u_true_snapshots" in data:
                us = data["u_true_snapshots"]      # (n_snaps, nv+1, nh+1)
                if snap_i < us.shape[0]:
                    ax.plot(xs, us[snap_i, row_i, :],
                            "k--", lw=LINE_WIDTH, label="Ground truth", zorder=3)
                    truth_plotted = True
        else:
            col_i  = min(fix_idx, nh)
            xs     = np.linspace(0, lv, nv + 1)
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

    # -------------------------------------------------------------------------
    # Panel 1: C cross section – fix x, vary z
    # -------------------------------------------------------------------------
    plot_c_xsec(
        runs_data, labels, idx=C_XSEC_X_IDX, axis="x",
        colors=colors, band_k=BAND_K,
        save_path=os.path.join(OUTPUT_DIR, "eps_C_xsec_fixedX.png"),
    )

    # -------------------------------------------------------------------------
    # Panel 2: C cross section – fix z, vary x
    # -------------------------------------------------------------------------
    plot_c_xsec(
        runs_data, labels, idx=C_XSEC_Z_IDX, axis="z",
        colors=colors, band_k=BAND_K,
        save_path=os.path.join(OUTPUT_DIR, "eps_C_xsec_fixedZ.png"),
    )

    # -------------------------------------------------------------------------
    # Panels 3 & 4: U-field
    # -------------------------------------------------------------------------
    if U_MODE == "temporal":
        plot_u_temporal(
            runs_data, labels, pt_idx=U_MONITOR_PT_IDX_1,
            colors=colors, band_k=BAND_K,
            save_path=os.path.join(OUTPUT_DIR, "eps_U_temporal_pt0.png"),
        )
        plot_u_temporal(
            runs_data, labels, pt_idx=U_MONITOR_PT_IDX_2,
            colors=colors, band_k=BAND_K,
            save_path=os.path.join(OUTPUT_DIR, "eps_U_temporal_pt1.png"),
        )
    else:   # "spatial"
        plot_u_spatial(
            runs_data, labels,
            t_idx=U_SPATIAL_T_IDX, axis=U_SPATIAL_AXIS_1, fix_idx=U_SPATIAL_FIX_1,
            colors=colors, band_k=BAND_K,
            save_path=os.path.join(OUTPUT_DIR, "eps_U_spatial_line1.png"),
        )
        plot_u_spatial(
            runs_data, labels,
            t_idx=U_SPATIAL_T_IDX, axis=U_SPATIAL_AXIS_2, fix_idx=U_SPATIAL_FIX_2,
            colors=colors, band_k=BAND_K,
            save_path=os.path.join(OUTPUT_DIR, "eps_U_spatial_line2.png"),
        )

    print(f"\nDone.  Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
