"""
run_batch.py
============
Batch sweep runner for QAPILM rectangular solver.

Usage
-----
    python run_batch.py                            # uses config.yaml + sweep.yaml in cwd
    python run_batch.py --config my_config.yaml --sweep my_sweep.yaml
    python run_batch.py --run baseline eps_0.05   # run only named sweep entries

Required packages (beyond standard library + numpy/matplotlib/scipy already used
by qapilm_rect.py):
    pip install pyyaml

See README.md for full instructions.
"""

import argparse
import copy
import csv
import json
import os
import sys
import traceback
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required.  Install it with:  pip install pyyaml"
    )

import numpy as np

# ---------------------------------------------------------------------------
# Import solver components from qapilm_rect.py (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from qapilm_rect import (
    ProblemConfig,
    BasisConfig,
    ModelConfig,
    SolverConfig,
    RunConfig,
    RectangularQAPILM,
    softplus,
)


# ===========================================================================
# Config helpers
# ===========================================================================

def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict (UTF-8 encoding)."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into a deep-copy of *base*.
    Override values take precedence; nested dicts are merged, not replaced.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _cfg_to_dataclasses(cfg: dict):
    """
    Convert the merged YAML dict into the five config dataclasses expected
    by RectangularQAPILM.  Returns (ProblemConfig, BasisConfig, ModelConfig,
    SolverConfig, RunConfig).
    """
    p_raw  = cfg.get("problem",      {})
    b_raw  = cfg.get("basis",        {})
    m_raw  = cfg.get("model",        {})
    s_raw  = cfg.get("solver",       {})
    r_raw  = cfg.get("run",          {})
    meas   = cfg.get("measurements", {})

    # ProblemConfig
    pconf = ProblemConfig(
        Lh           = float(p_raw.get("Lh",           ProblemConfig.Lh)),
        Lv           = float(p_raw.get("Lv",           ProblemConfig.Lv)),
        Lt           = float(p_raw.get("Lt",           ProblemConfig.Lt)),
        spatial_reso = tuple(p_raw.get("spatial_reso", list(ProblemConfig.spatial_reso))),
        Rcv          = float(p_raw.get("Rcv",          ProblemConfig.Rcv)),
        sigma        = float(p_raw.get("sigma",        ProblemConfig.sigma)),
        coeft        = tuple(p_raw.get("coeft",        list(ProblemConfig.coeft))),
        u0           = float(p_raw.get("u0",           ProblemConfig.u0)),
        alpha        = float(p_raw.get("alpha",        ProblemConfig.alpha)),
        const        = float(p_raw.get("const",        ProblemConfig.const)),
        filedir      = str(p_raw.get("filedir",        ProblemConfig.filedir)),
        file_rawS    = str(p_raw.get("file_rawS",      ProblemConfig.file_rawS)),
        file_Lmat    = str(p_raw.get("file_Lmat",      ProblemConfig.file_Lmat)),
        bcs          = tuple(p_raw.get("bcs",          list(ProblemConfig.bcs))),
        regen_fluc   = bool(p_raw.get("regen_fluc",    ProblemConfig.regen_fluc)),
        fluc_seed    = int(p_raw.get("fluc_seed",      ProblemConfig.fluc_seed)),
    )

    # BasisConfig
    bconf = BasisConfig(
        type             = str(b_raw.get("type",             BasisConfig.type)),
        orderx           = int(b_raw.get("orderx",           BasisConfig.orderx)),
        orderz           = int(b_raw.get("orderz",           BasisConfig.orderz)),
        wav_levels_x     = int(b_raw.get("wav_levels_x",     BasisConfig.wav_levels_x)),
        wav_levels_z     = int(b_raw.get("wav_levels_z",     BasisConfig.wav_levels_z)),
        rbf_centers_x    = int(b_raw.get("rbf_centers_x",    BasisConfig.rbf_centers_x)),
        rbf_centers_z    = int(b_raw.get("rbf_centers_z",    BasisConfig.rbf_centers_z)),
        rbf_shape        = float(b_raw.get("rbf_shape",       BasisConfig.rbf_shape)),
        bspline_nknots_x = int(b_raw.get("bspline_nknots_x", BasisConfig.bspline_nknots_x)),
        bspline_nknots_z = int(b_raw.get("bspline_nknots_z", BasisConfig.bspline_nknots_z)),
        bspline_degree   = int(b_raw.get("bspline_degree",   BasisConfig.bspline_degree)),
    )

    # ModelConfig
    mconf = ModelConfig(
        lam     = float(m_raw.get("lam",     ModelConfig.lam)),
        lamu    = float(m_raw.get("lamu",    ModelConfig.lamu)),
        lr      = float(m_raw.get("lr",      ModelConfig.lr)),
        ltol    = float(m_raw.get("ltol",    ModelConfig.ltol)),
        gtol    = float(m_raw.get("gtol",    ModelConfig.gtol)),
        itol    = int(m_raw.get("itol",      ModelConfig.itol)),
        epsilon = float(m_raw.get("epsilon", ModelConfig.epsilon)),
    )

    # SolverConfig
    sconf = SolverConfig(
        memory_mode      = str(s_raw.get("memory_mode",      SolverConfig.memory_mode)),
        store_u_snapshots= bool(s_raw.get("store_u_snapshots", SolverConfig.store_u_snapshots)),
    )

    # RunConfig  (loss_file will be overridden per-run below)
    rconf = RunConfig(
        results_dir  = str(r_raw.get("results_dir",  RunConfig.results_dir)),
        save_losses  = bool(r_raw.get("save_losses", RunConfig.save_losses)),
        loss_file    = str(r_raw.get("loss_file",    RunConfig.loss_file)),
        coeffs_csv   = str(r_raw.get("coeffs_csv",  RunConfig.coeffs_csv)),
    )

    # Measurement locations
    ukmat  = [tuple(pt) for pt in meas.get("ukmat",  [])]
    chkmat = [tuple(pt) for pt in meas.get("chkmat", [])]
    ukt    = list(meas.get("ukt", []))

    return pconf, bconf, mconf, sconf, rconf, ukmat, chkmat, ukt


# ===========================================================================
# Randomization utilities
# ===========================================================================

def _rand_seed_or_none(seed: int):
    """Set numpy random seed if seed != 0; otherwise leave the RNG as-is."""
    if seed != 0:
        np.random.seed(seed)


def _replace_coeft(pconf: "ProblemConfig", new_coeft: tuple) -> "ProblemConfig":
    """Return a new ProblemConfig identical to *pconf* except for coeft."""
    d = asdict(pconf)
    d["coeft"] = new_coeft
    return ProblemConfig(**d)


def _randomize_locations(ukmat_idx, chkmat_idx, numv: int, numh: int):
    """
    Return new ukmat and chkmat with the same number of points as the inputs
    but at uniformly random grid positions (without replacement if possible).

    Parameters
    ----------
    ukmat_idx  : list of (row, col) tuples for u observations
    chkmat_idx : list of (row, col) tuples for C observations
    numv       : maximum row index (inclusive)
    numh       : maximum column index (inclusive)
    """
    def _sample(n, numv, numh):
        total_cells = (numv + 1) * (numh + 1)
        if n > total_cells:
            # More points requested than cells — allow duplicates
            rows = np.random.randint(0, numv + 1, size=n)
            cols = np.random.randint(0, numh + 1, size=n)
        else:
            flat = np.random.choice(total_cells, size=n, replace=False)
            rows = flat // (numh + 1)
            cols = flat %  (numh + 1)
        return [tuple(rc) for rc in zip(rows.tolist(), cols.tolist())]

    new_ukmat  = _sample(len(ukmat_idx),  numv, numh)
    new_chkmat = _sample(len(chkmat_idx), numv, numh)
    return new_ukmat, new_chkmat


# ===========================================================================
# Single-run executor
# ===========================================================================

def run_one(cfg: dict, run_name: str, base_results_dir: str, coeffs_csv_path: str):
    """
    Execute one experiment described by *cfg* (already merged base+override).

    Supports N_mc Monte Carlo runs: the inverse solver is restarted N_mc times
    with independent random initial coefficients.  Mean C and mean U fields are
    computed by decoding each run's weight vector and averaging in field space
    (weights are never averaged).  When N_mc == 1 the behaviour is identical to
    the original single-run output.

    Additional outputs (for batch plotting):
    - ch_est_all.npy         : (N_mc, nv+1, nh+1) all individual MC C fields
    - u_est_all.npy          : (N_mc, nv+1, nh+1) individual MC U snapshots
    - u_temporal_all.npy     : (N_mc, n_pts, numt+1) U at monitor points over time
    - u_snapshots_all.npy    : (N_mc, n_snaps, nv+1, nh+1) U at snapshot years
    - u_true_temporal.npy    : (n_pts, numt+1) truth U at monitor points
    - u_true_snapshots.npy   : (n_snaps, nv+1, nh+1) truth U at snapshot years
    - t_coords.npy           : (numt+1,) physical time in years
    - snapshot_years.npy     : (n_snaps,) years corresponding to snapshot indices
    - monitor_points.npy     : (n_pts, 2) [row, col] monitor point indices

    Parameters
    ----------
    cfg               : merged configuration dict
    run_name          : short identifier for this sweep entry (used as folder name)
    base_results_dir  : parent results folder
    coeffs_csv_path   : path to the summary CSV file (appended after each run)
    """
    # Create per-run output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_results_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Number of Monte Carlo restarts (default 1 → original single-run behaviour)
    N_mc = max(1, int(cfg.get("monte_carlo", {}).get("N_mc", 1)))

    # ------------------------------------------------------------------
    # Randomize option  (default: none → backward compatible)
    # ------------------------------------------------------------------
    rand_raw  = cfg.get("randomize", {})
    rand_mode = str(rand_raw.get("mode", "none")).lower()  # "none"|"coeft"|"locations"
    rand_seed = int(rand_raw.get("seed", 0))

    # ------------------------------------------------------------------
    # Plot-output configuration (for batch plotting scripts)
    # ------------------------------------------------------------------
    plot_raw        = cfg.get("plot_output", {})
    snapshot_years  = list(plot_raw.get("snapshot_years", [0.1, 1.0, 2.0]))
    monitor_points  = [tuple(pt) for pt in plot_raw.get("monitor_points", [[10, 25], [5, 10]])]

    print(f"\n{'='*70}")
    print(f"  Starting run: {run_name}  →  {run_dir}")
    print(f"  Monte Carlo N_mc = {N_mc}")
    if rand_mode != "none":
        print(f"  Randomize mode   = {rand_mode}  (seed={rand_seed})")
    print(f"{'='*70}")

    # Build dataclass configs
    pconf, bconf, mconf, sconf, rconf, ukmat, chkmat, ukt_list = _cfg_to_dataclasses(cfg)

    # ------------------------------------------------------------------
    # Apply coeft randomization BEFORE building the solver
    # (solver.__init__ uses pconf.coeft to construct ground-truth C field)
    # ------------------------------------------------------------------
    if rand_mode == "coeft":
        _rand_seed_or_none(rand_seed)
        n_coeft = len(pconf.coeft)
        pconf = _replace_coeft(pconf, tuple(np.random.randn(n_coeft).tolist()))
        print(f"  Randomized coeft ({n_coeft} terms): {list(pconf.coeft)}")

    # Redirect per-run outputs into run_dir
    rconf.results_dir = run_dir
    # loss_file will be set per MC run inside the loop below
    rconf.loss_file   = os.path.join(run_dir, "QAPILM_Eps_loss.npy")

    # Convert measurement lists to numpy arrays / index tuples as expected by solver
    ukt_arr = np.array(ukt_list, dtype=int)
    ukmat_idx  = [tuple(pt) for pt in ukmat]
    chkmat_idx = [tuple(pt) for pt in chkmat]

    # ------------------------------------------------------------------
    # Instantiate solver (builds grid, loads/generates fluctuation field,
    # builds operator matrices, builds estimator bases)
    # ------------------------------------------------------------------
    solver = RectangularQAPILM(pconf, bconf, mconf, sconf, rconf)

    # ------------------------------------------------------------------
    # Apply location randomization AFTER building the solver
    # (we need numv/numh to clamp random indices to the grid)
    # ------------------------------------------------------------------
    if rand_mode == "locations":
        _rand_seed_or_none(rand_seed)
        ukmat_idx, chkmat_idx = _randomize_locations(
            ukmat_idx, chkmat_idx, solver.numv, solver.numh
        )
        print(f"  Randomized ukmat  ({len(ukmat_idx)} pts): {ukmat_idx}")
        print(f"  Randomized chkmat ({len(chkmat_idx)} pts): {chkmat_idx}")

    # Clamp monitor points to valid grid range
    monitor_pts_clamped = [
        (min(int(pt[0]), solver.numv), min(int(pt[1]), solver.numh))
        for pt in monitor_points
    ]

    # ------------------------------------------------------------------
    # Generate "ground truth" snapshot tensor via forward solver (once)
    # ------------------------------------------------------------------
    top, bot, left, right = pconf.bcs
    u0_mat = np.ones((solver.numv + 1, solver.numh + 1)) * pconf.u0
    numt   = solver.ilim

    print(f"  Running forward solver  (numt={numt}) …")
    utens, udeg = solver.forward_solver(solver.chm, numt, u0_mat, top, bot, left, right)

    # Time coordinate array (physical years)
    t_coords = np.arange(numt + 1) * solver.dt

    # Snapshot time indices (clamp to valid range)
    snap_t_idxs = [min(int(round(y / solver.dt)), numt) for y in snapshot_years]
    # Actual years corresponding to each index (may differ slightly due to rounding)
    snap_years_actual = [t_coords[i] for i in snap_t_idxs]

    # Extract truth temporal traces at monitor points: (n_pts, numt+1)
    u_true_temporal = np.array(
        [utens[:, r, c] for r, c in monitor_pts_clamped]
    )
    # Extract truth U snapshots at snapshot years: (n_snaps, nv+1, nh+1)
    u_true_snapshots = np.array([utens[t_i] for t_i in snap_t_idxs])

    # Clamp ukt indices to valid range
    ukt_arr = np.clip(ukt_arr, 0, numt)
    obs_last_t = int(np.max(ukt_arr))

    uk  = utens[ukt_arr, :, :]   # shape: (T, numv+1, numh+1)
    chk = solver.chm              # ground-truth permeability field

    # ------------------------------------------------------------------
    # Monte Carlo inverse loop
    # ------------------------------------------------------------------
    all_coefe      = []   # list of (nb,) arrays
    all_ch_est     = []   # list of (numv+1, numh+1) C fields
    all_u_est      = []   # list of (numv+1, numh+1) U snapshots at ukt_arr[-1]
    all_u_temporal = []   # list of (n_pts, numt+1) U temporal traces at monitor pts
    all_u_snapshots= []   # list of (n_snaps, numv+1, numh+1) U at snapshot years

    for mc_idx in range(N_mc):
        mc_label = f"MC {mc_idx + 1}/{N_mc}"
        print(f"  [{mc_label}] Running inverse solver  (epochs={mconf.itol}, ε={mconf.epsilon}) …")

        # Assign a per-run loss file; for N_mc == 1 keep the original name
        if N_mc == 1:
            rconf.loss_file = os.path.join(run_dir, "QAPILM_Eps_loss.npy")
        else:
            rconf.loss_file = os.path.join(run_dir, f"QAPILM_Eps_loss_mc{mc_idx:03d}.npy")

        if sconf.memory_mode == "stream":
            coefe = solver.inverse_solver_stream(
                ukt_arr, u0_mat, ukmat_idx, chkmat_idx, uk, chk
            )
        elif sconf.memory_mode == "full":
            coefe = solver.inverse_solver_full(
                ukt_arr, u0_mat, ukmat_idx, chkmat_idx, uk, chk
            )
        else:
            warnings.warn(
                f"Unknown memory_mode='{sconf.memory_mode}'; "
                "falling back to memory_mode='full' for backward compatibility.",
                stacklevel=2,
            )
            coefe = solver.inverse_solver_full(
                ukt_arr, u0_mat, ukmat_idx, chkmat_idx, uk, chk
            )

        # Decode weight vector → C field
        m_est  = solver.vec2mat2(solver.basese @ coefe)
        ch_est = softplus(m_est)

        # Run forward solver with estimated C to obtain estimated U field.
        utens_est, _ = solver.forward_solver(ch_est, numt, u0_mat, top, bot, left, right)
        # U snapshot at the last observation time
        u_est_snap = utens_est[obs_last_t].copy()

        # Extract temporal traces at monitor points: (n_pts, numt+1)
        u_temporal_run = np.array(
            [utens_est[:, r, c] for r, c in monitor_pts_clamped]
        )
        # Extract U snapshots at specified years: (n_snaps, nv+1, nh+1)
        u_snaps_run = np.array([utens_est[t_i] for t_i in snap_t_idxs])

        all_coefe.append(coefe.copy())
        all_ch_est.append(ch_est)
        all_u_est.append(u_est_snap)
        all_u_temporal.append(u_temporal_run)
        all_u_snapshots.append(u_snaps_run)

        print(f"  [{mc_label}] done")

    # After the loop rconf.loss_file points to the last MC run's loss file.
    # final_loss below reflects only that last iteration (representative scalar for CSV).

    # ------------------------------------------------------------------
    # Aggregate: mean and std fields (fields are aggregated, not weights)
    # ------------------------------------------------------------------
    mc_weights       = np.array(all_coefe)                         # (N_mc, nb)
    ch_est_arr       = np.array(all_ch_est)                        # (N_mc, nv+1, nh+1)
    u_temporal_arr   = np.array(all_u_temporal)                    # (N_mc, n_pts, numt+1)
    u_snapshots_arr  = np.array(all_u_snapshots)                   # (N_mc, n_snaps, nv+1, nh+1)
    u_est_arr        = np.array(all_u_est)                         # (N_mc, nv+1, nh+1)

    mean_ch_est      = np.mean(ch_est_arr,      axis=0)            # (nv+1, nh+1)
    std_ch_est       = np.std( ch_est_arr,      axis=0)            # (nv+1, nh+1)
    mean_u_est       = np.mean(u_est_arr,       axis=0)            # (nv+1, nh+1)
    u_temporal_mean  = np.mean(u_temporal_arr,  axis=0)            # (n_pts, numt+1)
    u_temporal_std   = np.std( u_temporal_arr,  axis=0)            # (n_pts, numt+1)
    u_snapshots_mean = np.mean(u_snapshots_arr, axis=0)            # (n_snaps, nv+1, nh+1)
    u_snapshots_std  = np.std( u_snapshots_arr, axis=0)            # (n_snaps, nv+1, nh+1)

    # Spatial grid coordinates (for plotting)
    x_coords = np.linspace(0.0, float(pconf.Lh), solver.numh + 1)  # (nh+1,)
    z_coords = np.linspace(0.0, float(pconf.Lv), solver.numv + 1)  # (nv+1,)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # Per-run weight vectors (all MC runs)
    np.save(os.path.join(run_dir, "mc_weights.npy"), mc_weights)
    # Mean C field (replaces single-run ch_est.npy)
    np.save(os.path.join(run_dir, "ch_est.npy"),     mean_ch_est)
    # Std of C field across MC realizations
    np.save(os.path.join(run_dir, "ch_est_std.npy"), std_ch_est)
    # Mean U field at last observation time
    np.save(os.path.join(run_dir, "u_mean_est.npy"), mean_u_est)
    # Ground-truth C field
    np.save(os.path.join(run_dir, "ch_true.npy"),    solver.chm)

    # Backward compat: for N_mc == 1 also write the original coefe.npy
    if N_mc == 1:
        np.save(os.path.join(run_dir, "coefe.npy"), mc_weights[0])

    # ------------------------------------------------------------------
    # Extended outputs for batch plotting
    # ------------------------------------------------------------------
    # All individual MC C fields: (N_mc, nv+1, nh+1)
    np.save(os.path.join(run_dir, "ch_est_all.npy"),      ch_est_arr)
    # All individual MC U snapshots at last obs time: (N_mc, nv+1, nh+1)
    np.save(os.path.join(run_dir, "u_est_all.npy"),       u_est_arr)
    # U temporal traces at monitor points: (N_mc, n_pts, numt+1)
    np.save(os.path.join(run_dir, "u_temporal_all.npy"),  u_temporal_arr)
    # U snapshots at specified years: (N_mc, n_snaps, nv+1, nh+1)
    np.save(os.path.join(run_dir, "u_snapshots_all.npy"), u_snapshots_arr)
    # Truth temporal traces: (n_pts, numt+1)
    np.save(os.path.join(run_dir, "u_true_temporal.npy"),  u_true_temporal)
    # Truth U snapshots: (n_snaps, nv+1, nh+1)
    np.save(os.path.join(run_dir, "u_true_snapshots.npy"), u_true_snapshots)
    # Physical time axis: (numt+1,)
    np.save(os.path.join(run_dir, "t_coords.npy"),          t_coords)
    # Snapshot years (actual, after rounding): (n_snaps,)
    np.save(os.path.join(run_dir, "snapshot_years.npy"),    np.array(snap_years_actual))
    # Monitor point indices: (n_pts, 2) as [row, col]
    np.save(os.path.join(run_dir, "monitor_points.npy"),    np.array(monitor_pts_clamped))

    # ------------------------------------------------------------------
    # Precomputed field statistics (new) – mean and std across MC ensemble
    # ------------------------------------------------------------------
    # These files allow batch_plot tools to reconstruct confidence bands
    # without re-running the forward solver or decoding mc_weights again.
    #
    #   ch_est_std.npy       : (nv+1, nh+1)        std of C across MC realizations
    #   u_temporal_mean.npy  : (n_pts, numt+1)      mean U at monitor pts over time
    #   u_temporal_std.npy   : (n_pts, numt+1)      std  U at monitor pts over time
    #   u_snapshots_mean.npy : (n_snaps, nv+1, nh+1) mean U at snapshot years
    #   u_snapshots_std.npy  : (n_snaps, nv+1, nh+1) std  U at snapshot years
    #   x_coords.npy         : (nh+1,)              horizontal grid positions (m)
    #   z_coords.npy         : (nv+1,)              vertical   grid positions (m)
    np.save(os.path.join(run_dir, "u_temporal_mean.npy"),   u_temporal_mean)
    np.save(os.path.join(run_dir, "u_temporal_std.npy"),    u_temporal_std)
    np.save(os.path.join(run_dir, "u_snapshots_mean.npy"),  u_snapshots_mean)
    np.save(os.path.join(run_dir, "u_snapshots_std.npy"),   u_snapshots_std)
    np.save(os.path.join(run_dir, "x_coords.npy"),          x_coords)
    np.save(os.path.join(run_dir, "z_coords.npy"),          z_coords)

    # ------------------------------------------------------------------
    # NPZ bundle: single-file archive with all stats + grids for plotting
    # ------------------------------------------------------------------
    # batch_stats.npz contains everything needed to produce confidence-band
    # plots without re-running any solver.  Keys:
    #   ch_true, ch_est_mean, ch_est_std
    #   u_temporal_mean, u_temporal_std, u_true_temporal
    #   u_snapshots_mean, u_snapshots_std, u_true_snapshots
    #   x_coords, z_coords, t_coords, snapshot_years, monitor_points
    np.savez(
        os.path.join(run_dir, "batch_stats.npz"),
        # C field
        ch_true          = solver.chm,
        ch_est_mean      = mean_ch_est,
        ch_est_std       = std_ch_est,
        # U temporal traces at monitor points
        u_temporal_mean  = u_temporal_mean,
        u_temporal_std   = u_temporal_std,
        u_true_temporal  = u_true_temporal,
        # U spatial snapshots
        u_snapshots_mean = u_snapshots_mean,
        u_snapshots_std  = u_snapshots_std,
        u_true_snapshots = u_true_snapshots,
        # Grids and axes
        x_coords         = x_coords,
        z_coords         = z_coords,
        t_coords         = t_coords,
        snapshot_years   = np.array(snap_years_actual),
        monitor_points   = np.array(monitor_pts_clamped),
    )
    print(f"  Saved field statistics → {os.path.join(run_dir, 'batch_stats.npz')}")

    # ------------------------------------------------------------------
    # Save run metadata (for batch plotting scripts and reproducibility)
    # ------------------------------------------------------------------
    # Persist the fully resolved, run-specific configuration for deterministic
    # post-hoc reconstruction in plotting tools.  Keys:
    #   - problem/basis/model/solver/run: exact dataclass values used in run
    #   - measurements: u/C observation indices and u observation times (ukt)
    #   - plot_output: monitor points and requested snapshot years
    # This enables replay of each MC realization from mc_weights.npy as:
    # coef -> decoded C field -> forward-solved U field.
    resolved_cfg = {
        "problem": asdict(pconf),
        "basis": asdict(bconf),
        "model": asdict(mconf),
        "solver": asdict(sconf),
        "run": asdict(rconf),
        "measurements": {
            "ukmat": [list(pt) for pt in ukmat_idx],
            "chkmat": [list(pt) for pt in chkmat_idx],
            "ukt": ukt_arr.tolist(),
        },
        "plot_output": {
            "snapshot_years": list(snapshot_years),
            "monitor_points": [list(pt) for pt in monitor_pts_clamped],
        },
    }
    resolved_cfg_path = os.path.join(run_dir, "run_config_resolved.json")
    with open(resolved_cfg_path, "w", encoding="utf-8") as fh:
        json.dump(resolved_cfg, fh, indent=2)
    print(f"  Saved resolved config → {resolved_cfg_path}")

    # The "mean estimate" is the pointwise mean of all MC C (or U) fields.
    # The confidence band is mean ± band_k * std; default band_k = 1.0 (68 %).
    run_metadata = {
        "run_name":         run_name,
        "timestamp":        timestamp,
        "N_mc":             N_mc,
        "basis_type":       bconf.type,
        "epsilon":          mconf.epsilon,
        "snapshot_years":   snap_years_actual,
        "monitor_points":   monitor_pts_clamped,
        "mean_estimate":    "pointwise mean of all MC realizations (not averaged weights)",
        "confidence_band":  "mean ± band_k * std  (default band_k = 1.0 → ±1σ ≈ 68%)",
        "saved_files": {
            # ---- precomputed statistics (primary input for batch_plot tools) ----
            "batch_stats.npz":        "NPZ bundle: all mean/std stats + grids (ch_true, ch_est_mean/std, u_temporal_mean/std, u_true_temporal, u_snapshots_mean/std, u_true_snapshots, x_coords, z_coords, t_coords, snapshot_years, monitor_points)",
            "ch_est.npy":             "mean C field  (nv+1, nh+1)",
            "ch_est_std.npy":         "std  C field across MC realizations  (nv+1, nh+1)",
            "u_temporal_mean.npy":    "mean U at monitor pts over time  (n_pts, numt+1)",
            "u_temporal_std.npy":     "std  U at monitor pts over time  (n_pts, numt+1)",
            "u_snapshots_mean.npy":   "mean U at snapshot years  (n_snaps, nv+1, nh+1)",
            "u_snapshots_std.npy":    "std  U at snapshot years  (n_snaps, nv+1, nh+1)",
            "x_coords.npy":           "horizontal grid positions in metres  (nh+1,)",
            "z_coords.npy":           "vertical   grid positions in metres  (nv+1,)",
            # ---- per-MC ensemble arrays (retained for backward compat) ---------
            "ch_est_all.npy":         "all MC C fields  (N_mc, nv+1, nh+1)",
            "u_mean_est.npy":         "mean U at last obs time  (nv+1, nh+1)",
            "u_est_all.npy":          "all MC U at last obs time  (N_mc, nv+1, nh+1)",
            "u_temporal_all.npy":     "U at monitor points (N_mc, n_pts, numt+1)",
            "u_snapshots_all.npy":    "U at snapshot years (N_mc, n_snaps, nv+1, nh+1)",
            "u_true_temporal.npy":    "truth U at monitor points (n_pts, numt+1)",
            "u_true_snapshots.npy":   "truth U at snapshot years (n_snaps, nv+1, nh+1)",
            "t_coords.npy":           "physical time axis in years (numt+1,)",
            "snapshot_years.npy":     "years for U snapshots (n_snaps,)",
            "monitor_points.npy":     "[row, col] monitor indices (n_pts, 2)",
            "ch_true.npy":            "ground-truth C field (nv+1, nh+1)",
            "mc_weights.npy":         "per-MC weight vectors (N_mc, nb)",
        },
    }
    meta_path = os.path.join(run_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, indent=2)
    print(f"  Saved run metadata → {meta_path}")

    # Save comparison plot (preserved output format; contents = mean C field)
    plot_path = os.path.join(run_dir, "triplot_C.png")
    c_title = "C estimated" if N_mc == 1 else "C estimated (mean)"
    solver.triplot2D(
        solver.chm, mean_ch_est,
        titles=["C truth", c_title, "log(truth/est)"],
        mode=1,
        psavepath=plot_path,
    )
    print(f"  Saved comparison plot → {plot_path}")

    # ------------------------------------------------------------------
    # Compute key metrics (based on mean C field)
    # ------------------------------------------------------------------
    cos_s = float(
        np.dot(solver.chm.flatten(), mean_ch_est.flatten()) /
        (np.linalg.norm(solver.chm.flatten()) * np.linalg.norm(mean_ch_est.flatten()))
    )
    rmse    = float(np.sqrt(np.mean((solver.chm - mean_ch_est) ** 2)))
    max_err = float(np.max(np.abs(solver.chm - mean_ch_est)))

    final_loss = (
        float(np.load(rconf.loss_file)[-1])
        if rconf.save_losses and os.path.exists(rconf.loss_file)
        else float("nan")
    )

    print(f"  cos_sim={cos_s:.4f}  RMSE={rmse:.4f}  max_err={max_err:.4f}  final_loss={final_loss:.6f}")

    # ------------------------------------------------------------------
    # Append row to summary CSV
    # ------------------------------------------------------------------
    row = {
        "run_name":   run_name,
        "run_dir":    run_dir,
        "N_mc":       N_mc,
        "basis_type": bconf.type,
        "epsilon":    mconf.epsilon,
        "regen_fluc": pconf.regen_fluc,
        "fluc_seed":  pconf.fluc_seed,
        "Rcv":        pconf.Rcv,
        "final_loss": final_loss,
        "cos_sim":    cos_s,
        "RMSE":       rmse,
        "max_err":    max_err,
    }
    # Include per-coef columns only for N_mc == 1 (backward compat)
    if N_mc == 1:
        row.update({f"coef_{i}": float(v) for i, v in enumerate(mc_weights[0])})

    file_exists = os.path.exists(coeffs_csv_path)
    with open(coeffs_csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Summary row appended → {coeffs_csv_path}")
    return row


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QAPILM batch sweep runner.  Loads config.yaml + sweep.yaml "
                    "and runs one experiment per sweep entry."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to base YAML config file  (default: config.yaml)"
    )
    parser.add_argument(
        "--sweep", default="sweep.yaml",
        help="Path to sweep YAML file  (default: sweep.yaml)"
    )
    parser.add_argument(
        "--run", nargs="*", metavar="NAME",
        help="Run only the named sweep entries (default: run all)"
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Override results directory from config.yaml"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load base config and sweep list
    # ------------------------------------------------------------------
    if not os.path.exists(args.config):
        sys.exit(f"Base config not found: {args.config}")
    if not os.path.exists(args.sweep):
        sys.exit(f"Sweep file not found: {args.sweep}")

    base_cfg   = _load_yaml(args.config)
    sweep_data = _load_yaml(args.sweep)

    sweeps = sweep_data.get("sweeps", [])
    if not sweeps:
        sys.exit("No sweep entries found in sweep.yaml under the 'sweeps:' key.")

    # Filter by --run names if provided
    if args.run:
        run_set = set(args.run)
        sweeps  = [s for s in sweeps if s.get("name", "") in run_set]
        if not sweeps:
            sys.exit(f"None of the requested run names found in sweep.yaml: {args.run}")

    # Determine output directory and summary CSV path
    results_dir = args.outdir or base_cfg.get("run", {}).get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)
    coeffs_csv  = os.path.join(
        results_dir,
        base_cfg.get("run", {}).get("coeffs_csv", "coeffs_batch.csv"),
    )

    print(f"Batch sweep: {len(sweeps)} run(s)  →  {results_dir}")
    print(f"Summary CSV: {coeffs_csv}\n")

    # ------------------------------------------------------------------
    # Execute each sweep entry
    # ------------------------------------------------------------------
    summary = []
    for entry in sweeps:
        run_name = entry.get("name", f"run_{len(summary)+1}")

        # Deep-copy base config and apply overrides (exclude the 'name' key)
        overrides = {k: v for k, v in entry.items() if k != "name"}
        merged_cfg = _deep_merge(base_cfg, overrides)

        try:
            row = run_one(merged_cfg, run_name, results_dir, coeffs_csv)
            summary.append(row)
        except Exception:
            print(f"\n[ERROR] Run '{run_name}' failed:")
            traceback.print_exc()
            summary.append({"run_name": run_name, "error": traceback.format_exc()})

    # ------------------------------------------------------------------
    # Print final summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  BATCH SWEEP COMPLETE")
    print(f"{'='*70}")
    for row in summary:
        name = row.get("run_name", "?")
        if "error" in row:
            print(f"  {name:30s}  FAILED")
        else:
            print(
                f"  {name:30s}  "
                f"cos_sim={row.get('cos_sim', float('nan')):.4f}  "
                f"RMSE={row.get('RMSE', float('nan')):.4f}  "
                f"loss={row.get('final_loss', float('nan')):.4e}"
            )
    print(f"\nSummary CSV written to: {coeffs_csv}")


if __name__ == "__main__":
    main()
