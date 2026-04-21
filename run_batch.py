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
import os
import sys
import traceback
import warnings
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
)


# ===========================================================================
# Config helpers
# ===========================================================================

def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as fh:
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
        type         = str(b_raw.get("type",         BasisConfig.type)),
        orderx       = int(b_raw.get("orderx",       BasisConfig.orderx)),
        orderz       = int(b_raw.get("orderz",       BasisConfig.orderz)),
        wav_levels_x = int(b_raw.get("wav_levels_x", BasisConfig.wav_levels_x)),
        wav_levels_z = int(b_raw.get("wav_levels_z", BasisConfig.wav_levels_z)),
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
# Single-run executor
# ===========================================================================

def run_one(cfg: dict, run_name: str, base_results_dir: str, coeffs_csv_path: str):
    """
    Execute one experiment described by *cfg* (already merged base+override).

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

    print(f"\n{'='*70}")
    print(f"  Starting run: {run_name}  →  {run_dir}")
    print(f"{'='*70}")

    # Build dataclass configs
    pconf, bconf, mconf, sconf, rconf, ukmat, chkmat, ukt_list = _cfg_to_dataclasses(cfg)

    # Redirect per-run outputs into run_dir
    rconf.results_dir = run_dir
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
    # Generate "ground truth" snapshot tensor via forward solver
    # ------------------------------------------------------------------
    top, bot, left, right = pconf.bcs
    u0_mat = np.ones((solver.numv + 1, solver.numh + 1)) * pconf.u0
    numt   = solver.ilim

    print(f"  Running forward solver  (numt={numt}) …")
    utens, udeg = solver.forward_solver(solver.chm, numt, u0_mat, top, bot, left, right)

    # ------------------------------------------------------------------
    # Extract observations at the requested time steps
    # ------------------------------------------------------------------
    # Clamp ukt indices to valid range
    ukt_arr = np.clip(ukt_arr, 0, numt)

    uk  = utens[ukt_arr, :, :]          # shape: (T, numv+1, numh+1)
    chk = solver.chm                    # ground-truth permeability field

    # ------------------------------------------------------------------
    # Run inverse solver (streaming, memory-optimised by default)
    # ------------------------------------------------------------------
    print(f"  Running inverse solver  (epochs={mconf.itol}, ε={mconf.epsilon}) …")
    if sconf.memory_mode == "stream":
        coefe = solver.inverse_solver_stream(ukt_arr, u0_mat, ukmat_idx, chkmat_idx, uk, chk)
    else:
        # "full" mode is not yet implemented in qapilm_rect.py; fall back to
        # the memory-optimised stream solver and notify the caller.
        warnings.warn(
            f"memory_mode='{sconf.memory_mode}' is not yet implemented; "
            "falling back to memory_mode='stream'.",
            stacklevel=2,
        )
        coefe = solver.inverse_solver_stream(ukt_arr, u0_mat, ukmat_idx, chkmat_idx, uk, chk)

    # ------------------------------------------------------------------
    # Reconstruct estimated C field and save outputs
    # ------------------------------------------------------------------
    from qapilm_rect import softplus
    m_est  = solver.vec2mat2(solver.basese @ coefe)
    ch_est = softplus(m_est)

    np.save(os.path.join(run_dir, "coefe.npy"),  coefe)
    np.save(os.path.join(run_dir, "ch_est.npy"), ch_est)
    np.save(os.path.join(run_dir, "ch_true.npy"), solver.chm)

    # Save comparison plots (preserved output format)
    plot_path = os.path.join(run_dir, "triplot_C.png")
    solver.triplot2D(
        solver.chm, ch_est,
        titles=["C truth", "C estimated", "log(truth/est)"],
        mode=1,
        psavepath=plot_path,
    )
    print(f"  Saved comparison plot → {plot_path}")

    # ------------------------------------------------------------------
    # Compute key metrics
    # ------------------------------------------------------------------
    cos_s = float(np.dot(solver.chm.flatten(), ch_est.flatten()) /
                  (np.linalg.norm(solver.chm.flatten()) * np.linalg.norm(ch_est.flatten())))
    rmse  = float(np.sqrt(np.mean((solver.chm - ch_est) ** 2)))
    max_err = float(np.max(np.abs(solver.chm - ch_est)))

    final_loss = float(np.load(rconf.loss_file)[-1]) if rconf.save_losses and os.path.exists(rconf.loss_file) else float("nan")

    print(f"  cos_sim={cos_s:.4f}  RMSE={rmse:.4f}  max_err={max_err:.4f}  final_loss={final_loss:.6f}")

    # ------------------------------------------------------------------
    # Append row to summary CSV
    # ------------------------------------------------------------------
    row = {
        "run_name":   run_name,
        "run_dir":    run_dir,
        "basis_type": bconf.type,
        "epsilon":    mconf.epsilon,
        "regen_fluc": pconf.regen_fluc,
        "fluc_seed":  pconf.fluc_seed,
        "Rcv":        pconf.Rcv,
        "final_loss": final_loss,
        "cos_sim":    cos_s,
        "RMSE":       rmse,
        "max_err":    max_err,
        **{f"coef_{i}": float(v) for i, v in enumerate(coefe)},
    }

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
