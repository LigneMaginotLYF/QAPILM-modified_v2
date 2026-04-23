# QAPILM-modified_v2

Reworked rectangular-case QAPILM solver for 2-D heterogeneous consolidation
inverse analysis, with batch-sweep support via YAML configuration files.

---

## Background

The solver recovers the heterogeneous coefficient-of-consolidation field
`C(x,z)` from sparse observations of both `C` and the excess pore-water
pressure `u(x,z,t)`.  `C` is parameterised as a weighted sum of basis
functions (polynomial-sine, DCT, Legendre, or Haar wavelet) passed through
a softplus activation.  The forward model is an explicit finite-difference
time march; the inverse step uses forward-mode sensitivity recursion with an
Adam optimiser and an ε-insensitive loss.

---

## Files

| File | Purpose |
|------|---------|
| `qapilm_rect.py` | Core solver: `RectangularQAPILM` class, config dataclasses, basis factory, forward/inverse solvers, plotting helpers |
| `config.yaml` | **Base configuration** – edit to change defaults for geometry, material, BCs, basis type, ε threshold, solver mode, etc. |
| `sweep.yaml` | **Batch sweep list** – one entry per experiment run; each entry overrides selected keys from `config.yaml` |
| `run_batch.py` | **Batch runner** – loads the two YAML files and executes all sweep entries, writing per-run results and a summary CSV |
| `vanilla_PINN_2D_rect.py` | **PINN solver** – physics-informed neural network for joint u/C recovery; generates observations via FD forward solver, trains two networks, saves timestamped models, and plots Real / QAPILM / PINN comparisons |
| `tools/plot_epsilon_batch.py` | **ε-threshold batch plotter** – reads multiple run directories and superimposes confidence bands for C cross sections and U temporal/spatial panels |
| `tools/plot_basis_batch.py` | **Basis-form batch plotter** – same cross-section plots plus parallel field comparison (truth / mean prediction / error) across basis types |

---

## Dependencies

Standard Python scientific stack plus **PyYAML** (and **PyTorch** for the PINN):

```bash
pip install numpy scipy matplotlib seaborn pyyaml
pip install torch          # for vanilla_PINN_2D_rect.py
```

---

## Quick start – single QAPILM run

```python
from qapilm_rect import (
    ProblemConfig, BasisConfig, ModelConfig,
    SolverConfig, RunConfig, RectangularQAPILM,
)

solver = RectangularQAPILM(
    ProblemConfig(), BasisConfig(), ModelConfig(),
    SolverConfig(), RunConfig(),
)
# u0 can be a scalar or a (nz+1, nx+1) matrix; both are handled correctly
utens, udeg = solver.forward_solver(solver.chm, solver.ilim, solver.p.u0, *solver.p.bcs)
```

---

## Batch sweep

### 1. Edit `config.yaml`

`config.yaml` holds the **base** configuration.  All keys are documented
inline.  The most important options are:

| Key | Description |
|-----|-------------|
| `problem.regen_fluc` | `false` → use pre-generated CSV fluctuation files; `true` → generate fresh random field |
| `problem.fluc_seed` | RNG seed when `regen_fluc: true` |
| `problem.filedir` / `file_rawS` / `file_Lmat` | Paths to legacy CSV fluctuation files (backward compatible) |
| `problem.bcs` | Boundary conditions `[top, bot, left, right]` (0 = Dirichlet, 1 = Neumann) |
| `basis.type` | Basis type: `poly` · `sin` · `dct` · `legendre` · `wavelet` (solo) or any duo like `poly+sin` · `dct+legendre` · `poly+wavelet`, etc. |
| `model.epsilon` | ε-insensitive loss threshold (relative to observation magnitude) |
| `solver.memory_mode` | `full` (legacy default) or `stream` (memory-optimised) |
| `solver.store_u_snapshots` | Whether to keep `u` snapshots during the inverse solve |
| `measurements.ukmat` / `chkmat` | Grid-index pairs `[row, col]` for `u` / `C` observations |
| `measurements.ukt` | Time-step indices at which `u` is observed |
| `monte_carlo.N_mc` | Number of Monte Carlo restarts (default `1` → single run) |

### 2. Edit `sweep.yaml`

`sweep.yaml` contains a list under the `sweeps:` key.  Each entry has a
`name:` and any number of override keys matching the structure of
`config.yaml`:

```yaml
sweeps:
  - name: "baseline"          # no overrides → pure default run

  - name: "eps_0.05"
    model:
      epsilon: 0.05

  - name: "basis_dct_regen"
    basis:
      type: "dct"
      orderx: 3
      orderz: 4
    problem:
      regen_fluc: true
      fluc_seed: 7
```

### 3. Run the batch

```bash
# Run all sweep entries defined in sweep.yaml
python run_batch.py

# Specify custom config/sweep files
python run_batch.py --config my_config.yaml --sweep my_sweep.yaml

# Run only selected named entries
python run_batch.py --run baseline eps_0.05 basis_dct_regen

# Override output directory
python run_batch.py --outdir ./my_results
```

### 4. Outputs

Each sweep entry creates a time-stamped sub-folder inside `results/`
(or `--outdir`):

```
results/
  baseline_20250101_120000/
    QAPILM_Eps_loss.npy      # loss curve for N_mc=1; for N_mc>1: QAPILM_Eps_loss_mc000.npy … mc{N_mc-1:03d}.npy
    mc_weights.npy           # all per-run weight vectors, shape (N_mc, nb)
    coefe.npy                # single-run coefe (only written when N_mc == 1)
    ch_est.npy               # mean estimated C field
    ch_est_all.npy           # all individual MC C fields, shape (N_mc, nv+1, nh+1)
    u_mean_est.npy           # mean estimated U field at last observation time
    u_est_all.npy            # all individual MC U snapshots, shape (N_mc, nv+1, nh+1)
    u_temporal_all.npy       # U at monitor points over time, shape (N_mc, n_pts, numt+1)
    u_snapshots_all.npy      # U at snapshot years, shape (N_mc, n_snaps, nv+1, nh+1)
    u_true_temporal.npy      # truth U at monitor points, shape (n_pts, numt+1)
    u_true_snapshots.npy     # truth U at snapshot years, shape (n_snaps, nv+1, nh+1)
    t_coords.npy             # physical time axis in years, shape (numt+1,)
    snapshot_years.npy       # years corresponding to u_*_snapshots.npy
    monitor_points.npy       # [row, col] indices of monitor points, shape (n_pts, 2)
    ch_true.npy              # ground-truth C field
    triplot_C.png            # three-panel comparison plot (mean C field)
  eps_0.05_20250101_120100/
    ...
  coeffs_batch.csv           # summary table: one row per run
```

`coeffs_batch.csv` contains columns for `run_name`, `N_mc`, `basis_type`,
`epsilon`, `regen_fluc`, `fluc_seed`, `Rcv`, `final_loss`, `cos_sim`,
`RMSE`, `max_err`, and (for `N_mc == 1`) all optimised coefficients
(`coef_0`, `coef_1`, …).

---

## Batch plotting scripts

### `tools/plot_epsilon_batch.py`

Reads run directories from a results folder and superimposes confidence
bands (mean ± std across the MC ensemble) for an epsilon-threshold batch
sweep.  Four plots are produced:

- **C cross section at x = const** (vary z) — `eps_C_xsec_fixedX.png`
- **C cross section at z = const** (vary x) — `eps_C_xsec_fixedZ.png`
- **U field panels** (× 2): either temporal evolution at fixed spatial
  points (`eps_U_temporal_pt*.png`) or spatial line at a fixed timestep
  (`eps_U_spatial_line*.png`) depending on `U_MODE`.

All tunable parameters are at the top of the script:

| Variable | Purpose |
|----------|---------|
| `RESULTS_DIR` | Parent folder containing run sub-directories |
| `GLOB_PATTERNS` | List of glob patterns to select runs (most-recent match per pattern) |
| `LABELS` | Legend labels (same order as `GLOB_PATTERNS`) |
| `C_XSEC_X_IDX` / `C_XSEC_Z_IDX` | Grid column/row indices for the two C cross sections |
| `U_MODE` | `"temporal"` (evolve in time at fixed point) or `"spatial"` (line at fixed t) |
| `U_MONITOR_PT_IDX_1/2` | Indices into the saved `monitor_points` array |
| `U_SPATIAL_T_IDX` | Timestep index for spatial mode |
| `OUTPUT_DIR` | Output folder for saved PNG files |

```bash
python tools/plot_epsilon_batch.py
```

The script requires that the run directories were created by `run_batch.py`
with `monte_carlo.N_mc > 1` (to produce non-trivial confidence bands).
For single-run results the confidence band collapses to a single line.

### `tools/plot_basis_batch.py`

Compares batch results across different basis-function configurations.
Produces **two sets** of figures:

**Set A** (same layout as `plot_epsilon_batch.py`):
- `basis_C_xsec_fixedX.png`, `basis_C_xsec_fixedZ.png`
- `basis_U_temporal_pt*.png` or `basis_U_spatial_line*.png`

**Set B** (parallel field plots with equal-sized panels):
- `basis_C_parallel.png` — C field rows: [truth | mean prediction | error]
- `basis_U_parallel_t{t}yr.png` — U field at each snapshot year:
  [truth | mean prediction | error], one row per basis type

```bash
python tools/plot_basis_batch.py
```

Tunable parameters include all of the above plus:

| Variable | Purpose |
|----------|---------|
| `U_SNAP_YEARS_TO_PLOT` | Physical years for U parallel plots |
| `ERROR_MODE` | `"log_ratio"` (log truth/pred) or `"diff"` (truth − pred) |

---

## Randomize option

`run_batch.py` supports an optional **randomize** mode that perturbs the
ground-truth or measurement setup before each run, useful for sensitivity
analysis and robustness testing.

Add a `randomize:` block to `config.yaml` or to a sweep entry in `sweep.yaml`:

```yaml
randomize:
  mode: "coeft"    # "none" (default) | "coeft" | "locations"
  seed: 42         # integer ≥ 1 for reproducibility; 0 = leave RNG unseeded
```

| Mode | Effect |
|------|--------|
| `"none"` | No randomization — backward-compatible default |
| `"coeft"` | Replaces the ground-truth trend polynomial coefficients (`problem.coeft`) with independent N(0, 1) random values, keeping the same number of coefficients |
| `"locations"` | Replaces `ukmat` and `chkmat` measurement locations with uniformly random grid positions while keeping the same number of measurement points |

**Example sweep entry** (vary coeft seed across runs):

```yaml
sweeps:
  - name: "rand_coeft_s1"
    randomize:
      mode: "coeft"
      seed: 1
  - name: "rand_coeft_s2"
    randomize:
      mode: "coeft"
      seed: 2
```

**Example sweep entry** (randomize measurement locations):

```yaml
sweeps:
  - name: "rand_locs_s1"
    randomize:
      mode: "locations"
      seed: 100
```

> **Backward compatibility**: when `randomize:` is absent from the config
> (or `mode: "none"`), the behaviour is identical to previous versions.

---

## Configuring basis types

Basis types are specified as a solo component name or two components joined
with `"+"`.  The polynomial prefix is **no longer forced** — each type is
clean and self-contained.

### Solo types

| `basis.type` | Description | `orderx` / `orderz` | `wav_levels_x/z` |
|--------------|-------------|---------------------|-----------------|
| `poly`       | 6-term 2D polynomial: 1, x, z, x², z², xz | — | — |
| `sin`        | Sine harmonics sin(2ᵏ x̄), sin(2ᵏ z̄) | ✓ | — |
| `dct`        | Discrete cosine cos(πk x̄), cos(πk z̄) | ✓ | — |
| `legendre`   | Legendre polynomials Pₖ(x̄), Pₖ(z̄) | ✓ | — |
| `wavelet`    | Haar wavelets | — | ✓ |

### Duo types (any two components joined with `+`)

| `basis.type` | Components concatenated |
|--------------|------------------------|
| `poly+sin`   | polynomial + sine (same as legacy `poly_sin`) |
| `poly+dct`   | polynomial + DCT |
| `poly+legendre` | polynomial + Legendre |
| `poly+wavelet`  | polynomial + Haar wavelets |
| `sin+dct`    | sine + DCT |
| `sin+legendre` | sine + Legendre |
| `sin+wavelet`  | sine + Haar wavelets |
| `dct+legendre` | DCT + Legendre |
| `dct+wavelet`  | DCT + Haar wavelets |
| `legendre+wavelet` | Legendre + Haar wavelets |

> **Backward-compatible alias**: `"poly_sin"` is silently mapped to
> `"poly+sin"`.  Existing config files using `type: "poly_sin"` will
> continue to work without changes.
>
> **Migration note**: the old names `"dct"`, `"legendre"`, and `"wavelet"`
> now produce **clean solo** versions (no poly prefix).  To reproduce the
> old poly-prefixed behaviour, change to `"poly+dct"`, `"poly+legendre"`,
> or `"poly+wavelet"` respectively.

Higher `orderx` / `orderz` increases the number of basis functions and
the expressiveness of the estimated `C` field, at the cost of more
computation and potential overfitting with sparse data.

---

## Notes

- **Backward compatibility**: existing output file names and figure formats
  produced by `qapilm_rect.py` are preserved unchanged.  When `N_mc == 1`
  the output folder is byte-for-byte identical to the original single-run
  output (plus the new `mc_weights.npy`, `ch_est_all.npy`, and auxiliary
  time-series files).
- **Memory**: `solver.memory_mode: full` is the backward-compatible default.
  Use `stream` to reduce memory to O(1) in the time dimension.
- **Reproducibility**: set `regen_fluc: true` and fix `fluc_seed` for
  deterministic runs independent of the legacy CSV files.
- **Monte Carlo UQ**: set `monte_carlo.N_mc` to a value greater than 1 to
  run the inverse solver multiple times with independent random initial
  coefficients.  All per-run weight vectors are saved to `mc_weights.npy`
  (shape `(N_mc, nb)`) and all individual C fields to `ch_est_all.npy`
  (shape `(N_mc, nv+1, nh+1)`) for downstream uncertainty quantification.
  The mean C and mean U fields replace the single-estimate outputs in
  `ch_est.npy` and `triplot_C.png`.
  **Computational cost**: each MC run performs one inverse solve **and** one
  additional forward solve (to decode the U field); total wall-time scales
  roughly as `N_mc × (1 inverse + 1 forward)` per sweep entry.
- **Batch plotting**: `tools/plot_epsilon_batch.py` and
  `tools/plot_basis_batch.py` load the extended output arrays
  (`ch_est_all.npy`, `u_temporal_all.npy`, `u_snapshots_all.npy`, etc.) to
  produce publication-ready comparison figures across multiple runs.
  These files are written automatically by `run_batch.py`; no extra
  configuration is required beyond the `plot_output:` block in `config.yaml`.
- **YAML encoding**: all YAML files (`config.yaml`, `sweep.yaml`, custom
  configs) must be saved with **UTF-8** encoding.  On Windows the default
  system encoding (e.g. GBK) can cause a `UnicodeDecodeError` when loading;
  the loader in `run_batch.py` always opens YAML files with
  `encoding="utf-8"` to avoid this.  When editing YAML files in a text
  editor, ensure the file is saved as UTF-8 (without BOM).

---

## PINN solver (`vanilla_PINN_2D_rect.py`)

`vanilla_PINN_2D_rect.py` provides a fully self-contained Physics-Informed
Neural Network (PINN) that solves the same 2-D consolidation problem as
`qapilm_rect.py` using PyTorch.

### PDE

```
u_t = C(x,z) · u_xx + C(x,z)·Rcv · u_zz + (∂C/∂x)·u_x + Rcv·(∂C/∂z)·u_z
```

Two networks are trained jointly:

| Network | Inputs | Output |
|---------|--------|--------|
| `UNet`  | x̄, z̄, t̄ (normalised) | u (pore pressure) |
| `CNet`  | x̄, z̄          | C > 0 (via softplus) |

### 1. Generating observations from the forward solver

**Option A – use the QAPILM FD solver** (recommended when a fully
configured QAPILM instance is available):

```python
from qapilm_rect import ProblemConfig, BasisConfig, ModelConfig, SolverConfig, RunConfig, RectangularQAPILM
from vanilla_PINN_2D_rect import PINN2DConsolidation, PINNObsConfig

# Build a QAPILM solver (ground-truth C is inside solver.chm)
solver = RectangularQAPILM(
    ProblemConfig(regen_fluc=True, fluc_seed=42),
    BasisConfig(), ModelConfig(), SolverConfig(), RunConfig(),
)

pinn = PINN2DConsolidation()
obs  = pinn.generate_observations_from_qapilm(solver)
```

**Option B – use the built-in minimal FD solver** (no external CSV files
needed):

```python
import numpy as np
from vanilla_PINN_2D_rect import PINN2DConsolidation, PINNGeomConfig

# Any (nz+1, nx+1) positive array works as ground-truth C
geom = PINNGeomConfig(Lh=10, Lv=5, Lt=2)
pinn = PINN2DConsolidation(geom=geom)

chm = np.ones((26, 51)) * 1.2          # uniform C (replace with real field)
obs = pinn.generate_observations_from_fd(chm)
```

### 2. Choosing measurement density

`PINNObsConfig` controls how observation locations are sampled:

```python
from vanilla_PINN_2D_rect import PINNObsConfig

obs_cfg = PINNObsConfig(
    density_mode = "grid",    # "grid" (regular sub-grid) or "random" (uniform random)
    u_density    = 0.05,      # fraction of spatial grid points used for u obs.
    c_density    = 0.05,      # fraction used for C obs.
    n_time_obs   = 5,         # number of observation time instants (auto-spaced)

    # Optional explicit overrides — when non-empty, density settings are ignored:
    u_locations  = [[2, 5], [10, 20]],   # [[row, col], …]
    c_locations  = [[0, 0], [12, 25]],
    time_indices = [100, 200, 300],       # exact time-step indices
)
```

### 3. Running the PINN and saving/loading timestamped models

**Training from Python:**

```python
from vanilla_PINN_2D_rect import (
    PINN2DConsolidation, PINNGeomConfig, PINNNetConfig,
    PINNTrainConfig, PINNObsConfig, PINNSaveConfig,
)

pinn = PINN2DConsolidation(
    geom      = PINNGeomConfig(),
    net_cfg   = PINNNetConfig(u_hidden_layers=4, u_hidden_width=64),
    train_cfg = PINNTrainConfig(epochs=5000, lr=1e-3),
    obs_cfg   = PINNObsConfig(density_mode="grid", u_density=0.05),
    save_cfg  = PINNSaveConfig(model_dir="./pinn_models"),
)

obs = pinn.generate_observations_from_fd(chm_true)
pinn.train(obs)
model_path = pinn.save()          # → ./pinn_models/pinn_YYYYMMDD_HHMMSS.pt
```

**Training from the command line** (uses a synthetic C field as demo):

```bash
# Quick-start with default settings (3000 epochs, built-in FD solver)
python vanilla_PINN_2D_rect.py

# Custom output directory
python vanilla_PINN_2D_rect.py --outdir ./my_pinn_models

# Custom epoch count
python vanilla_PINN_2D_rect.py --epochs 8000

# Use the full QAPILM FD solver (requires CSV fluctuation files)
python vanilla_PINN_2D_rect.py --use-qapilm

# Load a saved model and skip training
python vanilla_PINN_2D_rect.py --load pinn_models/pinn_20250101_120000.pt
```

**Loading a previously saved model:**

```python
from vanilla_PINN_2D_rect import PINN2DConsolidation

pinn = PINN2DConsolidation.load("pinn_models/pinn_20250101_120000.pt")
# All network weights and loss histories are restored.
```

Each saved model consists of two files:

```
pinn_models/
  pinn_20250101_120000.pt          # PyTorch checkpoint (weights + loss histories)
  pinn_20250101_120000_meta.json   # JSON metadata (configs, final loss, n_epochs)
```

### 4. Producing comparison plots

**Inline comparison plot (Real / QAPILM / PINN):**

```python
# After training or loading, call plot_comparison directly:
pinn.plot_comparison(
    true_C   = chm_true,       # ground-truth C field  (nz, nx)
    qapilm_C = ch_est,         # QAPILM estimate       (nz, nx)  — pass None to omit
    save_path = "comparison_C.png",
    mode      = 1,             # 1 → log(Real/PINN),  0 → Real-PINN
)
```

**Using the standalone `compare_and_plot()` helper** (minimal boilerplate,
can load fields from `.npy` files or arrays):

```python
from vanilla_PINN_2D_rect import PINN2DConsolidation, compare_and_plot

pinn = PINN2DConsolidation.load("pinn_models/pinn_20250101_120000.pt")

compare_and_plot(
    "results/baseline_20250101_120000/ch_true.npy",    # path or array
    "results/baseline_20250101_120000/ch_est.npy",     # QAPILM estimate
    pinn,
    save_path = "full_comparison.png",
)
```

The plot layout matches `RectangularQAPILM.triplot2D()`:

```
[ Real C | QAPILM C | PINN C | log(Real/PINN) ]
```
(the QAPILM panel is omitted when `qapilm_C=None`).

**Plotting loss history:**

```python
pinn.plot_loss_history(save_path="loss_history.png")
```

### PINN config reference

| Class | Key parameters |
|-------|---------------|
| `PINNGeomConfig` | `Lh`, `Lv`, `Lt`, `u0`, `Rcv`, `bcs` |
| `PINNNetConfig`  | `u_hidden_layers`, `u_hidden_width`, `c_hidden_layers`, `c_hidden_width`, `activation`, `use_mixed_formulation` |
| `PINNTrainConfig`| `epochs`, `lr`, `lr_decay`, `w_pde`, `w_bc`, `w_ic`, `w_data_u`, `w_data_c`, `n_colloc_pde` |
| `PINNObsConfig`  | `density_mode`, `u_density`, `c_density`, `u_locations`, `c_locations`, `time_indices`, `n_time_obs` |
| `PINNSaveConfig` | `model_dir`, `save_metadata` |

> **Mixed formulation** (`PINNNetConfig.use_mixed_formulation = True`):
> Introduces auxiliary networks for u_x and u_z so the PDE residual only
> requires first-order autograd, reducing training time by 2–5 ×.  Enable
> for large grids or long training runs.
