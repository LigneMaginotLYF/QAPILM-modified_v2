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

---

## Dependencies

Standard Python scientific stack plus **PyYAML**:

```bash
pip install numpy scipy matplotlib seaborn pyyaml
```

---

## Quick start – single run

```python
from qapilm_rect import (
    ProblemConfig, BasisConfig, ModelConfig,
    SolverConfig, RunConfig, RectangularQAPILM,
)
import numpy as np

solver = RectangularQAPILM(
    ProblemConfig(), BasisConfig(), ModelConfig(),
    SolverConfig(), RunConfig(),
)
u0 = np.ones((solver.numv + 1, solver.numh + 1))
utens, udeg = solver.forward_solver(solver.chm, solver.ilim, u0, *solver.p.bcs)
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
| `basis.type` | Basis type: `poly_sin` · `dct` · `legendre` · `wavelet` |
| `model.epsilon` | ε-insensitive loss threshold (relative to observation magnitude) |
| `solver.memory_mode` | `stream` (default, memory-optimised) or `full` (legacy) |
| `solver.store_u_snapshots` | Whether to keep `u` snapshots during the inverse solve |
| `measurements.ukmat` / `chkmat` | Grid-index pairs `[row, col]` for `u` / `C` observations |
| `measurements.ukt` | Time-step indices at which `u` is observed |

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
    QAPILM_Eps_loss.npy   # loss curve
    coefe.npy             # optimised basis coefficients
    ch_est.npy            # estimated C field
    ch_true.npy           # ground-truth C field
    triplot_C.png         # three-panel comparison plot
  eps_0.05_20250101_120100/
    ...
  coeffs_batch.csv        # summary table: one row per run
```

`coeffs_batch.csv` contains columns for `run_name`, `basis_type`,
`epsilon`, `regen_fluc`, `fluc_seed`, `Rcv`, `final_loss`, `cos_sim`,
`RMSE`, `max_err`, and all optimised coefficients (`coef_0`, `coef_1`, …).

---

## Configuring basis types

| `basis.type` | Additional keys | Notes |
|--------------|-----------------|-------|
| `poly_sin` | `orderx`, `orderz` | Polynomial + sine harmonics (default) |
| `dct` | `orderx`, `orderz` | Polynomial + discrete cosine terms |
| `legendre` | `orderx`, `orderz` | Polynomial + Legendre polynomials |
| `wavelet` | `wav_levels_x`, `wav_levels_z` | Polynomial + Haar wavelets |

Higher `orderx` / `orderz` increases the number of basis functions and
the expressiveness of the estimated `C` field, at the cost of more
computation and potential overfitting with sparse data.

---

## Notes

- **Backward compatibility**: existing output file names and figure formats
  produced by `qapilm_rect.py` are preserved unchanged.
- **Memory**: `solver.memory_mode: stream` (default) uses O(1) memory in
  the time dimension; switch to `full` only if you need the full `u` tensor.
- **Reproducibility**: set `regen_fluc: true` and fix `fluc_seed` for
  deterministic runs independent of the legacy CSV files.

