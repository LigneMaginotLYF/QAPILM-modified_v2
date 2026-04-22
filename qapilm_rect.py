import os
import itertools
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import diags
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ============================================================
# CONFIG STRUCTURES
# ============================================================
@dataclass
class ProblemConfig:
    # Geometry
    Lh: float = 10.0
    Lv: float = 5.0
    Lt: float = 2.0
    spatial_reso: tuple = (0.2, 0.2)

    # Material / field generation
    Rcv: float = 1.0
    sigma: float = 0.3
    coeft: tuple = (-1, -0.5, 1, 0.5, 2, -0.2)
    u0: float = 1.0
    alpha: float = 0.05
    const: float = -1.0

    # External fluctuation files (backward compatible)
    filedir: str = 'F:/博/PINN 2D consoli practice/250827 - Hindered by illness/Relocated to Python/lib/'
    file_rawS: str = 'rawSlistA-0.3sig-2by2.csv'
    file_Lmat: str = 'Lmat-2by2.csv'

    # BCs (top, bot, left, right)
    bcs: tuple = (0, 1, 0, 1)

    # Fluctuation control
    regen_fluc: bool = False
    fluc_seed: int = 123


@dataclass
class BasisConfig:
    # Basis type – one of the solo names or a duo joined with "+":
    #
    #  Solo (clean, no forced poly prefix):
    #    "poly"      – 6-term 2D polynomial  [1, x, z, x², z², xz]
    #    "sin"       – sine harmonics  [sin(2^k x), sin(2^k z)]
    #    "dct"       – discrete cosine  [cos(π k x), cos(π k z)]
    #    "legendre"  – Legendre polynomials  [P_k(x), P_k(z)]
    #    "chebyshev" – Chebyshev polynomials of the first kind  [T_k(x), T_k(z)]
    #    "wavelet"   – Haar wavelets
    #    "rbf"       – 1-D Gaussian RBFs along x and z separately
    #    "bspline"   – uniform cubic B-splines along x and z separately
    #
    #  Duo (any two components separated by "+"):
    #    e.g. "poly+sin", "poly+chebyshev", "rbf+bspline", etc.
    #
    #  Backward-compatible alias:
    #    "poly_sin"  →  treated as "poly+sin"
    type: str = "poly+sin"
    # Number of sine/DCT/Legendre/Chebyshev terms per axis
    orderx: int = 1
    orderz: int = 3
    # Haar wavelet decomposition levels (used by wavelet)
    wav_levels_x: int = 2
    wav_levels_z: int = 2
    # RBF: number of Gaussian RBF centres per axis and shape parameter
    rbf_centers_x: int = 5
    rbf_centers_z: int = 5
    rbf_shape: float = 1.0
    # B-spline: number of interior knots per axis and polynomial degree
    bspline_nknots_x: int = 6
    bspline_nknots_z: int = 6
    bspline_degree: int = 3


@dataclass
class ModelConfig:
    lam: float = 1.0
    lamu: float = 1.0
    lr: float = 0.1
    ltol: float = 1e-5
    gtol: float = 1e-9
    itol: int = 500
    epsilon: float = 0.1


@dataclass
class SolverConfig:
    # memory mode: "full" (legacy default) or "stream" (memory-optimized)
    memory_mode: str = "full"
    # store snapshots for u(t) only at ukt indices
    store_u_snapshots: bool = True


@dataclass
class RunConfig:
    # output format (kept)
    results_dir: str = "./results"
    save_losses: bool = True
    loss_file: str = "QAPILM_Eps_loss.npy"
    # CSV for batch collection (optional)
    coeffs_csv: str = "coeffs_batch.csv"


# ============================================================
# UTILITIES
# ============================================================
def create_tridiag_mat(n, main, upper, lower):
    diagonals = [main*np.ones(n), upper*np.ones(n-1), lower*np.ones(n-1)]
    offsets = [0, 1, -1]
    return diags(diagonals, offsets, shape=(n, n), format='csr')

def map_vectors(vectors, func):
    return np.array([func(v) for v in vectors])


def advanced_resize_matrix(matrix, target, interpolation='cubic', preserve_aspect_ratio=False, mode='reflect'):
    from scipy import ndimage
    input_shape = matrix.shape
    interpolation_map = {'nearest': 0, 'linear': 1, 'cubic': 3, 'lanczos': 5}
    order = interpolation_map[interpolation]

    if np.isscalar(target):
        scale_factor = target
        target_shape = (int(input_shape[0] * scale_factor), int(input_shape[1] * scale_factor))
    elif len(target) == 2:
        if preserve_aspect_ratio:
            aspect_ratio = input_shape[1] / input_shape[0]
            if target[0] is not None and target[1] is not None:
                scale_x = target[1] / input_shape[1]
                scale_y = target[0] / input_shape[0]
                scale_factor = min(scale_x, scale_y)
                target_shape = (int(input_shape[0] * scale_factor), int(input_shape[1] * scale_factor))
            elif target[0] is not None:
                scale_factor = target[0] / input_shape[0]
                target_shape = (target[0], int(input_shape[1] * scale_factor))
            else:
                scale_factor = target[1] / input_shape[1]
                target_shape = (int(input_shape[0] * scale_factor), target[1])
        else:
            target_shape = target
    else:
        raise ValueError("target must be scalar or len-2 tuple/list")

    target_shape = (max(1, target_shape[0]), max(1, target_shape[1]))
    zoom_factors = (target_shape[0] / input_shape[0], target_shape[1] / input_shape[1])
    resized = ndimage.zoom(matrix, zoom_factors, order=order, mode=mode)
    return np.array(resized)

def softplus(x):
    return np.log1p(np.exp(x))

def dsoftplus(x):
    return 1 / (1 + np.exp(-x))

def isoftplus(y):
    return np.log1p(np.expm1(y))

def epsilon_insensitive_loss_numpy(pred, real, epsilon=0.1):
    pred = np.asarray(pred)
    real = np.asarray(real)
    err = real - pred
    real_safe = np.where(np.abs(real) < 0.1, 0.1, np.abs(real))
    t = (epsilon * real_safe) ** 2
    err_sq = err ** 2
    loss = np.maximum(0, err_sq - t)
    mask = (err_sq > t).astype(float)
    grad = -2.0 * err * mask
    return loss, grad

def cos_sim(tenA, tenB):
    vecA = tenA.flatten()
    vecB = tenB.flatten()
    return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))


# ============================================================
# BASIS FACTORY
# ============================================================

# Valid solo component names.
_VALID_COMPONENTS = frozenset(["poly", "sin", "dct", "legendre", "chebyshev", "wavelet", "rbf", "bspline"])

# Backward-compatible aliases: old name → new canonical "+"-separated string.
_BASIS_ALIASES = {
    "poly_sin": "poly+sin",
}


def _parse_basis_type(type_str: str) -> list:
    """Parse a basis type string into an ordered list of component names.

    Accepted formats
    ----------------
    "<name>"      – solo component, e.g.  "poly",  "dct"
    "<A>+<B>"     – duo combination, e.g. "poly+sin",  "dct+legendre"

    The legacy alias "poly_sin" is silently mapped to "poly+sin".
    At most two components are supported; repeating the same component is
    an error.  Unknown component names raise a clear ValueError.
    """
    canonical = _BASIS_ALIASES.get(type_str, type_str)
    parts = [p.strip() for p in canonical.split("+")]

    if not parts or any(not p for p in parts):
        raise ValueError(
            f"Invalid basis type string: {type_str!r}.  "
            f"Use a component name or 'A+B'."
        )
    if len(parts) > 2:
        raise ValueError(
            f"Basis type {type_str!r} has {len(parts)} components; "
            "at most 2 are supported (solo or duo)."
        )
    unknown = [p for p in parts if p not in _VALID_COMPONENTS]
    if unknown:
        raise ValueError(
            f"Unknown basis component(s) {unknown!r} in type {type_str!r}.  "
            f"Valid components: {sorted(_VALID_COMPONENTS)}"
        )
    if len(parts) == 2 and parts[0] == parts[1]:
        raise ValueError(
            f"Duo basis type {type_str!r} repeats the same component."
        )
    return parts


class BasisFactory:
    """Builds basis-function vectors for the QAPILM estimator.

    The basis is determined by ``BasisConfig.type`` which may name a single
    component ("poly", "sin", "dct", "legendre", "wavelet") or two components
    joined with "+" (e.g. "poly+sin", "dct+legendre").  Each component
    contributes its terms independently; they are concatenated in order.

    The backward-compatible alias "poly_sin" is equivalent to "poly+sin".
    """

    def __init__(self, pconf: ProblemConfig, bconf: BasisConfig):
        self.p = pconf
        self.b = bconf
        self._components = _parse_basis_type(bconf.type)

    # ------------------------------------------------------------------
    # Per-component builders  (normalised coords x, z ∈ [0,1])
    # ------------------------------------------------------------------

    # ---- polynomial (2D, has cross-terms) ----------------------------
    def _poly_val(self, x, z):
        return np.array([1.0, x, z, x**2, z**2, x*z])

    def _poly_dx(self, x, z):
        return np.array([0.0, 1.0, 0.0, 2*x, 0.0, z]) / self.p.Lh

    def _poly_dz(self, x, z):
        return np.array([0.0, 0.0, 1.0, 0.0, 2*z, x]) / self.p.Lv

    # ---- sine harmonics ----------------------------------------------
    def _sin_val_x(self, x):
        return np.sin(2**np.arange(self.b.orderx) * x)

    def _sin_val_z(self, z):
        return np.sin(2**np.arange(self.b.orderz) * z)

    def _sin_dx_x(self, x):
        freqs = 2**np.arange(self.b.orderx)
        return np.cos(freqs * x) * freqs / self.p.Lh

    def _sin_dz_z(self, z):
        freqs = 2**np.arange(self.b.orderz)
        return np.cos(freqs * z) * freqs / self.p.Lv

    # ---- DCT ---------------------------------------------------------
    def _dct_val_x(self, x):
        k = np.arange(self.b.orderx)
        return np.cos(np.pi * k * x)

    def _dct_val_z(self, z):
        k = np.arange(self.b.orderz)
        return np.cos(np.pi * k * z)

    def _dct_dx_x(self, x):
        k = np.arange(self.b.orderx)
        return -np.pi * k * np.sin(np.pi * k * x) / self.p.Lh

    def _dct_dz_z(self, z):
        k = np.arange(self.b.orderz)
        return -np.pi * k * np.sin(np.pi * k * z) / self.p.Lv

    # ---- Legendre polynomials ----------------------------------------
    def _legendre_val(self, x_norm, order):
        from numpy.polynomial.legendre import legval
        xm = 2.0 * x_norm - 1.0   # map [0,1] → [-1,1]
        vals = []
        for deg in range(order):
            coeff = np.zeros(deg + 1)
            coeff[-1] = 1.0
            vals.append(legval(xm, coeff))
        return np.array(vals)

    def _legendre_dval(self, x_norm, order, L):
        eps = 1e-6
        return (
            self._legendre_val(x_norm + eps, order)
            - self._legendre_val(x_norm - eps, order)
        ) / (2.0 * eps * L)

    # ---- Haar wavelet ------------------------------------------------
    def _haar_val(self, x_norm, levels):
        vals = [1.0]
        for lev in range(1, levels + 1):
            num = 2**lev
            for k in range(num):
                lo, mid, hi = k / num, (k + 0.5) / num, (k + 1) / num
                if lo <= x_norm < mid:
                    vals.append(1.0)
                elif mid <= x_norm < hi:
                    vals.append(-1.0)
                else:
                    vals.append(0.0)
        return np.array(vals)

    def _haar_size(self, levels):
        return 1 + sum(2**lev for lev in range(1, levels + 1))

    # ---- Chebyshev polynomials of the first kind ---------------------
    def _chebyshev_val(self, x_norm, order):
        xm = 2.0 * x_norm - 1.0   # map [0,1] → [-1,1]
        vals = []
        for deg in range(order):
            coeff = np.zeros(deg + 1)
            coeff[-1] = 1.0
            vals.append(float(np.polynomial.chebyshev.chebval(xm, coeff)))
        return np.array(vals)

    def _chebyshev_dval(self, x_norm, order, L):
        eps = 1e-6
        return (
            self._chebyshev_val(x_norm + eps, order)
            - self._chebyshev_val(x_norm - eps, order)
        ) / (2.0 * eps * L)

    # ---- Gaussian RBF (1-D centres along each axis) -----------------
    def _rbf_val(self, x_norm, n_centers, shape):
        """Gaussian RBF values at x_norm ∈ [0,1] for n_centers equi-spaced centres."""
        centers = np.linspace(0.0, 1.0, n_centers)
        r2 = (x_norm - centers) ** 2
        return np.exp(-(shape ** 2) * r2)

    def _rbf_dval(self, x_norm, n_centers, shape, L):
        """Derivative of Gaussian RBFs w.r.t. the physical coordinate (length L)."""
        centers = np.linspace(0.0, 1.0, n_centers)
        r = x_norm - centers
        r2 = r ** 2
        return -2.0 * (shape ** 2) * r * np.exp(-(shape ** 2) * r2) / L

    # ---- Uniform cubic B-splines (1-D) --------------------------------
    def _bspline_basis(self, x_norm, n_knots, degree):
        """Evaluate all B-spline basis functions at x_norm ∈ [0,1].

        Uses a clamped uniform knot vector with *n_knots* interior knots and
        *degree* (default 3 = cubic).  Returns an array of length
        n_knots + degree + 1.
        """
        from scipy.interpolate import BSpline
        n_int = n_knots
        knots_int = np.linspace(0.0, 1.0, n_int + 2)[1:-1]  # interior knots
        t = np.concatenate(
            [np.zeros(degree + 1), knots_int, np.ones(degree + 1)]
        )
        n_basis = len(t) - degree - 1
        coeff_matrix = np.eye(n_basis)
        vals = np.array([
            BSpline(t, coeff_matrix[i], degree)(x_norm)
            for i in range(n_basis)
        ])
        return vals

    def _bspline_dbasis(self, x_norm, n_knots, degree, L):
        """Derivative of B-spline basis functions w.r.t. the physical coordinate."""
        eps = 1e-6
        hi = self._bspline_basis(min(x_norm + eps, 1.0), n_knots, degree)
        lo = self._bspline_basis(max(x_norm - eps, 0.0), n_knots, degree)
        return (hi - lo) / (2.0 * eps * L)


    # ------------------------------------------------------------------
    # Component dispatchers  (val / dx / dz for a named component)
    # ------------------------------------------------------------------

    def _comp_val(self, name, x, z):
        if name == "poly":
            return self._poly_val(x, z)
        if name == "sin":
            return np.concatenate([self._sin_val_x(x), self._sin_val_z(z)])
        if name == "dct":
            return np.concatenate([self._dct_val_x(x), self._dct_val_z(z)])
        if name == "legendre":
            return np.concatenate([
                self._legendre_val(x, self.b.orderx),
                self._legendre_val(z, self.b.orderz),
            ])
        if name == "chebyshev":
            return np.concatenate([
                self._chebyshev_val(x, self.b.orderx),
                self._chebyshev_val(z, self.b.orderz),
            ])
        if name == "wavelet":
            return np.concatenate([
                self._haar_val(x, self.b.wav_levels_x),
                self._haar_val(z, self.b.wav_levels_z),
            ])
        if name == "rbf":
            return np.concatenate([
                self._rbf_val(x, self.b.rbf_centers_x, self.b.rbf_shape),
                self._rbf_val(z, self.b.rbf_centers_z, self.b.rbf_shape),
            ])
        if name == "bspline":
            return np.concatenate([
                self._bspline_basis(x, self.b.bspline_nknots_x, self.b.bspline_degree),
                self._bspline_basis(z, self.b.bspline_nknots_z, self.b.bspline_degree),
            ])
        raise ValueError(f"Unknown component: {name!r}")

    def _comp_dx(self, name, x, z):
        """Derivative w.r.t. physical x of the component vector."""
        if name == "poly":
            return self._poly_dx(x, z)
        if name == "sin":
            return np.concatenate([
                self._sin_dx_x(x),
                np.zeros(self.b.orderz),
            ])
        if name == "dct":
            return np.concatenate([
                self._dct_dx_x(x),
                np.zeros(self.b.orderz),
            ])
        if name == "legendre":
            return np.concatenate([
                self._legendre_dval(x, self.b.orderx, self.p.Lh),
                np.zeros(self.b.orderz),
            ])
        if name == "chebyshev":
            return np.concatenate([
                self._chebyshev_dval(x, self.b.orderx, self.p.Lh),
                np.zeros(self.b.orderz),
            ])
        if name == "wavelet":
            # Haar wavelets are piecewise-constant → derivative ≈ 0
            return np.zeros(
                self._haar_size(self.b.wav_levels_x)
                + self._haar_size(self.b.wav_levels_z)
            )
        if name == "rbf":
            return np.concatenate([
                self._rbf_dval(x, self.b.rbf_centers_x, self.b.rbf_shape, self.p.Lh),
                np.zeros(self.b.rbf_centers_z),
            ])
        if name == "bspline":
            n_basis_z = self.b.bspline_nknots_z + self.b.bspline_degree + 1
            return np.concatenate([
                self._bspline_dbasis(x, self.b.bspline_nknots_x, self.b.bspline_degree, self.p.Lh),
                np.zeros(n_basis_z),
            ])
        raise ValueError(f"Unknown component: {name!r}")

    def _comp_dz(self, name, x, z):
        """Derivative w.r.t. physical z of the component vector."""
        if name == "poly":
            return self._poly_dz(x, z)
        if name == "sin":
            return np.concatenate([
                np.zeros(self.b.orderx),
                self._sin_dz_z(z),
            ])
        if name == "dct":
            return np.concatenate([
                np.zeros(self.b.orderx),
                self._dct_dz_z(z),
            ])
        if name == "legendre":
            return np.concatenate([
                np.zeros(self.b.orderx),
                self._legendre_dval(z, self.b.orderz, self.p.Lv),
            ])
        if name == "chebyshev":
            return np.concatenate([
                np.zeros(self.b.orderx),
                self._chebyshev_dval(z, self.b.orderz, self.p.Lv),
            ])
        if name == "wavelet":
            return np.zeros(
                self._haar_size(self.b.wav_levels_x)
                + self._haar_size(self.b.wav_levels_z)
            )
        if name == "rbf":
            return np.concatenate([
                np.zeros(self.b.rbf_centers_x),
                self._rbf_dval(z, self.b.rbf_centers_z, self.b.rbf_shape, self.p.Lv),
            ])
        if name == "bspline":
            n_basis_x = self.b.bspline_nknots_x + self.b.bspline_degree + 1
            return np.concatenate([
                np.zeros(n_basis_x),
                self._bspline_dbasis(z, self.b.bspline_nknots_z, self.b.bspline_degree, self.p.Lv),
            ])
        raise ValueError(f"Unknown component: {name!r}")

    # ------------------------------------------------------------------
    # Public API  (unchanged interface)
    # ------------------------------------------------------------------

    def basis(self, vector):
        x, z = np.array(vector) / np.array([self.p.Lh, self.p.Lv])
        return np.concatenate(
            [self._comp_val(c, x, z) for c in self._components], axis=None
        )

    def basis_dx(self, vector):
        x, z = np.array(vector) / np.array([self.p.Lh, self.p.Lv])
        return np.concatenate(
            [self._comp_dx(c, x, z) for c in self._components], axis=None
        )

    def basis_dz(self, vector):
        x, z = np.array(vector) / np.array([self.p.Lh, self.p.Lv])
        return np.concatenate(
            [self._comp_dz(c, x, z) for c in self._components], axis=None
        )


# ============================================================
# MAIN SOLVER
# ============================================================
class RectangularQAPILM:
    def __init__(self, pconf: ProblemConfig, bconf: BasisConfig, mconf: ModelConfig, sconf: SolverConfig, rconf: RunConfig):
        self.p = pconf
        self.b = bconf
        self.m = mconf
        self.s = sconf
        self.r = rconf

        # grid
        self.dx, self.dz = self.p.spatial_reso
        self.numh = int(self.p.Lh / self.dx)
        self.numv = int(self.p.Lv / self.dz)
        self.cpth = np.arange(0, self.p.Lh + 0.001, self.dx)
        self.cptv = np.arange(0, self.p.Lv + 0.001, self.dz)
        self.cpts2 = list(itertools.product(self.cpth, self.cptv))
        self.vec2mat2 = lambda vector: np.array(vector).reshape(self.numh+1, self.numv+1).transpose(1,0)

        # load random config files
        self.rawS, self.Lmat = self._load_external_files()

        # build truth and operators
        self._build_truth_and_operators()

        # estimator basis
        self._build_estimator_basis()

    def _load_external_files(self):
        rawS = np.loadtxt(open(os.path.join(self.p.filedir, self.p.file_rawS), 'rt'), delimiter=',')
        Lmat = np.loadtxt(open(os.path.join(self.p.filedir, self.p.file_Lmat), 'rt'), delimiter=',')
        return rawS, Lmat

    def _build_truth_and_operators(self):
        # truth trend bases (same as original)
        def baset(vector):
            x, z = (np.array(vector) / np.array([self.p.Lh, self.p.Lv]))
            return np.array([1, x, z, x**2, z**2, x*z])

        def basetdx(vector):
            x, z = (np.array(vector) / np.array([self.p.Lh, self.p.Lv]))
            return np.array([0, 1/self.p.Lh, 0, 2*x/self.p.Lh, 0, z/self.p.Lh])

        def basetdz(vector):
            x, z = (np.array(vector) / np.array([self.p.Lh, self.p.Lv]))
            return np.array([0, 0, 1/self.p.Lv, 0, 2*z/self.p.Lv, x/self.p.Lv])

        basest = map_vectors(self.cpts2, baset)
        trend = basest @ np.array(self.p.coeft)

        if self.p.regen_fluc:
            np.random.seed(self.p.fluc_seed)
            rawSlist = np.random.randn(*np.array(self.rawS).shape)
        else:
            rawSlist = np.array(self.rawS)

        fluc_o = self.p.sigma * np.array(self.Lmat).T @ rawSlist
        flucmat = fluc_o.reshape(50+1,25+1).transpose(1,0)
        res_fluc = advanced_resize_matrix(flucmat, [self.numv+1, self.numh+1])
        fluc = np.reshape(res_fluc.transpose(1,0), -1)

        self.trend = trend
        self.fluc = fluc
        self.Olist = np.exp(fluc) + trend - self.p.const
        self.chm = self.vec2mat2(self.Olist)
        self.chv = self.chm * self.p.Rcv

        basestdx = map_vectors(self.cpts2, basetdx)
        basestdz = map_vectors(self.cpts2, basetdz)
        trenddx = basestdx @ np.array(self.p.coeft)
        trenddz = basestdz @ np.array(self.p.coeft)

        o1 = create_tridiag_mat(self.numh+1, 0, 1, -1) / (self.dx*2)
        o2 = create_tridiag_mat(self.numv+1, 0, 1, -1) / (self.dz*2)
        self.d1 = self.vec2mat2(trenddx) + res_fluc * (o1.T.dot(res_fluc.T)).T
        self.d2 = self.vec2mat2(trenddz) + res_fluc * o2.dot(res_fluc)

        ah = self.p.alpha*(self.dx**2)/np.max(self.chm)
        az = self.p.alpha*(self.dz**2)/np.max(self.chv)
        bh = self.p.alpha*(self.dx*2)/np.max(self.d1+0.001)
        bz = self.p.alpha*(self.dz*2)/np.max(self.d2+0.001)
        self.dt = np.min([ah, az, bh, bz])
        self.ilim = int(self.p.Lt // self.dt)

        self.a1 = create_tridiag_mat(self.numh+1, -2, 1, 1) * self.dt / (self.dx**2)
        self.a2 = create_tridiag_mat(self.numv+1, -2, 1, 1) * self.dt / (self.dz**2)
        self.b1 = create_tridiag_mat(self.numh+1, 0, 1, -1) * self.dt / (self.dx*2)
        self.b2 = create_tridiag_mat(self.numv+1, 0, 1, -1) * self.dt / (self.dz*2)

        print('Initialization complete! dt =', self.dt, 'year')
        print('Cmax=', np.max(self.chm), '; Cmin=', np.min(self.chm))

    def _build_estimator_basis(self):
        factory = BasisFactory(self.p, self.b)
        self.basese = map_vectors(self.cpts2, factory.basis)
        self.basesedx = map_vectors(self.cpts2, factory.basis_dx)
        self.basesedz = map_vectors(self.cpts2, factory.basis_dz)

    # =========================================================
    # Forward solver
    # =========================================================
    def forward_solver(self, chm, numt, u0, top, bot, left, right):
        chv = chm * self.p.Rcv
        utens = np.zeros((numt+1, self.numv+1, self.numh+1), dtype=np.float64)
        utens[0,:,:] = u0
        udeg = np.ones(numt+1, dtype=np.float64)
        # udeg normalization requires a scalar reference value;
        # when u0 is a matrix (uniform IC) take its mean.
        u0_scalar = float(np.mean(u0)) if np.ndim(u0) > 0 else float(u0)

        for i in range(numt):
            cu = utens[i,:,:]
            utens[i+1,:,:] = cu + chm*((self.a1.T.dot(cu.T)).T) + chv*(self.a2.dot(cu)) + self.d1*((self.b1.T.dot(cu.T)).T) + self.d2*(self.b2.dot(cu))
            utens[i+1,:,0]  = left  * utens[i+1,:,1]
            utens[i+1,:,-1] = right * utens[i+1,:,-2]
            utens[i+1,0,:]  = top   * utens[i+1,1,:]
            utens[i+1,-1,:] = bot   * utens[i+1,-2,:]
            udeg[i+1] = np.mean(utens[i+1]) / u0_scalar
        return utens, udeg

    # =========================================================
    # Inverse: streaming (memory-optimized)
    # =========================================================
    def inverse_solver_stream(self, ukt, u0, ukmat, chkmat, uk, chk, wlst=None):
        top, bot, left, right = self.p.bcs
        nb = len(self.basese[0])

        coefe = np.random.randn(nb) * 0.2
        print('random init:', coefe)

        losses = []
        beta1, beta2 = 0.9, 0.999
        mvec = np.zeros(nb)
        vvec = np.zeros(nb)

        for j in range(self.m.itol):
            m = self.vec2mat2(self.basese @ coefe)
            chem = softplus(m)
            chev = chem * self.p.Rcv
            dcm = dsoftplus(m)
            dcv = dcm * self.p.Rcv
            ddcm = dcm * (1-dcm)
            ddcv = ddcm * self.p.Rcv

            trendedx = self.vec2mat2(self.basesedx @ coefe)
            trendedz = self.vec2mat2(self.basesedz @ coefe)
            d1e = dcm * trendedx
            d2e = dcv * trendedz

            # current fields only
            u = np.zeros((self.numv+1, self.numh+1), dtype=np.float64)
            u[:,:] = u0
            s = np.zeros((self.numv+1, self.numh+1, nb), dtype=np.float64)

            loss_t = 0.0
            gradients = np.zeros(nb)
            t_set = set(ukt.tolist())

            for i in range(int(ukt[-1])):
                cu = u
                u = cu + chem*((self.a1.T.dot(cu.T)).T) + chev*(self.a2.dot(cu)) + d1e*((self.b1.T.dot(cu.T)).T) + d2e*(self.b2.dot(cu))
                u[:,0]  = left  * u[:,1]
                u[:,-1] = right * u[:,-2]
                u[0,:]  = top   * u[1,:]
                u[-1,:] = bot   * u[-2,:]

                # update sensitivities for each basis (no time history)
                for k in range(nb):
                    bk = self.vec2mat2(self.basese[:,k])
                    bdxk = self.vec2mat2(self.basesedx[:,k])
                    bdzk = self.vec2mat2(self.basesedz[:,k])
                    chdk = dcm * bk
                    cvdk = chdk * self.p.Rcv
                    d1dk = ddcm * bk * trendedx + bdxk * dcm
                    d2dk = ddcv * bk * trendedz + bdzk * dcv

                    csk = s[:,:,k]
                    s[:,:,k] = csk + chdk*((self.a1.T.dot(cu.T)).T) + chem*((self.a1.T.dot(csk.T)).T) + \
                               cvdk*(self.a2.dot(cu)) + chev*(self.a2.dot(csk)) + \
                               d1dk*((self.b1.T.dot(cu.T)).T) + d1e*((self.b1.T.dot(csk.T)).T) + \
                               d2dk*(self.b2.dot(cu)) + d2e*(self.b2.dot(csk))
                    s[:,0,k]  = left  * s[:,1,k]
                    s[:,-1,k] = right * s[:,-2,k]
                    s[0,:,k]  = top   * s[1,:,k]
                    s[-1,:,k] = bot   * s[-2,:,k]

                # accumulate loss only at measurement times
                if i in t_set:
                    idx = np.where(ukt == i)[0][0]
                    for pt in ukmat:
                        uiloss, uigrad = epsilon_insensitive_loss_numpy(u[pt[0], pt[1]], uk[idx, pt[0], pt[1]], self.m.epsilon)
                        loss_t += np.mean(uiloss) * self.m.lamu
                        for k in range(nb):
                            gradients[k] += np.mean(uigrad * s[pt[0], pt[1], k]) * self.m.lamu

            # permeability loss
            for pt in chkmat:
                ciloss, cigrad = epsilon_insensitive_loss_numpy(chem[pt[0], pt[1]], chk[pt[0], pt[1]], self.m.epsilon)
                loss_t += self.m.lam * np.mean(ciloss) * (1 + self.p.Rcv)
                for k in range(nb):
                    gradients[k] += self.m.lam * np.mean(cigrad * chem * self.vec2mat2(self.basese[:,k])[pt[0], pt[1]]) * (1 + self.p.Rcv)

            losses.append(loss_t)
            if j % 10 == 0:
                print('Epoch', j, ': Loss=', loss_t, '; gradients=', gradients)

            mvec = mvec*beta1 + (1-beta1)*gradients
            vvec = vvec*beta2 + (1-beta2)*(gradients**2)
            mhat = mvec / (1-beta1**(j+1))
            vhat = vvec / (1-beta2**(j+1))
            update = mhat / (np.sqrt(vhat)+1e-11)
            coefe -= self.m.lr * update

            if loss_t <= self.m.ltol or np.sum(update**2) <= self.m.gtol:
                print('Early-stopped at epoch', j)
                break

        if self.r.save_losses:
            np.save(self.r.loss_file, losses)

        print('final loss=', loss_t)
        return coefe

    # =========================================================
    # Inverse: full history (legacy, memory-heavy)
    # =========================================================
    def inverse_solver_full(self, ukt, u0, ukmat, chkmat, uk, chk, wlst=None):
        top, bot, left, right = self.p.bcs
        nb = len(self.basese[0])

        coefe = np.random.randn(nb) * 0.2
        print('random init:', coefe)

        losses = []
        beta1, beta2 = 0.9, 0.999
        mvec = np.zeros(nb)
        vvec = np.zeros(nb)

        tmax = int(ukt[-1])
        t_set = set(ukt.tolist())

        for j in range(self.m.itol):
            m = self.vec2mat2(self.basese @ coefe)
            chem = softplus(m)
            chev = chem * self.p.Rcv
            dcm = dsoftplus(m)
            dcv = dcm * self.p.Rcv
            ddcm = dcm * (1-dcm)
            ddcv = ddcm * self.p.Rcv

            trendedx = self.vec2mat2(self.basesedx @ coefe)
            trendedz = self.vec2mat2(self.basesedz @ coefe)
            d1e = dcm * trendedx
            d2e = dcv * trendedz

            # full time history (legacy behavior; memory-heavy by design)
            u_hist = np.zeros((tmax + 1, self.numv+1, self.numh+1), dtype=np.float64)
            u_hist[0,:,:] = u0
            s_hist = np.zeros((tmax + 1, self.numv+1, self.numh+1, nb), dtype=np.float64)

            loss_t = 0.0
            gradients = np.zeros(nb)

            for i in range(tmax):
                cu = u_hist[i,:,:]
                un = cu + chem*((self.a1.T.dot(cu.T)).T) + chev*(self.a2.dot(cu)) + d1e*((self.b1.T.dot(cu.T)).T) + d2e*(self.b2.dot(cu))
                un[:,0]  = left  * un[:,1]
                un[:,-1] = right * un[:,-2]
                un[0,:]  = top   * un[1,:]
                un[-1,:] = bot   * un[-2,:]
                u_hist[i+1,:,:] = un

                # update sensitivities for each basis (with full time history)
                for k in range(nb):
                    bk = self.vec2mat2(self.basese[:,k])
                    bdxk = self.vec2mat2(self.basesedx[:,k])
                    bdzk = self.vec2mat2(self.basesedz[:,k])
                    chdk = dcm * bk
                    cvdk = chdk * self.p.Rcv
                    d1dk = ddcm * bk * trendedx + bdxk * dcm
                    d2dk = ddcv * bk * trendedz + bdzk * dcv

                    csk = s_hist[i,:,:,k]
                    sn = csk + chdk*((self.a1.T.dot(cu.T)).T) + chem*((self.a1.T.dot(csk.T)).T) + \
                         cvdk*(self.a2.dot(cu)) + chev*(self.a2.dot(csk)) + \
                         d1dk*((self.b1.T.dot(cu.T)).T) + d1e*((self.b1.T.dot(csk.T)).T) + \
                         d2dk*(self.b2.dot(cu)) + d2e*(self.b2.dot(csk))
                    sn[:,0]  = left  * sn[:,1]
                    sn[:,-1] = right * sn[:,-2]
                    sn[0,:]  = top   * sn[1,:]
                    sn[-1,:] = bot   * sn[-2,:]
                    s_hist[i+1,:,:,k] = sn

                # preserve existing time-index matching semantics
                if i in t_set:
                    idx = np.where(ukt == i)[0][0]
                    for pt in ukmat:
                        uiloss, uigrad = epsilon_insensitive_loss_numpy(u_hist[i+1, pt[0], pt[1]], uk[idx, pt[0], pt[1]], self.m.epsilon)
                        loss_t += np.mean(uiloss) * self.m.lamu
                        for k in range(nb):
                            gradients[k] += np.mean(uigrad * s_hist[i+1, pt[0], pt[1], k]) * self.m.lamu

            # permeability loss
            for pt in chkmat:
                ciloss, cigrad = epsilon_insensitive_loss_numpy(chem[pt[0], pt[1]], chk[pt[0], pt[1]], self.m.epsilon)
                loss_t += self.m.lam * np.mean(ciloss) * (1 + self.p.Rcv)
                for k in range(nb):
                    gradients[k] += self.m.lam * np.mean(cigrad * chem * self.vec2mat2(self.basese[:,k])[pt[0], pt[1]]) * (1 + self.p.Rcv)

            losses.append(loss_t)
            if j % 10 == 0:
                print('Epoch', j, ': Loss=', loss_t, '; gradients=', gradients)

            mvec = mvec*beta1 + (1-beta1)*gradients
            vvec = vvec*beta2 + (1-beta2)*(gradients**2)
            mhat = mvec / (1-beta1**(j+1))
            vhat = vvec / (1-beta2**(j+1))
            update = mhat / (np.sqrt(vhat)+1e-11)
            coefe -= self.m.lr * update

            if loss_t <= self.m.ltol or np.sum(update**2) <= self.m.gtol:
                print('Early-stopped at epoch', j)
                break

        if self.r.save_losses:
            np.save(self.r.loss_file, losses)

        print('final loss=', loss_t)
        return coefe

    # Backward-compatible API name retained for external callers.
    def inverse_solver(self, ukt, u0, ukmat, chkmat, uk, chk, wlst=None):
        return self.inverse_solver_full(ukt, u0, ukmat, chkmat, uk, chk, wlst=wlst)

    # =========================================================
    # plotting/output (format retained)
    # =========================================================
    def triplot2D(self, data1, data2, titles, mode, cmap1='viridis', cmap2='bwr', vmin12=None, vmax12=None, vmin3=None, vmax3=None, psavepath=None):
        data3 = np.log(data1/data2) if mode==1 else data1-data2
        xticks=[0,int(self.numh/4),int(self.numh/2),int(self.numh*3/4),self.numh]
        xtlabel=np.array([0,1/4,1/2,3/4,1])*self.p.Lh
        zticks=[0,int(self.numv/4),int(self.numv/2),int(self.numv*3/4),self.numv]
        ztlabel=[0,self.p.Lv/4,self.p.Lv/2,self.p.Lv*3/4,self.p.Lv]

        fig, axes = plt.subplots(1,3,figsize=(15,5), dpi=100)
        if vmin12 is None: vmin12 = min(np.min(data1), np.min(data2))
        if vmax12 is None: vmax12 = max(np.max(data1), np.max(data2))
        norm12 = colors.Normalize(vmin=vmin12, vmax=vmax12)

        im1 = axes[0].imshow(data1, cmap=cmap1, norm=norm12, origin='upper', aspect='auto', interpolation='bicubic')
        axes[0].set_title(titles[0]); axes[0].set_xlabel('r(m)'); axes[0].set_ylabel('z(m)')
        im2 = axes[1].imshow(data2, cmap=cmap1, norm=norm12, origin='upper', aspect='auto', interpolation='bicubic')
        axes[1].set_title(titles[1]); axes[1].set_xlabel('r(m)'); axes[1].set_ylabel('z(m)')

        if vmin3 is None: vmin3 = -np.abs(np.max(data3))
        if vmax3 is None: vmax3 = np.abs(np.max(data3))
        norm3 = colors.Normalize(vmin=vmin3, vmax=vmax3)
        im3 = axes[2].imshow(data3, cmap=cmap2, norm=norm3, origin='upper', aspect='auto', interpolation='bicubic')
        axes[2].set_title(titles[2]); axes[2].set_xlabel('X(m)'); axes[2].set_ylabel('Z(m)')

        for ax in axes:
            ax.set_xticks(xticks); ax.set_xticklabels(xtlabel)
            ax.set_yticks(zticks); ax.set_yticklabels(ztlabel)

        divider1 = make_axes_locatable(axes[1]); cax12 = divider1.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im2, cax=cax12)
        divider3 = make_axes_locatable(axes[2]); cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im3, cax=cax3)

        plt.tight_layout()
        if psavepath is None:
            plt.show()
        else:
            plt.savefig(psavepath, dpi=300, bbox_inches='tight')
            plt.close()
        return fig
