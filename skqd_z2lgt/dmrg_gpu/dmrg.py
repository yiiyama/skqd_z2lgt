"""Two-site DMRG on the active NumPy/CuPy backend.

Index conventions:
    MPS site tensor  A[l, s, r]
    MPO site tensor  W[a, b, s, t]  (= <s|op|t>)
    Left env         L[bra, w, ket]
    Right env        R[bra, w, ket]
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Callable, Optional
import numpy as np
from .backend import xp
from .sweeps import Sweeps

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# MPS construction
# --------------------------------------------------------------------------------------------------

def product_mps(bits: np.ndarray | None = None, num_sites: int | None = None,
                state: str = 'zero', noise: float = 0., seed: int = 0) -> list:
    """Product state. With ``bits`` given, the computational basis state; otherwise 'zero' or
    'plus' on every site.

    A nonzero ``noise`` adds a random admixture to each site. This is needed when starting two-site
    DMRG from a basis state of a Hamiltonian with off-diagonal terms acting on more than two sites
    (e.g. the XXX plaquette terms), whose effective two-site projection otherwise vanishes.
    """
    rng = np.random.default_rng(seed)
    if bits is not None:
        vecs = [np.eye(2)[int(b)] + noise * rng.normal(size=2) for b in bits]
        vecs = [v / np.linalg.norm(v) for v in vecs]
    elif state == 'plus':
        vecs = [np.full(2, np.sqrt(0.5))] * num_sites
    else:
        vecs = [np.eye(2)[0]] * num_sites
    return [xp.asarray_dev(v.reshape(1, 2, 1)) for v in vecs]


def random_mps(num_sites: int, bond_dim: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    mps = []
    dl = 1
    for site in range(num_sites):
        dr = min(bond_dim, 2 ** (site + 1), 2 ** (num_sites - site - 1))
        mps.append(xp.asarray_dev(rng.normal(size=(dl, 2, dr))))
        dl = dr
    return mps


def right_canonicalize(mps: list) -> list:
    """Bring the MPS into right-canonical form (site 0 carries the norm, then normalized)."""
    mps = list(mps)
    for site in range(len(mps) - 1, 0, -1):
        dl, d, dr = mps[site].shape
        q, r = xp.linalg.qr(mps[site].reshape(dl, d * dr).T)
        keep = q.shape[1]
        mps[site] = q.T.reshape(keep, d, dr)
        mps[site - 1] = xp.tensordot(mps[site - 1], r.T, axes=([2], [0]))
    mps[0] = mps[0] / xp.linalg.norm(mps[0])
    return mps


# --------------------------------------------------------------------------------------------------
# Environments and expectation values
# --------------------------------------------------------------------------------------------------

def update_left(env, a_bra, w, a_ket):
    tmp = xp.tensordot(env, a_ket, axes=([2], [0]))              # (al, a, t, kr)
    tmp = xp.tensordot(tmp, w, axes=([1, 2], [0, 3]))             # (al, kr, b, s)
    tmp = xp.tensordot(a_bra, tmp, axes=([0, 1], [0, 3]))         # (ar, kr, b)
    return tmp.transpose(0, 2, 1)


def update_right(env, a_bra, w, a_ket):
    tmp = xp.tensordot(a_ket, env, axes=([2], [2]))              # (kl, t, br, b)
    tmp = xp.tensordot(tmp, w, axes=([1, 3], [3, 1]))             # (kl, br, a, s)
    tmp = xp.tensordot(a_bra, tmp, axes=([1, 2], [3, 1]))         # (bl, kl, a)
    return tmp.transpose(0, 2, 1)


def overlap_left(env, a_bra, a_ket):
    """env[bra, ket] -> env'[bra', ket'] for the overlap of two different MPSs."""
    tmp = xp.tensordot(env, a_ket, axes=([1], [0]))              # (bl, t, kr)
    return xp.tensordot(a_bra, tmp, axes=([0, 1], [0, 1]))        # (br, kr)


def overlap_right(env, a_bra, a_ket):
    tmp = xp.tensordot(a_ket, env, axes=([2], [1]))              # (kl, t, br)
    return xp.tensordot(a_bra, tmp, axes=([1, 2], [1, 2]))        # (bl, kl)


def projected_state(olenv, a1, a2, orenv):
    """Projection of an MPS onto the current two-site variational space, flattened."""
    tmp = xp.tensordot(olenv, a1, axes=([1], [0]))               # (bl, t1, k)
    tmp = xp.tensordot(tmp, a2, axes=([2], [0]))                  # (bl, t1, t2, kr)
    tmp = xp.tensordot(tmp, orenv, axes=([3], [1]))               # (bl, t1, t2, br)
    return tmp.reshape(-1)


def overlap(mps_a: list, mps_b: list) -> float:
    env = xp.ones((1, 1), dtype=xp.dtype)
    for a, b in zip(mps_a, mps_b):
        env = overlap_left(env, a, b)
    return float(xp.scalar(env.reshape(-1)[0]))


def boundary_env():
    return xp.ones((1, 1, 1), dtype=xp.dtype)


def expectation(mps: list, mpo: list) -> float:
    """<psi|O|psi> / <psi|psi> for an MPO O on the device."""
    env = boundary_env()
    norm = boundary_env()
    ident = xp.asarray_dev(np.eye(2).reshape(1, 1, 2, 2))
    for a, w in zip(mps, mpo):
        env = update_left(env, a, w, a)
        norm = update_left(norm, a, ident, a)
    return float(xp.scalar(env.reshape(-1)[0])) / float(xp.scalar(norm.reshape(-1)[0]))


def update_left_double(env, a_bra, w, a_ket):
    """Left environment of <psi|W W|psi>: env[bra, w_top, w_bottom, ket]."""
    tmp = xp.tensordot(env, a_ket, axes=([3], [0]))              # (bl, wt, wb, t, kr)
    tmp = xp.tensordot(tmp, w, axes=([2, 3], [0, 3]))             # (bl, wt, kr, wb', m)
    tmp = xp.tensordot(tmp, w, axes=([1, 4], [0, 3]))             # (bl, kr, wb', wt', s)
    tmp = xp.tensordot(a_bra, tmp, axes=([0, 1], [0, 4]))         # (br, kr, wb', wt')
    return tmp.transpose(0, 3, 2, 1)


def energy_and_variance(mps: list, mpo: list) -> tuple[float, float]:
    """<H> and <H^2> - <H>^2 for a (not necessarily normalized) MPS.

    The variance vanishes for an exact eigenstate and is the standard, extrapolatable measure of
    the DMRG error. Cost per site ~ chi^3 w^2 d^2 (two MPO layers).
    """
    env1 = boundary_env()
    env2 = xp.ones((1, 1, 1, 1), dtype=xp.dtype)
    norm = xp.ones((1, 1), dtype=xp.dtype)
    for a, w in zip(mps, mpo):
        env1 = update_left(env1, a, w, a)
        env2 = update_left_double(env2, a, w, a)
        norm = xp.tensordot(a, xp.tensordot(norm, a, axes=([1], [0])), axes=([0, 1], [0, 1]))
    nrm = float(xp.scalar(norm.reshape(-1)[0]))
    energy = float(xp.scalar(env1.reshape(-1)[0])) / nrm
    h2 = float(xp.scalar(env2.reshape(-1)[0])) / nrm
    return energy, h2 - energy ** 2


def z_string_expectation(mps: list, sites) -> float:
    """<prod_{j in sites} Z_j> using a bond-dimension-1 MPO."""
    sites = set(int(s) for s in sites)
    ident = np.eye(2).reshape(1, 1, 2, 2)
    zop = np.diag([1., -1.]).reshape(1, 1, 2, 2)
    mpo = [xp.asarray_dev(zop if j in sites else ident) for j in range(len(mps))]
    return expectation(mps, mpo)


def sample_bitstrings(mps: list, num_samples: int, seed: int = 0) -> np.ndarray:
    """Perfect sampling from a right-canonical MPS (site 0 holds the norm). Host-side loop over
    sites, vectorized over samples on the device."""
    rng = np.random.default_rng(seed)
    nsites = len(mps)
    samples = np.zeros((num_samples, nsites), dtype=np.uint8)
    left = xp.ones((num_samples, 1), dtype=xp.dtype)
    for site, a in enumerate(mps):
        amps = xp.tensordot(left, a, axes=([1], [0]))            # (n, 2, r)
        probs = xp.asnumpy(xp.sum(amps ** 2, axis=2))
        probs /= probs.sum(axis=1, keepdims=True)
        bits = (rng.random(num_samples) < probs[:, 1]).astype(np.uint8)
        samples[:, site] = bits
        left = amps[xp.arange(num_samples), xp.asarray(bits.astype(np.int64))]
        left /= xp.linalg.norm(left, axis=1, keepdims=True)
    return samples


# --------------------------------------------------------------------------------------------------
# Local solver
# --------------------------------------------------------------------------------------------------

def lanczos_ground(matvec: Callable, v0, krylov_dim: int = 20, tol: float = 1.e-10,
                   max_restarts: int = 4):
    """Lowest eigenpair of a real-symmetric operator via restarted Lanczos.

    GPU-friendly: the Krylov basis is a device matrix V, reorthogonalization is block classical
    Gram-Schmidt applied twice (two GEMVs per step), and the tridiagonal coefficients stay on the
    device. The host is synchronized once per restart (to read the coefficients), not per step. A
    breakdown (invariant subspace found) is detected afterwards and the Krylov space truncated
    there.
    """
    size = v0.size
    kdim = max(1, min(krylov_dim, size))
    vec = v0.reshape(-1) / xp.linalg.norm(v0)
    energy = None
    for _ in range(max_restarts):
        basis = xp.zeros((kdim, size), dtype=vec.dtype)
        alphas = xp.zeros(kdim, dtype=vec.dtype)
        betas = xp.zeros(kdim, dtype=vec.dtype)
        basis[0] = vec
        for k in range(kdim):
            w = matvec(basis[k])
            alphas[k] = xp.dot(basis[k], w)
            active = basis[:k + 1]
            w = w - active.T @ (active @ w)
            w = w - active.T @ (active @ w)
            beta = xp.linalg.norm(w)
            betas[k] = beta
            if k + 1 < kdim:
                basis[k + 1] = w / xp.maximum(beta, 1.e-300)

        alpha_h = xp.asnumpy(alphas)
        beta_h = xp.asnumpy(betas)
        scale = max(1., float(np.max(np.abs(alpha_h))))
        breakdown = np.nonzero(beta_h[:kdim - 1] < tol * scale)[0]
        nkeep = int(breakdown[0]) + 1 if breakdown.size else kdim
        tri = np.diag(alpha_h[:nkeep]) + np.diag(beta_h[:nkeep - 1], 1) + \
            np.diag(beta_h[:nkeep - 1], -1)
        evals, evecs = np.linalg.eigh(tri)
        coeffs = xp.asarray(evecs[:, 0], dtype=vec.dtype)
        new_vec = coeffs @ basis[:nkeep]
        new_vec /= xp.linalg.norm(new_vec)
        converged = energy is not None and abs(evals[0] - energy) < tol * max(1., abs(evals[0]))
        energy = float(evals[0])
        vec = new_vec
        if breakdown.size or converged:
            break
    return energy, vec


def truncated_split(theta, maxdim: int, cutoff: float, method: str = 'svd',
                    direction: str = 'right', correction=None, mindim: int = 1):
    """Split theta[(l s1), (s2 r)] = left @ right, keeping at most maxdim states and discarding a
    total weight (sum of squared singular values / total) of at most cutoff, but at least mindim
    states when available.

    For direction='right' the left factor is an isometry (left-canonical site) and the right factor
    carries the normalized singular values; for 'left' the roles are reversed.

    method='svd' uses gesvd. method='eigh' diagonalizes the reduced density matrix, which is
    substantially faster on GPUs for large bond dimensions (slightly less accurate on tiny singular
    values). A ``correction`` (White's density-matrix perturbation) is added to the reduced density
    matrix of the side being kept as an isometry and forces the eigh path.
    """
    if method == 'svd' and correction is None:
        u, s, vh = xp.linalg.svd(theta, full_matrices=False)
        s2 = xp.asnumpy(s) ** 2
        keep, trunc_err = _num_kept(s2, maxdim, cutoff, mindim)
        s = s[:keep] / xp.linalg.norm(s[:keep])
        if direction == 'right':
            return u[:, :keep], s[:, None] * vh[:keep], trunc_err
        return u[:, :keep] * s[None, :], vh[:keep], trunc_err

    rho = theta @ theta.T if direction == 'right' else theta.T @ theta
    if correction is not None:
        rho = rho + correction
    evals, evecs = xp.linalg.eigh(rho)
    evecs = evecs[:, ::-1]
    s2 = np.clip(xp.asnumpy(evals)[::-1], 0., None)
    keep, trunc_err = _num_kept(s2, maxdim, cutoff, mindim)
    basis = evecs[:, :keep]
    if direction == 'right':
        right = basis.T @ theta
        return basis, right / xp.linalg.norm(right), trunc_err
    left = theta @ basis
    return left / xp.linalg.norm(left), basis.T, trunc_err


def _num_kept(s2: np.ndarray, maxdim: int, cutoff: float, mindim: int = 1) -> tuple[int, float]:
    """Smallest number of states with discarded weight <= cutoff, within [mindim, maxdim]."""
    discarded_tail = np.concatenate([np.cumsum(s2[::-1])[::-1], [0.]]) / s2.sum()
    keep = int(np.argmax(discarded_tail <= cutoff))
    keep = max(1, min(max(keep, mindim), maxdim, len(s2)))
    return keep, float(discarded_tail[keep])


def density_matrix_correction(theta, lenv, w1, w2, renv, direction: str, noise: float):
    """White's density-matrix perturbation noise * sum_b (H_b theta)(H_b theta)^T / norm for the
    side kept as an isometry, with H_b the MPO half-system operators acting on that side."""
    if direction == 'right':
        tmp = xp.tensordot(lenv, theta, axes=([2], [0]))              # (al, a, t1, t2, kr)
        tmp = xp.tensordot(tmp, w1, axes=([1, 2], [0, 3]))             # (al, t2, kr, b, s1)
        tmp = tmp.transpose(0, 4, 3, 1, 2)                             # (al, s1, b, t2, kr)
        tmp = tmp.reshape(theta.shape[0] * 2, -1)
    else:
        tmp = xp.tensordot(theta, renv, axes=([3], [2]))              # (kl, t1, t2, br, c)
        tmp = xp.tensordot(tmp, w2, axes=([2, 4], [3, 1]))             # (kl, t1, br, b, s2)
        tmp = tmp.transpose(4, 2, 0, 1, 3)                             # (s2, br, kl, t1, b)
        tmp = tmp.reshape(2 * theta.shape[3], -1)
    corr = tmp @ tmp.T
    trace = xp.trace(corr)
    return corr * (noise / xp.maximum(trace, 1.e-300))


# --------------------------------------------------------------------------------------------------
# DMRG driver
# --------------------------------------------------------------------------------------------------

@dataclass
class SweepRecord:
    sweep: int
    maxdim: int
    cutoff: float
    noise: float
    energy: float
    max_bond: int
    max_trunc_err: float
    seconds: float
    variance: Optional[float] = None


@dataclass
class DMRGResult:
    energy: float
    mps: list
    history: list[SweepRecord] = field(default_factory=list)


def two_site_dmrg(
    mpo: list,
    mps: list,
    sweeps: Sweeps,
    lanczos_tol: float = 1.e-10,
    svd_method: str = 'svd',
    energy_tol: Optional[float] = None,
    min_sweeps: int = 2,
    callback: Optional[Callable[[SweepRecord], None]] = None,
    orthogonal_to: Optional[list[list]] = None,
    weight: float = 10.,
    measure_variance: bool = False,
    start_sweep: int = 0,
    history: Optional[list] = None,
    on_sweep: Optional[Callable[[int, list, list], None]] = None
) -> DMRGResult:
    """Run two-site DMRG sweeps (one sweep = left->right + right->left).

    Args:
        mpo: Device MPO tensors.
        mps: Initial MPS (any gauge).
        sweeps: Per-sweep maxdim, mindim, cutoff, noise and Lanczos settings (ITensors-style).
            Noise is needed to escape product-like local minima when hopping terms span more
            than two sites.
        energy_tol: Stop early once the energy change between sweeps falls below this, like
            ITensors' DMRGObserver(energy_tol=...). Only checked after min_sweeps sweeps, on
            sweeps without noise, and when maxdim does not grow in the next sweep.
        orthogonal_to: Normalized MPSs (e.g. lower eigenstates) to project out by adding
            weight * sum_k |psi_k><psi_k| to the Hamiltonian, as in ITensors'
            dmrg(H, [psi0], psi; weight). The weight must exceed the gap to the target state.
        measure_variance: Compute <H^2> - <H>^2 of the MPO after every sweep (input to
            analysis.extrapolate). The projector of orthogonal_to is not included.
        start_sweep, history: Resume a schedule: sweeps before start_sweep are skipped and the
            history of the completed sweeps is prepended (see checkpoint).
        on_sweep: Called as on_sweep(isweep, mps, history) after every sweep (checkpointing).
    """
    nsites = len(mps)
    mps = right_canonicalize(mps)
    renvs = [None] * (nsites + 1)
    lenvs = [None] * (nsites + 1)
    renvs[nsites] = boundary_env()
    lenvs[0] = boundary_env()
    for site in range(nsites - 1, 0, -1):
        renvs[site] = update_right(renvs[site + 1], mps[site], mpo[site], mps[site])

    # Overlap environments <mps|psi_k> for the projected-out states
    others = list(orthogonal_to or [])
    olenvs = [[None] * (nsites + 1) for _ in others]
    orenvs = [[None] * (nsites + 1) for _ in others]
    for psi, olenv, orenv in zip(others, olenvs, orenvs):
        olenv[0] = xp.ones((1, 1), dtype=xp.dtype)
        orenv[nsites] = xp.ones((1, 1), dtype=xp.dtype)
        for site in range(nsites - 1, 0, -1):
            orenv[site] = overlap_right(orenv[site + 1], mps[site], psi[site])

    def make_matvec(site):
        lenv, w1, w2, renv = lenvs[site], mpo[site], mpo[site + 1], renvs[site + 2]
        projections = [projected_state(olenv[site], psi[site], psi[site + 1], orenv[site + 2])
                       for psi, olenv, orenv in zip(others, olenvs, orenvs)]

        def matvec(vec):
            theta = vec.reshape(lenv.shape[2], 2, 2, renv.shape[2])
            tmp = xp.tensordot(lenv, theta, axes=([2], [0]))           # (al, a, t1, t2, kr)
            tmp = xp.tensordot(tmp, w1, axes=([1, 2], [0, 3]))           # (al, t2, kr, b, s1)
            tmp = xp.tensordot(tmp, w2, axes=([3, 1], [0, 3]))           # (al, kr, s1, c, s2)
            tmp = xp.tensordot(tmp, renv, axes=([1, 3], [2, 1]))         # (al, s1, s2, br)
            out = tmp.reshape(-1)
            for proj in projections:
                out = out + weight * xp.vdot(proj, vec) * proj
            return out
        return matvec

    result = DMRGResult(energy=np.inf, mps=mps)
    result.history = list(history or [])
    last_energy = result.history[-1].energy if result.history else None
    for isweep, params in enumerate(sweeps):
        if isweep < start_sweep:
            continue
        t0 = time.time()
        max_err = 0.
        energy = None
        bonds = list(range(nsites - 1)) + list(range(nsites - 2, -1, -1))
        directions = ['right'] * (nsites - 1) + ['left'] * (nsites - 1)
        for site, direction in zip(bonds, directions):
            theta = xp.tensordot(mps[site], mps[site + 1], axes=([2], [0]))
            dl, dr = theta.shape[0], theta.shape[3]
            matvec = make_matvec(site)
            energy, vec = lanczos_ground(matvec, theta.reshape(-1), krylov_dim=params.krylovdim,
                                         tol=lanczos_tol, max_restarts=params.maxiter)
            theta = vec.reshape(dl, 2, 2, dr)
            correction = None
            if params.noise > 0.:
                correction = density_matrix_correction(theta, lenvs[site], mpo[site],
                                                       mpo[site + 1], renvs[site + 2], direction,
                                                       params.noise)
            left, right, err = truncated_split(theta.reshape(dl * 2, 2 * dr), params.maxdim,
                                               params.cutoff, method=svd_method,
                                               direction=direction, correction=correction,
                                               mindim=params.mindim)
            max_err = max(max_err, err)
            keep = left.shape[1]
            mps[site] = left.reshape(dl, 2, keep)
            mps[site + 1] = right.reshape(keep, 2, dr)
            if direction == 'right':
                lenvs[site + 1] = update_left(lenvs[site], mps[site], mpo[site], mps[site])
                for psi, olenv in zip(others, olenvs):
                    olenv[site + 1] = overlap_left(olenv[site], mps[site], psi[site])
            else:
                renvs[site + 1] = update_right(renvs[site + 2], mps[site + 1], mpo[site + 1],
                                               mps[site + 1])
                for psi, orenv in zip(others, orenvs):
                    orenv[site + 1] = overlap_right(orenv[site + 2], mps[site + 1], psi[site + 1])

        variance = energy_and_variance(mps, mpo)[1] if measure_variance else None
        xp.synchronize()
        record = SweepRecord(sweep=isweep, maxdim=params.maxdim, cutoff=params.cutoff,
                             noise=params.noise, energy=energy,
                             max_bond=max(a.shape[2] for a in mps), max_trunc_err=max_err,
                             seconds=time.time() - t0, variance=variance)
        result.history.append(record)
        result.energy = energy
        LOG.info('sweep %d maxdim %d cutoff %.0e noise %.0e E=%.12f bond=%d trunc=%.2e var=%s '
                 '(%.1fs)', isweep, params.maxdim, params.cutoff, params.noise, energy,
                 record.max_bond, max_err,
                 'n/a' if variance is None else f'{variance:.2e}', record.seconds)
        if callback is not None:
            callback(record)
        if on_sweep is not None:
            on_sweep(isweep, mps, result.history)

        if (energy_tol is not None and last_energy is not None and isweep + 1 >= min_sweeps
                and abs(energy - last_energy) < energy_tol
                and params.noise == 0.
                and (isweep + 1 == len(sweeps) or sweeps[isweep + 1].maxdim <= params.maxdim)):
            break
        last_energy = energy

    result.mps = mps  # right-canonical with norm on site 0 after the final left sweep
    return result
