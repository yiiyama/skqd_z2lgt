"""Block-sparse two-site DMRG with exact Gauss-law symmetry for the primal link model.

Symmetry: every Gauss-law operator G_v = prod_{l in v} Z_l commutes with H, giving the abelian group
Z2^V. A computational-basis link state s_l carries parity s_l at the two end vertices of the link,
so the charge of site j in state s is s * mask_j, with mask_j the bitmask of its end vertices.

Bond labels: the label of an MPS bond index is the vertex-parity bitmask of the links to its left.
Only labels that are reachable from the left boundary (label 0) and can be completed to the target
charge configuration at the right boundary are allowed. In particular, a vertex all of whose links
lie to the left must already carry its target parity, and a vertex with all links to the right has
parity 0. Only the "open" vertices with links on both sides of the cut are free, so a bond has at
most 2^(number of open vertices) labels (6 open vertices at most on the 156q lattice with the
snake ordering).

Storage: tensors are padded block arrays
    A[i, s, x, y]      left label index i, physical s, left/right block indices x/y
with the right label implied by charge conservation, label(right) = label(i) ^ s * mask. Blocks
are zero beyond the actual dimension of each label. Environments
    L[a, i, X, x]      MPO state a, ket label i; bra label = label(i) ^ delta_a
carry the operator charge delta_a of each MPO automaton state (the parity flipped by the X
operators already placed). The state is therefore exactly in the target sector at every step: no
penalty term and no leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Callable, Optional
import numpy as np
from .backend import xp
from .sweeps import Sweeps
from .dmrg import lanczos_ground, SweepRecord, DMRGResult, energy_and_variance

LOG = logging.getLogger(__name__)


def label_index(labels: np.ndarray, query) -> np.ndarray:
    """Index of each query label in the sorted label array, -1 if absent."""
    query = np.asarray(query, dtype=np.int64)
    if len(labels) == 0:
        return np.full(query.shape, -1)
    pos = np.searchsorted(labels, query)
    pos_c = np.minimum(pos, len(labels) - 1)
    return np.where(labels[pos_c] == query, pos_c, -1)


@dataclass
class Bond:
    labels: np.ndarray  # sorted int64 bitmasks
    dims: np.ndarray    # block dimension of each label

    @property
    def size(self) -> int:
        return len(self.labels)

    @property
    def pad(self) -> int:
        return max(1, int(self.dims.max(initial=1)))


class GaussLawChain:
    """Static symmetry data of the chain: site masks, target and allowed bond labels."""

    def __init__(self, masks: np.ndarray, target: int):
        self.masks = np.asarray(masks, dtype=np.int64)
        self.target = int(target)
        nsites = len(self.masks)
        if np.bitwise_or.reduce(self.masks) >= 2 ** 62:
            raise ValueError('At most 62 vertices are supported')

        # Vertices closed (all links placed) after bond j, and not yet started before bond j
        first = {}
        last = {}
        for site, mask in enumerate(self.masks):
            for bit in _bits(int(mask)):
                first.setdefault(bit, site)
                last[bit] = site
        self.closed = np.zeros(nsites + 1, dtype=np.int64)
        self.unstarted = np.zeros(nsites + 1, dtype=np.int64)
        for bit in first:
            for bond in range(nsites + 1):
                if last[bit] < bond:
                    self.closed[bond] |= 1 << bit
                if first[bit] >= bond:
                    self.unstarted[bond] |= 1 << bit

        forward = [np.array([0], dtype=np.int64)]
        for site in range(nsites):
            nxt = np.unique(np.concatenate([forward[-1], forward[-1] ^ self.masks[site]]))
            closed = self.closed[site + 1]
            forward.append(nxt[(nxt & closed) == (self.target & closed)])
        backward = [np.array([self.target], dtype=np.int64)]
        for site in range(nsites - 1, -1, -1):
            prv = np.unique(np.concatenate([backward[0], backward[0] ^ self.masks[site]]))
            backward.insert(0, prv[(prv & self.unstarted[site]) == 0])
        self.allowed = [np.intersect1d(f, b) for f, b in zip(forward, backward)]
        if any(len(a) == 0 for a in self.allowed):
            raise ValueError('Target charge configuration is not reachable (odd total charge?)')

    @property
    def num_sites(self) -> int:
        return len(self.masks)

    def site_map(self, site: int, left: Bond, right: Bond) -> np.ndarray:
        """(n_left, 2) index of the right label for each (left label, s), -1 if absent."""
        query = left.labels[:, None] ^ (np.array([0, 1], dtype=np.int64) * self.masks[site])
        return label_index(right.labels, query)


def _bits(value: int):
    bit = 0
    while value:
        if value & 1:
            yield bit
        value >>= 1
        bit += 1


def operator_charges(bond_keys: list[list], masks: np.ndarray) -> list[np.ndarray]:
    """Operator charge (parity flipped by the X operators placed so far) of each MPO state."""
    deltas = []
    for keys in bond_keys:
        delta = np.zeros(len(keys), dtype=np.int64)
        for ikey, key in enumerate(keys):
            if key in (('start',), ('final',)):
                continue
            # Open-term keys are ('start', (site, op), (site, op), ...)
            for item in key:
                if not isinstance(item, tuple):
                    continue
                site, op = item
                if op == 'X':
                    delta[ikey] ^= masks[site]
        deltas.append(delta)
    return deltas


class SymmetricMPS:
    """Padded block-sparse MPS in a fixed Gauss-law sector."""

    def __init__(self, chain: GaussLawChain, bonds: list[Bond], tensors: list):
        self.chain = chain
        self.bonds = bonds
        self.tensors = tensors

    @classmethod
    def random(cls, chain: GaussLawChain, block_dim: int = 1, seed: int = 0) -> 'SymmetricMPS':
        """Random state with every allowed label present, right-canonicalized."""
        rng = np.random.default_rng(seed)
        bonds = []
        for bond, labels in enumerate(chain.allowed):
            dim = 1 if bond in (0, chain.num_sites) else block_dim
            bonds.append(Bond(labels.copy(), np.full(len(labels), dim, dtype=np.int64)))
        tensors = []
        for site in range(chain.num_sites):
            left, right = bonds[site], bonds[site + 1]
            smap = chain.site_map(site, left, right)
            tensor = np.zeros((left.size, 2, left.pad, right.pad))
            for i, s in zip(*np.nonzero(smap >= 0)):
                r = smap[i, s]
                tensor[i, s, :left.dims[i], :right.dims[r]] = \
                    rng.normal(size=(left.dims[i], right.dims[r]))
            tensors.append(xp.asarray_dev(tensor))
        mps = cls(chain, bonds, tensors)
        mps.right_canonicalize()
        return mps

    @classmethod
    def from_bits(cls, chain: GaussLawChain, bits: np.ndarray, noise: float = 0.,
                  seed: int = 0) -> 'SymmetricMPS':
        """Basis state (in chain order), optionally with all other allowed labels added at weight
        ~noise so that the two-site update can populate them."""
        mps = cls.random(chain, block_dim=1, seed=seed)
        label = 0
        for site, bit in enumerate(bits):
            i = int(label_index(mps.bonds[site].labels, label))
            label ^= int(bit) * int(chain.masks[site])
            tensor = mps.tensors[site] * noise
            tensor[i, int(bit), 0, 0] = 1.
            mps.tensors[site] = tensor
        mps.right_canonicalize()
        return mps

    def site_map(self, site: int) -> np.ndarray:
        return self.chain.site_map(site, self.bonds[site], self.bonds[site + 1])

    def right_canonicalize(self):
        """Right-canonicalize by an LQ decomposition per left label, then normalize."""
        chain = self.chain
        for site in range(chain.num_sites - 1, 0, -1):
            left, right = self.bonds[site], self.bonds[site + 1]
            tensor = self.tensors[site]
            factors = []
            new_dims = np.zeros(left.size, dtype=np.int64)
            for i in range(left.size):
                mat = tensor[i, :, :left.dims[i], :].transpose(1, 0, 2).reshape(left.dims[i], -1)
                q, r = xp.linalg.qr(mat.T)
                new_dims[i] = q.shape[1]
                factors.append((q.T, r.T))  # mat = r.T @ q.T
            new_left = Bond(left.labels, new_dims)
            new_tensor = xp.zeros((left.size, 2, new_left.pad, right.pad), dtype=xp.dtype)
            for i, (qt, _) in enumerate(factors):
                new_tensor[i, :, :new_dims[i], :] = qt.reshape(new_dims[i], 2, right.pad) \
                    .transpose(1, 0, 2)
            # Push r.T into the previous tensor
            prev_left = self.bonds[site - 1]
            prev = self.tensors[site - 1]
            pmap = chain.site_map(site - 1, prev_left, left)
            new_prev = xp.zeros((prev_left.size, 2, prev_left.pad, new_left.pad), dtype=xp.dtype)
            for ip, s in zip(*np.nonzero(pmap >= 0)):
                i = pmap[ip, s]
                new_prev[ip, s, :, :new_dims[i]] = prev[ip, s, :, :left.dims[i]] @ factors[i][1]
            self.bonds[site] = new_left
            self.tensors[site] = new_tensor
            self.tensors[site - 1] = new_prev
        self.tensors[0] = self.tensors[0] / xp.linalg.norm(self.tensors[0])

    def to_dense(self) -> list:
        """Equivalent dense MPS (bond dimension = sum of block dimensions)."""
        offsets = [np.concatenate([[0], np.cumsum(b.dims)]) for b in self.bonds]
        dense = []
        for site, tensor in enumerate(self.tensors):
            left, right = self.bonds[site], self.bonds[site + 1]
            smap = self.site_map(site)
            out = xp.zeros((int(offsets[site][-1]), 2, int(offsets[site + 1][-1])), dtype=xp.dtype)
            for i, s in zip(*np.nonzero(smap >= 0)):
                r = smap[i, s]
                lo, ro = offsets[site][i], offsets[site + 1][r]
                out[lo:lo + left.dims[i], s, ro:ro + right.dims[r]] = \
                    tensor[i, s, :left.dims[i], :right.dims[r]]
            dense.append(out)
        return dense


# --------------------------------------------------------------------------------------------------
# Environments
# --------------------------------------------------------------------------------------------------

def _perm(bond: Bond, delta: np.ndarray) -> np.ndarray:
    """(w, n) index of the bra label label(i) ^ delta_a, -1 if absent."""
    return label_index(bond.labels, bond.labels[None, :] ^ delta[:, None])


def _nonzero_channels(w_host: np.ndarray):
    """(a, b, s, t, coeff) of the nonzero MPO elements."""
    return [(int(a), int(b), int(s), int(t), float(w_host[a, b, s, t]))
            for a, b, s, t in zip(*np.nonzero(w_host))]


def update_left_sym(env, perm_l, tensor, smap, w_elems, wr, right: Bond):
    new = xp.zeros((wr, right.size, right.pad, right.pad), dtype=xp.dtype)
    for a, b, s, t, coeff in w_elems:
        ket = np.nonzero((smap[:, t] >= 0) & (perm_l[a] >= 0))[0]
        if ket.size == 0:
            continue
        bra = perm_l[a][ket]
        m_bra = smap[bra, s]
        valid = m_bra >= 0
        ket, bra = ket[valid], bra[valid]
        if ket.size == 0:
            continue
        m_ket = smap[ket, t]
        tmp = env[a][ket] @ tensor[ket, t]                         # (k, X, y)
        res = tensor[bra, s].transpose(0, 2, 1) @ tmp               # (k, Y, y)
        new[b, m_ket] += coeff * res
    return new


def update_right_sym(env, perm_l, tensor, smap, w_elems, wl, left: Bond):
    new = xp.zeros((wl, left.size, left.pad, left.pad), dtype=xp.dtype)
    for a, b, s, t, coeff in w_elems:
        ket = np.nonzero((smap[:, t] >= 0) & (perm_l[a] >= 0))[0]
        if ket.size == 0:
            continue
        bra = perm_l[a][ket]
        valid = smap[bra, s] >= 0
        ket, bra = ket[valid], bra[valid]
        if ket.size == 0:
            continue
        m_ket = smap[ket, t]
        tmp = env[b][m_ket] @ tensor[ket, t].transpose(0, 2, 1)     # (k, Y, x)
        res = tensor[bra, s] @ tmp                                  # (k, X, x)
        new[a, ket] += coeff * res
    return new


# --------------------------------------------------------------------------------------------------
# DMRG
# --------------------------------------------------------------------------------------------------

def _group(elems, key: int) -> dict:
    groups: dict = {}
    for elem in elems:
        groups.setdefault(elem[key], []).append(elem)
    return groups


def _block_rows(array, rows, dims, ncols):
    """Stack array[i, s, :, :d_i] as rows (x of label i) x columns (t, z) for (i, s) in rows."""
    return xp.concatenate([array[i, s, :, :dims[i]].transpose(1, 0, 2).reshape(dims[i], ncols)
                           for i, s in rows])


def _density_matrix_corrections(theta, direction, lenv, renv, perm_l, rmap, mid, mid_idx,
                                mid_labels, left: Bond, w1_by_b, w2_by_b, delta_mid, noise):
    """White's correction per middle label (index into mid_labels), normalized to total trace
    `noise`. For a right move the kept side is the left one (rows); for a left move the right one
    (columns)."""
    ncols = theta.shape[2] * theta.shape[4]
    corr: dict = {}
    if direction == 'right':
        tl_cache = {}
        for b, elems in w1_by_b.items():
            proj = xp.zeros_like(theta)             # (bra i', bra s', t, X, z)
            for a, _, sp, sk, coeff in elems:
                ket = np.nonzero(perm_l[a] >= 0)[0]
                if ket.size == 0:
                    continue
                if a not in tl_cache:
                    tl_cache[a] = xp.matmul(lenv[a][:, None, None], theta)
                proj[perm_l[a][ket], sp] += coeff * tl_cache[a][ket, sk]
            for m in np.unique(mid_idx[mid_idx >= 0]):
                rows = [(int(i), int(s)) for i, s in zip(*np.nonzero(mid_idx == m))]
                mat = _block_rows(proj, rows, left.dims, ncols)
                corr[int(m)] = corr.get(int(m), 0.) + mat @ mat.T
    else:
        zero_r = xp.zeros((1,) + renv.shape[2:], dtype=xp.dtype)
        rsel = np.where(rmap >= 0, rmap, renv.shape[1])
        for b, elems in w2_by_b.items():
            proj = xp.zeros_like(theta)             # (ket i, ket s, bra t', X, Z')
            for _, c, tp, tk, coeff in elems:
                rg = xp.concatenate([renv[c], zero_r])[rsel[:, :, tk]]           # (n, 2, Z', z)
                proj[:, :, tp] += coeff * xp.matmul(theta[:, :, tk], rg.transpose(0, 1, 3, 2))
            target_idx = label_index(mid_labels, mid ^ delta_mid[b])
            target_idx = np.where(mid_idx >= 0, target_idx, -1)
            for m in np.unique(target_idx[target_idx >= 0]):
                rows = [(int(i), int(s)) for i, s in zip(*np.nonzero(target_idx == m))]
                mat = _block_rows(proj, rows, left.dims, ncols)
                corr[int(m)] = corr.get(int(m), 0.) + mat.T @ mat
    total = sum(float(xp.scalar(xp.trace(c))) for c in corr.values())
    if total <= 0.:
        return {}
    return {m: c * (noise / total) for m, c in corr.items()}


def _split_block(mat, direction: str, method: str, correction=None):
    """Return (s^2 on host, isometry, carry) with mat = iso @ carry ('right') or carry @ iso
    ('left'), ordered by decreasing singular value. With a density-matrix correction the
    decomposition is done on the corrected reduced density matrix."""
    if correction is not None or method == 'eigh':
        if direction == 'right':
            rho = mat @ mat.T + (correction if correction is not None else 0.)
            evals, vecs = xp.linalg.eigh(rho)
            vecs = vecs[:, ::-1]
            return np.clip(xp.asnumpy(evals)[::-1], 0., None), vecs, vecs.T @ mat
        rho = mat.T @ mat + (correction if correction is not None else 0.)
        evals, vecs = xp.linalg.eigh(rho)
        vecs = vecs[:, ::-1]
        return np.clip(xp.asnumpy(evals)[::-1], 0., None), vecs.T, mat @ vecs
    u, s, vh = xp.linalg.svd(mat, full_matrices=False)
    s2 = xp.asnumpy(s) ** 2
    if direction == 'right':
        return s2, u, s[:, None] * vh
    return s2, vh, u * s[None, :]


def symmetric_two_site_dmrg(
    chain: GaussLawChain,
    mpo_host: list[np.ndarray],
    deltas: list[np.ndarray],
    mps: SymmetricMPS,
    sweeps: Sweeps,
    lanczos_tol: float = 1.e-10,
    svd_method: str = 'svd',
    energy_tol: Optional[float] = None,
    min_sweeps: int = 2,
    callback: Optional[Callable[[SweepRecord], None]] = None,
    measure_variance: bool = False,
    dense_mpo: Optional[list] = None,
    seed: int = 0,
    start_sweep: int = 0,
    history: Optional[list] = None,
    on_sweep: Optional[Callable] = None
) -> tuple[DMRGResult, SymmetricMPS]:
    """Two-site DMRG within the Gauss-law sector of the chain.

    mpo_host must be the uncompressed automaton MPO (definite operator charge per state) with
    operator charges `deltas`.

    Noise: links whose state is fixed by the bond labels on both sides (a vertex closes there) tie
    the block dimensions of neighbouring bonds together, and the three-site plaquette terms cannot
    be resolved by a two-site window. Without help the bond dimension then stays frozen.
    For sweeps with noise > 0, White's density-matrix correction is added per block: for every
    MPO bond index b, the half-system operator is applied to theta and its contribution is added to
    the reduced density matrix of the middle label it maps to (shifted by the operator charge of
    b). This supplies the missing directions within the sector. Finish with noiseless sweeps.
    """
    del seed  # kept for interface stability
    nsites = chain.num_sites
    mpo = [xp.asarray_dev(w) for w in mpo_host]
    w_elems = [_nonzero_channels(w) for w in mpo_host]
    # Elements grouped by the MPO bond index between the two sites of a window:
    # site j as left site -> group by right index b; site j+1 as right site -> by left index b
    w_by_b = [_group(elems, key=1) for elems in w_elems]
    w2_by_b = [_group(elems, key=0) for elems in w_elems]
    # Two-site channels: M[a, c, u, v, s, t] = sum_b W1[a, b, u, s] W2[b, c, v, t]
    pair_channels = []
    for site in range(nsites - 1):
        comb = np.einsum('abus,bcvt->acuvst', mpo_host[site], mpo_host[site + 1])
        nz = np.nonzero(np.abs(comb).reshape(comb.shape[0], comb.shape[1], -1).max(axis=2))
        pair_channels.append([(int(a), int(c), xp.asarray_dev(comb[a, c]))
                              for a, c in zip(*nz)])

    def perm(bond_idx):
        return _perm(mps.bonds[bond_idx], deltas[bond_idx])

    lenvs: list = [None] * (nsites + 1)
    renvs: list = [None] * (nsites + 1)
    lenvs[0] = xp.ones((1, 1, 1, 1), dtype=xp.dtype)
    renvs[nsites] = xp.ones((1, 1, 1, 1), dtype=xp.dtype)
    for site in range(nsites - 1, 0, -1):
        renvs[site] = update_right_sym(renvs[site + 1], perm(site), mps.tensors[site],
                                       mps.site_map(site), w_elems[site],
                                       mpo_host[site].shape[0], mps.bonds[site])

    result = DMRGResult(energy=np.inf, mps=[])
    result.history = list(history or [])
    last_energy = result.history[-1].energy if result.history else None
    for isweep, params in enumerate(sweeps):
        if isweep < start_sweep:
            continue
        t0 = time.time()
        max_err = 0.
        energy = None
        bonds_seq = list(range(nsites - 1)) + list(range(nsites - 2, -1, -1))
        directions = ['right'] * (nsites - 1) + ['left'] * (nsites - 1)
        for site, direction in zip(bonds_seq, directions):
            left, right = mps.bonds[site], mps.bonds[site + 2]
            mid_labels = chain.allowed[site + 1]
            m1, m2 = chain.masks[site], chain.masks[site + 1]
            svals = np.array([0, 1], dtype=np.int64)
            mid = left.labels[:, None] ^ (svals * m1)[None, :]                  # (n, 2)
            mid_idx = label_index(mid_labels, mid)
            rmap = label_index(right.labels, mid[:, :, None] ^ (svals * m2)[None, None, :])
            rmap = np.where(mid_idx[:, :, None] >= 0, rmap, -1)                  # (n, 2, 2)

            # Current two-site tensor
            amap = mps.site_map(site)
            a2 = mps.tensors[site + 1]
            a2p = xp.concatenate([a2, xp.zeros((1,) + a2.shape[1:], dtype=xp.dtype)])
            a2g = a2p[np.where(amap >= 0, amap, a2.shape[0])]                   # (n, 2, 2, Dm, Dr)
            theta = xp.matmul(mps.tensors[site][:, :, None], a2g)              # (n, 2, 2, Dl, Dr)
            # The current middle bond may have fewer labels/dims than allowed; that is fine.

            lenv, renv = lenvs[site], renvs[site + 2]
            perm_l = perm(site)
            zero_r = xp.zeros((1,) + renv.shape[2:], dtype=xp.dtype)
            rsel = np.where(rmap >= 0, rmap, renv.shape[1])
            channels = {}
            for a, c, comb in pair_channels[site]:
                valid = np.nonzero(perm_l[a] >= 0)[0]
                if valid.size == 0:
                    continue
                rg = xp.concatenate([renv[c], zero_r])[rsel]                    # (n, 2, 2, Z, z)
                channels.setdefault(a, []).append((comb, rg, valid, perm_l[a][valid]))

            def matvec(vec, channels=channels, lenv=lenv, shape=theta.shape):
                th = vec.reshape(shape)
                out = xp.zeros_like(th)
                for a, chans in channels.items():
                    tl = xp.matmul(lenv[a][:, None, None], th)                 # (n, 2, 2, X, z)
                    for comb, rg, valid, bra in chans:
                        t3 = xp.matmul(tl, rg.transpose(0, 1, 2, 4, 3))       # (n, 2, 2, X, Z)
                        oc = xp.tensordot(comb, t3, axes=([2, 3], [1, 2]))     # (u, v, n, X, Z)
                        out[bra] += oc.transpose(2, 0, 1, 3, 4)[valid]
                return out.reshape(-1)

            energy, vec = lanczos_ground(matvec, theta.reshape(-1), krylov_dim=params.krylovdim,
                                         tol=lanczos_tol, max_restarts=params.maxiter)
            theta = vec.reshape(theta.shape)
            corrections = {}
            if params.noise > 0.:
                corrections = _density_matrix_corrections(
                    theta, direction, lenv, renv, perm_l, rmap, mid, mid_idx, mid_labels, left,
                    w_by_b[site], w2_by_b[site + 1], deltas[site + 1], params.noise)

            # Block decomposition over middle labels
            dr_pad = right.pad
            blocks = []
            for m in np.unique(mid_idx[mid_idx >= 0]):
                rows = [(int(i), int(s)) for i, s in zip(*np.nonzero(mid_idx == m))]
                # Rows (x of left label i, for each (i, s) in the block), columns (t, z)
                mat = _block_rows(theta, rows, left.dims, 2 * dr_pad)
                s2, iso, carry = _split_block(mat, direction, svd_method,
                                              corrections.get(int(m)))
                blocks.append((int(m), rows, s2, iso, carry))

            all_s2 = np.concatenate([b[2] for b in blocks])
            block_of = np.concatenate([np.full(len(b[2]), ib) for ib, b in enumerate(blocks)])
            order = np.argsort(-all_s2, kind='stable')
            s2_sorted = all_s2[order]
            total = s2_sorted.sum()
            tail = np.concatenate([np.cumsum(s2_sorted[::-1])[::-1], [0.]]) / total
            keep = int(np.argmax(tail <= params.cutoff))
            keep = max(1, min(max(keep, params.mindim), params.maxdim, len(s2_sorted)))
            max_err = max(max_err, float(tail[keep]))
            counts = np.bincount(block_of[order[:keep]], minlength=len(blocks))

            kept = [(blk, int(k)) for blk, k in zip(blocks, counts) if k > 0]
            new_bond = Bond(np.array([mid_labels[blk[0]] for blk, _ in kept], dtype=np.int64),
                            np.array([k for _, k in kept], dtype=np.int64))
            dpad = new_bond.pad
            new_a1 = xp.zeros((left.size, 2, left.pad, dpad), dtype=xp.dtype)
            new_a2 = xp.zeros((new_bond.size, 2, dpad, dr_pad), dtype=xp.dtype)
            for inew, ((m, rows, _, iso, carry), k) in enumerate(kept):
                if direction == 'right':
                    row_part, col_part = iso[:, :k], carry[:k]
                else:
                    row_part, col_part = carry[:, :k], iso[:k]
                offset = 0
                for i, s in rows:
                    d = int(left.dims[i])
                    new_a1[i, s, :d, :k] = row_part[offset:offset + d]
                    offset += d
                new_a2[inew, :, :k, :] = col_part.reshape(k, 2, dr_pad).transpose(1, 0, 2)
            if direction == 'right':
                new_a2 /= xp.linalg.norm(new_a2)
            else:
                new_a1 /= xp.linalg.norm(new_a1)
            mps.bonds[site + 1] = new_bond
            mps.tensors[site] = new_a1
            mps.tensors[site + 1] = new_a2

            if direction == 'right':
                lenvs[site + 1] = update_left_sym(lenvs[site], perm(site), new_a1,
                                                  mps.site_map(site), w_elems[site],
                                                  mpo_host[site].shape[1], new_bond)
            else:
                renvs[site + 1] = update_right_sym(renvs[site + 2], perm(site + 1), new_a2,
                                                   mps.site_map(site + 1), w_elems[site + 1],
                                                   mpo_host[site + 1].shape[0], new_bond)

        variance = None
        if measure_variance:
            variance = energy_and_variance(mps.to_dense(), dense_mpo or mpo)[1]
        xp.synchronize()
        record = SweepRecord(sweep=isweep, maxdim=params.maxdim, cutoff=params.cutoff,
                             noise=params.noise, energy=energy,
                             max_bond=max(int(b.dims.sum()) for b in mps.bonds),
                             max_trunc_err=max_err, seconds=time.time() - t0, variance=variance)
        result.history.append(record)
        result.energy = energy
        LOG.info('sym sweep %d maxdim %d E=%.12f bond=%d labels<=%d trunc=%.2e (%.1fs)', isweep,
                 params.maxdim, energy, record.max_bond, max(b.size for b in mps.bonds), max_err,
                 record.seconds)
        if callback is not None:
            callback(record)
        if on_sweep is not None:
            on_sweep(isweep, mps, result.history)
        if (energy_tol is not None and last_energy is not None and isweep + 1 >= min_sweeps
                and abs(energy - last_energy) < energy_tol
                and (isweep + 1 == len(sweeps) or sweeps[isweep + 1].maxdim <= params.maxdim)):
            break
        last_energy = energy

    result.mps = mps.to_dense()
    return result, mps
