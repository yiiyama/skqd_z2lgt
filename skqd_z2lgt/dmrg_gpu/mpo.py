"""Construction of MPOs for real Pauli-sum Hamiltonians (X and Z operators only).

MPO tensors are built on the host with NumPy and moved to the active device afterwards. Index
convention: W[a, b, s, t] = <s| (operator on this site) |t>, with a / b the left / right MPO bonds.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from qiskit.quantum_info import SparsePauliOp

PAULI = {
    'I': np.eye(2),
    'X': np.array([[0., 1.], [1., 0.]]),
    'Z': np.array([[1., 0.], [0., -1.]]),
}

Term = tuple[float, dict[int, str]]


def pauli_terms(operator: SparsePauliOp, atol: float = 1.e-12) -> list[Term]:
    """Convert a SparsePauliOp into a list of (coeff, {qubit: 'X' or 'Z'}).

    Qubit indices follow the qiskit convention (qubit q is the q-th character from the right of the
    Pauli label). Identity terms are kept with an empty op dict.
    """
    terms = []
    for pauli, coeff in zip(operator.paulis, operator.coeffs):
        if abs(coeff.imag) > atol:
            raise ValueError(f'Complex coefficient {coeff} for {pauli}')
        if np.any(pauli.x & pauli.z):
            raise ValueError(f'Y operators are not supported: {pauli}')
        ops = {int(q): 'X' for q in np.nonzero(pauli.x)[0]}
        ops.update({int(q): 'Z' for q in np.nonzero(pauli.z)[0]})
        # SparsePauliOp may carry a phase on the Pauli itself
        sign = (-1j) ** pauli.phase
        value = coeff * sign
        if abs(value.imag) > atol:
            raise ValueError(f'Complex term {value} for {pauli}')
        terms.append((float(value.real), ops))
    return terms


def site_order(terms: list[Term], num_sites: int, method: str = 'snake',
               positions: Optional[np.ndarray] = None) -> np.ndarray:
    """Return order[k] = qubit placed at MPS site k.

    - 'snake': geometric boustrophedon (see snake_order) in the direction with the smaller maximum
      number of terms crossing a chain cut. Requires positions; falls back to 'rcm' without them.
    - 'snake-rows' / 'snake-columns': snake in a fixed direction.
    - 'rcm': reverse Cuthill-McKee on the interaction graph (geometry-free bandwidth reduction).
    - 'natural': qubit order.
    """
    if method == 'natural':
        return np.arange(num_sites)
    if method.startswith('snake') and positions is None:
        method = 'rcm'
    if method == 'snake':
        candidates = [snake_order(positions, 'rows'), snake_order(positions, 'columns')]
        return min(candidates, key=lambda order: (cut_sizes(terms, order).max(),
                                                  cut_sizes(terms, order).mean()))
    if method == 'snake-rows':
        return snake_order(positions, 'rows')
    if method == 'snake-columns':
        return snake_order(positions, 'columns')
    if method != 'rcm':
        raise ValueError(f'Unknown ordering {method}')

    rows, cols = [], []
    for _, ops in terms:
        qubits = sorted(ops)
        for i, q1 in enumerate(qubits):
            for q2 in qubits[i + 1:]:
                rows += [q1, q2]
                cols += [q2, q1]
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_sites, num_sites)).tocsr()
    return np.asarray(reverse_cuthill_mckee(adjacency, symmetric_mode=True), dtype=int)


def snake_order(positions: np.ndarray, direction: str = 'rows') -> np.ndarray:
    """Boustrophedon ordering of sites with 2D positions.

    direction='rows': sites are grouped into rows of equal y (rounded to half units, which separates
    e.g. horizontal and diagonal links), rows are taken in increasing y, and within a row the sites
    are traversed in x alternately left-to-right and right-to-left. 'columns' swaps x and y.
    """
    positions = np.asarray(positions, dtype=float)
    stripe_axis = 1 if direction == 'rows' else 0
    stripe = np.round(positions[:, stripe_axis] * 2.) / 2.
    along = positions[:, 1 - stripe_axis]
    order = []
    for istripe, value in enumerate(np.unique(stripe)):
        members = np.nonzero(stripe == value)[0]
        sign = 1. if istripe % 2 == 0 else -1.
        order.extend(members[np.argsort(sign * along[members], kind='stable')])
    return np.array(order, dtype=int)


def cut_sizes(terms: list[Term], order: np.ndarray) -> np.ndarray:
    """Number of multi-site terms crossing each cut of the chain (a proxy for MPO and MPS cost)."""
    num_sites = len(order)
    site_of = np.empty(num_sites, dtype=int)
    site_of[order] = np.arange(num_sites)
    cuts = np.zeros(max(num_sites - 1, 1), dtype=int)
    for _, ops in terms:
        if len(ops) > 1:
            sites = site_of[list(ops)]
            cuts[sites.min():sites.max()] += 1
    return cuts


def permute_terms(terms: list[Term], order: np.ndarray) -> list[Term]:
    """Relabel term supports from qubit indices to MPS site indices."""
    qubit_to_site = np.empty(len(order), dtype=int)
    qubit_to_site[order] = np.arange(len(order))
    return [(coeff, {int(qubit_to_site[q]): op for q, op in ops.items()}) for coeff, ops in terms]


def build_mpo(terms: list[Term], num_sites: int, compress: bool = True,
              tol: float = 1.e-13) -> list[np.ndarray]:
    """Exact MPO of sum_t coeff_t prod_{j in t} op_{t,j}, optionally SVD-compressed
    (numerically lossless with the given relative tolerance). See automaton_mpo."""
    mpo = automaton_mpo(terms, num_sites)[0]
    if compress:
        mpo = compress_mpo(mpo, tol=tol)
    return mpo


def automaton_mpo(terms: list[Term], num_sites: int) -> tuple[list[np.ndarray], list[list]]:
    """Exact MPO via a finite-state automaton, together with the automaton state of every MPO bond
    index (bond_keys[j][a] for the bond left of site j).

    Automaton states on bond j (between sites j-1 and j) are 'start' (no operator placed yet),
    'final' (term complete) and the tuple of (site, op) placed so far for open terms. Terms sharing
    a prefix share states. The coefficient is attached to the last operator of each term. Since a
    state records which operators have been placed, it has a definite symmetry charge, which is
    what the block-sparse (symmetric) DMRG needs; SVD compression would mix states and lose it.
    """
    start, final = ('start',), ('final',)
    # Per site: (src, dst) -> op for automaton moves that do not complete a term, and
    # src -> [(op, coeff), ...] for moves into 'final'
    moves = [{(start, start): 'I', (final, final): 'I'} for _ in range(num_sites)]
    closes = [dict() for _ in range(num_sites)]

    for coeff, ops in terms:
        if not ops:
            # Constant offset, attached to site 0
            closes[0].setdefault(start, []).append(('I', coeff))
            continue
        sites = sorted(ops)
        key = start
        for site in range(sites[0], sites[-1]):
            op = ops.get(site, 'I')
            new_key = key + ((site, op),) if op != 'I' else key
            moves[site][(key, new_key)] = op
            key = new_key
        closes[sites[-1]].setdefault(key, []).append((ops[sites[-1]], coeff))

    def edges(site):
        yield from moves[site]
        yield from ((src, final) for src in closes[site])

    # Keep only states reachable from 'start' on bond 0 and co-reachable to 'final' on the last bond
    reachable = [set() for _ in range(num_sites + 1)]
    reachable[0].add(start)
    for site in range(num_sites):
        reachable[site + 1].update(dst for src, dst in edges(site) if src in reachable[site])
    coreachable = [set() for _ in range(num_sites + 1)]
    coreachable[num_sites].add(final)
    for site in range(num_sites - 1, -1, -1):
        coreachable[site].update(src for src, dst in edges(site) if dst in coreachable[site + 1])

    bond_states = []
    for bond in range(num_sites + 1):
        keys = sorted(reachable[bond] & coreachable[bond], key=_state_sort_key)
        bond_states.append({key: idx for idx, key in enumerate(keys)})

    mpo = []
    for site in range(num_sites):
        left, right = bond_states[site], bond_states[site + 1]
        tensor = np.zeros((len(left), len(right), 2, 2))
        for (src, dst), op in moves[site].items():
            if src in left and dst in right:
                tensor[left[src], right[dst]] += PAULI[op]
        if final in right:
            for src, contributions in closes[site].items():
                if src in left:
                    for op, coeff in contributions:
                        tensor[left[src], right[final]] += coeff * PAULI[op]
        mpo.append(tensor)

    bond_keys = [sorted(states, key=states.get) for states in bond_states]
    return mpo, bond_keys


def _state_sort_key(key):
    if key == ('start',):
        return (0, ())
    if key == ('final',):
        return (2, ())
    return (1, key)


def compress_mpo(mpo: list[np.ndarray], tol: float = 1.e-13) -> list[np.ndarray]:
    """SVD-compress an MPO with a left-to-right then right-to-left sweep.

    Singular values below tol * (largest singular value) are dropped, which removes only exact
    (numerical) linear dependencies of the automaton states.
    """
    mpo = [w.copy() for w in mpo]
    nsites = len(mpo)
    for _ in range(2):
        # Left to right
        for site in range(nsites - 1):
            wl, wr, d1, d2 = mpo[site].shape
            mat = mpo[site].transpose(0, 2, 3, 1).reshape(wl * d1 * d2, wr)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            keep = max(1, int(np.count_nonzero(s > tol * s[0])))
            mpo[site] = u[:, :keep].reshape(wl, d1, d2, keep).transpose(0, 3, 1, 2)
            carry = s[:keep, None] * vh[:keep]
            mpo[site + 1] = np.tensordot(carry, mpo[site + 1], axes=([1], [0]))
        # Right to left
        for site in range(nsites - 1, 0, -1):
            wl, wr, d1, d2 = mpo[site].shape
            mat = mpo[site].reshape(wl, wr * d1 * d2)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            keep = max(1, int(np.count_nonzero(s > tol * s[0])))
            mpo[site] = vh[:keep].reshape(keep, wr, d1, d2)
            carry = u[:, :keep] * s[None, :keep]
            mpo[site - 1] = np.tensordot(mpo[site - 1], carry, axes=([1], [0])).transpose(0, 3, 1, 2)
    return mpo


def mpo_to_dense(mpo: list[np.ndarray]) -> np.ndarray:
    """Contract an MPO into a dense matrix (site 0 = most significant index). For tests only."""
    result = mpo[0]  # (1, w, d, d)
    result = result[0]  # (w, d, d)
    for tensor in mpo[1:]:
        # result (w, D, D), tensor (w, w', d, d)
        result = np.einsum('aij,abkl->bikjl', result, tensor)
        w = result.shape[0]
        dim = result.shape[1] * result.shape[2]
        result = result.reshape(w, dim, dim)
    return result[0]


def terms_to_dense(terms: list[Term], num_sites: int) -> np.ndarray:
    """Dense matrix of a site-indexed term list (site 0 = most significant index). For tests."""
    dim = 2 ** num_sites
    matrix = np.zeros((dim, dim))
    for coeff, ops in terms:
        mat = np.eye(1)
        for site in range(num_sites):
            mat = np.kron(mat, PAULI[ops.get(site, 'I')])
        matrix += coeff * mat
    return matrix


def mpo_bond_dims(mpo: list[np.ndarray]) -> list[int]:
    return [w.shape[1] for w in mpo[:-1]]


def make_mpo(operator: SparsePauliOp | list[Term], num_sites: Optional[int] = None,
             order: Optional[np.ndarray] = None, compress: bool = True) -> list[np.ndarray]:
    """Convenience wrapper: Pauli operator (qubit-indexed) -> host MPO in the given site order."""
    if isinstance(operator, SparsePauliOp):
        num_sites = operator.num_qubits
        terms = pauli_terms(operator)
    else:
        terms = operator
    if order is None:
        order = np.arange(num_sites)
    return build_mpo(permute_terms(terms, order), num_sites, compress=compress)
