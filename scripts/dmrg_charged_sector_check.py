"""Charged-sector validation of the DMRG: long-string sector of (4, n) triangular lattices.

Part A: for lattices small enough for exact diagonalization (matrix-free Lanczos on the dual model),
        compare E0 and E1 from dual and primal DMRG with the exact values.
Part B: lambda scan for a larger lattice (default 4x7, charges at the ends of the middle row, the
        "long string" sector), with E0, E1, the gap, and convergence in the bond dimension.

Example:
    python scripts/dmrg_charged_sector_check.py --exact 4x4 4x5 4x6 --scan 4x7 -o charged.h5
"""
from argparse import ArgumentParser
import json
import logging
import time
import numpy as np
import h5py
from scipy.sparse.linalg import LinearOperator, eigsh
from skqd_z2lgt.dmrg_gpu.backend import xp
from skqd_z2lgt.dmrg_gpu.models import make_lattice, base_link_state, dual_problem
from skqd_z2lgt.dmrg_gpu.mpo import pauli_terms
from skqd_z2lgt.dmrg_gpu.runner import solve


def long_string_charges(lattice) -> list[int]:
    """Vertex ids at the two ends of the middle row of the lattice."""
    vertices = lattice.graph.attrs['vertices']
    rows = sorted({v.position[1] for v in vertices})
    middle = rows[len(rows) // 2]
    row = sorted((v.position[0], v.id) for v in vertices if v.position[1] == middle)
    return [row[0][1], row[-1][1]]


def exact_dual(lattice, coupling, charges, k=2):
    """Lowest k eigenvalues of the dual Ising model by matrix-free Lanczos."""
    hamiltonian = dual_problem(lattice, coupling, base_link_state(lattice, charges)).hamiltonian
    nq = hamiltonian.num_qubits
    index = np.arange(2 ** nq, dtype=np.int64)
    spins = [1. - 2. * ((index >> q) & 1) for q in range(nq)]
    diag = np.zeros(2 ** nq)
    flips = []
    for coeff, ops in pauli_terms(hamiltonian):
        if 'X' in ops.values():
            (qubit,) = ops
            flips.append((coeff, 1 << qubit))
        else:
            term = np.full(2 ** nq, coeff)
            for qubit in ops:
                term *= spins[qubit]
            diag += term
    del spins

    def matvec(vec):
        vec = vec.reshape(-1)
        out = diag * vec
        for coeff, mask in flips:
            out += coeff * vec[index ^ mask]
        return out

    op = LinearOperator((2 ** nq, 2 ** nq), matvec=matvec, dtype=np.float64)
    return np.sort(eigsh(op, k=k, which='SA', tol=1.e-12)[0])


def parse_shape(text):
    return tuple(int(x) for x in text.split('x'))


def main():
    parser = ArgumentParser()
    parser.add_argument('--exact', nargs='*', default=['4x4', '4x5', '4x6'])
    parser.add_argument('--exact-lambdas', type=float, nargs='+', default=[0.5, 1.0, 1.2, 1.5, 2.0])
    parser.add_argument('--scan', nargs='*', default=['4x7'])
    parser.add_argument('--scan-lambdas', type=float, nargs='+',
                        default=np.round(np.arange(0.1, 2.11, 0.2), 2).tolist())
    parser.add_argument('--chis', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--primal-lambdas', type=float, nargs='*', default=[1.2, 1.5])
    parser.add_argument('--sectors', choices=['both', 'vacuum', 'charged'], default='both',
                        help='Sectors for the exact comparison (part A).')
    parser.add_argument('--backend', default='auto')
    parser.add_argument('-o', '--out', default='charged_sector_check.h5')
    options = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    xp.configure(options.backend)

    def dual_sweeps(chi):
        return {'nsweeps': 10, 'maxdim': [16, 32, chi], 'cutoff': 1.e-12,
                'noise': [1.e-5, 1.e-6, 1.e-7, 0.]}

    def primal_sweeps(chi):
        return {'nsweeps': 14, 'maxdim': [32, 64, 128, chi], 'cutoff': 1.e-12,
                'noise': [1.e-3, 1.e-4, 1.e-5, 1.e-6, 0.]}

    records = []

    def emit(record):
        records.append(record)
        print(json.dumps(record), flush=True)

    # Part A: exact comparison
    for shape in options.exact:
        lattice = make_lattice(parse_shape(shape))
        sectors = {'both': ([], long_string_charges(lattice)), 'vacuum': ([],),
                   'charged': (long_string_charges(lattice),)}[options.sectors]
        for charges in sectors:
            for coupling in options.exact_lambdas:
                exact = exact_dual(lattice, coupling, charges)
                dual = solve(lattice, coupling, charges, model='dual', nstates=2,
                             sweeps=dual_sweeps(256), energy_tol=1.e-11)
                primal = solve(lattice, coupling, charges, model='primal', nstates=2, penalty=8.,
                               sweeps=primal_sweeps(256), energy_tol=1.e-11)
                emit({'part': 'exact', 'lattice': shape, 'charges': charges, 'lambda': coupling,
                      'E0_exact': exact[0], 'E1_exact': exact[1],
                      'dE0_dual': dual['energies'][0] - exact[0],
                      'dE1_dual': dual['energies'][1] - exact[1],
                      'dE0_primal': primal['energies'][0] - exact[0],
                      'dE1_primal': primal['energies'][1] - exact[1],
                      'primal_penalty': primal['state_penalty_energies'].tolist()})

    # Part B: lambda scan with chi convergence
    for shape in options.scan:
        lattice = make_lattice(parse_shape(shape))
        charges = long_string_charges(lattice)
        for coupling in options.scan_lambdas:
            record = {'part': 'scan', 'lattice': shape, 'charges': charges, 'lambda': coupling}
            for chi in options.chis:
                t0 = time.time()
                dual = solve(lattice, coupling, charges, model='dual', nstates=2,
                             sweeps=dual_sweeps(chi), energy_tol=1.e-11)
                vac = solve(lattice, coupling, [], model='dual', sweeps=dual_sweeps(chi),
                            energy_tol=1.e-11)
                record[f'chi{chi}'] = {'E0': dual['energies'][0], 'E1': dual['energies'][1],
                                       'gap': dual['energies'][1] - dual['energies'][0],
                                       'E0_vac': vac['energy'],
                                       'max_bond': int(dual['mps_bond_dims'].max()),
                                       'seconds': time.time() - t0}
            if coupling in options.primal_lambdas:
                primal = solve(lattice, coupling, charges, model='primal', penalty=8.,
                               sweeps=primal_sweeps(max(options.chis)), energy_tol=1.e-11)
                record['primal_E0'] = primal['energy']
                record['primal_penalty'] = primal['penalty_energy']
            emit(record)

    with h5py.File(options.out, 'w') as out:
        out.create_dataset('records', data=json.dumps(records))


if __name__ == '__main__':
    main()
