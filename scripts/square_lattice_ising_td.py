import os
import logging
import sys
from pathlib import Path
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import run_sqd
from skqd_z2lgt.perturbative_sqd import make_pt_core
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from square_ising import make_primal_graph, get_hamiltonian_coeffs


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('nx', type=int)
    parser.add_argument('ny', type=int)
    parser.add_argument('--charges', nargs=2)
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--mu')
    parser.add_argument('--out', default='square_lattice_ising_tqd.h5')
    parser.add_argument('--gpus')
    options = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger()

    if options.mu:
        if ',' in options.mu:
            mu_low, mu_high, nmu = options.mu.split(',')
            mus = np.linspace(float(mu_low), float(mu_high), int(nmu))
        else:
            mus = np.array([float(options.mu)])
    else:
        mus = np.linspace(1.5, 4.5, 21)

    # Number of plaquettes
    nx = options.nx
    ny = options.ny
    nplaq = nx * ny

    primal_graph = make_primal_graph(nx, ny, options.charges)
    z_coeffs, zz_coeffs = get_hamiltonian_coeffs(primal_graph)
    paulis = []
    coeffs = []
    for ip, c in z_coeffs.items():
        paulis.append('I' * (nplaq - ip - 1) + 'Z' + 'I' * ip)
        coeffs.append(c)
    for (ip1, ip2), c in zz_coeffs.items():
        paulis.append('I' * (nplaq - ip2 - 1) + 'Z' + 'I' * (ip2 - ip1 - 1) + 'Z' + 'I' * ip1)
        coeffs.append(c)
    paulis += ['I' * (nplaq - ip - 1) + 'X' + 'I' * ip for ip in range(nplaq)]
    coeffs += [0.] * nplaq
    paulis = np.array(paulis)
    coeffs = np.array(coeffs)

    hamiltonians = []
    for mu in mus:
        coeffs[-nplaq:] = mu
        hamiltonians.append(
            PauliSumXZ.from_paulisum((paulis, coeffs), force_real=True, add_padding=True)
        )
    hamiltonians = jax.tree.map(lambda *leaves: jnp.stack(leaves), *hamiltonians)

    @jax.jit
    def run_sqd_eigvalonly(sqd_states, hamiltonian):
        eigval = run_sqd(hamiltonian, sqd_states, states_size=sqd_states.shape[0],
                         return_eigvec=False)[0]
        return sqd_states, eigval

    sqd_states = make_pt_core(nplaq, options.order, True)
    energies = jax.lax.scan(run_sqd_eigvalonly, sqd_states, hamiltonians)[1]

    with h5py.File(options.out, 'w') as out:
        out.create_dataset('order', data=options.order)
        out.create_dataset('mus', data=mus)
        out.create_dataset('energies', data=energies)
