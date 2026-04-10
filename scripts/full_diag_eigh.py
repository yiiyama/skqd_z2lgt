import os
import sys
from pathlib import Path
from functools import partial
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.linalg import eigh, eigvalsh
from scipy.optimize import curve_fit
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding
from jax.experimental.ode import odeint
from qiskit.quantum_info import SparsePauliOp
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.extensions import denoising, perturbation_2q
from skqd_z2lgt.circuits import make_plaquette_circuits
from skqd_z2lgt.mwpm import minimum_weight_link_state
from skqd_z2lgt.sqd import sqd
from skqd_z2lgt.utils import read_bits
from skqd_z2lgt.tasks.common import make_dual_lattice
from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard
sys.path.append(str(Path(__file__).parent / 'lib'))
from unitary_krylov import make_hvec, make_trotter_uvec, integrate, simulate, sample, exact_diag, compute_gen_eigvals

configs = {
    4: [
    '''
* *
 *
* *
 *
* *
    ''',
    '''
* *
 * *
* *
 * *
* *
    ''',
    '''
* * *
 * *
* * *
 * *
* * *
    ''',
    ],
    6: [
    '''
* *
 *
* *
 *
* *
 *
* *
    ''',
    '''
* *
 * *
* *
 * *
* *
 * *
* *
    ''',
    ],
    8: [
    '''
* *
 *
* *
 *
* *
 *
* *
 *
* *
    ''',
    ]
}

excited_links = {
    4: [
        [5],
        [8],
        [11, 13],
        [14, 16],
        [17, 19, 22]
    ],
    6: [
        [],
        [12],
        [17]
    ],
    8: [
        [10],
        [16]
    ]
}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    jax.config.update('jax_enable_x64', True)

    plaquette_energy = 0.5

    filename = '/work/gp14/p14000/data/full_diag.h5'

    with h5py.File(filename, 'w', libver='latest') as out:
        for rows in [4, 6, 8]:
            for iconfig, (config, links) in enumerate(zip(configs[rows], excited_links[rows])):
                cols = iconfig + 1
                if out.get(f'row{rows}_col{cols}') is None:
                    print(rows, cols)
                    lattice = TriangularZ2Lattice(config)
                    nplaq = lattice.num_plaquettes

                    base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
                    base_link_state[::-1][links] = 1
                    dual_lattice = lattice.plaquette_dual(base_link_state)
                    hamiltonian = dual_lattice.make_hamiltonian(plaquette_energy)

                    eigvals, eigvecs = jnp.linalg.eigh(hamiltonian.to_matrix().real)
                    group = out.create_group(f'row{rows}_col{cols}')
                    group.create_dataset('eigvals', data=eigvals)
                    group.create_dataset('ground_state', data=eigvecs[:, 0])
                    group.create_dataset('gamma_0', data=eigvecs[0])
