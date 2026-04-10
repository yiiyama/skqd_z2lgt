import sys
from pathlib import Path
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
# from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard
from skqd_z2lgt.ground_locg import ground_locg
sys.path.append(str(Path(__file__).parent / 'lib'))
from unitary_krylov import make_hvec  # pylint: disable=wrong-import-position

logging.basicConfig(level=logging.INFO)
jax.config.update('jax_enable_x64', True)

excited_links = {
    (4, 1): [5],
    (4, 2): [8],
    (4, 3): [11, 13],
    (4, 4): [14, 16],
    (4, 5): [17, 19, 22],
    (4, 6): [20, 22, 25],
    (4, 7): [23, 25, 28, 31],
    (4, 8): [],
    (6, 1): [],
    (6, 2): [12],
    (6, 3): [17],
    (6, 4): [21, 24],
    (6, 5): [26, 29],
    (8, 1): [10],
    (8, 2): [16],
    (8, 3): [22, 24]
}

TEMPLATE = str(Path(__file__).parents[1] / 'data/sparsity_hstack/probs_{}x{}_centercharge.h5')


def run(config):
    lattice = TriangularZ2Lattice(config)
    base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
    base_link_state[::-1][excited_links[config]] = 1
    dual_lattice = lattice.plaquette_dual(base_link_state)
    nplaq = lattice.num_plaquettes
    hamiltonian = dual_lattice.make_hamiltonian(1.)

    if nplaq <= 28:
        # I have no idea why the "dtype switching" works, but for mid-scale lattices compute() won't
        # compile with float64
        hvec = make_hvec(hamiltonian, variable=True, dtype=np.complex128)

        @jax.jit
        def compute(plaquette_energy):
            xmat = jax.nn.one_hot(0, 2 ** nplaq, dtype=np.complex128)
            ground_state = ground_locg(lambda x: hvec(x, plaquette_energy), xmat)[1]
            probs = jnp.square(jnp.abs(ground_state))
            sort_idx = jnp.argsort(probs, descending=True)
            return probs[sort_idx], sort_idx

    else:
        hvec = make_hvec(hamiltonian, variable=True, dtype=np.float64)

        @jax.jit
        def compute(plaquette_energy):
            xmat = jax.nn.one_hot(0, 2 ** nplaq, dtype=np.float64)
            ground_state = ground_locg(lambda x: hvec(x, plaquette_energy), xmat)[1]
            return jnp.square(jnp.abs(ground_state))

    lambdas = np.linspace(0., 4., 21)

    with h5py.File(TEMPLATE.format(*config), 'a', libver='latest') as out:
        if 'lambdas' not in out:
            out.create_dataset('lambdas', data=lambdas)

        for ilmd, plaquette_energy in enumerate(lambdas):
            if f'probs_{ilmd}' in out:
                continue

            print('  ', plaquette_energy)
            if nplaq <= 28:
                sorted_probs, sort_idx = compute(plaquette_energy)
            else:
                probs = compute(plaquette_energy)
                sort_idx = np.argsort(probs)[::-1]
                sorted_probs = probs[sort_idx]

            s4 = np.searchsorted(np.cumsum(sorted_probs), 0.9999)
            out.create_dataset(f'probs_{ilmd}', data=sorted_probs[:s4])
            out.create_dataset(f'indices_{ilmd}', data=sort_idx[:s4])

            if nplaq == 30:
                sys.exit(0)


if __name__ == '__main__':
    # for conf in excited_links:
    for conf in [(4, 8)]:
        print(conf)
        run(conf)
