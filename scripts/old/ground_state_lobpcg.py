"""Compute the ground state energy and vector."""
import sys
from pathlib import Path
import numpy as np
import h5py
import jax
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard
sys.path.append(str(Path(__file__).parent / 'lib'))
from unitary_krylov import make_hvec  # pylint: disable=wrong-import-order


def compute_ground_state(config, excited_links, plaquette_energy, filename):
    """Compute the ground state."""
    lattice = TriangularZ2Lattice(config)
    base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
    base_link_state[::-1][excited_links] = 1
    dual_lattice = lattice.plaquette_dual(base_link_state)
    hamiltonian = dual_lattice.make_hamiltonian(plaquette_energy)
    nplaq = lattice.num_plaquettes
    dtype = np.float64 if nplaq > 28 else np.complex128
    hvec = make_hvec(hamiltonian, dtype=dtype)

    @jax.jit
    def compute():
        xmat = jax.nn.one_hot(0, 2 ** lattice.num_plaquettes, dtype=dtype)[:, None]
        # pylint: disable-next=unbalanced-tuple-unpacking
        eigvals, eigvecs, _ = lobpcg_standard(lambda x: -hvec(x.T).T, xmat)
        return -eigvals[0], eigvecs[:, 0]

    eigval, eigvec = compute()
    with h5py.File(filename, 'w', libver='latest') as out:
        out.create_dataset('eigval', data=eigval)
        out.create_dataset('eigvec', data=eigvec)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, nargs=2, required=True)
    parser.add_argument('--links', type=int, nargs='+')
    parser.add_argument('--plaquette-energy', type=float, required=True)
    parser.add_argument('--out', required=True)
    options = parser.parse_args()

    jax.config.update('jax_enable_x64', True)

    compute_ground_state(tuple(options.config), options.links, options.plaquette_energy,
                         options.out)
