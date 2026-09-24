import os
import sys
from pathlib import Path
import logging
import string
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec, Mesh
from jax.experimental.mesh_utils import create_hybrid_device_mesh
from rqutils.ground_locg import ground_locg
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.utils import minimum_weight_link_state
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from ising_hamiltonian import make_apply_h
from distributed import qubit_sharding

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('lattice')
    parser.add_argument('--charges', nargs=2, type=int)
    parser.add_argument('--mu')
    parser.add_argument('--out', default='.')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--tmux', type=int)
    parser.add_argument('--imu-range', nargs=2, type=int)
    parser.add_argument('--gpus')
    parser.add_argument('--mpi', nargs='?', const=True)
    options = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)

    if options.mu:
        mu_low, mu_high, nmu = options.mu.split(',')
        mus = np.linspace(float(mu_low), float(mu_high), int(nmu))
    else:
        mus = np.linspace(0.1, 2.1, 21)

    if options.imu_range:
        mus = mus[options.imu_range[0]:options.imu_range[1]]

    sharding = qubit_sharding(options.mpi, options.gpus)

    if options.lattice.endswith('.json'):
        lattice_name = Path(options.lattice).name[:-5]
        with open(options.lattice, 'r') as source:
            lattice = TriangularZ2Lattice.from_json(source.read())
    else:
        lattice_name = options.lattice
        nrow, ncol = map(int, options.lattice.split('x'))
        lattice = TriangularZ2Lattice((nrow, ncol))

    link_state = minimum_weight_link_state(options.charges, [], lattice)
    dual = lattice.plaquette_dual(link_state)
    hamiltonian = dual.make_hamiltonian(1.)
    apply_h = make_apply_h(hamiltonian, axis_type=AxisType.Explicit)

    nactiv = lattice.num_active_plaquettes
    ndim = 2 ** nactiv

    @jax.jit
    def locg(mu):
        eigval0, eigvec0 = ground_locg(apply_h, 0, args=(mu,), vspace=(ndim, np.float64))[:2]
        xinit = (
            jnp.bitwise_count(
                jax.lax.broadcasted_iota(np.int32, (ndim,), 0, out_sharding=sharding)
            ) == 1
        ).astype(np.float64) / jnp.sqrt(nactiv)
        eigval1 = ground_locg(apply_h, xinit, args=(mu,), orth=eigvec0)[0]
        return jnp.array([eigval0, eigval1])

    if options.tmux is None:
        eigvals = jax.vmap(locg)(mus)
    else:
        eigvals = jax.lax.scan(
            lambda _, mu: (_, locg(mu)),
            None,
            mus
        )[1].block_until_ready()

    if jax.process_index() == 0:
        filename = f'gaps_{lattice_name}{options.suffix}'
        if options.imu_range:
            filename += f'_mu{options.imu_range[0]}-{options.imu_range[1]}'
        filename += '.h5'
        eigvals = np.asarray(eigvals)
        with h5py.File(str(Path(options.out) / filename), 'w', libver='latest') as out:
            out.create_dataset('mus', data=mus)
            out.create_dataset('eigvals', data=eigvals)
