import os
import sys
from pathlib import Path
import logging
import string
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import AxisType
from rqutils.ground_locg import ground_locg
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.utils import minimum_weight_link_state
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from ising_hamiltonian import make_apply_h

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('lattice')
    parser.add_argument('--charges', nargs=2, type=int)
    parser.add_argument('--mu')
    parser.add_argument('--out', default='.')
    parser.add_argument('--mux', type=int)
    parser.add_argument('--gpus')
    parser.add_argument('--localmpi', action='store_true')
    options = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)

    if options.gpus:
        LOG.info('Parallelizing over %s', options.gpus)
        if options.gpus == 'mpi':
            from mpi4py import MPI
            jax.distributed.initialize(cluster_detection_method="mpi4py")
        elif options.localmpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            gpus = options.gpus.split(',')
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus[comm.Get_rank()]
            jax.distributed.initialize('localhost:10000', comm.Get_size(), comm.Get_rank())
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus

        ngpu = jax.device_count()
        nax = np.log2(ngpu).astype(int)
        if 2 ** nax != ngpu:
            raise ValueError('Invalid ngpu')
        mesh_shape = (2,) * nax
        axis_names = tuple(string.ascii_lowercase[:nax])
        jax.set_mesh(jax.make_mesh(mesh_shape, axis_names, axis_types=(AxisType.Explicit,) * nax))

    if options.lattice.endswith('.json'):
        lattice_name = Path(options.lattice).name[:-5]
        with open(options.lattice, 'r') as source:
            lattice = TriangularZ2Lattice.from_json(source.read())
    else:
        lattice_name = options.lattice
        nrow, ncol = map(int, options.lattice.split('x'))
        lattice = TriangularZ2Lattice((nrow, ncol))

    if options.charges is not None:
        link_state = minimum_weight_link_state(options.charges, [], lattice)
    else:
        link_state = np.zeros(lattice.num_links, dtype=np.uint8)
    dual = lattice.plaquette_dual(link_state)
    hamiltonian = dual.make_hamiltonian(1.)
    apply_h = make_apply_h(hamiltonian, axis_type=AxisType.Explicit)

    nactiv = lattice.num_active_plaquettes
    ndim = 2 ** nactiv

    if options.mu:
        mu_low, mu_high, nmu = options.mu.split(',')
        mus = np.linspace(float(mu_low), float(mu_high), int(nmu))
    else:
        mus = np.linspace(0.1, 2.1, 21)

    @jax.jit
    @jax.vmap
    @jax.jit
    def locg(mu):
        eigval0, eigvec0 = ground_locg(apply_h, 0, args=(mu,), vspace=(ndim, np.float64))[:2]
        xinit = (jnp.bitwise_count(jnp.arange(ndim)) == 1).astype(np.float64) / jnp.sqrt(nactiv)
        eigval1 = ground_locg(apply_h, xinit, args=(mu,), orth=eigvec0)[0]
        return jnp.array([eigval0, eigval1])

    if options.mux is None:
        eigvals = locg(mus)
    else:
        eigvals = jax.lax.scan(
            lambda _, mublock: (_, locg(mublock)),
            None,
            mus.reshape((-1, options.mux))
        )[1].reshape((-1, 2))

    filename = f'gaps_{lattice_name}.h5'
    if (proc_id := jax.process_index()) == 0:
        with h5py.File(str(Path(options.out) / filename), 'w', libver='latest') as out:
            out.create_dataset('eigvals', data=eigvals)
