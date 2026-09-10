import os
import sys
import string
from pathlib import Path
import logging
import numpy as np
import h5py
from rustworkx import adjacency_matrix
import jax
import jax.numpy as jnp
from jax.sharding import AxisType
from rqutils.ground_locg import ground_locg
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from ising_hamiltonian import make_apply_h
from face_area import compute_counts


@jax.jit(static_argnames=['apply_h', 'return_eigvec'])
def compute_area(counts, apply_h, return_eigvec=False):
    eigvec = ground_locg(apply_h, 0, vspace=(counts.shape[0], np.float64))[1]
    if return_eigvec:
        return counts @ jnp.square(eigvec), eigvec
    return counts @ jnp.square(eigvec)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('lattice')
    parser.add_argument('monopole', type=int)
    parser.add_argument('--out', default='.')
    parser.add_argument('--mu')
    parser.add_argument('--save-counts', action='store_true')
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
        name = Path(options.lattice).name[:-5]
        with open(options.lattice, 'r') as source:
            lattice = TriangularZ2Lattice.from_json(source.read())
    else:
        name = options.lattice
        nrow, ncol = map(int, options.lattice.split('x'))
        lattice = TriangularZ2Lattice((nrow, ncol))

    lattice.activate_plaquette(options.monopole, False)
    link_state = np.zeros(lattice.num_links, dtype=np.uint8)
    link_state[::-1][lattice.plaquette_links(options.monopole)] = 1
    dual = lattice.plaquette_dual(link_state)

    nactiv = lattice.num_active_plaquettes
    ndim = 2 ** nactiv

    pgraph = lattice.dual_graph.copy()
    for idx in pgraph.node_indices():
        if not isinstance(pgraph[idx], int):
            pgraph.remove_node(idx)
    amat = adjacency_matrix(pgraph).astype(np.uint8)
    seeds = np.delete(amat[options.monopole], options.monopole)
    bmat = np.delete(np.delete(amat, options.monopole, axis=0), options.monopole, axis=1)

    counts = compute_counts(bmat, seeds)

    if options.mu is None:
        mus = np.linspace(0.1, 2.6, 26)
    else:
        mumin, mumax = map(float, options.mu.split(',')[:2])
        nmu = int(options.mu.split(',')[2])
        mus = np.linspace(mumin, mumax, nmu)

    # eigvecs = np.empty(mus.shape + (ndim,))
    areas = np.empty_like(mus)

    for imu, mu in enumerate(mus):
        print('mu', mu)
        apply_h = make_apply_h(dual.make_hamiltonian(mu))
        areas[imu] = compute_area(counts, apply_h)
        
    output_name = str(Path(options.out) / f'{name}_{options.monopole}.h5')
    with h5py.File(output_name, 'w') as out:
        out.create_dataset('mus', data=mus)
        # out.create_dataset('eigvecs', data=eigvecs)
        out.create_dataset('areas', data=areas)
        if options.save_counts:
            out.create_dataset('counts', data=counts)
