import os
import sys
from pathlib import Path
import logging
import string
from functools import partial
import numpy as np
import h5py
import jax
from jax.sharding import PartitionSpec, AxisType, NamedSharding, auto_axes
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.ground_locg import ground_locg
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from ising_hamiltonian import make_apply_h


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('nrow', type=int)
    parser.add_argument('ncol', type=int)
    parser.add_argument('plaquette_energy', type=float)
    parser.add_argument('--gpus')
    parser.add_argument('--out', default='.')
    parser.add_argument('--xprof')
    parser.add_argument('--localmpi', action='store_true')
    options = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger()

    lattice = TriangularZ2Lattice((options.nrow, options.ncol))
    base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)

    nsites = (options.ncol - 1) // 2 + 1 + (options.nrow // 2 + 1) % 2
    vfirst = (lattice.num_vertices - nsites) // 2
    charged_vertices = [vfirst, vfirst + nsites - 1]
    links = []
    for ivtx in range(*charged_vertices):
        links.append(lattice.graph.edge_indices_from_endpoints(ivtx, ivtx + 1)[0])
    base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
    base_link_state[::-1][links] = 1
    dual_lattice = lattice.plaquette_dual(base_link_state)
    nplaq = lattice.num_plaquettes
    hamiltonian = dual_lattice.make_hamiltonian(options.plaquette_energy)

    apply_h = make_apply_h(hamiltonian, axis_type=AxisType.Explicit)

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
        mesh = jax.make_mesh(mesh_shape, axis_names, axis_types=(AxisType.Explicit,) * nax)
        sharding = NamedSharding(mesh, PartitionSpec(axis_names))
        ground_locg = partial(ground_locg, sharding=sharding)

    if (proc_id := jax.process_index()) == 0 and options.xprof:
        with jax.profiler.trace(options.xprof):
            eigval, eigvec, iter = ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64))
    else:
        eigval, eigvec, iter = ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64))

    filename = f'ground_{options.nrow}x{options.ncol}_l{options.plaquette_energy:.2f}.h5'
    if proc_id == 0:
        LOG.info('LOCG iterations: %d', iter)
        with h5py.File(str(Path(options.out) / filename), 'w', libver='latest') as out:
            out.create_dataset('eigval', data=eigval)
            out.create_dataset('eigvec', shape=eigvec.shape, dtype=eigvec.dtype)
    else:
        MPI.COMM_WORLD.recv(source=proc_id - 1, tag=11)

    LOG.info('Writing from process %d on indices %s', proc_id,
             list(shard.index for shard in eigvec.addressable_shards))

    with h5py.File(str(Path(options.out) / filename), 'a', libver='latest') as out:
        for shard in eigvec.addressable_shards:
            out['eigvec'][shard.index] = shard.data

    if proc_id < jax.process_count() - 1:
        MPI.COMM_WORLD.send(1, dest=proc_id + 1, tag=11)

    # LOG.info('compiling')
    # print(ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64), sharding=sh_single)[0])
    # LOG.info('tracing')
    # with jax.profiler.trace('/tmp/ground_4x8'):
    #     ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64), sharding=sh_single)
    # LOG.info('validation')
    # print(jnp.linalg.eigvalsh(hamiltonian.to_matrix())[0])
