import os
import sys
from pathlib import Path
import logging
import string
from functools import partial
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, AxisType, NamedSharding
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from ising_hamiltonian import make_apply_u


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('nrow', type=int)
    parser.add_argument('ncol', type=int)
    parser.add_argument('plaquette_energy', type=float)
    parser.add_argument('dt', type=float)
    parser.add_argument('nsteps', type=int)
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

    apply_u = make_apply_u(hamiltonian, axis_type=AxisType.Explicit)

    sharding = None
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

    @jax.jit
    def run():
        vec = jax.device_put(jax.nn.one_hot(0, 2 ** nplaq, dtype=np.complex128), sharding)
        for _ in range(options.nsteps):
            vec = apply_u(vec, options.dt)
        probs = jnp.square(jnp.abs(vec))
        return probs

    if (proc_id := jax.process_index()) == 0 and options.xprof:
        with jax.profiler.trace(options.xprof):
            probs = run()
    else:
        probs = run()

    filename = f'trotter_{options.nrow}x{options.ncol}_l{options.plaquette_energy:.2f}'
    filename += f'_dt{options.dt:.3f}_s{options.nsteps}.h5'
    if proc_id == 0:
        with h5py.File(str(Path(options.out) / filename), 'w', libver='latest') as out:
            out.create_dataset('probs', shape=probs.shape[0], dtype=np.float64)
    else:
        MPI.COMM_WORLD.recv(source=proc_id - 1, tag=11)

    LOG.info('Writing from process %d on indices %s', proc_id,
             list(shard.index for shard in probs.addressable_shards))

    with h5py.File(str(Path(options.out) / filename), 'a', libver='latest') as out:
        for shard in probs.addressable_shards:
            out['probs'][shard.index] = shard.data

    if proc_id < jax.process_count() - 1:
        MPI.COMM_WORLD.send(1, dest=proc_id + 1, tag=11)
