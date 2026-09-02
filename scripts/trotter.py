import os
import sys
from pathlib import Path
import logging
import string
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, AxisType, NamedSharding
from skqd_z2lgt.orchestration.common import make_dual_hamiltonian
from skqd_z2lgt.orchestration.open_output import open_output
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from ising_hamiltonian import make_apply_u


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--nsteps', type=int, required=True)
    parser.add_argument('--idt', type=int, default=0)
    parser.add_argument('--gpus')
    parser.add_argument('--xprof')
    parser.add_argument('--localmpi', action='store_true')
    options = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger()

    parameters = open_output(options.pkgpath)
    hamiltonian = make_dual_hamiltonian(parameters)
    apply_u = make_apply_u(hamiltonian, axis_type=AxisType.Explicit)
    dt = parameters.circuit.dts[options.idt]

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
        vec = (
            jax.lax.broadcasted_iota(np.int64, (2 ** lattice.num_plaquettes,), 0,
                                     out_sharding=sharding) == 0
        ).astype(np.complex128)
        for _ in range(options.nsteps):
            vec = apply_u(vec, dt)
        probs = jnp.square(jnp.abs(vec))
        return probs

    if (proc_id := jax.process_index()) == 0 and options.xprof:
        with jax.profiler.trace(options.xprof):
            probs = run()
    else:
        probs = run()

    path = Path(options.pkgpath) / f'trotter_idt{options.idt}_s{options.nsteps}.h5'
    if proc_id == 0:
        with h5py.File(str(path), 'w', libver='latest') as out:
            out.create_dataset('probs', shape=probs.shape[0], dtype=np.float64)
    else:
        MPI.COMM_WORLD.recv(source=proc_id - 1, tag=11)

    LOG.info('Writing from process %d on indices %s', proc_id,
             list(shard.index for shard in probs.addressable_shards))

    with h5py.File(str(path), 'a', libver='latest') as out:
        for shard in probs.addressable_shards:
            out['probs'][shard.index] = shard.data

    if proc_id < jax.process_count() - 1:
        MPI.COMM_WORLD.send(1, dest=proc_id + 1, tag=11)
