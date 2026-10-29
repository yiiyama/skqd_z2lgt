"""Subroutines to set up multi-GPU / multi-host mesh and shardings."""
import os
import string
import logging
import numpy as np
import h5py
import jax
from jax.sharding import AxisType, PartitionSpec, Mesh, NamedSharding
from jax.experimental.mesh_utils import create_hybrid_device_mesh

LOG = logging.getLogger(__name__)


def parse_gpus(gpus: str | list):
    if isinstance(gpus, list):
        return ','.join(f'{d}' for d in gpus)
    return gpus


def distributed_initialize(mpi: bool | str = False, gpus: str | list[int] = None):
    """Call distributed.initialize() or set CUDA_VISIBLE_DEVICES."""
    if mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        # Split the communicator by shared memory / node
        local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        # Get the local rank of the process within its node
        local_rank = local_comm.Get_rank()
        # Free the local communicator resources
        local_comm.Free()
        if gpus:
            visible_device = parse_gpus(gpus).split(',')[local_rank]
        else:
            # We assume one process per GPU is created
            visible_device = str(local_rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
        if mpi == 'local':
            comm = MPI.COMM_WORLD
            jax.distributed.initialize('localhost:10000', comm.Get_size(), comm.Get_rank())
        else:
            jax.distributed.initialize(cluster_detection_method="mpi4py")
        
        LOG.info('Process %d uses GPU %s', jax.process_index(), visible_device)
        if jax.process_index() == 0:
            LOG.info('All device slices: %s', [d.slice_index for d in jax.devices()])
    
    elif gpus:
        LOG.info('Parallelizing over %s', gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = parse_gpus(gpus)


def direct_sharding(mpi: bool | str = False, gpus: str | list = None, set_mesh: bool = True):
    """Return a NamedSharding over a mesh with shape (NGPU,)."""
    distributed_initialize(mpi, gpus)

    if (ndev := jax.device_count()) == 1:
        return None
    
    if mpi:
        unique_slices = set(getattr(d, 'slice_index', 0) for d in jax.devices())
        num_slices = len(unique_slices)
        ndev_per_slice = ndev // num_slices
        inner_mesh_shape = (1, ndev_per_slice)
        outer_mesh_shape = (num_slices, 1)
        axis_names = ('x', 'y')
        if jax.process_index() == 0:
            LOG.info('Inner mesh shape: %s, outer mesh shape: %s',
                     inner_mesh_shape, outer_mesh_shape)
        mesh = Mesh(create_hybrid_device_mesh(inner_mesh_shape, outer_mesh_shape),
                    axis_names=axis_names, axis_types=(AxisType.Explicit,) * 2)
    else:
        mesh = jax.make_mesh((ndev, 1), axis_names, axis_types=(AxisType.Explicit,) * 2)
    if set_mesh:
        jax.set_mesh(mesh)
    return NamedSharding(mesh, PartitionSpec(axis_names))



def qubit_sharding(mpi: bool | str = False, gpus: str | list = None, set_mesh: bool = True):
    """Return a NamedSharding over a mesh with shape (2,)*(log2 NGPU)."""
    distributed_initialize(mpi, gpus)

    if (ndev := jax.device_count()) == 1:
        return None
    
    nax = np.log2(ndev).astype(int)
    if 2 ** nax != ndev:
        raise ValueError('Invalid ndev')
    axis_names = tuple(string.ascii_lowercase[:nax])
    if mpi:
        unique_slices = set(getattr(d, 'slice_index', 0) for d in jax.devices())
        num_slices = len(unique_slices)
        ndev_per_slice = ndev // num_slices
        nax_per_slice = np.log2(ndev_per_slice).astype(int)
        if 2 ** nax_per_slice != ndev_per_slice:
            raise ValueError('Invalid ndev_per_slice')
        inner_mesh_shape = (1,) * (nax - nax_per_slice) + (2,) * nax_per_slice
        outer_mesh_shape = (2,) * (nax - nax_per_slice) + (1,) * nax_per_slice
        if jax.process_index() == 0:
            LOG.info('Inner mesh shape: %s, outer mesh shape: %s',
                     inner_mesh_shape, outer_mesh_shape)
        mesh = Mesh(create_hybrid_device_mesh(inner_mesh_shape, outer_mesh_shape),
                    axis_names=axis_names, axis_types=(AxisType.Explicit,) * nax)
    else:
        mesh = jax.make_mesh((2,) * nax, axis_names, axis_types=(AxisType.Explicit,) * nax)
    if set_mesh:
        jax.set_mesh(mesh)
    return NamedSharding(mesh, PartitionSpec(axis_names))


def save_multiprocess_array(
    out_path: str,
    dataset_name: str,
    array: jax.Array,
    outer_index: tuple[int, ...] = ()
):
    from mpi4py import MPI

    if (proc_id := jax.process_index()) != 0:
        MPI.COMM_WORLD.recv(source=proc_id - 1, tag=11)

    with h5py.File(out_path, 'a', libver='latest') as out:
        for shard in array.addressable_shards:
            out[dataset_name][outer_index + shard.index] = shard.data

    if proc_id < jax.process_count() - 1:
        MPI.COMM_WORLD.send(1, dest=proc_id + 1, tag=11)
