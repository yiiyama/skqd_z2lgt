import os
import string
import logging
from mpi4py import MPI
import numpy as np
import jax
from jax.sharding import AxisType, PartitionSpec, Mesh
from jax.experimental.mesh_utils import create_hybrid_device_mesh

LOG = logging.getLogger(__name__)


def qubit_sharding(mpi=False, gpus=None):
    """Return a NamedSharding over a mesh with shape (2,)*(log2 NGPU)."""
    if mpi:
        # Below: attempt at using one process per GPU. Somehow failed with "Invalid ordinal" at CUDA
        # boot.

        # comm = MPI.COMM_WORLD
        # # Split the communicator by shared memory / node
        # local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        # # Get the local rank of the process within its node
        # local_rank = local_comm.Get_rank()
        # # Free the local communicator resources
        # local_comm.Free()
        # if options.gpus:
        #     visible_device = options.gpus.split(',')[local_rank]
        # else:
        #     # We assume one process per GPU is created
        #     visible_device = str(local_rank)
        # LOG.info('Using GPU "%s"', visible_device)
        # os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
        if mpi == 'local':
            comm = MPI.COMM_WORLD
            jax.distributed.initialize('localhost:10000', comm.Get_size(), comm.Get_rank())
        else:
            jax.distributed.initialize(cluster_detection_method="mpi4py")
        
        # LOG.info('Process %d uses GPU %s', jax.process_index(), visible_device)
        LOG.info('All device slices: %s', [d.slice_index for d in jax.devices()])

    elif gpus:
        LOG.info('Parallelizing over %s', gpus)
        if isinstance(gpus, list):
            gpus = ','.join(f'{d}' for d in gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    
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
        LOG.info('Inner mesh shape: %s, outer mesh shape: %s',
                    inner_mesh_shape, outer_mesh_shape)
        mesh = Mesh(create_hybrid_device_mesh(inner_mesh_shape, outer_mesh_shape),
                    axis_names=axis_names, axis_types=(AxisType.Explicit,) * nax)
    else:
        mesh = jax.make_mesh((2,) * nax, axis_names, axis_types=(AxisType.Explicit,) * nax)
    jax.set_mesh(mesh)
    return PartitionSpec(axis_names)
