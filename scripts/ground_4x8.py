import sys
from pathlib import Path
import logging
import numpy as np
import h5py
import jax
from jax.sharding import PartitionSpec, AxisType, NamedSharding
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.ground_locg import ground_locg
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from ising_hamiltonian import make_apply_h

jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger()

mesh_qubit = jax.make_mesh((2, 2, 2), ('X', 'Y', 'Z'),
                           axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit))
sh_single = NamedSharding(mesh_qubit, PartitionSpec(('X', 'Y', 'Z')))
sh_qubit = NamedSharding(mesh_qubit, PartitionSpec('X', 'Y', 'Z'))

lattice = TriangularZ2Lattice((4, 8))
base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
base_link_state[::-1][[26, 28, 31, 34]] = 1
# base_link_state[::-1][[10, 13, 16]] = 1
# base_link_state[::-1][[20, 22, 25]] = 1
dual_lattice = lattice.plaquette_dual(base_link_state)
nplaq = lattice.num_plaquettes
hamiltonian = dual_lattice.make_hamiltonian(0.8)

apply_h = make_apply_h(hamiltonian)

eigval, eigvec, iter = ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64), sharding=sh_single)
print(iter)
with h5py.File('/data/iiyama/ground_4x8.h5', 'w', libver='latest') as out:
    out.create_dataset('eigval', data=eigval)
    out.create_dataset('eigvec', data=eigvec)
# LOG.info('compiling')
# print(ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64), sharding=sh_single)[0])
# LOG.info('tracing')
# with jax.profiler.trace('/tmp/ground_4x8'):
#     ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64), sharding=sh_single)
# LOG.info('validation')
# print(jnp.linalg.eigvalsh(hamiltonian.to_matrix())[0])
