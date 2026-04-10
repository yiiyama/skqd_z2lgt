import os
import sys
import logging
from pathlib import Path
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.ground_locg import ground_locg
from skqd_z2lgt.sqd import get_hamiltonian_arrays, sqd
from skqd_z2lgt.extensions import _plaquette_excitations
sys.path.append('/home/iiyama/src/skqd_z2lgt/lib')
from ising_hamiltonian import make_apply_h

jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.INFO)

out_dir = Path('/data/iiyama/2dz2/buildup')

nrow, ncol = map(int, sys.argv[1:3])
plaquette_energy = float(sys.argv[3])
if len(sys.argv) == 5:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[4]

lattice = TriangularZ2Lattice((nrow, ncol))
nsites = (ncol - 1) // 2 + 1 + (nrow // 2 + 1) % 2
vfirst = (lattice.num_vertices - nsites) // 2
links = []
for ivtx in range(vfirst, vfirst + nsites - 1):
    links.append(lattice.graph.edge_indices_from_endpoints(ivtx, ivtx + 1)[0])
base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
base_link_state[::-1][links] = 1
dual_lattice = lattice.plaquette_dual(base_link_state)
hamiltonian = dual_lattice.make_hamiltonian(plaquette_energy)
nq = hamiltonian.num_qubits

with h5py.File(out_dir / f'pt_states_{nq}plaq.h5', 'r', libver='latest') as source:
    states = np.concatenate([v[()] for v in source.values()], axis=0)

eigval, eigvec, states_u = sqd(hamiltonian, states, states_size=states.shape[0], return_states=True)

with h5py.File(out_dir / f'{nrow}x{ncol}_l{plaquette_energy:.1f}_pt.h5', 'w', libver='latest') as out:
    out.create_dataset('states', data=states_u)
    out.create_dataset('eigval', data=eigval)
    out.create_dataset('eigvec', data=eigvec)
