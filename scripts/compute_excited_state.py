import os
import sys
import logging
from pathlib import Path
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
sys.path.append('/home/iiyama/src/skqd_z2lgt/lib')
from ising_hamiltonian import make_apply_h

nrow, ncol = map(int, sys.argv[1:3])
if len(sys.argv) == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]

jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.INFO)

nplaq = nrow * ncol

print(nrow, ncol)
lattice = TriangularZ2Lattice((nrow, ncol))
nsites = (ncol - 1) // 2 + 1 + (nrow // 2 + 1) % 2
vfirst = (lattice.num_vertices - nsites) // 2
links = []
for ivtx in range(vfirst, vfirst + nsites - 1):
    links.append(lattice.graph.edge_indices_from_endpoints(ivtx, ivtx + 1)[0])
base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
base_link_state[::-1][links] = 1
dual_lattice = lattice.plaquette_dual(base_link_state)
hamiltonian = dual_lattice.make_hamiltonian(0.8)
apply_h = jax.jit(jax.vmap(make_apply_h(-hamiltonian), in_axes=(1,), out_axes=1))
xmat = np.zeros((2 ** nplaq, 2))
xmat[0, 0] = 1.
xmat[1 << np.arange(nplaq), 1] = 1. / np.sqrt(nplaq)
eigvals, eigvecs = lobpcg_standard(apply_h, xmat)[:2]

out_dir = '/data/iiyama/2dz2/buildup/excited_states'
try:
    os.makedirs(out_dir)
except FileExistsError:
    pass

with h5py.File(Path(out_dir) / f'{nrow}_{ncol}.h5', 'w', libver='latest') as out:
    out.create_dataset('eigvals', data=-eigvals)
    out.create_dataset('eigvecs', data=eigvecs)
