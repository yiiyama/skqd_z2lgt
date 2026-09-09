import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
import h5py
from rustworkx import adjacency_matrix
import jax
import jax.numpy as jnp
from jax.sharding import AxisType
from rqutils.ground_locg import ground_locg
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
sys.path.append('/home/iiyama/src/skqd_z2lgt/lib')
from ising_hamiltonian import make_apply_h
from face_area import compute_counts

jax.config.update('jax_enable_x64', True)
# jax.set_mesh(jax.make_mesh((2, 2), ('X', 'Y'), (AxisType.Explicit, AxisType.Explicit)))

nrow, ncol = 5, 5
monopole_idx = 11

lattice = TriangularZ2Lattice((nrow, ncol))
lattice.activate_plaquette(monopole_idx, False)
link_state = np.zeros(lattice.num_links, dtype=np.uint8)
link_state[::-1][lattice.plaquette_links(monopole_idx)] = 1
dual = lattice.plaquette_dual(link_state)

nactiv = lattice.num_active_plaquettes
ndim = 2 ** nactiv

pgraph = lattice.dual_graph.copy()
for idx in pgraph.node_indices():
    if not isinstance(pgraph[idx], int):
        pgraph.remove_node(idx)
amat = adjacency_matrix(pgraph).astype(np.uint8)
seeds = np.delete(amat[monopole_idx], monopole_idx)
bmat = np.delete(np.delete(amat, monopole_idx, axis=0), monopole_idx, axis=1)

counts = compute_counts(bmat, seeds)

mus = np.linspace(0.1, 1.6, 5)
eigvecs = np.empty(mus.shape + (ndim,))
areas = np.empty_like(mus)

for imu, mu in enumerate(mus):
    print('mu', mu)
    ham = dual.make_hamiltonian(mu)
    apply_h = make_apply_h(ham)
    eigvecs[imu] = ground_locg(apply_h, 0, vspace=(ndim, np.float64))[1]
    areas[imu] = counts @ jnp.square(eigvecs[imu])

with h5py.File(f'/data/iiyama/2dz2/monopole_area/{nrow}x{ncol}_{monopole_idx}.h5', 'w') as out:
    out.create_dataset('mus', data=mus)
    out.create_dataset('eigvecs', data=eigvecs)
    out.create_dataset('counts', data=counts)
    out.create_dataset('areas', data=areas)
