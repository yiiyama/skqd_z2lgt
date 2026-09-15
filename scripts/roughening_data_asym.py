import os
import sys
import numpy as np
import h5py
import jax
from rqutils.ground_locg import ground_locg
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
sys.path.append('/home/iiyama/src/skqd_z2lgt/lib')
from ising_hamiltonian import make_apply_h

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
jax.config.update('jax_enable_x64', True)

lattice = TriangularZ2Lattice('''
     * * *
  * * * *
 * * * *
  * * * *
 * * * *
* * *
''')

print('Flux string')
mus = np.linspace(0.1, 4.1, 20)
links = np.array([4, 14, 23, 31, 41])

ndim = 2 ** lattice.num_active_plaquettes
eigvals = np.empty(mus.shape + (len(links) + 1,))
eigvecs = np.empty(mus.shape + (len(links) + 1, ndim))
for imu, mu in enumerate(mus):
    for dist in range(len(links) + 1):
        print('mu =', mu, 'length', dist)
        link_state = np.zeros(lattice.num_links, dtype=np.uint8)
        link_state[::-1][links[:dist]] = 1
        ham = lattice.plaquette_dual(link_state).make_hamiltonian(mu)
        apply_h = make_apply_h(ham)
        eigval, eigvec, _ = ground_locg(apply_h, 0, vspace=(ndim, np.float64))
        eigvals[imu, dist] = eigval
        eigvecs[imu, dist] = eigvec

with h5py.File('/data/iiyama/2dz2/roughening/flux.h5', 'w') as out:
    out.create_dataset('mus', data=mus)
    out.create_dataset('eigvals', data=eigvals)
    out.create_dataset('eigvecs', data=eigvecs)

print('Excited states')
mus = np.linspace(0.1, 4.1, 20)
ndim = 2 ** lattice.num_active_plaquettes
eigvals = np.empty(mus.shape)
eigvecs = np.empty(mus.shape + (ndim,))
for imu, mu in enumerate(mus):
    print('mu =', mu)
    ham = lattice.plaquette_dual().make_hamiltonian(mu)
    apply_h = make_apply_h(ham)
    with h5py.File('/data/iiyama/2dz2/roughening/flux.h5') as source:
        ground_state = source['eigvecs'][imu, 0]
    xinit = np.zeros(ndim)
    xinit[1 << np.arange(lattice.num_active_plaquettes)] = 1. / np.sqrt(lattice.num_active_plaquettes)
    eigval, eigvec, _ = ground_locg(apply_h, xinit, orth=ground_state)
    eigvals[imu] = eigval
    eigvecs[imu] = eigvec

with h5py.File('/data/iiyama/2dz2/roughening/excited.h5', 'w') as out:
    out.create_dataset('mus', data=mus)
    out.create_dataset('eigvals', data=eigvals)
    out.create_dataset('eigvecs', data=eigvecs)

# print('Magnetic charge')
# mus = np.linspace(0.1, 4.1, 20)

# lattice.activate_plaquette(11, False)
# dual_vacuum = lattice.plaquette_dual()
# link_state = np.zeros(lattice.num_links, dtype=np.uint8)
# link_state[::-1][[19, 21, 23]] = 1
# dual_charged = lattice.plaquette_dual(link_state)

# ndim = 2 ** lattice.num_active_plaquettes
# eigvals = np.empty(mus.shape + (2,))
# eigvecs = np.empty(mus.shape + (2, ndim))
# for imu, mu in enumerate(mus):
#     print('mu =', mu, 'vacuum')
#     ham = dual_vacuum.make_hamiltonian(mu)
#     apply_h = make_apply_h(ham)
#     eigval, eigvec, _ = ground_locg(apply_h, 0, vspace=(ndim, np.float64))
#     eigvals[imu, 0] = eigval
#     eigvecs[imu, 0] = eigvec

#     print('mu =', mu, 'charged')
#     ham = dual_charged.make_hamiltonian(mu)
#     apply_h = make_apply_h(ham)
#     eigval, eigvec, _ = ground_locg(apply_h, 0, vspace=(ndim, np.float64))
#     eigvals[imu, 1] = eigval
#     eigvecs[imu, 1] = eigvec

# with h5py.File('/data/iiyama/2dz2/roughening/monopole.h5', 'w') as out:
#     out.create_dataset('mus', data=mus)
#     out.create_dataset('eigvals', data=eigvals)
#     out.create_dataset('eigvecs', data=eigvecs)
