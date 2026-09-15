import os
import sys
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from rqutils.ground_locg import ground_locg
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
sys.path.append('/home/iiyama/src/skqd_z2lgt/lib')
from ising_hamiltonian import make_apply_h

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
jax.config.update('jax_enable_x64', True)

lattice = TriangularZ2Lattice('''
 * * * * * * * *
* * * * * * * * *
 * * * * * * * *
''')

print('Flux string')
mus = np.linspace(0.1, 2.6, 6)
links = [[]]
# links.append(links[-1] + [28])
# links.append(links[-1] + [25, 31])
# links.append(links[-1] + [22, 34])
# links.append(links[-1] + [20, 37])
links.append(links[-1] + [31, 34])
links.append(links[-1] + [28, 37])
links.append(links[-1] + [25, 40])
links.append(links[-1] + [23, 43])

nplaq = lattice.num_active_plaquettes
ndim = 2 ** nplaq
nlink = lattice.num_links
eigvals = np.empty(mus.shape + (len(links) + 1,))
gaps = np.empty_like(mus)
fields = np.empty(mus.shape + (len(links) + 1, nlink))

lattice_plaquettes = np.sum(lattice.pl_matrix << np.arange(nplaq)[::-1, None], axis=0)

@jax.jit(static_argnames=['apply_h'])
def compute_field(apply_h):
    eigval, eigvec, _ = ground_locg(apply_h, 0, vspace=(ndim, np.float64))
    # plaq_states = ((jnp.arange(ndim)[:, None] >> jnp.arange(nplaq)[None, ::-1]) & 1).astype(np.uint8)
    # link_states = ((plaq_states @ lattice.pl_matrix) & 1).astype(np.uint8)
    plaq_states = jnp.arange(ndim)
    link_states = jnp.bitwise_count(jnp.bitwise_and(plaq_states[:, None], lattice_plaquettes[None, :])).astype(np.uint8) & 1
    probs = jnp.square(eigvec)
    field_expval = probs @ link_states
    return eigval, field_expval

@jax.jit(static_argnames=['apply_h'])
def compute_field_and_gap(apply_h):
    eigval, eigvec, _ = ground_locg(apply_h, 0, vspace=(ndim, np.float64))
    xinit = (jnp.bitwise_count(jnp.arange(ndim)) == 1).astype(np.float64) / jnp.sqrt(nplaq)
    eigval_exc = ground_locg(apply_h, xinit, orth=eigvec)[0]
    plaq_states = ((jnp.arange(ndim)[:, None] >> jnp.arange(nplaq)[None, ::-1]) & 1).astype(np.uint8)
    link_states = ((plaq_states @ lattice.pl_matrix) & 1).astype(np.uint8)
    probs = jnp.square(eigvec)
    field_expval = probs @ link_states
    return eigval, field_expval, eigval_exc - eigval

for imu, mu in enumerate(mus):
    for idist, ln in enumerate(links):
        print('mu =', mu, 'length', len(ln))
        link_state = np.zeros(lattice.num_links, dtype=np.uint8)
        link_state[::-1][ln] = 1
        ham = lattice.plaquette_dual(link_state).make_hamiltonian(mu)
        apply_h = make_apply_h(ham)
        if idist == 0:
            eigval, field_expval, gap = compute_field_and_gap(apply_h)
            gaps[imu] = gap
        else:
            eigval, field_expval = compute_field(apply_h)
        eigvals[imu, idist] = eigval
        fields[imu, idist] = field_expval

with h5py.File('/data/iiyama/2dz2/roughening/flux_and_gap.h5', 'w') as out:
    out.create_dataset('mus', data=mus)
    out.create_dataset('eigvals', data=eigvals)
    out.create_dataset('fields', data=fields)
    out.create_dataset('gaps', data=gaps)

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
