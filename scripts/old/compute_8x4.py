"""Compute the 8x4 eigenvector."""
import sys
import math
import logging
import time
import pathlib
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard
sys.path.append(str(pathlib.Path(__file__).parents[1] / 'lib'))
from unitary_krylov import make_hvec


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)
    jax.config.update('jax_enable_x64', True)

    lattice = TriangularZ2Lattice('''
    * * *
     * * *
    * * *
     * * *
    * * *
     * * *
    * * *
     * * *
    * * *
    ''')
    equator_links = [28, 30]
    # lattice = TriangularZ2Lattice('''
    # * *
    #  *
    # * *
    #  *
    # * *
    #  *
    # * *
    #  *
    # * *
    # ''')
    # equator_links = [10]

    nplaq = lattice.num_plaquettes
    filename = '/work/gp14/p14000/data/8x4_centercharge_ground_state.h5'

    base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
    base_link_state[::-1][equator_links] = 1
    dual_lattice = lattice.plaquette_dual(base_link_state)
    hamiltonian = dual_lattice.make_hamiltonian(1.)
    hvec = make_hvec(hamiltonian, variable=True, dtype=np.float64)

    lambdas = np.linspace(0., 4., 21)
    eigvals = np.empty_like(lambdas)
    s2s = np.empty_like(lambdas, dtype=np.int64)
    s3s = np.empty_like(s2s)
    s4s = np.empty_like(s2s)
    minimas = np.empty(lambdas.shape + (nplaq + 1,), dtype=np.float64)
    maximas = np.empty_like(minimas)
    medians = np.empty_like(minimas)
    means = np.empty_like(minimas)

    @jax.jit
    def compute(plaq_energy):
        xmat = jax.nn.one_hot(0, 2 ** nplaq, dtype=np.float64)[:, None]
        # pylint: disable-next=unbalanced-tuple-unpacking
        vals, vecs, _ = lobpcg_standard(lambda x: -hvec(x.T, plaq_energy).T, xmat)
        probs = jnp.square(jnp.abs(vecs[:, 0]))
        sort_idx = jnp.argsort(probs, stable=False, descending=True)
        probs = probs[sort_idx]
        cum_probs = jnp.cumsum(probs)
        s2 = jnp.searchsorted(cum_probs, 0.99)
        s3 = jnp.searchsorted(cum_probs, 0.999)
        s4 = jnp.searchsorted(cum_probs, 0.9999)
        minima = jnp.empty(nplaq + 1, dtype=np.float64)
        maxima = jnp.empty(nplaq + 1, dtype=np.float64)
        median = jnp.empty(nplaq + 1, dtype=np.float64)
        mean = jnp.empty(nplaq + 1, dtype=np.float64)
        for iex in range(nplaq + 1):
            idx = jnp.nonzero(jnp.bitwise_count(sort_idx) == iex, size=math.comb(nplaq, iex))
            ex_probs = probs[idx]
            minima = minima.at[iex].set(ex_probs[0])
            maxima = maxima.at[iex].set(ex_probs[-1])
            median = median.at[iex].set(jnp.quantile(ex_probs, 0.5))
            mean = mean.at[iex].set(jnp.mean(ex_probs))

        return -vals[0], (s2, s3, s4, minima, maxima, median, mean)

    for ilmd, plaquette_energy in enumerate(lambdas):
        LOG.info('Compute: %f', plaquette_energy)
        start = time.time()
        eigval, profile = compute(plaquette_energy)
        LOG.info('Done in %f s.', time.time() - start)
        eigvals[ilmd] = eigval
        s2s[ilmd] = profile[0]
        s3s[ilmd] = profile[1]
        s4s[ilmd] = profile[2]
        minimas[ilmd] = profile[3]
        maximas[ilmd] = profile[4]
        medians[ilmd] = profile[5]
        means[ilmd] = profile[6]

    with h5py.File(filename, 'w', libver='latest') as out:
        out.create_dataset('lambdas', data=lambdas)
        out.create_dataset('eigvals', data=eigvals)
        out.create_dataset('s2s', data=s2s)
        out.create_dataset('s3s', data=s3s)
        out.create_dataset('s4s', data=s4s)
        out.create_dataset('minimas', data=minimas)
        out.create_dataset('maximas', data=maximas)
        out.create_dataset('medians', data=medians)
        out.create_dataset('means', data=means)
