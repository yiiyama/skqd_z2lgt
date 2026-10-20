import sys
import logging
import math
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import h5py
import jax
from jax.sharding import PartitionSpec, get_abstract_mesh
import jax.numpy as jnp
from rqutils.ground_locg import ground_locg
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from ising_hamiltonian import make_apply_h_from_coeffs
from square_ising import make_primal_graph
from distributed import qubit_sharding, save_multiprocess_array
from kde import gaussian_kde


def make_apply_h(primal_graph):
    nvx = max(datum[0] for datum in primal_graph.nodes()) + 1
    nvy = max(datum[1] for datum in primal_graph.nodes()) + 1
    nx = nvx - 1
    ny = nvy - 1
    nplaq = nx * ny
    
    z_coeffs = defaultdict(float)
    zz_coeffs = defaultdict(float)
    for v1, v2, weight in primal_graph.edge_index_map().values():
        x1, y1 = primal_graph[v1]
        x2, y2 = primal_graph[v2]
        if x1 == x2:
            # Vertical primal link
            if x1 == 0:
                z_coeffs[y1 * nx] += weight
            elif x1 == nvx - 1:
                z_coeffs[(y1 + 1) * nx - 1] += weight
            else:
                zz_coeffs[(y1 * nx + x1 - 1, y1 * nx + x1)] += weight
        else:
            # Horizontal primal link
            if y1 == 0:
                z_coeffs[x1] += weight
            elif y1 == nvy - 1:
                z_coeffs[(y1 - 1) * nx + x1] += weight
            else:
                zz_coeffs[((y1 - 1) * nx + x1, y1 * nx + x1)] += weight

    return make_apply_h_from_coeffs(nplaq, z_coeffs, zz_coeffs)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('nx', type=int)
    parser.add_argument('ny', type=int)
    parser.add_argument('--charges', nargs=2)
    parser.add_argument('--mu')
    parser.add_argument('--out', default='.')
    parser.add_argument('--save-eigvec', action='store_true')
    parser.add_argument('--gpus')
    parser.add_argument('--mpi', nargs='?', const=True)
    options = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger()

    NKDE = 100

    def loginfo(msg, *args):
        if jax.process_index() == 0:
            LOG.info(msg, *args)

    if options.mu:
        if ',' in options.mu:
            mu_low, mu_high, nmu = options.mu.split(',')
            mus = np.linspace(float(mu_low), float(mu_high), int(nmu))
        else:
            mus = np.array([float(options.mu)])
    else:
        mus = np.linspace(1.5, 4.5, 21)

    sharding = qubit_sharding(options.mpi, options.gpus)

    # Number of plaquettes
    nx = options.nx
    ny = options.ny
    nplaq = nx * ny

    primal_graph = make_primal_graph(nx, ny, options.charges)
    apply_h = make_apply_h(primal_graph)
    
    if jax.process_index() == 0:
        with h5py.File(options.out, 'w') as out:
            out.create_dataset('mus', data=mus)
            out.create_dataset('eigvals', shape=mus.shape, dtype=np.float64)
            out.create_dataset('pkdes', shape=mus.shape + (nplaq + 1, NKDE), dtype=np.float64)
            out.create_dataset('kde_coords', shape=mus.shape + (nplaq + 1, NKDE), dtype=np.float64)
            out.create_dataset('ptots', shape=mus.shape + (nplaq + 1,), dtype=np.float64)
            if options.save_eigvec:
                out.create_dataset('eigvecs', shape=mus.shape + (2 ** nplaq,), dtype=np.float64)

    @jax.jit
    def get_ground_state(mu, weights):
        eigval, eigvec = ground_locg(apply_h, 0, args=(mu,), vspace=(2 ** nplaq, np.float64))[:2]
        probs = jnp.square(eigvec)
        pkdes = jnp.empty((nplaq + 1, NKDE))
        kde_coords = jnp.empty_like(pkdes)
        ptots = jnp.empty(nplaq + 1)

        for sidx, pidx in [(0, 0), (nplaq, 2 ** nplaq - 1)]:
            pkdes = pkdes.at[sidx].set(1. / NKDE)
            kde_coords = kde_coords.at[sidx].set(probs[pidx])
            ptots = ptots.at[sidx].set(probs[pidx])

        for weight in range(1, nplaq):
            mask = weights == weight
            coords, kde = gaussian_kde(probs, 100, scale='geom', mask=mask, batch_size=2)
            pkdes = pkdes.at[weight].set(kde)
            kde_coords = kde_coords.at[weight].set(coords)
            ptots = ptots.at[weight].set(jnp.sum(probs * mask))

        result = (eigval, pkdes, kde_coords, ptots)
        if options.save_eigvec:
            result += (eigvec,)
        return result

    local_device = list(sharding.addressable_devices)[0]
    indices = sharding.addressable_devices_indices_map((2 ** nplaq,))[local_device][0]
    local_weights = np.bitwise_count(
        np.arange(indices.start, indices.stop, dtype=np.int64)
    ).astype(np.uint8)
    weights = jax.make_array_from_process_local_data(sharding, local_weights)

    for imu, mu in enumerate(mus):
        loginfo('Starting mu=%.2f', mu)
        start = time.time()
        result = get_ground_state(mu, weights)
        eigval, pkdes, kde_coords, ptots = result[:4]
        if options.save_eigvec:
            eigvec = result[4]
        loginfo('Done in %.2f seconds. eigval=%.3f', time.time() - start, eigval)
        if jax.process_index() == 0:
            with h5py.File(options.out, 'a') as out:
                out['eigvals'][imu] = eigval
                out['pkdes'][imu] = pkdes
                out['kde_coords'][imu] = kde_coords
                out['ptots'][imu] = ptots

        if options.save_eigvec:
            if options.mpi:
                save_multiprocess_array(options.out, 'eigvecs', eigvec, outer_index=(imu,))
            else:
                with h5py.File(options.out, 'a') as out:
                    out['eigvecs'][imu] = eigvec
