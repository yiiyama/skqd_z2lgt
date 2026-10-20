import os
import logging
import sys
from pathlib import Path
import numpy as np
import h5py
import jax
from skqd_z2lgt.perturbative_sqd import make_pt_core
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from perturbation import make_perturbation_series
from square_ising import make_primal_graph


def get_perturbation_coeffs(primal_graph, max_order):
    nvx = max(datum[0] for datum in primal_graph.nodes()) + 1
    nvy = max(datum[1] for datum in primal_graph.nodes()) + 1
    nx = nvx - 1
    ny = nvy - 1
    nplaq = nx * ny
    nhlink = nx * (ny + 1)
    nvlink = (nx + 1) * ny
    nlink = nhlink + nvlink

    pt_states = make_pt_core(nplaq, max_order, False, False)[1:]

    encoding_mat = np.zeros((nplaq, nlink), dtype=np.uint8)
    for iy in range(ny):
        for ix in range(nx):
            ip = iy * nx + ix
            iltop = iy * nx + ix
            ilbottom = (iy + 1) * nx + ix
            illeft = nhlink + iy * (nx + 1) + ix
            ilright = nhlink + iy * (nx + 1) + ix + 1
            encoding_mat[ip, [iltop, ilbottom, illeft, ilright]] = 1

    base_link_state = np.zeros(nlink, dtype=np.uint8)
    base_link_state[primal_graph.filter_edges(lambda x: x > 0.)] = 1
    lengths = np.sum((pt_states @ encoding_mat + base_link_state) & 1, axis=1) - np.sum(base_link_state)

    pt_ecoeffs, pt_scoeffs = make_perturbation_series(nplaq, max_order, np.packbits(pt_states, axis=1), lengths)
    pt_ecoeffs = np.concatenate([[2 * np.sum(base_link_state, dtype=int) - nlink], pt_ecoeffs])
    return pt_ecoeffs, pt_scoeffs


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('nx', type=int)
    parser.add_argument('ny', type=int)
    parser.add_argument('--charges', nargs=2)
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--out', default='.')
    parser.add_argument('--gpus')
    options = parser.parse_args()

    if options.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus
    jax.config.update('jax_enable_x64', True)
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger()

    nx = options.nx
    ny = options.ny
    nplaq = nx * ny
    
    primal_graph = make_primal_graph(nx, ny, options.charges)
    pt_ecoeffs, pt_scoeffs = get_perturbation_coeffs(primal_graph, options.order)
    with h5py.File(options.out, 'w') as out:
        out.create_dataset('order', data=options.order)
        out.create_dataset('pt_ecoeffs', data=pt_ecoeffs)
        out.create_dataset('pt_scoeffs', data=pt_scoeffs)
