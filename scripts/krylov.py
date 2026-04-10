import sys
from pathlib import Path
import logging
from functools import partial
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard
sys.path.append(str(Path(__file__).parent / 'lib'))
from unitary_krylov import make_hvec, integrate  # pylint: disable=wrong-import-order

# Links to excite for equator-slicing charge positioning
excited_links = {
    (4, 1): [5],
    (4, 2): [8],
    (4, 3): [11, 13],
    (4, 4): [14, 16],
    (4, 5): [17, 19, 22],
    (4, 6): [20, 22, 25],
    (4, 7): [23, 25, 28, 31],
    (6, 1): [],
    (6, 2): [12],
    (6, 3): [17],
    (6, 4): [21, 24],
    (6, 5): [26, 29],
    (8, 1): [10],
    (8, 2): [16],
    (8, 3): [22, 24]
}

TEMPLATE = '/work/gp14/p14000/data/krylov/{}_{}_l{:.2f}.h5'
PSI_TEMPLATE = '/work/gp14/p14000/data/krylov/{}_{}_l{:.2f}_states.h5'

LOG = logging.getLogger(__name__)


@partial(jax.jit, static_argnums=[0, 1])
def compute_ground_state(hvec, nplaq, alpha):
    xmat = jax.nn.one_hot(0, 2 ** nplaq, dtype=np.complex128)[:, None]
    # pylint: disable-next=unbalanced-tuple-unpacking
    ground_state = lobpcg_standard(lambda x: -hvec(x.T).T, xmat)[1][:, 0]
    probs = jnp.square(jnp.abs(ground_state))
    sort_idx = jnp.argsort(probs, descending=True)
    sorted_probs = jnp.sort(probs, descending=True)
    cum_probs = jnp.cumsum(sorted_probs)
    largel = jnp.searchsorted(cum_probs, alpha)
    return sort_idx, sorted_probs, largel


@partial(jax.jit, static_argnums=[0, 1], static_argnames=['return_final_state'])
def compute_krylov_vectors(hvec, nplaq, tpoints, idx, psi0=None, return_final_state=False):
    psis = integrate(hvec, nplaq, tpoints, psi0=psi0)
    probs = jnp.square(jnp.abs(psis[1:, idx]))
    if return_final_state:
        return probs, psis[-1]
    return probs


def get_hvec(config, plaquette_energy):
    lattice = TriangularZ2Lattice(config)
    nplaq = lattice.num_plaquettes

    base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
    base_link_state[::-1][excited_links[config]] = 1
    dual_lattice = lattice.plaquette_dual(base_link_state)
    hamiltonian = dual_lattice.make_hamiltonian(plaquette_energy)

    hvec = make_hvec(hamiltonian, dtype=np.complex128)
    return hvec, nplaq


def main(config, plaquette_energy, max_steps, idt=None, step=None):
    dtvals = np.linspace(np.pi / 15., np.pi / 5., 11)
    ndt = dtvals.shape[0]

    if idt is None:
        idts = list(range(dtvals.shape[0]))
    else:
        idts = [idt]

    hvec_op, num_plaqs = get_hvec(config, plaquette_energy)

    with h5py.File(TEMPLATE.format(*(config + (plaquette_energy,))), 'a', libver='latest') as out:
        if 'dtvals' not in out:
            out.create_dataset('dtvals', data=dtvals)

        if 'largel' in out:
            largel = out['largel'][()]
            sort_idx = out['sort_idx'][()]
        else:
            sort_idx, sorted_probs, largel = compute_ground_state(hvec_op, num_plaqs, 0.999)
            largel = int(largel)
            sort_idx = np.array(sort_idx[:largel])
            out.create_dataset('ground_state', data=sorted_probs[:largel])
            out.create_dataset('largel', data=largel)
            out.create_dataset('sort_idx', data=sort_idx)

        if (dataset := out.get('krylov')) is None:
            dataset = out.create_dataset('krylov', shape=(ndt, max_steps, largel), dtype=np.float64)

        for idt in idts:
            dtval = dtvals[idt]
            LOG.info('dt %d: %.2f', idt, dtval)
            if step is None:
                tpoints = np.linspace(0., max_steps * dtval, max_steps + 1)
                krylov_probs = compute_krylov_vectors(hvec_op, num_plaqs, tpoints, sort_idx)
                LOG.info('Got krylov probs (shape %s dtype %s)', krylov_probs.shape,
                         krylov_probs.dtype)
                dataset[idt] = compute_krylov_vectors(hvec_op, num_plaqs, tpoints, sort_idx)
            else:
                LOG.info('Step %d', step)
                psi_path = PSI_TEMPLATE.format(*(config + (plaquette_energy,)))
                if step == 1:
                    psi_i = None
                else:
                    LOG.info('Reading psi_i from file')
                    with h5py.File(psi_path, 'r', libver='latest') as psi_file:
                        if psi_file['step'][idt] != step - 1:
                            raise RuntimeError('Need psi from the previous step')
                        psi_i = psi_file['psi'][idt]

                tpoints = np.linspace(step - 1, step, 2) * dtval
                krylov_probs, psi_f = compute_krylov_vectors(hvec_op, num_plaqs, tpoints,
                                                             sort_idx, psi0=psi_i,
                                                             return_final_state=True)
                LOG.info('Got krylov probs (shape %s dtype %s)', krylov_probs.shape,
                         krylov_probs.dtype)
                LOG.info('|psi_f[0]|=%f', np.abs(psi_f[0]))
                dataset[idt, step - 1] = krylov_probs
                if step != max_steps:
                    with h5py.File(psi_path, 'a', libver='latest') as psi_file:
                        step_ds = psi_file.get('step')
                        if (psi_ds := psi_file.get('psi')) is None:
                            step_ds = psi_file.create_dataset('step', shape=(ndt), dtype=np.int32)
                            psi_ds = psi_file.create_dataset('psi', shape=(ndt, 2 ** num_plaqs),
                                                             dtype=np.complex128)
                        step_ds[idt] = step
                        psi_ds[idt] = psi_f


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, nargs=2, required=True)
    parser.add_argument('--plaquette-energy', type=float, required=True)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--idt', type=int)
    parser.add_argument('--step', type=int)
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    jax.config.update('jax_enable_x64', True)

    main(tuple(options.config), options.plaquette_energy, options.max_steps, options.idt,
         options.step)
