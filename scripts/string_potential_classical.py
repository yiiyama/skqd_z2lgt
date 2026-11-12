"""Compute the energies for different charge configurations and coupling constants."""

import logging
from dataclasses import asdict
import json
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import SQDOptions, run_sqd
from skqd_z2lgt.perturbation import make_perturbation_series
from skqd_z2lgt.truncation_sqd import make_pt_core
from skqd_z2lgt.utils import minimum_weight_link_state
from skqd_z2lgt.dmrg_gpu.backend import xp
from skqd_z2lgt.dmrg_gpu.runner import solve, DEFAULT_SWEEPS
from skqd_z2lgt.dmrg_gpu.sweeps import Sweeps

LOG = logging.getLogger(__name__)


@jax.jit(static_argnames=['num_plaquettes', 'pt_order'])
def get_pt_ecoeffs(num_plaquettes, pt_order, plaq_states_p, pl_matrix, base_link_state):
    plaq_states = jnp.unpackbits(plaq_states_p, axis=1)[:, :lattice.num_plaquettes]
    base_length = jnp.sum(base_link_state, dtype=int)
    lengths = jnp.sum((plaq_states @ pl_matrix + base_link_state) & 1, axis=1, dtype=int)
    lengths -= base_length
    ecoeffs = make_perturbation_series(num_plaquettes, pt_order, plaq_states_p, lengths)[0]
    ecoeffs = jnp.concatenate([jnp.array([2 * base_length - base_link_state.shape[0]]), ecoeffs])
    return ecoeffs


def compute(lattice, charges, pt_order, mus):
    trunc_states_p = make_pt_core(lattice.num_plaquettes, pt_order, add_padding=False, packed=True)
    trunc_states_padded = make_pt_core(lattice.num_plaquettes, pt_order, add_padding=True,
                                       packed=True)

    link_state = minimum_weight_link_state(charges, [], lattice)
    dual = lattice.plaquette_dual(link_state)
    LOG.info('Computing perturbation series')
    ecoeffs = get_pt_ecoeffs(dual.num_plaquettes, pt_order, trunc_states_p[1:], lattice.pl_matrix,
                             link_state)
    pt_energies = np.sum(ecoeffs[None, :]
                          * np.power(mus[:, None], np.arange(0, pt_order + 1, 2)[None, :]),
                         axis=1)
    LOG.info('PT energies: %s', pt_energies)

    dmrg_energies = []
    tsd_energies = []

    for mu in mus:
        ## DMRG
        dconf = {
            'model': 'dual',
            'sweeps': dict(DEFAULT_SWEEPS),
            'penalty': 4.,
            'svd': 'svd',
            'ordering': 'snake',
            'energy_tol': None,
            'num_samples': 0,
            'seed': 0,
        }
        sweep_conf = dconf['sweeps']
        sweep_conf = {'table': sweep_conf} if isinstance(sweep_conf, str) else dict(sweep_conf)
        sweeps = Sweeps.from_config(sweep_conf)
        dconf['sweeps'] = sweeps.to_dict()
        backend = xp.configure('numpy', device=0, precision='double')
        model = 'dual'

        def progress(record, model=model):
            LOG.info(json.dumps(asdict(record)))

        dmrg_result = solve(
            lattice, mu, charges, model=model,
            charged_plaquettes=[], penalty=dconf['penalty'],
            sweeps=sweeps, ordering=dconf['ordering'], svd_method=dconf['svd'],
            energy_tol=dconf['energy_tol'], num_samples=dconf['num_samples'],
            seed=dconf['seed'], callback=progress,
            measure_variance=True,
            checkpoint=None,
            resume=False
        )
        LOG.info('DMRG energy %f', dmrg_result['energy'])
        dmrg_energies.append(dmrg_result['energy'])

        ## TSD
        hamiltonian = PauliSumXZ.from_paulisum(dual.make_hamiltonian(mu), force_real=True,
                                               add_padding=True)
        sqd_options = SQDOptions(return_eigvec=False)
        tsd_energy = run_sqd(hamiltonian, trunc_states_padded,
                             states_size=trunc_states_padded.shape[0], options=sqd_options)[0]
        tsd_energy = float(tsd_energy)
        LOG.info('TSD energy %f', tsd_energy)
        tsd_energies.append(tsd_energy)

    return pt_energies, np.array(dmrg_energies), np.array(tsd_energies)


if __name__ == '__main__':
    import os
    import re
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('lattice')
    parser.add_argument('ch1', type=int)
    parser.add_argument('ch2', type=int)
    parser.add_argument('--pt-order', type=int, default=6)
    parser.add_argument('--mu')
    parser.add_argument('--out')
    parser.add_argument('--gpu')
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    jax.config.update('jax_enable_x64', True)

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    if (matches := re.match('([0-9]+)x([0-9]+)', options.lattice)):
        lattice = TriangularZ2Lattice((int(matches.group(1)), int(matches.group(2))))
    else:
        raise ValueError(f'Invalid lattice specification: {options.lattice}')

    if options.mu:
        mu_low, mu_high, nmu = options.mu.split(',')
        mus = np.linspace(float(mu_low), float(mu_high), int(nmu))
    else:
        mus = np.linspace(0.1, 1.2, 12)

    charges = [options.ch1, options.ch2]
    
    pt_energies, dmrg_energies, tsd_energies = compute(lattice, charges, options.pt_order, mus)

    if (filename := options.out) is None:
        filename = f'{options.lattice}_{charges[0]}-{charges[1]}_pt{options.pt_order}.h5'
    with h5py.File(filename, 'w') as out:
        out.create_dataset('mus', data=mus)
        out.create_dataset('pt_order', data=options.pt_order)
        out.create_dataset('charges', data=charges)
        out.create_dataset('pt_energies', data=pt_energies)
        out.create_dataset('dmrg_energies', data=dmrg_energies)
        out.create_dataset('tsd_energies', data=tsd_energies)
