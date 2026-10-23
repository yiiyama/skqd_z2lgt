"""Compute the approximate ground state through DMRG."""
import os
import tempfile
import logging
from typing import Optional
from skqd_z2lgt.ising_dmrg import ising_dmrg  #, get_mps_probs
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.common import make_dual_hamiltonian
from skqd_z2lgt.orchestration.io import save_dmrg


def run_dmrg_and_sampling(parameters: Parameters, logger: Optional[logging.Logger] = None):
    with tempfile.NamedTemporaryFile() as tfile:
        filename = tfile.name

    hamiltonian = make_dual_hamiltonian(parameters)
    dmrg_params = parameters.dmrg
    julia_bin = 'julia'
    if dmrg_params.julia_sysimage:
        julia_bin = ['julia', '--sysimage', dmrg_params.julia_sysimage]

    energy = ising_dmrg(hamiltonian, filename=filename, nsweeps=dmrg_params.nsweeps,
                        maxdim=dmrg_params.maxdim, cutoff=dmrg_params.cutoff, julia_bin=julia_bin)
    # states, probs = get_mps_probs(filename, num_samples=dp.num_samples,
    #                               num_threads=os.cpu_count(), julia_bin=julia_bin)
    os.unlink(filename)
    # return energy, states, probs
    save_dmrg(parameters, energy)
    return energy


if __name__ == '__main__':
    import sys
    import argparse
    from skqd_z2lgt.orchestration.open_output import open_output

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()), stream=sys.stdout)
    run_dmrg_and_sampling(open_output(options.pkgpath), logging.getLogger('dmrg'))
