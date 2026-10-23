# pylint: disable=invalid-name
"""Preprocess raw data (link states with errors) and convert them to syndrome and plaquette data."""
import os
import logging
from typing import Optional
from skqd_z2lgt.error_correction import decode_all
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.common import make_dual_lattice
from skqd_z2lgt.orchestration.io import load_raw, save_reco


def preprocess_single_array(
    parameters: Parameters,
    etype: str,
    idt: int,
    step: int,
    logger: Optional[logging.Logger] = None
):
    """Preprocess one bit array parallelized over all cores and save the result."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Preprocessing %s, time interval %f, Trotter step %d',
                etype, parameters.circuit.dts[idt], step)

    dual_lattice = make_dual_lattice(parameters)
    batch_size = getattr(parameters.runtime, f'shots_{etype}') // (os.cpu_count() - 1)
    bit_array = load_raw(parameters, etype, idt, step)
    arrays = decode_all(bit_array, dual_lattice, batch_size=batch_size)
    save_reco(parameters, arrays, etype, idt, step, logger=logger)


if __name__ == '__main__':
    import sys
    import argparse
    from itertools import product
    from skqd_z2lgt.orchestration.open_output import open_output

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--etype')
    parser.add_argument('--idt')
    parser.add_argument('--step')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()), stream=sys.stdout)

    parameters = open_output(options.pkgpath)

    if options.etype:
        etypes = options.etype.split(',')
        idts = list(map(int, options.idt.split(',')))
        steps = list(map(int, options.step.split(',')))
        if not len(etypes) == len(idts) == len(steps):
            raise ValueError('Lengths of etype, idt, and step lists do not match')
        task_specs = list(zip(etypes, idts, steps))
    else:
        etypes = ['exp', 'ref']
        idts = list(range(len(parameters.circuit.dts)))
        steps = parameters.circuit.sampled_steps
        task_specs = list(product(etypes, idts, steps))

    if options.mpi:
        from mpi4py import MPI  # pylint: disable=no-name-in-module
        comm = MPI.COMM_WORLD
        task_specs = [task_specs[comm.Get_rank()]]

    for task_spec in task_specs:
        preprocess_single_array(parameters, *task_spec)
