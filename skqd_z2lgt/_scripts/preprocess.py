"""Convert raw quantum samples to syndrome & plaquette data."""
import sys
import argparse
import logging
from itertools import product
from skqd_z2lgt.orchestration.open_output import open_output
from skqd_z2lgt.orchestration.preprocess import preprocess_single_array

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
