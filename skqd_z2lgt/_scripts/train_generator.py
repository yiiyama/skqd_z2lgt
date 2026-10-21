import os
import sys
import argparse
import logging
from itertools import product
import jax
from skqd_z2lgt.orchestration.open_output import open_output
from skqd_z2lgt.orchestration.train_generator import train_single_model


parser = argparse.ArgumentParser()
parser.add_argument('pkgpath')
parser.add_argument('--mpi', action='store_true')
parser.add_argument('--idt')
parser.add_argument('--step')
parser.add_argument('--gpu')
parser.add_argument('--log-level', default='INFO')
args = parser.parse_args()

logging.basicConfig(level=getattr(logging, args.log_level.upper()), stream=sys.stdout)
logging.getLogger('jax').setLevel(logging.WARNING)

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update('jax_enable_x64', True)

parameters = open_output(args.pkgpath)

if args.idt:
    idts = list(map(int, args.idt.split(',')))
    steps = list(map(int, args.step.split(',')))
    if len(idts) != len(steps):
        raise ValueError('Lengths of idt and step lists do not match')
    task_specs = list(zip(idts, steps))
else:
    idts = list(range(len(parameters.circuit.dts)))
    steps = parameters.circuit.sampled_steps
    task_specs = list(product(idts, steps))

if args.mpi:
    from mpi4py import MPI  # pylint: disable=no-name-in-module
    comm = MPI.COMM_WORLD
    task_specs = [task_specs[comm.Get_rank()]]

if args.gpu and ',' in args.gpu:
    from concurrent.futures import ThreadPoolExecutor

    def train_in_thread(idev, idt, step):
        with jax.default_device(jax.devices()[idev]):
            train_single_model(parameters, idt, step)

    ndev = len(args.gpu.split(','))
    futures = []
    with ThreadPoolExecutor(max_workers=ndev) as executor:
        for itask, task_spec in enumerate(task_specs):
            idev = itask % ndev
            futures.append(executor.submit(train_in_thread, idev, *task_spec))
    for future in futures:
        future.result()

else:
    for task_spec in task_specs:
        train_single_model(parameters, *task_spec)
