"""Perform SQD."""
import os
import sys
import logging
import argparse
import jax
from jax.sharding import AxisType
from skqd_z2lgt.orchestration.open_output import open_output
from skqd_z2lgt.orchestration.diagonalize import diagonalize_in_mode


parser = argparse.ArgumentParser()
parser.add_argument('pkgpath')
parser.add_argument('mode')
parser.add_argument('--mpi', action='store_true')
parser.add_argument('--gpus', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--log-level', default='INFO')
options = parser.parse_args()

logging.basicConfig(level=getattr(logging, options.log_level.upper()), stream=sys.stdout)
logging.getLogger('jax').setLevel(logging.WARNING)

if options.gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update('jax_enable_x64', True)

parameters = open_output(options.pkgpath)

if options.mpi:
    jax.distributed.initialize(cluster_detection_method='mpi4py')

if (ndev := jax.device_count()) > 1:
    jax.set_mesh(jax.make_mesh((ndev,), ('x',), (AxisType.Explicit,)))

diagonalize_in_mode(parameters, options.mode)
