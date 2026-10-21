"""Call ITensorMPS DMRG."""
import sys
import logging
import argparse
from skqd_z2lgt.orchestration.open_output import open_output
from skqd_z2lgt.orchestration.dmrg import run_dmrg_and_sampling

parser = argparse.ArgumentParser()
parser.add_argument('pkgpath')
parser.add_argument('--log-level', default='INFO')
options = parser.parse_args()

logging.basicConfig(level=getattr(logging, options.log_level.upper()), stream=sys.stdout)

parameters = open_output(options.pkgpath)

run_dmrg_and_sampling(parameters)
