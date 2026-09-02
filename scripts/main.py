"""Full workflow."""
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import jax
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.open_output import open_output
from skqd_z2lgt.orchestration.dmrg import dmrg
from skqd_z2lgt.orchestration.sample_quantum import sample_quantum
from skqd_z2lgt.orchestration.preprocess import preprocess
from skqd_z2lgt.orchestration.train_generator import train_generator
from skqd_z2lgt.orchestration.diagonalize import diagonalize_in_mode

if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser(prog='skqd_z2lgt')
    parser.add_argument('parameters', metavar='PATH',
                        help='Path to a yaml file containing the workflow parameters.')
    parser.add_argument('--gpus', metavar='ID', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help='Logging level.')
    options = parser.parse_args()

    if options.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    logging.basicConfig(level=getattr(logging, options.log_level.upper()),
                        format='%(asctime)s:%(name)s:%(levelname)s %(message)s')
    logger = logging.getLogger('skqd_z2lgt')

    with open(options.parameters, 'r', encoding='utf-8') as source:
        parameters = Parameters(**yaml.load(source, yaml.Loader))

    open_output(parameters, logger)

    if parameters.dmrg:
        with ThreadPoolExecutor() as executor:
            dmrg_future = executor.submit(dmrg, parameters, logger=logger)

    sample_quantum(parameters, logger=logger)
    preprocess(parameters, logger=logger)
    train_generator(parameters, logger=logger)
    energy_rn = diagonalize_in_mode(parameters, 'rn', logger=logger)
    energy_cr = diagonalize_in_mode(parameters, 'cr', logger=logger)

    if parameters.dmrg:
        logger.info('DMRG energy: %f', dmrg_future.result())
    logger.info('SKQD energy (random bit flips): %f', energy_rn)
    logger.info('SKQD energy: %f', energy_cr)
