"""Full workflow."""
import os
import logging
import jax
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.open_output import open_output
from skqd_z2lgt.tasks.dmrg import dmrg
from skqd_z2lgt.tasks.sample_quantum import sample_quantum
from skqd_z2lgt.tasks.preprocess import preprocess
from skqd_z2lgt.tasks.train_generator import train_generator
from skqd_z2lgt.tasks.diagonalize import diagonalize_init, diagonalize

if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser(prog='skqd_z2lgt.py')
    parser.add_argument('parameters', metavar='PATH',
                        help='Path to a yaml file containing the workflow parameters.')
    parser.add_argument('--gpus', metavar='ID', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help='Logging level.')
    options = parser.parse_args()

    if options.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s:%(name)s:%(levelname)s %(message)s')
    logger = logging.getLogger('skqd_z2lgt')
    logger.setLevel(getattr(logging, options.log_level.upper()))

    with open(options.parameters, 'r', encoding='utf-8') as source:
        parameters = Parameters(**yaml.load(source, yaml.Loader))

    open_output(parameters)
    if parameters.dmrg:
        dmrg_energy = dmrg(parameters)
    raw_data = sample_quantum(parameters)
    reco_data = preprocess(parameters, raw_data)
    models = train_generator(parameters, reco_data[1])
    energy_norecov, states_init = diagonalize_init(parameters, reco_data[0])
    energy = diagonalize(parameters, reco_data[0], states_init, crbm_models=models)
    energy_random = diagonalize(parameters, reco_data[0], states_init, ref_data=reco_data[1])

    if parameters.dmrg:
        logger.info('DMRG energy: %f', dmrg_energy)
    logger.info('SKQD energy (no conf. recovery): %f', energy_norecov)
    logger.info('DMRG energy (random bit flips): %f', energy_random)
    logger.info('DMRG energy: %f', energy)
