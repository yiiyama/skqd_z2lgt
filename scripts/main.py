"""Full workflow."""
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import jax
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.open_output import open_output
from skqd_z2lgt.tasks.dmrg import dmrg
from skqd_z2lgt.tasks.sample_quantum import sample_quantum
from skqd_z2lgt.tasks.preprocess import preprocess
from skqd_z2lgt.tasks.train_generator import train_generator
from skqd_z2lgt.tasks.diagonalize import diagonalize_init, diagonalize, compile_models

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

    open_output(parameters)

    with ThreadPoolExecutor() as executor:
        if parameters.dmrg:
            dmrg_future = executor.submit(dmrg, parameters)
        raw_data = sample_quantum(parameters)
        reco_data = preprocess(parameters, raw_data)
        models_future = executor.submit(train_generator, parameters, reco_data[1])
        models_future.add_done_callback(lambda fut: compile_models(parameters, fut.result()))
        energy_init, states_init = diagonalize_init(parameters, reco_data[0])
        if jax.device_count() > 1:
            rnd_future = executor.submit(diagonalize, parameters, reco_data[0], energy_init,
                                         states_init, ref_data=reco_data[1], jax_device_id=1)
            rcv_future = executor.submit(diagonalize, parameters, reco_data[0], energy_init,
                                         states_init, crbm_models=models_future.result(),
                                         jax_device_id=0)
            energy_rnd = rnd_future.result()
            energy = rcv_future.result()
        else:
            energy_rnd = diagonalize(parameters, reco_data[0], energy_init, states_init,
                                     ref_data=reco_data[1])
            energy = diagonalize(parameters, reco_data[0], energy_init, states_init,
                                 crbm_models=models_future.result())

    if parameters.dmrg:
        logger.info('DMRG energy: %f', dmrg_future.result())
    logger.info('SKQD energy (no conf. recovery): %f', energy_init)
    logger.info('SKQD energy (random bit flips): %f', energy_rnd)
    logger.info('SKQD energy: %f', energy)
