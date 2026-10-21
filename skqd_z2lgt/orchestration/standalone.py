"""Full workflow."""
import os
import sys
from argparse import ArgumentParser
import logging
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Optional
import jax
from qiskit.primitives import BitArray
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.open_output import open_output, open_from_yaml
from skqd_z2lgt.orchestration.dmrg import run_dmrg_and_sampling
from skqd_z2lgt.orchestration.sample_quantum import get_trotter_circuits, compile_pubs
from skqd_z2lgt.orchestration.preprocess import preprocess_single_array
from skqd_z2lgt.orchestration.train_generator import train_single_model
from skqd_z2lgt.orchestration.diagonalize import diagonalize_in_mode
from skqd_z2lgt.orchestration.flows import (dmrg_flow, sample_quantum_flow, preprocess_flow,
                                            train_generator_flow)


def dmrg(parameters: Parameters, logger: Optional[logging.Logger] = None) -> float:
    """Standalone DMRG function."""
    return dmrg_flow(parameters, lambda: run_dmrg_and_sampling(parameters), logger)


def sample_quantum(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
) -> tuple[list[BitArray], list[BitArray]]:
    """Run the circuits on a backend and return the sampler results."""
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

    logger = logger or logging.getLogger(__name__)
    service = QiskitRuntimeService(instance=parameters.runtime.instance)

    def fetch_result_fn():
        return service.job(parameters.runtime.job_id).result()

    def sample_quantum_fn():
        backend = service.backend(parameters.runtime.backend, use_fractional_gates=True)
        layout, exp_circuits, ref_circuits = get_trotter_circuits(parameters, backend.target,
                                                                  logger)
        # Run primitive
        logger.info('Submitting a runtime job')
        pubs = compile_pubs(parameters, exp_circuits, ref_circuits)
        sampler = Sampler(backend, options=parameters.runtime.options)
        job = sampler.run(pubs)
        logger.info('Sampler job: %s', job.job_id())
        return layout, job.result(), job.job_id()

    return sample_quantum_flow(parameters, fetch_result_fn, sample_quantum_fn, logger)


def preprocess(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
):
    """Standalone preprocess function."""
    def convert_fn(tasks):
        for etype, idt, step in tasks:
            preprocess_single_array(parameters, etype, idt, step, logger=logger)

    preprocess_flow(parameters, convert_fn, logger=logger)


def train_generator(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
):
    """Standalone training function."""
    def train_fn(tasks):
        def train_in_thread(idev, idt, step):
            with jax.default_device(jax.devices()[idev]):
                train_single_model(parameters, idt, step)

        futures = []
        with ThreadPoolExecutor(jax.device_count()) as executor:
            for iproc, (idt, step) in enumerate(tasks):
                idev = iproc % jax.device_count()
                futures.append(executor.submit(train_in_thread, idev, idt, step))

        wait(futures)

    train_generator_flow(parameters, train_fn, logger=logger)


def run():
    parser = ArgumentParser(prog='skqd_z2lgt')
    parser.add_argument('parameters', metavar='PATH',
                        help='Path to the experiment package or a yaml file containing the workflow'
                             ' parameters.')
    parser.add_argument('--gpus', metavar='ID', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--upto', metavar='TASK', help='Stop at task.'
                        ' Value = dmrg, sample_quantum, preprocess, train_generator,'
                        ' diagonalize_rn, diagonalize_cr.')
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help='Logging level.')
    options = parser.parse_args()

    if options.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    logging.basicConfig(level=getattr(logging, options.log_level.upper()),
                        format='%(asctime)s:%(name)s:%(levelname)s %(message)s')
    logger = logging.getLogger('skqd_z2lgt')

    if os.path.isdir(options.parameters):
        parameters = open_output(options.parameters, logger)
    else:
        parameters = open_from_yaml(options.parameters, overwrite=options.overwrite)

    if parameters.dmrg:
        with ThreadPoolExecutor() as executor:
            dmrg_future = executor.submit(dmrg, parameters, logger=logger)
    if options.upto == 'dmrg':
        sys.exit(0)

    sample_quantum(parameters, logger=logger)
    if options.upto == 'sample_quantum':
        sys.exit(0)
    preprocess(parameters, logger=logger)
    if options.upto == 'preprocess':
        sys.exit(0)
    train_generator(parameters, logger=logger)
    if options.upto == 'train_generator':
        sys.exit(0)
    energy_rn = diagonalize_in_mode(parameters, 'rn', logger=logger)
    if options.upto == 'diagonalize_rn':
        sys.exit(0)
    energy_cr = diagonalize_in_mode(parameters, 'cr', logger=logger)
    if options.upto == 'diagonalize_cr':
        sys.exit(0)

    if parameters.dmrg:
        logger.info('DMRG energy: %f', dmrg_future.result())
    logger.info('SKQD energy (random bit flips): %f', energy_rn)
    logger.info('SKQD energy: %f', energy_cr)
