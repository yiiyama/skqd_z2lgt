"""Flows for program stages."""
from collections.abc import Callable
import logging
from pathlib import Path
from typing import Optional
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.io import (
    load_dmrg,
    load_raw,
    save_raw,
    check_reco,
    check_crbm_model,
    load_skqd_result
)


def dmrg_flow(
    parameters: Parameters,
    dmrg_fn: Callable,
    logger: Optional[logging.Logger] = None
) -> float:
    """Run DMRG on the dual Ising hamiltonian."""
    logger = logger or logging.getLogger(__name__)

    if (energy := load_dmrg(parameters, logger)) is not None:
        return energy

    logger.info('Invoking ITensorMPS DMRG function and sampling the MPS')
    # energy, states, probs = dmrg_fn()
    energy = dmrg_fn()
    return energy


def sample_quantum_flow(
    parameters: Parameters,
    fetch_result_fn: Callable,
    sample_quantum_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    """General flow for obtaining bitstring samples from quantum circuits."""
    logger = logger or logging.getLogger(__name__)

    try:
        load_raw(parameters)
    except FileNotFoundError:
        pass
    else:
        logger.info('Raw bitstrings already saved to file')
        return

    if parameters.runtime.job_id:
        logger.info('Fetching result of workload %s', parameters.runtime.job_id)
        pub_result = fetch_result_fn()
    else:
        logger.info('Running a new experiment')
        # Transpile and compose the circuits
        layout, pub_result, job_id = sample_quantum_fn()
        parameters.circuit.layout = layout
        parameters.runtime.job_id = job_id
        with open(Path(parameters.pkgpath) / 'parameters.json', 'w', encoding='utf-8') as out:
            out.write(parameters.model_dump_json())

    save_raw(parameters, pub_result, logger)


def preprocess_flow(
    parameters: Parameters,
    convert_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings."""
    logger = logger or logging.getLogger(__name__)

    tasks = []
    for etype in ['exp', 'ref']:
        for idt in range(len(parameters.circuit.dts)):
            for step in parameters.circuit.sampled_steps:
                if not check_reco(parameters, etype, idt, step, logger=logger):
                    tasks.append((etype, idt, step))

    if not tasks:
        logger.info('All link-state bitstrings already converted to syndrome and plaquette data')
        return

    logger.info('Correcting %d raw BitArrays to syndrome and plaquette data', len(tasks))
    convert_fn(tasks)


def train_generator_flow(
    parameters: Parameters,
    train_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    """General flow for training CRBM models."""
    logger = logger or logging.getLogger(__name__)

    tasks = []
    for idt in range(len(parameters.circuit.dts)):
        for step in parameters.circuit.sampled_steps:
            if not check_crbm_model(parameters, idt, step, logger=logger):
                tasks.append((idt, step))

    if not tasks:
        logger.info('All models already trained')
        return

    logger.info('Training %d CRBMs', len(tasks))
    train_fn(tasks)


def diagonalize_flow(
    parameters: Parameters,
    mode: str,  # 'cr' or 'rn'
    diagonalize_fn: Callable,
    logger: Optional[logging.Logger] = None
) -> float:
    """Project and diagonalize the Hamiltonian with configuration recovery or random bitflips."""
    logger = logger or logging.getLogger(__name__)

    try:
        energies = load_skqd_result(parameters, mode)[0]
    except FileNotFoundError:
        pass
    else:
        logger.info('SKQD result %s already saved to file.', mode)
        return float(energies[-1])

    return diagonalize_fn()
