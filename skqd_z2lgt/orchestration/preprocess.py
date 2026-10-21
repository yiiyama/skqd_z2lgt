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
