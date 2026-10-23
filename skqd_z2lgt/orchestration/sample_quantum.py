"""Compose the Krylov circuits and run the quantum runtime sampler."""
import os
import logging
from pathlib import Path
from typing import Optional
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend, Batch
from qiskit_ibm_runtime.executor_sampler import Sampler
from skqd_z2lgt.circuits import make_step_circuits, compose_trotter_circuits
from skqd_z2lgt.utils import minimum_weight_link_state
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.io import save_raw
from skqd_z2lgt.orchestration.common import make_lattice


def get_trotter_circuits(
    parameters: Parameters,
    target: Target,
    logger: Optional[logging.Logger] = None
) -> tuple[list[int], list[QuantumCircuit], list[QuantumCircuit]]:
    """Compose full Trotter simulation circuits.

    We first generate single-step circuit elements for the given lattice and base two-qubit gate,
    compile them, then compose the resulting ISA circuits into multi-step Trotter simulation
    circuits. Both the forward-evolution (Krylov) circuits and forward-backward (reference) circuits
    are returned.

    Args:
        parameters: Workflow parameters.
        target: Backend target.

    Returns:
        Physical qubit layout and lists (length configuration.num_steps) of forward-evolution and
        forward-backward circuits.
    """
    logger = logger or logging.getLogger(__name__)

    lattice = make_lattice(parameters)
    if lattice.layout is not None:
        parameters.circuit.layout = lattice.layout

    initial_state = minimum_weight_link_state(parameters.lgt.charged_vertices,
                                              parameters.lgt.charged_plaquettes, lattice)

    steps = parameters.circuit.sampled_steps
    all_circuits = {}
    for idt, dt in enumerate(parameters.circuit.dts):
        snippets, layout = make_step_circuits(
            lattice, parameters.lgt.plaquette_energy, dt, target, initial_state=initial_state,
            layout=parameters.circuit.layout,
            optimization_level=parameters.circuit.optimization_level
        )
        circuits = compose_trotter_circuits(snippets[0], snippets[1], snippets[3], snippets[5],
                                            steps)
        all_circuits |= {('exp', idt, s): c for s, c in zip(steps, circuits)}
        circuits = compose_trotter_circuits(snippets[0], snippets[2], snippets[4], snippets[5],
                                            steps)
        all_circuits |= {('ref', idt, s): c for s, c in zip(steps, circuits)} 
    return layout, all_circuits


def filter_out_completed(
    parameters: Parameters,
    circuits: dict[tuple[str, int, int], QuantumCircuit],
    task_map: dict[str, list],
    service: Optional[QiskitRuntimeService] = None,
    logger: Optional[logging.Logger] = None
):
    """Drop circuits that already have sampled results locally or in the cloud."""
    if service is None:
        service = QiskitRuntimeService()

    logger = logger or logging.getLogger(__name__)
    
    raw_data_path = Path(parameters.pkgpath) / 'data/raw'
    # Drop circuits whose raw data already exist
    for etype, idt, step in list(circuits.keys()):
        if os.path.exists(raw_data_path / f'{etype}_dt{idt}_k{step}.h5'):
            logger.info('Raw data %s_dt%d_k%d exists', etype, idt, step)
            circuits.pop((etype, idt, step))

    # Check the existing task map and fetch missing results. Drop circuits whose results are
    # successfully recovered
    for job_id, key in task_map.items():
        key = tuple(key)
        if key in circuits:
            logger.info('Fetching job result for %s_dt%d_k%d', etype, idt, step)    
            job = service.job(job_id)
            result = job.result()
            save_raw(parameters, result[0].data.c, *key)
            circuits.pop(key)

    return circuits


def run_sampler(
    parameters: Parameters,
    circuits: dict[tuple[str, int, int], QuantumCircuit],
    backend: IBMBackend,
    logger: Optional[logging.Logger] = None
):
    """Submit sampler jobs and fetch the results."""
    logger = logger or logging.getLogger(__name__)

    logger.info('Submitting jobs for %d circuits', len(circuits))

    jobs = {}
    with Batch(backend=backend):
        sampler = Sampler()
        for (etype, idt, step), circuit in circuits.items():
            shots = getattr(parameters.runtime, f'shots_{etype}')
            job = sampler.run([circuit], shots=shots)
            jobs[(etype, idt, step)] = job

    task_map = {}
    for (etype, idt, step), job in jobs.items():
        save_raw(parameters, job.result()[0].data.c, etype, idt, step)
        task_map[job.job_id()] = [etype, idt, step]

    return task_map
