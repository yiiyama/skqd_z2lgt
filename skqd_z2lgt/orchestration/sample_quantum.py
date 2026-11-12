"""Compose the Krylov circuits and run the quantum runtime sampler."""
import os
import logging
from typing import Optional
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from qiskit_ibm_runtime import QiskitRuntimeService
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


def submit_sampler_job(
    parameters: Parameters,
    circuit: QuantumCircuit,
    etype: str,
    idt: int,
    step: int,
    run_name: Optional[str] = None
) -> str:
    job_tags = [f'{etype}_dt{idt}_k{step}']
    if run_name:
        job_tags.append(run_name)
    sampler = Sampler(options={'environment': {'job_tags': job_tags}})
    shots = getattr(parameters.runtime, f'shots_{etype}')
    job = sampler.run([circuit], shots=shots)
    return job.job_id()


def retrieve_samples(
    parameters: Parameters,
    job_id: str,
    etype: str,
    idt: int,
    step: int,
    service: Optional[QiskitRuntimeService] = None
):
    if service is None:
        service = QiskitRuntimeService()
    job = service.job(job_id)
    save_raw(parameters, job.result()[0].data.c, etype, idt, step)
