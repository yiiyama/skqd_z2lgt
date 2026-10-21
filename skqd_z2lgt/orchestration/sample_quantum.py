"""Compose the Krylov circuits and run the quantum runtime sampler."""
import logging
from typing import Optional
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from skqd_z2lgt.circuits import make_step_circuits, compose_trotter_circuits
from skqd_z2lgt.utils import minimum_weight_link_state
from skqd_z2lgt.orchestration.parameters import Parameters
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

    exp_circuits = []
    ref_circuits = []
    for dt in parameters.circuit.dts:
        circuits, layout = make_step_circuits(
            lattice, parameters.lgt.plaquette_energy, dt, target, initial_state=initial_state,
            layout=parameters.circuit.layout,
            optimization_level=parameters.circuit.optimization_level
        )
        steps = parameters.circuit.sampled_steps
        exp_circuits.append(
            compose_trotter_circuits(circuits[0], circuits[1], circuits[3], circuits[5], steps)
        )
        ref_circuits.append(
            compose_trotter_circuits(circuits[0], circuits[2], circuits[4], circuits[5], steps)
        )
    return layout, exp_circuits, ref_circuits


def compile_pubs(parameters, exp_circuits, ref_circuits):
    pubs = [(circ, [], parameters.runtime.shots_exp) for circ in sum(exp_circuits, [])]
    pubs += [(circ, [], parameters.runtime.shots_ref) for circ in sum(ref_circuits, [])]
    return pubs
