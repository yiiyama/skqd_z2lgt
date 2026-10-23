"""Common functions for skqd_z2lgt tasks."""
from pathlib import Path
from qiskit.quantum_info import SparsePauliOp
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from heavyhex_qft.plaquette_dual import PlaquetteDual
from skqd_z2lgt.utils import minimum_weight_link_state
from skqd_z2lgt.orchestration.parameters import Parameters


def make_lattice(parameters: Parameters) -> TriangularZ2Lattice:
    lconf = parameters.lgt.lattice
    if isinstance(lconf, str) and lconf.endswith('.json'):
        lconf = Path(lconf)
        if not lconf.is_absolute():
            lconf = Path(parameters.pkgpath) / lconf
        with open(lconf, 'r') as source:
            return TriangularZ2Lattice.from_json(source.read())
    return TriangularZ2Lattice(lconf)


def make_dual_lattice(parameters: Parameters) -> PlaquetteDual:
    """Return the Ising hamiltonian for the given charge sector."""
    lattice = make_lattice(parameters)
    base_link_state = minimum_weight_link_state(parameters.lgt.charged_vertices,
                                                parameters.lgt.charged_plaquettes, lattice)
    dual_lattice = lattice.plaquette_dual(base_link_state)
    return dual_lattice


def make_dual_hamiltonian(parameters: Parameters) -> SparsePauliOp:
    dual_lattice = make_dual_lattice(parameters)
    return dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy)
