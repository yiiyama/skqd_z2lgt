"""Hamiltonians of the 2D Z2 LGT for DMRG: dual Ising model and primal model with Gauss-law penalty.

Both models are constructed from the same inputs as the SKQD workflow (lattice configuration,
plaquette energy K, charged vertices / plaquettes), and the charge sector is fixed through the same
minimum-weight base link state, so the ground-state energies of the two models must coincide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.utils import minimum_weight_link_state
from .mpo import Term, pauli_terms


@dataclass
class LGTProblem:
    """A DMRG problem: physical Hamiltonian, optional penalty terms and a reference product state.

    Qubit indexing follows the SparsePauliOp convention of the underlying heavyhex_qft object.
    """
    name: str
    num_qubits: int
    hamiltonian: SparsePauliOp
    penalty_terms: list[Term]
    initial_bits: Optional[np.ndarray]  # Indexed by qubit
    gauss_supports: list[list[int]]  # Qubits of each Gauss-law operator (primal only)
    gauss_targets: np.ndarray  # Target eigenvalue (+1/-1) of each Gauss-law operator
    positions: Optional[np.ndarray] = None  # 2D position of each qubit (plaquette or link centre)

    def terms(self) -> list[Term]:
        return pauli_terms(self.hamiltonian) + list(self.penalty_terms)


def make_lattice(lattice_config) -> TriangularZ2Lattice:
    if isinstance(lattice_config, (list, tuple)):
        lattice_config = tuple(lattice_config)
    return TriangularZ2Lattice(lattice_config)


def base_link_state(lattice: TriangularZ2Lattice, charged_vertices: list[int],
                    charged_plaquettes: Optional[list[int]] = None) -> np.ndarray:
    """Base link state of the charge sector, identical to the SKQD workflow
    (skqd_z2lgt.orchestration.common.make_dual_lattice)."""
    return minimum_weight_link_state(list(charged_vertices), list(charged_plaquettes or []),
                                     lattice)


def link_bits_by_qubit(lattice: TriangularZ2Lattice, link_state: np.ndarray) -> np.ndarray:
    """Convert a heavyhex_qft link-state array (bit for link qubit nl-1-i at position i) to an array
    indexed by the qubit index of the primal SparsePauliOp."""
    return np.asarray(link_state, dtype=np.uint8)[::-1].copy()


def dual_problem(lattice: TriangularZ2Lattice, plaquette_energy: float,
                 link_state: np.ndarray) -> LGTProblem:
    """Dual Ising Hamiltonian on the active plaquettes (exactly what SKQD diagonalizes)."""
    dual = lattice.plaquette_dual(link_state)
    hamiltonian = dual.make_hamiltonian(plaquette_energy)
    plaquettes = lattice.graph.attrs['plaquettes']
    positions = np.array([plaquettes[pid].position for pid in lattice.active_plaquette_ids],
                         dtype=float)
    return LGTProblem(name='dual', num_qubits=hamiltonian.num_qubits, hamiltonian=hamiltonian,
                      penalty_terms=[], initial_bits=None, gauss_supports=[],
                      gauss_targets=np.zeros(0), positions=positions)


def gauss_law(lattice: TriangularZ2Lattice, link_state: np.ndarray):
    """Gauss-law operators G_v = prod_{l in v} Z_l as qubit supports, and their target eigenvalues
    (-1)^(charge at v) in the sector of the link state."""
    lid_to_qubit = lattice._lid_to_bit  # pylint: disable=protected-access
    bits = link_bits_by_qubit(lattice, link_state)
    supports = []
    targets = []
    for vid in lattice.vertex_ids:
        qubits = sorted(int(lid_to_qubit[lid]) for lid in lattice.vertex_links(vid))
        supports.append(qubits)
        targets.append(1. - 2. * (int(np.sum(bits[qubits])) % 2))
    return supports, np.array(targets)


def primal_problem(lattice: TriangularZ2Lattice, plaquette_energy: float, link_state: np.ndarray,
                   penalty: float) -> LGTProblem:
    """Primal link Hamiltonian H = -sum Z_l - K sum_p XXX plus
    penalty * sum_v (1 - q_v G_v) / 2, which projects onto the charge sector of the link state.
    """
    if lattice.num_active_plaquettes != lattice.num_plaquettes:
        # Inactive plaquettes give additional conserved fluxes that the dual model fixes through
        # the base link state; these would need extra (nonlocal) penalty terms here.
        raise NotImplementedError('Primal penalty DMRG with inactive plaquettes is not supported')

    hamiltonian = lattice.make_hamiltonian(plaquette_energy)
    supports, targets = gauss_law(lattice, link_state)
    penalty_terms: list[Term] = []
    if penalty != 0.:
        for qubits, target in zip(supports, targets):
            penalty_terms.append((0.5 * penalty, {}))
            penalty_terms.append((-0.5 * penalty * target, {q: 'Z' for q in qubits}))

    links = lattice.graph.attrs['links']
    positions = np.zeros((lattice.num_links, 2))
    for lid in lattice.link_ids:
        positions[lattice._lid_to_bit[lid]] = links[lid].position  # pylint: disable=protected-access
    return LGTProblem(name='primal', num_qubits=hamiltonian.num_qubits, hamiltonian=hamiltonian,
                      penalty_terms=penalty_terms,
                      initial_bits=link_bits_by_qubit(lattice, link_state),
                      gauss_supports=supports, gauss_targets=targets, positions=positions)
