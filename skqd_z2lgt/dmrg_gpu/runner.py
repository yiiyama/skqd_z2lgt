"""High-level entry point: build a Z2 LGT problem, run DMRG on the active backend, measure."""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional
import numpy as np
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from .backend import xp
from .mpo import automaton_mpo, build_mpo, pauli_terms, permute_terms, site_order, mpo_bond_dims, cut_sizes
from .models import base_link_state, dual_problem, primal_problem
from .sweeps import Sweeps
from .dmrg import (two_site_dmrg, product_mps, random_mps, expectation, z_string_expectation,
                   sample_bitstrings, SweepRecord, overlap,
                   energy_and_variance)
from .analysis import extrapolate
from .checkpoint import (PROBLEM_KEYS, check_compatible, load_checkpoint, problem_meta,
                         save_checkpoint)
from .symmetric import GaussLawChain, SymmetricMPS, operator_charges, symmetric_two_site_dmrg

LOG = logging.getLogger(__name__)

DEFAULT_SWEEPS = {
    'nsweeps': 8,
    'maxdim': [16, 32, 64, 128, 256],
    'cutoff': [1.e-10],
    'noise': [1.e-3, 1.e-4, 1.e-5, 1.e-6, 0.],
}


def solve(
    lattice: TriangularZ2Lattice,
    plaquette_energy: float,
    charged_vertices: list[int],
    model: str = 'dual',
    charged_plaquettes: Optional[list[int]] = None,
    penalty: float = 4.,
    sweeps: Optional[Sweeps | dict | str] = None,
    ordering: str = 'snake',
    svd_method: str = 'svd',
    init: str = 'auto',
    init_bond_dim: int = 8,
    init_noise: float = 0.1,
    seed: int = 0,
    energy_tol: Optional[float] = None,
    num_samples: int = 0,
    callback: Optional[Callable[[SweepRecord], None]] = None,
    nstates: int = 1,
    excited_weight: float = 20.,
    measure_variance: bool = False,
    initial_state=None,
    checkpoint: Optional[str] = None,
    resume: bool = False
) -> dict:
    """Ground state of the Z2 LGT in the given charge sector.

    Args:
        model: 'dual' (Ising model on plaquettes, as in SKQD), 'primal' (link model with Gauss-law
            penalty) or 'primal-sym' (link model with exact Gauss-law symmetry, block-sparse).
        penalty: Energy cost per violated Gauss law (primal only).
        ordering: Mapping of qubits onto the MPS chain: 'snake' (default), 'snake-rows',
            'snake-columns', 'rcm' or 'natural' (see mpo.site_order).
        init: 'auto' (base link state for primal, random for dual), 'random', 'plus' or 'basis'.
        sweeps: Sweep schedule as a Sweeps object, a dict of its fields, or an ITensors-style
            table string (see skqd_z2lgt.dmrg_gpu.sweeps). Default: DEFAULT_SWEEPS.
        init_noise: Random admixture added to each site of the 'basis' initial state.
        nstates: Number of lowest eigenstates in the sector. States beyond the ground state are
            found by DMRG with the lower states projected out (excited_weight * |psi_k><psi_k|);
            the weight must exceed the gaps. For the primal model, keep penalty above the gaps
            as well, otherwise a Gauss-law-violating state can be returned.
        measure_variance: Measure the energy variance after every sweep, which enables the
            variance extrapolation in 'extrapolation'. The final variance is always reported.
        initial_state: Start from this state instead of `init` (the 'state' entry of a previous
            result for the same model, lattice, charges and ordering; e.g. a warm start at a
            different coupling, see scan.scan). Perturb it first (checkpoint.perturb_state) and
            use noise in the first sweeps to avoid staying in a previous minimum.
        checkpoint: Path of an .npz checkpoint written after every sweep (ground state only).
        resume: Continue from `checkpoint` if it exists. The checkpoint must describe the same
            problem (model, lattice, coupling, charges, ordering, penalty); the remaining sweeps
            of the current schedule are run.

    Returns:
        Dict with the physical energy (penalty excluded), sweep history, and diagnostics.
    """
    sweeps = Sweeps.from_config(DEFAULT_SWEEPS if sweeps is None else sweeps)
    link_state = base_link_state(lattice, charged_vertices, charged_plaquettes)
    if model == 'dual':
        problem = dual_problem(lattice, plaquette_energy, link_state)
    elif model == 'primal':
        problem = primal_problem(lattice, plaquette_energy, link_state, penalty)
    elif model == 'primal-sym':
        # Exact Gauss-law symmetry: no penalty terms
        problem = primal_problem(lattice, plaquette_energy, link_state, 0.)
        if nstates != 1:
            raise NotImplementedError('nstates > 1 is not implemented for primal-sym')
    else:
        raise ValueError(f'Unknown model {model}')

    nsites = problem.num_qubits
    all_terms = problem.terms()
    order = site_order(all_terms, nsites, ordering, positions=problem.positions)
    host_mpo = build_mpo(permute_terms(all_terms, order), nsites)
    mpo = [xp.asarray_dev(w) for w in host_mpo]
    LOG.info('%s model: %d sites, ordering %s (max cut %d), MPO bond dims max %d', model, nsites,
             ordering, cut_sizes(all_terms, order).max(), max(mpo_bond_dims(host_mpo), default=1))

    meta = problem_meta(model, lattice.graph.attrs['configuration'], plaquette_energy,
                        charged_vertices, charged_plaquettes, ordering, order, nsites, penalty,
                        sweeps.to_dict())
    start_sweep, history = 0, None
    if checkpoint is not None and nstates != 1:
        raise NotImplementedError('Checkpointing is implemented for the ground state only')
    if checkpoint is not None and resume and os.path.exists(checkpoint):
        saved_meta, initial_state, history = load_checkpoint(checkpoint)
        check_compatible(saved_meta, meta, PROBLEM_KEYS + ('plaquette_energy',))
        start_sweep = history[-1].sweep + 1 if history else 0
        LOG.info('Resuming from %s after sweep %d', checkpoint, start_sweep - 1)
    on_sweep = None
    if checkpoint is not None:
        def on_sweep(_, state, hist):
            save_checkpoint(checkpoint, meta, state, hist)

    if initial_state is not None:
        expected_type = SymmetricMPS if model == 'primal-sym' else list
        if not isinstance(initial_state, expected_type) or (
                expected_type is list and len(initial_state) != nsites):
            raise ValueError('initial_state does not match the model / number of sites')

    if init == 'auto':
        init = 'basis' if problem.initial_bits is not None else 'random'

    def initial_mps(state_seed):
        if init == 'basis':
            bits = problem.initial_bits if problem.initial_bits is not None else np.zeros(nsites)
            return product_mps(bits=np.asarray(bits)[order], noise=init_noise, seed=state_seed)
        if init == 'plus':
            return product_mps(num_sites=nsites, state='plus')
        return random_mps(nsites, init_bond_dim, seed=state_seed)

    results = []
    final_state = None
    if model == 'primal-sym':
        sym_result, final_state = _solve_symmetric(
            problem, order, mpo, sweeps, svd_method, energy_tol, callback, measure_variance,
            init_bond_dim, init, seed, init_noise, initial_state, start_sweep, history, on_sweep)
        results.append(sym_result)
    for istate in range(len(results), nstates):
        if nstates > 1:
            LOG.info('State %d', istate)
        first = istate == 0
        start = initial_state if first and initial_state is not None else \
            initial_mps(seed + istate)
        results.append(two_site_dmrg(mpo, start, sweeps,
                                     svd_method=svd_method, energy_tol=energy_tol,
                                     callback=callback,
                                     orthogonal_to=[res.mps for res in results],
                                     weight=excited_weight, measure_variance=measure_variance,
                                     start_sweep=start_sweep if first else 0,
                                     history=history if first else None,
                                     on_sweep=on_sweep if first else None))
    result = results[0]
    if final_state is None:
        final_state = result.mps

    # Physical energy without the penalty
    physical_mpo = [xp.asarray_dev(w) for w in
                    build_mpo(permute_terms(pauli_terms(problem.hamiltonian), order), nsites)]
    energies = np.array([expectation(res.mps, physical_mpo) for res in results])
    energy = energies[0]

    qubit_to_site = np.empty(nsites, dtype=int)
    qubit_to_site[order] = np.arange(nsites)
    gauss = np.array([z_string_expectation(result.mps, qubit_to_site[support])
                      for support in problem.gauss_supports])

    output = {
        'model': model,
        'num_sites': nsites,
        'energy': energy,
        'dmrg_energy': result.energy,
        'penalty_energy': result.energy - energy,
        'gauss_expectations': gauss,
        'gauss_targets': problem.gauss_targets,
        'gauss_supports': problem.gauss_supports,
        'order': order,
        'mpo_bond_dims': np.array(mpo_bond_dims(host_mpo)),
        'mps_bond_dims': np.array([a.shape[2] for a in result.mps[:-1]]),
        'history': result.history,
        'link_state': link_state,
        'mps': result.mps,
        'energies': energies,
        'excited_mps': [res.mps for res in results[1:]],
        'excited_histories': [res.history for res in results[1:]],
        'state_penalty_energies': np.array([expectation(res.mps, mpo) for res in results])
                                  - energies,
        'state_overlaps': np.array([[overlap(r1.mps, r2.mps) for r2 in results]
                                    for r1 in results]),
        # Variance of the DMRG Hamiltonian (incl. penalty), equal to that of H within the sector
        'variances': np.array([energy_and_variance(res.mps, mpo)[1] for res in results]),
        'extrapolation': extrapolate(result.history),
        # Reusable final state (dense MPS list, or SymmetricMPS for primal-sym)
        'state': final_state,
    }
    output['variance'] = output['variances'][0]
    if num_samples > 0:
        # Samples indexed by qubit (column q = qubit q of the SparsePauliOp)
        samples = sample_bitstrings(result.mps, num_samples, seed=seed)
        output['samples'] = samples[:, qubit_to_site]
    return output


def _solve_symmetric(problem, order, dense_mpo, sweeps, svd_method, energy_tol, callback,
                     measure_variance, init_bond_dim, init, seed, init_noise, initial_state=None,
                     start_sweep=0, history=None, on_sweep=None):
    """Exact-Gauss-law DMRG; returns (DMRGResult whose mps is the equivalent dense MPS,
    final SymmetricMPS)."""
    nsites = problem.num_qubits
    vertex_of = [[] for _ in range(nsites)]
    for ivert, support in enumerate(problem.gauss_supports):
        for qubit in support:
            vertex_of[qubit].append(ivert)
    masks = np.array([sum(1 << v for v in vertex_of[q]) for q in order], dtype=np.int64)
    target = sum(1 << v for v, q in enumerate(problem.gauss_targets) if q < 0)
    chain = GaussLawChain(masks, target)
    terms = permute_terms(pauli_terms(problem.hamiltonian), order)
    mpo_host, bond_keys = automaton_mpo(terms, nsites)
    deltas = operator_charges(bond_keys, masks)
    LOG.info('primal-sym: max allowed labels per bond %d, automaton MPO bond max %d',
             max(len(a) for a in chain.allowed), max(w.shape[1] for w in mpo_host[:-1]))
    if initial_state is not None:
        if not np.array_equal(initial_state.chain.masks, masks) or \
                initial_state.chain.target != target:
            raise ValueError('initial_state belongs to a different chain or charge sector')
        sym_mps = initial_state
    elif init in ('auto', 'basis'):
        sym_mps = SymmetricMPS.from_bits(chain, np.asarray(problem.initial_bits)[order],
                                         noise=init_noise, seed=seed)
    else:
        sym_mps = SymmetricMPS.random(chain, block_dim=max(1, init_bond_dim // 8), seed=seed)
    result, final = symmetric_two_site_dmrg(chain, mpo_host, deltas, sym_mps, sweeps,
                                            svd_method=svd_method, energy_tol=energy_tol,
                                            callback=callback, measure_variance=measure_variance,
                                            dense_mpo=dense_mpo, seed=seed,
                                            start_sweep=start_sweep, history=history,
                                            on_sweep=on_sweep)
    return result, final
