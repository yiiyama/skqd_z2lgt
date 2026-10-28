"""Coupling scans with warm starts, guarded against getting stuck in a previous minimum.

A warm start reuses the converged state of the neighbouring coupling. This is much cheaper than a
cold start, but the state can stay trapped in the basin of the previous solution, especially across
level crossings or in regions where the ground state changes character. Safeguards:

1. The inherited state is perturbed (checkpoint.perturb_state, relative size `perturbation`) and
   the warm schedule starts with noise (White's density-matrix correction), which lets DMRG leave
   the previous basin.
2. Cold anchors: independent cold starts at the first and last coupling and at every
   `cold_every`-th point. If a cold start ends lower than the warm one by more than
   `tolerance`, the point is flagged and the scan continues from the cold state.
3. Optional bidirectional scan: forward and backward warm passes; per point the lower energy is
   kept and the difference (hysteresis) is reported and flagged above `tolerance`.

Since DMRG is variational, the minimum over all runs at a point is the best estimate, and every
disagreement is recorded instead of being silently resolved.
"""
from __future__ import annotations

import logging
from typing import Optional
import numpy as np
from .checkpoint import perturb_state
from .runner import solve
from .sweeps import Sweeps

LOG = logging.getLogger(__name__)

DEFAULT_WARM_SWEEPS = {'nsweeps': 6, 'maxdim': [256], 'cutoff': [1.e-10],
                       'noise': [1.e-4, 1.e-5, 1.e-6, 0.]}


def scan(
    lattice,
    couplings: list[float],
    charged_vertices: list[int],
    model: str = 'dual',
    sweeps=None,
    warm_sweeps=None,
    warm_start: bool = True,
    perturbation: float = 1.e-2,
    cold_every: Optional[int] = 4,
    bidirectional: bool = False,
    tolerance: float = 1.e-8,
    seed: int = 0,
    **solve_kwargs
) -> list[dict]:
    """Ground-state energies along a list of couplings.

    Args:
        sweeps: Schedule for cold starts (default: solve's default).
        warm_sweeps: Schedule for warm starts; should begin with noise (DEFAULT_WARM_SWEEPS).
        warm_start: If False, every point is a cold start.
        perturbation: Relative random perturbation applied to an inherited state.
        cold_every: Cold anchor spacing (None: only first and last point).
        bidirectional: Also scan backwards with warm starts.
        tolerance: Energy difference above which warm/cold or forward/backward disagreements are
            flagged.
        solve_kwargs: Passed to solve (penalty, ordering, svd_method, ...).

    Returns:
        One dict per coupling: energy (minimum over runs), variance, source of the minimum, the
        energies of all runs, and flags.
    """
    couplings = list(couplings)
    npoints = len(couplings)
    warm = Sweeps.from_config(warm_sweeps or DEFAULT_WARM_SWEEPS)

    def run(coupling, state=None, label='cold', run_seed=seed):
        kwargs = dict(solve_kwargs)
        if state is None:
            res = solve(lattice, coupling, charged_vertices, model=model, sweeps=sweeps,
                        seed=run_seed, **kwargs)
        else:
            res = solve(lattice, coupling, charged_vertices, model=model, sweeps=warm,
                        initial_state=perturb_state(state, perturbation, seed=run_seed),
                        seed=run_seed, **kwargs)
        LOG.info('lambda=%.4f %s E=%.12f var=%.2e', coupling, label, res['energy'],
                 res['variance'])
        return res

    def anchor(index):
        if not warm_start:
            return True
        if index in (0, npoints - 1):
            return True
        return cold_every is not None and index % cold_every == 0

    runs: list[dict] = [{} for _ in range(npoints)]
    states: list = [None] * npoints

    def best_state(index):
        entries = runs[index]
        label = min(entries, key=lambda key: entries[key]['energy'])
        return entries[label]['state']

    def sweep_pass(indices, direction):
        previous = None
        for index in indices:
            coupling = couplings[index]
            if warm_start and previous is not None:
                runs[index][f'warm-{direction}'] = run(coupling, previous, f'warm-{direction}',
                                                       seed + index)
            if 'cold' not in runs[index] and (anchor(index) or previous is None):
                runs[index]['cold'] = run(coupling, None, 'cold', seed + index)
            # Continue from the lowest state found so far at this point (re-anchors after a
            # stuck warm start)
            previous = best_state(index)
            states[index] = previous

    sweep_pass(range(npoints), 'fwd')
    if bidirectional and warm_start:
        sweep_pass(range(npoints - 1, -1, -1), 'bwd')

    results = []
    for coupling, entries in zip(couplings, runs):
        energies = {label: float(res['energy']) for label, res in entries.items()}
        source = min(energies, key=energies.get)
        flags = []
        if 'cold' in energies:
            for label, energy in energies.items():
                if label != 'cold' and energies['cold'] < energy - tolerance:
                    flags.append(f'{label} stuck above cold start by {energy - energies["cold"]:.2e}')
        if 'warm-fwd' in energies and 'warm-bwd' in energies:
            hyst = abs(energies['warm-fwd'] - energies['warm-bwd'])
            if hyst > tolerance:
                flags.append(f'hysteresis {hyst:.2e}')
        for flag in flags:
            LOG.warning('lambda=%.4f: %s', coupling, flag)
        best = entries[source]
        results.append({'coupling': coupling, 'energy': energies[source], 'source': source,
                        'variance': float(best['variance']), 'energies': energies,
                        'flags': flags, 'extrapolation': best['extrapolation'],
                        'max_bond': int(np.max(best['mps_bond_dims'], initial=1)),
                        'state': best['state']})
    return results
