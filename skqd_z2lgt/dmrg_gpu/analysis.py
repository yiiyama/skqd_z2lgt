"""Extrapolation of DMRG energies to zero truncation error / zero variance."""
from __future__ import annotations

from typing import Optional
import numpy as np


def extrapolation_points(history: list) -> list:
    """Final noiseless sweep of each distinct maxdim, in schedule order.

    Points from sweeps with noise are excluded because the noise biases both the energy and the
    truncation error. Only maxdim values where the bond dimension was actually saturated carry
    information about the truncation, so a point is used only if its truncation error is positive.
    """
    last = {}
    for record in history:
        if record.noise == 0.:
            last[record.maxdim] = record
    return [rec for rec in last.values() if rec.max_trunc_err > 0.]


def _linear_fit(x: np.ndarray, y: np.ndarray, npoints: int) -> dict:
    order = np.argsort(x)[:npoints]  # Points with the smallest x are the most reliable
    x, y = x[order], y[order]
    if len(x) < 2 or np.ptp(x) == 0.:
        return {}
    slope, intercept = np.polyfit(x, y, 1)
    return {'energy': float(intercept), 'slope': float(slope), 'x': x.tolist(), 'y': y.tolist()}


def extrapolate(history: list, npoints: int = 4) -> dict:
    """Linear extrapolation of E in the truncation error (max discarded weight of the final sweep
    at each maxdim) and, when available, in the energy variance.

    Returns a dict with the best variational energy ('energy_best'), the extrapolated values
    ('truncation' / 'variance', each with energy, slope and the fitted points), and an error
    estimate 'error' = |energy_best - extrapolated|, taken from the variance fit when available.
    The extrapolation distance is a conservative error bar, not a statistical one.
    """
    points = extrapolation_points(history)
    result: dict = {'energy_best': float(min(rec.energy for rec in history)),
                    'npoints': len(points)}
    if len(points) < 2:
        return result
    energies = np.array([rec.energy for rec in points])
    trunc = np.array([rec.max_trunc_err for rec in points])
    if fit := _linear_fit(trunc, energies, npoints):
        result['truncation'] = fit
    variances = [rec.variance for rec in points]
    if all(v is not None for v in variances):
        if fit := _linear_fit(np.array(variances, dtype=float), energies, npoints):
            result['variance'] = fit
    best_fit: Optional[dict] = result.get('variance') or result.get('truncation')
    if best_fit:
        result['energy_extrapolated'] = best_fit['energy']
        result['error'] = abs(result['energy_best'] - best_fit['energy'])
    return result
