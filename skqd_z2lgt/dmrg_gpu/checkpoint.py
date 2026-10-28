"""MPS checkpoint/restart and state perturbation for warm starts.

A checkpoint is a single .npz file holding the MPS tensors (dense, or block-sparse with the bond
labels and dimensions for primal-sym), the sweep history and a JSON metadata record describing the
problem (model, lattice, coupling, charges, ordering, sweep schedule, completed sweeps). Files are
written atomically (temporary file + rename), so an interrupted write never corrupts the previous
checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from typing import Optional
import numpy as np
from .backend import xp
from .dmrg import SweepRecord
from .symmetric import Bond, SymmetricMPS, GaussLawChain

# Metadata entries that must match for a resume (the coupling may differ for a warm start)
PROBLEM_KEYS = ('model', 'lattice', 'charged_vertices', 'charged_plaquettes', 'ordering', 'order',
                'num_sites', 'penalty')


def save_checkpoint(path: str, meta: dict, state, history: list) -> None:
    arrays = {}
    if isinstance(state, SymmetricMPS):
        arrays['kind'] = np.array('symmetric')
        for ibond, bond in enumerate(state.bonds):
            arrays[f'labels_{ibond}'] = bond.labels
            arrays[f'dims_{ibond}'] = bond.dims
        arrays['masks'] = state.chain.masks
        arrays['target'] = np.array(state.chain.target)
        tensors = state.tensors
    else:
        arrays['kind'] = np.array('dense')
        tensors = state
    for site, tensor in enumerate(tensors):
        arrays[f'site_{site}'] = xp.asnumpy(tensor)
    arrays['meta'] = np.array(json.dumps(meta))
    arrays['history'] = np.array(json.dumps([asdict(rec) for rec in history]))
    tmp = f'{path}.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_checkpoint(path: str):
    """Return (meta, state, history). state is a list of device tensors or a SymmetricMPS."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        history = [SweepRecord(**rec) for rec in json.loads(str(data['history']))]
        nsites = meta['num_sites']
        tensors = [xp.asarray_dev(data[f'site_{site}']) for site in range(nsites)]
        if str(data['kind']) == 'symmetric':
            chain = GaussLawChain(data['masks'], int(data['target']))
            bonds = [Bond(data[f'labels_{ibond}'], data[f'dims_{ibond}'])
                     for ibond in range(nsites + 1)]
            state = SymmetricMPS(chain, bonds, tensors)
        else:
            state = tensors
    return meta, state, history


def check_compatible(meta: dict, expected: dict, keys=PROBLEM_KEYS) -> None:
    """Raise if a checkpoint does not describe the same problem."""
    mismatches = {key: (meta.get(key), expected.get(key)) for key in keys
                  if meta.get(key) != expected.get(key)}
    if mismatches:
        raise ValueError(f'Checkpoint does not match the requested problem: {mismatches}')


def perturb_state(state, amplitude: float, seed: int = 0):
    """Return a copy of an MPS with a random perturbation of relative size `amplitude` per site.

    Used before reusing a converged state as a warm start, so that DMRG is not started exactly in
    the previous minimum. For a SymmetricMPS the perturbation is restricted to the symmetry-allowed
    blocks, so the state stays in its Gauss-law sector.
    """
    rng = np.random.default_rng(seed)
    if isinstance(state, SymmetricMPS):
        tensors = []
        for site, tensor in enumerate(state.tensors):
            left, right = state.bonds[site], state.bonds[site + 1]
            smap = state.site_map(site)
            noise = np.zeros(tensor.shape)
            for i, s in zip(*np.nonzero(smap >= 0)):
                r = smap[i, s]
                noise[i, s, :left.dims[i], :right.dims[r]] = \
                    rng.normal(size=(left.dims[i], right.dims[r]))
            scale = float(xp.scalar(xp.linalg.norm(tensor))) / max(np.linalg.norm(noise), 1.e-300)
            tensors.append(tensor + amplitude * scale * xp.asarray_dev(noise))
        new = SymmetricMPS(state.chain, [Bond(b.labels.copy(), b.dims.copy())
                                         for b in state.bonds], tensors)
        new.right_canonicalize()
        return new
    out = []
    for tensor in state:
        noise = rng.normal(size=tensor.shape)
        scale = float(xp.scalar(xp.linalg.norm(tensor))) / max(np.linalg.norm(noise), 1.e-300)
        out.append(tensor + amplitude * scale * xp.asarray_dev(noise))
    return out


def problem_meta(model: str, lattice_config, plaquette_energy: float, charged_vertices,
                 charged_plaquettes, ordering: str, order, num_sites: int, penalty: float,
                 sweeps: Optional[dict] = None) -> dict:
    return {'model': model, 'lattice': str(lattice_config), 'plaquette_energy': plaquette_energy,
            'charged_vertices': list(map(int, charged_vertices)),
            'charged_plaquettes': list(map(int, charged_plaquettes or [])),
            'ordering': ordering, 'order': list(map(int, order)), 'num_sites': int(num_sites),
            'penalty': float(penalty) if model == 'primal' else 0., 'sweeps': sweeps}
