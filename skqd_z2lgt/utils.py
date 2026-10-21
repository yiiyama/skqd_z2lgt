"""Utility functions."""
from typing import Optional
import numpy as np
from scipy.sparse import csc_matrix
import h5py
from pymatching import Matching
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice


def read_bits(
    dataset: np.ndarray | h5py.Dataset,
    num_bits: Optional[int] = None,
    align: str = 'left',
    offset: int = 0
) -> np.ndarray:
    bits = np.unpackbits(dataset, axis=-1)
    num_bits = num_bits or dataset.attrs['num_bits']
    if align == 'left':
        return bits[..., offset:num_bits + offset]
    return bits[..., -(offset + num_bits):bits.shape[-1] - offset]


def save_bits(group: h5py.Group, name: str, bits: np.ndarray) -> h5py.Dataset:
    dataset = group.create_dataset(name, data=np.packbits(bits, axis=-1))
    dataset.attrs['num_bits'] = bits.shape[-1]
    return dataset


def minimum_weight_link_state(
    electric_charges: list[int],
    magnetic_charges: list[int],
    lattice: TriangularZ2Lattice,
    matching: Optional[Matching] = None
) -> np.ndarray:
    """Return a link state with minimal excitations corresponding to the charge distribution."""
    if matching is None:
        matching = Matching(csc_matrix(lattice.vl_matrix))

    if not electric_charges and not magnetic_charges:
        return np.zeros(lattice.num_links, dtype=np.uint8)

    if len(electric_charges) % 2 == 1:
        raise ValueError('There must be an even number of electric charges')

    charge_config = np.zeros(lattice.num_vertices, dtype=int)
    charge_config[[lattice.vertex_id_to_idx(vid) for vid in electric_charges]] = 1
    link_state = matching.decode(charge_config)

    if magnetic_charges:
        active_pids = set(lattice.active_plaquette_ids)
        if any(pid in active_pids for pid in magnetic_charges):
            raise ValueError('Active plaquette listed as magnetic charge')
        # Flip minimum loops around magnetic charges
        mcharges = np.zeros(lattice.num_plaquettes, dtype=np.uint8)
        mcharges[[lattice.plaquette_id_to_idx(pid) for pid in magnetic_charges]] = 1
        link_state ^= (mcharges @ lattice.pl_matrix) & 1

    return link_state
