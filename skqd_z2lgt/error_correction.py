"""Functions for minimum-distance correction of observed link bitstrings."""
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
import numpy as np
from scipy.sparse import csc_matrix
from pymatching import Matching
from qiskit.primitives import BitArray
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from heavyhex_qft.utils import as_bitarray
from heavyhex_qft.plaquette_dual import PlaquetteDual
from skqd_z2lgt.utils import read_bits


def correct_link_state(
    link_state: np.ndarray,
    dual_lattice: TriangularZ2Lattice,
    matching: Optional[Matching] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lattice = dual_lattice.primal
    if not matching:
        matching = Matching(csc_matrix(lattice.vl_matrix))
    magnetic_mask = np.ones(dual_lattice.num_plaquettes, dtype=bool)
    active_idx = [lattice.plaquette_id_to_idx(pid) for pid in lattice.active_plaquette_ids]
    magnetic_mask[active_idx] = False
    syndrome, plaq_state = _decode(as_bitarray(link_state), dual_lattice, matching, magnetic_mask)
    # Need to recompose the plaquette state
    full_plaq_state = np.zeros(lattice.num_plaquettes, dtype=np.uint8)
    full_plaq_state[~magnetic_mask] = plaq_state
    corrected_link_state = dual_lattice.encode_plaq_to_link(full_plaq_state)
    return syndrome, corrected_link_state


def _decode(link_state, dual_lattice, matching, magnetic_mask):
    """Get the syndrome of the link state and apply the MWPM correction.

    When the base link state of the dual lattice is nonzero, we have three ways to correct:
    - XOR with the base link state, compute the syndrome, apply MWPM, correct
    - Compute the syndrome, apply MWPM, XOR with the base state, correct
    - Compute the syndrome, XOR with the base syndrome, apply MWPM, correct
    Either way results in a unique (independent of the true link state) map from link bit-flip
    errors to Wilson loops, but the obtained loops will be different. For our purpose of learning
    the association between errors and loops (more specifically conditional probability of loops for
    given error syndromes), there is likely no real difference in the outcome between the two
    approaches. We choose the first approach ("neutralize" the state) here.
    """
    lattice = dual_lattice.primal
    # Get the electric charges = syndrome for MWPM correction
    e_syndrome = lattice.get_charge_config(link_state ^ dual_lattice.base_link_state)
    correction = matching.decode(e_syndrome)
    corrected_link_state = link_state ^ correction
    # If the corrected link state have excitations *wrt base* at magnetic charges, flip
    plaq_state = dual_lattice.decode_link_to_plaq(corrected_link_state)
    if magnetic_mask is None:
        return e_syndrome, plaq_state
    syndrome = np.concatenate([e_syndrome, plaq_state[magnetic_mask]])
    return syndrome, plaq_state[~magnetic_mask]


def decode_all(
    bit_array: BitArray,
    dual_lattice: PlaquetteDual,
    shuffle: bool = False,
    batch_size: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the counts dict to input data for correction learning."""
    array = bit_array.array
    num_bits = bit_array.num_bits
    shots = array.shape[0]

    if batch_size <= 0:
        out = _decode_batch(array, num_bits, dual_lattice)
    else:
        with ProcessPoolExecutor() as executor:
            futures = []
            start = 0
            while start < shots:
                end = start + batch_size
                fut = executor.submit(_decode_batch, array[start:end], num_bits, dual_lattice)
                futures.append((start, end, fut))
                start = end

        nact = dual_lattice.primal.num_active_plaquettes
        nsyn = dual_lattice.primal.num_vertices + (dual_lattice.num_plaquettes - nact)
        out = (
            np.empty((shots, nsyn), dtype=np.uint8),
            np.empty((shots, nact), dtype=np.uint8)
        )

        for start, end, fut in futures:
            batch_out = fut.result()
            out[0][start:end] = batch_out[0]
            out[1][start:end] = batch_out[1]

    if shuffle:
        indices = np.arange(shots)
        np.random.default_rng().shuffle(indices)
        out = (out[0][indices], out[1][indices])

    return out


def _decode_batch(batch_array, num_bits, dual_lattice):
    lattice = dual_lattice.primal
    active_idx = [lattice.plaquette_id_to_idx(pid) for pid in lattice.active_plaquette_ids]
    if (nmag := lattice.num_plaquettes - len(active_idx)) == 0:
        magnetic_mask = None
    else:
        magnetic_mask = np.ones(dual_lattice.num_plaquettes, dtype=bool)
        magnetic_mask[active_idx] = False

    out = (
        np.empty((batch_array.shape[0], lattice.num_vertices + nmag), dtype=np.uint8),
        np.empty((batch_array.shape[0], len(active_idx)), dtype=np.uint8)
    )

    matching = Matching(csc_matrix(lattice.vl_matrix))
    link_states = read_bits(batch_array, num_bits=num_bits, align='right')
    for ishot, link_state in enumerate(link_states):
        syndrome, plaq_state = _decode(link_state, dual_lattice, matching, magnetic_mask)
        out[0][ishot] = syndrome
        out[1][ishot] = plaq_state

    return out
