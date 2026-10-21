"""Input / output operations."""
import os
from pathlib import Path
import logging
from typing import Optional
import numpy as np
from scipy.sparse import csr_array
import h5py
from qiskit.primitives import BitArray, PrimitiveResult
from skqd_z2lgt.utils import read_bits, save_bits
from skqd_z2lgt.orchestration.parameters import Parameters

RecoData = tuple[np.ndarray, np.ndarray]  # (syndrome, plaquette data)


def load_dmrg(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
) -> float:
    path = Path(parameters.pkgpath) / 'dmrg.h5'
    if os.path.exists(path):
        logger.info('DMRG result already exists in the output file.')
        with h5py.File(path, 'r', libver='latest') as source:
            return source['energy'][()]


def save_dmrg(
    parameters: Parameters,
    energy: float,
    # mps_states: np.ndarray,
    # mps_probs: np.ndarray,
    logger: Optional[logging.Logger] = None
):
    path = Path(parameters.pkgpath) / 'dmrg.h5'
    with h5py.File(path, 'w', libver='latest') as out:
        out.create_dataset('energy', data=energy)
        # out.create_dataset('mps_states', data=states)
        # out.create_dataset('mps_probs', data=probs)


def save_raw(
    parameters: Parameters,
    pub_result: PrimitiveResult,
    logger: Optional[logging.Logger] = None
):
    """Save the sampled bitstrings to files."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving raw link data')
    dirpath = Path(parameters.pkgpath) / 'data' / 'raw'
    os.makedirs(dirpath, exist_ok=True)

    ires = 0
    for etype in ['exp', 'ref']:
        for idt, dt in enumerate(parameters.circuit.dts):
            for step in parameters.circuit.sampled_steps:
                path = dirpath / f'{etype}_dt{idt}_k{step}.h5'
                with h5py.File(path, 'w', libver='latest') as out:
                    out.attrs['delta_t'] = dt
                    bit_array = pub_result[ires].data.c
                    dataset = out.create_dataset('link', data=bit_array.array)
                    dataset.attrs['num_bits'] = bit_array.num_bits
                    ires += 1


def load_raw(
    parameters: Parameters,
    etype: Optional[str] = None,
    idt: Optional[int] = None,
    step: Optional[int] = None
) -> (tuple[list[list[BitArray]], list[list[BitArray]]] | list[list[BitArray]] | list[BitArray]
      | BitArray):
    """Load the sampled bitstrings from files."""
    def read_bit_array(et, itm, st):
        filename = Path(parameters.pkgpath) / 'data' / 'raw' / f'{et}_dt{itm}_k{st}.h5'
        with h5py.File(filename, 'r', libver='latest') as source:
            dataset = source['link']
            return BitArray(dataset[()], int(dataset.attrs['num_bits']))

    def read_dt_arrays(et, itm):
        return [read_bit_array(et, itm, st) for st in parameters.circuit.sampled_steps]

    def read_et_arrays(et):
        return [read_dt_arrays(et, itm) for itm in range(len(parameters.circuit.dts))]

    if etype is None:
        return tuple(read_et_arrays(et) for et in ['exp', 'ref'])
    if idt is None:
        return read_et_arrays(etype)
    if step is None:
        return read_dt_arrays(etype, idt)
    return read_bit_array(etype, idt, step)


def save_reco(
    parameters: Parameters,
    arrays: RecoData,
    etype: str,
    idt: int,
    step: int,
    logger: Optional[logging.Logger] = None
):
    """Save syndrome and plaquette data to files."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving reco data for etype=%s idt=%d step=%d', etype, idt, step)

    path = Path(parameters.pkgpath) / 'data' / 'reco' / f'{etype}_dt{idt}_k{step}.h5'
    os.makedirs(path.parent, exist_ok=True)
    with h5py.File(path, 'w', libver='latest') as out:
        for name, array in zip(['syndrome', 'plaq'], arrays):
            save_bits(out, name, array)


def check_reco(
    parameters: Parameters,
    etype: str,
    idt: int,
    step: int,
    logger: Optional[logging.Logger] = None
) -> bool:
    logger = logger or logging.getLogger(__name__)
    path = Path(parameters.pkgpath) / 'data' / 'reco' / f'{etype}_dt{idt}_k{step}.h5'
    if os.path.exists(path):
        logger.info('Reco data for etype=%s idt=%d step=%d already exists.', etype, idt, step)
        return True
    return False


def load_reco(
    parameters: Parameters,
    etype: Optional[str] = None,
    idt: Optional[int] = None,
    step: Optional[int] = None
) -> (tuple[list[list[RecoData]], list[list[RecoData]]] | list[list[RecoData]] | list[RecoData]
      | RecoData):
    """Load syndrome and plaquette data from files."""
    def read_reco_data(et, itm, st):
        path = Path(parameters.pkgpath) / 'data' / 'reco' / f'{et}_dt{itm}_k{st}.h5'
        with h5py.File(path, 'r', libver='latest') as source:
            return tuple(read_bits(source[name]) for name in ['syndrome', 'plaq'])

    def read_dt_data(et, itm):
        return [read_reco_data(et, itm, st) for st in parameters.circuit.sampled_steps]

    def read_et_data(et):
        return [read_dt_data(et, itm) for itm in range(len(parameters.circuit.dts))]

    if etype is None:
        return tuple(read_et_data(et) for et in ['exp', 'ref'])
    if idt is None:
        return read_et_data(etype)
    if step is None:
        return read_dt_data(etype, idt)
    return read_reco_data(etype, idt, step)


def check_crbm_model(
    parameters: Parameters,
    idt: int,
    step: int,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Check existence of a saved model without loading."""
    logger = logger or logging.getLogger(__name__)
    path = Path(parameters.pkgpath) / 'crbm' / f'dt{idt}_k{step}.h5'
    if os.path.exists(path):
        logger.info('CRBM model for dt=%f step=%d already exists.', parameters.circuit.dts[idt],
                    step)
        return True
    return False


def load_crbm_train_records(
    parameters: Parameters,
    idt: int,
    step: int,
) -> dict[str, np.ndarray]:
    path = Path(parameters.pkgpath) / 'crbm' / f'dt{idt}_k{step}.h5'
    with h5py.File(path, 'r', libver='latest') as source:
        return {key: value[()] for key, value in source['records'].items()}


def check_skqd_result(
    parameters: Parameters,
    result_name: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    logger = logger or logging.getLogger(__name__)
    path = Path(parameters.pkgpath) / f'{result_name}.h5'
    if os.path.exists(path):
        logger.info('SKQD result %s exists.', result_name)
        return True
    return False


def load_skqd_result(
    parameters: Parameters,
    mode: str
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Return the saved SKQD result."""
    path = Path(parameters.pkgpath) / f'skqd_{mode}.h5'
    with h5py.File(path, 'r', libver='latest') as source:
        # sqd_states = read_bits(source['sqd_states'])
        energies = source['energies'][()]
        # eigvec = source['eigvec'][()]
        subspace_dims = source['subspace_dims'][()]
    # return sqd_states, energy, eigvec
    return energies, subspace_dims


def save_skqd_result(
    parameters: Parameters,
    mode: str,
    energies: np.ndarray,
    subspace_dims: np.ndarray,
    # states: np.ndarray,
    # eigvec: np.ndarray,
    # ham_proj: Optional[np.ndarray] = None
):
    """Save the SKQD result to file."""
    path = Path(parameters.pkgpath) / f'skqd_{mode}.h5'
    with h5py.File(path, 'w', libver='latest') as out:
        out.create_dataset('energies', data=energies)
        out.create_dataset('subspace_dims', data=subspace_dims)
        # save_bits(out, 'states', states)
        # out.create_dataset('eigvec', data=eigvec)
        # if ham_proj is not None:
        #     group = out.create_group('ham_proj')
        #     group.create_dataset('data', data=ham_proj.data)
        #     group.create_dataset('indices', data=ham_proj.indices)
        #     group.create_dataset('indptr', data=ham_proj.indptr)


def load_projected_hamiltonian(source: h5py.Group):
    """Construct a CSR array from saved Hamiltonian data."""
    subgroup = source['ham_proj']
    data = subgroup['data'][()]
    indices = subgroup['indices'][()]
    indptr = subgroup['indptr'][()]
    shape = (indptr.shape[0] - 1,) * 2
    return csr_array((data, indices, indptr), shape=shape)
