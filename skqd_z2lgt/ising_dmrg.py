"""Wrapper for ising_dmrg.jl."""
import os
import sys
import tempfile
import subprocess
from typing import Optional
import numpy as np
import h5py
from qiskit.quantum_info import SparsePauliOp


def ising_dmrg(
    hamiltonian: SparsePauliOp,
    filename: Optional[str] = None,
    nsweeps: int = 5,
    maxdim: Optional[list[int]] = None,
    cutoff: float = 1.e-10,
    julia_bin: str | list[str] = 'julia'
):
    """Call ising_dmrg.jl. There must be a much smarter way to do this."""
    zz_indices = []
    zz_coeffs = []
    z_indices = []
    z_coeffs = []
    x_indices = []
    x_coeffs = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        pstr = pauli.to_label()
        if pstr.count('Z') == 2:
            first = pstr.index('Z')
            zz_indices.append([first, pstr.index('Z', first + 1)])
            zz_coeffs.append(coeff)
        elif pstr.count('Z') == 1:
            z_indices.append(pstr.index('Z'))
            z_coeffs.append(coeff)
        elif pstr.count('X') == 1:
            x_indices.append(pstr.index('X'))
            x_coeffs.append(coeff)
        else:
            raise RuntimeError(pstr)

    maxdim = maxdim or [10, 20, 100, 100]
    if nsweeps > 4 and len(maxdim) < nsweeps:
        maxdim += [200] * (nsweeps - 4)

    is_tempfile = False
    if not filename:
        is_tempfile = True
        with tempfile.NamedTemporaryFile() as tfile:
            filename = tfile.name

    with h5py.File(filename, 'w', libver='latest') as out:
        out.create_dataset('num_qubits', data=hamiltonian.num_qubits)
        out.create_dataset('zz_indices', data=np.array(zz_indices))
        out.create_dataset('zz_coeffs', data=np.array(zz_coeffs))
        out.create_dataset('z_indices', data=np.array(z_indices))
        out.create_dataset('z_coeffs', data=np.array(z_coeffs))
        out.create_dataset('x_indices', data=np.array(x_indices))
        out.create_dataset('x_coeffs', data=np.array(x_coeffs))
        out.create_dataset('nsweeps', data=nsweeps)
        out.create_dataset('maxdim', data=maxdim)
        out.create_dataset('cutoff', data=cutoff)

    program = os.path.join(
        os.path.dirname(__file__),
        '_julia',
        'ising_dmrg.jl'
    )

    if isinstance(julia_bin, str):
        julia_bin = [julia_bin]

    try:
        proc = subprocess.run(julia_bin + [program, filename],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.output + '\n')
        sys.stderr.write(exc.stderr + '\n')
        sys.stderr.flush()
        raise
    
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()

    with h5py.File(filename, 'r', libver='latest') as source:
        energy = source['energy'][()]

    if is_tempfile:
        os.unlink(filename)

    return energy


def get_mps_probs(
    filename: str,
    num_samples: int = 100000,
    num_threads: Optional[int] = None,
    julia_bin: str | list[str] = 'julia'
) -> tuple[np.ndarray, np.ndarray]:
    """Call mps_sparsity.jl and get the list of probable computational basis states."""
    program = os.path.join(
        os.path.dirname(__file__),
        '_julia',
        'mps_sparsity.jl'
    )

    if isinstance(julia_bin, str):
        julia_bin = [julia_bin]

    if num_threads is not None:
        julia_bin += ['-t', f'{num_threads}']

    proc = subprocess.run(julia_bin + [program, filename, str(num_samples)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()

    with h5py.File(filename, 'r', libver='latest') as source:
        states = source['states'][()]
        probs = source['probs'][()]

    return states, probs


def get_mps_coverage(
    filename: str,
    samples_filename: str,
    icp: int,
    ndt: int,
    nstep: int,
    nexp: int,
    julia_bin: str | list[str] = 'julia'
) -> float:
    program = os.path.join(
        os.path.dirname(__file__),
        '_julia',
        'mps_coverage.jl'
    )

    if isinstance(julia_bin, str):
        julia_bin = [julia_bin]

    with tempfile.NamedTemporaryFile() as tfile:
        out_filename = tfile.name

    args = [filename, samples_filename, f'c{icp}/dt{ndt}/step{nstep + 1}/exp{nexp}', out_filename]

    proc = subprocess.run(julia_bin + [program] + args,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()

    with h5py.File(out_filename, 'r', libver='latest') as source:
        prob = source['prob'][()]

    os.unlink(out_filename)

    return prob
